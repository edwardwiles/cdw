# ============================================================================
# Addendum "test persistent preallocation and eliminate redundant full price
# tensors", Backend B: drop the raw `price0` tensor entirely, keep only
# `pTσ0`. Every ranking/tie/O(1)-winner-update site is mechanically rewritten
# with FLIPPED comparison direction (argmin price == argmax pTσ, since
# pTσ = price^(1-σ), σ>1 makes x -> x^(1-σ) strictly DECREASING on x>0 --
# see docs/fullA_price_tensor_audit.md §3 for the full derivation this file
# implements). This is a faithful, order-reversed mirror of
# lfix_incremental.jl -- every function here has a named counterpart there;
# read the two side by side if auditing.
#
# ADDITIVE ONLY: does not modify lfix_incremental.jl, composite_gradient.jl,
# bandwidth_quantile.jl, or LFixBaseCache. Defines a PARALLEL struct
# (`LFixBaseCacheB`) and parallel function set; nothing existing calls into
# this file. Equivalence with the reference (Backend "Reference") is verified
# in test_lfix_pTsigma_only.jl.
#
# BONUS not just a memory saving: `pTsigma_cell!` below computes ONLY the
# sigma-transformed value, skipping the raw-price FLOP path entirely
# (`price = constCons_od ./ U[:,o].^(-μ)` is a genuinely separate computation
# from `pTσ = constConsσ_od ./ Uσ[:,o].^(-μ)`, using a DIFFERENT array (U vs
# Uσ) -- dropping price0 also removes that arithmetic, not merely its
# storage).
# ============================================================================
include(joinpath(@__DIR__, "lfix_incremental.jl"))    # aod_pow_cell, TiedWinnerError, lfix_from_q, affected_cells, cf_contrib_at
include(joinpath(@__DIR__, "winner_certificate.jl"))  # constCons_matrix -- reused for a PRICE-space tie check, see detect_pTσ_ties's docstring below

"""
    pTsigma_cell(θ_full, ctx, o, d) -> pTσ::Vector{W}

ONLY the sigma-transformed value (mirrors `price_and_pTsigma_cell`'s `pTσ` output exactly,
bit-for-bit -- same formula, just does not also compute the discarded `price`).
"""
function pTsigma_cell(θ_full::AbstractVector, ctx, o::Int, d::Int)
    γo = ctx.γ; σ = θ_full[2]; μ = θ_full[1]
    AodPow = aod_pow_cell(θ_full, ctx, o, d)
    wPow_o = γo.wHat[o]^(1 - σ)
    constConsσ_od = wPow_o * (AodPow * γo.τ[o, d])^(1 - σ)
    return constConsσ_od ./ (@view(γo.Uσ[:, o]) .^ (-μ))
end

"In-place variant of `pTsigma_cell` -- writes into caller-supplied `psbuf`."
function pTsigma_cell!(psbuf::AbstractVector, θ_full::AbstractVector, ctx, o::Int, d::Int)
    γo = ctx.γ; σ = θ_full[2]; μ = θ_full[1]
    AodPow = aod_pow_cell(θ_full, ctx, o, d)
    wPow_o = γo.wHat[o]^(1 - σ)
    constConsσ_od = wPow_o * (AodPow * γo.τ[o, d])^(1 - σ)
    psbuf .= constConsσ_od ./ (@view(γo.Uσ[:, o]) .^ (-μ))
    return nothing
end

"""
    max_and_secondmax(col) -> (wmax, wo, gap)

Mirror of `winners_v2.jl::min_and_secondmin`, tracking the MAXIMUM (winner in pTσ-space is the
origin with the HIGHEST pTσ, since pTσ = price^(1-σ) reverses order). Same first-occurrence
tie-break convention (`isless`-based, matches `findmax`).
"""
function max_and_secondmax(col)
    n = length(col)
    @assert n >= 1 "max_and_secondmax: empty column"
    m1 = col[1]; idx1 = 1
    m2 = oftype(m1, -Inf)
    @inbounds for i in 2:n
        v = col[i]
        if isless(m1, v)
            m2 = m1
            m1 = v; idx1 = i
        elseif isless(m2, v)
            m2 = v
        end
    end
    gap = n >= 2 ? (m1 - m2) : oftype(m1, Inf)
    return m1, idx1, gap
end

"""
    max_secondthirdmax_with_idx(col) -> (m1, idx1, m2, idx2, m3, idx3)

Mirror of `lfix_incremental.jl::min_secondthirdmin_with_idx`: rank-1/2/3 by DESCENDING value
(pTσ-space winner = argmax). `idx3`/`m3` are `0`/`-Inf` if `D<3`.
"""
function max_secondthirdmax_with_idx(col)
    n = length(col)
    m1 = col[1]; idx1 = 1
    m2 = oftype(m1, -Inf); idx2 = 0
    m3 = oftype(m1, -Inf); idx3 = 0
    @inbounds for i in 2:n
        v = col[i]
        if isless(m1, v)
            m3 = m2; idx3 = idx2
            m2 = m1; idx2 = idx1
            m1 = v; idx1 = i
        elseif isless(m2, v)
            m3 = m2; idx3 = idx2
            m2 = v; idx2 = i
        elseif isless(m3, v)
            m3 = v; idx3 = i
        end
    end
    return m1, idx1, m2, idx2, m3, idx3
end

"""
    detect_pTσ_ties(pTσ0, D, W; tol=0.0) -> Vector{Tuple{Int,Int}}

Mirror of `detect_price_ties`: scans for (draw,destination) pairs where 2+ origins are tied at
the row MAXIMUM of pTσ0 (within `tol`). MATHEMATICALLY, a tie in price is EXACTLY a tie in pTσ
(strict monotone bijection on positive reals, `pTσ=price^(1-σ)`) -- but NUMERICALLY this does
NOT transfer at `tol=0.0` (bit-exact) reliability: `pTσ0` here is computed via constConsσ/Uσ, an
INDEPENDENTLY-COMPUTED floating-point pathway from `price0`'s constCons/U (different array,
different power chain) -- even though `Uσ == U.^(1-σ)` exactly at context-build time
(confirmed, see docs/fullA_price_tensor_audit.md §1), `(U^(1-σ))^(-μ)` and `(U^(-μ))^(1-σ)` are
NOT guaranteed bit-identical in floating-point arithmetic, so an exact price tie can land a few
ULPs apart in the SEPARATELY-computed pTσ0. **Found empirically this session**
(`test_lfix_pTsigma_only.jl`'s own adversarial fixture failed to trigger `TiedWinnerError` here
at `tol=0.0` despite an exact, confirmed price tie) -- NOT a theoretical concern, a real one.
`winner_certificate.jl::build_winner_ref` already hit this EXACT issue for its OWN log-score
representation and defends against it by checking ties in the ORIGINAL price/UPow
representation directly (its own comment: "so an exact price tie is caught even if the log
round-trip would not reproduce it") -- `build_lfix_base_cache_B` below does the SAME thing
(computes `constCons`/`UPow` TRANSIENTLY, O(W*D)+O(D^2), NOT a stored W*D*D tensor, discarded
immediately after the tie check) rather than trusting this function at `tol=0.0`. This function
is kept for documentation/comparison purposes only -- NOT called by `build_lfix_base_cache_B`.
"""
function detect_pTσ_ties(pTσ0::Array{Float64,3}, D::Int, W::Int; tol::Float64 = 0.0)
    tied = Tuple{Int,Int}[]
    @inbounds for d in 1:D, ω in 1:W
        mx = pTσ0[ω, 1, d]
        for o in 2:D
            pTσ0[ω, o, d] > mx && (mx = pTσ0[ω, o, d])
        end
        n_at_max = 0
        for o in 1:D
            pTσ0[ω, o, d] >= mx - tol && (n_at_max += 1)
        end
        n_at_max > 1 && push!(tied, (ω, d))
    end
    return tied
end

"""
    detect_price_ties_from_factors(θ_full, ctx, D, W) -> Vector{Tuple{Int,Int}}

ROBUST tie check used by `build_lfix_base_cache_B`: computes `constCons` (D×D, via
`constCons_matrix`) and `UPow = ctx.U .^ (-μ)` (W×D) -- exactly the factors `price_{sod} =
constCons_{od}/UPow_{so}` decomposes into -- TRANSIENTLY (both discarded on return, never
stored in `LFixBaseCacheB`) and checks ties in genuine price space, mirroring
`build_winner_ref`'s own defensive check (winner_certificate.jl) exactly. O(W*D)+O(D^2)
transient memory, not O(W*D^2) -- Backend B's persistent-memory goal is unaffected by this
one-time construction-time check.
"""
function detect_price_ties_from_factors(θ_full::AbstractVector, ctx, D::Int, W::Int)
    μ = θ_full[1]
    constCons, _, _ = constCons_matrix(θ_full, ctx)
    UPow = ctx.U .^ (-μ)
    tied = Tuple{Int,Int}[]
    @inbounds for d in 1:D, ω in 1:W
        mn = constCons[1, d] / UPow[ω, 1]
        for o in 2:D
            p = constCons[o, d] / UPow[ω, o]
            p < mn && (mn = p)
        end
        n_at_min = 0
        for o in 1:D
            (constCons[o, d] / UPow[ω, o]) <= mn && (n_at_min += 1)
        end
        n_at_min > 1 && push!(tied, (ω, d))
    end
    return tied
end

"""
    update_winner_o1_B(pTσ_wo, wo, pTσ_ro, ro, o_changed, new_pTσ) -> (new_wo, new_pTσ_wo, new_ro, new_pTσ_ro, exact)

Mirror of `update_winner_o1`, EVERY comparison direction flipped (`<=`->`>=`, `<`->`>`) since
higher pTσ now wins. Same case-analysis proof, order-reversed -- see that function's docstring
for the full derivation this mirrors.
"""
function update_winner_o1_B(pTσ_wo::Float64, wo::Int, pTσ_ro::Float64, ro::Int, o_changed::Int, new_pTσ::Float64)
    if o_changed == wo
        if new_pTσ >= pTσ_ro
            return wo, new_pTσ, ro, pTσ_ro, true
        else
            return ro, pTσ_ro, o_changed, new_pTσ, false
        end
    else
        if new_pTσ > pTσ_wo
            return o_changed, new_pTσ, wo, pTσ_wo, true
        elseif o_changed == ro
            return wo, pTσ_wo, o_changed, new_pTσ, false
        else
            new_ro = new_pTσ > pTσ_ro ? o_changed : ro
            new_pTσ_ro = new_pTσ > pTσ_ro ? new_pTσ : pTσ_ro
            return wo, pTσ_wo, new_ro, new_pTσ_ro, true
        end
    end
end

"""
    LFixBaseCacheB

Backend B twin of `LFixBaseCache`: identical fields EXCEPT `price0` is gone, and the
winner/runner-up price-LEVEL fields are replaced by their pTσ-VALUE equivalents
(`winner_pTσ0`/`runnerup_pTσ0`; `third_pTσ0` already existed in the original under this exact
name/meaning -- no rename needed there, it does double duty for ranking AND the economic
moment in this backend).
"""
struct LFixBaseCacheB
    D::Int; oci::Int; W::Int; μ::Float64; σ::Float64; baseIndex::Int
    gammafac::Float64
    SW::Vector{Float64}
    denom::Vector{Float64}
    CONST_d::Vector{Float64}
    pTσ0::Array{Float64,3}          # W x D x D -- the ONLY dense tensor
    winner0::Matrix{Int}
    winner_pTσ0::Matrix{Float64}
    runnerup0::Matrix{Int}
    runnerup_pTσ0::Matrix{Float64}
    third0::Matrix{Int}
    third_pTσ0::Matrix{Float64}
    contrib0::Matrix{Float64}
    λstar::Vector{Float64}
    ζstar::Float64
    q0::Vector{Float64}
    wPrime_bi::Float64; τPrime_bi::Float64; LPrime_bi::Float64
    Uσ_bi::Vector{Float64}
    λ_cf::Float64
    cf_contrib0::Vector{Float64}
end

"Backend B twin of `build_lfix_base_cache`. Same formula chain, price0 never computed at all."
function build_lfix_base_cache_B(x_free0::AbstractVector, ctx, base::BaseDualState; validate_dense::Bool = false)
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = γo.SamplingWeights[1:W]
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]
    λstar = base.λstar

    pTσ0 = Array{Float64}(undef, W, D, D)
    for d in 1:D, o in 1:D
        pTsigma_cell!(@view(pTσ0[:, o, d]), base.θ_full0, ctx, o, d)
    end

    tied_pairs = detect_price_ties_from_factors(base.θ_full0, ctx, D, W)
    isempty(tied_pairs) || throw(TiedWinnerError(length(tied_pairs), tied_pairs[1:min(5, end)]))

    winner0 = Matrix{Int}(undef, W, D)
    winner_pTσ0 = Matrix{Float64}(undef, W, D)
    runnerup0 = Matrix{Int}(undef, W, D)
    runnerup_pTσ0 = Matrix{Float64}(undef, W, D)
    third0 = Matrix{Int}(undef, W, D)
    third_pTσ0 = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        m1, idx1, m2, idx2, m3, idx3 = max_secondthirdmax_with_idx(@view(pTσ0[ω, :, d]))
        winner0[ω, d] = idx1; winner_pTσ0[ω, d] = m1
        runnerup0[ω, d] = idx2; runnerup_pTσ0[ω, d] = m2
        third0[ω, d] = idx3; third_pTσ0[ω, d] = idx3 == 0 ? -Inf : m3
    end

    CONST_d = zeros(D)
    for d in 1:D
        s = 0.0
        for o in 1:D
            d1 = d + (o - 1) * D
            s += λstar[d1] * (-γo.P[d1] * denom[d])
        end
        CONST_d[d] = s
    end

    contrib0 = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, ω in 1:W
        wo = winner0[ω, d]
        d1w = d + (wo - 1) * D
        contrib0[ω, d] = (SW[ω] / gammafac) * (CONST_d[d] + λstar[d1w] * winner_pTσ0[ω, d])
    end

    wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
    wPrime_bi = wPrime[bi]
    τPrime_bi = γo.τPrime[bi, bi]
    LPrime_bi = γo.LPrime[bi]
    Uσ_bi = γo.Uσ[:, bi] .^ (-μ)
    d1_cf = D^2 + 1
    λ_cf = oci - 1 >= d1_cf ? λstar[d1_cf] : 0.0

    AodPow_bibi0 = aod_pow_cell(base.θ_full0, ctx, bi, bi)
    γ_prime_bi0 = base.θ_full0[3+D]
    constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi0 * τPrime_bi)^(1 - σ)
    denom_cf0 = γ_prime_bi0^σ * wPrime_bi_gdp(wPrime_bi, LPrime_bi)
    raw_cf0 = constConsσ_bibi ./ Uσ_bi .- denom_cf0
    cf_contrib0 = λ_cf .* (raw_cf0 ./ gammafac .* SW)

    q0 = [-base.ζstar - sum(@view(contrib0[s, :])) - cf_contrib0[s] for s in 1:W]

    if validate_dense
        K = zeros(W); Gfull = zeros(W, obj.d)
        obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
        q0_true = [-base.ζstar - dot(λstar, @view(Gfull[s, 1:oci-1])) for s in 1:W]
        maxerr = maximum(abs.(q0_true .- q0))
        maxerr < 1e-8 || error("build_lfix_base_cache_B: self-validation FAILED, max|q0_true-q0_cache|=$maxerr")
    end

    return LFixBaseCacheB(D, oci, W, μ, σ, bi, gammafac, SW, denom, CONST_d, pTσ0,
        winner0, winner_pTσ0, runnerup0, runnerup_pTσ0, third0, third_pTσ0, contrib0,
        λstar, base.ζstar, q0, wPrime_bi, τPrime_bi, LPrime_bi, Uσ_bi, λ_cf, cf_contrib0)
end

"""
New DISPATCH methods (not edits) for the existing generic functions `cf_contrib_at`
(lfix_incremental.jl) and `gamma_component_analytic` (composite_gradient.jl): both only ever
read fields that exist identically on `LFixBaseCacheB` (`baseIndex`/`σ`/`wPrime_bi`/
`τPrime_bi`/`λ_cf`/`gammafac`/`Uσ_bi`/`SW`), so the bodies are copied verbatim from the
originals, just re-annotated to accept the new cache type. Julia's multiple dispatch makes this
purely additive (a new method on an existing generic function name) -- neither source file is
modified.
"""
function cf_contrib_at(cache::LFixBaseCacheB, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    raw_cf = constConsσ_bibi ./ cache.Uσ_bi .- denom_cf
    return cache.λ_cf .* (raw_cf ./ cache.gammafac .* cache.SW)
end

function gamma_component_analytic(cache::LFixBaseCacheB, base::BaseDualState, g::Float64)
    σ = cache.σ
    mean_mSW = dot(base.m_star, cache.SW) / cache.W
    return -cache.λ_cf * σ * g^(σ - 1) * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi) / cache.gammafac * mean_mSW
end

"Backend B twin of `dest_contrib_block_local`: full recompute of all D origins' pTσ, then max-scan."
function dest_contrib_block_local_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int)
    D = cache.D; W = cache.W
    pTσ_d = Matrix{Float64}(undef, W, D)
    for o in 1:D
        pTσ_d[:, o] .= pTsigma_cell(θ_full, ctx, o, d)
    end
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        _, wo, _ = max_and_secondmax(@view(pTσ_d[ω, :]))
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_d[ω, wo])
    end
    return contrib
end

"Backend B twin of `dest_contrib_incremental` (Tier 2, O(D) rescan fallback for 2+ changed origins)."
function dest_contrib_incremental_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    new_pTσ = Dict{Int,Vector{Float64}}()
    for o in changed_origins
        new_pTσ[o] = pTsigma_cell(θ_full, ctx, o, d)
    end
    contrib = Vector{Float64}(undef, W)
    col = Vector{Float64}(undef, D)
    @inbounds for ω in 1:W
        for o in 1:D
            col[o] = haskey(new_pTσ, o) ? new_pTσ[o][ω] : cache.pTσ0[ω, o, d]
        end
        _, wo, _ = max_and_secondmax(col)
        pTσ_wo = col[wo]
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"Backend B twin of `dest_contrib_incremental_top3` (Tier 2b, exact top-3-cache-based, <=2 changed origins)."
function dest_contrib_incremental_top3_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) > 2 && return dest_contrib_incremental_B(cache, ctx, θ_full, d, changed_origins)

    D = cache.D; W = cache.W
    Cd = changed_origins
    new_pTσ = Dict{Int,Vector{Float64}}()
    for o in Cd
        new_pTσ[o] = pTsigma_cell(θ_full, ctx, o, d)
    end
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_pTσ = -Inf
        if !(r1 in Cd)
            best_o = r1; best_pTσ = cache.winner_pTσ0[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_pTσ = cache.runnerup_pTσ0[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_pTσ = cache.third_pTσ0[ω, d]
        end
        bo = best_o; bpTσ = best_pTσ
        for o in Cd
            v = new_pTσ[o][ω]
            # Remediation task Part E (finding F5): canonical exact-tie convention -- lowest
            # origin index wins, matching every generic-rescan tier (note the direction here is
            # `>` since higher pTsigma wins, not lower price -- the tie-break itself is still
            # "lowest index"). See lfix_factorized.jl's identical fix for the full rationale.
            if v > bpTσ || (v == bpTσ && o < bo)
                bpTσ = v; bo = o
            end
        end
        if bo == 0
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = haskey(new_pTσ, o) ? new_pTσ[o][ω] : cache.pTσ0[ω, o, d]
            end
            _, bo, _ = max_and_secondmax(col)
            bpTσ = col[bo]
        end
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * bpTσ)
    end
    return contrib
end

"Backend B twin of `dest_contrib_incremental_o1` (Tier 3, TRUE O(1) single-changed-origin update)."
function dest_contrib_incremental_o1_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int}; multi_method::Symbol = :top3)
    if length(changed_origins) != 1
        multi_method === :top3 && return dest_contrib_incremental_top3_B(cache, ctx, θ_full, d, changed_origins)
        multi_method === :generic && return dest_contrib_incremental_B(cache, ctx, θ_full, d, changed_origins)
        error("dest_contrib_incremental_o1_B: multi_method must be :top3 or :generic, got $multi_method")
    end

    D = cache.D; W = cache.W
    o = changed_origins[1]
    new_pTσ = pTsigma_cell(θ_full, ctx, o, d)
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        wo, pTσ_wo, _, _, _exact = update_winner_o1_B(
            cache.winner_pTσ0[ω, d], cache.winner0[ω, d],
            cache.runnerup_pTσ0[ω, d], cache.runnerup0[ω, d],
            o, new_pTσ[ω])
        d1w = d + (wo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"Backend B twin of `lfix_incremental_at`. Same orchestration, dest_contrib_*_B tiers."
function lfix_incremental_at_B(cache::LFixBaseCacheB, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64; tier::Symbol = :incremental, multi_method::Symbol = :top3)
    w = copy(w0); w[coord_idx] = new_val
    z = pivot_expand(w[2:end], pe)
    Aod_theta = exp.(z)
    x_free = vcat(w[1], vec(Aod_theta))
    θ_full = CS.reconstruct_full(x_free, ctx.m)

    cells = affected_cells(pe, coord_idx)
    affected_dests = unique(last.(cells))
    cf_touched = coord_idx == 1 || any(((o, d),) -> o == cache.baseIndex && d == cache.baseIndex, cells)

    q = copy(cache.q0)
    for d in affected_dests
        old_contrib = @view cache.contrib0[:, d]
        new_contrib = if tier == :block_local
            dest_contrib_block_local_B(cache, ctx, θ_full, d)
        elseif tier == :incremental
            origins_here = [o for (o, dd) in cells if dd == d]
            dest_contrib_incremental_B(cache, ctx, θ_full, d, origins_here)
        elseif tier == :incremental_o1
            origins_here = [o for (o, dd) in cells if dd == d]
            dest_contrib_incremental_o1_B(cache, ctx, θ_full, d, origins_here; multi_method = multi_method)
        else
            error("lfix_incremental_at_B: unknown tier=$tier")
        end
        q .-= new_contrib .- old_contrib
    end
    if cf_touched
        new_cf = cf_contrib_at(cache, θ_full, ctx)
        q .-= new_cf .- cache.cf_contrib0
    end

    return lfix_from_q(q, cache.ζstar)
end

# ----------------------------------------------------------------------------
# Bandwidth selection (mirrors composite_gradient.jl's count_winner_flips* +
# select_bandwidth, and bandwidth_quantile.jl's closed-form quantile lookup).
# ----------------------------------------------------------------------------

"Backend B twin of `count_winner_flips_multi` (O(D) rescan fallback, flip COUNTING only)."
function count_winner_flips_multi_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    new_pTσ = Dict{Int,Vector{Float64}}()
    for o in changed_origins
        new_pTσ[o] = pTsigma_cell(θ_full, ctx, o, d)
    end
    col = Vector{Float64}(undef, D)
    flips = 0
    @inbounds for ω in 1:W
        for o in 1:D
            col[o] = haskey(new_pTσ, o) ? new_pTσ[o][ω] : cache.pTσ0[ω, o, d]
        end
        _, wo, _ = max_and_secondmax(col)
        flips += (wo != cache.winner0[ω, d])
    end
    return flips
end

"Backend B twin of `count_winner_flips_multi_top3`."
function count_winner_flips_multi_top3_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    length(changed_origins) > 2 && return count_winner_flips_multi_B(cache, ctx, θ_full, d, changed_origins)
    Cd = changed_origins
    new_pTσ = Dict{Int,Vector{Float64}}()
    for o in Cd
        new_pTσ[o] = pTsigma_cell(θ_full, ctx, o, d)
    end
    flips = 0
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_pTσ = -Inf
        if !(r1 in Cd)
            best_o = r1; best_pTσ = cache.winner_pTσ0[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_pTσ = cache.runnerup_pTσ0[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_pTσ = cache.third_pTσ0[ω, d]
        end
        bo = best_o; bpTσ = best_pTσ
        for o in Cd
            v = new_pTσ[o][ω]
            # Remediation task Part E (finding F5): canonical exact-tie convention -- lowest
            # origin index wins, matching every generic-rescan tier (note the direction here is
            # `>` since higher pTsigma wins, not lower price -- the tie-break itself is still
            # "lowest index"). See lfix_factorized.jl's identical fix for the full rationale.
            if v > bpTσ || (v == bpTσ && o < bo)
                bpTσ = v; bo = o
            end
        end
        if bo == 0
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = haskey(new_pTσ, o) ? new_pTσ[o][ω] : cache.pTσ0[ω, o, d]
            end
            _, bo, _ = max_and_secondmax(col)
        end
        flips += (bo != r1)
    end
    return flips
end

"Backend B twin of `count_winner_flips`."
function count_winner_flips_B(cache::LFixBaseCacheB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int}; multi_method::Symbol = :top3)
    if length(changed_origins) != 1
        multi_method === :top3 && return count_winner_flips_multi_top3_B(cache, ctx, θ_full, d, changed_origins)
        multi_method === :generic && return count_winner_flips_multi_B(cache, ctx, θ_full, d, changed_origins)
        error("count_winner_flips_B: multi_method must be :top3 or :generic, got $multi_method")
    end
    o = changed_origins[1]
    new_pTσ = pTsigma_cell(θ_full, ctx, o, d)
    flips = 0
    @inbounds for ω in 1:cache.W
        wo, _, _, _, _ = update_winner_o1_B(cache.winner_pTσ0[ω, d], cache.winner0[ω, d],
                                             cache.runnerup_pTσ0[ω, d], cache.runnerup0[ω, d],
                                             o, new_pTσ[ω])
        flips += (wo != cache.winner0[ω, d])
    end
    return flips
end

"Backend B twin of `select_bandwidth` (bisection). Same algorithm, count_winner_flips_B."
function select_bandwidth_B(cache::LFixBaseCacheB, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6,
        multi_method::Symbol = :top3)
    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth_B: coord_idx=$coord_idx has no affected A_od cells"
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += count_winner_flips_B(cache, ctx, θ_full, d, origins_here; multi_method = multi_method)
        end
        return total_flips / (cache.W * length(affected_dests))
    end

    h = h0
    lo_frac, hi_frac = target_mass_frac
    m = mass_at(h)
    n_iter = 0
    while n_iter < max_iter
        if m < lo_frac && h < h_ceil
            h = min(h * 2, h_ceil)
        elseif m > hi_frac && h > h_floor
            h = max(h / 2, h_floor)
        else
            break
        end
        m = mass_at(h)
        n_iter += 1
        (h == h_ceil || h == h_floor) && break
    end
    return h, m, (n_iter = n_iter, hit_floor = h == h_floor, hit_ceil = h == h_ceil)
end

"""
    a_block_fd_component_B(cache, ctx, pe, w0, coord_idx, h; multi_method=:top3, max_h_shrinks=4) -> Float64

Backend B twin of `a_block_fd_component` (composite_gradient.jl), including the AUD-12
nonfinite-both-sides retry discipline.
"""
function a_block_fd_component_B(cache::LFixBaseCacheB, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64;
        multi_method::Symbol = :top3, max_h_shrinks::Int = 4)
    h_try = h
    for attempt in 1:(max_h_shrinks + 1)
        Lp = lfix_incremental_at_B(cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h_try; tier = :incremental_o1, multi_method = multi_method)
        Lm = lfix_incremental_at_B(cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h_try; tier = :incremental_o1, multi_method = multi_method)
        if isfinite(Lp) && isfinite(Lm)
            return (Lp - Lm) / (2h_try)
        elseif isfinite(Lp) || isfinite(Lm)
            L0 = lfix_incremental_at_B(cache, ctx, pe, w0, coord_idx, w0[coord_idx]; tier = :incremental_o1, multi_method = multi_method)
            isfinite(L0) || break
            return isfinite(Lp) ? (Lp - L0) / h_try : (L0 - Lm) / h_try
        end
        h_try /= 4
    end
    return NaN
end

"""
    composite_gradient_at_B(x_free0, ctx, pe; base=nothing, multi_method=:top3) -> (g, meta)

Backend B twin of `composite_gradient_at` -- THE main entry point for a complete gradient call
using ONLY the pTσ0-backed cache (no price0 ever constructed).
"""
function composite_gradient_at_B(x_free0::AbstractVector, ctx, pe; base::Union{Nothing,BaseDualState} = nothing, multi_method::Symbol = :top3)
    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache_B(x_free0, ctx, base)
    D = ctx.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])

    h_used = zeros(D2); switch_mass = zeros(D2)
    for k in 2:D2
        h, m, _ = select_bandwidth_B(cache, ctx, pe, w0, k; multi_method = multi_method)
        h_used[k] = h; switch_mass[k] = m
        g[k] = a_block_fd_component_B(cache, ctx, pe, w0, k, h; multi_method = multi_method)
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               winner0 = copy(cache.winner0), gamma_component = g[1])
end
