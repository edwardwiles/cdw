# ============================================================================
# Addendum "test persistent preallocation and eliminate redundant full price
# tensors", Backend C: factorized/on-demand representation. NEITHER price0
# NOR pTσ0 is ever materialized as a dense W*D*D tensor. Persistent state is
# O(W*D) + O(D^2), matching the addendum's own §5 spec:
#
#   S_{sod} = log price_{sod} = logCC_{od} + mulU_{so}      (log-space factorization,
#                                                             not the raw-power K*B
#                                                             the addendum sketches --
#                                                             equivalent, numerically
#                                                             safer, and this codebase
#                                                             ALREADY HAS it validated,
#                                                             see below)
#   pTσ_{sod} = exp((1-σ) * S_{sod})                          (on-demand reconstruction,
#                                                             O(1) per queried cell, no
#                                                             stored tensor at all)
#
# REUSES, VERBATIM, the ALREADY-VALIDATED prior art in winner_certificate.jl
# (continuations 7/8, NOT built this session):
#   - `constCons_matrix` / `logCC0` -- the D×D bilateral log-price reference.
#   - `WinnerRefCache` / `build_winner_ref` -- the O(W*D)+O(D^2) winner/runner-up/
#     third-place ranking scan (top3_scan), NO dense W*D*D array ever built.
#   - `coord_winner_update!`'s exact top-3-cache proof (same logic re-derived
#     inline below rather than calling that exact function, because it writes
#     the FULL W×D winner matrix on every call -- see this file's own
#     `dest_contrib_incremental_top3_C` docstring for why a lighter,
#     single-destination-column version is used instead for the hot L_fix
#     gradient loop).
#
# Per docs/fullA_price_tensor_audit.md §4, this is exactly the unification
# that audit doc recommends: `LFixBaseCache`'s consumption pattern (400
# sequential single/double-coordinate perturbations per gradient call, each
# needing EXACT top-3) wired onto `WinnerRefCache`'s ALREADY-VALIDATED
# factorized representation, rather than a third representation invented from
# scratch.
#
# SCOPE: implements the `:incremental_o1`-equivalent (top-3-cache) tier only
# (the tier a real gradient call actually uses by default) -- NOT separate
# `:block_local`/`:generic` tiers, since Backend C's entire point is the O(D)
# memory footprint; a full O(D)-rescan generic fallback is still provided
# (`dest_contrib_incremental_generic_C`) for the >2-changed-origin defensive
# case, matching every other tier's own established fallback discipline.
# ============================================================================
include(joinpath(@__DIR__, "lfix_incremental.jl"))     # aod_pow_cell, TiedWinnerError, lfix_from_q, affected_cells, cf_contrib_at, wPrime_bi_gdp
include(joinpath(@__DIR__, "winner_certificate.jl"))    # constCons_matrix, WinnerRefCache, build_winner_ref, top3_scan

"pTσ reconstructed on demand from a log-price score S = log(price) -- O(1), no stored tensor. pTσ = price^(1-σ) = exp((1-σ)*S)."
@inline pTσ_from_score(S::Float64, σ::Float64) = exp((1 - σ) * S)

"""
    LFixBaseCacheC

Backend C cache: persistent state is `ref::WinnerRefCache` (O(W*D)+O(D^2): `logCC0`, `mulU`,
winner/runnerup/third INDICES and SCORES `sw`/`sr`/`st3` -- log-price, not raw price or pTσ)
plus the L_fix-specific bookkeeping (`CONST_d`, `contrib0`, `q0`, cf-column pieces) common to
every backend. `contrib0` IS still a dense W×D matrix (unavoidable -- it is the actual economic
moment the outer objective needs), but the two W×D×D tensors (`price0`, `pTσ0`) that the
Reference and Backend B both build are GONE entirely; `ref`'s own footprint is O(W*D), a
D-fold reduction at this problem's D=20 (~5x net reduction after `contrib0`'s own O(W*D)
overhead is counted, see the benchmark report).
"""
struct LFixBaseCacheC
    D::Int; oci::Int; W::Int; μ::Float64; σ::Float64; baseIndex::Int
    gammafac::Float64
    SW::Vector{Float64}
    denom::Vector{Float64}
    CONST_d::Vector{Float64}
    ref::WinnerRefCache
    contrib0::Matrix{Float64}
    λstar::Vector{Float64}
    ζstar::Float64
    q0::Vector{Float64}
    wPrime_bi::Float64; τPrime_bi::Float64; LPrime_bi::Float64
    Uσ_bi::Vector{Float64}
    λ_cf::Float64
    cf_contrib0::Vector{Float64}
end

"""
Backend C twin of `build_lfix_base_cache`. Reuses `build_winner_ref` VERBATIM for the ranking
scan (its own O(W*D^2)-TIME, O(W*D)-MEMORY scan and tie-check are unchanged, unmodified,
untouched) -- only the L_fix-specific `contrib0`/`q0`/cf-column assembly is new here, and it
reconstructs the winner's pTσ ON DEMAND (`pTσ_from_score`) rather than reading a dense tensor.
"""
function build_lfix_base_cache_C(x_free0::AbstractVector, ctx, base::BaseDualState; validate_dense::Bool = false)
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = γo.SamplingWeights[1:W]
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]
    λstar = base.λstar

    ref = build_winner_ref(x_free0, ctx)   # throws TiedWinnerError on ties, unmodified

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
        wo = ref.winner[ω, d]
        d1w = d + (wo - 1) * D
        pTσ_wo = pTσ_from_score(ref.sw[ω, d], σ)
        contrib0[ω, d] = (SW[ω] / gammafac) * (CONST_d[d] + λstar[d1w] * pTσ_wo)
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
        maxerr < 1e-8 || error("build_lfix_base_cache_C: self-validation FAILED, max|q0_true-q0_cache|=$maxerr")
    end

    return LFixBaseCacheC(D, oci, W, μ, σ, bi, gammafac, SW, denom, CONST_d, ref, contrib0,
        λstar, base.ζstar, q0, wPrime_bi, τPrime_bi, LPrime_bi, Uσ_bi, λ_cf, cf_contrib0)
end

function cf_contrib_at(cache::LFixBaseCacheC, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    raw_cf = constConsσ_bibi ./ cache.Uσ_bi .- denom_cf
    return cache.λ_cf .* (raw_cf ./ cache.gammafac .* cache.SW)
end

function gamma_component_analytic(cache::LFixBaseCacheC, base::BaseDualState, g::Float64)
    σ = cache.σ
    mean_mSW = dot(base.m_star, cache.SW) / cache.W
    return -cache.λ_cf * σ * g^(σ - 1) * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi) / cache.gammafac * mean_mSW
end

"Backend C twin of `dest_contrib_incremental` (Tier 2, O(D) rescan fallback, >2 changed origins): logCC/mulU column rescan, no stored tensor."
function dest_contrib_incremental_generic_C(cache::LFixBaseCacheC, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W; σ = cache.σ
    ref = cache.ref
    _, logCC′, _ = constCons_matrix(θ_full, ctx)
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        bo = 1; bs = logCC′[1, d] + ref.mulU[ω, 1]
        for o in 2:D
            v = logCC′[o, d] + ref.mulU[ω, o]
            v < bs && (bs = v; bo = o)
        end
        pTσ_wo = pTσ_from_score(bs, σ)
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"""
    dest_contrib_incremental_top3_C(cache, ctx, θ_full, d, changed_origins) -> Vector{W}

Backend C's main tier: EXACT top-3-cache-based update mirroring `dest_contrib_incremental_top3`
(lfix_incremental.jl) / `dest_contrib_incremental_top3_B` (lfix_pTsigma_only.jl) structurally,
but sourcing winner/runnerup/third from `cache.ref` (log-price SCORES, ascending/argmin
convention -- SAME direction as the raw-price Reference, unlike Backend B's flipped-argmax
pTσ convention, since log-price preserves the original argmin ordering) and reconstructing pTσ
on demand at the resolved winner (`pTσ_from_score`) rather than indexing a dense tensor.

Deliberately does NOT call `winner_certificate.jl::coord_winner_update!` even though that
function implements the identical proof -- `coord_winner_update!`'s signature
`copyto!(winner_out, ref.winner)`s the FULL W×D winner matrix on every single call, which would
mean re-copying the entire winner matrix on every one of the ~2*(D^2-1) probes in a full
gradient call (unnecessary allocation/copy this backend's whole point is to avoid). This
function only ever writes the D affected column's W-length contribution vector, matching every
other backend's own per-destination scoping.
"""
function dest_contrib_incremental_top3_C(cache::LFixBaseCacheC, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) > 2 && return dest_contrib_incremental_generic_C(cache, ctx, θ_full, d, changed_origins)

    D = cache.D; W = cache.W; σ = cache.σ
    ref = cache.ref
    Cd = changed_origins
    _, logCC′, _ = constCons_matrix(θ_full, ctx)
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        r1 = ref.winner[ω, d]; r2 = ref.runnerup[ω, d]; r3 = ref.third[ω, d]
        best_o = 0; best_s = Inf
        if !(r1 in Cd)
            best_o = r1; best_s = ref.sw[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_s = ref.sr[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_s = ref.st3[ω, d]
        end
        bo = best_o; bs = best_s
        for o in Cd
            v = logCC′[o, d] + ref.mulU[ω, o]
            if v < bs
                bs = v; bo = o
            end
        end
        if bo == 0
            # extremely defensive: all of top-3 were changed (D<=3 & |Cd|>=3), unreachable for
            # |Cd|<=2 with D>=3 -- same defensive fallback every other tier's own top-3 path has.
            bo = 1; bs = logCC′[1, d] + ref.mulU[ω, 1]
            for o in 2:D
                v = logCC′[o, d] + ref.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
        end
        pTσ_wo = pTσ_from_score(bs, σ)
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"Backend C twin of `lfix_incremental_at`. Only :incremental_o1 (top-3 cache, the default tier a real gradient call uses) is implemented -- see this file's own docstring for scope."
function lfix_incremental_at_C(cache::LFixBaseCacheC, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64)
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
        origins_here = [o for (o, dd) in cells if dd == d]
        new_contrib = dest_contrib_incremental_top3_C(cache, ctx, θ_full, d, origins_here)
        q .-= new_contrib .- old_contrib
    end
    if cf_touched
        new_cf = cf_contrib_at(cache, θ_full, ctx)
        q .-= new_cf .- cache.cf_contrib0
    end

    return lfix_from_q(q, cache.ζstar)
end

# ----------------------------------------------------------------------------
# Bandwidth selection
# ----------------------------------------------------------------------------

"Backend C twin of `count_winner_flips_multi_top3` (flip COUNTING only, top-3-cache based)."
function count_winner_flips_C(cache::LFixBaseCacheC, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    ref = cache.ref
    Cd = changed_origins
    _, logCC′, _ = constCons_matrix(θ_full, ctx)
    flips = 0
    @inbounds for ω in 1:W
        r1 = ref.winner[ω, d]; r2 = ref.runnerup[ω, d]; r3 = ref.third[ω, d]
        best_o = 0; best_s = Inf
        if !(r1 in Cd)
            best_o = r1; best_s = ref.sw[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_s = ref.sr[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_s = ref.st3[ω, d]
        end
        bo = best_o; bs = best_s
        for o in Cd
            v = logCC′[o, d] + ref.mulU[ω, o]
            if v < bs
                bs = v; bo = o
            end
        end
        if bo == 0
            bo = 1; bs = logCC′[1, d] + ref.mulU[ω, 1]
            for o in 2:D
                v = logCC′[o, d] + ref.mulU[ω, o]
                v < bs && (bs = v; bo = o)
            end
        end
        flips += (bo != r1)
    end
    return flips
end

"Backend C twin of `select_bandwidth` (bisection over count_winner_flips_C)."
function select_bandwidth_C(cache::LFixBaseCacheC, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6)
    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth_C: coord_idx=$coord_idx has no affected A_od cells"
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += count_winner_flips_C(cache, ctx, θ_full, d, origins_here)
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

"Backend C twin of `a_block_fd_component`, including the AUD-12 nonfinite-both-sides retry discipline."
function a_block_fd_component_C(cache::LFixBaseCacheC, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; max_h_shrinks::Int = 4)
    h_try = h
    for attempt in 1:(max_h_shrinks + 1)
        Lp = lfix_incremental_at_C(cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h_try)
        Lm = lfix_incremental_at_C(cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h_try)
        if isfinite(Lp) && isfinite(Lm)
            return (Lp - Lm) / (2h_try)
        elseif isfinite(Lp) || isfinite(Lm)
            L0 = lfix_incremental_at_C(cache, ctx, pe, w0, coord_idx, w0[coord_idx])
            isfinite(L0) || break
            return isfinite(Lp) ? (Lp - L0) / h_try : (L0 - Lm) / h_try
        end
        h_try /= 4
    end
    return NaN
end

"Backend C twin of `composite_gradient_at` -- THE main entry point, no price0/pTσ0 tensor ever built."
function composite_gradient_at_C(x_free0::AbstractVector, ctx, pe; base::Union{Nothing,BaseDualState} = nothing)
    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache_C(x_free0, ctx, base)
    D = ctx.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])

    h_used = zeros(D2); switch_mass = zeros(D2)
    for k in 2:D2
        h, m, _ = select_bandwidth_C(cache, ctx, pe, w0, k)
        h_used[k] = h; switch_mass[k] = m
        g[k] = a_block_fd_component_C(cache, ctx, pe, w0, k, h)
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               winner0 = copy(cache.ref.winner), gamma_component = g[1])
end
