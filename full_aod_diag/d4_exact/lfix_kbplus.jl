# ============================================================================
# Finalization task Phase 4 (2026-07-22): "Backend :kbplus" -- the reference-aligned,
# no-W-scale-exponentiation factorization requested in the finalization brief, tested FIRST
# per the brief's own instruction ("Preferred reference-aligned implementation to evaluate
# first"), before trusting Backend C+'s exp((1-sigma)*(logCC+mulU)) log-exp reconstruction.
#
# ALGEBRA (mapped onto this codebase's own trusted formulas, not the paper's raw notation --
# per the brief's own instruction to use the code's source of truth):
#
#   Reference (lfix_incremental.jl, price_and_pTsigma_cell!):
#     price = constCons_od / (U[:,o]^(-mu))
#     pTsigma = constConsSigma_od / (USigma[:,o]^(-mu))         (USigma = U.^(1-sigma), fixed
#                                                                 once at context-build time,
#                                                                 ctx.gamma.Usigma)
#
# This file computes EXACTLY that ratio, persistently: `USigmaPow[s,o] = USigma[s,o]^(-mu)` is
# O(W*D), built ONCE per context (mu is fixed data in the current parameterization -- read
# fresh from base.theta_full0 every build, never cached across a mu change, same discipline as
# every other backend's mu/sigma handling). `constConsSigma_od = constCons_od^(1-sigma)` is
# O(D^2), recomputed on every coordinate probe (cheap -- NOT W-scale) alongside the logCC
# ranking-score recompute every other factorized backend already does.
#
# WINNER RANKING is UNCHANGED from Backend C+: reuses `winner_certificate.jl`'s
# `WinnerRefCache`/`build_winner_ref` VERBATIM (log-score S = logCC + mulU, already exp-free
# for ranking/tie purposes -- see docs/fullA_price_tensor_audit.md sec 3-4). The ONLY
# difference from Backend C+ is the VALUE reconstruction at a resolved winner: C+ computes
# `exp((1-sigma)*S)` (one exp call per queried cell, i.e. O(W) exp calls per coordinate probe);
# this backend computes `constConsSigma[o,d] / USigmaPow[s,o]` (one division, zero
# exp/log/pow calls per queried cell, matching the brief's explicit "no W-scale exp, log, or ^
# in any coordinate probe" requirement).
#
# `cf_contrib_at`/`gamma_component_analytic` (the baseIndex/baseIndex "cf" column and the
# gamma-component-1 analytic derivative) are REUSED UNCHANGED from lfix_factorized.jl: both
# already use the identical ratio pattern (`constConsSigma_bibi ./ Uσ_bi`, `Uσ_bi` built once
# via a single W-length `.^(-mu)` at cache-build time, no exp anywhere) -- they were already
# "kbplus-shaped", nothing to change.
# ============================================================================
include(joinpath(@__DIR__, "lfix_factorized.jl"))   # reuses: constCons_matrix, WinnerRefCache,
# build_winner_ref, cf_contrib_at, gamma_component_analytic (all UNCHANGED), TiedWinnerError,
# affected_cells, lfix_from_q, aod_pow_cell, wPrime_bi_gdp -- transitively via lfix_incremental.jl

"pTσ reconstructed as a RATIO -- constConsσ_od / USigmaPow_so -- no exp/log/pow call, O(1) per queried cell."
@inline pTσ_from_ratio(constConsσ_od::Float64, USigmaPow_so::Float64) = constConsσ_od / USigmaPow_so

"""
    LFixBaseCacheKB

Backend :kbplus cache: identical to `LFixBaseCacheC` (same `ref::WinnerRefCache` ranking state)
plus the one new persistent array this backend needs: `USigmaPow` (W×D, `Uσ^(-μ)`, built once
per base point, read-only thereafter within that base point's lifetime).
"""
struct LFixBaseCacheKB
    D::Int; oci::Int; W::Int; μ::Float64; σ::Float64; baseIndex::Int
    gammafac::Float64
    SW::Vector{Float64}
    denom::Vector{Float64}
    CONST_d::Vector{Float64}
    ref::WinnerRefCache
    USigmaPow::Matrix{Float64}
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
Backend :kbplus twin of `build_lfix_base_cache_C`. Ranking scan (`build_winner_ref`) is
VERBATIM, unmodified -- only `contrib0`'s winner-value reconstruction differs (ratio, not exp).
"""
function build_lfix_base_cache_KB(x_free0::AbstractVector, ctx, base::BaseDualState; validate_dense::Bool = false)
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = γo.SamplingWeights[1:W]
    denom = [γo.wHat[d] * γo.L[d] for d in 1:D]
    λstar = base.λstar

    ref = build_winner_ref(x_free0, ctx)   # throws TiedWinnerError on ties, unmodified

    USigmaPow = γo.Uσ[1:W, :] .^ (-μ)   # O(W*D), once per base point -- the ONLY new persistent array

    constCons0, _, _ = constCons_matrix(base.θ_full0, ctx)   # O(D^2), negligible -- redundant with
    # build_winner_ref's own internal call (not exposed outward), re-derived here rather than
    # touching that already-validated function's signature.
    constConsσ0 = constCons0 .^ (1 - σ)   # O(D^2)

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
        pTσ_wo = pTσ_from_ratio(constConsσ0[wo, d], USigmaPow[ω, wo])
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
        maxerr < 1e-8 || error("build_lfix_base_cache_KB: self-validation FAILED, max|q0_true-q0_cache|=$maxerr")
    end

    return LFixBaseCacheKB(D, oci, W, μ, σ, bi, gammafac, SW, denom, CONST_d, ref, USigmaPow, contrib0,
        λstar, base.ζstar, q0, wPrime_bi, τPrime_bi, LPrime_bi, Uσ_bi, λ_cf, cf_contrib0)
end

"Backend :kbplus twin of `cf_contrib_at` -- UNCHANGED formula (already ratio-based, no exp anywhere), retyped for LFixBaseCacheKB."
function cf_contrib_at(cache::LFixBaseCacheKB, θ_full::AbstractVector, ctx)
    bi = cache.baseIndex; σ = cache.σ
    AodPow_bibi = aod_pow_cell(θ_full, ctx, bi, bi)
    γ_prime_bi = θ_full[3+ctx.D]
    constConsσ_bibi = cache.wPrime_bi^(1 - σ) * (AodPow_bibi * cache.τPrime_bi)^(1 - σ)
    denom_cf = γ_prime_bi^σ * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi)
    raw_cf = constConsσ_bibi ./ cache.Uσ_bi .- denom_cf
    return cache.λ_cf .* (raw_cf ./ cache.gammafac .* cache.SW)
end

"Backend :kbplus twin of `gamma_component_analytic` -- UNCHANGED formula, retyped for LFixBaseCacheKB."
function gamma_component_analytic(cache::LFixBaseCacheKB, base::BaseDualState, g::Float64)
    σ = cache.σ
    mean_mSW = dot(base.m_star, cache.SW) / cache.W
    return -cache.λ_cf * σ * g^(σ - 1) * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi) / cache.gammafac * mean_mSW
end

"Backend :kbplus twin of `dest_contrib_incremental_generic_C` (Tier 2, O(D) rescan fallback, >2 changed origins): ratio reconstruction, no exp."
function dest_contrib_incremental_generic_KB(cache::LFixBaseCacheKB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W; σ = cache.σ
    ref = cache.ref
    constCons′, logCC′, _ = constCons_matrix(θ_full, ctx)
    constConsσ′ = constCons′ .^ (1 - σ)   # O(D^2), negligible -- NOT W-scale
    contrib = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        bo = 1; bs = logCC′[1, d] + ref.mulU[ω, 1]
        for o in 2:D
            v = logCC′[o, d] + ref.mulU[ω, o]
            v < bs && (bs = v; bo = o)
        end
        pTσ_wo = pTσ_from_ratio(constConsσ′[bo, d], cache.USigmaPow[ω, bo])
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"""
    dest_contrib_incremental_top3_KB(cache, ctx, θ_full, d, changed_origins) -> Vector{W}

Backend :kbplus's main tier -- structurally IDENTICAL to `dest_contrib_incremental_top3_C`
(same top-3-cache case analysis, same winner-resolution loop over `changed_origins`), differing
ONLY in the final value-reconstruction line: `pTσ_from_ratio` (division) instead of
`pTσ_from_score` (exp call).
"""
function dest_contrib_incremental_top3_KB(cache::LFixBaseCacheKB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    length(changed_origins) > 2 && return dest_contrib_incremental_generic_KB(cache, ctx, θ_full, d, changed_origins)

    D = cache.D; W = cache.W; σ = cache.σ
    ref = cache.ref
    Cd = changed_origins
    constCons′, logCC′, _ = constCons_matrix(θ_full, ctx)
    constConsσ′ = constCons′ .^ (1 - σ)   # O(D^2), negligible -- NOT W-scale
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
            # Remediation task Part E (finding F5): canonical exact-tie convention --
            # lowest origin index wins (matches every generic-rescan tier's own natural
            # behavior, which scans o=1..D with strict `<`). Previously `if v < bs` alone,
            # which on an exact tie kept whichever candidate was considered FIRST (the
            # cached top-3 survivor, regardless of its index vs the tying changed origin's)
            # -- disagreed with the generic-rescan tiers whenever the survivor's index
            # exceeded the tying origin's. Measure-zero in exact arithmetic; see
            # test_winner_forced_tie.jl.
            if v < bs || (v == bs && o < bo)
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
        # bo is the resolved winner index; its constConsσ MAY be either a changed cell (bo in
        # Cd, use the freshly-recomputed constConsσ′) or an unchanged persistent cell (bo not in
        # Cd, still safe to read from constConsσ′ since constCons_matrix recomputes the FULL D×D
        # matrix every call -- O(D^2), not selectively updated -- exactly matching how logCC′
        # above is already handled by every other backend, including the trusted Reference).
        pTσ_wo = pTσ_from_ratio(constConsσ′[bo, d], cache.USigmaPow[ω, bo])
        d1w = d + (bo - 1) * D
        contrib[ω] = (cache.SW[ω] / cache.gammafac) * (cache.CONST_d[d] + cache.λstar[d1w] * pTσ_wo)
    end
    return contrib
end

"Backend :kbplus twin of `lfix_incremental_at_C`."
function lfix_incremental_at_KB(cache::LFixBaseCacheKB, ctx, pe, w0::AbstractVector, coord_idx::Int, new_val::Float64)
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
        new_contrib = dest_contrib_incremental_top3_KB(cache, ctx, θ_full, d, origins_here)
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

"Backend :kbplus twin of `count_winner_flips_C` (flip COUNTING only, top-3-cache based -- ranking is unaffected by the ratio-vs-exp change, so this is IDENTICAL to C+'s own, just retyped)."
function count_winner_flips_KB(cache::LFixBaseCacheKB, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
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
            # Remediation task Part E (finding F5): canonical exact-tie convention --
            # lowest origin index wins (matches every generic-rescan tier's own natural
            # behavior, which scans o=1..D with strict `<`). Previously `if v < bs` alone,
            # which on an exact tie kept whichever candidate was considered FIRST (the
            # cached top-3 survivor, regardless of its index vs the tying changed origin's)
            # -- disagreed with the generic-rescan tiers whenever the survivor's index
            # exceeded the tying origin's. Measure-zero in exact arithmetic; see
            # test_winner_forced_tie.jl.
            if v < bs || (v == bs && o < bo)
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

"Backend :kbplus twin of `select_bandwidth_C`."
function select_bandwidth_KB(cache::LFixBaseCacheKB, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6)
    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth_KB: coord_idx=$coord_idx has no affected A_od cells"
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += count_winner_flips_KB(cache, ctx, θ_full, d, origins_here)
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

"Backend :kbplus twin of `a_block_fd_component_C`, including the AUD-12 nonfinite-both-sides retry discipline."
function a_block_fd_component_KB(cache::LFixBaseCacheKB, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64; max_h_shrinks::Int = 4)
    h_try = h
    for attempt in 1:(max_h_shrinks + 1)
        Lp = lfix_incremental_at_KB(cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h_try)
        Lm = lfix_incremental_at_KB(cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h_try)
        if isfinite(Lp) && isfinite(Lm)
            return (Lp - Lm) / (2h_try)
        elseif isfinite(Lp) || isfinite(Lm)
            L0 = lfix_incremental_at_KB(cache, ctx, pe, w0, coord_idx, w0[coord_idx])
            isfinite(L0) || break
            return isfinite(Lp) ? (Lp - L0) / h_try : (L0 - Lm) / h_try
        end
        h_try /= 4
    end
    return NaN
end

"Backend :kbplus twin of `composite_gradient_at_C` -- THE main (allocating) entry point, no price0/pTσ0 tensor and no W-scale exp/log/pow anywhere."
function composite_gradient_at_KB(x_free0::AbstractVector, ctx, pe; base::Union{Nothing,BaseDualState} = nothing)
    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache_KB(x_free0, ctx, base)
    D = ctx.D; D2 = D^2
    z0 = log.(reshape(x_free0[2:end], D, D))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])

    h_used = zeros(D2); switch_mass = zeros(D2)
    for k in 2:D2
        h, m, _ = select_bandwidth_KB(cache, ctx, pe, w0, k)
        h_used[k] = h; switch_mass[k] = m
        g[k] = a_block_fd_component_KB(cache, ctx, pe, w0, k, h)
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               winner0 = copy(cache.ref.winner), gamma_component = g[1])
end
