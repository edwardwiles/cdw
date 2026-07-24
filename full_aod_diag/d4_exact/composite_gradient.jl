# ============================================================================
# Continuation 4, Phase 3: composite hybrid gradient of Delta(w) in the
# reduced 16-dim (D=4) coordinate w = [gamma'_focal; z_free[1:D^2-1]].
#
# Two pieces, matching lfix_incremental.jl's own dependency-graph finding:
#   - gamma'_focal (w[1]): EXACT closed-form derivative of L_fix (no A_od
#     cell changes, no winner-switching -- derived below from cf_contrib_at's
#     own closed form, cross-validated against ForwardDiff.derivative and
#     central FD in test_composite_gradient.jl).
#   - A-block (w[2:end]): central FD over lfix_incremental_at(...;
#     tier=:incremental_o1), the O(1)-winner-update tier from Phase 2 --
#     O(1)-ish per probe, NO inner dual re-solve. Bandwidth h is chosen PER
#     COORDINATE by a documented adaptive selector (switching-mass target +
#     floor/ceiling + h-vs-h/2 slope-stability diagnostic), not a fixed
#     FIXED_H=0.01 (flagged as a real gap in docs/fullA_d4_final_report.md
#     sec 4 item 4 -- h-sensitivity matters a lot near the upper candidate).
#
# The composite gradient requires exactly ONE inner dual solve per outer
# iterate (to build the BaseDualState + LFixBaseCache at the CURRENT point --
# already effectively paid for by the objective/constraint evaluation at that
# same point), vs `eval_grad_central_fd`'s 2*n_free = 32 FULL inner re-solves.
# ============================================================================
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))
using LinearAlgebra: norm

# ----------------------------------------------------------------------------
# Gamma component: exact closed form
# ----------------------------------------------------------------------------
"""
    gamma_component_analytic(cache, base, g) -> Float64

Exact d(L_fix)/d(gamma'_focal) evaluated AT the cache's own base point
(g == w0[1]), derived by hand from `cf_contrib_at`'s closed form (the ONLY
piece of L_fix that depends on gamma'_focal -- see lfix_incremental.jl's
header). Chain of derivation (verify against ForwardDiff/central-FD in
test_composite_gradient.jl before trusting for anything beyond that check):

  cf_contrib_s(g) = lambda_cf * ( (constConsσ_bibi / Uσ_bi[s] - g^σ * wPrime_bi*LPrime_bi)
                                    / gammafac * SW[s] )
  -- constConsσ_bibi depends on AodPow[bi,bi] and mu, NEITHER of which change
     when only w[1]=g is perturbed (Aod cells untouched, mu fixed) -- so it's
     a CONSTANT w.r.t. g, cached once in `cache`.
  d(cf_contrib_s)/dg = -lambda_cf * sigma * g^(sigma-1) * wPrime_bi*LPrime_bi / gammafac * SW[s]

  q_s(g) = q0_s - (cf_contrib_s(g) - cf_contrib0_s)  =>  dq_s/dg = -d(cf_contrib_s)/dg

  L_fix(g) = -(mean_s Psi(q_s(g)) + zeta*)
  d(L_fix)/dg = -mean_s[ dPsi(q_s(g)) * dq_s/dg ]

At g == w0[1] (q(g)==q0 exactly), dPsi(q0_s) == base.m_star[s] BY CONSTRUCTION
(solve_base_state populates m_star as obj.arg1 = dPsi(arg0) at exactly this
q0 -- oracle.jl's own documented invariant, reused not re-derived). So:

  d(L_fix)/dg|_{w0[1]} = -mean_s[ m_star[s] * (-d(cf_contrib_s)/dg) ]
                       = mean_s[ m_star[s] * d(cf_contrib_s)/dg ]
                       = -lambda_cf*sigma*g^(sigma-1)*wPrime_bi*LPrime_bi/gammafac * mean_s[m_star[s]*SW[s]]

This is an EXACT local derivative (not a small-h approximation) -- the only
floating-point error is machine roundoff, no discretization error at all,
since the gamma coordinate is genuinely smooth (no MinInd! branching).
"""
function gamma_component_analytic(cache::LFixBaseCache, base::BaseDualState, g::Float64)
    σ = cache.σ
    mean_mSW = dot(base.m_star, cache.SW) / cache.W
    return -cache.λ_cf * σ * g^(σ - 1) * wPrime_bi_gdp(cache.wPrime_bi, cache.LPrime_bi) / cache.gammafac * mean_mSW
end

# ----------------------------------------------------------------------------
# A-block: adaptive bandwidth selector + central FD
# ----------------------------------------------------------------------------
"""
    count_winner_flips(cache, ctx, θ_full, d, changed_origins) -> Int

Counts how many of the W draws' cached winner at destination `d` would flip
under the given perturbed θ_full, restricted to the `changed_origins` (1 or 2
per the dependency graph). Uses the SAME `update_winner_o1` case analysis as
`dest_contrib_incremental_o1` (only the winner identity matters here, not the
downstream contribution value) -- reuses that exact logic rather than a
separate re-derivation.
"""
function count_winner_flips(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int}; multi_method::Symbol = :top3)
    if length(changed_origins) != 1
        multi_method === :top3 && return count_winner_flips_multi_top3(cache, ctx, θ_full, d, changed_origins)
        multi_method === :generic && return count_winner_flips_multi(cache, ctx, θ_full, d, changed_origins)
        error("count_winner_flips: multi_method must be :top3 or :generic, got $multi_method")
    end
    o = changed_origins[1]
    new_price, _ = price_and_pTsigma_cell(θ_full, ctx, o, d)
    flips = 0
    @inbounds for ω in 1:cache.W
        wo, _, _, _, _ = update_winner_o1(cache.winner_price0[ω, d], cache.winner0[ω, d],
                                            cache.runnerup_price0[ω, d], cache.runnerup0[ω, d],
                                            o, new_price[ω])
        flips += (wo != cache.winner0[ω, d])
    end
    return flips
end

"""
    count_winner_flips_multi(cache, ctx, θ_full, d, changed_origins) -> Int

ORIGINAL fallback for the two-changed-origins-in-one-destination case: full O(D)
rescan (correctness over speed, matching dest_contrib_incremental's own fallback
discipline). KEPT as the trusted reference / documented fallback per this
investigation's additive discipline -- `count_winner_flips`'s default now routes
to `count_winner_flips_multi_top3` instead (Continuation 8, see that function's
docstring); reach this original via `count_winner_flips(...; multi_method=:generic)`.
Exact equivalence between the two is verified (not assumed) in
`test_winner_top3_equivalence.jl`, including synthetic forced-2-changed-origin
cases beyond what a real D=4 run naturally exercises (per that file's own
docstring, this branch is RARE in a real run -- 3 of 15 A-block coordinates at
D=4, see `docs/winner_certificate_report.md` sec 2's audit).
"""
function count_winner_flips_multi(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    new_price = Dict{Int,Vector{Float64}}()
    for o in changed_origins
        p, _ = price_and_pTsigma_cell(θ_full, ctx, o, d)
        new_price[o] = p
    end
    col = Vector{Float64}(undef, D)
    flips = 0
    @inbounds for ω in 1:W
        for o in 1:D
            col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
        end
        _, wo, _ = min_and_secondmin(col)
        flips += (wo != cache.winner0[ω, d])
    end
    return flips
end

"""
    count_winner_flips_multi_top3(cache, ctx, θ_full, d, changed_origins) -> Int

Continuation 8 deliverable: O(1)-per-draw replacement for `count_winner_flips_multi`'s
O(D) rescan, using the cached top-3 ranking (`cache.winner0/runnerup0/third0` and
their price levels -- `lfix_incremental.jl`'s LFixBaseCache extended additively with
`third0`/`third_price0` this continuation, computed for free from the already-built
dense `price0` array) exactly the way `winner_certificate.jl::coord_winner_update!`
does for the analogous `WinnerRefCache`. EXACT (not approximate): with <=2 changed
origins the best surviving UNCHANGED origin is at worst rank 3 (same proof,
reused not re-derived -- see `coord_winner_update!`'s docstring). Only counts
FLIPS relative to `cache.winner0` (matching `count_winner_flips_multi`'s own
return contract exactly) -- verified bit-identical to `count_winner_flips_multi`
across a real coordinate sweep AND synthetic forced-2-changed-origin cases in
`test_winner_top3_equivalence.jl`.
"""
function count_winner_flips_multi_top3(cache::LFixBaseCache, ctx, θ_full::AbstractVector, d::Int, changed_origins::AbstractVector{Int})
    D = cache.D; W = cache.W
    length(changed_origins) > 2 && return count_winner_flips_multi(cache, ctx, θ_full, d, changed_origins)
    Cd = changed_origins
    new_price = Dict{Int,Vector{Float64}}()
    for o in Cd
        p, _ = price_and_pTsigma_cell(θ_full, ctx, o, d)
        new_price[o] = p
    end
    flips = 0
    @inbounds for ω in 1:W
        r1 = cache.winner0[ω, d]; r2 = cache.runnerup0[ω, d]; r3 = cache.third0[ω, d]
        best_o = 0; best_p = Inf
        if !(r1 in Cd)
            best_o = r1; best_p = cache.winner_price0[ω, d]
        elseif !(r2 in Cd)
            best_o = r2; best_p = cache.runnerup_price0[ω, d]
        elseif r3 != 0 && !(r3 in Cd)
            best_o = r3; best_p = cache.third_price0[ω, d]
        end
        bo = best_o; bp = best_p
        for o in Cd
            v = new_price[o][ω]
            # Remediation task Part E (finding F5): canonical exact-tie convention -- lowest
            # origin index wins, matching every generic-rescan tier. See lfix_factorized.jl's
            # identical fix for the full rationale.
            if v < bp || (v == bp && o < bo)
                bp = v; bo = o
            end
        end
        if bo == 0
            # extremely defensive: all of top-3 were changed (needs D<=3 & |Cd|>=3);
            # unreachable for |Cd|<=2 with D>=3, same as coord_winner_update!'s own
            # defensive branch. Full O(D) column rescan.
            col = Vector{Float64}(undef, D)
            for o in 1:D
                col[o] = haskey(new_price, o) ? new_price[o][ω] : cache.price0[ω, o, d]
            end
            _, bo, _ = min_and_secondmin(col)
        end
        flips += (bo != r1)
    end
    return flips
end

"""
    select_bandwidth(cache, ctx, pe, w0, coord_idx; kwargs...) -> (h, switch_mass, meta)

Adaptive bandwidth selector for A-block coordinate `coord_idx` (2:D^2 in the
reduced w vector), per continuation-4 task sec "A-block" requirement (not a
universal FIXED_H). Targets a SWITCHING-MASS fraction (fraction of the W
draws, across affected destinations, whose winner flips under a +h probe) in
`target_mass_frac = (lo, hi)`: too small means the probe stays inside one
smooth cell and would just reproduce Method-A's known-wrong
winner-boundary-dropping gradient (see derivative_methods.jl's
`method_A_pathwise_ad` docstring); too large means the secant averages over
so many kinks it stops being a local estimate. Simple geometric bisection
(doubling/halving) with an explicit floor/ceiling, matching
`docs/fullA_d4_final_report.md` sec 4 item 4's documented h=0.1 failure point
as the ceiling's justification. Also computes the h-vs-h/2 slope-stability
diagnostic the task requires (reported, not used to override the mass-based
choice -- two independent criteria, both logged).
"""
function select_bandwidth(cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int;
        h0::Float64 = 0.01, h_floor::Float64 = 1e-4, h_ceil::Float64 = 0.1,
        target_mass_frac::Tuple{Float64,Float64} = (0.003, 0.03), max_iter::Int = 6,
        multi_method::Symbol = :top3)

    cells = affected_cells(pe, coord_idx)
    @assert !isempty(cells) "select_bandwidth: coord_idx=$coord_idx has no affected A_od cells (gamma coordinate uses the analytic path, not this)"
    affected_dests = unique(last.(cells))

    function mass_at(h::Float64)
        w = copy(w0); w[coord_idx] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        total_flips = 0
        for d in affected_dests
            origins_here = [o for (o, dd) in cells if dd == d]
            total_flips += count_winner_flips(cache, ctx, θ_full, d, origins_here; multi_method = multi_method)
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
    a_block_fd_component(cache, ctx, pe, w0, coord_idx, h) -> Float64

Central FD of L_fix (via `lfix_incremental_at`, tier=:incremental_o1) at
coordinate `coord_idx`, step `h`. Falls back to a one-sided difference if one
probe's inner reconstruction is non-finite (mirrors
`run_d4_optimized_fd.jl::eval_grad_central_fd`'s own documented fallback
discipline for the SAME reason: a NaN silently propagating into a KNITRO
Jacobian callback is fatal, not merely wrong).

AUD-12 fix (twin of lfix_buffer_reuse.jl::a_block_fd_component!): both +/-h nonfinite no longer
silently returns 0.0. See that function's docstring for the full rationale -- geometric h
shrink-and-retry first, a one-sided secant if only one side is finite (checked at every h,
matching the original discipline before retries were added), NaN (explicit
derivative-unavailable, never a value indistinguishable from a genuine zero) only once every h
and the base value itself are exhausted.
"""
function a_block_fd_component(cache::LFixBaseCache, ctx, pe, w0::AbstractVector, coord_idx::Int, h::Float64;
        multi_method::Symbol = :top3, max_h_shrinks::Int = 4)
    h_try = h
    for attempt in 1:(max_h_shrinks + 1)
        Lp = lfix_incremental_at(cache, ctx, pe, w0, coord_idx, w0[coord_idx] + h_try; tier = :incremental_o1, multi_method = multi_method)
        Lm = lfix_incremental_at(cache, ctx, pe, w0, coord_idx, w0[coord_idx] - h_try; tier = :incremental_o1, multi_method = multi_method)
        if isfinite(Lp) && isfinite(Lm)
            return (Lp - Lm) / (2h_try)
        elseif isfinite(Lp) || isfinite(Lm)
            L0 = lfix_incremental_at(cache, ctx, pe, w0, coord_idx, w0[coord_idx]; tier = :incremental_o1, multi_method = multi_method)
            isfinite(L0) || break
            return isfinite(Lp) ? (Lp - L0) / h_try : (L0 - Lm) / h_try
        end
        h_try /= 4
    end
    return NaN
end

"""
    composite_gradient_at(x_free0, ctx, pe; base=nothing) -> (g, meta)

THE main entry point. Solves the inner dual at `x_free0` (unless `base` is
supplied, e.g. reused from the objective evaluation that just ran at the same
point), builds the LFixBaseCache, and returns the full n_free=D^2-dim
composite gradient in reduced w-coordinates: gamma component analytic, A-block
via adaptive-h central FD over the O(1)-incremental tier. `meta` carries the
per-coordinate chosen h, switching mass, h-vs-h/2 slope-stability ratio, and
the base's winner_hash (for the refresh policy's "large winner-hash change"
trigger).
"""
function composite_gradient_at(x_free0::AbstractVector, ctx, pe; base::Union{Nothing,BaseDualState} = nothing, multi_method::Symbol = :top3)
    base = base === nothing ? solve_base_state(x_free0, ctx) : base
    cache = build_lfix_base_cache(x_free0, ctx, base)
    D = ctx.D
    # Ddest (destination count) -- Ddest==D unless row_idx excludes ROW (Part A, 2026-07-23).
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    D2 = D * Ddest
    z0 = log.(reshape(x_free0[2:end], D, Ddest))
    w0 = vcat(x_free0[1], pivot_reduce(z0, pe))

    g = zeros(D2)
    g[1] = gamma_component_analytic(cache, base, w0[1])

    h_used = zeros(D2); switch_mass = zeros(D2); slope_ratio = fill(NaN, D2)
    for k in 2:D2
        h, m, selmeta = select_bandwidth(cache, ctx, pe, w0, k; multi_method = multi_method)
        h_used[k] = h; switch_mass[k] = m
        g[k] = a_block_fd_component(cache, ctx, pe, w0, k, h; multi_method = multi_method)
        g_half = a_block_fd_component(cache, ctx, pe, w0, k, h / 2; multi_method = multi_method)
        denom = max(abs(g[k]), abs(g_half), 1e-12)
        slope_ratio[k] = abs(g[k] - g_half) / denom
    end

    return g, (base = base, cache = cache, w0 = w0, h_used = h_used, switch_mass = switch_mass,
               slope_ratio = slope_ratio, winner0 = copy(cache.winner0), gamma_component = g[1])
end

# ----------------------------------------------------------------------------
# Refresh policy: wires derivative_methods.jl::should_refresh into a live
# decision loop (task requirement -- was "an existing, unwired policy
# function" before this continuation).
# ----------------------------------------------------------------------------
mutable struct HybridGradientPolicy
    refresh_every::Int
    gap_tol::Float64
    iters_since_refresh::Int
    last_optimized_slope::Float64      # ||A-block|| of the last EXPENSIVE gradient (NOT the gamma component -- see decide_gradient!'s docstring for why)
    last_winner0::Union{Nothing,Matrix{Int}}   # FULL cached winner matrix from the last gradient call
    winner_jump_frac::Float64          # trigger threshold: FRACTION of the W*D winner entries that must
                                        # differ from the last call before refreshing. NOTE: an earlier
                                        # version of this policy compared a single `hash(winner0)` for
                                        # exact equality -- at W=8000 draws, SOME winner somewhere flips on
                                        # essentially every outer step, so an exact-hash check would trigger
                                        # a refresh almost every call regardless of the configured
                                        # threshold (silently degrading "hybrid" to "always expensive",
                                        # defeating the entire point). Caught before any real run, not
                                        # after -- fixed to a genuine fractional-change comparison.
    last_rejected::Bool
    n_refresh::Int
    n_cheap::Int
    log::Vector{NamedTuple}
end

HybridGradientPolicy(; refresh_every::Int = 5, gap_tol::Float64 = 0.05, winner_jump_frac::Float64 = 0.02) =
    HybridGradientPolicy(refresh_every, gap_tol, 0, NaN, nothing, winner_jump_frac, false, 0, 0, NamedTuple[])

"""
    decide_gradient!(policy, x_free0, ctx, pe, eval_grad_expensive; base=nothing) -> (g, meta)

Live decision: on trigger (periodic / disagreement / winner-fraction jump --
`rejected_step` is a documented no-op this session, see below -- via
`should_refresh` plus the winner-fraction extension), calls the supplied
EXPENSIVE gradient function `eval_grad_expensive(w0)::Vector` (intended to be
`eval_grad_central_fd` from run_d4_optimized_fd.jl, i.e. the full
optimized-value Delta_FD gradient); otherwise computes the cheap composite
gradient. Exact hard value/feasibility (computed separately by the caller's
own `eval_F`) always governs acceptance -- this function only decides which
GRADIENT to hand KNITRO.

NOT wired this session (documented gap, not a silent omission): "rejected-step"
and "failed random-directional check" triggers from the task's suggested
list. `policy.last_rejected` exists but nothing ever sets it true -- reading
KNITRO's own accept/reject decision per callback would need a
`KN_set_newpoint_callback`, not implemented here given the time this
continuation had; the periodic + disagreement + winner-fraction-jump triggers
below are the three that are genuinely live.

DISAGREEMENT METRIC, IMPORTANT (found empirically this session, do not revert
without re-reading): `should_refresh`'s "cheap_slope vs last_optimized_slope"
comparison is fed the A-BLOCK GRADIENT NORM here, NOT the gamma component. A
first version used the gamma component (matching derivative_methods.jl's own
generic naming) and found it triggered an expensive refresh on EVERY SINGLE
call, with zero cheap calls ever taken, in both a synthetic multi-step test
and a real short KNITRO run -- not because the cheap gradient was wrong (the
gamma component is analytically EXACT, see gamma_component_analytic), but
because Delta_dual's own h=0.01 FD estimate of the gamma slope has such
severe curvature-driven bias (documented in this file's own commit message
and test_composite_gradient.jl) that it drifts by 15-40% between successive
KNITRO iterates even for tiny steps, near this candidate. Comparing an EXACT
cheap value against a NOISY expensive one on a coordinate where the two
disagree for reasons having nothing to do with staleness silently defeated
the entire point of "hybrid" (100% expensive calls, zero savings). The
A-block norm does not suffer the same gamma-specific bias and is a more
meaningful proxy for "has the cheap linearization drifted since the last
anchor" -- still an IMPERFECT proxy (both slopes are still evaluated at
different h -- composite's adaptive per-coordinate h vs the expensive
function's own fixed h=0.01), flagged as a real follow-up, not silently
assumed solved.
"""
function decide_gradient!(policy::HybridGradientPolicy, x_free0::AbstractVector, ctx, pe,
        eval_grad_expensive::Function; base::Union{Nothing,BaseDualState} = nothing)

    g_cheap, meta = composite_gradient_at(x_free0, ctx, pe; base = base)
    cheap_slope = norm(@view g_cheap[2:end])

    winner_jump_frac_actual = policy.last_winner0 === nothing ? 0.0 :
        count(meta.winner0 .!= policy.last_winner0) / length(meta.winner0)
    winner_jumped = policy.last_winner0 !== nothing && winner_jump_frac_actual > policy.winner_jump_frac

    (do_refresh, reason) = should_refresh(policy.iters_since_refresh, cheap_slope, policy.last_optimized_slope;
                                            refresh_every = policy.refresh_every, gap_tol = policy.gap_tol)
    if !do_refresh && policy.last_rejected
        do_refresh, reason = true, :rejected_step
    end
    if !do_refresh && winner_jumped
        do_refresh, reason = true, :winner_fraction_jump
    end
    if !do_refresh && isnan(policy.last_optimized_slope)
        do_refresh, reason = true, :first_call
    end

    if do_refresh
        g = eval_grad_expensive(meta.w0)
        policy.iters_since_refresh = 0
        policy.last_optimized_slope = norm(@view g[2:end])
        policy.n_refresh += 1
        push!(policy.log, (kind = :refresh, reason = reason, a_block_norm = policy.last_optimized_slope, cheap_slope = cheap_slope,
                            winner_jump_frac = winner_jump_frac_actual))
        policy.last_winner0 = meta.winner0
        policy.last_rejected = false
        return g, merge(meta, (gradient_source = :expensive, refresh_reason = reason))
    else
        policy.iters_since_refresh += 1
        policy.n_cheap += 1
        push!(policy.log, (kind = :cheap, reason = :none, a_block_norm = cheap_slope, cheap_slope = cheap_slope,
                            winner_jump_frac = winner_jump_frac_actual))
        policy.last_winner0 = meta.winner0
        return g_cheap, merge(meta, (gradient_source = :cheap, refresh_reason = :none))
    end
end
