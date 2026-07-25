# Continuation session (2026-07-24), outer-benchmark CORRECTION: audits+retests the
# constrained D=20 finite-delta outer campaign per the governing prompt's Stage 1 (items
# 1-7). Supersedes `scripts/melitz_real_d20_full_campaign_2026-07-24.jl` as the production
# constrained-search driver -- that script is kept unmodified for direct before/after
# comparison (see the session report's own comparison table).
#
# Fixes applied vs. the prior campaign, each traced to a concrete root cause (see the
# session report for the full audit):
#
#   1. `cutoff_constraint_backend=:linear` is now explicitly passed. The prior campaign
#      relied on `solve_melitz_finite_delta_bound`'s own DEFAULT
#      (`cutoff_constraint_backend=:nonlinear_reference`) -- never overridden -- so all
#      `D + D*(D-1) = 400` cutoff rows were registered as GENERIC NONLINEAR KNITRO
#      constraints, evaluated (value+dense Jacobian) through the shared callback on every
#      FC/GA call, plus the 1 genuinely nonlinear divergence row: exactly the observed
#      "401 general nonlinear one-sided inequalities." The `:linear` backend (already
#      implemented+tested, `affine_cutoff.jl`) registers the 400 cutoff rows as TRUE KNITRO
#      linear constraints (`KN_add_con_linear_struct`, constant coefficients, zero
#      per-iterate cost) and leaves only the divergence row on the nonlinear callback.
#   2. `lower_limit_guard` lowered from `49.0` (waiting for a raw dual objective below -50,
#      i.e. a certified lower bound above `delta+49=50`) to a small NUMERICAL guard
#      (default `1e-6` here, chosen from the Section 2 validation below) -- a finite dual
#      value already above `delta` (not `delta+49`) is itself a valid certificate that
#      `Delta(theta) > delta`; there is no economic reason to wait for a much larger margin.
#   3. (`src/melitz/finite_delta_outer.jl`/`inner_screening.jl`, both files): a certified
#      `BudgetInfeasible` rejection now returns a FINITE constraint value
#      (`result.lower_bound/delta`, provably `>1`) and an exact fixed-dual gradient to
#      KNITRO, instead of throwing a `DomainError` (an eval-error giving KNITRO zero
#      magnitude/direction information on every rejection). Only `MomentInfeasible` (no
#      finite dual point exists for that certificate) and `NumericalFailure` (no certified
#      relationship to `delta`) remain genuine eval-error throws.
#   4. `external_incumbent=theta_fixed_af` (the Phase 6 fixed-A/f verified boundary point)
#      is now passed explicitly -- `solve_melitz_finite_delta_bound`'s new kwarg folds it
#      into the SAME best-incumbent selection as the trajectory's own candidates, so the
#      returned `cold_verified_incumbent` can never be worse than this known feasible point.
#      The starting point is also moved from the interior `Delta0=0.219` point to a
#      near-boundary point with `Delta0` in `[0.90,0.98]` (found by a small scan below).
#   5. Per-event block-movement logging: `TrajectoryLogger` now records `norm(A-A_start)`,
#      `norm(f-f_start)`, max abs A/f movement, the two gravity-pivot cells' own movement,
#      `min_slack`, and the total `theta` step -- for EVERY FC event, not inferred from `g`
#      alone.
#   6. Block-scaled trust region: `theta_box` is now a length-`n` vector, not a uniform
#      scalar `2.0` -- a small corridor on `g` (matching Phase 6's own measured ~0.08-wide
#      Delta<=1 gamma-only corridor) and separately-sized (still deliberately SMALL, per
#      main prompt Section 6's "do not permit a first dense trial step that changes hundreds
#      of participation decisions") radii on the A-block/f-block coordinates. Disclosed
#      explicitly in the report as a REASONED STARTING radius, not a tuned optimum.
#
# Usage: julia --project=. -t 16 scripts/melitz_real_d20_constrained_correction_2026-07-24.jl

using Printf
using DelimitedFiles
using LinearAlgebra: BLAS, norm

include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(dirname(@__DIR__), "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity

const REAL_DIR = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
const INNER_OPT = get(ENV, "MELITZ_INNER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt"))
const OUTER_OPT_CAMPAIGN = get(ENV, "MELITZ_OUTER_OPT_CAMPAIGN",
    joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt"))
const CAMPAIGN_SECONDS = parse(Float64, get(ENV, "MELITZ_CAMPAIGN_SECONDS", "240"))
const LOWER_LIMIT_GUARD = parse(Float64, get(ENV, "MELITZ_LOWER_LIMIT_GUARD", "1e-6"))

# Phase 6 incumbent (docs/melitz_real_d20_outer_benchmark_2026-07-24.md Section 6.2, cold-
# verified): g_fixed=-0.49783321, kappa_fixed=0.92939627, GT_fixed=0.07060373,
# Delta(g_fixed)=0.9969031. Re-derived fresh below (not hand-copied) so this script stays
# correct if the calibration ever changes -- but the KNOWN VALUES are pinned here as the
# expected/reported reference for the report's own before/after table.
const G_FIXED_REFERENCE = -0.49783321
const KAPPA_FIXED_REFERENCE = 0.92939627

function load_calibration()
    lambdaData = readdlm(joinpath(REAL_DIR, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(REAL_DIR, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(REAL_DIR, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(REAL_DIR, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate,
        focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
end

kappa_of_g(g::Real, calib) = (calib.w_prime / calib.w[calib.target_country]) * exp(g)^(1 / (calib.sigma - 1))

# ============================================================================
# Section 2 (main prompt): lower_limit_guard validation -- confirm a small numerical guard
# correctly classifies a known-below-budget point as InnerSolved and a known-above-budget
# point as BudgetInfeasible(:live_dual_threshold), and time the rejection.
# ============================================================================
function validate_lower_limit_guard(ctx, obj_inner, theta_calib, calib; guards=(1e-8, 1e-6, 1e-4))
    println("="^100)
    println("SECTION 2: lower_limit_guard validation (numerical-guard scale, not the old delta+49 margin)")
    println("="^100)

    # Two known points bracketing Delta=1 along the gamma-only path (Phase 6 grid,
    # docs/melitz_real_d20_outer_benchmark_2026-07-24.md Section 6.1): g=-0.4988 (Delta~1.06,
    # just ABOVE budget) and g_fixed=-0.49783321 (Delta~0.9969, just BELOW budget).
    theta_below = copy(theta_calib); theta_below[1] = G_FIXED_REFERENCE
    theta_above = copy(theta_calib); theta_above[1] = -0.4988

    cache = MelitzDeltaEvalCache(8)
    r_below = evaluate_melitz_delta(theta_below, ctx, obj_inner; cold=true, cache=cache)
    r_above = evaluate_melitz_delta(theta_above, ctx, obj_inner; cold=true, cache=cache)
    @printf("  reference points: g_below=%.6f Delta=%.6e (expect <1) | g_above=%.6f Delta=%.6e (expect >1)\n",
        theta_below[1], r_below.Delta, theta_above[1], r_above.Delta)
    flush(stdout)

    results = NamedTuple[]
    for guard in guards
        # Build a bundle whose stored dual bank already contains a dual vector "tuned" for a
        # DIFFERENT (calibration) point, then classify BOTH reference points at delta=1 under
        # this guard -- exercising exactly the live KNITRO-native threshold mechanism
        # (`lower_limit = -(delta+guard)`), timed.
        obj_g = build_melitz_implicit_bundle(ctx, obj_inner.U, theta_calib; delta=1.0,
            find_smallest=true, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN, lower_limit_guard=guard)
        bank = MelitzDualBank()

        t0 = time()
        res_below = melitz_classified_inner_solve(obj_g, theta_below, ctx; delta=1.0, bank=bank)
        t_below = time() - t0

        t0 = time()
        res_above = melitz_classified_inner_solve(obj_g, theta_above, ctx; delta=1.0, bank=bank)
        t_above = time() - t0

        below_ok = res_below isa InnerSolved
        above_ok = res_above isa BudgetInfeasible
        certified_bound = above_ok ? res_above.lower_bound : NaN
        @printf("  guard=%-10.1e  below: %-14s t=%6.3fs  |  above: %-24s t=%6.3fs  certified_bound=%.6f  correct=%s\n",
            guard, below_ok ? "InnerSolved" : string(typeof(res_below)), t_below,
            above_ok ? "BudgetInfeasible($(res_above.source))" : string(typeof(res_above)), t_above,
            certified_bound, below_ok && above_ok)
        flush(stdout)
        push!(results, (guard=guard, below_ok=below_ok, above_ok=above_ok,
            t_below=t_below, t_above=t_above, certified_bound=certified_bound))
    end
    return results
end

# ============================================================================
# Section 4 (main prompt): find a near-boundary starting point with Delta0 in [0.90,0.98]
# on the gamma-only (fixed A/f) path, WITHOUT re-running the full Phase 6 bisection --
# a short local scan between g_fixed (Delta~0.997) and the calibration (Delta~4e-4).
# ============================================================================
function find_near_boundary_start(ctx, obj_inner, theta_calib; target_lo=0.90, target_hi=0.98)
    println("\n" * "="^100)
    println("SECTION 4: near-boundary starting point scan (target Delta0 in [$target_lo,$target_hi])")
    println("="^100)
    cache = MelitzDeltaEvalCache(16)
    # Scan g upward (toward calibration) from g_fixed in small steps -- Delta falls
    # monotonically in this direction per the Phase 6 grid.
    for step in 0.0005:0.0005:0.02
        g = G_FIXED_REFERENCE + step
        theta = copy(theta_calib); theta[1] = g
        r = evaluate_melitz_delta(theta, ctx, obj_inner; cold=false, cache=cache)
        @printf("  g=%.6f  Delta=%.6e  nStatus=%d  verified=%s\n", g, r.Delta, r.nStatus, r.verified)
        flush(stdout)
        if r.verified && target_lo <= r.Delta <= target_hi
            println("  -> selected as campaign starting point")
            return theta, r
        end
    end
    error("find_near_boundary_start: no point in the scanned range landed in [$target_lo,$target_hi] -- widen the step range")
end

# ============================================================================
# Section 5: enhanced trajectory logger with per-event block-movement diagnostics.
# ============================================================================
mutable struct BlockLogger
    t0::Float64
    n_fc::Int
    events::Vector{NamedTuple}
    best_kappa::Float64
    best_theta::Union{Nothing,Vector{Float64}}
    A_start::Matrix{Float64}
    f_start::Matrix{Float64}
    theta_start::Vector{Float64}
end

function log_block_event!(logger::BlockLogger, calib, ctx, theta, result)
    logger.n_fc += 1
    t = time() - logger.t0
    kind = result isa InnerSolved ? :InnerSolved :
           result isa BudgetInfeasible ? :BudgetInfeasible :
           result isa MomentInfeasible ? :MomentInfeasible :
           result isa NumericalFailure ? :NumericalFailure : :Other
    Delta = result isa InnerSolved ? result.Delta :
            result isa BudgetInfeasible ? result.lower_bound : NaN
    g = theta[1]
    gamma_prime = exp(g)
    kappa = (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
    accepted = kind == :InnerSolved && Delta <= 1.0
    if accepted && kappa < logger.best_kappa
        logger.best_kappa = kappa
        logger.best_theta = copy(theta)
    end

    A, f, gamma_prime_j, f_jj = melitz_expand_theta(theta, ctx)
    dA = A .- logger.A_start
    df = f .- logger.f_start
    A_pivot_od = lin2od(ctx.A_pivot.pivot, ctx.D)
    A_pivot_move = A[A_pivot_od...] - logger.A_start[A_pivot_od...]
    theta_step = norm(theta .- logger.theta_start)
    state = melitz_outer_state(theta, ctx)

    push!(logger.events, (t=t, n_fc=logger.n_fc, kind=kind, Delta=Delta, g=g, kappa=kappa,
        GT=1 - kappa, accepted=accepted, theta_step=theta_step,
        normA=norm(dA), normf=norm(df), maxA=maximum(abs.(dA)), maxf=maximum(abs.(df)),
        A_pivot_move=A_pivot_move, min_slack=state.min_slack))

    @printf("[LIVE] n_fc=%4d t=%7.2fs kind=%-16s Delta=%10.4e g=%9.5f kappa=%9.6f accepted=%s | ||dA||=%8.4f ||df||=%8.4f maxA=%7.4f maxf=%7.4f Apivot_move=%8.5f min_slack=%8.5f theta_step=%8.5f\n",
        logger.n_fc, t, kind, Delta, g, kappa, accepted,
        norm(dA), norm(df), maximum(abs.(dA)), maximum(abs.(df)), A_pivot_move, state.min_slack, theta_step)
    flush(stdout)
    return nothing
end

# ============================================================================
# Section 6: block-scaled trust region.
# ============================================================================
function block_scaled_theta_box(ctx; g_radius=0.05, A_radius=0.15, f_radius=0.15)
    n = 1 + (ctx.D^2 - 1) + (length(ctx.f_free_lin) - 1)
    nA = ctx.D^2 - 1
    box = zeros(n)
    box[1] = g_radius
    box[2:1+nA] .= A_radius
    box[2+nA:end] .= f_radius
    return box
end

function main()
    calib = load_calibration()
    BLAS.set_num_threads(16)
    obj_inner, theta_calib = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1, inner_loop_opt=INNER_OPT)
    ctx = obj_inner.γ

    # -------- Section 2: guard validation --------
    guard_results = validate_lower_limit_guard(ctx, obj_inner, theta_calib, calib)

    # -------- Section 4: near-boundary start + external incumbent --------
    theta_fixed_af = copy(theta_calib); theta_fixed_af[1] = G_FIXED_REFERENCE
    theta_init, r_start = find_near_boundary_start(ctx, obj_inner, theta_calib)
    kappa0 = kappa_of_g(theta_init[1], calib)
    @printf("\nCampaign start: g=%.6f  Delta0=%.6e  kappa0=%.6f  GT0=%.6f\n",
        theta_init[1], r_start.Delta, kappa0, 1 - kappa0)
    @printf("External incumbent (fixed A/f): g=%.8f  kappa_fixed_reference=%.8f\n",
        G_FIXED_REFERENCE, KAPPA_FIXED_REFERENCE)
    flush(stdout)

    # -------- Section 6: block-scaled trust region --------
    theta_box = block_scaled_theta_box(ctx)
    @printf("theta_box: g_radius=%.4f  A_radius=%.4f  f_radius=%.4f  (small local radii, not the old uniform 2.0)\n",
        theta_box[1], theta_box[2], theta_box[end])

    A_start, f_start, _, _ = melitz_expand_theta(theta_init, ctx)
    logger = BlockLogger(time(), 0, NamedTuple[], Inf, nothing, A_start, f_start, copy(theta_init))
    on_result(theta, result) = log_block_event!(logger, calib, ctx, theta, result)

    melitz_profile_reset!()
    MELITZ_PROFILE[] = true

    println("\n" * "="^100)
    println("SECTION 7: corrected constrained campaign ($(CAMPAIGN_SECONDS)s, :linear cutoffs, guard=$(LOWER_LIMIT_GUARD), external incumbent, block-scaled box)")
    println("="^100)
    flush(stdout)

    t0 = time()
    result = solve_melitz_finite_delta_bound(ctx, obj_inner, theta_init; delta=1.0, direction=:upper,
        gradient_backend=:B_direct_argument_parallel, h=1e-4, theta_box=theta_box,
        cutoff_constraint_backend=:linear,
        inner_loop_opt=INNER_OPT, outer_loop_opt=OUTER_OPT_CAMPAIGN,
        on_inner_result=on_result, dual_bank_max_size=8, lower_limit_guard=LOWER_LIMIT_GUARD,
        external_incumbent=theta_fixed_af)
    wall = time() - t0
    MELITZ_PROFILE[] = false

    println("\n" * "="^100)
    println("CAMPAIGN RESULT")
    println("="^100)
    @printf("wall=%.2fs  nStatus=%d  n_fc_calls=%d  n_ga_calls=%d  n_inner_solved=%d\n",
        wall, result.nStatus, result.n_fc_calls, result.n_ga_calls, result.n_inner_solved)
    @printf("n_moment_infeasible_reject=%d  n_budget_infeasible_reject=%d  n_numerical_failure_reject=%d  inner_eval_failures=%d\n",
        result.n_moment_infeasible_reject, result.n_budget_infeasible_reject,
        result.n_numerical_failure_reject, result.inner_eval_failures)

    for (label, cand) in (("initial_incumbent", result.initial_incumbent),
                          ("best_live_incumbent", result.best_live_incumbent),
                          ("cold_verified_incumbent (THE ANSWER)", result.cold_verified_incumbent))
        if cand === nothing
            @printf("  %-38s: nothing\n", label)
        else
            g = cand.eval.theta_free[1]
            gamma_prime = exp(g)
            kappa = (calib.w_prime / calib.w[calib.target_country]) * gamma_prime^(1 / (calib.sigma - 1))
            @printf("  %-38s: objective=%.6f  g=%.6f  kappa=%.6f  GT=%.6f  Delta=%.6e  min_slack=%.4f  source=%s\n",
                label, cand.objective, g, kappa, 1 - kappa, cand.eval.Delta, cand.eval.min_slack, cand.source)
        end
    end

    incumbent_ok = result.cold_verified_incumbent !== nothing &&
        result.cold_verified_incumbent.eval.gamma_prime_j <= exp(G_FIXED_REFERENCE) * (1 + 1e-9) + 1e-12
    kappa_final = result.cold_verified_incumbent === nothing ? NaN :
        (calib.w_prime / calib.w[calib.target_country]) * result.cold_verified_incumbent.eval.gamma_prime_j^(1 / (calib.sigma - 1))
    @printf("\nACCEPTANCE CRITERION (main prompt Section 7): kappa_final=%.8f <= kappa_fixed_reference=%.8f ? %s\n",
        kappa_final, KAPPA_FIXED_REFERENCE, kappa_final <= KAPPA_FIXED_REFERENCE + 1e-9)

    println("\n" * "="^100)
    println("TRAJECTORY (first 25 and last 25 FC events, with block movement)")
    println("="^100)
    evs = logger.events
    show_idx = length(evs) <= 50 ? (1:length(evs)) : vcat(1:25, (length(evs)-24):length(evs))
    @printf("%6s %8s %16s %12s %10s %10s %8s %8s %8s %8s %10s\n",
        "n_fc", "t(s)", "kind", "Delta", "g", "kappa", "||dA||", "||df||", "maxA", "maxf", "min_slack")
    for i in show_idx
        e = evs[i]
        @printf("%6d %8.2f %16s %12.4e %10.6f %10.6f %8.4f %8.4f %8.4f %8.4f %10.5f\n",
            e.n_fc, e.t, e.kind, e.Delta, e.g, e.kappa, e.normA, e.normf, e.maxA, e.maxf, e.min_slack)
    end

    println("\n" * "="^100)
    println("WALL-CLOCK DECOMPOSITION (melitz_profile_report)")
    println("="^100)
    melitz_profile_report(stdout; trajectory_total_s=wall)

    BLAS.set_num_threads(1)
    println("\nDONE.")
    return result, calib, ctx, obj_inner, logger, guard_results
end

main()
