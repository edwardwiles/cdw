# Sequential reduced-q-subspace outer-search backend, continued: Phase 6 (Hessian treatment),
# Phase 8 (sequential stage controller + its own small KNITRO NLP). Depends on
# `reduced_q_subspace.jl` (Phases 1-5/7), included immediately before this file.
#
# Deliberately a SEPARATE, SIMPLER KNITRO wiring than `finite_delta_outer.jl`'s own
# `solve_melitz_finite_delta_bound`/`melitz_build_finite_delta_callbacks` (no exact-point
# cache, no dual-polish/origin-block prescreens, no multi-candidate live tracking beyond one
# running best-incumbent) -- an experimental controller does not need production's full
# engineering sophistication to be a genuine, correct KNITRO NLP (Rule 11 only requires the
# PRODUCTION backend stay unchanged, which it does: zero lines of `finite_delta_outer.jl`
# are touched by this file). What IS reused, not reimplemented: the typed inner-solve API
# (`solve_melitz_delta!`/`MelitzInnerSession`, `inner_session.jl`), the classification types
# (`inner_screening.jl`), `evaluate_melitz_delta_from_solution`/`melitz_classify_outer_feasibility`
# (`delta_star.jl`/`finite_delta_outer.jl`), and every Phase 1-5/7 function above.

using LinearAlgebra: norm

# ============================================================================
# Phase 8 (objective direction): one tested comparator for "is candidate A a more extreme
# (better) verified bound than candidate B under this direction", replacing the informal,
# error-prone "higher/lower kappa is better" narrative that the governing prompt's own
# correction #2 found backwards for the :upper case in the prior session's Phase 12 report
# (docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md's own table analysis text claimed the
# experimental backend found a "better (higher) kappa" in both :upper runs -- but the upper
# GT% bound is obtained by MINIMIZING kappa, so a HIGHER kappa there is a WORSE, less extreme,
# solve, not a better one; the underlying KNITRO incumbent-selection logic in
# `solve_melitz_finite_delta_bound` was never wrong -- `signed_objective` already encodes this
# correctly -- only the DIAGNOSTIC narrative comparing kappa values after the fact was
# backwards for :upper).
# ============================================================================

"""
    melitz_reduced_q_more_extreme_kappa(direction::Symbol, kappa_a::Real, kappa_b::Real) -> Bool

`true` iff candidate `a` (with divergence-implied Frechet parameter `kappa_a`) is the MORE
EXTREME (better) verified bound than candidate `b` under `direction`:

  - `:upper` GT% bound: obtained by MINIMIZING kappa (thereby maximizing `100*(1-kappa)`) --
    `a` is better iff `kappa_a < kappa_b`.
  - `:lower` GT% bound: obtained by MAXIMIZING kappa (thereby minimizing `100*(1-kappa)`) --
    `a` is better iff `kappa_a > kappa_b`.

`melitz_reduced_q_more_extreme_gt` is the equivalent comparator directly on GT% values
(`100*(1-kappa)`), for a caller that already has GT% rather than kappa on hand -- both
comparators must and do agree at every point (regression test), since GT%=100*(1-kappa) is a
strictly decreasing affine function of kappa.
"""
function melitz_reduced_q_more_extreme_kappa(direction::Symbol, kappa_a::Real, kappa_b::Real)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower, got $direction"))
    return direction == :upper ? kappa_a < kappa_b : kappa_a > kappa_b
end

function melitz_reduced_q_more_extreme_gt(direction::Symbol, gt_a::Real, gt_b::Real)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower, got $direction"))
    return direction == :upper ? gt_a > gt_b : gt_a < gt_b
end

# ============================================================================
# Phase 8 record/result types.
# ============================================================================

"""
    MelitzReducedQTrialRecord

One KNITRO trial-point evaluation within one stage (Phase 9 instrumentation, the subset
captured directly by the reduced NLP's own callback -- the fuller per-trial instrumentation
table the governing prompt's Phase 9 also asks for (coordinatewise-assembled prediction,
one-sided fully-reoptimized change, screening/timing breakdown) is produced separately by the
D4 shakedown/comparison scripts, which call these same source functions directly with full
logging -- kept out of this hot KNITRO-callback-adjacent struct to avoid extra per-call cost
on a genuine bounded-runtime search).
"""
struct MelitzReducedQTrialRecord
    stage_id::Int
    x_reduced::Vector{Float64}
    theta_full::Vector{Float64}
    classification::Symbol   # :FiniteSolved / :AboveEvaluationCap / :InfiniteDeltaCertified / :NumericalFailure / :cap_screened
    Delta::Float64           # NaN unless :FiniteSolved
    signed_objective::Float64
    accepted_incumbent::Bool
end

"""
    MelitzReducedQStageResult

One stage's outcome: the `MelitzReducedQStage` geometry used, KNITRO's own terminal status,
per-classification counts (Rule 2: the four typed results, kept distinct; `n_cap_screened`
counts the Phase 7 screen's own safe exits separately, never folded into `n_above_cap` so the
screening-vs-real-solve split stays auditable), the full per-trial trace, the best verified
`FiniteSolved` incumbent found THIS stage (or `nothing`), and wall-clock time.
"""
struct MelitzReducedQStageResult
    stage::MelitzReducedQStage
    nStatus::Int
    n_finite_solved::Int
    n_above_cap::Int
    n_infinite_delta::Int
    n_numerical_failure::Int
    n_cap_screened::Int
    trials::Vector{MelitzReducedQTrialRecord}
    best_incumbent::Union{Nothing,NamedTuple}
    wall_s::Float64
    x_reduced_final::Vector{Float64}
end

# ============================================================================
# Phase 6/8: one bounded reduced-NLP KNITRO solve for a single stage.
# ============================================================================

"""
    melitz_solve_reduced_q_stage!(ctx, session, stage, x_reduced_init; delta, direction,
        theta_box_g=2.0, theta_box_A=2.0, max_iterations=60, max_seconds=180.0,
        outer_loop_opt=<default>, cap_screen=true, gamma_h=1e-6, s_crossing_target=50)
        -> MelitzReducedQStageResult

Phase 8 Step 3: registers and bounded-solves ONE reduced KNITRO NLP over
`x_reduced=(welfare,A_free...,s)` -- `s` the ONLY q-related variable KNITRO ever sees this
stage (Rule: "KNITRO should optimize jointly over (welfare coordinate, A_free, s), not over
all individual q coordinates").

Objective: `find_smallest ? x_reduced[1] : -x_reduced[1]` (`direction==:upper`
`find_smallest=true`), IDENTICAL in form to production's own `signed_objective`
(`finite_delta_outer.jl`) -- linear, registered together with the ONE divergence constraint
via a single eval callback (`cb_F!`/`cb_G!`), mirroring `melitz_register_finite_delta_knitro_problem!`'s
own `:linear` cutoff-backend branch.

Divergence constraint (`c[1] <= 1`, dimensionless): `DeltaStar(theta_full(x_reduced))/delta`,
via the typed `solve_melitz_delta!(session, theta_full, policy)` (Rule 2 -- the four
classifications are used AS-IS, never reimplemented). `FiniteSolved` reports the genuine
value/gradient; `AboveEvaluationCap`/`InfiniteDeltaCertified` report the SAME fixed sentinel
`cap/delta` with a ZERO gradient (mirrors production's own documented convention exactly,
`finite_delta_outer.jl`'s `melitz_build_finite_delta_callbacks` docstring); `NumericalFailure`
throws `DomainError` (an ordinary KNITRO eval-error, per Rule 3 -- NEVER reported/labeled
"infeasible"). `cap_screen=true` (default) applies the Phase 7 one-sided-safe fixed-dual
screen BEFORE calling `solve_melitz_delta!` at all, using the anchor's own verified dual as
`x_ref` -- a screened-out point is recorded identically to a real `AboveEvaluationCap` result
(same sentinel/gradient), just without the KNITRO-facing cost of a real inner solve.

Cutoff constraints (Phase 2): registered ONCE as true KNITRO linear rows
(`KN_add_con_linear_struct`, `C_r`/`b_r` from `melitz_reduced_affine_cutoff_system`) -- no
per-iterate cost, no callback involvement, exactly mirroring production's own `:linear`
cutoff-constraint-backend registration pattern.

Hessian (Phase 6): `hessopt=6` (KNITRO's limited-memory BFGS/L-BFGS quasi-Newton option) --
documented here as the ONE choice this experimental backend makes; no exact outer Hessian is
derived or supplied (per the governing prompt's own explicit instruction not to attempt one),
and every INNER Hessian remains exact/structured/matrix-free (Rule 8, untouched -- this
function never touches the inner CC dual solve's own Hessian machinery at all).

`theta_box_g`/`theta_box_A` are symmetric artificial boxes around `x_reduced_init`'s own
welfare/A values (mirrors production's own `theta_box` convention, Section 8.F-flagged as
non-economic); `s`'s own box is `stage.s_lo`/`stage.s_hi` (Phase 4's EXACT feasible interval),
further intersected with `[-1,1]` if the caller's `stage` was built with a wider `s_box`.
"""
function melitz_solve_reduced_q_stage!(ctx, session::MelitzInnerSession, stage::MelitzReducedQStage,
                                        x_reduced_init::AbstractVector{Float64};
                                        delta::Real, direction::Symbol,
                                        theta_box_g::Real=2.0, theta_box_A::Real=2.0,
                                        max_iterations::Int=60, max_seconds::Real=180.0,
                                        outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..",
                                            "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
                                        cap_screen::Bool=true, gamma_h::Real=1e-6,
                                        s_crossing_target::Integer=50)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower, got $direction"))
    melitz_reduced_q_check_ctx(ctx)
    obj = session.obj
    policy = session.policy
    find_smallest = direction == :upper
    D = ctx.D
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    length(x_reduced_init) == n_reduced || throw(ArgumentError(
        "melitz_solve_reduced_q_stage!: x_reduced_init must have length 2+nA=$n_reduced"))

    C_r, b_r, _ = melitz_reduced_affine_cutoff_system(stage, ctx)
    m_cut = size(C_r, 1)
    cap = melitz_policy_cap(policy)
    divergence_sentinel = cap / Float64(delta)

    trials = MelitzReducedQTrialRecord[]
    n_finite = Ref(0); n_above = Ref(0); n_inf = Ref(0); n_fail = Ref(0); n_screened = Ref(0)
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    signed_obj(x) = find_smallest ? x[1] : -x[1]

    # A currently-valid dual (the theta_init/stage-anchor's own verified optimum) -- used ONLY
    # as the FIXED reference dual for the Phase 7 screen; weak duality (see reduced_q_subspace.jl's
    # Phase 7 header) makes -obj(x_ref) a valid DeltaStar lower bound at ANY trial theta,
    # optimal or not.
    x_ref = copy(obj.x)

    function try_update_incumbent!(x_reduced, theta_full, Delta, x_dual, nStatus)
        r_eval = evaluate_melitz_delta_from_solution(theta_full, ctx, obj, Delta, x_dual, nStatus;
            full_equilibrium_check=false)
        cls = melitz_classify_outer_feasibility(r_eval, delta)
        cls.outer_feasible || return false
        cand = (x_reduced=copy(x_reduced), theta_full=copy(theta_full), Delta=Delta,
                objective=signed_obj(x_reduced), r=r_eval, x=copy(x_dual), nStatus=nStatus)
        if best[] === nothing || cand.objective < best[].objective
            best[] = cand
            return true
        end
        return false
    end

    # `inner_eval` is called from BOTH `cb_F!` and `cb_G!` (KNITRO's own `eval_fcga=no`
    # convention -- see finite_delta_outer.jl's own header comment -- calls the objective/
    # constraint callback and the gradient callback SEPARATELY at the same accepted trial
    # point for most outer iterates). It therefore must NOT increment the shared
    # classification counters or push to `trials` itself (that would double-count every
    # ordinary trial point, one increment from each callback) -- classification bookkeeping
    # happens ONLY in `cb_F!`, keyed off the SAME `kind` symbol this function returns, so
    # `trials` and the four typed counters plus `n_cap_screened` always sum to exactly
    # `length(trials)` (Phase 9 instrumentation invariant, regression-tested).
    function inner_eval(x_reduced::Vector{Float64})
        theta_full = melitz_reduced_full_theta(x_reduced, stage, ctx)
        if cap_screen
            lb = melitz_reduced_q_cap_screen(x_reduced, stage, ctx, obj, x_ref, cap)
            lb !== nothing && return theta_full, :cap_screened, NaN, nothing, -1
        end
        result = solve_melitz_delta!(session, theta_full, policy)
        if result isa FiniteSolved
            return theta_full, :FiniteSolved, result.Delta, result.x, result.nStatus
        elseif result isa AboveEvaluationCap
            return theta_full, :AboveEvaluationCap, NaN, nothing, -1
        elseif result isa InfiniteDeltaCertified
            return theta_full, :InfiniteDeltaCertified, NaN, nothing, -1
        else
            return theta_full, :NumericalFailure, NaN, nothing, result.nStatus
        end
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        x_reduced = collect(evalRequest.x)
        evalResult.obj[1] = signed_obj(x_reduced)
        theta_full, kind, Delta, x_dual, nStatus = inner_eval(x_reduced)
        if kind == :FiniteSolved
            n_finite[] += 1
            evalResult.c[1] = Delta / delta
            accepted = try_update_incumbent!(x_reduced, theta_full, Delta, x_dual, nStatus)
            push!(trials, MelitzReducedQTrialRecord(stage.stage_id, x_reduced, theta_full, kind, Delta,
                evalResult.obj[1], accepted))
            return 0
        elseif kind in (:AboveEvaluationCap, :InfiniteDeltaCertified, :cap_screened)
            kind == :AboveEvaluationCap ? (n_above[] += 1) :
            kind == :InfiniteDeltaCertified ? (n_inf[] += 1) : (n_screened[] += 1)
            evalResult.c[1] = divergence_sentinel
            push!(trials, MelitzReducedQTrialRecord(stage.stage_id, x_reduced, theta_full, kind, NaN,
                evalResult.obj[1], false))
            return 0
        else   # :NumericalFailure -- never labeled infeasible (Rule 3)
            n_fail[] += 1
            push!(trials, MelitzReducedQTrialRecord(stage.stage_id, x_reduced, theta_full, kind, NaN,
                evalResult.obj[1], false))
            throw(DomainError(x_reduced[1],
                "reduced-q outer callback: NumericalFailure -- inner CC dual solve returned " *
                "nStatus=$nStatus with no certificate obtained; rejecting this trial point, " *
                "no DeltaStar value invented (not called infeasible)."))
        end
    end

    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        x_reduced = collect(evalRequest.x)
        evalResult.objGrad .= 0.0
        evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        theta_full, kind, Delta, x_dual, nStatus = inner_eval(x_reduced)
        if kind == :FiniteSolved
            g_reduced = zeros(n_reduced)
            melitz_reduced_q_gradient!(g_reduced, x_reduced, stage, ctx, obj, x_dual;
                gamma_h=gamma_h, s_crossing_target=s_crossing_target)
            evalResult.jac[1:n_reduced] .= g_reduced ./ delta
        else
            evalResult.jac[1:n_reduced] .= 0.0
            kind == :NumericalFailure && throw(DomainError(x_reduced[1],
                "reduced-q outer gradient callback: NumericalFailure (not called infeasible)."))
        end
        return 0
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, outer_loop_opt)
    KNITRO.KN_set_int_param_by_name(kc, "hessopt", Cint(6))    # Phase 6: L-BFGS quasi-Newton
    KNITRO.KN_set_int_param_by_name(kc, "maxit", Cint(max_iterations))
    KNITRO.KN_set_double_param_by_name(kc, "maxtime_real", Float64(max_seconds))

    xIndices = melitz_kn_add_vars!(kc, n_reduced)
    lo = collect(Float64.(x_reduced_init))
    hi = collect(Float64.(x_reduced_init))
    lo[1] -= theta_box_g; hi[1] += theta_box_g
    lo[2:1+nA] .-= theta_box_A; hi[2:1+nA] .+= theta_box_A
    lo[end] = max(lo[end] - 1.0, stage.s_lo)
    hi[end] = min(hi[end] + 1.0, stage.s_hi)
    lo[end] = min(lo[end], x_reduced_init[end])   # never exclude the initial point itself
    hi[end] = max(hi[end], x_reduced_init[end])
    KNITRO.KN_set_var_lobnds_all(kc, lo)
    KNITRO.KN_set_var_upbnds_all(kc, hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, collect(Float64.(x_reduced_init)))

    cIndices = melitz_kn_add_cons!(kc, 1 + m_cut)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1.0)
    if m_cut > 0
        KNITRO.KN_set_con_lobnds(kc, m_cut, cIndices[2:end], -b_r)
        nnz = m_cut * n_reduced
        indexCons_lin = repeat(cIndices[2:end], inner=n_reduced)
        indexVars_lin = repeat(xIndices, outer=m_cut)
        coefs_lin = vec(permutedims(C_r))
        KNITRO.KN_add_con_linear_struct(kc, nnz, indexCons_lin, indexVars_lin, coefs_lin)
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, [cIndices[1]], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons=fill(cIndices[1], n_reduced), jacIndexVars=xIndices)

    t0 = time()
    KNITRO.KN_solve(kc)
    nStatus, _, x_final_raw, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)
    wall = time() - t0

    return MelitzReducedQStageResult(stage, Int(nStatus), n_finite[], n_above[], n_inf[], n_fail[],
        n_screened[], trials, best[], wall, collect(Float64.(x_final_raw)))
end

# ============================================================================
# Phase 8: the sequential stage controller.
# ============================================================================

"""
    MelitzReducedQSearchResult

The sequential search's overall outcome (Phase 8): the retained best verified incumbent
(`theta`/`Delta`/`x`/`objective`, argmin of `signed_objective` over the initial point and
every stage's own verified incumbent -- Rule 12, "every expanded search must retain the
verified restricted incumbent"), every stage's own `MelitzReducedQStageResult`, and whether
the loop stopped from the 2-consecutive-no-improvement rule or the hard 5-stage cap.
"""
struct MelitzReducedQSearchResult
    incumbent::NamedTuple
    stages::Vector{MelitzReducedQStageResult}
    stopped_reason::Symbol   # :no_improvement_streak / :max_stages / :no_direction_found
end

"""
    melitz_run_reduced_q_sequential_search(ctx, obj, theta_init; delta, direction,
        policy=CappedEvaluation(10.0), n_stages_max=5, bandwidth_policy=..., target_switches=100,
        s_crossing_target=50, theta_box_g=2.0, theta_box_A=2.0, max_iterations_per_stage=60,
        max_seconds_per_stage=180.0, outer_loop_opt=<default>, cap_screen=true) -> MelitzReducedQSearchResult

Phase 8: the full sequential reduced-q-subspace controller.

  1. Cold-verifies `theta_init` (must classify `FiniteSolved`) -- installed as the initial
     incumbent (Rule 12) BEFORE any stage runs, exactly mirroring production's own
     `initial_incumbent` convention.
  2. At each stage: builds ONE reduced-q direction at the current anchor
     (`melitz_build_reduced_q_stage`, Phases 3-4 -- retains the previous stage's own
     direction as an optional alternative via `previous_direction`); if no useful direction
     is found (Phase 3's own "do not invent a useful direction"), runs a welfare-plus-A-only
     stage instead (`s` pinned at `0.0`, zero-width box -- a genuine, still-typed-classified
     KNITRO solve, not skipped silently) and stops the search afterward (no further q
     directions to try).
  3. Runs one bounded reduced KNITRO solve (`melitz_solve_reduced_q_stage!`).
  4. Retains the best VERIFIED (Rule 6: `FiniteSolved`, within-budget, genuinely improving the
     relevant GT% bound per `melitz_reduced_q_more_extreme_kappa`) trial point as the new
     anchor if it improves on the running incumbent; otherwise keeps the incumbent unchanged.
  5. Stops after `n_stages_max` stages (hard cap, default 5) or 2 CONSECUTIVE stages with no
     verified improvement, whichever comes first -- never calls this "convergence" (that label
     is reserved for a genuine KNITRO-reported converged status backed by an independent
     first-order check, which this controller does not itself perform; `stopped_reason`
     records only the STOPPING RULE that fired, not a convergence claim).
"""
function melitz_run_reduced_q_sequential_search(ctx, obj, theta_init::AbstractVector;
                                                 delta::Real, direction::Symbol,
                                                 policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0),
                                                 n_stages_max::Int=5,
                                                 bandwidth_policy::MelitzQBandwidthPolicy=PowerScaledQBandwidth(1e-3, 80_000, 0.5),
                                                 target_switches::Integer=100, s_crossing_target::Integer=50,
                                                 theta_box_g::Real=2.0, theta_box_A::Real=2.0,
                                                 max_iterations_per_stage::Int=60, max_seconds_per_stage::Real=180.0,
                                                 outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..",
                                                     "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"),
                                                 cap_screen::Bool=true, dual_bank_max_size::Int=8)
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower, got $direction"))
    melitz_reduced_q_check_ctx(ctx)
    nA, nq, n_full, n_reduced = melitz_reduced_q_dim(ctx)
    find_smallest = direction == :upper
    signed_obj_full(theta) = find_smallest ? theta[1] : -theta[1]

    session = MelitzInnerSession(obj, ctx, policy; dual_bank_max_size=dual_bank_max_size)
    r0 = solve_melitz_delta!(session, collect(Float64.(theta_init)), policy)
    r0 isa FiniteSolved || throw(ArgumentError(
        "melitz_run_reduced_q_sequential_search: theta_init must classify FiniteSolved to " *
        "seed the search (Rule 12 requires a verified starting incumbent); got " *
        "$(nameof(typeof(r0)))."))

    incumbent = (theta=collect(Float64.(theta_init)), Delta=r0.Delta, x=copy(r0.x),
                 objective=signed_obj_full(theta_init))
    anchor = copy(incumbent.theta)
    anchor_x = copy(incumbent.x)

    stage_results = MelitzReducedQStageResult[]
    prev_direction = nothing
    no_improve_streak = 0
    stopped_reason = :max_stages

    for stage_id in 1:n_stages_max
        melitz_update_operator_at_theta!(obj.op, anchor, ctx)
        stage = melitz_build_reduced_q_stage(anchor, anchor_x, ctx, obj, stage_id;
            bandwidth_policy=bandwidth_policy, target_switches=target_switches,
            previous_direction=prev_direction)

        if stage === nothing
            # Phase 3/8 Step 6: no useful q direction -- run one welfare-plus-A-only stage
            # (s pinned at 0.0 via a zero-width box) instead of inventing a direction, then
            # stop (no further q movement is possible from this anchor without a new,
            # meaningfully nonzero, coordinatewise signal).
            dummy_stage = MelitzReducedQStage(zeros(nq), zeros(nq), stage_id, bandwidth_policy,
                copy(anchor), 0.0, 0.0, melitz_reduced_q_stage_fingerprint(stage_id, zeros(nq), zeros(nq),
                    bandwidth_policy, ctx, obj.op.W, nothing))
            x_reduced_init = vcat(anchor[1], anchor[2:1+nA], 0.0)
            stage_res = melitz_solve_reduced_q_stage!(ctx, session, dummy_stage, x_reduced_init;
                delta=delta, direction=direction, theta_box_g=theta_box_g, theta_box_A=theta_box_A,
                max_iterations=max_iterations_per_stage, max_seconds=max_seconds_per_stage,
                outer_loop_opt=outer_loop_opt, cap_screen=cap_screen, s_crossing_target=s_crossing_target)
            push!(stage_results, stage_res)
            if stage_res.best_incumbent !== nothing && stage_res.best_incumbent.objective < incumbent.objective
                cand = stage_res.best_incumbent
                incumbent = (theta=cand.theta_full, Delta=cand.Delta, x=cand.x, objective=cand.objective)
            end
            stopped_reason = :no_direction_found
            break
        end

        x_reduced_init = vcat(anchor[1], anchor[2:1+nA], 0.0)
        stage_res = melitz_solve_reduced_q_stage!(ctx, session, stage, x_reduced_init;
            delta=delta, direction=direction, theta_box_g=theta_box_g, theta_box_A=theta_box_A,
            max_iterations=max_iterations_per_stage, max_seconds=max_seconds_per_stage,
            outer_loop_opt=outer_loop_opt, cap_screen=cap_screen, s_crossing_target=s_crossing_target)
        push!(stage_results, stage_res)

        improved = false
        if stage_res.best_incumbent !== nothing
            cand = stage_res.best_incumbent
            if cand.objective < incumbent.objective
                incumbent = (theta=cand.theta_full, Delta=cand.Delta, x=cand.x, objective=cand.objective)
                anchor = copy(cand.theta_full)
                anchor_x = copy(cand.x)
                improved = true
            end
        end

        if improved
            no_improve_streak = 0
            prev_direction = copy(stage.q_basis_free ./ max(norm(stage.q_basis_free), 1e-300))
        else
            no_improve_streak += 1
            if no_improve_streak >= 2
                stopped_reason = :no_improvement_streak
                break
            end
        end
    end

    return MelitzReducedQSearchResult(incumbent, stage_results, stopped_reason)
end
