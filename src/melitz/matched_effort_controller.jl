# Gate 2 matched-effort ablation controllers (2026-07-29 reduced-q validation session).
#
# Method A ("welfare plus exact A, q frozen") and Method C ("sequential production (A,f)")
# additive stage/restart controllers, matching Method B's own (`melitz_run_reduced_q_sequential_search`,
# `reduced_q_controller.jl`) stage/continuation/incumbent-retention conventions exactly, so a
# matched-effort comparison isolates "is the gain from reduced q specifically" rather than
# from continuation, exact A, or extra evaluations alone (governing prompt Gate 2).
#
# Method A reuses `melitz_solve_reduced_q_stage!` DIRECTLY (Rule 10: no second, script-only
# implementation) with a zero-width `s` box every stage (`s_lo=s_hi=0.0`) -- q is pinned at the
# anchor, so only the welfare coordinate and the exact-A block move. This is not a new KNITRO
# wiring: it is the EXISTING reduced-q stage solver with the q-direction disabled, isolating
# exactly "continuation + exact A" per the governing prompt's own Method A description.
#
# Method C wraps the EXISTING production `solve_melitz_finite_delta_bound` (`finite_delta_outer.jl`,
# UNTOUCHED) in the same stage-loop shape: continuation from the best verified incumbent,
# identical incumbent retention, same per-stage KNITRO `maxit`/`maxtime_real` budget, same
# stopping rules (max stages, 2-consecutive-no-improvement). Zero lines of `finite_delta_outer.jl`
# are modified.

"""
    melitz_run_welfare_plus_a_sequential_search(ctx, obj, theta_init; delta, direction,
        policy=CappedEvaluation(10.0), n_stages_max=5, theta_box_g=2.0, theta_box_A=2.0,
        max_iterations_per_stage=40, max_seconds_per_stage=120.0, outer_loop_opt=<default>,
        cap_screen=true) -> MelitzReducedQSearchResult

Gate 2 Method A. Identical stage/continuation/incumbent-retention/stopping-rule shape to
`melitz_run_reduced_q_sequential_search` (`reduced_q_controller.jl`), but the q-block is
PINNED at the anchor every stage (a zero-width `s` box, `s_lo=s_hi=0.0`) via
`melitz_solve_reduced_q_stage!` -- reused directly, not reimplemented. Isolates continuation +
exact-A-block gradient value alone, with NO q movement of any kind (not even the reduced
scalar `s`).
"""
function melitz_run_welfare_plus_a_sequential_search(ctx, obj, theta_init::AbstractVector;
                                                       delta::Real, direction::Symbol,
                                                       policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0),
                                                       n_stages_max::Int=5,
                                                       theta_box_g::Real=2.0, theta_box_A::Real=2.0,
                                                       max_iterations_per_stage::Int=40,
                                                       max_seconds_per_stage::Real=120.0,
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
        "melitz_run_welfare_plus_a_sequential_search: theta_init must classify FiniteSolved " *
        "to seed the search (Rule 12); got $(nameof(typeof(r0)))."))

    incumbent = (theta=collect(Float64.(theta_init)), Delta=r0.Delta, x=copy(r0.x),
                 objective=signed_obj_full(theta_init))
    anchor = copy(incumbent.theta)

    stage_results = MelitzReducedQStageResult[]
    no_improve_streak = 0
    stopped_reason = :max_stages
    dummy_policy = FixedRawQBandwidth(0.0)   # never used (s pinned at 0), only needed for the struct

    for stage_id in 1:n_stages_max
        melitz_update_operator_at_theta!(obj.op, anchor, ctx)
        dummy_stage = MelitzReducedQStage(zeros(nq), zeros(nq), stage_id, dummy_policy,
            copy(anchor), 0.0, 0.0, melitz_reduced_q_stage_fingerprint(stage_id, zeros(nq), zeros(nq),
                dummy_policy, ctx, obj.op.W, nothing))
        x_reduced_init = vcat(anchor[1], anchor[2:1+nA], 0.0)
        stage_res = melitz_solve_reduced_q_stage!(ctx, session, dummy_stage, x_reduced_init;
            delta=delta, direction=direction, theta_box_g=theta_box_g, theta_box_A=theta_box_A,
            max_iterations=max_iterations_per_stage, max_seconds=max_seconds_per_stage,
            outer_loop_opt=outer_loop_opt, cap_screen=cap_screen)
        push!(stage_results, stage_res)

        improved = false
        if stage_res.best_incumbent !== nothing
            cand = stage_res.best_incumbent
            if cand.objective < incumbent.objective
                incumbent = (theta=cand.theta_full, Delta=cand.Delta, x=cand.x, objective=cand.objective)
                anchor = copy(cand.theta_full)
                improved = true
            end
        end

        if improved
            no_improve_streak = 0
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

"""
    MelitzProductionStageResult

One Method-C stage's outcome, structurally mirroring `MelitzReducedQStageResult`'s own field
names/shape (`n_finite_solved`/`n_above_cap`/`n_infinite_delta`/`n_numerical_failure`/
`n_cap_screened`, always `0` here -- production `(A,f)` has no cap-screen concept) so
`melitz_typed_counters_from_reduced_q_stage` can be reused for reporting without a THIRD
counter-construction function.
"""
struct MelitzProductionStageResult
    stage_id::Int
    nStatus::Int
    n_finite_solved::Int
    n_above_cap::Int
    n_infinite_delta::Int
    n_numerical_failure::Int
    n_cap_screened::Int
    n_trials::Int              # res.inner_solve_count, independently recorded
    best_theta::Union{Nothing,Vector{Float64}}
    best_Delta::Float64
    best_objective::Float64
    wall_s::Float64
end

"""
    melitz_run_production_stage_sequential_search(ctx, obj, theta_init; delta, direction,
        policy=CappedEvaluation(10.0), n_stages_max=5, max_iterations_per_stage=40,
        max_seconds_per_stage=120.0, inner_loop_opt, outer_loop_opt=<default>) -> MelitzReducedQSearchResult

Gate 2 Method C: wraps the UNMODIFIED production `solve_melitz_finite_delta_bound`
(`finite_delta_outer.jl`, `gradient_backend=:auto`, `:logf`) in the SAME stage/continuation/
incumbent-retention/stopping-rule shape as Method A/B -- each stage is one independent KNITRO
run (`maxit=max_iterations_per_stage`, matching Method A/B's own per-stage KNITRO budget
exactly) started from `theta_init = <previous stage's best verified incumbent>`; the running
incumbent across stages is the best verified `FiniteSolved` point seen so far (never regresses,
mirroring Rule 12). Zero lines of `finite_delta_outer.jl` are touched.
"""
function melitz_run_production_stage_sequential_search(ctx, obj, theta_init::AbstractVector;
                                                         delta::Real, direction::Symbol,
                                                         policy::MelitzInnerSolvePolicy=CappedEvaluation(10.0),
                                                         n_stages_max::Int=5,
                                                         max_iterations_per_stage::Int=40,
                                                         max_seconds_per_stage::Real=120.0,
                                                         inner_loop_opt::AbstractString,
                                                         outer_loop_opt::AbstractString=joinpath(@__DIR__, "..", "..",
                                                             "melitz_outer_finite_delta_alg_direct_2026-07-27.opt"))
    direction in (:upper, :lower) || throw(ArgumentError("direction must be :upper or :lower, got $direction"))
    find_smallest = direction == :upper
    signed_obj_full(theta) = find_smallest ? theta[1] : -theta[1]

    session = MelitzInnerSession(obj, ctx, policy)
    r0 = solve_melitz_delta!(session, collect(Float64.(theta_init)), policy)
    r0 isa FiniteSolved || throw(ArgumentError(
        "melitz_run_production_stage_sequential_search: theta_init must classify FiniteSolved " *
        "to seed the search (Rule 12); got $(nameof(typeof(r0)))."))

    incumbent = (theta=collect(Float64.(theta_init)), Delta=r0.Delta, x=copy(r0.x),
                 objective=signed_obj_full(theta_init))
    anchor = copy(incumbent.theta)

    stage_results = MelitzProductionStageResult[]
    no_improve_streak = 0
    stopped_reason = :max_stages

    # A temporary per-stage maxit override file is avoided by passing an explicit small option
    # tweak via KNITRO's own `outer_loop_opt` file (already sets maxit via the file); instead we
    # rely on the SAME option file Method A/B use, and additionally cap `maxit` at the Julia
    # level is not exposed as a solve_melitz_finite_delta_bound kwarg -- the option FILE already
    # pins `maxit`/`maxtime_real`. To match Method A/B's own `max_iterations_per_stage`/
    # `max_seconds_per_stage` exactly without editing finite_delta_outer.jl, this controller
    # writes a temporary option file that is the SAME base file with `maxit`/`maxtime_real`
    # overridden -- disclosed, not silent.
    tmp_opt = melitz_matched_effort_option_file(outer_loop_opt, max_iterations_per_stage, max_seconds_per_stage)

    for stage_id in 1:n_stages_max
        t0 = time()
        res = solve_melitz_finite_delta_bound(ctx, obj, anchor; delta=delta, direction=direction,
            gradient_backend=:auto, backend=:matrix_free, policy=policy,
            inner_loop_opt=inner_loop_opt, outer_loop_opt=tmp_opt)
        wall = time() - t0

        counters = melitz_typed_counters_from_direct_result(res)
        inc = res.cold_verified_incumbent
        best_theta = inc === nothing ? nothing : inc.eval.theta_free
        best_Delta = inc === nothing ? NaN : inc.eval.Delta
        best_obj = inc === nothing ? Inf : signed_obj_full(inc.eval.theta_free)

        stage_res = MelitzProductionStageResult(stage_id, res.nStatus, counters.n_finite_solved,
            counters.n_above_cap_evaluated, counters.n_infinite_delta, counters.n_numerical_failure,
            0, res.inner_solve_count, best_theta, best_Delta, best_obj, wall)
        push!(stage_results, stage_res)

        improved = false
        if best_theta !== nothing && best_obj < incumbent.objective
            xdual = melitz_recover_lfd(obj, best_theta)
            incumbent = (theta=collect(Float64.(best_theta)), Delta=best_Delta, x=copy(xdual.dual_x), objective=best_obj)
            anchor = copy(incumbent.theta)
            improved = true
        end

        if improved
            no_improve_streak = 0
        else
            no_improve_streak += 1
            if no_improve_streak >= 2
                stopped_reason = :no_improvement_streak
                break
            end
        end
    end

    return MelitzReducedQSearchResult(incumbent, MelitzReducedQStageResult[], stopped_reason), stage_results
end

"""
    melitz_matched_effort_option_file(base_opt_path, maxit, maxtime_real) -> String

Writes a temporary KNITRO option file identical to `base_opt_path` except `maxit`/
`maxtime_real` are overridden to the matched-effort per-stage budget -- lets Method C match
Method A/B's own `max_iterations_per_stage`/`max_seconds_per_stage` exactly without editing
`finite_delta_outer.jl` (which has no direct `maxit` kwarg of its own; it is driven entirely by
`outer_loop_opt`, matching every other production caller's convention).

**Any pre-existing `maxit`/`maxtime_real` line in `base_opt_path` is DROPPED, not merely
shadowed by a later line.** An earlier version of this function appended the override lines
after a verbatim copy of the base file, relying on "KNITRO option files honor the last
occurrence of a duplicate key" -- `melitz_outer_finite_delta_alg_direct_2026-07-27.opt` (the
base file every caller uses) already sets `maxit 25`, so that version produced a file with TWO
`maxit` lines. This was live-diagnosed as the proximate trigger of a reproducible KNITRO-level
segfault (`KN_solve` -> `KTR_solve` -> `KN_load_qcqp` crashing inside `libknitro.so` itself,
confirmed reproducible in an isolated single-testset run, not a resource-accumulation artifact)
-- KNITRO's own option-file parser is not guaranteed safe against a duplicate key within one
`KN_load_param_file` call. Filtering the duplicate out entirely, rather than relying on
override-by-shadowing, removes the ambiguity structurally.
"""
function melitz_matched_effort_option_file(base_opt_path::AbstractString, maxit::Integer, maxtime_real::Real)
    tmp = tempname() * "_matched_effort.opt"
    is_overridden_key(line) = begin
        s = strip(line)
        isempty(s) && return false
        key = first(split(s))
        lowercase(key) in ("maxit", "maxtime_real")
    end
    open(tmp, "w") do io
        for line in eachline(base_opt_path)
            is_overridden_key(line) && continue
            println(io, line)
        end
        println(io, "maxit  ", maxit)
        println(io, "maxtime_real  ", maxtime_real)
    end
    return tmp
end
