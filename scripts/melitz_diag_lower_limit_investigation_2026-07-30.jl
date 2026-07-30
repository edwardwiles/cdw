# Diagnostic (2026-07-30): investigate whether the reduced-q backend's high NumericalFailure
# counts (91/46/157/111 in the corrected Phase 12 CSV; substantial per-stage counts in Gate 2's
# own Method B rows) are caused by a silently-mis-propagated `lower_limit` (a documented,
# repeatedly-recurring bug class in this codebase per inner_screening.jl's own header
# comments -- "a genuinely InfiniteDeltaCertified real-D20 point... because the requested cap
# never propagated into obj.lower_limit at bundle-construction time"), as flagged by the user.
#
# Method: run the EXACT same sequential reduced-q search as Gate 2's own Method B, extract real
# :NumericalFailure trial points (theta_full), then RE-SOLVE each one directly via the public
# `solve_melitz_delta!` API with an `on_result` hook, printing `obj.lower_limit`,
# `melitz_policy_cap(session.policy)`, the raw KNITRO `nStatus`, and the raw objective value at
# the final iterate -- direct empirical evidence, not code-reading inference alone.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf
println("Julia threads: ", Threads.nthreads()); flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=policy, backend=:matrix_free, forbid_dense_fallback=true)
ctx = obj.γ

@printf("Bundle-construction check: obj.lower_limit=%.6f  melitz_policy_lower_limit(policy)=%.6f  MATCH=%s\n",
        obj.lower_limit, melitz_policy_lower_limit(policy), obj.lower_limit == melitz_policy_lower_limit(policy))
flush(stdout)

outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")
result = melitz_run_reduced_q_sequential_search(ctx, obj, theta0; delta=0.5, direction=:upper,
    policy=policy, n_stages_max=5, max_iterations_per_stage=40, max_seconds_per_stage=120.0,
    outer_loop_opt=outer_opt)

println("\nSearch complete. stopped_reason=", result.stopped_reason, "  n_stages=", length(result.stages))
all_trials = vcat([s.trials for s in result.stages]...)
n_fail = count(t -> t.classification == :NumericalFailure, all_trials)
n_above = count(t -> t.classification == :AboveEvaluationCap, all_trials)
n_finite = count(t -> t.classification == :FiniteSolved, all_trials)
@printf("Total trials=%d  FiniteSolved=%d  AboveEvaluationCap=%d  NumericalFailure=%d\n",
        length(all_trials), n_finite, n_above, n_fail)
flush(stdout)

fail_trials = filter(t -> t.classification == :NumericalFailure, all_trials)
println("\n" * "="^100)
println("Re-solving ", min(5, length(fail_trials)), " real NumericalFailure trial points directly, with instrumentation:")
println("="^100)

# Build a FRESH session (mirrors exactly what melitz_run_reduced_q_sequential_search itself
# builds internally) so this diagnostic uses the identical policy/obj pairing.
session = MelitzInnerSession(obj, ctx, policy)
@printf("Session check: session.obj.lower_limit=%.6f  session.policy=%s  melitz_policy_cap(session.policy)=%.6f\n",
        session.obj.lower_limit, session.policy, melitz_policy_cap(session.policy))
flush(stdout)

for (i, t) in enumerate(fail_trials[1:min(5, length(fail_trials))])
    theta_full = t.theta_full
    println("\n--- Re-solve attempt $i (original classification: $(t.classification)) ---")
    @printf("  Pre-solve: session.obj.lower_limit=%.6f\n", session.obj.lower_limit)
    crossing_hits = Ref(0)
    result2 = solve_melitz_delta!(session, theta_full, policy; on_result=(th, r) -> begin
        if r isa AboveEvaluationCap
            crossing_hits[] += 1
        end
    end)
    @printf("  Re-solve classification: %s\n", nameof(typeof(result2)))
    @printf("  Post-solve: session.obj.lower_limit=%.6f (should be UNCHANGED)\n", session.obj.lower_limit)
    if result2 isa NumericalFailure
        @printf("  nStatus=%d\n", result2.nStatus)
    elseif result2 isa AboveEvaluationCap
        @printf("  certified_lower_bound=%.6e  source=%s\n", result2.certified_lower_bound, result2.source)
    elseif result2 isa FiniteSolved
        @printf("  Delta=%.6e  nStatus=%d\n", result2.Delta, result2.nStatus)
    end
    flush(stdout)
end

println("\n" * "="^100)
println("Direct low-level check: does KNITRO's own inner solve actually RESPECT the configured")
println("lower_limit early-stop for a KNOWN failing point, i.e. does obj.threshold_crossed[] ever")
println("fire true for one of these theta values, or does it ALWAYS stay false (which would be")
println("the smoking gun the user is worried about)?")
println("="^100)
for (i, t) in enumerate(fail_trials[1:min(5, length(fail_trials))])
    theta_full = t.theta_full
    melitz_bundle_prepare_at_theta!(obj, theta_full)
    obj.threshold_crossed[] = false
    val, x, nStatus = melitz_bundle_inner_solve!(obj, theta_full)
    @printf("  trial %d: raw inner solve -> val=%.6e  nStatus=%d  threshold_crossed=%s  lower_limit=%.6f\n",
            i, val, nStatus, obj.threshold_crossed[], obj.lower_limit)
    flush(stdout)
end

println("\nDiagnostic complete.")
flush(stdout)
