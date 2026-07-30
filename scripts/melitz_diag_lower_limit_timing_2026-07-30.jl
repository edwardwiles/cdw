# Follow-up to melitz_diag_lower_limit_investigation_2026-07-30.jl: the first diagnostic
# confirmed obj.lower_limit is correctly set to -cap at every checkpoint and that
# NumericalFailure trials return raw KNITRO nStatus=-300 (a genuine dual-side status, not a
# sentinel -- the -1e10 "objective" IS a codebase-level sentinel, documented in cc_bundle.jl/
# finite_delta_outer.jl, deliberately masking the unreliable failed-solve objective, separate
# from nStatus which passes through unchanged). This script adds WALL-CLOCK timing per
# re-solve (inner_loop_opt has outlev=0, so no visible KNITRO log either way) to check whether
# KNITRO detects nStatus=-300 quickly (healthy -- a fast, confident dual-infeasibility/
# unboundedness read) or grinds toward its full maxit=10000/maxtime_real=90s budget first
# (would suggest the early-stop mechanism is not engaging promptly).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf

CAP = 10.0
policy = CappedEvaluation(CAP)
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=policy, backend=:matrix_free, forbid_dense_fallback=true)
ctx = obj.γ
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")

result = melitz_run_reduced_q_sequential_search(ctx, obj, theta0; delta=0.5, direction=:upper,
    policy=policy, n_stages_max=5, max_iterations_per_stage=40, max_seconds_per_stage=120.0,
    outer_loop_opt=outer_opt)
all_trials = vcat([s.trials for s in result.stages]...)
fail_trials = filter(t -> t.classification == :NumericalFailure, all_trials)
println("n_fail=", length(fail_trials)); flush(stdout)

session = MelitzInnerSession(obj, ctx, policy)
println("\nTiming each NumericalFailure re-solve directly (raw melitz_bundle_inner_solve!, outlev=0):")
times = Float64[]
for (i, t) in enumerate(fail_trials[1:min(20, length(fail_trials))])
    melitz_bundle_prepare_at_theta!(obj, t.theta_full)
    t0 = time()
    val, x, nStatus = melitz_bundle_inner_solve!(obj, t.theta_full)
    elapsed = time() - t0
    push!(times, elapsed)
    @printf("  trial %2d: nStatus=%d  elapsed=%.4fs  ||x||=%.4e\n", i, nStatus, elapsed, norm(x))
    flush(stdout)
end
@printf("\nSummary over %d re-solved failures: median=%.4fs  min=%.4fs  max=%.4fs  mean=%.4fs\n",
        length(times), sort(times)[cld(length(times),2)], minimum(times), maximum(times), sum(times)/length(times))
println("(inner policy budget: maxit=10000, maxtime_real=90s -- for reference)")
flush(stdout)
