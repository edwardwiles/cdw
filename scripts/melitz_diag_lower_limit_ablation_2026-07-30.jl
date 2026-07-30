# Direct ablation, per user request (2026-07-30): pull real NumericalFailure theta_full
# points out of an actual Method B run, then re-run the canonical Delta* recovery
# (melitz_recover_lfd) on EACH one under three different obj.lower_limit settings, directly
# mutating obj.lower_limit between runs (not inferring anything from logs) --
#   (a) lower_limit=-10   (current production default under CappedEvaluation(10.0))
#   (b) lower_limit=-Inf  (the check effectively OFF -- the exact failure mode the user is
#       asking about: does removing it make KNITRO take dramatically longer/more iterations?)
#   (c) lower_limit=-1    (a much TIGHTER threshold -- does our own check now fire BEFORE
#       KNITRO's native unboundedness detector, changing the outcome?)
# Reports iterations, wall time, nStatus, and threshold_crossed for each combination.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra

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

# Use a VERBOSE inner options file (outlev=4) so # of iterations is visible in KN_get_number_iters.
saved_opt = obj.inner_loop_opt
obj.inner_loop_opt = "/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/9e9fa36e-7ddf-4905-95c2-3de5708055d0/scratchpad/melitz_inner_verbose.opt"

n_show = min(8, length(fail_trials))
println("\n", rpad("trial",6), rpad("lower_limit",13), rpad("nStatus",10), rpad("elapsed_s",12), rpad("threshold_crossed",20), "final_obj")
println("-"^90)

for (i, t) in enumerate(fail_trials[1:n_show])
    theta_full = t.theta_full
    for ll in (-10.0, -Inf, -1.0)
        obj.lower_limit = ll
        obj.threshold_crossed[] = false
        melitz_bundle_prepare_at_theta!(obj, theta_full)
        t0 = time()
        val, x, nStatus = melitz_bundle_inner_solve!(obj, theta_full)
        elapsed = time() - t0
        @printf("%-6d%-13.4g%-10d%-12.5f%-20s%.4e\n", i, ll, nStatus, elapsed, string(obj.threshold_crossed[]), val)
        flush(stdout)
    end
    println()
end

obj.lower_limit = -10.0   # restore
obj.inner_loop_opt = saved_opt
println("Diagnostic complete.")
flush(stdout)
