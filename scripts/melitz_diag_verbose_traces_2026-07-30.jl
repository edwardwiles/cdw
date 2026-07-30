# Direct response to user pushback (2026-07-30): show the ACTUAL KNITRO iteration-by-iteration
# trace for real NumericalFailure points, not aggregate stats. Temporarily swaps
# obj.inner_loop_opt to a verbose (outlev=4) copy of the SAME options file so we can see the
# raw KNITRO log for the inner CC dual solve at 8 real failing theta_full points, restoring the
# original afterward. Also prints f/lower_limit/threshold_crossed state explicitly around each
# call so the iteration log can be read against the exact governing check
# (cc_algo/PsiObjectiveBundle.jl: `if f <= lower_limit ... else return f end`).

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

const VERBOSE_OPT = "/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/9e9fa36e-7ddf-4905-95c2-3de5708055d0/scratchpad/melitz_inner_verbose.opt"
saved_opt = obj.inner_loop_opt
obj.inner_loop_opt = VERBOSE_OPT

n_shown = min(8, length(fail_trials))
println("\n" * "#"^100)
println("# Showing FULL KNITRO iteration logs for $n_shown real NumericalFailure points (outlev=4)")
println("#"^100)
for (i, t) in enumerate(fail_trials[1:n_shown])
    theta_full = t.theta_full
    println("\n" * "="^100)
    println("TRIAL $i / theta_full[1:3]=", theta_full[1:3], " ...")
    println("="^100)
    obj.threshold_crossed[] = false
    melitz_bundle_prepare_at_theta!(obj, theta_full)
    val, x, nStatus = melitz_bundle_inner_solve!(obj, theta_full)
    @printf(">>> trial %d SUMMARY: nStatus=%d  raw_returned_val=%.4e  threshold_crossed=%s  lower_limit=%.4f  ||x_final||=%.4e\n",
            i, nStatus, val, obj.threshold_crossed[], obj.lower_limit, norm(x))
    flush(stdout)
end

obj.inner_loop_opt = saved_opt   # restore
println("\nDiagnostic complete. obj.inner_loop_opt restored to: ", obj.inner_loop_opt)
flush(stdout)
