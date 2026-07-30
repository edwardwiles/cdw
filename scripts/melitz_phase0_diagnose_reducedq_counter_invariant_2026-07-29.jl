# Phase 0 diagnostic: does the per-stage invariant
#   n_finite_solved + n_above_cap + n_infinite_delta + n_numerical_failure + n_cap_screened == length(trials)
# (asserted for a 1-stage smoke run by test/melitz/runtests.jl's own regression test) actually
# hold for EVERY stage of a genuine multi-stage Phase-12-style run? The original Phase 12 CSV's
# own aggregate n_accepted_steps (sum of length(trials) across stages) is systematically SMALLER
# than the aggregate classification sum across stages, by a consistent ~1.34-1.42x ratio in all
# four sequential_reduced_q rows -- reproducing (not merely resembling) the ratio the prior
# session's own regression test caught and reportedly fixed (34 trials-vs-counters mismatch on
# 25 real trials, 34/25=1.36). This script isolates ONE cell (delta=0.1, direction=:upper, the
# same fixture Phase 12 used) and checks the invariant stage-by-stage, live.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf
println("Julia threads: ", Threads.nthreads()); flush(stdout)

outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx = obj.γ

result = melitz_run_reduced_q_sequential_search(ctx, obj, theta0; delta=0.1, direction=:upper,
    policy=CappedEvaluation(10.0), n_stages_max=5, max_iterations_per_stage=40, max_seconds_per_stage=120.0,
    outer_loop_opt=outer_opt)

println("\nstopped_reason = ", result.stopped_reason)
println("n_stages = ", length(result.stages))
total_classified = 0
total_trials = 0
for (i, sr) in enumerate(result.stages)
    classified = sr.n_finite_solved + sr.n_above_cap + sr.n_infinite_delta + sr.n_numerical_failure + sr.n_cap_screened
    ntrials = length(sr.trials)
    global total_classified += classified
    global total_trials += ntrials
    ok = classified == ntrials
    @printf("  stage %d: n_finite=%d n_above=%d n_inf=%d n_fail=%d n_screened=%d  classified_sum=%d  length(trials)=%d  MATCH=%s\n",
            i, sr.n_finite_solved, sr.n_above_cap, sr.n_infinite_delta, sr.n_numerical_failure, sr.n_cap_screened,
            classified, ntrials, ok)
    # Also check: does trials itself contain duplicate x_reduced entries (would indicate KNITRO
    # re-invoking cb_F! for the same point, e.g. after a thrown NumericalFailure exception)?
    xs = [t.x_reduced for t in sr.trials]
    n_unique = length(unique(xs))
    @printf("    length(trials)=%d, n_unique(x_reduced)=%d, n_duplicate_rows=%d\n", ntrials, n_unique, ntrials-n_unique)
    kinds = [t.classification for t in sr.trials]
    for k in unique(kinds)
        cnt = count(==(k), kinds)
        @printf("    trials classification %-20s count=%d\n", string(k), cnt)
    end
end
@printf("\nTOTAL: classified_sum=%d  total_trials=%d  MATCH=%s  ratio=%.4f\n",
        total_classified, total_trials, total_classified==total_trials, total_classified/max(total_trials,1))
flush(stdout)
