# ============================================================================
# Phase 1 (continuation 3), CORRECTED re-profile. Two things changed vs the
# original profile_components.jl (continuation 2):
#
#   1. The Phase 1A/B/C fixes are applied (oracle_fast.jl: reused obj.H
#      instead of a second moments! call, allocation-free winner scan,
#      preallocated KKT-residual reduction) -- run AFTER those fixes, not
#      before, per the user's explicit request.
#   2. "inner_solve" is no longer reported as one opaque bucket. Per the
#      user's explicit correction: inner_loop_internal's own "inner_solve"
#      time is NOT "the CC multiplier optimization" -- it is a moment-matrix
#      BUILD (inner_moment_build, same kind of O(D^2 W) cost as the
#      now-eliminated redundant second moments! call) followed by the ACTUAL
#      dual optimization (inner_knitro_dual_solve), which is itself further
#      split into its two KNITRO callback types (inner_dual_fg_callback --
#      FUSED objective+gradient in one callback, cannot be split further
#      without patching cc_algo/PsiObjectiveBundle.jl's callable method,
#      which this investigation's additive-only discipline avoids --
#      inner_dual_hessian_callback) plus KNITRO's own un-instrumented SQP/
#      barrier overhead (reported as "inner_knitro_dual_solve (exclusive)",
#      derived as inclusive-total MINUS the two nested callback totals, not
#      double-counted into the top-level percentage breakdown).
#
# Runs BOTH the OLD (oracle_profiled.jl) and NEW (oracle_fast.jl) profilers
# back to back, same warm-up discipline, same N=50 reps, same calibration
# point and W=8000/D=4 setting as the original profile_components.jl, so the
# before/after comparison is apples-to-apples. Old raw numbers are preserved
# in profile_components.csv (continuation 2's own artifact, untouched);
# this script's own OUTDIR is separate and dated.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_profiled.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
using Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_components_v2")
mkpath(OUTDIR)
const N_REPS = 50

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

function stats_row(label, times; alloc = Float64[])
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted)/n
    σ = n > 1 ? sqrt(sum((t-μ)^2 for t in sorted)/(n-1)) : 0.0
    mab = isempty(alloc) ? NaN : sum(alloc)/length(alloc)
    tab = isempty(alloc) ? NaN : sum(alloc)
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            p90_s = sorted[clamp(ceil(Int, 0.9*n), 1, n)], p95_s = sorted[clamp(ceil(Int, 0.95*n), 1, n)],
            mean_s = μ, std_s = σ, mean_alloc_bytes = mab, total_alloc_bytes = tab, mean_gc_s = NaN, total_gc_s = NaN)
end

println("="^78); println("OLD PROFILER (oracle_profiled.jl, Phase 1 (continuation 2) code, UNCHANGED)"); println("="^78)
flush(stdout)
for warm in (true, false)
    evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = warm)
end
prof_reset!()
old_total_warm = Float64[]; old_total_cold = Float64[]
for rep in 1:N_REPS
    t0 = time_ns(); evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = true)
    push!(old_total_warm, (time_ns() - t0) / 1e9)
end
for rep in 1:N_REPS
    t0 = time_ns(); evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = false)
    push!(old_total_cold, (time_ns() - t0) / 1e9)
end
old_rows = prof_summary()
push!(old_rows, stats_row("TOTAL_evaluate_fullA_warm", old_total_warm))
push!(old_rows, stats_row("TOTAL_evaluate_fullA_cold", old_total_cold))
write_csv_rows(joinpath(OUTDIR, "profile_OLD_raw.csv"), old_rows)

println("\n" * "="^78); println("NEW PROFILER (oracle_fast.jl, Phase 1 fixes + nested inner-solve timers)"); println("="^78)
flush(stdout)
for warm in (true, false)
    evaluate_fullA_fast(x0, ctx; cache = nothing, warm = warm)
end

# ---- WARM reps measured in their OWN prof_reset! scope -- do NOT pool with cold reps. A warmed,
#      already-converged inner solve does ~1 fg call / 0 hessian calls; a cold one does ~10/~9 (see
#      TEST 5 in test_oracle_fast.jl) -- pooling the two into one "median" per nested-callback label
#      would be a bimodal-distribution artifact, exactly the kind of double-counting/mis-attribution
#      this re-profile exists to eliminate. ----
prof_reset!()
new_total_warm = Float64[]
new_fg_calls = Int[]; new_hess_calls = Int[]
for rep in 1:N_REPS
    t0 = time_ns(); _, meta = evaluate_fullA_fast(x0, ctx; cache = nothing, warm = true)
    push!(new_total_warm, (time_ns() - t0) / 1e9)
    push!(new_fg_calls, meta.n_fg_calls); push!(new_hess_calls, meta.n_hess_calls)
end
new_rows_warm = prof_summary()
push!(new_rows_warm, stats_row("TOTAL_evaluate_fullA_fast_warm", new_total_warm))
write_csv_rows(joinpath(OUTDIR, "profile_NEW_warm_only.csv"), new_rows_warm)

prof_reset!()
new_total_cold = Float64[]
cold_fg_calls = Int[]; cold_hess_calls = Int[]
for rep in 1:N_REPS
    t0 = time_ns(); _, meta = evaluate_fullA_fast(x0, ctx; cache = nothing, warm = false)
    push!(new_total_cold, (time_ns() - t0) / 1e9)
    push!(cold_fg_calls, meta.n_fg_calls); push!(cold_hess_calls, meta.n_hess_calls)
end
new_rows_cold = prof_summary()
push!(new_rows_cold, stats_row("TOTAL_evaluate_fullA_fast_cold", new_total_cold))
write_csv_rows(joinpath(OUTDIR, "profile_NEW_cold_only.csv"), new_rows_cold)

new_rows = new_rows_warm   # the WARMED breakdown below uses warm-only stats, not pooled

# ---- derive the exclusive inner_knitro_dual_solve time (its own recorded total is INCLUSIVE of
#      the nested fg/hessian callback totals, by the @prof macro's own documented nesting behavior) ----
label_total(rows, lbl) = begin
    r = filter(r -> r.label == lbl, rows)
    isempty(r) ? 0.0 : r[1].mean_s * r[1].n   # total = mean*n (n reps, each ONE call to this label at the warmed calibration point)
end
solve_incl_total = label_total(new_rows, "inner_knitro_dual_solve")
fg_total = label_total(new_rows, "inner_dual_fg_callback")
hess_total = label_total(new_rows, "inner_dual_hessian_callback")
solve_excl_total = solve_incl_total - fg_total - hess_total
n_solve = (filter(r -> r.label == "inner_knitro_dual_solve", new_rows)[1]).n

println("\n" * "="^78); println("DERIVED: inner_knitro_dual_solve EXCLUSIVE (KNITRO's own SQP/line-search overhead, not attributable to a callback)"); println("="^78)
@printf("  inclusive total = %.4fs (n=%d)  fg_callback total = %.4fs  hessian_callback total = %.4fs\n", solve_incl_total, n_solve, fg_total, hess_total)
@printf("  EXCLUSIVE total = %.4fs (%.1f%% of inclusive)   mean exclusive per warmed call ~= %.6fs\n",
        solve_excl_total, 100*solve_excl_total/max(solve_incl_total,1e-12), solve_excl_total/max(n_solve,1))

# ---- top-level warmed % breakdown, NOT double-counting nested labels ----
total_warm_median = (filter(r -> r.label == "TOTAL_evaluate_fullA_fast_warm", new_rows)[1]).median_s
top_level_labels_warm = ["reconstruct_full", "inner_moment_build", "inner_knitro_dual_solve",
    "moments_reuse", "primal_weight_recovery", "primal_divergence_compute", "kkt_residual_compute",
    "gravity_compute", "moment_resid_compute", "winner_compute"]
println("\n" * "="^78); println("WARMED TOP-LEVEL BREAKDOWN (median seconds, % of TOTAL_evaluate_fullA_fast_warm=$(round(total_warm_median,digits=6))s)"); println("="^78)
println("(inner_knitro_dual_solve is INCLUSIVE of the two nested callback rows below it -- do not sum this column naively)")
breakdown_rows = NamedTuple[]
for lbl in top_level_labels_warm
    r = filter(r -> r.label == lbl, new_rows)
    isempty(r) && continue
    med = r[1].median_s
    @printf("  %-28s median=%.6fs  pct=%.1f%%\n", lbl, med, 100*med/total_warm_median)
    push!(breakdown_rows, (label = lbl, median_s = med, pct_of_total = 100*med/total_warm_median))
end
for (lbl, tot) in (("  -> inner_dual_fg_callback", fg_total), ("  -> inner_dual_hessian_callback", hess_total))
    med_est = tot / max(n_solve, 1)
    @printf("  %-28s median~=%.6fs  pct~=%.1f%% (nested inside inner_knitro_dual_solve, shown for reference)\n", lbl, med_est, 100*med_est/total_warm_median)
end
excl_med_est = solve_excl_total / max(n_solve,1)
@printf("  %-28s median~=%.6fs  pct~=%.1f%% (KNITRO's own overhead, exclusive)\n", "  -> (exclusive remainder)", excl_med_est, 100*excl_med_est/total_warm_median)
push!(breakdown_rows, (label = "inner_dual_fg_callback (nested)", median_s = fg_total/max(n_solve,1), pct_of_total = 100*(fg_total/max(n_solve,1))/total_warm_median))
push!(breakdown_rows, (label = "inner_dual_hessian_callback (nested)", median_s = hess_total/max(n_solve,1), pct_of_total = 100*(hess_total/max(n_solve,1))/total_warm_median))
push!(breakdown_rows, (label = "inner_knitro_dual_solve (exclusive)", median_s = excl_med_est, pct_of_total = 100*excl_med_est/total_warm_median))
write_csv_rows(joinpath(OUTDIR, "profile_NEW_breakdown.csv"), breakdown_rows)

total_cold_median = (filter(r -> r.label == "TOTAL_evaluate_fullA_fast_cold", new_rows_cold)[1]).median_s
cold_solve_incl_total = label_total(new_rows_cold, "inner_knitro_dual_solve")
cold_fg_total = label_total(new_rows_cold, "inner_dual_fg_callback")
cold_hess_total = label_total(new_rows_cold, "inner_dual_hessian_callback")
cold_n_solve = (filter(r -> r.label == "inner_knitro_dual_solve", new_rows_cold)[1]).n
println("\n" * "="^78); println("COLD-START BREAKDOWN, for contrast (median seconds, % of TOTAL_evaluate_fullA_fast_cold=$(round(total_cold_median,digits=6))s)"); println("="^78)
for lbl in top_level_labels_warm
    r = filter(r -> r.label == lbl, new_rows_cold)
    isempty(r) && continue
    med = r[1].median_s
    @printf("  %-28s median=%.6fs  pct=%.1f%%\n", lbl, med, 100*med/total_cold_median)
end
@printf("    -> inner_dual_fg_callback     median~=%.6fs (n_fg_calls mean=%.1f per cold solve)\n", cold_fg_total/max(cold_n_solve,1), sum(cold_fg_calls)/length(cold_fg_calls))
@printf("    -> inner_dual_hessian_callback median~=%.6fs (n_hess_calls mean=%.1f per cold solve)\n", cold_hess_total/max(cold_n_solve,1), sum(cold_hess_calls)/length(cold_hess_calls))
@printf("    -> (exclusive remainder)       median~=%.6fs\n", (cold_solve_incl_total-cold_fg_total-cold_hess_total)/max(cold_n_solve,1))

println("\n" * "="^78); println("BEFORE/AFTER TOTAL WALL TIME (warmed, median)"); println("="^78)
old_total_median = (filter(r -> r.label == "TOTAL_evaluate_fullA_warm", old_rows)[1]).median_s
@printf("  OLD (oracle_profiled.jl, pre-Phase-1):  median=%.6fs\n", old_total_median)
@printf("  NEW (oracle_fast.jl, post-Phase-1):     median=%.6fs\n", total_warm_median)
@printf("  SPEEDUP: %.3fx\n", old_total_median / total_warm_median)

println("\n" * "="^78); println("n_fg_calls / n_hess_calls at the WARMED calibration point (across $N_REPS reps)"); println("="^78)
@printf("  n_fg_calls:   min=%d max=%d mean=%.2f\n", minimum(new_fg_calls), maximum(new_fg_calls), sum(new_fg_calls)/length(new_fg_calls))
@printf("  n_hess_calls: min=%d max=%d mean=%.2f\n", minimum(new_hess_calls), maximum(new_hess_calls), sum(new_hess_calls)/length(new_hess_calls))
println("  (VERIFIES the L_fix/optimized-value claim below: a warm-started, already-converged inner")
println("   solve needs very few/zero Hessian calls; a cold solve needs many -- see phaseD/L_fix profile.)")

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "OLD total warm median: ", old_total_median, "s")
    println(io, "NEW total warm median: ", total_warm_median, "s")
    println(io, "speedup: ", old_total_median/total_warm_median, "x")
    println(io, "inner_knitro_dual_solve inclusive total: ", solve_incl_total, "s over ", n_solve, " calls")
    println(io, "  fg_callback total: ", fg_total, "s  hessian_callback total: ", hess_total, "s")
    println(io, "  EXCLUSIVE (KNITRO overhead) total: ", solve_excl_total, "s (", 100*solve_excl_total/max(solve_incl_total,1e-12), "% of inclusive)")
    println(io, "n_fg_calls (warmed): min=", minimum(new_fg_calls), " max=", maximum(new_fg_calls), " mean=", sum(new_fg_calls)/length(new_fg_calls))
    println(io, "n_hess_calls (warmed): min=", minimum(new_hess_calls), " max=", maximum(new_hess_calls), " mean=", sum(new_hess_calls)/length(new_hess_calls))
end

println("\nWrote ", OUTDIR)
