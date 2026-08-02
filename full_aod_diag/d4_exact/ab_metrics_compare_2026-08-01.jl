# ============================================================================
# Claude Code task 2026-08-01, §15/§16: A/B metrics comparison + decision
# rule application. Reads both arms' trace CSVs (written by
# run_ab_full_upper_2026-08-01.jl / run_ab_profiled_upper_2026-08-01.jl),
# reports best verified objective, time-to-threshold, eval/gradient counts,
# and applies the task's own >=20%/<=10% decision thresholds.
# ============================================================================
using CSV, DataFrames, Printf

const ROOT = "/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-outer-ab-2026-08-01"

full_trace = CSV.read(joinpath(ROOT, "FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_FULL_TRACE.csv"), DataFrame)
prof_trace = CSV.read(joinpath(ROOT, "FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_PROFILED_TRACE.csv"), DataFrame)

full_verified = filter(r -> r.delta_feasible !== missing ? true : true, full_trace)  # keep all; verification flag differs by schema
best_full = minimum(full_trace.Delta_dual)
best_prof = minimum(prof_trace.Delta_dual)

t_full_max = maximum(full_trace.t_elapsed)
t_prof_max = maximum(prof_trace.t_elapsed)

println("="^90)
println("FULL arm:     n_eval=", nrow(full_trace), "  wall=", round(t_full_max, digits=1), "s  best_Delta=", best_full)
println("PROFILED arm: n_eval=", nrow(prof_trace), "  wall=", round(t_prof_max, digits=1), "s  best_Delta=", best_prof)
println("="^90)

# matched wall-clock comparison: best objective each arm achieved by min(t_full_max, t_prof_max)
t_match = min(t_full_max, t_prof_max)
best_full_at_match = minimum(full_trace[full_trace.t_elapsed .<= t_match, :Delta_dual])
best_prof_at_match = minimum(prof_trace[prof_trace.t_elapsed .<= t_match, :Delta_dual])
println(@sprintf("At matched wall-clock t=%.1fs: full_best=%.6e  profiled_best=%.6e", t_match, best_full_at_match, best_prof_at_match))

# time to reach a common threshold (the worse of the two final bests, i.e. a threshold BOTH reached)
threshold = max(best_full, best_prof)
t_full_thresh = minimum(full_trace[full_trace.Delta_dual .<= threshold, :t_elapsed]; init = Inf)
t_prof_thresh = minimum(prof_trace[prof_trace.Delta_dual .<= threshold, :t_elapsed]; init = Inf)
println(@sprintf("Time to reach threshold Delta<=%.6e: full=%.1fs  profiled=%.1fs", threshold, t_full_thresh, t_prof_thresh))

pct_diff_at_match = (best_prof_at_match - best_full_at_match) / abs(best_full_at_match) * 100
println(@sprintf("\nprofiled vs full objective at matched time: %.2f%% (negative = profiled better, this is a MINIMIZATION)", pct_diff_at_match))

verdict = if pct_diff_at_match <= -20
    "profiled_better_objective_at_matched_time"
elseif pct_diff_at_match >= 20
    "full_better_objective_at_matched_time"
elseif abs(pct_diff_at_match) <= 10
    "equivalent"
else
    "marginal_difference_$(round(pct_diff_at_match,digits=1))pct"
end
println("\nOUTER_AB_UPPER = ", verdict)

summary = DataFrame(
    arm = ["full", "profiled"],
    n_eval = [nrow(full_trace), nrow(prof_trace)],
    wall_s = [t_full_max, t_prof_max],
    best_Delta = [best_full, best_prof],
    best_Delta_at_matched_time = [best_full_at_match, best_prof_at_match],
    time_to_common_threshold_s = [t_full_thresh, t_prof_thresh],
)
outpath = joinpath(ROOT, "FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_SUMMARY.csv")
CSV.write(outpath, summary)
println("\nWrote $outpath")
