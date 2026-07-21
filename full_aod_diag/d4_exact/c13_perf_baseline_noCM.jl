# Continuation 13 perf comparison, run A: regular (no CM) delta=1 upper bound, FULL PRODUCTION
# SETTINGS -- literally the reviewed production driver (c10_d20_production_driver.jl's
# run_polish_checkpointed), not a reimplementation. Wraps the whole call in @prof "TOTAL_RUN" so
# the already-extensive existing instrumentation (compressed inner solve FG/Hessian/moment-build,
# KKT/gravity/moment-resid computation, etc. -- all already @prof-labeled in oracle_fast.jl/
# compressed_live.jl) gives a wall-time breakdown without touching any trusted file.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Printf, Serialization

const DRAW_SEED = 20260719
const MAXTIME = get(ENV, "C13_PERF_MAXTIME", "1800") |> x -> parse(Float64, x)
const CKPT_DIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_perf_comparison", "noCM")
mkpath(CKPT_DIR)

lp(xs...) = (println(xs...); flush(stdout))

# Compute the calibration starting point once, deterministically (same seed/config
# run_polish_checkpointed will use internally) -- NOT passed into the function (its own signature
# takes g_start/zfree_start, not a pre-built ctx), but reproducing the exact same values since the
# seed+config are identical.
Random.seed!(DRAW_SEED)
ctx0 = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe0))
g_start = w_calib[1]; zfree_start = w_calib[2:end]
lp(">>> calibration g_start=", g_start, "  length(zfree_start)=", length(zfree_start))

prof_reset!()
t_wall0 = time()
result = @prof "TOTAL_RUN" run_polish_checkpointed("noCM_prod", true, g_start, zfree_start;
    maxtime_real = MAXTIME, hessopt_tag = "sr1", W_in = 80000, delta_in = 1.0,
    draw_seed_in = DRAW_SEED, ckpt_dir = CKPT_DIR, checkpoint_interval_s = 90.0)
t_wall1 = time()

lp("="^100)
lp("RUN A (no CM, full production) SUMMARY")
lp("="^100)
lp(@sprintf("wall_ext=%.1fs (harness wall=%.1fs)  n_eval=%d  n_grad_calls=%d  kappa=%s  knitro_status=%d",
    result.wall_ext, t_wall1 - t_wall0, result.n_eval, result.n_grad_calls, string(result.kappa), result.knitro_status))
lp("screen_counts: ", result.screen_counts)

rows = prof_summary()
total_time_by_label = Dict(r.label => r.n * r.mean_s for r in rows)
total_run_time = get(total_time_by_label, "TOTAL_RUN", result.wall_ext)
others = sort([(l, t) for (l, t) in total_time_by_label if l != "TOTAL_RUN"], by = x -> -x[2])

lp("")
lp("="^100)
lp("WALL-TIME BREAKDOWN (@prof labels, total seconds, sorted descending)")
lp("="^100)
for (label, tot) in others
    pct = 100 * tot / total_run_time
    lp(@sprintf("  %-45s  total=%10.2fs  (%.1f%% of TOTAL_RUN)", label, tot, pct))
end
sum_others = sum(x[2] for x in others; init = 0.0)
unlabeled = total_run_time - sum_others
lp(@sprintf("  %-45s  total=%10.2fs  (%.1f%% of TOTAL_RUN)", "[unlabeled / gradient-loop / KNITRO overhead]", unlabeled, 100*unlabeled/total_run_time))
lp(@sprintf("\nTOTAL_RUN (outer wrapper) = %.1fs. Sum of inner @prof labels = %.1fs.", total_run_time, sum_others))

write_csv_rows(joinpath(CKPT_DIR, "prof_summary.csv"), rows)
serialize(joinpath(CKPT_DIR, "result.jls"), (kappa=result.kappa, wall_ext=result.wall_ext, n_eval=result.n_eval,
    n_grad_calls=result.n_grad_calls, knitro_status=result.knitro_status, best_feasible=result.best_feasible,
    screen_counts=result.screen_counts))
lp("Profile CSV written: ", joinpath(CKPT_DIR, "prof_summary.csv"))
lp("DONE")
