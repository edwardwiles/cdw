# Continuation 13 perf comparison, run B: delta=1 upper bound WITH L=50 common marginals,
# production CM bundle (cumulative Architecture C -- confirmed faster/better-conditioned than
# interval-native at real D20 scale, see docs/fullA_c13_cumulative_vs_interval_native_d20.log;
# Architecture B moment construction). Same calibration start, same delta, same SR1
# wallclock-budget outer optimizer settings as run A (c13_perf_baseline_noCM.jl) to the extent
# this driver and the true production driver's machinery allow -- see the comparison report for
# the honest list of what could NOT be held identical (compressed representation +
# pairwise/witness screening are real-production-only features; the CM columns' dense-appended
# block does not support the compressed representation, per Continuation 12's finding).
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, Random, Serialization

const DRAW_SEED = 20260719
const MAXTIME = get(ENV, "C13_PERF_MAXTIME", "1800") |> x -> parse(Float64, x)
const CKPT_DIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_perf_comparison", "CM_L50")
mkpath(CKPT_DIR)

lp(xs...) = (println(xs...); flush(stdout))

# SAME explicit seed-before-setup convention c10_d20_production_driver.jl uses, for the SAME
# reason (context_real_d20.jl's real-data U draws are otherwise unseeded/non-reproducible across
# processes) -- ensures run A and run B see bit-identical ctx.U.
Random.seed!(DRAW_SEED)
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe))
lp(">>> calibration g_start=", w_calib[1], "  length(zfree_start)=", length(w_calib)-1)

snaps = nested_grid_sequence([10, 20, 50])
L = 50
t_setup = @elapsed pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
lp(">>> CM production context (L=$L) built in ", round(t_setup, digits=2), "s.  ncm=", pcx.aug.ncm, "  d_total=", pcx.ctx_cm.obj.d)

prof_reset!()
t_wall0 = time()
result = @prof "TOTAL_RUN" run_cm_upper(pcx, ctx, pe, copy(w_calib);
    delta = 1.0, maxtime_real = MAXTIME, opt_file = "csw_outer_wallclock_sr1.opt", verbose = true)
t_wall1 = time()

lp("="^100)
lp("RUN B (CM L=50) SUMMARY")
lp("="^100)
lp(@sprintf("wall=%.1fs (harness wall=%.1fs)  n_eval=%d  n_grad=%d  kappa=%s  knitro_status=%d",
    result.wall, t_wall1 - t_wall0, result.n_eval, result.n_grad, string(result.kappa), result.knitro_status))

rows = prof_summary()
total_time_by_label = Dict(r.label => r.n * r.mean_s for r in rows)
total_run_time = get(total_time_by_label, "TOTAL_RUN", result.wall)
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
serialize(joinpath(CKPT_DIR, "result.jls"), (kappa=result.kappa, wall=result.wall, n_eval=result.n_eval, n_grad=result.n_grad, knitro_status=result.knitro_status, best=result.best))
lp("Profile CSV written: ", joinpath(CKPT_DIR, "prof_summary.csv"))
lp("DONE")
