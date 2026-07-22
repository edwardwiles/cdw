# Continuation 14, Task 2: combined CM-Hessian benchmark -- the 2x2 matrix (serial bins / Julia-
# thread-parallel bins) x (BLAS=1 / BLAS in {8,10,20}) -- cells A(serial,BLAS1), B(threaded,BLAS1),
# C(serial,BLAS{8,10,20}), D(threaded,BLAS{8,10,20}) -- at the genuinely hard CM L=50 point and the
# near-infeasible CM point from c14_find_hard_cm_point.jl, plus calibration and one accepted
# CM outer-trajectory checkpoint (stage_L50_latest) as cheaper reference points.
#
# This is the ONE experiment diag/fullA-inner-blas-threading's own report explicitly flagged as
# not run: that branch's "does the ~2.9x component-level threaded-Hessian win survive a live
# KN_solve" check (report sec 12) used an EASY ~2s cold calibration-adjacent solve and found only
# 1.04x end-to-end, concluding the Hessian callback is amortized across too few Newton iterations
# to matter for that scenario. Here we re-run the SAME live-wiring check
# (`cm_production_value_v2_hess`, cm_hessian_threaded.jl) on a point that needs 7-10 Hessian calls
# at 20-35s (matching the REAL full CM L=50 outer trajectory's own per-solve profile, see
# c14_find_hard_cm_point.jl header) -- the exact scenario that report flagged as untested.
#
# Correctness gate (not just a speed test): every cell's Delta_dual is diffed against a REF cell
# (serial bins, BLAS.gemm! -- i.e. the UNTOUCHED production `hessian_cm_structured!` via
# `archC_hess_cb_builder`, cm_hessian_architectures.jl) to ~1e-13.
#
# Run with: JULIA_NUM_THREADS=10 OPENBLAS_NUM_THREADS=1 julia --project=. full_aod_diag/d4_exact/c14_cm_hessian_benchmark.jl
# (BLAS thread count is varied AT RUNTIME via BLAS.set_num_threads inside this script -- Julia
# thread count is fixed at process launch, hence -t 10: matches the source branch's own measured
# `build_bin_tables!` peak (3, "Section 10 -- Julia-thread scaling"), so "threaded bins" cells use
# Threads.nthreads()=10 throughout.)
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
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
using Printf, LinearAlgebra, Random, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c14_parallel_prod")
mkpath(OUTDIR)

lp("=== c14_cm_hessian_benchmark === ", Dates.now(), "  Threads.nthreads()=", Threads.nthreads())
W = 80000; DELTA = 1.0
Random.seed!(20260719)
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
lp(@sprintf(">>> ctx built in %.1fs. D=%d W=%d", time()-t0, D, W))

x_free_calib = ctx.θ0_up[ctx.free_idx]
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
z0 = log.(Aod_theta_natural)
zfree0 = pivot_reduce(reshape(z0, D, D), pe)
gp0 = ctx.θ0_up[3+D]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf_calib = x_free_from_w(vcat(gp0 * 1.01, zfree0))

snaps = nested_grid_sequence([10, 20, 50])
cfg = CMConfig(common_marginals = true, cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50],
               cm_basis = :cumulative, cm_hessian_backend = :structured, contrasts = :anchored)
t0 = time()
pcx = build_cm_production_context_v2(ctx, CS, cfg; L = 50)
cctx = build_cm_bin_ctx(ctx, pcx.aug)   # established workaround, see cm_hessian_threaded.jl docstring
lp(@sprintf(">>> CM production context (L=50) built in %.1fs. ncm=%d d_total=%d",
    time()-t0, pcx.aug.ncm, pcx.ctx_cm.obj.d))

fixture_path = joinpath(OUTDIR, "hard_cm_point.jls")
isfile(fixture_path) || error("c14_cm_hessian_benchmark.jl: run c14_find_hard_cm_point.jl first -- $fixture_path not found")
fixture = deserialize(fixture_path)
lp(">>> loaded fixture: hard_point=", fixture.hard_point.label, " (n_hess=", fixture.hard_point.n_hess,
   " wall=", round(fixture.hard_point.wall, digits=1), "s)  near_infeasible=", fixture.near_infeasible_point.label)

# ---- one accepted CM outer-trajectory checkpoint (stage_L50), if cheaply available ----
traj_xf = nothing
ckpt_path = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "c13_d20_cm_continuation", "stage_L50_latest.jls")
if isfile(ckpt_path)
    payload = deserialize(ckpt_path)
    if payload.best_w !== nothing
        traj_xf = x_free_from_w(payload.best_w)
        lp(">>> loaded accepted CM outer-trajectory checkpoint stage_L50 (kappa=", payload.kappa, ")")
    end
end

points = NamedTuple[]
push!(points, (label = "hard_cm_point", xf = fixture.hard_point.xf, full_grid = true))
push!(points, (label = "near_infeasible_cm_point", xf = fixture.near_infeasible_point.xf, full_grid = true))
push!(points, (label = "calibration", xf = xf_calib, full_grid = false))
if traj_xf !== nothing
    push!(points, (label = "cm_trajectory_stage_L50", xf = traj_xf, full_grid = false))
end

"""
Force a genuinely COLD inner solve: `pcx.ctx_cm.obj.x` is production's own warm-start cache
(`PsiObjectiveBundleImplicit.x`, defaults to `NaN .* ones(outer_constr_index)`, overwritten with
the converged (zeta,lambda) on every successful solve, and `inner_loop_initial_values` reuses it
as the NEXT solve's starting point whenever `use_cached_x=true` AND `norm(obj.x) < 1e6`, per
cc_algo/inner_loop_functions.jl). Without this reset, every cell in this benchmark after the
first would trivially warm-start from the IMMEDIATELY PRIOR cell's own converged solution AT THE
SAME xf -- 1 FG call, 0 further Newton iterations, testing nothing about Hessian-construction
cost. Caught live during this benchmark's first run (all non-REF cells showed n_fg=1, iters=0,
wall~2s regardless of threaded/BLAS settings -- a real bug, not a threading win) -- fixed here.
"""
function force_cold_start!()
    fill!(pcx.ctx_cm.obj.x, NaN)
    return nothing
end

"One benchmarked cell: run cm_production_value_v2_hess, return timing/correctness/allocation stats."
function run_cell(label, xf; threaded_bins, blas_threads, use_syrk, tls)
    BLAS.set_num_threads(blas_threads)
    force_cold_start!()
    prof_reset!()
    iters0 = CS.INNER_ITERS_TOTAL[]
    stats = @timed cm_production_value_v2_hess(xf, pcx, cctx; threaded_bins = threaded_bins, tls = tls, use_syrk = use_syrk)
    (K, base, n_fg, n_hess) = stats.value
    iters = CS.INNER_ITERS_TOTAL[] - iters0
    rows = prof_summary()
    hess_row = findfirst(r -> r.label == "inner_dual_hessian_callback_archC_v2", rows)
    hess_total_s = hess_row === nothing ? NaN : rows[hess_row].n * rows[hess_row].mean_s
    hess_alloc = hess_row === nothing ? NaN : rows[hess_row].total_alloc_bytes
    hess_gc_s = hess_row === nothing ? NaN : rows[hess_row].total_gc_s
    # Remediation fix (task Part A, F1): was `-base.ζstar`, which omits mean(Psi(q*)).
    Delta_dual = delta_dual_from_base(pcx.ctx_cm.obj, base)
    lp(@sprintf("  [%s] threaded=%-5s blas=%-2d syrk=%-5s  wall=%8.3fs  hess_cb_total=%7.3fs(n=%d)  n_fg=%3d iters=%3d  alloc=%.2eB  gc=%.3fs  status=%d  Delta_dual=%.12f",
        label, string(threaded_bins), blas_threads, string(use_syrk), stats.time, hess_total_s, n_hess, n_fg, iters,
        stats.bytes, stats.gctime, base.inner_status, Delta_dual))
    return (label = label, threaded_bins = threaded_bins, blas_threads = blas_threads, use_syrk = use_syrk,
            wall = stats.time, hess_cb_total_s = hess_total_s, hess_cb_n = n_hess, hess_cb_alloc_bytes = hess_alloc,
            hess_cb_gc_s = hess_gc_s, n_fg = n_fg, iters = iters, total_alloc_bytes = stats.bytes, total_gc_s = stats.gctime,
            inner_status = base.inner_status, Delta_dual = Delta_dual)
end

all_rows = NamedTuple[]
for pt in points
    lp(""); lp("="^110); lp("POINT: ", pt.label); lp("="^110)
    tls = build_thread_local_scratch(cctx)

    # ---- REF: unmodified production hessian_cm_structured! (serial, gemm), BLAS=1 ----
    BLAS.set_num_threads(1)
    force_cold_start!()
    prof_reset!()
    θ_full0 = CS.reconstruct_full(pt.xf, pcx.ctx_cm.m)
    ref_stats = @timed inner_loop_internal_archgeneric(pcx.ctx_cm.obj, θ_full0; hess_cb_builder = _o -> archC_hess_cb_builder(cctx))
    (K_ref, x_ref, status_ref, nfg_ref, nhess_ref) = ref_stats.value
    Delta_ref = -x_ref[1]
    ref_rows = prof_summary()
    ref_hess_row = findfirst(r -> r.label == "inner_dual_hessian_callback_archC", ref_rows)
    ref_hess_total_s = ref_hess_row === nothing ? NaN : ref_rows[ref_hess_row].n * ref_rows[ref_hess_row].mean_s
    lp(@sprintf("  [%s] REF (production, serial+gemm, BLAS=1)  wall=%8.3fs  n_hess=%d  status=%d  Delta_dual=%.12f",
        pt.label, ref_stats.time, nhess_ref, status_ref, Delta_ref))
    push!(all_rows, (label = pt.label, threaded_bins = false, blas_threads = 1, use_syrk = false,
        wall = ref_stats.time, hess_cb_total_s = NaN, hess_cb_n = nhess_ref, hess_cb_alloc_bytes = NaN,
        hess_cb_gc_s = NaN, n_fg = nfg_ref, iters = -1, total_alloc_bytes = ref_stats.bytes,
        total_gc_s = ref_stats.gctime, inner_status = status_ref, Delta_dual = Delta_ref, cell = "REF",
        diff_vs_ref = 0.0))

    cells = pt.full_grid ?
        [("A", false, 1), ("B", true, 1), ("C_blas8", false, 8), ("C_blas10", false, 10), ("C_blas20", false, 20),
         ("D_blas8", true, 8), ("D_blas10", true, 10), ("D_blas20", true, 20)] :
        [("A", false, 1), ("B", true, 1), ("C_blas10", false, 10), ("D_blas10", true, 10)]

    for (cellname, threaded, blas_n) in cells
        r = run_cell(pt.label, pt.xf; threaded_bins = threaded, blas_threads = blas_n, use_syrk = true, tls = tls)
        diff = abs(r.Delta_dual - Delta_ref)
        hess_speedup = (isnan(ref_hess_total_s) || isnan(r.hess_cb_total_s) || r.hess_cb_total_s == 0) ? NaN : ref_hess_total_s / r.hess_cb_total_s
        lp(@sprintf("      -> cell=%-9s |Delta_dual - REF| = %.3e   speedup(wall) = %.3fx   speedup(hess_cb) = %.3fx",
            cellname, diff, ref_stats.time / r.wall, hess_speedup))
        push!(all_rows, merge(r, (cell = cellname, diff_vs_ref = diff)))
    end
end

BLAS.set_num_threads(1)   # restore a sane default before writing output / exiting
lp(""); lp("="^110); lp("SUMMARY TABLE"); lp("="^110)
lp(@sprintf("%-28s %-10s %-9s %-6s %-6s %10s %10s %8s %8s %12s", "point", "cell", "threaded", "blas", "syrk", "wall_s", "hess_cb_s", "n_hess", "n_fg", "diff_vs_ref"))
for r in all_rows
    lp(@sprintf("%-28s %-10s %-9s %-6d %-6s %10.3f %10.3f %8d %8d %12.3e",
        r.label, get(r, :cell, "?"), string(r.threaded_bins), r.blas_threads, string(r.use_syrk),
        r.wall, r.hess_cb_total_s, r.hess_cb_n, r.n_fg, get(r, :diff_vs_ref, 0.0)))
end

write_csv_rows(joinpath(OUTDIR, "cm_hessian_benchmark.csv"), all_rows)
serialize(joinpath(OUTDIR, "cm_hessian_benchmark_raw.jls"), all_rows)
lp(""); lp(">>> wrote ", joinpath(OUTDIR, "cm_hessian_benchmark.csv"))
lp("DONE_C14_CM_HESSIAN_BENCHMARK")
