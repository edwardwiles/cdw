# ============================================================================
# CM threaded Architecture-C bounded sweep across P0/P1/P2 (allocation/Hessian port task §6),
# filling the gap the source port branch's own test_cm_threaded_hessian.jl disclosed (one hard
# point only, isolated Hessian-callback timing, not a P0/P1/P2 x complete-solve comparison).
#
# Bounded per this session's plan (full Julia-thread x BLAS-thread x 3-point matrix is not
# attempted -- disclosed reduction, matching this branch's own house style): at each of P0
# (calibration), P1 (early/moderate real point), P2 (deepest/hardest real point reached from a
# longer real trajectory), compares 3 configs at Threads.nthreads()=20 (bin-table parallelism):
#   serial      : threaded_bins=false, BLAS=1  (original architecture, pre-port baseline)
#   threaded    : threaded_bins=true,  BLAS=1  (production default as of this port)
#   threaded+b4 : threaded_bins=true,  BLAS=4  (oversubscription check: does adding BLAS threads
#                 on top of Julia-thread bin-table parallelism help or hurt)
#
# P1/P2 are harvested from two independent real run_cm_upper_checkpointed calls of different
# wall budgets from calibration (production default threaded_bins=true throughout the harvest
# itself), using `res.best.w` (the outer vector at the best verified feasible incumbent) --
# genuine production-driver-visited points, not synthetic.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/cm_threaded_sweep_p0p1p2_2026-07-25.jl <outdir>
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Dates, Serialization, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = abspath(ARGS[1])
mkpath(OUTDIR)
const P1_HARVEST_S = parse(Float64, get(ENV, "P1_HARVEST_S", "40.0"))
const P2_HARVEST_S = parse(Float64, get(ENV, "P2_HARVEST_S", "150.0"))
const SWEEP_BUDGET_S = parse(Float64, get(ENV, "SWEEP_BUDGET_S", "25.0"))
const L = parse(Int, get(ENV, "BENCH_L", "50"))

lp(">>> cm_threaded_sweep_p0p1p2 starting ", now(), " Julia threads=", Threads.nthreads())

t0 = time()
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)
w_calib = vcat(g0, zfree0)
snaps = nested_grid_sequence([10, 20, 50]); probs = snaps[L]
lp(">>> context build wall = ", round(time() - t0, digits = 3), "s  n_free=", length(w_calib))

# ---------------------------------------------------------------------------
# Harvest P1 (moderate) and P2 (hard/deep) via independent real runs from calibration.
# ---------------------------------------------------------------------------
r1 = run_cm_upper_checkpointed(w_calib; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
    draw_seed = 20260719, L = L, contrasts = :orthonormal, probs = probs, maxtime_real = P1_HARVEST_S,
    ckpt_dir = joinpath(OUTDIR, "ckpt_harvest_p1"), label = "harvest_p1", checkpoint_interval_s = 9999.0,
    destination_sample = :exclude_row)
p1_w = r1.best !== nothing ? r1.best.w : w_calib
lp(">>> P1 harvest: n_eval=", r1.n_eval, " best=", r1.best === nothing ? "nothing (fallback to calib)" : "gp=$(r1.best.gp) Delta=$(r1.best.Delta)")

r2 = run_cm_upper_checkpointed(w_calib; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
    draw_seed = 20260719, L = L, contrasts = :orthonormal, probs = probs, maxtime_real = P2_HARVEST_S,
    ckpt_dir = joinpath(OUTDIR, "ckpt_harvest_p2"), label = "harvest_p2", checkpoint_interval_s = 9999.0,
    destination_sample = :exclude_row)
p2_w = r2.best !== nothing ? r2.best.w : w_calib
lp(">>> P2 harvest: n_eval=", r2.n_eval, " best=", r2.best === nothing ? "nothing (fallback to calib)" : "gp=$(r2.best.gp) Delta=$(r2.best.Delta)")

points = Dict(:P0 => w_calib, :P1 => p1_w, :P2 => p2_w)
serialize(joinpath(OUTDIR, "cm_sweep_p1p2_points.jls"), points)

# ---------------------------------------------------------------------------
# Config sweep.
# ---------------------------------------------------------------------------
configs = [
    (name = "serial", threaded_bins = false, blas = 1),
    (name = "threaded", threaded_bins = true, blas = 1),
    (name = "threaded_blas4", threaded_bins = true, blas = 4),
]
results = NamedTuple[]
for pname in (:P0, :P1, :P2), c in configs
    w = points[pname]
    tt0 = time()
    r = run_cm_upper_checkpointed(w; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
        draw_seed = 20260719, L = L, contrasts = :orthonormal, probs = probs, maxtime_real = SWEEP_BUDGET_S,
        ckpt_dir = joinpath(OUTDIR, "ckpt_$(pname)_$(c.name)"), label = "sweep_$(pname)_$(c.name)",
        checkpoint_interval_s = 9999.0, destination_sample = :exclude_row,
        threaded_bins = c.threaded_bins, blas_threads = c.blas)
    wall = time() - tt0
    push!(results, (point = pname, config = c.name, threaded_bins = c.threaded_bins, blas = c.blas,
                     wall = wall, n_eval = r.n_eval, n_grad = r.n_grad,
                     best_gp = r.best === nothing ? NaN : r.best.gp,
                     best_Delta = r.best === nothing ? NaN : r.best.Delta,
                     knitro_status = r.knitro_status))
    lp(">>> ", pname, " ", c.name, " (threaded_bins=", c.threaded_bins, ", blas=", c.blas, "): wall=",
       round(wall, digits = 2), "s n_eval=", r.n_eval, " n_grad=", r.n_grad, " best_Delta=", results[end].best_Delta)
end

open(joinpath(OUTDIR, "cm_threaded_sweep_p0p1p2_2026-07-25.csv"), "w") do io
    println(io, "point,config,threaded_bins,blas,wall_s,n_eval,n_grad,best_gp,best_Delta,knitro_status")
    for r in results
        println(io, r.point, ",", r.config, ",", r.threaded_bins, ",", r.blas, ",", round(r.wall, digits = 3), ",",
                r.n_eval, ",", r.n_grad, ",", r.best_gp, ",", r.best_Delta, ",", r.knitro_status)
    end
end
lp(">>> DONE wall_total=", round(time() - t0, digits = 2), "s")
