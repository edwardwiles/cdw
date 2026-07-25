# ============================================================================
# Origin-ZC bounded BLAS benchmark (allocation/Hessian port task §7).
#
# "Do not automatically replace origin-ZC's dense Architecture A Hessian. Perform ONE bounded
# benchmark: current dense Architecture A; BLAS threads 1,4,8,10,20; P0 and one near-budget
# point; one 300-second direct outer run." -- taken literally: this does NOT sweep a P2/hard
# point or attempt Architecture C on origin-ZC (deliberately, per that section's explicit
# instruction not to force Architecture C onto mean/pairwise-product moments).
#
# P0 = genuine calibrated start (real feasible eta0 = log(mean(U^k)) per origin, K_mean=1, same
# construction as test_backend_manifest_cm_originzc.jl). P_near = harvested via one real,
# moderate-budget run_originzc_upper_checkpointed call from P0.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/originzc_blas_sweep_2026-07-25.jl <outdir>
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_checkpoint.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl"]
    include(joinpath(_D4E, f))
end
using Dates, Serialization, Printf, LinearAlgebra, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = abspath(ARGS[1])
mkpath(OUTDIR)
const HARVEST_S = parse(Float64, get(ENV, "HARVEST_S", "60.0"))
const SWEEP_BUDGET_S = parse(Float64, get(ENV, "SWEEP_BUDGET_S", "30.0"))
const BLAS_SETTINGS = [1, 4, 8, 10, 20]

lp(">>> originzc_blas_sweep starting ", now(), " Julia threads=", Threads.nthreads())

t0 = time()
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)
w_calib = vcat(g0, zfree0)
layout1 = OriginByPowerLayout(ctx0.D, 1, 1)
nu0_1 = Vector{Float64}(undef, n_eta(layout1))
for o in 1:ctx0.D
    nu0_1[target_index(layout1, o, 1)] = mean(@view ctx0.U[:, o])
end
w_p0 = vcat(w_calib, log.(nu0_1))
lp(">>> context build wall = ", round(time() - t0, digits = 3), "s  n_free=", length(w_p0))

# ---------------------------------------------------------------------------
# Harvest a near-budget point.
# ---------------------------------------------------------------------------
rh = run_originzc_upper_checkpointed(w_p0; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
    draw_seed = 20260719, maxtime_real = HARVEST_S, ckpt_dir = joinpath(OUTDIR, "ckpt_harvest"),
    label = "harvest", checkpoint_interval_s = 9999.0,
    distribution_restriction = :origin_specific_moments, K_mean = 1, destination_sample = :exclude_row)
w_near = rh.best !== nothing ? rh.best.w : w_p0
lp(">>> harvest: n_eval=", rh.n_eval, " best=", rh.best === nothing ? "nothing (fallback to P0)" : "gp=$(rh.best.gp) Delta=$(rh.best.Delta)")

points = Dict(:P0 => w_p0, :P_near => w_near)
serialize(joinpath(OUTDIR, "originzc_sweep_points.jls"), points)

# ---------------------------------------------------------------------------
# BLAS sweep, Architecture A (dense) retained throughout -- this benchmark does not attempt to
# change origin-ZC's Hessian architecture.
# ---------------------------------------------------------------------------
results = NamedTuple[]
for pname in (:P0, :P_near), b in BLAS_SETTINGS
    w = points[pname]
    tt0 = time()
    r = run_originzc_upper_checkpointed(w; W = 80_000, delta = 1.0, draw_design = :pseudorandom,
        draw_seed = 20260719, maxtime_real = SWEEP_BUDGET_S, ckpt_dir = joinpath(OUTDIR, "ckpt_$(pname)_blas$(b)"),
        label = "sweep_$(pname)_blas$(b)", checkpoint_interval_s = 9999.0,
        distribution_restriction = :origin_specific_moments, K_mean = 1, destination_sample = :exclude_row,
        blas_threads = b)
    wall = time() - tt0
    push!(results, (point = pname, blas_threads = b, wall = wall, n_eval = r.n_eval, n_grad = r.n_grad,
                     best_gp = r.best === nothing ? NaN : r.best.gp,
                     best_Delta = r.best === nothing ? NaN : r.best.Delta,
                     knitro_status = r.knitro_status))
    lp(">>> ", pname, " BLAS=", b, ": wall=", round(wall, digits = 2), "s n_eval=", r.n_eval,
       " n_grad=", r.n_grad, " best_Delta=", results[end].best_Delta)
end

open(joinpath(OUTDIR, "originzc_blas_sweep_2026-07-25.csv"), "w") do io
    println(io, "point,blas_threads,wall_s,n_eval,n_grad,best_gp,best_Delta,knitro_status")
    for r in results
        println(io, r.point, ",", r.blas_threads, ",", round(r.wall, digits = 3), ",", r.n_eval, ",",
                r.n_grad, ",", r.best_gp, ",", r.best_Delta, ",", r.knitro_status)
    end
end
lp(">>> DONE wall_total=", round(time() - t0, digits = 2), "s")
