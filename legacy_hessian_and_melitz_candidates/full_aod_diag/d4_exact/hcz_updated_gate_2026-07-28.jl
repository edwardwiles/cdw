# H_CZ (bin x ZC cross-block, cm_meanzc-only) updated real-D20 timing + correctness gate (2026-07-28
# followup, user-requested). Prior evidence was inherited from an earlier session's sub-block
# profiling only (1.3-1.4x at 20 workers, loses to serial at 4/8) -- this gets FRESH numbers on the
# current merged production defaults (cross_hessian_threaded=true, zc_gram_backend=:reference --
# :blas_gemm confirmed unsafe this same session, see docs/HZZ_BLAS_GEMM_CORRECTNESS_FINDING_2026-07-28.md).
#
# Methodology: ONE real driver call (through run_cm_upper_checkpointed, never a low-level helper)
# to capture a real, converged/feasible live dual state via the CMZC_LIVE_PCX_STASH handle, then
# post-hoc recompute the H_CZ raw-table-fill primitive (bin_zc_cross_hessian_fill!/_threaded!)
# serial vs threaded at several worker counts on that SAME frozen state -- same safe pattern already
# used successfully in diag_cmzc_originzc_d20_profile_2026-07-28.jl and the CM+ZC isolated gate
# (does NOT hit the :operator-backend post-hoc-recompute bug found in the ZC-centering D20 gate,
# since this operates on the already-built cctx.hzz_centered.ZcS, not a fresh archA/archC hess
# builder call).
const D4X = @__DIR__
cd(D4X)
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

lp("Threads.nthreads()=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())
lp("Defaults: CROSS_HESSIAN_THREADED_DEFAULT[]=", CROSS_HESSIAN_THREADED_DEFAULT[],
   " ZC_GRAM_BACKEND_DEFAULT[]=", ZC_GRAM_BACKEND_DEFAULT[])

const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hcz_updated_gate_2026-07-28")
mkpath(OUTROOT)
const REPS = 8
"min/mean seconds over REPS calls, after one untimed warmup call."
function timeit(f::Function; reps::Int = REPS)
    f()
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return (minimum(ts), sum(ts) / reps)
end

function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
w0 = cmzc_w0(ctx0, pe0, theta0, xy0)

CMZC_LIVE_PCX_STASH[] = nothing
out = joinpath(OUTROOT, "driver"); rm(out; force = true, recursive = true); mkpath(out)
t0 = time()
result = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
    cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
    ckpt_dir = out, run_id = "hcz_updated", label = "hcz_updated",
    checkpoint_interval_s = 3600.0, maxtime_real = 60.0, verbose = true)
wall = time() - t0
lp("[DRIVER] wall=", @sprintf("%.1f", wall), "s status=", result.knitro_status, " n_eval=", result.n_eval, " n_grad=", result.n_grad)
handle = CMZC_LIVE_PCX_STASH[]
handle === nothing && error("no live handle captured -- increase maxtime_real")

cctx = handle.cctx; ctx_cm = handle.ctx_cm; obj_z = ctx_cm.obj
w_z = copy(obj_z.arg2); M_z = obj_z.M
normS = norm(w_z)
lp("[STATE] norm(dual state)=", normS)
(isfinite(normS) && normS > 0) || error("dual state uninitialized -- no real Hessian callback fired")

op = cctx.hzz_zc_op
Lb_z = cctx.L
refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])
cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, size(w_z, 1))
refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, w_z; fill_S = true)   # need ZcS filled
nz = n_restriction(op)
cctx.bin_zc_cross = ensure_bin_zc_cross_scratch!(cctx.bin_zc_cross, cctx.D, Lb_z, nz)
bin_zc_ws = cctx.bin_zc_cross

bin_zc_cross_hessian_fill!(bin_zc_ws, cctx.Bidx, cctx.hzz_centered.ZcS)
ZBinCScum_ref = copy(bin_zc_ws.ZBinCScum)
tmin_serial, tmean_serial = timeit(() -> bin_zc_cross_hessian_fill!(bin_zc_ws, cctx.Bidx, cctx.hzz_centered.ZcS))
lp(@sprintf("[H_CZ serial]      min=%.4fs mean=%.4fs", tmin_serial, tmean_serial))

rows = NamedTuple[]
push!(rows, (workers = 1, mode = "serial", t_min_s = tmin_serial, t_mean_s = tmean_serial, maxdiff = 0.0))

for workers in (4, 8, 20)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill_threaded!(bin_zc_ws, cctx.Bidx, cctx.hzz_centered.ZcS; workers = workers))
    maxdiff = maximum(abs.(bin_zc_ws.ZBinCScum .- ZBinCScum_ref))
    speedup = tmin_serial / tmin
    lp(@sprintf("[H_CZ threaded w=%2d] min=%.4fs mean=%.4fs speedup=%.2fx maxdiff=%.3e", workers, tmin, tmean, speedup, maxdiff))
    push!(rows, (workers = workers, mode = "threaded", t_min_s = tmin, t_mean_s = tmean, maxdiff = maxdiff))
end

csvpath = joinpath(D4X, "..", "..", "docs", "HCZ_UPDATED_GATE_2026-07-28.csv")
mkpath(dirname(csvpath))
open(csvpath, "w") do io
    println(io, "workers,mode,t_min_s,t_mean_s,maxdiff_vs_serial,speedup_vs_serial")
    for r in rows
        speedup = tmin_serial / r.t_min_s
        println(io, "$(r.workers),$(r.mode),$(r.t_min_s),$(r.t_mean_s),$(r.maxdiff),$(speedup)")
    end
end
lp("Wrote ", csvpath)
lp("DONE.")
