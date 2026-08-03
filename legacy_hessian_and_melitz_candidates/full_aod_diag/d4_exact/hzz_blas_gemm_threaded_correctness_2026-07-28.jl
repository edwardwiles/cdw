# Direct correctness check: does :blas_gemm produce the SAME H_ZZ (and hence packed Hessian) as
# :reference when cross_hessian_threaded=true (today's real production default)?
#
# Motivation: the D=4 gate (test_threaded_cross_hessian_d4.jl) explicitly sets
# cross_hessian_threaded=false before testing zc_gram_backend variants -- meaning :blas_gemm's
# "bit-exact vs :reference" evidence was NEVER validated with threaded H_EC/H_EZ simultaneously
# engaged. A real D=20 run at cross_hessian_threaded=true + zc_gram_backend=:blas_gemm just failed
# catastrophically (nStatus=-502, blown-up dual state with values ~1e7-1e11) -- this script isolates
# whether that's because :blas_gemm's OUTPUT differs from :reference under this untested
# combination, run via the real public driver (cross_hessian_threaded=true, zc_gram_backend=
# :reference -- i.e. TODAY'S actual safe production default) to get a real converged dual state,
# then recomputes H_ZZ BOTH ways on that SAME frozen state and compares.
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
const OUTROOT = joinpath(D4X, "..", "..", "results", "hzz_blas_gemm_threaded_correctness_2026-07-28")
mkpath(OUTROOT)

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

# Real driver call at TODAY's actual safe defaults (cross_hessian_threaded=true, zc_gram_backend=
# :reference) -- captures a real converged/feasible dual state via the live-pcx stash.
CMZC_LIVE_PCX_STASH[] = nothing
out = joinpath(OUTROOT, "driver_ref"); rm(out; force = true, recursive = true); mkpath(out)
result = run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[50],
    cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
    ckpt_dir = out, run_id = "hzz_correctness_ref", label = "hzz_correctness_ref",
    checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = true)
lp("[DRIVER] status=", result.knitro_status, " n_eval=", result.n_eval, " n_grad=", result.n_grad)
handle = CMZC_LIVE_PCX_STASH[]
handle === nothing && error("no live handle captured -- increase maxtime_real")

cctx = handle.cctx; ctx_cm = handle.ctx_cm; obj_z = ctx_cm.obj
w_z = copy(obj_z.arg2); M_z = obj_z.M
normS = norm(w_z)
lp("[STATE] norm(dual state)=", normS, " (finite&&>0 required for a real captured callback)")
(isfinite(normS) && normS > 0) || error("dual state uninitialized")

op = cctx.hzz_zc_op
refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])
cctx.hzz_centered = ensure_zc_centered_scratch!(cctx.hzz_centered, op, size(w_z, 1))
nx = n_restriction(op)

# --- :reference ---
refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, w_z; fill_S = true)
HZZ_ref = Matrix{Float64}(undef, nx, nx)
zc_restriction_gram!(HZZ_ref, cctx.hzz_centered, op, M_z)
lp("[REFERENCE] norm(HZZ)=", norm(HZZ_ref), " max=", maximum(abs, HZZ_ref))

# --- :blas_gemm, on the SAME frozen w_z/M_z ---
refresh_zc_centered!(cctx.hzz_centered, op, cctx.hzz_zc_ws, w_z; fill_S = false)
cctx.raw_zc_ws = ensure_zc_raw_weighted_workspace!(cctx.raw_zc_ws, op, size(w_z, 1))
refresh_zc_raw_target_vector!(cctx.raw_zc_ws, cctx.hzz_zc_ws, op)
HZZ_gemm = Matrix{Float64}(undef, nx, nx)
zc_gram_dispatch!(HZZ_gemm, :blas_gemm, nothing, op, cctx.raw_zc_ws, w_z, M_z; workers = 1)
lp("[BLAS_GEMM]  norm(HZZ)=", norm(HZZ_gemm), " max=", maximum(abs, HZZ_gemm))

maxdiff = maximum(abs.(HZZ_ref .- HZZ_gemm))
relmaxdiff = maxdiff / max(1e-300, maximum(abs, HZZ_ref))
lp("\n=== RESULT: maxdiff=", maxdiff, "  relative=", relmaxdiff, " ===")
lp(maxdiff < 1e-6 ? "PASS -- :blas_gemm matches :reference on this real threaded-H_EC/H_EZ state" :
   "FAIL -- :blas_gemm DIVERGES from :reference on this real threaded-H_EC/H_EZ state -- likely root cause of the D=20 solve failure")
lp("DONE.")
