# Part B: H_ZZ backend bake-off via the real checkpointed driver (2026-07-29).
#
# Reproduces the exact setup that originally triggered the reproducible :blas_gemm failure
# (docs/HZZ_BLAS_GEMM_CORRECTNESS_FINDING_2026-07-28.md, evidence pulled from Dropbox): real
# D=20/W=100,000, cm_meanzc, run_cm_upper_checkpointed, draw_seed=20260719, delta=1.0,
# OPENBLAS_NUM_THREADS=1, run alone. Loops over ALL SIX H_ZZ backends automatically in one process
# (sequential, not concurrent -- avoids any KNITRO-concurrency confound), each a short maxtime_real
# smoke, capturing full inner-solve status/iteration/kappa. Adapted from the already-merged
# hzz_resource_gate_worker_2026-07-28.jl (same w0 construction, same real driver entry points) --
# NOT a direct archC_meanzc_base_state/archC_meanzc_verified_state call (see
# docs/PREEXISTING_DIRECT_CALL_FAILURE_2026-07-29.md for why that path is currently broken on
# clean production HEAD, independent of this task).
#
# Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=<repo-root> -t 20 \
#            full_aod_diag/d4_exact/hzz_backend_direct_bakeoff_2026-07-29.jl [FAMILY]
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
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(D4X, f))
end
using Random, Printf, LinearAlgebra, Statistics, Dates
lp(xs...) = (println(xs...); flush(stdout))

const FAMILY = length(ARGS) >= 1 ? ARGS[1] : "cm_meanzc"
const W = 100_000
const DELTA = 1.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hzz_backend_bakeoff_2026-07-29", FAMILY)
mkpath(OUTROOT)

BLAS.set_num_threads(1)
lp("[$FAMILY] Threads.nthreads()=", Threads.nthreads(), " BLAS threads=", BLAS.get_num_threads())

function originzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    return vcat(w_a_calib, log.(nu0))
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

# built once, shared across every backend run (theta-fixed, draw_seed-fixed w0 construction)
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
w0 = FAMILY == "origin_zc" ? originzc_w0(ctx0, pe0, theta0, xy0) : cmzc_w0(ctx0, pe0, theta0, xy0)

const BACKENDS = length(ARGS) >= 2 ? Symbol.(split(ARGS[2], ",")) : [:reference, :centered_syrk, :blas_syrk, :blas_gemm, :threaded_packed]
results = NamedTuple[]
for backend in BACKENDS
    ZC_GRAM_BACKEND_DEFAULT[] = backend
    ZC_GRAM_THREADED_WORKERS_DEFAULT[] = backend === :threaded_packed ? 20 : 1
    lp("\n=== FAMILY=$FAMILY backend=$backend ===")
    t0 = time()
    local result, errmsg
    errmsg = ""
    try
        if FAMILY == "origin_zc"
            result = run_originzc_upper_checkpointed(w0;
                W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
                distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
                ckpt_dir = OUTROOT, run_id = "bakeoff_$(FAMILY)_$(backend)", label = "bakeoff_$(FAMILY)_$(backend)",
                checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = false)
        else
            result = run_cm_upper_checkpointed(w0;
                W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
                probs = nested_grid_sequence([10, 20, 50])[50],
                cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
                ckpt_dir = OUTROOT, run_id = "bakeoff_$(FAMILY)_$(backend)", label = "bakeoff_$(FAMILY)_$(backend)",
                checkpoint_interval_s = 3600.0, maxtime_real = 25.0, verbose = false)
        end
    catch e
        errmsg = sprint(showerror, e)
        result = nothing
    end
    wall = time() - t0
    if result === nothing
        lp("[RESULT $FAMILY|$backend] EXCEPTION after wall=", @sprintf("%.1f", wall), "s: ", errmsg[1:min(400, length(errmsg))])
        push!(results, (family = FAMILY, backend = backend, wall = wall, status = "EXCEPTION", n_eval = -1, n_grad = -1, kappa = NaN, errmsg = errmsg))
    else
        status_str = string(result.knitro_status)
        kappa_val = hasproperty(result, :kappa) ? result.kappa : NaN
        lp("[RESULT $FAMILY|$backend] wall=", @sprintf("%.1f", wall), "s status=", status_str,
           " n_eval=", result.n_eval, " n_grad=", result.n_grad, " kappa=", kappa_val)
        push!(results, (family = FAMILY, backend = backend, wall = wall, status = status_str, n_eval = result.n_eval, n_grad = result.n_grad, kappa = kappa_val, errmsg = ""))
    end
end

csvpath = joinpath(D4X, "..", "..", "docs", "HZZ_BACKEND_BAKEOFF_2026-07-29_$(FAMILY).csv")
mkpath(dirname(csvpath))
open(csvpath, "w") do io
    write(io, "family,backend,wall_s,status,n_eval,n_grad,kappa\n")
    for r in results
        write(io, "$(r.family),$(r.backend),$(r.wall),$(r.status),$(r.n_eval),$(r.n_grad),$(r.kappa)\n")
    end
end
lp("\nWrote ", csvpath)
lp("\n=== SUMMARY ===")
for r in results
    lp(rpad(string(r.backend), 18), rpad(r.status, 14), " n_eval=", rpad(string(r.n_eval), 6), " n_grad=", rpad(string(r.n_grad), 6), " kappa=", r.kappa)
end
lp("DONE.")
