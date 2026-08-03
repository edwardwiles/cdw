# H_ZZ backend x BLAS/Julia-thread complete-inner-solve sweep worker (2026-07-29).
#
# One (backend, OPENBLAS_NUM_THREADS, JULIA_NUM_THREADS) combination per OS process, per the task's
# explicit "run separate processes with fixed OPENBLAS_NUM_THREADS/JULIA_NUM_THREADS, never
# BLAS.set_num_threads inside a callback" instruction. Deliberately does NOT call
# BLAS.set_num_threads or override ZC_GRAM_THREADED_WORKERS_DEFAULT/CROSS_HESSIAN_WORKERS_DEFAULT --
# both already resolve from Threads.nthreads() at process start (resolve_cross_hessian_workers_default),
# so launching with `-t N` reproduces exactly what production would do at that thread count.
#
# Real D=20/W=100,000 cm_meanzc, run_cm_upper_checkpointed, draw_seed=20260719, delta=1.0,
# maxtime_real=60.0 (the task's own "60-second outer smoke" tier -- a genuine KNITRO outer solve,
# not truncated before convergence at this scale per the prior post-fix bake-off).
#
# Usage: OPENBLAS_NUM_THREADS=<N> OMP_NUM_THREADS=<N> julia --project=<repo-root> -t <M> \
#            hzz_thread_sweep_worker_2026-07-29.jl <backend> <results_csv_path>
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

length(ARGS) >= 2 || error("usage: julia hzz_thread_sweep_worker_2026-07-29.jl <backend> <results_csv_path>")
const BACKEND = Symbol(ARGS[1])
const CSVPATH = ARGS[2]
const W = 100_000
const DELTA = 1.0
const MAXTIME = 60.0
const OUTROOT = joinpath(D4X, "..", "..", "results", "hzz_thread_sweep_2026-07-29")
mkpath(OUTROOT)

const OPENBLAS_ENV = get(ENV, "OPENBLAS_NUM_THREADS", "unset")
const JULIA_ENV = get(ENV, "JULIA_NUM_THREADS", "unset")
lp("[$BACKEND] launch: OPENBLAS_NUM_THREADS(env)=", OPENBLAS_ENV, " JULIA_NUM_THREADS(env)=", JULIA_ENV,
   " Threads.nthreads()=", Threads.nthreads(), " BLAS.get_num_threads()=", BLAS.get_num_threads())

function cmzc_w0(ctx0, pe0, theta0, xy0)
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    D = ctx0.D
    gp_calib = x_free_calib[1]
    z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, ctx0.D_dest)), pe0)
    a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
    w_a_calib = vcat(gp_calib, a_calib)
    return vcat(w_a_calib, log.(Float64.(factorial.(1:1))))
end

t_setup0 = time()
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
w0 = cmzc_w0(ctx0, pe0, theta0, xy0)
lp("[$BACKEND] setup done in ", @sprintf("%.1f", time() - t_setup0), "s")

ZC_GRAM_BACKEND_DEFAULT[] = BACKEND
run_id = "sweep_$(BACKEND)_ob$(OPENBLAS_ENV)_jt$(JULIA_ENV)_$(getpid())"

function do_solve(w0, run_id)
    try
        result = run_cm_upper_checkpointed(w0;
            W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal,
            probs = nested_grid_sequence([10, 20, 50])[50],
            cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
            ckpt_dir = OUTROOT, run_id = run_id, label = run_id,
            checkpoint_interval_s = 3600.0, maxtime_real = MAXTIME, verbose = false)
        return result, ""
    catch e
        return nothing, sprint(showerror, e)
    end
end

t0 = time()
result, errmsg = do_solve(w0, run_id)
wall = time() - t0

status_str = "EXCEPTION"
n_eval = -1; n_grad = -1; kappa_val = NaN
if result !== nothing
    status_str = string(result.knitro_status)
    n_eval = result.n_eval
    n_grad = result.n_grad
    kappa_val = hasproperty(result, :kappa) ? result.kappa : NaN
    lp("[$BACKEND] wall=", @sprintf("%.1f", wall), "s status=", status_str,
       " n_eval=", n_eval, " n_grad=", n_grad, " kappa=", kappa_val)
else
    lp("[$BACKEND] EXCEPTION after wall=", @sprintf("%.1f", wall), "s: ", errmsg[1:min(400, length(errmsg))])
end

header_needed = !isfile(CSVPATH)
open(CSVPATH, "a") do io
    if header_needed
        write(io, "backend,openblas_threads_env,julia_threads_env,nthreads_actual,blas_threads_actual,wall_s,status,n_eval,n_grad,kappa,pid\n")
    end
    write(io, "$(BACKEND),$(OPENBLAS_ENV),$(JULIA_ENV),$(Threads.nthreads()),$(BLAS.get_num_threads()),$(wall),$(status_str),$(n_eval),$(n_grad),$(kappa_val),$(getpid())\n")
end
lp("[$BACKEND] appended row to ", CSVPATH)
lp("[$BACKEND] DONE.")
