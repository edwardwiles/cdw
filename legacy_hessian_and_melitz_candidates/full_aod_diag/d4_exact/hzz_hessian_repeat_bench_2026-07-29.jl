# H_ZZ backend x thread-count benchmark, take 2 (2026-07-29): repeated REAL full-Hessian-assembly
# calls at a FIXED, real, solved point, instead of a full outer KNITRO solve.
#
# Rationale (user-directed correction): the first sweep (hzz_thread_sweep_worker_2026-07-29.jl)
# measured a complete `run_cm_upper_checkpointed` outer solve capped at maxtime_real=60s, which only
# completes ~2-3 outer evaluations in that window. Each evaluation's wall-clock is dominated by the
# inner economic-equilibrium solve, H_EE/H_EC/H_EZ/H_CZ, checkpointing, and outer-Newton
# path-dependence -- none of which H_ZZ touches. H_ZZ is a small slice of that total, so the outer-
# solve signal is mostly noise for isolating ITS effect. This script instead calls the actual
# production Hessian-assembly function (`hessian_cm_structured_v2!`, cm_hessian_architectures.jl --
# the SAME function cb_G! calls every outer iteration, all blocks H_EE..H_ZZ together, real D=20
# data, real thread config) many times in a tight loop at one fixed, genuinely-solved point,
# isolating H_ZZ's backend/thread effect from outer-loop noise while still exercising the real
# code path (so real oversubscription effects, if any, still show up -- this is NOT the same as
# isolated H_ZZ-kernel-only timing, which the project's own prior session already found misleading).
#
# W=80,000 (not 100,000): matches this project's own established, already-gated real-D20 test
# (test_cm_meanzc_hcz_hzz_direct_d20.jl) for obtaining the one-time solved point via
# `archC_meanzc_base_state` -- W=100,000 direct calls to that function are separately known-fragile
# (docs/PREEXISTING_DIRECT_CALL_FAILURE_2026-07-29.md, nStatus=-400 at certain (W,contrasts)
# combos), unrelated to this task. W=80,000 is still this project's own documented floor for
# "any reported production-scale number" (memory: d20-realdata-w-sensitivity, melitz-small-w-
# numerically-finicky) -- not a shortcut. The Hessian-assembly cost this script measures is a
# per-callback cost that does NOT depend on which point NCORE/ncm dimensions come from, only on W
# and the model's structural dimensions, both held fixed and real here.
#
# All backends tested via the SAME persistent `cctx` (flipping `cctx.zc_gram_backend` /
# `cctx.zc_gram_workers` between backends) -- safe now that the shared-scratch ZcS fill-gate bug is
# fixed (fill_S=true unconditionally, production/fullA-exact@4d7b5b5); the prior "build a fresh cctx
# per backend" warning was specifically about that now-fixed bug.
#
# Usage: OPENBLAS_NUM_THREADS=<N> OMP_NUM_THREADS=<N> julia --project=<repo-root> -t <M> \
#            hzz_hessian_repeat_bench_2026-07-29.jl <backend1,backend2,...> <results_csv_path>
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random, Statistics

length(ARGS) >= 2 || error("usage: julia hzz_hessian_repeat_bench_2026-07-29.jl <backend1,backend2,...> <results_csv_path>")
const BACKENDS = Symbol.(split(ARGS[1], ","))
const CSVPATH = ARGS[2]
const W = 80_000
const NREPS = 30
lp(xs...) = (println(xs...); flush(stdout))

const OPENBLAS_ENV = get(ENV, "OPENBLAS_NUM_THREADS", "unset")
const JULIA_ENV = get(ENV, "JULIA_NUM_THREADS", "unset")
lp("launch: OPENBLAS_NUM_THREADS(env)=", OPENBLAS_ENV, " JULIA_NUM_THREADS(env)=", JULIA_ENV,
   " Threads.nthreads()=", Threads.nthreads(), " BLAS.get_num_threads()=", BLAS.get_num_threads(),
   " backends=", BACKENDS)

nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]

lp("Building real D=20 context (W=$W, delta=1.0)..."); t_setup0 = time()
ctx = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]
K_mean, K_pair = 1, 1
ν0 = nu0vec(K_mean)
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
ctx_cm = merge(ctx, (obj = aug.obj_cm,))
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
lp("setup done in ", @sprintf("%.1f", time() - t_setup0), "s")

lp("Solving one real feasible point (archC_meanzc_base_state, W=$W)..."); t_solve0 = time()
base = archC_meanzc_base_state(x_free_calib, ν0, ctx_cm, cctx)
lp("solve done in ", @sprintf("%.1f", time() - t_solve0), "s, inner_status=", base.inner_status)
base.inner_status in (0, -100, -101, -103) || error("archC_meanzc_base_state did not return a feasible point (status=$(base.inner_status)) -- cannot build a representative benchmark point")

n = cctx.NCORE + cctx.ncm
x0 = vcat(base.ζstar, base.λstar)
_archC_prep_for_hessian!(ctx_cm.obj, x0)
h = Vector{Float64}(undef, n * (n + 1) ÷ 2)

header_needed = !isfile(CSVPATH)
open(CSVPATH, "a") do io
    if header_needed
        write(io, "backend,openblas_threads_env,julia_threads_env,nthreads_actual,blas_threads_actual,workers,n_reps,median_s,mean_s,min_s,max_s,pid\n")
    end
    for backend in BACKENDS
        cctx.zc_gram_backend = backend
        workers = backend === :threaded_packed ? Threads.nthreads() : 1
        cctx.zc_gram_workers = workers
        # warmup (JIT + first-touch)
        hessian_cm_structured_v2!(h, ctx_cm.obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)
        times = Vector{Float64}(undef, NREPS)
        for r in 1:NREPS
            t0 = time()
            hessian_cm_structured_v2!(h, ctx_cm.obj, cctx; threaded_bins = cctx.use_threaded_bins, tls = cctx.tls, use_syrk = true)
            times[r] = time() - t0
        end
        med = median(times); avg = mean(times); mn = minimum(times); mx = maximum(times)
        lp("[$backend] workers=$workers median=", @sprintf("%.5f", med), "s mean=", @sprintf("%.5f", avg),
           "s min=", @sprintf("%.5f", mn), "s max=", @sprintf("%.5f", mx), "s (n=$NREPS)")
        write(io, "$(backend),$(OPENBLAS_ENV),$(JULIA_ENV),$(Threads.nthreads()),$(BLAS.get_num_threads()),$(workers),$(NREPS),$(med),$(avg),$(mn),$(mx),$(getpid())\n")
        flush(io)
    end
end
lp("DONE.")
