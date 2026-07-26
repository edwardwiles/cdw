# ============================================================================
# port/shared-winner-pair-core-hessian-production-2026-07-25, final gate task §5:
# real outer-loop matched A/B for origin-ZC, dense-reference H_EE vs the shared
# exact winner-pair H_EE, through the REAL production driver
# (run_originzc_upper_checkpointed) unmodified. Adapted directly from the
# restricted-immutable-workspace port's own origin-ZC A/B harness
# (restricted_workspace_outer_ab_originzc.jl) -- same K_mean=K_pair=1, delta=1,
# W=80000, seed=20260719, destination_sample=:exclude_row, calibrated start,
# same explicit outer algorithm/screens/cache/checkpoint/supervisor settings.
# The ONLY difference between the two arms is which global Ref
# (ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[]) `build_originzc_core_hess_ctx`
# resolves H_EE's backend from -- set once before calling the real driver,
# same discipline as matched_outer_benchmark_cm_2026-07-25.jl.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia \
#   --project=. -t 20 full_aod_diag/d4_exact/winner_pair_outer_ab_originzc_2026-07-25.jl \
#   <dense|winner_pair> <budget_s> <ckpt_dir> <label>
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_checkpoint.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_originzc_cplus.jl", "cm_originzc_config.jl", "cm_originzc_checkpoint.jl",
          "cm_checkpoint_fingerprint.jl", "direction_bounds.jl", "blas_thread_policy.jl", "production_backend_manifest.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Statistics, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const VARIANT = ARGS[1]
VARIANT in ("dense", "winner_pair") || error("VARIANT must be dense|winner_pair, got $VARIANT")
const BUDGET = parse(Float64, ARGS[2])
const CKPT_DIR = abspath(ARGS[3])
const LABEL = ARGS[4]
mkpath(CKPT_DIR)

ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[] = VARIANT == "dense" ? :dense_reference : :exact_winner_pair_parallel
lp(">>> VARIANT=", VARIANT, " -> ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[]=", ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT[],
   " workers=", ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT[])

const K = 1
const DELTA = 1.0
const W = 80_000
const DRAW_SEED = 20260719
const DESTINATION_SAMPLE = :exclude_row

lp("="^100)
lp("Origin-ZC winner-pair matched outer A/B: variant=", VARIANT, " K_mean=K_pair=", K, " delta=", DELTA,
   " budget=", BUDGET, "s label=", LABEL, " julia_threads=", Threads.nthreads(),
   " OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", "unset"), "  ", Dates.now())
lp("="^100)

reset_core_hessian_counters!()

t_setup0 = time()
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
    destination_sample = DESTINATION_SAMPLE)
pe = build_pivot_elimination(ctx)
D = ctx.D
t_setup = time() - t_setup0

gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*_ctx_ddest(ctx)], D, _ctx_ddest(ctx)))
zfree0 = pivot_reduce(z0, pe)

layout = OriginByPowerLayout(D, K, K)
nu0 = Vector{Float64}(undef, n_eta(layout))
for k in 1:K
    Uk = ctx.U .^ k
    for o in 1:D
        nu0[target_index(layout, o, k)] = mean(@view Uk[:, o])
    end
end
eta0 = log.(nu0)
w0 = vcat(gp0, zfree0, eta0)
lp(">>> context build wall=", round(t_setup, digits = 3), "s D=", D, " D_dest=", _ctx_ddest(ctx),
   " start point: gp0=", gp0, " n_eta=", length(eta0), " Delta_target(delta)=", DELTA)

manifest = resolve_origin_zc_manifest(; octx = nothing, blas_threads = nothing)
lp(">>> configured (pre-solve) manifest: core_hessian_backend=", VARIANT == "dense" ? :dense_reference : :exact_winner_pair_parallel,
   " core_hessian_workers=", ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT[], " core_hessian_worker_policy=", core_hessian_worker_policy_label())

GC.gc()
gc_num_before = Base.gc_num()
t_meas0 = time()
alloc_meas = @allocated begin
    global res = run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
        maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "originzc_winnerpair_ab_$(VARIANT)_$(LABEL)",
        label = "$(VARIANT)_$(LABEL)", checkpoint_interval_s = 30.0, cm_gradient_backend = :cplus,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = K, K_pair = K,
        power_target_layout = :origin_by_power, destination_sample = DESTINATION_SAMPLE)
end
t_meas = time() - t_meas0
gc_num_after = Base.gc_num()
gc_diff = Base.GC_Diff(gc_num_after, gc_num_before)
peak_rss_kb = try
    parse(Int, split(read(`grep VmHWM /proc/self/status`, String))[2])
catch
    -1
end

lp()
lp("A/B RESULT variant=", VARIANT, " knitro_status=", res.knitro_status, " wall=", round(t_meas, digits = 3),
   "s alloc_bytes=", alloc_meas, " (", round(alloc_meas / 1e9, digits = 3), " GB) gc_time_s=",
   round(gc_diff.total_time / 1e9, digits = 3), " gc_count=", gc_diff.pause,
   " peak_rss_kb=", peak_rss_kb, " n_eval=", res.n_eval, " n_grad=", res.n_grad, " kappa=", res.kappa)

print_core_hessian_counters()
counters_nt = resolve_core_hessian_counters_manifest()

t_verify = NaN; verify_ok = false; verify_Delta = NaN
if res.best === nothing
    lp("NO FEASIBLE INCUMBENT FOUND in this budget.")
else
    lp("best_feasible: gp=", res.best.gp, " Delta=", res.best.Delta, " n_eval=", res.best.n_eval, " t=", res.best.t)
    D2_econ = length(w0) - n_eta(layout)
    wbest = res.best.w
    xf_best = x_free_from_w(wbest[1:D2_econ], pe)
    νfull_best = exp.(wbest[D2_econ+1:end])
    pcx = build_originzc_production_context(ctx, CS, layout)
    t_verify0 = time()
    _, base_cold, verify_cold = cm_originzc_production_value_verified(xf_best, νfull_best, pcx)
    t_verify = time() - t_verify0
    verify_Delta = verify_cold.Delta_dual
    verify_ok = is_verified_success(verify_cold) && isfinite(verify_Delta) && verify_Delta <= DELTA + 1e-6
    lp("COLD-VERIFY: wall=", round(t_verify, digits = 3), "s Delta_dual=", verify_Delta,
       " (checkpoint recorded ", res.best.Delta, ") |diff|=", abs(verify_Delta - res.best.Delta),
       " verified_success=", is_verified_success(verify_cold), " ok=", verify_ok,
       " class=", classify_inner_result(verify_cold))
end

open(joinpath(CKPT_DIR, "summary_originzc_$(VARIANT)_$(LABEL).txt"), "w") do io
    println(io, "RUN origin-ZC winner-pair A/B summary (variant=", VARIANT, ") -- ", now())
    println(io, "K_mean=", K, " K_pair=", K, " delta=", DELTA, " W=", W, " destination_sample=", DESTINATION_SAMPLE)
    println(io, "julia_threads=", Threads.nthreads())
    println(io, "core_hessian_backend_requested=", VARIANT == "dense" ? :dense_reference : :exact_winner_pair_parallel)
    println(io, "core_hessian_workers=", ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT[])
    println(io, "core_hessian_worker_policy=", core_hessian_worker_policy_label())
    println(io, "setup_wall_s=", t_setup)
    println(io, "measured_wall_s=", t_meas)
    println(io, "measured_alloc_bytes=", alloc_meas)
    println(io, "measured_alloc_gb=", alloc_meas / 1e9)
    println(io, "measured_gc_time_s=", gc_diff.total_time / 1e9)
    println(io, "measured_gc_count=", gc_diff.pause)
    println(io, "peak_rss_kb=", peak_rss_kb)
    println(io, "n_eval=", res.n_eval)
    println(io, "n_grad=", res.n_grad)
    println(io, "knitro_status=", res.knitro_status)
    println(io, "kappa=", res.kappa)
    println(io, "best_Delta=", res.best === nothing ? "nothing" : res.best.Delta)
    println(io, "best_n_eval=", res.best === nothing ? "nothing" : res.best.n_eval)
    println(io, "best_t=", res.best === nothing ? "nothing" : res.best.t)
    println(io, "cold_verify_wall_s=", t_verify)
    println(io, "cold_verify_Delta_dual=", verify_Delta)
    println(io, "cold_verify_ok=", verify_ok)
    println(io, "winner_pair_hessian_calls=", counters_nt.winner_pair_hessian_calls)
    println(io, "winner_pair_serial_calls=", counters_nt.winner_pair_serial_calls)
    println(io, "winner_pair_parallel_calls=", counters_nt.winner_pair_parallel_calls)
    println(io, "dense_core_fallback_calls=", counters_nt.dense_core_fallback_calls)
    println(io, "compressed_core_rebuilds=", counters_nt.compressed_core_rebuilds)
end

serialize(joinpath(CKPT_DIR, "res_meas_originzc_$(VARIANT)_$(LABEL).jls"),
    (best_feasible = res.best, trace = res.trace, knitro_status = res.knitro_status,
     n_eval = res.n_eval, n_grad = res.n_grad, wall = t_meas, kappa = res.kappa))

lp()
lp("OUTER A/B DONE (", VARIANT, "/", LABEL, ") -- ", Dates.now())
