# ============================================================================
# Matched-benchmark harness, CM (flexible common-marginals) family (section 2 of
# PRODUCTION_ALLOCATION_FIXES_PORT_REPORT_2026-07-25.md's requirements).
#
# Companion to matched_outer_benchmark_u_2026-07-25.jl -- see that file's header for why a
# matched comparison requires BOTH families to solve the same direct constrained upper-bound
# problem (min_{A,g_p} g_p s.t. Delta*(A,g_p) <= 1) from the same genuine calibrated point
# (ctx.θ0_up), with the same explicit outer algorithm. CM already used this formulation via
# run_cm_upper_checkpointed in the original 2026-07-25 audit (audit_run_cm_2026-07-25.jl) --
# this port carries that config forward unchanged except for requesting the explicit outer
# algorithm choice (knitro_outer_algorithm.jl) via pin_outer_algorithm=true below, instead of
# leaving run_cm_upper_checkpointed at its own default (the .opt file's algorithm=auto) -- this
# is opt-in for this matched-benchmark harness only, not a change to the driver's own default.
#
# Config matches section 11 exactly: D=20, D_dest=19 (destination_sample=:exclude_row), W=80000,
# seed=20260719, delta=1, L=50, calibrated start, 20 Julia threads, one process at a time, 300s
# budget (all overridable via ENV for shorter smoke runs).
#
# ADDITIVE/DIAGNOSTIC ONLY: calls the production entry point (run_cm_upper_checkpointed)
# unmodified in behavior.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=<n> OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/matched_outer_benchmark_cm_2026-07-25.jl <outdir>
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Dates, Serialization, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

const OUTDIR = abspath(ARGS[1])
const T_WARMUP = parse(Float64, get(ENV, "BENCH_WARMUP_S", "20.0"))
const T_MEASURED = parse(Float64, get(ENV, "BENCH_MEASURED_S", "300.0"))
mkpath(OUTDIR)
mkpath(joinpath(OUTDIR, "ckpt_warmup"))
mkpath(joinpath(OUTDIR, "ckpt_measured"))

lp(">>> matched_outer_benchmark_cm_2026-07-25 starting ", now())
lp(">>> Julia threads=", Threads.nthreads(), " OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", "unset"),
   " OMP_NUM_THREADS=", get(ENV, "OMP_NUM_THREADS", "unset"))
lp(">>> host load (uptime): "); run(`uptime`)
lp(">>> outer algorithm=", PRODUCTION_OUTER_ALGORITHM, " (Interior/CG) hessopt=", PRODUCTION_OUTER_HESSOPT, " (L-BFGS) -- see knitro_outer_algorithm.jl")

const W = parse(Int, get(ENV, "BENCH_W", "80000"))
const L = parse(Int, get(ENV, "BENCH_L", "50"))
const DELTA = parse(Float64, get(ENV, "BENCH_DELTA", "1.0"))
const DRAW_SEED = parse(Int, get(ENV, "BENCH_SEED", "20260719"))
const CM_CONTRASTS = :orthonormal   # approved production contrast basis (2026-07-22 review)
const CM_EXTENSION = Symbol(get(ENV, "BENCH_CM_EXTENSION", "cm_only"))
const CM_GRADIENT_BACKEND = :cplus  # production default
const CM_DESTINATION_SAMPLE = Symbol(get(ENV, "BENCH_DESTINATION_SAMPLE", "exclude_row"))
const CM_HESSIAN_BACKEND = Symbol(get(ENV, "BENCH_CM_HESSIAN_BACKEND", "structured"))
const BLAS_THREADS = haskey(ENV, "BENCH_BLAS_THREADS") ? parse(Int, ENV["BENCH_BLAS_THREADS"]) : nothing   # allocation/Hessian port task §6.3
lp(">>> active Hessian backend=:", CM_HESSIAN_BACKEND, ", contrasts=", CM_CONTRASTS,
   " cm_extension=", CM_EXTENSION, " cm_gradient_backend=", CM_GRADIENT_BACKEND, " destination_sample=", CM_DESTINATION_SAMPLE)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

# ---------------------------------------------------------------------------
# Context build -- genuine calibrated start via ctx.θ0_up (NOT zfree=0, see CLAUDE.md); same
# convention as matched_outer_benchmark_u_2026-07-25.jl.
# ---------------------------------------------------------------------------
t_setup0 = time()
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = CM_DESTINATION_SAMPLE)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))
t_setup = time() - t_setup0
lp(">>> context build wall = ", round(t_setup, digits = 3), "s  D=", D, " Ddest=", Ddest,
   " n_free=", length(w_calib), " g0=", w_calib[1], " ||zfree0||=", norm(w_calib[2:end]))

# ---------------------------------------------------------------------------
# Warm-up at the calibrated start (throwaway ckpt dir).
# ---------------------------------------------------------------------------
t_warm0 = time()
res_warm = run_cm_upper_checkpointed(w_calib;
    W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
    L = L, contrasts = CM_CONTRASTS, probs = probs,
    cm_hessian_backend = CM_HESSIAN_BACKEND, cm_grid_rule = :nested_family,
    maxtime_real = T_WARMUP, ckpt_dir = joinpath(OUTDIR, "ckpt_warmup"),
    run_id = "matched_bench_cm_warmup", label = "warmup", checkpoint_interval_s = 9999.0,
    cm_gradient_backend = CM_GRADIENT_BACKEND, cm_extension = CM_EXTENSION,
    destination_sample = CM_DESTINATION_SAMPLE, verbose = true, blas_threads = BLAS_THREADS,
    pin_outer_algorithm = true)
t_warm = time() - t_warm0
lp(">>> warm-up wall = ", round(t_warm, digits = 3), "s  n_eval=", res_warm.n_eval,
   " n_grad=", res_warm.n_grad, " status=", res_warm.knitro_status)

prof_reset!()
GC.gc()
gc_num_before = Base.gc_num()

# ---------------------------------------------------------------------------
# MEASURED outer solve: min g_p s.t. Delta*(A,g_p) <= 1, from the genuine calibrated point.
# ---------------------------------------------------------------------------
t_meas0 = time()
alloc_meas = @allocated begin
    global res_meas = run_cm_upper_checkpointed(w_calib;
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
        L = L, contrasts = CM_CONTRASTS, probs = probs,
        cm_hessian_backend = CM_HESSIAN_BACKEND, cm_grid_rule = :nested_family,
        maxtime_real = T_MEASURED, ckpt_dir = joinpath(OUTDIR, "ckpt_measured"),
        run_id = "matched_bench_cm_measured", label = "measured", checkpoint_interval_s = 30.0,
        cm_gradient_backend = CM_GRADIENT_BACKEND, cm_extension = CM_EXTENSION,
        destination_sample = CM_DESTINATION_SAMPLE, verbose = true, blas_threads = BLAS_THREADS,
        pin_outer_algorithm = true)
end
t_meas = time() - t_meas0
gc_num_after = Base.gc_num()
gc_diff = Base.GC_Diff(gc_num_after, gc_num_before)
peak_rss_kb = try
    parse(Int, split(read(`grep VmHWM /proc/self/status`, String))[2])
catch
    -1
end

lp(">>> MEASURED RUN DONE wall=", round(t_meas, digits = 3), "s alloc_bytes=", alloc_meas,
   " (", round(alloc_meas / 1e9, digits = 3), " GB)",
   " gc_time_s=", round(gc_diff.total_time / 1e9, digits = 3),
   " n_eval=", res_meas.n_eval, " n_grad=", res_meas.n_grad,
   " status=", res_meas.knitro_status, " peak_rss_kb=", peak_rss_kb)
lp(">>> best=", res_meas.best === nothing ? "nothing" :
    "Delta=$(res_meas.best.Delta) gp=$(res_meas.best.gp) n_eval=$(res_meas.best.n_eval) t=$(res_meas.best.t)")

# ---------------------------------------------------------------------------
# Cold-verify the best incumbent, mirroring cm_cold_verify.jl's own production verification
# pattern: rebuild pcx via build_cm_production_context and call cm_production_value_verified.
# ---------------------------------------------------------------------------
t_verify = NaN; verify_ok = false; verify_Delta = NaN
if res_meas.best !== nothing
    pcxV = build_cm_production_context(ctx0, CS; L = L, contrasts = CM_CONTRASTS, probs = probs)
    w_best = res_meas.best.w
    xf_best = x_free_from_w(w_best, pe0)
    t_verify0 = time()
    K_v, base_v, verify_v = cm_production_value_verified(xf_best, pcxV)
    t_verify = time() - t_verify0
    verify_Delta = verify_v.Delta_dual
    verify_ok = is_verified_success(verify_v) && isfinite(verify_Delta) && verify_Delta <= DELTA + 1e-6
    lp(">>> COLD-VERIFY best incumbent: wall=", round(t_verify, digits = 3), "s Delta_dual=", verify_Delta,
       " (vs reported ", res_meas.best.Delta, ", diff=", abs(verify_Delta - res_meas.best.Delta), ") ok=", verify_ok)
end

# ---------------------------------------------------------------------------
# Dump prof_summary(), trace, top-level result to disk.
# ---------------------------------------------------------------------------
rows = prof_summary()
write_csv_rows(joinpath(OUTDIR, "prof_summary_cm.csv"), rows)
write_csv_rows(joinpath(OUTDIR, "trace_cm.csv"),
    NamedTuple[(idx = t.idx, t = t.t, gp = t.gp, Delta = t.Delta, feasible = t.feasible, verified = t.verified) for t in res_meas.trace])

open(joinpath(OUTDIR, "summary_cm.txt"), "w") do io
    println(io, "RUN CM (flexible common marginals, matched min g_p s.t. Delta*<=1 formulation) summary -- ", now())
    println(io, "outer_algorithm=", PRODUCTION_OUTER_ALGORITHM, " outer_hessopt=", PRODUCTION_OUTER_HESSOPT)
    println(io, "cm_hessian_backend=", CM_HESSIAN_BACKEND)
    println(io, "setup_wall_s=", t_setup)
    println(io, "warmup_wall_s=", t_warm, " warmup_n_eval=", res_warm.n_eval, " warmup_n_grad=", res_warm.n_grad)
    println(io, "measured_wall_s=", t_meas)
    println(io, "measured_alloc_bytes=", alloc_meas)
    println(io, "measured_alloc_gb=", alloc_meas / 1e9)
    println(io, "measured_gc_time_s=", gc_diff.total_time / 1e9)
    println(io, "measured_gc_count=", gc_diff.pause)
    println(io, "peak_rss_kb=", peak_rss_kb)
    println(io, "n_eval=", res_meas.n_eval)
    println(io, "n_grad=", res_meas.n_grad)
    println(io, "knitro_status=", res_meas.knitro_status)
    println(io, "kappa=", res_meas.kappa)
    println(io, "best_Delta=", res_meas.best === nothing ? "nothing" : res_meas.best.Delta)
    println(io, "best_n_eval=", res_meas.best === nothing ? "nothing" : res_meas.best.n_eval)
    println(io, "best_t=", res_meas.best === nothing ? "nothing" : res_meas.best.t)
    println(io, "cold_verify_wall_s=", t_verify)
    println(io, "cold_verify_Delta_dual=", verify_Delta)
    println(io, "cold_verify_ok=", verify_ok)
    println(io, "avg_alloc_bytes_per_eval=", res_meas.n_eval > 0 ? alloc_meas / res_meas.n_eval : NaN)
    println(io, "avg_alloc_bytes_per_grad=", res_meas.n_grad > 0 ? alloc_meas / res_meas.n_grad : NaN)
end

serialize(joinpath(OUTDIR, "res_meas_cm.jls"), (best_feasible = res_meas.best, trace = res_meas.trace,
    knitro_status = res_meas.knitro_status, n_eval = res_meas.n_eval, n_grad = res_meas.n_grad,
    wall = res_meas.wall, kappa = res_meas.kappa))

lp(">>> RUN_CM_DONE")
