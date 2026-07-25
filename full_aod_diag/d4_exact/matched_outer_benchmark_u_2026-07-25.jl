# ============================================================================
# Matched-benchmark harness, UNRESTRICTED family (section 2 of
# PRODUCTION_ALLOCATION_FIXES_PORT_REPORT_2026-07-25.md's requirements).
#
# The 2026-07-25 wall-clock audit's own unrestricted run used run_profile_checkpointed (the
# fixed-g_p, minimum-Delta profile routine), while its CM run used run_cm_upper_checkpointed
# (the direct constrained upper-bound problem: min_{A,g_p} g_p s.t. Delta*(A,g_p) <= 1). That is
# NOT a matched scientific comparison -- the two families were solving different outer problems.
# Every production before/after gate in this porting task must instead use run_polish_checkpointed
# here for the unrestricted family, i.e. the SAME direct min g_p s.t. Delta*<=1 formulation as
# run_cm_upper_checkpointed (matched_outer_benchmark_cm_2026-07-25.jl), from the SAME genuine
# calibrated point (ctx.θ0_up -- see CLAUDE.md: A_od==1/zfree==0 is NOT this point, do not
# construct or compare against that reparameterization reference by mistake), with the SAME
# explicit outer algorithm (knitro_outer_algorithm.jl's set_production_outer_algorithm!, applied
# automatically inside run_polish_checkpointed/run_cm_upper_checkpointed as of this port --
# neither family relies on algorithm=auto).
#
# Config matches section 11 exactly: D=20, D_dest=19 (destination_sample=:exclude_row), W=80000,
# seed=20260719, delta=1, calibrated start, 20 Julia threads, one process at a time, 300s budget
# (all overridable via ENV for shorter smoke runs).
#
# ADDITIVE/DIAGNOSTIC ONLY: calls the production entry point (run_polish_checkpointed)
# unmodified in behavior.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=<n> OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/matched_outer_benchmark_u_2026-07-25.jl <outdir>
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))

const OUTDIR = abspath(ARGS[1])
const T_WARMUP = parse(Float64, get(ENV, "BENCH_WARMUP_S", "20.0"))
const T_MEASURED = parse(Float64, get(ENV, "BENCH_MEASURED_S", "300.0"))
const W = parse(Int, get(ENV, "BENCH_W", "80000"))
const DELTA = parse(Float64, get(ENV, "BENCH_DELTA", "1.0"))
const DRAW_SEED = parse(Int, get(ENV, "BENCH_SEED", "20260719"))
const DESTINATION_SAMPLE = Symbol(get(ENV, "BENCH_DESTINATION_SAMPLE", "exclude_row"))
mkpath(OUTDIR)
mkpath(joinpath(OUTDIR, "ckpt_warmup"))
mkpath(joinpath(OUTDIR, "ckpt_measured"))

lp(">>> matched_outer_benchmark_u_2026-07-25 starting ", now())
lp(">>> Julia threads=", Threads.nthreads(), " OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", "unset"),
   " OMP_NUM_THREADS=", get(ENV, "OMP_NUM_THREADS", "unset"))
lp(">>> host load (uptime): "); run(`uptime`)
lp(">>> outer algorithm=", PRODUCTION_OUTER_ALGORITHM, " (Interior/CG) hessopt=", PRODUCTION_OUTER_HESSOPT, " (L-BFGS) -- see knitro_outer_algorithm.jl")

# ---------------------------------------------------------------------------
# Context build -- genuine calibrated start via ctx.θ0_up (NOT zfree=0, see CLAUDE.md).
# ---------------------------------------------------------------------------
t_setup0 = time()
ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = DESTINATION_SAMPLE)
print_active_layout_banner(ctx0, "unrestricted")
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), D, Ddest), pe0)
t_setup = time() - t_setup0
lp(">>> context build wall = ", round(t_setup, digits = 3), "s  D=", D, " Ddest=", Ddest,
   " n_free=", length(zfree0) + 1, " g0=", g0, " ||zfree0||=", norm(zfree0))

# ---------------------------------------------------------------------------
# Warm-up at the calibrated start (throwaway ckpt dir) -- pays first-call JIT/compile cost
# before the measured window. NOTE the argument order: run_polish_checkpointed's positional
# signature is (label, find_smallest_in, g_start_in, zfree_start_in) -- find_smallest comes
# BEFORE g_start here, the reverse of run_profile_checkpointed's (label, g_in, find_smallest_in,
# zfree_start_in). Mixing these up silently swaps a Float64 into a Bool slot.
# ---------------------------------------------------------------------------
t_warm0 = time()
res_warm = run_polish_checkpointed("warmup", true, g0, zfree0;
    maxtime_real = T_WARMUP, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = joinpath(OUTDIR, "ckpt_warmup"), checkpoint_interval_s = 9999.0,
    destination_sample = DESTINATION_SAMPLE)
t_warm = time() - t_warm0
lp(">>> warm-up wall = ", round(t_warm, digits = 3), "s  n_eval=", res_warm.n_eval,
   " n_grad=", res_warm.n_grad_calls, " status=", res_warm.knitro_status)

prof_reset!()
GC.gc()
gc_num_before = Base.gc_num()

# ---------------------------------------------------------------------------
# MEASURED outer solve: min g_p s.t. Delta*(A,g_p) <= 1, from the genuine calibrated point.
# ---------------------------------------------------------------------------
t_meas0 = time()
alloc_meas = @allocated begin
    global res_meas = run_polish_checkpointed("measured", true, g0, zfree0;
        maxtime_real = T_MEASURED, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = joinpath(OUTDIR, "ckpt_measured"), checkpoint_interval_s = 30.0,
        destination_sample = DESTINATION_SAMPLE)
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
   " n_eval=", res_meas.n_eval, " n_grad=", res_meas.n_grad_calls,
   " status=", res_meas.knitro_status, " peak_rss_kb=", peak_rss_kb)
lp(">>> screens(pw/wt/wn/env/wr/sn/pass)=", res_meas.screen_counts)
lp(">>> best_feasible=", res_meas.best_feasible === nothing ? "nothing" :
    "gp=$(res_meas.best_feasible.gp) Delta=$(res_meas.best_feasible.Delta) n_eval=$(res_meas.best_feasible.n_eval) t=$(res_meas.best_feasible.t_elapsed)")

# ---------------------------------------------------------------------------
# Cold-verify the best incumbent: re-evaluate at exactly that w with warm=false.
# ---------------------------------------------------------------------------
t_verify = NaN; verify_ok = false; verify_Delta = NaN
if res_meas.best_feasible !== nothing
    ctxV = res_meas.ctx; peV = res_meas.pe
    rscV = build_ranged_screen_context(ctxV)
    scV = ScreenCounters(); n_evalV = Ref(0)
    xf_best = x_free_from_w(res_meas.best_feasible.w, peV)
    t_verify0 = time()
    r_verify, meta_verify = screened_eval(xf_best, ctxV, rscV, scV, n_evalV; warm = false)
    t_verify = time() - t_verify0
    verify_ok = r_verify.inner_status in FEASIBLE_CODES && isfinite(r_verify.Delta_dual) && r_verify.Delta_dual <= DELTA + 1e-6
    verify_Delta = r_verify.Delta_dual
    lp(">>> COLD-VERIFY best incumbent: wall=", round(t_verify, digits = 3), "s status=", r_verify.inner_status,
       " Delta_dual=", verify_Delta, " (vs reported ", res_meas.best_feasible.Delta, ", diff=",
       abs(verify_Delta - res_meas.best_feasible.Delta), ") ok=", verify_ok)
end

# ---------------------------------------------------------------------------
# Dump prof_summary(), trace, screen counts, top-level result to disk.
# ---------------------------------------------------------------------------
rows = prof_summary()
write_csv_rows(joinpath(OUTDIR, "prof_summary_u.csv"), rows)
write_csv_rows(joinpath(OUTDIR, "trace_u.csv"),
    NamedTuple[(idx = t.idx, t_elapsed = t.t_elapsed, gp = t.gp, Delta_dual = t.Delta_dual, inner_status = t.inner_status,
      feasible = t.feasible) for t in res_meas.trace])
write_csv_rows(joinpath(OUTDIR, "screen_rejections_u.csv"),
    NamedTuple[(stage = r.stage, o = r.o, d = r.d, n_eval = r.n_eval) for r in res_meas.screen_rejections])

open(joinpath(OUTDIR, "summary_u.txt"), "w") do io
    println(io, "RUN U (unrestricted, matched min g_p s.t. Delta*<=1 formulation) summary -- ", now())
    println(io, "outer_algorithm=", PRODUCTION_OUTER_ALGORITHM, " outer_hessopt=", PRODUCTION_OUTER_HESSOPT)
    println(io, "setup_wall_s=", t_setup)
    println(io, "warmup_wall_s=", t_warm, " warmup_n_eval=", res_warm.n_eval, " warmup_n_grad=", res_warm.n_grad_calls)
    println(io, "measured_wall_s=", t_meas)
    println(io, "measured_alloc_bytes=", alloc_meas)
    println(io, "measured_alloc_gb=", alloc_meas / 1e9)
    println(io, "measured_gc_time_s=", gc_diff.total_time / 1e9)
    println(io, "measured_gc_count=", gc_diff.pause)
    println(io, "peak_rss_kb=", peak_rss_kb)
    println(io, "n_eval=", res_meas.n_eval)
    println(io, "n_grad=", res_meas.n_grad_calls)
    println(io, "n_checkpoint_reuse_hits=", res_meas.n_checkpoint_reuse_hits)
    println(io, "knitro_status=", res_meas.knitro_status, " (", res_meas.native_outer_diag.status_name, ")")
    println(io, "native_outer_iters=", res_meas.native_outer_diag.n_iters)
    println(io, "screen_counts=", res_meas.screen_counts)
    println(io, "best_gp=", res_meas.best_feasible === nothing ? "nothing" : res_meas.best_feasible.gp)
    println(io, "best_Delta=", res_meas.best_feasible === nothing ? "nothing" : res_meas.best_feasible.Delta)
    println(io, "best_n_eval=", res_meas.best_feasible === nothing ? "nothing" : res_meas.best_feasible.n_eval)
    println(io, "best_t_elapsed=", res_meas.best_feasible === nothing ? "nothing" : res_meas.best_feasible.t_elapsed)
    println(io, "cold_verify_wall_s=", t_verify)
    println(io, "cold_verify_Delta_dual=", verify_Delta)
    println(io, "cold_verify_ok=", verify_ok)
    println(io, "avg_alloc_bytes_per_eval=", res_meas.n_eval > 0 ? alloc_meas / res_meas.n_eval : NaN)
    println(io, "avg_alloc_bytes_per_grad=", res_meas.n_grad_calls > 0 ? alloc_meas / res_meas.n_grad_calls : NaN)
end

serialize(joinpath(OUTDIR, "res_meas_u.jls"), (best_feasible = res_meas.best_feasible, trace = res_meas.trace,
    screen_counts = res_meas.screen_counts, knitro_status = res_meas.knitro_status,
    n_eval = res_meas.n_eval, n_grad_calls = res_meas.n_grad_calls, wall_ext = res_meas.wall_ext))

lp(">>> RUN_U_DONE")
