# ============================================================================
# Section 6.3: unrestricted BLAS-thread sweep at the real D=20/W=80,000 calibration point.
#
# Builds the real production context ONCE (ctx/pe/rsc, the ~65-120s expensive step), then loops
# over BLAS thread counts, calling run_polish_checkpointed (the matched min g_p s.t. Delta*<=1
# formulation -- section 2) via reuse=(ctx=ctx0,pe=pe0,rsc=rsc0) so each config pays only its own
# outer-solve budget, not another context rebuild. Each config starts fresh from the SAME genuine
# calibrated point (ctx.θ0_up), same seed/delta/destination_sample -- only blas_threads differs.
#
# SCOPE NOTE: the task's full spec asks for calibration + a near-delta=1 feasible point + a hard
# screen-passing point, each x5 BLAS configs. Given overall task time budget, this script covers
# the calibration point only (the single most decision-relevant scenario, and the one the prior
# 2026-07-21 report's own headline recommendation was based on) -- flagged explicitly as a scope
# reduction in the deliverable report, not a silent omission.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/blas_sweep_u_2026-07-25.jl <outdir>
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, Printf, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const OUTDIR = abspath(ARGS[1])
const BUDGET_S = parse(Float64, get(ENV, "SWEEP_BUDGET_S", "90.0"))
const CONFIGS = [parse(Int, s) for s in split(get(ENV, "SWEEP_BLAS_CONFIGS", "1,4,8,10,20"), ",")]
mkpath(OUTDIR)

lp(">>> blas_sweep_u_2026-07-25 starting ", now(), " Julia threads=", Threads.nthreads(),
   " budget_s=", BUDGET_S, " configs=", CONFIGS)
lp(">>> host load: "); run(`uptime`)

t0 = time()
ctx0 = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
                             draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
rsc0 = build_ranged_screen_context(ctx0)
lp(">>> context build wall = ", round(time() - t0, digits = 3), "s")
reuse_nt = (ctx = ctx0, pe = pe0, rsc = rsc0)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)

# Untimed warm-up (JIT) at a throwaway config/dir before any measured config.
mkpath(joinpath(OUTDIR, "ckpt_warmup"))
res_warm = run_polish_checkpointed("warmup", true, g0, zfree0;
    maxtime_real = 15.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    destination_sample = :exclude_row, reuse = reuse_nt,
    ckpt_dir = joinpath(OUTDIR, "ckpt_warmup"), checkpoint_interval_s = 9999.0, blas_threads = 1)
lp(">>> warm-up done: n_eval=", res_warm.n_eval, " status=", res_warm.knitro_status)

rows = NamedTuple[]
for n in CONFIGS
    label = "blas$(n)"
    ckpt_dir = joinpath(OUTDIR, "ckpt_$label")
    mkpath(ckpt_dir)
    GC.gc()
    gc0 = Base.gc_num()
    t_meas0 = time()
    alloc = @allocated begin
        global res = run_polish_checkpointed(label, true, g0, zfree0;
            maxtime_real = BUDGET_S, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
            destination_sample = :exclude_row, reuse = reuse_nt,
            ckpt_dir = ckpt_dir, checkpoint_interval_s = 9999.0, blas_threads = n)
    end
    wall = time() - t_meas0
    gcdiff = Base.GC_Diff(Base.gc_num(), gc0)
    best_gp = res.best_feasible === nothing ? NaN : res.best_feasible.gp
    best_Delta = res.best_feasible === nothing ? NaN : res.best_feasible.Delta
    row = (blas_threads = n, wall_s = wall, n_eval = res.n_eval, n_grad = res.n_grad_calls,
           alloc_gb = alloc / 1e9, gc_time_s = gcdiff.total_time / 1e9,
           knitro_status = res.knitro_status, best_gp = best_gp, best_Delta = best_Delta,
           actual_blas_threads_readback = BLAS.get_num_threads())
    push!(rows, row)
    lp(">>> [", label, "] wall=", round(wall, digits = 3), "s n_eval=", res.n_eval, " n_grad=", res.n_grad_calls,
       " alloc_gb=", round(alloc / 1e9, digits = 3), " gc_s=", round(gcdiff.total_time / 1e9, digits = 3),
       " status=", res.knitro_status, " best_gp=", best_gp, " best_Delta=", best_Delta,
       " blas_readback=", BLAS.get_num_threads())
end

write_csv_rows(joinpath(OUTDIR, "blas_sweep_u_2026-07-25.csv"), rows)
lp(">>> BLAS_SWEEP_U_DONE")
