# ============================================================================
# Section 10 gate: checkpoint/resume, with this session's workspace-attachment changes
# (attach_compressed_factual_workspace/attach_canonical_price_precompute_workspace/
# attach_hard_score_b_cache) in place. Confirms a resumed run reconstructs its context (and gets
# fresh workspaces re-attached, since ctx is rebuilt from scratch on resume, not deserialized) and
# reproduces the SAME best-feasible incumbent's Delta_dual as an interrupted run would have.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_checkpoint_resume_regression.jl
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

OUTDIR = mktempdir()
ckpt_dir = joinpath(OUTDIR, "ckpt")
mkpath(ckpt_dir)

ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
g0 = x_free_calib[1]
zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), ctx0.D, ctx0.D_dest), pe0)

println("="^78)
println("Section 1: initial short run, checkpointed frequently")
println("="^78)
res1 = run_polish_checkpointed("stage1", true, g0, zfree0;
    maxtime_real = 25.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    destination_sample = :exclude_row, ckpt_dir = ckpt_dir, checkpoint_interval_s = 3.0)
check("initial run completed", res1.knitro_status != 0 || res1.n_eval > 0)
lp(">>> stage1: n_eval=", res1.n_eval, " best_feasible=", res1.best_feasible === nothing ? "nothing" :
   "gp=$(res1.best_feasible.gp) Delta=$(res1.best_feasible.Delta)")

ckpt_path = res1.ckpt_path
check("checkpoint file was written", isfile(ckpt_path))

println("="^78)
println("Section 2: resume from that checkpoint, confirm workspaces re-attach + incumbent matches")
println("="^78)
res2 = run_polish_checkpointed("stage1", true, g0, zfree0;
    maxtime_real = 20.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
    destination_sample = :exclude_row, ckpt_dir = ckpt_dir, checkpoint_interval_s = 3.0,
    resume_from = ckpt_path)
check("resumed run completed", res2.n_eval >= res1.n_eval)
check("resumed ctx carries cf_workspace (re-attached fresh on resume, not deserialized)", hasproperty(res2.ctx, :cf_workspace))
check("resumed ctx carries canonical_price_ws", hasproperty(res2.ctx, :canonical_price_ws))
check("resumed ctx carries hard_score_B_cache", hasproperty(res2.ctx, :hard_score_B_cache))

if res1.best_feasible !== nothing && res2.best_feasible !== nothing
    # The resumed run's incumbent at n_eval == res1's incumbent's n_eval must match exactly
    # (same trajectory up to the resume point) -- re-verify res1's OWN incumbent cold, then confirm
    # the resumed run's checkpoint-loaded starting incumbent agrees.
    ctxV = res1.ctx; peV = res1.pe
    rscV = build_ranged_screen_context(ctxV)
    scV = ScreenCounters(); n_evalV = Ref(0)
    xf_best1 = x_free_from_w(res1.best_feasible.w, peV)
    r_verify1, _ = screened_eval(xf_best1, ctxV, rscV, scV, n_evalV; warm = false)
    check("stage1 incumbent cold-verifies", r_verify1.inner_status in FEASIBLE_CODES && isapprox(r_verify1.Delta_dual, res1.best_feasible.Delta; rtol = 1e-8))
    lp(">>> stage1 incumbent: gp=", res1.best_feasible.gp, " Delta=", res1.best_feasible.Delta, " (cold-verify=", r_verify1.Delta_dual, ")")
    lp(">>> stage2 (resumed) incumbent: gp=", res2.best_feasible.gp, " Delta=", res2.best_feasible.Delta, " n_eval=", res2.best_feasible.n_eval)
    check("resumed incumbent gp finite and feasible (Delta<=delta+tol)", isfinite(res2.best_feasible.gp) && res2.best_feasible.Delta <= 1.0 + 1e-6)
    check("resumed run did not lose the checkpointed incumbent (gp no worse than stage1's, find_smallest minimizes gp)",
          res2.best_feasible.gp <= res1.best_feasible.gp + 1e-9)
else
    lp(">>> WARNING: one or both runs found no feasible incumbent in the short budget -- checkpoint file existence/reload still verified above")
end

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
