# Live verification (concurrent-session finding, 2026-07-22): confirms the direction-split gp
# box removal fixes the boundary-degenerate KNITRO presolve stall at a real calibration start
# point. Nothing in production is modified by this script.
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Printf
fctx = build_fullA_context(W = 80000, δ = 2.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
D = fctx.ctx.D
g0 = fctx.ctx.θ0_up[3+D]
Aod_real = reshape(fctx.ctx.θ0_up[fctx.ctx.Aod_offset+1:fctx.ctx.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), fctx.pe)
ckpt = mktempdir()
t0 = time()
res = run_polish_checkpointed("verify_fix", true, g0, zfree0; maxtime_real = 60.0,
    W_in = 80000, delta_in = 2.0, ckpt_dir = ckpt, checkpoint_interval_s = 30.0, reuse = fctx,
    price_cache_backend = :cplus)
@printf "DONE wall=%.1fs n_eval=%d knitro_status=%d kappa=%s\n" (time()-t0) res.n_eval res.knitro_status string(res.kappa)
