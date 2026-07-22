# ============================================================================
# Closure task Phase 3C: validate (not merely assert) the direction-split gp box removal
# (commit a69fb21 on this branch). Real D=20/W=80,000, delta=1.0, one shared context (built
# once via build_fullA_context, reused across all 3 runs -- same efficient pattern
# remediation_b1_verify_direction_box_fix.jl already established for this exact question).
#
# Required checks:
#   1. upper search (find_smallest=true) still minimizes gp -- incumbent gp < g_start=g_F.
#   2. lower search (find_smallest=false) still maximizes gp -- incumbent gp > g_start=g_F.
#   3. the exact (unperturbed) calibration start evaluates successfully for both directions
#      (>=1 real outer eval -- the exact symptom the removed box's boundary-degenerate presolve
#      previously prevented: 0-iteration KN_RC_TIME_LIMIT_INFEAS stall).
#   4. a start point on the "wrong side" of g_F for its own direction (which the OLD
#      validate_gp_in_direction_box gate would have hard-rejected before any solving) is no
#      longer rejected -- run_polish_checkpointed accepts it and proceeds.
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))
using Printf

lp(xs...) = (println(xs...); flush(stdout))
n_pass = Ref(0); n_fail = Ref(0)
check(name, cond) = (cond ? (lp("  PASS: ", name); n_pass[] += 1) : (lp("  FAIL: ", name); n_fail[] += 1))

const BUDGET = 60.0
fctx = build_fullA_context(W = 80000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
ctx = fctx.ctx; pe = fctx.pe
D = ctx.D
Aod_real = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe)
gF = frechet_benchmark_gp(ctx)
gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
lp(">>> g_F (Frechet benchmark / calibration gp) = ", gF, "   box=[", gp_lo, ", ", gp_hi, "]")

lp("\n=== 1/4: UPPER (find_smallest=true) from the exact unperturbed calibration start ===")
ckpt_u = mktempdir()
res_u = run_polish_checkpointed("c31_upper", true, gF, zfree0; maxtime_real = BUDGET,
    W_in = 80000, delta_in = 1.0, ckpt_dir = ckpt_u, checkpoint_interval_s = 200.0,
    reuse = fctx, price_cache_backend = :cplus)
lp("  n_eval=", res_u.n_eval, " knitro_status=", res_u.knitro_status,
   " best=", res_u.best_feasible === nothing ? "nothing" : "gp=$(res_u.best_feasible.gp)")
check("upper run produced at least 1 verified eval", res_u.n_eval >= 1)
check("upper incumbent gp < g_F (minimizing gp, as expected for find_smallest=true)",
      res_u.best_feasible !== nothing && res_u.best_feasible.gp < gF)

lp("\n=== 2/4: LOWER (find_smallest=false) from the exact unperturbed calibration start ===")
fctx_lower = build_fullA_context(W = 80000, δ = 1.0, find_smallest = false, draw_design = :pseudorandom, draw_seed = 20260719)
ckpt_l = mktempdir()
res_l = run_polish_checkpointed("c31_lower", false, gF, zfree0; maxtime_real = BUDGET,
    W_in = 80000, delta_in = 1.0, ckpt_dir = ckpt_l, checkpoint_interval_s = 200.0,
    reuse = fctx_lower, price_cache_backend = :cplus)
lp("  n_eval=", res_l.n_eval, " knitro_status=", res_l.knitro_status,
   " best=", res_l.best_feasible === nothing ? "nothing" : "gp=$(res_l.best_feasible.gp)")
check("lower run produced at least 1 verified eval", res_l.n_eval >= 1)
check("lower incumbent gp > g_F (maximizing gp, as expected for find_smallest=false)",
      res_l.best_feasible !== nothing && res_l.best_feasible.gp > gF)

lp("\n=== 3/4: clean start confirmed for both (no 0-iteration presolve stall) ===")
check("upper run reached >0 outer iterations", res_u.n_eval >= 1)
check("lower run reached >0 outer iterations", res_l.n_eval >= 1)

lp("\n=== 4/4: start point on the OLD split's \"wrong side\" is no longer rejected ===")
# Under the OLD (removed) direction-split box: upper's box was [gp_lo, g_F] -- a start ABOVE
# g_F was the "wrong side" and validate_gp_in_direction_box would hard-error immediately, before
# any KNITRO call. Pick a start strictly between g_F and gp_hi (old-upper-box-violating, but
# inside the actual current full-range box) and confirm run_polish_checkpointed proceeds.
wrong_side_gp = gF + 0.4 * (gp_hi - gF)
lp("  wrong-side (for upper) start gp = ", wrong_side_gp, "  (g_F=", gF, ", gp_hi=", gp_hi, ")")
threw_old_style_rejection = false
res_wrongside = nothing
try
    ckpt_w = mktempdir()
    global res_wrongside = run_polish_checkpointed("c31_upper_wrongside", true, wrong_side_gp, zfree0;
        maxtime_real = 30.0, W_in = 80000, delta_in = 1.0, ckpt_dir = ckpt_w,
        checkpoint_interval_s = 200.0, reuse = fctx, price_cache_backend = :cplus)
catch e
    global threw_old_style_rejection = occursin("direction", lowercase(sprint(showerror, e))) ||
                                        occursin("wrong side", lowercase(sprint(showerror, e)))
    threw_old_style_rejection && lp("  (would-be old-style rejection message): ", sprint(showerror, e))
    threw_old_style_rejection || rethrow()
end
check("a start on the old split's \"wrong side\" is NOT rejected as a direction-box violation",
      !threw_old_style_rejection && res_wrongside !== nothing)

lp("\n>>> RESULT: ", n_pass[], "/", n_pass[] + n_fail[], " checks passed")
n_fail[] == 0 || error("c31_phase3c_direction_box_validation.jl: $(n_fail[]) check(s) FAILED")
