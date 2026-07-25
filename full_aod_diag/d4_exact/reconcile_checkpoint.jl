# Reconciles an interrupted matched-comparison arm from its last checkpoint: loads the
# accumulated best_feasible incumbent, reconstructs its outer point, and cold-verifies it via a
# fresh warm=false production evaluation (screened_eval / screened_eval_flexible_A) --
# independent of anything the interrupted outer solve itself cached.
# Usage: julia --project=. reconcile_checkpoint.jl <fixed|flexible> <ckpt_path> <delta>
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_flexible_theta_A.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const ARM = ARGS[1]
const CKPT_PATH = ARGS[2]
const DELTA = parse(Float64, ARGS[3])
const W = 80_000
const DRAW_SEED = 20260719

lp(">>> RECONCILE arm=", ARM, " ckpt=", CKPT_PATH, " delta=", DELTA)

if ARM == "fixed"
    ckpt = load_checkpoint(CKPT_PATH)
    lp("checkpoint: schema=", ckpt.schema, " n_eval=", ckpt.n_eval, " knitro_iter=", ckpt.knitro_iter,
       " wall_elapsed=", ckpt.wall_elapsed, " checkpoint_reason=", ckpt.checkpoint_reason)
    b = ckpt.best_feasible
    b === nothing && error("checkpoint has no best_feasible incumbent -- nothing to reconcile")
    lp("best_feasible: ", b)

    ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = ckpt.find_smallest,
        draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed, destination_sample = ckpt.destination_sample)
    ctx.draw_meta.checksum_uniform == ckpt.draw_checksum_uniform ||
        error("draw checksum mismatch on reconstruction -- refusing to reconcile against a different draw realization")
    pe = build_pivot_elimination(ctx)
    # best_feasible NamedTuple fields (run_polish_checkpointed's own construction):
    # (gp=.., w=.., Delta=.., gravity=.., kkt=.., inner_status=.., t_elapsed=.., n_eval=..)
    w_champ = b.w
    xf = x_free_from_w(w_champ, pe)
    rsc = build_ranged_screen_context(ctx)
    r_cv, _ = screened_eval(xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
    σ = ctx.σ
    κ = 1 - w_champ[1]^(σ / (σ - 1))
    lp(">>> RECONCILED (fixed): gp=", w_champ[1], " kappa=", κ, " cold-verify Delta_dual=", r_cv.Delta_dual,
       " (checkpoint recorded Delta=", b.Delta, ") inner_status=", r_cv.inner_status,
       " gravity=", r_cv.gravity_value, " feasible=", r_cv.Delta_dual <= DELTA + 1e-6)
    lp(">>> RECONCILE_DONE arm=fixed delta=", DELTA, " kappa=", κ, " n_eval=", ckpt.n_eval, " wall=", ckpt.wall_elapsed)
elseif ARM == "flexible"
    ckpt = load_checkpoint_flexA(CKPT_PATH)
    lp("checkpoint: schema=", ckpt.schema, " n_eval=", ckpt.n_eval, " knitro_iter=", ckpt.knitro_iter,
       " wall_elapsed=", ckpt.wall_elapsed, " checkpoint_reason=", ckpt.checkpoint_reason,
       " theta_lo=", ckpt.theta_lo, " theta_hi=", ckpt.theta_hi)
    b = ckpt.best_feasible
    b === nothing && error("checkpoint has no best_feasible incumbent -- nothing to reconcile")
    lp("best_feasible: ", b)

    ctx_base = d20_real_setup_design(W = W, δ = DELTA, find_smallest = ckpt.find_smallest,
        draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed, destination_sample = ckpt.destination_sample)
    ctx_base.draw_meta.checksum_uniform == ckpt.draw_checksum_uniform ||
        error("draw checksum mismatch on reconstruction -- refusing to reconcile against a different draw realization")
    ctx = make_flexible_theta(ctx_base; theta_lo = ckpt.theta_lo, theta_hi = ckpt.theta_hi, A_coordinate_mode = ckpt.A_coordinate_mode)
    xy = precompute_aspace_XY(ctx)
    rsc = build_ranged_screen_context(ctx)
    w_champ = b.w   # [eta_theta; gp; a_nonpivot]
    r_cv, _, d_cv = screened_eval_flexible_A(w_champ, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    σ = ctx.σ
    κ = 1 - w_champ[2]^(σ / (σ - 1))
    lp(">>> RECONCILED (flexible): gp=", w_champ[2], " theta=", d_cv.theta, " kappa=", κ,
       " cold-verify Delta_dual=", r_cv.Delta_dual, " (checkpoint recorded Delta=", b.Delta, ")",
       " inner_status=", r_cv.inner_status, " gravity=", r_cv.gravity_value, " feasible=", r_cv.Delta_dual <= DELTA + 1e-6)
    lp(">>> RECONCILE_DONE arm=flexible delta=", DELTA, " kappa=", κ, " n_eval=", ckpt.n_eval, " wall=", ckpt.wall_elapsed)
else
    error("ARG1 must be fixed or flexible")
end
