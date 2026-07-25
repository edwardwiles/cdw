# Reconciles an interrupted unified-driver matched-comparison arm from its last
# D20CheckpointUnified checkpoint: loads the accumulated best_feasible incumbent, reconstructs
# its outer point, and cold-verifies it via a fresh warm=false production evaluation.
# Usage: julia --project=. reconcile_checkpoint_unified.jl <legacy_z|powered_aspace> <ckpt_path> <delta>
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const A_MODE = Symbol(ARGS[1])
const CKPT_PATH = ARGS[2]
const DELTA = parse(Float64, ARGS[3])
const W = 80_000

lp(">>> RECONCILE-UNIFIED A_coordinate_mode=", A_MODE, " ckpt=", CKPT_PATH, " delta=", DELTA)

ckpt = load_checkpoint_unified(CKPT_PATH)
lp("checkpoint: schema=", ckpt.schema, " n_eval=", ckpt.n_eval, " knitro_iter=", ckpt.knitro_iter,
   " wall_elapsed=", ckpt.wall_elapsed, " checkpoint_reason=", ckpt.checkpoint_reason,
   " A_coordinate_mode=", ckpt.A_coordinate_mode)
b = ckpt.best_feasible
b === nothing && error("checkpoint has no best_feasible incumbent -- nothing to reconcile")
lp("best_feasible: gp=", b.gp, " theta=", get(b, :theta, NaN), " Delta=", b.Delta, " n_eval=", b.n_eval)

ctx0 = d20_real_setup_design(W = ckpt.W, δ = DELTA, find_smallest = ckpt.find_smallest,
    draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed, destination_sample = ckpt.destination_sample)
ctx0.draw_meta.checksum_uniform == ckpt.draw_checksum_uniform ||
    error("draw checksum mismatch on reconstruction -- refusing to reconcile against a different draw realization")

layout = make_layout(trade_elasticity_mode = ckpt.trade_elasticity_mode, A_coordinate_mode = ckpt.A_coordinate_mode,
                      gp_coordinate_mode = ckpt.gp_coordinate_mode)
xy = precompute_aspace_XY(ctx0)
theta_star = 1.0 / ctx0.μHat
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
rsc = build_ranged_screen_context(ctx0)

d0 = decode_outer_unified(b.w, ctx0, layout, pgc, xy)
r_cv, _ = screened_eval(d0.xf, ctx0, rsc, ScreenCounters(), Ref(0); warm = false)
σ = ctx0.σ
κ = 1 - b.gp^(σ / (σ - 1))
lp(">>> RECONCILED-UNIFIED: gp=", b.gp, " kappa=", κ, " cold-verify Delta_dual=", r_cv.Delta_dual,
   " (checkpoint recorded Delta=", b.Delta, ") inner_status=", r_cv.inner_status,
   " gravity=", r_cv.gravity_value, " feasible=", r_cv.Delta_dual <= DELTA + 1e-6)
lp(">>> RECONCILE_UNIFIED_DONE A_coordinate_mode=", A_MODE, " delta=", DELTA, " kappa=", κ,
   " n_eval=", ckpt.n_eval, " wall=", ckpt.wall_elapsed)
