# Addendum §7: bounded confirmation of gp_coordinate_mode within fixed+powered_aspace: raw gp vs
# scaled-log gp (u_g = s_g*log(gp/gp_star)). NOT changing the default (raw stays default per §7);
# this is an experimental opt-in confirmation run only.
#
# Usage: julia --project=. matched_comparison_gp_scaling.jl <raw|scaled_log> <delta> <budget_s> <ckpt_dir> <rep_idx>
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))

const GP_MODE = Symbol(ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const CKPT_DIR = abspath(ARGS[4])
const REP = ARGS[5]
mkpath(CKPT_DIR)

const W = 80_000
const DRAW_SEED = 20260719
const FIND_SMALLEST = true

GP_MODE in (:raw, :scaled_log) || error("ARG1 must be raw or scaled_log")

lp(">>> GP-SCALING COMPARISON gp_coordinate_mode=", GP_MODE, " delta=", DELTA, " budget=", BUDGET, "s rep=", REP)

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = GP_MODE)
gs = GP_MODE == :scaled_log ? GpScale(gp0, 1.0) : nothing
xy = precompute_aspace_XY(ctx0)
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout, gs)

label = "gpscale_$(GP_MODE)_delta$(DELTA)_rep$(REP)"
res = run_polish_checkpointed_unified(label, FIND_SMALLEST, w_start;
    layout = layout, gp_scale = gs, maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 30.0,
    price_cache_backend = :cplus, destination_sample = :exclude_row)
lp(">>> RESULT gp_coordinate_mode=", GP_MODE, " rep=", REP, ": knitro_status=", res.knitro_status,
   " wall=", round(res.wall_ext, digits = 1), " n_eval=", res.n_eval, " kappa=", res.kappa)
lp(">>> ARM_DONE gp_mode=", GP_MODE, " delta=", DELTA, " rep=", REP, " kappa=", res.kappa, " wall=", res.wall_ext, " n_eval=", res.n_eval)
