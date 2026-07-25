# Addendum §6: matched comparison of FIXED-theta coordinate choice (legacy_z vs powered_aspace),
# both raw gp, at the genuine calibrated D=20 post-omit-ROW start point. Both arms use the SAME
# unified driver (run_polish_checkpointed_unified) with layout differing ONLY in A_coordinate_mode
# -- everything else (data, draws, screens, cache, incumbent logic, algorithm=auto+SR1) identical
# by construction (one shared driver, one shared option file).
#
# Usage: julia --project=. matched_comparison_fixed_coordinate_modes.jl <legacy_z|powered_aspace> <delta> <budget_s> <ckpt_dir> [suffix]
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))

const A_MODE = Symbol(ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const CKPT_DIR = abspath(ARGS[4])
const SUFFIX = length(ARGS) >= 5 ? ARGS[5] : ""
mkpath(CKPT_DIR)

const W = 80_000
const DRAW_SEED = 20260719
const FIND_SMALLEST = true

A_MODE in (:legacy_z, :powered_aspace) || error("ARG1 must be legacy_z or powered_aspace")

lp(">>> FIXED-COORDINATE-MODE COMPARISON A_coordinate_mode=", A_MODE, " delta=", DELTA, " budget=", BUDGET, "s")
lp(">>> Julia threads: ", Threads.nthreads())

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = A_MODE, gp_coordinate_mode = :raw)
xy = precompute_aspace_XY(ctx0)
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)

label = "matched_fixed_$(A_MODE)_delta$(DELTA)$(SUFFIX)"
res = run_polish_checkpointed_unified(label, FIND_SMALLEST, w_start;
    layout = layout, maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 30.0,
    price_cache_backend = :cplus, destination_sample = :exclude_row)
lp(">>> RESULT A_coordinate_mode=", A_MODE, ": knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
   " n_eval=", res.n_eval, " kappa=", res.kappa,
   " best=", res.best_feasible === nothing ? "nothing" : "gp=$(res.best_feasible.gp) Delta=$(res.best_feasible.Delta)")

if res.best_feasible !== nothing
    d_cv = decode_outer_unified(res.best_feasible.w, res.ctx, layout, res.pgc, res.xy)
    r_cv, _ = screened_eval(d_cv.xf, res.ctx, build_ranged_screen_context(res.ctx), ScreenCounters(), Ref(0); warm = false)
    lp(">>> cold-verify: inner_status=", r_cv.inner_status, " Delta_dual=", r_cv.Delta_dual, " (vs live ", res.best_feasible.Delta, ")")
end
lp(">>> ARM_DONE A_coordinate_mode=", A_MODE, " delta=", DELTA, " kappa=", res.kappa, " wall=", res.wall_ext, " n_eval=", res.n_eval)
