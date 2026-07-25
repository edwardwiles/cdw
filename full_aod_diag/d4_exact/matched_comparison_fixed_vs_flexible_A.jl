# Task §15: matched practical-value comparison, post-omit-ROW production context.
# Usage: julia --project=. -t 20 matched_comparison_fixed_vs_flexible_A.jl <fixed|flexible> <delta> <budget_s> <ckpt_dir> [label_suffix]
#
# Both arms start from the IDENTICAL genuine calibrated point (ctx0.θ0_up's own A_od/gp block,
# NOT zfree=0 -- see CLAUDE.md), same data/draws (seed 20260719, W=80000), same
# algorithm=auto+SR1 (csw_outer_wallclock_sr1.opt, loaded by BOTH run_polish_checkpointed and
# run_polish_checkpointed_flexible_theta_A by default), same screens/cache/incumbent logic
# (both call into the SAME screened_eval / DualBank / SafeExactCache machinery), same
# destination_sample=:exclude_row. Cold-verifies the champion at the end via a fresh warm=false
# screened_eval call, independent of anything the outer solve itself cached.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_flexible_theta_A.jl"))
using Dates

lp(xs...) = (println(xs...); flush(stdout))

const ARM = ARGS[1]            # "fixed" | "flexible"
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const CKPT_DIR = abspath(ARGS[4])
const SUFFIX = length(ARGS) >= 5 ? ARGS[5] : ""
mkpath(CKPT_DIR)

const W = 80_000
const DRAW_SEED = 20260719
const FIND_SMALLEST = true   # upper-kappa direction, matching the brief's §0 established baseline numbers

ARM in ("fixed", "flexible") || error("ARG1 must be fixed or flexible")

lp(">>> MATCHED COMPARISON arm=", ARM, " delta=", DELTA, " budget=", BUDGET, "s ckpt_dir=", CKPT_DIR)
lp(">>> Julia threads: ", Threads.nthreads())

# ---- Build the genuine calibrated starting point (ctx0.θ0_up's own A_od/gp block) ----
ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
theta_star = 1.0 / ctx0.μHat
sigma = ctx0.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
lp(">>> calibration: gp0=", gp0, " theta_star=", theta_star, " theta_bounds=[", theta_min, ",", theta_max, "]")

if ARM == "fixed"
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D; Ddest = ctx0.D_dest
    zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), D, Ddest), pe0)
    label = "matched_fixed_delta$(DELTA)$(SUFFIX)"
    res = run_polish_checkpointed(label, FIND_SMALLEST, gp0, zfree0;
        maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 30.0,
        price_cache_backend = :cplus, destination_sample = :exclude_row)
    lp(">>> FIXED result: knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
       " n_eval=", res.n_eval, " kappa=", res.kappa,
       " best=", res.best_feasible === nothing ? "nothing" : "gp=$(res.best_feasible.gp) Delta=$(res.best_feasible.Delta)")

    # Cold-verify champion
    if res.best_feasible !== nothing
        pe_v = build_pivot_elimination(res.ctx)
        xf_v = x_free_from_w(res.best_feasible.w, pe_v)
        r_cv, _ = screened_eval(xf_v, res.ctx, build_ranged_screen_context(res.ctx), ScreenCounters(), Ref(0); warm = false)
        lp(">>> FIXED cold-verify: inner_status=", r_cv.inner_status, " Delta_dual=", r_cv.Delta_dual,
           " (vs live ", res.best_feasible.Delta, ")")
    end
    lp(">>> ARM_DONE arm=fixed delta=", DELTA, " kappa=", res.kappa, " wall=", res.wall_ext, " n_eval=", res.n_eval)
else
    ctx = make_flexible_theta(ctx0; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
    xy = precompute_aspace_XY(ctx)
    D = ctx.D; Ddest = ctx.D_dest
    logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
    pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
    w_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)
    label = "matched_flexible_delta$(DELTA)$(SUFFIX)"
    res = run_polish_checkpointed_flexible_theta_A(label, FIND_SMALLEST, w_start_a;
        theta_lo = theta_min, theta_hi = theta_max,
        maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 30.0,
        price_cache_backend = :cplus, destination_sample = :exclude_row)
    lp(">>> FLEXIBLE result: knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
       " n_eval=", res.n_eval, " kappa=", res.kappa,
       " best=", res.best_feasible === nothing ? "nothing" : "gp=$(res.best_feasible.gp) theta=$(get(res.best_feasible,:theta,NaN)) Delta=$(res.best_feasible.Delta)")

    if res.best_feasible !== nothing
        w_v = res.best_feasible.w
        r_cv, _, d_cv = screened_eval_flexible_A(w_v, res.ctx, build_ranged_screen_context(res.ctx), ScreenCounters(), Ref(0), res.xy; warm = false)
        lp(">>> FLEXIBLE cold-verify: inner_status=", r_cv.inner_status, " Delta_dual=", r_cv.Delta_dual,
           " theta=", d_cv.theta, " (vs live ", res.best_feasible.Delta, ")")
    end
    lp(">>> ARM_DONE arm=flexible delta=", DELTA, " kappa=", res.kappa, " wall=", res.wall_ext, " n_eval=", res.n_eval)
end
