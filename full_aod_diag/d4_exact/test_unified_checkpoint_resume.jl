# Checkpoint/resume regression gates for run_polish_checkpointed_unified (task §14 / Phase 2).
# Real D=20/W=80,000/:exclude_row throughout (the unified driver only builds via
# d20_real_setup_design). Each case rebuilds ctx from scratch (~65-90s) -- no `reuse=` kwarg on
# this driver yet, so this is intentionally a small, bounded set of cases, not a sweep.
#
# Cases:
#  1. Happy-path save -> resume (fixed_aspace): resume must succeed, tolerances must check out,
#     and n_eval must continue increasing (not reset).
#  2. Layout mismatch on resume (fixed_aspace checkpoint, fixed_legacyz resume attempt): must
#     error, not silently resume under the wrong coordinate convention.
#  3. destination_sample mismatch on resume: must error.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_unified_checkpoint_resume.jl
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0
const FIND_SMALLEST = true
CKPT_DIR = mktempdir()

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace)
layout_z = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z)
xy = precompute_aspace_XY(ctx0)
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0/theta_star*0.999, mu_probe2 = 1.0/theta_star*1.001)
w_start_a = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_a)

lp(">>> Case 1: happy-path save -> resume (fixed_aspace)")
res1 = run_polish_checkpointed_unified("ckresume", FIND_SMALLEST, w_start_a;
    layout = layout_a, maxtime_real = 25.0, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 5.0, price_cache_backend = :cplus,
    destination_sample = :exclude_row)
n_eval_1 = res1.n_eval
latest_path = joinpath(CKPT_DIR, "ckresume_unified_latest.jls")
check(isfile(latest_path), "checkpoint file written after run 1")

res2 = run_polish_checkpointed_unified("ckresume", FIND_SMALLEST, w_start_a;
    layout = layout_a, maxtime_real = 25.0, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 5.0, price_cache_backend = :cplus,
    destination_sample = :exclude_row, resume_from = latest_path)
check(res2.n_eval >= n_eval_1, "resumed n_eval ($(res2.n_eval)) continues from run 1 ($(n_eval_1)), not reset")
check(res2.best_feasible !== nothing, "resumed run has a best_feasible incumbent")

lp(">>> Case 2: layout mismatch on resume (checkpoint=fixed_aspace, resume attempt=fixed_legacyz)")
mismatch_caught = false
try
    run_polish_checkpointed_unified("ckresume", FIND_SMALLEST, w_start_a;
        layout = layout_z, maxtime_real = 5.0, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 5.0, price_cache_backend = :cplus,
        destination_sample = :exclude_row, resume_from = latest_path)
catch e
    global mismatch_caught = occursin("LAYOUT mismatch", sprint(showerror, e))
end
check(mismatch_caught, "layout mismatch on resume raises the expected error")

lp(">>> Case 3: destination_sample mismatch on resume")
ds_mismatch_caught = false
try
    run_polish_checkpointed_unified("ckresume", FIND_SMALLEST, w_start_a;
        layout = layout_a, maxtime_real = 5.0, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 5.0, price_cache_backend = :cplus,
        destination_sample = :all_legacy, resume_from = latest_path)
catch e
    global ds_mismatch_caught = occursin("destination_sample MISMATCH", sprint(showerror, e))
end
check(ds_mismatch_caught, "destination_sample mismatch on resume raises the expected error")

lp(isempty(FAILURES) ? "ALL UNIFIED CHECKPOINT/RESUME GATES PASS" : "FAILURES: $(FAILURES)")
