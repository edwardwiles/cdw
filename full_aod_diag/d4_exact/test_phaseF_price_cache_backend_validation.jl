# Phase F remediation (2026-07-26): confirm price_cache_backend=:bogus now fails FAST with a
# clear, actionable error (before any KNITRO solve), instead of silently building a
# lfix_c_ws=nothing that would previously crash with a confusing MethodError deep inside cb_G!
# mid-solve. Uses D=4 (needs_outer_moment_jacobian=false, no real D=20 setup needed) purely to
# exercise run_polish_checkpointed_unified's own early validation logic quickly -- the function
# itself is written for D=20 real data (d20_real_setup_design), so this test instead calls the
# validation logic directly via a minimal reproduction, since a full D=20 ctx build is not needed
# to prove this specific fix.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

# Real D=20 setup (this specific check runs BEFORE any KNITRO solve, so the wall cost here is
# just the ~65-90s ctx build, not a solve budget).
layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
ctx0 = d20_real_setup_design(W = 80_000, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
xy = precompute_aspace_XY(ctx0)
pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)

CKPT_DIR = mktempdir()

threw = false
msg = ""
try
    run_polish_checkpointed_unified("stage", true, w_start;
        layout = layout, maxtime_real = 1.0, W_in = 80_000, delta_in = 1.0, draw_seed_in = 20260719,
        ckpt_dir = CKPT_DIR, price_cache_backend = :bogus, destination_sample = :exclude_row)
catch e
    global threw = true
    global msg = sprint(showerror, e)
end
check(threw, "price_cache_backend=:bogus raises an error")
check(occursin("price_cache_backend", msg) && occursin("not supported", msg),
      "error message is the NEW clear/actionable one, not a downstream MethodError (msg=$msg)")
check(!occursin("MethodError", msg), "error is NOT a MethodError (i.e. did not reach cb_G! before failing)")

lp("==================== SUMMARY ====================")
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES: ", FAILURES)
    exit(1)
end
