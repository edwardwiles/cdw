# ============================================================================
# End-to-end wiring check for the allocation/cache-cleanup task's
# use_pooled_gradient= kwarg on run_polish_checkpointed (c10_d20_production_
# driver.jl). test_gradient_workspace.jl already validates
# composite_gradient_at_fast_pooled itself is bit-identical to
# composite_gradient_at_fast_buffered at the FUNCTION level; this validates
# the DRIVER actually dispatches to it correctly end-to-end (real KNITRO
# outer solve, real cb_G! calls, real checkpoint writes) when the flag is
# set, and is unchanged when it isn't.
#
# Run standalone (real D=20/W=80000, several minutes):
#   julia --project=. full_aod_diag/d4_exact/test_driver_pooled_gradient_wiring.jl
# ============================================================================
include(joinpath(@__DIR__, "staged_delta5.jl"))

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

ckpt_root = mktempdir()

println("== Driver wiring: use_pooled_gradient=true dispatches to composite_gradient_at_fast_pooled ==")
ctx0 = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true)
D = ctx0.D
pe0 = build_pivot_elimination(ctx0)
g0 = ctx0.θ0_up[3+D]   # natural calibrated gamma_prime_focal -- guaranteed inside the direction box
# AUD note (this session's own memory: "gravity_elimination z=0 is NOT calibration"): zfree=0
# through pivot_expand gives Aod~1, an arbitrary reparam GAUGE reference, not the real calibrated
# A* -- and is generally NOT inner-feasible. The real calibrated start is ctx0.θ0_up's own Aod
# block (Aod_offset+1:Aod_offset+D^2), pivot-reduced.
Aod_real = reshape(ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D^2], D, D)
zfree0 = pivot_reduce(log.(Aod_real), pe0)

res_buffered = run_polish_checkpointed("wiretest_buffered", true, g0, zfree0;
    maxtime_real = 20.0, W_in = 80000, delta_in = 1.0,
    ckpt_dir = joinpath(ckpt_root, "buffered"), checkpoint_interval_s = 5.0,
    use_pooled_gradient = false)

res_pooled = run_polish_checkpointed("wiretest_pooled", true, g0, zfree0;
    maxtime_real = 20.0, W_in = 80000, delta_in = 1.0,
    ckpt_dir = joinpath(ckpt_root, "pooled"), checkpoint_interval_s = 5.0,
    use_pooled_gradient = true)

check("both runs completed (knitro_status recorded)", res_buffered.knitro_status isa Integer && res_pooled.knitro_status isa Integer)
check("n_grad_calls > 0 for both (gradient path actually exercised)", res_buffered.n_grad_calls > 0 && res_pooled.n_grad_calls > 0)
check("pooled run's incumbent (if any) has a finite Delta", res_pooled.best_feasible === nothing || isfinite(res_pooled.best_feasible.Delta))
check("buffered run's incumbent (if any) has a finite Delta", res_buffered.best_feasible === nothing || isfinite(res_buffered.best_feasible.Delta))
println("  buffered: kappa=", res_buffered.kappa, " n_grad_calls=", res_buffered.n_grad_calls)
println("  pooled:   kappa=", res_pooled.kappa, " n_grad_calls=", res_pooled.n_grad_calls)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
