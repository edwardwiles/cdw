# ============================================================================
# Closure task Phase 3B regression test: CMExpectedSolveFailure must be the ONLY exception
# type run_cm_upper/run_cm_upper_checkpointed's cb_F! silently converts into a rejected point.
# A bare ErrorException from an ordinary programming bug (not the documented inner-solve-failed
# signal) must propagate and abort the run, not be silently swallowed.
#
# Real D=20/W=80,000/L=50 context (the only validated CM production path); kept to a tiny
# maxtime_real budget (a handful of seconds) since only 1-2 evaluations are needed either way.
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
n_pass = Ref(0); n_fail = Ref(0)
function check(name, cond)
    if cond
        lp("  PASS: ", name); n_pass[] += 1
    else
        lp("  FAIL: ", name); n_fail[] += 1
    end
end

Random.seed!(20260719)
ctx = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true)
pe = build_pivot_elimination(ctx)
snaps = nested_grid_sequence([10, 20, 50])
L = 50
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
x_free_calib = ctx.θ0_up[ctx.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], ctx.D, ctx.D)), pe))
xf_calib = vcat(w_calib[1], vec(exp.(pivot_expand(w_calib[2:end], pe))))

lp(">>> Test group 1: raise side -- archC_verified_state/cm_production_value_verified really")
lp("    throw CMExpectedSolveFailure (not a bare ErrorException) for a genuine inner-solve failure.")
# A grossly pathological x_free (A_od scaled by 1e8) is expected to make the inner CC dual solve
# genuinely infeasible/unbounded (nStatus outside (0,-100,-101,-103)) -- same style of pathological
# point the sequential-linearized branch's own test_recover_lfd_unsuccessful_solve.jl uses.
raised_type = nothing
scales_tried = Float64[]
for scale in (1e8, 1e15, 1e30, 1e60, 1e150)
    raised_type !== nothing && break
    global scales_tried
    push!(scales_tried, scale)
    xf_pathological = copy(xf_calib)
    xf_pathological[2:end] .*= scale
    # also push gp itself far outside [gp_lo, gp_hi] -- the inner CC dual solve doesn't depend
    # on gp's bound feasibility directly, but a wildly-off gp still changes theta_full via
    # reconstruct_full and can independently destabilize the inner solve.
    xf_pathological[1] = scale
    try
        cm_production_value_verified(xf_pathological, pcx)
    catch e
        global raised_type = typeof(e)
    end
end
lp("  scales tried before an exception appeared (or exhausted): ", scales_tried)
check("a genuinely pathological point raises SOME exception", raised_type !== nothing)
check("that exception is CMExpectedSolveFailure, not a bare ErrorException",
      raised_type === CMExpectedSolveFailure)

lp(">>> Test group 2: catch-site idiom -- exactly reproduces cb_F!'s own `e isa CMExpectedSolveFailure || rethrow()`")
function replicate_catch_site(f)
    try
        f()
        return :no_exception
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        return :gracefully_rejected
    end
end
r1 = replicate_catch_site(() -> throw(CMExpectedSolveFailure("synthetic: inner solve failed")))
check("a genuine CMExpectedSolveFailure is caught and converted to a graceful rejection", r1 == :gracefully_rejected)

propagated_bug = false
try
    replicate_catch_site(() -> error("injected programming bug -- must NOT be swallowed"))
catch e
    global propagated_bug = e isa ErrorException && !(e isa CMExpectedSolveFailure)
end
check("an injected bare ErrorException (programming bug) is NOT caught -- propagates", propagated_bug)

propagated_methoderror = false
not_callable = 1   # binding first: a bare literal `(1)(2)` parses as multiplication (=2), not a call
try
    replicate_catch_site(() -> not_callable(2))   # MethodError: an Int is not callable
catch e
    global propagated_methoderror = e isa MethodError
end
check("an injected MethodError is NOT caught -- propagates", propagated_methoderror)

lp(">>> Test group 3: end-to-end -- run_cm_upper_checkpointed itself does not silently absorb an")
lp("    injected programming bug as a rejected point (global monkey-patch, restored after test;")
lp("    this is a standalone `julia script.jl` process, so no other session is affected).")
const _cm_production_value_verified_ORIGINAL = cm_production_value_verified
global _injected_bug_calls = 0
function cm_production_value_verified(x_free0::AbstractVector, pcx_arg)
    global _injected_bug_calls += 1
    error("cm_production_value_verified: INJECTED programming bug for test_cm_expected_solve_failure_typed.jl -- must propagate, not be silently rejected")
end
ckpt_dir_bug = mktempdir()
propagated_e2e = false
try
    run_cm_upper_checkpointed(w_calib; W = 80000, delta = 1.0, draw_design = :pseudorandom,
        draw_seed = 20260719, L = L, contrasts = :anchored, probs = snaps[L],
        cm_hessian_backend = :structured, cm_grid_rule = :nested_family,
        maxtime_real = 30.0, ckpt_dir = ckpt_dir_bug, run_id = "test_bug", label = "bugtest",
        checkpoint_interval_s = 100.0, verbose = false)
    lp("  (run_cm_upper_checkpointed returned normally -- UNEXPECTED, see below)")
catch e
    global propagated_e2e = true
    lp("  run_cm_upper_checkpointed raised (expected): ", sprint(showerror, e))
end
# restore
function cm_production_value_verified(x_free0::AbstractVector, pcx_arg)
    return _cm_production_value_verified_ORIGINAL(x_free0, pcx_arg)
end
check("injected bug calls happened (monkeypatch reached)", _injected_bug_calls > 0)
check("run_cm_upper_checkpointed did NOT silently complete/reject -- it propagated the bug", propagated_e2e)

lp("\n>>> RESULT: ", n_pass[], "/", n_pass[] + n_fail[], " assertions passed")
n_fail[] == 0 || error("test_cm_expected_solve_failure_typed.jl: $(n_fail[]) assertion(s) FAILED")
