# ============================================================================
# Part B restoration verification (2026-07-23 release): confirms the newly-wired
# cm_screen_bridge.jl screens do NOT false-reject the CM production path's own
# calibration point, and that screen-on / screen-off return the same K/Delta_dual
# there (the certificate functions themselves -- pairwise_certificate,
# screen_hard_winners -- are UNCHANGED, reused verbatim from infeasibility_screen.jl,
# already covered by test_infeasibility_screen.jl; this test covers the NEW call
# site only). Real D=20/W=80,000/L=50 context, matching
# test_cm_expected_solve_failure_typed.jl's proven setup pattern.
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
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
check("ctx.pairwise precomputed by default (build_screen=true default)", ctx.pairwise !== nothing)
check("ctx.witness precomputed by default", ctx.witness !== nothing)

pe = build_pivot_elimination(ctx)
snaps = nested_grid_sequence([10, 20, 50])
L = 50
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
x_free_calib = ctx.θ0_up[ctx.free_idx]
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], ctx.D, ctx.D)), pe))
xf_calib = vcat(w_calib[1], vec(exp.(pivot_expand(w_calib[2:end], pe))))

lp(">>> Group 1: screen precheck alone does not false-reject the calibration point")
sc = CMScreenCounters()
precheck_ok = true
try
    cm_screen_precheck!(xf_calib, pcx.ctx_cm; counters = sc)
catch e
    global precheck_ok = false
    lp("  UNEXPECTED exception from cm_screen_precheck! at calibration point: ", sprint(showerror, e))
end
check("cm_screen_precheck! passes at calibration (no false rejection)", precheck_ok)
check("passed counter incremented", sc.passed == 1)
check("no pairwise/witness/winner rejections at calibration", sc.pairwise == 0 && sc.witness == 0 && sc.winner == 0)

lp(">>> Group 2: screened vs unscreened evaluation agree bit-for-bit on the feasible calibration point")
K_unscreened, base_unscreened, verify_unscreened = cm_production_value_verified(xf_calib, pcx)
K_screened, base_screened, verify_screened = cm_production_value_verified_screened(xf_calib, pcx)
check("K (kappa objective) identical screened vs unscreened", K_unscreened == K_screened)
check("Delta_dual identical screened vs unscreened",
      get(verify_unscreened, :Delta_dual, NaN) == get(verify_screened, :Delta_dual, NaN))
check("inner_status identical screened vs unscreened",
      get(verify_unscreened, :inner_status, nothing) == get(verify_screened, :inner_status, nothing))

lp(">>> Group 3: screened wrapper still raises CMExpectedSolveFailure (not a new exception type)")
lp("    for a genuinely pathological point (same construction as test_cm_expected_solve_failure_typed.jl),")
lp("    i.e. restoration did not change failure-mode typing for callers.")
xf_pathological = copy(xf_calib)
xf_pathological[2:end] .*= 1e30
xf_pathological[1] = 1e30
raised_screened = nothing
try
    cm_production_value_verified_screened(xf_pathological, pcx)
catch e
    global raised_screened = typeof(e)
end
check("pathological point still raises SOME exception under the screened wrapper", raised_screened !== nothing)
check("that exception is still CMExpectedSolveFailure", raised_screened === CMExpectedSolveFailure)

lp("\n>>> RESULT: ", n_pass[], "/", n_pass[] + n_fail[], " assertions passed")
n_fail[] == 0 || error("test_cm_screen_restoration.jl: $(n_fail[]) assertion(s) FAILED")
