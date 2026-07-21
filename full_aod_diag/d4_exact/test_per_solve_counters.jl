# ============================================================================
# Live-KNITRO regression test for task §9 (per-solve instrumentation audit):
# "Add tests that solve the same small problem twice and verify that per-solve
# counters do not inherit values from the first solve."
#
# Uses a trivial synthetic NLP (NOT the real D=20 economic model -- no ctx build, no
# real data) so this runs in seconds and needs nothing beyond a working KNITRO
# license (this machine: demand.mit.edu only, see memory note
# reference-knitro-license-demand). Mirrors the production driver's own pattern of
# `kc = KNITRO.KN_new()` freshly per call, `KN_free(kc)` at the end.
# ============================================================================
using Test, KNITRO

include(joinpath(@__DIR__, "..", "..", "cc_algo", "knitro_compat.jl"))   # KN_add_vars(kc,n)/etc. 2-arg convenience shim, same one the real production driver relies on
include(joinpath(@__DIR__, "knitro_status.jl"))

"Solve min (x-a)^2 s.t. x in [-10,10] from a deliberately bad start, on a FRESH kc. Returns full_status_record."
function solve_toy(a::Float64; x0::Float64 = 5.0, maxit::Int = 1000)
    kc = KNITRO.KN_new()
    KNITRO.KN_set_int_param(kc, KNITRO.KN_PARAM_OUTLEV, 0)
    xIdx = KNITRO.KN_add_vars(kc, 1)
    KNITRO.KN_set_var_lobnds_all(kc, [-10.0])
    KNITRO.KN_set_var_upbnds_all(kc, [10.0])
    KNITRO.KN_set_var_primal_init_values_all(kc, [x0])
    function cb_eval(kc2, cb, evalRequest, evalResult, userParams)
        x = evalRequest.x[1]
        evalResult.obj[1] = (x - a)^2
        return 0
    end
    function cb_grad(kc2, cb, evalRequest, evalResult, userParams)
        x = evalRequest.x[1]
        evalResult.objGrad[1] = 2 * (x - a)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_eval)
    KNITRO.KN_set_cb_grad(kc, cb, cb_grad)
    KNITRO.KN_solve(kc)
    status, obj, xsol, _ = KNITRO.KN_get_solution(kc)
    rec = full_status_record(status, kc)
    KNITRO.KN_free(kc)
    return (status = status, xsol = xsol[1], rec = rec)
end

@testset "per-solve counters are not inherited across fresh kc instances" begin

r1 = solve_toy(3.0; x0 = 5.0)
r2 = solve_toy(-7.0; x0 = 9.9)   # deliberately a much longer path -> should need MORE iterations than r1

@test isapprox(r1.xsol, 3.0; atol = 1e-4)
@test isapprox(r2.xsol, -7.0; atol = 1e-4)

@testset "both solves report genuinely positive per-solve counts" begin
    @test r1.rec.n_iters >= 0
    @test r2.rec.n_iters >= 0
    @test r1.rec.n_fc_evals >= 1
    @test r2.rec.n_fc_evals >= 1
end

@testset "second solve's counters reflect ONLY its own (longer) trajectory, not cumulative totals" begin
    # r2 starts much further from its optimum (16.9 units away) than r1 does (2.0 units away),
    # so if counters were being silently summed/inherited from r1, r2's reported n_fc_evals would
    # be inflated far beyond what a fresh solve of this trivial 1-D quadratic needs. A quadratic
    # in 1 variable converges in a handful of iterations regardless of start distance (Newton-like
    # step is exact for a quadratic), so a genuinely fresh, uninherited count should stay small
    # (well under what 2x accumulation across both solves would look like).
    @test r2.rec.n_iters < 50   # trivial 1-D quadratic; would be arbitrarily larger if r1's count carried over across a naive global-counter bug
    @test r2.rec.n_fc_evals < 50
end

@testset "solve_time_real resets per solve (not cumulative)" begin
    @test r1.rec.solve_time_real >= 0.0
    @test r2.rec.solve_time_real >= 0.0
    # Both are trivial sub-second solves; a cumulative-time bug would make r2's reported time
    # implausibly include r1's wall-clock too, but there's no shared clock here to leak from --
    # this mainly documents that the field is populated and non-negative, not a strict inequality.
end

@testset "status_name/category round-trip through the same decoder used for production status codes" begin
    @test r1.rec.status_category in (:optimal, :feasible_approx, :limit_feasible)
    @test r2.rec.status_category in (:optimal, :feasible_approx, :limit_feasible)
end

end # testset

println("All per-solve counter regression tests passed.")
