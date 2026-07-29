# Quick standalone smoke test for the new :B_direct_argument_aq_experimental gradient
# backend (Phase 11) before running the full test suite. Throwaway, D4 only.

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Printf

data = generate_fake_melitz_data(; D=4, seed=29, W=20_000)
obj, theta0 = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx = obj.γ

# 1. direct closure test: does the gradient function run and produce a finite, sane vector?
obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@assert lfd0.lfd_ok
x0 = copy(lfd0.dual_x)
gfn = make_melitz_gradient_delta_direct_aq_experimental(PowerScaledQBandwidth(1e-3, 80_000, 0.5))
g = zeros(length(theta0))
melitz_update_operator_at_theta!(obj.op, theta0, ctx)
gfn(g, theta0, ctx, obj, x0)
println("gradient (first 10): ", g[1:10])
@assert all(isfinite, g) "gradient contains non-finite entries"
println("SMOKE 1 PASSED: closure runs, all finite")

# 2. wrong-parameterization guard
obj_f, theta0_f = build_melitz_psi_bundle(data; policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx_f = obj_f.γ
function check_throws_argerror()
    try
        gfn(zeros(length(theta0_f)), theta0_f, ctx_f, obj_f, x0)
        return false
    catch e
        return e isa ArgumentError
    end
end
@assert check_throws_argerror() "expected ArgumentError for :logf ctx"
println("SMOKE 2 PASSED: :logf ctx correctly rejected")

# 3. full dispatch through solve_melitz_finite_delta_bound (bounded, small maxit)
println("\n--- dispatch smoke test via solve_melitz_finite_delta_bound ---")
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_alg_direct_2026-07-27.opt")
result = solve_melitz_finite_delta_bound(ctx, obj, theta0; delta=0.5, direction=:upper,
    gradient_backend=:B_direct_argument_aq_experimental, backend=:matrix_free,
    policy=CappedEvaluation(10.0), inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
@printf("dispatch result nStatus=%d  inner_solves=%d\n", result.nStatus, result.inner_solve_count)
println("SMOKE 3 PASSED: dispatch runs without crash")

println("\nALL SMOKE TESTS PASSED")
