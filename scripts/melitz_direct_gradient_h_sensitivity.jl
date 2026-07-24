# Continuation4: is the direct-backend vs analytic-backend gap an h-truncation effect
# (analytic backend linearizes Psi at arg0_base via dPsi once; direct backend evaluates the
# true nonlinear Psi at both displaced points) that shrinks as h->0, or a genuine formula bug?
using Printf, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
outer_opt = joinpath(dirname(@__DIR__), "melitz_outer_finite_delta.opt")
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj_inner, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt, needs_outer_moment_jacobian=false)
ctx = obj_inner.γ
n = length(theta0)

CS = CounterfactualSensitivity
obj_ref = build_melitz_implicit_bundle(ctx, obj_inner.U, theta0; delta=1.0, find_smallest=true,
    gradient_backend=:B_argument_localized_parallel, h=1e-4, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
obj_ref.use_cached_x = false; obj_ref.x .= NaN
_, x, nStatus = CS.inner_loop_internal(obj_ref, theta0)
@printf("inner nStatus=%d\n", nStatus)

# how many draws have arg0_base near the Psi kink (arg0<=1 vs >1)?
outer_constr_index = obj_ref.outer_constr_index
arg0_base = zeros(size(obj_ref.U,1))
LinearAlgebra.BLAS.gemv!('N', 1.0, @view(obj_ref.H[:, 2:1+outer_constr_index]), -x, 0.0, arg0_base)
@printf("arg0_base: min=%.4f max=%.4f  frac <=1: %.4f\n", minimum(arg0_base), maximum(arg0_base), mean(arg0_base .<= 1.0))

probe_coords = [1, 18, 19]
for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
    local_jac_ref = zeros(n)
    dummy_g = zeros(n)
    obj_ref2 = build_melitz_implicit_bundle(ctx, obj_inner.U, theta0; delta=1.0, find_smallest=true,
        gradient_backend=:B_argument_localized_parallel, h=h, inner_loop_opt=inner_opt, outer_loop_opt=outer_opt)
    obj_ref2.H .= obj_ref.H; obj_ref2.x .= x
    obj_ref2(x, dummy_g, theta0; jac=local_jac_ref)

    direct_serial = make_melitz_gradient_delta_direct_serial(h)
    local_jac_direct = zeros(n)
    direct_serial(local_jac_direct, theta0, ctx, obj_ref2, x)

    @printf("h=%.0e:\n", h)
    for r in probe_coords
        rel = abs(local_jac_direct[r] - local_jac_ref[r]) / max(1.0, abs(local_jac_ref[r]))
        @printf("  coord %2d: analytic=% .6e  direct=% .6e  rel_diff=%.3e\n", r, local_jac_ref[r], local_jac_direct[r], rel)
    end
end
