# Continuation4 smoke test: exercise the new :B_direct_argument_serial/_parallel backends
# through the REAL production combined callback (melitz_fixed_point_probe), not just the
# standalone validation script -- confirms cb_G!'s new branch actually registers/runs
# correctly inside a real (degenerate, 0-DOF) KNITRO problem, end to end.
using Printf
include(joinpath(@__DIR__, "melitz_finite_delta_campaign.jl"))

inner_opt = joinpath(dirname(@__DIR__), "melitz_inner_loop_options.opt")
data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj_inner, theta0 = build_melitz_psi_bundle(data; inner_loop_opt=inner_opt, needs_outer_moment_jacobian=false)
ctx = obj_inner.γ

for backend in (:B, :B_argument_localized_parallel, :B_direct_argument_serial, :B_direct_argument_parallel)
    r = melitz_fixed_point_probe(ctx, obj_inner, theta0; delta=1e-2, direction=:upper,
        gradient_backend=backend, inner_loop_opt=inner_opt)
    @printf("backend=%-28s nStatus=%d eval_failed=%s obj_value=%.6f c[1]=%.6e\n",
        backend, r.nStatus, r.eval_failed, r.obj_value, isempty(r.c) ? NaN : r.c[1])
end
println("SMOKE TEST DONE")
