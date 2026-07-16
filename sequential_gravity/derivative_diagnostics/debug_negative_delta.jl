ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

# Reproduce the exact theta from the lower-bound full-A best-feasible endpoint by re-deriving it
# directly (gamma'=0.997561 area) -- instead, just probe at the FRECHET point first as a sanity
# baseline (known-good, delta* should be tiny positive), then at a hand-picked off-benchmark point.
θtest = copy(θr0)
frozen = freeze_gravity_linearization(θtest, seq_gravcol, grad_R_theta)
@printf("baseline (Frechet) ok=%s R=%.3e\n", frozen.ok, frozen.R)
moments_fn = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen.θ, frozen.Rcol, frozen.gcol, frozen.dRdθ)
obj = build_fixed_dual_bundle(γ, U, length(θtest), D+2, moments_fn; find_smallest=false)
δstar, xstar, nSt = inner_loop(obj, θtest)
@printf("Frechet: delta*=%.8f nStatus=%d  x*[1:3]=%s\n", δstar, nSt, xstar[1:3])
Qcheck = dual_criterion_fixed_x(θtest, obj, xstar)
@printf("cross-check via dual_criterion_fixed_x at same x*: Q=%.8f (should equal delta* above)\n", Qcheck)

println("\n--- now a genuinely off-benchmark, larger-gamma' point (gamma'=0.9975, same A*) ---")
θtest2 = copy(θr0); θtest2[3] = 0.9975
frozen2 = freeze_gravity_linearization(θtest2, seq_gravcol, grad_R_theta)
@printf("ok=%s R=%.3e\n", frozen2.ok, frozen2.R)
if frozen2.ok
    moments_fn2 = make_frozen_gravity_moments(EK_moments_focal_norm_directgp!, D, frozen2.θ, frozen2.Rcol, frozen2.gcol, frozen2.dRdθ)
    obj2 = build_fixed_dual_bundle(γ, U, length(θtest2), D+2, moments_fn2; find_smallest=false)
    δstar2, xstar2, nSt2 = inner_loop(obj2, θtest2)
    @printf("delta*=%.8f nStatus=%d x*[1:3]=%s\n", δstar2, nSt2, xstar2[1:3])
    Qcheck2 = dual_criterion_fixed_x(θtest2, obj2, xstar2)
    @printf("cross-check: Q=%.8f\n", Qcheck2)
else
    println("theta2 not gravity-feasible either")
end
