# ============================================================================
# Part 4 driver (Stage A, D=4): validates the FULL-(D+2) fixed-dual criterion
# and FD gradient against (a) the exact identity at a real sequential
# iterate, (b) fully re-solved profile finite differences, at BOTH the
# Frechet benchmark and an off-benchmark target (the Frechet point has a
# near-zero true slope, which inflates relative-error metrics -- see
# derivative_methods_report.md Part 4's lesson -- so this driver always tests
# an off-benchmark target too).
#
# Loads run_profiled_production.jl with SKIP_BATCH_LOOP=true purely for its
# setup/functions (economy, theta_r0, seq_gravcol, grad_R_theta, KBOUNDS,
# EK_moments_focal_norm_directgp!, ...) -- no KNITRO outer search is
# triggered by the include itself.
#
#   DVAL=4 WVAL=8000 julia --project=. sequential_gravity/derivative_diagnostics/run_full_d2_validation.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf, Random, LinearAlgebra

println("\n" * "="^78); println(">>> Freezing a REAL sequential-iterate gravity linearization at the Frechet benchmark"); println("="^78)
frozen0 = freeze_gravity_linearization(θr0, seq_gravcol, grad_R_theta)
@printf("theta0 = Frechet benchmark. R_mean=%.4e  Rcol(R_beta)=%.4e  ok=%s\n", frozen0.R, frozen0.Rcol, frozen0.ok)
@printf("dRdθ (nonzero only for Acol block; mu frozen, gamma'_focal has none by construction):\n")
for i in eachindex(frozen0.dRdθ)
    @printf("  dRdθ[%d] = % .6e\n", i, frozen0.dRdθ[i])
end

println("\n" * "="^78); println(">>> PART 4.1: full-(D+2) fixed-dual identity test at the Frechet point"); println("="^78)
r0 = test_full_fixed_dual_identity(θr0, frozen0, γ, U, D; find_smallest=true)
@printf("delta*=%.10f  Q_full=%.10f  abs_diff=%.3e  rel_diff=%.3e  nStatus=%d\n", r0.δ_star, r0.Q_fixed, r0.abs_diff, r0.rel_diff, r0.nStatus)
identity_ok_frechet = r0.rel_diff < 1e-6
println(">>> Identity PASSES at Frechet point? ", identity_ok_frechet)
identity_ok_frechet || error("STOPPING per task instructions: Q_full(theta_k,x_k*) != delta*(theta_k) at the Frechet point. Diagnose before computing derivatives.")

println("\n" * "="^78); println(">>> Freezing a REAL sequential-iterate gravity linearization at an OFF-BENCHMARK target"); println("="^78)
γp_lo, γp_hi = KBOUNDS.γp_lo, KBOUNDS.γp_hi
θ_off = copy(θr0); θ_off[3] = θr0[3] - 0.15 * (θr0[3] - γp_lo)
frozen_off = freeze_gravity_linearization(θ_off, seq_gravcol, grad_R_theta)
@printf("theta_off: gamma'_focal=%.6f (Frechet=%.6f). R_mean=%.4e  Rcol=%.4e  ok=%s\n", θ_off[3], θr0[3], frozen_off.R, frozen_off.Rcol, frozen_off.ok)

println("\n" * "="^78); println(">>> PART 4.1b: full-(D+2) fixed-dual identity test at the off-benchmark point"); println("="^78)
r_off = test_full_fixed_dual_identity(θ_off, frozen_off, γ, U, D; find_smallest=true)
@printf("delta*=%.10f  Q_full=%.10f  abs_diff=%.3e  rel_diff=%.3e  nStatus=%d\n", r_off.δ_star, r_off.Q_fixed, r_off.abs_diff, r_off.rel_diff, r_off.nStatus)
identity_ok_off = r_off.rel_diff < 1e-6
println(">>> Identity PASSES at off-benchmark point? ", identity_ok_off)
identity_ok_off || error("STOPPING per task instructions: Q_full(theta_k,x_k*) != delta*(theta_k) off-benchmark. Diagnose before computing derivatives.")

println("\nPART 4.1 DONE -- both identity checks pass. Proceeding to FD gradient / profile comparison in a follow-up driver.")
