ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

# Re-audit the upper-bound D=4 endpoint (gamma'=0.892861, best-feasible full-A point) with the
# FIXED freeze_gravity_linearization (delta=Inf internally, gravity_ok decoupled from budget_ok).
gp_full = 0.892861
θfull = copy(θr0); θfull[3] = gp_full
@printf("--- exact_inner_divergence_at(moved-A endpoint, gamma'=%.6f) ---\n", gp_full)
frozen_moved = freeze_gravity_linearization(θfull, seq_gravcol, grad_R_theta)
@printf("gravity_ok=%s R=%.4e div_p(if gravity_ok)=%s\n", frozen_moved.ok, frozen_moved.R,
        frozen_moved.ok ? @sprintf("%.6f", frozen_moved.div_p) : "n/a")
audit_moved = exact_inner_divergence_at(θfull)
@printf("delta*_movedA = %s  gravity_ok=%s  nStatus=%s\n", audit_moved.δ_star, audit_moved.gravity_ok, audit_moved.nStatus)

@printf("\n--- exact_fixedA_divergence_at(gamma'=%.6f, A=A*) ---\n", gp_full)
θfixed = copy(θr0); θfixed[3] = gp_full
frozen_fixed = freeze_gravity_linearization(θfixed, seq_gravcol, grad_R_theta)
@printf("gravity_ok=%s R=%.4e div_p(if gravity_ok)=%s\n", frozen_fixed.ok, frozen_fixed.R,
        frozen_fixed.ok ? @sprintf("%.6f", frozen_fixed.div_p) : "n/a")
audit_fixed = exact_fixedA_divergence_at(gp_full, θr0)
@printf("delta*_fixedA = %s  gravity_ok=%s  nStatus=%s\n", audit_fixed.δ_star, audit_fixed.gravity_ok, audit_fixed.nStatus)

println("\n--- verbose seq_gravcol trace, moved-A endpoint (delta=Inf, so budget never blocks) ---")
seq_gravcol(θfull; δ=Inf, verbose=true)
println("\n--- verbose seq_gravcol trace, fixed-A endpoint (delta=Inf) ---")
seq_gravcol(θfixed; δ=Inf, verbose=true)
