ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra

# Reproduce the exact case flagged in the report: A=A* (theta_r0's Acol), gamma'=0.997561.
θtest = copy(θr0); θtest[3] = 0.997561
@printf("theta[3] (gamma') = %.6f, Acol = %s (A*)\n", θtest[3], θtest[4:3+D])

println("\n--- verbose seq_gravcol trace at this theta ---")
col, R, Rcol, umat, p, ok = seq_gravcol(θtest; δ=1.0, verbose=true)
@printf("\nFINAL: R=%.4e ok=%s\n", R, ok)

println("\n--- also check: does recover_lfd (blind D+1) succeed here at all? ---")
p0, ok0 = recover_lfd(θtest, EK_moments_focal_norm_directgp!, D + 1)
@printf("blind recover_lfd: ok=%s  divergence(p0)=%.6e  max|p0*W-1|=%.4e\n", ok0, divergence_of(p0), maximum(abs, p0 .* W .- 1))

println("\n--- and: what IS the destination-inversion convergence at the FRECHET p (uniform-ish) vs this p? ---")
# Compare gravity residual using p=uniform (uniform reweighting, i.e. "no counterfactual stretch at all")
# vs the p implied by this gamma' target, HOLDING A fixed at A* in both cases.
log_x = build_log_x(Uσ, θtest[1]); uf = focal_u(θtest)
p_unif = fill(1.0 / W, W)
um_unif = zeros(D, D); um_unif[:, focal] .= uf
for d in omitted
    inv = invert_destination(log_x, p_unif, λData[:, d]; ref=ref, ρ=ρ, tol=DEST_INV_TOL, maxit=150, ls_iters=50)
    um_unif[:, d] .= inv.u_full
    @printf("  dest %d (p=uniform): converged=%s iters=%d share_err=%.2e\n", d, inv.converged, inv.iterations, inv.max_abs_share_error)
end
R_unif = gravity_residual(um_unif, logτ, logw, σ).R_mean
@printf("R_mean at A=A*, p=UNIFORM (no counterfactual reweighting at all): %.4e\n", R_unif)

if ok0
    um_p, stats_p, all_ok_p = let
        um = zeros(D, D); um[:, focal] .= uf
        stats = Vector{Any}(undef, D); allok = true
        for d in omitted
            inv = invert_destination(log_x, p0, λData[:, d]; ref=ref, ρ=ρ, tol=DEST_INV_TOL, maxit=150, ls_iters=50)
            um[:, d] .= inv.u_full; stats[d] = inv.stats
            @printf("  dest %d (p=blind-LFD @ gamma'=%.4f): converged=%s iters=%d share_err=%.2e\n", d, θtest[3], inv.converged, inv.iterations, inv.max_abs_share_error)
            inv.converged || (allok = false)
        end
        um, stats, allok
    end
    R_p0 = gravity_residual(um_p, logτ, logw, σ).R_mean
    @printf("R_mean at A=A*, p=blind-LFD(gamma'=%.4f) BEFORE any sequential gravity-augmentation: %.4e\n", θtest[3], R_p0)
end
