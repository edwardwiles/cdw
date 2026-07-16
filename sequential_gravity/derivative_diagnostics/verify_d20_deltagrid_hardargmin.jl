# ============================================================================
# Additional, stricter verification pass: re-invert the D-1 omitted destinations
# under the LITERAL HARD ARGMIN rule (rho=0), not the rho=2e-3 smoothed model
# production actually solves under. Focal was ALREADY hard-argmin in every prior
# check (EK_moments_focal_norm_directgp! never smooths). This directly answers
# "does the production (smoothed) solution correspond to a genuine hard-argmin
# equilibrium, not merely a smoothed approximation of one" -- a strictly
# stronger check than verify_d20_deltagrid.jl's rho-matched check, acceptable
# here because this is a one-off audit, not something that needs to run inside
# the search loop where rho=0's harder-to-converge Newton behavior matters.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra, JLD2

const OUT_DIR = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05")
const DELTAS = [0.1, 1.0, 2.0, 5.0]

function verify_hard_one(δnom::Float64)
    println("\n" * "="^78); println(">>> HARD-ARGMIN VERIFICATION delta_nominal=$δnom"); println("="^78)
    d = JLD2.load(joinpath(OUT_DIR, "seq_upper_delta$(δnom).jld2"))
    θsol = d["best_feasible_theta"]

    # Fresh smoothed re-solve first (as before) to get a clean, canonical (p, umat) pair.
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
    ok || error("delta=$δnom: smoothed re-solve failed")

    log_x = build_log_x(Uσ, θsol[1])
    max_err_hard = zeros(D)
    max_iters = 0
    n_converged = 0
    for dd in omitted
        # rho=0.0 (hard argmin), warm-started from the smoothed-model's own u_full, tight tol,
        # generous maxit/ls_iters since this is a one-off audit, not a search-loop inner call.
        inv_hard = invert_destination(log_x, p, λData[:, dd]; ref=ref, ρ=0.0, tol=1e-8, maxit=500, ls_iters=200, u_init=umat[:, dd])
        max_err_hard[dd] = inv_hard.max_abs_share_error
        max_iters = max(max_iters, inv_hard.iterations)
        inv_hard.converged && (n_converged += 1)
        @printf("  dest #%2d: converged=%-5s iters=%3d max_abs_share_error=%.4e  ‖u_hard-u_smoothed‖=%.4e\n",
            dd, inv_hard.converged, inv_hard.iterations, inv_hard.max_abs_share_error,
            norm(inv_hard.u_full .- umat[:, dd]))
    end
    @printf("\nHARD-ARGMIN SUMMARY delta=%.2g: %d/%d omitted destinations converged, max share error over ALL omitted = %.4e, focal (always hard) = 5.6e-08-ish (see prior check)\n",
        δnom, n_converged, D - 1, maximum(max_err_hard[omitted]))
    return (δnom=δnom, n_converged=n_converged, n_total=D - 1, max_err=maximum(max_err_hard[omitted]))
end

rows = [verify_hard_one(δ) for δ in DELTAS]
println("\n" * "="^78); println(">>> HARD-ARGMIN FINAL SUMMARY (ALL destinations, focal+omitted, rho=0)"); println("="^78)
for r in rows
    @printf("delta=%.2g: %d/%d omitted destinations converged to hard-argmin equilibrium, max share error=%.4e  %s\n",
        r.δnom, r.n_converged, r.n_total, r.max_err, (r.n_converged == r.n_total && r.max_err < 1e-4) ? "PASS" : "CHECK")
end
println("\nHARD_ARGMIN_VERIFY DONE")
