# ============================================================================
# Independent verification of the D=20 real-data delta-grid results
# (batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta*.jld2).
#
# For each saved (delta, gamma'_focal, A_od) solution:
#   1. Freshly (no warm start) re-run the destination inversion (seq_gravcol) to recover
#      the FULL D x D competitiveness matrix umat (not just the focal column) and the LFD p.
#   2. Check GRAVITY holds: gravity_residual(umat,...).R_mean small.
#   3. Check ALL D trade-share moments hold (focal AND the D-1 omitted destinations),
#      independently recomputed from (umat, p) via dest_share -- NOT reusing
#      invert_destination's own internal accounting, so this is a genuine independent
#      check, not just re-reading a number the solver already reported.
#   4. Compute the EXACT delta* at this point (a fresh, cold KNITRO inner solve on the
#      full D+2-moment problem, via exact_inner_divergence_at) and compare to the nominal
#      delta budget used to produce it.
#   5. Compute exact delta*_fixedA(gamma'_focal) -- the same gamma' target, but with A
#      pinned at A* -- via exact_fixedA_divergence_at, to confirm the moved-A result
#      genuinely outperforms the fixed-A* baseline at the SAME gamma'.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     REAL_DATA_DIR=.../real_data/noah_D20 \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/verify_d20_deltagrid.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf, LinearAlgebra, JLD2, DelimitedFiles

const OUT_DIR = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05")
const DELTAS = [0.1, 1.0, 2.0, 5.0]

function verify_one(δnom::Float64)
    println("\n" * "="^78); println(">>> VERIFYING delta_nominal=$δnom"); println("="^78)
    path = joinpath(OUT_DIR, "seq_upper_delta$(δnom).jld2")
    d = JLD2.load(path)
    θsol = d["best_feasible_theta"]
    θsol === nothing && error("delta=$δnom has no best_feasible_theta saved -- cannot verify")
    gp_saved = d["best_feasible_gp"]; κ_saved = d["best_feasible_kappa"]
    @printf("saved: gamma'_focal=%.6f  kappa=%.6f\n", gp_saved, κ_saved)
    @assert isapprox(θsol[3], gp_saved; atol=1e-9) "theta_star[3] doesn't match the saved gamma' -- data integrity problem"

    # ---- 1. Fresh, cold, independent re-solve of the sequential loop at theta_sol ----
    t0 = time()
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
    t_reinvert = time() - t0
    @printf("fresh seq_gravcol re-solve: ok=%s  wall=%.1fs\n", ok, t_reinvert)
    ok || error("delta=$δnom: fresh re-solve did NOT converge (gravity or blind-LFD failure) -- saved solution is NOT reproducible")

    # ---- 2. Gravity check (independently recomputed from the freshly re-solved umat) ----
    gr = gravity_residual(umat, logτ, logw, σ)
    @printf("GRAVITY: R_mean=%.4e  R_sum=%.4e  (tol=5e-4)  %s\n", gr.R_mean, gr.R_sum, abs(gr.R_mean) <= 5e-4 ? "PASS" : "FAIL")

    # ---- 3. ALL D trade-share moments (focal + omitted), independently recomputed from (umat,p) ----
    # IMPORTANT: the omitted destinations were inverted under the SMOOTHED (softmax, rho=global
    # rho=2e-3) model throughout seq_gravcol/invert_destination -- NOT hard-max (rho=0). An
    # earlier version of this check used rho=0.0 unconditionally and found spurious 1e-3-3e-3
    # "errors" that were entirely an artifact of checking the WRONG model against data solved
    # under a DIFFERENT model (proof: re-running invert_destination fresh at rho=global rho
    # reproduces lambdaData to ~1e-8, matching its own internal DEST_INV_TOL; the same u vector
    # checked at rho=0.0 gives a spuriously different answer). Use rho=global rho here (the
    # model actually solved), and separately report the rho=0 hard-max gap as an informational
    # (not pass/fail) diagnostic of how much the smoothing itself biases the exact hard model.
    #
    # SECOND correction (caught re-checking this very script's own output): the FOCAL column is
    # NOT smoothed at all -- EK_moments_focal_norm_directgp! uses a literal hard argmin
    # (`if price < best; best=price; bo=o; end`, rho=0 conceptually), never `invert_destination`
    # (focal's u comes directly from focal_u(theta), not an inversion). Only the OMITTED
    # destinations go through invert_destination's rho=global-rho smoothing. Applying rho=global
    # rho uniformly (as an earlier version of this fix did) is correct for omitted but WRONG for
    # focal -- confirmed by it reproducing exactly the growing-with-delta error pattern the
    # rho=0-everywhere version originally showed for the omitted destinations, now showing up on
    # focal instead. Use the CORRECT rho per destination: 0 for focal, global rho for omitted.
    log_x = build_log_x(Uσ, θsol[1])
    logp = log.(p)
    share_errs = zeros(D); share_errs_hardmax = zeros(D)
    for dd in 1:D
        ρ_dd = dd == focal ? 0.0 : ρ
        model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=ρ_dd)
        share_errs[dd] = maximum(abs.(model_shares .- λData[:, dd]))
        model_shares_hard, _ = dest_share(log_x, logp, umat[:, dd]; ρ=0.0)
        share_errs_hardmax[dd] = maximum(abs.(model_shares_hard .- λData[:, dd]))
    end
    max_err_focal = share_errs[focal]
    max_err_omitted = maximum(share_errs[omitted])
    @printf("TRADE SHARES (focal at rho=0 hard-argmin, omitted at rho=%.4g smoothed -- the model actually solved for each): max|model-empirical| focal=%.2e  max over omitted=%.2e  %s\n",
        ρ, max_err_focal, max_err_omitted, max(max_err_focal, max_err_omitted) < 1e-4 ? "PASS" : "FAIL")
    worst_dest = argmax(share_errs)
    @printf("  worst destination: #%d, max share error=%.3e%s\n", worst_dest, share_errs[worst_dest], worst_dest == focal ? " (focal)" : "")
    @printf("  [informational only] hard-max (rho=0) gap vs smoothed-model solution, omitted destinations only: max=%.3e (expected O(rho)=O(%.0e), not a pass/fail check)\n",
        maximum(share_errs_hardmax[omitted]), ρ)

    # Also cross-check via the DIRECT moment formula the CC solver itself enforces
    # (E_p[G_o] = 0 exactly at the solution, for the D focal-share moments) -- independent of
    # the dest_share-based check above, since it recomputes G from scratch via the production
    # moments function rather than reusing umat/dest_share machinery at all.
    Kchk = zeros(W); Gchk = zeros(W, D + 1)
    EK_moments_focal_norm_directgp!(Kchk, Gchk, θsol, U, (γ=γ,))
    Gmom_err = maximum(abs.(sum(p .* Gchk[:, 1:D], dims=1)))
    @printf("  cross-check via E_p[G_focal]=0 (direct CC moment formula): max|E_p[G]|=%.3e\n", Gmom_err)

    # ---- 4. Exact delta* at this point ----
    t0 = time()
    audit = exact_inner_divergence_at(θsol)
    t_audit = time() - t0
    @printf("EXACT delta*(moved A) = %.6f   (nominal budget = %.2f, ratio=%.4f)   gravity_ok=%s  wall=%.1fs\n",
        audit.δ_star, δnom, audit.δ_star / δnom, audit.gravity_ok, t_audit)

    # ---- 5. Exact delta*_fixedA(gamma') at A* ----
    t0 = time()
    audit_fixed = exact_fixedA_divergence_at(θsol[3], θr0)
    t_fixed = time() - t0
    # A delta_star >= 1e9 is KNITRO's hard-failure sentinel (inner_loop_internal's -1e10 fallback,
    # sign-flipped) -- the inner (zeta,lambda) solve failed OUTRIGHT at this (gamma',A*), not just
    # "a large but finite divergence". Report this as INFEASIBLE explicitly rather than a literal
    # ~1e10 number (and rather than a nonsensical ~1e10 "Delta_delta").
    fixedA_infeasible = !audit_fixed.gravity_ok || audit_fixed.δ_star >= 1e9
    if fixedA_infeasible
        @printf("EXACT delta*_fixedA(same gamma', A=A*) = INFEASIBLE (gravity_ok=%s, raw solver value=%.3e -- A* cannot reach this gamma' at all, not merely at higher cost)  wall=%.1fs\n",
            audit_fixed.gravity_ok, audit_fixed.δ_star, t_fixed)
        Δδ = Inf
    else
        Δδ = audit_fixed.δ_star - audit.δ_star
        @printf("EXACT delta*_fixedA(same gamma', A=A*) = %.6f   Delta_delta = %.6f  (positive => moving A helped)  gravity_ok=%s  wall=%.1fs\n",
            audit_fixed.δ_star, Δδ, audit_fixed.gravity_ok, t_fixed)
    end

    return (δnom=δnom, gp=θsol[3], κ=κ_saved, R_mean=gr.R_mean, max_share_err_focal=max_err_focal,
            max_share_err_omitted=max_err_omitted, Gmom_err=Gmom_err, δ_star_movedA=audit.δ_star,
            δ_star_fixedA=audit_fixed.δ_star, Δδ=Δδ, fixedA_infeasible=fixedA_infeasible,
            gravity_ok_moved=audit.gravity_ok, gravity_ok_fixed=audit_fixed.gravity_ok)
end

rows = [verify_one(δ) for δ in DELTAS]

println("\n" * "="^78); println(">>> SUMMARY"); println("="^78)
@printf("%6s %10s %10s %12s %14s %14s %14s %14s %10s\n",
    "delta", "gamma'", "kappa", "R_mean", "max_shr_err", "delta*_moved", "delta*_fixed", "Δδ", "delta*/δnom")
for r in rows
    fixedA_str = r.fixedA_infeasible ? "INFEASIBLE" : @sprintf("%.6f", r.δ_star_fixedA)
    Δδ_str = r.fixedA_infeasible ? "n/a (A* infeasible)" : @sprintf("%.6f", r.Δδ)
    @printf("%6.2f %10.6f %10.6f %12.2e %14.2e %14.6f %14s %14s %10.4f\n",
        r.δnom, r.gp, r.κ, r.R_mean, max(r.max_share_err_focal, r.max_share_err_omitted),
        r.δ_star_movedA, fixedA_str, Δδ_str, r.δ_star_movedA / r.δnom)
end

open(joinpath(@__DIR__, "verify_d20_deltagrid_results.csv"), "w") do io
    writedlm(io, ["delta_nom" "gamma_p" "kappa" "R_mean" "max_share_err_focal" "max_share_err_omitted" "Gmom_err" "delta_star_movedA" "delta_star_fixedA_or_INFEASIBLE" "Delta_delta_or_Inf" "gravity_ok_moved" "gravity_ok_fixed"], ',')
    for r in rows
        fixedA_val = r.fixedA_infeasible ? "INFEASIBLE" : r.δ_star_fixedA
        writedlm(io, [[r.δnom r.gp r.κ r.R_mean r.max_share_err_focal r.max_share_err_omitted r.Gmom_err r.δ_star_movedA fixedA_val r.Δδ r.gravity_ok_moved r.gravity_ok_fixed]], ',')
    end
end

println("\nVERIFY_D20_DELTAGRID DONE")
