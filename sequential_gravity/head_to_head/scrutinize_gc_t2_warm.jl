# ============================================================================
# Ad-hoc scrutiny (2026-07-16): GC's T2/warm result (kappa=0.10555, div_p=0.110
# vs budget=1.0, relDeltaA=6.96 -- a huge A_od movement) is MUCH better than
# LC's own carefully-validated delta=1.0 result (kappa=0.081518) and even beats
# T3's own looser-budget reference (kappa=0.091491). That pattern (extreme
# A_od movement + big win + audit failure) is exactly what should be
# double-checked, not taken at face value. This does a FRESH, independent
# re-solve (same pattern as verify_d20_deltagrid.jl) -- NOT reusing GC's own
# eval_candidate/fitness machinery at all -- to rule out a subtle bug in how
# "feasible" was determined during the search.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra

d = JLD2.load(joinpath(@__DIR__, "out_gc", "gc_T2_warm.jld2"))
θsol = Float64.(d["theta_star"])
@printf("Loaded GC T2/warm theta_star: gp=%.6f (saved kappa=%.6f, saved div_p=%.6f, saved relDeltaA=%.3f)\n",
        θsol[3], d["kappa"], d["div_p"], d["relDeltaA"])
flush(stdout)

# ---- 1. Fresh, cold, independent re-solve (maxit raised, no warm start, no reuse of GC's own state) ----
t0 = time()
col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
@printf("\nFresh seq_gravcol re-solve: ok=%s wall=%.1fs\n", ok, time() - t0)
if !ok
    println("FRESH RE-SOLVE FAILED TO CONVERGE -- this alone is a red flag, the saved solution may not be reproducible.")
else
    gr = gravity_residual(umat, logτ, logw, σ)
    @printf("GRAVITY: R_mean=%.4e (saved R was %.4e)  %s\n", gr.R_mean, d["R"], abs(gr.R_mean) <= 5e-4 ? "PASS" : "FAIL")

    div_p_fresh = divergence_of(p)
    @printf("DIVERGENCE(p) fresh = %.6f  (saved div_p was %.6f, budget=1.0)  %s\n",
            div_p_fresh, d["div_p"], div_p_fresh <= 1.0 ? "WITHIN BUDGET" : "OVER BUDGET")

    # ---- 2. Independent trade-share check at EVERY destination (focal at rho=0, omitted at rho=global rho) ----
    log_x = build_log_x(Uσ, θsol[1])
    logp = log.(p)
    worst_err = 0.0
    for dd in 1:D
        ρ_dd = dd == focal ? 0.0 : ρ
        model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=ρ_dd)
        err = maximum(abs.(model_shares .- λData[:, dd]))
        global worst_err = max(worst_err, err)
    end
    @printf("TRADE SHARES: max|model-empirical| over all %d destinations = %.3e  %s\n",
            D, worst_err, worst_err < 1e-4 ? "PASS" : "FAIL")

    # ---- 3. Direct CC moment cross-check ----
    Kchk = zeros(W); Gchk = zeros(W, D + 1)
    EK_moments_focal_norm_directgp!(Kchk, Gchk, θsol, U, (γ=γ,))
    Gmom_err = maximum(abs.(sum(p .* Gchk[:, 1:D], dims=1)))
    @printf("CC MOMENT E_p[G_focal]=0 cross-check: max|E_p[G]|=%.3e\n", Gmom_err)

    # ---- 4. relDeltaA sanity ----
    Acol_star = θr0[4:3+D]
    relΔA = norm(θsol[4:3+D] .- Acol_star) / norm(Acol_star)
    @printf("relDeltaA (fresh) = %.4f  (saved was %.4f)\n", relΔA, d["relDeltaA"])

    println("\n" * "="^78)
    if ok && abs(gr.R_mean) <= 5e-4 && div_p_fresh <= 1.0 + 1e-6 && worst_err < 1e-4
        println(">>> RESULT: independently re-verified -- gravity, divergence budget, and ALL D trade shares")
        println(">>> genuinely hold at this point. The kappa=0.1055 result appears to be a REAL feasible point,")
        println(">>> not a bug artifact. It is a striking result (beats LC's own careful search) that still")
        println(">>> deserves scrutiny of WHY the local method (LC) missed this basin, but it isn't fabricated.")
    else
        println(">>> RESULT: independent re-verification FOUND A PROBLEM -- see failures above. Do not trust")
        println(">>> this point without further investigation.")
    end
    println("="^78)
end
println("\nSCRUTINIZE_GC_T2_WARM DONE")
