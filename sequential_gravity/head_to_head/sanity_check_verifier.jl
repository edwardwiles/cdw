# ============================================================================
# Sanity-check the VERIFIER itself (2026-07-16): run the exact same independent
# trade-share check used in scrutinize_gc_t2_warm.jl against a point we
# ALREADY independently confirmed is trustworthy (LC's T1/Astar, which
# reproduced the known reference value earlier this session), to rule out a
# bug in the CHECK before concluding GC's T2/warm point is genuinely bad.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra

d = JLD2.load(joinpath(@__DIR__, "out_lc", "lc_T1_Astar.jld2"))
θsol = Float64.(d["best_feasible_theta"])
@printf("Loaded LC T1/Astar theta: gp=%.6f (saved kappa=%.6f, saved audited_delta_star=%.6f, budget=0.1)\n",
        θsol[3], d["best_feasible_kappa"], d["audited_delta_star"])
flush(stdout)

t0 = time()
col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=Inf, maxit=100, tol=5e-4)
@printf("\nFresh seq_gravcol re-solve: ok=%s wall=%.1fs\n", ok, time() - t0)

gr = gravity_residual(umat, logτ, logw, σ)
@printf("GRAVITY: R_mean=%.4e (saved R_mean_at_solution was %.4e)  %s\n",
        gr.R_mean, d["R_mean_at_solution"], abs(gr.R_mean) <= 5e-4 ? "PASS" : "FAIL")

div_p_fresh = divergence_of(p)
@printf("DIVERGENCE(p) fresh = %.6f  (budget=0.1)  %s\n", div_p_fresh, div_p_fresh <= 0.1 ? "WITHIN BUDGET" : "OVER BUDGET")

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

Kchk = zeros(W); Gchk = zeros(W, D + 1)
EK_moments_focal_norm_directgp!(Kchk, Gchk, θsol, U, (γ=γ,))
Gmom_err = maximum(abs.(sum(p .* Gchk[:, 1:D], dims=1)))
@printf("CC MOMENT E_p[G_focal]=0 cross-check: max|E_p[G]|=%.3e\n", Gmom_err)

println("\n" * "="^78)
if ok && abs(gr.R_mean) <= 5e-4 && worst_err < 1e-4
    println(">>> VERIFIER SANITY CHECK: PASSED on a known-good point. The check methodology itself")
    println(">>> is sound -- the GC T2/warm FAIL result is not an artifact of the verification script.")
else
    println(">>> VERIFIER SANITY CHECK: FAILED even on a known-good point -- there IS a bug in the")
    println(">>> verification script itself, and the GC T2/warm FAIL result should not be trusted yet.")
end
println("="^78)
println("\nSANITY_CHECK_VERIFIER DONE")
