# ============================================================================
# Re-audit (2026-07-16, post recover_lfd fix): LC's two other 1e10-audited
# points (T2/rand1, T3/warm) -- were these also spurious like GC's T2/warm, or
# was that an isolated case? Simplified check per user's guidance: verify
# gravity DIRECTLY on the fixed A_od (gravity_residual, no linearized-moment
# reconstruction needed) plus trade shares at every destination.
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra

function check_point(label::String, θsol::Vector{Float64}, budget::Float64)
    println("\n" * "="^78); println(">>> $label"); println("="^78)
    col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=budget, maxit=100, tol=5e-4)
    @printf("POST-FIX seq_gravcol(budget=%.2f): ok=%s\n", budget, ok)
    if !ok
        println(">>> Now correctly rejected as infeasible by the fixed recover_lfd.")
        return
    end
    gr = gravity_residual(umat, logτ, logw, σ)
    div_p = divergence_of(p)
    log_x = build_log_x(Uσ, θsol[1])
    logp = log.(p)
    worst_err = 0.0
    for dd in 1:D
        ρ_dd = dd == focal ? 0.0 : ρ
        model_shares, _ = dest_share(log_x, logp, umat[:, dd]; ρ=ρ_dd)
        err = maximum(abs.(model_shares .- λData[:, dd]))
        global worst_err = max(worst_err, err)
    end
    @printf("GRAVITY: R_mean=%.4e  DIVERGENCE: div_p=%.6f (budget=%.2f)  TRADE SHARES: worst_err=%.3e  %s\n",
            gr.R_mean, div_p, budget, worst_err, worst_err < 1e-4 ? "GENUINE" : "STILL SPURIOUS")
end

d1 = JLD2.load(joinpath(@__DIR__, "out_lc", "lc_T2_rand1.jld2"))
check_point("LC T2/rand1 (was: kappa=$(d1["best_feasible_kappa"]), audited_delta_star=1e10)",
            Float64.(d1["best_feasible_theta"]), 1.0)

d2 = JLD2.load(joinpath(@__DIR__, "out_lc", "lc_T3_warm.jld2"))
check_point("LC T3/warm (was: kappa=$(d2["best_feasible_kappa"]), audited_delta_star=1e10)",
            Float64.(d2["best_feasible_theta"]), 2.0)

println("\nREAUDIT_LC_POINTS DONE")
