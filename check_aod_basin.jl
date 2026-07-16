using JLD2, Printf

dir = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/batch_out_realD20_W80000"
AOD_STAR = 832716.0469855476  # theta_r0's constant A[.,focal] value in the gamma_focal===1 gauge

for delta in ["0.1", "1.0", "2.0"]
    path = joinpath(dir, "seq_upper_delta$(delta).jld2")
    isfile(path) || continue
    d = JLD2.load(path)
    theta_star = d["theta_star"]
    Aod = theta_star[4:end]
    rel_dev = (Aod .- AOD_STAR) ./ AOD_STAR
    @printf("\n=== delta=%s  kappa=%.6f  nStatus=%d ===\n", delta, d["kappa"], d["nStatus"])
    @printf("  A_od range: [%.6g, %.6g]   A_od* = %.6g\n", minimum(Aod), maximum(Aod), AOD_STAR)
    @printf("  max |relative deviation from A_od*|: %.6e\n", maximum(abs, rel_dev))
    @printf("  mean |relative deviation from A_od*|: %.6e\n", sum(abs, rel_dev) / length(rel_dev))
    @printf("  A_od values: %s\n", Aod)

    if haskey(d, "best_feasible_theta") && d["best_feasible_theta"] !== nothing
        bAod = d["best_feasible_theta"][4:end]
        brel = (bAod .- AOD_STAR) ./ AOD_STAR
        @printf("  [best-feasible] kappa=%.6f  max|rel dev|=%.6e\n", d["best_feasible_kappa"], maximum(abs, brel))
    end
end
