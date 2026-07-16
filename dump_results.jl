using JLD2
dir = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/batch_out_realD20"
for bound in ["lower", "upper"], delta in ["0.1", "1.0", "10.0"]
    path = joinpath(dir, "seq_$(bound)_delta$(delta).jld2")
    isfile(path) || continue
    d = JLD2.load(path)
    println("--- $bound delta=$delta ---")
    println("  KNITRO-terminal: kappa=$(d["kappa"])  gamma_p=$(d["gamma_p"])  nStatus=$(d["nStatus"])  gravity_feasible=$(d["gravity_feasible"])  R_mean=$(d["R_mean_at_solution"])  wall=$(d["wall"])")
    println("  best-feasible:   kappa=$(d["best_feasible_kappa"])  gamma_p=$(d["best_feasible_gp"])  gravity_ok=$(d["best_feasible_gravity_ok"])")
    println("  inner_solves=$(d["inner_solves"])  cold=$(d["cold_inner"])  warm=$(d["warm_started_inner"])")
end
