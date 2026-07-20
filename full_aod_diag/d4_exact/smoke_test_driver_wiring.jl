# Smoke test: exercise run_profile_checkpointed end-to-end (real KNITRO,
# short maxtime_real) after wiring screened_eval -> evaluate_fullA_screened_ranged,
# to confirm the driver-level wiring change (not just the standalone screen
# functions already validated) works: ctx/rsc construction, warm-cache seed,
# a few real outer iterations, checkpoint save with the new 7-field
# screen_counts shape, and the final report line.
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using DelimitedFiles

zfree = vec(readdlm(joinpath(@__DIR__, "..", "..", "diagnostics", "infeasibility_points",
                              "feasible_delta1_anchor_pushed_r8_zfree.csv")))
g0 = 0.9517637267993538   # gamma_prime_focal_at_checkpoint from that same catalogue point's json

ckpt_dir = joinpath(@__DIR__, "..", "..", "results", "smoke_test_driver_wiring")
rm(ckpt_dir; recursive = true, force = true)

result = run_profile_checkpointed("smoke", g0, true, zfree;
    maxtime_real = 60.0, hessopt_tag = "sr1", W_in = 80000, delta_in = 1.0,
    ckpt_dir = ckpt_dir, checkpoint_interval_s = 30.0)

println("\n=== SMOKE TEST RESULT ===")
println("knitro_status = ", result.knitro_status)
println("n_eval = ", result.n_eval)
println("screen_counts = ", result.screen_counts)
println("best = ", result.best)
@assert result.n_eval > 0 "no evaluations ran"
@assert haskey(result.screen_counts, :envelope) "screen_counts missing new :envelope field -- wiring did not take effect"
@assert haskey(result.screen_counts, :winning_range) "screen_counts missing new :winning_range field"
@assert haskey(result.screen_counts, :safety_net) "screen_counts missing new :safety_net field"
println("SMOKE TEST PASSED")
