# ============================================================================
# Continuation 10, Phase 7: smoke test for the QMC draw-injection infra
# (qmc_context_real_d20.jl + qmc_draws.jl). Small W (not the real 80000) for
# speed -- purpose is to verify WIRING correctness, not to produce any
# reportable number:
#   1. d20_real_setup_qmc, fed the SAME seed-888 pseudorandom U production
#      would draw itself, reproduces d20_real_setup's ctx EXACTLY (theta0_up,
#      gp0, bounds, Delta at the calibration point) -- i.e. the fork is a
#      faithful byte-for-byte copy of the production chain except at the one
#      intended injection point.
#   2. halton_U / sobol_U produce valid (finite, positive, correct-shape) F*
#      draw matrices and wire cleanly through d20_real_setup_qmc to a working
#      evaluate_fullA call.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # unify-random-draw-production-pipeline 2026-07-30: qmc_context_real_d20.jl deleted, d20_real_setup now takes U= directly
include(joinpath(@__DIR__, "qmc_draws.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "winners.jl"))
using Sobol, Random, Statistics, LinearAlgebra

const W_SMOKE = parse(Int, get(ENV, "QMC_SMOKE_W", "80000"))

println("="^80); println("Phase 7 wiring smoke test (W=", W_SMOKE, ")"); println("="^80)

# ---- 1. Equivalence: qmc fork w/ seed-888 pseudorandom U == production d20_real_setup ----
ctx_prod = d20_real_setup(W = W_SMOKE, find_smallest = true)
U_prod_seed888 = pseudorandom_U(W_SMOKE, ctx_prod.D; seed = 888)
ctx_qmc_same = d20_real_setup(W = W_SMOKE, U = U_prod_seed888, find_smallest = true)

println("[1] D match: ", ctx_prod.D == ctx_qmc_same.D)
println("    theta0_up match: ", ctx_prod.θ0_up == ctx_qmc_same.θ0_up, " (max abs diff=", maximum(abs.(ctx_prod.θ0_up .- ctx_qmc_same.θ0_up)), ")")
println("    gp0 match: ", ctx_prod.θ0_up[3+ctx_prod.D] == ctx_qmc_same.θ0_up[3+ctx_qmc_same.D])
println("    bounds match: ", ctx_prod.bounds == ctx_qmc_same.bounds)
println("    U match: ", ctx_prod.U == ctx_qmc_same.U)

r_prod = evaluate_fullA(ctx_prod.θ0_up[vcat(ctx_prod.free_idx)], ctx_prod; warm = false)
r_qmc  = evaluate_fullA(ctx_qmc_same.θ0_up[vcat(ctx_qmc_same.free_idx)], ctx_qmc_same; warm = false)
println("    r_prod inner_status=", r_prod.inner_status, " error_reason=", r_prod.error_reason)
println("    r_qmc  inner_status=", r_qmc.inner_status, " error_reason=", r_qmc.error_reason)
println("    Delta_dual match at calibration point: prod=", r_prod.Delta_dual, " qmc-fork=", r_qmc.Delta_dual,
        " diff=", abs(r_prod.Delta_dual - r_qmc.Delta_dual))
status_match = r_prod.inner_status == r_qmc.inner_status
delta_match = (isnan(r_prod.Delta_dual) && isnan(r_qmc.Delta_dual)) || abs(r_prod.Delta_dual - r_qmc.Delta_dual) < 1e-10
equivalence_ok = ctx_prod.θ0_up == ctx_qmc_same.θ0_up && ctx_prod.U == ctx_qmc_same.U && status_match && delta_match
println("    EQUIVALENCE: ", equivalence_ok ? "PASS" : "FAIL")

# ---- 2. Halton / Sobol wiring ----
D = ctx_prod.D
U_halton = halton_U(W_SMOKE, D; seed = 101)
U_sobol  = sobol_U(W_SMOKE, D; seed = 101)
println("\n[2] Halton U: shape=", size(U_halton), " finite=", all(isfinite, U_halton), " positive=", all(U_halton .>= 0), " mean=", round(mean(U_halton), digits=3), " (Exp(1) target 1.0)")
println("    Sobol  U: shape=", size(U_sobol), " finite=", all(isfinite, U_sobol), " positive=", all(U_sobol .>= 0), " mean=", round(mean(U_sobol), digits=3), " (Exp(1) target 1.0)")

ctx_halton = d20_real_setup(W = W_SMOKE, U = U_halton, find_smallest = true)
ctx_sobol  = d20_real_setup(W = W_SMOKE, U = U_sobol, find_smallest = true)
r_halton = evaluate_fullA(ctx_halton.θ0_up[vcat(ctx_halton.free_idx)], ctx_halton; warm = false)
r_sobol  = evaluate_fullA(ctx_sobol.θ0_up[vcat(ctx_sobol.free_idx)], ctx_sobol; warm = false)
println("    Halton ctx: inner_status=", r_halton.inner_status, " Delta=", r_halton.Delta_dual)
println("    Sobol  ctx: inner_status=", r_sobol.inner_status, " Delta=", r_sobol.Delta_dual)
wiring_ok = r_halton.inner_status in (0,-100,-101,-103) && r_sobol.inner_status in (0,-100,-101,-103) && isfinite(r_halton.Delta_dual) && isfinite(r_sobol.Delta_dual)
println("    WIRING: ", wiring_ok ? "PASS" : "FAIL")

println("\n", "="^80)
println("SMOKE TEST SUMMARY: equivalence_ok=", equivalence_ok, " wiring_ok=", wiring_ok)
println("="^80)
