# ============================================================================
# Re-runs the ACTUAL standard constrained method (outer_solve_nested_cached,
# extremize gamma' subject to delta*<=budget, KNITRO SQP) -- the same method
# that produced the section-7.2 D=20 real-data result
# (kappa=0.081518, USE_VAR_SCALING=true, SCALING_POWER=0.5) -- but from a
# RANDOMIZED Acol starting point instead of the usual cold start at theta_r0
# (A*). gamma'_focal's own starting value is left at theta_r0's own value
# (only Acol is randomized), per explicit user request.
#
# IMPORTANT: run_profiled_production.jl's own batch loop (`run_one_bound`)
# does NOT thread `scaling_power` through to `outer_solve_nested_cached` at
# all (silently defaults to 1.0, not the 0.5 that actually produced the good
# result) -- confirmed by direct inspection of the current checked-out file.
# This script therefore calls `outer_solve_nested_cached` DIRECTLY with
# `scaling_power=0.5` explicit, to faithfully match the original run's
# actual settings rather than whatever the current batch-loop entry point
# happens to default to.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true SEED=1 SIGMA=0.5 \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/run_scaled_constrained_d20_randomstart.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using Printf, JLD2, LinearAlgebra, Random

const SEED = parse(Int, get(ENV, "SEED", "1"))
const SIGMA = parse(Float64, get(ENV, "SIGMA", "0.5"))
const DELTA = 1.0
const GRAD_METHOD = :fixed_dual_fd_full
const SCALING_POWER = 0.5

@assert D == 20 && W == 80000

Random.seed!(SEED)
θinit = copy(θr0)
noise = SIGMA .* randn(D)
θinit[4:3+D] .*= exp.(noise)
relΔA0 = norm(θinit[4:3+D] .- θr0[4:3+D]) / norm(θr0[4:3+D])

println("="^78)
println(">>> Standard CONSTRAINED search (use_var_scaling=true, scaling_power=$SCALING_POWER), D=20 REAL DATA, W=80000")
println("    RANDOM Acol start: seed=$SEED sigma=$SIGMA  relΔA(start vs A*)=$(round(relΔA0, digits=4))")
println("="^78)
flush(stdout)

t0 = time()
# NOTE: the 5th returned value (`best_κ` per outer_solve_nested_cached's own naming) is a
# documented MISNOMER (see full_d2_correction_report.md section 4 item 2) -- it actually
# stores the raw K[1]=gamma'_focal target, not real kappa. Matching the established pattern
# in run_full_d2_outer_loop_test.jl, this driver ignores it and derives gp from best_θ[3]
# directly, converting via gp2kappa explicitly.
gp, θstar, nStatus, best_θ, _best_κ_misnomer, best_warm, cache = outer_solve_nested_cached(
    true, θinit; use_exact_grad = true, δ = DELTA, gradient_method = GRAD_METHOD,
    use_var_scaling = true, scaling_power = SCALING_POWER)
wall = time() - t0

κ_raw = gp2kappa(gp)
@printf("\n  raw KNITRO endpoint: gamma'=%.6f -> kappa=%.6f  status=%d  wall=%.1fs\n", gp, κ_raw, nStatus, wall)

if best_θ === nothing
    println("  best-feasible: NONE FOUND")
    best_gp = NaN; κ_best = NaN; relΔA_best = NaN
else
    best_gp = best_θ[3]
    κ_best = gp2kappa(best_gp)
    relΔA_best = norm(best_θ[4:3+D] .- θr0[4:3+D]) / norm(θr0[4:3+D])
    @printf("  best-feasible: gamma'=%.6f -> kappa=%.6f  relΔA(vs A*)=%.4f\n", best_gp, κ_best, relΔA_best)
end

out_path = joinpath(@__DIR__, "scaled_constrained_D20_randomstart_seed$(SEED)_sigma$(SIGMA).jld2")
JLD2.save(out_path, Dict(
    "D" => D, "W" => W, "delta" => DELTA, "bound" => "upper", "gradient_method" => String(GRAD_METHOD),
    "use_var_scaling" => true, "scaling_power" => SCALING_POWER,
    "seed" => SEED, "sigma" => SIGMA, "theta_init" => θinit, "relΔA_init" => relΔA0,
    "gamma_p" => gp, "kappa" => κ_raw, "nStatus" => nStatus, "wall" => wall,
    "best_feasible_theta" => best_θ, "best_feasible_gp" => best_gp, "best_feasible_kappa" => κ_best,
    "relΔA_best" => relΔA_best,
    "starting_point_source" => "randomized Acol (seed=$SEED, sigma=$SIGMA)",
))
println("\nSaved: $out_path")
println("\nSCALED CONSTRAINED D20 RANDOM-START RUN DONE")
