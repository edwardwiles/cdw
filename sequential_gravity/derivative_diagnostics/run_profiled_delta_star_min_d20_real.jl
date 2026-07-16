# ============================================================================
# D=20 REAL-DATA test of the profiled delta*(A_od) minimization reformulation
# (see run_profiled_delta_star_min.jl for the D=4 version and full rationale,
# and full_d2_correction_report.md section 7/7.1/7.2 for the motivating
# gamma'-vs-A_od gradient-scale problem this is meant to sidestep).
#
# Per EXPLICIT user request: do NOT re-run the constrained full-A search here
# (it took ~46 min with KNITRO variable scaling -- report section 7.2).
# Instead, reuse EXACTLY the already-computed result from that run, saved at
#   sequential_gravity/batch_out_realD20_W80000_fixeddualfdfull_scaled05/seq_upper_delta1.0.jld2
# Specs of that saved run, confirmed by direct inspection of the jld2 (not
# just the report's prose) and cross-checked against its own run log
# (sequential_gravity/d20_upper_delta1_W80000_scaled05_run.log):
#   D=20, W=80000, REAL data (FAKEDATA=3, REAL_DATA_DIR=real_data/noah_D20,
#   matching run_d20_upper_delta1_W80000_fixeddualfdfull.sh's own config),
#   bound=upper (find_smallest=true), delta budget=1.0,
#   gradient_method=fixed_dual_fd_full, use_var_scaling=true (scaling_power=0.5
#   per the report -- not itself a key in the jld2, but the directory name
#   ("_scaled05") and the run log's "use_var_scaling=true" line both confirm
#   this is the section-7.2 run, not the unscaled section-7 one).
#   best_feasible_gp = 0.950259956422648 -> kappa = 0.081518 -- this is the
#   "GT" (gamma'_focal target) the user asked to reuse exactly.
# The run log's own point estimate (gamma'_focal(F*)=0.987762,
# kappa=0.020314) matches the jld2's kappa_point_estimate=0.0203135... to 4
# decimals -- confirms this driver's freshly-built theta_r0 (same D/W/
# real-data config) lines up with the saved run's setup; asserted below too.
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true MAXTIME=1800 \
#     julia -t 19 --project=. sequential_gravity/derivative_diagnostics/run_profiled_delta_star_min_d20_real.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))

using BlackBoxOptim, Printf, JLD2, LinearAlgebra

const SAVED_PATH = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05", "seq_upper_delta1.0.jld2")
const MAXTIME = parse(Float64, get(ENV, "MAXTIME", "1800"))   # seconds per minimization run (2 runs total)

@assert D == 20 "this driver is D=20-specific"
@assert W == 80000 "this driver is W=80000-specific (matches the saved scaled run)"

saved = JLD2.load(SAVED_PATH)
@assert saved["D"] == 20 && saved["delta"] == 1.0 && saved["bound"] == "upper" &&
        saved["gradient_method"] == "fixed_dual_fd_full" && saved["use_var_scaling"] == true "saved jld2 spec mismatch -- not the section-7.2 scaled run"

gammap_star = saved["best_feasible_gp"]
κ_saved = saved["best_feasible_kappa"]
θ_saved_best = Float64.(saved["best_feasible_theta"])
Aod_constrained = θ_saved_best[4:3+D]

# Sanity: this driver's freshly-built θr0 (same D/W/real-data config) must carry the
# SAME mu/sigma as the saved run's theta -- confirms we're pointed at the identical setup,
# not silently using a different real-data snapshot.
@printf("  sanity check: this run's θr0[1:2] = %s\n", θr0[1:2])
@printf("  sanity check: saved best_feasible_theta[1:2] = %s\n", θ_saved_best[1:2])
@assert isapprox(θr0[1], θ_saved_best[1]; rtol = 1e-8) && isapprox(θr0[2], θ_saved_best[2]; rtol = 1e-8) "MISMATCH: this driver's real-data setup does not match the saved D=20 scaled run -- do not trust results below"
println("  sanity check PASSED: mu/sigma match the saved run exactly.")

@printf("\nReusing saved GT (NOT re-solving the constrained search): gamma'_focal* = %.12f -> kappa = %.6f\n", gammap_star, κ_saved)
@printf("  (from %s)\n\n", SAVED_PATH)

const mu0 = θr0[1]
const sigma0 = θr0[2]
const Acol_star = θr0[4:3+D]

neval = Ref(0)
function delta_star_of_Aod(Aod::AbstractVector{Float64})
    neval[] += 1
    θ = vcat(mu0, sigma0, gammap_star, Aod)
    exact_inner_divergence_at(θ).δ_star
end

search_range = [(θ_lo[3+i], θ_hi[3+i]) for i in 1:D]

t0 = time()
δ_at_Astar = delta_star_of_Aod(Acol_star)
@printf("  delta*(A_od = A*)                    = %.8f  (%.1fs)\n", δ_at_Astar, time() - t0)
t0 = time()
δ_at_constrained = delta_star_of_Aod(Aod_constrained)
@printf("  delta*(A_od = scaled-search endpoint) = %.8f  (%.1fs)\n", δ_at_constrained, time() - t0)

function run_minimization(label::String, x0::Vector{Float64})
    println("\n  --- minimizing delta*(A_od) from $label (MaxTime=$(MAXTIME)s) ---")
    t0 = time()
    neval[] = 0
    res = bboptimize(delta_star_of_Aod, x0; SearchRange = search_range,
        Method = :generating_set_search, MaxTime = MAXTIME, TraceMode = :compact, TraceInterval = 30.0)
    wall = time() - t0
    xbest = best_candidate(res); δbest = best_fitness(res)
    @printf("  [%s] delta*_min = %.8f  (nevals=%d, wall=%.1fs)\n", label, δbest, neval[], wall)
    relΔA = norm(xbest .- Acol_star) / norm(Acol_star)
    @printf("  [%s] rel‖A_od_min - A*‖ = %.4f\n", label, relΔA)
    (label = label, x0 = x0, xbest = xbest, δbest = δbest, nevals = neval[], wall = wall, relΔA = relΔA)
end

res_fromAstar = run_minimization("A_od = A*", copy(Acol_star))
res_fromConstrained = run_minimization("A_od = scaled-constrained-search endpoint", copy(Aod_constrained))

println("\n" * "="^78); println(">>> SUMMARY (D=20 real data)"); println("="^78)
@printf("  gamma'_focal* (GT, reused from saved scaled run)   = %.12f (kappa=%.6f)\n", gammap_star, κ_saved)
@printf("  original delta BUDGET used by that constrained run  = 1.000000\n")
@printf("  exact delta*(A*, same gamma')                       = %.8f\n", δ_at_Astar)
@printf("  exact delta*(scaled-search A_od, same gamma')       = %.8f\n", δ_at_constrained)
@printf("  minimized delta*, start=A*                          = %.8f  (nevals=%d, wall=%.1fs)\n",
    res_fromAstar.δbest, res_fromAstar.nevals, res_fromAstar.wall)
@printf("  minimized delta*, start=scaled-search A_od          = %.8f  (nevals=%d, wall=%.1fs)\n",
    res_fromConstrained.δbest, res_fromConstrained.nevals, res_fromConstrained.wall)
δmin = min(res_fromAstar.δbest, res_fromConstrained.δbest)
gap = 1.0 - δmin
@printf("\n  gap = budget - min(delta*_min over both starts) = 1.000000 - %.6f = %.6f\n", δmin, gap)
if gap > 0.02
    println("  => MEANINGFULLY LOWER than the budget: real headroom left on the table even after KNITRO-scaling.")
elseif gap < -0.02
    println("  => the reformulation's minimum EXCEEDS the budget -- unexpected, needs investigation")
    println("     (the scaled-search's own best-feasible point is itself delta*<=1-feasible by construction, so")
    println("     the minimizer starting AT that point should never do worse -- check optimizer convergence / MaxTime).")
else
    println("  => approximately EQUAL to the budget: the scaled constrained search already found (close to) the true optimum.")
end

out_path = joinpath(@__DIR__, "profiled_delta_star_min_D20_W80000_delta1.0_upper_real.jld2")
JLD2.save(out_path, Dict(
    "D" => D, "W" => W, "delta_budget" => 1.0, "bound" => "upper",
    "gammap_star" => gammap_star, "kappa_saved" => κ_saved, "source_jld2" => SAVED_PATH,
    "Acol_star" => Acol_star, "Aod_constrained" => Aod_constrained,
    "delta_at_Astar" => δ_at_Astar, "delta_at_constrained" => δ_at_constrained,
    "res_fromAstar" => res_fromAstar, "res_fromConstrained" => res_fromConstrained,
))
println("\nSaved: $out_path")
println("\nPROFILED DELTA* MINIMIZATION TEST (D=20 REAL DATA) DONE")
