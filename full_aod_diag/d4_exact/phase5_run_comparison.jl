# ============================================================================
# Phase 5 driver: load the 5-start sequential/profiled production multistart
# results (produced by sequential_gravity/run_profiled_production_phase5_multistart.jl
# in the PRODUCTION worktree, START_ID=0..4, BOUND=both), pick the best
# feasible kappa per bound direction across all 5 starts (per
# sequential_methodology.tex "Checks, validation, and multistart": "the
# reported number at each delta is the best across all five starts"),
# reconstruct the full-A point for the winning (upper-bound) start, evaluate
# it in the exact full-A oracle, and print the final comparison table.
#
# Usage: julia --project=. full_aod_diag/d4_exact/phase5_run_comparison.jl <OUT_DIR_GLOB_ROOT>
#   where OUT_DIR_GLOB_ROOT is the production-worktree batch_out_phase5_multistart_<TS> dir
#   containing start0/ .. start4/ subdirs.
# ============================================================================
include(joinpath(@__DIR__, "phase5_sequential_reconstruction.jl"))
using JLD2, Printf

const SEQ_BATCH_ROOT = length(ARGS) >= 1 ? ARGS[1] :
    error("usage: julia phase5_run_comparison.jl <batch_out_phase5_multistart_TS dir>")

const FIXED_A_BENCHMARK_KAPPA = 0.1439805232
const FULLA_INCUMBENT_KAPPA = 0.17176461388430053
const FULLA_INCUMBENT_GP = 0.8930839180420251

function load_seq_result(start_id::Int, bound::String; δ::Real = 1)
    # NOTE: matches run_profiled_production*.jl's own result_path naming, which interpolates the
    # RAW DELTA_GRID element (params.δ_ref=1, an Int literal) -- "delta1.jld2", not "delta1.0.jld2".
    # delta must therefore be passed as an Int here (δ=1) to match on disk.
    path = joinpath(SEQ_BATCH_ROOT, "start$start_id", "seq_$(bound)_delta$(δ).jld2")
    isfile(path) || return nothing
    d = JLD2.load(path)
    get(d, "done", false) === true || return nothing
    return d
end

function best_across_starts(bound::String; δ::Real = 1, n_starts::Int = 5)
    best = nothing
    for sid in 0:n_starts-1
        d = load_seq_result(sid, bound; δ = δ)
        d === nothing && (@printf("  start%d (%s): NO RESULT FILE\n", sid, bound); continue)
        @assert d["D"] == 4 "start$sid $bound result has D=$(d["D"]), expected D=4 -- stale/mismatched file, per the investigation's documented past mistake (memory: two candidate JLD2 files that LOOKED D=4 were actually stale D=10 results)"
        bκ = d["best_feasible_kappa"]; bok = d["best_feasible_gravity_ok"]
        rκ = d["kappa"]; rok = d["gravity_feasible"]; nSt = d["nStatus"]
        @printf("  start%d (%s): raw kappa=%.6f (feasible=%s, nStatus=%d)  best-feasible kappa=%s (feasible=%s)\n",
                sid, bound, rκ, rok, nSt, isnan(bκ) ? "NaN" : @sprintf("%.6f", bκ), bok)
        if bok && !isnan(bκ)
            better = bound == "upper" ? (best === nothing || bκ > best.bκ) : (best === nothing || bκ < best.bκ)
            better && (best = (start_id = sid, bκ = bκ, θ = d["best_feasible_theta"], d = d))
        end
    end
    return best
end

println("="^100)
println("Sequential/profiled production multistart results -- 5 starts, BOTH bounds, delta=1, D=4, W=8000")
println("="^100)
println("\n--- LOWER bound (gamma'_focal MAXIMIZED -> kappa MINIMIZED) ---")
best_lower = best_across_starts("lower")
println("\n--- UPPER bound (gamma'_focal MINIMIZED -> kappa MAXIMIZED) ---")
best_upper = best_across_starts("upper")

println("\n" * "="^100)
if best_upper === nothing
    println("NO FEASIBLE UPPER-BOUND POINT FOUND ACROSS ANY OF THE 5 STARTS.")
else
    @printf("Best sequential UPPER-bound result: START_ID=%d, best_feasible_kappa=%.6f\n",
            best_upper.start_id, best_upper.bκ)
end
if best_lower === nothing
    println("NO FEASIBLE LOWER-BOUND POINT FOUND ACROSS ANY OF THE 5 STARTS.")
else
    @printf("Best sequential LOWER-bound result: START_ID=%d, best_feasible_kappa=%.6f\n",
            best_lower.start_id, best_lower.bκ)
end
println("="^100)

# ---- Reconstruct + exact full-A evaluation, for whichever bound(s) have a feasible best result ----
function reconstruct_and_verify(label::String, best; δ::Real = 1.0, find_smallest::Bool = true)
    best === nothing && (println("\n[$label] SKIPPED: no feasible sequential result to reconstruct."); return nothing)
    println("\n" * "-"^100)
    @printf("[%s] Reconstructing full-A point from sequential START_ID=%d best-feasible theta\n", label, best.start_id)
    println("-"^100)
    rc = build_reconstruction_context(; δ = δ, find_smallest = find_smallest)
    x_free, diag = reconstruct_fullA_point(rc, best.θ; δ = δ)

    @printf("  Fresh cold seq_gravcol re-solve at theta_seq: R_mean=%.4e  gravity_ok=%s  divergence(p)=%.4e  delta_ok=%s\n",
            diag.seq_R_mean, diag.seq_gravity_ok, diag.seq_divergence_p, diag.seq_delta_ok)
    @printf("  Focal-column round-trip check (Acol vs AodPow-derived Aod_theta): max abs err = %.3e\n",
            diag.focal_roundtrip_max_abs_err)

    cache = oracle_cache_for(rc.ctx)
    result = evaluate_fullA(x_free, rc.ctx; cache = cache, use_cache = false, mode = :hard, warm = true, tag = label)

    @printf("\n  evaluate_fullA result:\n")
    @printf("    inner_status = %d   solved = %s\n", result.inner_status, result.inner_status in (0,-100,-101,-103))
    @printf("    gamma_focal_prime = %.12f   (theta_seq[3] = %.12f, match=%s)\n",
            result.gamma_focal_prime, best.θ[3], result.gamma_focal_prime == best.θ[3])
    κ_exact = 1 - result.gamma_focal_prime^(rc.σ/(rc.σ-1))
    @printf("    kappa (exact full-A oracle) = %.10f\n", κ_exact)
    @printf("    gravity_R_mean = %.6e   gravity_R_sum = %.6e   gravity_R_beta = %.6e\n",
            result.gravity_R_mean, result.gravity_R_sum, result.gravity_R_beta)
    @printf("    max_abs_moment_resid = %.6e  (over %d moments)\n", result.max_abs_moment_resid, length(result.moment_resid))
    @printf("    Delta_dual = %.8f   Delta_primal = %.8f   primal_dual_gap = %.3e   Delta - delta = %.6e  (delta=%.4g)\n",
            result.Delta_dual, result.Delta_primal, result.primal_dual_gap, result.Delta_minus_delta, δ)
    @printf("    mean_m_resid = %.3e   max_abs_moment_kkt_resid = %.3e\n", result.mean_m_resid, result.max_abs_moment_kkt_resid)
    @printf("    m_mean=%.6f m_min=%.6f m_max=%.6f  weight_norm_resid=%.3e\n", result.m_mean, result.m_min, result.m_max, result.weight_norm_resid)
    @printf("    winner_hash = %s\n", string(result.winner_hash))
    @printf("    elapsed: total=%.2fs inner=%.2fs post=%.2fs\n", result.elapsed.total, result.elapsed.inner, result.elapsed.post)

    exact_feasible = result.inner_status in (0,-100,-101,-103) &&
                      abs(result.gravity_R_mean) < 5e-4 &&
                      result.Delta_dual <= δ * (1 + 1e-6) + 1e-10 &&
                      result.max_abs_moment_resid < 1e-3
    @printf("\n  EXACT_FEASIBLE (full-A oracle, hard gates: solved status, |R_mean|<5e-4, Delta<=delta, max|moment_resid|<1e-3): %s\n", exact_feasible)

    return (label = label, x_free = x_free, diag = diag, result = result, κ_exact = κ_exact, exact_feasible = exact_feasible, start_id = best.start_id)
end

upper_recon = reconstruct_and_verify("UPPER", best_upper; δ = 1.0, find_smallest = true)
lower_recon = reconstruct_and_verify("LOWER", best_lower; δ = 1.0, find_smallest = false)

println("\n" * "="^100)
println("FINAL COMPARISON TABLE (D=4, W=8000, delta=1, seedFakeData=889, seedU=888, baseIndex=2, sigma=2.5)")
println("="^100)
@printf("  %-45s %14s\n", "candidate", "kappa")
@printf("  %-45s %14.10f\n", "fixed_A_benchmark (A held at A*)", FIXED_A_BENCHMARK_KAPPA)
@printf("  %-45s %14.10f  (gamma_focal_prime=%.10f)\n", "full-A incumbent (maxit=40, UNVERIFIED robust-local)", FULLA_INCUMBENT_KAPPA, FULLA_INCUMBENT_GP)
if upper_recon !== nothing
    @printf("  %-45s %14.10f  (START_ID=%d, exact_feasible=%s)\n",
            "sequential reconstructed (UPPER, best of 5)", upper_recon.κ_exact, upper_recon.start_id, upper_recon.exact_feasible)
else
    println("  sequential reconstructed (UPPER, best of 5)   NO FEASIBLE RESULT")
end
if lower_recon !== nothing
    @printf("  %-45s %14.10f  (START_ID=%d, exact_feasible=%s)\n",
            "sequential reconstructed (LOWER, best of 5)", lower_recon.κ_exact, lower_recon.start_id, lower_recon.exact_feasible)
end
println("="^100)

if upper_recon !== nothing && upper_recon.exact_feasible
    if upper_recon.κ_exact > FULLA_INCUMBENT_KAPPA
        println("\n>>> Sequential reconstructed point EXCEEDS the full-A incumbent. <<<")
    elseif upper_recon.κ_exact > FIXED_A_BENCHMARK_KAPPA
        println("\n>>> Sequential reconstructed point beats the fixed-A floor but FALLS SHORT of the full-A incumbent. <<<")
    else
        println("\n>>> Sequential reconstructed point falls short of even the fixed-A floor (should not happen for a converged run -- flag as a problem). <<<")
    end
end
println("\nPHASE5_COMPARISON_DONE")
