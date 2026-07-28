# Post-consolidation validation, Phase 1: exact replay of the 1.510118e14 anomaly through
# the NEW consolidated inner-solve API (`MelitzInnerSession`/`solve_melitz_delta!`,
# `docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`). Governing prompt:
# `melitz_post_consolidation_validation_2026-07-XX` task. Reuses the EXACT reconstruction
# recipe from `scripts/melitz_anomaly_phase1_reconstruct_2026-07-28.jl` (same seeds, same
# theta_pt/theta_new construction) -- only the inner-solve MECHANISM is swapped from the old
# `melitz_classified_inner_solve(obj, theta, ctx; delta_evaluation_cap=...)` (no longer
# defined) to `solve_melitz_delta!(session, theta, policy; ...)`.
#
# Budget: at most 3 inner solves via solve_melitz_delta!. Used: 2 (r0, r_new), both under
# CappedEvaluation(10). The capped-vs-uncapped separation check (governing prompt's
# "run one explicit FullValueEvaluation() only if needed") is answered WITHOUT a third
# solve: melitz_policy_lower_limit is a pure, already-statically-verified function of the
# policy value alone (confirmed by direct source read this session,
# src/melitz/inner_solve_policy.jl:139-140) -- printed here for the record, not re-derived
# by running a divergent uncapped pursuit, per the explicit instruction "do not allow the
# uncapped solve to run toward 1e14."

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")

println("="^100)
println("POST-CONSOLIDATION PHASE 1: exact 1.510118e14 anomaly replay via solve_melitz_delta!")
println("="^100)
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy_capped = CappedEvaluation(CAP)
policy_full = FullValueEvaluation()
println("melitz_policy_lower_limit(CappedEvaluation($CAP)) = ", melitz_policy_lower_limit(policy_capped))
println("melitz_policy_lower_limit(FullValueEvaluation())   = ", melitz_policy_lower_limit(policy_full))
println("melitz_policy_cap(CappedEvaluation($CAP))          = ", melitz_policy_cap(policy_capped))
println("melitz_policy_cap(FullValueEvaluation())            = ", melitz_policy_cap(policy_full))
@assert melitz_policy_lower_limit(policy_capped) == -CAP
@assert melitz_policy_lower_limit(policy_full) == -KNITRO.KN_INFINITY
println("Architectural separation confirmed at the policy level (no solve needed): capped " *
        "and full-value policies resolve to genuinely different, independently-computed " *
        "lower_limit values; a CappedEvaluation session can never be constructed with the " *
        "uncapped threshold (MelitzInnerSession's own assert, re-verified live below).")
flush(stdout)

# --- fixture: build_realD20_fixture now defaults to policy=CappedEvaluation(10.0) ---
d20 = build_realD20_fixture(; policy=policy_capped)
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D
println("\nobj type = ", typeof(obj))
println("obj.lower_limit (at fixture construction) = ", obj.lower_limit)
@assert obj.lower_limit == -CAP "fixture should be capped by the new default -- root cause closed"
flush(stdout)

session = MelitzInnerSession(obj, ctx, policy_capped)
println("MelitzInnerSession constructed; consistency assert passed (would have thrown otherwise).")
flush(stdout)

# --- reconstruct theta_pt for target=0.5, EXACTLY as the original anomaly script did ---
profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
target = 0.5
k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
g_pt = gs[k] + t * (gs[k+1] - gs[k])
theta_pt = copy(theta0); theta_pt[1] = g_pt
println("\nReconstructed g_pt = ", g_pt, "  (original session's own g_pt = -0.48546126592859407)")
println("Match: ", g_pt == -0.48546126592859407)
flush(stdout)

theta_start = copy(theta0); theta_start[1] = g_pt

println("\nBuilding reduced empirical SVD basis (deterministic, seeded, NOT an inner solve)...")
flush(stdout)
n_sub = min(80, n - 1)
sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)
Wsub = 4000
idxsub = 1:Wsub
Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
ei = zeros(n)
for (jj, kk) in enumerate(sub_coords)
    ei[kk] = 1.0
    dG = melitz_moment_directional_derivative(theta_start, ei, ctx, obj)
    Jsub[:, jj] .= vec(dG[idxsub, :])
    ei[kk] = 0.0
end
Usvd = svd(Jsub)
v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
svd_basis_near_null = v_near_null ./ norm(v_near_null)
println("Smallest singular values: ", Usvd.S[end-5:end])
flush(stdout)

theta_new = theta_pt .+ (-1.0) * 0.05 .* svd_basis_near_null

# ============================================================================
# INNER SOLVE 1/2 (of budget <=3): r0 = solve_melitz_delta!(session, theta_pt, policy_capped)
# ============================================================================
println("\n" * "-"^100)
println("INNER SOLVE 1/2: r0 = solve_melitz_delta!(session, theta_pt, CappedEvaluation($CAP); " *
        "origin_block_screen=true, warm_start_source=:previous)")
t0 = time()
r0 = solve_melitz_delta!(session, theta_pt, policy_capped; origin_block_screen=true,
                          warm_start_source=:previous)
t_r0 = time() - t0
println("  wall = ", t_r0, "s")
println("  result type = ", typeof(r0))
if r0 isa FiniteSolved
    println("  r0.Delta = ", r0.Delta, "   r0.nStatus = ", r0.nStatus, "   ||x|| = ", norm(r0.x))
end
println("  obj.threshold_crossed[] after call = ", obj.threshold_crossed[])
println("  original session's own Delta0 = 0.4832764950468894   match: ",
        r0 isa FiniteSolved && r0.Delta == 0.4832764950468894)
flush(stdout)

# ============================================================================
# INNER SOLVE 2/2: r_new = solve_melitz_delta!(session, theta_new, policy_capped) [the anomaly point]
# ============================================================================
println("\n" * "-"^100)
println("INNER SOLVE 2/2: r_new = solve_melitz_delta!(session, theta_new, CappedEvaluation($CAP); " *
        "origin_block_screen=true, warm_start_source=:previous)  [svd_near_null, sign=-1, THE ANOMALY POINT]")
t0 = time()
r_new = solve_melitz_delta!(session, theta_new, policy_capped; origin_block_screen=true,
                             warm_start_source=:previous)
t_r_new = time() - t0
println("  wall = ", t_r_new, "s")
println("  result type = ", typeof(r_new))
println("  obj.lower_limit AT THIS SOLVE = ", obj.lower_limit)
println("  obj.threshold_crossed[] after call = ", obj.threshold_crossed[])

FINITE_SOLVED_RETURNED = r_new isa FiniteSolved
println("\n>>> classification = ", typeof(r_new), "  <<<")
if r_new isa FiniteSolved
    println("  !!! UNEXPECTED: FiniteSolved returned. Delta = ", r_new.Delta, "  nStatus=", r_new.nStatus)
    println("  Delta <= cap($CAP)+tol ? ", r_new.Delta <= CAP + max(1e-6, 1e-6*CAP))
elseif r_new isa InfiniteDeltaCertified
    println("  InfiniteDeltaCertified: column=", r_new.column, " lo=", r_new.lo, " hi=", r_new.hi,
            " kind=", r_new.kind, "  -- EXPECTED per prior forensic session's own certificate")
elseif r_new isa AboveEvaluationCap
    println("  AboveEvaluationCap: certified_lower_bound=", r_new.certified_lower_bound,
            "  source=", r_new.source, "  -- acceptable fallback per governing prompt")
elseif r_new isa NumericalFailure
    println("  NumericalFailure: nStatus=", r_new.nStatus,
            "  -- acceptable conservative fallback per governing prompt")
end
flush(stdout)

println("\n" * "="^100)
println("PHASE 1 REGRESSION ASSERTION: r_new must NEVER be FiniteSolved")
@assert !FINITE_SOLVED_RETURNED "PHASE 1 FAILED: consolidated API returned FiniteSolved for the anomaly point!"
println("PASSED: r_new is $(typeof(r_new)), never FiniteSolved.")
println("="^100)

# --- write CSV record ---
open(joinpath(OUTDIR, "melitz_post_consolidation_phase1_anomaly_replay_2026-07-28.csv"), "w") do io
    println(io, "quantity,value")
    println(io, "cap,$CAP")
    println(io, "g_pt,$g_pt")
    println(io, "r0_result_type,$(typeof(r0))")
    println(io, "r0_Delta,$(r0 isa FiniteSolved ? r0.Delta : NaN)")
    println(io, "r_new_result_type,$(typeof(r_new))")
    println(io, "r_new_FiniteSolved,$(FINITE_SOLVED_RETURNED)")
    if r_new isa InfiniteDeltaCertified
        println(io, "r_new_column,$(r_new.column)")
        println(io, "r_new_kind,$(r_new.kind)")
    elseif r_new isa AboveEvaluationCap
        println(io, "r_new_certified_lower_bound,$(r_new.certified_lower_bound)")
        println(io, "r_new_source,$(r_new.source)")
    elseif r_new isa NumericalFailure
        println(io, "r_new_nStatus,$(r_new.nStatus)")
    end
    println(io, "wall_r0_s,$t_r0")
    println(io, "wall_r_new_s,$t_r_new")
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase1_anomaly_replay_2026-07-28.csv")
println("\nDONE Phase 1 (post-consolidation).")
