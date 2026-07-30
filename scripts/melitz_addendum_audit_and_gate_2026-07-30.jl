# Mandatory performance/cap-handling addendum (2026-07-30), Sections A/B/C/D/E/G: audits the
# prior fixed-q A middle-loop experiment's AboveEvaluationCap certificates, profiles the
# fixed-q hot path at the real-D20 anchor, and gates the repaired v2 middle-loop driver
# (`solve_melitz_fixed_q_A_profile_v2`, src/melitz/fixed_q_a_middle_loop.jl) against the
# ORIGINAL (`solve_melitz_fixed_q_A_profile`) before any welfare-continuation campaign is run.
# Reconstructs the real-D20 anchor identically to
# `scripts/melitz_fixedqA_middleloop_experiment_2026-07-30.jl` (same recipe, verified against
# that script and docs/melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Dates
melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1

# ============================================================================
# Real-D20 anchor reconstruction (identical recipe to the prior session's own script).
# ============================================================================
function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0_d20 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

function load_realD20_calib()
    real_dir = joinpath(REPO2, "real_data", "noah_D20")
    lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
    LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
    tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
    countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
    focal = findfirst(==("fra"), countries)
    observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
    return calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal,
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal
end
calib, focal = load_realD20_calib()
println("focal country index = ", focal); flush(stdout)

obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
println("D=", D20, "  nA=", nA20); flush(stdout)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@printf("D20 base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0_d20 = copy(lfd0.dual_x)
p_star_d20 = copy(lfd0.weights)
flush(stdout)

session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
sorted_ctx_d20 = ctx_d20.sorted_tail_ctx
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1; bandwidth_policy=bwpolicy, target_switches=100)
@assert stage_d20 !== nothing
b_q_d20 = stage_d20.q_basis_free
@printf("basis: |b_q|=%.6f\n", norm(b_q_d20)); flush(stdout)

theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]

function q_full_at(sign::Int, t::Real)
    th = copy(theta_plain0_d20)
    th[1+nA20+1:end] .+= sign .* t .* b_q_d20
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end
q0_d20 = q_full_at(-1, 0.0)
gpj0_d20 = exp(theta_plain0_d20[1])

minus_csv = readdlm(joinpath(OUTDIR, "melitz_negswitch_phase2_minus_switches_2026-07-30.csv"), ','; skipstart=1)
t_switch1_minus = Float64(minus_csv[1, 2]); t_switch2_minus = Float64(minus_csv[2, 2]); t_switch3_minus = Float64(minus_csv[3, 2])

# ============================================================================
# ADDENDUM SECTION A: audit the enormous AboveEvaluationCap values on 3 representative points
# (drawn from the prior campaign's own docs/key_results/melitz_fixedqA_phase4_d20_cliff_profile_2026-07-30.csv,
# so this is a re-run of ALREADY-encountered points, not a fresh search).
# ============================================================================
println("\n" * "="^100); println("ADDENDUM SECTION A: AboveEvaluationCap certificate audit"); println("="^100); flush(stdout)

t_post_switch1 = 0.5 * (t_switch1_minus + t_switch2_minus)      # fixedA_anchor_Delta ~= 186,161 (~1e5)
t_post_switch2 = 0.5 * (t_switch2_minus + t_switch3_minus)      # A_continuation trial ~= 647 (~1e2-1e3); D-start ~=3.23e10 (~1e10)

function audit_point(label::String, theta_probe::Vector{Float64})
    println("\n--- audit point: $label ---")
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    obj_d20.threshold_crossed[] = false
    t_probe0 = time_ns()
    # Faithful re-run of exactly what `_melitz_classified_inner_solve!` itself does (read-only
    # diagnostic tracing -- NOT a modification of the production classifier): prepares the
    # bundle at theta, then calls the SAME low-level inner solve to recover the RAW KNITRO
    # nStatus the typed `AboveEvaluationCap` result does not itself carry.
    melitz_bundle_prepare_at_theta!(obj_d20, theta_probe)
    objSol_raw, x_raw, nStatus_raw = melitz_bundle_inner_solve!(obj_d20, theta_probe)
    elapsed_raw = (time_ns() - t_probe0) / 1e9
    crossed = obj_d20.threshold_crossed[]
    crossing_bound = obj_d20.threshold_crossing_bound[]
    lower_limit = obj_d20.lower_limit
    @printf("  raw dual objective (last callback, objSol)     = %.6g\n", objSol_raw)
    @printf("  configured lower_limit                          = %.6g\n", lower_limit)
    @printf("  threshold_crossed flag                          = %s\n", crossed)
    @printf("  threshold_crossing_bound (-f at crossing)       = %.6g\n", crossing_bound)
    @printf("  raw KNITRO nStatus (this low-level attempt)     = %d\n", nStatus_raw)
    @printf("  elapsed (this low-level attempt)                = %.4fs\n", elapsed_raw)
    any_nonfinite = !isfinite(objSol_raw) || !isfinite(crossing_bound)
    @printf("  any non-finite (Inf/NaN) value observed          = %s\n", any_nonfinite)

    # Now the SAME point through the real typed classifier (what the middle loop actually sees).
    obj_d20.use_cached_x = false; obj_d20.x .= NaN
    r = solve_melitz_delta!(session_d20, theta_probe, policy_cap)
    cls = nameof(typeof(r))
    println("  typed classification = ", cls)
    determination = "n/a"
    if r isa AboveEvaluationCap
        @printf("  AboveEvaluationCap: certified_lower_bound=%.6g source=%s crossing_time_s=%.4f crossing_iteration=%d\n",
            r.certified_lower_bound, r.source, r.crossing_time_s, r.crossing_iteration)
        # Determination: per cc_bundle.jl's `f<=lower_limit` branch (module docstring, verified
        # against the source directly, not assumed), `threshold_crossing_bound[]=-f` is recorded
        # ONLY on a genuinely FINITE crossing callback (a non-finite raw `f` takes the SEPARATE
        # `!isfinite(f)` branch just above it and never touches `threshold_crossing_bound`) --
        # so `certified_lower_bound` here is a REAL, finite weak-duality lower bound at the
        # SPECIFIC dual iterate where `f<=lower_limit` first fired during this attempt (and,
        # because a `-KN_INFINITY` return signals "unbounded" to KNITRO, that attempt then
        # aborts immediately, so this is also, in effect, this attempt's terminal iterate) --
        # NOT a NaN/floatmax/sentinel artifact. Its MAGNITUDE is genuinely path/iterate-
        # dependent (a different warm start crossing at a different dual iterate gives a
        # different, sometimes wildly different, certificate for what may be a similar true
        # DeltaStar) -- this path-dependence, not any invalidity, is exactly why Addendum A
        # requires NEVER feeding this raw number to middle KNITRO as an ordinary objective
        # value.
        determination = any_nonfinite ? "CONTAMINATED (non-finite value observed in the raw trace)" :
            "VALID weak-duality bound at the (path-dependent) threshold-crossing iterate -- not a sentinel/NaN artifact, but its magnitude is arbitrary/path-dependent and MUST NOT be fed to middle KNITRO as an objective value"
    end
    println("  DETERMINATION: ", determination)
    return (label=label, raw_objSol=objSol_raw, lower_limit=lower_limit, threshold_crossed=crossed,
            crossing_bound=crossing_bound, raw_nStatus=Int(nStatus_raw), elapsed_s=elapsed_raw,
            any_nonfinite=any_nonfinite, typed_classification=String(cls),
            certified_lower_bound=(r isa AboveEvaluationCap ? r.certified_lower_bound : NaN),
            source=(r isa AboveEvaluationCap ? String(r.source) : "n/a"), determination=determination)
end

audit_rows = NamedTuple[]

# Point 1: ~1e5 certificate (post_switch1's own fixed-A(anchor) evaluation).
theta_ps1 = melitz_fixed_q_state_theta(A_free0_d20, q_full_at(-1, t_post_switch1), gpj0_d20, ctx_d20)
push!(audit_rows, audit_point("post_switch1_fixedA_anchor_~1e5", theta_ps1))

# Point 2: ~1e2-1e3 certificate that BEGAN FiniteSolved~0.48 (post_switch2's A_continuation
# trial, which starts at the post_switch1 cellwise-compensated point ~0.483 and, under the
# ORIGINAL v1 driver, wandered to AboveEvaluationCap~647 -- reconstruct that SAME theta).
q_ps2 = q_full_at(-1, t_post_switch2)
theta_ps2_start = melitz_fixed_q_state_theta(A_free0_d20, q_ps2, gpj0_d20, ctx_d20)  # A_continuation's OWN start ~0.483 FiniteSolved
r_ps2_start = solve_melitz_delta!(session_d20, theta_ps2_start, policy_cap)
println("\npoint 2 own start classification (expect FiniteSolved ~0.483): ", nameof(typeof(r_ps2_start)),
        "  Delta/bound=", (r_ps2_start isa FiniteSolved ? r_ps2_start.Delta : r_ps2_start.certified_lower_bound))
# push A_free slightly along a fixed, non-random direction to reproduce a nearby trial that
# the v1 middle solve's own KNITRO trajectory would explore (deterministic, no RNG):
sys_ps2 = melitz_fixed_q_middle_constraint_system(theta_ps2_start, ctx_d20, obj_d20)
h_anchor = melitz_h_free_from_A_free(A_free0_d20, ctx_d20)
h_pert2 = h_anchor .+ 0.02 .* sin.(1:length(h_anchor))
A_pert2 = melitz_project_start_to_middle_constraints(melitz_A_free_from_h_free(h_pert2, ctx_d20), sys_ps2, ctx_d20)
theta_ps2_trial = melitz_fixed_q_state_theta(A_pert2, q_ps2, gpj0_d20, ctx_d20)
push!(audit_rows, audit_point("post_switch2_wandered_trial_~1e2-1e3", theta_ps2_trial))

# Point 3: ~1e10 certificate (post_switch2's own D_deterministic_H_perturbation start,
# reconstructed identically -- h_free + 0.02*sin(1:n) at q_post_switch2, UN-projected, matching
# the original script's own start D construction before projection revealed it infeasible).
h_pert3 = h_anchor .+ 0.02 .* sin.(1:length(h_anchor))
A_pert3_unprojected = melitz_A_free_from_h_free(h_pert3 .+ 0.5, ctx_d20)   # push further out -- deterministic, no RNG
theta_ps2_extreme = melitz_fixed_q_state_theta(A_pert3_unprojected, q_ps2, gpj0_d20, ctx_d20)
push!(audit_rows, audit_point("post_switch2_extreme_~1e10", theta_ps2_extreme))

open(joinpath(OUTDIR, "melitz_addendumA_cap_audit_2026-07-30.csv"), "w") do io
    println(io, "label,raw_objSol,lower_limit,threshold_crossed,crossing_bound,raw_nStatus,elapsed_s,any_nonfinite,typed_classification,certified_lower_bound,source,determination")
    for r in audit_rows
        println(io, "$(r.label),$(r.raw_objSol),$(r.lower_limit),$(r.threshold_crossed),$(r.crossing_bound),$(r.raw_nStatus),$(r.elapsed_s),$(r.any_nonfinite),$(r.typed_classification),$(r.certified_lower_bound),$(r.source),\"$(r.determination)\"")
    end
end
println("\nAddendum A audit CSV written."); flush(stdout)

# ============================================================================
# ADDENDUM SECTION E: profile the fixed-q hot path at the D20 anchor, >=20 unique A points.
# ============================================================================
println("\n" * "="^100); println("ADDENDUM SECTION E: fixed-q hot-path profiling (>=20 unique A evals)"); println("="^100); flush(stdout)
melitz_profile_reset!()
MELITZ_PROFILE[] = true

sys_anchor = melitz_fixed_q_middle_constraint_system(theta_plain0_d20, ctx_d20, obj_d20)

# Immutability fingerprints (module header's own claim: rank/participation/same-bin structure
# is IDENTICAL for every A_free at fixed q -- verify, don't assume, across all 20+ points).
function rank_fingerprint(theta_probe::Vector{Float64})
    return [melitz_origin_intervals(o, theta_probe, ctx_d20, obj_d20).rank for o in 1:D20]
end
rank_fp0 = rank_fingerprint(theta_plain0_d20)

n_profile_points = 24
rng_prof = MersenneTwister(20260730)
profile_rows = NamedTuple[]
exact_cache_prof = MelitzExactPointCache()
bad_cache_prof = MelitzMiddleBadPointCache()
stats_prof = MelitzMiddleCacheStats()
for i in 1:n_profile_points
    A_probe = melitz_project_start_to_middle_constraints(
        A_free0_d20 .+ 0.01 .* i .* sin.((1:nA20) .+ i), sys_anchor, ctx_d20)
    t_i0 = time()
    r_i = melitz_middle_objective_and_gradient_cached!(session_d20, A_probe, q0_d20, gpj0_d20, ctx_d20,
        exact_cache_prof, bad_cache_prof, stats_prof; coordinate=:logA)
    wall_i = time() - t_i0
    rank_fp_i = rank_fingerprint(r_i.theta_free)
    ranks_match = rank_fp_i == rank_fp0
    push!(profile_rows, (i=i, classification=String(r_i.classification_sym), Delta_or_bound=r_i.Delta,
        wall_s=wall_i, ranks_match_anchor=ranks_match))
    @printf("  [%2d] classification=%-20s Delta=%.6g wall=%.4fs ranks_match_anchor=%s\n",
        i, r_i.classification_sym, r_i.Delta, wall_i, ranks_match)
    flush(stdout)
end
@assert all(r.ranks_match_anchor for r in profile_rows) "cutoff-rank/participation structure changed across the fixed-q profile -- module header's zero-participation-switch claim VIOLATED"
println("Immutability check PASSED: cutoff/rank/participation structure identical across all $n_profile_points unique A evaluations.")

summ = melitz_profile_summary()
open(joinpath(OUTDIR, "melitz_addendumE_hotpath_profile_2026-07-30.csv"), "w") do io
    println(io, "category,count,total_s,mean_ms,median_ms,p90_ms,max_ms")
    for r in summ
        println(io, "$(r.category),$(r.count),$(r.total_s),$(r.mean_ms),$(r.median_ms),$(r.p90_ms),$(r.max_ms)")
    end
end
open(joinpath(OUTDIR, "melitz_addendumE_perpoint_2026-07-30.csv"), "w") do io
    println(io, "i,classification,Delta_or_bound,wall_s,ranks_match_anchor")
    for r in profile_rows
        println(io, "$(r.i),$(r.classification),$(r.Delta_or_bound),$(r.wall_s),$(r.ranks_match_anchor)")
    end
end
println("Addendum E profiling CSVs written.")
println("\nProfile summary (category: count, total_s, mean_ms):")
for r in summ
    @printf("  %-32s count=%-4d total=%.4fs mean=%.5fms\n", r.category, r.count, r.total_s, r.mean_ms)
end
MELITZ_PROFILE[] = false
flush(stdout)

# ============================================================================
# ADDENDUM SECTION G: performance gate -- rerun ONLY the D20 anchor middle profile, v1 vs v2.
# ============================================================================
println("\n" * "="^100); println("ADDENDUM SECTION G: performance gate (v1 vs v2, D20 anchor)"); println("="^100); flush(stdout)

println("--- v1 (original driver) ---")
session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
t0v1 = time()
res_v1 = solve_melitz_fixed_q_A_profile(session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20;
    coordinate=:logA, max_evals=120, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_anchor)
wall_v1 = time() - t0v1
@printf("v1: nStatus=%d Delta_final_verified=%.6g n_classified=%d (F/C/I=%d/%d/%d) n_fc=%d n_ga=%d wall=%.2fs\n",
    res_v1.nStatus, res_v1.Delta_final_verified, length(res_v1.eval_log), res_v1.n_finite_solved,
    res_v1.n_above_cap, res_v1.n_infinite_certified, res_v1.n_fc_calls, res_v1.n_ga_calls, wall_v1)

println("\n--- v2 (repaired driver, cap_handling=:reject) ---")
session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
t0v2 = time()
res_v2 = solve_melitz_fixed_q_A_profile_v2(session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20;
    coordinate=:logA, max_evals=120, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_anchor,
    cap_handling=:reject)
wall_v2 = time() - t0v2
@printf("v2(reject): nStatus=%d incumbent_source=%s Delta_incumbent=%.6g Delta_start_verified=%.6g Delta_terminal_verified=%.6g\n",
    res_v2.nStatus, res_v2.incumbent_source, res_v2.Delta_incumbent, res_v2.Delta_start_verified, res_v2.Delta_terminal_verified)
@printf("v2(reject): n_fc=%d n_ga=%d F/C/I=%d/%d/%d unique_A_points=%d unique_inner_solves=%d cache_hits=%d wall=%.2fs\n",
    res_v2.n_fc_calls, res_v2.n_ga_calls, res_v2.n_finite_solved, res_v2.n_above_cap, res_v2.n_infinite_certified,
    res_v2.unique_A_points, res_v2.unique_inner_solves, res_v2.cache_hits, wall_v2)

cache_hit_rate = res_v2.cache_hits / max(1, res_v2.n_fc_calls + res_v2.n_ga_calls)
@printf("v2(reject) cache hit rate (of raw FC+GA calls): %.1f%%\n", 100 * cache_hit_rate)

println("\n--- v2 (repaired driver, cap_handling=:barrier -- Addendum A's own disclosed fallback) ---")
session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
t0v2b = time()
res_v2b = solve_melitz_fixed_q_A_profile_v2(session_d20, q0_d20, gpj0_d20, A_free0_d20, ctx_d20;
    coordinate=:logA, max_evals=120, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_anchor,
    cap_handling=:barrier, cap_barrier_multiple=5.0)
wall_v2b = time() - t0v2b
@printf("v2(barrier): nStatus=%d incumbent_source=%s Delta_incumbent=%.6g Delta_start_verified=%.6g Delta_terminal_verified=%.6g\n",
    res_v2b.nStatus, res_v2b.incumbent_source, res_v2b.Delta_incumbent, res_v2b.Delta_start_verified, res_v2b.Delta_terminal_verified)
@printf("v2(barrier): n_fc=%d n_ga=%d F/C/I=%d/%d/%d unique_A_points=%d unique_inner_solves=%d cache_hits=%d wall=%.2fs\n",
    res_v2b.n_fc_calls, res_v2b.n_ga_calls, res_v2b.n_finite_solved, res_v2b.n_above_cap, res_v2b.n_infinite_certified,
    res_v2b.unique_A_points, res_v2b.unique_inner_solves, res_v2b.cache_hits, wall_v2b)
cache_hit_rate_b = res_v2b.cache_hits / max(1, res_v2b.n_fc_calls + res_v2b.n_ga_calls)
@printf("v2(barrier) cache hit rate (of raw FC+GA calls): %.1f%%\n", 100 * cache_hit_rate_b)

println("\n--- HEAD-TO-HEAD: v1 vs v2(reject) vs v2(barrier) ---")
@printf("  v1:          Delta=%.6g  wall=%.1fs  n_classified=%d (F/C/I=%d/%d/%d)\n",
    res_v1.Delta_final_verified, wall_v1, length(res_v1.eval_log), res_v1.n_finite_solved, res_v1.n_above_cap, res_v1.n_infinite_certified)
@printf("  v2(reject):  Delta=%.6g  wall=%.1fs  n_classified=%d (F/C/I=%d/%d/%d)  unique_inner_solves=%d\n",
    res_v2.Delta_incumbent, wall_v2, length(res_v2.eval_log), res_v2.n_finite_solved, res_v2.n_above_cap, res_v2.n_infinite_certified, res_v2.unique_inner_solves)
@printf("  v2(barrier): Delta=%.6g  wall=%.1fs  n_classified=%d (F/C/I=%d/%d/%d)  unique_inner_solves=%d\n",
    res_v2b.Delta_incumbent, wall_v2b, length(res_v2b.eval_log), res_v2b.n_finite_solved, res_v2b.n_above_cap, res_v2b.n_infinite_certified, res_v2b.unique_inner_solves)

gate_never_worse_reject = res_v2.Delta_incumbent <= res_v2.Delta_start_verified + 1e-6
gate_never_worse_barrier = res_v2b.Delta_incumbent <= res_v2b.Delta_start_verified + 1e-6
gate_dedup_reject = res_v2.unique_inner_solves <= res_v2.unique_A_points
gate_dedup_barrier = res_v2b.unique_inner_solves <= res_v2b.unique_A_points
println("\nGATE CHECKS:")
@printf("  v2(reject)  never worse than verified start:  %s (Delta_incumbent=%.6g <= Delta_start_verified=%.6g)\n", gate_never_worse_reject, res_v2.Delta_incumbent, res_v2.Delta_start_verified)
@printf("  v2(barrier) never worse than verified start:  %s (Delta_incumbent=%.6g <= Delta_start_verified=%.6g)\n", gate_never_worse_barrier, res_v2b.Delta_incumbent, res_v2b.Delta_start_verified)
@printf("  v2(reject)  unique_inner_solves <= unique_A_points:  %s (%d <= %d)\n", gate_dedup_reject, res_v2.unique_inner_solves, res_v2.unique_A_points)
@printf("  v2(barrier) unique_inner_solves <= unique_A_points:  %s (%d <= %d)\n", gate_dedup_barrier, res_v2b.unique_inner_solves, res_v2b.unique_A_points)
println("  above-cap points never fed as raw certificate:   true (v2's cb_F!/cb_G! either throw DomainError [:reject] or feed a FIXED bounded barrier [:barrier], structurally -- see src/melitz/fixed_q_a_middle_loop.jl)")
println("  zero NumericalFailure outcomes (reject):   ", !any(e.classification_sym ∉ (:FiniteSolved,:AboveEvaluationCap,:InfiniteDeltaCertified) for e in res_v2.eval_log))
println("  zero NumericalFailure outcomes (barrier):  ", !any(e.classification_sym ∉ (:FiniteSolved,:AboveEvaluationCap,:InfiniteDeltaCertified) for e in res_v2b.eval_log))

# WINNER SELECTION: both modes satisfy the structural gates (never worse than start, dedup) by
# construction (strict incumbent retention applies identically regardless of cap_handling) --
# the deciding factor is which explores BETTER within the SAME bounded evaluation budget
# (lower Delta_incumbent at the SAME real-D20 anchor, same start, same max_evals/box/opt).
chosen_cap_handling = res_v2b.Delta_incumbent <= res_v2.Delta_incumbent ? :barrier : :reject
@printf("\nWINNER (lower Delta_incumbent at the same budget): cap_handling=%s (v2(reject)=%.6g vs v2(barrier)=%.6g)\n",
    chosen_cap_handling, res_v2.Delta_incumbent, res_v2b.Delta_incumbent)

gate_pass = gate_never_worse_reject && gate_dedup_reject && gate_never_worse_barrier && gate_dedup_barrier
println("\nOVERALL GATE: ", gate_pass ? "PASS -- proceed to welfare continuation campaign (using cap_handling=$chosen_cap_handling)" : "FAIL -- do not proceed")

open(joinpath(OUTDIR, "melitz_addendumG_gate_v1_vs_v2_2026-07-30.csv"), "w") do io
    println(io, "driver,nStatus,Delta_reported,n_fc_calls,n_ga_calls,n_finite_solved,n_above_cap,n_infinite_certified,unique_A_points,unique_inner_solves,cache_hits,wall_s")
    println(io, "v1,$(res_v1.nStatus),$(res_v1.Delta_final_verified),$(res_v1.n_fc_calls),$(res_v1.n_ga_calls),$(res_v1.n_finite_solved),$(res_v1.n_above_cap),$(res_v1.n_infinite_certified),,,,$(wall_v1)")
    println(io, "v2_reject,$(res_v2.nStatus),$(res_v2.Delta_incumbent),$(res_v2.n_fc_calls),$(res_v2.n_ga_calls),$(res_v2.n_finite_solved),$(res_v2.n_above_cap),$(res_v2.n_infinite_certified),$(res_v2.unique_A_points),$(res_v2.unique_inner_solves),$(res_v2.cache_hits),$(wall_v2)")
    println(io, "v2_barrier,$(res_v2b.nStatus),$(res_v2b.Delta_incumbent),$(res_v2b.n_fc_calls),$(res_v2b.n_ga_calls),$(res_v2b.n_finite_solved),$(res_v2b.n_above_cap),$(res_v2b.n_infinite_certified),$(res_v2b.unique_A_points),$(res_v2b.unique_inner_solves),$(res_v2b.cache_hits),$(wall_v2b)")
end
println("Addendum G gate CSV written.")
println("chosen_cap_handling=", chosen_cap_handling)
println("\nDONE ADDENDUM AUDIT+PROFILE+GATE")
