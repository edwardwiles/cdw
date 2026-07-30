# Phase 3 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30):
# fixed-q hot-path profile of the REPAIRED v2 middle-loop driver at three representative D20
# welfare points, using the existing production profiling instrumentation
# (src/melitz/profiling.jl, MELITZ_PROFILE[]) plus two NEW categories added this session
# (:middle_theta_reconstruction, :middle_cache_lookup -- fixed_q_a_middle_loop.jl,
# melitz_middle_objective_and_gradient_cached!) to separate the ADDENDUM-D cache-overhead cost
# from the genuine per-A-point inner-solve cost, which the prior session's Section E profile
# (docs/key_results/melitz_addendumE_hotpath_profile_2026-07-30.csv) did not yet exist to
# distinguish (that profile predates the v2 cache wrapper's own instrumentation).
#
# Three points (governing prompt's own choice): (1) the profiled anchor (GT=6.2906%); (2) an
# interior feasible continuation point (GT=6.5281%, idx=3 on the stored upper path); (3) the
# near-boundary point (GT=7.0969%, Delta*=0.499, idx=6 -- the decisive extreme point).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Serialization
LinearAlgebra.BLAS.set_num_threads(1)
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT_D20 = joinpath(REPO2, "melitz_middle_loop_opt_2026-07-30.opt")
const D20_MIDDLE_BOX = 0.1
const MAX_MIDDLE_EVALS = 150   # >=20 unique A evaluations required; 150 gives headroom
const CAP_HANDLING = :barrier
const CAP_BARRIER_MULTIPLE = 5.0

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
obj_d20, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx_d20 = obj_d20.γ
D20 = ctx_d20.D; nA20 = D20^2 - 1
session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

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
theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]
g0 = theta_plain0_d20[1]
function q_full_at_g(g_target::Real)
    th = copy(theta_plain0_d20)
    th[1] = g_target
    _, _, _, _, q = expand_free_theta_logcutoff(th, ctx_d20)
    return q
end

# Load the stored upper-path row states (idx=3 interior, idx=6 near-boundary) precomputed by
# melitz_phase2_precompute_states_2026-07-30.jl (SAME inputs Phase 2's replay used -- reuse
# rather than re-derive).
struct RowInputState
    idx::Int; g_target::Float64; prev_A_free::Vector{Float64}; prev_q::Matrix{Float64}
    prev_p_star::Vector{Float64}; stored_GT::Float64; stored_Delta::Float64
    stored_classification::Symbol; stored_accepted::Bool; stored_best_start::Symbol
end
loaded = deserialize(joinpath(@__DIR__, "melitz_phase2_row_states_2026-07-30.jls"))
row_states = Dict(rs.idx => rs for rs in loaded.row_states)

points = [
    (label="anchor_g0", g=g0, prev_A_free=A_free0_d20, prev_q=q_full_at_g(g0)),
    (label="interior_idx3_GT6.53pct", g=row_states[3].g_target, prev_A_free=row_states[3].prev_A_free, prev_q=row_states[3].prev_q),
    (label="near_boundary_idx6_GT7.10pct", g=row_states[6].g_target, prev_A_free=row_states[6].prev_A_free, prev_q=row_states[6].prev_q),
]

all_rows = NamedTuple[]
for pt in points
    println("\n", "="^100); println("PROFILING POINT: ", pt.label, "  g=", pt.g); flush(stdout)
    gpj_target = exp(pt.g)
    q_target = q_full_at_g(pt.g)
    theta_for_constraints = melitz_fixed_q_state_theta(pt.prev_A_free, q_target, gpj_target, ctx_d20)
    sys_t = melitz_fixed_q_middle_constraint_system(theta_for_constraints, ctx_d20, obj_d20)
    # Deterministic (non-random) fixed sinusoidal perturbation of the continuation start, purely
    # to force enough KNITRO exploration steps for a statistically meaningful hot-path sample
    # (a warm, already-near-optimal continuation start can converge in as few as ~4 evaluations,
    # too thin for percentile timing breakdowns) -- matches this repo's own established
    # "Start D: deterministic log-H perturbation" convention (doc
    # melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md Phase 4), not a random multistart.
    nA_here = length(pt.prev_A_free)
    det_perturb = 0.03 .* sin.((1:nA_here) .* 0.7)
    A_cont = melitz_project_start_to_middle_constraints(copy(pt.prev_A_free) .+ det_perturb, sys_t, ctx_d20)

    # Immutability fingerprint BEFORE the middle solve.
    rank_fp_before = [copy(melitz_origin_intervals(o, theta_for_constraints, ctx_d20, obj_d20).rank) for o in 1:D20]
    sense_fp_before = copy(sys_t.sense)

    melitz_profile_reset!()
    MELITZ_PROFILE[] = true
    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    t0 = time()
    res = solve_melitz_fixed_q_A_profile_v2(session_d20, q_target, gpj_target, A_cont, ctx_d20;
        coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
        cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    wall_total = time() - t0
    MELITZ_PROFILE[] = false

    @printf("  classification=%s Delta=%.6g unique_A=%d unique_solves=%d cache_hits=%d wall=%.2fs\n",
        nameof(typeof(res.r_incumbent)), res.Delta_incumbent, res.unique_A_points, res.unique_inner_solves,
        res.cache_hits, wall_total)
    if res.unique_A_points < 20
        @warn "governing prompt asks for >=20 unique A evaluations per point; this point converged in fewer" point=pt.label unique_A_points=res.unique_A_points
    end

    # Immutability fingerprint AFTER the middle solve, at the incumbent's own theta.
    rank_fp_after = [copy(melitz_origin_intervals(o, res.theta_free_incumbent, ctx_d20, obj_d20).rank) for o in 1:D20]
    sys_after = melitz_fixed_q_middle_constraint_system(res.theta_free_incumbent, ctx_d20, obj_d20)
    ranks_match = all(rank_fp_before[o] == rank_fp_after[o] for o in 1:D20)
    sense_match = sense_fp_before == sys_after.sense

    println("\n  -- Immutability fingerprints --")
    println("  cutoff/rank structure bit-identical anchor->incumbent: ", ranks_match)
    println("  same-bin/ordering sense vector bit-identical:          ", sense_match)
    flush(stdout)

    println("\n  -- Profile report --")
    rows = melitz_profile_report(stdout; trajectory_total_s=wall_total)
    flush(stdout)

    for r in rows
        push!(all_rows, (point=pt.label, category=String(r.category), count=r.count, total_s=r.total_s,
            mean_ms=r.mean_ms, median_ms=r.median_ms, p90_ms=r.p90_ms, max_ms=r.max_ms,
            wall_total_s=wall_total, unique_A_points=res.unique_A_points, unique_inner_solves=res.unique_inner_solves,
            cache_hits=res.cache_hits, Delta_incumbent=res.Delta_incumbent, ranks_match=ranks_match, sense_match=sense_match))
    end
end

outpath = joinpath(OUTDIR, "melitz_phase3_hotpath_profile_2026-07-30.csv")
open(outpath, "w") do io
    println(io, "point,category,count,total_s,mean_ms,median_ms,p90_ms,max_ms,wall_total_s,unique_A_points,unique_inner_solves,cache_hits,Delta_incumbent,ranks_match,sense_match")
    for r in all_rows
        println(io, join([r.point, r.category, r.count, r.total_s, r.mean_ms, r.median_ms, r.p90_ms, r.max_ms,
            r.wall_total_s, r.unique_A_points, r.unique_inner_solves, r.cache_hits, r.Delta_incumbent,
            r.ranks_match, r.sense_match], ","))
    end
end
println("\nWrote ", outpath)
println("\nPHASE 3 HOT-PATH PROFILE COMPLETE")
