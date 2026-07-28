# Post-consolidation validation, Phase 4: replay archived real-D20 outer trial points.
#
# DISCLOSED SUBSTITUTION (read before interpreting results): the governing prompt asks to
# "use saved complete theta vectors from the recent nested searches" (gamma-only,
# gamma+technology, gamma+participation, full joint). Checked directly: NONE of those nested
# block-search CSVs (`docs/key_results/melitz_phase9_realD20_nested_block_search_2026-07-28.csv`
# and siblings) persist a full theta vector -- only scalar summaries (dg, dlogA-norm,
# dlogf-norm, DeltaStar, nStatus). No serialized/JLD2 theta array exists on disk for any of
# them either (checked: no `Serialization`/`JLD2` usage in any 2026-07-28 phase script). A
# literal "load and replay" is therefore impossible without either (a) re-running a full outer
# KNITRO search (explicitly out of scope: "do not rerun an outer optimizer"), or (b) picking a
# different, genuinely-persisted source of individual archived real-D20 points.
#
# This script uses (b): `docs/key_results/melitz_phase10_realD20_coordinate_basis_comparison_2026-07-28.csv`
# -- 20 individually-classified real-D20 points (2 base points + 3 af_random + 2 svd directions,
# x2 signs, x2 targets) from the SAME 2026-07-28 gradient-redundancy session, fully and exactly
# reconstructible (seeded RNG, deterministic construction, identical to Phase 1's own anomaly
# reconstruction recipe) -- and replays 8 of them (representative: both base points, the FiniteSolved
# svd_steepest direction at both targets and both signs, the NumericalFailure af_random_1 and
# svd_near_null directions) via the consolidated API. This covers the "finite within-budget"
# and "prior NumericalFailure" categories directly. Neither a persisted "prior AboveEvaluationCap"
# nor a persisted "prior InfiniteDeltaCertified" real-D20 OUTER-trial point exists in this
# session's saved artifacts (disclosed gap) -- Phase 1's own anomaly point (InfiniteDeltaCertified
# under origin_block_screen) and Phase 2's cap-50 rows (AboveEvaluationCap, different cap) are the
# closest available substitutes and are cross-referenced in the final report instead of re-derived
# here a second time.
#
# ADDITIONALLY (structural check of the higher-level `solve_melitz_finite_delta_bound` driver,
# not just the standalone classifier): re-runs ONE of the four Phase 9 nested-block-search
# configurations (`gamma_only`, the cheapest at ~45s) end-to-end through the consolidated API,
# from the identical starting point/box/options, and compares nStatus/dg/DeltaStar to the
# historically-reported row (nStatus=0, dg=-0.010048147673314078, DeltaStar=0.8603593934221736).

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)

d20 = build_realD20_fixture(; policy=policy)
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
function g_for_target(target)
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    return gs[k] + t * (gs[k+1] - gs[k])
end

# ============================================================================
# Part A: 8 representative coordinate-basis-comparison points (Phase 10 of the gradient-
# redundancy session), exactly reconstructed.
# ============================================================================
println("\n" * "="^100); println("PART A: 8 representative archived real-D20 points (Phase 10 coordinate-basis table)"); println("="^100)

rows_a = NamedTuple[]
n_solves_a = Ref(0)

for target in (0.5, 0.8)
    g_pt = g_for_target(target)
    theta_pt = copy(theta0); theta_pt[1] = g_pt
    session = MelitzInnerSession(obj, ctx, policy)   # one shared bank per target, matching original script
    r0 = solve_melitz_delta!(session, theta_pt, policy; warm_start_source=:previous)
    n_solves_a[] += 1
    @assert r0 isa FiniteSolved "base point target=$target not FiniteSolved: $(typeof(r0))"
    Delta0 = r0.Delta
    println("target=$target  g=$g_pt  Delta0=$Delta0  (historical Delta0: ",
            target == 0.5 ? "0.4832764950468894" : "0.7626745479773996", ")")

    af_dirs = [begin
        v = zeros(n); v[2:end] .= randn(MersenneTwister(1000 + i), n - 1); v ./ norm(v)
    end for i in 1:3]
    # svd_basis reconstruction: identical to Phase 1's own near-null build, at the target=0.5
    # starting point (the original script built ONE svd_basis at the target=0.5 point and
    # reused it for BOTH target loops -- reconstructed identically here).
    n_sub = min(80, n - 1)
    sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)
    if target == 0.5   # build once, at the target=0.5 point, exactly as the original script did
        global theta_start_for_basis = theta_pt
        Wsub = 4000; idxsub = 1:Wsub
        Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
        ei = zeros(n)
        for (jj, kk) in enumerate(sub_coords)
            ei[kk] = 1.0
            dG = melitz_moment_directional_derivative(theta_start_for_basis, ei, ctx, obj)
            Jsub[:, jj] .= vec(dG[idxsub, :])
            ei[kk] = 0.0
        end
        global Usvd = svd(Jsub)
    end
    v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
    v_steepest = zeros(n); v_steepest[sub_coords] .= Usvd.V[:, 1]
    svd_dirs = [v_near_null ./ norm(v_near_null), v_steepest ./ norm(v_steepest)]
    svd_names = ["svd_near_null", "svd_steepest"]

    # Representative subset (not all 10 directions x2 signs=20 per target): af_random_1 (both
    # signs), svd_near_null (sign=+1 only -- sign=-1@target=0.5 IS the anomaly, already
    # covered exhaustively by Phase 1), svd_steepest (both signs).
    selected = [("af_random_1", af_dirs[1], 1.0), ("af_random_1", af_dirs[1], -1.0),
                ("svd_near_null", svd_dirs[1], 1.0),
                ("svd_steepest", svd_dirs[2], 1.0), ("svd_steepest", svd_dirs[2], -1.0)]
    step_norm = 0.05
    for (label, dvec, sign) in selected
        theta_new = theta_pt .+ sign * step_norm .* dvec
        r_new = solve_melitz_delta!(session, theta_new, policy; warm_start_source=:previous)
        n_solves_a[] += 1
        finite_new = r_new isa FiniteSolved
        push!(rows_a, (target_delta=target, g=g_pt, direction=label, sign=sign, step_norm=step_norm,
            Delta0=Delta0, result_type=string(typeof(r_new)),
            Delta_after_step=(finite_new ? r_new.Delta : NaN)))
        @printf("  [%-14s sign=%+.0f] result=%-22s Delta=%s\n", label, sign, typeof(r_new),
                finite_new ? @sprintf("%.4e", r_new.Delta) : "NA")
        flush(stdout)
    end
end
println("\nPart A total solves: ", n_solves_a[], " (budget: <=10)")
flush(stdout)

# ============================================================================
# Part B: one nested-block-search structural re-run (gamma_only, via solve_melitz_finite_delta_bound)
# ============================================================================
println("\n" * "="^100); println("PART B: gamma_only nested-block re-run via solve_melitz_finite_delta_bound"); println("="^100)

g_start = g_for_target(0.5)
theta_start = copy(theta0); theta_start[1] = g_start
sqp_base = joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt")
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")

function make_opt(base, tag; delta, maxit, maxtime_real)
    tmpdir = joinpath(OUTDIR, "tmp_opt_post_consolidation_2026-07-28"); mkpath(tmpdir)
    lines = readlines(base)
    lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
    push!(lines, @sprintf("delta           %.6g", delta))
    push!(lines, @sprintf("maxit           %d", maxit))
    push!(lines, @sprintf("maxtime_real    %.1f", maxtime_real))
    path = joinpath(tmpdir, tag * ".opt")
    open(io -> foreach(l -> println(io, l), lines), path, "w")
    return path
end
function block_box(n, D, g_radius, A_radius, f_radius)
    nA = D^2 - 1; b = zeros(n); b[1] = g_radius; b[2:1+nA] .= A_radius; b[2+nA:end] .= f_radius; b
end

opt = make_opt(sqp_base, "phase4_gamma_only_recert_2026-07-28"; delta=0.05, maxit=1000, maxtime_real=180.0)
box = block_box(n, D, 0.5, 0.0, 0.0)   # gamma-only: A/f radius = 0
vs = ones(n); vs[1] = 1e-4

MELITZ_PROFILE[] = true; melitz_profile_reset!()
t0 = time()
res = solve_melitz_finite_delta_bound(ctx, obj, theta_start; delta=1.0, direction=:upper,
    policy=policy, gradient_backend=:auto, theta_box=box,
    cutoff_constraint_backend=:linear, inner_loop_opt=inner_opt, outer_loop_opt=opt,
    var_scale=vs, var_center=collect(Float64.(theta_start)),
    backend=:matrix_free, forbid_dense_fallback=true, objective_scale=:auto)
wall = time() - t0
MELITZ_PROFILE[] = false

cv = res.cold_verified_incumbent
if cv === nothing
    println("gamma_only re-run: NO cold_verified_incumbent (nStatus=$(res.nStatus)) -- DIVERGENT from historical (which produced a real incumbent)")
    row_b = (block="gamma_only", nStatus=res.nStatus, wall=wall, dg=NaN, DeltaStar=NaN)
else
    theta_final = cv.eval.theta_free
    dg = theta_final[1] - theta_start[1]
    row_b = (block="gamma_only", nStatus=res.nStatus, wall=wall, dg=dg, DeltaStar=cv.eval.Delta)
    println("gamma_only re-run: nStatus=$(res.nStatus)  wall=$(round(wall,digits=1))s  dg=$dg  DeltaStar=$(cv.eval.Delta)")
    println("  historical:       nStatus=0             wall=45.2s               dg=-0.010048147673314078  DeltaStar=0.8603593934221736")
end
flush(stdout)

open(joinpath(OUTDIR, "melitz_post_consolidation_phase4_archived_d20_replay_2026-07-28.csv"), "w") do io
    println(io, "part,target_delta,g,direction,sign,step_norm,Delta0,result_type,Delta_after_step")
    for r in rows_a
        println(io, "A,$(r.target_delta),$(r.g),$(r.direction),$(r.sign),$(r.step_norm),$(r.Delta0),$(r.result_type),$(r.Delta_after_step)")
    end
    println(io, "B,,,gamma_only_nested_rerun,,,,nStatus=$(row_b.nStatus);dg=$(row_b.dg),$(row_b.DeltaStar)")
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase4_archived_d20_replay_2026-07-28.csv")
println("\nDONE Phase 4 (post-consolidation).")
