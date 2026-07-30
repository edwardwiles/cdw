# Phase 7 (profiledA_parallel_speed_and_cutoff_portfolio governing prompt, 2026-07-30):
# structured cutoff-anchor portfolio. Six economically/numerically meaningful free-cutoff
# anchors, NOT arbitrary random vectors:
#
#   1. current_calibration      -- the anchor itself (t=0 along every perturbation direction).
#   2. rank_spaced               -- a small deterministic per-rank nudge to the free-q block
#      (breaks any exact same-bin ties into a strict rank ordering) while leaving A/f/gravity
#      untouched (q-gravity/A-gravity/focal-normalization are re-derived fresh from the SAME
#      pivot machinery the production theta representation already uses, so they hold
#      structurally by construction, not by explicit re-projection).
#   3. origin_block_perturbed    -- moderate shift of ALL free q cells with origin=Korea (idx14,
#      the audited negative-switch-cliff origin, melitz_d20_negative_switch_geometry_audit_2026-07-30.md).
#   4. destination_block_perturbed -- moderate shift of ALL free q cells with destination=focal
#      (France) -- the calibration's own reference country.
#   5. reduced_q_pre_switch      -- EXACT reused point from the negative-switch audit's own
#      documented reduced-q direction (b_q, PowerScaledQBandwidth(1e-3,80_000,0.5),
#      target_switches=100), t=6.31e-3, sign=-1 ("pre_switch1" -- before the first minus-side
#      switch, previously verified FiniteSolved with fixed-A Delta~=anchor).
#   6. reduced_q_post_switch     -- SAME direction, t=1.26e-2, sign=-1 ("post_switch2" -- PAST
#      the audited cliff; fixed-A DeltaStar there is AboveEvaluationCap at ~493K, but the
#      SAME session's own profiled-A search found Phi=0.293, FiniteSolved -- a previously
#      discovered good D20-derived cutoff state, reused here as anchor #6 per the governing
#      prompt's own "one or two previously discovered good D4/D20-derived cutoff states" ask).
#
# For each: profile A (solve_melitz_fixed_q_A_profile_v2) at the CALIBRATION welfare point g0,
# require FiniteSolved, record Delta*, reject anchors with pathological scaling/immediate
# support cliffs. Does NOT optimize the cutoff vector itself in this session (governing prompt).
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
const MAX_MIDDLE_EVALS = 150
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
        p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6), focal, countries
end
calib, focal, countries = load_realD20_calib()
korea_idx = findfirst(==("kor"), countries)
korea_idx === nothing && (korea_idx = 14)   # documented index from the negative-switch audit
println("focal=", focal, " (", countries[focal], ")  korea_idx=", korea_idx, " (", get(countries, korea_idx, "?"), ")"); flush(stdout)

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
q_free0_d20 = theta_plain0_d20[2+nA20:end]
g0 = theta_plain0_d20[1]
gpj0 = exp(g0)

session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@assert lfd0.lfd_ok
x0_d20 = copy(lfd0.dual_x)

q_pivot = build_q_gravity_pivot(ctx_d20)
od_of_free = [lin2od(q_pivot.other[k], D20) for k in eachindex(q_pivot.other)]

# --- Anchor 5/6: reduced-q direction (EXACT reuse of the negative-switch audit's own basis). ---
bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1;
    bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20.q_basis_free
@printf("reduced-q basis: |b_q|=%.6f\n", norm(b_q_d20)); flush(stdout)

function theta_plain_with_q_free(new_q_free::Vector{Float64})
    th = copy(theta_plain0_d20)
    th[2+nA20:end] .= new_q_free
    return th
end

anchors = NamedTuple[]
push!(anchors, (label="current_calibration", q_free=copy(q_free0_d20)))

# Anchor 2: rank-spaced -- small deterministic per-rank nudge (breaks exact ties, does not
# re-order any existing strict inequality since the nudge magnitude is far below the smallest
# observed same-bin/ordering gap at real-D20 scale).
rank_nudge = [1e-6 * k for k in eachindex(q_free0_d20)]
push!(anchors, (label="rank_spaced", q_free=q_free0_d20 .+ rank_nudge))

# Anchor 3: origin-block perturbation (Korea's own outbound free q cells).
origin_mask = [od[1] == korea_idx for od in od_of_free]
q_free_origin = copy(q_free0_d20)
q_free_origin[origin_mask] .-= 0.02
push!(anchors, (label="origin_block_korea", q_free=q_free_origin))

# Anchor 4: destination-block perturbation (focal country's own inbound free q cells).
dest_mask = [od[2] == focal for od in od_of_free]
q_free_dest = copy(q_free0_d20)
q_free_dest[dest_mask] .+= 0.02
push!(anchors, (label="destination_block_focal", q_free=q_free_dest))

# Anchors 5/6: reduced-q direction, pre- and post- the audited minus-side cliff.
push!(anchors, (label="reduced_q_pre_switch", q_free=q_free0_d20 .+ (-1.0) .* 6.31e-3 .* b_q_d20))
push!(anchors, (label="reduced_q_post_switch", q_free=q_free0_d20 .+ (-1.0) .* 1.26e-2 .* b_q_d20))

results = NamedTuple[]
for anc in anchors
    println("\n", "="^100); println("ANCHOR: ", anc.label); flush(stdout)
    theta_anchor_plain = theta_plain_with_q_free(anc.q_free)
    _, f_anchor, _, _, q_anchor_full = expand_free_theta_logcutoff(theta_anchor_plain, ctx_d20)
    A_anchor_full = exp.(reshape(pivot_expand(A_free0_d20, ctx_d20.A_pivot), D20, D20))

    # Reject up front if the reconstructed A/f are pathologically scaled (>30 log-units from the
    # calibration's own values -- an obvious sign of a nonsensical cutoff configuration, not a
    # judgment call made after the fact).
    logA_spread = maximum(abs.(log.(A_anchor_full)))
    logf_spread = maximum(abs.(log.(f_anchor)))
    pathological = !all(isfinite, A_anchor_full) || !all(isfinite, f_anchor) || logA_spread > 60 || logf_spread > 60
    if pathological
        println("  REJECTED: pathological scaling (logA_spread=", logA_spread, " logf_spread=", logf_spread, ")")
        push!(results, (label=anc.label, rejected=true, reason="pathological_scaling", Delta=NaN,
            classification="rejected", unique_A_points=0, unique_inner_solves=0, wall_s=0.0))
        continue
    end

    sys_t = melitz_fixed_q_middle_constraint_system(theta_anchor_plain, ctx_d20, obj_d20)
    A_start = melitz_project_start_to_middle_constraints(copy(A_free0_d20), sys_t, ctx_d20)

    session_d20.obj.use_cached_x = false; session_d20.obj.x .= NaN
    t0 = time()
    local res
    local ok = true
    try
        res = solve_melitz_fixed_q_A_profile_v2(session_d20, q_anchor_full, gpj0, A_start, ctx_d20;
            coordinate=:logA, max_evals=MAX_MIDDLE_EVALS, box=D20_MIDDLE_BOX, outer_loop_opt=MIDDLE_OPT_D20, sys=sys_t,
            cap_handling=CAP_HANDLING, cap_barrier_multiple=CAP_BARRIER_MULTIPLE)
    catch e
        ok = false
        @warn "anchor $(anc.label): solve threw" exception=(e, catch_backtrace())
    end
    wall = time() - t0
    if !ok
        push!(results, (label=anc.label, rejected=true, reason="threw", Delta=NaN,
            classification="error", unique_A_points=0, unique_inner_solves=0, wall_s=wall))
        continue
    end
    finite = res.r_incumbent isa FiniteSolved
    @printf("  classification=%s Delta=%.6g wall=%.1fs unique_A=%d unique_solves=%d\n",
        nameof(typeof(res.r_incumbent)), res.Delta_incumbent, wall, res.unique_A_points, res.unique_inner_solves)
    push!(results, (label=anc.label, rejected=!finite, reason=finite ? "" : "not_finite",
        Delta=res.Delta_incumbent, classification=String(nameof(typeof(res.r_incumbent))),
        unique_A_points=res.unique_A_points, unique_inner_solves=res.unique_inner_solves, wall_s=wall))
    flush(stdout)
end

outpath = joinpath(OUTDIR, "melitz_phase7_cutoff_portfolio_2026-07-30.csv")
open(outpath, "w") do io
    println(io, "label,rejected,reason,Delta,classification,unique_A_points,unique_inner_solves,wall_s")
    for r in results
        println(io, join([r.label, r.rejected, r.reason, r.Delta, r.classification, r.unique_A_points,
            r.unique_inner_solves, r.wall_s], ","))
    end
end
println("\nWrote ", outpath)
n_verified = count(r -> !r.rejected, results)
println("Verified (FiniteSolved) anchors: ", n_verified, " / ", length(results))
println("\nPHASE 7 CUTOFF-ANCHOR PORTFOLIO COMPLETE")
