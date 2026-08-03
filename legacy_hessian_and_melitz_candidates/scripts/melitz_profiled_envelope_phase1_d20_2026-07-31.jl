# Phase 1 validation (real D20): exact smooth profiled envelope derivative
# (profiled_envelope_gradient.jl) vs fully reprofiled central differences over (g,q_free),
# every endpoint genuinely reoptimizing A via solve_melitz_fixed_q_A_profile_v2. Three states:
# the D20 profiled anchor, and the best available cold-verified upper/lower incumbents from
# the just-completed delta=0.5 production campaign (docs/key_results/production_delta0p5_2026-07-31,
# commit d0904c7 -- this session's own worktree base, read via `git log` before this script was
# written, not re-run).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
WT = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profiled-q-envelope-gradient-2026-07-31"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(WT, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization

melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT = joinpath(WT, "melitz_middle_loop_opt_2026-07-30.opt")
OUTDIR = joinpath(WT, "docs", "key_results")
PRODDIR = joinpath(WT, "docs", "key_results", "production_delta0p5_2026-07-31")

# ============================================================================
# D20 fixture (identical recipe to the production campaign's own scripts).
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
D20 = ctx_d20.D; nA20 = D20^2 - 1; nq20 = D20^2 - 2
println("D=", D20, "  nA=", nA20, "  nq=", nq20); flush(stdout)

obj_d20.use_cached_x = false; obj_d20.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj_d20, theta0_d20)
@printf("D20 base-point solve: %.2fs  Delta0=%.10f  lfd_ok=%s  nStatus=%d\n", time() - t0, lfd0.Delta, lfd0.lfd_ok, lfd0.nStatus)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
flush(stdout)

session_d20 = MelitzInnerSession(obj_d20, ctx_d20, policy_cap)

theta_plain0_d20 = melitz_unpower_theta_free(theta0_d20, ctx_d20)
A_free0_d20 = theta_plain0_d20[2:1+nA20]
g0 = theta_plain0_d20[1]
_, _, _, _, q0_d20 = expand_free_theta_logcutoff(theta_plain0_d20, ctx_d20)

function q_full_at_theta(theta_plain::Vector{Float64}, ctx)
    _, _, _, _, q = expand_free_theta_logcutoff(theta_plain, ctx)
    return q
end

function profile_A(session, q_target::Matrix{Float64}, gpj_target::Float64, A_free_start::Vector{Float64}, ctx;
                    max_evals::Int=150, box::Real=0.1)
    theta_for_sys = melitz_fixed_q_state_theta(A_free_start, q_target, gpj_target, ctx)
    sys = melitz_fixed_q_middle_constraint_system(theta_for_sys, ctx, session.obj)
    A_proj = melitz_project_start_to_middle_constraints(copy(A_free_start), sys, ctx)
    session.obj.use_cached_x = false; session.obj.x .= NaN
    res = solve_melitz_fixed_q_A_profile_v2(session, q_target, gpj_target, A_proj, ctx;
        coordinate=:logA, max_evals=max_evals, box=box, outer_loop_opt=MIDDLE_OPT,
        sys=sys, cap_handling=:barrier, cap_barrier_multiple=5.0)
    return res, sys
end

const ROWS = NamedTuple[]

function run_state(label::String, session, ctx, theta_plain_star0::Vector{Float64};
                    hs_q=(1e-4,), hs_g=(1e-3,), active_tol=1e-8, extra_dirs=Tuple{Symbol,Vector{Float64}}[])
    D_ = ctx.D
    nA_ = D_^2 - 1
    nq_ = D_^2 - 2
    println("\n" * "="^90); println("STATE: $label"); println("="^90); flush(stdout)

    A_free_star0 = theta_plain_star0[2:1+nA_]
    g0_ = theta_plain_star0[1]
    q0 = q_full_at_theta(theta_plain_star0, ctx)

    # Cold-reverify this is genuinely a middle optimum's own solved state before computing the
    # envelope derivative (profiled_envelope_derivative does its own internal cold re-verify too).
    env = profiled_envelope_derivative(session, theta_plain_star0, q0, ctx)
    @printf("  envelope: dPhi_dg=%.8g  |dPhi_dq_free|=%.6g  n_active_rows=%d  Delta_check=%.10f  nStatus=%d\n",
        env.dPhi_dg, norm(env.dPhi_dq_free), env.n_active, env.Delta, env.nStatus)
    flush(stdout)

    rng = MersenneTwister(2026)
    dirs = Vector{Tuple{Symbol,Vector{Float64}}}()
    # coordinate direction ON the focal-origin row (exact_q_smooth_gradient is nonzero ONLY
    # there) -- a genuinely informative coordinate test, not a trivially-zero one.
    c_free, pivot_idx, other = melitz_cached_f_pivot_parts(ctx)
    f_free_lin = ctx.f_free_lin
    j = ctx.target_country
    focal_free_idx = findfirst(k -> lin2od(f_free_lin[other[k]], D_)[1] == j, eachindex(other))
    if focal_free_idx !== nothing
        push!(dirs, (:coord_focal_row, (v = zeros(nq_); v[focal_free_idx] = 1.0; v)))
    end
    d = randn(rng, nq_); d ./= norm(d)
    push!(dirs, (:random_dense_1, d))
    weights = [abs(c_free[other[k]] / c_free[pivot_idx]) for k in eachindex(other)]
    kmax = argmax(weights)
    push!(dirs, (:gravity_pivot_sensitive, (v = zeros(nq_); v[kmax] = 1.0; v)))
    append!(dirs, extra_dirs)

    q_free_idx = (1 + nA_ + 1):length(theta_plain_star0)
    fp0 = env.chamber_rank

    for (dname, dvec) in dirs
        for h in hs_q
            theta_p = copy(theta_plain_star0); theta_p[q_free_idx] .+= h .* dvec
            theta_m = copy(theta_plain_star0); theta_m[q_free_idx] .-= h .* dvec
            q_p = q_full_at_theta(theta_p, ctx)
            q_m = q_full_at_theta(theta_m, ctx)
            fp_p = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_p, exp(g0_), ctx), ctx, session.obj)
            fp_m = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_m, exp(g0_), ctx), ctx, session.obj)
            same_chamber = (fp_p == fp0) && (fp_m == fp0)
            t0d = time()
            res_p, _ = profile_A(session, q_p, exp(g0_), A_free_star0, ctx)
            res_m, _ = profile_A(session, q_m, exp(g0_), A_free_star0, ctx)
            wall = time() - t0d
            okp = res_p.r_incumbent isa FiniteSolved
            okm = res_m.r_incumbent isa FiniteSolved
            predicted = dot(env.dPhi_dq_free, dvec)
            secant = (okp && okm) ? (res_p.Delta_incumbent - res_m.Delta_incumbent) / (2h) : NaN
            relerr = (okp && okm) ? abs(predicted - secant) / max(1e-10, abs(secant)) : NaN
            sign_agree = (okp && okm) ? (sign(predicted) == sign(secant)) : missing
            @printf("  [q %-24s h=%.0e] chamber_ok=%s pred=%.6g secant=%.6g relerr=%.4g sign_agree=%s (F/F=%s/%s) wall=%.1fs\n",
                dname, h, same_chamber, predicted, secant, relerr, string(sign_agree), okp, okm, wall)
            flush(stdout)
            push!(ROWS, (state=label, kind="q", direction=String(dname), h=h, same_chamber=same_chamber,
                predicted=predicted, secant=secant, relerr=relerr, sign_agree=sign_agree,
                plus_finite=okp, minus_finite=okm, plus_nStatus=res_p.nStatus, minus_nStatus=res_m.nStatus,
                plus_Delta=res_p.Delta_incumbent, minus_Delta=res_m.Delta_incumbent))
        end
    end

    for h in hs_g
        gp = g0_ + h; gm = g0_ - h
        q_p = q_full_at_theta(vcat(gp, theta_plain_star0[2:end]), ctx)
        q_m = q_full_at_theta(vcat(gm, theta_plain_star0[2:end]), ctx)
        fp_p = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_p, exp(gp), ctx), ctx, session.obj)
        fp_m = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_m, exp(gm), ctx), ctx, session.obj)
        same_chamber = (fp_p == fp0) && (fp_m == fp0)
        t0d = time()
        res_p, _ = profile_A(session, q_p, exp(gp), A_free_star0, ctx)
        res_m, _ = profile_A(session, q_m, exp(gm), A_free_star0, ctx)
        wall = time() - t0d
        okp = res_p.r_incumbent isa FiniteSolved
        okm = res_m.r_incumbent isa FiniteSolved
        predicted = env.dPhi_dg
        secant = (okp && okm) ? (res_p.Delta_incumbent - res_m.Delta_incumbent) / (2h) : NaN
        relerr = (okp && okm) ? abs(predicted - secant) / max(1e-10, abs(secant)) : NaN
        sign_agree = (okp && okm) ? (sign(predicted) == sign(secant)) : missing
        @printf("  [g                        h=%.0e] chamber_ok=%s pred=%.6g secant=%.6g relerr=%.4g sign_agree=%s (F/F=%s/%s) wall=%.1fs\n",
            h, same_chamber, predicted, secant, relerr, string(sign_agree), okp, okm, wall)
        flush(stdout)
        push!(ROWS, (state=label, kind="g", direction="welfare", h=h, same_chamber=same_chamber,
            predicted=predicted, secant=secant, relerr=relerr, sign_agree=sign_agree,
            plus_finite=okp, minus_finite=okm, plus_nStatus=res_p.nStatus, minus_nStatus=res_m.nStatus,
            plus_Delta=res_p.Delta_incumbent, minus_Delta=res_m.Delta_incumbent))
    end
    return env
end

# --- State 1: D20 profiled anchor (profile A fresh at the anchor's own g0,q0) ---
res_anchor, _ = profile_A(session_d20, q0_d20, exp(g0), A_free0_d20, ctx_d20)
println("\nD20 anchor pre-check: nStatus=$(res_anchor.nStatus) Delta=$(res_anchor.Delta_incumbent) wall=$(round(res_anchor.wall_s,digits=1))s"); flush(stdout)
@assert res_anchor.r_incumbent isa FiniteSolved
theta_anchor_star = res_anchor.theta_free_incumbent

# gravity-pivot-sensitive + the "previous reduced-q direction" for the anchor specifically
bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
x0_d20 = copy(lfd0.dual_x)
stage_d20 = melitz_build_reduced_q_stage(theta0_d20, x0_d20, ctx_d20, obj_d20, 1; bandwidth_policy=bwpolicy, target_switches=100)
b_q_d20 = stage_d20 === nothing ? nothing : stage_d20.q_basis_free
extra = b_q_d20 === nothing ? Tuple{Symbol,Vector{Float64}}[] : [(:reduced_q_direction, b_q_d20 ./ norm(b_q_d20))]

run_state("D20_anchor", session_d20, ctx_d20, theta_anchor_star; extra_dirs=extra)

# --- State 2: best upper incumbent (reduced_q_pre_switch, GT=7.496042%) ---
ckpt_upper = deserialize(joinpath(PRODDIR, "checkpoints", "reduced_q_pre_switch_upper.jls"))
best_upper = ckpt_upper.points[ckpt_upper.most_extreme_idx]
println("\nD20 upper incumbent loaded: GT=$(best_upper.GT) Delta=$(best_upper.Delta) classification=$(best_upper.classification)"); flush(stdout)
run_state("D20_upper_incumbent_reduced_q_pre_switch", session_d20, ctx_d20, best_upper.theta_free)

# --- State 3: best lower incumbent (current_calibration, GT=1.252141%) ---
ckpt_lower = deserialize(joinpath(PRODDIR, "checkpoints", "current_calibration_lower.jls"))
best_lower = ckpt_lower.points[ckpt_lower.most_extreme_idx]
println("\nD20 lower incumbent loaded: GT=$(best_lower.GT) Delta=$(best_lower.Delta) classification=$(best_lower.classification)"); flush(stdout)
run_state("D20_lower_incumbent_current_calibration", session_d20, ctx_d20, best_lower.theta_free)

open(joinpath(OUTDIR, "melitz_profiled_envelope_phase1_d20_2026-07-31.csv"), "w") do io
    println(io, "state,kind,direction,h,same_chamber,predicted,secant,relerr,sign_agree,plus_finite,minus_finite,plus_nStatus,minus_nStatus,plus_Delta,minus_Delta")
    for r in ROWS
        println(io, "$(r.state),$(r.kind),$(r.direction),$(r.h),$(r.same_chamber),$(r.predicted),$(r.secant),$(r.relerr),$(r.sign_agree),$(r.plus_finite),$(r.minus_finite),$(r.plus_nStatus),$(r.minus_nStatus),$(r.plus_Delta),$(r.minus_Delta)")
    end
end
println("\nDone. n_rows=", length(ROWS)); flush(stdout)
