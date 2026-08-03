# Phase 1 validation (D4 only, fast iteration): exact smooth profiled envelope derivative
# (profiled_envelope_gradient.jl) vs fully reprofiled central differences over (g,q_free),
# every endpoint genuinely reoptimizing A via solve_melitz_fixed_q_A_profile_v2.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
WT = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profiled-q-envelope-gradient-2026-07-31"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(WT, "src", "melitz", "include_melitz.jl"))
using Printf, LinearAlgebra, Random

melitz_thread_startup_report()
LinearAlgebra.BLAS.set_num_threads(1)
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

CAP = 10.0
policy_cap = CappedEvaluation(CAP)
MIDDLE_OPT = joinpath(WT, "melitz_middle_loop_opt_2026-07-30.opt")

# ============================================================================
# D4 fixture (seed=29, the whitelisted-robust seed per feedback-melitz-d4-seed-fragility).
# ============================================================================
FIXTURE4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta04 = build_melitz_psi_bundle(FIXTURE4; outer_parameterization=:logcutoff, policy=policy_cap,
    backend=:matrix_free, forbid_dense_fallback=true)
ctx4 = obj4.γ
D4 = ctx4.D; nA4 = D4^2 - 1; nq4 = D4^2 - 2
session4 = MelitzInnerSession(obj4, ctx4, policy_cap)
theta_plain04 = melitz_unpower_theta_free(theta04, ctx4)
A04, f04, gpj04, fjj04, q04 = expand_free_theta_logcutoff(theta_plain04, ctx4)
A_free04 = theta_plain04[2:1+nA4]
g0 = log(gpj04)
println("D4: D=$D4 nA=$nA4 nq=$nq4 g0=$g0"); flush(stdout)

function q_full_at_theta(theta_plain::Vector{Float64}, ctx)
    _, _, _, _, q = expand_free_theta_logcutoff(theta_plain, ctx)
    return q
end

"""Profile A at (g,q_full) starting from `A_free_start` (unprojected OK -- projected here)."""
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

function run_state(label::String, session, ctx, q0::Matrix{Float64}, g0_::Float64, A_free_start0::Vector{Float64};
                    hs_q=(1e-5, 1e-4), hs_g=(1e-4, 1e-3), n_random_dirs::Int=2, active_tol=1e-8)
    D_ = ctx.D
    nA_ = D_^2 - 1
    nq_ = D_^2 - 2
    println("\n" * "="^90); println("STATE: $label"); println("="^90); flush(stdout)

    # --- 1. Profile A at the anchor (g0,q0) ---
    res0, sys0 = profile_A(session, q0, exp(g0_), A_free_start0, ctx)
    println("  anchor profile: nStatus=$(res0.nStatus) Delta_incumbent=$(res0.Delta_incumbent) " *
            "wall=$(round(res0.wall_s,digits=1))s unique_A=$(res0.unique_A_points)"); flush(stdout)
    res0.r_incumbent isa FiniteSolved || (@warn "$label: anchor not FiniteSolved, skipping"; return)
    Phi0 = res0.Delta_incumbent
    A_free_star0 = res0.A_free_incumbent
    theta_star0 = res0.theta_free_incumbent

    # --- 2. Exact envelope derivative at the profiled optimum ---
    env = profiled_envelope_derivative(session, theta_star0, q0, ctx; sys=sys0, A_free_incumbent=A_free_star0,
                                        active_tol=active_tol)
    @printf("  envelope: dPhi_dg=%.8g  |dPhi_dq_free|=%.6g  n_active_rows=%d/%d  Delta_check=%.10f (vs %.10f)\n",
        env.dPhi_dg, norm(env.dPhi_dq_free), env.n_active, length(env.active_rows), env.Delta, Phi0)
    flush(stdout)

    # --- 3. Direction bank over q_free ---
    rng = MersenneTwister(2026)
    dirs = Vector{Tuple{Symbol,Vector{Float64}}}()
    push!(dirs, (:coord_1, (v = zeros(nq_); v[1] = 1.0; v)))
    if nq_ >= 3
        push!(dirs, (:coord_mid, (v = zeros(nq_); v[cld(nq_, 2)] = 1.0; v)))
    end
    for k in 1:n_random_dirs
        d = randn(rng, nq_); d ./= norm(d)
        push!(dirs, (Symbol("random_dense_$k"), d))
    end
    # gravity-pivot-sensitive direction: the free q coordinate with the largest adjoint weight
    # onto the reconstructed q/f-pivot cell (|c_free[i]/c_free[pivot]|), i.e. the coordinate
    # whose movement most strongly moves the q-gravity pivot's own reconstructed cell.
    c_free, pivot_idx, other = melitz_cached_f_pivot_parts(ctx)
    weights = [abs(c_free[other[k]] / c_free[pivot_idx]) for k in eachindex(other)]
    kmax = argmax(weights)
    push!(dirs, (:gravity_pivot_sensitive, (v = zeros(nq_); v[kmax] = 1.0; v)))

    theta_plain_star0 = copy(theta_star0)  # already :logcutoff plain (A_free,g,q_free layout)
    q_free_idx = (1 + nA_ + 1):length(theta_plain_star0)
    fp0 = env.chamber_rank

    for (dname, d) in dirs
        for h in hs_q
            theta_p = copy(theta_plain_star0); theta_p[q_free_idx] .+= h .* d
            theta_m = copy(theta_plain_star0); theta_m[q_free_idx] .-= h .* d
            q_p = q_full_at_theta(theta_p, ctx)
            q_m = q_full_at_theta(theta_m, ctx)
            fp_p = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_p, exp(g0_), ctx), ctx, session.obj)
            fp_m = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_m, exp(g0_), ctx), ctx, session.obj)
            same_chamber = (fp_p == fp0) && (fp_m == fp0)
            res_p, _ = profile_A(session, q_p, exp(g0_), A_free_star0, ctx)
            res_m, _ = profile_A(session, q_m, exp(g0_), A_free_star0, ctx)
            okp = res_p.r_incumbent isa FiniteSolved
            okm = res_m.r_incumbent isa FiniteSolved
            predicted = dot(env.dPhi_dq_free, d)
            secant = (okp && okm) ? (res_p.Delta_incumbent - res_m.Delta_incumbent) / (2h) : NaN
            relerr = (okp && okm) ? abs(predicted - secant) / max(1e-10, abs(secant)) : NaN
            sign_agree = (okp && okm) ? (sign(predicted) == sign(secant)) : missing
            @printf("  [q %-24s h=%.0e] chamber_ok=%s pred=%.6g secant=%.6g relerr=%.4g sign_agree=%s (F/F=%s/%s)\n",
                dname, h, same_chamber, predicted, secant, relerr, string(sign_agree), okp, okm)
            flush(stdout)
            push!(ROWS, (state=label, kind="q", direction=String(dname), h=h, same_chamber=same_chamber,
                predicted=predicted, secant=secant, relerr=relerr, sign_agree=sign_agree,
                plus_finite=okp, minus_finite=okm, plus_nStatus=res_p.nStatus, minus_nStatus=res_m.nStatus))
        end
    end

    # --- 4. g direction ---
    for h in hs_g
        gp = g0_ + h; gm = g0_ - h
        q_p = q_full_at_theta(vcat(gp, theta_plain_star0[2:end]), ctx)
        q_m = q_full_at_theta(vcat(gm, theta_plain_star0[2:end]), ctx)
        fp_p = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_p, exp(gp), ctx), ctx, session.obj)
        fp_m = melitz_chamber_fingerprint(melitz_fixed_q_state_theta(A_free_star0, q_m, exp(gm), ctx), ctx, session.obj)
        same_chamber = (fp_p == fp0) && (fp_m == fp0)
        res_p, _ = profile_A(session, q_p, exp(gp), A_free_star0, ctx)
        res_m, _ = profile_A(session, q_m, exp(gm), A_free_star0, ctx)
        okp = res_p.r_incumbent isa FiniteSolved
        okm = res_m.r_incumbent isa FiniteSolved
        predicted = env.dPhi_dg
        secant = (okp && okm) ? (res_p.Delta_incumbent - res_m.Delta_incumbent) / (2h) : NaN
        relerr = (okp && okm) ? abs(predicted - secant) / max(1e-10, abs(secant)) : NaN
        sign_agree = (okp && okm) ? (sign(predicted) == sign(secant)) : missing
        @printf("  [g                        h=%.0e] chamber_ok=%s pred=%.6g secant=%.6g relerr=%.4g sign_agree=%s (F/F=%s/%s)\n",
            h, same_chamber, predicted, secant, relerr, string(sign_agree), okp, okm)
        flush(stdout)
        push!(ROWS, (state=label, kind="g", direction="welfare", h=h, same_chamber=same_chamber,
            predicted=predicted, secant=secant, relerr=relerr, sign_agree=sign_agree,
            plus_finite=okp, minus_finite=okm, plus_nStatus=res_p.nStatus, minus_nStatus=res_m.nStatus))
    end
    return env, res0
end

run_state("D4_anchor_seed29", session4, ctx4, q04, g0, A_free04)

# --- D4 second state: the anchor's own Delta (3.4e-6) sits far below the middle-loop KNITRO
# solver's own opttol_abs=1e-4 -- any two independent middle solves near it disagree by an
# amount comparable to their own convergence noise, swamping the (tiny) true slope (h*slope ~
# 1e-9 at h=1e-4). Construct a genuinely informative second D4 state by offsetting welfare
# substantially away from the near-perfect calibration point (Delta* becomes O(0.01-1), well
# above KNITRO's own absolute tolerance) BEFORE profiling A there -- this is the required
# "two D4 states" per the governing prompt, chosen deliberately (not randomly) to avoid the
# degenerate near-zero-divergence floor the anchor itself sits at.
g_offset = g0 - 0.08   # gscan (melitz_profiled_envelope_d4_gscan_2026-07-31.jl): Delta=0.1096
                       # FiniteSolved here, with comfortable margin to both edges (delta_g=+0.08
                       # AboveEvaluationCap, delta_g=-0.18 AboveEvaluationCap) -- the anchor's
                       # own Delta (3.4e-6) sits far below KNITRO's own opttol_abs=1e-4, making
                       # it uninformative for this validation (disclosed in the report).
res_off, sys_off = profile_A(session4, q04, exp(g_offset), A_free04, ctx4)
println("\nD4 offset-state pre-check: g_offset=$g_offset nStatus=$(res_off.nStatus) Delta=$(res_off.Delta_incumbent)"); flush(stdout)
run_state("D4_offset_g0plus1", session4, ctx4, q04, g_offset, res_off.A_free_incumbent)

OUTDIR = joinpath(WT, "docs", "key_results")
mkpath(OUTDIR)
open(joinpath(OUTDIR, "melitz_profiled_envelope_phase1_d4_2026-07-31.csv"), "w") do io
    println(io, "state,kind,direction,h,same_chamber,predicted,secant,relerr,sign_agree,plus_finite,minus_finite,plus_nStatus,minus_nStatus")
    for r in ROWS
        println(io, "$(r.state),$(r.kind),$(r.direction),$(r.h),$(r.same_chamber),$(r.predicted),$(r.secant),$(r.relerr),$(r.sign_agree),$(r.plus_finite),$(r.minus_finite),$(r.plus_nStatus),$(r.minus_nStatus)")
    end
end
println("\nDone. n_rows=", length(ROWS)); flush(stdout)
