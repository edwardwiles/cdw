# Phase 6 (real D20, mandatory bounded smoke test): predictor-corrector welfare continuation
# using the LFD-preserving constructor. Governing prompt:
# melitz_joint_Aq_feasibility_preserving_search_2026-07-30. NOT a claim of outer convergence
# -- a short, bounded (<=5 steps/direction) smoke test only.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))
theta0 = theta_q_rows[("realD20_seed1_W80000", 0.5)]

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

obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx = obj.γ
D = ctx.D; nA = D^2-1; nq = D^2-2
sorted_ctx = ctx.sorted_tail_ctx

obj.use_cached_x = false; obj.x .= NaN
lfd0 = melitz_recover_lfd(obj, theta0)
@assert lfd0.lfd_ok
@printf("anchor: Delta0=%.10f\n", lfd0.Delta); flush(stdout)

wage_ratio = ctx.w_prime / ctx.w[ctx.target_country]
wm0 = melitz_welfare_metrics_from_g(theta0[1], wage_ratio, ctx.sigma)
@printf("anchor welfare: g=%.6f GT=%.6f%% (theoretical, closed-form)\n", wm0.g, 100*wm0.gains_from_trade)
flush(stdout)

function g_for_GT(GT_target::Real, wage_ratio::Real, sigma::Real)
    kappa_ratio_target = 1 - GT_target
    gamma_prime_target = kappa_ratio_target^(sigma - 1) / wage_ratio^(sigma - 1)
    return log(gamma_prime_target)
end
# sanity check: g_for_GT(wm0.gains_from_trade,...) reproduces theta0[1]
g_check = g_for_GT(wm0.gains_from_trade, wage_ratio, ctx.sigma)
@assert isapprox(g_check, theta0[1]; atol=1e-9) "g<->GT inversion mismatch: $g_check vs $(theta0[1])"

function classify_theta_free(theta_free_lc, obj_lc, ctx_lc)
    session = MelitzInnerSession(obj_lc, ctx_lc, policy_cap)
    return solve_melitz_delta!(session, theta_free_lc, policy_cap)
end
function classify_summary(r)
    if r isa FiniteSolved
        return (kind="FiniteSolved", Delta=r.Delta, cert=NaN)
    elseif r isa AboveEvaluationCap
        return (kind="AboveEvaluationCap", Delta=NaN, cert=r.certified_lower_bound)
    elseif r isa InfiniteDeltaCertified
        return (kind="InfiniteDeltaCertified", Delta=NaN, cert=NaN)
    else
        return (kind="NumericalFailure", Delta=NaN, cert=NaN)
    end
end

theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
A_anchor, f_anchor, gpj_anchor, _, q_anchor = expand_free_theta_logcutoff(theta_plain0, ctx)
q_free_free_anchor = theta_plain0[2+nA:end]

STEP_PP = 0.05 / 100   # 0.05 gains-from-trade percentage points, as a fraction
MAX_STEPS = 5
MAX_SHRINKS = 3

results = NamedTuple[]

function run_continuation(direction_sign::Int, label::String)
    println("\n" * "="^100); println("Continuation direction: $label"); println("="^100); flush(stdout)
    theta_cur = copy(theta0)
    theta_plain_cur = copy(theta_plain0)
    qff_cur = copy(q_free_free_anchor)
    p_cur = copy(lfd0.weights)
    GT_cur = wm0.gains_from_trade
    Delta_cur = lfd0.Delta
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)

    for step in 1:MAX_STEPS
        step_pp = STEP_PP
        accepted = false
        local stC, rC, sC, tC_corr, t_inner
        for shrink in 1:MAX_SHRINKS
            GT_target = GT_cur + direction_sign * step_pp
            g_new = g_for_GT(GT_target, wage_ratio, ctx.sigma)

            tC0 = time()
            stC = melitz_construct_lfd_preserving_state(theta_cur, p_cur, ctx, obj, g_new, qff_cur;
                correct=true, max_corrector_rounds=8)
            tC_corr = time() - tC0

            if !stC.feasible
                @printf("[%s step=%d shrink=%d] step_pp=%.4f%% -> INFEASIBLE witness (max|trade_res|=%.2e focal=%.2e gravA=%.2e gravF=%.2e); shrinking\n",
                    label, step, shrink, 100*step_pp, maximum(abs.(stC.trade_residuals)), stC.focal_residual,
                    stC.gravity_A_residual, stC.gravity_f_residual)
                step_pp /= 2
                flush(stdout)
                continue
            end

            t_inner0 = time()
            rC = classify_theta_free(stC.theta_free, obj, ctx)
            t_inner = time() - t_inner0
            sC = classify_summary(rC)
            if sC.kind == "FiniteSolved" && sC.Delta <= 10.0
                accepted = true
                break
            else
                # Step 1E invariant: a verified witness implies the reoptimized point cannot be
                # AboveEvaluationCap/InfiniteDeltaCertified. Flag loudly (do NOT silently accept)
                # if this ever happens -- would indicate a corrector/witness bug, not shrink further
                # for THIS reason (the task's own critical-invariant #1).
                @printf("[%s step=%d shrink=%d] WITNESS SAYS FEASIBLE BUT REOPT CLASSIFIED %s (cert=%s) -- INVARIANT VIOLATION CANDIDATE, investigate\n",
                    label, step, shrink, sC.kind, sC.cert)
                flush(stdout)
                step_pp /= 2
            end
        end

        if !accepted
            @printf("[%s step=%d] could not accept after %d shrinks -- stopping continuation\n", label, step, MAX_SHRINKS)
            flush(stdout)
            break
        end

        n_switch_cells = count(!=(:ok), stC.A_status)  # should always be 0 for :ok-only recovery
        dA = norm(vec(stC.A) .- vec(A_anchor))
        dq = norm(vec(stC.q) .- vec(q_anchor))
        df = norm(vec(stC.f) .- vec(f_anchor))
        wm_new = melitz_welfare_metrics_from_g(g_for_GT(GT_cur + direction_sign*step_pp, wage_ratio, ctx.sigma), wage_ratio, ctx.sigma)
        @printf("[%s step=%d] ACCEPTED: GT %.4f%% -> %.4f%%  Delta*(reopt)=%.6f (was %.6f)  corrector=%.3fs inner=%.3fs  dA=%.3e dq=%.3e df=%.3e  divergence(p*)=%.6f\n",
            label, step, 100*GT_cur, 100*wm_new.gains_from_trade, sC.Delta, Delta_cur, tC_corr, t_inner,
            dA, dq, df, stC.divergence_pstar)
        flush(stdout)

        push!(results, (label=label, step=step, GT_from=GT_cur, GT_to=wm_new.gains_from_trade,
            step_pp_used=step_pp, Delta_before=Delta_cur, Delta_after=sC.Delta, kind=sC.kind,
            corrector_time_s=tC_corr, inner_time_s=t_inner, dA_norm=dA, dq_norm=dq, df_norm=df,
            divergence_pstar=stC.divergence_pstar, corrector_rounds=length(stC.corrector_trace),
            gravA_witness=stC.gravity_A_residual, gravF_witness=stC.gravity_f_residual,
            focal_witness=stC.focal_residual))

        # update state: reoptimized LFD becomes the new p*, new theta becomes current anchor
        lfd_new = melitz_recover_lfd(obj, stC.theta_free)
        @assert lfd_new.lfd_ok
        theta_cur = copy(stC.theta_free)
        theta_plain_cur = melitz_unpower_theta_free(theta_cur, ctx)
        A_cur2, f_cur2, gpj_cur2, _, q_cur2 = expand_free_theta_logcutoff(theta_plain_cur, ctx)
        qff_cur = theta_plain_cur[2+nA:end]
        p_cur = lfd_new.weights
        GT_cur = wm_new.gains_from_trade
        Delta_cur = sC.Delta
        melitz_update_operator_at_theta!(obj.op, theta_cur, ctx)
    end
end

run_continuation(+1, "upper_GT")
run_continuation(-1, "lower_GT")

# ================================================================================================
# Comparison: gamma-only continuation from the SAME anchor (no q/A correction at all -- literally
# only move g, hold q_free_free AND A_free at anchor), same step budget, same number of attempts.
# ================================================================================================
function run_gamma_only(direction_sign::Int, label::String)
    println("\n" * "="^100); println("Gamma-only comparison: $label"); println("="^100); flush(stdout)
    GT_cur = wm0.gains_from_trade
    gamma_rows = NamedTuple[]
    for step in 1:MAX_STEPS
        step_pp = STEP_PP
        GT_target = GT_cur + direction_sign * step_pp
        g_new = g_for_GT(GT_target, wage_ratio, ctx.sigma)
        theta_trial = copy(theta_plain0)
        theta_trial[1] = g_new
        r = classify_theta_free(theta_trial, obj, ctx)
        s = classify_summary(r)
        @printf("[gamma-only %s step=%d] GT_target=%.4f%%  kind=%s  Delta/cert=%.6g\n",
            label, step, 100*GT_target, s.kind, isnan(s.Delta) ? s.cert : s.Delta)
        flush(stdout)
        push!(gamma_rows, (label=label, step=step, GT_target=GT_target, kind=s.kind, Delta=s.Delta, cert=s.cert))
        s.kind == "FiniteSolved" || break
        GT_cur = GT_target
    end
    return gamma_rows
end
gamma_upper = run_gamma_only(+1, "upper_GT")
gamma_lower = run_gamma_only(-1, "lower_GT")

open(joinpath(OUTDIR, "melitz_aq_phase6_predictor_corrector_2026-07-30.csv"), "w") do io
    println(io, "label,step,GT_from,GT_to,step_pp_used,Delta_before,Delta_after,kind,corrector_time_s,inner_time_s,dA_norm,dq_norm,df_norm,divergence_pstar,corrector_rounds,gravA_witness,gravF_witness,focal_witness")
    for r in results
        println(io, join([r.label,r.step,r.GT_from,r.GT_to,r.step_pp_used,r.Delta_before,r.Delta_after,r.kind,
            r.corrector_time_s,r.inner_time_s,r.dA_norm,r.dq_norm,r.df_norm,r.divergence_pstar,
            r.corrector_rounds,r.gravA_witness,r.gravF_witness,r.focal_witness], ","))
    end
end
open(joinpath(OUTDIR, "melitz_aq_phase6_gamma_only_comparison_2026-07-30.csv"), "w") do io
    println(io, "label,step,GT_target,kind,Delta,cert")
    for r in vcat(gamma_upper, gamma_lower)
        println(io, join([r.label,r.step,r.GT_target,r.kind,r.Delta,r.cert], ","))
    end
end
println("\nDONE PHASE 6")
