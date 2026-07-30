# Phase 5 (real D20): retest the known origin-14 cutoff cliff at the first 10 negative and
# first 10 positive chamber transitions, comparing:
#   A. pure q, fixed A
#   B. prior cellwise compensation (uncorrected, Step 1B alone)
#   C. prior two-lever discrete corrector (`lfd_preserving_state.jl`, 523c4af)
#   D. new hybrid chamber corrector (`hybrid_chamber_corrector.jl`)
# Governing prompt: melitz_hybrid_chamber_lfd_corrector_2026-07-30, Phase 5.
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
println("focal country index = ", focal); flush(stdout)

obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
    inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
    forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
ctx = obj.γ
D = ctx.D; nA = D^2-1; nq = D^2-2
sorted_ctx = ctx.sorted_tail_ctx
println("D=", D, "  nA=", nA, "  nq=", nq); flush(stdout)

obj.use_cached_x = false; obj.x .= NaN
t0 = time()
lfd0 = melitz_recover_lfd(obj, theta0)
@printf("anchor solve: %.2fs  Delta0=%.10f  lfd_ok=%s\n", time()-t0, lfd0.Delta, lfd0.lfd_ok)
@assert lfd0.lfd_ok
@assert isapprox(lfd0.Delta, 0.483276; atol=1e-4)
x0 = copy(lfd0.dual_x)
p_star = copy(lfd0.weights)
flush(stdout)

bwpolicy = PowerScaledQBandwidth(1e-3, 80_000, 0.5)
stage = melitz_build_reduced_q_stage(theta0, x0, ctx, obj, 1; bandwidth_policy=bwpolicy, target_switches=100)
@assert stage !== nothing
b_q = stage.q_basis_free
println("|b_q|=", norm(b_q)); flush(stdout)

theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
A_anchor, f_anchor, gpj_anchor, _, q_anchor = expand_free_theta_logcutoff(theta_plain0, ctx)
g_anchor = theta_plain0[1]
q_free_free_anchor = theta_plain0[2+nA:end]

N_SWITCHES = 10
minus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=-1, n_switches=60, t_max=1.0)
plus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=+1, n_switches=60, t_max=1.0)

function distinct_ts(events, n)
    ts = Float64[]
    for e in events
        (isempty(ts) || e.t > ts[end]*(1+1e-6)) && push!(ts, e.t)
        length(ts) >= n && break
    end
    return ts
end
ts_minus = distinct_ts(minus_events, N_SWITCHES)
ts_plus = distinct_ts(plus_events, N_SWITCHES)
println("minus distinct t (first $N_SWITCHES): ", ts_minus)
println("plus distinct t (first $N_SWITCHES): ", ts_plus)
flush(stdout)

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
function classify_theta_free(theta_free_lc::AbstractVector, obj_lc, ctx_lc)
    session = MelitzInnerSession(obj_lc, ctx_lc, policy_cap)
    return solve_melitz_delta!(session, theta_free_lc, policy_cap)
end

rows = NamedTuple[]
q_free_free_at(sign::Int, t::Real) = q_free_free_anchor .+ sign .* t .* b_q

function process_switch(ki::Int, sign::Int, t::Real, ev)
    label_sign = sign > 0 ? "plus" : "minus"
    qff = q_free_free_at(sign, t)

    # --- A: pure q, fixed A ---
    theta_A = copy(theta_plain0)
    theta_A[1+nA+1:end] .= qff
    rA = classify_theta_free(theta_A, obj, ctx)
    sA = classify_summary(rA)

    # --- B: cellwise-compensated, uncorrected ---
    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    stB = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff; correct=false)
    rB = classify_theta_free(stB.theta_free, obj, ctx)
    sB = classify_summary(rB)

    # --- C: prior two-lever discrete corrector ---
    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    tC0 = time()
    stC = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff;
        correct=true, max_corrector_rounds=8)
    tC_corr = time() - tC0
    rC = classify_theta_free(stC.theta_free, obj, ctx)
    sC = classify_summary(rC)

    # --- D: new hybrid chamber corrector ---
    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    tD0 = time()
    stD = melitz_construct_hybrid_chamber_state(theta0, p_star, ctx, obj, g_anchor, qff;
        gravity_tol=1e-10, focal_tol=1e-9, moment_tol=1e-9, max_macro_rounds=5,
        discrete_max_depth=3, discrete_beam_width=100, discrete_lever_pool_size=60,
        discrete_half_window=100, discrete_max_candidates=2000, continuous_max_iters=50)
    tD_corr = time() - tD0
    rD = classify_theta_free(stD.theta_free, obj, ctx)
    sD = classify_summary(rD)

    @printf("[%s k=%d] switch=(o=%d,d=%d,%s)  A(pure-q)=%s\n", label_sign, ki, ev.o, ev.d, String(ev.dir), sA.kind)
    @printf("[%s k=%d] B(uncorrected): feasible=%s gravA=%.3e gravF=%.3e focal=%.3e kind=%s\n",
        label_sign, ki, stB.feasible, stB.gravity_A_residual, stB.gravity_f_residual, stB.focal_residual, sB.kind)
    @printf("[%s k=%d] C(2-lever,%.2fs): feasible=%s gravA=%.3e gravF=%.3e focal=%.3e kind=%s Delta/cert=%.6g\n",
        label_sign, ki, tC_corr, stC.feasible, stC.gravity_A_residual, stC.gravity_f_residual, stC.focal_residual,
        sC.kind, isnan(sC.Delta) ? sC.cert : sC.Delta)
    @printf("[%s k=%d] D(hybrid,%.2fs): feasible=%s gravA=%.3e gravF=%.3e focal=%.3e kind=%s Delta/cert=%.6g  disc_cand=%d cont_it=%d cont_status=%s macro=%d rt_exact=%s\n",
        label_sign, ki, tD_corr, stD.feasible, stD.gravity_A_residual, stD.gravity_f_residual, stD.focal_residual,
        sD.kind, isnan(sD.Delta) ? sD.cert : sD.Delta, stD.discrete_candidates_examined, stD.continuous_iters,
        stD.continuous_status, stD.macro_rounds, stD.roundtrip_exact)
    flush(stdout)

    dA_norm_C = norm(vec(stC.A) .- vec(A_anchor)); dA_norm_D = norm(vec(stD.A) .- vec(A_anchor))
    dq_norm_C = norm(vec(stC.q) .- vec(q_anchor)); dq_norm_D = norm(vec(stD.q) .- vec(q_anchor))
    df_norm_C = norm(vec(stC.f) .- vec(f_anchor)); df_norm_D = norm(vec(stD.f) .- vec(f_anchor))
    divp = stD.divergence_pstar

    melitz_update_operator_at_theta!(obj.op, theta0, ctx)

    push!(rows, (side=label_sign, k=ki, t=t, switch_o=ev.o, switch_d=ev.d, switch_dir=String(ev.dir),
        A_kind=sA.kind,
        B_feasible=stB.feasible, B_gravA=stB.gravity_A_residual, B_gravF=stB.gravity_f_residual,
        B_focal=stB.focal_residual, B_kind=sB.kind,
        C_feasible=stC.feasible, C_gravA=stC.gravity_A_residual, C_gravF=stC.gravity_f_residual,
        C_focal=stC.focal_residual, C_kind=sC.kind, C_Delta=sC.Delta, C_cert=sC.cert, C_time_s=tC_corr,
        D_feasible=stD.feasible, D_gravA=stD.gravity_A_residual, D_gravF=stD.gravity_f_residual,
        D_focal=stD.focal_residual, D_kind=sD.kind, D_Delta=sD.Delta, D_cert=sD.cert, D_time_s=tD_corr,
        D_disc_cand=stD.discrete_candidates_examined, D_cont_iters=stD.continuous_iters,
        D_cont_status=String(stD.continuous_status), D_macro=stD.macro_rounds, D_rt_exact=stD.roundtrip_exact,
        dA_norm_C=dA_norm_C, dA_norm_D=dA_norm_D, dq_norm_C=dq_norm_C, dq_norm_D=dq_norm_D,
        df_norm_C=df_norm_C, df_norm_D=df_norm_D, divergence_pstar=divp))
end

println("\n" * "="^100); println("MINUS side (first $N_SWITCHES switches)"); println("="^100); flush(stdout)
for (ki, tk) in enumerate(ts_minus)
    gap_hi = ki < length(ts_minus) ? ts_minus[ki+1]-tk : tk
    gap_lo = ki==1 ? tk : tk-ts_minus[ki-1]
    eps = max(0.15*min(gap_lo,gap_hi), tk*1e-8)
    process_switch(ki, -1, tk+eps, minus_events[ki])
end

println("\n" * "="^100); println("PLUS side (first $N_SWITCHES switches)"); println("="^100); flush(stdout)
for (ki, tk) in enumerate(ts_plus)
    gap_hi = ki < length(ts_plus) ? ts_plus[ki+1]-tk : tk
    gap_lo = ki==1 ? tk : tk-ts_plus[ki-1]
    eps = max(0.15*min(gap_lo,gap_hi), tk*1e-8)
    process_switch(ki, +1, tk+eps, plus_events[ki])
end

open(joinpath(OUTDIR, "melitz_hybrid_phase5_d20_cliff_retest_2026-07-30.csv"), "w") do io
    println(io, "side,k,t,switch_o,switch_d,switch_dir,A_kind," *
        "B_feasible,B_gravA,B_gravF,B_focal,B_kind," *
        "C_feasible,C_gravA,C_gravF,C_focal,C_kind,C_Delta,C_cert,C_time_s," *
        "D_feasible,D_gravA,D_gravF,D_focal,D_kind,D_Delta,D_cert,D_time_s,D_disc_cand,D_cont_iters,D_cont_status,D_macro,D_rt_exact," *
        "dA_norm_C,dA_norm_D,dq_norm_C,dq_norm_D,df_norm_C,df_norm_D,divergence_pstar")
    for r in rows
        println(io, join([r.side,r.k,r.t,r.switch_o,r.switch_d,r.switch_dir,r.A_kind,
            r.B_feasible,r.B_gravA,r.B_gravF,r.B_focal,r.B_kind,
            r.C_feasible,r.C_gravA,r.C_gravF,r.C_focal,r.C_kind,r.C_Delta,r.C_cert,r.C_time_s,
            r.D_feasible,r.D_gravA,r.D_gravF,r.D_focal,r.D_kind,r.D_Delta,r.D_cert,r.D_time_s,r.D_disc_cand,r.D_cont_iters,r.D_cont_status,r.D_macro,r.D_rt_exact,
            r.dA_norm_C,r.dA_norm_D,r.dq_norm_C,r.dq_norm_D,r.df_norm_C,r.df_norm_D,r.divergence_pstar], ","))
    end
end

println("\nDONE PHASE 5 (D20 cliff retest, A/B/C/D comparison)")
