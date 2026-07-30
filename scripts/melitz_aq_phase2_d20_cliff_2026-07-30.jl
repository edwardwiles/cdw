# Phase 2/3/4 (real D20): test the LFD-preserving constructor at the audited origin-14
# cutoff cliff (docs/melitz_d20_negative_switch_geometry_audit_2026-07-30.md). Governing
# prompt: melitz_joint_Aq_feasibility_preserving_search_2026-07-30.
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
println("Julia threads: ", Threads.nthreads(), "  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads()); flush(stdout)

const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
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

function build_bundle(paramzn::Symbol)
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=paramzn)
    return obj
end

obj = build_bundle(:logcutoff)
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

minus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=-1, n_switches=30, t_max=1.0)
plus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=+1, n_switches=30, t_max=1.0)
println("first minus switch: o=", minus_events[1].o, " d=", minus_events[1].d, " t=", minus_events[1].t, " dir=", minus_events[1].dir)
println("first plus switch: o=", plus_events[1].o, " d=", plus_events[1].d, " t=", plus_events[1].t, " dir=", plus_events[1].dir)
flush(stdout)

function distinct_ts(events, n)
    ts = Float64[]
    for e in events
        (isempty(ts) || e.t > ts[end]*(1+1e-6)) && push!(ts, e.t)
        length(ts) >= n && break
    end
    return ts
end
ts_minus = distinct_ts(minus_events, 6)
ts_plus = distinct_ts(plus_events, 6)
println("minus distinct t (first 6): ", ts_minus)
println("plus distinct t (first 6): ", ts_plus)
flush(stdout)

function classify_summary(r)
    if r isa FiniteSolved
        return (kind="FiniteSolved", Delta=r.Delta, cert=NaN, src="")
    elseif r isa AboveEvaluationCap
        return (kind="AboveEvaluationCap", Delta=NaN, cert=r.certified_lower_bound, src=String(r.source))
    elseif r isa InfiniteDeltaCertified
        return (kind="InfiniteDeltaCertified", Delta=NaN, cert=NaN, src=String(r.kind))
    else
        return (kind="NumericalFailure", Delta=NaN, cert=NaN, src="")
    end
end

function classify_theta_free(theta_free_lc::AbstractVector, obj_lc, ctx_lc)
    session = MelitzInnerSession(obj_lc, ctx_lc, policy_cap)
    return solve_melitz_delta!(session, theta_free_lc, policy_cap)
end

rows = NamedTuple[]
function q_free_free_at(sign::Int, t::Real)
    return q_free_free_anchor .+ sign .* t .* b_q
end

function process_switch(ki::Int, sign::Int, t::Real, ev)
    label_sign = sign > 0 ? "plus" : "minus"
    qff = q_free_free_at(sign, t)

    # --- Endpoint A: pure q, A held at anchor ---
    theta_A = copy(theta_plain0)
    theta_A[1+nA+1:end] .= qff
    rA = classify_theta_free(theta_A, obj, ctx)
    sA = classify_summary(rA)
    @printf("[%s k=%d] Endpoint A (pure q): %s  Delta/cert=%.6g\n", label_sign, ki, sA.kind, isnan(sA.Delta) ? sA.cert : sA.Delta)
    flush(stdout)

    # --- Endpoint B: cellwise-compensated, uncorrected ---
    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    stB = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff; correct=false)
    rB = classify_theta_free(stB.theta_free, obj, ctx)
    sB = classify_summary(rB)
    @printf("[%s k=%d] Endpoint B (cellwise, uncorrected): max|trade_res|=%.3e focal_res=%.3e gravA=%.3e gravF=%.3e feasible=%s reopt=%s Delta/cert=%.6g\n",
        label_sign, ki, maximum(abs.(stB.trade_residuals)), stB.focal_residual, stB.gravity_A_residual,
        stB.gravity_f_residual, stB.feasible, sB.kind, isnan(sB.Delta) ? sB.cert : sB.Delta)
    flush(stdout)

    # --- Endpoint C: fully corrected ---
    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    tC0 = time()
    stC = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff;
        correct=true, max_corrector_rounds=8)
    tC_corr = time() - tC0
    rC = classify_theta_free(stC.theta_free, obj, ctx)
    sC = classify_summary(rC)
    @printf("[%s k=%d] Endpoint C (corrected, %.2fs): max|trade_res|=%.3e focal_res=%.3e gravA=%.3e gravF=%.3e feasible=%s reopt=%s Delta/cert=%.6g  n_switches_A_recovered=%d\n",
        label_sign, ki, tC_corr, maximum(abs.(stC.trade_residuals)), stC.focal_residual, stC.gravity_A_residual,
        stC.gravity_f_residual, stC.feasible, sC.kind, isnan(sC.Delta) ? sC.cert : sC.Delta,
        length(stC.corrector_trace))
    flush(stdout)

    dA_norm = norm(vec(stC.A) .- vec(A_anchor))
    dq_norm = norm(vec(stC.q) .- vec(q_anchor))
    df_norm = norm(vec(stC.f) .- vec(f_anchor))
    divp = stC.divergence_pstar

    melitz_update_operator_at_theta!(obj.op, theta0, ctx)  # restore between switches

    push!(rows, (side=label_sign, k=ki, t=t, switch_o=ev.o, switch_d=ev.d, switch_dir=String(ev.dir),
        A_kind=sA.kind, A_Delta=sA.Delta, A_cert=sA.cert,
        B_max_trade_res=maximum(abs.(stB.trade_residuals)), B_focal_res=stB.focal_residual,
        B_gravA=stB.gravity_A_residual, B_gravF=stB.gravity_f_residual, B_feasible=stB.feasible,
        B_kind=sB.kind, B_Delta=sB.Delta, B_cert=sB.cert,
        C_max_trade_res=maximum(abs.(stC.trade_residuals)), C_focal_res=stC.focal_residual,
        C_gravA=stC.gravity_A_residual, C_gravF=stC.gravity_f_residual, C_feasible=stC.feasible,
        C_kind=sC.kind, C_Delta=sC.Delta, C_cert=sC.cert, C_corrector_rounds=length(stC.corrector_trace),
        C_corrector_time_s=tC_corr, dA_norm=dA_norm, dq_norm=dq_norm, df_norm=df_norm,
        divergence_pstar=divp))
end

println("\n" * "="^100); println("MINUS side (first 5 switches)"); println("="^100); flush(stdout)
for (ki, tk) in enumerate(ts_minus[1:5])
    gap_hi = ki < length(ts_minus) ? ts_minus[ki+1]-tk : tk
    gap_lo = ki==1 ? tk : tk-ts_minus[ki-1]
    eps = max(0.15*min(gap_lo,gap_hi), tk*1e-8)
    process_switch(ki, -1, tk+eps, minus_events[ki])
end

println("\n" * "="^100); println("PLUS side (first 5 switches)"); println("="^100); flush(stdout)
for (ki, tk) in enumerate(ts_plus[1:5])
    gap_hi = ki < length(ts_plus) ? ts_plus[ki+1]-tk : tk
    gap_lo = ki==1 ? tk : tk-ts_plus[ki-1]
    eps = max(0.15*min(gap_lo,gap_hi), tk*1e-8)
    process_switch(ki, +1, tk+eps, plus_events[ki])
end

open(joinpath(OUTDIR, "melitz_aq_phase2_d20_cliff_2026-07-30.csv"), "w") do io
    println(io, "side,k,t,switch_o,switch_d,switch_dir,A_kind,A_Delta,A_cert,B_max_trade_res,B_focal_res,B_gravA,B_gravF,B_feasible,B_kind,B_Delta,B_cert,C_max_trade_res,C_focal_res,C_gravA,C_gravF,C_feasible,C_kind,C_Delta,C_cert,C_corrector_rounds,C_corrector_time_s,dA_norm,dq_norm,df_norm,divergence_pstar")
    for r in rows
        println(io, join([r.side,r.k,r.t,r.switch_o,r.switch_d,r.switch_dir,r.A_kind,r.A_Delta,r.A_cert,
            r.B_max_trade_res,r.B_focal_res,r.B_gravA,r.B_gravF,r.B_feasible,r.B_kind,r.B_Delta,r.B_cert,
            r.C_max_trade_res,r.C_focal_res,r.C_gravA,r.C_gravF,r.C_feasible,r.C_kind,r.C_Delta,r.C_cert,
            r.C_corrector_rounds,r.C_corrector_time_s,r.dA_norm,r.dq_norm,r.df_norm,r.divergence_pstar], ","))
    end
end

# Detail on the SPECIFIC audited destination-16/19 switch (minus side, k=1: o=14,d=19).
println("\n" * "="^100); println("Destination-16/19 detail at first minus switch"); println("="^100)
ev1 = minus_events[1]
o14, d19 = ev1.o, ev1.d
qff_below = q_free_free_at(-1, ts_minus[1] - 0.15*ts_minus[1])
qff_above = q_free_free_at(-1, ts_minus[1] + 0.15*(ts_minus[2]-ts_minus[1]))
suffix_o14 = melitz_origin_suffix_tail(sorted_ctx, o14, p_star)
for (label, qff) in (("below", qff_below), ("above", qff_above))
    theta_trial = vcat(g_anchor, theta_plain0[2:1+nA], qff)
    _, _, _, _, q_trial = expand_free_theta_logcutoff(theta_trial, ctx)
    H16_before = melitz_T_od(q_anchor[o14, 16], o14, sorted_ctx, suffix_o14)
    T16_trial = melitz_T_od(q_trial[o14, 16], o14, sorted_ctx, suffix_o14)
    T19_trial = melitz_T_od(q_trial[o14, d19], o14, sorted_ctx, suffix_o14)
    @printf("label=%-6s q[14,16]=%.6f q[14,19]=%.6f  T16(p*)=%.10f (anchor H16=%.10f)  T19(p*)=%.10f\n",
        label, q_trial[o14,16], q_trial[o14,d19], T16_trial, H16_before, T19_trial)
end

serialize(joinpath(SCRATCH, "phase2_d20_state.jls"), (theta0=theta0, x0=x0, p_star=p_star, calib=calib,
    focal=focal, D=D, b_q=b_q, stage=stage, ts_minus=ts_minus, ts_plus=ts_plus,
    minus_events=minus_events, plus_events=plus_events, Delta0=lfd0.Delta))
println("\nDONE PHASE 2/3/4 (D20 cliff)")
