# Phase 0 (real D20): reproduce the prior joint-(A,q) session's Phase 2 table using the OLD
# two-lever corrector (`melitz_construct_lfd_preserving_state`, `lfd_preserving_state.jl`,
# committed 523c4af), AND audit -- for every B/C endpoint -- the explicit constructed state
# against the state obtained via reduce_to_free_theta_logcutoff -> expand_free_theta_logcutoff.
# Governing prompt: melitz_hybrid_chamber_lfd_corrector_2026-07-30, Phase 0.
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

minus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=-1, n_switches=30, t_max=1.0)
plus_events = melitz_q_direction_exact_switches(theta_plain0, b_q, ctx, sorted_ctx; sign=+1, n_switches=30, t_max=1.0)

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
q_free_free_at(sign::Int, t::Real) = q_free_free_anchor .+ sign .* t .* b_q

"""
Phase 0 round-trip audit: given an EXPLICIT (A,f,gamma_prime_j) state (e.g. from
`melitz_construct_lfd_preserving_state`), reduce it to free theta and re-expand, then compare
DIRECTLY to the explicit state -- in log space for A/f (scale-appropriate), raw for q.
Returns (max_abs_logA, max_abs_logf, max_abs_q, max_rel_logA, max_rel_logf).
"""
function roundtrip_audit(A_explicit, f_explicit, gpj_explicit, q_explicit, ctx)
    theta_free_recovered = reduce_to_free_theta_logcutoff(A_explicit, f_explicit, gpj_explicit, ctx)
    theta_plain_rt = melitz_unpower_theta_free(theta_free_recovered, ctx)
    A_rt, f_rt, gpj_rt, _, q_rt = expand_free_theta_logcutoff(theta_plain_rt, ctx)
    dlogA = abs.(log.(A_rt) .- log.(A_explicit))
    dlogf = abs.(log.(f_rt) .- log.(f_explicit))
    dq = abs.(q_rt .- q_explicit)
    return (max_abs_logA=maximum(dlogA), max_abs_logf=maximum(dlogf), max_abs_q=maximum(dq),
            gpj_diff=abs(gpj_rt - gpj_explicit),
            argmax_logA=Tuple(argmax(dlogA)), argmax_logf=Tuple(argmax(dlogf)))
end

function process_switch(ki::Int, sign::Int, t::Real, ev)
    label_sign = sign > 0 ? "plus" : "minus"
    qff = q_free_free_at(sign, t)

    theta_A = copy(theta_plain0)
    theta_A[1+nA+1:end] .= qff
    rA = classify_theta_free(theta_A, obj, ctx)
    sA = classify_summary(rA)

    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    stB = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff; correct=false)
    rB = classify_theta_free(stB.theta_free, obj, ctx)
    sB = classify_summary(rB)
    rtB = roundtrip_audit(stB.A, stB.f, stB.gamma_prime_j, stB.q, ctx)

    melitz_update_operator_at_Afg!(obj.op, A_anchor, f_anchor, gpj_anchor, ctx)
    tC0 = time()
    stC = melitz_construct_lfd_preserving_state(theta0, p_star, ctx, obj, g_anchor, qff;
        correct=true, max_corrector_rounds=8)
    tC_corr = time() - tC0
    rC = classify_theta_free(stC.theta_free, obj, ctx)
    sC = classify_summary(rC)
    rtC = roundtrip_audit(stC.A, stC.f, stC.gamma_prime_j, stC.q, ctx)

    @printf("[%s k=%d] A(pure-q)=%s B: feasible=%s gravA=%.3e gravF=%.3e kind=%s | ROUNDTRIP B: dlogA=%.3e dlogf=%.3e dq=%.3e\n",
        label_sign, ki, sA.kind, stB.feasible, stB.gravity_A_residual, stB.gravity_f_residual, sB.kind,
        rtB.max_abs_logA, rtB.max_abs_logf, rtB.max_abs_q)
    @printf("[%s k=%d] C(corrected,%.1fs): feasible=%s gravA=%.3e gravF=%.3e kind=%s Delta/cert=%.6g | ROUNDTRIP C: dlogA=%.3e (@%s) dlogf=%.3e (@%s) dq=%.3e\n",
        label_sign, ki, tC_corr, stC.feasible, stC.gravity_A_residual, stC.gravity_f_residual, sC.kind,
        isnan(sC.Delta) ? sC.cert : sC.Delta, rtC.max_abs_logA, rtC.argmax_logA, rtC.max_abs_logf, rtC.argmax_logf, rtC.max_abs_q)
    flush(stdout)

    # Critical correctness check: does the round-trip discrepancy track the gravity residual
    # itself (as the theory predicts: reduce/expand re-imposes gravity EXACTLY via the pivot,
    # so any nonzero gravity residual in the explicit state must show up as a pivot-cell
    # discrepancy of comparable order), or is it decoupled?
    ratio_A = rtC.max_abs_logA / max(abs(stC.gravity_A_residual), 1e-300)
    ratio_F = rtC.max_abs_logf / max(abs(stC.gravity_f_residual), 1e-300)
    @printf("[%s k=%d] discrepancy/residual ratio: A=%.3f  F=%.3f  (order-1 expected if pivot-driven)\n",
        label_sign, ki, ratio_A, ratio_F)
    flush(stdout)

    melitz_update_operator_at_theta!(obj.op, theta0, ctx)

    push!(rows, (side=label_sign, k=ki, t=t, switch_o=ev.o, switch_d=ev.d, switch_dir=String(ev.dir),
        A_kind=sA.kind,
        B_feasible=stB.feasible, B_gravA=stB.gravity_A_residual, B_gravF=stB.gravity_f_residual, B_kind=sB.kind,
        B_rt_dlogA=rtB.max_abs_logA, B_rt_dlogf=rtB.max_abs_logf, B_rt_dq=rtB.max_abs_q,
        C_feasible=stC.feasible, C_gravA=stC.gravity_A_residual, C_gravF=stC.gravity_f_residual, C_kind=sC.kind,
        C_Delta=sC.Delta, C_cert=sC.cert, C_corrector_time_s=tC_corr,
        C_rt_dlogA=rtC.max_abs_logA, C_rt_dlogf=rtC.max_abs_logf, C_rt_dq=rtC.max_abs_q,
        ratio_A=ratio_A, ratio_F=ratio_F))
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

open(joinpath(OUTDIR, "melitz_hybrid_phase0_roundtrip_audit_2026-07-30.csv"), "w") do io
    println(io, "side,k,t,switch_o,switch_d,switch_dir,A_kind,B_feasible,B_gravA,B_gravF,B_kind,B_rt_dlogA,B_rt_dlogf,B_rt_dq,C_feasible,C_gravA,C_gravF,C_kind,C_Delta,C_cert,C_corrector_time_s,C_rt_dlogA,C_rt_dlogf,C_rt_dq,ratio_A,ratio_F")
    for r in rows
        println(io, join([r.side,r.k,r.t,r.switch_o,r.switch_d,r.switch_dir,r.A_kind,
            r.B_feasible,r.B_gravA,r.B_gravF,r.B_kind,r.B_rt_dlogA,r.B_rt_dlogf,r.B_rt_dq,
            r.C_feasible,r.C_gravA,r.C_gravF,r.C_kind,r.C_Delta,r.C_cert,r.C_corrector_time_s,
            r.C_rt_dlogA,r.C_rt_dlogf,r.C_rt_dq,r.ratio_A,r.ratio_F], ","))
    end
end

println("\nDONE PHASE 0 (roundtrip audit of the prior session's two-lever corrector)")
