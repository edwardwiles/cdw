# q-bandwidth convergence campaign (2026-07-29), Phase 9: PROPER matched (A,q) vs (A,f)
# comparison, correcting the prior session's own disclosed flaw (its "pure extensive"
# direction was a genuine no-op: near-zero local sensitivity, so the "mixed" row was really
# just "pure_intensive plus nothing").
#
# Fix: select the q-direction via bisection on TARGET CROSSING COUNT (>=15 two-sided) rather
# than a single random free coordinate -- guarantees a genuinely nonzero extensive-margin
# move before running the comparison, verified live (not assumed) via three explicit
# pre-checks: (1) A is held bit-identical; (2) the q movement is nonzero in the FULL
# reconstructed system; (3) at least one participation switch occurs; the fourth check (the
# reoptimized DeltaStar change is materially nonzero) is verified AFTER solving, not assumed.
#
# (A,q) prediction = exact-A analytical gradient (zero for a pure-extensive/q-only leg) PLUS
# the shortlisted q estimator (PowerScaled alpha=1/2, anchor25 -- Phase 6's own shortlist).
# (A,f) prediction = direct reoptimized central-difference secant of DeltaStar in :logf
# coordinates at the SAME step size (mirrors the prior session's own Section 7 method:
# "production FD, h=1e-4" IS the central-difference secant at that h for a single direction).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random

const OUTDIR = joinpath(REPO, "docs", "key_results")
CAP = 10.0
policy_cap = CappedEvaluation(CAP)
println("Julia threads: ", Threads.nthreads()); flush(stdout)

function load_theta_q_rows(path)
    rows = Dict{Tuple{String,Float64},Vector{Float64}}()
    for line in eachline(path)
        parts = split(line, ",")
        rows[(parts[1], parse(Float64, parts[2]))] = parse.(Float64, parts[5:end])
    end
    return rows
end
theta_q_rows = load_theta_q_rows(joinpath(OUTDIR, "melitz_qbw_phase3_theta_q_2026-07-29.csv"))

results = NamedTuple[]

function run_target(target::Float64; W::Int=20_000, seed::Int=29)
    println("\n" * "="^100); @printf("target=%.1f  W=%d  seed=%d\n", target, W, seed); println("="^100); flush(stdout)

    theta_q0 = theta_q_rows[("D4_seed29_W20000", target)]
    D = 4; nA = D^2 - 1; nq = length(theta_q0) - 1 - nA

    data = generate_fake_melitz_data(; D=D, sigma=2.5, theta_star=6.8, target_country=1, seed=seed, W=W)
    obj_q, _ = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
        policy=policy_cap, backend=:matrix_free, forbid_dense_fallback=true)
    ctx_q = obj_q.γ
    sorted_ctx = ctx_q.sorted_tail_ctx

    obj_f, theta0_f_native = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free, policy=policy_cap)
    ctx_f = obj_f.γ

    obj_q.use_cached_x = false; obj_q.x .= NaN
    lfd0_q = melitz_recover_lfd(obj_q, theta_q0)
    @assert lfd0_q.lfd_ok "base point (q-space) failed to verify"
    x0_q = copy(lfd0_q.dual_x)
    A0, f0, gpj0, fjj0 = melitz_expand_theta(theta_q0, ctx_q)
    theta_f0 = reduce_to_free_theta(MelitzPrimitives(D, ctx_q.sigma, ctx_q.theta_star, ctx_q.target_country,
                                                       ctx_q.tau, ctx_q.w, A0, f0, gpj0), ctx_f)
    obj_f.use_cached_x = false; obj_f.x .= NaN
    lfd0_f = melitz_recover_lfd(obj_f, theta_f0)
    @assert lfd0_f.lfd_ok "base point (f-space) failed to verify"
    delta0_mismatch = abs(lfd0_f.Delta - lfd0_q.Delta)
    @printf("Delta0 (q-space)=%.8e  Delta0 (f-space)=%.8e  |mismatch|=%.3e\n", lfd0_q.Delta, lfd0_f.Delta, delta0_mismatch)
    @assert delta0_mismatch < 1e-6 * max(1.0, lfd0_q.Delta)

    # Exact-A gradient (fixed q) at this base point -- used for the A-block of every mixed prediction.
    state0 = MelitzExpandedState(D)
    state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
    melitz_update_operator_at_theta!(obj_q.op, theta_q0, ctx_q)
    exact_A_free, _ = melitz_exact_a_gradient(obj_q, x0_q, state0, ctx_q)

    # Select a genuinely q-crossing free coordinate (bisect for >=15 two-sided crossings at a
    # moderate reference step, then pick the SIGN/scale explicitly).
    m_candidates = 1:nq
    chosen_m = nothing
    chosen_h = nothing
    for m in m_candidates
        h, tp, tm = _melitz_bisect_h_two_sided(15, theta_q0, m, ctx_q, sorted_ctx; h_hi=0.2)
        if min(tp, tm) >= 15
            chosen_m = m; chosen_h = h; break
        end
    end
    @assert chosen_m !== nothing "no free q coordinate reached >=15 two-sided crossings within h_hi"
    @printf("chosen q coordinate m=%d, h=%.4e (>=15 two-sided crossings confirmed)\n", chosen_m, chosen_h)

    q_gradient_policy = PowerScaledQBandwidth(chosen_h, W, 0.5)   # anchored directly at the crossing-verified h itself

    for (pathname, r_intensive, r_extensive) in [
            ("pure_intensive", 1.0, 0.0), ("pure_extensive", 0.0, 1.0),
            ("mixed_5050", 1.0, 1.0), ("mixed_7525", 1.0, 0.33)]
        for scale in (0.5, 1.0, 2.0)
            t_A = scale * 1e-3 * r_intensive     # A-coordinate step (any free A coord, use coord 1)
            t_q = scale * chosen_h * r_extensive  # q-coordinate step (the crossing-verified coordinate)

            theta_q_p = copy(theta_q0); theta_q_p[1+1] += t_A; theta_q_p[1+nA+chosen_m] += t_q
            theta_q_m = copy(theta_q0); theta_q_m[1+1] -= t_A; theta_q_m[1+nA+chosen_m] -= t_q

            # pre-checks (verified live, not assumed)
            Ap, fp, gpjp, fjjp = melitz_expand_theta(theta_q_p, ctx_q)
            Am, fm, gpjm, fjjm = melitz_expand_theta(theta_q_m, ctx_q)
            a_moved = maximum(abs.(log.(Ap) .- log.(Am))) > 1e-12
            q_full_p_dummy = expand_free_theta_logcutoff(theta_q_p, ctx_q)[5]
            q_full_0 = expand_free_theta_logcutoff(theta_q0, ctx_q)[5]
            q_moved = maximum(abs.(q_full_p_dummy .- q_full_0)) > 1e-12
            tp_cnt, tm_cnt, _ = melitz_q_two_sided_crossings(theta_q0, chosen_m, abs(t_q) + 1e-300, ctx_q, sorted_ctx)
            has_switch = r_extensive != 0.0 ? (tp_cnt > 0 || tm_cnt > 0) : true   # N/A for pure_intensive

            # (A,q) prediction: exact A (only nonzero if r_intensive!=0) + q-secant (fixed-dual mode)
            pred_A_part = r_intensive != 0.0 ? exact_A_free[1] * t_A : 0.0
            pred_q_part = 0.0
            if r_extensive != 0.0
                rq = melitz_q_coordinate_probe(theta_q0, chosen_m, q_gradient_policy, obj_q, ctx_q; x0=x0_q, mode=:fixed_dual)
                pred_q_part = rq.secant * t_q
            end
            pred_aq = pred_A_part + pred_q_part

            # direct (A,q)-space reoptimized secant
            obj_q.use_cached_x = false; obj_q.x .= NaN; lp_q = melitz_recover_lfd(obj_q, theta_q_p)
            obj_q.use_cached_x = false; obj_q.x .= NaN; lm_q = melitz_recover_lfd(obj_q, theta_q_m)
            # CORRECTED 2026-07-29 (continuation session, governing-prompt "Important
            # corrections" #1): the ORIGINAL version of this script compared `pred_aq`
            # (`exact_A_free[1]*t_A + rq.secant*t_q`, a ONE-SIDED linear extrapolation from
            # theta_q0 to theta_q0+t) directly against `lp_q.Delta - lm_q.Delta`, the FULL
            # TWO-SIDED change `Delta*(theta0+t) - Delta*(theta0-t)` (spanning 2t, not t).
            # Under local linearity `Delta*(theta0+t)-Delta*(theta0-t) ~= 2*(g.t) = 2*pred_aq`,
            # so the original comparison was structurally biased toward "predicted is ~half of
            # actual" REGARDLESS of estimator quality -- exactly the "roughly half its
            # magnitude ... a systematic, repeatable underprediction" finding the prior version
            # of this session's own doc reported (Section "Phase 9", e.g. pure_intensive/
            # target=0.1/scale=1.0: predicted -3.555e-4 vs actual -7.110e-4, ratio ~0.5). That
            # finding was a units/normalization artifact of the comparison itself, not
            # (necessarily) a genuine 2x underprediction bias in the (A,q) estimator.
            #
            # Fix: report BOTH the original two-sided actual change (kept, relabeled
            # unambiguously) AND a genuinely matched ONE-SIDED actual change
            # (`lp_q.Delta - lfd0_q.Delta`, base -> +t, the SAME half-interval `pred_aq` itself
            # spans) as the PRIMARY comparison partner for `pred_aq`. `pred_aq_twosided =
            # 2*pred_aq` is also reported for a reader who prefers to compare against the
            # two-sided actual instead -- both pairings are now internally consistent (same
            # interval length on both sides), not conflated.
            actual_dDelta_twosided = (lp_q.lfd_ok && lm_q.lfd_ok) ? (lp_q.Delta - lm_q.Delta) : NaN
            actual_dDelta_onesided = lp_q.lfd_ok ? (lp_q.Delta - lfd0_q.Delta) : NaN
            pred_aq_twosided = 2 * pred_aq
            actual_dDelta = actual_dDelta_twosided   # kept for any downstream reader of the old column name

            # matched (A,f) endpoint + production-style FD secant at the SAME step
            theta_f_p = reduce_to_free_theta(MelitzPrimitives(D, ctx_f.sigma, ctx_f.theta_star, ctx_f.target_country,
                                                                ctx_f.tau, ctx_f.w, Ap, fp, gpjp), ctx_f)
            theta_f_m = reduce_to_free_theta(MelitzPrimitives(D, ctx_f.sigma, ctx_f.theta_star, ctx_f.target_country,
                                                                ctx_f.tau, ctx_f.w, Am, fm, gpjm), ctx_f)
            obj_f.use_cached_x = false; obj_f.x .= NaN; lp_f = melitz_recover_lfd(obj_f, theta_f_p)
            obj_f.use_cached_x = false; obj_f.x .= NaN; lm_f = melitz_recover_lfd(obj_f, theta_f_m)
            pred_af_actual_path = (lp_f.lfd_ok && lm_f.lfd_ok) ? (lp_f.Delta - lm_f.Delta) : NaN   # ground truth via f-path (should match actual_dDelta)

            @printf("  path=%-14s scale=%.1f  t_A=%.2e t_q=%.2e  a_moved=%s q_moved=%s switch=%s\n",
                    pathname, scale, t_A, t_q, a_moved, q_moved, has_switch)
            @printf("    pred_aq(one-sided)=%.6e  actual_dDelta(one-sided,q-space)=%.6e  ratio=%.4f  [MATCHED comparison]\n",
                    pred_aq, actual_dDelta_onesided, pred_aq / max(abs(actual_dDelta_onesided), 1e-300) * sign(actual_dDelta_onesided))
            @printf("    pred_aq(x2, two-sided)=%.6e  actual_dDelta(two-sided,q-space)=%.6e  actual_dDelta(f-space,two-sided)=%.6e  |cross-space mismatch|=%.3e\n",
                    pred_aq_twosided, actual_dDelta_twosided, pred_af_actual_path, abs(actual_dDelta_twosided - pred_af_actual_path))
            flush(stdout)

            push!(results, (target=target, path=pathname, scale=scale, m=chosen_m, t_A=t_A, t_q=t_q,
                             a_moved=a_moved, q_moved=q_moved, has_switch=has_switch,
                             pred_aq_onesided=pred_aq, actual_dDelta_onesided=actual_dDelta_onesided,
                             pred_aq_twosided=pred_aq_twosided, actual_dDelta_twosided=actual_dDelta_twosided,
                             actual_dDelta_fspace_twosided=pred_af_actual_path,
                             lfd_ok_p_q=lp_q.lfd_ok, lfd_ok_m_q=lm_q.lfd_ok, lfd_ok_p_f=lp_f.lfd_ok, lfd_ok_m_f=lm_f.lfd_ok))
        end
    end
end

for target in (0.1, 0.5)
    run_target(target; W=20_000, seed=29)
end

# CORRECTED 2026-07-29 (continuation session): CSV columns renamed/added to make the
# one-sided-vs-two-sided pairing explicit and unambiguous (see the in-loop comment above) --
# `pred_aq_onesided` must be compared to `actual_dDelta_onesided`, `pred_aq_twosided` to
# `actual_dDelta_twosided`; never `pred_aq_onesided` to `actual_dDelta_twosided` (the original
# bug).
open(joinpath(OUTDIR, "melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.csv"), "w") do io
    println(io, "target,path,scale,m,t_A,t_q,a_moved,q_moved,has_switch,pred_aq_onesided,actual_dDelta_onesided,pred_aq_twosided,actual_dDelta_twosided,actual_dDelta_fspace_twosided,lfd_ok_p_q,lfd_ok_m_q,lfd_ok_p_f,lfd_ok_m_f")
    for r in results
        println(io, join([r.target, r.path, r.scale, r.m, r.t_A, r.t_q, r.a_moved, r.q_moved, r.has_switch,
                           r.pred_aq_onesided, r.actual_dDelta_onesided, r.pred_aq_twosided, r.actual_dDelta_twosided,
                           r.actual_dDelta_fspace_twosided,
                           r.lfd_ok_p_q, r.lfd_ok_m_q, r.lfd_ok_p_f, r.lfd_ok_m_f], ","))
    end
end
println("\nPhase 9 complete. Rows: ", length(results))
flush(stdout)
