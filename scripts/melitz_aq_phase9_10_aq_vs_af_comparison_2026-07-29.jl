# A_q separation and gradient diagnostics session (2026-07-29), Phase 9/10: matched
# economic-direction comparison of the (A,q) coordinate system (exact A-block gradient +
# fixed-dual q-block secants) against the current production (A,f) coordinate system
# (finite-bandwidth direct-sorted gradient), plus a fixed-dual intensive/switching
# decomposition for the extensive/mixed cases. D=4 only (disclosed scope reduction, session
# time budget) -- three matched directions: pure intensive (A move, q fixed), pure extensive
# (q move, A fixed), mixed (both).

const REPO = joinpath(@__DIR__, "..")
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Random, LinearAlgebra, Printf, DelimitedFiles

const OUTDIR = joinpath(REPO, "docs", "key_results")
isdir(OUTDIR) || mkpath(OUTDIR)

function fixed_dual_delta!(obj::MelitzCCBundle, theta::AbstractVector, x0::AbstractVector, ctx)
    melitz_update_operator_at_theta!(obj.op, theta, ctx)
    return -obj(x0)
end
function reoptimized_delta!(obj::MelitzCCBundle, theta::AbstractVector)
    obj.use_cached_x = false; obj.x .= NaN
    lfd = melitz_recover_lfd(obj, theta)
    return lfd.Delta, lfd.nStatus, lfd.lfd_ok
end
"Fixed-dual central secant for one coordinate of theta (raw step h)."
function coord_secant(obj, theta0, k, h, x0, ctx)
    tp = copy(theta0); tp[k] += h
    tm = copy(theta0); tm[k] -= h
    Bp = fixed_dual_delta!(obj, tp, x0, ctx)
    Bm = fixed_dual_delta!(obj, tm, x0, ctx)
    melitz_update_operator_at_theta!(obj.op, theta0, ctx)
    return (Bp - Bm) / (2h)
end

D, seed, W = 4, 29, 20_000
data = generate_fake_melitz_data(; D=D, seed=seed, W=W)

# Two ctx/obj pairs sharing the SAME economic primitives (same data), different outer coord.
obj_q, theta0_q = build_melitz_psi_bundle(data; outer_parameterization=:logcutoff,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx_q = obj_q.γ
obj_f, theta0_f = build_melitz_psi_bundle(data; outer_parameterization=:logf,
    policy=CappedEvaluation(10.0), backend=:matrix_free, forbid_dense_fallback=true)
ctx_f = obj_f.γ
nA = D^2 - 1
n = length(theta0_q)

obj_q.use_cached_x = false; obj_q.x .= NaN
lfd0q = melitz_recover_lfd(obj_q, theta0_q)
@assert lfd0q.lfd_ok
x0q = copy(lfd0q.dual_x)
obj_f.use_cached_x = false; obj_f.x .= NaN
lfd0f = melitz_recover_lfd(obj_f, theta0_f)
@assert lfd0f.lfd_ok
x0f = copy(lfd0f.dual_x)
Delta0 = lfd0q.Delta
@printf("Delta0 (:logcutoff)=%.6e  Delta0 (:logf)=%.6e  (should match: same economic point)\n", lfd0q.Delta, lfd0f.Delta)
flush(stdout)

A0, f0, gpj0, fjj0 = melitz_expand_theta(theta0_q, ctx_q)
state0 = MelitzExpandedState(D); state0.A .= A0; state0.f .= f0; state0.gamma_prime_j = gpj0; state0.f_jj = fjj0
melitz_update_operator_at_theta!(obj_q.op, theta0_q, ctx_q)
exact_free, _ = melitz_exact_a_gradient(obj_q, x0q, state0, ctx_q)

# (:logf) production gradient at theta0_f, matched bandwidth.
h_prod = 1e-4
gfun = make_melitz_gradient_delta_direct_sorted_serial(h_prod)
grad_f = zeros(n)
gfun(grad_f, theta0_f, ctx_f, obj_f, x0f)
grad_f ./= 1e10   # gfun's own convention: d(1e10*Delta)/dtheta (matches Object A in the 2026-07-27 script)

# q-block fixed-dual secants (bandwidth matched to the step sizes used below), (A,q) system.
h_q = 1e-4
nq = n - 1 - nA
q_secants = zeros(nq)
for m in 1:nq
    q_secants[m] = coord_secant(obj_q, theta0_q, 1 + nA + m, h_q, x0q, ctx_q)
end

function displaced_point(theta_q_new, ctx_q)
    A, f, gpj, fjj = melitz_expand_theta(theta_q_new, ctx_q)
    return A, f, gpj
end

function to_logf_theta(A, f, gpj, ctx_f)
    p = MelitzPrimitives(ctx_f.D, ctx_f.sigma, ctx_f.theta_star, ctx_f.target_country, ctx_f.tau, ctx_f.w, A, f, gpj)
    return melitz_reduce_theta(p, ctx_f)
end

function run_case(label, theta_q_new, q_pred_terms)
    A, f, gpj = displaced_point(theta_q_new, ctx_q)
    theta_f_new = to_logf_theta(A, f, gpj, ctx_f)

    Delta_actual, nS, ok = reoptimized_delta!(obj_q, theta_q_new)
    dDelta_actual = Delta_actual - Delta0

    # (A,q) system linear prediction: exact A-block dot + q-block secant dot (both w.r.t.
    # theta0_q's own free-coordinate displacement).
    dtheta_q = theta_q_new .- theta0_q
    dA_free = dtheta_q[2:1+nA]
    dq_free = dtheta_q[2+nA:end]
    pred_aq = dot(exact_free, dA_free) + dot(q_secants, dq_free)

    # (A,f) system linear prediction: production registered gradient dot with the ACTUAL
    # displacement in :logf coordinates (which generally touches BOTH A and f blocks even for
    # a "pure A, fixed-q" economic move).
    dtheta_f = theta_f_new .- theta0_f
    pred_af = dot(grad_f, dtheta_f)

    melitz_update_operator_at_theta!(obj_q.op, theta0_q, ctx_q)

    @printf("[%s] nStatus=%d ok=%s  dDelta_actual=%.6e  pred(A,q)=%.6e (err=%.3e)  pred(A,f)=%.6e (err=%.3e)\n",
        label, nS, ok, dDelta_actual, pred_aq, pred_aq - dDelta_actual, pred_af, pred_af - dDelta_actual)
    flush(stdout)
    return (label=label, nStatus=nS, ok=ok, dDelta_actual=dDelta_actual, pred_aq=pred_aq,
            err_aq=pred_aq - dDelta_actual, pred_af=pred_af, err_af=pred_af - dDelta_actual,
            abs_ratio_aq=abs((pred_aq-dDelta_actual)/max(abs(dDelta_actual),1e-12)),
            abs_ratio_af=abs((pred_af-dDelta_actual)/max(abs(dDelta_actual),1e-12)))
end

rows = NamedTuple[]

# Pure intensive: move one ordinary A free coordinate, hold q fixed.
r_int = 1e-3
theta_int = copy(theta0_q); theta_int[3] += r_int
push!(rows, run_case("pure_intensive", theta_int, nothing))

# Pure extensive: move one ordinary q free coordinate, hold A fixed.
r_ext = 1e-4
theta_ext = copy(theta0_q); theta_ext[2+nA] += r_ext
push!(rows, run_case("pure_extensive", theta_ext, nothing))

# Mixed: move both simultaneously.
theta_mix = copy(theta0_q); theta_mix[3] += r_int; theta_mix[2+nA] += r_ext
push!(rows, run_case("mixed", theta_mix, nothing))

open(joinpath(OUTDIR, "melitz_aq_phase9_aq_vs_af_2026-07-29.csv"), "w") do io
    println(io, "label,nStatus,ok,dDelta_actual,pred_aq,err_aq,pred_af,err_af,abs_ratio_aq,abs_ratio_af")
    for r in rows
        println(io, join([r.label, r.nStatus, r.ok, r.dDelta_actual, r.pred_aq, r.err_aq,
                           r.pred_af, r.err_af, r.abs_ratio_aq, r.abs_ratio_af], ","))
    end
end
println("\n(A,q) vs (A,f) comparison complete. CSV written.")
flush(stdout)
