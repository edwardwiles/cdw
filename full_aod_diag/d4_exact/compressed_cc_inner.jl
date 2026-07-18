# ============================================================================
# Compressed CC-inner dual bundle: objective, gradient, and Hessian-vector
# products for the fixed-moment inner dual, computed ENTIRELY from the
# compressed winner-form representation (compressed_moments.jl) -- the dense
# W x D^2 factual moment matrix is never materialized.
#
# ADDITIVE / DIAGNOSTIC. Does not modify cc_algo/PsiObjectiveBundle.jl or any
# production inner solver. Mirrors PsiObjectiveBundleImplicit's math exactly
# (verified against its body in cc_algo/PsiObjectiveBundle.jl:181-210):
#
#   x = (ζ, λ),  λ ∈ R^{oci-1}       (H[:,1]=K, H[:,2]=1, H[:,3:1+oci]=G[:,1:oci-1])
#   q_s   = -ζ - λ'G_{s,1:oci-1}                         (= obj.arg0)
#   f     = mean_s Ψ(q_s) + ζ
#   g_ζ   = 1 - mean_s Ψ'(q_s)
#   g_λ   = -(1/M) Σ_s Ψ'(q_s) G_{s,·}                    (transpose contraction)
#   Hess (ζ,λ)-block, directional along p=(p_ζ,p_λ):
#     let u_s = p_ζ + p_λ'G_{s,·},  r_s = Ψ''(q_s) u_s
#     (Hp)_ζ = (1/M) Σ_s r_s
#     (Hp)_λ = (1/M) Σ_s r_s G_{s,·}                      (transpose contraction)
#
# The forward contraction (λ'G_s, p_λ'G_s) is compressed_dual_contraction.
# The transpose contraction (Σ_s w_s G_{s,j}) is compressed_transpose_contraction
# below: ONE O(W·D) pass accumulating winner-origin bucket totals B[o,d] and a
# destination/global total T, then an O(D^2) target-share correction per column
# -- exactly the accumulation the task brief describes. No dense G, no dense
# Hessian.
# ============================================================================

"""
    compressed_transpose_contraction(weights, cf) -> Vector{oci-1}

Exact `v[j] = Σ_s weights[s] · G_{s,j}` for every inner-dual column j, from the
compressed representation. O(W·D) (winner-bucket accumulation) + O(D^2).
"""
function compressed_transpose_contraction(weights::AbstractVector, cf::CompressedFactual)
    D = cf.D; W = cf.W; ncol = cf.oci - 1
    length(weights) == W || error("weights length $(length(weights)) != W=$W")
    B = zeros(D, D)          # B[o,d] = Σ_{s: winner_sd=o} SW_s·weights_s·v_{s,d}
    T = 0.0                  # T = Σ_s SW_s·weights_s
    Bcf = 0.0                # counterfactual-column bucket
    @inbounds for s in 1:W
        ws = cf.SW[s] * weights[s]
        T += ws
        for d in 1:D
            B[cf.winner[s, d], d] += ws * cf.wval[s, d]
        end
        if cf.cf_col > 0
            Bcf += ws * cf.cf_raw[s]
        end
    end
    v = zeros(ncol)
    @inbounds for d in 1:D, o in 1:D
        j = d + (o - 1) * D
        # Σ_s w_s G_{s,j} = nrm·gdiv·(B[o,d] − P_od·denom_d·T) − nrm·usePMM·PMM_j·T
        v[j] = cf.nrm[j] * cf.gdiv[j] * (B[o, d] - cf.Pmat[o, d] * cf.denom[d] * T) -
               cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    if cf.cf_col > 0
        j = cf.cf_col
        v[j] = cf.nrm[j] * cf.gdiv[j] * Bcf - cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    return v
end

"""
    compressed_cc_value_grad(ζ, λ, cf; Psi!, dPsi!) -> (f, g_ζ, g_λ)

CC-inner dual objective + gradient w.r.t. (ζ, λ) from the compressed moments.
`Psi!`, `dPsi!` are the bundle's Ψ / Ψ' (pass ctx.obj.Psi!, ctx.obj.dPsi!).
Matches PsiObjectiveBundleImplicit's f and g exactly (M = W normalization).
"""
function compressed_cc_value_grad(ζ::Real, λ::AbstractVector, cf::CompressedFactual;
                                  Psi!, dPsi!)
    W = cf.W; M = W
    contr = compressed_dual_contraction(λ, cf)          # λ'G_s
    q = similar(contr)
    @inbounds @. q = -ζ - contr
    Psq = similar(q); Psi!(Psq, q)
    dPsq = similar(q); dPsi!(dPsq, q)
    f = sum(Psq) / M + ζ
    g_ζ = 1.0 - sum(dPsq) / M
    g_λ = compressed_transpose_contraction(dPsq, cf)
    @. g_λ = -(1.0 / M) * g_λ
    return f, g_ζ, g_λ, q, dPsq
end

"""
    compressed_cc_hvp(q, p_ζ, p_λ, cf; ddPsi!) -> (Hp_ζ, Hp_λ)

Exact Hessian-vector product of the (ζ,λ)-block along p=(p_ζ,p_λ), from the
compressed moments. `q` is the arg0 vector returned by compressed_cc_value_grad
(so Ψ'' is evaluated at the same point). O(W·D), no dense Hessian.
"""
function compressed_cc_hvp(q::AbstractVector, p_ζ::Real, p_λ::AbstractVector, cf::CompressedFactual;
                           ddPsi!)
    W = cf.W; M = W
    cpλ = compressed_dual_contraction(p_λ, cf)          # p_λ'G_s
    ddPsq = similar(q); ddPsi!(ddPsq, q)
    r = similar(q)
    @inbounds @. r = ddPsq * (p_ζ + cpλ)
    Hp_ζ = sum(r) / M
    Hp_λ = compressed_transpose_contraction(r, cf)
    @. Hp_λ = (1.0 / M) * Hp_λ
    return Hp_ζ, Hp_λ
end

# ============================================================================
# HESSIAN-CALLBACK ADAPTER DECISION (continuation 8, live-integration).
#
# KNITRO's registered inner-loop Hessian callback in this codebase is DENSE,
# not HVP-style: `ek_inner.opt` sets `hessopt exact` (=1), and
# `oracle_fast.jl::inner_loop_KNITRO_profiled` registers it via
# `KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, ...)` -- KNITRO asks for
# the full packed dense (1+ncol)x(1+ncol) Hessian block every call, not a
# matrix-vector product. `compressed_cc_hvp` above computes exact HVPs in
# O(W*D) each, but building the FULL dense Hessian via (1+ncol) HVP calls
# (one per basis vector) would cost O(ncol*W*D), which at D=4 (ncol=17) is
# NOT cheaper than the dense path's direct O(W*ncol) BLAS `gemm!` (a single,
# highly-optimized multithreaded call in `hessian!`) -- ncol sequential
# scalar-loop HVP calls lose to one BLAS gemm in practice. A truly-compressed
# dense-Hessian callback is therefore not the right target here.
#
# DECISION (matches the task brief's explicitly sanctioned fallback): the
# compressed live integration (`compressed_live.jl`) keeps the Hessian
# callback DENSE -- it materializes `obj.H`'s G columns from the
# already-built `CompressedFactual` via `materialize_dense_factual!` (O(W*D^2)
# writes, but reusing the ALREADY-COMPUTED `cf.winner`/`cf.wval`, so it skips
# the winner-search and the D-1-losers'-sigma-value work the ORIGINAL dense
# `moments!` build would redo) and then calls the UNCHANGED, existing
# `hessian!(h, obj)` (cc_algo/PsiObjectiveBundle.jl) verbatim -- not
# reimplemented, so its correctness is inherited, not re-proven.
#
# REFINEMENT beyond the brief's "once per Hessian call": since theta (hence G)
# is FIXED for the entire inner KNITRO solve (only zeta,lambda vary),
# `compressed_live.jl` materializes LAZILY and ONCE PER INNER SOLVE (cached on
# first Hessian call, reused for every subsequent Hessian call in that same
# solve), not once per call. This is strictly cheaper than "once per Hessian
# call" whenever KNITRO calls the Hessian callback more than once per inner
# solve (common), and never more expensive when it calls it exactly once.
#
# EXACTNESS: this makes the compressed live path's HESSIAN CALLBACK
# bit-identical to the dense path's (same `hessian!` call on an equivalent
# dense G) -- NOT an approximation, just a lazily-cached materialization. The
# genuinely-compressed, no-dense-G-ever path is the OBJECTIVE/GRADIENT (FG)
# callback only (`compressed_cc_value_grad`, used every FG call, of which
# there are typically many more per inner solve than Hessian calls -- see
# `docs/compressed_live_integration_report.md` for the measured FG-vs-Hessian
# call-count ratio and the resulting speedup).
# ============================================================================

"""
    compressed_moment_resid(cf::CompressedFactual, weights) -> Vector{oci-1}

MOMENT-RESIDUAL formula (mean_s w_s*G_s over the inner-dual columns), from
compressed moments only. `weights` is typically `ones(W)` (unweighted mean,
matching `evaluate_fullA_fast`'s `moment_resid` diagnostic) or `m_weights`
(the recovered LFD primal weights, matching its `max_abs_moment_kkt_resid`
diagnostic). This is EXACTLY `compressed_transpose_contraction(weights, cf) /
sum-normalization` depending on which mean convention the caller wants -- see
`compressed_live.jl` callers for the exact normalization used at each call
site (mirrors `oracle_fast.jl`'s own `moment_resid` (divide by W) vs
`max_abs_moment_kkt_resid` (divide by W, using primal weights) conventions).
Provided here as a thin, explicitly-named wrapper around
`compressed_transpose_contraction` (no new math) so a caller doesn't have to
re-derive which contraction primitive is the right one for a residual sum.
"""
compressed_moment_resid(cf::CompressedFactual, weights::AbstractVector) =
    compressed_transpose_contraction(weights, cf)

