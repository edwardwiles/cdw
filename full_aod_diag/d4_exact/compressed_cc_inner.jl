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
