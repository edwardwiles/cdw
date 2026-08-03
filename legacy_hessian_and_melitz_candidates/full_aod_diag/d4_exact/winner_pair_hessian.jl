# ============================================================================
# diag/compressed-hessian-operator-audit-2026-07-25, Phase 5: exact winner-pair
# construction of the core-factual (H_EE, extended with the ones/zeta column
# and the optional counterfactual price-index column) Hessian block, built
# DIRECTLY from CompressedFactual -- never materializes the dense W x m
# factual moment matrix, and never runs the generic BLAS.gemm! G'SG used by
# the current production `hessian!` (PsiObjectiveBundleImplicitMethodB_fullA.jl)
# / `hessian_cm_structured!`'s H_EE carve-out (cm_hessian_architectures.jl).
#
# DERIVATION (see docs/UNRESTRICTED_WINNER_PAIR_HESSIAN_DERIVATION_2026-07-25.md
# for the full writeup). From compressed_moments.jl's own exact formula
# (materialize_dense_factual!):
#
#   G[w,j] = SW[w] * ( Qtilde[w,j] - pi[j] )        j = 1..oci-1 (bilateral + cf col)
#   Qtilde[w, j(o,slot)] = kappa0[j] * wval[w,slot] * 1{winner[w,slot] == o}   (bilateral)
#   Qtilde[w, cf_col]    = kappa0[cf_col] * cf_raw[w]                          (cf column, dense)
#   pi[j]  = kappa0[j] * Pmat[o,slot] * denom[slot]  +  nrm[j] * usePMM * PMM[j]   (bilateral)
#   pi[cf] = nrm[cf_col] * usePMM * PMM[cf_col]                                    (cf column)
#   kappa0[j] = nrm[j] * gdiv[j]
#
# i.e. G = Q - nu*pi' (rank-one correction), Q[w,j] = SW[w]*Qtilde[w,j], nu = SW.
# (The task brief's schematic "E_F = Q - 1*pi'" assumes uniform sampling
# weight; this repo's SamplingWeights are not assumed uniform, so the
# rank-one correction here is genuinely nu*pi', nu=SW, not literally the
# all-ones vector -- verified against compressed_moments.jl's own formula,
# not assumed.)
#
# The Hessian's extended dual-coefficient matrix is Ghat = [1 | G] (column 0
# is the literal all-ones zeta coefficient, NOT SW-scaled -- distinct object
# from nu). With S = diag(arg2) (curvature weights from ddPsi!):
#
#   H = (1/M) * Ghat' S Ghat
#   H[0,0]     = sum(S) / M
#   H[0,1:m]   = (u - t0*pi) / M,          u[j] = sum_w S[w]*Q[w,j],  t0 = sum_w S[w]*nu[w]
#   H[1:m,1:m] = (Q'SQ - r*pi' - pi*r' + s0*pi*pi') / M,   r = Q'S*nu,  s0 = nu'S*nu
#
# Q'SQ is the only O(W*Ddest^2)-shaped term (vs O(W*m^2) for a dense G'SG):
# for a FIXED destination slot, a draw's winner is a single origin, so
# (Q'SQ)[(o,slot),(o',slot)] is exactly zero for o != o' -- cross terms only
# arise ACROSS DIFFERENT destination slots, which is the textbook winner-pair
# accumulation: for every draw w and pair of active destinations (slot,slot'),
# scatter S[w]*y[w,slot]*y[w,slot'] into Hessian coordinate
# ((winner[w,slot],slot),(winner[w,slot'],slot')), y[w,slot] = Q[w, j(winner[w,slot],slot)].
#
# Everything below is exact (no approximation) -- validated numerically
# against production `hessian!` at D=4 in
# validate_winner_pair_hessian_d4.jl before this file is used for anything
# beyond that validation.
# ============================================================================

"""
    WinnerPairHessCtx

Precomputed, theta-fixed scratch for the winner-pair Hessian. Built once per
outer point (same lifetime as `CompressedFactual`), reused across every
Hessian-callback call within one inner KNITRO solve (only `obj.arg2`, i.e.
S, changes call to call).
"""
struct WinnerPairHessCtx
    D::Int
    Ddest::Int
    W::Int
    ncolI::Int              # = cf.oci - 1 (bilateral + optional cf column)
    has_cf::Bool
    kappa0::Vector{Float64}      # length ncolI
    pi_vec::Vector{Float64}      # length ncolI
    nu::Vector{Float64}          # = SW, length W
    # per-(w,slot) winner-scaled value, ALREADY kappa0-scaled: y[w,slot] = kappa0[j(winner[w,slot],slot)]*wval[w,slot]
    y::Matrix{Float64}           # W x Ddest
    winner::Matrix{Int}          # W x Ddest (alias of cf.winner, kept for convenience)
    cf_raw_scaled::Vector{Float64}  # kappa0[cf_col]*cf_raw[w], length W (empty if !has_cf)
    # ---- persistent per-callback scratch (theta-fixed SIZE, reused every Hessian-callback call
    # within one inner solve -- ONLY obj.arg2/arg0 change call to call, so these buffers are
    # allocated ONCE here, matching the production convention of obj.H_copy/∂∂f_∂∂x
    # (PsiObjectiveBundleImplicitMethodB_fullA.jl) and cctx.Ews/cctx.Hfull
    # (cm_hessian_architectures.jl) -- NOT re-allocated per callback, addendum-driven fix ----
    Snu_buf::Vector{Float64}     # W
    Snu2_buf::Vector{Float64}    # W
    u_buf::Vector{Float64}       # ncolI
    r_buf::Vector{Float64}       # ncolI
    QQ_buf::Matrix{Float64}      # ncolI x ncolI (upper triangle only ever written)
end

"""
    build_winner_pair_ctx(cf::CompressedFactual) -> WinnerPairHessCtx

O(W*Ddest) construction (dominated by computing `y`), theta-fixed -- build
once per outer point/inner solve, exactly analogous to the lazy
`materialize_dense_factual_structured!` cache it replaces.
"""
function build_winner_pair_ctx(cf::CompressedFactual)
    D = cf.D; Ddest = cf.D_dest; W = cf.W; ncolI = cf.oci - 1
    has_cf = cf.cf_col > 0

    kappa0 = Vector{Float64}(undef, ncolI)
    pi_vec = Vector{Float64}(undef, ncolI)
    @inbounds for slot in 1:Ddest, o in 1:D
        j = slot + (o - 1) * Ddest
        k0 = cf.nrm[j] * cf.gdiv[j]
        kappa0[j] = k0
        pi_vec[j] = k0 * cf.Pmat[o, slot] * cf.denom[slot] + cf.nrm[j] * cf.usePMM * cf.PMM[j]
    end

    y = Matrix{Float64}(undef, W, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = cf.winner[w, slot]
            j = slot + (o - 1) * Ddest
            y[w, slot] = kappa0[j] * cf.wval[w, slot]
        end
    end

    cf_raw_scaled = Float64[]
    if has_cf
        jcf = cf.cf_col
        k0cf = cf.nrm[jcf] * cf.gdiv[jcf]
        kappa0[jcf] = k0cf
        pi_vec[jcf] = cf.nrm[jcf] * cf.usePMM * cf.PMM[jcf]
        cf_raw_scaled = k0cf .* cf.cf_raw
    end

    return WinnerPairHessCtx(D, Ddest, W, ncolI, has_cf, kappa0, pi_vec, copy(cf.SW), y, cf.winner, cf_raw_scaled,
        Vector{Float64}(undef, W), Vector{Float64}(undef, W),
        Vector{Float64}(undef, ncolI), Vector{Float64}(undef, ncolI),
        Matrix{Float64}(undef, ncolI, ncolI))
end

"""
    winner_pair_hessian!(h::AbstractVector, obj, wctx::WinnerPairHessCtx)

Fills the packed row-major upper-triangular Hessian `h` (length
n*(n+1)/2, n = obj.outer_constr_index = 1 + wctx.ncolI), IDENTICAL
packing convention to `hessian!`/`hessian_cm_structured!`. Requires
`obj.arg2` to already hold the current curvature weights (i.e.
`ddPsi!(arg2, arg0)` must have been called at the current dual point first
-- same precondition as `hessian_cm_structured!`, see `_archC_prep_for_hessian!`).

O(W*Ddest^2 + W*Ddest) total, O(Ddest^2) extra memory for the dense H_EE
scratch (small: Ddest<=~20 in production) -- never allocates or touches a
W x m array.
"""
function winner_pair_hessian!(h::AbstractVector, obj, wctx::WinnerPairHessCtx)
    # Addendum fairness fix: production's hessian! (cc_algo/PsiObjectiveBundle.jl:598-618) pays
    # the _enter_callback!/_exit_callback! cross-thread reentrancy-guard cost (memory:
    # knitro-concurrency-test-design / AUD-02 cross-thread callback guard) on every call --
    # reproduced here so isolated-callback timing comparisons are apples-to-apples, not an
    # artifact of this candidate skipping a guard the baseline pays for.
    CS._enter_callback!(obj)   # this file loads at Main scope (unlike cc_algo/PsiObjectiveBundle.jl, which is CS.include'd) -- qualify the guard call
    try
    # Same convention as hessian!/hessian_cm_structured!: caller guarantees obj.arg0 reflects the
    # CURRENT dual point (true for the real KNITRO FG-then-H callback order; the D=4 validation
    # script and _archC_prep_for_hessian! both set it explicitly off-KNITRO), this function derives
    # arg2 (curvature weights) itself via ddPsi!, not precomputed by the caller.
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2                    # curvature weights, length W (diag of the weighting matrix)
    M = obj.M
    D = wctx.D; Ddest = wctx.Ddest; W = wctx.W
    ncolI = wctx.ncolI
    n = 1 + ncolI
    kappa0 = wctx.kappa0; pi_vec = wctx.pi_vec; nu = wctx.nu; y = wctx.y

    length(h) == n * (n + 1) ÷ 2 || error("winner_pair_hessian!: length(h)=$(length(h)) != n(n+1)/2 for n=$n")

    # ---- scalar accumulators over draws: S_sum, t0, s0, and per-draw Snu, Snu2 ----
    # (Snu/Snu2/u/r/QQ are PERSISTENT scratch owned by wctx -- allocated once in
    # build_winner_pair_ctx, overwritten (not reallocated) on every call here.)
    S_sum = 0.0; t0 = 0.0; s0 = 0.0
    Snu = wctx.Snu_buf
    Snu2 = wctx.Snu2_buf
    @inbounds for w in 1:W
        Sw = S[w]; nuw = nu[w]
        S_sum += Sw
        snu = Sw * nuw
        Snu[w] = snu
        t0 += snu
        snu2 = snu * nuw
        Snu2[w] = snu2
        s0 += snu2
    end

    # ---- u[j] = sum_w S[w]*nu[w]*Qtilde[w,j] = sum_w Snu[w]*Qtilde[w,j]  (H[0,1:m] numerator) ----
    # ---- r[j] = sum_w S[w]*nu[w]^2*Qtilde[w,j] = sum_w Snu2[w]*Qtilde[w,j]  (row/col target correction) ----
    u = wctx.u_buf; r = wctx.r_buf
    fill!(u, 0.0); fill!(r, 0.0)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = wctx.winner[w, slot]
            j = slot + (o - 1) * Ddest
            yv = y[w, slot]
            u[j] += Snu[w] * yv
            r[j] += Snu2[w] * yv
        end
    end
    if wctx.has_cf
        jcf = ncolI  # cf column is always the LAST inner-dual column (cf.cf_col == D*Ddest+1 == ncolI when present)
        crs = wctx.cf_raw_scaled
        uu = 0.0; rr = 0.0
        @inbounds for w in 1:W
            uu += Snu[w] * crs[w]
            rr += Snu2[w] * crs[w]
        end
        u[jcf] = uu; r[jcf] = rr
    end

    # ---- Q'SQ, bilateral-bilateral block: dense (D x Ddest) x (D x Ddest) scratch, winner-pair accumulation ----
    # QQ[(o,slot),(o',slot')], but only the (winner[w,slot],slot) entry is ever touched per draw/slot-pair,
    # so build it as an accumulator indexed by (o,slot,o',slot') via nested D-scratch per slot-pair is too
    # large (D^2*Ddest^2); instead accumulate directly into the full (D*Ddest) dense Hessian scratch at the
    # exact winner coordinates -- O(W*Ddest^2) additions total, O((D*Ddest)^2) memory for the scratch itself
    # (unavoidable: this IS the output size, same as the dense architectures already allocate).
    QQ = wctx.QQ_buf
    fill!(QQ, 0.0)   # O(ncolI^2), negligible next to the O(W*Ddest^2) accumulation below -- simpler/safer than tracking exactly which upper-triangle entries need clearing
    @inbounds for slot in 1:Ddest
        for slotp in slot:Ddest
            if slot == slotp
                # diagonal block: o must equal o' (single winner per draw/slot) -- O(W) per slot
                for w in 1:W
                    o = wctx.winner[w, slot]
                    j = slot + (o - 1) * Ddest
                    QQ[j, j] += Snu2[w] * y[w, slot] * y[w, slot]
                end
            else
                for w in 1:W
                    o = wctx.winner[w, slot]; op = wctx.winner[w, slotp]
                    j = slot + (o - 1) * Ddest
                    jp = slotp + (op - 1) * Ddest
                    v = Snu2[w] * y[w, slot] * y[w, slotp]
                    if j <= jp
                        QQ[j, jp] += v
                    else
                        QQ[jp, j] += v
                    end
                end
            end
        end
    end
    if wctx.has_cf
        jcf = ncolI
        crs = wctx.cf_raw_scaled
        # cf-cf
        qcc = 0.0
        @inbounds for w in 1:W
            qcc += Snu2[w] * crs[w] * crs[w]
        end
        QQ[jcf, jcf] += qcc
        # bilateral-cf cross, O(W*Ddest)
        @inbounds for slot in 1:Ddest
            for w in 1:W
                o = wctx.winner[w, slot]
                j = slot + (o - 1) * Ddest
                v = Snu2[w] * y[w, slot] * crs[w]
                if j <= jcf
                    QQ[j, jcf] += v
                else
                    QQ[jcf, j] += v   # unreachable given cf is always last column, kept for generality
                end
            end
        end
    end

    # ---- assemble H[1:m,1:m] = (QQ - r*pi' - pi*r' + s0*pi*pi') / M into upper triangle, write to h ----
    # ---- assemble H[0,1:m] = (u - t0*pi)/M, H[0,0] = S_sum/M ----
    invM = 1.0 / M
    k = 1
    h[k] = S_sum * invM; k += 1
    @inbounds for j in 1:ncolI
        h[k] = (u[j] - t0 * pi_vec[j]) * invM
        k += 1
    end
    @inbounds for i in 1:ncolI
        for j in i:ncolI
            qq = i == j ? QQ[i, i] : (i <= j ? QQ[i, j] : QQ[j, i])
            val = qq - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0 * pi_vec[i] * pi_vec[j]
            h[k] = val * invM
            k += 1
        end
    end
    return h
    finally
        CS._exit_callback!(obj)
    end
end
