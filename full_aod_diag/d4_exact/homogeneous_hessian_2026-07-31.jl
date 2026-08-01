# ============================================================================
# Task §11: H_EE under the homogeneous (profiled-destination-scales) moment
# definitions.
# ADDITIVE ONLY -- does not modify core_exact_hessian.jl's
# build_winner_pair_ctx/winner_pair_hessian!, both reused as the structural
# template and correctness reference for the UNCHANGED parts.
#
# CORRECTION to this session's own earlier audit finding: the audit
# (FULL_TO_PROFILED_PIPELINE_CALL_GRAPH_2026-07-31.md, Topic 7) classified
# H_EE as "unchanged," reasoning that winner_pair_hessian! operates entirely
# on cf::CompressedFactual's winner/wval fields, decoupled from A-coordinate
# layout. That reasoning is INCOMPLETE: `build_winner_pair_ctx` also reads
# `cf.denom[slot]` directly (via `pi_vec[j] = kappa0[j]*Pmat[o,slot]*
# denom[slot] + ...`), the SAME fixed-target dependency homogeneous_
# contraction_2026-07-31.jl found and fixed in the FG kernels. Found by
# reading the real kernel body (core_exact_hessian.jl:373-408) after the FG
# fix, not assumed to transfer from the FG finding.
#
# UNLIKE FG, this is NOT a one-line scalar swap: pi_vec (constant per
# column) is used throughout winner_pair_hessian! multiplied against
# AGGREGATE scalar sums (t0, r[i], s0) that only factor cleanly because
# pi_vec doesn't vary by draw. Under the homogeneous moment, the "target"
# lambda_j*wval[w,slot(j)] IS per-draw, so the Hessian's quadratic-form cross
# terms need two NEW accumulators the old kernel never computes:
#   T1[slot]      = Σ_w Snu[w]*wval[w,slot]                  (Ddest-vector)
#   R2[j,slotp]   = Σ_w Snu2[w]*y[w,slot(j)]*1_j(w)*wval[w,slotp]  (ncolI x Ddest)
#   R4[slot,slotp]= Σ_w Snu2[w]*wval[w,slot]*wval[w,slotp]    (Ddest x Ddest)
# (QQ, the y*y cross term, is UNCHANGED -- it never involves the target at
# all.) See PROFILED_HEE_DERIVATION_2026-07-31.md for the full algebra.
# Same asymptotic cost as the original, O(W*Ddest^2)+O(W*Ddest).
# ============================================================================

isdefined(Main, :dest_slot) || error("homogeneous_hessian_2026-07-31.jl requires cc_algo/active_layout.jl to be included first.")

struct HomogeneousWinnerPairHessCtx
    D::Int
    Ddest::Int
    W::Int
    ncolI::Int
    has_cf::Bool
    kappa0::Vector{Float64}
    lambda_coef::Vector{Float64}   # length ncolI: Pmat[o(j),slot(j)] (bilateral) or gp^sigma (cf_col)
    target_slot::Vector{Int}        # length ncolI: slot(j) (bilateral) or bi_slot (cf_col)
    nu::Vector{Float64}
    y::Matrix{Float64}              # W x Ddest, kappa0-scaled winner value (bilateral, UNCHANGED formula)
    winner::Matrix{Int}
    ycf::Vector{Float64}            # kappa0[cf_col]*(cf_raw[w]+denom_cf), length W (empty if !has_cf)
end

function build_homogeneous_winner_pair_ctx(cf::CompressedFactual, ctx, θ_full::AbstractVector)
    D = cf.D; Ddest = cf.D_dest; W = cf.W; ncolI = cf.oci - 1
    has_cf = cf.cf_col > 0

    kappa0 = Vector{Float64}(undef, ncolI)
    lambda_coef = Vector{Float64}(undef, ncolI)
    target_slot = Vector{Int}(undef, ncolI)
    @inbounds for slot in 1:Ddest, o in 1:D
        j = slot + (o - 1) * Ddest
        k0 = cf.nrm[j] * cf.gdiv[j]
        kappa0[j] = k0
        lambda_coef[j] = cf.Pmat[o, slot]
        target_slot[j] = slot
    end

    y = Matrix{Float64}(undef, W, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = cf.winner[w, slot]
            j = slot + (o - 1) * Ddest
            y[w, slot] = kappa0[j] * cf.wval[w, slot]
        end
    end

    ycf = Float64[]
    if has_cf
        bi = ctx.bi
        bi_slot = dest_slot(ctx, bi)
        σ = θ_full[2]; gp = θ_full[3 + D]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        jcf = cf.cf_col
        k0cf = cf.nrm[jcf] * cf.gdiv[jcf]
        kappa0[jcf] = k0cf
        lambda_coef[jcf] = gpσ
        target_slot[jcf] = bi_slot
        ycf = k0cf .* (cf.cf_raw .+ denom_cf)
    end

    return HomogeneousWinnerPairHessCtx(D, Ddest, W, ncolI, has_cf, kappa0, lambda_coef, target_slot,
        copy(cf.SW), y, cf.winner, ycf)
end

"""
    homogeneous_winner_pair_hessian!(h, obj, wctx) -> h

Fills the SAME packed row-major upper-triangular Hessian layout
`winner_pair_hessian!` does (n=1+ncolI), under the homogeneous moment
definitions. `obj.arg0`/`obj.arg2` must already be populated (same
precondition as the original).
"""
function homogeneous_winner_pair_hessian!(h::AbstractVector, obj, wctx::HomogeneousWinnerPairHessCtx)
    CS._enter_callback!(obj)
    try
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    M = obj.M
    Ddest = wctx.Ddest; W = wctx.W; ncolI = wctx.ncolI
    n = 1 + ncolI
    kappa0 = wctx.kappa0; lambda_coef = wctx.lambda_coef; target_slot = wctx.target_slot
    nu = wctx.nu; y = wctx.y

    length(h) == n * (n + 1) ÷ 2 || error("homogeneous_winner_pair_hessian!: length(h)=$(length(h)) != n(n+1)/2 for n=$n")

    Snu = Vector{Float64}(undef, W)
    Snu2 = Vector{Float64}(undef, W)
    S_sum = 0.0
    @inbounds for w in 1:W
        Sw = S[w]; nuw = nu[w]
        S_sum += Sw
        snu = Sw * nuw
        Snu[w] = snu
        Snu2[w] = snu * nuw
    end

    # T1[slot] = Σ_w Snu[w]*wval[w,slot]  (needs raw wval, i.e. y[w,slot]/kappa0[winner col] -- recompute
    # directly from cf.wval via wctx; simplest: reconstruct from y using the SAME winner cell's kappa0)
    T1 = zeros(Ddest)
    R4 = zeros(Ddest, Ddest)
    wval = Matrix{Float64}(undef, W, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = wctx.winner[w, slot]
            j = slot + (o - 1) * Ddest
            wval[w, slot] = y[w, slot] / kappa0[j]
        end
    end
    @inbounds for w in 1:W
        for slot in 1:Ddest
            T1[slot] += Snu[w] * wval[w, slot]
        end
        for slot in 1:Ddest, slotp in slot:Ddest
            v = Snu2[w] * wval[w, slot] * wval[w, slotp]
            R4[slot, slotp] += v
            slot != slotp && (R4[slotp, slot] += v)
        end
    end

    u = zeros(ncolI); r = zeros(ncolI)
    QQ = zeros(ncolI, ncolI)
    R2 = zeros(ncolI, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = wctx.winner[w, slot]
            j = slot + (o - 1) * Ddest
            yv = y[w, slot]
            u[j] += Snu[w] * yv
            r[j] += Snu2[w] * yv
            for slotp in 1:Ddest
                R2[j, slotp] += Snu2[w] * yv * wval[w, slotp]
            end
        end
    end
    @inbounds for slot in 1:Ddest
        for slotp in slot:Ddest
            if slot == slotp
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
                    QQ[j, jp] += v
                    QQ[jp, j] += v
                end
            end
        end
    end

    if wctx.has_cf
        jcf = ncolI
        ycf = wctx.ycf
        bi_slot = target_slot[jcf]
        uu = 0.0; rr = 0.0
        @inbounds for w in 1:W
            uu += Snu[w] * ycf[w]
            rr += Snu2[w] * ycf[w]
            for slotp in 1:Ddest
                R2[jcf, slotp] += Snu2[w] * ycf[w] * wval[w, slotp]
            end
        end
        u[jcf] = uu; r[jcf] = rr
        qcc = 0.0
        @inbounds for w in 1:W
            qcc += Snu2[w] * ycf[w] * ycf[w]
        end
        QQ[jcf, jcf] += qcc
        @inbounds for slot in 1:Ddest
            for w in 1:W
                o = wctx.winner[w, slot]
                j = slot + (o - 1) * Ddest
                v = Snu2[w] * y[w, slot] * ycf[w]
                QQ[j, jcf] += v
                QQ[jcf, j] += v
            end
        end
    end

    # Lam[j] = kappa0[j]*lambda_coef[j] -- the ORIGINAL kernel's pi_vec[j] already has kappa0[j]
    # baked in (pi_vec[j] = kappa0[j]*Pmat[o,slot]*denom[slot]); every homogeneous replacement term
    # below must carry the SAME kappa0 scaling that y/QQ/R2 already do, not just lambda_coef alone.
    Lam = kappa0 .* lambda_coef

    invM = 1.0 / M
    k = 1
    h[k] = S_sum * invM; k += 1
    @inbounds for j in 1:ncolI
        h[k] = (u[j] - Lam[j] * T1[target_slot[j]]) * invM
        k += 1
    end
    @inbounds for i in 1:ncolI
        for j in i:ncolI
            val = QQ[i, j] - Lam[j] * R2[i, target_slot[j]] - Lam[i] * R2[j, target_slot[i]] +
                  Lam[i] * Lam[j] * R4[target_slot[i], target_slot[j]]
            h[k] = val * invM
            k += 1
        end
    end
    return h
    finally
        CS._exit_callback!(obj)
    end
end
