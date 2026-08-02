# ============================================================================
# Claude Code task 2026-08-01, §8: H_EE rebuilt in the REDUCED basis (the
# prototype's homogeneous_hessian_2026-07-31.jl still uses
# ncolI = cf.oci - 1, the FULL anchor-inclusive dimension).
# ADDITIVE ONLY -- does not modify homogeneous_hessian_2026-07-31.jl.
#
# DERIVATION (why dropping the anchor's j-index is exact, not a further
# approximation on top of §6/§7's reduction): the dual objective is
# sum_w Psi(r_w), r_w = -zeta - sum_j E_{w,j}*lambda_j. Every economic moment
# column (retained or anchor) decomposes as
#   E_{w,j} = y_part[w,j] - Lam[j]*wval[w,slot(j)],
#   y_part[w,j] = kappa0[j]*wval[w,slot(j)] * 1{o(j) == winner[w,slot(j)]}
#   (sparse: nonzero for at most one j per (w,slot)), Lam[j] = kappa0[j]*
#   Pmat[o(j),slot(j)] (a FIXED per-column constant, not draw-dependent).
# The Hessian block QQ[i,j] = sum_w Psi''(r_w)*E_{w,i}*E_{w,j} expands into
# four pieces (see homogeneous_hessian_2026-07-31.jl's own header for the
# original derivation over the FULL index set); nothing in that expansion
# assumes j ranges over every origin at a slot -- it holds for ANY subset of
# (o,slot) columns, because y_part/Lam are defined per-column, not
# per-complete-slot. Restricting j to the RETAINED (o,slot) pairs only is
# therefore exact: the "dense" pieces (T1[slot], R4[slot,slotp], the
# gradient/Hessian's -Lam[j]*T1[target_slot[j]] and -Lam[i]*Lam[j]*
# R4[...] terms) are UNCHANGED full sums over every draw (they only ever
# needed wval, which is defined regardless of who wins); the "sparse" pieces
# (u[j], r[j], R2[j,*], QQ[j,*]) simply accumulate NOTHING on any draw where
# the winner happens to be the OMITTED anchor origin (there is no j for it),
# which is exactly the task's own description: "there is no direct one-hot
# winner coefficient but every retained moment still receives its
# -lambda*M term" (the T1/R4 pieces ARE that term, and they don't care who
# won). `test_reduced_homogeneous_hessian_2026-08-01.jl` verifies this
# against BOTH a finite-difference Hessian of the reduced dual objective AND
# an explicit dense reduced-G Hessian (`Gred' * Diagonal(Psi''(r)) * Gred`),
# per task §8's "use exact/dense comparisons in addition to finite
# differences where possible."
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("reduced_homogeneous_hessian_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")

struct ReducedHomogeneousWinnerPairHessCtx
    D::Int
    Ddest::Int
    W::Int
    ncolI::Int                      # == layout.total_reduced_economic_moments (1 + this = packed Hessian dim)
    has_cf::Bool
    kappa0::Vector{Float64}          # length ncolI
    lambda_coef::Vector{Float64}     # length ncolI: Pmat[o(j),slot(j)] (bilateral) or gp^sigma (france)
    target_slot::Vector{Int}         # length ncolI
    nu::Vector{Float64}              # sampling weights (cf.SW)
    y::Matrix{Float64}               # W x Ddest: kappa0[winner's reduced col]*wval[w,slot], 0.0 if winner is the anchor
    wval::Matrix{Float64}            # W x Ddest: raw per-draw destination value, ALWAYS defined (winner-independent)
    winner_reduced_col::Matrix{Int}  # W x Ddest: reduced j of the winner, or 0 if the winner is the omitted anchor
    ycf::Vector{Float64}             # length W (empty if !has_cf): kappa0[france]*(cf_raw+denom_cf), winner-independent
end

"""
    build_reduced_homogeneous_winner_pair_ctx(cf, ctx, θ_full, layout) -> ReducedHomogeneousWinnerPairHessCtx

Builds the reduced-basis Hessian context. `layout.total_reduced_economic_moments`
is the packed Hessian's `ncolI` (NOT `cf.oci-1`).
"""
function build_reduced_homogeneous_winner_pair_ctx(cf::CompressedFactual, ctx, θ_full::AbstractVector,
                                                     layout::ProfiledEconomicMomentLayout)
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    ncolI = layout.total_reduced_economic_moments
    has_cf = layout.france_ratio_reduced_j > 0

    kappa0 = Vector{Float64}(undef, ncolI)
    lambda_coef = Vector{Float64}(undef, ncolI)
    target_slot = Vector{Int}(undef, ncolI)
    @inbounds for k in eachindex(layout.retained_full_factual_j)
        o = layout.retained_origin[k]; slot = layout.retained_slot[k]
        j_full = layout.retained_full_factual_j[k]
        k0 = cf.nrm[j_full] * cf.gdiv[j_full]
        kappa0[k] = k0
        lambda_coef[k] = cf.Pmat[o, slot]
        target_slot[k] = slot
    end

    winner_reduced_col = zeros(Int, W, Ddest)
    y = zeros(W, Ddest)
    wval = copy(cf.wval)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = cf.winner[w, slot]
            j_full = slot + (o - 1) * Ddest
            k = layout.full_factual_to_reduced[j_full]
            winner_reduced_col[w, slot] = k
            k != 0 && (y[w, slot] = kappa0[k] * wval[w, slot])
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
        jcf = layout.france_ratio_reduced_j
        j_full_cf = cf.cf_col
        k0cf = cf.nrm[j_full_cf] * cf.gdiv[j_full_cf]
        kappa0[jcf] = k0cf
        lambda_coef[jcf] = gpσ
        target_slot[jcf] = bi_slot
        ycf = k0cf .* (cf.cf_raw .+ denom_cf)
    end

    return ReducedHomogeneousWinnerPairHessCtx(D, Ddest, W, ncolI, has_cf, kappa0, lambda_coef, target_slot,
        copy(cf.SW), y, wval, winner_reduced_col, ycf)
end

"""
    reduced_homogeneous_winner_pair_hessian!(h, obj, wctx) -> h

Fills the packed row-major upper-triangular Hessian, dimension
`n = 1 + wctx.ncolI` (`ncolI == layout.total_reduced_economic_moments`, NOT
`cf.oci-1`). Same precondition as the full kernel: `obj.arg0` must already
hold `q = -zeta - t` at the evaluation point.
"""
function reduced_homogeneous_winner_pair_hessian!(h::AbstractVector, obj, wctx::ReducedHomogeneousWinnerPairHessCtx)
    CS._enter_callback!(obj)
    try
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    M = obj.M
    Ddest = wctx.Ddest; W = wctx.W; ncolI = wctx.ncolI
    n = 1 + ncolI
    kappa0 = wctx.kappa0; lambda_coef = wctx.lambda_coef; target_slot = wctx.target_slot
    nu = wctx.nu; y = wctx.y; wval = wctx.wval; winner_reduced_col = wctx.winner_reduced_col

    length(h) == n * (n + 1) ÷ 2 || error("reduced_homogeneous_winner_pair_hessian!: length(h)=$(length(h)) != n(n+1)/2 for n=$n")

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

    # T1/R4: FULL sums over every draw, winner-independent (task: "every retained moment still
    # receives its -lambda*M term" even on draws where the anchor origin wins).
    T1 = zeros(Ddest)
    R4 = zeros(Ddest, Ddest)
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

    # u/r/R2/QQ: SPARSE winner-indicator pieces -- accumulate nothing when the winner at (w,slot)
    # is the omitted anchor (winner_reduced_col[w,slot]==0). This is the whole dimension reduction.
    u = zeros(ncolI); r = zeros(ncolI)
    QQ = zeros(ncolI, ncolI)
    R2 = zeros(ncolI, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            j = winner_reduced_col[w, slot]
            j == 0 && continue
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
                    j = winner_reduced_col[w, slot]
                    j == 0 && continue
                    QQ[j, j] += Snu2[w] * y[w, slot] * y[w, slot]
                end
            else
                for w in 1:W
                    j = winner_reduced_col[w, slot]; jp = winner_reduced_col[w, slotp]
                    (j == 0 || jp == 0) && continue
                    v = Snu2[w] * y[w, slot] * y[w, slotp]
                    QQ[j, jp] += v
                    QQ[jp, j] += v
                end
            end
        end
    end

    if wctx.has_cf
        jcf = ncolI   # france is always the last reduced column by construction (layout.france_ratio_reduced_j == n_bilateral+1 == ncolI)
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
                j = winner_reduced_col[w, slot]
                j == 0 && continue
                v = Snu2[w] * y[w, slot] * ycf[w]
                QQ[j, jcf] += v
                QQ[jcf, j] += v
            end
        end
    end

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
