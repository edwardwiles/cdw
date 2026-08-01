# ============================================================================
# Claude Code task 2026-08-01, §6/§7: the TRULY REDUCED forward/transpose
# operator the prototype never built (homogeneous_contraction_2026-07-31.jl's
# kernels are homogeneous-VALUED but still full-DIMENSIONAL -- ncolI ==
# cf.oci-1, one column per origin per destination INCLUDING the anchor).
# ADDITIVE ONLY -- does not modify homogeneous_contraction_2026-07-31.jl,
# reused unchanged as one of two correctness references below (the other is
# a from-scratch dense reduced-G construction).
#
# DERIVATION (why this reduction is exact, not approximate): the full
# homogeneous kernel's forward pass computes, per destination slot,
#   acc_slot(w) = (kappa[winner(w,slot),slot] - Cbar[slot]) * wval(w,slot),
#   Cbar[slot]  = sum_{o=1}^{D} kappa[o,slot]*Pmat[o,slot].
# Both this task's brief AND direct algebraic inspection agree: setting
# kappa[anchor(slot),slot] == 0 (i.e. never assigning that column a beta
# coefficient at all, rather than fixing it to a computed dual value) leaves
# every OTHER term unchanged -- Cbar[slot] simply becomes a sum over the
# D-1 retained origins, and the per-draw winner lookup naturally contributes
# 0 (no direct one-hot term) whenever the anchor happens to win, while the
# "-Cbar[slot]*wval" part -- built from the D-1 RETAINED origins only --
# still applies on every draw regardless of who wins. This is exactly the
# task's suggested implementation (`kappa[anchor,s]=0`), and
# `test_reduced_homogeneous_contraction_2026-08-01.jl` TEST 3 below proves it
# numerically equals calling the OLD full kernel with the anchor's beta
# coefficient forced to exactly 0.0 (not merely close) -- i.e. this file
# performs no new arithmetic, only a genuine reduction in which coordinates
# are FREE (the anchor was already structurally forced to contribute nothing
# once beta_anchor==0; this file simply never allocates a beta slot for it).
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("reduced_homogeneous_contraction_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")

"""
    reduced_homogeneous_dual_contraction(β, cf, ctx, θ_full, layout) -> Vector{Float64}  (length W)

Forward contraction under the REDUCED homogeneous moment definitions:
`length(β) == layout.total_reduced_economic_moments` (NOT `cf.oci-1`). No
anchor column exists in `β` at all -- this is the actual dimension reduction
task §3 requires, not merely a value change.
"""
function reduced_homogeneous_dual_contraction(β::AbstractVector, cf::CompressedFactual, ctx, θ_full::AbstractVector,
                                               layout::ProfiledEconomicMomentLayout)
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(β) == layout.total_reduced_economic_moments ||
        error("reduced_homogeneous_dual_contraction: β length $(length(β)) != total_reduced_economic_moments = $(layout.total_reduced_economic_moments)")

    κ = zeros(D, Ddest)          # anchor cells structurally left at 0.0 -- no β entry ever written there
    Cbar = zeros(Ddest)
    @inbounds for k in eachindex(layout.retained_full_factual_j)
        o = layout.retained_origin[k]; slot = layout.retained_slot[k]
        j_full = layout.retained_full_factual_j[k]
        kk = β[k] * cf.nrm[j_full] * cf.gdiv[j_full]
        κ[o, slot] = kk
        Cbar[slot] += kk * cf.Pmat[o, slot]
    end

    κ_cf = 0.0; const_cf = 0.0; bi_slot = 0; gpσ = 0.0
    has_france = layout.france_ratio_reduced_j > 0
    if has_france
        cf.cf_col > 0 || error("reduced_homogeneous_dual_contraction: layout claims a France ratio moment but cf.cf_col==0")
        bi = ctx.bi
        bi_slot = dest_slot(ctx, bi)
        σ = θ_full[2]; gp = θ_full[3 + D]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        j_cf_full = cf.cf_col
        κ_cf = β[layout.france_ratio_reduced_j] * cf.nrm[j_cf_full] * cf.gdiv[j_cf_full]
        const_cf = κ_cf * denom_cf
    end

    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for k in eachindex(layout.retained_full_factual_j)
            j_full = layout.retained_full_factual_j[k]
            pmmterm += β[k] * cf.nrm[j_full] * cf.PMM[j_full]
        end
        if has_france
            j_cf_full = cf.cf_col
            pmmterm += β[layout.france_ratio_reduced_j] * cf.nrm[j_cf_full] * cf.PMM[j_cf_full]
        end
    end

    t = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = const_cf
        for slot in 1:Ddest
            acc += (κ[cf.winner[w, slot], slot] - Cbar[slot]) * cf.wval[w, slot]
        end
        if has_france
            acc += κ_cf * cf.cf_raw[w] - κ_cf * gpσ * cf.wval[w, bi_slot]
        end
        t[w] = cf.SW[w] * (acc - pmmterm)
    end
    return t
end

"""
    reduced_homogeneous_transpose_contraction!(v, weights, cf, ctx, θ_full, layout, B, Tslot) -> v

Transpose contraction under the REDUCED layout: `length(v) ==
layout.total_reduced_economic_moments`. Produces NO output entry for any
anchor moment (there is no slot in `v` for one). `B`/`Tslot` are caller
scratch, same shapes as the full kernel's (`D x Ddest`, `Ddest`).
"""
function reduced_homogeneous_transpose_contraction!(v::AbstractVector{Float64}, weights::AbstractVector,
                                                     cf::CompressedFactual, ctx, θ_full::AbstractVector,
                                                     layout::ProfiledEconomicMomentLayout,
                                                     B::AbstractMatrix{Float64}, Tslot::AbstractVector{Float64})
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(weights) == W || error("reduced_homogeneous_transpose_contraction!: weights length $(length(weights)) != W=$W")
    length(v) == layout.total_reduced_economic_moments ||
        error("reduced_homogeneous_transpose_contraction!: v length $(length(v)) != total_reduced_economic_moments = $(layout.total_reduced_economic_moments)")

    fill!(B, 0.0); fill!(Tslot, 0.0)
    T = 0.0
    Bcf = 0.0
    has_france = layout.france_ratio_reduced_j > 0
    @inbounds for s in 1:W
        ws = cf.SW[s] * weights[s]
        T += ws
        for slot in 1:Ddest
            wv = cf.wval[s, slot]
            B[cf.winner[s, slot], slot] += ws * wv
            Tslot[slot] += ws * wv
        end
        if has_france
            Bcf += ws * cf.cf_raw[s]
        end
    end

    @inbounds for k in eachindex(layout.retained_full_factual_j)
        o = layout.retained_origin[k]; slot = layout.retained_slot[k]
        j_full = layout.retained_full_factual_j[k]
        v[k] = cf.nrm[j_full] * cf.gdiv[j_full] * (B[o, slot] - cf.Pmat[o, slot] * Tslot[slot]) -
               cf.nrm[j_full] * cf.usePMM * cf.PMM[j_full] * T
    end

    if has_france
        bi = ctx.bi
        bi_slot = dest_slot(ctx, bi)
        σ = θ_full[2]; gp = θ_full[3 + D]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        j_full = cf.cf_col
        v[layout.france_ratio_reduced_j] = cf.nrm[j_full] * cf.gdiv[j_full] * (Bcf + denom_cf * T - gpσ * Tslot[bi_slot]) -
               cf.nrm[j_full] * cf.usePMM * cf.PMM[j_full] * T
    end
    return v
end

"""
    expand_reduced_beta_to_full(β_reduced, layout, ncolI_full) -> Vector{Float64}

Embeds a reduced β into the OLD full-dimensional β-space (`ncolI_full ==
cf.oci-1`), with every anchor entry forced to exactly `0.0`. Used ONLY as a
cross-check helper (task §16 failure-diagnosis requirement: compare the
reduced kernel to the full kernel evaluated at the algebraically equivalent
zero-anchor point) -- never used on any real solve path, since the whole
point of this file is to never allocate the anchor coordinate at all.
"""
function expand_reduced_beta_to_full(β_reduced::AbstractVector, layout::ProfiledEconomicMomentLayout, ncolI_full::Int)
    β_full = zeros(ncolI_full)
    @inbounds for k in eachindex(layout.retained_full_factual_j)
        β_full[layout.retained_full_factual_j[k]] = β_reduced[k]
    end
    if layout.france_ratio_reduced_j > 0
        β_full[ncolI_full] = β_reduced[layout.france_ratio_reduced_j]   # cf.cf_col == ncolI_full by construction (cf_col = D*Ddest+1 = ncolI_full)
    end
    return β_full
end
