# ============================================================================
# Task §9: compressed forward/transpose contraction under the HOMOGENEOUS
# (profiled-destination-scales) moment definitions, operating on the SAME
# CompressedFactual the real operator pipeline uses.
# ADDITIVE ONLY -- does not modify compressed_moments.jl's
# compressed_dual_contraction[!]/compressed_transpose_contraction[!], both
# reused unchanged as the correctness reference these are checked against.
#
# WHERE THE OLD TARGET ACTUALLY LIVES (found by reading the real kernels,
# not assumed): compressed_moments.jl's compressed_dual_contraction!/
# compressed_transpose_contraction! encode the OLD absolute moment's target
# via ONE fixed scalar per destination, `cf.denom[slot]`, multiplied by a
# GLOBAL weighted count (`T`/`Csum`, pulled OUTSIDE the per-draw loop since
# it doesn't vary by draw). The homogeneous moment
# (`Q_od(w) - lambda_od*M_d(w)`, homogeneous_moments_2026-07-31.jl) replaces
# that fixed `denom[slot]` with the model's own PER-DRAW `cf.wval[w,slot]`
# (== M_d(w) exactly, already a field of CompressedFactual -- no
# reconstruction needed). Concretely: the OLD kernels compute a single
# w-independent constant and add it once; these kernels move the
# corresponding term INSIDE the per-draw loop, multiplying by `wval[w,slot]`
# per draw instead. Same O(W*Ddest)+O(D*Ddest) cost, same CompressedFactual
# fields, no dense anything, no new struct.
#
# France's cf_col gets the analogous treatment: target
# `gp^sigma*wPrime_bi*LPrime_bi` (fixed, theory doc section 2.5) becomes
# `gp^sigma*M_f(w)` (per-draw, == `gp^sigma*cf.wval[w,bi_slot]`).
# ============================================================================

isdefined(Main, :dest_slot) || error("homogeneous_contraction_2026-07-31.jl requires cc_algo/active_layout.jl to be included first.")

"""
    homogeneous_dual_contraction(β, cf, ctx, θ_full) -> Vector{Float64}  (length W)

Forward contraction `t[w] = Σ_j β[j]*G_new_{w,j}` under the homogeneous
moment definitions. Exact analogue of `compressed_dual_contraction`
(`compressed_moments.jl`), which this does not modify.
"""
function homogeneous_dual_contraction(β::AbstractVector, cf::CompressedFactual, ctx, θ_full::AbstractVector)
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(β) == cf.oci - 1 || error("homogeneous_dual_contraction: β length $(length(β)) != oci-1 = $(cf.oci-1)")

    κ = Matrix{Float64}(undef, D, Ddest)
    Cbar = zeros(Ddest)   # Cbar[slot] = Σ_o κ[o,slot]*Pmat[o,slot] -- multiplies wval[w,slot] PER DRAW below, no denom factor
    @inbounds for slot in 1:Ddest
        acc = 0.0
        for o in 1:D
            j = slot + (o - 1) * Ddest
            k = β[j] * cf.nrm[j] * cf.gdiv[j]
            κ[o, slot] = k
            acc += k * cf.Pmat[o, slot]
        end
        Cbar[slot] = acc
    end

    κ_cf = 0.0; const_cf = 0.0; bi_slot = 0; gpσ = 0.0
    if cf.cf_col > 0
        bi = ctx.bi
        bi_slot = dest_slot(ctx, bi)
        σ = θ_full[2]; gp = θ_full[3 + D]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        κ_cf = β[cf.cf_col] * cf.nrm[cf.cf_col] * cf.gdiv[cf.cf_col]
        const_cf = κ_cf * denom_cf
    end

    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for j in 1:(cf.oci - 1)
            pmmterm += β[j] * cf.nrm[j] * cf.PMM[j]
        end
    end

    t = Vector{Float64}(undef, W)
    @inbounds for w in 1:W
        acc = const_cf
        for slot in 1:Ddest
            acc += (κ[cf.winner[w, slot], slot] - Cbar[slot]) * cf.wval[w, slot]
        end
        if cf.cf_col > 0
            acc += κ_cf * cf.cf_raw[w] - κ_cf * gpσ * cf.wval[w, bi_slot]
        end
        t[w] = cf.SW[w] * (acc - pmmterm)
    end
    return t
end

"""
    homogeneous_transpose_contraction!(v, weights, cf, ctx, θ_full, B, Tslot) -> v

Transpose contraction `v[j] = Σ_w weights[w]*G_new_{w,j}` under the
homogeneous moment definitions, writing into caller-supplied `v`. Exact
analogue of `compressed_transpose_contraction!` (`compressed_moments.jl`),
which this does not modify. `Tslot` (length `Ddest`, caller-supplied scratch)
replaces the OLD kernel's single global `T` with a PER-DESTINATION weighted
`M_d` sum -- computed in the SAME O(W*Ddest) winner-accumulation pass that
already builds `B`.
"""
function homogeneous_transpose_contraction!(v::AbstractVector{Float64}, weights::AbstractVector, cf::CompressedFactual,
                                             ctx, θ_full::AbstractVector,
                                             B::AbstractMatrix{Float64}, Tslot::AbstractVector{Float64})
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    length(weights) == W || error("homogeneous_transpose_contraction!: weights length $(length(weights)) != W=$W")
    fill!(B, 0.0); fill!(Tslot, 0.0)
    T = 0.0
    Bcf = 0.0
    @inbounds for s in 1:W
        ws = cf.SW[s] * weights[s]
        T += ws
        for slot in 1:Ddest
            wv = cf.wval[s, slot]
            B[cf.winner[s, slot], slot] += ws * wv
            Tslot[slot] += ws * wv
        end
        if cf.cf_col > 0
            Bcf += ws * cf.cf_raw[s]
        end
    end
    @inbounds for slot in 1:Ddest, o in 1:D
        j = slot + (o - 1) * Ddest
        v[j] = cf.nrm[j] * cf.gdiv[j] * (B[o, slot] - cf.Pmat[o, slot] * Tslot[slot]) -
               cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    if cf.cf_col > 0
        bi = ctx.bi
        bi_slot = dest_slot(ctx, bi)
        σ = θ_full[2]; gp = θ_full[3 + D]
        gpσ = gp^σ
        wPrime_bi = 1.0
        denom_cf = gpσ * wPrime_bi * ctx.γ.LPrime[bi]
        j = cf.cf_col
        v[j] = cf.nrm[j] * cf.gdiv[j] * (Bcf + denom_cf * T - gpσ * Tslot[bi_slot]) -
               cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    return v
end
