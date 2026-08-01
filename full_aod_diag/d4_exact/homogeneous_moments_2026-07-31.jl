# ============================================================================
# Task §8 (core piece) / §1.2: homogeneous factual moment.
# ADDITIVE ONLY -- reuses destination_M_d (recover_full_a_2026-07-31.jl)
# unchanged; requires that file included first.
#
# Replaces the OLD moment  E_F[Q_od(w)] - lambda_od*denom[d] = 0  (denom[d] a
# FIXED DATA constant, NOT invariant to a destination-column rescale -- this
# is what currently pins the destination scale, per
# PROFILED_DESTINATION_SCALES_MASTER_2026-07-31.md's resolution of the
# "is the scale really redundant" question) with the NEW moment
#   E_F[Q_od(w) - lambda_od*M_d(w)] = 0
# using the MODEL's OWN M_d(w) instead of the fixed denom[d]. Because both
# Q_od(w) and M_d(w) scale by the SAME factor kappa^(mu*(sigma-1)) under a
# common destination-d column rescale (theory doc section 2.1, numerically
# confirmed), this new moment is exactly homogeneous: its value scales by
# kappa^(mu*(sigma-1)) too (not merely "stays zero if zero" -- the exact
# proportional-rescaling property, tested directly below), which is exactly
# why the destination scale becomes genuinely unidentified under this moment
# set and can be validly profiled out via a fixed anchor gauge instead of
# left as a KNITRO search direction.
# ============================================================================

isdefined(Main, :destination_M_d) || error("homogeneous_moments_2026-07-31.jl requires recover_full_a_2026-07-31.jl to be included first.")

"""
    homogeneous_factual_moment(θ_full, ctx; d_list=1:ctx.D) -> Dict{Int,Matrix{Float64}}

For every destination `d` in `d_list`, returns a `W x D` matrix whose column
`o` is the per-draw homogeneous moment `Q_od(w) - lambda_od*M_d(w)`. Reuses
`destination_M_d` for `M_d(w)` and reconstructs `Q_od(w)` the same way that
function does internally (documented in its own header), so the two are
mutually consistent by construction.
"""
function homogeneous_factual_moment(θ_full::AbstractVector{Float64}, ctx; d_list = 1:ctx.D)
    D = ctx.D
    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, ctx.nTotalMoments)
    ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
    μ = θ_full[1]; σ = θ_full[2]
    lambda = reshape(ctx.γ.P, (D, D))'
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = ctx.γ.SamplingWeights[1:W]
    wscale = SW ./ gammafac
    out = Dict{Int,Matrix{Float64}}()
    for d in d_list
        denom_d = ctx.γ.wHat[d] * ctx.γ.L[d]
        Q = zeros(W, D)
        for o in 1:D
            d1 = d + (o - 1) * D
            @. Q[:, o] = G[:, d1] / wscale + lambda[o, d] * denom_d
        end
        M = vec(sum(Q, dims = 2))
        H = zeros(W, D)
        for o in 1:D
            @. H[:, o] = Q[:, o] - lambda[o, d] * M
        end
        out[d] = H
    end
    return out
end

# ============================================================================
# Task §1.3 / theory doc section 2.5: France homogeneous ratio moment.
#
# OLD (absolute) target: rho_f_absolute = gp^sigma * wPrime_bi * LPrime_bi
# (compressed_moments.jl:262-266's own cf_raw/denom_cf construction --
# wPrime_bi==1 exactly, confirmed). NEW (homogeneous) target coefficient:
# rho_f_ratio = gp^sigma (theory doc section 2.5's derivation: matches the
# old target exactly at the gamma-normalized point, using the confirmed
# identity LPrime_bi == L_bi -- population is physically invariant across
# the factual/counterfactual scenario).
#
# CORRECTION NOTE: an EARLIER pass in this session concluded rho_f == gp
# (identity, no sigma power), by misreading K (the OUTER KNITRO objective
# value -- gp/gamma'_focal itself, being extremized -- confirmed by tracing
# cc_algo/PsiObjectiveBundle.jl's callable, which never reads column 1 of H
# in its inner-dual/outer-constraint computation) as if it were the France
# moment's target. The actual France moment is G's own cf_col, an INNER-DUAL
# column (confirmed live: D=4 test context has outer_constr_index==obj.d==18,
# numMomentInnerSimple==17==D^2+1, i.e. gravity alone is the outer column).
# See PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md section 0's correction
# and section 2.5 for the full derivation.
# ============================================================================

"""
    homogeneous_france_moment(θ_full, ctx) -> Vector{Float64}

Per-draw homogeneous France ratio moment `Phi_ff(w) - gp^sigma * M_f(w)`,
where `Phi_ff(w)` is reconstructed from the live `moments!` output's `cf_col`
column (adding back its own internal `denom_cf` target, the same pattern
`homogeneous_factual_moment` uses for the bilateral columns) and `M_f(w)` is
France's (baseIndex's) factual destination total via `destination_M_d`.
"""
function homogeneous_france_moment(θ_full::AbstractVector{Float64}, ctx)
    D = ctx.D
    W = size(ctx.U, 1)
    bi = ctx.bi
    K = zeros(W); G = zeros(W, ctx.nTotalMoments)
    ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
    μ = θ_full[1]; σ = θ_full[2]
    gp = θ_full[3 + D]
    gammafac = spgamma(μ * (1 - σ) + 1)
    SW = ctx.γ.SamplingWeights[1:W]
    wscale = SW ./ gammafac
    cf_col = D^2 + 1
    wPrime_bi = 1.0   # confirmed exact (compressed_moments.jl: wPrime built by inserting 1.0 at bi)
    denom_cf = gp^σ * wPrime_bi * ctx.γ.LPrime[bi]
    Phi_ff = G[:, cf_col] ./ wscale .+ denom_cf
    M_f = destination_M_d(θ_full, ctx; d_list = [bi])[bi]
    return Phi_ff .- gp^σ .* M_f
end
