# ============================================================================
# Diagnostic 1 (+ Diagnostic 8-lite): analytical vs AD vs finite-difference
# Jacobian of the trade-flow moments w.r.t. log(Aod_theta), at the Frechet
# benchmark. Requires diagnostics/context.jl to be included first.
#
# All three Jacobians are D^2 x D^2 matrices using the SAME two index
# conventions (see context.jl's idx_out/idx_in):
#   row  = idx_out(o,d) = d+(o-1)*D   (moment for origin o, destination d)
#   col  = idx_in(j,d)  = j+(d-1)*D   (perturbation of Aod_theta[j,d])
# Off-block entries (different d) are exactly zero by construction (Aod_theta
# at destination d only enters that destination's D moments) and are not
# computed by the AD/FD paths either, so the comparison is well-posed only
# within each destination's D x D block; we still build/compare the full
# D^2 x D^2 matrix since off-block zeros are themselves a testable prediction.
# ============================================================================

"""
    analytical_jacobian_full(λ, β, D; gamma_norm=1.0)

∂E_F[g_od]/∂log(Aod_theta[j,d]) at the Frechet benchmark, derived from the
user's economic-setup formula in α=log(b) coordinates,
  ∂E[g_od]/∂α_od = β λ_od + (1-β) λ_od^2      (j=o)
  ∂E[g_od]/∂α_jd = (1-β) λ_od λ_jd            (j≠o)
chain-ruled to the code's actual free coordinate log(Aod_theta[j,d]) via
dα_jd/d log(Aod_theta[j,d]) = μ(σ-1) = 1/β:
  ∂E[g_od]/∂log(Aod_theta[j,d]) = λ_od + c λ_od^2       (j=o),   c=(1-β)/β
                                 = c λ_od λ_jd           (j≠o)
Zero for d'≠d (different destination) by construction.

`gamma_norm` = gamma(μ(1-σ)+1), the SAME Frechet-moment normalizing constant
production's own `EK_moments_gammanorm_directgp!` divides its raw moments by
(`G[:,1:simple_end] /= gamma(μ*(1-σ)+1)`, moments_gammanorm.jl). This factor is
NOT part of the user's abstract economic-setup formula (which is stated in raw
E[U^(σ-1)]-normalized units) but IS baked into the code's actual G object, so
must be divided through here too for an apples-to-apples comparison against
the AD/FD Jacobians (both computed on production's actual, already-normalized
moments). Verified empirically: dividing by this constant closes an initially
puzzling, remarkably UNIFORM ~18% gap (ratio 0.81-0.83 across all 25 D=5
diagonal entries, tightly clustered around 1/gamma(mu*(1-sigma)+1) to <2%,
i.e. within Monte Carlo noise) between the raw analytical formula and AD.
"""
function analytical_jacobian_full(λ::AbstractMatrix{Float64}, β::Float64, D::Int; gamma_norm::Float64=1.0)
    J = zeros(D^2, D^2)
    c = (1 - β) / β
    for d in 1:D, o in 1:D
        row = idx_out(o, d, D)
        for j in 1:D
            col = idx_in(j, d, D)
            J[row, col] = ((j == o) ? (λ[o, d] + c * λ[o, d]^2) : (c * λ[o, d] * λ[j, d])) / gamma_norm
        end
    end
    return J
end

"""
AD Jacobian via ForwardDiff.jacobian over the D^2 Aod_theta entries (mirrors
production's own AD path). Uses `tradeshare_share_means` (the share-recovered
object, E_F=λ_od at the benchmark) so it is directly comparable to
`analytical_jacobian_full` -- production's own G is a deviation object with
E_F=0 (see context.jl); since share = G/denom[d] + const, this differs from
differentiating raw G only by a per-destination constant row rescaling, and
production's own AD/envelope gradient code differentiates exactly this same
G internally, so no information about winner-boundary handling is lost by
working in share units here.
"""
function ad_jacobian_full(θbase::Vector{Float64}, Aod0::Vector{Float64}, U, γobj, D::Int, nTotalMoments::Int, Aod_offset::Int)
    # differentiate w.r.t. LOG(Aod_theta) directly (Aodvec = Aod0.*exp(logdelta), eval at logdelta=0) --
    # NOT the raw level, since Aod0 is generally != 1 (per-destination constants from the gammanorm
    # gauge conversion), so a raw-level ForwardDiff.jacobian would silently need an extra
    # elementwise *Aod0[col] rescaling to match the log-derivative convention used everywhere else.
    f = logdelta -> tradeshare_share_means(make_full_theta(θbase, Aod0 .* exp.(logdelta), Aod_offset, D), U, γobj, D, nTotalMoments)
    return ForwardDiff.jacobian(f, zeros(length(Aod0)))
end

"""
    smooth_only_jacobian_full(λ, β, D; gamma_norm=1.0)

The "smooth part" AD alone can see (winner indicator held mechanically fixed
by MinInd!'s zero-derivative Bool branch): for j=o, decomposing the total
analytical formula β λ_od + (1-β)λ_od^2 into
  SMOOTH(α-units) = λ_od                          -- exact identity:
    d(pricesTempσ[o]*1{o wins})/dα_od = 1{o wins}*d(pricesTempσ[o])/dα_od
    (product rule; second term is exactly 0 since pricesInd carries zero AD
    partials), and d(pricesTempσ[o])/dα_od = pricesTempσ[o] exactly (b_od is
    literally e^α_od), so E[·] = E[pricesTempσ[o]*1{o wins}] = λ_od*denom[d].
  BOUNDARY(α-units) = (β-1)λ_od(1-λ_od)             -- the remainder.
  (Check: λ_od + (β-1)λ_od(1-λ_od) = β λ_od + (1-β)λ_od^2. ✓)
Chain-ruled to log(Aod_theta) via 1/β and gamma-normalized: SMOOTH_δ =
λ_od/(β·gamma_norm), zero off-diagonal (AD's off-diagonal has NO channel at
all, smooth or otherwise, into a DIFFERENT origin's moment -- see
jacobian_checks.jl module docstring). This is the quantity AD should
reproduce almost exactly (a mathematical near-identity, not an approximation,
up to the target λ_od being the DATA value vs AD's own simulated value at
finite W).
"""
function smooth_only_jacobian_full(λ::AbstractMatrix{Float64}, β::Float64, D::Int; gamma_norm::Float64=1.0)
    J = zeros(D^2, D^2)
    for d in 1:D, o in 1:D
        J[idx_out(o, d, D), idx_in(o, d, D)] = λ[o, d] / (β * gamma_norm)
    end
    return J
end

"""
    fd_jacobian_full(θbase, Aod0, U, γobj, D, nTotalMoments, Aod_offset, h)

Central-difference Jacobian, perturbing log(Aod_theta[j,d]) by ±h (i.e.
Aod_theta[j,d] *= exp(±h)) and FULLY recomputing moments (hard MinInd!, no
Duals at all -- the gold-standard "exact recompute" reference). Same simulation
draws U reused for every column (common random numbers).
"""
function fd_jacobian_full(θbase::Vector{Float64}, Aod0::Vector{Float64}, U, γobj, D::Int,
        nTotalMoments::Int, Aod_offset::Int, h::Float64)
    n = D^2
    J = zeros(n, n)
    for k in 1:n
        Aplus = copy(Aod0); Aplus[k] *= exp(h)
        Aminus = copy(Aod0); Aminus[k] *= exp(-h)
        gplus = tradeshare_share_means(make_full_theta(θbase, Aplus, Aod_offset, D), U, γobj, D, nTotalMoments)
        gminus = tradeshare_share_means(make_full_theta(θbase, Aminus, Aod_offset, D), U, γobj, D, nTotalMoments)
        J[:, k] = (gplus .- gminus) ./ (2h)
    end
    return J
end

"""
    jacobian_error_report(J_ref, J_test, D; label)

Splits errors into diagonal (j==o, same d) vs off-diagonal (j≠o, same d) vs
cross-destination (d'≠d, should be exactly 0 in both) blocks and reports
max/mean abs and relative error for each.
"""
function jacobian_error_report(J_ref::Matrix{Float64}, J_test::Matrix{Float64}, D::Int; label::String="")
    diag_abs = Float64[]; diag_rel = Float64[]
    off_abs = Float64[]; off_rel = Float64[]
    cross_abs = Float64[]
    for d in 1:D, o in 1:D
        row = idx_out(o, d, D)
        for j in 1:D
            col = idx_in(j, d, D)
            e = abs(J_test[row, col] - J_ref[row, col])
            r = e / max(abs(J_ref[row, col]), 1e-12)
            if j == o
                push!(diag_abs, e); push!(diag_rel, r)
            else
                push!(off_abs, e); push!(off_rel, r)
            end
        end
        for dp in 1:D
            dp == d && continue
            for j in 1:D
                col = idx_in(j, dp, D)
                push!(cross_abs, abs(J_test[row, col] - J_ref[row, col]))
            end
        end
    end
    return (label=label,
        diag_max_abs=maximum(diag_abs), diag_mean_abs=mean(diag_abs),
        diag_max_rel=maximum(diag_rel), diag_mean_rel=mean(diag_rel),
        off_max_abs=maximum(off_abs), off_mean_abs=mean(off_abs),
        off_max_rel=maximum(off_rel), off_mean_rel=mean(off_rel),
        cross_max_abs=maximum(cross_abs))
end

"""
    winner_switch_report(ctx, j_focus, d_focus, o_other, hs)

Diagnostic 8-lite: for the (j_focus,d_focus) perturbation, at each h in hs,
(1) recomputes exact winners at destination d_focus (via winners_at_destination)
before/after perturbing Aod_theta[j_focus,d_focus] by +h, reporting the
fraction of draws whose winner switches; (2) the realized central-difference
derivative estimate for the DIAGONAL entry (o=j_focus) and one OFF-DIAGONAL
entry (o=o_other), so convergence to the nonzero analytical value (not to 0)
can be checked as h shrinks.
"""
function winner_switch_report(θbase::Vector{Float64}, Aod0::Vector{Float64}, U, γobj, D::Int,
        nTotalMoments::Int, Aod_offset::Int, j_focus::Int, d_focus::Int, o_other::Int, hs::Vector{Float64})
    θ0 = make_full_theta(θbase, Aod0, Aod_offset, D)
    H0 = full_H(θ0, U, γobj, nTotalMoments)
    w0 = winners_at_destination(H0, γobj, D, d_focus)

    rows = NamedTuple[]
    k = idx_in(j_focus, d_focus, D)
    row_diag = idx_out(j_focus, d_focus, D)
    row_off = idx_out(o_other, d_focus, D)
    denom_d = γobj.wHat[d_focus] * γobj.L[d_focus]   # rescale raw-G derivative to share units, matching analytical_jacobian_full
    for h in hs
        Aplus = copy(Aod0); Aplus[k] *= exp(h)
        Aminus = copy(Aod0); Aminus[k] *= exp(-h)
        θp = make_full_theta(θbase, Aplus, Aod_offset, D)
        θm = make_full_theta(θbase, Aminus, Aod_offset, D)
        Hp = full_H(θp, U, γobj, nTotalMoments)
        Hm = full_H(θm, U, γobj, nTotalMoments)
        wp = winners_at_destination(Hp, γobj, D, d_focus)
        frac_switch = mean(w0 .!= wp)

        gplus_diag = mean(view(Hp, :, 2 + row_diag)); gminus_diag = mean(view(Hm, :, 2 + row_diag))
        gplus_off = mean(view(Hp, :, 2 + row_off)); gminus_off = mean(view(Hm, :, 2 + row_off))
        d_diag = (gplus_diag - gminus_diag) / (2h) / denom_d
        d_off = (gplus_off - gminus_off) / (2h) / denom_d

        push!(rows, (h=h, frac_switch=frac_switch, fd_diag=d_diag, fd_offdiag=d_off))
    end
    return rows
end
