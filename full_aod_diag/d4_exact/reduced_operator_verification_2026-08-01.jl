# ============================================================================
# Claude Code task 2026-08-01, §11/§12: independent post-solve verification
# for the REDUCED (profiled-destination-scales) inner solve, mirroring
# `verify_inner_solution_operator_unrestricted!` (operator_verification.jl)
# exactly but built on the reduced kernels (reduced_homogeneous_contraction_
# 2026-08-01.jl) instead of `economic_forward!`/`economic_transpose!`. Feeds
# the SAME `verify_namedtuple_from_operator` (operator_verification.jl) the
# dense/full/reference path already uses -- that function only needs
# `(r, f, kkt_resid)`, `obj.dPsi!`, `W`, `nStatus`, so the resulting `verify`
# NamedTuple is DIRECTLY comparable (same fields, same formulas) to the
# reference path's, exactly what task §11's decisive equivalence test needs.
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :reduced_homogeneous_dual_contraction) || error("reduced_operator_verification_2026-08-01.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :verify_namedtuple_from_operator) || error("reduced_operator_verification_2026-08-01.jl requires operator_verification.jl to be included first.")

"""
    verify_inner_solution_reduced_profiled!(zeta, beta, cf, ctx, θ_full, layout, obj, W) -> (r=.., f=.., g_beta=.., kkt_resid=..)

Independently recomputes `r = -zeta*1 - Gred*beta`, the exact objective, and
`g_beta` from the REDUCED forward/transpose kernels -- same contract as
`verify_inner_solution_operator_unrestricted!`, so its output feeds
`verify_namedtuple_from_operator` unchanged.
"""
function verify_inner_solution_reduced_profiled!(zeta::Float64, beta::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector, layout::ProfiledEconomicMomentLayout, obj, W::Int)
    length(beta) == layout.total_reduced_economic_moments ||
        error("verify_inner_solution_reduced_profiled!: length(beta)=$(length(beta)) != total_reduced_economic_moments=$(layout.total_reduced_economic_moments)")

    t = reduced_homogeneous_dual_contraction(beta, cf, ctx, θ_full, layout)
    r = -zeta .- t

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_beta = zeros(layout.total_reduced_economic_moments)
    B = zeros(cf.D, cf.D_dest); Tslot = zeros(cf.D_dest)
    reduced_homogeneous_transpose_contraction!(g_beta, dPsi_r, cf, ctx, θ_full, layout, B, Tslot)
    g_beta .*= -(1.0 / W)

    return (r = r, f = f, g_beta = g_beta, kkt_resid = maximum(abs, g_beta))
end
