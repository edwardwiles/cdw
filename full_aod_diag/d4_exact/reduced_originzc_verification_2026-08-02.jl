# ZC lane task (2026-08-02): independent post-solve verification for the REDUCED (profiled-
# economic-layout) origin-ZC inner solve, combining `verify_inner_solution_reduced_profiled!`'s
# reduced-economic contraction (reduced_operator_verification_2026-08-01.jl -- built for the
# unrestricted family, which has no restriction block) with `verify_inner_solution_operator_
# originzc!`'s restriction_forward!/restriction_transpose! terms (operator_verification.jl --
# built for the FULL/dense-economic-width origin-ZC operator path). Neither existing verifier
# covers "reduced economic width + a real ZC restriction block" -- this is that combination,
# needed because the ZC lane's own reduced origin-ZC context genuinely has both.
#
# Draw-level identity this enforces (task's own "Phase I: ZC-only outer gradient" requirement):
#     q = -zeta - t_economic - t_Z
# where t_economic is the REDUCED homogeneous economic contraction and t_Z is the SAME
# restriction_forward! every other origin-ZC path (dense or operator) already uses unchanged.
# ADDITIVE ONLY -- does not modify any existing verifier.

isdefined(Main, :reduced_homogeneous_dual_contraction) || error("reduced_originzc_verification_2026-08-02.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :restriction_forward!) || error("reduced_originzc_verification_2026-08-02.jl requires zc_restriction_operator.jl to be included first.")
isdefined(Main, :verify_namedtuple_from_operator) || error("reduced_originzc_verification_2026-08-02.jl requires operator_verification.jl to be included first.")

"""
    verify_inner_solution_reduced_originzc!(zeta, beta, lambda_mean, lambda_pair, cf, ctx, θ_full,
        layout, zc_op, zc_layout, nu_full, obj, W) -> (r=.., f=.., g_beta=.., g_mean=.., g_pair=..,
        kkt_resid=..)

`beta` is the reduced economic dual block (length `layout.total_reduced_economic_moments`,
`layout::ProfiledEconomicMomentLayout`); `lambda_mean`/`lambda_pair` are the restriction duals
(`zc_layout::MeanZCTargetLayout`, `zc_op::ZCRestrictionOperator` -- same objects
`build_originzc_augmented_obj`/`build_originzc_core_hess_ctx` already build). Returns the SAME
field set `verify_namedtuple_from_operator` expects (mirrors both source verifiers' own contract).
"""
function verify_inner_solution_reduced_originzc!(zeta::Float64, beta::AbstractVector{Float64},
        lambda_mean::AbstractVector{Float64}, lambda_pair::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector,
        layout::ProfiledEconomicMomentLayout, zc_op::ZCRestrictionOperator, zc_layout,
        nu_full::AbstractVector{Float64}, obj, W::Int)
    length(beta) == layout.total_reduced_economic_moments ||
        error("verify_inner_solution_reduced_originzc!: length(beta)=$(length(beta)) != total_reduced_economic_moments=$(layout.total_reduced_economic_moments)")

    zc_ws = ZCRestrictionWorkspace(zc_op)
    refresh_zc_targets!(zc_ws, zc_op, zc_layout, nu_full)

    t_economic = reduced_homogeneous_dual_contraction(beta, cf, ctx, θ_full, layout)
    r = -zeta .- t_economic
    restriction_forward!(r, lambda_mean, lambda_pair, zc_op, zc_ws)   # subtracts t_Z in place, same convention as verify_inner_solution_operator_originzc!

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_beta = zeros(layout.total_reduced_economic_moments)
    B = zeros(cf.D, cf.D_dest); Tslot = zeros(cf.D_dest)
    reduced_homogeneous_transpose_contraction!(g_beta, dPsi_r, cf, ctx, θ_full, layout, B, Tslot)
    g_beta .*= -(1.0 / W)

    g_mean = zeros(n_mean(zc_op)); g_pair = zeros(n_pair(zc_op))
    restriction_transpose!(g_mean, g_pair, dPsi_r, zc_op, zc_ws)

    g_lambda = vcat(g_beta, g_mean, g_pair)
    return (r = r, f = f, g_lambda = g_lambda, g_beta = g_beta, g_mean = g_mean, g_pair = g_pair,
        kkt_resid = maximum(abs, g_lambda))
end
