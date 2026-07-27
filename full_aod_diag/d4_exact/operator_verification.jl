# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 2 §9: operator-based post-solve
# verification -- independently recomputes r = -zeta*1 - G*lambda, the exact objective, and
# g_lambda = -(1/W)*G'*Psi'(r) from the SAME shared economic + family-specific restriction
# operators the FG callback uses, WITHOUT reading dense obj.H/G.
#
# SCOPE (honest): this file implements and validates `verify_inner_solution_operator_originzc!`
# for origin-ZC ONLY -- the smallest, newest-operator family, built as a genuine, gated proof that
# operator-based verification is achievable with the SAME shared operators this branch already
# built and validated (economic_forward!/economic_transpose!, restriction_forward!/
# restriction_transpose!). Extending this to CM+ZC/flexible-CM/common-Frechet (whose own
# verification today reads dense obj.H via `obj(inner_x, constr=...)` and `CS.select_G_from_H`,
# and whose CM-block verification also still needs the `skip_cm_fill_ref`-toggled dense CM-column
# fill) is NOT done in this pass -- see docs/OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md for
# the explicit remaining-work list. `skip_cm_fill_ref` itself is NOT removed by this file (task
# §10's ask) -- that requires the CM/Frechet family verification to also go operator-based first,
# which this file does not attempt.
#
# "Independent" here means: recomputes from `cf`/`op`/`layout` (immutable/campaign-level state)
# with FRESH scratch buffers, never touching the live FG callback's own `st.arg0`/`st.arg1`/etc --
# so a bug that corrupted `st`'s own scratch would NOT be silently reproduced by this check.
# ================================================================================================

isdefined(Main, :economic_forward!) || include(joinpath(@__DIR__, "economic_operator.jl"))
isdefined(Main, :ZCRestrictionOperator) || include(joinpath(@__DIR__, "zc_restriction_operator.jl"))
isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))

"""
    verify_inner_solution_operator_originzc!(zeta, lambda, cf, op, layout, nu_full, obj, W) -> NamedTuple

Independently recomputes, from operator state only (no dense `obj.H`/`G` read):
- `r = -zeta*1 - E*lambda_E - R*lambda_R` (economic_forward!/restriction_forward!);
- the exact objective `f = mean(Psi(r)) + zeta`;
- the full dual gradient `g_lambda = -(1/W)*[E;R]'*Psi'(r)` (economic_transpose!/restriction_transpose!);
- the KKT residual `max|g_lambda|` (a stationarity check at a converged lambda* -- `g_lambda` should
  be ~0 at the true optimum, mirroring `kkt_residual_blas`'s own role in the dense verifier, just
  computed from the operator's own transpose instead of a dense `G'm_weights` product).
Uses FRESH scratch (own `EconomicFGWorkspace`/`ZCRestrictionWorkspace`), independent of whatever
`OriginZCOperatorState` the live FG callback used.
"""
function verify_inner_solution_operator_originzc!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, op::ZCRestrictionOperator, layout, nu_full::AbstractVector{Float64},
        obj, W::Int)
    ncore1 = cf.oci - 1
    λ_E = @view lambda[1:ncore1]
    λ_mean = @view lambda[ncore1+1:ncore1+n_mean(op)]
    λ_pair = @view lambda[ncore1+n_mean(op)+1:ncore1+n_mean(op)+n_pair(op)]

    ws = economic_operator_workspace(cf)
    zc_ws = ZCRestrictionWorkspace(op)
    refresh_zc_targets!(zc_ws, op, layout, nu_full)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, ws)
    r .-= econ_buf
    restriction_forward!(r, λ_mean, λ_pair, op, zc_ws)

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, ws)
    g_E .*= -(1.0 / W)
    g_mean = zeros(n_mean(op)); g_pair = zeros(n_pair(op))
    restriction_transpose!(g_mean, g_pair, dPsi_r, op, zc_ws)

    g_lambda = vcat(g_E, g_mean, g_pair)
    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda))
end
