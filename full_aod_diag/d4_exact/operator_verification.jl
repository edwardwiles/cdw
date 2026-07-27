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

isdefined(Main, :CMLookupState) || include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))

"""
    verify_inner_solution_operator_cmmeanzc!(zeta, lambda, cf, zc_op, zc_layout, nu_vec, L, nO,
        origins, refIndex1, bins, R, obj, W) -> NamedTuple

port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26 Phase A item 8: CM+ZC
analogue of `verify_inner_solution_operator_originzc!`, extended to `G=[E|Z|C]` (economic + Z
restriction + CM-grid). Reuses the SAME free-function kernels `CMMeanZCOperatorState`'s own FG
callable calls (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`/
`build_weighted_histogram!`/`cumulative_backward_gradient!`, cm_lookup_kernels.jl -- unchanged,
already independently validated) against FRESH scratch buffers allocated here, never touching the
live `CMMeanZCOperatorState`'s own `st.arg0`/`st.hist_h`/etc -- same independence contract as the
origin-ZC verifier. `lambda = [lambda_E; lambda_mean; lambda_pair; lambda_cm]`, matching
`CMMeanZCOperatorState`'s own `x` layout (minus `zeta`).
"""
function verify_inner_solution_operator_cmmeanzc!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, zc_op::ZCRestrictionOperator, zc_layout, nu_vec::AbstractVector{Float64},
        L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
        R::Union{Nothing,Matrix{Float64}}, obj, W::Int)
    ncore1 = cf.oci - 1
    ncm = nO * L
    λ_E = @view lambda[1:ncore1]
    λ_mean = @view lambda[ncore1+1:ncore1+n_mean(zc_op)]
    λ_pair = @view lambda[ncore1+n_mean(zc_op)+1:ncore1+n_mean(zc_op)+n_pair(zc_op)]
    λ_cm = @view lambda[ncore1+n_mean(zc_op)+n_pair(zc_op)+1:ncore1+n_mean(zc_op)+n_pair(zc_op)+ncm]

    econ_ws = economic_operator_workspace(cf)
    zc_ws = ZCRestrictionWorkspace(zc_op)
    refresh_zc_targets!(zc_ws, zc_op, zc_layout, nu_vec)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, econ_ws)
    r .-= econ_buf
    restriction_forward!(r, λ_mean, λ_pair, zc_op, zc_ws)

    D_bins = size(bins, 2)
    nbins = L + 1
    λmat_stored = reshape(λ_cm, nO, L)
    λmat_block = zeros(nO, L)
    apply_contrast!(λmat_block, λmat_stored, R)
    λmat_ext = zeros(nO, L + 1)
    suffix_sums!(λmat_ext, λmat_block)
    cm_contrib = zeros(W)
    cumulative_forward_contribution!(cm_contrib, bins, refIndex1, origins, λmat_ext)
    r .-= cm_contrib

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)
    g_mean = zeros(n_mean(zc_op)); g_pair = zeros(n_pair(zc_op))
    restriction_transpose!(g_mean, g_pair, dPsi_r, zc_op, zc_ws)

    hist_partials = [zeros(D_bins, nbins)]
    hist_h = zeros(D_bins, nbins)
    build_weighted_histogram!(hist_h, hist_partials, bins, dPsi_r, D_bins, nbins)
    Hpre = zeros(D_bins, L)
    g_block = zeros(nO, L)
    cumulative_backward_gradient!(g_block, Hpre, hist_h, refIndex1, origins, L, W)
    g_stored = zeros(nO, L)
    apply_contrast!(g_stored, g_block, R)

    g_lambda = vcat(g_E, g_mean, g_pair, vec(g_stored))
    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda))
end

"""
    verify_inner_solution_operator_cm!(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R, obj, W) -> NamedTuple

shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: flexible-CM
analogue of `verify_inner_solution_operator_originzc!`/`_cmmeanzc!`, extended to `G=[E|C]`
(economic + CM-grid ONLY -- no ZC mean/pair restriction block, matching `CMLookupState`'s own
`x = [zeta; lambda_E; lambda_cm]` layout, cm_lookup_kernels.jl). Reuses the SAME free-function
kernels (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`/
`build_weighted_histogram!`/`cumulative_backward_gradient!`) `CMMeanZCOperatorState`'s own verifier
above calls, against FRESH scratch buffers allocated here -- same independence contract (never
touches a live `CMLookupState`'s own `st.arg0`/`st.hist_h`/etc).
"""
function verify_inner_solution_operator_cm!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int,
        bins::Matrix{<:Unsigned}, R::Union{Nothing,Matrix{Float64}}, obj, W::Int)
    ncore1 = cf.oci - 1
    ncm = nO * L
    λ_E = @view lambda[1:ncore1]
    λ_cm = @view lambda[ncore1+1:ncore1+ncm]
    length(lambda) == ncore1 + ncm || error("verify_inner_solution_operator_cm!: length(lambda)=$(length(lambda)) != ncore1+ncm=$(ncore1+ncm)")

    econ_ws = economic_operator_workspace(cf)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, econ_ws)
    r .-= econ_buf

    D_bins = size(bins, 2)
    nbins = L + 1
    λmat_stored = reshape(λ_cm, nO, L)
    λmat_block = zeros(nO, L)
    apply_contrast!(λmat_block, λmat_stored, R)
    λmat_ext = zeros(nO, L + 1)
    suffix_sums!(λmat_ext, λmat_block)
    cm_contrib = zeros(W)
    cumulative_forward_contribution!(cm_contrib, bins, refIndex1, origins, λmat_ext)
    r .-= cm_contrib

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)

    hist_partials = [zeros(D_bins, nbins)]
    hist_h = zeros(D_bins, nbins)
    build_weighted_histogram!(hist_h, hist_partials, bins, dPsi_r, D_bins, nbins)
    Hpre = zeros(D_bins, L)
    g_block = zeros(nO, L)
    cumulative_backward_gradient!(g_block, Hpre, hist_h, refIndex1, origins, L, W)
    g_stored = zeros(nO, L)
    apply_contrast!(g_stored, g_block, R)

    g_lambda = vcat(g_E, vec(g_stored))
    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda))
end

isdefined(Main, :CMFrechetLookupState) || include(joinpath(@__DIR__, "cm_frechet_lookup_kernels.jl"))

"""
    verify_inner_solution_operator_cm_frechet!(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R,
        level_targets, obj, W) -> NamedTuple

shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: common-Frechet
analogue of `verify_inner_solution_operator_cm!`, extended to `G=[E|C|Level]` (economic + CM-grid +
the common-level anchor block that turns flexible CM into fixed Frechet). This is the genuinely NEW
verifier of the 4 built this session -- the level block has no existing verification-side
precedent, only a production FG kernel (`CMFrechetLookupState`'s callable, `cm_frechet_lookup_
kernels.jl`) to independently re-derive against, using FRESH scratch instead of a live `st`'s own
`arg0`/`arg1`/`Hpre`/etc. `lambda = [lambda_E; lambda_cm; lambda_level]`, matching
`CMFrechetLookupState`'s own `x = [zeta; lambda_core; lambda_cm; lambda_level]` layout.
"""
function verify_inner_solution_operator_cm_frechet!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int,
        bins::Matrix{<:Unsigned}, R::Union{Nothing,Matrix{Float64}}, level_targets::Vector{Float64},
        obj, W::Int)
    D_bins = size(bins, 2)
    ncore1 = cf.oci - 1
    ncm = nO * L
    length(lambda) == ncore1 + ncm + L ||
        error("verify_inner_solution_operator_cm_frechet!: length(lambda)=$(length(lambda)) != ncore1+ncm+L=$(ncore1+ncm+L)")
    λ_E = @view lambda[1:ncore1]
    λ_cm = @view lambda[ncore1+1:ncore1+ncm]
    λ_level = @view lambda[ncore1+ncm+1:ncore1+ncm+L]

    econ_ws = economic_operator_workspace(cf)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, λ_E, cf, econ_ws)
    r .-= econ_buf

    nbins = L + 1
    λmat_stored = reshape(λ_cm, nO, L)
    λmat_block = zeros(nO, L)
    apply_contrast!(λmat_block, λmat_stored, R)
    λmat_ext = zeros(nO, L + 1)
    suffix_sums!(λmat_ext, λmat_block)
    cm_contrib = zeros(W)
    cumulative_forward_contribution!(cm_contrib, bins, refIndex1, origins, λmat_ext)
    r .-= cm_contrib

    invsqrtD = 1.0 / sqrt(D_bins)
    P_level = zeros(L + 1)
    frechet_level_suffix_sums!(P_level, λ_level)
    level_contrib = zeros(W)
    frechet_level_forward_sum!(level_contrib, bins, D_bins, P_level)
    const_term = 0.0
    @inbounds for l in 1:L
        const_term += λ_level[l] * level_targets[l]
    end
    @inbounds for s in 1:W
        r[s] -= invsqrtD * level_contrib[s] - const_term
    end

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    sum_dPsi = sum(dPsi_r)
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)

    hist_partials = [zeros(D_bins, nbins)]
    hist_h = zeros(D_bins, nbins)
    build_weighted_histogram!(hist_h, hist_partials, bins, dPsi_r, D_bins, nbins)
    Hpre = zeros(D_bins, L)
    prefix_sums!(Hpre, hist_h, L)

    g_block = zeros(nO, L)
    cumulative_backward_gradient_from_prefix!(g_block, Hpre, refIndex1, origins, L, W)
    g_stored = zeros(nO, L)
    apply_contrast!(g_stored, g_block, R)

    g_level = zeros(L)
    frechet_level_backward_gradient!(g_level, Hpre, D_bins, L, W, invsqrtD, level_targets, sum_dPsi)

    g_lambda = vcat(g_E, vec(g_stored), g_level)
    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda))
end

"""
    verify_inner_solution_operator_unrestricted!(zeta, lambda, cf, obj, W) -> NamedTuple

shared-FG-verification-and-A-gradient release (2026-07-27), Phase A continuation: unrestricted
analogue of `verify_inner_solution_operator_originzc!`/`_cmmeanzc!`/`_cm!`, for `G=E` ONLY -- no
restriction block at all (the unrestricted family has none). The smallest of the 4 remaining
verifiers by construction: unrestricted's own production FG (Addendum Part A) is already fully
`economic_forward!`/`economic_transpose!`-based, so this is genuinely the same math, just
independently recomputed against fresh scratch instead of the live FG callback's own `st.arg0`/
`st.arg1`.
"""
function verify_inner_solution_operator_unrestricted!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, obj, W::Int)
    ncore1 = cf.oci - 1
    length(lambda) == ncore1 || error("verify_inner_solution_operator_unrestricted!: length(lambda)=$(length(lambda)) != ncore1=$ncore1 (unrestricted has no restriction block)")
    ws = economic_operator_workspace(cf)

    r = fill(-zeta, W)
    econ_buf = zeros(W)
    economic_forward!(econ_buf, lambda, cf, ws)
    r .-= econ_buf

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    g_lambda = zeros(ncore1)
    economic_transpose!(g_lambda, dPsi_r, cf, ws)
    g_lambda .*= -(1.0 / W)

    record_operator_verification!()
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda))
end
