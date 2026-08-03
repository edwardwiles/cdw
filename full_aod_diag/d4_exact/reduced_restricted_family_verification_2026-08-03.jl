# ============================================================================
# profiled-inner-readiness-2026-08-03, task §7: independent post-solve verification for the
# remaining 3 REDUCED restricted-family evaluators (flexible_CM, common_frechet, CM+ZC/cm_meanzc).
# origin_ZC already has `verify_inner_solution_reduced_originzc!`
# (reduced_originzc_verification_2026-08-02.jl) -- this file follows that EXACT pattern: combine
# `reduced_homogeneous_dual_contraction`/`_transpose_contraction!` (REDUCED economic width) with
# the family-specific restriction terms, reusing the SAME free-function kernels
# (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`/`build_weighted_histogram!`/
# `cumulative_backward_gradient!`/`frechet_level_*`/`restriction_forward!`/`restriction_transpose!`)
# `_verify_inner_solution_operator_cm_core`/`verify_inner_solution_operator_cmmeanzc!`
# (operator_verification.jl) already use for FULL's dense economic width -- the CM-grid/level/ZC
# restriction math does not depend on the economic layout's width at all (same insight the
# origin-ZC verifier already exploited), so this is genuinely ONE set of formulas, reused, not
# duplicated. FRESH scratch throughout -- never touches a live FG callback's own state.
#
# ONE shared core (`_verify_inner_solution_reduced_cm_core`) behind both
# `verify_inner_solution_reduced_cm!` (flexible_CM, level_targets=nothing) and
# `verify_inner_solution_reduced_cm_frechet!` (common_frechet, level_targets given), mirroring
# `_verify_inner_solution_operator_cm_core`'s own FULL-side split exactly.
# `verify_inner_solution_reduced_cmzc!` (CM+ZC/cm_meanzc) additionally combines the ZC mean/pair
# restriction terms (from `verify_inner_solution_operator_cmmeanzc!`) with the CM-grid terms --
# genuinely THREE blocks (reduced-econ + ZC + CM-grid), no level.
#
# Lambda layouts match each family's own live FG callback x-layout exactly (verified by direct
# code read, not assumed):
#   flexible_CM   (profiled_reduced_lookup_kernels_2026-08-02.jl):        [econ(n_econ); cm(ncm)]
#   common_frechet(profiled_reduced_frechet_lookup_kernels_2026-08-02.jl):[econ(n_econ); cm(ncm_cm); level(ncm_level)]
#   cm_meanzc/CM+ZC(profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl):[econ(n_econ); mean(n_mean); pair(n_pair); cm(ncm)]
# (all exclude zeta, matching `verify_inner_solution_operator_cm!`/`_cmmeanzc!`'s own `lambda` convention.)
# ============================================================================

isdefined(Main, :reduced_homogeneous_dual_contraction) || error("reduced_restricted_family_verification_2026-08-03.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first.")
isdefined(Main, :apply_contrast!) || error("reduced_restricted_family_verification_2026-08-03.jl requires cm_lookup_kernels.jl to be included first.")
isdefined(Main, :restriction_forward!) || error("reduced_restricted_family_verification_2026-08-03.jl requires zc_restriction_operator.jl to be included first.")
isdefined(Main, :verify_namedtuple_from_operator) || error("reduced_restricted_family_verification_2026-08-03.jl requires operator_verification.jl to be included first.")

"""
    _verify_inner_solution_reduced_cm_core(zeta, lambda, cf, ctx, θ_full, layout, L, nO, origins,
        refIndex1, bins, R, obj, W, level_targets) -> NamedTuple

Shared implementation behind `verify_inner_solution_reduced_cm!` (flexible_CM, `level_targets=nothing`)
and `verify_inner_solution_reduced_cm_frechet!` (common_frechet, `level_targets` a `Vector{Float64}`).
Mirrors `_verify_inner_solution_operator_cm_core` (operator_verification.jl) exactly, with the dense
`economic_forward!`/`economic_transpose!` calls replaced by `reduced_homogeneous_dual_contraction`/
`reduced_homogeneous_transpose_contraction!` -- the ONLY difference between the reduced and FULL
verifiers, matching this whole task's own design constraint (don't invent new architecture).
"""
function _verify_inner_solution_reduced_cm_core(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector, layout::ProfiledEconomicMomentLayout,
        L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
        R::Union{Nothing,Matrix{Float64}}, obj, W::Int, level_targets::Union{Nothing,Vector{Float64}})
    D_bins = size(bins, 2)
    n_econ = layout.total_reduced_economic_moments
    ncm = nO * L
    expected_len = level_targets === nothing ? n_econ + ncm : n_econ + ncm + L
    length(lambda) == expected_len ||
        error("_verify_inner_solution_reduced_cm_core: length(lambda)=$(length(lambda)) != expected=$(expected_len)")
    β_econ = @view lambda[1:n_econ]
    λ_cm = @view lambda[n_econ+1:n_econ+ncm]
    λ_level = level_targets === nothing ? nothing : (@view lambda[n_econ+ncm+1:n_econ+ncm+L])

    t_economic = reduced_homogeneous_dual_contraction(β_econ, cf, ctx, θ_full, layout)
    r = -zeta .- t_economic

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
    if level_targets !== nothing
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
    end

    Psi_r = similar(r); obj.Psi!(Psi_r, r)
    f = sum(Psi_r) / W + zeta

    dPsi_r = similar(r); obj.dPsi!(dPsi_r, r)
    sum_dPsi = sum(dPsi_r)
    g_econ = zeros(n_econ)
    B = zeros(cf.D, cf.D_dest); Tslot = zeros(cf.D_dest)
    reduced_homogeneous_transpose_contraction!(g_econ, dPsi_r, cf, ctx, θ_full, layout, B, Tslot)
    g_econ .*= -(1.0 / W)

    hist_partials = [zeros(D_bins, nbins)]
    hist_h = zeros(D_bins, nbins)
    build_weighted_histogram!(hist_h, hist_partials, bins, dPsi_r, D_bins, nbins)
    Hpre = zeros(D_bins, L)
    g_block = zeros(nO, L)
    if level_targets === nothing
        cumulative_backward_gradient!(g_block, Hpre, hist_h, refIndex1, origins, L, W)
    else
        prefix_sums!(Hpre, hist_h, L)
        cumulative_backward_gradient_from_prefix!(g_block, Hpre, refIndex1, origins, L, W)
    end
    g_stored = zeros(nO, L)
    apply_contrast!(g_stored, g_block, R)

    g_level_local = nothing
    g_lambda = if level_targets === nothing
        vcat(g_econ, vec(g_stored))
    else
        g_level = zeros(L)
        frechet_level_backward_gradient!(g_level, Hpre, D_bins, L, W, invsqrtD, level_targets, sum_dPsi)
        g_level_local = g_level
        vcat(g_econ, vec(g_stored), g_level)
    end
    kkt_resid_E = isempty(g_econ) ? 0.0 : maximum(abs, g_econ)
    kkt_resid_cm = maximum(abs, g_stored)
    kkt_resid_level = g_level_local === nothing ? nothing : maximum(abs, g_level_local)
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_cm = kkt_resid_cm, kkt_resid_level = kkt_resid_level)
end

"""
    verify_inner_solution_reduced_cm!(zeta, lambda, cf, ctx, θ_full, layout, L, nO, origins,
        refIndex1, bins, R, obj, W) -> NamedTuple

Independent REDUCED-economic-layout verifier for flexible_CM. `lambda = [beta_econ(n_econ);
lambda_cm(ncm)]`, matching `ReducedCMLookupState`'s own `x = [zeta; econ; cm]` layout (no gravity).
"""
function verify_inner_solution_reduced_cm!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector, layout::ProfiledEconomicMomentLayout,
        L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
        R::Union{Nothing,Matrix{Float64}}, obj, W::Int)
    return _verify_inner_solution_reduced_cm_core(zeta, lambda, cf, ctx, θ_full, layout, L, nO,
        origins, refIndex1, bins, R, obj, W, nothing)
end

"""
    verify_inner_solution_reduced_cm_frechet!(zeta, lambda, cf, ctx, θ_full, layout, L, nO, origins,
        refIndex1, bins, R, level_targets, obj, W) -> NamedTuple

Independent REDUCED-economic-layout verifier for common_frechet. `lambda = [beta_econ(n_econ);
lambda_cm(ncm_cm); lambda_level(ncm_level)]`, matching `ReducedCMFrechetLookupState`'s own
`x = [zeta; econ; cm; level]` layout (no gravity).
"""
function verify_inner_solution_reduced_cm_frechet!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector, layout::ProfiledEconomicMomentLayout,
        L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
        R::Union{Nothing,Matrix{Float64}}, level_targets::Vector{Float64}, obj, W::Int)
    return _verify_inner_solution_reduced_cm_core(zeta, lambda, cf, ctx, θ_full, layout, L, nO,
        origins, refIndex1, bins, R, obj, W, level_targets)
end

"""
    verify_inner_solution_reduced_cmzc!(zeta, lambda, cf, ctx, θ_full, layout, zc_op, zc_layout,
        nu_full, L, nO, origins, refIndex1, bins, R, obj, W) -> NamedTuple

Independent REDUCED-economic-layout verifier for CM+ZC (cm_meanzc): reduced-economic + ZC mean/pair
restriction (`restriction_forward!`/`restriction_transpose!`, same as origin_ZC's own verifier) +
CM-grid restriction (same core loop as `verify_inner_solution_reduced_cm!` above) -- genuinely THREE
blocks, no level. `lambda = [beta_econ(n_econ); lambda_mean(n_mean); lambda_pair(n_pair);
lambda_cm(ncm)]`, matching `ReducedCMMeanZCOperatorState`'s own `x = [zeta; econ; mean; pair; cm]`
layout (no gravity).
"""
function verify_inner_solution_reduced_cmzc!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, ctx, θ_full::AbstractVector, layout::ProfiledEconomicMomentLayout,
        zc_op::ZCRestrictionOperator, zc_layout, nu_full::AbstractVector{Float64},
        L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int, bins::Matrix{<:Unsigned},
        R::Union{Nothing,Matrix{Float64}}, obj, W::Int)
    D_bins = size(bins, 2)
    n_econ = layout.total_reduced_economic_moments
    ncm = nO * L
    expected_len = n_econ + n_mean(zc_op) + n_pair(zc_op) + ncm
    length(lambda) == expected_len ||
        error("verify_inner_solution_reduced_cmzc!: length(lambda)=$(length(lambda)) != expected=$expected_len")
    β_econ = @view lambda[1:n_econ]
    λ_mean = @view lambda[n_econ+1:n_econ+n_mean(zc_op)]
    λ_pair = @view lambda[n_econ+n_mean(zc_op)+1:n_econ+n_mean(zc_op)+n_pair(zc_op)]
    λ_cm = @view lambda[n_econ+n_mean(zc_op)+n_pair(zc_op)+1:n_econ+n_mean(zc_op)+n_pair(zc_op)+ncm]

    zc_ws = ZCRestrictionWorkspace(zc_op)
    refresh_zc_targets!(zc_ws, zc_op, zc_layout, nu_full)

    t_economic = reduced_homogeneous_dual_contraction(β_econ, cf, ctx, θ_full, layout)
    r = -zeta .- t_economic
    restriction_forward!(r, λ_mean, λ_pair, zc_op, zc_ws)

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
    g_econ = zeros(n_econ)
    B = zeros(cf.D, cf.D_dest); Tslot = zeros(cf.D_dest)
    reduced_homogeneous_transpose_contraction!(g_econ, dPsi_r, cf, ctx, θ_full, layout, B, Tslot)
    g_econ .*= -(1.0 / W)
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

    g_lambda = vcat(g_econ, g_mean, g_pair, vec(g_stored))
    kkt_resid_E = isempty(g_econ) ? 0.0 : maximum(abs, g_econ)
    kkt_resid_mean = isempty(g_mean) ? 0.0 : maximum(abs, g_mean)
    kkt_resid_pair = isempty(g_pair) ? 0.0 : maximum(abs, g_pair)
    kkt_resid_cm = maximum(abs, g_stored)
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_mean = kkt_resid_mean, kkt_resid_pair = kkt_resid_pair,
            kkt_resid_cm = kkt_resid_cm)
end
