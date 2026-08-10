# ============================================================================
# CM+ZC-CROSS (2026-08-09): full K_pair^2 cross-power extension of the COMMON-MARGINALS
# mean/pairwise-zero-covariance restriction block -- the CM+ZC analog of OZC-CROSS
# (cm_originzc_cross_moments.jl), which this mirrors closely. See cm_meanzc_moments.jl for the base
# family this extends.
#
# Restriction: for every unordered origin pair (o,p), o<p (`packed_pair_index`, UNCHANGED canonical
# order), and every ORDERED level pair (k1,k2) in {1,...,K_pair}x{1,...,K_pair} (K_pair^2 total per
# origin pair, including the pre-existing diagonal k1=k2):
#     E_F[z_o(w)^k1 * z_p(w)^k2] = nu_k1 * nu_k2
# No new outer parameters -- nu_k is the SAME scalar-per-level quantity CM+ZC's mean block already
# defines (`n_eta(SharedByPowerCrossLayout) = K_mean`, UNCHANGED); only the pair-restriction COUNT
# (K_pair^2*npair instead of K_pair*npair) and the cross-term math change.
#
# WHAT IS REUSED COMPLETELY UNCHANGED (confirmed by direct reading, 2026-08-09 -- the same
# column-count-agnostic property OZC-CROSS already established and relies on):
#   * `build_raw_cross_pair_matrix_levels` / `cross_pair_level_index` (cm_originzc_cross_moments.jl)
#     -- reused VERBATIM. Those build origin-indexed raw feature columns from the per-level Frechet
#     powers; nothing in them knows or cares whether nu is shared or origin-specific.
#   * `ZCRestrictionOperator` (zc_restriction_operator.jl) -- keyed off `length(Zpairraw_all)`, so
#     `n_pair(op)` becomes `K_pair^2*npair` automatically; `refresh_zc_targets!` calls
#     `pair_targets(layout, nu, klin, D)` and therefore dispatches to `SharedByPowerCrossLayout`'s
#     own method automatically.
#   * `CMMeanZCOperatorState` (cm_meanzc_lookup_kernels.jl) -- slices lambda by `n_mean(op)`/
#     `n_pair(op)`, never by K_pair's meaning.
#   * The CM-grid block: column layout is `[econ | mean | pair | CM-grid | gravity]`
#     (cm_meanzc_moments.jl:36-43) and every CM offset derives from `aug.n_pair`/`n_pair(op)`, so a
#     K_pair^2 pair block auto-shifts the CM columns correctly. The pair block is mathematically and
#     index-wise independent of the CM grid.
#
# WHAT IS GENUINELY NEW MATH (do not read this as a copy of the base family's formula): the
# nu-gradient. The base family writes ONE component per level (`out[k] = mean_m*total`, using
# `d_pair_dnu = -2nu`, cm_meanzc_moments.jl:589-608) because level k's moment columns depend on
# nu_k ONLY -- the outer Jacobian is block-diagonal in k. That is FALSE for the cross grid: the
# `(k1,k2)` pair block depends on nu_{k1} AND nu_{k2}, so the derivative must ACCUMULATE
# `-nu_{k2}` into slot `k1` and `-nu_{k1}` into slot `k2` across every `klin`. See
# `d_delta_dual_d_nu_cross_vec` below.
#
# :operator ONLY, by construction -- exactly like OZC-CROSS, and for the same reason: production
# CM+ZC already defaults to `moment_representation=:operator` (cm_meanzc_production.jl:185) and
# CLAUDE.md forbids adding a `:dense_reference` fallback nobody would use. A caller asking for
# `:dense_reference` gets a hard error, not a silent substitution.
# ============================================================================

using LinearAlgebra: dot

isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
isdefined(Main, :build_cm_meanzc_augmented_obj) || include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
isdefined(Main, :cross_pair_level_index) || include(joinpath(@__DIR__, "cm_originzc_cross_moments.jl"))
isdefined(Main, :SharedByPowerCrossLayout) || include(joinpath(@__DIR__, "cm_meanzc_cross_target_layout.jl"))

"""
    n_meanzc_cross_moments(D::Int, K_mean::Int, K_pair::Int) -> Int

Total new inner moments for CM+ZC-CROSS: `K_mean*D` (mean block, UNCHANGED formula/meaning) plus
`K_pair^2 * D(D-1)/2` (cross-pair block). Contrast `n_meanzc_moments`'s `K_pair*D(D-1)/2`
(diagonal-only, base family). Excludes the CM-grid block (`ncm`), exactly as `n_meanzc_moments`
does.
"""
function n_meanzc_cross_moments(D::Int, K_mean::Int, K_pair::Int)
    K_mean >= 1 || error("n_meanzc_cross_moments: K_mean must be >= 1, got $K_mean")
    0 <= K_pair <= K_mean || error("n_meanzc_cross_moments: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
    return K_mean * D + K_pair^2 * div(D * (D - 1), 2)
end

"""
    build_cm_meanzc_cross_augmented_obj(ctx, CS; L, K_mean, include_truncated_moment, K_pair,
                                        contrasts, meanzc_basis, probs, refIndex1, aml) -> NamedTuple

CM+ZC-CROSS analog of `build_cm_meanzc_augmented_obj`'s `:operator` branch (cm_meanzc_moments.jl).
Everything about the CM-grid block (`precalc_common_marginals_cdf`, `ncm`, `n_families`,
`ncm_cdf`/`ncm_pow`) is byte-identical to that function; the only differences are:

  1. `Zpairraw_all` is the `K_pair^2` cross-power grid (`build_raw_cross_pair_matrix_levels`,
     cm_originzc_cross_moments.jl, reused verbatim) instead of the `K_pair` diagonal levels, so
     `n_pair = K_pair^2*npair`;
  2. a `layout::SharedByPowerCrossLayout` field is returned on the `aug` NamedTuple, which
     `build_cm_meanzc_bin_ctx` picks up (`hasproperty(aug, :layout)`) to construct the ZC operator's
     layout instead of hardcoding `SharedByPowerLayout` -- the ONE edit this family needs in
     otherwise-unmodified shared code.

`aml` (Variant D focal `k*=sigma-1` mean-row omission): identical mechanism to the base family's
own. It touches ONLY the mean block -- `n_mean` becomes `mean_offset_from_aml(aml)[end]` (a ROW
count; see the base function's own comment at cm_meanzc_moments.jl:510-518 for why this must NOT be
`aml.n_eta_active`, which is a completely different scale under a SHARED layout: `n_eta = K_mean`,
not `K_mean*D`). The PAIR block is COMPLETELY UNCHANGED by `aml`, exactly as for the base family
and for OZC-CROSS.

`moment_representation` is not a kwarg here: this family is `:operator`-only (see file header).
"""
function build_cm_meanzc_cross_augmented_obj(ctx, CS; L::Int, K_mean::Int, include_truncated_moment::Bool,
                                              K_pair::Int, contrasts::Symbol = :anchored,
                                              meanzc_basis::Symbol = :direct,
                                              probs::Union{Nothing,AbstractVector{Float64}} = nothing,
                                              refIndex1::Int = ctx.γ.refIndex1,
                                              aml::Union{Nothing,ActiveMeanLayout} = nothing)
    K_mean >= 1 || error("build_cm_meanzc_cross_augmented_obj: K_mean must be >= 1, got $K_mean")
    1 <= K_pair <= K_mean || error("build_cm_meanzc_cross_augmented_obj: K_pair must satisfy 1 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean (K_pair=0 means 'no pair block at all' -- use the base family, build_cm_meanzc_augmented_obj, for that)")
    meanzc_basis in (:direct, :anchored) || error("build_cm_meanzc_cross_augmented_obj: meanzc_basis must be :direct or :anchored, got $meanzc_basis")

    layout = SharedByPowerCrossLayout(K_mean, K_pair)
    aml === nothing || aml.base isa SharedByPowerCrossLayout ||
        error("build_cm_meanzc_cross_augmented_obj: aml.base must be a SharedByPowerCrossLayout, got $(typeof(aml.base))")
    aml === nothing || (aml.base.K_mean == K_mean && aml.base.K_pair == K_pair) ||
        error("build_cm_meanzc_cross_augmented_obj: aml.base (K_mean=$(aml.base.K_mean),K_pair=$(aml.base.K_pair)) disagrees with (K_mean=$K_mean,K_pair=$K_pair)")

    obj0 = ctx.obj
    ncore_econ = obj0.d
    # CM-grid block: byte-identical to build_cm_meanzc_augmented_obj's own construction.
    σHat = include_truncated_moment ? ctx.σ : nothing
    μHat_cm = include_truncated_moment ? ctx.μHat : nothing
    CM, z, origins = precalc_common_marginals_cdf(ctx.U, refIndex1, L; include_truncated_moment = include_truncated_moment,
                                                   σHat = σHat, μHat = μHat_cm, contrasts = contrasts, probs = probs)
    ncm = size(CM, 2)
    @assert ncm == n_cm_moments(ctx.D, L; include_truncated_moment = include_truncated_moment)
    nO = length(origins)
    ncm_cdf = nO * L
    ncm_pow = include_truncated_moment ? nO * L : 0
    n_families = include_truncated_moment ? 2 : 1

    # K_pair=0 in this call: only Zraw_all (the per-level Frechet-power features) is needed here --
    # the diagonal-only Zpairraw_all this function would otherwise also build is unused (this family
    # builds its OWN Zpairraw_all, from Zraw_all, via build_raw_cross_pair_matrix_levels).
    Zraw_all, _ = build_raw_mean_pair_matrix_levels(ctx.U, K_mean, 0; μ = ctx.μHat)
    Zpairraw_all = build_raw_cross_pair_matrix_levels(Zraw_all, K_pair)

    D = ctx.D
    npair = div(D * (D - 1), 2)
    n_mean_dense = K_mean * D
    n_mean = (aml !== nothing && aml.active) ? mean_offset_from_aml(aml)[end] : n_mean_dense
    n_pair = K_pair^2 * npair
    @assert n_mean_dense + n_pair == n_meanzc_cross_moments(D, K_mean, K_pair)

    d_new = ncore_econ + n_mean + n_pair + ncm
    outer_constr_index_new = obj0.outer_constr_index + n_mean + n_pair + ncm

    obj_cm = OperatorPsiBundle(δ = obj0.δ, find_smallest = obj0.find_smallest,
        γ = obj0.γ, l = obj0.l, outer_constr_index = outer_constr_index_new,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x, threshold_state = obj0.threshold_state,
        inner_loop_opt = obj0.inner_loop_opt)
    @assert obj_cm.outer_constr_index == d_new

    return (obj_cm = obj_cm, CM = CM, z = z, origins = origins, ncore = ncore_econ, ncm = ncm,
            L = L, contrasts = contrasts, refIndex1 = refIndex1,
            Zraw_all = Zraw_all, Zpairraw_all = Zpairraw_all, K_mean = K_mean, K_pair = K_pair,
            n_mean = n_mean, n_pair = n_pair, meanzc_basis = meanzc_basis,
            ncore_econ = ncore_econ, core_cf_ref = Ref{Any}(nothing), moments_skip! = nothing,
            include_truncated_moment = include_truncated_moment, n_families = n_families,
            ncm_cdf = ncm_cdf, ncm_pow = ncm_pow, aml = aml, layout = layout)
end

# ----------------------------------------------------------------------------
# Outer-gradient envelope formula, extended to the K_pair^2 grid. THIS IS THE GENUINELY NEW MATH
# for CM+ZC-CROSS (see file header): the base family's `d_delta_dual_d_nu_vec` writes one
# independent component per level; the cross grid couples levels, so we must accumulate.
# ----------------------------------------------------------------------------

"""
    d_delta_dual_d_nu_cross_vec(λstar, aug, νvec; mean_m) -> Vector{Float64}   (length K_mean)
    d_delta_dual_d_eta_nu_cross_vec(λstar, aug, νvec; mean_m) -> Vector{Float64}

CM+ZC-CROSS analog of `d_delta_dual_d_nu_vec`/`d_delta_dual_d_eta_nu_vec` (cm_meanzc_moments.jl):
the IDENTICAL envelope-derivative principle (`d(Delta_dual)/d(nu) = mean_m * Σ_j λ_j* ·
d(target_j)/d(nu)`, with `d(Delta_dual)/d(eta) = nu * d(Delta_dual)/d(nu)` by the `nu=exp(eta)`
chain rule), extended to the `K_pair^2` ordered cross-power grid.

MEAN BLOCK -- byte-identical to the base formula: level `k`'s mean columns have target `nu_k` for
every origin, `d(target)/d(nu_k)` given UNCHANGED by `d_mean_dnu_direct`/`d_mean_dnu_anchored`, and
depend on no other level.

PAIR BLOCK -- the genuine extension, and the reason this cannot be a copy of the base function.
Block `klin` has target `nu_{k1}*nu_{k2}` (`(k1,k2) = cross_pair_level_index(layout.K_pair)[klin]`)
for EVERY pair, so it contributes `-nu_{k2}*Σ_j λ_pair,klin,j*` to `d(Delta_dual)/d(nu_{k1})` AND
`-nu_{k1}*Σ_j λ_pair,klin,j*` to `d(Delta_dual)/d(nu_{k2})` -- two different slots, accumulated,
rather than the base family's single `d_pair_dnu = -2nu_k` written into slot `k`. When `k1==k2` the
two accumulations land in the SAME slot and sum to exactly `-2*nu_k*Σ_j λ*`, i.e. this formula
reduces to the base family's diagonal one term-for-term (verified numerically at D4: with K_pair=1
the only combo is (1,1)).

`Σ_j λ_pair,klin,j*` (a plain sum over pairs, not a `dot` against a per-pair derivative vector) is
correct here and ONLY here because the target is origin-free under the shared-nu layout -- the same
simplification `meanzc_fixed_contribution`'s own `νvec[k]^2*sum(λ_pair_k)` already makes for the
base family. OZC-CROSS, whose targets ARE origin-specific, necessarily uses a per-pair loop instead
(`d_delta_dual_d_eta_origin_cross_vec`, cm_originzc_cross_production.jl).
"""
function d_delta_dual_d_nu_cross_vec(λstar::AbstractVector{Float64}, aug, νvec::AbstractVector{Float64}; mean_m::Float64)
    layout = aug.layout
    layout isa SharedByPowerCrossLayout || error("d_delta_dual_d_nu_cross_vec: aug.layout must be a SharedByPowerCrossLayout, got $(typeof(layout))")
    K_mean = aug.K_mean; K_pair = aug.K_pair
    length(νvec) == K_mean || error("d_delta_dual_d_nu_cross_vec: length(νvec)=$(length(νvec)) != aug.K_mean=$K_mean")
    ncore_econ = aug.ncore_econ
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    d_mean = aug.meanzc_basis === :direct ? d_mean_dnu_direct(D) : d_mean_dnu_anchored(D, aug.refIndex1)
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    expected_len = (ncore_econ - 1) + K_mean * D + K_pair^2 * npair
    length(λstar) >= expected_len ||
        error("d_delta_dual_d_nu_cross_vec: length(λstar)=$(length(λstar)) < expected_len=$expected_len " *
              "(ncore_econ=$ncore_econ, K_mean=$K_mean, D=$D, K_pair=$K_pair, npair=$npair) -- " *
              "aug/λstar dimension mismatch, refusing to silently misindex.")
    d_nu = zeros(K_mean)
    for k in 1:K_mean
        λ_mean_k = @view λstar[mean_start+(k-1)*D : mean_start+k*D-1]
        d_nu[k] += dot(λ_mean_k, d_mean)
    end
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        λ_pair_klin = @view λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        s = sum(λ_pair_klin)
        d_nu[k1] -= νvec[k2] * s
        d_nu[k2] -= νvec[k1] * s
    end
    d_nu .*= mean_m
    return d_nu
end

function d_delta_dual_d_eta_nu_cross_vec(λstar::AbstractVector{Float64}, aug, νvec::AbstractVector{Float64}; mean_m::Float64)
    d_nu = d_delta_dual_d_nu_cross_vec(λstar, aug, νvec; mean_m = mean_m)
    return νvec .* d_nu   # chain rule, nu = exp(eta)
end

"""
    d_delta_dual_d_eta_active_and_nustar_shared_cross(λstar, aug, aml::ActiveMeanLayout, νvec_eff;
                                                       mean_m) -> (eta_grad_active, d_delta_d_nu_star)

CM+ZC-CROSS analog of `d_delta_dual_d_eta_active_and_nustar_shared` (cm_meanzc_moments.jl) --
Variant D (focal `k*=sigma-1` mean-row omission) applied to the `K_pair^2` cross grid. The mean-block
loop is copied from the base function (ragged-aware via `aml.mean_active_origins`/
`mean_offset_from_aml`; unchanged, since Variant D only ever touches the mean block). The pair-block
loop is `d_delta_dual_d_nu_cross_vec`'s cross-grid accumulation, with the ragged-aware
`pair_start0 = ncore_econ + mean_offset[end]` offset -- the pair block itself is completely
unaffected by `aml`.

THE ONE PLACE THIS DIFFERS MATERIALLY FROM THE DIAGONAL FAMILY, and the reason it is FD-checked
explicitly rather than assumed: under the shared layout `aml.dense_omit_idx == aml.kstar`, and the
focal `nu_star` sits at level `k*`, which appears in cross-pair blocks at BOTH `(k*,k2)` for every
`k2` AND `(k1,k*)` for every `k1` -- `2*K_pair - 1` distinct blocks, versus the diagonal family's
single `(k*,k*)` block. So `d_delta_d_nu_star` collects strictly more terms here. The
accumulate-into-both-slots loop below handles that automatically (each block contributes to whichever
of `k1`/`k2` equals `k*`, and to BOTH when `k1==k2==k*`), but the D4 FD gate
`verify_cmzc_cross_gradient_d4_2026-08-09.jl` checks it numerically rather than by inspection.

`νvec_eff` is the FULL dense (length `K_mean`) shared-nu vector with the derived `nu_star` already
scattered into position `aml.kstar` (via `scatter_nu_eff`) -- pair targets always need the shared
value at every level, including the derived one.
"""
function d_delta_dual_d_eta_active_and_nustar_shared_cross(λstar::AbstractVector{Float64}, aug, aml::ActiveMeanLayout,
                                                            νvec_eff::AbstractVector{Float64}; mean_m::Float64)
    layout = aml.base
    layout isa SharedByPowerCrossLayout || error("d_delta_dual_d_eta_active_and_nustar_shared_cross: aml.base must be a SharedByPowerCrossLayout, got $(typeof(layout))")
    K_mean = aug.K_mean; K_pair = aug.K_pair
    length(νvec_eff) == K_mean || error("d_delta_dual_d_eta_active_and_nustar_shared_cross: length(νvec_eff)=$(length(νvec_eff)) != aug.K_mean=$K_mean")
    ncore_econ = aug.ncore_econ
    D = size(aug.Zraw_all[1], 2)
    npair = D * (D - 1) ÷ 2
    mean_start = ncore_econ
    mean_offset = mean_offset_from_aml(aml)
    n_mean_active = mean_offset[end]
    pair_start0 = ncore_econ + n_mean_active   # ragged-aware: active mean count, not K_mean*D
    expected_len = (ncore_econ - 1) + n_mean_active + K_pair^2 * npair
    length(λstar) >= expected_len ||
        error("d_delta_dual_d_eta_active_and_nustar_shared_cross: length(λstar)=$(length(λstar)) < expected_len=$expected_len " *
              "(ncore_econ=$ncore_econ, n_mean_active=$n_mean_active, K_pair=$K_pair, npair=$npair) -- " *
              "aug/λstar dimension mismatch, refusing to silently misindex.")
    d_nu = zeros(K_mean)
    # Mean-block loop: same as d_delta_dual_d_eta_active_and_nustar_shared's own (mean block is
    # unaffected by the cross-pair extension).
    for k in 1:K_mean
        active_k = aml.mean_active_origins[k]
        λ_mean_k = @view λstar[mean_start+mean_offset[k] : mean_start+mean_offset[k+1]-1]
        d_mean_dense = aug.meanzc_basis === :direct ? d_mean_dnu_direct(D) : d_mean_dnu_anchored(D, aug.refIndex1)
        d_mean_active = length(active_k) == D ? d_mean_dense : d_mean_dense[active_k]
        d_nu[k] += dot(λ_mean_k, d_mean_active)
    end
    # Pair-block loop: d_delta_dual_d_nu_cross_vec's cross-grid accumulation, ragged offset.
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        λ_pair_klin = @view λstar[pair_start0+(klin-1)*npair : pair_start0+klin*npair-1]
        s = sum(λ_pair_klin)
        d_nu[k1] -= νvec_eff[k2] * s
        d_nu[k2] -= νvec_eff[k1] * s
    end
    d_nu .*= mean_m
    eta_grad_active, _ = gather_active_grad(aml, νvec_eff .* d_nu)
    # d_delta_d_nu_star is the RAW (un-nu-multiplied) d(Delta)/d(nu_star) -- gathered from d_nu
    # directly, NOT from the eta-scaled vector (matching the base function's own identical reasoning:
    # no eta exists at the omitted coordinate anymore, so there is no exp() chain rule to apply
    # there; the caller chain-rules it into gp/A_dd instead).
    _, d_delta_d_nu_star = gather_active_grad(aml, d_nu)
    return eta_grad_active, d_delta_d_nu_star
end
