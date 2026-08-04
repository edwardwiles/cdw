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
# verification today reads dense obj.H via `obj(inner_x, constr=...)` and `CS.select_G_from_H`)
# is NOT done in this pass -- see docs/OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md for
# the explicit remaining-work list. (`verify_inner_solution_operator_cm!`/`_cm_frechet!` below
# were added in a LATER pass -- see their own docstrings -- and do cover flexible-CM/common-
# Fréchet's :operator verification_backend branch; this header note describes this file's
# original, narrower scope at the time it was written.) The now-removed `skip_cm_fill_ref` Ref
# (docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md) was UNRELATED to this file's own scope --
# CM/Frechet's dense-CM-column fill is unconditional today regardless of verification_backend.
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
    # Phase 10 audit (2026-08-02): explicit per-block KKT residual breakdown -- checklist items
    # "reduced economic moments" (kkt_resid_E), "Z moments" (kkt_resid_mean/kkt_resid_pair), and
    # "France ratio" (france_ratio_resid, the SPECIFIC g_E[cf.cf_col] coordinate -- cf.cf_col is
    # the France/cf-row inner-dual column per compressed_moments.jl:151/249-266). None of this is
    # new math: g_E/g_mean/g_pair were already computed above for g_lambda; this only NAMES and
    # exposes the sub-maxima that were previously only visible merged into a single scalar
    # kkt_resid = maximum(abs, g_lambda) (which already correctly bounds every one of these, since
    # max over a concatenation cannot hide a large sub-block -- see this file's own audit notes).
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_mean = isempty(g_mean) ? 0.0 : maximum(abs, g_mean)
    kkt_resid_pair = isempty(g_pair) ? 0.0 : maximum(abs, g_pair)
    france_ratio_resid = (cf.cf_col > 0 && cf.cf_col <= length(g_E)) ? abs(g_E[cf.cf_col]) : NaN
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_mean = kkt_resid_mean, kkt_resid_pair = kkt_resid_pair,
            france_ratio_resid = france_ratio_resid)
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
    # Phase 10 audit (2026-08-02): same per-block breakdown as verify_inner_solution_operator_originzc!
    # above, plus kkt_resid_cm for the CM-grid block (C in the E/Z/C family split) -- reuses the
    # already-computed g_E/g_mean/g_pair/g_stored, no new math.
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_mean = isempty(g_mean) ? 0.0 : maximum(abs, g_mean)
    kkt_resid_pair = isempty(g_pair) ? 0.0 : maximum(abs, g_pair)
    kkt_resid_cm = maximum(abs, g_stored)
    france_ratio_resid = (cf.cf_col > 0 && cf.cf_col <= length(g_E)) ? abs(g_E[cf.cf_col]) : NaN
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_mean = kkt_resid_mean, kkt_resid_pair = kkt_resid_pair,
            kkt_resid_cm = kkt_resid_cm, france_ratio_resid = france_ratio_resid)
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
    return _verify_inner_solution_operator_cm_core(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R, obj, W, nothing)
end

isdefined(Main, :CMFrechetLookupState) || include(joinpath(@__DIR__, "cm_frechet_lookup_kernels.jl"))

"""
    verify_inner_solution_operator_cm_frechet!(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R,
        level_targets, obj, W) -> NamedTuple

Harmonization task (2026-07-28): thin public wrapper matching this family's existing external
call signature (`level_targets` positioned before `obj, W`, unlike a trailing optional argument,
so this couldn't just become an optional-argument extension of `verify_inner_solution_operator_cm!`
without breaking every existing caller). Delegates to the shared
`_verify_inner_solution_operator_cm_core` below -- see that function's own docstring.
"""
function verify_inner_solution_operator_cm_frechet!(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int,
        bins::Matrix{<:Unsigned}, R::Union{Nothing,Matrix{Float64}}, level_targets::Vector{Float64},
        obj, W::Int)
    return _verify_inner_solution_operator_cm_core(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R, obj, W, level_targets)
end

"""
    _verify_inner_solution_operator_cm_core(zeta, lambda, cf, L, nO, origins, refIndex1, bins, R, obj, W, level_targets)

Harmonization task (2026-07-28): the ONE shared implementation behind both
`verify_inner_solution_operator_cm!` (flexible CM, CM+ZC -- `level_targets=nothing`) and
`verify_inner_solution_operator_cm_frechet!` (common Fréchet -- `level_targets` a `Vector{Float64}`)
-- previously two independent, near-duplicate functions (economic + CM-grid verification
byte-identical; common Fréchet's copy additionally computed the level-block forward/backward
contribution). `G=[E|C]` when `level_targets===nothing`, `G=[E|C|Level]` otherwise.
"""
function _verify_inner_solution_operator_cm_core(zeta::Float64, lambda::AbstractVector{Float64},
        cf::CompressedFactual, L::Int, nO::Int, origins::Vector{Int}, refIndex1::Int,
        bins::Matrix{<:Unsigned}, R::Union{Nothing,Matrix{Float64}}, obj, W::Int,
        level_targets::Union{Nothing,Vector{Float64}})
    D_bins = size(bins, 2)
    ncore1 = cf.oci - 1
    ncm = nO * L
    expected_len = level_targets === nothing ? ncore1 + ncm : ncore1 + ncm + L
    length(lambda) == expected_len ||
        error("_verify_inner_solution_operator_cm_core: length(lambda)=$(length(lambda)) != expected=$(expected_len)")
    λ_E = @view lambda[1:ncore1]
    λ_cm = @view lambda[ncore1+1:ncore1+ncm]
    λ_level = level_targets === nothing ? nothing : (@view lambda[ncore1+ncm+1:ncore1+ncm+L])

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
    g_E = zeros(ncore1)
    economic_transpose!(g_E, dPsi_r, cf, econ_ws)
    g_E .*= -(1.0 / W)

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
        vcat(g_E, vec(g_stored))
    else
        g_level = zeros(L)
        frechet_level_backward_gradient!(g_level, Hpre, D_bins, L, W, invsqrtD, level_targets, sum_dPsi)
        g_level_local = g_level
        vcat(g_E, vec(g_stored), g_level)
    end
    record_operator_verification!()
    # Phase 10 audit (2026-08-02): per-block breakdown shared by flexible-CM (level_targets===
    # nothing, kkt_resid_level=nothing) and common-Frechet (kkt_resid_level = the "F" level-anchor
    # block's own residual) -- reuses g_E/g_stored/g_level already computed above, no new math.
    kkt_resid_E = isempty(g_E) ? 0.0 : maximum(abs, g_E)
    kkt_resid_cm = maximum(abs, g_stored)
    kkt_resid_level = g_level_local === nothing ? nothing : maximum(abs, g_level_local)
    france_ratio_resid = (cf.cf_col > 0 && cf.cf_col <= length(g_E)) ? abs(g_E[cf.cf_col]) : NaN
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = kkt_resid_E, kkt_resid_cm = kkt_resid_cm, kkt_resid_level = kkt_resid_level,
            france_ratio_resid = france_ratio_resid)
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
    # Phase 10 audit (2026-08-02): unrestricted's G=E only, so kkt_resid_E == kkt_resid exactly --
    # added for API symmetry with the 4 restricted verifiers' new breakdown fields, plus the
    # France-ratio-column residual (france_ratio_resid, cf.cf_col -- see the ZC verifier's own
    # comment above for what this column is).
    france_ratio_resid = (cf.cf_col > 0 && cf.cf_col <= length(g_lambda)) ? abs(g_lambda[cf.cf_col]) : NaN
    return (r = r, f = f, g_lambda = g_lambda, kkt_resid = maximum(abs, g_lambda),
            kkt_resid_E = maximum(abs, g_lambda), france_ratio_resid = france_ratio_resid)
end

# ================================================================================================
# Section 6 (verification-defaults task, 2026-07-27): production wiring layer. Each family's
# dense `*_verified_state` tail (`archC_verified_state`/`archC_meanzc_verified_state`/
# `archOZ_verified_state`/`archC_frechet_verified_state`, cm_production_bundle.jl/
# cm_meanzc_production.jl/cm_originzc_production.jl/cm_frechet_cplus.jl -- plus unrestricted's own
# tail in compressed_live.jl::evaluate_fullA_fast_compressed) builds an IDENTICAL-SHAPED `verify`
# NamedTuple (fields: inner_status, Delta_dual, Delta_primal, primal_dual_gap, weight_norm_resid,
# mean_m_resid, max_abs_moment_kkt_resid, m_mean, m_min, m_max) from
# `CS.select_G_from_H(obj,obj.H)` + `obj(inner_x, constr=...)`. `verify_namedtuple_from_operator`
# below builds the SAME-shaped NamedTuple from an operator-verifier's own `(r,f,g_lambda,
# kkt_resid)` return, with NO dense G read anywhere in this function:
#   - Delta_dual = -ov.f (sign verified from oracle.jl's own documented "constr[1] = -f*1e10"
#     comment, not assumed -- see oracle.jl ~line 404-410).
#   - m_weights = dPsi(ov.r), recomputed via the SAME obj.dPsi! both backends already call (ov.r
#     is already returned by every verify_inner_solution_operator_*! function).
#   - max_abs_moment_kkt_resid = ov.kkt_resid directly (both are the same max|g_lambda| quantity,
#     see kkt_residual_blas's own docstring: "max_j |sum_ω m_weights[ω]*G[ω,j]|/W" == max|g_lambda|
#     up to the sign already absorbed by abs()).
# Because the returned NamedTuple has the identical field set the dense path produces,
# `classify_inner_result`/`is_cacheable_result`/`is_verified_success` (oracle.jl) consume either
# one identically -- this is what makes the "cache admission decision"/"incumbent admission
# decision" comparisons in the gate scripts a literal function-level comparison, not a re-derived
# approximation.
# ================================================================================================

"""
    verify_namedtuple_from_operator(ov, obj, W, nStatus) -> (m_weights, verify)

See file header above. `ov` is any `verify_inner_solution_operator_*!` return value (must have
`r`, `f`, `kkt_resid` fields). Returns `(m_weights, verify)`: `m_weights` for building a
`BaseDualState` the same way the dense path does, `verify` for `classify_inner_result` etc.
Requires `primal_divergence` (oracle.jl) to already be defined in the caller's session -- same
implicit dependency the dense `*_verified_state` tails already have (this file has never itself
included oracle.jl, matching its existing include-discipline for CS.select_G_from_H etc.).
"""
function verify_namedtuple_from_operator(ov, obj, W::Int, nStatus::Integer)
    m_weights = similar(ov.r)
    obj.dPsi!(m_weights, ov.r)
    # m_weights_all_finite (verifier-underflow-fix-2026-08-04, ported from campaign branch commits
    # 5eb8c99/6ade48d): cc_algo/Psi.jl's dPsi! is exp(r) for r<=1 and e*r for r>1 -- BOTH branches
    # are mathematically strictly positive for any finite r (the r>1 branch only fires when r>1, so
    # e*r>e>0 there always), so m_weights[i]==0.0 exactly can ONLY happen via Float64 underflow of
    # exp(r) for a very negative r (roughly r<-745, since exp(-745) is already below the smallest
    # representable positive double) -- confirmed live: a real K=3 cm_meanzc point had
    # m_weights[2184]=0.0 from r[2184]=-999.4 (a single far-tail Monte Carlo draw out of W=100,000
    # with an astronomically small, but genuinely positive, reweighted probability), everything else
    # about that solve (KNITRO inner_status=0, primal_dual_gap/mean_m_resid/max_abs_moment_kkt_resid
    # all passing tolerance by 8-9 orders of magnitude) was excellent. `classify_inner_result`'s old
    # strict `mmin > tol.m_min_floor` rejected this as if m<=0 were itself a sign of a bad solve --
    # it isn't; m is provably >=0 by construction here, so relaxing that to `>=` does essentially no
    # protective work on its own. THIS field is the real safeguard the relaxation needs: any genuine
    # optimization failure (a NaN/Inf leaking into r from a diverged/pathological solve, not a benign
    # far-tail underflow) shows up as a non-finite entry in m_weights, caught here directly rather
    # than only indirectly through whichever aggregate statistic happens to blow up.
    m_weights_all_finite = all(isfinite, m_weights)
    # m_weights_all_nonnegative/underflow_zero_count/r_min/r_max (post-verifier-fix W=100k rerun +
    # K=3 campaign task, §2.2): diagnostic-only fields so a campaign report can show whether/how
    # often the previously-rejected extreme (far-tail-underflow) region is actually being visited by
    # the fixed gate, without re-deriving m_weights or r from scratch downstream.
    m_weights_all_nonnegative = all(m -> m >= 0.0, m_weights)
    underflow_zero_count = count(==(0.0), m_weights)
    r_min, r_max = extrema(ov.r)
    s_m_weights = sum(m_weights)
    Delta_dual = -ov.f
    Delta_primal = primal_divergence(m_weights)
    mean_m_resid = abs(s_m_weights / W - 1.0)
    verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              weight_norm_resid = abs(sum(x -> x / s_m_weights, m_weights) - 1.0),
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = ov.kkt_resid,
              m_weights_all_finite = m_weights_all_finite,
              m_weights_all_nonnegative = m_weights_all_nonnegative,
              underflow_zero_count = underflow_zero_count,
              r_min = r_min, r_max = r_max,
              m_mean = s_m_weights / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
    return (m_weights, verify)
end

# ================================================================================================
# Phase 10 audit (2026-08-02): checklist item "recovered full factual shares" -- NONE of the 5
# `verify_inner_solution_operator_*!` functions above ever reconstruct the full A_od/factual
# winner allocation from the converged (zeta*,lambda*)/its LFD and confirm it reproduces the
# correct factual outcome; that was a genuine gap (grep-confirmed: no caller of
# `recover_gamma_normalized_full_A_from_lfd` exists in this file before this addition). This is
# ADDITIVE, family-agnostic (needs only `cf`/`m_weights`/`theta_full`/`ctx`, no family-specific
# restriction state), and built entirely from ALREADY-VALIDATED reused helpers -- no new economics:
#   - `recover_gamma_normalized_full_A_from_lfd` (reduced_recovery_from_lfd_2026-08-01.jl) recovers
#     the gamma-normalized full A_od implied by THIS solve's own LFD (`m_weights = dPsi(ov.r)`,
#     already returned by every verifier above).
#   - `destination_Q_od`/argmax-winner comparison (recover_full_a_2026-07-31.jl,
#     `test_recover_full_a_2026-07-31.jl`'s own Test 2 pattern) confirms recovery changes only the
#     destination-column SCALE, not WHO wins each draw or the within-destination share ratios --
#     i.e. the recovered full shares reproduce the exact same factual outcome the solve was run
#     against, not merely "some economically plausible" outcome.
# Requires recover_full_a_2026-07-31.jl + reduced_recovery_from_lfd_2026-08-01.jl (which in turn
# requires active_layout.jl for active_destinations/dest_slot) included by the caller first, same
# include-discipline as this file's own economic/ZC/CM includes above.
# ================================================================================================

isdefined(Main, :active_destinations) || include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
isdefined(Main, :destination_Q_od) || include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
isdefined(Main, :recover_gamma_normalized_full_A_from_lfd) || include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))

"""
    verify_recovered_full_factual_shares(theta_full, ctx, cf, m_weights; d_list=active_destinations(ctx)) -> NamedTuple

Phase 10 audit (2026-08-02) checklist item "recovered full factual shares", shared across all 5
families (family-agnostic -- takes only the economic-core `cf`/`m_weights`, no restriction-block
state). `theta_full` must be the SAME full theta vector `cf` was built at; `m_weights` must be
`dPsi(ov.r)` from a `verify_inner_solution_operator_*!` call at that same converged solve (exactly
what `verify_namedtuple_from_operator` above already computes as its own `m_weights` return).

Recovers the gamma-normalized full A_od implied by this solve's own LFD
(`recover_gamma_normalized_full_A_from_lfd`), then checks that recovering it and re-evaluating the
model reproduces the IDENTICAL winner (argmax) at every draw and destination, and the identical
within-destination share ratios, as `theta_full` itself -- i.e. the recovered full factual shares
are economically consistent with (reproduce) the factual outcome the inner solve was run against,
not merely dimensionally valid. `max_winner_mismatch == 0` and small `max_share_ratio_diff` is a
PASS; any winner mismatch is a hard failure (the recovery changed who wins some draw, which the
theory this recovery is built on says should never happen -- see
`reduced_recovery_from_lfd_2026-08-01.jl`'s own header).
"""
function verify_recovered_full_factual_shares(theta_full::AbstractVector{Float64}, ctx, cf::CompressedFactual,
        m_weights::AbstractVector{Float64}; d_list = active_destinations(ctx))
    Aod_offset = ctx.Aod_offset
    D = ctx.D
    Ddest = length(active_destinations(ctx))
    z_full, c, gamma_tilde = recover_gamma_normalized_full_A_from_lfd(theta_full, ctx, cf, m_weights; d_list = d_list)
    theta_recovered = copy(theta_full)
    theta_recovered[Aod_offset+1:Aod_offset+D*Ddest] .= vec(exp.(z_full))

    max_winner_mismatch = 0
    max_share_ratio_diff = 0.0
    for d in d_list
        Qw = destination_Q_od(theta_full, ctx, d)
        Qr = destination_Q_od(theta_recovered, ctx, d)
        Mw = vec(sum(Qw, dims = 2)); Mr = vec(sum(Qr, dims = 2))
        winw = [argmax(@view Qw[w, :]) for w in 1:size(Qw, 1)]
        winr = [argmax(@view Qr[w, :]) for w in 1:size(Qr, 1)]
        nmis = sum(winw .!= winr)
        max_winner_mismatch = max(max_winner_mismatch, nmis)
        rdiff = maximum(abs.((Qw ./ Mw) .- (Qr ./ Mr)))
        max_share_ratio_diff = max(max_share_ratio_diff, rdiff)
    end
    return (z_full = z_full, c = c, gamma_tilde = gamma_tilde,
            max_winner_mismatch = max_winner_mismatch, max_share_ratio_diff = max_share_ratio_diff)
end

"""
    *_VERIFICATION_BACKEND_DEFAULT::Ref{Symbol}

One global Ref per family, `:dense_reference` | `:operator`, mirroring this codebase's existing
backend-toggle discipline (`CM_INNER_FG_BACKEND_DEFAULT`/`ORIGINZC_FG_BACKEND_DEFAULT`/etc.,
core_exact_hessian.jl) -- NOT a `CMBinHessCtx` struct field, deliberately: `CMBinHessCtx` lives in
cm_hessian_architectures.jl, an explicitly off-limits Hessian-backend file for this task (two other
agents are concurrently changing which Hessian backends are active on different branches); a global
Ref threads through with zero struct-plumbing risk, same pattern `UNRESTRICTED_CORE_HESSIAN_BACKEND`
(compressed_live.jl) already uses for a cross-cutting backend choice outside `CMBinHessCtx`.
Defaults flipped `:dense_reference` -> `:operator` for ALL FIVE families (2026-07-27) after
Section 6.1's D=4 AND real D=20/W=80,000 comparison gates ALL PASSED cleanly for every family --
draw-level dual index, objective, complete dual gradient (full vector), KKT residual,
feasibility/moment residual, status classification, cache admission decision, incumbent admission
decision all agreed between backends at every tested config, both scales. See
docs/FIVE_FAMILY_OPERATOR_VERIFICATION_DEFAULT_RELEASE_2026-07-27.md for the full per-family
pass/fail table with real numbers. `:dense_reference` remains available as an explicit,
named, non-default backend (pass `verification_backend = :dense_reference` at any call site).
"""
const CM_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)
const CM_MEANZC_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)
const ORIGINZC_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)
const CM_FRECHET_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)
const UNRESTRICTED_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)
