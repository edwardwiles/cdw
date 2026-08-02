# ============================================================================
# Claude Code task 2026-08-01 (profiled-restricted-production-outer-bridge), §5:
# real, in-place, operator-only `restriction_contrib0!` implementations for the
# four restricted families (flexible CM, common Frechet, ZC-only/origin-ZC,
# CM+ZC), reusing ONLY the already-validated forward operator kernels every
# family's own real production FG callback already calls
# (cm_lookup_kernels.jl::apply_contrast!/suffix_sums!/cumulative_forward_
# contribution!, cm_frechet_lookup_kernels.jl::frechet_level_suffix_sums!/
# frechet_level_forward_sum!, zc_restriction_operator.jl::restriction_forward!).
# No dense G, no moment-matrix materialization, no winner recomputation --
# every function below is a thin, allocation-free composition of existing
# kernels against a persistent, caller-owned workspace (task §7).
#
# CONVENTION (derived and EMPIRICALLY VERIFIED against each family's own real
# production FG state, not assumed -- see
# profiled_restricted_q_decomposition_gate_2026-08-01.jl):
#   Every family's real dual_index!/FG callback builds
#     arg0 = -zeta - econ_buf - restriction_raw
#   where `econ_buf` (economic_forward!/compressed_dual_contraction!) is
#   ALREADY SW-weighted internally (compressed_moments.jl:
#   `t[w] = cf.SW[w]*(acc-pmmterm)`), but EVERY restriction forward kernel
#   (cm_forward_contribution!'s cm_contrib, restriction_forward!'s Rλ, the
#   Frechet level contribution) is NOT SW-weighted -- confirmed by direct
#   reading of interval_forward_contribution!/cumulative_forward_contribution!
#   (cm_lookup_kernels.jl) and restriction_forward! (zc_restriction_operator.jl):
#   neither ever multiplies by `cf.SW`.
#   The five-accessor contract (profiled_outer_gradient_layout_contract_2026-08-01.jl)
#   requires `restriction_contrib0(fctx,ev)` to be UNWEIGHTED -- the caller
#   (build_shared_profiled_lfix_cache) applies `SW[w]` exactly once, together
#   with the economic block's own pre-SW `const_part`/`contrib0` terms. So
#   every function below returns `restriction_raw[w] / SW[w]`, NOT
#   `restriction_raw[w]` directly -- dividing by SW here is what makes the
#   caller's single multiplication reconstruct the true (already-SW-weighted)
#   `restriction_raw` term. This was verified, not guessed (see the gate
#   script: at this D4 config SW happens to be uniform == 1.0 for every draw,
#   so the /SW step is a no-op numerically here, but the convention is
#   general and is what the contract's own docstring requires).
#
# STATUS NOTE (honest, see this branch's final report): these operators
# compute the RESTRICTION block only. Combining them with a genuinely REDUCED
# (anchor-excluded) economic dual, as `profiled_economic_layout`'s
# `ProfiledEconomicMomentLayout` type requires, needs a reduced/pivoted inner
# KNITRO solve per restricted family (a `build_profiled_operator_bundle`
# analogue) that does not exist in this worktree for any restricted family --
# confirmed empirically: `build_cm_production_context`'s `cctx.NCORE-1`
# equals `cf.oci-1` (17 at D4), the DENSE economic width, NOT
# `total_reduced_economic_moments` (13 at D4). This reduced-inner-solve
# machinery is being built separately (uncommitted, in-flight) in the sibling
# `architecture/profiled-restricted-inner-endtoend-2026-08-01` worktree
# (confirmed via one permitted read of its `profiled_restricted_accessors_
# 2026-08-01.jl`: `CMBinHessCtx`/`OriginZCCoreHessCtx` there carry NEW fields
# -- `ncore_core`, `profiled_layout`, `econ_ctx`, `n_eta` -- that do not exist
# on this worktree's own `CMBinHessCtx`/`OriginZCCoreHessCtx`, meaning the
# actual reduced economic-block wiring lives in files this task's own
# forbidden list bars editing -- cm_hessian_architectures.jl's moments!
# wrappers/struct definitions). A closed-form "gauge-shift" attempt to derive
# a reduced beta directly from a real DENSE solved beta (shift kappa[o,slot]
# by kappa[anchor,slot] before dropping the anchor column, exploiting
# Pmat[:,slot] summing to 1) was tried and empirically FAILS (residual
# max|q_truth-q_recon| ~0.81 at D4, nowhere near machine precision) -- so
# this is a genuine, non-trivial reduction, not a simple reparametrization
# safe to improvise. See PROFILED_RESTRICTED_Q_DECOMPOSITION_GATE_2026-08-01.csv
# for the full honest gate results (restriction-block-only validation against
# the DENSE economic operator, clearly distinct from the full profiled/reduced
# gate the five-accessor contract ultimately needs).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :apply_contrast!) || error("profiled_restriction_contrib0_operators_2026-08-01.jl requires cm_lookup_kernels.jl to be included first.")
isdefined(Main, :ZCRestrictionOperator) || error("profiled_restriction_contrib0_operators_2026-08-01.jl requires zc_restriction_operator.jl to be included first.")

# ----------------------------------------------------------------------------
# Flexible CM: restriction block is C (CM-grid) dual only.
# ----------------------------------------------------------------------------

"Persistent scratch for `restriction_contrib0_flexcm!` -- sized once per campaign (nO, L, W fixed for the whole run)."
struct FlexCMRestrictionWorkspace
    λmat_block::Matrix{Float64}   # (nO, L)
    λmat_ext::Matrix{Float64}     # (nO, L+1)
    raw::Vector{Float64}          # (W,) cm_contrib scratch, pre-SW-division
end
FlexCMRestrictionWorkspace(nO::Int, L::Int, W::Int) =
    FlexCMRestrictionWorkspace(zeros(nO, L), zeros(nO, L + 1), zeros(W))

"""
    restriction_contrib0_flexcm!(dest, λ_cm, bins, refIndex1, origins, R, SW, ws::FlexCMRestrictionWorkspace) -> dest

Flexible CM's restriction_contrib0!: `dest[w] = cm_contrib[w] / SW[w]`, `cm_contrib` computed via
the EXACT SAME free-function chain `cm_forward_contribution!` (cm_lookup_kernels.jl) uses internally
(`apply_contrast!` -> `suffix_sums!` -> `cumulative_forward_contribution!`), just against this
caller-owned workspace instead of a live `CMLookupState`'s own scratch (task §7: never touch a live
FG state's own buffers). `λ_cm` is the family's solved CM-grid dual slice (length `nO*L`, `(oi-1)*L+l`
convention, matching `CMLookupState`'s own `x[2+ncore1:1+ncore1+ncm]`).
"""
function restriction_contrib0_flexcm!(dest::AbstractVector{Float64}, λ_cm::AbstractVector{Float64},
        bins::AbstractMatrix{<:Unsigned}, refIndex1::Int, origins::Vector{Int},
        R::Union{Nothing,Matrix{Float64}}, SW::AbstractVector{Float64}, ws::FlexCMRestrictionWorkspace)
    nO = length(origins); L = size(ws.λmat_block, 2)
    length(λ_cm) == nO * L || error("restriction_contrib0_flexcm!: length(λ_cm)=$(length(λ_cm)) != nO*L=$(nO*L)")
    λmat_stored = reshape(λ_cm, nO, L)
    apply_contrast!(ws.λmat_block, λmat_stored, R)
    suffix_sums!(ws.λmat_ext, ws.λmat_block)
    cumulative_forward_contribution!(ws.raw, bins, refIndex1, origins, ws.λmat_ext)
    dest .= ws.raw ./ SW
    return dest
end

# ----------------------------------------------------------------------------
# Common Frechet: restriction block is C (CM-grid, cumulative basis) + F (level) dual.
# ----------------------------------------------------------------------------

"Persistent scratch for `restriction_contrib0_frechet!`."
struct FrechetRestrictionWorkspace
    λmat_block::Matrix{Float64}   # (nO, L)
    λmat_ext::Matrix{Float64}     # (nO, L+1)
    cm_raw::Vector{Float64}       # (W,)
    P_level::Vector{Float64}      # (L+1,)
    level_raw::Vector{Float64}    # (W,)
end
FrechetRestrictionWorkspace(nO::Int, L::Int, W::Int) =
    FrechetRestrictionWorkspace(zeros(nO, L), zeros(nO, L + 1), zeros(W), zeros(L + 1), zeros(W))

"""
    restriction_contrib0_frechet!(dest, λ_cm, λ_level, bins, refIndex1, origins, R, D, level_targets, SW, ws) -> dest

Common Frechet's restriction_contrib0!: `dest[w] = (cm_contrib[w] + invsqrtD*level_contrib[w] -
const_term) / SW[w]`, where `const_term = sum_l λ_level[l]*level_targets[l]`, EXACTLY the
`_verify_inner_solution_operator_cm_core`/`CMFrechetLookupState`'s own forward-pass formula
(operator_verification.jl / cm_frechet_lookup_kernels.jl), reusing
`frechet_level_suffix_sums!`/`frechet_level_forward_sum!` unchanged plus the SAME CM-grid chain
`restriction_contrib0_flexcm!` uses (common Fréchet's own CM block is always the `:suffix`/cumulative
basis, matching `cm_forward_contribution!`'s module docstring).
"""
function restriction_contrib0_frechet!(dest::AbstractVector{Float64}, λ_cm::AbstractVector{Float64},
        λ_level::AbstractVector{Float64}, bins::AbstractMatrix{<:Unsigned}, refIndex1::Int,
        origins::Vector{Int}, R::Union{Nothing,Matrix{Float64}}, D::Int, level_targets::Vector{Float64},
        SW::AbstractVector{Float64}, ws::FrechetRestrictionWorkspace)
    nO = length(origins); L = size(ws.λmat_block, 2)
    length(λ_cm) == nO * L || error("restriction_contrib0_frechet!: length(λ_cm)=$(length(λ_cm)) != nO*L=$(nO*L)")
    length(λ_level) == L || error("restriction_contrib0_frechet!: length(λ_level)=$(length(λ_level)) != L=$L")
    λmat_stored = reshape(λ_cm, nO, L)
    apply_contrast!(ws.λmat_block, λmat_stored, R)
    suffix_sums!(ws.λmat_ext, ws.λmat_block)
    cumulative_forward_contribution!(ws.cm_raw, bins, refIndex1, origins, ws.λmat_ext)

    frechet_level_suffix_sums!(ws.P_level, λ_level)
    frechet_level_forward_sum!(ws.level_raw, bins, D, ws.P_level)
    invsqrtD = 1.0 / sqrt(D)
    const_term = 0.0
    @inbounds for l in eachindex(λ_level)
        const_term += λ_level[l] * level_targets[l]
    end
    @inbounds for w in eachindex(dest)
        raw = ws.cm_raw[w] + invsqrtD * ws.level_raw[w] - const_term
        dest[w] = raw / SW[w]
    end
    return dest
end

# ----------------------------------------------------------------------------
# ZC-only (origin-ZC): restriction block is Z (mean+pair) dual only.
# ----------------------------------------------------------------------------

"""
    restriction_contrib0_originzc!(dest, λ_mean, λ_pair, op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace, SW) -> dest

ZC-only's restriction_contrib0!: `dest[w] = (R*lambda)[w] / SW[w]`. `restriction_forward!`
(zc_restriction_operator.jl) ACCUMULATES `-(Rλ)` into its `arg0` argument (its own documented FG
convention); calling it against a caller-owned buffer freshly zeroed here recovers `-(Rλ)` in that
buffer, so `dest` is set to `-buffer/SW = (Rλ)/SW` -- unchanged existing kernel, no restriction
formula re-derived. `ws` must already have been refreshed for the CURRENT outer point's targets via
`refresh_zc_targets!(ws, op, layout, νfull)` (task §7: caller's responsibility, not redone here --
matches `refresh_zc_targets!`'s own existing call-site discipline throughout the codebase).
"""
function restriction_contrib0_originzc!(dest::AbstractVector{Float64}, λ_mean::AbstractVector{Float64},
        λ_pair::AbstractVector{Float64}, op::ZCRestrictionOperator, ws::ZCRestrictionWorkspace,
        SW::AbstractVector{Float64})
    fill!(dest, 0.0)
    restriction_forward!(dest, λ_mean, λ_pair, op, ws)   # dest now holds -(R*lambda)
    dest .= (-1.0) .* dest ./ SW
    return dest
end

# ----------------------------------------------------------------------------
# CM+ZC: restriction block is C (CM-grid) + Z (mean+pair) dual.
# ----------------------------------------------------------------------------

"""
    restriction_contrib0_cmzc!(dest, λ_cm, λ_mean, λ_pair, bins, refIndex1, origins, R, zc_op, zc_ws, SW, cm_ws) -> dest

CM+ZC's restriction_contrib0!: sum of the CM-grid contribution (built via the SAME
`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!` chain
`restriction_contrib0_flexcm!` uses, raw/pre-SW, into `cm_ws.raw`) and the Z (mean/pair)
contribution (`restriction_forward!`'s own `-(Rλ)` accumulation, into `dest`), combined with ONE
division by `SW`. No allocation per call (both raw terms land in caller-owned scratch). Same
caveat as `restriction_contrib0_originzc!`: `zc_ws` must already be refreshed for the current outer
point via `refresh_zc_targets!`.
"""
function restriction_contrib0_cmzc!(dest::AbstractVector{Float64}, λ_cm::AbstractVector{Float64},
        λ_mean::AbstractVector{Float64}, λ_pair::AbstractVector{Float64},
        bins::AbstractMatrix{<:Unsigned}, refIndex1::Int, origins::Vector{Int},
        R::Union{Nothing,Matrix{Float64}}, zc_op::ZCRestrictionOperator, zc_ws::ZCRestrictionWorkspace,
        SW::AbstractVector{Float64}, cm_ws::FlexCMRestrictionWorkspace)
    nO = length(origins); L = size(cm_ws.λmat_block, 2)
    length(λ_cm) == nO * L || error("restriction_contrib0_cmzc!: length(λ_cm)=$(length(λ_cm)) != nO*L=$(nO*L)")
    λmat_stored = reshape(λ_cm, nO, L)
    apply_contrast!(cm_ws.λmat_block, λmat_stored, R)
    suffix_sums!(cm_ws.λmat_ext, cm_ws.λmat_block)
    cumulative_forward_contribution!(cm_ws.raw, bins, refIndex1, origins, cm_ws.λmat_ext)   # raw CM contribution, pre-SW

    fill!(dest, 0.0)
    restriction_forward!(dest, λ_mean, λ_pair, zc_op, zc_ws)   # dest now holds -(R*lambda)
    @inbounds for w in eachindex(dest)
        dest[w] = (cm_ws.raw[w] - dest[w]) / SW[w]
    end
    return dest
end
