# ============================================================================
# Legacy-H removal (2026-07-28): a genuinely separate production-operator bundle type, with NO H
# field and NO H-sized preallocation at all -- not a shrunk/optional H, an absent one. Complements
# (does not replace) `PsiObjectiveBundleImplicit`, which remains the `:dense_reference` type,
# unchanged, for explicit correctness comparison.
#
# SHARED across all restricted families, by design -- not per-family. The original
# `PsiObjectiveBundleImplicit` is itself family-agnostic (one type, a pluggable `moments!::Function`
# field supplies the per-family behavior); an earlier draft of this file named the type
# `OperatorCMBundle` as if it needed to be flexible-CM-specific, which was a mistake caught on
# review -- direct comparison of `wrap_moments_with_cm_archB` (flexible-CM),
# `wrap_moments_with_cm_meanzc` (CM+ZC), `wrap_moments_with_originzc` (origin-ZC), and
# `wrap_moments_with_cm_frechet_archB` (common-Fréchet) shows their priming logic is IDENTICAL --
# `cf = cf_build(θ_econ, ctx; check_ties=false)`, `fill_K_directgp!(K, θ_econ, ctx)`,
# `grav_raw = compressed_gravity_raw(θ_econ, ctx)`, `fill_gravity_column_into!(...)`,
# `core_cf_ref[] = cf` -- only the θ-slicing (each family strips its own trailing
# ν/η parameters before calling this) differs, and that slicing already happens in each family's
# own entry-point function before `prime_operator!` is called, not inside it.
#
# Why this is safe (not merely "should be safe"): this session empirically validated, via
# test_shared_core_hessian_d4_gates.jl (real KNITRO D=4, all 4 sections, 40/40 PASS), that under
# each restricted family's default production configuration (operator FG backend, `:winner_bin`
# cross-Hessian backend, `core_hessian_backend` never `:dense_reference` outside this repo's own
# comparison harnesses) neither the economic block NOR the family-specific restriction block
# (CM-grid / mean+pair / level) of the legacy dense G is ever read anywhere on the production
# FG/Hessian path for flexible-CM, CM+ZC, and origin-ZC -- all three now unconditionally skip both
# blocks (`skip_fill=true`, this session's fix). The ONLY things the priming step still needs to
# produce, for those three families, are K and the gravity column -- both O(W), not O(W*d).
# `H_save` (the returned inner-objective value) is `K[1] * (-1)^find_smallest`, exactly as
# `PsiObjectiveBundleImplicit`'s own priming computes it from `H[1,1]` today -- `K` IS `H`'s former
# first column, nothing about that formula changes. Common-Fréchet's own skip is NOT yet validated
# at real D=20 for the actual operator-FG code path (see
# OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md for the in-progress investigation) --
# `prime_operator!` itself is family-agnostic either way; wiring it into common-Fréchet's own
# entry point is safe to do once that family's own skip is validated, not before.
#
# What this does NOT do: it does not touch the outer-loop A-gradient/theta-derivative machinery
# (out of scope per this task). Only flexible-CM's entry point (`build_cm_production_context`) is
# wired to construct this type so far this session; CM+ZC/origin-ZC's own entry points can adopt
# the SAME type and the SAME `prime_operator!` function without redefining either -- purely a
# wiring change in their own `inner_loop_internal_*_operator` functions, the natural next step.
# ============================================================================

using Parameters: @with_kw

@with_kw mutable struct OperatorPsiBundle{T}
    δ                   ::Float64
    find_smallest       ::Bool
    γ                   ::T
    l                   ::Int64
    inequality_index    ::Array{Int64,1}
    complement_index    ::Array{Int64,2} = [0 0]
    U                   ::Array{Float64,2}
    M                   ::Int64          = size(U)[1]
    N                   ::Int64          = M
    outer_constr_index  ::Int64
    inner_loop_opt      ::String
    lower_limit         ::Float64        = -KNITRO.KN_INFINITY
    use_cached_x        ::Bool           = false
    Psi!                ::Function       = Psi!
    dPsi!               ::Function       = dPsi!
    ddPsi!              ::Function       = ddPsi!
    # No H, no H_copy, no jac_h, no moments! field -- outer-gradient/jacobian machinery is out of
    # this task's scope and is never invoked on this type (it is only ever constructed for the
    # inner dual solve). K and the single gravity column are the ONLY per-draw storage this bundle
    # owns -- O(W), not O(W*(2+d)) -- and neither is family-specific (see header).
    K                   ::Array{Float64,1} = zeros(M)
    grav_col            ::Array{Float64,1} = zeros(M)
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)
    threshold_state     ::ThresholdAbortState = ThresholdAbortState()
end

# Fail-fast interface (task §6): dense G/H access is a language-level impossibility for this type
# (no such field), but these explicit methods give a clear error message rather than a bare
# `FieldError`/`MethodError` if something reaches for them anyway.
select_G_from_H(::OperatorPsiBundle, args...) =
    error("select_G_from_H: dense G/H access is forbidden for OperatorPsiBundle -- this type has no H field. Use the :dense_reference construction path (PsiObjectiveBundleImplicit) if you genuinely need it.")

# Mirrors cc_algo/inner_loop_functions.jl's PsiObjectiveBundleImplicit methods exactly (same
# formulas, same fields, all of which OperatorPsiBundle also carries) -- these are the ONLY
# dispatch methods any family's production inner-solve call chain needs specialized for this new
# obj type; everything else in those chains is either untyped (`obj::Any`) or reads plain fields,
# which OperatorPsiBundle already provides, shared across every family that uses it.
CS.inner_loop_number_variables(obj::OperatorPsiBundle) = obj.outer_constr_index
CS.inner_loop_lower_bounds(obj::OperatorPsiBundle) =
    vcat(-KNITRO.KN_INFINITY, [i ∈ obj.inequality_index ? 0.0 : -KNITRO.KN_INFINITY for i in 1:obj.outer_constr_index-1])
CS.inner_loop_initial_values(obj::OperatorPsiBundle) =
    obj.use_cached_x && LinearAlgebra.norm(obj.x) < 1e6 ? obj.x : zeros(obj.outer_constr_index)
CS.inner_loop_complementarity_constraints(kc, obj::OperatorPsiBundle) =
    KNITRO.KN_set_compcons(kc, zeros(Int32, length(obj.complement_index[:, 1])), Int32.(obj.complement_index[:, 1] .+ 0), Int32.(obj.complement_index[:, 2] .+ 0))

"""
    prime_operator!(obj::OperatorPsiBundle, θ_econ, ctx, core_cf_ref) -> nothing

Legacy-H removal (2026-07-28): the operator-only priming step, SHARED across every restricted
family (see file header for the side-by-side confirmation that all four families' existing
priming closures do exactly this same sequence). Replaces `wrap_moments_with_cm_archB` and its 3
siblings for this bundle type entirely (not reusing them -- those closures' internal
column-indexing assumes a full-width `G` matching the original `[K|ones|G]` layout, which this
bundle deliberately does not have). Builds `cf`, publishes it for the Hessian callback (identical
to the existing closures' own `core_cf_ref[] = cf`), fills `K` and the gravity column directly --
no `materialize_dense_factual_structured!`, no restriction-column fill, no intermediate
`Gtmp`/copy at all. `θ_econ` is the economic-only parameter slice (each family's own entry-point
function strips its own trailing ν/η restriction parameters before calling this, exactly as each
existing closure already does internally).
"""
function prime_operator!(obj::OperatorPsiBundle, θ_econ::AbstractVector, ctx, core_cf_ref::Ref{Any})
    cf = cf_build(θ_econ, ctx; check_ties = false)
    fill_K_directgp!(obj.K, θ_econ, ctx)
    grav_raw = compressed_gravity_raw(θ_econ, ctx)
    fill_gravity_column_into!(obj.grav_col, grav_raw, ctx, 1)
    core_cf_ref[] = cf
    obj.H_save = obj.K[1] * (-1.0)^obj.find_smallest
    return nothing
end

_dense_H_or_nothing(obj::OperatorPsiBundle) = nothing
