# ============================================================================
# True no-H operator bundle (2026-07-28 continuation): a genuinely separate production-operator
# bundle type, with NO H field, NO H_copy field, NO K field (renamed `payoff`, since even the NAME
# "K" is on the forbidden-field list, not just the legacy [K|ones|G] LAYOUT), NO `ones` field, and
# NO `moments!` field -- not a shrunk/optional H, an absent one. Complements (does not replace)
# `PsiObjectiveBundleImplicit`, which remains the `:dense_reference` type, unchanged, for explicit
# correctness comparison.
#
# `economic_state`/`restriction_state` are genuine bundle-owned state HANDLES, not duplicated
# storage: `economic_state` is the SAME `Ref{Any}` box the family's `CMBinHessCtx.core_cf_ref`
# already is (aliased, not copied, by `prime_operator!`'s caller) -- the bundle now owns a
# reference to its own economic (compressed-factual) state rather than that state living only on a
# side context the bundle doesn't know about. `restriction_state` is the owning family's bin/
# contrast context itself (e.g. the `CMBinHessCtx` for flexible-CM/CM+ZC, `nothing` for
# unrestricted) -- see `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md` for why this, not a nested
# `dual_workspace`/`hessian_workspace` struct-of-structs, is the right shape here: the shared
# Hessian/verification code (`hessian_cm_structured!`, `verify_inner_solution_operator_cm!`, etc.)
# already does unconditional flat-field reads (`@unpack M, arg0, arg2, ddPsi! = obj`, bare
# `obj.Psi!`) for BOTH bundle types, validated and explicitly out of this task's scope to touch --
# nesting those fields under a workspace sub-struct would require rewriting that already-correct
# shared code for no correctness benefit, so they stay flat, exactly as `PsiObjectiveBundleImplicit`
# itself already has them.
#
# SHARED across all restricted families, by design -- not per-family. The original
# `PsiObjectiveBundleImplicit` is itself family-agnostic (one type, a pluggable `moments!::Function`
# field supplies the per-family behavior); an earlier draft of this file named the type
# `OperatorCMBundle` as if it needed to be flexible-CM-specific, which was a mistake caught on
# review -- direct comparison of `wrap_moments_with_cm_archB` (flexible-CM),
# `wrap_moments_with_cm_meanzc` (CM+ZC), `wrap_moments_with_originzc` (origin-ZC), and
# `wrap_moments_with_cm_frechet_archB` (common-Fréchet) shows their priming logic is IDENTICAL --
# `cf = cf_build(θ_econ, ctx; check_ties=false)`, `fill_K_directgp!(payoff, θ_econ, ctx)`,
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
# produce, for those three families, are `payoff` and the gravity column -- both O(W), not O(W*d).
# `H_save` (the returned inner-objective value) is `payoff[1] * (-1)^find_smallest`, exactly as
# `PsiObjectiveBundleImplicit`'s own priming computes it from `H[1,1]` today -- `payoff` IS `H`'s
# former first column, nothing about that formula changes. Common-Fréchet's own skip is NOT yet validated
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
    # No default: a struct-level default here (formerly -KNITRO.KN_INFINITY) is exactly the
    # anti-pattern this repo's CLAUDE.md forbids for scientific/settings parameters -- it let 5
    # REDUCED bundle-construction functions silently omit this kwarg and get a value for which
    # `f <= lower_limit` can never fire, instead of the UndefKeywordError that would have caught
    # the gap the moment each was written (found live 2026-08-03, see
    # docs/audits/profiled-inner-readiness-2026-08-03/).
    lower_limit         ::Float64
    use_cached_x        ::Bool           = false
    # Qualified `CS.Psi!`/`CS.dPsi!`/`CS.ddPsi!` (not bare `Psi!`): unlike `PsiObjectiveBundleImplicit`
    # (defined INSIDE the `CS` module, cc_algo/PsiObjectiveBundle.jl, where a bare `Psi!` default
    # already resolves to `CS.Psi!`), this file is `include`d at top level (Main), so an
    # unqualified default would look for `Main.Psi!` and fail with `UndefVarError`.
    Psi!                ::Function       = CS.Psi!
    dPsi!               ::Function       = CS.dPsi!
    ddPsi!              ::Function       = CS.ddPsi!
    # No H, no H_copy, no jac_h, no moments! field, no K field (renamed `payoff`) -- outer-
    # gradient/jacobian machinery is out of this task's scope and is never invoked on this type (it
    # is only ever constructed for the inner dual solve). `payoff` and the single gravity column are
    # the ONLY per-draw storage this bundle owns -- O(W), not O(W*(2+d)) -- and neither is
    # family-specific (see header).
    payoff              ::Array{Float64,1} = zeros(M)
    grav_col            ::Array{Float64,1} = zeros(M)
    H_save              ::Float64          = 0.0
    arg0                ::Array{Float64,1} = zeros(M)
    arg1                ::Array{Float64,1} = zeros(M)
    arg2                ::Array{Float64,1} = zeros(M)
    x                   ::Array{Float64,1} = NaN .* ones(outer_constr_index)
    threshold_state     ::ThresholdAbortState = ThresholdAbortState()
    # Bundle-owned state handles (task §Part A struct sketch's `economic_state`/`restriction_state`)
    # -- see file header for why these are aliases into existing shared boxes, not new storage.
    economic_state      ::Union{Nothing,Base.RefValue{Any}} = nothing
    restriction_state    ::Any                               = nothing
end

# Fail-fast interface (task §6): dense G/H access is a language-level impossibility for this type
# (no such field), but these explicit methods give a clear error message rather than a bare
# `FieldError`/`MethodError` if something reaches for them anyway.
select_G_from_H(::OperatorPsiBundle, args...) =
    error("select_G_from_H: dense G/H access is forbidden for OperatorPsiBundle -- this type has no H field. Use the :dense_reference construction path (PsiObjectiveBundleImplicit) if you genuinely need it.")
# `obj.H`/`obj.moments!` on an OperatorPsiBundle already fail immediately with Julia's own default
# `getproperty` ("type OperatorPsiBundle has no field H") -- no custom method needed or added; the
# absence is structural (fieldnames(typeof(obj)) genuinely does not contain :H/:moments!/:K/:ones),
# not a runtime-guarded illusion of absence.

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
    prime_operator!(obj::OperatorPsiBundle, θ_econ, ctx, core_cf_ref; restriction_state=nothing) -> nothing

True no-H operator bundle (2026-07-28): the operator-only priming step, SHARED across every
restricted family (see file header for the side-by-side confirmation that all four families'
existing priming closures do exactly this same sequence). Replaces `wrap_moments_with_cm_archB` and
its 3 siblings for this bundle type entirely (not reusing them -- those closures' internal
column-indexing assumes a full-width `G` matching the original `[K|ones|G]` layout, which this
bundle deliberately does not have). Builds `cf`, publishes it for the Hessian callback (identical
to the existing closures' own `core_cf_ref[] = cf`), fills `payoff` and the gravity column directly
-- no `materialize_dense_factual_structured!`, no restriction-column fill, no intermediate
`Gtmp`/copy at all. `θ_econ` is the economic-only parameter slice (each family's own entry-point
function strips its own trailing ν/η restriction parameters before calling this, exactly as each
existing closure already does internally). `obj.economic_state` is set to ALIAS `core_cf_ref`
itself (not a copy) -- `obj.economic_state[] === core_cf_ref[]` always holds, so the bundle
genuinely owns a handle to the same economic state the family's Hessian callback reads via
`cctx.core_cf_ref`, with zero duplicated computation or storage.
"""
function prime_operator!(obj::OperatorPsiBundle, θ_econ::AbstractVector, ctx, core_cf_ref::Ref{Any};
                          restriction_state = nothing)
    cf = cf_build(θ_econ, ctx; check_ties = false)
    fill_K_directgp!(obj.payoff, θ_econ, ctx)
    # compressed_gravity_raw/fill_gravity_column_into! into obj.grav_col REMOVED (2026-07-31,
    # Brazil-Korea gravity-exclusion task): confirmed dead -- obj.grav_col has no read-site
    # anywhere in the codebase; the real production objective (obj.H_save) is built from
    # obj.payoff alone. The actual gravity restriction is enforced exactly by pivot_expand
    # upstream of every real outer iterate (GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md); this
    # was pure wasted per-eval computation with no consumer. obj.grav_col field itself is left in
    # place (cheap, avoids a struct-layout change) but is no longer written here.
    core_cf_ref[] = cf
    obj.economic_state = core_cf_ref
    restriction_state !== nothing && (obj.restriction_state = restriction_state)
    obj.H_save = obj.payoff[1] * (-1.0)^obj.find_smallest
    return nothing
end

_dense_H_or_nothing(obj::OperatorPsiBundle) = nothing
_dense_H_copy_or_nothing(obj::OperatorPsiBundle) = nothing
