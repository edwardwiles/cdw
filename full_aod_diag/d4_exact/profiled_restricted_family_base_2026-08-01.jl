# ============================================================================
# Claude Code task 2026-08-01 (all-families completion): shared reduced base
# object for the FOUR RESTRICTED families' augmented-obj builders.
#
# DESIGN RESOLUTION (this session, direct source read, not assumed): every
# restricted-family augmented-obj builder already reads its economic-block
# width GENERICALLY off `ctx.obj.d`/`ctx.obj.outer_constr_index`
# (`build_cm_augmented_obj_archB`: `ncore = obj0.d`; `build_originzc_augmented_obj`/
# `build_cm_meanzc_augmented_obj`: the analogous `ncore_econ = obj0.d`), and
# `wrap_moments_with_cm_archB`'s dense economic-column materialization
# (`materialize_dense_factual_structured!(@view(Gtmp[:,1:pregrav]), cf)`) is
# ALREADY skipped unconditionally on the production `skip_fill=true` path
# (`moment_representation=:operator`, the current default) -- confirmed by
# direct read, cm_hessian_architectures.jl:314/334/338 (`if !skip_fill`
# guards). This means the economic-block WIDTH (`pregrav`/`ncore`) is a pure
# bookkeeping quantity on the production path: feeding these builders a
# `ctx.obj`-shaped object whose `.d`/`.outer_constr_index` already reflect the
# REDUCED width (`1 + layout.total_reduced_economic_moments`, the SAME
# quantity `build_profiled_operator_bundle` already uses for the unrestricted
# family, profiled_operator_bundle_2026-08-01.jl:64) requires ZERO changes to
# `wrap_moments_with_cm_archB`/`build_cm_augmented_obj_archB` themselves -- the
# only thing needed is a way to construct that reduced base object and thread
# it in. This file provides exactly that, plus the additive `base_obj`
# keyword on `build_cm_augmented_obj_archB` (cm_hessian_architectures.jl) that
# consumes it -- see that function's own updated docstring.
#
# What this file does NOT do (honest scope cut, not a silent gap -- see
# PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md for the full remaining
# plan): it does not wire the reduced H_EE kernel
# (`reduced_homogeneous_winner_pair_hessian!`) into `_fill_cm_HEE!`, does not
# implement the H_EC/H_EF/H_EZ "gather retained rows only" step in
# `hessian_cm_structured!`/`archA_partitioned_hess_cb_builder`, and does not
# touch `build_originzc_augmented_obj`/`build_cm_meanzc_augmented_obj` (the
# ZC-only/CM+ZC analogues) at all -- only the dimension-bookkeeping
# prerequisite common to all four families is built and tested here.
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))

"""
    build_reduced_base_obj_for_family(ctx, layout::ProfiledEconomicMomentLayout, CS) -> reduced_obj0

Shallow-copies `ctx.obj` (a `CS.PsiObjectiveBundleImplicit`) with `.d` and
`.outer_constr_index` overridden to the profiled/reduced economic-block width
`n = 1 + layout.total_reduced_economic_moments` -- the SAME quantity
`build_profiled_operator_bundle` computes for the unrestricted family
(profiled_operator_bundle_2026-08-01.jl:64), so all five families share
literally the same reduced-width formula, matching the mission's "one shared
profiled economic layout" requirement (mission §4).

Every OTHER field (`γ`, `δ`, `find_smallest`, `l`, `U`, `N`, `lower_limit`,
`inner_loop_opt`, etc.) is copied from `ctx.obj` UNCHANGED -- these describe
the CC inner-dual objective's own delta-grid/weight structure, independent of
how many economic-moment columns feed it (same reasoning
`build_profiled_operator_bundle`'s own docstring already gives for reusing
`ref_obj`'s shared outer parameters).

`.moments!`/`.moments_jacobian!` are ALSO copied unchanged from `ctx.obj`
(the FULL, D*Ddest-shaped closure) -- this reduced object is consumed only
for its `.d`/`.outer_constr_index` bookkeeping by
`build_cm_augmented_obj_archB`'s `ncore = obj0.d` line; a restricted-family
caller immediately replaces `.moments!` with its own `wrap_moments_with_*`
closure. KNOWN, DOCUMENTED GAP (not fixed here): `wrap_moments_with_cm_archB`
threads `obj0.moments!` through as the `core_moments!` TiedWinnerError
fallback argument -- if that fallback is ever actually reached (documented
elsewhere as provably unreachable in production once `check_ties=false`,
cm_hessian_architectures.jl:270-286), it would try to write the FULL
(unreduced) pregrav width into a `Gtmp` sized for the REDUCED width. Since the
fallback is unreachable on the current production `check_ties=false` path,
this is a latent-but-inert gap, not a live correctness bug; flagged here for
whoever removes that dead-code fallback path or reintroduces a live tie
check.
"""
function build_reduced_base_obj_for_family(ctx, layout::ProfiledEconomicMomentLayout, CS)
    obj0 = ctx.obj
    n = 1 + layout.total_reduced_economic_moments
    return CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest, γ = obj0.γ,
        (moments!) = obj0.moments!, moments_jacobian! = obj0.moments_jacobian!,
        d = n, outer_constr_index = n,
        inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
        l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit,
        use_cached_x = obj0.use_cached_x,
        threshold_state = obj0.threshold_state,
        outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
        needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
end
