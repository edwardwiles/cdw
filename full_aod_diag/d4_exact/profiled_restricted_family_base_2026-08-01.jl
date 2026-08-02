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
# CORRECTION (later same session, see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md's own
# "important correction" section): the claim above ("requires ZERO changes... on the production
# path") holds ONLY for dimension bookkeeping. It does NOT mean a reduced family's `obj_cm.moments!`
# can run a real FG callback -- `skip_fill=true` is a narrow priming mechanism (skips ALL of G, not
# just economic columns) never installed as any family's PRIMARY `moments!` in the default
# (`:dense_reference` inner-FG-backend) configuration; the primary `moments!` genuinely calls
# `materialize_dense_factual_structured!` every callback, which requires the FULL (unreduced)
# `cf.oci-1` width and errors immediately on a reduced-width view. `materialize_dense_factual_
# structured_reduced!` below closes this gap (added later this session): a genuine reduced dense-G
# materialization, needed for `wrap_moments_with_cm_archB` to support a real solve under
# `profiled_layout!==nothing` -- see that function's own updated docstring.
#
# H_EE/H_EC wiring (`_fill_cm_HEE!`/`hessian_cm_structured!`) was completed later this session too
# (in `cm_hessian_architectures.jl` directly, D4-verified bit-identical) -- see the master doc's
# full accounting for exactly what remains: `hessian_cm_structured_v2!` (threaded), H_EF (common
# Fréchet), H_EZ (ZC-only/CM+ZC), and everything past a D4 Hessian-formula gate (D20, inner
# equivalence, outer gradient, performance, W500k).
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

isdefined(Main, :materialize_dense_factual_structured!) || error("profiled_restricted_family_base_2026-08-01.jl requires structured_moment_build.jl to be included first.")

"""
    materialize_dense_factual_structured_reduced!(Gview, cf::CompressedFactual, layout::ProfiledEconomicMomentLayout;
        scratch_full::Union{Nothing,Matrix{Float64}}=nothing) -> Gview

Reduced/anchor-omitting analog of `materialize_dense_factual_structured!`, closing the FG-callback
gap flagged in this file's own header correction (and PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md).
`Gview` must be sized `(cf.W, layout.total_reduced_economic_moments)` -- columns
`1:n_bilateral` are `layout.retained_full_factual_j`-selected (in that order), the LAST column (if
`layout.france_ratio_reduced_j > 0`) is the France/counterfactual-price-index column.

DESIGN: rather than re-deriving `structured_fill_chunk!`'s rank-one-fixed-term + winner-scatter
formula in reduced-index form (risking a fresh bug in a hand-rewritten formula), this GATHERS from
the existing, already-validated `materialize_dense_factual_structured!` computed into a full-width
scratch buffer -- the same "gather at assembly, don't touch the underlying validated primitive"
design this session already used for H_EC. Cost: `O(W*D*Ddest)` to build the full scratch (same
complexity class as the old full fill; not FLOP-optimal, matching this session's own stated
priority of correctness/dimension-reduction over cross-block FLOP reduction) plus a cheap `O(W*
n_reduced)` column-gather. `scratch_full`, if supplied, is reused (not reallocated) across repeated
calls -- callers that call this every FG callback (the real use case) should own a persistent buffer
sized `(cf.W, cf.oci-1)` and pass it in.
"""
function materialize_dense_factual_structured_reduced!(Gview::AbstractMatrix, cf, layout::ProfiledEconomicMomentLayout;
        scratch_full::Union{Nothing,Matrix{Float64}} = nothing)
    W = cf.W
    n_reduced = layout.total_reduced_economic_moments
    size(Gview) == (W, n_reduced) || error("materialize_dense_factual_structured_reduced!: size(Gview)=$(size(Gview)) != (W,n_reduced)=($W,$n_reduced)")
    ncol_full = cf.oci - 1
    Gfull = scratch_full === nothing ? Matrix{Float64}(undef, W, ncol_full) : scratch_full
    size(Gfull) == (W, ncol_full) || error("materialize_dense_factual_structured_reduced!: scratch_full size $(size(Gfull)) != (W,cf.oci-1)=($W,$ncol_full)")
    materialize_dense_factual_structured!(Gfull, cf)
    n_bilateral = length(layout.retained_full_factual_j)
    @views Gview[:, 1:n_bilateral] .= Gfull[:, layout.retained_full_factual_j]
    layout.france_ratio_reduced_j > 0 && (@views Gview[:, end] .= Gfull[:, cf.cf_col])
    return Gview
end

# ============================================================================
# FORMULATION-CONSISTENCY BUGFIX (2026-08-01, found live via direct autodiff cross-check after the
# user correctly flagged the reduced flexible-CM solve hitting KNITRO's iteration limit -- unusual
# for this class of problem and rightly treated as suspicious rather than accepted at face value):
#
# `reduced_homogeneous_winner_pair_hessian!` (H_EE) and `winner_pair_cross_hessian_cm_block!`'s
# `use_profiled_correction=true` path (H_EC, after this session's own Lam_homog fix) both implement
# the HOMOGENEOUS moment formulation (`homogeneous_contraction_2026-07-31.jl`'s own `kappa`/`Cbar`
# convention, deliberately DIFFERENT from the STRUCTURED formulation
# `materialize_dense_factual_structured!`/`structured_fill_chunk!` production `G` is built from --
# confirmed via direct comparison, NOT a reparametrization of the same moments: their linear
# functionals `t(w)=sum_j beta_j*G[w,j]` disagree by ~30-95% at matched beta, not a constant scale
# factor). `materialize_dense_factual_structured_reduced!` above is therefore the WRONG G for a
# reduced context whose Hessian goes through the homogeneous-formulation kernels -- it was validated
# internally consistent with ITSELF (Part 1 of test_profiled_flexcm_d4_fg_and_solve_gate_2026-08-01.jl
# still passes, correctly, since gathering from the FULL structured G is exactly what it's supposed
# to do) but not consistent with the Hessian it was paired with, which is why the D4 KNITRO solve
# ground through the 100-iteration default limit (linear, not quadratic, convergence -- the classic
# signature of a Hessian that doesn't match the objective's actual curvature) instead of the ~4
# iterations to machine precision the FULL model achieves on the identical problem. This function
# closes that gap: reuses ONLY the already-validated `reduced_homogeneous_dual_contraction` (this
# branch's own, pre-existing, autodiff-confirmed-to-machine-precision homogeneous kernel), never
# re-deriving its formula by hand (two earlier hand-derivation attempts this session each introduced
# a fresh, different bug -- see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md's full account).
# ============================================================================

isdefined(Main, :reduced_homogeneous_dual_contraction) || error("profiled_restricted_family_base_2026-08-01.jl requires reduced_homogeneous_contraction_2026-08-01.jl to be included first (for materialize_homogeneous_dense_G_reduced!).")

"""
    materialize_homogeneous_dense_G_reduced!(Gview, cf, ctx, θ_full, layout) -> Gview

Dense HOMOGENEOUS-formulation `G` for the reduced economic block, sized `(cf.W,
layout.total_reduced_economic_moments)` -- the correct FG counterpart to
`reduced_homogeneous_winner_pair_hessian!` (H_EE) and the `use_profiled_correction=true` path of
`winner_pair_cross_hessian_cm_block!` (H_EC), REPLACING `materialize_dense_factual_structured_reduced!`
above wherever a reduced context's Hessian goes through those kernels (i.e. whenever
`profiled_layout !== nothing`, unconditionally -- there is no case where mixing formulations is
correct).

DESIGN: since `reduced_homogeneous_dual_contraction(β, cf, ctx, θ_full, layout)` is LINEAR in `β`,
column `j` of `G` is EXACTLY `reduced_homogeneous_dual_contraction(e_j, cf, ctx, θ_full, layout)` for
the `j`-th unit vector `e_j` -- guaranteed correct by construction (no hand-derived closed form,
which twice produced a fresh bug earlier this session when attempted). Cost:
`O(n_reduced)` calls to an `O(W)` function = `O(W*n_reduced)`, the same complexity class as the
structured version it replaces; not yet optimized to a genuine O(1)-per-column closed form (a real,
documented follow-up, not attempted here given this function's own cautionary history).
"""
function materialize_homogeneous_dense_G_reduced!(Gview::AbstractMatrix, cf, ctx, θ_full::AbstractVector,
        layout::ProfiledEconomicMomentLayout)
    W = cf.W
    n_reduced = layout.total_reduced_economic_moments
    size(Gview) == (W, n_reduced) || error("materialize_homogeneous_dense_G_reduced!: size(Gview)=$(size(Gview)) != (W,n_reduced)=($W,$n_reduced)")
    e = zeros(n_reduced)
    @inbounds for j in 1:n_reduced
        e[j] = 1.0
        Gview[:, j] .= reduced_homogeneous_dual_contraction(e, cf, ctx, θ_full, layout)
        e[j] = 0.0
    end
    return Gview
end
