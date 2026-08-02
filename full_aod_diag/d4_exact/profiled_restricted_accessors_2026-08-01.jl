# ============================================================================
# Restricted-inner-endtoend task (2026-08-01), addendum: stable typed accessor
# surface for the SEPARATE outer-gradient/A-gp workstream. DIAGNOSTIC/SHARED-
# READ-ONLY -- wraps already-existing, already-validated objects
# (ProfiledEconomicMomentLayout, AnchorSpec, the reduced family contexts this
# session built/reused), adds no new numerical logic of its own. Does not
# touch anything the outer-gradient workstream owns (A/gp outer-gradient
# wrappers, ProfiledLFixCache, incremental winner-update gradient code,
# gravity-pivot chain rule, outer A/B harnesses) -- this file is purely
# READ access to reduced-layout structure, not gradient computation.
#
# Naming matches the addendum's own requested names exactly:
#   profiled_economic_layout(ctx)
#   economic_dual_range(cctx_or_octx)
#   restriction_dual_ranges(cctx_or_octx)
#   profiled_anchor_spec(ctx)
#   profiled_outer_coordinate_layout(ctx)
# ============================================================================

isdefined(Main, :ProfiledEconomicMomentLayout) || error("profiled_restricted_accessors_2026-08-01.jl requires profiled_economic_moment_layout_2026-08-01.jl to be included first.")
isdefined(Main, :build_profiled_ab_spec_pe) || error("profiled_restricted_accessors_2026-08-01.jl requires profiled_outer_evaluator_2026-08-01.jl to be included first (for build_profiled_ab_spec_pe).")

"""
    profiled_anchor_spec(ctx; global_overrides=Dict{Int,Int}()) -> AnchorSpec

The anchor MAP (one omitted destination-slot cell per active destination) shared by every layer of
the profiled architecture -- the outer relative-A coordinate (`relative_a_coordinate_2026-07-31.jl`),
the reduced economic-moment layout below, and (via `build_profiled_ab_spec_pe`) the outer coordinate
layout. Thin wrapper around `build_anchor_spec_from_ctx` (`profiled_economic_moment_layout_2026-08-01.jl`)
-- rectangular-safe (correct for a non-contiguous omitted destination too, unlike
`default_anchor_spec`'s square/contiguous-only own-cell rule).
"""
profiled_anchor_spec(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}()) =
    build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)

"""
    profiled_economic_layout(ctx; global_overrides=Dict{Int,Int}(), has_france_ratio=nothing) -> ProfiledEconomicMomentLayout

The shared, family-agnostic reduced economic-moment layout (task §6): one retained bilateral
homogeneous-factual-moment column per (origin,active-destination) cell except the anchor cell, plus
at most one France counterfactual-ratio column. IDENTICAL across all five families -- built once from
`profiled_anchor_spec(ctx)` and a probe `CompressedFactual` (only used to read `cf.cf_col > 0`, a
fixed structural fact of `ctx`, not point-dependent -- any valid `cf` suffices). `has_france_ratio`
can be supplied directly (e.g. by a caller that already has a `cf` in hand) to skip the probe build.
"""
function profiled_economic_layout(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}(),
        has_france_ratio::Union{Nothing,Bool} = nothing)
    spec = profiled_anchor_spec(ctx; global_overrides = global_overrides)
    hfr = has_france_ratio
    if hfr === nothing
        x_free_calib = ctx.θ0_up[ctx.free_idx]
        θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
        cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
        hfr = cf_probe.cf_col > 0
    end
    return build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = hfr)
end

"""
    profiled_outer_coordinate_layout(ctx; global_overrides=Dict{Int,Int}()) -> (spec, gauge, pe)

The OUTER relative-A coordinate layout (task §14's "profiled start must be produced by reducing the
exact full calibration point" convention) -- `(AnchorSpec, gauge, PivotGravityElimOnRetained)`, built
against the GENUINE calibration z-matrix (`ctx.θ0_up`'s own A_od block), never the gravity-elimination
pivot's `zfree=0` reference point (this repo's own standing CLAUDE.md warning: A_od==1 is not
calibration). Thin wrapper around `build_profiled_ab_spec_pe`
(`profiled_outer_evaluator_2026-08-01.jl`) -- the SAME function `evaluate_profiled_point`/
`reduce_calibration_to_w_profiled` already use, not a new construction.

NOTE for the outer-gradient workstream: this uses the SAME `spec` (`AnchorSpec`) as
`profiled_economic_layout(ctx)` above internally would if called with the same `global_overrides` --
both layers share one anchor MAP, per this session's own scope note. This function returns the
OUTER-coordinate triple (origin-fast `i=o+(d-1)*D` indexing, `relative_a_coordinate_2026-07-31.jl`
convention), which is NOT directly comparable index-for-index to `profiled_economic_layout`'s own
`retained_full_factual_j` (destination-fast `j=slot+(o-1)*Ddest` indexing,
`compressed_moments.jl`/`cc_algo/active_layout.jl` convention) -- see that struct's own docstring for
the full "never compare these two `j`/`i` numberings directly" warning.
"""
profiled_outer_coordinate_layout(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}()) =
    build_profiled_ab_spec_pe(ctx; global_overrides = global_overrides)

# ----------------------------------------------------------------------------
# economic_dual_range / restriction_dual_ranges: per-built-context dual-vector
# index ranges. Dual index 1 is always the zeta/intercept column (every
# family's own x[1]=zeta, x[2:end]=beta convention) -- these accessors
# describe how x[2:end] (equivalently, obj.d's own column layout: index k in
# G/H corresponds to dual index k+1) is carved up for a GIVEN, already-built
# reduced family context. Multiple dispatch per family's own concrete context
# type -- each method reads ONLY fields that context type already carries
# (cctx.NCORE/cctx.ncm/cctx.frechet_ext_cache for CM-family contexts,
# octx.NCORE/octx.n_eta for origin-ZC), no new bookkeeping invented.
# ----------------------------------------------------------------------------

"""
    economic_dual_range(cctx::CMBinHessCtx) -> UnitRange{Int}
    economic_dual_range(octx::OriginZCCoreHessCtx) -> UnitRange{Int}

Dual-vector index range (1-based). Dual index 1 is `zeta`. `G`'s column `k` (`k=1:d`) corresponds to
dual index `k+1`, for every `k`. `G`'s OWN economic columns are `1:pregrav`
(`pregrav = layout.total_reduced_economic_moments`, confirmed directly from
`materialize_homogeneous_dense_G_reduced!`'s own fill target `Gtmp[:,1:pregrav]`), NOT `1:NCORE` --
`NCORE = 1 + pregrav`, but `G`'s column `NCORE` is the GRAVITY column
(`fill_gravity_column_into!(@view(Gtmp[:, ncore_full]), ...)`, present in EVERY ONE of
`wrap_moments_with_cm_archB`/`wrap_moments_with_cm_frechet_archB`/`wrap_moments_with_originzc`, same
pattern, `ncore_full == NCORE`), a SEPARATE "sole outer-only column" (this file's own top-of-file
notation comment) -- NOT part of either the economic block or any restriction block. So the economic
block occupies dual indices `2:NCORE` (`NCORE-1 = pregrav` slots), and `gravity_dual_index` below is
the SEPARATE single dual index `1+NCORE`, sitting between `economic_dual_range` and
`restriction_dual_ranges`.

This was gotten wrong twice before landing here (first as `2:NCORE` with no separate gravity
accounting, silently absorbing gravity's dual index into whichever range happened to be adjacent
depending on which bug was live; caught both times only by `q_decomposition`'s own numerical
cross-check against a directly-reconstructed `G[:,1:pregrav]*beta`, NOT by `verify_dual_ranges_
partition` alone, which cannot distinguish "right split, right total" from "wrong split, right
total" -- see this file's own commit history for the full account). Treat this docstring, not the
git history, as authoritative.

CM+ZC (`cctx.ncore_core < cctx.NCORE`, the widened-core case, `CMZC_WIDENED_CORE_FINDING_2026-08-01.md`)
is EXPLICITLY NOT SUPPORTED here -- errors loudly rather than silently returning a range that
conflates the true economic block with the widened mean/pair-ZC columns.
"""
function economic_dual_range(cctx::CMBinHessCtx)
    cctx.ncore_core == cctx.NCORE ||
        error("economic_dual_range: cctx.ncore_core=$(cctx.ncore_core) != cctx.NCORE=$(cctx.NCORE) -- " *
              "this is a CM+ZC widened-core context (see CMZC_WIDENED_CORE_FINDING_2026-08-01.md); " *
              "economic_dual_range's simple 2:NCORE range is not well-defined for it (would " *
              "conflate the true economic block with the widened mean/pair-ZC columns). Not supported.")
    return 2:cctx.NCORE
end
economic_dual_range(octx::OriginZCCoreHessCtx) = 2:octx.NCORE

"""
    gravity_dual_index(cctx::CMBinHessCtx) -> Int
    gravity_dual_index(octx::OriginZCCoreHessCtx) -> Int

The single dual-vector index (1-based) for the gravity moment -- `G`'s own column `NCORE` (see
`economic_dual_range`'s docstring), sitting between the economic block and the first restriction
block. `1 + cctx.NCORE`.
"""
gravity_dual_index(cctx::CMBinHessCtx) = 1 + cctx.NCORE
gravity_dual_index(octx::OriginZCCoreHessCtx) = 1 + octx.NCORE

"""
    restriction_dual_ranges(cctx::CMBinHessCtx) -> NamedTuple
    restriction_dual_ranges(octx::OriginZCCoreHessCtx) -> NamedTuple

Dual-vector index ranges (1-based) for each restriction block, in COLUMN ORDER, starting immediately
after `economic_dual_range`'s own last index. Keys/count/order depend on which family `cctx`/`octx`
was built for:
  - flexible CM (`cctx.frechet_ext_cache === nothing`): `(C = <ncm columns>,)`.
  - common Fréchet (`cctx.frechet_ext_cache isa CMFrechetExtension`): `(C = <ncm_cm columns>,
    F = <ncm_level columns>)` -- CM-grid block then level block, matching
    `wrap_moments_with_cm_frechet_archB`'s own `[core | CM | level | gravity]` column layout.
  - ZC-only (`octx::OriginZCCoreHessCtx`): `(Z = <n_eta columns>,)`.
CM+ZC is not supported (see `economic_dual_range`'s identical guard) -- `cctx.frechet_ext_cache`
alone cannot distinguish CM+ZC from flexible CM (both have it `=== nothing`), so this dispatches on
`cctx.ncore_core == cctx.NCORE` FIRST, matching `economic_dual_range`'s own check.
"""
function restriction_dual_ranges(cctx::CMBinHessCtx)
    cctx.ncore_core == cctx.NCORE ||
        error("restriction_dual_ranges: cctx.ncore_core=$(cctx.ncore_core) != cctx.NCORE=$(cctx.NCORE) -- " *
              "this is a CM+ZC widened-core context; not supported (see economic_dual_range's identical guard).")
    start = cctx.NCORE + 2   # immediately after economic_dual_range's own last index, 1+NCORE
    ext = cctx.frechet_ext_cache
    if ext isa CMFrechetExtension
        ncm_level = cctx.L
        ncm_cm = cctx.ncm - ncm_level
        return (C = start:(start + ncm_cm - 1), F = (start + ncm_cm):(start + cctx.ncm - 1))
    else
        return (C = start:(start + cctx.ncm - 1),)
    end
end
function restriction_dual_ranges(octx::OriginZCCoreHessCtx)
    start = octx.NCORE + 2   # immediately after economic_dual_range's own last index, 1+NCORE
    return (Z = start:(start + octx.n_eta - 1),)
end

"""
    verify_dual_ranges_partition(cctx_or_octx, total_dual_dim::Int) -> Bool

Structural self-check (not a numerical gate -- see `economic_dual_range`'s own docstring for why this
alone is NOT sufficient to catch a wrong-but-same-total-count split): confirms `economic_dual_range`/
`gravity_dual_index`/`restriction_dual_ranges` together, plus dual index 1 (zeta), partition
`1:total_dual_dim` exactly -- no gaps, no overlaps, no out-of-bounds. `total_dual_dim` should be
`1 + obj.outer_constr_index` for the reduced `obj` this context was built alongside (dual vector =
[zeta; economic; gravity; restrictions...], length `1 + outer_constr_index`).
"""
function verify_dual_ranges_partition(ctx_hess, total_dual_dim::Int)
    econ = economic_dual_range(ctx_hess)
    grav = gravity_dual_index(ctx_hess)
    restr = restriction_dual_ranges(ctx_hess)
    all_ranges = vcat([econ, grav:grav], collect(values(restr)))
    covered = falses(total_dual_dim)
    covered[1] = true   # zeta, index 1, not part of either accessor's own range
    for r in all_ranges
        for i in r
            (1 <= i <= total_dual_dim) || return false
            covered[i] && return false   # overlap
            covered[i] = true
        end
    end
    return all(covered)
end

# ============================================================================
# Addendum (2026-08-01): additional hooks requested by the outer-bridge
# workstream. Still read-only / structural -- no outer gradient, no
# restriction-parameter gradient assembly, no outer runner, no A/B harness.
# ============================================================================

"""
    stable_inner_layout_fields(ctx; global_overrides=Dict{Int,Int}()) -> NamedTuple

A stable, minimal bundle of structural facts the outer bridge needs without reaching into any
built reduced context's own internal fields directly. Every field here is read from `ctx`/the
shared layout objects above -- no new computation.
"""
function stable_inner_layout_fields(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    layout = profiled_economic_layout(ctx; global_overrides = global_overrides)
    spec = profiled_anchor_spec(ctx; global_overrides = global_overrides)
    return (D = ctx.D, Ddest = layout.Ddest,
            destination_ids = layout.destination_ids, anchor_origin_by_slot = layout.anchor_origin_by_slot,
            n_bilateral_reduced = length(layout.retained_full_factual_j),
            has_france_ratio = layout.france_ratio_reduced_j > 0,
            n_economic_reduced = layout.total_reduced_economic_moments,
            anchor_spec = spec, economic_layout = layout)
end

"""
    restriction_outer_parameter_layout(cctx::CMBinHessCtx) -> NamedTuple
    restriction_outer_parameter_layout(octx::OriginZCCoreHessCtx) -> NamedTuple

For a built reduced family context, the PHYSICAL outer parameter each restriction dual-vector slot
(from `restriction_dual_ranges`) corresponds to -- structural metadata only, read directly off
fields the context/its `aug` already carry (theta-independent, computed once at context-build time),
no new numerical logic:
  - flexible CM: `(C = (origins=cctx.origins, refIndex1=cctx.refIndex1, z=<per-level threshold>, L=cctx.L),)`
    -- column `(oi-1)*L + l` of the C block is the CDF contrast for origin `origins[oi]` (vs.
    `refIndex1`) at threshold level `l` of `L`.
  - common Fréchet: `(C = <as above>, F = (level_targets = ext.level_targets, L = cctx.L),)` -- column
    `l` of the F block targets `level_targets[l]`.
  - ZC-only: `(Z = (K_mean = octx.fg_layout/hzz_zc_layout's own K_mean, K_pair = ..., D = ...),)` --
    columns `1:K_mean*D` are per-(origin,mean-power-level) targets, remaining `K_pair*npair` columns
    are per-(pair,pair-power-level) targets (SAME layout `n_eta`/`mean_targets`/`pair_targets`
    already use to interpret `nu_full` -- `cm_originzc_moments.jl`).

Requires `cctx.z`/`cctx.frechet_ext_cache` (CM-family) or `octx.hzz_zc_layout` (origin-ZC) to already
be populated -- call AFTER the context has been used in at least one real FG/Hessian callback (same
precondition `restriction_dual_ranges` has for `frechet_ext_cache`).
"""
function restriction_outer_parameter_layout(cctx::CMBinHessCtx)
    cctx.ncore_core == cctx.NCORE ||
        error("restriction_outer_parameter_layout: CM+ZC widened-core context; not supported (see economic_dual_range's identical guard).")
    C_meta = (origins = cctx.origins, refIndex1 = cctx.refIndex1, z = cctx.z, L = cctx.L)
    ext = cctx.frechet_ext_cache
    if ext isa CMFrechetExtension
        return (C = C_meta, F = (level_targets = ext.level_targets, L = cctx.L))
    else
        return (C = C_meta,)
    end
end
function restriction_outer_parameter_layout(octx::OriginZCCoreHessCtx)
    layout = octx.hzz_zc_layout
    layout === nothing && error("restriction_outer_parameter_layout: octx.hzz_zc_layout is nothing -- octx.n_eta must be > 0 (this context has no restriction block).")
    return (Z = (K_mean = layout.K_mean, K_pair = layout.K_pair, D = octx.econ_ctx.D),)
end

"""
    _q_gravity(obj, gdi::Int, λstar::AbstractVector{Float64}) -> Vector{Float64}

Gravity's own per-draw dual contribution -- `G`'s column `NCORE` (`gdi = gravity_dual_index(...)`,
`H`'s column `gdi` under the SAME 1:1 dual-index-to-H-column correspondence `E = H[:,2:1+NCORE]`
establishes), read DIRECTLY from `obj.H` (already populated by the caller's own prior `moments!`
call -- never re-derived via `compressed_gravity_raw` independently, which would risk yet another
hand-derivation bug of the kind this session's own `materialize_homogeneous_dense_G_reduced!`
docstring warns against). Sign/scale matches `_archC_prep_for_hessian!`'s/`operator_prep_for_hessian!`'s
own `arg0 = -H_view*x` convention exactly (each dual-indexed column contributes `-H[:,k]*x[k-1]`)
-- gravity is A/gp-DEPENDENT (it is itself a function of the A_od block, confirmed empirically: this
session's own D4 gate showed `q_restriction` picking up a spurious ~2x-the-economic-delta shift
under an A_od perturbation before this piece was separated out), so it must NOT be folded into
`q_restriction`.
"""
function _q_gravity(obj, gdi::Int, λstar::AbstractVector{Float64})
    G_view = CS.select_G_from_H(obj, obj.H)
    ncore = gdi - 1
    return -G_view[:, ncore] .* λstar[ncore]
end

"""
    q_decomposition(cctx_or_octx, obj, ζstar::Float64, λstar::AbstractVector{Float64}) -> NamedTuple

Diagnostic accessor for the complete per-draw dual index `q` at a VERIFIED FIXED dual point
(`ζstar`, `λstar` -- e.g. `base.ζstar`/`base.λstar` from `archC_base_state`/`archC_frechet_base_state`/
`archOZ_base_state`), decomposed into `q_economic` (A/gp-DEPENDENT -- the only piece an A/gp outer
gradient call needs to re-differentiate), `q_gravity` (ALSO A/gp-DEPENDENT -- gravity is itself a
function of the A_od block, see `_q_gravity`'s own docstring), and `q_restriction` (A/gp-INDEPENDENT
at fixed restriction parameters -- held fixed during an A/gp gradient call, per the outer bridge's
own stated convention).

Design (matches this session's own "safe by linearity, verify against production, never hand-derive
a parallel formula" discipline used throughout Phase 8): `q_total` is read DIRECTLY from `obj.arg0`
after re-priming it at `x=[ζstar;λstar]` via the EXISTING, already-validated
`_prep_dual_index_for_archC!`/`_prep_dual_index_for_archA!` (the same dual-index refresh every real
Hessian callback in this codebase already calls -- NOT re-derived here). `q_economic` is computed via
the EXISTING, already-validated `reduced_homogeneous_dual_contraction` (this session's own repeatedly
gated "safe by linearity" kernel) at the economic sub-vector of `λstar`. `q_gravity` is read directly
off `obj.H`'s own gravity column (see `_q_gravity`). `q_restriction` is obtained by SUBTRACTION
(`q_total - q_economic - q_gravity`, using `q_total`'s own already-correct sign convention, whatever
it is) rather than by independently re-deriving a restriction forward formula (which this session's
own `materialize_homogeneous_dense_G_reduced!` docstring documents TWICE produced a fresh, different
bug when hand-derived) -- so `q_restriction` is correct BY CONSTRUCTION relative to production's own
`q_total`, not by trusting a parallel derivation.

NORMALIZATION / SW WEIGHTING (explicit, per the outer bridge's own requirement): `q_total`/
`q_economic`/`q_gravity`/`q_restriction` are all per-draw, UNWEIGHTED by sampling weights `SW`/`nu` --
weighting (if any) happens downstream, inside `Psi!`/`dPsi!`/`ddPsi!`'s own `arg1`/`arg2` consumers
and the `sum(...)/M` normalization in the objective functor (`PsiObjectiveBundle.jl`), never inside
`q` itself. `θ_full` passed to `reduced_homogeneous_dual_contraction` must be the CURRENT outer
point's theta (the same one this `cctx`/`octx`'s `cf`/`profiled_theta_ref` were built/published at,
AND the same one `obj.H` was last refreshed at via a real `moments!` call -- `q_gravity` reads `obj.H`
directly, so a stale `obj.H` silently produces a stale `q_gravity`) -- if the caller is mid an A/gp
probe with the restriction dual fixed but theta perturbed, `q_economic`/`q_gravity` genuinely change
(A/gp-DEPENDENT) while `q_restriction`, evaluated at the SAME fixed `λstar` restriction sub-vector, is
unaffected by the perturbation to the extent the restriction columns themselves don't depend on theta
(true for every restriction type this session touched -- CM-grid/level/mean-pair-ZC columns are all
built from `Bidx`/`U` alone, theta-independent, confirmed by direct read of
`fill_cm_columns_from_bins!`/`fill_frechet_level_columns_from_bins!`/`mean_columns_direct!`/
`pair_columns!`'s own signatures -- none take `θ`).

✅ RESOLVED (2026-08-02, integration/profiled-restricted-production-ready-2026-08-02): the ~2x
discrepancy was a missing sign negation in `q_economic`, not a missing/mis-scaled component.
`reduced_homogeneous_dual_contraction` returns the RAW `+G*β` contraction (the same convention
`materialize_homogeneous_dense_G_reduced!` uses to build `G`'s own columns directly, unnegated), but
`q_total`/`obj.arg0` is built via a uniform `-H*x` convention for EVERY dual-indexed column
(`_archC_prep_for_hessian!`'s own `BLAS.gemv!('N', 1.0, H[:, 2:1+outer_constr_index], -x, 0.0, arg0)`),
and `_q_gravity` already correctly applies that negation (`-G_view[:,ncore] .* λstar[ncore]`).
`q_economic` did not. Fix: negate it (`q_economic = -reduced_homogeneous_dual_contraction(...)`),
consistent with the SAME convention the outer bridge's own already-passing q-decomposition gate uses
independently (`profiled_restricted_q_decomposition_gate_2026-08-01.jl`'s own
`q_recon = -zeta - reduced_homogeneous_dual_contraction(...) - restriction_contrib0`). Verified live:
pre-fix `max|Δq_economic under pert|=1.065e-3`, `max|Δq_restriction under pert|=2.129e-3` (ratio
exactly 2.0); post-fix `max|Δq_restriction under pert|=2.776e-17` (machine precision). The three
earlier index-bug fixes (gravity/economic boundary, economic column slicing, gravity's own dual slot)
were real and necessary but insufficient on their own -- this sign fix is what closed the gap to zero.
See `Q_DECOMPOSITION_KNOWN_ISSUE_2026-08-01.md` for the original investigation log (superseded by this
resolution) and the integration master report for the full root-cause writeup.
"""
function q_decomposition(cctx::CMBinHessCtx, obj, cf, θ_full::AbstractVector, ζstar::Float64, λstar::AbstractVector{Float64})
    layout = cctx.profiled_layout
    layout === nothing && error("q_decomposition: cctx.profiled_layout is nothing -- this accessor is for reduced/profiled contexts only.")
    x = vcat(ζstar, λstar)
    _prep_dual_index_for_archC!(cctx, obj, x)
    q_total = copy(obj.arg0)
    er = economic_dual_range(cctx)
    # er (economic_dual_range) now spans EXACTLY G's economic columns 1:pregrav (dual indices
    # 2:NCORE, see that accessor's own corrected docstring) -- a direct dual-index-to-lambda_star
    # mapping (dual index k -> lambda_star[k-1]), no further offset needed.
    β_econ = λstar[er .- 1]
    # SIGN FIX (2026-08-02, resolves the q-decomposition known issue): arg0/q_total is built via the
    # SAME -H*x convention for every dual-indexed column (_archC_prep_for_hessian!'s own
    # `BLAS.gemv!('N', 1.0, H[:, 2:1+outer_constr_index], -x, 0.0, arg0)`, and _q_gravity's own
    # `-G_view[:,ncore] .* λstar[ncore]`). reduced_homogeneous_dual_contraction, however, returns the
    # RAW +G*β contraction unnegated (exactly the convention materialize_homogeneous_dense_G_reduced!
    # uses to build G's own columns, i.e. G[:,j] itself, not -G[:,j]). Without this negation,
    # q_economic carried the WRONG sign relative to q_total/q_gravity, so
    # `q_restriction = q_total - q_economic - q_gravity` picked up an extra `+2*(true q_economic)`
    # term -- exactly the reproducible "q_restriction changes by ~2x q_economic's own change"
    # discrepancy (Q_DECOMPOSITION_KNOWN_ISSUE_2026-08-01.md). Confirmed live: pre-fix
    # max|Δq_economic|=1.065e-3, max|Δq_restriction|=2.129e-3 (ratio exactly 2.0).
    q_economic = -reduced_homogeneous_dual_contraction(β_econ, cf, cctx.econ_ctx, collect(θ_full), layout)
    q_gravity = _q_gravity(obj, gravity_dual_index(cctx), λstar)
    q_restriction = q_total .- q_economic .- q_gravity
    return (q_total = q_total, q_economic = q_economic, q_gravity = q_gravity, q_restriction = q_restriction,
            economic_dual_range = er, gravity_dual_index = gravity_dual_index(cctx),
            restriction_dual_ranges = restriction_dual_ranges(cctx))
end
function q_decomposition(octx::OriginZCCoreHessCtx, obj, cf, θ_full::AbstractVector, ζstar::Float64, λstar::AbstractVector{Float64})
    layout = octx.profiled_layout
    layout === nothing && error("q_decomposition: octx.profiled_layout is nothing -- this accessor is for reduced/profiled contexts only.")
    x = vcat(ζstar, λstar)
    _prep_dual_index_for_archA!(octx, obj, x)
    q_total = copy(obj.arg0)
    er = economic_dual_range(octx)
    # er (economic_dual_range) now spans EXACTLY G's economic columns 1:pregrav (dual indices
    # 2:NCORE, see that accessor's own corrected docstring) -- a direct dual-index-to-lambda_star
    # mapping (dual index k -> lambda_star[k-1]), no further offset needed.
    β_econ = λstar[er .- 1]
    # SIGN FIX (2026-08-02): see the identical fix + rationale in the CMBinHessCtx method above.
    q_economic = -reduced_homogeneous_dual_contraction(β_econ, cf, octx.econ_ctx, collect(θ_full), layout)
    q_gravity = _q_gravity(obj, gravity_dual_index(octx), λstar)
    q_restriction = q_total .- q_economic .- q_gravity
    return (q_total = q_total, q_economic = q_economic, q_gravity = q_gravity, q_restriction = q_restriction,
            economic_dual_range = er, gravity_dual_index = gravity_dual_index(octx),
            restriction_dual_ranges = restriction_dual_ranges(octx))
end
