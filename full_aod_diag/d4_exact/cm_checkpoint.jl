# ============================================================================
# CM production checkpoint schema (allocation/cache cleanup task, §11).
#
# GAP (confirmed by direct code reading): `run_cm_upper` (cm_outer_driver.jl) has NO
# checkpoint/resume support at all -- it is a single-shot KNITRO call with no periodic
# save and no resume path. The one place CM runs ARE checkpointed today
# (`c13_d20_cm_upper_continuation.jl`'s `save_stage`) writes a bare, ad-hoc NamedTuple
# (`L, probs, kappa, knitro_status, wall, n_eval, n_grad, best_w, best_Delta, xsol,
# timestamp`) with NO schema version, NO draw-design/checksum validation, NO KNITRO-version
# field, and no corresponding load/resume logic whatsoever -- it is write-only, used only to
# hand `best_w` forward into the NEXT (coarser->finer) grid stage, never to resume an
# INTERRUPTED run of the SAME stage. This is a materially weaker guarantee than the
# unrestricted path's schema-3 `D20Checkpoint` (c10_d20_production_driver.jl), which
# validates draw checksums, KNITRO version, and refuses to resume an incompatible config.
#
# This file adds `CMCheckpoint` (same rigor as `D20Checkpoint`, plus the CM-specific fields
# the brief lists: grid size, exact cutpoints, contrasts/basis, Hessian backend) and
# `run_cm_upper_checkpointed`, a NEW wrapper (additive -- `run_cm_upper`/`cm_outer_driver.jl`
# are UNCHANGED) that periodically checkpoints the best exact-feasible incumbent (matching
# `run_polish_checkpointed`'s own `:new_best`/`:wall_interval`/`:stage_complete` discipline)
# and can resume from a prior `CMCheckpoint`, rebuilding `ctx`/`pcx` from the checkpoint's own
# recorded (W, δ, draw_design, draw_seed, L, contrasts, probs) and hard-refusing resume if the
# regenerated draw checksums don't match (same guarantee schema-3 gives the unrestricted path).
#
# Verified in test_cm_checkpoint.jl: an interrupted-then-resumed run (fresh process) reproduces
# the same best-feasible incumbent's Delta/kappa to bit-for-bit agreement.
# ============================================================================
using Serialization, Dates

isdefined(Main, :with_blas_threads) || include(joinpath(@__DIR__, "blas_thread_policy.jl"))   # allocation/Hessian port task §6.3
isdefined(Main, :set_production_outer_algorithm!) || include(joinpath(@__DIR__, "knitro_outer_algorithm.jl"))   # allocation/Hessian port task §1.3/§4: opt-in pinned outer algorithm for matched benchmarks only -- see that file's module docstring; NOT applied unless a caller passes pin_outer_algorithm=true
isdefined(Main, :print_production_backend_manifest) || include(joinpath(@__DIR__, "production_backend_manifest.jl"))   # allocation/Hessian port task §2: central production backend manifest
isdefined(Main, :CMProductionEvalKey) || include(joinpath(@__DIR__, "cm_exact_cache_production.jl"))   # Phase C remediation (2026-07-26): exact-point cache for this driver's real pcx shape
isdefined(Main, :is_better_polish) || include(joinpath(@__DIR__, "incumbent_logic.jl"))   # 2026-07-28 lower-direction wiring: pure, KNITRO-free find_smallest-aware incumbent comparison, reused (not re-derived) from the unrestricted family's own validated helper
isdefined(Main, :CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED) || include(joinpath(@__DIR__, "cm_hessian_subblock_profiling.jl"))   # D=20 profiling task (2026-07-28): opt-in live-pcx stash this function writes below, default off
isdefined(Main, :prepare_production_run) || include(joinpath(@__DIR__, "production_bundle_api.jl"))   # architecture/production-operator-bundle-hardening-2026-07-30
isdefined(Main, :default_gravity_exclude_cells_brazil_korea) || include(joinpath(@__DIR__, "country_resolve.jl"))
isdefined(Main, :aod_pow_matrix) || include(joinpath(@__DIR__, "compressed_live.jl"))   # k=(sigma-1) narrow fix: aod_pow_matrix
isdefined(Main, :autarky_cf_scalars) || include(joinpath(@__DIR__, "autarky_cf.jl"))   # k=(sigma-1) narrow fix: autarky_cf_scalars
isdefined(Main, :CallbackHealthRecord) || include(joinpath(@__DIR__, "cm_callback_health.jl"))   # 2026-08-06 outer-production-closeout: fake-success guard + assert_two_family_capabilities!

"""
    assert_two_family_capabilities!(label, pcx, is_meanzc, is_frechet, include_truncated_moment, threaded_bins)

2026-08-06 outer-production-closeout task, section 4: replaces the blanket
`include_truncated_moment=true` refusal that used to live at the top of `run_cm_upper_checkpointed`
(see that function's own RELAXATION HISTORY comment) with a truthful, POST-build capability check
driven by the ACTUAL constructed `pcx`/`cctx` this run will use -- not a static claim that "the code
was written to support this". A caller may proceed only when every applicable check below passes;
each one inspects a real field on the live context, and each corresponds to a concrete historical
bug this session's investigation found (missing `cctx.Pow`, missing `cctx.tls` under
`threaded_bins=true`).
"""
function assert_two_family_capabilities!(label::AbstractString, pcx, is_meanzc::Bool, is_frechet::Bool,
                                          include_truncated_moment::Bool, threaded_bins::Bool)
    include_truncated_moment || return nothing   # single-family (:cm_only, CDF-basis-only) path -- untouched
    cctx = pcx.cctx
    cctx === nothing &&
        error("$label: include_truncated_moment=true requires an operator-backed CMBinHessCtx (cctx) -- " *
              "got nothing (this pcx carries a dense-reference bundle). Dense is not a production path.")
    cctx.n_families == 2 ||
        error("$label: include_truncated_moment=true requires a genuinely two-family context " *
              "(cctx.n_families==2), got n_families=$(cctx.n_families) -- the two-family CM spec " *
              "(eq.35+eq.36) was requested but the constructed context does not actually carry it.")
    cctx.Pow === nothing &&
        error("$label: two-family capability check FAILED -- cctx.Pow is nothing despite " *
              "cctx.n_families==2. This is exactly the missing-Pow-wiring defect fixed in commit " *
              "895b99b (cm_meanzc_lookup_production.jl) for CM+ZC -- refusing to proceed rather than " *
              "silently constructing an operator state with the wrong dual layout.")
    if is_meanzc
        cctx.inner_fg_backend === :operator ||
            error("$label: two-family CM+ZC requires inner_fg_backend=:operator (the only backend with " *
                  "a two-family-aware forward/backward FG kernel, CMMeanZCOperatorState) -- got " *
                  ":$(cctx.inner_fg_backend).")
        cctx.meanzc_zc_op === nothing &&
            error("$label: two-family CM+ZC requires cctx.meanzc_zc_op (ZCRestrictionOperator) to be built -- got nothing.")
    end
    if threaded_bins && cctx.use_threaded_bins
        cctx.tls === nothing &&
            error("$label: threaded_bins=true requires cctx.tls to be constructed " *
                  "(ThreadLocalBinScratch) -- got nothing. This is exactly the " *
                  "threaded_bins=true/tls=nothing defect the D4 common-Frechet two-family Hessian " *
                  "gate caught -- refusing to silently fall back to serial rather than erroring " *
                  "before KNITRO starts.")
    end
    return nothing
end

const CM_CHECKPOINT_SCHEMA = 10

"""
    meanzc_profiled_nu_value(xf, ctx) -> Float64

k=(sigma-1) exact-collinearity narrow fix (see `run_cm_upper_checkpointed`'s own
`meanzc_profiled_level` docstring for the full mechanism/evidence). Computes the SAME
target the autarky/counterfactual price-index moment (`autarky_cf.jl`, column D^2+1)
already enforces for the focal country `bi=baseIndex`: `E[z_bi(w)^(sigma-1)] =
cf_denom/cf_num`. `xf` is the economic free-parameter vector (`x_free`, i.e. `[gp;
vec(A_od free)]`, the SAME argument `cm_meanzc_production_value_verified_screened`/
`cm_meanzc_production_gradient(_cplus)` already take).
"""
function meanzc_profiled_nu_value(xf::AbstractVector{Float64}, ctx)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    AodPow = aod_pow_matrix(θ_full, ctx)
    γ_prime_bi = θ_full[3+ctx.D]
    cf_num, cf_denom, _ = autarky_cf_scalars(ctx.obj, AodPow, ctx.σ, γ_prime_bi)
    return cf_denom / cf_num
end
# Bumped 8 -> 9 (transformed-A restricted-family port, 2026-07-26 production-audit task addendum;
# whole-tree CMCheckpointV* grep confirms V9 unclaimed -- cm_originzc_checkpoint.jl currently ends
# at V7, so this bump does not collide there; that file's own analogous bump uses V10, keeping the
# two files' shared-namespace numbering interleaved and non-colliding, same discipline as every
# prior bump on either file): adds `A_coordinate_mode::Symbol` (:legacy_z | :powered_aspace) to the
# persisted schema. `zfree` remains ALWAYS canonical z-space regardless of this field's value (the
# outer search coordinate the checkpoint's OWN run actually used) -- this field exists purely so a
# resume knows which coordinate w0 was reconstructed in at write time, matching
# outer_coordinate_layout.jl's D20CheckpointUnified's identical "zfree always genuine z-space, plus
# a separate mode-tag field" discipline for the unrestricted family.
# Bumped 6 -> 8 (fixed-Frechet-as-CM-plus-anchor production port, 2026-07-25/26; skips 7, already
# taken by cm_originzc_checkpoint.jl's own CMCheckpointV7 -- see CM_CHECKPOINT_SCHEMA's own comment
# block below and the whole-tree CMCheckpointV* grep this bump was checked against): adds
# `marginal_restriction::Symbol` (:common_flexible | :common_frechet) to the persisted schema, so a
# checkpoint records which restriction family (plain flexible CM vs CM-plus-common-level-anchor
# fixed Frechet) it was written under -- CM_frechet's own thresholds/targets/basis are NOT
# separately persisted, since they are already fully determined, deterministically, by the
# EXISTING persisted fields (cm_L, cm_probs, cm_contrasts, ctx.D) exactly the same way flexible
# CM's own thresholds already are -- no new non-deterministic state to capture.
# Bumped 4 -> 6 (destination_sample production wiring, exclude-ROW-destination release,
# 2026-07-24): adds destination_sample, row_idx, D_dest to the persisted schema, so a checkpoint
# records WHICH destination-sample regime (:all_legacy square D x D vs :exclude_row true-shrink
# D x (D-1)) it was written under -- required because the two regimes have DIFFERENT n_free (D^2
# vs D*D_dest), so resuming under a mismatched regime would silently misinterpret the persisted
# `zfree` vector's length/meaning. New type name for the same Julia-Serialization reason
# CMCheckpointV3/V4 themselves exist (see the 2->3 bump's comment below). NOTE: the natural next
# name "CMCheckpointV5" is ALREADY TAKEN -- cm_originzc_checkpoint.jl (a separate, already-existing
# extension for the origin-specific-ZC restriction family) defines `CMCheckpointV5` for an
# unrelated field set (distribution_restriction/origin_K_mean/etc.). Reusing that name here would
# silently redefine/shadow a DIFFERENT struct with a DIFFERENT field layout, so this bump jumps
# straight to `CMCheckpointV6`, skipping 5. cm_originzc_checkpoint.jl's own destination_sample bump
# (this same release) similarly uses `CMCheckpointV7` (extending V5), not V6.
# Bumped 3 -> 4 (CM+moments(+ZC) production integration, 2026-07-23): adds cm_extension,
# meanzc_K_mean, meanzc_K_pair, meanzc_basis, eta_nu, moment_layout_version to the persisted
# schema, generalizing the checkpoint to cover the (:cm_only | :cm_plus_equal_means |
# :cm_plus_equal_means_zero_covariance | :cm_plus_moments) extension family at any (K_mean,K_pair)
# -- ONE checkpoint type for the whole family, not a parallel CMMeanZCCheckpoint universe, so the
# same production stage runner/supervisor/cold-verifier handles every arm. Same Serialization
# gotcha as the 2->3 bump applies (see that comment below) -- CMCheckpointV3 kept permanently
# unchanged, CMCheckpointV4 is a new type name.
# Bumped 1 -> 2 (remediation task Part A, finding F1): schema-1 checkpoints computed cb_F!'s
# reported/constrained Delta as `-base.ζstar`, which silently omits mean(Psi(q*)) and overstates
# the divergence at tail-active points (any draw with recovered weight m* > e). A schema-1
# checkpoint's `best_feasible.Delta` and `feasible` flags are NOT trustworthy -- do not resume
# from one. Use `migrate_cm_checkpoint_v1_candidate` to recover just the incumbent vector as a
# fresh start point, then cold-re-evaluate it with `cm_production_value_verified` before trusting
# any Delta/feasibility for it.
#
# Bumped 2 -> 3 (Part II.4 follow-up, 2026-07-23): adds `cm_gradient_backend` to the persisted
# schema (see `CMCheckpointV3`'s own field comment). IMPORTANT Julia-Serialization gotcha,
# discovered live this session (not assumed): `Serialization` resolves a struct field-for-field by
# the TYPE NAME recorded inside the file, looked up in the CURRENT session -- it is NOT safe to
# just add a field to the EXISTING `CMCheckpoint` struct under the same name, because every old
# file's embedded type reference (`Main.CMCheckpoint`) would then resolve to the NEW, longer
# layout and misread the byte stream (confirmed empirically: this throws `EOFError`, not a clean/
# catchable type error, both for `deserialize(path)::CMCheckpoint` AND for an attempted read into
# a DIFFERENTLY-NAMED struct with the old layout -- the type lookup happens via the name IN THE
# FILE, not the annotation at the call site). The correct fix is to version the TYPE NAME: leave
# `CMCheckpoint` (below) permanently unchanged as the schema-1/2 legacy layout -- every existing
# checkpoint, including the entire completed `production_runs/cm_campaign_2026-07-22`, remains
# loadable through it forever -- and introduce `CMCheckpointV3` as a distinct type for schema>=3.
# `load_cm_checkpoint` tries `CMCheckpointV3` first, falls back to `CMCheckpoint`, and upgrades.
# Verified against a real copy of chain1/delta_0.1/stage_latest.jls, not merely asserted.

"""
    CMCheckpoint

Legacy (schema 1/2) checkpoint layout. Kept PERMANENTLY UNCHANGED for backward-compat reads only
-- every schema-1/2 file ever written (including the entire 2026-07-22 real production campaign)
has this exact type name+layout embedded in its own serialized bytes, and Julia's `Serialization`
looks up structs by that embedded name, not by whatever the current top-of-tree definition is (see
the `CM_CHECKPOINT_SCHEMA` comment above for how this was discovered). `save_cm_checkpoint` never
constructs this type -- new checkpoints are always `CMCheckpointV3`.
"""
struct CMCheckpoint
    schema::Int
    run_id::String
    label::String
    branch::Symbol                 # :cm_upper (only direction wired today; kept for parity/future :cm_lower)
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    # ---- CM configuration (task §11's explicit field list) ----
    cm_L::Int
    cm_probs::Vector{Float64}       # exact cutpoints -- NOT re-derived from L on resume, taken verbatim
    cm_contrasts::Symbol
    cm_grid_rule::Symbol            # :equal | :nested_family -- metadata only (probs is authoritative)
    cm_basis::Symbol                # :cumulative | :interval
    cm_hessian_backend::Symbol      # :structured | :dense_reference
    # ---- outer-point / incumbent state ----
    g::Float64
    zfree::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any              # NamedTuple or nothing
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol       # :new_best | :wall_interval | :stage_complete | :stage_complete_unverified
    knitro_version::String
end

"""
    CMCheckpointV3

CM-production checkpoint layout, schema>=3 (Part II.4 follow-up, 2026-07-23). Identical to
`CMCheckpoint` except for one new field, `cm_gradient_backend` (see below) -- given a NEW type
name specifically because Julia's `Serialization` cannot safely add a field to an existing struct
name (see `CM_CHECKPOINT_SCHEMA`'s comment). This is the type `save_cm_checkpoint` always
constructs going forward; `CMCheckpoint` (above) is retained only for reading old files.
"""
struct CMCheckpointV3
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol     # :reference | :cplus. Previously recorded ONLY in an
                                     # unversioned sidecar (<label>_gradient_backend.txt), never
                                     # validated on resume -- a checkpoint file alone could not
                                     # reveal which backend produced it. Now part of the schema.
    g::Float64
    zfree::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
end

"""
    CMCheckpointV4

CM-production checkpoint layout, schema>=4 (CM+moments(+ZC) production integration,
2026-07-23). Identical to `CMCheckpointV3` except six new fields, appended at the end:
`cm_extension`, `meanzc_K_mean`, `meanzc_K_pair`, `meanzc_basis`, `eta_nu`,
`moment_layout_version`. New type name for the same Julia-Serialization reason `CMCheckpointV3`
itself exists (see `CM_CHECKPOINT_SCHEMA`'s comment) -- `CMCheckpointV3` is retained permanently,
read-only, for every schema-3 file the CM-C+ campaign already wrote.
"""
struct CMCheckpointV4
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol            # :cm_only | :cm_plus_equal_means | :cm_plus_equal_means_zero_covariance | :cm_plus_moments
    meanzc_K_mean::Int              # 0 for :cm_only
    meanzc_K_pair::Int              # 0 for :cm_only
    meanzc_basis::Symbol            # :direct (only option implemented); irrelevant when cm_extension==:cm_only
    moment_layout_version::Int      # bump if wrap_moments_with_cm_meanzc's column order ever changes
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}         # length meanzc_K_mean; Float64[] for :cm_only
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
end

"""
    CMCheckpointV6

CM-production checkpoint layout, schema>=6 (destination_sample production wiring, exclude-ROW-
destination release, 2026-07-24). Identical to `CMCheckpointV4` except three new fields, appended
at the end: `destination_sample`, `row_idx`, `D_dest`. New type name for the Julia-Serialization
reason documented at `CM_CHECKPOINT_SCHEMA` above (also explains why this skips V5, already taken
by cm_originzc_checkpoint.jl's unrelated struct) -- `CMCheckpointV4` is retained permanently,
read-only, for every schema-4 file the CM-C+/meanzc campaigns already wrote (all of which are, by
construction, :all_legacy -- destination_sample did not exist as a runtime option at schema 4).
"""
struct CMCheckpointV6
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    # ---- NEW (schema 6): omit-ROW-destination true-shrink production option ----
    destination_sample::Symbol         # :all_legacy | :exclude_row
    row_idx::Union{Nothing,Int}        # nothing for :all_legacy; the excluded destination's index otherwise
    D_dest::Int                        # destination count; D_dest==D for :all_legacy
end

const MEANZC_MOMENT_LAYOUT_VERSION = 1   # wrap_moments_with_cm_meanzc's column order, cm_meanzc_moments.jl

"""
    CMCheckpointV8

CM-production checkpoint layout, schema>=8 (fixed-Frechet-as-CM-plus-anchor production port,
2026-07-25/26). Identical to `CMCheckpointV6` except one new field, appended at the end:
`marginal_restriction`. New type name for the Julia-Serialization reason documented at
`CM_CHECKPOINT_SCHEMA` above (also explains why this skips V7, already taken by
cm_originzc_checkpoint.jl's unrelated struct) -- `CMCheckpointV6` is retained permanently,
read-only, for every schema-6 file the destination_sample-era campaigns already wrote (all of
which are, by construction, :common_flexible -- :common_frechet did not exist as a runtime option
at schema 6).
"""
struct CMCheckpointV8
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
    # ---- NEW (schema 8): fixed-Frechet-as-CM-plus-anchor marginal restriction mode ----
    marginal_restriction::Symbol   # :common_flexible | :common_frechet
end

"""
    CMCheckpointV9

Identical to `CMCheckpointV8` except one new field, appended at the end: `A_coordinate_mode`
(transformed-A restricted-family port, 2026-07-26). `CMCheckpointV8` is retained permanently,
read-only, for every schema-8 file already written (all of which are, by construction,
`:legacy_z` -- `:powered_aspace` did not exist as a runtime option at schema 8).
"""
struct CMCheckpointV9
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
    marginal_restriction::Symbol
    # ---- NEW (schema 9): transformed-A restricted-family port ----
    A_coordinate_mode::Symbol   # :legacy_z | :powered_aspace
end

"""
    CMCheckpointV10

2026-08-05 truncated-power task: the CM moment-feature-family spec (`cm_moment_spec`) now varies
(previously every checkpoint ever written implicitly meant "eq.35 CDF-only", the ONLY spec that
existed) -- this is NOT a safely-inferable field the way every prior schema bump was (schema
6->8's `marginal_restriction` or schema 8->9's `A_coordinate_mode` could each only ever have had
one specific value in every pre-existing file; here, an old file's `zfree`/`dual_warm_start`/
`best_feasible` were computed against a DIFFERENT, half-width inner CM block -- resuming from one
under the new spec would silently corrupt the resumed state, not just mislabel it). Per this
task's brief: old (schema<=9, `CMCheckpointV9`-and-earlier) CDF-only checkpoints/dual banks must
HARD-REFUSE, not auto-upgrade -- see `load_cm_checkpoint`'s own schema-10 gate below, which
deliberately does NOT extend the schema-2..9 auto-upgrade chain to schema 10.

New fields (appended at the end, mirroring every prior schema bump's own convention):
- `cm_moment_spec::Symbol`: `:cdf_only` (`cm_feature_family_count=1`, eq.35 only) or
  `:cdf_plus_truncated_power_1msigma` (`cm_feature_family_count=2`, eq.35+eq.36) -- the two
  values `common_marginals_moments.jl::build_cm_augmented_obj`'s `include_truncated_moment`
  can actually produce.
- `cm_feature_family_count::Int`: `1` or `2`, redundant with `cm_moment_spec` (kept as a separate
  plain integer so a dimension check never has to string-compare a Symbol).
- `cm_feature_schema_version::Int`: bumped whenever the FEATURE CONSTRUCTION itself (not just the
  family count) changes in a way that would change the numeric CM block content at identical
  `(U, z, contrasts, σ)` -- `1` for this task's own construction (see
  `common_marginals_moments.jl::precalc_common_marginals_cdf`).
- `cm_feature_operator_checksum::String`: `hash(...)` of the resolved `(cm_moment_spec, L,
  cm_feature_schema_version, contrasts, ncm)` tuple (see `cm_feature_operator_fingerprint` below)
  -- a cheap, deterministic fingerprint of "which exact CM operator produced this checkpoint's
  dual vector," checked on resume in ADDITION to the existing draw-checksum/cm_L/cm_contrasts
  checks (belt-and-suspenders: catches a mismatch even if some future change touches the feature
  construction without bumping any of the human-maintained enum fields above).
"""
struct CMCheckpointV10
    schema::Int
    run_id::String
    label::String
    branch::Symbol
    find_smallest::Bool
    delta::Float64
    W::Int
    draw_seed::Int
    draw_design::Symbol
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    cm_L::Int
    cm_probs::Vector{Float64}
    cm_contrasts::Symbol
    cm_grid_rule::Symbol
    cm_basis::Symbol
    cm_hessian_backend::Symbol
    cm_gradient_backend::Symbol
    cm_extension::Symbol
    meanzc_K_mean::Int
    meanzc_K_pair::Int
    meanzc_basis::Symbol
    moment_layout_version::Int
    g::Float64
    zfree::Vector{Float64}
    eta_nu::Vector{Float64}
    logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}
    bandwidth_cache::Dict{Int,Float64}
    best_feasible::Any
    n_eval::Int
    n_grad::Int
    wall_elapsed::Float64
    wall_budget_remaining::Float64
    checkpoint_reason::Symbol
    knitro_version::String
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
    marginal_restriction::Symbol
    A_coordinate_mode::Symbol
    # ---- NEW (schema 10): CM moment-feature-family spec (2026-08-05 truncated-power task) ----
    cm_moment_spec::Symbol
    cm_feature_family_count::Int
    cm_feature_schema_version::Int
    cm_feature_operator_checksum::String
end

"""
    cm_feature_operator_fingerprint(cm_moment_spec, L, cm_feature_schema_version, contrasts, ncm) -> String

Deterministic fingerprint of the CM feature operator that produced a checkpoint's CM dual block --
see `CMCheckpointV10`'s own docstring for why this exists alongside the human-maintained enum
fields.
"""
cm_feature_operator_fingerprint(cm_moment_spec::Symbol, L::Int, cm_feature_schema_version::Int,
                                 contrasts::Symbol, ncm::Int) =
    string(hash((cm_moment_spec, L, cm_feature_schema_version, contrasts, ncm)))

"Atomic-ish checkpoint write, same discipline as `save_checkpoint` (D20Checkpoint): serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV10)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Atomic-ish checkpoint write, same discipline as `save_checkpoint` (D20Checkpoint): serialize to a .tmp file then mv, so a crash mid-write never leaves a half-written checkpoint."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV6)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Same discipline as the V6 method above -- new method (multiple dispatch), the V6 method is unchanged/untouched."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV8)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Same discipline as the V6/V8 methods above -- new method (multiple dispatch), those methods unchanged/untouched."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV9)
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Upgrades a legacy schema-2 `CMCheckpoint` to `CMCheckpointV3`, filling `cm_gradient_backend = :reference` -- CORRECT (not a guess) for every schema-2 file that exists, since :reference was the kwarg's own default throughout schema-2's entire lifetime and the only value the real 2026-07-22 campaign's stage runner ever passed (confirmed by direct read of cm_production_stage_runner.jl, which never sets cm_gradient_backend)."
function upgrade_schema2(old::CMCheckpoint)
    return CMCheckpointV3(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        :reference,   # cm_gradient_backend -- implicit, correct default for schema 2
        old.g, old.zfree, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version)
end

"Upgrades a schema-3 `CMCheckpointV3` (CM-C+ campaign, no meanzc extension ever existed at that schema) to `CMCheckpointV4`, filling cm_extension=:cm_only, meanzc_K_mean=meanzc_K_pair=0, meanzc_basis=:direct, eta_nu=Float64[] -- CORRECT (not a guess): the meanzc extension point did not exist anywhere in the codebase when any schema-3 file was written, so :cm_only is the only value consistent with those files' own provenance."
function upgrade_schema3(old::CMCheckpointV3)
    return CMCheckpointV4(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend,
        :cm_only, 0, 0, :direct, MEANZC_MOMENT_LAYOUT_VERSION,
        old.g, old.zfree, Float64[], old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version)
end

"Upgrades a schema-4 `CMCheckpointV4` (destination_sample did not exist as a runtime option at that schema) to `CMCheckpointV6`, filling destination_sample=:all_legacy, row_idx=nothing, D_dest=20 -- CORRECT (not a guess): `run_cm_upper_checkpointed` hardcodes `d20_real_setup_design` (this checkpoint system is D=20-only by construction), and every schema-4 file was written before `destination_sample` existed as a real option anywhere in this codebase, so :all_legacy/D_dest=20 is the only value consistent with those files' own provenance."
function upgrade_schema4_to_v6(old::CMCheckpointV4)
    return CMCheckpointV6(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version,
        :all_legacy, nothing, 20)
end

"Upgrades a schema-6 `CMCheckpointV6` (:common_frechet did not exist as a runtime option at that schema) to `CMCheckpointV8`, filling marginal_restriction=:common_flexible -- CORRECT (not a guess): every schema-6 file was written before :common_frechet existed anywhere in this codebase, so :common_flexible is the only value consistent with those files' own provenance."
function upgrade_schema6_to_v8(old::CMCheckpointV6)
    return CMCheckpointV8(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version, old.destination_sample, old.row_idx, old.D_dest,
        :common_flexible)
end

"Upgrades a schema-8 `CMCheckpointV8` (:powered_aspace did not exist as a runtime option at that schema) to `CMCheckpointV9`, filling A_coordinate_mode=:legacy_z -- CORRECT (not a guess): every schema-8 file was written before :powered_aspace existed anywhere in the restricted-family drivers, so :legacy_z is the only value consistent with those files' own provenance."
function upgrade_schema8_to_v9(old::CMCheckpointV8)
    return CMCheckpointV9(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version, old.destination_sample, old.row_idx, old.D_dest,
        old.marginal_restriction, :legacy_z)
end

"""
    load_cm_checkpoint(path) -> CMCheckpointV9

Tries the CURRENT (schema 9, `CMCheckpointV9`) shape first; falls back to schema-8
(`CMCheckpointV8`, upgraded via `upgrade_schema8_to_v9`), then schema-6 (`CMCheckpointV6`,
upgraded via `upgrade_schema6_to_v8` then `upgrade_schema8_to_v9`), then schema-4
(`CMCheckpointV4`, upgraded via `upgrade_schema4_to_v6` then the same chain), then schema-3
(`CMCheckpointV3`, upgraded via `upgrade_schema3` then the same chain), then legacy schema-1/2
(`CMCheckpoint`, upgraded via `upgrade_schema2` then the same chain). Schema-1 files are still
hard-refused below (semantically untrustworthy Delta) -- this fallback chain only concerns byte
LAYOUT, not schema-1's own known defect. Always returns a `CMCheckpointV9` (uniform shape for
every caller downstream of this function, regardless of which schema the file on disk actually is).
"""
function load_cm_checkpoint(path::AbstractString)
    # 2026-08-05 truncated-power task: schema 10 is NOT auto-upgraded from -- see CMCheckpointV10's
    # own docstring for why (an old file's zfree/dual_warm_start/best_feasible were computed
    # against a DIFFERENT, half-width inner CM block; silently reinterpreting them under the new
    # spec would corrupt the resumed state, not just mislabel it). Try schema 10 first; any older
    # schema that successfully deserializes under the pre-existing chain is HARD-REFUSED below
    # (not upgraded) -- deliberately breaking the schema-2..9 auto-upgrade chain at this one step.
    local ckpt, is_pre10
    try
        ckpt = deserialize(path)::CMCheckpointV10
        is_pre10 = false
    catch e10
        (e10 isa TypeError || e10 isa EOFError || e10 isa MethodError) || rethrow()
        ckpt = try
            deserialize(path)::CMCheckpointV9
        catch e00
            (e00 isa TypeError || e00 isa EOFError || e00 isa MethodError) || rethrow()
            try
                upgrade_schema8_to_v9(deserialize(path)::CMCheckpointV8)
            catch e0
                (e0 isa TypeError || e0 isa EOFError || e0 isa MethodError) || rethrow()
                try
                    upgrade_schema8_to_v9(upgrade_schema6_to_v8(deserialize(path)::CMCheckpointV6))
                catch e1b
                    (e1b isa TypeError || e1b isa EOFError || e1b isa MethodError) || rethrow()
                    try
                        upgrade_schema8_to_v9(upgrade_schema6_to_v8(upgrade_schema4_to_v6(deserialize(path)::CMCheckpointV4)))
                    catch e1
                        (e1 isa TypeError || e1 isa EOFError || e1 isa MethodError) || rethrow()
                        try
                            upgrade_schema8_to_v9(upgrade_schema6_to_v8(upgrade_schema4_to_v6(upgrade_schema3(deserialize(path)::CMCheckpointV3))))
                        catch e2
                            (e2 isa TypeError || e2 isa EOFError || e2 isa MethodError) || rethrow()
                            local old
                            try
                                old = deserialize(path)::CMCheckpoint
                            catch
                                error("load_cm_checkpoint($path): failed to deserialize under CMCheckpointV10, " *
                                      "CMCheckpointV9, CMCheckpointV8, CMCheckpointV6, CMCheckpointV4, CMCheckpointV3, " *
                                      "AND the legacy CMCheckpoint (schema 1/2) layout -- this file is not a recognized " *
                                      "CM checkpoint (corrupt, truncated, or an even older/unrelated format).")
                            end
                            upgrade_schema8_to_v9(upgrade_schema6_to_v8(upgrade_schema4_to_v6(upgrade_schema3(upgrade_schema2(old)))))
                        end
                    end
                end
            end
        end
        is_pre10 = true
    end
    if ckpt.schema == 1
        error("load_cm_checkpoint($path): schema=1, expected $(CM_CHECKPOINT_SCHEMA) -- schema-1 " *
              "checkpoints stored Delta as `-zeta_star` (remediation task Part A, finding F1), NOT the " *
              "canonical Delta_dual = -(mean(Psi(q*))+zeta*); their best_feasible.Delta/feasible flags " *
              "are NOT trustworthy and must not be resumed from directly. Call " *
              "`migrate_cm_checkpoint_v1_candidate($path)` to recover the incumbent w-vector as a fresh " *
              "START POINT only, then cold-re-evaluate it with cm_production_value_verified before " *
              "trusting any Delta/feasibility for it.")
    end
    if is_pre10
        error("load_cm_checkpoint($path): schema=$(ckpt.schema), a pre-2026-08-05 CDF-only CM checkpoint " *
              "(cm_moment_spec did not exist -- every such file implicitly means eq.35-CDF-only). Schema " *
              "$(CM_CHECKPOINT_SCHEMA) checkpoints may impose eq.35+eq.36 (a WIDER, numerically DIFFERENT " *
              "CM dual block) -- resuming a pre-schema-10 file's zfree/dual_warm_start/best_feasible under " *
              "the new spec would silently reinterpret a solve of a different economic problem, not just " *
              "mislabel it. Refusing to auto-upgrade (unlike every prior schema bump). Start a fresh run " *
              "instead; if you specifically want the OLD eq.35-only spec, pass include_truncated_moment=" *
              "false (schema $(CM_CHECKPOINT_SCHEMA) checkpoints written with cm_feature_family_count=1 " *
              "remain readable via this same schema-10-only path, they are not what triggered this error).")
    end
    ckpt.schema == CM_CHECKPOINT_SCHEMA ||
        error("load_cm_checkpoint($path): schema=$(ckpt.schema), expected $(CM_CHECKPOINT_SCHEMA) -- " *
              "unrecognized schema (not caught by the pre-10 hard-refusal above either).")
    return ckpt
end

"""
    migrate_cm_checkpoint_v1_candidate(path) -> (w_candidate, provenance)

Recover ONLY the incumbent w-vector (`g`, `zfree`) from a pre-remediation schema-1 `CMCheckpoint`
as a possible fresh start point -- per task Part A, a schema-1 checkpoint's `best_feasible.Delta`
and `feasible` flag were computed via the F1-buggy `-zeta_star` shorthand and must NOT be trusted.
`provenance.stored_Delta_DO_NOT_TRUST` is returned only for audit/comparison logging; callers must
cold-re-evaluate `w_candidate` with `cm_production_value_verified` (or
`run_cm_upper_checkpointed`'s own cb_F!, which now does this correctly) before treating it as
feasible or as an incumbent. The `CMCheckpoint` struct layout is unchanged between schema 1 and 2
(only the semantics of the stored Delta changed), so plain `deserialize` reconstructs it directly.
"""
function migrate_cm_checkpoint_v1_candidate(path::AbstractString)
    raw = deserialize(path)::CMCheckpoint
    raw.schema == 1 || error("migrate_cm_checkpoint_v1_candidate($path): expected schema=1, got $(raw.schema)")
    w_candidate = vcat(raw.g, raw.zfree)
    stored_Delta = raw.best_feasible === nothing ? NaN : raw.best_feasible.Delta
    provenance = (run_id = raw.run_id, label = raw.label, n_eval = raw.n_eval,
                  checkpoint_reason = raw.checkpoint_reason, stored_Delta_DO_NOT_TRUST = stored_Delta)
    return (w_candidate = w_candidate, provenance = provenance)
end

"""
    run_cm_upper_checkpointed(ctx, w0; L, contrasts=:anchored, probs, delta=1.0,
        maxtime_real=180.0, ckpt_dir, run_id, label, checkpoint_interval_s=90.0,
        resume_from=nothing, cm_hessian_backend=:structured, kwargs...) -> NamedTuple

Checkpointed CM outer loop, same objective/constraint/gradient path as `run_cm_upper`
(cm_outer_driver.jl, UNCHANGED, reused not reimplemented) but with `D20Checkpoint`-grade
resume support. `resume_from` (a `CMCheckpoint` path) regenerates `ctx`/`pcx` from the
checkpoint's own recorded provenance and REFUSES to resume (hard error, not a silent
best-effort) if the regenerated draw checksums don't match -- same guarantee schema-3 gives
the unrestricted path.
"""
function run_cm_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        find_smallest::Bool = true,   # 2026-07-28 lower-direction wiring: true="upper" (minimize gp,
        # the real larger-kappa branch), false="lower" (maximize gp) -- see direction_bounds.jl's own
        # audit for the evidenced find_smallest<->upper/lower mapping. Default true preserves every
        # pre-existing caller's exact behavior byte-for-byte.
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        L::Int = 10, contrasts::Symbol = :anchored, probs::Union{Nothing,AbstractVector{Float64}} = nothing,
        include_truncated_moment::Bool,   # 2026-08-05 truncated-power task: REQUIRED, no default (repo
        # rule: never default a scientific parameter that changes which restriction is imposed).
        # `true` imposes eq.35+eq.36 (the corrected flexible-CM/CM+ZC production spec, requires
        # moment_representation=:dense_reference internally -- see build_cm_production_context's own
        # guard); `false` reproduces the pre-2026-08-05 eq.35-only behavior byte-for-byte. Ignored
        # (any value accepted) when marginal_restriction=:common_frechet -- that family's CM sub-
        # block is a deliberate single-family carve-out, see cm_frechet_level.jl.
        cm_hessian_backend::Symbol = :structured, cm_grid_rule::Symbol = :equal,
        threaded_bins::Bool = true,   # allocation/Hessian port task §6: pass-through to
        # build_cm_production_context/build_cm_bin_ctx -- true (production default) selects the
        # threaded Architecture-C Hessian; false is an explicit benchmark-only opt-out.
        inner_fg_backend::Symbol = CM_INNER_FG_BACKEND_DEFAULT[],   # Phase B1 remediation
        # (2026-07-26): :dense_reference (default -- unchanged) | :cm_lookup (validated O(W*(D-1))
        # lookup FG kernel, ONLY valid for plain flexible CM -- marginal_restriction=
        # :common_flexible AND cm_extension=:cm_only; silently ignored by the meanzc/frechet
        # branches below, which call their own separate build_*_production_context functions that
        # do not accept this kwarg. Hessian backend/math is completely unaffected either way.
        blas_threads::Union{Nothing,Int} = nothing,   # allocation/Hessian port task §6.3: set once
        # right after ctx build (see blas_thread_policy.jl) -- nothing (default) leaves the ambient
        # process BLAS thread count (e.g. OPENBLAS_NUM_THREADS) untouched, zero behavior change.
        pin_outer_algorithm::Bool = false,   # allocation/Hessian port task §1.3/§4: opt-in explicit
        # algorithm=2(Interior/CG)+hessopt=6(L-BFGS) via knitro_outer_algorithm.jl, for matched
        # benchmark A/Bs only. false (default): unchanged existing behavior (.opt file's
        # algorithm=auto, as before this kwarg existed).
        # sigma3 campaign prep (2026-07-30): DIFFERENT, separate opt-in from pin_outer_algorithm
        # above -- forces algorithm=Direct+hessopt=SR1(3)/BFGS(6) via knitro_outer_algorithm.jl's
        # set_outer_algorithm_direct!, not the CG+L-BFGS config `pin_outer_algorithm` pins to.
        # `nothing` (default): zero behavior change. Takes priority over pin_outer_algorithm if
        # both are somehow set (errors instead -- see the call site below).
        outer_direct_hessopt::Union{Nothing,Symbol} = nothing,
        maxtime_real::Float64 = 180.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "cm_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        use_dual_bank::Bool = false, dual_bank_size::Int = 8,   # Phase D remediation (2026-07-26),
        # KEEP_OPT_IN per five-family finish task §2 (2026-07-26): RestrictedDualBank/
        # cm_dual_bank_production.jl -- distance-only warm-start selection (see that file's header
        # for why this differs from the unrestricted family's own KKT-scored DualBank). Validated
        # only on a small D=4 sequence so far; default reverted to false (opt-in) here pending the
        # real-trajectory benchmark in RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md. true: a
        # nearby prior successful dual is offered as the inner solve's warm start instead of always
        # using the single obj.x slot. false (default): zero overhead, byte-identical to every
        # pre-existing production run.
        use_exact_cache::Bool = true,   # Phase C remediation (2026-07-26): exact-point cache
        # (CMProductionEvalKey/cm_exact_cache_production.jl) for this driver's own real
        # (ctx_cm,cctx) shape -- previously unwired despite cm_config.jl's SafeExactCache{CMEvalKey}
        # infrastructure existing in the tree (built against an incompatible pcx.cfg shape this
        # driver never constructs). true (new default): identical outer point + identical
        # scientific context + a valid solved state skips the inner solve entirely. false: zero
        # overhead, byte-identical to every pre-existing production run.
        cm_gradient_backend::Symbol = :cplus,   # CM-C+ production integration 2026-07-23 (docs/CM_PRODUCTION_STATE_2026-07-23.md,
        # docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md): :cplus (PRODUCTION DEFAULT --
        # cm_production_gradient_cplus/composite_gradient_at_Cplus_from_cache, lfix_cm_cplus.jl;
        # 60/60 D=4 + 5/5 real D=20/W=80000/L=50 gates, cosine 1.0 / zero sign mismatches against
        # :reference, ~4.9x-6.8x faster + ~85x fewer allocations per gradient callback) |
        # :reference (documented fallback/validation backend -- cm_production_gradient/
        # composite_gradient_at_fast, the original, byte-identical-to-pre-C+ path; retained for
        # audit/comparison and as an explicit override). Part II.4 follow-up (2026-07-23): now VALIDATED against a resumed
        # checkpoint's own persisted `cm_gradient_backend` (schema>=3; schema-2 checkpoints are
        # treated as :reference, see `upgrade_schema2`). A mismatch is a HARD ERROR unless
        # `allow_backend_switch=true` is also passed -- see that kwarg's own doc for the policy
        # rationale (base-state/incumbent correctness is backend-independent, proven in
        # docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md and empirically confirmed to 1e-13..1e-16
        # across every real D=20 point tested this session, but the persisted `bandwidth_cache`
        # was tuned under the ORIGINAL backend's own selection formula and is cleared on a
        # deliberate switch rather than silently reused across backends).
        allow_backend_switch::Bool = false,   # explicit override required to resume under a
        # DIFFERENT cm_gradient_backend than the checkpoint was written with. Ignored for a fresh
        # (non-resumed) run. Switching is not silently allowed even though it is
        # correctness-preserving (see above) -- the brief's own instruction is to make this an
        # explicit, audited choice, not an invisible default.
        heartbeat_interval_s::Union{Nothing,Float64} = nothing,   # remediation task Part B:
        # opt-in liveness watchdog (nothing = off, zero overhead, the default). When set, a
        # background Timer logs, every heartbeat_interval_s, how long it has been since the last
        # cb_F!/cb_G! callback RETURNED. Purpose: distinguish, with real evidence, "ordinary long
        # callback" (heartbeats show a callback in flight, but n_eval/n_grad keep advancing
        # release-over-release) from "stuck inside a single callback" (one callback's wall time
        # alone exceeds several heartbeat intervals with nothing else changing) from "process
        # killed externally" (log simply stops -- see `signal 15: Terminated`, which is Julia's
        # own SIGTERM handler dumping a backtrace, NOT evidence of an uncaught exception or an
        # internal crash; a genuine uncaught Julia exception self-terminates with an ERROR/
        # nonzero exit code and needs no external signal -- confirmed live in
        # test_threaded_exception_propagation.jl). Does NOT attempt to time individual
        # sub-phases (moment construction / Hessian / BLAS / GC) -- that finer breakdown is Part
        # D's scope (full production timing instrumentation), not duplicated here.
        cm_extension::Symbol = :cm_only,   # CM+moments(+ZC) production integration (2026-07-23):
        # :cm_only (default, UNCHANGED behavior -- dispatches to the pre-existing
        # cm_production_value_verified/cm_production_gradient(_cplus) path exactly as before,
        # byte-identical) | :cm_plus_equal_means | :cm_plus_equal_means_zero_covariance |
        # :cm_plus_moments (the general K_mean/K_pair escape hatch). Resolved via
        # meanzc_resolve_K/CMMeanZCConfig (cm_meanzc_config.jl), the SAME validated logic
        # CMMeanZCConfig uses, not re-derived here.
        meanzc_K_mean::Int = 0, meanzc_K_pair::Int = 0,   # only consulted when cm_extension=:cm_plus_moments
        meanzc_basis::Symbol = :direct,
        meanzc_nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
        # sigma3 campaign prep (2026-07-30): passthrough to d20_real_setup_design's own kwargs of
        # the same name. DEFAULT FLIPPED 2026-08-01 (user-directed) -- see
        # c10_d20_production_driver_unified.jl's own identical comment for the full rationale and
        # the audit that scoped this change to only the 3 real production driver functions.
        exclude_diagonal_gravity::Bool = true,
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}} = default_gravity_exclude_cells_brazil_korea(),
        σHat::Union{Nothing,Float64} = 3.0,
        destination_sample::Symbol = :exclude_row,   # exclude-ROW-destination production release
        # (2026-07-24): :exclude_row (PRODUCTION DEFAULT -- true D_origin/D_dest dimension shrink,
        # ROW dropped as a destination only; validated real D=20/W=80000 both cm_gradient_backend
        # values, see lfix_cplus_exclude_row_validation.jl) | :all_legacy (square D x D, explicit
        # reproduction-only opt-out, byte-identical to every pre-existing CM production run).
        marginal_restriction::Symbol = :common_flexible,   # fixed-Frechet-as-CM-plus-anchor
        # production port (2026-07-25/26): :common_flexible (PRODUCTION DEFAULT -- plain flexible
        # CM, (D-1)*L restrictions, byte-identical to every pre-existing CM production run) |
        # :common_frechet (fixed Frechet as CM plus a common-level anchor, D*L restrictions --
        # cm_frechet_level.jl/cm_frechet_hessian.jl/cm_frechet_cplus.jl; opt-in, currently requires
        # cm_extension=:cm_only, i.e. not yet combined with the meanzc extension -- see the guard
        # just below).
        meanzc_profiled_level::Union{Nothing,Int} = nothing,   # k=(sigma-1) exact-collinearity
        # narrow fix (user-approved, diagnostic/cmzc-k2-singularity-2026-08-05 --
        # docs/audits/cmzc-k2-singularity-2026-08-05/MASTER.md). The autarky/counterfactual
        # price-index moment (base economic block, column D^2+1, autarky_cf.jl) already enforces,
        # for country bi=baseIndex, E[z_bi(w)^(sigma-1)] = cf_denom/cf_num -- an EXACT affine
        # function of the SAME quantity a mean-ZC restriction at level k=(sigma-1) ALSO restricts,
        # to a separate, gp-independent target nu_k. At the calibration point the two coincide;
        # moving gp breaks that and both become jointly infeasible (confirmed live, D4/W5000/
        # sigma3: fails nStatus=-300 at every |gp perturbation| from 1e-6 to 1e-2). Default
        # `nothing`: zero behavior change for every existing caller. When set to k0, exactly ONE
        # substitution is made, at the single upstream point where the outer KNITRO guess `w` is
        # unpacked into `nuvec` (both in cb_F! and cb_G!, plus the backend-switch and final
        # re-verification call sites): `nuvec[k0]` is set to `cf_denom/cf_num` (computed from the
        # SAME theta_econ the autarky moment itself uses) instead of `exp(w[D2_econ+k0])`.
        # Everything downstream (cm_meanzc_production_value_verified_screened,
        # cm_meanzc_production_gradient(_cplus), the inner KNITRO dual solve, the structured/
        # operator Hessian) receives `nuvec` exactly as before and is NOT modified in any way.
        # The ONE necessary consequence, applied ONLY in cb_G! (also purely local, no other file
        # touched): (a) w's own eta_nu_{k0} coordinate now has EXACTLY ZERO effect on the
        # objective (nuvec[k0] no longer reads from it at all), so the gradient KNITRO receives
        # for that coordinate is set to 0 (not the raw d(Delta)/d(eta_nu_k0) the unmodified
        # gradient function still returns, since that quantity is real and well-defined but is not
        # the correct partial derivative w.r.t. w's own now-unused slot); (b) since nuvec[k0] is
        # now an implicit function of gp, the standard d(Delta)/d(gp) returned by the unmodified
        # gradient function (which differentiates holding nuvec fixed, since it's passed as a
        # plain argument) is missing the chain-rule term d(Delta)/d(nu_k0) * d(nu_k0)/d(gp) --
        # added back in explicitly, using d(nu_k0)/d(gp) = sigma*nu_k0/gp (closed form, since
        # cf_denom = gp^sigma * const and cf_num does not depend on gp).
        A_coordinate_mode::Symbol = :powered_aspace,   # transformed-A restricted-family port
        # (2026-07-26 five-family finish task §8): :powered_aspace (NEW PRODUCTION DEFAULT, fixed-
        # theta only -- promoted after test_cm_aspace_coordinate_gates.jl's real D=20/W=80,000
        # equivalence gate: a<->z round-trip to <1e-9, a-space decode reconstructs the IDENTICAL
        # logA_full as the direct z-space path at the calibration point, gradient rescale matches
        # gradient_transform_unified exactly -- 6/6 PASS) | :legacy_z (z_nonpivot=log(Aod_theta), byte-
        # identical to every pre-existing CM-family production run) | :powered_aspace (the
        # theta-decoupled a-space coordinate, cm_aspace_coordinate.jl -- fixed-theta ONLY; a
        # pointwise-affine, constant-slope-(-theta) reparametrization of z_nonpivot, reusing the
        # EXISTING pe::PivotGravityElim/pivot_expand/pivot_reduce and the EXISTING z-space
        # restriction gradient kernels (cm_production_gradient_cplus/cm_meanzc_production_gradient_
        # cplus/cm_frechet_production_gradient_cplus, ALL unchanged) -- only the outer decode/
        # encode/gradient-rescale boundary changes. w0/resume must be constructed in the SAME
        # coordinate this kwarg selects (see cm_w0_from_calibration).
        )   # architecture/production-operator-bundle-hardening-2026-07-30: the moment_representation
        # kwarg that previously lived here is REMOVED, not defaulted -- production runners must not
        # accept a representation choice at all (task §2). All three branches below (flexible_cm,
        # common_frechet, cm_meanzc) now always construct OperatorPsiBundle via
        # prepare_production_run. A dense reference bundle is available only through
        # DenseReferenceDiagnostics.prepare_context, never from this driver.
    # 2026-08-05/06 truncated-power task, RELAXATION HISTORY (see git log for the exact prior wording
    # at each step -- summarized here so the CURRENT state is legible without archaeology):
    #   1. Originally: unconditional hard-refuse for ANY include_truncated_moment=true call.
    #   2. Relaxed to allow cm_extension=:cm_only (plain flexible CM) only, with CM+ZC (any other
    #      cm_extension) still refused -- at that point genuinely correct: CM+ZC's own widened-row
    #      H_CZ cross Hessian and :operator FG (CMMeanZCOperatorState) had NOT yet been given the
    #      analogous two-family extension flexible CM already had.
    #   3. 2026-08-06 (paired-basis-preconditioning pilot continuation): that gap is now closed --
    #      `hessian_cm_structured!`'s own top-of-body comment (cm_hessian_architectures.jl) confirms
    #      H_CZ's widened cross-block is dense-H-free for n_families==2 (bin_zc_cross_hessian_fill!/
    #      _block!, all 3 hcz_prep backends, given "_pow" companion tables), and
    #      `CMMeanZCOperatorState` (cm_meanzc_lookup_kernels.jl) already carries the full `Pow`-gated
    #      two-family forward/backward extension mirroring `CMLookupState`'s own (both dated
    #      2026-08-05, i.e. built in the SAME pass that fixed flexible CM -- this driver-level guard
    #      was simply never updated to match once that work landed). The one remaining gap tonight
    #      (`verify_inner_solution_operator_cmmeanzc!`'s own hardcoded single-family dual-vector
    #      width under `verification_backend=:operator`) is now ALSO fixed (operator_verification.jl).
    #      Re-verified end-to-end at real W=100,000/L=50, cm_extension=:cm_plus_moments -- see
    #      key_results/cmmeanzc_w100k_l50_evidence.txt in this session's Dropbox push. All FOUR
    #      cm_extension values (:cm_only/:cm_plus_equal_means/:cm_plus_equal_means_zero_covariance/
    #      :cm_plus_moments) resolve through the SAME meanzc_resolve_K/CMMeanZCConfig machinery, so
    #      the relaxation is not narrowed to :cm_plus_moments specifically. common_frechet remains
    #      excluded here (unaffected by this change) -- its own two-family gate lives inside
    #      build_cm_frechet_production_context (cm_frechet_level.jl), which still requires
    #      moment_representation=:dense_reference (CMFrechetLookupState/the outer-gradient q0-fold
    #      are not yet extended for two families).
    # 2026-08-06 (outer-production-closeout task, section 4): the blanket refusal that used to live
    # here was removed (see git history / the RELAXATION HISTORY comment above) with only a prose
    # justification, not a runtime check -- "do not merely delete a guard, replace it with a
    # truthful capability check" is the explicit rule for this pass. `assert_two_family_capabilities!`
    # (defined below, cm_checkpoint.jl) is called AFTER `pcx`/`cctx` are actually built (see the
    # call site right after `prepare_production_run` below) and inspects REAL fields on the
    # constructed context -- cctx.n_families, cctx.Pow, cctx.inner_fg_backend, cctx.tls -- rather
    # than trusting that "the code was written to support this" implies "this specific run's
    # context actually has the capability". A caller may proceed only when every check passes.
    lp(xs...) = (println(xs...); flush(stdout))
    # Release fix (2026-07-23, origin-ZC K<=2 release, section 4.1): resolve ckpt_dir to an
    # absolute path BEFORE any real-data/model setup runs -- see the identical fix and full
    # rationale in run_originzc_upper_checkpointed (cm_originzc_checkpoint.jl). Applied here
    # too since this is the shared CM-family entry point and is exposed to the exact same
    # cd()-during-real-data-setup hazard.
    ckpt_dir = abspath(ckpt_dir)
    mkpath(ckpt_dir)

    cm_gradient_backend in (:reference, :cplus) ||
        error("run_cm_upper_checkpointed($label): cm_gradient_backend must be :reference|:cplus, got :$cm_gradient_backend")
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_cm_upper_checkpointed($label): destination_sample must be :exclude_row|:all_legacy, got :$destination_sample")
    marginal_restriction in (:common_flexible, :common_frechet) ||
        error("run_cm_upper_checkpointed($label): marginal_restriction must be :common_flexible|:common_frechet, got :$marginal_restriction")
    A_coordinate_mode in (:legacy_z, :powered_aspace) ||
        error("run_cm_upper_checkpointed($label): A_coordinate_mode must be :legacy_z|:powered_aspace, got :$A_coordinate_mode")
    lp("[", label, "] A_coordinate_mode=", A_coordinate_mode,
       A_coordinate_mode == :powered_aspace ? " (transformed-A, PRODUCTION DEFAULT since five-family finish task §8)" : " (legacy-z, explicit replication mode)")
    lp("[", label, "] cm_gradient_backend=", cm_gradient_backend,
       cm_gradient_backend == :cplus ? " (production default)" : " (fallback/validation backend)",
       " destination_sample=", destination_sample, destination_sample == :exclude_row ? " (production default)" : " (legacy/reproduction-only)")
    write(joinpath(ckpt_dir, "$(label)_gradient_backend.txt"),
          "cm_gradient_backend=$(cm_gradient_backend)\nrun_id=$(run_id)\nrecorded_at=$(Dates.now())\n")

    # CM+moments(+ZC) production integration (2026-07-23): resolve (K_mean,K_pair) via the SAME
    # validated CMMeanZCConfig logic cm_meanzc_config.jl already establishes -- not re-derived
    # here. is_meanzc==false takes the exact pre-existing :cm_only code path at every branch
    # below (byte-identical behavior, confirmed by test_cm_meanzc_d4_gates.jl's own CM-only
    # regression gate and by this function's own :cm_only smoke test).
    # 2026-08-05 truncated-power task BUGFIX: `CMMeanZCConfig`'s own field default
    # (`cm::CMConfig = CMConfig()`, cm_meanzc_config.jl) stopped being constructible the moment
    # `CMConfig.cm_moment_families` became a required (no-default) field -- this call previously
    # relied on that default and ran UNCONDITIONALLY (even for include_truncated_moment=false),
    # so it silently broke EVERY call to this driver, not just two-family ones (confirmed live:
    # UndefKeywordError inside CMConfig() at this exact line, for an include_truncated_moment=true
    # smoke run -- a pre-existing regression from the earlier `cm_moment_families` field addition,
    # never caught because no test in that session exercised this real driver, only the diagnostic
    # archC_base_state path). Pass `cm_moment_families` explicitly, matching this driver's own
    # `include_truncated_moment` -- `meanzc_resolve_K` itself never reads `cfg.cm`'s other fields,
    # so no other CMConfig field needs threading through here.
    meanzc_K_mean, meanzc_K_pair = meanzc_resolve_K(CMMeanZCConfig(cm_extension = cm_extension,
        meanzc_K_mean = meanzc_K_mean, meanzc_K_pair = meanzc_K_pair, meanzc_basis = meanzc_basis,
        cm = CMConfig(cm_moment_families = include_truncated_moment ? 2 : 1)))
    is_meanzc = cm_extension !== :cm_only
    is_meanzc && lp("[", label, "] cm_extension=", cm_extension, " K_mean=", meanzc_K_mean,
                     " K_pair=", meanzc_K_pair, " meanzc_basis=", meanzc_basis)
    (marginal_restriction === :common_frechet && is_meanzc) &&
        error("run_cm_upper_checkpointed($label): marginal_restriction=:common_frechet is not yet " *
              "combined with cm_extension=:$cm_extension (the meanzc extension) -- these are " *
              "orthogonal but the combination has not been implemented/validated. Use cm_extension=" *
              ":cm_only with marginal_restriction=:common_frechet, or marginal_restriction=" *
              ":common_flexible with the meanzc extension.")
    marginal_restriction === :common_frechet &&
        lp("[", label, "] marginal_restriction=common_frechet (fixed Frechet as CM plus a common-level anchor)")
    if meanzc_profiled_level !== nothing
        is_meanzc || error("run_cm_upper_checkpointed($label): meanzc_profiled_level requires cm_extension!=:cm_only (is_meanzc)")
        1 <= meanzc_profiled_level <= meanzc_K_mean ||
            error("run_cm_upper_checkpointed($label): meanzc_profiled_level=$meanzc_profiled_level out of range 1:$meanzc_K_mean")
        lp("[", label, "] k=(sigma-1) narrow fix ACTIVE: meanzc_profiled_level=", meanzc_profiled_level,
           " -- nu_", meanzc_profiled_level, " replaced by cf_denom/cf_num (autarky_cf.jl) at every outer evaluation, not read from the KNITRO guess.")
    end

    resumed = resume_from === nothing ? nothing : load_cm_checkpoint(resume_from)
    backend_switched = false   # Part II.4 follow-up -- set true below only on an explicit, audited cross-backend resume

    if resumed !== nothing
        # 2026-07-28 lower-direction wiring: direction is a more fundamental identity than any of
        # the backend/config checks below -- refuse outright on mismatch, no override (matches this
        # function's own established no-escape-hatch discipline for the other structural fields).
        resumed.find_smallest == find_smallest ||
            error("run_cm_upper_checkpointed($label): direction MISMATCH on resume -- checkpoint " *
                  "was written with find_smallest=$(resumed.find_smallest) (branch=:$(resumed.branch)), " *
                  "this call requests find_smallest=$find_smallest -- refusing to silently resume a " *
                  "different upper/lower direction under the same checkpoint.")
        # CM+moments(+ZC) integration: the moment-column layout (hence the outer vector's own
        # dimension and meaning) is FIXED by (cm_extension,K_mean,K_pair,meanzc_basis) at
        # checkpoint-write time -- unlike cm_gradient_backend (a pure outer-gradient-kernel
        # choice, provably safe to switch, see below), there is no safe override here: refuse
        # outright, no allow_*_switch escape hatch.
        (resumed.cm_extension == cm_extension && resumed.meanzc_K_mean == meanzc_K_mean &&
         resumed.meanzc_K_pair == meanzc_K_pair &&
         (cm_extension === :cm_only || resumed.meanzc_basis == meanzc_basis)) ||
            error("run_cm_upper_checkpointed($label): meanzc config MISMATCH on resume -- checkpoint " *
                  "was written with cm_extension=:$(resumed.cm_extension), K_mean=$(resumed.meanzc_K_mean), " *
                  "K_pair=$(resumed.meanzc_K_pair), meanzc_basis=:$(resumed.meanzc_basis); this call " *
                  "requests cm_extension=:$cm_extension, K_mean=$meanzc_K_mean, K_pair=$meanzc_K_pair, " *
                  "meanzc_basis=:$meanzc_basis -- refusing to resume under a different moment-column " *
                  "layout (the outer vector's own dimension/meaning depends on K_mean; there is no safe " *
                  "override for this, unlike cm_gradient_backend).")
        # exclude-ROW-destination release (2026-07-24): destination_sample changes n_free (D^2 vs
        # D*D_dest) exactly like cm_extension/K_mean does -- no safe override, hard-refuse on
        # mismatch (same discipline as the meanzc check just above).
        resumed.destination_sample == destination_sample ||
            error("run_cm_upper_checkpointed($label): destination_sample MISMATCH on resume -- " *
                  "checkpoint was written with destination_sample=:$(resumed.destination_sample), this " *
                  "call requests :$destination_sample -- refusing to resume under a different " *
                  "destination-sample regime (D^2 vs D*D_dest free-parameter dimension differs).")
        # fixed-Frechet-as-CM-plus-anchor production port: marginal_restriction changes the
        # restriction column count (hence outer moment/dual-vector dimension and meaning) exactly
        # like cm_extension/destination_sample do -- no safe override, hard-refuse on mismatch
        # (same discipline as those two checks).
        resumed.marginal_restriction == marginal_restriction ||
            error("run_cm_upper_checkpointed($label): marginal_restriction MISMATCH on resume -- " *
                  "checkpoint was written with marginal_restriction=:$(resumed.marginal_restriction), " *
                  "this call requests :$marginal_restriction -- refusing to resume under a different " *
                  "restriction family (CM-only vs CM-plus-level-anchor restriction-column count differs).")
        # 2026-08-05 truncated-power task: cm_feature_family_count changes the CM block width (hence
        # the inner dual-vector dimension) exactly like cm_extension/destination_sample/
        # marginal_restriction do -- no safe override, hard-refuse on mismatch. (load_cm_checkpoint
        # already refuses every pre-schema-10 file outright, so `resumed` here always has these
        # fields -- this check catches a schema-10-vs-schema-10 family-count mismatch, e.g. an old
        # single-family schema-10 run resumed under a new two-family request.)
        resumed.cm_feature_family_count == (include_truncated_moment ? 2 : 1) ||
            error("run_cm_upper_checkpointed($label): CM feature-family-count MISMATCH on resume -- " *
                  "checkpoint was written with cm_moment_spec=:$(resumed.cm_moment_spec) " *
                  "(cm_feature_family_count=$(resumed.cm_feature_family_count)), this call requests " *
                  "include_truncated_moment=$include_truncated_moment " *
                  "(cm_feature_family_count=$(include_truncated_moment ? 2 : 1)) -- refusing to resume " *
                  "under a different CM restriction spec (the CM dual block's width/meaning differs).")
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        L = resumed.cm_L; probs = resumed.cm_probs; contrasts = resumed.cm_contrasts
        cm_hessian_backend = resumed.cm_hessian_backend; cm_grid_rule = resumed.cm_grid_rule
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
        # Part II.4 follow-up (2026-07-23): backend provenance check. resumed.cm_gradient_backend
        # is always present now (schema-2 files are upgraded to :reference by load_cm_checkpoint).
        backend_switched = resumed.cm_gradient_backend != cm_gradient_backend
        if backend_switched
            if !allow_backend_switch
                error("run_cm_upper_checkpointed($label): checkpoint was written with " *
                      "cm_gradient_backend=:$(resumed.cm_gradient_backend), but this call requests " *
                      ":$(cm_gradient_backend) -- refusing to silently switch backends on resume. " *
                      "Pass allow_backend_switch=true if this is intentional (base-state/incumbent " *
                      "correctness is backend-independent -- see the kwarg's own docstring -- but the " *
                      "switch is audited, not silent, and clears the persisted bandwidth_cache).")
            end
            lp("[", label, "] *** BACKEND SWITCH ON RESUME *** checkpoint backend=:",
               resumed.cm_gradient_backend, " -> requested backend=:", cm_gradient_backend,
               " (allow_backend_switch=true, explicit override) -- clearing persisted bandwidth_cache",
               " (was tuned under the OLD backend's own selection formula, not safe to reuse as-is).")
            write(joinpath(ckpt_dir, "$(label)_backend_switch_audit.txt"),
                  "BACKEND SWITCH ON RESUME\n" *
                  "resumed_from=$(resume_from)\n" *
                  "checkpoint_backend=$(resumed.cm_gradient_backend)\n" *
                  "requested_backend=$(cm_gradient_backend)\n" *
                  "run_id=$(run_id)\nlabel=$(label)\nswitched_at=$(Dates.now())\n" *
                  "bandwidth_cache_cleared=true\n" *
                  "n_eval_at_switch=$(resumed.n_eval)\nn_grad_at_switch=$(resumed.n_grad)\n")
        end
        # AUD-10 fix (matches c10_d20_production_driver.jl's identical resume warning): the
        # checkpoint's own best_feasible[] was already gated by is_verified_success in cb_F!
        # above (or is nothing) -- it remains the correct scientific incumbent regardless of
        # whether the RAW terminal solver state that triggered :stage_complete_unverified was
        # itself verified.
        if resumed.checkpoint_reason == :stage_complete_unverified
            lp("WARNING: resuming from a checkpoint whose terminal point FAILED verification at write ",
               "time (checkpoint_reason=:stage_complete_unverified, AUD-04/AUD-10) -- best_feasible[] ",
               "inside this checkpoint is still the correct scientific incumbent; only the raw terminal ",
               "solver-state fields are suspect.")
        end
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed, destination_sample = destination_sample, exclude_diagonal_gravity = exclude_diagonal_gravity, gravity_exclude_cells = gravity_exclude_cells, σHat = σHat)
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)   # Phase E remediation (2026-07-26): cf_build (moments! closures below) reuses this instead of allocating fresh every call
    pe = build_pivot_elimination(ctx)
    # Transformed-A restricted-family port: theta_cm/xy_cm are only actually used when
    # A_coordinate_mode==:powered_aspace (cheap to compute unconditionally regardless -- O(D*Ddest),
    # negligible next to ctx build -- so both branches below can share one code path).
    # LAZY on purpose: cm_fixed_theta/precompute_cm_aspace_xy live in the additive
    # cm_aspace_coordinate.jl file, NOT included by the ~55 existing callers of
    # run_cm_upper_checkpointed that never pass A_coordinate_mode (implicit :legacy_z default) --
    # calling them unconditionally would break every one of those callers with a hard include-list
    # requirement they don't need. Only evaluated when A_coordinate_mode=:powered_aspace is
    # actually requested, at which point the caller MUST have included cm_aspace_coordinate.jl
    # (clear isdefined guard below, not a bare UndefVarError).
    if A_coordinate_mode == :powered_aspace
        isdefined(Main, :cm_fixed_theta) ||
            error("run_cm_upper_checkpointed($label): A_coordinate_mode=:powered_aspace requires " *
                  "cm_aspace_coordinate.jl to be included (defines cm_fixed_theta/precompute_cm_aspace_xy/" *
                  "cm_z_from_a/cm_a_from_z) -- add it to this script's include list, after gravity_elimination.jl.")
        theta_cm = cm_fixed_theta(ctx)
        xy_cm = precompute_cm_aspace_xy(ctx)
    else
        theta_cm = NaN
        xy_cm = nothing
    end
    # Transformed-A restricted-family port: coordinate-aware decode (w_econ=[gp;A_nonpivot_native]
    # -> xf) and zfree extraction (w_econ -> ALWAYS canonical z-space, for checkpointing), shared by
    # cb_F!/cb_G!/do_checkpoint/xf_switch/xf_final below. :legacy_z is a pure passthrough to the
    # EXISTING x_free_from_w/pivot_reduce (byte-identical, zero behavior change from before this
    # port); :powered_aspace converts a_nonpivot->z_nonpivot first (cm_aspace_coordinate.jl), then
    # reuses the SAME x_free_from_w/pivot_expand unchanged.
    xf_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe) :
        x_free_from_w(w_econ, pe)
    zfree_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe) : w_econ[2:end]

    if resumed !== nothing
        if ctx.draw_meta.checksum_uniform != resumed.draw_checksum_uniform ||
           ctx.draw_meta.checksum_transformed != resumed.draw_checksum_transformed
            error("run_cm_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated " *
                  "draws (design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's own " *
                  "recorded checksums. Refusing to resume from a different problem instance.")
        end
        # resumed.zfree is ALWAYS canonical z-space (by construction -- see do_checkpoint below),
        # regardless of which A_coordinate_mode the checkpoint's own run searched in -- so resuming
        # under a DIFFERENT A_coordinate_mode than the checkpoint was written under is safe by
        # construction (not an approximation/assumption the way a gradient-backend switch is) and
        # needs no explicit opt-in, just a transparent log line.
        resumed_coord_mode = hasproperty(resumed, :A_coordinate_mode) ? resumed.A_coordinate_mode : :legacy_z
        resumed_coord_mode == A_coordinate_mode ||
            lp("[", label, "] A_coordinate_mode on resume differs from checkpoint (checkpoint=:",
               resumed_coord_mode, ", requested=:", A_coordinate_mode, ") -- safe (checkpoint zfree is ",
               "always canonical z-space), reconstructing w0 in the requested coordinate.")
        A_native0 = A_coordinate_mode == :powered_aspace ? cm_a_from_z(resumed.zfree, theta_cm, xy_cm, pe) : resumed.zfree
        w0 = vcat(resumed.g, A_native0, resumed.eta_nu)
    elseif w0 === nothing
        error("run_cm_upper_checkpointed($label): w0 required for a fresh (non-resumed) run " *
              (is_meanzc ? "-- must be vcat(gp, A_nonpivot_native, eta_nu) with length(eta_nu)==$(meanzc_K_mean), " *
                           "A_nonpivot_native in whichever coordinate A_coordinate_mode selects (see cm_w0_from_calibration)" : ""))
    end

    probs === nothing && error("run_cm_upper_checkpointed($label): probs required (exact cutpoints, not re-derived from L)")
    is_frechet = marginal_restriction === :common_frechet   # guarded mutually exclusive with is_meanzc above
    mode_label = is_meanzc ? "cm_plus_meanzc" : (is_frechet ? "cm_common_frechet" : "cm_flexible")
    # architecture/production-operator-bundle-hardening-2026-07-30 (task §4): the ONLY call in this
    # function that decides bundle representation -- hardcoded :operator in every branch, not a
    # passthrough kwarg. prepare_production_run wraps the result in a type-safe ProductionContext,
    # derives the live backend manifest, and fatally asserts the OperatorPsiBundle invariant before
    # this driver does anything else with pcx.
    family_tag_pre = is_meanzc ? :cm_meanzc : (is_frechet ? :common_frechet : :flexible_cm)
    # 2026-08-06 (paired-basis-preconditioning pilot continuation): NOTE this driver's own
    # prepare_production_run FATALLY requires an OperatorPsiBundle result (production_bundle_api.jl
    # -- "Production runners may only construct OperatorPsiBundle... use DenseReferenceDiagnostics.
    # prepare_context instead" for a genuine dense bundle) -- so moment_representation=:dense_
    # reference is NOT a usable fallback through THIS driver for any family, cm_meanzc included; it
    # would just move the crash from build_cm_meanzc_production_context's own guard to this
    # function's fatal assert. cm_meanzc (is_meanzc) therefore still requires include_truncated_
    # moment=false to reach here UNTIL CMMeanZCOperatorState's two-family FG (cm_meanzc_lookup_
    # kernels.jl -- the code already exists, built the same 2026-08-05 pass as CMLookupState's own)
    # is independently verified against dense reference the way CMLookupState/the Hessian side
    # already were this session -- see the verification task tracked alongside this comment. Until
    # then, moment_representation=:operator stays unconditional here (unchanged from before); the
    # actual gate is build_cm_meanzc_production_context's own internal guard (cm_meanzc_production.jl),
    # which still correctly refuses include_truncated_moment=true + moment_representation=:operator.
    # 2026-08-06 (outer-production-closeout task, section 9.1): the is_frechet branch below never
    # passed `include_truncated_moment`/`threaded_bins` through to `build_cm_frechet_production_context`
    # at all -- since `include_truncated_moment` became a required (no-default) kwarg on that
    # function during the 2026-08-05/06 truncated-power task, this meant EVERY call to this driver
    # with `marginal_restriction=:common_frechet` threw `UndefKeywordError` before ever reaching
    # KNITRO, regardless of any other setting -- the real public common-Frechet path was completely
    # unreachable, not merely TLS-limited. Fixed by passing both through, mirroring the other two
    # branches exactly. `build_cm_frechet_production_context`'s OWN internal capability gate (it
    # requires `moment_representation=:dense_reference` for `include_truncated_moment=true`, which
    # conflicts with `prepare_production_run`'s hard ban on dense bundles below) still correctly
    # refuses two-family common-Frechet through this driver -- this fix makes plain (single-family)
    # common-Frechet reachable again, it does not relax the two-family restriction.
    prepared = prepare_production_run(family_tag_pre, "run_cm_upper_checkpointed",
        () -> is_meanzc ?
            build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = meanzc_K_mean, K_pair = meanzc_K_pair,
                include_truncated_moment = include_truncated_moment,
                contrasts = contrasts, meanzc_basis = meanzc_basis, probs = probs, moment_representation = :operator) :
            is_frechet ?
            build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, probs = probs,
                cm_hessian_backend = cm_hessian_backend, threaded_bins = threaded_bins,
                include_truncated_moment = include_truncated_moment, moment_representation = :operator) :
            build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, probs = probs, threaded_bins = threaded_bins,
                include_truncated_moment = include_truncated_moment,
                inner_fg_backend = inner_fg_backend, moment_representation = :operator))
    pcx = prepared.ctx.inner
    assert_two_family_capabilities!(label, pcx, is_meanzc, is_frechet, include_truncated_moment, threaded_bins)
    pcx = with_screen_counters(pcx)   # 2026-07-24 release (Part B step 7): attach live screen counters for this run
    # D=20 profiling task (flexible_cm/common_frechet, 2026-07-28): stash a live handle to this
    # run's own (cctx, obj) the instant it is built -- ONLY while
    # CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] is true (default false, see
    # cm_hessian_subblock_profiling.jl). Lets a profiling/gate script reach the SAME mutable state
    # every real Hessian/FG callback for THIS run reads/writes, entirely through this public,
    # confirmed-working entry point -- never a second, direct low-level call.
    CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] && (CM_LIVE_PCX_STASH[] = pcx)
    exact_cache = use_exact_cache ? cm_production_exact_cache() : nothing   # Phase C remediation (2026-07-26)
    family_tag = is_meanzc ? :cm_meanzc : (is_frechet ? :common_frechet : :flexible_cm)
    dual_bank = use_dual_bank ? RestrictedDualBank(dual_bank_size) : nothing   # Phase D remediation (2026-07-26)
    # ZC Hessian backend production integration (2026-08-01): cm_meanzc's own validated production
    # BLAS-thread recommendation (8, see ZC_GRAM_BACKEND_DEFAULT's docstring) is applied here ONLY
    # when the caller passed no explicit `blas_threads` AND this run is actually cm_meanzc --
    # `run_cm_upper_checkpointed` is SHARED by flexible_cm/common_frechet/cm_meanzc, and neither
    # sibling family was part of this optimization's validation, so their existing zero-behavior-
    # change default (`nothing` -> ambient thread count untouched) is deliberately left alone.
    effective_blas_threads = blas_threads !== nothing ? blas_threads :
        (family_tag === :cm_meanzc ? ZC_GRAM_BLAS_THREADS_DEFAULT[] : nothing)
    effective_blas_threads !== nothing && BLAS.set_num_threads(effective_blas_threads)   # allocation/Hessian port task §6.3 -- process-scoped (not restored), see blas_thread_policy.jl
    print_active_layout_banner(ctx, mode_label)
    print_screen_startup_banner(mode_label)
    th = pcx.ctx_cm.obj.threshold_state
    println("[threshold-config] mode=", mode_label,
            " requested_delta=", delta, " resolved_active_threshold=", th.threshold,
            " stored_in_objective_bundle=", pcx.ctx_cm.obj.threshold_state.threshold)
    flush(stdout)
    # moment_representation threading task (2026-07-29): the REAL bundle_type, read off pcx AFTER
    # the builder call above -- never an asserted literal (this is exactly the class of bug this
    # task exists to fix: a static print claiming OperatorPsiBundle while ctx.obj was actually
    # still PsiObjectiveBundleImplicit). Printed for both branches below.
    real_bundle_type = Symbol(nameof(typeof(pcx.ctx_cm.obj)))
    if is_frechet
        print_frechet_startup_manifest((marginal_restriction = marginal_restriction, contrasts = contrasts,
            cm_hessian_backend = cm_hessian_backend), ctx.D, L)
        lp("[", label, "] core_hessian_backend=", pcx.cctx === nothing ? "dense_reference" : "exact_winner_pair_parallel (Architecture C)")
        lp("[backend-manifest]   bundle_type=", real_bundle_type)
    else
        print_production_backend_manifest(resolve_flexible_cm_manifest(; cctx = pcx.cctx, blas_threads = effective_blas_threads,
            cm_extension = cm_extension, meanzc_K_mean = meanzc_K_mean, meanzc_K_pair = meanzc_K_pair,
            bundle_type = real_bundle_type))   # allocation/Hessian port task §2; effective_blas_threads
            # (not the raw kwarg) so the manifest records what was ACTUALLY applied, including
            # cm_meanzc's own ZC_GRAM_BLAS_THREADS_DEFAULT[] auto-selection above.
    end
    write_backend_manifest_atomic(prepared.manifest, joinpath(ckpt_dir, "$(label)_backend_manifest.json"))   # architecture/production-operator-bundle-hardening-2026-07-30 (task §7): live manifest, replaces the static prints above as the source of truth

    # D2_econ = length of the (gp, zfree) economic block only -- length(w0) itself is
    # D2_econ + meanzc_K_mean when is_meanzc, matching cm_meanzc_production.jl's own convention
    # (g_ext has length D^2 + K_mean, D^2 == D2_econ).
    D2_econ = length(w0) - (is_meanzc ? meanzc_K_mean : 0)

    nu_bounds = is_meanzc ?
        (meanzc_nu_bounds === nothing ? meanzc_default_nu_bounds(ctx, meanzc_K_mean) : meanzc_nu_bounds) :
        NTuple{2,Float64}[]
    is_meanzc && lp("[", label, "] eta_nu box (per level, log-nu units): ", nu_bounds)

    # pool/workspace for cm_gradient_backend=:cplus only -- zero cost (nothing allocated) when
    # the :reference fallback backend is explicitly selected instead.
    cplus_pool = cm_gradient_backend == :cplus ? build_grad_workspace_pool(size(ctx.obj.U, 1)) : nothing
    cplus_ws = cm_gradient_backend == :cplus ? build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.obj.U, 1)) : nothing

    # Production integration 2026-07-23: an explicit, audited backend switch on resume must cold-
    # verify the resumed incumbent before it is trusted going forward, not merely inherit whatever
    # verification it received under the OLD backend at write time. The inner dual solve (and
    # hence Delta_dual/feasibility/verified-ness) is backend-independent -- only the outer cb_G!
    # gradient differs across cm_gradient_backend -- so this is expected to reproduce the
    # checkpoint's own recorded Delta to numerical precision; a failure here would mean the
    # resumed incumbent is not safe to carry across the switch.
    if backend_switched && resumed.best_feasible !== nothing
        xf_switch = xf_from_w_econ(resumed.best_feasible.w[1:D2_econ])
        verify_switch = if is_meanzc
            νvec_switch = exp.(resumed.best_feasible.w[D2_econ+1:end])
            meanzc_profiled_level === nothing || (νvec_switch[meanzc_profiled_level] = meanzc_profiled_nu_value(xf_switch, ctx))
            (_, _, vs) = cm_meanzc_production_value_verified_screened(xf_switch, νvec_switch, pcx; counters = pcx.screen_counters); vs
        elseif is_frechet
            (_, _, vs) = cm_frechet_production_value_verified_screened(xf_switch, pcx; counters = pcx.screen_counters); vs
        else
            (_, _, vs) = cm_production_value_verified_screened(xf_switch, pcx; counters = pcx.screen_counters); vs
        end
        is_verified_success(verify_switch) ||
            error("run_cm_upper_checkpointed($label): backend switch on resume requested (allow_backend_switch=true), " *
                  "but the resumed incumbent (gp=$(resumed.best_feasible.gp) Delta=$(resumed.best_feasible.Delta)) " *
                  "FAILED independent cold re-verification under the new backend (class=$(classify_inner_result(verify_switch))) " *
                  "-- refusing to carry it forward as best_feasible[] across the switch.")
        lp("[", label, "] backend-switch cold re-verification of resumed incumbent: Delta_dual=",
           verify_switch.Delta_dual, " (checkpoint recorded ", resumed.best_feasible.Delta, ") |diff|=",
           abs(verify_switch.Delta_dual - resumed.best_feasible.Delta), " -- PASSED.")
    end

    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo_econ = vcat(gp_lo, w0[2:D2_econ] .- z_halfwidth)
    w_hi_econ = vcat(gp_hi, w0[2:D2_econ] .+ z_halfwidth)
    w_lo = is_meanzc ? vcat(w_lo_econ, [nu_bounds[k][1] for k in 1:meanzc_K_mean]) : w_lo_econ
    w_hi = is_meanzc ? vcat(w_hi_econ, [nu_bounds[k][2] for k in 1:meanzc_K_mean]) : w_hi_econ

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    if outer_direct_hessopt !== nothing
        pin_outer_algorithm && error("run_cm_upper_checkpointed($label): outer_direct_hessopt and pin_outer_algorithm are mutually exclusive (different, incompatible pinned outer configs) -- set at most one.")
        outer_direct_hessopt in (:sr1, :bfgs) || error("run_cm_upper_checkpointed($label): outer_direct_hessopt must be :sr1 or :bfgs, got :$outer_direct_hessopt")
        set_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS)
    end
    pin_outer_algorithm && set_production_outer_algorithm!(kc)   # opt-in only; default leaves opt_file's algorithm=auto in effect
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], delta)

    last_F_state = Ref{Any}(nothing)
    best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
    n_eval = Ref(resumed !== nothing ? resumed.n_eval : 0)
    n_grad = Ref(resumed !== nothing ? resumed.n_grad : 0)
    trace = NamedTuple[]
    bandwidth_cache = (resumed !== nothing && !backend_switched) ? copy(resumed.bandwidth_cache) : Dict{Int,Float64}()
    t_start = time()
    prior_wall = resumed !== nothing ? resumed.wall_elapsed : 0.0
    last_ckpt_t = Ref(time())
    last_activity_t = Ref(time())   # updated at the END of every cb_F!/cb_G! call
    last_activity_kind = Ref(:none)
    heartbeat_timer = nothing
    if heartbeat_interval_s !== nothing
        heartbeat_timer = Timer(heartbeat_interval_s; interval = heartbeat_interval_s) do _
            since = time() - last_activity_t[]
            lp("[", label, "] HEARTBEAT t=", round(time() - t_start, digits = 1),
               "s  last_callback=", last_activity_kind[], "  ", round(since, digits = 1),
               "s since last callback RETURNED  n_eval=", n_eval[], " n_grad=", n_grad[],
               since > 4 * heartbeat_interval_s ?
                   "  ** no callback has returned for >4 heartbeat intervals -- either an ordinary" *
                   " long single callback (e.g. a slow/near-infeasible inner solve) or a stall;" *
                   " this heartbeat cannot distinguish those two without per-phase timers (Part D)" : "")
        end
    end
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        # zfree_now is ALWAYS canonical z-space regardless of A_coordinate_mode (converted from
        # a-space here if needed) -- matches D20CheckpointUnified's own "zfree always genuine
        # z-space" discipline (outer_coordinate_layout.jl), so a checkpoint is resumable under
        # EITHER A_coordinate_mode without re-deriving anything.
        zfree_now = zfree_from_w_econ(w_current[1:D2_econ])
        eta_nu_now = is_meanzc ? w_current[D2_econ+1:end] : Float64[]
        logA_full = pivot_expand(zfree_now, pe)
        dual_warm_src = (is_meanzc || is_frechet) ? pcx.ctx_cm.obj.x : ctx.obj.x
        # 2026-08-05 truncated-power task: schema-10 CM feature-family metadata. include_truncated_moment
        # is always false by the time this closure can run for the non-frechet branches (the guard near
        # the top of this function already refused true outright) -- frechet's CM sub-block is always
        # single-family regardless of the caller's include_truncated_moment value, so it is recorded as
        # cm_feature_family_count=1 unconditionally for that branch.
        cm_family_count_now = is_frechet ? 1 : (include_truncated_moment ? 2 : 1)
        cm_moment_spec_now = cm_family_count_now == 2 ? :cdf_plus_truncated_power_1msigma : :cdf_only
        cm_feature_schema_version_now = 1
        cm_feature_checksum_now = cm_feature_operator_fingerprint(cm_moment_spec_now, L, cm_feature_schema_version_now, contrasts, pcx.aug.ncm)
        ckpt = CMCheckpointV10(CM_CHECKPOINT_SCHEMA, run_id, label, (find_smallest ? :cm_upper : :cm_lower), find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            L, collect(probs), contrasts, cm_grid_rule, :cumulative, cm_hessian_backend, cm_gradient_backend,
            cm_extension, meanzc_K_mean, meanzc_K_pair, meanzc_basis, MEANZC_MOMENT_LAYOUT_VERSION,
            w_current[1], copy(zfree_now), copy(eta_nu_now), logA_full, copy(dual_warm_src), copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version,
            destination_sample, ctx.row_idx, ctx.D_dest,
            marginal_restriction, A_coordinate_mode,
            cm_moment_spec_now, cm_family_count_now, cm_feature_schema_version_now, cm_feature_checksum_now)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        νvec = is_meanzc ? exp.(w[D2_econ+1:end]) : Float64[]
        is_meanzc && meanzc_profiled_level !== nothing && (νvec[meanzc_profiled_level] = meanzc_profiled_nu_value(xf, ctx))
        local base, verify
        cache_key = exact_cache === nothing ? nothing :
            CMProductionEvalKey(collect(xf), collect(νvec), delta, find_smallest, pcx.ctx_cm.obj.inner_loop_opt,
                family_tag, L, contrasts, meanzc_K_mean, meanzc_K_pair, A_coordinate_mode,
                context_fingerprint(pcx.ctx_cm))
        try
            base, verify = cm_cache_lookup_or_compute!(exact_cache, cache_key, () -> begin
                if is_meanzc
                    _, b, v = cm_meanzc_production_value_verified_screened(xf, νvec, pcx; counters = pcx.screen_counters,
                        dual_bank = dual_bank, eval_id = n_eval[])
                elseif is_frechet
                    _, b, v = cm_frechet_production_value_verified_screened(xf, pcx; counters = pcx.screen_counters,
                        dual_bank = dual_bank, eval_id = n_eval[])
                else
                    _, b, v = cm_production_value_verified_screened(xf, pcx; counters = pcx.screen_counters,
                        dual_bank = dual_bank, eval_id = n_eval[])
                end
                return b, v
            end)
        catch e
            # Closure task Phase 3B: narrowed further to the dedicated CMExpectedSolveFailure
            # type (cm_production_bundle.jl) -- see cm_outer_driver.jl's identical fix for the
            # full rationale (a bare ErrorException is also what an ordinary programming bug
            # raises, so `e isa ErrorException` alone could silently swallow a real bug).
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_cm_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        # Remediation fix (task Part A, finding F1): this used to read `Δ = -base.ζstar`, which
        # silently omits mean(Psi(q*)) and overstates the divergence whenever any recovered
        # weight m* exceeds e (the hybrid divergence's quadratic-branch threshold). `verify`
        # (returned by cm_production_value_verified, computed in archC_verified_state) already
        # carries the canonical Delta_dual = -(mean(Psi(q*))+zeta*) == cbuf[1]/1e10 -- use it
        # directly rather than recomputing or re-deriving. Verified live at real D=20/W=80,000/
        # L=50 points (remediation_a1_verify_delta_dual_identity.jl): the identity holds to
        # machine precision, and the two diverge (by a real, if usually small, amount) at
        # tail-active points.
        Δ = verify.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
        # AUD-04 gate (matches c10_d20_production_driver.jl's cb_F! pattern): feasibility
        # (Delta<=delta) alone is not a verified solve -- KNITRO's own statuses 0/-100/-101/-103
        # are tolerance-based stops, not an optimality certificate. Require
        # classify_inner_result(verify) == VerifiedSolved before this point may become the
        # incumbent -- see docs/fullA_independent_audit_remediation.md AUD-04.
        verified = is_verified_success(verify)
        is_new_best = feasible && verified &&
            is_better_polish(w[1], best_feasible[] === nothing ? nothing : best_feasible[].gp, find_smallest)
        if is_new_best
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, n_eval = n_eval[], t = prior_wall + (time() - t_start))
            do_checkpoint(:new_best, collect(w))
        elseif time() - last_ckpt_t[] >= checkpoint_interval_s
            do_checkpoint(:wall_interval, collect(w))
        end
        push!(trace, (idx = n_eval[], t = time() - t_start, gp = w[1], Delta = Δ, feasible = feasible, verified = verified))
        if verbose && (n_eval[] <= 3 || n_eval[] % 20 == 0)
            lp("  eval ", n_eval[], " t=", round(time() - t_start, digits = 1), "s gp=", w[1], " Delta=", Δ, " feasible=", feasible, " verified=", verified)
        end
        last_activity_t[] = time(); last_activity_kind[] = :cb_F!
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        νvec = is_meanzc ? exp.(w[D2_econ+1:end]) : Float64[]
        is_meanzc && meanzc_profiled_level !== nothing && (νvec[meanzc_profiled_level] = meanzc_profiled_nu_value(xf, ctx))
        shared = last_F_state[]
        matched = shared !== nothing && shared.w == w
        base = matched ? shared.base : nothing
        verify_c = matched ? shared.verify : nothing
        gfull, meta = if is_meanzc
            if cm_gradient_backend == :cplus
                cm_meanzc_production_gradient_cplus(xf, νvec, pcx, ctx, pe, cplus_pool, cplus_ws;
                    base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
            else
                cm_meanzc_production_gradient(xf, νvec, pcx, ctx, pe;
                    base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
            end
        elseif is_frechet
            if cm_gradient_backend == :cplus
                # 2026-08-06: pass verify = verify_c too, matching the is_meanzc branch above --
                # cm_frechet_production_gradient_cplus now falls back to a fresh verified re-solve
                # whenever EITHER base OR verify is missing (see that function's own bug-fix
                # comment); passing the already-matched verify_c avoids a redundant re-solve on the
                # common (matched) case.
                cm_frechet_production_gradient_cplus(xf, pcx, ctx, pe, cplus_pool, cplus_ws; base = base, verify = verify_c, threaded = true,
                    h_mode = :cached, bandwidth_cache = bandwidth_cache)
            else
                cm_frechet_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
                    h_mode = :cached, bandwidth_cache = bandwidth_cache)
            end
        else
            if cm_gradient_backend == :cplus
                cm_production_gradient_cplus(xf, pcx, ctx, pe, cplus_pool, cplus_ws; base = base, threaded = true,
                    h_mode = :cached, bandwidth_cache = bandwidth_cache)
            else
                cm_production_gradient(xf, pcx, ctx, pe; base = base, threaded = true,
                    h_mode = :cached, bandwidth_cache = bandwidth_cache)
            end
        end
        # k=(sigma-1) narrow fix: the ONE necessary consequence of overriding nuvec[k0] above (see
        # meanzc_profiled_level's own docstring for the full derivation) -- neither line touches
        # cm_meanzc_production_gradient(_cplus)/d_delta_dual_d_eta_nu_vec, both of which still
        # correctly compute d(Delta)/d(eta_nu_k) treating nuvec as a plain fixed argument (exactly
        # as before); only the ASSEMBLY of the final KNITRO-facing gradient vector changes, here.
        if is_meanzc && meanzc_profiled_level !== nothing
            k0 = meanzc_profiled_level
            # d_eta_k0 = d(Delta)/d(eta_nu_k0) = nu_k0 * d(Delta)/d(nu_k0) (eta=log(nu), still
            # correct as computed -- d_delta_dual_d_eta_nu_vec = nu .* d_delta_dual_d_nu_vec).
            # The needed correction is d(Delta)/d(nu_k0) * d(nu_k0)/d(gp), NOT d_eta_k0 *
            # d(nu_k0)/d(gp) -- the two nu_k0 factors (one implicit in d_eta_k0, one in
            # d(nu_k0)/d(gp)=sigma*nu_k0/gp) cancel exactly: d(Delta)/d(nu_k0) = d_eta_k0/nu_k0, so
            # the correction is (d_eta_k0/nu_k0)*(sigma*nu_k0/gp) = d_eta_k0*sigma/gp.
            d_eta_k0 = gfull[D2_econ+k0]
            gfull[1] += d_eta_k0 * ctx.σ / w[1]  # chain rule: nu_k0 now an implicit function of gp
            gfull[D2_econ+k0] = 0.0              # w's own eta_nu_k0 coordinate has ZERO effect on
            # the objective now (nuvec[k0] no longer reads from it) -- its correct partial
            # derivative is exactly 0, not the d(Delta)/d(eta_nu_k0) quantity just consumed above.
        end
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        # Transformed-A restricted-family port: gfull is ALWAYS the z-space gradient (the shared
        # numerical kernel -- cm_production_gradient_cplus/cm_meanzc_production_gradient_cplus/
        # cm_frechet_production_gradient_cplus, ALL unchanged) regardless of A_coordinate_mode;
        # rescale the A-block (indices 2:D2_econ; index 1 is d/dgp, D2_econ+1:end is the meanzc
        # eta_nu block, neither touched by the A-coordinate) by the constant scalar -theta_cm to
        # convert into the coordinate KNITRO is actually searching, exactly mirroring
        # outer_coordinate_layout.jl::gradient_transform_unified's own `g[2:end] .*= (-theta)` line
        # (cross-validated bit-for-bit against that function, test_cm_aspace_coordinate_gates.jl).
        if A_coordinate_mode == :powered_aspace
            gfull[2:D2_econ] .*= -theta_cm
        end
        evalResult.jac .= gfull
        last_activity_t[] = time(); last_activity_kind[] = :cb_G!
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    if outer_direct_hessopt !== nothing
        assert_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS; context = "run_cm_upper_checkpointed($label)")
    end
    pin_outer_algorithm && assert_outer_algorithm_explicit!(kc; context = "run_cm_upper_checkpointed($label)")
    try
        KNITRO.KN_solve(kc)
    finally
        heartbeat_timer !== nothing && close(heartbeat_timer)
    end
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    # AUD-04/AUD-10 fix (matches c10_d20_production_driver.jl's run_profile_checkpointed/
    # run_polish_checkpointed final-checkpoint gate): re-verify the terminal point independently
    # before checkpointing it as a normal :stage_complete -- KNITRO's own reported outer status
    # says nothing about whether the INNER dual solve at that point passed the AUD-04
    # residual/gap checks. best_feasible[] (already gated by is_verified_success in cb_F! above)
    # remains the correct resume/incumbent state regardless of this outcome.
    xsol_v = collect(xsol)
    xf_final = xf_from_w_econ(xsol_v[1:D2_econ])
    local verify_final
    try
        if is_meanzc
            νvec_final = exp.(xsol_v[D2_econ+1:end])
            meanzc_profiled_level === nothing || (νvec_final[meanzc_profiled_level] = meanzc_profiled_nu_value(xf_final, ctx))
            _, _, verify_final = cm_meanzc_production_value_verified_screened(xf_final, νvec_final, pcx; counters = pcx.screen_counters)
        elseif is_frechet
            _, _, verify_final = cm_frechet_production_value_verified_screened(xf_final, pcx; counters = pcx.screen_counters)
        else
            _, _, verify_final = cm_production_value_verified_screened(xf_final, pcx; counters = pcx.screen_counters)
        end
    catch e
        # Closure task Phase 3B: narrowed further to the dedicated CMExpectedSolveFailure type
        # (cm_production_bundle.jl) -- see cm_outer_driver.jl's identical fix for the full
        # rationale. A genuine programming bug here must propagate (this is post-solve diagnostic
        # verification, not a KNITRO callback, so there is no DomainError-vs-KN_RC_CALLBACK_ERR
        # contract to preserve by catching broadly).
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)   # inner solve failed outright at the terminal point -- unverified by construction
    end
    if is_verified_success(verify_final)
        final_ckpt = do_checkpoint(:stage_complete, collect(xsol))
    else
        lp("[", label, "] WARNING: terminal point failed verification (class=", classify_inner_result(verify_final),
           ") -- checkpointing as :stage_complete_unverified, NOT :stage_complete. best_feasible[]=",
           best_feasible[] === nothing ? "nothing" : "gp=$(best_feasible[].gp) Delta=$(best_feasible[].Delta)",
           " remains the correct resume/incumbent state (AUD-10).")
        final_ckpt = do_checkpoint(:stage_complete_unverified, collect(xsol))
    end
    print_screen_summary(pcx; label = label)
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"),
            screen_summary = as_namedtuple(pcx.screen_counters))
end

"""
    run_cm_lower_checkpointed(w0=nothing; kwargs...)

2026-07-28 lower-direction wiring: thin wrapper around `run_cm_upper_checkpointed` with
`find_smallest=false` hardcoded (the real lower-kappa direction, per `direction_bounds.jl`'s own
audit). Added for naming parity/discoverability -- the `branch::Symbol` checkpoint field's own
comment (`:cm_upper (only direction wired today; kept for parity/future :cm_lower)`) anticipated
exactly this sibling. Forwards every other argument unchanged; defaults `label` to `"cm_lower"`
instead of silently inheriting `run_cm_upper_checkpointed`'s own `"cm_upper"` default.
"""
function run_cm_lower_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing; label::String = "cm_lower", kwargs...)
    return run_cm_upper_checkpointed(w0; find_smallest = false, label = label, kwargs...)
end
