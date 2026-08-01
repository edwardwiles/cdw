# ============================================================================
# Checkpoint schema + opt-in production launcher for the origin-specific
# pairwise-zero-covariance restriction (task brief Section 12/13).
#
# Extends the CM checkpoint schema WITHOUT mutating CMCheckpoint/V3/V4
# (frozen, per cm_checkpoint.jl's own Serialization-gotcha discipline --
# CMCheckpointV5 is a NEW type name). cm_checkpoint.jl is NOT modified at
# all: `load_cm_checkpoint_v5` below tries V5 first, then falls back to the
# EXISTING `load_cm_checkpoint` (itself a V4/V3/legacy fallback chain),
# upgraded via `upgrade_schema4` -- zero edits to any existing file.
#
# `run_originzc_upper_checkpointed` is a SEPARATE new function (not a change
# to `run_cm_upper_checkpointed`'s body) -- this is the "explicit opt-in, not
# default" the task brief asks for: callers must call a different function
# by name to reach this restriction family at all.
# ============================================================================
using Serialization, Dates
using LinearAlgebra: BLAS
isdefined(Main, :with_blas_threads) || include(joinpath(@__DIR__, "blas_thread_policy.jl"))   # allocation/Hessian port task §6.3/§7
isdefined(Main, :print_production_backend_manifest) || include(joinpath(@__DIR__, "production_backend_manifest.jl"))   # allocation/Hessian port task §2
isdefined(Main, :set_production_outer_algorithm!) || include(joinpath(@__DIR__, "knitro_outer_algorithm.jl"))   # sigma3 campaign prep (2026-07-30): this file previously had no outer-algorithm-pinning dependency at all; needed now for set_outer_algorithm_direct!/assert_outer_algorithm_direct! (outer_direct_hessopt kwarg)
isdefined(Main, :CMProductionEvalKey) || include(joinpath(@__DIR__, "cm_exact_cache_production.jl"))   # Phase C remediation (2026-07-26)
isdefined(Main, :is_better_polish) || include(joinpath(@__DIR__, "incumbent_logic.jl"))   # 2026-07-28 lower-direction wiring: pure, KNITRO-free find_smallest-aware incumbent comparison, reused (not re-derived) from the unrestricted family's own validated helper
isdefined(Main, :prepare_production_run) || include(joinpath(@__DIR__, "production_bundle_api.jl"))   # architecture/production-operator-bundle-hardening-2026-07-30
isdefined(Main, :default_gravity_exclude_cells_brazil_korea) || include(joinpath(@__DIR__, "country_resolve.jl"))

const CM_CHECKPOINT_SCHEMA_V5 = 5
# Bumped 4 -> 5 (origin-specific-ZC integration, 2026-07-23): adds
# distribution_restriction, power_target_layout, origin_D to the persisted
# schema. `cm_extension`/`meanzc_K_mean`/`meanzc_K_pair`/`meanzc_basis`/
# `eta_nu`/`moment_layout_version` (V4 fields) are RETAINED, now describing
# the CM-family restriction only; `distribution_restriction`/`K_mean`/
# `K_pair` (new fields, `originzc_K_mean`/`originzc_K_pair` to avoid a name
# clash with the retained `meanzc_K_mean`/`meanzc_K_pair`) describe the
# ORIGIN-family restriction. A single checkpoint uses exactly one family
# non-trivially -- `save_originzc_checkpoint` asserts this (see below),
# never silently combines them (out of scope for this task).
const ORIGINZC_MOMENT_LAYOUT_VERSION = 1   # wrap_moments_with_originzc's column order, cm_originzc_moments.jl

const CM_CHECKPOINT_SCHEMA_V7 = 7
# Bumped 5 -> 7 (destination_sample production wiring, exclude-ROW-destination release,
# 2026-07-24): adds destination_sample, row_idx, D_dest -- same rationale/fields as
# cm_checkpoint.jl's CMCheckpointV4->V6 bump (see that file's CM_CHECKPOINT_SCHEMA comment for the
# full "why not D^2 vs D*D_dest" explanation). Skips 6 deliberately: cm_checkpoint.jl's OWN
# destination_sample bump already claimed CMCheckpointV6 for the CM-family schema (extending V4);
# this file's origin-family schema extends V5 instead, so it takes the next globally-unused
# number, V7, keeping every CMCheckpoint* type name/schema number in this codebase distinct.

"""
    CMCheckpointV5

Schema-5 checkpoint layout: `CMCheckpointV4`'s complete field set (CM-family
restriction, unchanged), PLUS `distribution_restriction`, `origin_K_mean`,
`origin_K_pair`, `power_target_layout`, `origin_D`, `origin_moment_layout_version`
(origin-family restriction). `eta_nu` (inherited field) holds whichever
family's current eta vector is live: length `meanzc_K_mean` for a CM-family
run, length `n_eta(layout)` (`origin_K_mean*origin_D` for `:origin_by_power`,
`origin_K_mean` for `:shared_by_power`) for an origin-family run.
"""
struct CMCheckpointV5
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
    # ---- NEW (schema 5): origin-specific-ZC (no CM) restriction family ----
    distribution_restriction::Symbol   # :unrestricted | :origin_specific_moments | :origin_specific_moments_zero_covariance
    origin_K_mean::Int                 # 0 for :unrestricted
    origin_K_pair::Int                 # 0 for :unrestricted or :origin_specific_moments
    power_target_layout::Symbol        # :none | :shared_by_power | :origin_by_power
    origin_D::Int                      # 0 unless power_target_layout==:origin_by_power
    origin_moment_layout_version::Int  # bump if wrap_moments_with_originzc's column order ever changes
    # ---- outer-point / incumbent state ----
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
end

"Atomic-ish checkpoint write for CMCheckpointV5 (same discipline as `save_cm_checkpoint`)."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV5)
    (ckpt.cm_extension === :cm_only || ckpt.distribution_restriction === :unrestricted) ||
        error("save_cm_checkpoint: a single checkpoint must use exactly one restriction family non-trivially -- " *
              "got cm_extension=:$(ckpt.cm_extension) AND distribution_restriction=:$(ckpt.distribution_restriction) " *
              "both active. Combining CM with the origin-specific restriction is out of scope for this task.")
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"""
Upgrades a schema-4 `CMCheckpointV4` (origin-specific-ZC did not exist at
that schema) to `CMCheckpointV5`, filling distribution_restriction=:unrestricted,
origin_K_mean=origin_K_pair=0, power_target_layout=(old.cm_extension===:cm_only
? :none : :shared_by_power), origin_D=0, origin_moment_layout_version=0 --
CORRECT (not a guess): the origin-specific-ZC extension point did not exist
anywhere in the codebase when any schema-4 file was written, and every V4
meanzc arm used ONE shared nu_k (SharedByPowerLayout), so :shared_by_power is
the only value consistent with those files' own provenance.
"""
function upgrade_schema4(old::CMCheckpointV4)
    power_layout = old.cm_extension === :cm_only ? :none : :shared_by_power
    return CMCheckpointV5(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        :unrestricted, 0, 0, power_layout, 0, 0,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version)
end

"""
Drops the schema-6-only fields (`destination_sample`/`row_idx`/`D_dest`) to view a
`CMCheckpointV6` as a `CMCheckpointV5` -- lossless for every V5-only reader (a V6 file's other 27
fields describe the SAME CM-family run a V5 file would; `destination_sample` is simply extra
metadata a V5-only caller never asked about). Needed because `cm_checkpoint.jl`'s
`load_cm_checkpoint` now returns `CMCheckpointV6` (destination_sample production wiring,
2026-07-24), not `CMCheckpointV4` -- `load_cm_checkpoint_v5`'s own fallback below must adapt to
that.
"""
function downgrade_v6_to_v5(v6::CMCheckpointV6)
    power_layout = v6.cm_extension === :cm_only ? :none : :shared_by_power
    return CMCheckpointV5(v6.schema, v6.run_id, v6.label, v6.branch, v6.find_smallest, v6.delta,
        v6.W, v6.draw_seed, v6.draw_design, v6.draw_checksum_uniform, v6.draw_checksum_transformed,
        v6.cm_L, v6.cm_probs, v6.cm_contrasts, v6.cm_grid_rule, v6.cm_basis, v6.cm_hessian_backend,
        v6.cm_gradient_backend, v6.cm_extension, v6.meanzc_K_mean, v6.meanzc_K_pair, v6.meanzc_basis,
        v6.moment_layout_version,
        :unrestricted, 0, 0, power_layout, 0, 0,
        v6.g, v6.zfree, v6.eta_nu, v6.logA_full, v6.dual_warm_start, v6.bandwidth_cache, v6.best_feasible,
        v6.n_eval, v6.n_grad, v6.wall_elapsed, v6.wall_budget_remaining, v6.checkpoint_reason,
        v6.knitro_version)
end

"""
    load_cm_checkpoint_v5(path) -> CMCheckpointV5

Tries schema-5 (`CMCheckpointV5`) first; falls back to `load_cm_checkpoint`
(cm_checkpoint.jl, itself a V6/V4/V3/legacy fallback chain as of the 2026-07-24 destination_sample
release) downgraded via `downgrade_v6_to_v5`. Always returns a `CMCheckpointV5`. SUPERSEDED as
`run_originzc_upper_checkpointed`'s own loader by `load_cm_checkpoint_v7` below (which also
carries destination_sample/row_idx/D_dest) -- kept for any caller that specifically wants a V5
view.
"""
function load_cm_checkpoint_v5(path::AbstractString)
    try
        return deserialize(path)::CMCheckpointV5
    catch e
        (e isa TypeError || e isa EOFError || e isa MethodError) || rethrow()
        return downgrade_v6_to_v5(load_cm_checkpoint(path))
    end
end

"""
    CMCheckpointV7

Schema-7 checkpoint layout (destination_sample production wiring, exclude-ROW-destination
release, 2026-07-24): `CMCheckpointV5`'s complete field set (CM-family + origin-family
restrictions, unchanged), PLUS `destination_sample`, `row_idx`, `D_dest` (same fields/rationale as
cm_checkpoint.jl's `CMCheckpointV6` -- see `CM_CHECKPOINT_SCHEMA_V7`'s comment for why this skips
6, already claimed by that unrelated CM-family bump).
"""
struct CMCheckpointV7
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
    distribution_restriction::Symbol
    origin_K_mean::Int
    origin_K_pair::Int
    power_target_layout::Symbol
    origin_D::Int
    origin_moment_layout_version::Int
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
    # ---- NEW (schema 7): omit-ROW-destination true-shrink production option ----
    destination_sample::Symbol
    row_idx::Union{Nothing,Int}
    D_dest::Int
end

"Atomic-ish checkpoint write for CMCheckpointV7 (same discipline/assertion as the V5 method above)."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV7)
    (ckpt.cm_extension === :cm_only || ckpt.distribution_restriction === :unrestricted) ||
        error("save_cm_checkpoint: a single checkpoint must use exactly one restriction family non-trivially -- " *
              "got cm_extension=:$(ckpt.cm_extension) AND distribution_restriction=:$(ckpt.distribution_restriction) " *
              "both active. Combining CM with the origin-specific restriction is out of scope for this task.")
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

const CM_CHECKPOINT_SCHEMA_V10 = 10
# Bumped 7 -> 10 (transformed-A restricted-family port, 2026-07-26 production-audit task addendum;
# whole-tree CMCheckpointV* grep: V8/V9 already claimed by cm_checkpoint.jl's own transformed-A
# bump -- see that file's CM_CHECKPOINT_SCHEMA comment -- so this bump takes the next globally
# unused number, V10, same interleaved-shared-namespace discipline as every prior bump on either
# file): adds `A_coordinate_mode::Symbol` to the persisted schema. `zfree` remains ALWAYS
# canonical z-space regardless of this field's value, same discipline as cm_checkpoint.jl's V9.

"""
    CMCheckpointV10

Identical to `CMCheckpointV7` except one new field, appended at the end: `A_coordinate_mode`.
`CMCheckpointV7` is retained permanently, read-only, for every schema-7 file already written (all
of which are, by construction, `:legacy_z`).
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
    distribution_restriction::Symbol
    origin_K_mean::Int
    origin_K_pair::Int
    power_target_layout::Symbol
    origin_D::Int
    origin_moment_layout_version::Int
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
    # ---- NEW (schema 10): transformed-A restricted-family port ----
    A_coordinate_mode::Symbol   # :legacy_z | :powered_aspace
end

"Atomic-ish checkpoint write for CMCheckpointV10 (same discipline/assertion as the V5/V7 methods above)."
function save_cm_checkpoint(path::AbstractString, ckpt::CMCheckpointV10)
    (ckpt.cm_extension === :cm_only || ckpt.distribution_restriction === :unrestricted) ||
        error("save_cm_checkpoint: a single checkpoint must use exactly one restriction family non-trivially -- " *
              "got cm_extension=:$(ckpt.cm_extension) AND distribution_restriction=:$(ckpt.distribution_restriction) " *
              "both active. Combining CM with the origin-specific restriction is out of scope for this task.")
    tmp = path * ".tmp"
    serialize(tmp, ckpt)
    mv(tmp, path; force = true)
    return path
end

"Upgrades a schema-7 `CMCheckpointV7` (:powered_aspace did not exist as a runtime option at that schema) to `CMCheckpointV10`, filling A_coordinate_mode=:legacy_z -- CORRECT (not a guess), same reasoning as cm_checkpoint.jl's upgrade_schema8_to_v9."
function upgrade_schema7_to_v10(old::CMCheckpointV7)
    return CMCheckpointV10(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        old.distribution_restriction, old.origin_K_mean, old.origin_K_pair, old.power_target_layout,
        old.origin_D, old.origin_moment_layout_version,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version, old.destination_sample, old.row_idx, old.D_dest,
        :legacy_z)
end

"""
    load_cm_checkpoint_v10(path) -> CMCheckpointV10

Tries schema-10 (`CMCheckpointV10`) first; falls back to `load_cm_checkpoint_v7` (its own full
V7/V5/CM-family fallback chain) upgraded via `upgrade_schema7_to_v10`. Always returns a
`CMCheckpointV10`. This is the loader `run_originzc_upper_checkpointed` uses as of the
transformed-A restricted-family port (superseding `load_cm_checkpoint_v7` for that call site,
which remains defined/usable on its own for any other caller that still wants a plain V7 view).
"""
function load_cm_checkpoint_v10(path::AbstractString)
    try
        return deserialize(path)::CMCheckpointV10
    catch e1
        (e1 isa TypeError || e1 isa EOFError || e1 isa MethodError) || rethrow()
        return upgrade_schema7_to_v10(load_cm_checkpoint_v7(path))
    end
end

"""
Upgrades a schema-5 `CMCheckpointV5` (destination_sample did not exist as a runtime option at that
schema) to `CMCheckpointV7`, filling destination_sample=:all_legacy, row_idx=nothing, D_dest=20 --
CORRECT (not a guess), same reasoning as cm_checkpoint.jl's `upgrade_schema4_to_v6`:
`run_originzc_upper_checkpointed` hardcodes `d20_real_setup_design` (D=20-only by construction),
and every schema-5 file was written before `destination_sample` existed anywhere in this codebase.
"""
function upgrade_schema5_to_v7(old::CMCheckpointV5)
    return CMCheckpointV7(old.schema, old.run_id, old.label, old.branch, old.find_smallest, old.delta,
        old.W, old.draw_seed, old.draw_design, old.draw_checksum_uniform, old.draw_checksum_transformed,
        old.cm_L, old.cm_probs, old.cm_contrasts, old.cm_grid_rule, old.cm_basis, old.cm_hessian_backend,
        old.cm_gradient_backend, old.cm_extension, old.meanzc_K_mean, old.meanzc_K_pair, old.meanzc_basis,
        old.moment_layout_version,
        old.distribution_restriction, old.origin_K_mean, old.origin_K_pair, old.power_target_layout,
        old.origin_D, old.origin_moment_layout_version,
        old.g, old.zfree, old.eta_nu, old.logA_full, old.dual_warm_start, old.bandwidth_cache, old.best_feasible,
        old.n_eval, old.n_grad, old.wall_elapsed, old.wall_budget_remaining, old.checkpoint_reason,
        old.knitro_version,
        :all_legacy, nothing, 20)
end

"""
Upgrades a `load_cm_checkpoint`-shaped result (CM-family only -- such a file is never an
origin-family run, `run_cm_upper_checkpointed` never writes distribution_restriction) to
`CMCheckpointV7`. UNLIKE `upgrade_schema5_to_v7`, this carries the REAL
`destination_sample`/`row_idx`/`D_dest` straight across (a CM-family file may genuinely have
`destination_sample=:exclude_row` -- routing through `downgrade_v6_to_v5`'s lossy V5 view, which
drops those fields, would silently and incorrectly reset a real :exclude_row provenance back to
:all_legacy).

RENAMED from `upgrade_schema6_to_v7` (transformed-A restricted-family port, 2026-07-26): the
argument's static type tracks whatever `load_cm_checkpoint` (cm_checkpoint.jl) currently returns
-- `CMCheckpointV9` as of that port (was `CMCheckpointV6` when this function was first written;
`CMCheckpointV9` is a strict field superset of `CMCheckpointV6` under identical field names, so
every field this function reads is unaffected -- it simply never reads the two CM-family-only
fields V9 adds, `marginal_restriction`/`A_coordinate_mode`, exactly as it never read schema-8's
`marginal_restriction` either).
"""
function upgrade_load_cm_checkpoint_result_to_v7(src::CMCheckpointV9)
    power_layout = src.cm_extension === :cm_only ? :none : :shared_by_power
    return CMCheckpointV7(src.schema, src.run_id, src.label, src.branch, src.find_smallest, src.delta,
        src.W, src.draw_seed, src.draw_design, src.draw_checksum_uniform, src.draw_checksum_transformed,
        src.cm_L, src.cm_probs, src.cm_contrasts, src.cm_grid_rule, src.cm_basis, src.cm_hessian_backend,
        src.cm_gradient_backend, src.cm_extension, src.meanzc_K_mean, src.meanzc_K_pair, src.meanzc_basis,
        src.moment_layout_version,
        :unrestricted, 0, 0, power_layout, 0, 0,
        src.g, src.zfree, src.eta_nu, src.logA_full, src.dual_warm_start, src.bandwidth_cache, src.best_feasible,
        src.n_eval, src.n_grad, src.wall_elapsed, src.wall_budget_remaining, src.checkpoint_reason,
        src.knitro_version,
        src.destination_sample, src.row_idx, src.D_dest)
end

"""
    load_cm_checkpoint_v7(path) -> CMCheckpointV7

Tries schema-7 (`CMCheckpointV7`) first; falls back to schema-5 (`CMCheckpointV5`, upgraded via
`upgrade_schema5_to_v7`, correctly implying destination_sample=:all_legacy since V5 predates that
option entirely); falls back to `load_cm_checkpoint` (cm_checkpoint.jl's own V9/V8/V6/V4/V3/legacy
fallback chain, always returns `CMCheckpointV9`) upgraded via
`upgrade_load_cm_checkpoint_result_to_v7` (which carries a REAL destination_sample straight
across, unlike routing through the lossy V5 view). Always returns a `CMCheckpointV7`. This is the
loader `run_originzc_upper_checkpointed` uses (superseding `load_cm_checkpoint_v5` for that call
site).
"""
function load_cm_checkpoint_v7(path::AbstractString)
    try
        return deserialize(path)::CMCheckpointV7
    catch e1
        (e1 isa TypeError || e1 isa EOFError || e1 isa MethodError) || rethrow()
        try
            return upgrade_schema5_to_v7(deserialize(path)::CMCheckpointV5)
        catch e2
            (e2 isa TypeError || e2 isa EOFError || e2 isa MethodError) || rethrow()
            return upgrade_load_cm_checkpoint_result_to_v7(load_cm_checkpoint(path))
        end
    end
end

"""
    run_originzc_upper_checkpointed(w0; W, delta, draw_design, draw_seed, distribution_restriction,
        K_mean, K_pair, power_target_layout=:origin_by_power, meanzc_basis=:direct,
        nu_bounds=nothing, maxtime_real, opt_file, ckpt_dir, run_id, label, checkpoint_interval_s,
        resume_from=nothing, cm_gradient_backend=:cplus, ...) -> NamedTuple

Checkpointed outer loop for the origin-specific-ZC (no CM) restriction family
-- SEPARATE function from `run_cm_upper_checkpointed` (explicit opt-in per
task brief Section 12: callers must call a DIFFERENT function by name to
reach this restriction at all; `run_cm_upper_checkpointed` itself is
untouched). Structural analog of `run_cm_upper_checkpointed`, minus
everything CM-specific (no `L`/`probs`/`contrasts`/`cm_hessian_backend`
kwargs -- this arm has no CM-grid block).

Production-audit remediation correction (2026-07-26, baseline static audit finding, docs/
PRODUCTION_5X7_AUDIT_BASELINE_AND_REMEDIATION_SIZING_2026-07-26.md): the previous version of this
docstring claimed this arm "always uses Architecture A" -- stale relative to the
shared-winner-pair-core-hessian-production-2026-07-25 port. H_EE (the economic-core Hessian block)
dispatches to the SAME shared `exact_winner_pair_parallel` backend every other family uses by
default (`cm_originzc_production.jl`'s `_originzc_hess_cb_builder`/`archA_partitioned_hess_cb_builder`,
`core_exact_hessian.jl`) -- only H_ER/H_RR (the restriction and cross Hessian blocks) remain dense
("Architecture A" in the narrower sense of that dense BLAS gemm), because this family's restriction
dimension is small and fixed-size (no CM threshold grid to structure), not because H_EE itself is
dense.
"""
function run_originzc_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        find_smallest::Bool = true,   # 2026-07-28 lower-direction wiring: true="upper" (minimize gp,
        # the real larger-kappa branch), false="lower" (maximize gp) -- see direction_bounds.jl's own
        # audit for the evidenced find_smallest<->upper/lower mapping. Default true preserves every
        # pre-existing caller's exact behavior byte-for-byte.
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        maxtime_real::Float64 = 180.0, opt_file::String = "csw_outer_wallclock_sr1.opt",
        z_halfwidth::Float64 = 30.0,
        ckpt_dir::AbstractString, run_id::String = string(Dates.now()), label::String = "originzc_upper",
        checkpoint_interval_s::Float64 = 90.0, resume_from::Union{Nothing,AbstractString} = nothing,
        verbose::Bool = true,
        use_dual_bank::Bool = false, dual_bank_size::Int = 8,   # Phase D remediation (2026-07-26),
        # KEEP_OPT_IN per five-family finish task §2 (2026-07-26): same RestrictedDualBank/
        # cm_dual_bank_production.jl as run_cm_upper_checkpointed; default reverted to false here
        # pending RESTRICTED_DUAL_BANK_FINAL_DECISION_2026-07-26.md.
        use_exact_cache::Bool = true,   # Phase C remediation (2026-07-26): same
        # CMProductionEvalKey/cm_exact_cache_production.jl exact-point cache as
        # run_cm_upper_checkpointed. true (new default): identical outer point + identical
        # scientific context + valid solved state skips the inner solve entirely. false: zero
        # overhead, byte-identical to every pre-existing production run.
        cm_gradient_backend::Symbol = :cplus,
        allow_backend_switch::Bool = false,
        distribution_restriction::Symbol,   # REQUIRED, no default -- explicit opt-in (task brief Section 12)
        K_mean::Int, K_pair::Int = 0,
        power_target_layout::Symbol = :origin_by_power, meanzc_basis::Symbol = :direct,
        nu_bounds::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
        # sigma3 campaign prep (2026-07-30): passthrough to d20_real_setup_design's own kwargs of
        # the same name. DEFAULT FLIPPED 2026-08-01 (user-directed) -- see
        # c10_d20_production_driver_unified.jl's own identical comment for the full rationale and
        # the audit that scoped this change to only the 3 real production driver functions.
        exclude_diagonal_gravity::Bool = true,
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}} = default_gravity_exclude_cells_brazil_korea(),
        σHat::Union{Nothing,Float64} = 3.0,
        # sigma3 campaign prep (2026-07-30): this driver had NO outer-algorithm-pinning mechanism
        # at all before this (unlike run_cm_upper_checkpointed/run_polish_checkpointed_unified) --
        # opt_file above leaves algorithm=auto, which knitro_outer_algorithm.jl's own 2026-07-25
        # audit found resolves inconsistently by family. outer_direct_hessopt forces
        # algorithm=Direct+hessopt=SR1(3)/BFGS(6) via set_outer_algorithm_direct!. `nothing`
        # (default): zero behavior change.
        outer_direct_hessopt::Union{Nothing,Symbol} = nothing,
        destination_sample::Symbol = :exclude_row,   # exclude-ROW-destination production release
        # (2026-07-24): same option/semantics/production-default as run_cm_upper_checkpointed's
        # own destination_sample kwarg.
        blas_threads::Union{Nothing,Int} = nothing,   # allocation/Hessian port task §6.3/§7: set once
        # right after ctx build (see blas_thread_policy.jl) -- nothing (default) leaves the ambient
        # process BLAS thread count untouched, zero behavior change. Added for section 7's bounded
        # origin-ZC BLAS benchmark; origin-ZC retains Architecture A (dense) regardless of this.
        A_coordinate_mode::Symbol = :powered_aspace,   # transformed-A restricted-family port
        # (2026-07-26 five-family finish task §8): same option/semantics/NEW production default as
        # run_cm_upper_checkpointed's own A_coordinate_mode kwarg -- the shared decode/encode/
        # gradient-rescale boundary (cm_aspace_coordinate.jl, pe::PivotGravityElim/pivot_expand/
        # pivot_reduce) is family-agnostic and identically exercised by test_cm_aspace_coordinate_
        # gates.jl's real D=20 equivalence gate (6/6 PASS); the origin-family eta/nu restriction
        # block is entirely orthogonal to this choice (same as CM's own eta_nu), untouched either
        # way. :powered_aspace (fixed-theta only) | :legacy_z (byte-identical to every pre-existing
        # origin-ZC production run, explicit replication mode).
        )   # architecture/production-operator-bundle-hardening-2026-07-30: the
        # moment_representation kwarg that previously lived here is REMOVED, not defaulted --
        # production runners must not accept a representation choice at all (task §2). This
        # function now always constructs OperatorPsiBundle via prepare_production_run below. A
        # dense reference bundle is available only through DenseReferenceDiagnostics.prepare_context,
        # never from this driver.
    lp(xs...) = (println(xs...); flush(stdout))
    # Release fix (2026-07-23, section 4.1): resolve ckpt_dir to an absolute path
    # BEFORE any real-data/model setup runs. A relative ckpt_dir silently
    # resolved against whatever the process's cwd happened to be at the time
    # each later joinpath(ckpt_dir, ...) call executed -- found live during the
    # K=2 D=20 shakedown, where a real-data setup file further down the include
    # chain calls cd() as a side effect, causing the shakedown's relative
    # ckpt_dir to land at the worktree root instead of the launch-time directory
    # (see ORIGIN_SPECIFIC_ZC_K12_INTEGRATION_REPORT_2026-07-23.md). abspath()
    # here is computed against the cwd at call time, i.e. before any such cd().
    ckpt_dir = abspath(ckpt_dir)
    mkpath(ckpt_dir)

    distribution_restriction !== :unrestricted ||
        error("run_originzc_upper_checkpointed($label): distribution_restriction=:unrestricted has no restriction to run " *
              "-- this function is the explicit opt-in entry point for the origin-specific-ZC family; use the plain " *
              "unrestricted driver directly instead.")
    cm_gradient_backend in (:reference, :cplus) ||
        error("run_originzc_upper_checkpointed($label): cm_gradient_backend must be :reference|:cplus, got :$cm_gradient_backend")
    destination_sample in (:exclude_row, :all_legacy) ||
        error("run_originzc_upper_checkpointed($label): destination_sample must be :exclude_row|:all_legacy, got :$destination_sample")
    A_coordinate_mode in (:legacy_z, :powered_aspace) ||
        error("run_originzc_upper_checkpointed($label): A_coordinate_mode must be :legacy_z|:powered_aspace, got :$A_coordinate_mode")
    lp("[", label, "] A_coordinate_mode=", A_coordinate_mode,
       A_coordinate_mode == :powered_aspace ? " (transformed-A, PRODUCTION DEFAULT since five-family finish task §8)" : " (legacy-z, explicit replication mode)")
    lp("[", label, "] distribution_restriction=", distribution_restriction, " power_target_layout=", power_target_layout,
       " K_mean=", K_mean, " K_pair=", K_pair, " cm_gradient_backend=", cm_gradient_backend,
       " destination_sample=", destination_sample, destination_sample == :exclude_row ? " (production default)" : " (legacy/reproduction-only)")

    cfg = OriginZCConfig(distribution_restriction = distribution_restriction, K_mean = K_mean, K_pair = K_pair,
                          power_target_layout = power_target_layout, meanzc_basis = meanzc_basis, nu_bounds = nu_bounds)
    K_mean_r, K_pair_r = originzc_resolve_K(cfg)   # raises on any config inconsistency

    resumed = resume_from === nothing ? nothing : load_cm_checkpoint_v10(resume_from)
    backend_switched = false

    if resumed !== nothing
        # 2026-07-28 lower-direction wiring: direction is a more fundamental identity than any of
        # the backend/config checks below -- refuse outright on mismatch, no override (matches this
        # function's own established no-escape-hatch discipline for the other structural fields).
        resumed.find_smallest == find_smallest ||
            error("run_originzc_upper_checkpointed($label): direction MISMATCH on resume -- checkpoint " *
                  "was written with find_smallest=$(resumed.find_smallest) (branch=:$(resumed.branch)), " *
                  "this call requests find_smallest=$find_smallest -- refusing to silently resume a " *
                  "different upper/lower direction under the same checkpoint.")
        (resumed.distribution_restriction == distribution_restriction && resumed.origin_K_mean == K_mean_r &&
         resumed.origin_K_pair == K_pair_r && resumed.power_target_layout == power_target_layout) ||
            error("run_originzc_upper_checkpointed($label): distribution_restriction/K/layout MISMATCH on resume -- " *
                  "checkpoint has distribution_restriction=:$(resumed.distribution_restriction), " *
                  "K_mean=$(resumed.origin_K_mean), K_pair=$(resumed.origin_K_pair), " *
                  "power_target_layout=:$(resumed.power_target_layout); this call requests " *
                  "distribution_restriction=:$distribution_restriction, K_mean=$K_mean_r, K_pair=$K_pair_r, " *
                  "power_target_layout=:$power_target_layout -- refusing to resume under a different moment layout.")
        resumed.origin_moment_layout_version == ORIGINZC_MOMENT_LAYOUT_VERSION ||
            error("run_originzc_upper_checkpointed($label): origin_moment_layout_version MISMATCH -- checkpoint=" *
                  "$(resumed.origin_moment_layout_version), current=$(ORIGINZC_MOMENT_LAYOUT_VERSION). Refusing to resume.")
        resumed.destination_sample == destination_sample ||
            error("run_originzc_upper_checkpointed($label): destination_sample MISMATCH on resume -- " *
                  "checkpoint was written with destination_sample=:$(resumed.destination_sample), this " *
                  "call requests :$destination_sample -- refusing to resume under a different " *
                  "destination-sample regime.")
        W = resumed.W; delta = resumed.delta; draw_design = resumed.draw_design; draw_seed = resumed.draw_seed
        lp("[", label, "] RESUMING from ", resume_from, " (n_eval=", resumed.n_eval, " n_grad=", resumed.n_grad,
           " wall_elapsed=", round(resumed.wall_elapsed, digits = 1), "s)")
        backend_switched = resumed.cm_gradient_backend != cm_gradient_backend
        if backend_switched && !allow_backend_switch
            error("run_originzc_upper_checkpointed($label): checkpoint was written with cm_gradient_backend=:" *
                  "$(resumed.cm_gradient_backend), but this call requests :$(cm_gradient_backend) -- refusing to " *
                  "silently switch backends on resume. Pass allow_backend_switch=true if intentional.")
        end
    end

    ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = find_smallest, draw_design = draw_design, draw_seed = draw_seed, destination_sample = destination_sample, exclude_diagonal_gravity = exclude_diagonal_gravity, gravity_exclude_cells = gravity_exclude_cells, σHat = σHat)
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)   # Phase E remediation (2026-07-26): cf_build (moments! closures below) reuses this instead of allocating fresh every call
    pe = build_pivot_elimination(ctx)
    D = ctx.D
    # Transformed-A restricted-family port: same lazy-computation/isdefined-guard discipline as
    # run_cm_upper_checkpointed (cm_checkpoint.jl) -- see that function's identical block for the
    # full rationale (existing callers that never pass A_coordinate_mode must not be forced to
    # include cm_aspace_coordinate.jl).
    if A_coordinate_mode == :powered_aspace
        isdefined(Main, :cm_fixed_theta) ||
            error("run_originzc_upper_checkpointed($label): A_coordinate_mode=:powered_aspace requires " *
                  "cm_aspace_coordinate.jl to be included (defines cm_fixed_theta/precompute_cm_aspace_xy/" *
                  "cm_z_from_a/cm_a_from_z) -- add it to this script's include list, after gravity_elimination.jl.")
        theta_cm = cm_fixed_theta(ctx)
        xy_cm = precompute_cm_aspace_xy(ctx)
    else
        theta_cm = NaN
        xy_cm = nothing
    end
    xf_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        x_free_from_w(vcat(w_econ[1], cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe)), pe) :
        x_free_from_w(w_econ, pe)
    zfree_from_w_econ(w_econ) = A_coordinate_mode == :powered_aspace ?
        cm_z_from_a(w_econ[2:end], theta_cm, xy_cm, pe) : w_econ[2:end]
    layout = originzc_make_layout(cfg, D)
    originzc_validate_bounds(cfg, layout)

    if resumed !== nothing
        (power_target_layout != :origin_by_power || resumed.origin_D == D) ||
            error("run_originzc_upper_checkpointed($label): D MISMATCH on resume -- checkpoint origin_D=" *
                  "$(resumed.origin_D), current D=$D -- refusing to resume under a different origin dimension.")
        (ctx.draw_meta.checksum_uniform == resumed.draw_checksum_uniform &&
         ctx.draw_meta.checksum_transformed == resumed.draw_checksum_transformed) ||
            error("run_originzc_upper_checkpointed($label): draw checksum MISMATCH on resume -- regenerated draws " *
                  "(design=:$(draw_design), seed=$(draw_seed)) do not match the checkpoint's own recorded checksums.")
        # resumed.zfree is ALWAYS canonical z-space regardless of which A_coordinate_mode the
        # checkpoint's own run searched in -- safe to resume under a DIFFERENT A_coordinate_mode
        # (same reasoning as run_cm_upper_checkpointed's identical resume block).
        resumed_coord_mode = hasproperty(resumed, :A_coordinate_mode) ? resumed.A_coordinate_mode : :legacy_z
        resumed_coord_mode == A_coordinate_mode ||
            lp("[", label, "] A_coordinate_mode on resume differs from checkpoint (checkpoint=:",
               resumed_coord_mode, ", requested=:", A_coordinate_mode, ") -- safe (checkpoint zfree is ",
               "always canonical z-space), reconstructing w0 in the requested coordinate.")
        A_native0 = A_coordinate_mode == :powered_aspace ? cm_a_from_z(resumed.zfree, theta_cm, xy_cm, pe) : resumed.zfree
        w0 = vcat(resumed.g, A_native0, resumed.eta_nu)
    elseif w0 === nothing
        error("run_originzc_upper_checkpointed($label): w0 required for a fresh (non-resumed) run -- must be " *
              "vcat(gp, zfree, eta) with length(eta)==$(n_eta(layout))")
    end

    # architecture/production-operator-bundle-hardening-2026-07-30 (task §4): the ONLY call in this
    # function that decides bundle representation -- hardcoded :operator, not a passthrough kwarg.
    # prepare_production_run wraps the result in a type-safe ProductionContext, derives the live
    # backend manifest, and fatally asserts the OperatorPsiBundle invariant before this driver does
    # anything else with pcx.
    prepared = prepare_production_run(:origin_zc, "run_originzc_upper_checkpointed",
        () -> build_originzc_production_context(ctx, CS, layout; moment_representation = :operator))
    pcx = prepared.ctx.inner
    pcx = with_screen_counters(pcx)   # 2026-07-24 release (Part B step 7): attach live screen counters for this run
    exact_cache = use_exact_cache ? cm_production_exact_cache() : nothing   # Phase C remediation (2026-07-26)
    dual_bank = use_dual_bank ? RestrictedDualBank(dual_bank_size) : nothing   # Phase D remediation (2026-07-26)
    blas_threads !== nothing && BLAS.set_num_threads(blas_threads)   # allocation/Hessian port task §6.3/§7 -- process-scoped (not restored), see blas_thread_policy.jl
    print_active_layout_banner(ctx, "origin_zc")
    print_screen_startup_banner("origin_zc")
    th = pcx.ctx_cm.obj.threshold_state
    println("[threshold-config] mode=origin_zc requested_delta=", delta,
            " resolved_active_threshold=", th.threshold, " stored_in_objective_bundle=", th.threshold)
    flush(stdout)
    print_production_backend_manifest(resolve_origin_zc_manifest(; octx = pcx.octx, blas_threads = blas_threads,
        bundle_type = Symbol(nameof(typeof(pcx.ctx_cm.obj)))))   # 2026-07-25 continuation: pass the REAL octx this driver just built via build_originzc_production_context -- was previously called with no octx at all, so it always fell back to reporting the pre-port dense_architecture_a path regardless of what actually ran. moment_representation threading task (2026-07-29): bundle_type is now also the REAL type, read off pcx AFTER build_originzc_production_context above.
    write_backend_manifest_atomic(prepared.manifest, joinpath(ckpt_dir, "$(label)_backend_manifest.json"))   # architecture/production-operator-bundle-hardening-2026-07-30 (task §7): live manifest, replaces the static print above as the source of truth
    D2_econ = length(w0) - n_eta(layout)

    bounds = cfg.nu_bounds === nothing ? originzc_default_nu_bounds(ctx, layout) : cfg.nu_bounds
    lp("[", label, "] eta box (per coordinate, log-nu units): n=", length(bounds))

    cplus_pool = cm_gradient_backend == :cplus ? build_grad_workspace_pool(size(ctx.obj.U, 1)) : nothing
    cplus_ws = cm_gradient_backend == :cplus ? build_lfix_factorized_workspace(ctx.D, ctx.D_dest, size(ctx.obj.U, 1)) : nothing

    if backend_switched && resumed.best_feasible !== nothing
        xf_switch = xf_from_w_econ(resumed.best_feasible.w[1:D2_econ])
        νvec_switch = exp.(resumed.best_feasible.w[D2_econ+1:end])
        (_, _, vs) = cm_originzc_production_value_verified_screened(xf_switch, νvec_switch, pcx; counters = pcx.screen_counters)
        is_verified_success(vs) ||
            error("run_originzc_upper_checkpointed($label): backend switch on resume requested, but the resumed " *
                  "incumbent FAILED independent cold re-verification under the new backend -- refusing to carry it forward.")
        lp("[", label, "] backend-switch cold re-verification of resumed incumbent: Delta_dual=", vs.Delta_dual,
           " (checkpoint recorded ", resumed.best_feasible.Delta, ") -- PASSED.")
    end

    D2 = length(w0)
    gp_lo, gp_hi = ctx.bounds.γp_lo, ctx.bounds.γp_hi
    w_lo_econ = vcat(gp_lo, w0[2:D2_econ] .- z_halfwidth)
    w_hi_econ = vcat(gp_hi, w0[2:D2_econ] .+ z_halfwidth)
    w_lo = vcat(w_lo_econ, [bounds[k][1] for k in 1:n_eta(layout)])
    w_hi = vcat(w_hi_econ, [bounds[k][2] for k in 1:n_eta(layout)])

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    if outer_direct_hessopt !== nothing
        outer_direct_hessopt in (:sr1, :bfgs) || error("run_originzc_upper_checkpointed($label): outer_direct_hessopt must be :sr1 or :bfgs, got :$outer_direct_hessopt")
        set_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS)
    end
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
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
    knitro_version = try
        KNITRO.KN_get_release()
    catch
        "unknown"
    end

    function do_checkpoint(reason::Symbol, w_current::Vector{Float64})
        zfree_now = zfree_from_w_econ(w_current[1:D2_econ])   # ALWAYS canonical z-space
        eta_now = w_current[D2_econ+1:end]
        logA_full = pivot_expand(zfree_now, pe)
        dual_warm_src = pcx.ctx_cm.obj.x
        ckpt = CMCheckpointV10(CM_CHECKPOINT_SCHEMA_V10, run_id, label, (find_smallest ? :cm_upper : :cm_lower), find_smallest, delta, W, draw_seed,
            draw_design, ctx.draw_meta.checksum_uniform, ctx.draw_meta.checksum_transformed,
            0, Float64[], :anchored, :equal, :cumulative, :dense_reference, cm_gradient_backend,
            :cm_only, 0, 0, :direct, 0,
            distribution_restriction, K_mean_r, K_pair_r, power_target_layout, layout_D(layout), ORIGINZC_MOMENT_LAYOUT_VERSION,
            w_current[1], copy(zfree_now), copy(eta_now), logA_full, copy(dual_warm_src), copy(bandwidth_cache),
            best_feasible[], n_eval[], n_grad[], prior_wall + (time() - t_start),
            maxtime_real - (time() - t_start), reason, knitro_version,
            destination_sample, ctx.row_idx, ctx.D_dest, A_coordinate_mode)
        path = joinpath(ckpt_dir, "$(label)_latest.jls")
        save_cm_checkpoint(path, ckpt)
        last_ckpt_t[] = time()
        return ckpt
    end

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        νvec = exp.(w[D2_econ+1:end])
        local base, verify
        # L/contrasts have no meaning for origin-ZC (no CM grid) -- 0/:none sentinels; family_tag
        # (:origin_zc, fixed per driver) plus a FRESH per-call cache (never shared across driver
        # invocations/configs) are what actually prevent cross-config collision here.
        cache_key = exact_cache === nothing ? nothing :
            CMProductionEvalKey(collect(xf), collect(νvec), delta, find_smallest, pcx.ctx_cm.obj.inner_loop_opt,
                :origin_zc, 0, :none, K_mean, K_pair, A_coordinate_mode, context_fingerprint(pcx.ctx_cm))
        try
            base, verify = cm_cache_lookup_or_compute!(exact_cache, cache_key, () -> begin
                _, b, v = cm_originzc_production_value_verified_screened(xf, νvec, pcx; counters = pcx.screen_counters,
                    dual_bank = dual_bank, eval_id = n_eval[])
                return b, v
            end)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            reject_point(w[1], "run_originzc_upper_checkpointed($label): infeasible/failed inner solve at this point")
        end
        Δ = verify.Delta_dual
        evalResult.obj[1] = find_smallest ? w[1] : -w[1]
        evalResult.c[1] = Δ
        n_eval[] += 1
        last_F_state[] = (w = copy(w), base = base, verify = verify)
        feasible = isfinite(Δ) && Δ <= delta + 1e-6
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
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        xf = xf_from_w_econ(w[1:D2_econ])
        νvec = exp.(w[D2_econ+1:end])
        shared = last_F_state[]
        matched = shared !== nothing && shared.w == w
        base = matched ? shared.base : nothing
        verify_c = matched ? shared.verify : nothing
        gfull, meta = if cm_gradient_backend == :cplus
            cm_originzc_production_gradient_cplus(xf, νvec, pcx, ctx, pe, cplus_pool, cplus_ws;
                base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        else
            cm_originzc_production_gradient(xf, νvec, pcx, ctx, pe;
                base = base, verify = verify_c, threaded = true, h_mode = :cached, bandwidth_cache = bandwidth_cache)
        end
        n_grad[] += 1
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = find_smallest ? 1.0 : -1.0
        # Transformed-A restricted-family port: gfull is ALWAYS the z-space gradient (unchanged
        # cm_originzc_production_gradient_cplus/cm_originzc_production_gradient); rescale the
        # A-block (indices 2:D2_econ) by the constant scalar -theta_cm when the outer search is
        # actually in a-space, same as run_cm_upper_checkpointed's identical block.
        if A_coordinate_mode == :powered_aspace
            gfull[2:D2_econ] .*= -theta_cm
        end
        evalResult.jac .= gfull
        return 0
    end

    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    if outer_direct_hessopt !== nothing
        assert_outer_algorithm_direct!(kc, outer_direct_hessopt === :sr1 ? KNITRO_HESSOPT_SR1 : KNITRO_HESSOPT_BFGS; context = "run_originzc_upper_checkpointed($label)")
    end
    KNITRO.KN_solve(kc)
    wall_ext = time() - t_start
    nStatus, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    σ = ctx.σ
    b = best_feasible[]
    κ = b === nothing ? NaN : 1 - b.gp^(σ / (σ - 1))

    xsol_v = collect(xsol)
    xf_final = xf_from_w_econ(xsol_v[1:D2_econ])
    local verify_final
    try
        νvec_final = exp.(xsol_v[D2_econ+1:end])
        _, _, verify_final = cm_originzc_production_value_verified_screened(xf_final, νvec_final, pcx; counters = pcx.screen_counters)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        verify_final = (inner_status = -300,)
    end
    final_ckpt = if is_verified_success(verify_final)
        do_checkpoint(:stage_complete, collect(xsol))
    else
        lp("[", label, "] WARNING: terminal point failed verification -- checkpointing as :stage_complete_unverified.")
        do_checkpoint(:stage_complete_unverified, collect(xsol))
    end
    print_screen_summary(pcx; label = label)
    return (knitro_status = nStatus, wall = wall_ext, n_eval = n_eval[], n_grad = n_grad[],
            best = b, kappa = κ, xsol = collect(xsol), trace = trace, final_checkpoint = final_ckpt,
            ckpt_path = joinpath(ckpt_dir, "$(label)_latest.jls"),
            screen_summary = as_namedtuple(pcx.screen_counters))
end

"""
    run_originzc_lower_checkpointed(w0=nothing; kwargs...)

2026-07-28 lower-direction wiring: thin wrapper around `run_originzc_upper_checkpointed` with
`find_smallest=false` hardcoded (the real lower-kappa direction, per `direction_bounds.jl`'s own
audit). Forwards every other argument unchanged; defaults `label` to `"originzc_lower"` instead
of silently inheriting `run_originzc_upper_checkpointed`'s own `"originzc_upper"` default.
"""
function run_originzc_lower_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing; label::String = "originzc_lower", kwargs...)
    return run_originzc_upper_checkpointed(w0; find_smallest = false, label = label, kwargs...)
end
