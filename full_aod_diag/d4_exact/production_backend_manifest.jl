# ============================================================================
# Central, machine-readable production backend manifest (allocation/Hessian port task §2).
#
# One place that RESOLVES (not just prints) which backend each public production driver is
# actually using -- core representation, Hessian architecture, thread counts, checkpoint schema,
# screen stack -- as a structured NamedTuple, plus a human-readable printer and a JSON writer.
# Every field is populated from values the calling driver already has in scope at the real call
# site (the actual kwarg it resolved, the actual struct field it built), NOT a separate hardcoded
# "expected" description that could drift from what the driver really does -- see
# PUBLIC_ENTRY_POINT_BACKEND_ASSERTIONS_2026-07-25.md for the test that checks these against each
# real public driver's live output.
#
# This supersedes (does not duplicate) the two existing per-topic startup banners
# (`print_active_layout_banner`, cc_algo/active_layout.jl; `print_screen_startup_banner`,
# cm_screen_bridge.jl) -- those remain unchanged and continue to print alongside this one; this
# file adds the fields neither of them covers (Hessian backend, thread counts, checkpoint schema).
#
# Do not scatter backend defaults across hidden kwargs / environment variables / benchmark-only
# wrappers / separate option files per the task's explicit prohibition (§2) -- this file, plus the
# `blas_threads`/`cm_hessian_backend`/`threaded_bins` kwargs already threaded through the three
# checkpointed drivers, is the single reconciliation point.
# ============================================================================

using Dates
using LinearAlgebra: BLAS

const CHECKPOINT_SCHEMA_UNRESTRICTED = 4        # c10_d20_production_driver.jl :: CHECKPOINT_SCHEMA (D20CheckpointV4)
# CM_CHECKPOINT_SCHEMA (=6, cm_checkpoint.jl) and CM_CHECKPOINT_SCHEMA_V7 (=7,
# cm_originzc_checkpoint.jl) are used directly below -- both already exist as named constants in
# their own files (loaded before this one by every driver), not redefined here to avoid the
# Serialization type-name-collision gotcha documented in cm_checkpoint.jl's own header.

"Ordered screen stack currently active in production for the given family -- matches the literal list each driver's own startup banner already prints (c10_d20_production_driver.jl / cm_checkpoint.jl / cm_originzc_checkpoint.jl); kept here as one place so the JSON manifest and the printed banner cannot silently diverge."
function production_screen_stack(family::Symbol)
    if family === :unrestricted
        return ["pairwise_certificate", "screen_hard_winners", "envelope(EXACT_INFEASIBLE_PREWINNER_ENVELOPE)", "winning_range", "safety_net_moment_range"]
    else   # :flexible_cm, :cm_meanzc, :origin_zc all share the CM-family screen stack
        return ["pairwise_certificate", "screen_hard_winners"]
    end
end

"""
    resolve_unrestricted_manifest(; hessian_backend, blas_threads, julia_threads=Threads.nthreads(),
        trade_elasticity_mode=:fixed, A_coordinate_mode=:legacy_z, gp_coordinate_mode=:raw,
        theta_bounds=nothing, outer_dimension=nothing)

Resolves the unrestricted family's backend manifest from values the calling driver
(`run_profile_checkpointed`/`run_polish_checkpointed`/`run_polish_checkpointed_unified`) already
has in scope. The `trade_elasticity_mode`/`A_coordinate_mode`/`gp_coordinate_mode`/
`theta_*`/`outer_dimension` fields (transformed-A/flexible-theta production port task §20) all
default to the pre-port fixed/legacy-z/raw values, so `run_profile_checkpointed`/
`run_polish_checkpointed`'s existing unqualified calls are byte-identical in every field they
already emitted -- only `run_polish_checkpointed_unified` passes the non-default values.
"""
function resolve_unrestricted_manifest(; hessian_backend::Symbol = UNRESTRICTED_CORE_HESSIAN_BACKEND[], blas_threads::Union{Nothing,Int},
        trade_elasticity_mode::Symbol = :fixed, A_coordinate_mode::Symbol = :legacy_z,
        gp_coordinate_mode::Symbol = :raw,
        theta_bounds::Union{Nothing,Tuple{Float64,Float64}} = nothing,
        outer_dimension::Union{Nothing,Int} = nothing,
        bundle_type::Symbol = :PsiObjectiveBundleImplicit)   # unrestricted operator-bundle wiring
        # task (2026-07-29): the calling driver's ACTUAL `typeof(ctx.obj)`, not an asserted literal
        # (that mismatch -- campaign_unrestricted_runner.jl hardcoding "bundle_type=OperatorPsiBundle"
        # while ctx.obj was never actually converted -- is exactly the bug this field closes). Only
        # `run_polish_checkpointed_unified` passes this explicitly (post-`build_unrestricted_
        # operator_ctx`); pre-port callers (`run_profile_checkpointed`/`run_polish_checkpointed`)
        # never pass it and keep reporting the true pre-port value unchanged.
    return (
        family = :unrestricted,
        bundle_type = bundle_type,
        core_top1_engine = :canonical_log_additive,           # print_active_layout_banner's own literal
        outer_gradient_top3_engine = :cplus,                   # print_active_layout_banner's own literal
        core_moment_representation = :compressed_winner_form,   # port/shared-winner-pair-core-hessian-production-2026-07-25
        hessian_backend = hessian_backend,                     # kept for backward compat with pre-port callers/printers
        # port/shared-winner-pair-core-hessian-production-2026-07-25 (task §5): granular backend fields.
        # H_EE IS the whole Hessian for this family -- no cross/restriction block exists to report.
        core_hessian_backend = UNRESTRICTED_CORE_HESSIAN_BACKEND[],
        core_hessian_workers = UNRESTRICTED_CORE_HESSIAN_WORKERS[],
        core_hessian_worker_policy = core_hessian_worker_policy_label(),
        core_hessian_storage = UNRESTRICTED_CORE_HESSIAN_STORAGE[],
        cross_hessian_backend = :none,
        restriction_hessian_backend = :none,
        full_hessian_assembly = :packed_direct,   # winner-pair kernels write KNITRO's packed triangle natively, no dense round-trip
        knitro_hessian_format = :dense_rowmajor_packed_upper_triangle,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CHECKPOINT_SCHEMA_UNRESTRICTED,
        screen_stack = production_screen_stack(:unrestricted),
        trade_elasticity_mode = trade_elasticity_mode,
        A_coordinate_mode = A_coordinate_mode,
        A_coordinate_mapping_version = A_coordinate_mode == :legacy_z ? nothing : AMAP_VERSION,
        gp_coordinate_mode = gp_coordinate_mode,
        theta_coordinate = trade_elasticity_mode == :flexible ? :eta_theta : nothing,
        theta_bounds = theta_bounds,
        theta_derivative_backend = trade_elasticity_mode == :flexible ? :fixed_dual_secant : nothing,
        theta_aware_dual_bank = trade_elasticity_mode == :flexible,   # dual_bank_zfree, outer_coordinate_layout.jl (task §12 fix)
        outer_dimension = outer_dimension,
    )
end

"""
    resolve_flexible_cm_manifest(; cctx, blas_threads, cm_extension=:cm_only, meanzc_K_mean=0, meanzc_K_pair=0)

Resolves the flexible-CM (and, when `cm_extension != :cm_only`, CM+mean/ZC) family's backend
manifest. `cctx` is the real `CMBinHessCtx` the driver just built (`pcx.cctx`) -- `threaded_bins`
and `hessian_backend` are read directly off it, not asserted.
"""
function resolve_flexible_cm_manifest(; cctx, blas_threads::Union{Nothing,Int},
        cm_extension::Symbol = :cm_only, meanzc_K_mean::Int = 0, meanzc_K_pair::Int = 0,
        meanzc_target_layout::Symbol = :shared_by_power,   # CM+ZC-CROSS (2026-08-09): :shared_by_power
        # (diagonal, the pre-existing family -- default preserves every existing caller's reported
        # family symbol byte-for-byte) | :shared_by_power_cross. Reported as a DISTINCT family symbol
        # because a startup manifest that labelled a K_pair^2 cross run as plain `cm_meanzc` would be
        # actively misleading in exactly the artifact (the run's own provenance record) an analyst
        # later uses to tell the two apart.
        bundle_type::Symbol = :PsiObjectiveBundleImplicit)   # moment_representation threading task
        # (2026-07-29): the calling driver's ACTUAL `typeof(pcx.ctx_cm.obj)` -- see
        # resolve_unrestricted_manifest's identical field for the full rationale. Only
        # run_cm_upper_checkpointed passes this explicitly; other/older callers keep reporting the
        # pre-port default unchanged.
    is_meanzc = cm_extension !== :cm_only
    is_meanzc_cross = is_meanzc && meanzc_target_layout === :shared_by_power_cross
    family = is_meanzc_cross ? :cm_meanzc_cross : (is_meanzc ? :cm_meanzc : :flexible_cm)
    # port/shared-winner-pair-core-hessian-production-2026-07-25 (task §5), CORRECTED 2026-07-25
    # continuation: this manifest is printed at DRIVER STARTUP, before any inner solve has run --
    # `cctx.core_cf_ref[]` is thus *always* still `nothing`/unset at print time regardless of
    # whether the shared backend will be used (a real bug this session found: `core_active` was
    # always false here, so every startup manifest print silently claimed
    # `dense_reference_fallback_this_point` even when `:exact_winner_pair_parallel` was correctly
    # configured and WOULD be used from the very first Hessian callback onward). Report the
    # CONFIGURED backend directly -- whether it's actually reached on every call is what the
    # runtime counters (task §2, `CORE_HESSIAN_COUNTERS`/`resolve_core_hessian_counters_manifest`)
    # are for, checked AFTER a solve, not at this startup print.
    threaded_label = cctx.use_threaded_bins ? :threaded_architecture_c_with_winner_pair_core : :architecture_c_with_winner_pair_core
    nt = (
        family = family,
        bundle_type = bundle_type,
        core_top1_engine = :canonical_log_additive,
        outer_gradient_top3_engine = :cplus,
        core_moment_representation = :compressed_winner_form,   # allocation/Hessian port task §5
        cm_restriction_basis = :cumulative,
        cm_internal_storage = :bin_index,
        hessian_backend = cctx.core_hessian_backend === :dense_reference ?
            (cctx.use_threaded_bins ? :threaded_architecture_c : :serial_architecture_c) : threaded_label,
        threaded_bins = cctx.use_threaded_bins,
        core_hessian_backend = cctx.core_hessian_backend,
        core_hessian_workers = cctx.core_hessian_workers,
        core_hessian_worker_policy = core_hessian_worker_policy_label(),
        core_hessian_storage = cctx.core_hessian_storage,
        cross_hessian_backend = :cm_bin_prefix,          # H_EC -- unchanged (Phase B audit: already near-optimal, see docs)
        # Phase F remediation (production-audit continuation, 2026-07-26): this label was keyed on
        # `is_meanzc` alone, but R-congruence is actually gated on `cctx.R !== nothing`, which
        # tracks `contrasts == :orthonormal` (orthonormal_contrast_matrix), NOT on whether the
        # meanzc extension is active. run_cm_upper_checkpointed's own default is
        # `contrasts=:anchored` for BOTH the plain and meanzc branches, so under ordinary default
        # settings this label previously claimed congruence was active for CM+ZC when it was not
        # (found in the baseline static audit, docs/PRODUCTION_5X7_AUDIT_BASELINE_AND_
        # REMEDIATION_SIZING_2026-07-26.md, item B5). Keying on the real condition fixes the label;
        # the Hessian computation itself was already correct either way -- this is a reporting-only
        # fix, no numerical change.
        restriction_hessian_backend = cctx.R !== nothing ? :cm_bin_prefix_plus_congruence : :cm_bin_prefix,   # H_CC
        full_hessian_assembly = :dense_scratch_then_pack,   # cctx.Hfull dense corner-insertion, then one pack loop (unchanged)
        knitro_hessian_format = :dense_rowmajor_packed_upper_triangle,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CM_CHECKPOINT_SCHEMA,
        screen_stack = production_screen_stack(:flexible_cm),
    )
    # ZC Hessian backend production integration (2026-08-01), requirement: every live manifest
    # records the selected H_ZZ/H_CZ/H_EZ backends, not just the coarser hessian_backend/
    # cross_hessian_backend/restriction_hessian_backend labels above -- read directly off `cctx`
    # (not asserted), same discipline as every other field here. Meaningful for cm_meanzc only
    # (flexible_cm/common_frechet have no Z-restriction block; the fields exist on the shared
    # CMBinHessCtx struct but are never dispatched on for those families).
    is_meanzc && return merge(nt, (K_mean = meanzc_K_mean, K_pair = meanzc_K_pair,
        zc_gram_backend = cctx.zc_gram_backend, hcz_prep_backend = cctx.hcz_prep_backend,
        zc_ez_backend = cctx.zc_ez_backend))
    return nt
end

"""
    resolve_common_frechet_manifest(; cctx, blas_threads)

Phase F remediation (production-audit continuation, 2026-07-26): structured, JSON-able manifest
resolver for the common-Frechet family -- previously the ONLY family among the four restricted
families with no such resolver (its own startup manifest went through the print-only
`print_frechet_startup_manifest`, `cm_frechet_level.jl:322`, which cannot be serialized to JSON or
compared programmatically the way `resolve_flexible_cm_manifest`/`resolve_origin_zc_manifest`
already can be). Mirrors `resolve_flexible_cm_manifest`'s fields/style exactly (common-Frechet
shares the SAME `CMBinHessCtx`/`build_cm_bin_ctx` machinery for its economic core -- confirmed in
the baseline audit's Area 5/6 findings: H_EE dispatch is unchanged/shared, only the level-block
Hessian differs), adding the Frechet-specific fields (`frechet_feature_set`, `frechet_basis`,
`frechet_grid_size`, `frechet_level_count`, `level_hessian_backend`) `print_frechet_startup_manifest`
already prints as loose strings.
"""
function resolve_common_frechet_manifest(; cctx, blas_threads::Union{Nothing,Int})
    threaded_label = cctx.use_threaded_bins ? :threaded_architecture_c_with_winner_pair_core : :architecture_c_with_winner_pair_core
    level_hessian_backend = cctx.use_threaded_bins ? :threaded_architecture_c_frechet_level : :serial_architecture_c_frechet_level
    return (
        family = :common_frechet,
        core_top1_engine = :canonical_log_additive,
        outer_gradient_top3_engine = :cplus,
        core_moment_representation = :compressed_winner_form,
        cm_restriction_basis = :cumulative,
        cm_internal_storage = :bin_index,
        frechet_feature_set = :cdf_only,
        frechet_basis = :cm_contrasts_plus_common_level,
        frechet_grid_size = cctx.L,
        frechet_level_count = 1,
        hessian_backend = cctx.core_hessian_backend === :dense_reference ?
            (cctx.use_threaded_bins ? :threaded_architecture_c : :serial_architecture_c) : threaded_label,
        level_hessian_backend = level_hessian_backend,   # hessian_cm_frechet_structured_v2!/_! (task Phase 0 gate: threaded is the real production dispatch, 4.6x faster, ~1e-14 agreement)
        threaded_bins = cctx.use_threaded_bins,
        core_hessian_backend = cctx.core_hessian_backend,
        core_hessian_workers = cctx.core_hessian_workers,
        core_hessian_worker_policy = core_hessian_worker_policy_label(),
        core_hessian_storage = cctx.core_hessian_storage,
        cross_hessian_backend = :cm_bin_prefix,          # H_EC + H_E-level share the SAME bin/prefix tables, no new O(W) pass
        restriction_hessian_backend = :cm_bin_prefix,    # H_CC + H_level, common-Frechet has no free restriction parameters (unlike CM+ZC's eta_nu), so congruence is never applicable here
        full_hessian_assembly = :dense_scratch_then_pack,
        knitro_hessian_format = :dense_rowmajor_packed_upper_triangle,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CM_CHECKPOINT_SCHEMA,   # common-Frechet checkpoints/resumes through the SAME cm_checkpoint.jl infrastructure as flexible-CM
        screen_stack = production_screen_stack(:common_frechet),
    )
end

"""
    resolve_origin_zc_manifest(; octx, blas_threads)

port/shared-winner-pair-core-hessian-production-2026-07-25: `octx` is the
real `OriginZCCoreHessCtx` the driver built (`ctx_cm.octx`). `octx=nothing`
(a caller that built `ctx_cm` before this port, or via some other path)
reports the pre-port monolithic dense Architecture A unconditionally.

CORRECTED 2026-07-25 continuation: reports `octx`'s CONFIGURED backend
directly (not gated on whether a `CompressedFactual` has been built yet) --
this manifest prints at driver STARTUP, before any inner solve, so
`octx.core_cf_ref[]` is always still unset at print time regardless of what
backend will actually run. See `resolve_flexible_cm_manifest`'s identical fix.
"""
function resolve_origin_zc_manifest(; octx = nothing, blas_threads::Union{Nothing,Int},
        power_target_layout::Symbol = :origin_by_power,   # OZC-CROSS (2026-08-09): see
        # resolve_flexible_cm_manifest's identical `meanzc_target_layout` kwarg for the rationale.
        # Default preserves every existing caller's reported family symbol byte-for-byte.
        bundle_type::Symbol = :PsiObjectiveBundleImplicit)   # moment_representation threading task
        # (2026-07-29): the calling driver's ACTUAL `typeof(pcx.ctx_cm.obj)`, not asserted -- mirrors
        # resolve_unrestricted_manifest's identical fix. Only run_originzc_upper_checkpointed passes
        # this explicitly; the one other caller (a standalone benchmark script) never did and keeps
        # reporting the true pre-port value unchanged.
    core_configured = octx !== nothing && octx.core_hessian_backend !== :dense_reference
    is_cross = power_target_layout === :origin_by_power_cross
    return (
        family = is_cross ? :origin_zc_cross : :origin_zc,
        bundle_type = bundle_type,
        core_representation = core_configured ? :compressed_winner_form : :compressed,
        restriction_representation = is_cross ? :pairwise_zero_covariance_cross_power_grid : :pairwise_zero_covariance,
        hessian_backend = core_configured ? :partitioned_winner_pair_core_dense_restriction : :dense_architecture_a,
        core_hessian_backend = octx === nothing ? :dense_reference : octx.core_hessian_backend,
        core_hessian_workers = octx === nothing ? 0 : octx.core_hessian_workers,
        core_hessian_worker_policy = core_hessian_worker_policy_label(),
        core_hessian_storage = octx === nothing ? :none : octx.core_hessian_storage,
        cross_hessian_backend = :dense_exact,      # H_ER -- retained dense (task §4.4/§12), computed once, H_RE never independently
        restriction_hessian_backend = :dense_exact, # H_RR -- retained dense
        full_hessian_assembly = core_configured ? :dense_scratch_partitioned_then_pack : :dense_scratch_monolithic_then_pack,
        knitro_hessian_format = :dense_rowmajor_packed_upper_triangle,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CM_CHECKPOINT_SCHEMA_V7,
        screen_stack = production_screen_stack(:origin_zc),
        # ZC Hessian backend production integration (2026-08-01): the ZC-specific H_ZZ/H_EZ
        # backend selections (origin-ZC has no H_CZ -- no CM-grid block) -- read directly off
        # `octx`, not asserted, same discipline as every other field here.
        zc_gram_backend = octx === nothing ? :none : octx.zc_gram_backend,
        zc_ez_backend = octx === nothing ? :none : octx.zc_ez_backend,
    )
end

"""
    resolve_core_hessian_counters_manifest() -> NamedTuple

2026-07-25 continuation (task §2): the RUNTIME counterpart to the (static, resolved-at-setup)
manifest above -- proves which backend actually EXECUTED, not just which one was requested.
Reads `CORE_HESSIAN_COUNTERS[]` (`core_exact_hessian.jl`) live; call this AFTER a solve/benchmark
run, not at setup time (unlike `resolve_*_manifest`, which is meaningful before any callback has
fired). `dense_fallback_reason_counts` is flattened into individual `fallback_<reason>` fields
(zero-valued reasons included) so the JSON writer's generic NamedTuple serialization handles it
without a special case.
"""
function resolve_core_hessian_counters_manifest()
    c = CORE_HESSIAN_COUNTERS[]
    reason_fields = NamedTuple(Symbol("fallback_", r) => get(c.dense_fallback_reason_counts, r, 0) for r in CORE_HESSIAN_FALLBACK_REASONS)
    return merge((
        winner_pair_hessian_calls = c.winner_pair_hessian_calls,
        winner_pair_serial_calls = c.winner_pair_serial_calls,
        winner_pair_parallel_calls = c.winner_pair_parallel_calls,
        dense_core_fallback_calls = c.dense_core_fallback_calls,
        compressed_core_rebuilds = c.compressed_core_rebuilds,
    ), reason_fields)
end

"Human-readable startup print of a resolved manifest NamedTuple -- flushed immediately, same discipline as the existing [winner-engine]/[screen-stack]/[threshold-config] banners this supplements."
function print_production_backend_manifest(nt)
    println("[backend-manifest] family=", nt.family)
    for k in propertynames(nt)
        k === :family && continue
        v = getproperty(nt, k)
        println("[backend-manifest]   ", k, "=", v isa AbstractVector ? join(v, ",") : v)
    end
    flush(stdout)
    return nothing
end

"Minimal dependency-free JSON serialization of a resolved manifest NamedTuple (no JSON.jl dependency assumed for this diagnostic-only writer)."
function manifest_to_json_fragment(nt)
    io = IOBuffer()
    print(io, "  {\n")
    ks = propertynames(nt)
    for (i, k) in enumerate(ks)
        v = getproperty(nt, k)
        vs = if v isa AbstractVector
            "[" * join(("\"" * string(x) * "\"" for x in v), ", ") * "]"
        elseif v isa Symbol || v isa AbstractString
            "\"" * string(v) * "\""
        elseif v isa Bool || v isa Number
            string(v)
        else
            "\"" * string(v) * "\""
        end
        print(io, "    \"", k, "\": ", vs, i == length(ks) ? "\n" : ",\n")
    end
    print(io, "  }")
    return String(take!(io))
end

"""
    write_production_backend_manifest_json(path, manifests::Vector; generated_at=now())

Writes the required `PRODUCTION_BACKEND_MANIFEST_2026-07-25.json` deliverable: an array of
resolved manifests (one per family actually exercised), each as returned by
`resolve_unrestricted_manifest`/`resolve_flexible_cm_manifest`/`resolve_origin_zc_manifest`.
"""
function write_production_backend_manifest_json(path::AbstractString, manifests::AbstractVector; generated_at = now())
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"generated_at\": \"", generated_at, "\",")
        println(io, "  \"manifests\": [")
        for (i, nt) in enumerate(manifests)
            print(io, manifest_to_json_fragment(nt))
            println(io, i == length(manifests) ? "" : ",")
        end
        println(io, "  ]")
        println(io, "}")
    end
    return nothing
end
