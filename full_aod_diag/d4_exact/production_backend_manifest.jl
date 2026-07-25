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
    resolve_unrestricted_manifest(; hessian_backend, blas_threads, julia_threads=Threads.nthreads())

Resolves the unrestricted family's backend manifest from values the calling driver
(`run_profile_checkpointed`/`run_polish_checkpointed`) already has in scope.
"""
function resolve_unrestricted_manifest(; hessian_backend::Symbol, blas_threads::Union{Nothing,Int})
    return (
        family = :unrestricted,
        core_top1_engine = :canonical_log_additive,           # print_active_layout_banner's own literal
        outer_gradient_top3_engine = :cplus,                   # print_active_layout_banner's own literal
        core_moment_representation = :compressed,               # moment_representation=:compressed production default
        hessian_backend = hessian_backend,                     # :dense_exact (serial) unless a caller opts into a different backend
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CHECKPOINT_SCHEMA_UNRESTRICTED,
        screen_stack = production_screen_stack(:unrestricted),
    )
end

"""
    resolve_flexible_cm_manifest(; cctx, blas_threads, cm_extension=:cm_only, meanzc_K_mean=0, meanzc_K_pair=0)

Resolves the flexible-CM (and, when `cm_extension != :cm_only`, CM+mean/ZC) family's backend
manifest. `cctx` is the real `CMBinHessCtx` the driver just built (`pcx.cctx`) -- `threaded_bins`
and `hessian_backend` are read directly off it, not asserted.
"""
function resolve_flexible_cm_manifest(; cctx, blas_threads::Union{Nothing,Int},
        cm_extension::Symbol = :cm_only, meanzc_K_mean::Int = 0, meanzc_K_pair::Int = 0)
    is_meanzc = cm_extension !== :cm_only
    family = is_meanzc ? :cm_meanzc : :flexible_cm
    nt = (
        family = family,
        core_top1_engine = :canonical_log_additive,
        outer_gradient_top3_engine = :cplus,
        core_moment_representation = :compressed_winner_form,   # allocation/Hessian port task §5
        cm_restriction_basis = :cumulative,
        cm_internal_storage = :bin_index,
        hessian_backend = cctx.use_threaded_bins ? :threaded_architecture_c : :serial_architecture_c,
        threaded_bins = cctx.use_threaded_bins,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CM_CHECKPOINT_SCHEMA,
        screen_stack = production_screen_stack(:flexible_cm),
    )
    is_meanzc && return merge(nt, (K_mean = meanzc_K_mean, K_pair = meanzc_K_pair))
    return nt
end

"""
    resolve_origin_zc_manifest(; blas_threads)

Resolves the origin-specific-ZC family's backend manifest. Always Architecture A (generic dense
Hessian) -- see ORIGIN_ZC_HESSIAN_DIAGNOSIS_2026-07-25.md for why this is retained rather than
forcing Architecture C onto a restriction basis it was never designed for.
"""
function resolve_origin_zc_manifest(; blas_threads::Union{Nothing,Int})
    return (
        family = :origin_zc,
        core_representation = :compressed,
        restriction_representation = :pairwise_zero_covariance,
        hessian_backend = :dense_architecture_a,
        julia_threads = Threads.nthreads(),
        blas_threads = something(blas_threads, BLAS.get_num_threads()),
        checkpoint_schema = CM_CHECKPOINT_SCHEMA_V7,
        screen_stack = production_screen_stack(:origin_zc),
    )
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
