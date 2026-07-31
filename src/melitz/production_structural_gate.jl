# Phase 14 runtime safeguards (2026-07-31 legacy-H removal audit,
# docs/melitz_legacy_H_removal_audit_2026-07-31.md). Two small, cheap, always-derive-from-
# the-live-object helpers -- neither changes any economic/numerical behavior.
#
# `melitz_live_backend_manifest(obj)` answers "what did this run actually construct" by
# INSPECTING `obj` (type, fieldnames, per-field byte sizes, legacy-field presence) and the
# process-global Phase 10 counters -- never printing a claim that isn't re-derived from the
# object at call time. `melitz_assert_production_bundle!(obj)` is the fatal gate: call it
# once, immediately before a production KNITRO campaign begins, to hard-fail rather than
# silently proceed if the live bundle is ever not the H-less production type, or if any
# dense-path counter is nonzero since the caller's own `melitz_backend_counters_reset!()`.

const MELITZ_FORBIDDEN_BUNDLE_FIELDS = (:H, :H_copy, :G, :K, :ones, Symbol("moments!"), :jac_h_theta)
# note: MelitzCCBundle's own `jac_h`/`needs_outer_moment_jacobian` fields are NOT forbidden --
# they are the documented always-empty/always-false duck-type-compatibility fields (cc_bundle.jl
# struct docstring), never a legacy dense theta-jacobian; `jac_h_theta` above is a distinct
# name reserved only so this list stays exact if that ever changed.

"""
    melitz_live_backend_manifest(obj) -> NamedTuple

Derives a full backend/field manifest from the LIVE object `obj` -- never a hardcoded string.
Includes the concrete bundle type, whether any forbidden legacy-dense field name is present
on that type, a byte-size inventory of every array field, and the current Phase 10 dense/
matrix-free usage counters (`melitz_backend_counters_snapshot()`). Safe to call on any bundle
type (dense legacy or `MelitzCCBundle`) -- purely inspects, mutates nothing.
"""
function melitz_live_backend_manifest(obj)
    T = typeof(obj)
    fns = fieldnames(T)
    present_forbidden = Tuple(f for f in MELITZ_FORBIDDEN_BUNDLE_FIELDS if f in fns)
    field_bytes = NamedTuple{fns}(ntuple(i -> begin
        v = getfield(obj, fns[i])
        v isa AbstractArray ? Base.summarysize(v) : 0
    end, length(fns)))
    return (
        bundle_type = T,
        fieldnames = fns,
        forbidden_fields_present = present_forbidden,
        is_structurally_h_less = isempty(present_forbidden),
        total_bytes = Base.summarysize(obj),
        field_bytes = field_bytes,
        hessian_backend = hasproperty(obj, :hessian_backend) ? getfield(obj, :hessian_backend) : missing,
        counters = melitz_backend_counters_snapshot(),
    )
end

"""
    melitz_print_live_backend_manifest(obj)

Prints `melitz_live_backend_manifest(obj)` in a fixed, readable format -- the Phase 14 "live
backend manifest," derived from the object every time it is called, never a cached or
hardcoded string.
"""
function melitz_print_live_backend_manifest(obj)
    m = melitz_live_backend_manifest(obj)
    println("Melitz live backend manifest:")
    println("  bundle_type              = ", m.bundle_type)
    println("  is_structurally_h_less    = ", m.is_structurally_h_less)
    println("  forbidden_fields_present  = ", m.forbidden_fields_present)
    println("  total bundle bytes        = ", m.total_bytes)
    println("  hessian_backend           = ", m.hessian_backend)
    println("  dense_moment_calls        = ", m.counters.dense_moment_calls)
    println("  dense_inner_*_calls       = (", m.counters.dense_inner_objective_calls, ", ",
            m.counters.dense_inner_gradient_calls, ", ", m.counters.dense_inner_hessian_calls, ")")
    println("  dense_G_materializations  = ", m.counters.dense_G_materializations)
    println("  production_dense_screen_calls = ", m.counters.production_dense_screen_calls)
    return m
end

"""
    melitz_assert_production_bundle!(obj; require_zero_dense_counters::Bool=true)

Phase 14's fatal production assertion. Call once, immediately before a real production
KNITRO campaign begins (after `melitz_backend_counters_reset!()` has been called at the
start of that campaign, so counters reflect only this run). Throws `ErrorException`
immediately if:

1. `typeof(obj)` is not `MelitzCCBundle` (a caller that deliberately wants the dense
   reference bundle for a diagnostic run should NOT call this assertion at all -- it is a
   PRODUCTION-only gate, not a general-purpose type check); or
2. any name in `MELITZ_FORBIDDEN_BUNDLE_FIELDS` is a field of `typeof(obj)` (should be
   structurally impossible for `MelitzCCBundle` today -- this is the belt-and-suspenders
   check for a future accidental field addition); or
3. `require_zero_dense_counters=true` (default) and any of
   `dense_moment_calls`/`dense_inner_objective_calls`/`dense_inner_gradient_calls`/
   `dense_inner_hessian_calls`/`dense_G_materializations`/`production_dense_screen_calls`
   is nonzero.

Returns the manifest (so a caller can log it) on success.
"""
function melitz_assert_production_bundle!(obj; require_zero_dense_counters::Bool=true)
    m = melitz_live_backend_manifest(obj)
    m.bundle_type === MelitzCCBundle || error(
        "melitz_assert_production_bundle!: expected bundle type MelitzCCBundle for a " *
        "production run, got $(m.bundle_type). If this is a deliberate dense-reference " *
        "diagnostic run, do not call this assertion.")
    isempty(m.forbidden_fields_present) || error(
        "melitz_assert_production_bundle!: MelitzCCBundle unexpectedly has forbidden " *
        "legacy-dense field(s) $(m.forbidden_fields_present) -- this should be structurally " *
        "impossible; the struct definition (cc_bundle.jl) must have changed.")
    if require_zero_dense_counters
        c = m.counters
        bad = filter(kv -> kv[1] in (:dense_moment_calls, :dense_inner_objective_calls,
            :dense_inner_gradient_calls, :dense_inner_hessian_calls, :dense_G_materializations,
            :production_dense_screen_calls) && kv[2] != 0, pairs(c))
        isempty(bad) || error(
            "melitz_assert_production_bundle!: nonzero dense-path counter(s) since the last " *
            "melitz_backend_counters_reset!(): $(collect(bad)) -- a dense fallback fired " *
            "during what was supposed to be a pure matrix-free production run.")
    end
    return m
end
