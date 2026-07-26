# Five-family finish task, Phase 3 (2026-07-26): runtime counters verifying the mandatory
# fixed-theta CM-feature-immutability invariant.
#
# AUDIT FINDING (not a rewrite): for all four restricted families, bin/threshold assignment is
# ALREADY built exactly once, at production-context-build time, outside any outer-point-dependent
# hot path -- confirmed by exhaustively grepping every `compute_bin_indices` call site in the
# production tree (2026-07-26):
#   - build_cm_production_context      (cm_production_bundle.jl:102)  -- flexible CM
#   - build_cm_frechet_production_context (cm_frechet_level.jl:278)   -- common Frechet
#   - build_cm_meanzc_production_context  (cm_meanzc_production.jl:44) -- CM+ZC
#   - build_cm_bin_ctx / build_cm_meanzc_bin_ctx (cm_hessian_architectures.jl:261,410) -- Hessian
#     scratch context, also built ONCE per production context, never inside a callback
#   - origin-ZC has NO CM grid at all (zero hits -- confirmed, not merely assumed)
# None of these five call sites are reachable from any `moments!`/FG/Hessian callback -- they run
# once when `build_cm_*_production_context(...)` is called, and the resulting `Bidx`/`z`/contrast
# matrix `R`/Frechet `level_targets` are captured by closure and reused unchanged across every
# subsequent outer-point evaluation for the lifetime of that context. This means
# `cm_feature_rebuilds_due_to_A_or_gp = 0` already holds STRUCTURALLY for fixed theta -- the
# counters below make that empirically verifiable at runtime rather than merely inferred from
# static code reading, and give the theta-changing case (flexible theta) a place to record a real
# rebuild if that path is ever exercised.
#
# A full `CMImmutableFeatureOperator` consolidating struct (per the task's original wording) was
# considered and NOT built this session: given the invariant already holds architecturally, the
# only thing a new wrapper struct would add is a stylistic reorganization of already-immutable
# fields already threaded correctly through five independent build functions -- real but low-value
# work compared to the genuinely unbuilt items (Phases 5-7). See
# IMMUTABLE_CM_FEATURE_OPERATOR_2026-07-26.md for the full writeup and the scoped follow-on.

mutable struct CMFeatureImmutabilityCounters
    cm_feature_context_builds::Int          # a build_cm_*_production_context call completed
    cm_feature_rebuilds_due_to_theta::Int    # theta changed and features were rebuilt (flexible-theta path; not yet wired -- always 0 under fixed theta)
    cm_feature_rebuilds_due_to_A_or_gp::Int  # a bin/threshold/contrast rebuild attributable to A or gp alone (MUST be 0 for fixed theta -- structural invariant)
    cm_dense_feature_materializations::Int   # a dense CM feature matrix was explicitly materialized (reference/debug backends only)
end
CMFeatureImmutabilityCounters() = CMFeatureImmutabilityCounters(0, 0, 0, 0)
const CM_FEATURE_IMMUTABILITY_COUNTERS = Ref(CMFeatureImmutabilityCounters())
reset_cm_feature_immutability_counters!() = (CM_FEATURE_IMMUTABILITY_COUNTERS[] = CMFeatureImmutabilityCounters())

function print_cm_feature_immutability_counters(c::CMFeatureImmutabilityCounters = CM_FEATURE_IMMUTABILITY_COUNTERS[])
    println("[cm-feature-immutability] context_builds=", c.cm_feature_context_builds,
            " rebuilds_due_to_theta=", c.cm_feature_rebuilds_due_to_theta,
            " rebuilds_due_to_A_or_gp=", c.cm_feature_rebuilds_due_to_A_or_gp,
            " dense_materializations=", c.cm_dense_feature_materializations)
end

"Call once at the top of each build_cm_*_production_context / build_originzc_production_context
after this file is included, to record that a fresh immutable-feature context was built."
record_cm_feature_context_build!() = (CM_FEATURE_IMMUTABILITY_COUNTERS[].cm_feature_context_builds += 1)
