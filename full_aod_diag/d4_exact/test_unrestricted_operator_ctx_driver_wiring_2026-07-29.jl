# ================================================================================================
# SUPERSEDED 2026-07-30 (architecture/production-operator-bundle-hardening-2026-07-30).
#
# This script originally called run_polish_checkpointed_unified with an explicit
# moment_representation kwarg, both ways, to prove the 2026-07-29 postmortem's first incident
# (unrestricted's real production driver silently staying on the dense bundle) was fixed. As of
# this task, that kwarg no longer exists on the driver's signature at all -- there is no
# representation choice left to thread through and check both ways; the driver unconditionally
# constructs OperatorPsiBundle via prepare_production_run (production_bundle_api.jl), which is
# itself asserted, not merely tested once.
#
# The property this script used to prove is now structurally guaranteed rather than empirically
# checked -- but the regression class it guarded against is exactly what
# test_all_family_real_production_entrypoints_operator_bundle.jl checks on every run, for all 5
# families at once, with zero overrides. Run that instead. This file is retained only as a
# historical pointer, not executable as a meaningful check any more.
#
# See also: PRODUCTION_BUNDLE_CONSTRUCTION_CALL_GRAPH_2026-07-30.md,
# DENSE_REFERENCE_REACHABILITY_AUDIT_2026-07-30.md.
# ================================================================================================
println("test_unrestricted_operator_ctx_driver_wiring_2026-07-29.jl: SUPERSEDED by " *
        "test_all_family_real_production_entrypoints_operator_bundle.jl (architecture/" *
        "production-operator-bundle-hardening-2026-07-30) -- run that instead.")
