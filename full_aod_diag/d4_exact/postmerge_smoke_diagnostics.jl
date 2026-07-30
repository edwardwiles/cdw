# Post-merge smoke diagnostics (2026-07-28). The real public drivers (`run_cm_upper_checkpointed`,
# `run_originzc_upper_checkpointed`, `run_profile_checkpointed`) already unconditionally print their
# own live `production_backend_manifest.jl` manifest (economic FG backend, H_EE/H_EC/H_CC backend
# labels, checkpoint schema, screen stack) at startup -- see e.g. cm_checkpoint.jl:914's
# `print_production_backend_manifest(resolve_flexible_cm_manifest(...))` call, which fires on every
# real call, not just this smoke. This file adds only what that manifest does NOT already cover:
# the bundle's structural field absence (proven fresh, same code, in this session's premerge gate)
# and the dynamic no-dense-G runtime counters (`no_dense_g_counters.jl`), read AFTER the real solve.
function print_no_h_bundle_facts(family::AbstractString)
    println("-"^90)
    # moment_representation threading task (2026-07-29) correction: this block used to assert
    # "bundle_type = OperatorPsiBundle" unconditionally for every family, sourced from NOTHING
    # (not even MOMENT_REPRESENTATION[], which only governs flexible_cm) -- FALSE for cm_meanzc and
    # origin_zc at the time it was written (both builders hardcoded :dense_reference and the real
    # drivers never threaded a kwarg to override it; confirmed live 2026-07-29, see
    # cm_meanzc_production.jl::build_cm_meanzc_production_context / cm_originzc_production.jl::
    # build_originzc_production_context history). Both are now genuinely :operator by default too
    # (flipped + validated D=4 + real D=20/W=100,000 through their real drivers end-to-end,
    # test_meanzc_originzc_driver_wiring_2026-07-29.jl), so the claim below is no longer false --
    # but this print is called at campaign-script STARTUP, before any real ctx/pcx exists, so it
    # can only ever describe the CONFIGURED default, never a verified live fact. The REAL,
    # per-call bundle_type -- derived from the actual ctx.obj this specific run built, not asserted
    # -- is what run_cm_upper_checkpointed/run_originzc_upper_checkpointed's own
    # [backend-manifest] bundle_type= line reports (production_backend_manifest.jl); that is the
    # authoritative source, this is a config disclosure only.
    println("NO-H BUNDLE CONFIG: ", family, "  (moment_representation default = ", MOMENT_REPRESENTATION[],
            " -- only governs flexible_cm; common_frechet/cm_meanzc/origin_zc each resolve their own default independently)")
    println("  :operator (production default for ALL FIVE families as of 2026-07-29) constructs OperatorPsiBundle;")
    println("  :dense_reference is the explicit opt-out, unchanged PsiObjectiveBundleImplicit.")
    println("  has_H_field = false, has_H_copy_field = false, has_moments!_field = false, has_K_field = false (renamed payoff)")
    println("  -- structural absence proven fresh this session in TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md, same commit")
    println("  See each real run's own [backend-manifest] bundle_type= line below for the ACTUAL type used.")
    println("-"^90)
end

function reset_no_h_counters!()
    reset_no_dense_g_counters!()
end

function print_no_h_counters(family::AbstractString)
    c = NO_DENSE_G_COUNTERS[]
    println("-"^90)
    println("NO-H RUNTIME COUNTERS: ", family)
    println("  full_G_materializations              = ", c.full_G_materializations, "  (must be 0)")
    println("  dense_economic_G_materializations     = ", c.dense_economic_G_materializations, "  (must be 0)")
    println("  dense_CM_G_materializations           = ", c.dense_CM_G_materializations, "  (must be 0)")
    println("  dense_ZC_G_materializations           = ", c.dense_ZC_G_materializations, "  (must be 0)")
    println("  dense_Frechet_G_materializations      = ", c.dense_Frechet_G_materializations, "  (must be 0)")
    println("  generic_dense_FG_calls (production moments!/select_G_from_H calls) = ", c.generic_dense_FG_calls, "  (must be 0)")
    println("  operator_FG_calls                     = ", c.operator_FG_calls)
    println("  hessian_weight_dense_recomputes (must be 0 in production default) = ", c.hessian_weight_dense_recomputes)
    println("-"^90)
end
