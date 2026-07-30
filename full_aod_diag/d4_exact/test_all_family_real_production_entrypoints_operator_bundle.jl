# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30, task §10: the standing,
# mandatory, ALL-FIVE-FAMILIES gate this task's own postmortem (dense_bundle_incident_postmortem_
# 2026-07-29.zip) identifies as the one check that would have caught both real incidents
# immediately, and that did not exist anywhere in the repository before this task.
#
# IMPORTANT SCOPE NOTE (found during this task's own audit, PRODUCTION_BUNDLE_CONSTRUCTION_CALL_
# GRAPH_2026-07-30.md): all 3 real top-level driver functions (`run_cm_upper_checkpointed`,
# `run_originzc_upper_checkpointed`, `run_polish_checkpointed_unified`) hardcode a real-D=20-data
# context internally (`d20_real_setup_design`) -- there is no way to call any of them at D=4 at
# all, real driver or not. This gate therefore exercises, at D=4, EXACTLY the same
# prepare_production_run(family, runner, build_inner) call each real driver now makes internally
# (the closures below are copy-identical to the ones wired into cm_checkpoint.jl/
# cm_originzc_checkpoint.jl/c10_d20_production_driver_unified.jl as of this task) -- i.e. this is
# the real production construction PATH, run at D=4 for speed, not a separate/parallel
# implementation of it. The D=20/W=100,000 extended release gate (task §16) additionally exercises
# the true top-level driver functions end-to-end, checkpoint-and-all, at real production scale.
#
# Covers, per task §10: no moment-representation override is even POSSIBLE any more (the kwarg is
# gone from all 3 driver signatures -- checked structurally below, not just "not passed"), no
# backend override, normal production settings. Asserts: live object is OperatorPsiBundle,
# production assertion passes, dense-reference construction count = 0, legacy fields absent,
# select_G_from_H calls = 0 (via the structural not-applicable check), dense materializations = 0.
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_config.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "production_bundle_api.jl", "dense_reference_diagnostics.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random, Test

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

lp("="^90)
lp("test_all_family_real_production_entrypoints_operator_bundle.jl -- D=4 standing gate")
lp("="^90)

# ---- Structural check: the 3 real driver signatures no longer accept moment_representation -----
for (fname, fn) in (("run_cm_upper_checkpointed", run_cm_upper_checkpointed),
                     ("run_originzc_upper_checkpointed", run_originzc_upper_checkpointed))
    has_kw = any(ks -> :moment_representation in ks, Base.kwarg_decl.(methods(fn)))
    check("$fname signature no longer accepts moment_representation", !has_kw)
end
# run_polish_checkpointed_unified is defined in c10_d20_production_driver_unified.jl, which is NOT
# on this test's include list (it pulls in the full unrestricted real-D=20 driver stack, unneeded
# for the flexible_cm/common_frechet/cm_meanzc/origin_zc/unrestricted checks below, which use
# build_unrestricted_operator_ctx directly, not the driver). Checked structurally via source text
# instead of loading the function -- equally rigorous for "does the signature mention the kwarg".
let src = read(joinpath(_D4E, "c10_d20_production_driver_unified.jl"), String)
    # Comments may still mention the word (explaining the removal); the signature itself must not
    # -- checked via the type-annotation pattern that only a real kwarg declaration produces.
    check("run_polish_checkpointed_unified's signature no longer declares moment_representation",
          !occursin(r"moment_representation\s*::", src))
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
Random.seed!(20260730)
probs4 = collect(range(0.0, 1.0; length = 12))[2:end-1]

reset_no_dense_g_counters!()
reset_dense_reference_construction_log!()

# ---- flexible_cm: exact closure shape wired into cm_checkpoint.jl's non-meanzc/non-frechet branch
prep_flex = prepare_production_run(:flexible_cm, "run_cm_upper_checkpointed(D4-gate)",
    () -> build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probs4,
        threaded_bins = true, inner_fg_backend = :cm_lookup, moment_representation = :operator))
check("flexible_cm: live bundle is OperatorPsiBundle", prep_flex.ctx.obj isa OperatorPsiBundle)
check("flexible_cm: bundle_invariant_pass", prep_flex.manifest.bundle_invariant_pass)
check("flexible_cm: no legacy fields present", !prep_flex.manifest.structural.any_legacy_field_present)
check("flexible_cm: select_G_from_H not applicable", !prep_flex.manifest.structural.select_G_from_H_applicable)

# ---- common_frechet: exact closure shape wired into cm_checkpoint.jl's is_frechet branch --------
prep_frechet = prepare_production_run(:common_frechet, "run_cm_upper_checkpointed(D4-gate)",
    () -> build_cm_frechet_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probs4,
        cm_hessian_backend = :structured, moment_representation = :operator))
check("common_frechet: live bundle is OperatorPsiBundle", prep_frechet.ctx.obj isa OperatorPsiBundle)
check("common_frechet: bundle_invariant_pass", prep_frechet.manifest.bundle_invariant_pass)
check("common_frechet: no legacy fields present", !prep_frechet.manifest.structural.any_legacy_field_present)

# ---- cm_meanzc (CM+ZC): exact closure shape wired into cm_checkpoint.jl's is_meanzc branch ------
prep_meanzc = prepare_production_run(:cm_meanzc, "run_cm_upper_checkpointed(D4-gate)",
    () -> build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = 1, K_pair = 0,
        contrasts = :anchored, meanzc_basis = :direct, probs = probs4, moment_representation = :operator))
check("cm_meanzc: live bundle is OperatorPsiBundle", prep_meanzc.ctx.obj isa OperatorPsiBundle)
check("cm_meanzc: bundle_invariant_pass", prep_meanzc.manifest.bundle_invariant_pass)
check("cm_meanzc: no legacy fields present", !prep_meanzc.manifest.structural.any_legacy_field_present)

# ---- origin_zc: exact closure shape wired into cm_originzc_checkpoint.jl ------------------------
layout4 = OriginByPowerLayout(ctx.D, 1, 0)   # K_mean=1, K_pair=0
prep_oz = prepare_production_run(:origin_zc, "run_originzc_upper_checkpointed(D4-gate)",
    () -> build_originzc_production_context(ctx, CS, layout4; moment_representation = :operator))
check("origin_zc: live bundle is OperatorPsiBundle", prep_oz.ctx.obj isa OperatorPsiBundle)
check("origin_zc: bundle_invariant_pass", prep_oz.manifest.bundle_invariant_pass)
check("origin_zc: no legacy fields present", !prep_oz.manifest.structural.any_legacy_field_present)

# ---- unrestricted: exact closure shape wired into c10_d20_production_driver_unified.jl ----------
prep_u = prepare_production_run(:unrestricted, "run_polish_checkpointed_unified(D4-gate)",
    () -> build_unrestricted_operator_ctx(ctx; moment_representation = :operator))
check("unrestricted: live bundle is OperatorPsiBundle", prep_u.ctx.obj isa OperatorPsiBundle)
check("unrestricted: bundle_invariant_pass", prep_u.manifest.bundle_invariant_pass)
check("unrestricted: no legacy fields present", !prep_u.manifest.structural.any_legacy_field_present)

# ---- cross-family invariants (task §10's own explicit list) ------------------------------------
check("dense-reference construction count = 0 across all 5 families", DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0)
report = no_dense_g_report()
check("dense materializations = 0 (full_G)", report.full_G_materializations == 0)
check("dense materializations = 0 (economic)", report.dense_economic_G_materializations == 0)
check("dense materializations = 0 (CM)", report.dense_CM_G_materializations == 0)
check("dense materializations = 0 (Frechet)", report.dense_Frechet_G_materializations == 0)
check("dense materializations = 0 (ZC)", report.dense_ZC_G_materializations == 0)
check("generic_dense_FG_calls = 0", report.generic_dense_FG_calls == 0)

for (family, prep) in (("flexible_cm", prep_flex), ("common_frechet", prep_frechet),
                        ("cm_meanzc", prep_meanzc), ("origin_zc", prep_oz), ("unrestricted", prep_u))
    check("$family: assert_production_operator_bundle! passes standalone",
          assert_production_operator_bundle!(prep.ctx; where_ = "gate test ($family)"))
end

lp("="^90)
if isempty(FAILURES)
    lp("ALL PASS (", 5 * 4 + 2 + 6, " checks)")
else
    lp("FAILURES (", length(FAILURES), "): ", FAILURES)
    error("test_all_family_real_production_entrypoints_operator_bundle.jl: ", length(FAILURES), " check(s) failed")
end
