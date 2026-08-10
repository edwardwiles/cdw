# NEGATIVE test for the schema-11 `meanzc_target_layout` resume guard (2026-08-09).
#
# The POSITIVE case (a cross checkpoint resumed as cross) is covered by
# production_smoke_cmzc_cross_2026-08-09.jl's own `resume` mode. This covers the case that actually
# matters for correctness: resuming a CROSS checkpoint while requesting the DIAGONAL layout (or vice
# versa) must HARD-REFUSE, not silently reinterpret the stored dual_warm_start under a different
# restriction.
#
# Why this needs its own test rather than inspection: the two layouts agree on EVERY other persisted
# field -- cm_extension, meanzc_K_mean, meanzc_K_pair, meanzc_basis, and even eta_nu's length are
# identical -- so before schema 11 there was literally nothing in the file to distinguish them, and
# every other resume guard would have passed. See CMCheckpointV11's docstring.
#
# The guard fires in run_cm_upper_checkpointed's resume block, which runs BEFORE `ctx` is built, so
# this test costs a compile and no solver time.
#
# Usage: julia --project=. -t 2 .../test_cmzc_cross_resume_guard_2026-08-09.jl <path_to_cross_checkpoint.jls>
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf

const CKPT = ARGS[1]
isfile(CKPT) || error("test_cmzc_cross_resume_guard: no checkpoint at $CKPT")

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name); flush(stdout)
end

c = load_cm_checkpoint(CKPT)
println("checkpoint: schema=$(c.schema) meanzc_target_layout=$(c.meanzc_target_layout) " *
        "cm_extension=$(c.cm_extension) K=($(c.meanzc_K_mean),$(c.meanzc_K_pair)) |eta_nu|=$(length(c.eta_nu))")
check("test fixture is a CROSS checkpoint", c.meanzc_target_layout === :shared_by_power_cross)

SNAPS = nested_grid_sequence([10, 20, 50])
PROBS = SNAPS[c.cm_L]

"Call the driver requesting `layout`, resuming from the cross checkpoint. Returns the error message, or \"NO ERROR\"."
function try_resume(layout::Symbol)
    try
        run_cm_upper_checkpointed(nothing;
            W = c.W, delta = c.delta, draw_design = c.draw_design, draw_seed = c.draw_seed,
            L = c.cm_L, contrasts = c.cm_contrasts, probs = PROBS, include_truncated_moment = true,
            cm_extension = c.cm_extension, meanzc_K_mean = c.meanzc_K_mean, meanzc_K_pair = c.meanzc_K_pair,
            meanzc_target_layout = layout,
            meanzc_profiled_level = 2, A_coordinate_mode = c.A_coordinate_mode,
            inner_lower_limit = -10.0, destination_sample = c.destination_sample,
            exclude_diagonal_gravity = true,
            gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
            ckpt_dir = mktempdir(), run_id = "guardtest", label = "guardtest",
            resume_from = CKPT, checkpoint_interval_s = 1e9, maxtime_real = 1.0, verbose = false)
        return "NO ERROR"
    catch e
        return sprint(showerror, e)
    end
end

println("\n--- requesting :shared_by_power (DIAGONAL) against a CROSS checkpoint -> must REFUSE ---")
msg = try_resume(:shared_by_power)
println("  ", first(msg, 420))
check("mismatched layout resume is REFUSED", occursin("meanzc_target_layout MISMATCH", msg))
check("refusal names both the stored and the requested layout",
      occursin("shared_by_power_cross", msg) && occursin("this call requests :shared_by_power", msg))
check("refusal is not a generic/unrelated error",
      !occursin("UndefKeyword", msg) && !occursin("MethodError", msg))

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
