# ================================================================================================
# fullA-lower-limit-and-hotpath-2026-08-06, task P0/section 5/6/14: standing gate proving
# `inner_lower_limit` is the SINGLE required source of truth for the inner KNITRO objective's
# `lower_limit` backstop, all the way from `d20_real_setup` (the one place it is constructed)
# through every real family-specific production bundle, and that the dead
# ThresholdAbortState/resolve_threshold_for_delta early-abort mechanism is no longer constructed
# with a live (non-Inf) threshold anywhere on this path.
#
# Real D20 data, small W (5,000) for speed -- this is a propagation/wiring proof, not a scientific
# campaign; task §5 explicitly permits D20/W=5,000-20,000 here rather than W=100,000.
# ================================================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_hessian_subblock_profiling.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_config.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_cplus.jl", "cm_meanzc_lookup_production.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "production_bundle_api.jl", "dense_reference_diagnostics.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

lp("="^90)
lp("test_lower_limit_production_source_of_truth_2026-08-06.jl")
lp("="^90)

# ---- 1. Structural: the 3 real driver functions + the two shared context builders all declare
#         inner_lower_limit with NO default (required kwarg) ----------------------------------
for (fname, fn) in (("d20_real_setup", d20_real_setup), ("d20_real_setup_design", d20_real_setup_design),
                     ("run_cm_upper_checkpointed", run_cm_upper_checkpointed),
                     ("run_originzc_upper_checkpointed", run_originzc_upper_checkpointed))
    has_kw = any(ks -> :inner_lower_limit in ks, Base.kwarg_decl.(methods(fn)))
    check("$fname declares inner_lower_limit kwarg", has_kw)
end
let src = read(joinpath(_D4E, "c10_d20_production_driver_unified.jl"), String)
    check("run_polish_checkpointed_unified's signature declares inner_lower_limit (no default)",
          occursin(r"inner_lower_limit\s*::\s*Float64\s*,", src))
end

# ---- 2. Omitting inner_lower_limit is a hard UndefKeywordError, not a silent default ----------
let threw = false, var = nothing
    try
        d20_real_setup(W = 2000, δ = 1.0, find_smallest = true, build_screen = false)
    catch e
        threw = e isa UndefKeywordError
        var = threw ? e.var : nothing
    end
    check("d20_real_setup omitting inner_lower_limit -> UndefKeywordError", threw)
    check("...naming :inner_lower_limit specifically", var === :inner_lower_limit)
end

# ---- 3. -50 supplied explicitly is ACCEPTED (diagnostic override, task §5's 4th test case) ----
let threw = false
    try
        ctx_neg50 = d20_real_setup(W = 2000, δ = 1.0, find_smallest = true, build_screen = false,
                                    inner_lower_limit = -50.0)
        threw = !(ctx_neg50.obj.lower_limit == -50.0)
    catch e
        threw = true
    end
    check("d20_real_setup(inner_lower_limit=-50.0) accepted as explicit diagnostic override", !threw)
end

# ---- 4. Real production value: -10.0 propagates from d20_real_setup all the way to the LIVE
#         objective bundle each family's own FG callback reads `.lower_limit` off, AND the
#         dead threshold-abort mechanism is inert (threshold=Inf, not resolve_threshold_for_delta's
#         10.0) -- proving section 4's removal took effect, not just that the line was deleted. ----
const W_SMOKE = 5000
Random.seed!(20260806)
ctx = d20_real_setup(W = W_SMOKE, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
                      build_screen = true, inner_lower_limit = -10.0)
check("d20_real_setup: ctx.obj.lower_limit == -10.0 (production value)", ctx.obj.lower_limit == -10.0)
check("d20_real_setup: ctx.obj.threshold_state.threshold == Inf (dead abort confirmed inert)",
      isinf(ctx.obj.threshold_state.threshold))
check("d20_real_setup: ctx.obj.threshold_state.triggered == false", ctx.obj.threshold_state.triggered == false)

probsL = collect(range(0.0, 1.0; length = 12))[2:end-1]

prep_flex = prepare_production_run(:flexible_cm, "run_cm_upper_checkpointed(LL10-gate)",
    () -> build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probsL,
        threaded_bins = true, inner_fg_backend = :cm_lookup, moment_representation = :operator,
        include_truncated_moment = false))
check("flexible_cm: live bundle lower_limit == -10.0", prep_flex.ctx.obj.lower_limit == -10.0)

prep_frechet = prepare_production_run(:common_frechet, "run_cm_upper_checkpointed(LL10-gate)",
    () -> build_cm_frechet_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probsL,
        cm_hessian_backend = :structured, moment_representation = :operator,
        include_truncated_moment = false))
check("common_frechet: live bundle lower_limit == -10.0", prep_frechet.ctx.obj.lower_limit == -10.0)

prep_meanzc = prepare_production_run(:cm_meanzc, "run_cm_upper_checkpointed(LL10-gate)",
    () -> build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = 1, K_pair = 0,
        contrasts = :anchored, meanzc_basis = :direct, probs = probsL, moment_representation = :operator,
        include_truncated_moment = false))
check("cm_meanzc (CM+ZC): live bundle lower_limit == -10.0", prep_meanzc.ctx.obj.lower_limit == -10.0)

layout_oz = OriginByPowerLayout(ctx.D, 1, 0)
prep_oz = prepare_production_run(:origin_zc, "run_originzc_upper_checkpointed(LL10-gate)",
    () -> build_originzc_production_context(ctx, CS, layout_oz; moment_representation = :operator))
check("origin_zc: live bundle lower_limit == -10.0", prep_oz.ctx.obj.lower_limit == -10.0)

prep_u = prepare_production_run(:unrestricted, "run_polish_checkpointed_unified(LL10-gate)",
    () -> build_unrestricted_operator_ctx(ctx; moment_representation = :operator))
check("unrestricted: live bundle lower_limit == -10.0", prep_u.ctx.obj.lower_limit == -10.0)

for (family, prep) in (("flexible_cm", prep_flex), ("common_frechet", prep_frechet),
                        ("cm_meanzc", prep_meanzc), ("origin_zc", prep_oz), ("unrestricted", prep_u))
    check("$family: live bundle threshold_state inert (Inf)", isinf(prep.ctx.obj.threshold_state.threshold))
end

lp("="^90)
if isempty(FAILURES)
    lp("ALL PASS (", 5 + 2 + 3 + 5 + 5, " checks)")
else
    lp("FAILURES (", length(FAILURES), "): ", FAILURES)
    error("test_lower_limit_production_source_of_truth_2026-08-06.jl: ", length(FAILURES), " check(s) failed")
end
