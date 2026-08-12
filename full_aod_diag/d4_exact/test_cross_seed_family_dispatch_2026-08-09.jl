# Real-context dispatch gate for the CROSS kinds in the reproducible multistart-seed campaign
# (2026-08-09). `test_cmzc_cross_wiring_2026-08-09.jl` covers the spec/descriptor/digest layer
# (solver-free); THIS covers the part that layer cannot: that `build_family` and `evaluate_family`
# actually dispatch `:origin_zc_cross` / `:cm_zc_cross` to the right layout + context builder, and
# that a real verified evaluation comes back through the SAME screened evaluators the diagonal kinds
# use.
#
# Run at K=2/2 rather than the preset's 3/3 purely for wall-clock (the cross inner solve is ~55s at
# K=2/2 vs ~293s at K=3/3, D20/W=100k -- see bench_cmzc_cross_ncore_ext_2026-08-09.jl). The dispatch
# being tested is K-independent.
#
# Usage: julia --project=. -t 10 .../test_cross_seed_family_dispatch_2026-08-09.jl [K] [W] [--build-only]
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
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "fast_range_screen.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(_D4E, f))
end
using Printf

const KK = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const W  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000
const BUILD_ONLY = "--build-only" in ARGS

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name); flush(stdout)
end

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
D = ctx.D
npair = div(D * (D - 1), 2)
x_free = ctx.θ0_up[ctx.free_idx]
println("ctx built: D=$D W=$W sigma=$(ctx.σ) bi=$(ctx.bi)")

specs = [origin_zc_cross_family_spec(:OZC_CROSS_TEST; K_mean = KK, K_pair = KK),
         cm_zc_cross_family_spec(:CMZC_CROSS_TEST; K_mean = KK, K_pair = KK, L = 50)]

for spec in specs
    println("\n==== build_family dispatch: $(spec.id) (kind=$(spec.kind)) ====")
    fb = build_family(ctx, spec)
    check("$(spec.id): is_cross_seed_kind", is_cross_seed_kind(spec.kind))
    if spec.kind == :origin_zc_cross
        check("$(spec.id): layout is OriginByPowerCrossLayout", fb.layout isa OriginByPowerCrossLayout)
        check("$(spec.id): n_eta == K_mean*D", n_eta(fb.layout) == KK * D)
        check("$(spec.id): aug.n_pair == K_pair^2*npair", fb.pcx.aug.n_pair == KK^2 * npair)
    else
        check("$(spec.id): layout is SharedByPowerCrossLayout", fb.layout isa SharedByPowerCrossLayout)
        check("$(spec.id): n_eta == K_mean", n_eta(fb.layout) == KK)
        check("$(spec.id): aug.n_pair == K_pair^2*npair", fb.pcx.aug.n_pair == KK^2 * npair)
        check("$(spec.id): cctx picked up the CROSS layout", fb.pcx.cctx.hzz_zc_layout isa SharedByPowerCrossLayout)
    end
    # Variant D (profiled_level_for) applies identically to both -- aml.base must be the CROSS layout,
    # or d_delta_dual_d_eta_active_and_nustar_*_cross's own type assertion would reject it later.
    if fb.aml !== nothing
        check("$(spec.id): aml.base is the SAME cross layout object", fb.aml.base === fb.layout)
        check("$(spec.id): aml active at kstar=sigma-1=$(Int(ctx.σ)-1)", fb.aml.active && fb.aml.kstar == Int(ctx.σ) - 1)
    else
        println("    (no Variant D at this spec: profiled_level_for returned nothing)")
    end
    check("$(spec.id): derived_focal_nu dispatches (not nothing when aml active)",
          fb.aml === nothing || derived_focal_nu(ctx, fb, x_free) !== nothing)

    if !BUILD_ONLY
        println("    running evaluate_family (real companion-LFD solve + real cross solve) ...")
        t0 = time()
        r = evaluate_family(ctx, fb, x_free; eval_id = 1)
        @printf("    -> Delta_star=%.10e  verified=%s  inner_status=%d  class=%s  wall=%.1fs\n",
                r.Delta_star, string(r.verified), r.inner_status, string(r.verification_class), time() - t0)
        flush(stdout)
        check("$(spec.id): evaluate_family returns the cross kind", r.kind == spec.kind)
        check("$(spec.id): Delta_star is finite", isfinite(r.Delta_star))
        check("$(spec.id): inner solve reached an accepted status", r.inner_status in (0, -100, -101, -103))
        check("$(spec.id): nu_values length == n_eta_active", length(r.nu_values) == (fb.aml === nothing ? n_eta(fb.layout) : fb.aml.n_eta_active))
    end
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
