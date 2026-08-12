# ================================================================================================
# "L = 50 means 50 equally sized buckets" -- gate the USER-FACING PROMISE on every route that can
# state an L (2026-08-12, user-directed).
#
# The point of this file is that the promise is about a WORD, not about one function. It was
# possible, before this task, for `L = 50` to mean three different restrictions depending on which
# door you came in through (the campaign's injected dyadic grid; the driver's bare
# `range(1/L,(L-1)/L,length=L)` default; family #7's own `k/G`). So each route gets its own check
# here, and each check re-derives the BUCKET MASSES from the cutpoints rather than asserting the
# cutpoint formula -- a test that only compared against `(1:(L-1))./L` would pass just as happily if
# "equal mass" were the wrong requirement.
#
# No KNITRO, no D=20 context: these are grid-resolution assertions, and keeping them cheap is what
# lets them be run on every touch of the CM config surface.
# ================================================================================================

const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          # MERGE 2026-08-12: the CROSS families (production, 2026-08-09). REQUIRED here, not optional:
          # the merged `multistart_seed_generator.jl` hard-checks `OriginByPowerCrossLayout` /
          # `SharedByPowerCrossLayout` / both cross context builders in its dependency guard, so this
          # orchestrator dies at include time without them -- caught by running the merged
          # orchestrator, not by anything that merely compiles. Order copied from
          # `test_cross_seed_family_dispatch_2026-08-09.jl`, which is the file that exercises them.
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "knitro_status.jl", "knitro_version_check.jl",
          "c10_d20_production_driver.jl", "flexible_theta.jl", "flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl", "c10_d20_production_driver_unified.jl",
          "multistart_seed_generator.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "cm_pairwise_quantile_config.jl"]
    include(joinpath(_D4E, f))
end
using Test, TOML

"Bucket masses induced by a cutpoint vector on (0,1): the thing 'equally sized buckets' is ABOUT."
bucket_masses(p::AbstractVector{Float64}) = [p[1]; diff(p); 1 - p[end]]

"Assert `p` cuts (0,1) into exactly `L` buckets of mass 1/L each."
function assert_L_equal_buckets(p::AbstractVector{Float64}, L::Int, who::AbstractString)
    m = bucket_masses(p)
    @test length(p) == L - 1
    @test length(m) == L
    @test all(x -> isapprox(x, 1 / L; atol = 1e-12), m)
    @test issorted(p) && 0.0 < p[1] && p[end] < 1.0
    println("  OK  ", rpad(who, 52), L, " buckets, every mass ", 1 / L)
end

@testset "L means L equal-mass buckets, on every route that can state an L" begin

    @testset "route 1: the grid function itself" begin
        for L in (2, 5, 10, 20, 37, 50, 100)
            assert_L_equal_buckets(cm_equal_mass_probs(L), L, "cm_equal_mass_probs($L)")
        end
    end

    @testset "route 2: resolve_cm_probs (seed specs + orchestrator both go through it)" begin
        for L in (10, 20, 50)
            assert_L_equal_buckets(resolve_cm_probs(L), L, "resolve_cm_probs($L)")
        end
    end

    @testset "route 3: the FamilySeedSpec constructors the paper preset uses" begin
        for (name, spec) in (("cm_only_family_spec", cm_only_family_spec(:CM; L = 50)),
                             ("common_frechet_family_spec", common_frechet_family_spec(:CF; L = 50)),
                             ("cm_zc_family_spec", cm_zc_family_spec(:CMZC; K_mean = 3, K_pair = 3, L = 50)))
            assert_L_equal_buckets(spec.probs, spec.L, "$name(L=50).probs")
            # and the LEVEL count the CM builders are handed is derived from the grid, not from L
            @test cm_n_levels(spec) == length(spec.probs) == spec.L - 1
        end
    end

    @testset "route 4: paper_upper_v1.toml's own L, through fam_kwargs()'s translation" begin
        # Re-implements fam_kwargs()'s two lines rather than importing the orchestrator (which needs
        # 4 CLI args and a campaign root). If those two lines change, this test is SUPPOSED to be
        # updated with them -- it is pinning the translation, not testing that the file exists.
        manifest = TOML.parsefile(joinpath(_D4E, "..", "..", "protocols", "paper_upper_v1.toml"))
        for fam in ("COMMON_MARGINALS", "COMMON_FRECHET", "CM_PLUS_ZC")
            kwargs = manifest["families"][fam]["kwargs"]
            @test !haskey(kwargs, "probs")          # if this ever fails the injection stopped applying
            L_manifest = kwargs["L"]
            @test L_manifest == 50
            probs = resolve_cm_probs(L_manifest)
            L_driver = length(probs)                 # fam_kwargs(): kw = merge(kw, (L = length(probs), probs = probs))
            assert_L_equal_buckets(probs, L_manifest, "paper_upper_v1 $fam L=$L_manifest")
            @test L_driver == 49
        end
    end

    @testset "route 5: run_cm_upper_checkpointed with a BARE L and no probs" begin
        # The gap this task closed. Before it, this route fell through to
        # precalc_common_marginals_cdf's `range(1/L,(L-1)/L,length=L)` default -- 51 buckets at
        # L=50, 49 of mass 0.0195918 and 2 of mass 0.02. We assert the driver's resolution WITHOUT
        # running a solve, by pinning the two lines the driver executes when probs === nothing.
        for L in (10, 50)
            probs = cm_equal_mass_probs(L)           # driver: probs = cm_equal_mass_probs(L)
            assert_L_equal_buckets(probs, L, "run_cm_upper_checkpointed(L=$L, probs=nothing)")
            @test length(probs) == L - 1             # driver: L = length(probs)
        end
        # NEGATIVE CONTROL: the old fall-through really was not equal-mass, so this gate is not
        # asserting something that was already true.
        old_default = collect(range(1 / 50, 49 / 50, length = 50))
        m_old = bucket_masses(old_default)
        @test length(m_old) == 51
        @test !all(x -> isapprox(x, 0.02; atol = 1e-10), m_old)
        @test sort(unique(round.(m_old, digits = 8))) == [0.01959184, 0.02]
    end

    @testset "route 6: family #7's own CM grid (separate function, must agree numerically)" begin
        for G in (10, 50)
            assert_L_equal_buckets(cm_pq_probs_grid(G), G, "cm_pq_probs_grid($G)")
            @test cm_pq_probs_grid(G) == cm_equal_mass_probs(G)
        end
    end

    @testset "the two grids this replaced are NOT equal-mass (so the promise has content)" begin
        dyadic = nested_grid_sequence([10, 20, 50])[50]
        @test length(bucket_masses(dyadic)) == 51
        @test sort(unique(round.(bucket_masses(dyadic), digits = 10))) == [0.015625, 0.03125]
        # cm_equal_grid_probs is DELIBERATELY unchanged (see its docstring) -- pin that, so a future
        # session changing it has to come here and decide on purpose.
        @test cm_equal_grid_probs(50) == collect(range(1 / 50, 49 / 50, length = 50))
        @test length(bucket_masses(cm_equal_grid_probs(50))) == 51
    end
end
