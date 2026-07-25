# D=4 correctness gates for the addendum's unified outer-coordinate-layout architecture
# (task addendum §5): z<->a round trips, identical economic A, identical gravity residual,
# identical Delta*, C+ gradient transformed by (-theta*), fixed+legacy_z reproduces the
# EXISTING untouched production driver's own numbers (cross-check against x_free_from_w/
# build_pivot_elimination directly, not just internal self-consistency).
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_unified_layout_d4.jl
using Test, Random
using LinearAlgebra: norm

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))

const FAILURES = String[]
function record!(ts)
    for r in ts.results
        (r isa Test.Fail || r isa Test.Error) && push!(FAILURES, string(ts.description))
    end
end

# Rectangular D=4/D_dest=3 sample, matching production's own row_idx convention.
# d_exact_setup_scaled() (unlike d20_real_setup_design) does not populate pairwise/witness --
# evaluate_fullA_screened_ranged's own fallback logic dereferences ctx.pairwise/ctx.witness
# unconditionally (c10_d20_production_driver.jl:407's screened_eval passes them through to it).
# nothing is a valid value here (triggers a fresh precompute_pairwise_M(ctx) per call, fine at
# this test's D=4 scale) -- same fix already applied in test_flexible_theta_aspace_d4.jl,
# omitted by mistake when this file was written; not a production bug.
ctx = merge(d_exact_setup_scaled(D = 4, W = 4000, row_idx = 4), (pairwise = nothing, witness = nothing))
theta_star = 1.0 / ctx.μHat
sigma = ctx.σ
D = ctx.D; Ddest = ctx.D_dest
rsc = build_ranged_screen_context(ctx)
xy = precompute_aspace_XY(ctx)

x_free_calib = CS.pack_free(ctx.θ0_up, ctx.m)
gp0 = x_free_calib[1]
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))

pe_legacy = build_pivot_elimination(ctx)   # EXISTING, unmodified production pivot
pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)

println("D=4 unified layout gates: theta_star=$theta_star D=$D Ddest=$Ddest"); flush(stdout)

ts_all = @testset "Addendum: unified outer-coordinate layout, D=4 gates" begin

    @testset "1. fixed+legacy_z+raw reproduces EXISTING production driver's own xf exactly" begin
        layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
        w_legacy = vcat(gp0, pivot_reduce(logA_full0, pe_legacy))
        d = decode_outer_unified(w_legacy, ctx, layout, pgc, xy)
        xf_ref = x_free_from_w(w_legacy, pe_legacy)   # existing, unmodified production function
        @test d.xf ≈ xf_ref rtol=1e-8
        println("  max|xf_unified - xf_legacy| = $(maximum(abs.(d.xf .- xf_ref)))")

        r_unified, _ = screened_eval(d.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        r_legacy, _ = screened_eval(xf_ref, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        @test r_unified.Delta_dual ≈ r_legacy.Delta_dual rtol=1e-8
        @test r_unified.inner_status in FEASIBLE_CODES
        println("  Delta_dual: unified=$(r_unified.Delta_dual) legacy=$(r_legacy.Delta_dual)")
    end

    @testset "2. fixed+powered_aspace+raw reproduces the SAME economic point as fixed+legacy_z" begin
        layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
        layout_z = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
        w_a = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_a)
        w_z = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_z)
        d_a = decode_outer_unified(w_a, ctx, layout_a, pgc, xy)
        d_z = decode_outer_unified(w_z, ctx, layout_z, pgc, xy)
        @test d_a.xf ≈ d_z.xf rtol=1e-8
        println("  max|xf_a - xf_z| = $(maximum(abs.(d_a.xf .- d_z.xf)))")

        r_a, _ = screened_eval(d_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        r_z, _ = screened_eval(d_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        @test r_a.inner_status in FEASIBLE_CODES
        @test r_z.inner_status in FEASIBLE_CODES
        @test abs(r_a.gravity_value) < 1e-8
        @test r_a.Delta_dual ≈ r_z.Delta_dual rtol=1e-8
        println("  Delta_dual: a-space=$(r_a.Delta_dual) z-space=$(r_z.Delta_dual) gravity_a=$(r_a.gravity_value)")
    end

    @testset "3. z<->a round trip exact at fixed theta_star" begin
        rng = MersenneTwister(2026)
        a_test = randn(rng, D, Ddest) .* 3.0
        z_rt = z_from_a(a_test, theta_star, xy)
        a_rt = a_from_z(z_rt, theta_star, xy)
        @test maximum(abs.(a_rt .- a_test)) < 1e-10
    end

    @testset "4. DECISIVE chain-rule check: direct a-space FD == direct z-space FD (fixed theta, no composite_gradient dependency)" begin
        layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
        layout_z = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)
        w_a0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_a)
        w_z0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_z)
        rng = MersenneTwister(11)
        dir = randn(rng, D * Ddest - 1); dir ./= norm(dir)
        h = 1e-4
        w_plus_a = copy(w_a0); w_plus_a[2:end] .+= h .* dir
        w_minus_a = copy(w_a0); w_minus_a[2:end] .-= h .* dir
        d_plus_a = decode_outer_unified(w_plus_a, ctx, layout_a, pgc, xy)
        d_minus_a = decode_outer_unified(w_minus_a, ctx, layout_a, pgc, xy)
        r_plus_a, _ = screened_eval(d_plus_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        r_minus_a, _ = screened_eval(d_minus_a.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        fd_a = (r_plus_a.Delta_dual - r_minus_a.Delta_dual) / (2h)

        dir_z = (-theta_star) .* dir
        w_plus_z = copy(w_z0); w_plus_z[2:end] .+= h .* dir_z
        w_minus_z = copy(w_z0); w_minus_z[2:end] .-= h .* dir_z
        d_plus_z = decode_outer_unified(w_plus_z, ctx, layout_z, pgc, xy)
        d_minus_z = decode_outer_unified(w_minus_z, ctx, layout_z, pgc, xy)
        r_plus_z, _ = screened_eval(d_plus_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        r_minus_z, _ = screened_eval(d_minus_z.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false)
        fd_z = (r_plus_z.Delta_dual - r_minus_z.Delta_dual) / (2h)

        rel_err = abs(fd_a - fd_z) / max(abs(fd_z), 1e-8)
        println("  decisive fixed-theta chain-rule check: fd_a=$fd_a fd_z(scaled dir)=$fd_z rel_err=$rel_err")
        @test rel_err < 1e-4
    end

    @testset "5. gradient_transform_unified matches manual (-theta) rescale" begin
        layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
        gfull_z = vcat(0.5, randn(MersenneTwister(3), D * Ddest - 1))
        g_transformed = gradient_transform_unified(gfull_z, theta_star, gp0, layout_a, nothing)
        @test g_transformed[1] == gfull_z[1]   # raw gp: untouched
        @test g_transformed[2:end] ≈ gfull_z[2:end] .* (-theta_star)
    end

    @testset "6. gp_coordinate_mode=:scaled_log encode/decode round trip + gradient chain rule" begin
        gs = GpScale(gp0, 2.5)
        layout_scaled = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :scaled_log)
        u_g = encode_gp(gp0, layout_scaled, gs)
        @test abs(u_g) < 1e-10   # gp==gp_star -> u_g==0 exactly
        gp_rt = decode_gp(u_g, layout_scaled, gs)
        @test gp_rt ≈ gp0 rtol=1e-12
        # perturb and round-trip
        u_g2 = 0.05
        gp2 = decode_gp(u_g2, layout_scaled, gs)
        u_g2_rt = encode_gp(gp2, layout_scaled, gs)
        @test u_g2_rt ≈ u_g2 rtol=1e-10
        # gradient chain rule: d(Delta)/d(u_g) = d(Delta)/d(gp) * gp/s_g
        dDelta_dgp = 0.37
        dDelta_dug = rescale_gp_gradient(dDelta_dgp, gp0, layout_scaled, gs)
        @test dDelta_dug ≈ dDelta_dgp * gp0 / gs.s_g rtol=1e-12
    end

    @testset "7. reduce_to_w_unified / decode_outer_unified round trip, both A-coordinate modes" begin
        for (lbl, layout) in (("legacy_z", make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :legacy_z, gp_coordinate_mode = :raw)),
                               ("powered_aspace", make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)))
            w0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)
            d0 = decode_outer_unified(w0, ctx, layout, pgc, xy)
            logA_recon = pivot_expand_cheap(d0.z_nonpivot, pgc, d0.mu)
            w_rt = reduce_to_w_unified(d0.theta, d0.gp, logA_recon, pgc, xy, layout)
            @test w_rt ≈ w0 rtol=1e-10
            println("  [$lbl] round-trip max|diff| = $(maximum(abs.(w_rt .- w0)))")
        end
    end

    @testset "8. cache A/B/A (fixed+powered_aspace): identical point -> exact hit" begin
        layout_a = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
        w_a0 = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout_a)
        exact_cache = SafeExactCache()
        d0 = decode_outer_unified(w_a0, ctx, layout_a, pgc, xy)
        r1, _ = screened_eval(d0.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = false, exact_cache = exact_cache)
        size1 = length(exact_cache)
        r2, _ = screened_eval(d0.xf, ctx, rsc, ScreenCounters(), Ref(0); warm = true, exact_cache = exact_cache)
        size2 = length(exact_cache)
        @test size2 == size1
        @test r1.Delta_dual === r2.Delta_dual
        @test get(r2, :cache_hit, false) == true
        println("  cache sizes: $size1 -> $size2 (exact hit confirmed)")
    end
end

for ts in ts_all.results
    ts isa Test.DefaultTestSet && record!(ts)
end
println(isempty(FAILURES) ? "ALL UNIFIED-LAYOUT D=4 GATES PASS" : "FAILURES: $(join(FAILURES, ", "))")
flush(stdout)
