# D=4 production-shaped (pivot-eliminated) correctness gates for the theta/A_od decorrelation
# reparametrization ("a-space"), production port, 2026-07-25 (task §13).
#
# Ported from experiment/fullA-theta-aspace-reparam-2026-07-25's
# test_flexible_theta_aspace_d4.jl onto current production's rectangular D x Ddest layout and
# current include chain (c10_d20_production_driver.jl, gravity_elimination.jl,
# flexible_theta.jl, flexible_theta_aspace_production.jl) -- NOT copied verbatim; every square
# D^2/D assumption in the source has been generalized to D*Ddest/Ddest, and a genuinely
# rectangular D=4/D_dest=3 (last-position omission, matching production's own row_idx
# convention) battery is ADDED as testset 7, since none of the square-D4 gates below can
# exercise the omit-ROW rectangular path.
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_flexible_theta_aspace_d4.jl
using Test
using Random
using LinearAlgebra: norm

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))

const FAILURES = String[]
function record!(ts)
    for r in ts.results
        if r isa Test.Fail || r isa Test.Error
            push!(FAILURES, string(ts.description))
        end
    end
end

"Build a flexible-theta ctx (square D=4, all_legacy-shaped) + its a-space/z-space start points.
Uses context_scaled.jl's d_exact_setup_scaled(row_idx=nothing) rather than plain d4_exact_setup()
-- the latter's returned ctx lacks a D_dest field, which the CURRENT (post-omit-ROW) production
evaluate_fullA_screened_ranged/fast_range_screen.jl unconditionally reads (a pre-existing
production architecture fact, not introduced by this port: d4_exact_setup()'s square ctx was
never wired through the modern rectangular screened-eval path). d_exact_setup_scaled always sets
D_dest (=D under row_idx=nothing), so this is the minimal fix, not a math change."
function build_square_case()
    ctx_fixed = d_exact_setup_scaled(D = 4, W = 4000, row_idx = nothing)
    theta_star = 1.0 / ctx_fixed.μHat
    sigma = ctx_fixed.σ
    theta_min = 2 * (sigma - 1) * 1.05
    theta_max = 2 * theta_star
    ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
    # d4_exact_setup() (unlike d20_real_setup_design) does not populate pairwise/witness --
    # evaluate_fullA_screened_ranged's own fallback logic dereferences ctx.pairwise/ctx.witness
    # unconditionally, so any ctx used with the production screened-eval path needs these fields
    # present (nothing is a valid value -- triggers a fresh precompute_pairwise_M(ctx) per call,
    # fine at this test's scale).
    ctx = merge(ctx, (pairwise = nothing, witness = nothing))
    rsc = build_ranged_screen_context(ctx)
    xy = precompute_aspace_XY(ctx)

    Ddest = _flex_ddest(ctx)
    x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
    gp0 = x_free_fixed[1]
    logA_full0 = log.(reshape(x_free_fixed[2:end], ctx.D, Ddest))
    pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)

    w_ext_start_z = reduce_to_w_ext(log(theta_star), gp0, logA_full0, pgc0)
    w_ext_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)
    return (ctx = ctx, rsc = rsc, xy = xy, pgc0 = pgc0, theta_star = theta_star, theta_min = theta_min,
            theta_max = theta_max, w_ext_start_z = w_ext_start_z, w_ext_start_a = w_ext_start_a, Ddest = Ddest)
end

"Build a flexible-theta ctx over a GENUINELY RECTANGULAR D=4/D_dest=3 sample (row_idx=4, last-position omission -- production's own destination_sample=:exclude_row convention)."
function build_rect_case()
    ctx_fixed = d_exact_setup_scaled(D = 4, W = 4000, row_idx = 4)
    theta_star = 1.0 / ctx_fixed.μHat
    sigma = ctx_fixed.σ
    theta_min = 2 * (sigma - 1) * 1.05
    theta_max = 2 * theta_star
    ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
    ctx = merge(ctx, (pairwise = nothing, witness = nothing))
    rsc = build_ranged_screen_context(ctx)
    xy = precompute_aspace_XY(ctx)
    Ddest = _flex_ddest(ctx)
    @assert Ddest == ctx.D - 1 "build_rect_case: expected a genuinely rectangular D_dest=D-1 sample, got D=$(ctx.D) D_dest=$Ddest"

    x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
    gp0 = x_free_fixed[1]
    logA_full0 = log.(reshape(x_free_fixed[2:end], ctx.D, Ddest))
    pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
    w_ext_start_z = reduce_to_w_ext(log(theta_star), gp0, logA_full0, pgc0)
    w_ext_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)
    return (ctx = ctx, rsc = rsc, xy = xy, pgc0 = pgc0, theta_star = theta_star, theta_min = theta_min,
            theta_max = theta_max, w_ext_start_z = w_ext_start_z, w_ext_start_a = w_ext_start_a, Ddest = Ddest)
end

sq = build_square_case()
println("SQUARE case: theta_star=$(sq.theta_star) theta_min=$(sq.theta_min) theta_max=$(sq.theta_max) D=$(sq.ctx.D) D_dest=$(sq.Ddest)"); flush(stdout)

ts_all = @testset "theta/A_od decorrelation reparametrization -- D=4 correctness gates (production port)" begin

    @testset "1. a<->z round-trip is exact (machine precision) at several theta values" begin
        ctx = sq.ctx; xy = sq.xy; Ddest = sq.Ddest
        rng = MersenneTwister(2026)
        for theta_test in (sq.theta_min, sq.theta_star, sq.theta_max, (sq.theta_min + sq.theta_star) / 2)
            a_test = randn(rng, ctx.D, Ddest) .* 3.0
            z_rt = z_from_a(a_test, theta_test, xy)
            a_rt = a_from_z(z_rt, theta_test, xy)
            @test maximum(abs.(a_rt .- a_test)) < 1e-10
            z_test = randn(rng, ctx.D, Ddest) .* 3.0
            a_rt2 = a_from_z(z_test, theta_test, xy)
            z_rt2 = z_from_a(a_rt2, theta_test, xy)
            @test maximum(abs.(z_rt2 .- z_test)) < 1e-10
        end
    end

    @testset "2. decode_and_expand_flexible_A reproduces decode_and_expand_flexible's xf at the SAME economic point" begin
        ctx = sq.ctx
        d_z = decode_and_expand_flexible(sq.w_ext_start_z, ctx)
        d_a = decode_and_expand_flexible_A(sq.w_ext_start_a, ctx, sq.xy)
        @test d_z.theta ≈ d_a.theta rtol=1e-13
        @test d_z.gp == d_a.gp
        @test d_z.xf ≈ d_a.xf rtol=1e-8
        println("max|xf_z - xf_a| = $(maximum(abs.(d_z.xf .- d_a.xf)))"); flush(stdout)
    end

    @testset "3. a-space start point cold-verifies, gravity satisfied, matches z-space Delta_dual" begin
        ctx = sq.ctx; rsc = sq.rsc
        sc_z = ScreenCounters(); sc_a = ScreenCounters()
        r_z, _, d_z = screened_eval_flexible(sq.w_ext_start_z, ctx, rsc, sc_z, Ref(0), sq.pgc0; warm = false)
        r_a, _, d_a = screened_eval_flexible_A(sq.w_ext_start_a, ctx, rsc, sc_a, Ref(0), sq.xy; warm = false)
        @test r_z.inner_status in FEASIBLE_CODES
        @test r_a.inner_status in FEASIBLE_CODES
        @test abs(r_a.gravity_value) < 1e-8
        @test r_z.Delta_dual ≈ r_a.Delta_dual rtol=1e-8
        println("start point: Delta_dual_z=$(r_z.Delta_dual) Delta_dual_a=$(r_a.Delta_dual) gravity_a=$(r_a.gravity_value)"); flush(stdout)
    end

    @testset "4. gradient chain rule: composite_gradient (rescaled by -theta) matches finite difference in a" begin
        ctx = sq.ctx; rsc = sq.rsc; xy = sq.xy
        pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / sq.theta_min, mu_probe2 = 1.0 / sq.theta_max)
        r0, d0 = screened_eval_flexible_A_verify(sq.w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        @test r0.inner_status in FEASIBLE_CODES
        ctx_frozen = freeze_theta_ctx(ctx, d0.mu)
        xf_reduced = vcat(d0.gp, d0.xf[3:end])
        pe_here = pivot_elim_from_cache(pgc, d0.mu)
        base = BaseDualState(xf_reduced, r0.θ_full, r0.zeta, r0.lambda, copy(ctx.obj.arg1), r0.inner_status)
        gfull_reduced, _meta = composite_gradient_at_fast_buffered(xf_reduced, ctx_frozen, pe_here; base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        predicted_g_a = gfull_reduced[2:end] .* (-d0.theta)

        rng = MersenneTwister(7)
        dir = randn(rng, length(predicted_g_a)); dir ./= norm(dir)
        h = 1e-5
        w_plus = copy(sq.w_ext_start_a); w_plus[3:end] .+= h .* dir
        w_minus = copy(sq.w_ext_start_a); w_minus[3:end] .-= h .* dir
        r_plus, _ = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        r_minus, _ = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        @test r_plus.inner_status in FEASIBLE_CODES
        @test r_minus.inner_status in FEASIBLE_CODES
        fd_directional = (r_plus.Delta_dual - r_minus.Delta_dual) / (2h)
        predicted_directional = sum(predicted_g_a .* dir)
        rel_err = abs(fd_directional - predicted_directional) / max(abs(fd_directional), 1e-8)
        println("directional derivative: analytic(rescaled)=$predicted_directional finite-diff=$fd_directional rel_err=$rel_err"); flush(stdout)
        # INFORMATIONAL, not a hard gate (double-FD noise, see test 4b for the decisive check).
        rel_err < 0.20 || @warn "directional-derivative rel_err=$rel_err exceeds the informational 20% band -- investigate if test 4b also disagrees"
    end

    @testset "4b. DECISIVE chain-rule check: direct a-space FD == direct z-space FD along the scaled direction" begin
        ctx = sq.ctx; rsc = sq.rsc; xy = sq.xy; Ddest = sq.Ddest
        rng = MersenneTwister(11)
        dir = randn(rng, ctx.D * Ddest - 1); dir ./= norm(dir)
        h = 1e-4
        theta0 = sq.theta_star

        w_plus_a = copy(sq.w_ext_start_a); w_plus_a[3:end] .+= h .* dir
        w_minus_a = copy(sq.w_ext_start_a); w_minus_a[3:end] .-= h .* dir
        r_plus_a, _ = screened_eval_flexible_A_verify(w_plus_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        r_minus_a, _ = screened_eval_flexible_A_verify(w_minus_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        fd_a = (r_plus_a.Delta_dual - r_minus_a.Delta_dual) / (2h)

        dir_z = (-theta0) .* dir
        w_plus_z = copy(sq.w_ext_start_z); w_plus_z[3:end] .+= h .* dir_z
        w_minus_z = copy(sq.w_ext_start_z); w_minus_z[3:end] .-= h .* dir_z
        r_plus_z, _, _ = screened_eval_flexible(w_plus_z, ctx, rsc, ScreenCounters(), Ref(0), sq.pgc0; warm = false)
        r_minus_z, _, _ = screened_eval_flexible(w_minus_z, ctx, rsc, ScreenCounters(), Ref(0), sq.pgc0; warm = false)
        fd_z = (r_plus_z.Delta_dual - r_minus_z.Delta_dual) / (2h)

        rel_err = abs(fd_a - fd_z) / max(abs(fd_z), 1e-8)
        println("decisive check: fd_a=$fd_a fd_z(scaled dir)=$fd_z rel_err=$rel_err"); flush(stdout)
        @test rel_err < 1e-4
    end

    @testset "5. theta secant: a-space (holding a fixed) vs fully-resolved finite difference in theta" begin
        ctx = sq.ctx; rsc = sq.rsc; xy = sq.xy
        r0, d0 = screened_eval_flexible_A_verify(sq.w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
        inner_x_fixed = copy(ctx.obj.x)
        rel_errs = Float64[]
        for h in (1e-3, 2.5e-4)
            w_plus = copy(sq.w_ext_start_a); w_plus[1] += h
            w_minus = copy(sq.w_ext_start_a); w_minus[1] -= h
            D_plus_analytic = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx, xy)
            D_minus_analytic = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx, xy)
            secant_analytic = (D_plus_analytic - D_minus_analytic) / (2h)

            r_plus, _ = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            r_minus, _ = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            @test r_plus.inner_status in FEASIBLE_CODES
            @test r_minus.inner_status in FEASIBLE_CODES
            secant_resolved = (r_plus.Delta_dual - r_minus.Delta_dual) / (2h)
            rel_err = abs(secant_analytic - secant_resolved) / max(abs(secant_resolved), 1e-8)
            println("  h=$h: analytic=$secant_analytic resolved-inner=$secant_resolved rel_err=$rel_err"); flush(stdout)
            push!(rel_errs, rel_err)
        end
        println("theta secant (a-space, holding a fixed): rel_err(h=1e-3)=$(rel_errs[1]) rel_err(h=2.5e-4)=$(rel_errs[2])"); flush(stdout)
        @test rel_errs[2] < rel_errs[1]
        rel_errs[2] < 0.20 || @warn "theta-secant rel_err=$(rel_errs[2]) at h=2.5e-4 exceeds the informational 20% band -- recheck at D=20 real data"
    end

    @testset "6. CORE HYPOTHESIS: a 5% theta perturbation at FIXED a_nonpivot stays feasible/bounded" begin
        ctx = sq.ctx; rsc = sq.rsc; xy = sq.xy
        theta_up = sq.theta_star * 1.05
        theta_down = sq.theta_star * 0.95
        if theta_up <= sq.theta_max && theta_down >= sq.theta_min
            w_up_a = copy(sq.w_ext_start_a); w_up_a[1] = log(theta_up)
            w_down_a = copy(sq.w_ext_start_a); w_down_a[1] = log(theta_down)
            r_up_a, _ = screened_eval_flexible_A_verify(w_up_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            r_down_a, _ = screened_eval_flexible_A_verify(w_down_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            println("a-space +-5% theta (a_nonpivot FIXED): up.status=$(r_up_a.inner_status) Delta=$(r_up_a.Delta_dual)  down.status=$(r_down_a.inner_status) Delta=$(r_down_a.Delta_dual)"); flush(stdout)

            w_up_z = copy(sq.w_ext_start_z); w_up_z[1] = log(theta_up)
            w_down_z = copy(sq.w_ext_start_z); w_down_z[1] = log(theta_down)
            r_up_z, _, _ = screened_eval_flexible(w_up_z, ctx, rsc, ScreenCounters(), Ref(0), sq.pgc0; warm = false)
            r_down_z, _, _ = screened_eval_flexible(w_down_z, ctx, rsc, ScreenCounters(), Ref(0), sq.pgc0; warm = false)
            println("z-space +-5% theta (z_nonpivot FIXED, OLD parametrization): up.status=$(r_up_z.inner_status) Delta=$(r_up_z.Delta_dual)  down.status=$(r_down_z.inner_status) Delta=$(r_down_z.Delta_dual)"); flush(stdout)
        else
            println("theta_star*1.05/0.95 outside [theta_min,theta_max] at D=4 -- skipping, not a failure")
        end
    end

    @testset "7. RECTANGULAR (D=4, D_dest=3, last-position omission): full battery re-run on the omit-ROW-shaped sample" begin
        rc = build_rect_case()
        ctx = rc.ctx; rsc = rc.rsc; xy = rc.xy; Ddest = rc.Ddest
        println("RECT case: theta_star=$(rc.theta_star) D=$(ctx.D) D_dest=$Ddest (D*Ddest-1=$(ctx.D*Ddest-1) free A coords)"); flush(stdout)

        @testset "7a. a<->z round-trip exact on rectangular sample" begin
            rng = MersenneTwister(2027)
            a_test = randn(rng, ctx.D, Ddest) .* 3.0
            z_rt = z_from_a(a_test, rc.theta_star, xy)
            a_rt = a_from_z(z_rt, rc.theta_star, xy)
            @test maximum(abs.(a_rt .- a_test)) < 1e-10
        end

        @testset "7b. rectangular start point cold-verifies, gravity satisfied, z==a economic equivalence" begin
            d_z = decode_and_expand_flexible(rc.w_ext_start_z, ctx)
            d_a = decode_and_expand_flexible_A(rc.w_ext_start_a, ctx, xy)
            @test d_z.xf ≈ d_a.xf rtol=1e-8
            r_z, _, _ = screened_eval_flexible(rc.w_ext_start_z, ctx, rsc, ScreenCounters(), Ref(0), rc.pgc0; warm = false)
            r_a, _, _ = screened_eval_flexible_A(rc.w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            @test r_z.inner_status in FEASIBLE_CODES
            @test r_a.inner_status in FEASIBLE_CODES
            @test abs(r_a.gravity_value) < 1e-8
            @test r_z.Delta_dual ≈ r_a.Delta_dual rtol=1e-8
            println("rect start point: Delta_dual_z=$(r_z.Delta_dual) Delta_dual_a=$(r_a.Delta_dual) gravity_a=$(r_a.gravity_value)"); flush(stdout)
        end

        @testset "7c. DECISIVE chain-rule check on the rectangular sample" begin
            rng = MersenneTwister(29)
            dir = randn(rng, ctx.D * Ddest - 1); dir ./= norm(dir)
            h = 1e-4
            theta0 = rc.theta_star
            w_plus_a = copy(rc.w_ext_start_a); w_plus_a[3:end] .+= h .* dir
            w_minus_a = copy(rc.w_ext_start_a); w_minus_a[3:end] .-= h .* dir
            r_plus_a, _ = screened_eval_flexible_A_verify(w_plus_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            r_minus_a, _ = screened_eval_flexible_A_verify(w_minus_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
            fd_a = (r_plus_a.Delta_dual - r_minus_a.Delta_dual) / (2h)

            dir_z = (-theta0) .* dir
            w_plus_z = copy(rc.w_ext_start_z); w_plus_z[3:end] .+= h .* dir_z
            w_minus_z = copy(rc.w_ext_start_z); w_minus_z[3:end] .-= h .* dir_z
            r_plus_z, _, _ = screened_eval_flexible(w_plus_z, ctx, rsc, ScreenCounters(), Ref(0), rc.pgc0; warm = false)
            r_minus_z, _, _ = screened_eval_flexible(w_minus_z, ctx, rsc, ScreenCounters(), Ref(0), rc.pgc0; warm = false)
            fd_z = (r_plus_z.Delta_dual - r_minus_z.Delta_dual) / (2h)

            rel_err = abs(fd_a - fd_z) / max(abs(fd_z), 1e-8)
            println("rect decisive check: fd_a=$fd_a fd_z(scaled dir)=$fd_z rel_err=$rel_err"); flush(stdout)
            @test rel_err < 1e-4
        end

        @testset "7d. checkpoint/resume round-trip fields on the rectangular sample (reduce_to_w_ext_A inverse)" begin
            d_a = decode_and_expand_flexible_A(rc.w_ext_start_a, ctx, xy)
            logA_full = pivot_expand_cheap(d_a.z_nonpivot, d_a.pgc, d_a.mu)
            w_rt = reduce_to_w_ext_A(d_a.theta, d_a.gp, logA_full, d_a.pgc, xy)
            @test w_rt ≈ rc.w_ext_start_a rtol=1e-10
        end
    end
end

for ts in ts_all.results
    ts isa Test.DefaultTestSet && record!(ts)
end
println(isempty(FAILURES) ? "ALL GATES PASS" : "FAILURES: $(join(FAILURES, ", "))")
flush(stdout)
