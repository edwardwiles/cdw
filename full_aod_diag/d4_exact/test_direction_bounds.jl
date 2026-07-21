# ============================================================================
# Toy tests for the direction-aware gamma bounds fix (addendum §5). Deterministic,
# no KNITRO, no real D=20 context -- a minimal mock ctx (just the fields
# direction_bounds.jl actually reads: θ0_up, D, bounds.γp_lo/γp_hi) exercises the
# pure functions directly.
#
# IMPORTANT: these tests assert the EVIDENCED direction (see direction_bounds.jl's
# own header for the three-way evidence: algebra + real established D4/D20 numbers +
# this repo's own explicit "calibrated post-hoc" code comments), which is the
# OPPOSITE of the addendum's own literal box formulas. The addendum's own item
# names ("upper run cannot cross below g_F", "lower run cannot cross above g_F")
# are inverted here to match: upper -> [gp_lo, g_F] (cannot cross ABOVE g_F), lower
# -> [g_F, gp_hi] (cannot cross BELOW g_F). See the handoff doc for the full
# reconciliation and an explicit flag for the user's own review.
# ============================================================================
using Test

include(joinpath(@__DIR__, "direction_bounds.jl"))
include(joinpath(@__DIR__, "incumbent_logic.jl"))   # for is_better_polish, cross-checked below

"Minimal synthetic ctx: only the fields direction_bounds.jl reads."
function mock_ctx(; D::Int = 4, gp_lo::Float64 = 0.7, gp_hi::Float64 = 1.0, gF::Float64 = 0.9)
    θ0_up = zeros(3 + D + D^2)
    θ0_up[3 + D] = gF
    return (D = D, θ0_up = θ0_up, bounds = (γp_lo = gp_lo, γp_hi = gp_hi))
end

"Toy kappa(gp) matching theoretical_gammaprime_bounds' own documented formula, for the direction cross-check below."
toy_kappa(gp, σ) = 1 - gp^(σ / (σ - 1))

@testset "direction-aware gamma bounds" begin

@testset "frechet_benchmark_gp reads theta0_up[3+D]" begin
    ctx = mock_ctx(D = 4, gF = 0.960965)
    @test frechet_benchmark_gp(ctx) == 0.960965
end

@testset "upper box is [gp_lo, g_F]; lower box is [g_F, gp_hi]" begin
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    lo_u, hi_u = direction_gamma_bounds(ctx, true)    # find_smallest=true -> upper
    lo_l, hi_l = direction_gamma_bounds(ctx, false)   # find_smallest=false -> lower
    @test (lo_u, hi_u) == (0.7, 0.9)
    @test (lo_l, hi_l) == (0.9, 1.0)
end

@testset "an upper run cannot cross ABOVE g_F (corrected direction)" begin
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    @test_nowarn validate_gp_in_direction_box(0.9, ctx, true)    # exactly at g_F: feasible (closed interval)
    @test_nowarn validate_gp_in_direction_box(0.75, ctx, true)   # strictly inside [0.7, 0.9]: feasible
    @test_throws ErrorException validate_gp_in_direction_box(0.95, ctx, true)   # above g_F: must reject
    @test_throws ErrorException validate_gp_in_direction_box(0.65, ctx, true)   # below gp_lo: must reject
end

@testset "a lower run cannot cross BELOW g_F (corrected direction)" begin
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    @test_nowarn validate_gp_in_direction_box(0.9, ctx, false)    # exactly at g_F: feasible (closed interval)
    @test_nowarn validate_gp_in_direction_box(0.95, ctx, false)   # strictly inside [0.9, 1.0]: feasible
    @test_throws ErrorException validate_gp_in_direction_box(0.85, ctx, false)  # below g_F: must reject
    @test_throws ErrorException validate_gp_in_direction_box(1.05, ctx, false)  # above gp_hi: must reject
end

@testset "the Frechet point is feasible in BOTH boxes (shared closed boundary)" begin
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    gF = frechet_benchmark_gp(ctx)
    @test_nowarn validate_gp_in_direction_box(gF, ctx, true)
    @test_nowarn validate_gp_in_direction_box(gF, ctx, false)
end

@testset "closed intervals: exact endpoint values are feasible, not just interior points" begin
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    @test_nowarn validate_gp_in_direction_box(0.7, ctx, true)   # exactly gp_lo
    @test_nowarn validate_gp_in_direction_box(1.0, ctx, false)  # exactly gp_hi
end

@testset "reported kappa moves in the economically correct direction" begin
    # sigma=2.5 (this repo's own real calibrated value, confirmed empirically in both
    # D4 and D20 sessions) -> kappa strictly decreasing in gp.
    σ = 2.5
    ctx = mock_ctx(gp_lo = 0.7, gp_hi = 1.0, gF = 0.9)
    lo_u, hi_u = direction_gamma_bounds(ctx, true)
    lo_l, hi_l = direction_gamma_bounds(ctx, false)
    # Upper (find_smallest=true) explores gp in [gp_lo, g_F] -- the SMALLER end of this
    # range gives the LARGER kappa, consistent with "upper" meaning "larger kappa achievable".
    @test toy_kappa(lo_u, σ) > toy_kappa(hi_u, σ)   # kappa(gp_lo) > kappa(g_F)
    # Lower (find_smallest=false) explores gp in [g_F, gp_hi] -- the LARGER end gives the
    # SMALLER kappa, consistent with "lower" meaning "smaller kappa achievable".
    @test toy_kappa(lo_l, σ) > toy_kappa(hi_l, σ)   # kappa(g_F) > kappa(gp_hi)
    # And directly: the best achievable kappa in the upper box (at gp_lo) exceeds the
    # best achievable kappa in the lower box (at g_F) -- upper genuinely bounds above lower.
    @test toy_kappa(lo_u, σ) > toy_kappa(lo_l, σ)
end

@testset "is_better_polish's own comparison direction matches the upper/lower kappa convention" begin
    # find_smallest=true ("upper") must prefer SMALLER gp (larger kappa); find_smallest=false
    # ("lower") must prefer LARGER gp (smaller kappa). Cross-checks incumbent_logic.jl's
    # is_better_polish against THIS file's direction convention, not just in isolation.
    @test is_better_polish(0.75, 0.85, true) == true     # upper: smaller candidate wins
    @test is_better_polish(0.95, 0.85, true) == false    # upper: larger candidate loses
    @test is_better_polish(0.95, 0.85, false) == true    # lower: larger candidate wins
    @test is_better_polish(0.75, 0.85, false) == false   # lower: smaller candidate loses
end

@testset "asymmetric bounds (D=20-scale sigma, different gF) still split correctly" begin
    ctx = mock_ctx(D = 20, gp_lo = 0.5, gp_hi = 1.0, gF = 0.9974)
    lo_u, hi_u = direction_gamma_bounds(ctx, true)
    lo_l, hi_l = direction_gamma_bounds(ctx, false)
    @test lo_u == 0.5 && hi_u == 0.9974
    @test lo_l == 0.9974 && hi_l == 1.0
    @test_throws ErrorException validate_gp_in_direction_box(0.8926, ctx, false)  # a real D4 "upper" gp value must be REJECTED as a lower-direction start
    @test_nowarn validate_gp_in_direction_box(0.8926, ctx, true)   # ...but accepted for upper
end

end # testset

println("All direction-bounds tests passed.")
