# D=4 origin-specific-ZC C+-versus-Reference gate. Mirrors
# test_cm_meanzc_cplus_equivalence.jl's structure exactly, for the no-CM
# origin-by-power arm: full gradient (economic block + every eta_{o,k}
# coordinate) compared between cm_gradient_backend=:reference
# (cm_originzc_production_gradient) and :cplus
# (cm_originzc_production_gradient_cplus), at (K_mean,K_pair) in
# {(1,0),(1,1),(2,0),(2,2)}, plus ONE (3,2) structural smoke test.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
using Test, Printf, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D

nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)
const CONFIGS = [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc"), (2, 0, "K2_mean_only"), (2, 2, "K2_mean_zc")]

W = size(ctx.U, 1)
pool = build_grad_workspace_pool(W)
ws = build_lfix_factorized_workspace(ctx.D, W)

println("="^100)
println("Origin-ZC gate: full gradient, :reference vs :cplus, incl. every eta_{o,k} coordinate")
println("="^100)
@testset "origin-ZC: :reference vs :cplus full-gradient equivalence" begin
    for (K_mean, K_pair, label) in CONFIGS
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        pcx = build_originzc_production_context(ctx, CS, layout)

        ν0 = nu0_origin(K_mean, D)
        base0, verify0 = archOZ_verified_state(x_free_calib, ν0, pcx.ctx_cm)
        @test is_verified_success(verify0)

        g_ref, _ = cm_originzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
            threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        g_cplus, _ = cm_originzc_production_gradient_cplus(x_free_calib, ν0, pcx, ctx, pe, pool, ws; base = base0, verify = verify0,
            threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

        D2 = ctx.D^2
        n_eta_here = n_eta(layout)
        @test length(g_ref) == D2 + n_eta_here
        @test length(g_cplus) == D2 + n_eta_here

        econ_ref = g_ref[1:D2]; econ_cplus = g_cplus[1:D2]
        maxdiff = maximum(abs.(econ_ref .- econ_cplus))
        relmax = maximum(abs.(econ_ref .- econ_cplus) ./ max.(abs.(econ_ref), 1e-8))
        cossim = dot(econ_ref, econ_cplus) / (norm(econ_ref) * norm(econ_cplus))
        nsign = sum((econ_ref .> 1e-8) .& (econ_cplus .< -1e-8)) + sum((econ_ref .< -1e-8) .& (econ_cplus .> 1e-8))
        @printf "  %-14s econ block: max|diff|=%.3e  max relative=%.3e  cosine=%.12f  sign_mismatches=%d\n" label maxdiff relmax cossim nsign
        @test nsign == 0
        @test cossim > 1 - 1e-8
        @test maxdiff < 1e-8

        eta_ref = g_ref[D2+1:end]; eta_cplus = g_cplus[D2+1:end]
        eta_maxdiff = maximum(abs.(eta_ref .- eta_cplus))
        @printf "  %-14s eta block (n=%d): max|diff|=%.3e\n" label n_eta_here eta_maxdiff
        @test eta_ref ≈ eta_cplus atol = 1e-10
        for j in 1:n_eta_here
            @test sign(eta_ref[j]) == sign(eta_cplus[j]) || abs(eta_ref[j]) < 1e-8
        end
    end
end
println()

println("="^100)
println("Structural (K_mean,K_pair)=(3,2) smoke test -- detects hidden K=1/K=2 hard-coding")
println("="^100)
@testset "K_mean=3,K_pair=2 structural smoke (dimension/config-generality check)" begin
    layout = OriginByPowerLayout(D, 3, 2)
    pcx = build_originzc_production_context(ctx, CS, layout)
    ν0 = nu0_origin(3, D)
    base0, verify0 = archOZ_verified_state(x_free_calib, ν0, pcx.ctx_cm)
    @test is_verified_success(verify0)
    g_ref, _ = cm_originzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
        threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    g_cplus, _ = cm_originzc_production_gradient_cplus(x_free_calib, ν0, pcx, ctx, pe, pool, ws; base = base0, verify = verify0,
        threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    @test length(g_ref) == ctx.D^2 + n_eta(layout)
    @test g_ref[ctx.D^2+1:end] ≈ g_cplus[ctx.D^2+1:end] atol = 1e-10
    @printf "  K=3,K_pair=2  n_eta=%d  gradient lengths match, eta blocks agree\n" n_eta(layout)
end

println("\nALL ORIGIN-ZC C+ EQUIVALENCE GATES DONE.")
