# D=4 mean/ZC-specific C+-versus-Reference gate (2026-07-23 CM-C+ integration).
# Per the release addendum: "Do not rerun the generic CM-C+ release battery merely
# because mean/ZC is being added. Run only the mean/ZC-specific C+-versus-Reference
# gates specified in the main prompt and multiple-K addendum." This file is that gate:
# full gradient (economic block + every eta_nu_k coordinate) compared between
# cm_gradient_backend=:reference (cm_meanzc_production_gradient) and :cplus
# (cm_meanzc_production_gradient_cplus), at (K_mean,K_pair) in {(1,0),(1,1),(2,0),(2,2)},
# plus ONE (3,2) structural smoke test (dimension/config-generality check, not a
# genuine additional K-value gate).
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
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
using Test, Printf, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]

const L = 10
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
const CONFIGS = [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc"), (2, 0, "K2_mean_only"), (2, 2, "K2_mean_zc")]

W = size(ctx.U, 1)
pool = build_grad_workspace_pool(W)
ws = build_lfix_factorized_workspace(ctx.D, W)

println("="^100)
println("Mean/ZC-specific gate: full gradient, :reference vs :cplus, incl. every eta_nu_k coordinate")
println("="^100)
@testset "CM+moments(+ZC): :reference vs :cplus full-gradient equivalence" begin
    for (K_mean, K_pair, label) in CONFIGS
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = :direct)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        bins = cm_bin_indices_for(ctx, aug)
        pcx = (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)

        ν0 = nu0vec(K_mean)
        # ONE shared verified base-state solve -- the inner dual solve is backend-independent
        # (only the outer gradient kernel differs), per the base-state addendum.
        base0, verify0 = archC_meanzc_verified_state(x_free_calib, ν0, ctx_cm, cctx)
        @test is_verified_success(verify0)

        g_ref, _ = cm_meanzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
            threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
        g_cplus, _ = cm_meanzc_production_gradient_cplus(x_free_calib, ν0, pcx, ctx, pe, pool, ws; base = base0, verify = verify0,
            threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())

        D2 = ctx.D^2
        @test length(g_ref) == D2 + K_mean
        @test length(g_cplus) == D2 + K_mean

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
        @printf "  %-14s eta_nu block: reference=%s  cplus=%s  max|diff|=%.3e\n" label eta_ref eta_cplus eta_maxdiff
        @test eta_ref ≈ eta_cplus atol = 1e-10
        for k in 1:K_mean
            @test sign(eta_ref[k]) == sign(eta_cplus[k]) || abs(eta_ref[k]) < 1e-8
        end
    end
end
println()

println("="^100)
println("Structural (K_mean,K_pair)=(3,2) smoke test -- detects hidden K=1/K=2 hard-coding, not a full gate")
println("="^100)
@testset "K_mean=3,K_pair=2 structural smoke (dimension/config-generality check)" begin
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 3, K_pair = 2, meanzc_basis = :direct)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    bins = cm_bin_indices_for(ctx, aug)
    pcx = (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)
    ν0 = nu0vec(3)

    base0, verify0 = archC_meanzc_verified_state(x_free_calib, ν0, ctx_cm, cctx)
    @test is_verified_success(verify0)
    g_ref, _ = cm_meanzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0,
        threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    g_cplus, _ = cm_meanzc_production_gradient_cplus(x_free_calib, ν0, pcx, ctx, pe, pool, ws; base = base0, verify = verify0,
        threaded = false, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    D2 = ctx.D^2
    @test length(g_ref) == D2 + 3
    @test length(g_cplus) == D2 + 3
    econ_maxdiff = maximum(abs.(g_ref[1:D2] .- g_cplus[1:D2]))
    eta_maxdiff = maximum(abs.(g_ref[D2+1:end] .- g_cplus[D2+1:end]))
    @printf "  K3_pair2 smoke: econ max|diff|=%.3e  eta_nu(3 levels) max|diff|=%.3e  eta_ref=%s\n" econ_maxdiff eta_maxdiff g_ref[D2+1:end]
    @test econ_maxdiff < 1e-8
    @test eta_maxdiff < 1e-10
end
println()
println("All mean/ZC-specific :reference-vs-:cplus gates passed.")
