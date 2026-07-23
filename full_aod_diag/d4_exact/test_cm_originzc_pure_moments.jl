# D=4 pure/inner gates for the origin-specific pairwise-zero-covariance
# restriction (no common marginals). See docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md.
# Needs a real KNITRO license -- this file actually solves the inner CC dual problem.
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
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
using Test, Printf, LinearAlgebra, Random, Statistics, NLsolve

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D

# Draws are Exp(1) (genExpRands!): E[z^k] = k!.
nu0_shared(K::Int) = [Float64(factorial(k)) for k in 1:K]
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

println("="^100)
println("1) target-layout dimension/indexing sanity")
println("="^100)
@testset "target layout dims" begin
    for (K_mean, K_pair) in [(1, 1), (2, 2)]
        shared = SharedByPowerLayout(K_mean, K_pair)
        orig = OriginByPowerLayout(D, K_mean, K_pair)
        @test n_eta(shared) == K_mean
        @test n_eta(orig) == K_mean * D
        @test n_originzc_moments(D, K_mean, K_pair) == n_meanzc_moments(D, K_mean, K_pair)
        # unordered-pair count
        npair = D * (D - 1) ÷ 2
        @test length(packed_pair_index(D)) == npair
    end
end

println("="^100)
println("2) origin-specific moment construction vs dense reference")
println("="^100)
@testset "moment column construction matches dense reference" begin
    for (K_mean, K_pair) in [(1, 1), (2, 2)]
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        aug = build_originzc_augmented_obj(ctx, CS, layout)
        @test aug.n_mean == K_mean * D
        @test aug.n_pair == K_pair * (D * (D - 1) ÷ 2)

        η0 = log.(nu0_origin(K_mean, D))
        θ_ext = vcat(CS.reconstruct_full(x_free_calib, ctx.m), exp.(η0))
        n = aug.obj_cm.outer_constr_index
        Gfull = Matrix{Float64}(undef, size(ctx.U, 1), aug.obj_cm.d)
        Kbuf = zeros(size(ctx.U, 1))
        aug.obj_cm.moments!(Kbuf, Gfull, θ_ext, ctx.U, aug.obj_cm)

        pregrav = aug.ncore_econ - 1
        for k in 1:K_mean
            cols = pregrav+(k-1)*D+1 : pregrav+k*D
            νo_k = nu0_origin(K_mean, D)[(k-1)*D+1:k*D]
            dense_ref = ctx.U .^ k .- νo_k'
            @test maximum(abs.(Gfull[:, cols] .- dense_ref)) < 1e-10
        end
        mean_end = pregrav + aug.n_mean
        npair = D * (D - 1) ÷ 2
        pairs = packed_pair_index(D)
        for k in 1:K_pair
            cols = mean_end+(k-1)*npair+1 : mean_end+k*npair
            νfull = nu0_origin(K_mean, D)
            dense_ref = Matrix{Float64}(undef, size(ctx.U, 1), npair)
            for (j, (o, p)) in enumerate(pairs)
                dense_ref[:, j] .= (ctx.U[:, o] .^ k) .* (ctx.U[:, p] .^ k) .- νfull[(k-1)*D+o] * νfull[(k-1)*D+p]
            end
            @test maximum(abs.(Gfull[:, cols] .- dense_ref)) < 1e-10
        end
        @printf "  K_mean=%d K_pair=%d  n_mean=%d n_pair=%d  moment columns match dense reference\n" K_mean K_pair aug.n_mean aug.n_pair
    end
end

println("="^100)
println("3) inner solve + canonical Delta_dual (K=1, K=2)")
println("="^100)
results = Dict{Tuple{Int,Int},Any}()
@testset "inner solve succeeds, Delta_dual finite and sane" begin
    for (K_mean, K_pair) in [(1, 0), (1, 1), (2, 0), (2, 2)]
        layout = OriginByPowerLayout(D, K_mean, K_pair)
        pcx = build_originzc_production_context(ctx, CS, layout)
        νfull0 = nu0_origin(K_mean, D)
        K, base, verify = cm_originzc_production_value_verified(x_free_calib, νfull0, pcx)
        @test verify.inner_status in (0, -100, -101, -103)
        @test isfinite(verify.Delta_dual)
        @test verify.Delta_dual >= -1e-8
        results[(K_mean, K_pair)] = (pcx = pcx, base = base, verify = verify, νfull0 = νfull0)
        @printf "  K_mean=%d K_pair=%d  Delta_dual=%.6f  gap=%.2e  kkt_resid=%.2e  status=%d\n" K_mean K_pair verify.Delta_dual verify.primal_dual_gap verify.max_abs_moment_kkt_resid verify.inner_status
    end
end

println("="^100)
println("4) mean-only-arm implementation-equivalence: profiled origin-moments-only ~ unrestricted")
println("="^100)
@testset "mean-only arm reproduces unrestricted Delta (profiled over nu)" begin
    # unrestricted baseline (K_mean=0 case, i.e. plain ctx.obj, no meanzc wrapper at all)
    Kb, baseB, verifyB = cm_production_value_verified_unrestricted = let
        obj0 = ctx.obj
        θ_econ0 = CS.reconstruct_full(x_free_calib, ctx.m)
        Kx, x, nStatus, _, _ = inner_loop_internal_archgeneric(obj0, θ_econ0; hess_cb_builder = archA_hess_cb_builder)
        nStatus in (0, -100, -101, -103) || error("unrestricted baseline solve failed nStatus=$nStatus")
        ζstar = x[1]; λstar = collect(x[2:end])
        G = CS.select_G_from_H(obj0, obj0.H)
        ncon = obj0.d - obj0.outer_constr_index + 2
        cbuf = zeros(ncon)
        obj0(x, constr = @view(cbuf[1:ncon]))
        Δ = cbuf[1] / 1e10
        (Kx, nothing, (Delta_dual = Δ,))
    end
    Δ_unrestricted = verifyB.Delta_dual

    r = results[(1, 0)]
    Δ_at_draw_moments = r.verify.Delta_dual
    @printf "  Delta_unrestricted=%.8f   Delta_origin_mean_only(at draw moments, unprofiled)=%.8f   diff=%.3e\n" Δ_unrestricted Δ_at_draw_moments (Δ_at_draw_moments - Δ_unrestricted)
    # Evaluating AT the draws' own raw moments does NOT make the mean-defining constraint
    # non-binding (it forces the REWEIGHTED distribution to match that specific numeric value,
    # which is generally a real restriction relative to the unconstrained reweighting) -- the
    # task brief's equivalence claim is about the PROFILED (Delta-minimizing) nu, not an
    # arbitrary one. Profile via NLsolve root-finding on the analytic eta gradient (first-order
    # condition for the eta-only minimization at fixed economic outer point) -- canonical solver,
    # not a hand-rolled search.
    pcx = r.pcx
    function grad_resid!(F, η)
        νfull = exp.(η)
        base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
        F .= d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull; mean_m = verify.m_mean)
        return nothing
    end
    η0 = log.(r.νfull0)
    sol = nlsolve(grad_resid!, η0; autodiff = :finite, ftol = 1e-9, iterations = 100)
    @test converged(sol)
    η_star = sol.zero
    _, _, verify_star = cm_originzc_value_verified_from_eta(x_free_calib, η_star, pcx.ctx_cm)
    Δ_profiled = verify_star.Delta_dual
    @printf "  nlsolve converged=%s  Delta_origin_mean_only(profiled)=%.8f  diff vs unrestricted=%.3e\n" converged(sol) Δ_profiled (Δ_profiled - Δ_unrestricted)
    @test isapprox(Δ_profiled, Δ_unrestricted; atol = 1e-5)
end

println("="^100)
println("5) ZC nesting: Delta_unrestricted <= Delta_origin_ZC (K=1, K=2)")
println("="^100)
@testset "pairwise-ZC restriction weakly increases Delta relative to mean-only" begin
    for K_mean in (1, 2)
        Δ_mean_only = results[(K_mean, 0)].verify.Delta_dual
        Δ_zc = results[(K_mean, K_mean)].verify.Delta_dual
        @printf "  K=%d  Delta_mean_only=%.8f  Delta_zc=%.8f  (zc - mean_only)=%.3e\n" K_mean Δ_mean_only Δ_zc (Δ_zc - Δ_mean_only)
        @test Δ_zc >= Δ_mean_only - 1e-6
    end
end

println("="^100)
println("6) analytic eta_{o,k} derivative vs reoptimized central finite differences")
println("="^100)
@testset "analytic d(Delta_dual)/d(eta_{o,k}) matches reoptimized FD" begin
    for (K_mean, K_pair) in [(1, 1), (2, 2)]
        r = results[(K_mean, K_pair)]
        pcx = r.pcx; base = r.base; verify = r.verify; νfull0 = r.νfull0
        g_analytic = d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull0; mean_m = verify.m_mean)
        h1 = 1e-4
        g_fd1 = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull0, pcx.ctx_cm; h = h1)
        maxdiff1 = maximum(abs.(g_analytic .- g_fd1))
        cos1 = dot(g_analytic, g_fd1) / (norm(g_analytic) * norm(g_fd1) + 1e-300)
        @printf "  K_mean=%d K_pair=%d  n_eta=%d  h=%.0e  max|analytic-fd|=%.3e  cosine=%.8f\n" K_mean K_pair length(g_analytic) h1 maxdiff1 cos1
        ok = maxdiff1 < 1e-3 && cos1 > 0.999
        if !ok
            h2 = 1e-5
            g_fd2 = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull0, pcx.ctx_cm; h = h2)
            maxdiff2 = maximum(abs.(g_analytic .- g_fd2))
            cos2 = dot(g_analytic, g_fd2) / (norm(g_analytic) * norm(g_fd2) + 1e-300)
            @printf "  (bandwidth check) h=%.0e  max|analytic-fd|=%.3e  cosine=%.8f\n" h2 maxdiff2 cos2
            ok = maxdiff2 < 1e-3 && cos2 > 0.999
        end
        @test ok
    end
end

println("\nALL ORIGIN-ZC PURE/INNER GATES DONE.")
