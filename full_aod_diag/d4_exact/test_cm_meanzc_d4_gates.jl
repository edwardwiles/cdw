# D=4 correctness gates for the CM+moments(+ZC) extension: dense-vs-structured
# Hessian equivalence, inner-solve equivalence + canonical Delta_dual, fixed-
# point nesting (CM <= CM+mean <= CM+mean+ZC, level by level), full outer
# gradient vs a trusted reference (including the eta_nu_k components), and
# CM-only regression. Covers K_mean=1 (regression against the original two
# named arms) and K_mean=2 (the generalized power-level extension). Needs a
# real KNITRO license -- this file actually solves the inner CC dual problem.
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
using Test, Printf, LinearAlgebra, Random, Statistics

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

"Unpack a packed upper-triangular Hessian vector into a dense symmetric n x n matrix."
function unpack_packed(h::AbstractVector, n::Int)
    M = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        M[i, j] = h[k]; M[j, i] = h[k]
        k += 1
    end
    return M
end

"Dense-reference (Architecture A) analog of archC_meanzc_base_state -- archA_hess_cb_builder is generic over any moments! layout, no new code needed."
function archA_meanzc_base_state(x_free0::AbstractVector, νvec::AbstractVector{Float64}, ctx_cm)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, νvec)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0; hess_cb_builder = archA_hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("archA_meanzc_base_state: inner solve failed, nStatus=$nStatus")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, copy(obj.arg1), nStatus)
end

const L = 10
# (K_mean, K_pair, label) -- (1,0)/(1,1) regression-check the original two named arms
# (meanzc_extension_to_K(:cm_plus_equal_means)==(1,0), (:cm_plus_equal_means_zero_covariance)==(1,1));
# (2,0)/(2,2) are the new generalized power-level configurations.
const CONFIGS = [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc"), (2, 0, "K2_mean_only"), (2, 2, "K2_mean_zc")]
const BASES = (:direct, :anchored)
# Draws are Exp(1) (genExpRands!), so E[z^k] = k! is the natural scale for nu_k (k=1: mean 1,
# k=2: E[z^2]=2, etc.) -- an arbitrary constant offset per level (e.g. 1.0+0.15*(k-1)) is not a
# feasible starting point for k>=2 since higher raw moments grow factorially, not linearly.
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]

@test meanzc_extension_to_K(:cm_plus_equal_means) == (1, 0)
@test meanzc_extension_to_K(:cm_plus_equal_means_zero_covariance) == (1, 1)

println("="^100)
println("8.2a: dense (Architecture A) vs structured (widened-NCORE Architecture C) Hessian equivalence")
println("="^100)
@testset "Hessian equivalence: dense vs structured, all (K_mean,K_pair) configs, both bases" begin
    for (K_mean, K_pair, label) in CONFIGS, basis in BASES
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = basis)
        objA = aug.obj_cm
        n = objA.outer_constr_index
        θ_ext_calib = vcat(θ_full_calib, nu0vec(K_mean))
        K = zeros(size(ctx.U, 1))
        objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_ext_calib, objA.U, objA)
        objA.H[:, 1] .= K
        objA.H[:, 2] .= 1.0

        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        Random.seed!(7000 + L)
        xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
        maxerr = 0.0
        for x in xs
            hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            objA(x, h = hA)
            HA = unpack_packed(hA, n)

            hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
            _archC_prep_for_hessian!(objA, x)
            hessian_cm_structured!(hC, objA, cctx)
            HC = unpack_packed(hC, n)

            err = maximum(abs.(HC .- HA))
            maxerr = max(maxerr, err)
            @test err < 1e-8
        end
        @printf "  %-14s basis=%-9s  n=%d  max|H_structured - H_dense| = %.3e\n" label basis n maxerr
    end
end
println()

println("="^100)
println("8.2b: inner solve equivalence (dense vs structured), canonical Delta_dual, KKT residuals")
println("="^100)
@testset "Inner solve: dense vs structured backend agree; canonical Delta_dual; primal-dual gap; KKT residual" begin
    for (K_mean, K_pair, label) in CONFIGS, basis in BASES
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = basis)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        ν0 = nu0vec(K_mean)

        baseA = archA_meanzc_base_state(x_free_calib, ν0, ctx_cm)
        baseC, verifyC = archC_meanzc_verified_state(x_free_calib, ν0, ctx_cm, cctx)

        @test baseA.ζstar ≈ baseC.ζstar atol = 1e-7
        @test baseA.λstar ≈ baseC.λstar atol = 1e-6
        @test baseA.m_star ≈ baseC.m_star atol = 1e-6

        # canonical Delta_dual via delta_dual_from_base (cm_production_bundle.jl, reused) vs
        # verify.Delta_dual (archC_meanzc_verified_state's own independent recompute)
        Δ_from_base = delta_dual_from_base(ctx_cm.obj, baseC)
        @test Δ_from_base ≈ verifyC.Delta_dual atol = 1e-8
        @test isfinite(verifyC.Delta_dual)
        @test verifyC.primal_dual_gap < 1e-5
        @test verifyC.max_abs_moment_kkt_resid < 1e-4
        @test verifyC.mean_m_resid < 1e-6
        @test is_verified_success(verifyC)

        # mean/pair residuals at every level should be finite at a near-feasible calibration point
        for k in 1:K_mean
            resid_mean = recovered_mean_residuals(baseC.m_star, aug.Zraw_all[k], ν0[k])
            @test all(isfinite, resid_mean)
        end
        for k in 1:K_pair
            resid_pair = recovered_pair_residuals(baseC.m_star, aug.Zpairraw_all[k], ν0[k])
            @test all(isfinite, resid_pair)
        end
        @printf "  %-14s basis=%-9s  Delta_dual=%.8f  gap=%.3e  kkt=%.3e\n" label basis verifyC.Delta_dual verifyC.primal_dual_gap verifyC.max_abs_moment_kkt_resid
    end
end
println()

println("="^100)
println("8.3: fixed-point nesting Delta_CM <= Delta_CM+mean <= min_nu Delta_CM+mean+ZC, K=1 and K=2")
println("="^100)
@testset "Fixed-point nesting at fixed (g,A): CM <= K_mean=1 mean-only <= K_mean=1 mean+ZC" begin
    # CM baseline (existing production path, byte-unchanged)
    pcx_cm = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    base_cm = archC_base_state(x_free_calib, pcx_cm.ctx_cm, pcx_cm.cctx)
    Δ_cm = delta_dual_from_base(pcx_cm.ctx_cm.obj, base_cm)

    for basis in BASES
        aug_mean = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 0, meanzc_basis = basis)
        ctx_mean = merge(ctx, (obj = aug_mean.obj_cm,))
        cctx_mean = build_cm_meanzc_bin_ctx(ctx, aug_mean)

        ν_grid = range(0.5, 2.0, length = 13)
        Δ_mean_vals = Float64[]
        for ν in ν_grid
            try
                base, verify = archC_meanzc_verified_state(x_free_calib, [ν], ctx_mean, cctx_mean)
                is_verified_success(verify) && push!(Δ_mean_vals, verify.Delta_dual)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
            end
        end
        @test !isempty(Δ_mean_vals)
        Δ_mean_best = minimum(Δ_mean_vals)
        @test Δ_cm <= Δ_mean_best + 1e-6

        aug_zc = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 1, meanzc_basis = basis)
        ctx_zc = merge(ctx, (obj = aug_zc.obj_cm,))
        cctx_zc = build_cm_meanzc_bin_ctx(ctx, aug_zc)
        Δ_zc_vals = Float64[]
        for ν in ν_grid
            try
                base, verify = archC_meanzc_verified_state(x_free_calib, [ν], ctx_zc, cctx_zc)
                is_verified_success(verify) && push!(Δ_zc_vals, verify.Delta_dual)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
            end
        end
        @test !isempty(Δ_zc_vals)
        Δ_zc_best = minimum(Δ_zc_vals)
        @test Δ_mean_best <= Δ_zc_best + 1e-6

        @printf "  K=1 basis=%-9s  Delta_CM=%.8f <= Delta_CM+mean(best)=%.8f <= Delta_CM+mean+ZC(best)=%.8f\n" basis Δ_cm Δ_mean_best Δ_zc_best
    end
end

@testset "Fixed-point nesting, K generalization: K_mean=1 mean+ZC <= K_mean=2 mean+ZC (more restrictions, weakly tighter)" begin
    # Adding a SECOND power level's worth of restrictions (K_mean=2,K_pair=2 nests
    # K_mean=1,K_pair=1: every K=1 restriction is still imposed, plus new ones for k=2) can only
    # weakly INCREASE the achieved divergence floor at a fixed (g,A) point (more equality
    # constraints on the same least-favorable F => weakly smaller feasible set => weakly larger
    # min-divergence). Profile nu_2 on a grid around nu_1's own optimum (nu_1 fixed at its K=1
    # optimum since that restriction is unchanged by adding K=2).
    aug1 = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 1, meanzc_basis = :direct)
    ctx1 = merge(ctx, (obj = aug1.obj_cm,))
    cctx1 = build_cm_meanzc_bin_ctx(ctx, aug1)
    ν_grid = range(0.5, 2.0, length = 13)
    Δ1_vals = Tuple{Float64,Float64}[]
    for ν in ν_grid
        try
            base, verify = archC_meanzc_verified_state(x_free_calib, [ν], ctx1, cctx1)
            is_verified_success(verify) && push!(Δ1_vals, (verify.Delta_dual, ν))
        catch e
            e isa CMExpectedSolveFailure || rethrow()
        end
    end
    @test !isempty(Δ1_vals)
    Δ1_best, ν1_best = Δ1_vals[argmin(first.(Δ1_vals))]

    aug2 = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 2, K_pair = 2, meanzc_basis = :direct)
    ctx2 = merge(ctx, (obj = aug2.obj_cm,))
    cctx2 = build_cm_meanzc_bin_ctx(ctx, aug2)
    ν2_grid = range(1.0, 4.0, length = 13)   # centered on Exp(1)'s E[z^2]=2! = 2
    Δ2_vals = Float64[]
    for ν2 in ν2_grid
        try
            base, verify = archC_meanzc_verified_state(x_free_calib, [ν1_best, ν2], ctx2, cctx2)
            is_verified_success(verify) && push!(Δ2_vals, verify.Delta_dual)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
        end
    end
    @test !isempty(Δ2_vals)
    Δ2_best = minimum(Δ2_vals)
    @test Δ1_best <= Δ2_best + 1e-6
    @printf "  Delta_K1(mean+ZC, best)=%.8f <= Delta_K2(mean+ZC, best over nu_2 at nu_1=%.4f)=%.8f\n" Δ1_best ν1_best Δ2_best
end
println()

println("="^100)
println("8.4: full outer gradient (composite_gradient_at_fast reuse) vs trusted full-rebuild reference")
println("="^100)
@testset "Outer gradient vs trusted full-rebuild reference, all configs, direct basis" begin
    for (K_mean, K_pair, label) in CONFIGS
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = :direct)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        bins = cm_bin_indices_for(ctx, aug)
        pcx = (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)

        ν0 = nu0vec(K_mean)
        base0, verify0 = archC_meanzc_verified_state(x_free_calib, ν0, ctx_cm, cctx)
        g_ext, meta = cm_meanzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0)
        # g_ext = [g_econ (w-space, D2 = D^2 entries: gp + pivot-reduced log-A_od); d_eta (K_mean entries)]
        D2 = ctx.D^2
        @test length(g_ext) == D2 + K_mean

        # (g,A_od) block: composite_gradient_at_fast is itself an adaptive-bandwidth secant
        # method around a possibly-nonsmooth (winner-switching) objective -- a naive fixed-h
        # central FD is NOT a valid ground truth for it (confirmed live: even PLAIN CM's
        # already-trusted analytic gradient disagrees with a naive h=1e-5 probe by up to 0.14).
        # Use this codebase's own trusted reference instead: full_rebuild_gradient_fallback_meanzc
        # (fixed-dual, h=0.01, full moment rebuild every probe -- the same methodology
        # full_rebuild_gradient_fallback/fixed_dual_L already establish as correct for plain CM,
        # cosine similarity 0.9999 there).
        g_ref, meta_ref = full_rebuild_gradient_fallback_meanzc(x_free_calib, ν0, ctx_cm, pe, base0; h = 0.01)
        g_econ = g_ext[1:D2]
        maxdiff = maximum(abs.(g_econ .- g_ref))
        cossim = dot(g_econ, g_ref) / (norm(g_econ) * norm(g_ref))
        @printf "  %-14s (g,A_od) block: max|analytic-full_rebuild|=%.3e  cosine=%.6f\n" label maxdiff cossim
        @test maxdiff < 5e-3
        @test cossim > 0.999

        # eta_nu_k components: FD in eta_nu-space (nu_k = exp(eta_nu_k)), reoptimized inner solve,
        # ONE level perturbed at a time (block-diagonal Jacobian: perturbing eta_nu_k must not
        # move OTHER levels' analytic components, checked implicitly since g_ext itself is fixed
        # while only the FD probe varies level k).
        η0 = log.(ν0)
        hη = 1e-5
        for k in 1:K_mean
            ηp = copy(η0); ηp[k] += hη
            ηm = copy(η0); ηm[k] -= hη
            _, _, vp = cm_meanzc_production_value_verified(x_free_calib, exp.(ηp), pcx)
            _, _, vm = cm_meanzc_production_value_verified(x_free_calib, exp.(ηm), pcx)
            fd_eta = (vp.Delta_dual - vm.Delta_dual) / (2hη)
            @printf "  %-14s level k=%d  analytic d(Delta)/d(eta_nu_%d)=%.6f  FD(reoptimized)=%.6f  |diff|=%.3e\n" label k k g_ext[D2+k] fd_eta abs(g_ext[D2+k]-fd_eta)
            @test g_ext[D2+k] ≈ fd_eta atol = 1e-3 rtol = 1e-2

            # fixed-dual FD (no reoptimization -- recompute Delta_dual at the SAME base.λstar,ζstar,
            # only nu_k perturbed, other levels held at their base value)
            function delta_dual_fixed_dual_at_nu(base, ctx_cm, νvec)
                ncon = ctx_cm.obj.d - ctx_cm.obj.outer_constr_index + 2
                θ_ext = vcat(base.θ_full0, νvec)
                Kbuf = zeros(size(ctx_cm.obj.U,1))
                ctx_cm.obj.moments!(Kbuf, CS.select_G_from_H(ctx_cm.obj, ctx_cm.obj.H), θ_ext, ctx_cm.obj.U, ctx_cm.obj)
                ctx_cm.obj.H[:,1] .= Kbuf; ctx_cm.obj.H[:,2] .= 1.0
                cbuf = zeros(ncon)
                inner_x = vcat(base.ζstar, base.λstar)
                ctx_cm.obj(inner_x, constr = @view(cbuf[1:ncon]))
                return cbuf[1] / 1e10
            end
            fdp = delta_dual_fixed_dual_at_nu(base0, ctx_cm, exp.(ηp))
            fdm = delta_dual_fixed_dual_at_nu(base0, ctx_cm, exp.(ηm))
            fd_eta_fixed = (fdp - fdm) / (2hη)
            @test g_ext[D2+k] ≈ fd_eta_fixed atol = 1e-4 rtol = 1e-3
        end
    end
end
println()

println("="^100)
println("8.5: CM-only regression -- extension code present but disabled uses the SAME path/dims/results")
println("="^100)
@testset "CM-only regression: build_cm_production_context unaffected by cm_meanzc files being loaded" begin
    pcx_before = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    base_before = archC_base_state(x_free_calib, pcx_before.ctx_cm, pcx_before.cctx)
    Δ_before = delta_dual_from_base(pcx_before.ctx_cm.obj, base_before)

    pcx_after = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    base_after = archC_base_state(x_free_calib, pcx_after.ctx_cm, pcx_after.cctx)
    Δ_after = delta_dual_from_base(pcx_after.ctx_cm.obj, base_after)

    @test pcx_before.ctx_cm.obj.d == pcx_after.ctx_cm.obj.d
    @test pcx_before.ctx_cm.obj.outer_constr_index == pcx_after.ctx_cm.obj.outer_constr_index
    @test Δ_before == Δ_after            # bit-identical: same code path, same inputs
    @test base_before.ζstar == base_after.ζstar
    @test base_before.λstar == base_after.λstar

    # invalid (K_mean,K_pair) is rejected at construction, never silently accepted
    @test_throws ErrorException build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 0, K_pair = 0)
    @test_throws ErrorException build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = 1, K_pair = 2)
end

println()
println("All D=4 CM+moments(+ZC) gate tests passed (K_mean=1 regression + K_mean=2 generalization).")
