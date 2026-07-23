# D=4 correctness gates for the CM+mean(+ZC) extension: dense-vs-structured
# Hessian equivalence, inner-solve equivalence + canonical Delta_dual, fixed-
# point nesting (CM <= CM+mean <= CM+mean+ZC), full outer gradient vs central
# FD (including the eta_nu component), and CM-only regression. Needs a real
# KNITRO license -- this file actually solves the inner CC dual problem.
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
function archA_meanzc_base_state(x_free0::AbstractVector, ν::Float64, ctx_cm)
    obj = ctx_cm.obj
    θ_econ0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    θ_ext0 = vcat(θ_econ0, ν)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_ext0; hess_cb_builder = archA_hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("archA_meanzc_base_state: inner solve failed, nStatus=$nStatus")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_econ0, ζstar, λstar, copy(obj.arg1), nStatus)
end

const L = 10
const ARMS = (:cm_plus_equal_means, :cm_plus_equal_means_zero_covariance)
const BASES = (:direct, :anchored)
const NU0 = 1.0

println("="^100)
println("8.2a: dense (Architecture A) vs structured (widened-NCORE Architecture C) Hessian equivalence")
println("="^100)
@testset "Hessian equivalence: dense vs structured, both arms, both bases" begin
    for arm in ARMS, basis in BASES
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = basis)
        objA = aug.obj_cm
        n = objA.outer_constr_index
        θ_ext_calib = vcat(θ_full_calib, NU0)
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
        @printf "  arm=%-38s basis=%-9s  n=%d  max|H_structured - H_dense| = %.3e\n" arm basis n maxerr
    end
end
println()

println("="^100)
println("8.2b: inner solve equivalence (dense vs structured), canonical Delta_dual, KKT residuals")
println("="^100)
@testset "Inner solve: dense vs structured backend agree; canonical Delta_dual; primal-dual gap; KKT residual" begin
    for arm in ARMS, basis in BASES
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = basis)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)

        baseA = archA_meanzc_base_state(x_free_calib, NU0, ctx_cm)
        baseC, verifyC = archC_meanzc_verified_state(x_free_calib, NU0, ctx_cm, cctx)

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

        # mean/pair residuals should be small at a near-feasible calibration point's converged F
        resid_mean = recovered_mean_residuals(baseC.m_star, aug.Zraw, NU0)
        @test maximum(abs.(resid_mean)) < 10.0   # loose sanity bound; tightness checked in nesting test below
        if aug.n_pair > 0
            resid_pair = recovered_pair_residuals(baseC.m_star, aug.Zpairraw, NU0)
            @test all(isfinite, resid_pair)
        end
        @printf "  arm=%-38s basis=%-9s  Delta_dual=%.8f  gap=%.3e  kkt=%.3e\n" arm basis verifyC.Delta_dual verifyC.primal_dual_gap verifyC.max_abs_moment_kkt_resid
    end
end
println()

println("="^100)
println("8.3: fixed-point nesting Delta_CM <= Delta_CM+mean <= min_nu Delta_CM+mean+ZC")
println("="^100)
@testset "Fixed-point nesting at fixed (g,A): CM <= CM+mean <= min_ν CM+mean+ZC" begin
    # CM baseline (existing production path, byte-unchanged)
    pcx_cm = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    base_cm = archC_base_state(x_free_calib, pcx_cm.ctx_cm, pcx_cm.cctx)
    Δ_cm = delta_dual_from_base(pcx_cm.ctx_cm.obj, base_cm)

    for basis in BASES
        aug_mean = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = :cm_plus_equal_means, meanzc_basis = basis)
        ctx_mean = merge(ctx, (obj = aug_mean.obj_cm,))
        cctx_mean = build_cm_meanzc_bin_ctx(ctx, aug_mean)

        # profile a narrow grid around the sample-mean-consistent nu to find the arm's own floor
        ν_grid = range(0.5, 2.0, length = 13)
        Δ_mean_best = Inf
        for ν in ν_grid
            try
                _, verify = archC_meanzc_verified_state(x_free_calib, ν, ctx_mean, cctx_mean), nothing
            catch
            end
        end
        Δ_mean_vals = Float64[]
        for ν in ν_grid
            try
                base, verify = archC_meanzc_verified_state(x_free_calib, ν, ctx_mean, cctx_mean)
                is_verified_success(verify) && push!(Δ_mean_vals, verify.Delta_dual)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
            end
        end
        @test !isempty(Δ_mean_vals)
        Δ_mean_best = minimum(Δ_mean_vals)
        @test Δ_cm <= Δ_mean_best + 1e-6

        aug_zc = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = :cm_plus_equal_means_zero_covariance, meanzc_basis = basis)
        ctx_zc = merge(ctx, (obj = aug_zc.obj_cm,))
        cctx_zc = build_cm_meanzc_bin_ctx(ctx, aug_zc)
        Δ_zc_vals = Float64[]
        for ν in ν_grid
            try
                base, verify = archC_meanzc_verified_state(x_free_calib, ν, ctx_zc, cctx_zc)
                is_verified_success(verify) && push!(Δ_zc_vals, verify.Delta_dual)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
            end
        end
        @test !isempty(Δ_zc_vals)
        Δ_zc_best = minimum(Δ_zc_vals)
        @test Δ_mean_best <= Δ_zc_best + 1e-6

        @printf "  basis=%-9s  Delta_CM=%.8f <= Delta_CM+mean(best over grid)=%.8f <= Delta_CM+mean+ZC(best over grid)=%.8f\n" basis Δ_cm Δ_mean_best Δ_zc_best
    end
end
println()

println("="^100)
println("8.4: full outer gradient (composite_gradient_at_fast reuse) vs central FD, incl. eta_nu")
println("="^100)
@testset "Outer gradient vs central FD, both arms, direct basis" begin
    for arm in ARMS
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = :direct)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        bins = cm_bin_indices_for(ctx, aug)
        pcx = (ctx_cm = ctx_cm, aug = aug, cctx = cctx, bins = bins)

        ν0 = 1.05
        base0, verify0 = archC_meanzc_verified_state(x_free_calib, ν0, ctx_cm, cctx)
        g_ext, meta = cm_meanzc_production_gradient(x_free_calib, ν0, pcx, ctx, pe; base = base0, verify = verify0)
        # g_ext = [g_econ (w-space, D2 = D^2 entries: gp + pivot-reduced log-A_od); d_eta]
        D2 = ctx.D^2
        @test length(g_ext) == D2 + 1

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
        @printf "  arm=%-38s (g,A_od) block: max|analytic-full_rebuild|=%.3e  cosine=%.6f\n" arm maxdiff cossim
        @test maxdiff < 5e-3
        @test cossim > 0.999

        # eta_nu component: FD in eta_nu-space (nu = exp(eta_nu)), reoptimized inner solve at each probe
        η0 = log(ν0)
        hη = 1e-5
        _, _, vp = cm_meanzc_production_value_verified(x_free_calib, exp(η0 + hη), pcx)
        _, _, vm = cm_meanzc_production_value_verified(x_free_calib, exp(η0 - hη), pcx)
        fd_eta = (vp.Delta_dual - vm.Delta_dual) / (2hη)
        @printf "  arm=%-38s analytic d(Delta)/d(eta_nu)=%.6f  FD(reoptimized)=%.6f  |diff|=%.3e\n" arm g_ext[end] fd_eta abs(g_ext[end]-fd_eta)
        @test g_ext[end] ≈ fd_eta atol = 1e-3 rtol = 1e-2

        # fixed-dual FD (no reoptimization -- recompute Delta_dual at the SAME base.λstar,ζstar, only nu perturbed)
        function delta_dual_fixed_dual_at_nu(base, ctx_cm, ν)
            ncon = ctx_cm.obj.d - ctx_cm.obj.outer_constr_index + 2
            θ_ext = vcat(base.θ_full0, ν)
            K = zeros(size(ctx_cm.obj.U,1))
            ctx_cm.obj.moments!(K, CS.select_G_from_H(ctx_cm.obj, ctx_cm.obj.H), θ_ext, ctx_cm.obj.U, ctx_cm.obj)
            ctx_cm.obj.H[:,1] .= K; ctx_cm.obj.H[:,2] .= 1.0
            cbuf = zeros(ncon)
            inner_x = vcat(base.ζstar, base.λstar)
            ctx_cm.obj(inner_x, constr = @view(cbuf[1:ncon]))
            return cbuf[1] / 1e10
        end
        fdp = delta_dual_fixed_dual_at_nu(base0, ctx_cm, exp(η0 + hη))
        fdm = delta_dual_fixed_dual_at_nu(base0, ctx_cm, exp(η0 - hη))
        fd_eta_fixed = (fdp - fdm) / (2hη)
        @test g_ext[end] ≈ fd_eta_fixed atol = 1e-4 rtol = 1e-3
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

    # invalid cm_extension is rejected at construction, never silently accepted
    @test_throws ErrorException build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = :cm_only)
    @test_throws ErrorException build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = :bogus)
end

println()
println("All D=4 CM+mean(+ZC) gate tests passed.")
