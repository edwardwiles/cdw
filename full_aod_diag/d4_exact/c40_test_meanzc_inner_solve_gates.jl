# Task brief Sections 6.2 (inner obj/grad/Hessian equivalence), 6.3 (nested
# restriction monotonicity + decomposition), 6.4 (nu-gradient tests), and 6.5
# (CM-only regression) at D=4. Needs a real KNITRO license (this file actually
# solves the inner CC dual problem, unlike c40_test_meanzc_pure_moments.jl).
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
include(joinpath(@__DIR__, "mean_zero_cov_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
using Test, Printf, LinearAlgebra, Random, Statistics

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)

"Unpack a packed upper-triangular Hessian vector into a dense symmetric n x n matrix (c13_validate_hessian_archs.jl's helper, reused verbatim)."
function unpack_packed(h::AbstractVector, n::Int)
    M = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        M[i, j] = h[k]; M[j, i] = h[k]
        k += 1
    end
    return M
end

"Dense-reference (Architecture A) analog of `archC_base_state`, for this file's mean/pair-aware `ctx_cm`/`aug` (generic -- archA_hess_cb_builder is the unmodified generic dense Hessian, works for ANY moments! layout)."
function archA_base_state_meanzc(x_free0::AbstractVector, ctx_cm)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder = archA_hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("archA_base_state_meanzc: inner solve failed, nStatus=$nStatus")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

const L = 10
const ARMS = (:cm_plus_mean, :cm_plus_mean_zero_covariance)
const BASES = (:direct, :anchored)

println("="^100)
println("SECTION 6.2a: dense (Architecture A) vs structured (widened-NCORE Architecture C) Hessian equivalence")
println("="^100)
@testset "Hessian equivalence: dense vs structured, both arms, both bases" begin
    for arm in ARMS, basis in BASES
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = basis, nu_ref = Ref(1.0))
        objA = aug.obj_cm
        n = objA.outer_constr_index
        K = zeros(size(ctx.U, 1))
        objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full_calib, objA.U, objA)
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
        @printf "  arm=%-28s basis=%-9s  n=%d  max|H_structured - H_dense| = %.3e\n" arm basis n maxerr
    end
end
println()

println("="^100)
println("SECTION 6.2b: inner solve equivalence (dense vs structured backend), recovered weights, canonical Delta_dual identity")
println("="^100)
@testset "Inner solve: dense vs structured backend agree on (zeta*, Delta_dual, recovered weights)" begin
    for arm in ARMS, basis in BASES
        nu_ref = Ref(1.0)
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = basis, nu_ref = nu_ref)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)

        base_dense = archA_base_state_meanzc(x_free_calib, ctx_cm)
        base_struct = archC_base_state(x_free_calib, ctx_cm, cctx)

        @test base_dense.inner_status in (0, -100, -101, -103)
        @test base_struct.inner_status in (0, -100, -101, -103)
        @test isapprox(base_dense.ζstar, base_struct.ζstar; atol = 1e-6)
        @test isapprox(base_dense.λstar, base_struct.λstar; atol = 1e-5)

        Δ_dense = delta_dual_from_base(ctx_cm.obj, base_dense)
        Δ_struct = delta_dual_from_base(ctx_cm.obj, base_struct)
        @test isapprox(Δ_dense, Δ_struct; atol = 1e-6)

        m_dense = base_dense.m_star; m_struct = base_struct.m_star
        @test isapprox(sum(m_dense) / length(m_dense), 1.0; atol = 1e-3)
        @test isapprox(sum(m_struct) / length(m_struct), 1.0; atol = 1e-3)

        # primal moment residual recovery: mean residual should be ~0 at a verified solution
        # (this IS the imposed restriction -- the CM+mean(+ZC) equality moments)
        resid_mean = recovered_mean_residuals(m_struct, aug.Zraw, nu_ref[])
        @printf "  arm=%-28s basis=%-9s  Delta_dual(dense)=%.8f  Delta_dual(struct)=%.8f  max|mean resid|=%.3e\n" arm basis Δ_dense Δ_struct maximum(abs.(resid_mean))
        @test maximum(abs.(resid_mean)) < 1e-3

        if arm === :cm_plus_mean_zero_covariance
            resid_pair = recovered_pair_residuals(m_struct, aug.Zpairraw, nu_ref[])
            @printf "      max|pair resid|=%.3e\n" maximum(abs.(resid_pair))
            @test maximum(abs.(resid_pair)) < 1e-2
        end
    end
end
println()

println("="^100)
println("SECTION 6.5: CM-only regression -- cm_extension=:cm_only path is the pre-existing, byte-unmodified build_cm_augmented_obj")
println("="^100)
@testset "CM-only path untouched by this file's additions" begin
    aug_plain = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    ctx_cm_plain = merge(ctx, (obj = aug_plain.obj_cm,))
    r1 = evaluate_fullA(x_free_calib, ctx_cm_plain; use_cache = false, warm = false)
    r2 = evaluate_fullA(x_free_calib, ctx_cm_plain; use_cache = false, warm = false)
    @test r1.zeta == r2.zeta   # bit-identical repeat call through the SAME unmodified path
    # calling build_cm_meanzc_augmented_obj with cm_extension=:cm_only must be REJECTED --
    # there is no such thing as "the mean/pair machinery in cm_only mode," per Section 4's
    # "cannot represent an invalid state" requirement.
    @test_throws ErrorException build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = :cm_only)
    cfg = MeanZCConfig()
    @test cfg.cm_extension === :cm_only
    @test meanzc_active(cfg) == false
    @test meanzc_cache_key(cfg, nothing) == (cm_extension = :cm_only,)
end
println()

println("="^100)
println("SECTION 6.4: nu-gradient tests -- analytic d(Delta_dual)/d(eta_nu) vs central finite differences")
println("="^100)
"Fixed-dual central FD of Delta_dual in eta_nu: RESOLVE-FREE, holds (zeta*,lambda*) fixed, only recomputes Delta_dual at the perturbed nu via delta_dual_from_base's own obj(...) recompute (mirrors the math note's envelope-theorem approximation being tested)."
function fixed_dual_delta_at_eta(aug, base, η)
    aug.nu_ref[] = exp(η)
    obj = aug.obj_cm
    # rebuild G at the new nu (moments! reads nu_ref[] fresh every call, theta unchanged) --
    # delta_dual_from_base's own obj(...) call does NOT recompute moments!, it only re-evaluates
    # Psi at (zeta*,lambda*) against whatever G is CURRENTLY stored in obj.H, so a genuine
    # fixed-dual-but-new-G approximation requires this explicit rebuild first.
    obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), base.θ_full0, obj.U, obj)
    obj.H[:, 2] .= 1.0
    return delta_dual_from_base(obj, base)
end
"Independently-reoptimized central FD: fully RE-SOLVES the inner CC dual problem at the perturbed nu (the genuinely independent ground truth)."
function reoptimized_delta_at_eta(x_free0, ctx_cm, cctx, aug, η)
    aug.nu_ref[] = exp(η)
    base = archC_base_state(x_free0, ctx_cm, cctx)
    return delta_dual_from_base(ctx_cm.obj, base)
end

@testset "d(Delta_dual)/d(eta_nu): fixed-dual FD and reoptimized FD, both arms, several bandwidths" begin
    for arm in ARMS
        nu_ref = Ref(1.0)
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = :direct, nu_ref = nu_ref)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)

        η0 = log(1.0)
        nu_ref[] = exp(η0)
        base0 = archC_base_state(x_free_calib, ctx_cm, cctx)
        _, verify0 = archC_verified_state(x_free_calib, ctx_cm, cctx)
        d_analytic = d_delta_dual_d_eta_nu(base0.λstar, aug, nu_ref[]; mean_m = verify0.m_mean)

        for h in (0.02, 0.01, 0.005)
            d_fixed = (fixed_dual_delta_at_eta(aug, base0, η0 + h) - fixed_dual_delta_at_eta(aug, base0, η0 - h)) / (2h)
            d_reopt = (reoptimized_delta_at_eta(x_free_calib, ctx_cm, cctx, aug, η0 + h) -
                       reoptimized_delta_at_eta(x_free_calib, ctx_cm, cctx, aug, η0 - h)) / (2h)
            nu_ref[] = exp(η0)   # restore -- callers must not leave nu_ref stale for the next iteration
            sign_ok_fixed = sign(d_analytic) == sign(d_fixed) || abs(d_analytic) < 1e-8
            sign_ok_reopt = sign(d_analytic) == sign(d_reopt) || abs(d_analytic) < 1e-8
            relerr_fixed = abs(d_analytic - d_fixed) / max(1e-8, abs(d_fixed))
            relerr_reopt = abs(d_analytic - d_reopt) / max(1e-8, abs(d_reopt))
            @printf "  arm=%-28s h=%.4f  analytic=%.6e  fixed-dual FD=%.6e (relerr=%.3e)  reoptimized FD=%.6e (relerr=%.3e)\n" arm h d_analytic d_fixed relerr_fixed d_reopt relerr_reopt
            @test sign_ok_fixed
            @test sign_ok_reopt
            @test relerr_fixed < 0.1
            @test relerr_reopt < 0.1
        end

        # mean-only arm: confirm the derivative touches no pair storage (aug.Zpairraw === nothing)
        # and involves no pair term (already checked at the pure-moment level; here just confirm
        # the arm actually reached this point without needing Zpairraw)
        if arm === :cm_plus_mean
            @test aug.Zpairraw === nothing
        end
    end
end
println()

println("="^100)
println("SECTION 6.3: nested restriction monotonicity -- Delta_CM <= min_nu Delta_CM+mean <= min_nu Delta_CM+mean+ZC")
println("="^100)
"""
Simple bounded 1-D golden-section-free profile: bracket + coarse grid + local refine, over
eta_nu in log(nu_lo)..log(nu_hi). The task's hard finite-support interval (Section 3) is a
NECESSARY condition for a common mean to be feasible, not a guarantee that the inner CC dual
problem solves cleanly at every point in it (e.g. near the extreme endpoints, where the implied
reweighting can become numerically degenerate/infeasible for KNITRO, nStatus=-300) -- so a
solve failure at a given eta is treated as +Inf (excluded from the argmin), not a fatal error.
Errors out only if EVERY probed point fails (a genuine problem, not an edge effect).
"""
function profile_over_eta(f::Function, η_lo::Float64, η_hi::Float64; ngrid::Int = 15, nrefine::Int = 25)
    function safe_f(η)
        try
            return f(η)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            return Inf
        end
    end
    ηs = range(η_lo, η_hi, length = ngrid)
    vals = [safe_f(η) for η in ηs]
    all(isinf, vals) && error("profile_over_eta: every grid point failed to solve over [$η_lo, $η_hi]")
    i = argmin(vals)
    η_best = ηs[i]; v_best = vals[i]
    lo2 = max(η_lo, η_best - step(ηs)); hi2 = min(η_hi, η_best + step(ηs))
    for η in range(lo2, hi2, length = nrefine)
        v = safe_f(η)
        if v < v_best
            v_best = v; η_best = η
        end
    end
    return η_best, v_best
end

@testset "Nesting inequalities hold up to solver tolerance (D=4)" begin
    nu_lo, nu_hi = nu_feasible_interval(ctx.U)
    η_lo, η_hi = log(nu_lo), log(nu_hi)
    @printf "  nu feasible interval: [%.6f, %.6f]  (eta range [%.4f, %.4f])\n" nu_lo nu_hi η_lo η_hi

    aug_plain = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    ctx_cm_plain = merge(ctx, (obj = aug_plain.obj_cm,))
    r_plain = evaluate_fullA(x_free_calib, ctx_cm_plain; use_cache = false, warm = false)
    Δ_cm = r_plain.Delta_dual   # canonical (NOT -r_plain.zeta, the F1-buggy proxy)
    @printf "  Delta_CM (no mean/ZC restriction) = %.8f\n" Δ_cm

    results = Dict{Symbol,Float64}()
    for arm in ARMS
        nu_ref = Ref(1.0)
        aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = :direct, nu_ref = nu_ref)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        cctx = build_cm_meanzc_bin_ctx(ctx, aug)
        function neg_of_delta(η)
            nu_ref[] = exp(η)
            base = archC_base_state(x_free_calib, ctx_cm, cctx)
            return delta_dual_from_base(ctx_cm.obj, base)
        end
        η_best, Δ_best = profile_over_eta(neg_of_delta, η_lo, η_hi)
        results[arm] = Δ_best
        @printf "  arm=%-28s  min_nu Delta = %.8f  at nu*=%.6f (eta*=%.4f)\n" arm Δ_best exp(η_best) η_best
    end

    Δ_mean = results[:cm_plus_mean]
    Δ_zc = results[:cm_plus_mean_zero_covariance]
    tol = 1e-4
    @printf "  Delta_CM=%.8f  <=  min_nu Delta_CM+mean=%.8f  <=  min_nu Delta_CM+mean+ZC=%.8f  ?\n" Δ_cm Δ_mean Δ_zc
    @printf "  mean_incr = %.3e   ZC_incr = %.3e\n" (Δ_mean - Δ_cm) (Δ_zc - Δ_mean)
    @test Δ_cm <= Δ_mean + tol
    @test Δ_mean <= Δ_zc + tol

    # second inequality re-checked at several COMMON fixed nu values (not just each arm's own
    # optimum) -- guards against a profiling failure masquerading as a moment bug (Section 6.3)
    for ν_fixed in range(max(nu_lo, 0.7), min(nu_hi, 1.4), length = 5)
        η_fixed = log(ν_fixed)
        vals = Dict{Symbol,Float64}()
        failed = false
        for arm in ARMS
            nu_ref = Ref(1.0)
            aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, cm_extension = arm, meanzc_basis = :direct, nu_ref = nu_ref)
            ctx_cm = merge(ctx, (obj = aug.obj_cm,))
            cctx = build_cm_meanzc_bin_ctx(ctx, aug)
            nu_ref[] = ν_fixed
            try
                base = archC_base_state(x_free_calib, ctx_cm, cctx)
                vals[arm] = delta_dual_from_base(ctx_cm.obj, base)
            catch e
                e isa CMExpectedSolveFailure || rethrow()
                failed = true
                @printf "    fixed nu=%.4f: SKIPPED (arm=%s inner solve infeasible at this nu -- edge effect, not a moment bug)\n" ν_fixed arm
                break
            end
        end
        failed && continue
        @printf "    fixed nu=%.4f: Delta_CM+mean=%.8f  Delta_CM+mean+ZC=%.8f  (diff=%.3e)\n" ν_fixed vals[:cm_plus_mean] vals[:cm_plus_mean_zero_covariance] (vals[:cm_plus_mean_zero_covariance] - vals[:cm_plus_mean])
        @test vals[:cm_plus_mean] <= vals[:cm_plus_mean_zero_covariance] + tol
    end
end

println("\nAll Section 6.2/6.3/6.4/6.5 inner-solve gates passed (or reported above).")
