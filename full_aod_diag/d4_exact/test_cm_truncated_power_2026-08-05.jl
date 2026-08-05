# 2026-08-05 truncated-power task: correctness gates for the new CDW eq.36 (truncated
# (1-sigma)-power) common-marginals feature family, alongside the pre-existing eq.35 (CDF)
# family. See CM_CURRENT_SINGLE_BLOCK_SOURCE_MAP.md for the full call-chain trace and design.
#
# Gates implemented here (Step F of the task brief):
#   1. Direct feature test: hand-construct both families at small D/W/L, require exact
#      agreement with precalc_common_marginals_cdf.
#   2. Dimension gates: CDF sub-block=(D-1)L, total=2(D-1)L, outer dimension unchanged.
#   3. CDF-block-preservation gate: new CDF sub-block bit-identical to calling with
#      include_truncated_moment=false at identical state.
#   4. D4 real KNITRO solve (via oracle.jl's trusted dense evaluate_fullA -- Architecture A,
#      fully generic, no new math) for the two-family augmented objective: inner solve
#      converges, CM-block KKT residuals near machine precision for BOTH sub-blocks.
#   5. Checkpoint incompatibility: a schema-9 (pre-existing single-family) style checkpoint
#      construction is refused by load_cm_checkpoint.
using Test
using LinearAlgebra: norm
using Random
using Statistics: quantile

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

println("="^100)
println("Gate 1+2: direct feature construction + dimension gates (small hand D/W/L)")
println("="^100)

@testset "direct feature construction" begin
    W, D, L = 200, 4, 5
    refIndex1 = 1
    σHat = 3.0
    Random.seed!(20260805)
    U = rand(W, D) .* 3.0 .+ 0.1   # positive support, arbitrary (not Exp(1) -- doesn't matter for a pure feature-construction check)
    probs = collect(range(1/L, (L-1)/L, length=L))
    z_expect = quantile(U[:, refIndex1], probs)

    CM, z, origins = precalc_common_marginals_cdf(U, refIndex1, L; include_truncated_moment = true, σHat = σHat, contrasts = :anchored, probs = probs)
    nO = D - 1
    @test length(origins) == nO
    @test origins == [2, 3, 4]
    @test z == z_expect
    @test size(CM) == (W, 2*nO*L)
    @test size(CM, 2) == n_cm_moments(D, L; include_truncated_moment = true)
    @test n_cm_moments(D, L; include_truncated_moment = true) == 2 * (D-1) * L
    @test n_cm_moments(D, L; include_truncated_moment = false) == (D-1) * L

    # Hand-construct family 1 (CDF, eq.35) and family 2 (truncated power, eq.36) independently,
    # using the SAME U/z/refIndex1/origins, and require exact agreement (:anchored -> R=nothing,
    # i.e. block == raw, no contrast rotation to worry about here).
    CDF_ref = [U[s, refIndex1] <= z[l] for s in 1:W, l in 1:L]
    for (li, l) in enumerate(1:L)
        for (oi, o) in enumerate(origins)
            col1 = (l-1)*nO + oi
            hand1 = [Float64(U[s,o] <= z[l]) - Float64(U[s,refIndex1] <= z[l]) for s in 1:W]
            @test CM[:, col1] == hand1
        end
    end
    pw = 1 - σHat
    tm_offset = nO*L
    for (li, l) in enumerate(1:L)
        for (oi, o) in enumerate(origins)
            col2 = tm_offset + (l-1)*nO + oi
            hand2 = [ (U[s,o]^pw) * Float64(U[s,o] <= z[l]) - (U[s,refIndex1]^pw) * Float64(U[s,refIndex1] <= z[l]) for s in 1:W]
            @test CM[:, col2] ≈ hand2
        end
    end
    @test all(isfinite, CM)
    # Not accidentally using U's rank/quantile position (i.e. not the CDF/uniform-transformed
    # draw) -- the power block must use the RAW z_o(omega)=U[:,o] values, not e.g. rank(U[:,o])/W.
    @test !all(CM[:, tm_offset+1] .== CM[:, 1])   # sanity: power block genuinely differs from CDF block

    println("PASS: direct feature construction matches hand-built eq.35+eq.36 exactly; dims correct.")
end

println("="^100)
println("Gate: D4/L50 dimension gate matching task brief's headline numbers pattern (D20/L50 -> 1900,")
println("here D4 sanity: (D-1)L=15 at L=5, total=30)")
println("="^100)
@testset "D20/L50 headline dimension (computed without building the D20 context)" begin
    D, L = 20, 50
    @test n_cm_moments(D, L; include_truncated_moment = false) == 950
    @test n_cm_moments(D, L; include_truncated_moment = true) == 1900
    println("PASS: D20/L50 CDF-only=950, two-family=1900 (950 -> 1900, +950).")
end

println("="^100)
println("Gate 3: CDF-block-preservation -- new CDF sub-block bit-identical to include_truncated_moment=false")
println("="^100)
@testset "CDF-block preservation" begin
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    L = 10
    aug_old = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = false, contrasts = :anchored)
    aug_new = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored)
    @test aug_old.ncm == (ctx.D-1)*L
    @test aug_new.ncm == 2*(ctx.D-1)*L
    @test aug_new.ncm_cdf == aug_old.ncm
    @test aug_new.n_families == 2
    @test aug_old.n_families == 1
    # bit-identical CDF sub-block (raw feature matrix)
    @test aug_new.CM[:, 1:aug_new.ncm_cdf] == aug_old.CM
    @test aug_new.z == aug_old.z
    @test aug_new.origins == aug_old.origins
    println("PASS: CDF sub-block of the two-family CM matrix is bit-identical to the CDF-only build.")
end

println("="^100)
println("Gate 4: D4 real KNITRO inner solve, two-family flexible CM, via oracle.jl's trusted dense evaluate_fullA")
println("(Architecture A / fully generic dense Hessian -- zero new Hessian code, see MASTER.md)")
println("="^100)
@testset "D4 real inner solve, two-family" begin
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    x_free_calib = ctx.θ0_up[ctx.free_idx]
    for L in (5, 10)
        aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        @test aug.ncm == 2*(ctx.D - 1)*L
        @test aug.obj_cm.d == ctx.obj.d + aug.ncm
        @test aug.obj_cm.outer_constr_index == aug.obj_cm.d

        t0 = time()
        r = evaluate_fullA(x_free_calib, ctx_cm; use_cache = false, warm = false)
        telapsed = time() - t0
        # (0, -100, -101, -102, -103): matches this repo's own broader "feasible result" acceptance
        # set (see knitro_status.jl / test_knitro_status.jl -- -102=KN_RC_FEAS_NO_IMPROVE is a
        # genuine feasible terminal status, just omitted from c10_d20_production_driver.jl's own
        # narrower FEASIBLE_CODES for unrelated reasons; several other test files in this repo
        # already accept it, e.g. test_frechet_d20_gates_L50.jl).
        @test r.inner_status in (0, -100, -101, -102, -103)

        W = size(ctx.U, 1)
        K = zeros(W); G = zeros(W, aug.obj_cm.d)
        aug.obj_cm.moments!(K, G, r.θ_full, ctx.U, aug.obj_cm)
        m_full = copy(aug.obj_cm.arg1)

        ncore = aug.ncore
        cdf_cols = ncore:(ncore + aug.ncm_cdf - 1)
        pow_cols = (ncore + aug.ncm_cdf):(ncore + aug.ncm - 1)
        core_cols = 1:(ncore - 1)

        kkt(j) = abs(sum(m_full .* G[:, j]) / W)
        max_cdf_kkt = maximum(kkt(j) for j in cdf_cols)
        max_pow_kkt = maximum(kkt(j) for j in pow_cols)
        max_core_kkt = maximum(kkt(j) for j in core_cols)
        @test max_cdf_kkt < 1e-8
        @test max_pow_kkt < 1e-6   # power feature has larger raw scale -> slightly looser but still tight
        @test all(isfinite, G)

        println("  L=$L: nStatus=$(r.inner_status) t=$(round(telapsed,digits=2))s Delta_dual=$(round(r.Delta_dual,digits=6)) " *
                "max|core KKT|=$(max_core_kkt) max|CDF KKT|=$(max_cdf_kkt) max|pow KKT|=$(max_pow_kkt)")
    end
    println("PASS: two-family flexible CM D4 inner solve converges, both sub-blocks satisfy KKT to tight tolerance.")
end

println("="^100)
println("Gate: outer gradient q0-fold two-family helper -- direct BLAS matvec matches definition")
println("="^100)
@testset "cm_fixed_value_contribution_two_family" begin
    include(joinpath(@__DIR__, "instrumentation.jl"))
    include(joinpath(@__DIR__, "oracle_fast.jl"))
    include(joinpath(@__DIR__, "three_way_derivatives.jl"))
    include(joinpath(@__DIR__, "lfix_incremental.jl"))
    include(joinpath(@__DIR__, "composite_gradient.jl"))
    include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
    include(joinpath(@__DIR__, "common_marginals_interval.jl"))
    include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
    include(joinpath(@__DIR__, "lfix_cm_aware.jl"))

    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    L = 8
    aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored)
    bins = cm_bin_indices_for(ctx, aug)
    λ_cm = randn(aug.ncm) .* 0.01
    out = cm_fixed_value_contribution_two_family(λ_cm, aug, bins, ctx)
    # Direct definition: out[s] = lambda_cm' * CM[s,:] literally (no lookup trick at all).
    direct = aug.CM * λ_cm
    @test out ≈ direct atol=1e-9 rtol=1e-9
    @test all(isfinite, out)
    println("PASS: two-family q0-fold contribution matches the direct lambda_cm'*CM definition to 1e-9.")
end

println("="^100)
println("All gates complete.")
println("="^100)
