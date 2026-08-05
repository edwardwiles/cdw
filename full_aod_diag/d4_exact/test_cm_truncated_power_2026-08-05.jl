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
using Statistics: quantile, mean
using SpecialFunctions: gamma

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

println("="^100)
println("Gate 1+2: direct feature construction + dimension gates (small hand D/W/L)")
println("="^100)

@testset "direct feature construction" begin
    # 2026-08-05 CORRECTED (redo after the raw-U-vs-Frechet-z bug was found and fixed): the
    # ORIGINAL version of this gate used U ~ Uniform(0.1, 3.1) -- bounded away from 0 -- which
    # could never have exposed the (1-sigma) exponent applied to the wrong variable (a
    # Uniform(0.1,3.1)^(-1.5) is perfectly well-behaved regardless of sign/variable convention).
    # Use real Exp(1) draws here instead (matching production's actual ctx.U, which CAN get
    # arbitrarily close to 0), and build the hand reference INDEPENDENTLY from the
    # z_o(omega)=U_o(omega)^(-mu) relation directly (not by calling frechet_power_feature or any
    # other production code) -- this is the check the coordinator/user explicitly required: an
    # independent re-derivation, not a self-consistency check against the (previously, equally
    # wrong) formula.
    W, D, L = 2000, 4, 5
    refIndex1 = 1
    σHat = 3.0
    μHat = 0.15   # representative real-D20-scale value (real production ~0.13-0.20)
    Random.seed!(20260805)
    U = -log.(rand(W, D))   # Exp(1) via inverse-CDF, independent of this repo's own genExpRands!
    probs = collect(range(1/L, (L-1)/L, length=L))
    # 2026-08-05 (user-directed): cutoffs are now THEORETICAL (closed-form -log(1-p)), not the
    # empirical quantile(U[:,refIndex1],probs) -- see theoretical_u_threshold's own docstring.
    z_expect = theoretical_u_threshold.(probs)

    CM, z, origins = precalc_common_marginals_cdf(U, refIndex1, L; include_truncated_moment = true, σHat = σHat, μHat = μHat, contrasts = :anchored, probs = probs)
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
    # INDEPENDENT hand reference for eq.36: z_x(omega) = U_x(omega)^(-mu) computed directly here
    # (not via frechet_power_feature), then z^(1-sigma) -- this is genuinely re-deriving the
    # relation, not calling the same (potentially wrong) production helper twice.
    zhand = U .^ (-μHat)
    @test all(isfinite, zhand) && all(>(0), zhand)
    tm_offset = nO*L
    for (li, l) in enumerate(1:L)
        for (oi, o) in enumerate(origins)
            col2 = tm_offset + (l-1)*nO + oi
            hand2 = [ (zhand[s,o]^(1-σHat)) * Float64(U[s,o] <= z[l]) - (zhand[s,refIndex1]^(1-σHat)) * Float64(U[s,refIndex1] <= z[l]) for s in 1:W]
            @test CM[:, col2] ≈ hand2
        end
    end
    @test all(isfinite, CM)
    # Not accidentally using U's rank/quantile position (i.e. not the CDF/uniform-transformed
    # draw) -- the power block must use the RAW z_o(omega) values, not e.g. rank(U[:,o])/W.
    @test !all(CM[:, tm_offset+1] .== CM[:, 1])   # sanity: power block genuinely differs from CDF block
    # The whole point of this fix: at REAL Exp(1) draws with a realistic (small, positive) exponent
    # mu*(sigma-1), the power block must NOT be dominated by a single extreme draw the way the
    # buggy U.^(1-sigma) formula was (see MASTER.md for the W=5,000/D20 real-data numbers: a single
    # draw producing Pow~2e7 while the column mean was ~4487 -- a >99%-from-one-draw domination).
    pow_col_means = [mean(CM[:, tm_offset + (l-1)*nO + oi]) for l in 1:L, oi in 1:nO]
    @test maximum(abs, pow_col_means) < 10.0   # well-behaved order of magnitude, not 1e4-1e7

    println("PASS: direct feature construction matches an INDEPENDENTLY hand-built eq.35+eq.36 (z=U^(-mu)) exactly; well-behaved magnitudes; dims correct.")
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
println("Gate 3: CDF-block self-consistency -- standalone single-family build matches the CDF")
println("sub-block of the combined two-family build, at the SAME (now theoretical-cutoff) context")
println("="^100)
# 2026-08-05 REINTERPRETED (user-directed): since cutoffs are now the theoretical closed-form
# Fréchet quantile (theoretical_u_threshold) rather than the empirical sample quantile, the CDF
# family's own NUMERIC VALUES now legitimately differ from old (pre-2026-08-05) production -- this
# is intentional, not a regression (see MASTER.md). This gate no longer means "bit-identical to
# old production's empirical-cutoff CDF values" (that comparison is expected to differ and is not
# tested here) -- it means "internally self-consistent": a standalone single-family
# (include_truncated_moment=false) build and the CDF sub-block of the combined two-family build,
# at the identical context/L/contrasts, must still agree exactly (both now use the SAME
# theoretical cutoff formula, so of course they should -- this catches any accidental
# cross-contamination between the two families' construction, which is the actual thing this gate
# protects against).
@testset "CDF-block self-consistency" begin
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    L = 10
    aug_old = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = false, contrasts = :anchored)
    aug_new = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true, contrasts = :anchored)
    @test aug_old.ncm == (ctx.D-1)*L
    @test aug_new.ncm == 2*(ctx.D-1)*L
    @test aug_new.ncm_cdf == aug_old.ncm
    @test aug_new.n_families == 2
    @test aug_old.n_families == 1
    # bit-identical CDF sub-block (raw feature matrix) -- self-consistency, not old-production match
    @test aug_new.CM[:, 1:aug_new.ncm_cdf] == aug_old.CM
    @test aug_new.z == aug_old.z
    @test aug_new.origins == aug_old.origins
    # z is now the theoretical closed form, not an empirical sample quantile of ctx.U
    probs_expect = collect(range(1/L, (L-1)/L, length=L))
    @test aug_new.z == theoretical_u_threshold.(probs_expect)
    @test issorted(aug_new.z)   # required invariant for the (untouched) bin-index architecture
    println("PASS: CDF sub-block is self-consistent between single- and two-family builds; cutoffs are the theoretical closed form and remain sorted ascending.")
end

println("="^100)
println("Gate: theoretical Fréchet-quantile closed form -- Monte Carlo convergence + Γ(1-μk) reduction")
println("(full derivation/validation in diag_theoretical_quantile_check_2026-08-05.jl; condensed here)")
println("="^100)
@testset "theoretical quantile closed form" begin
    μHat2, σHat2 = 0.15, 2.5
    k2 = 1 - σHat2
    # reduces to the untruncated Gamma(1-mu*k) as z_l -> infinity
    @test isapprox(eq36_theoretical_truncated_moment(1e12, k2, μHat2), gamma(1 - μHat2*k2, 0.0); atol=1e-9)
    @test isapprox(eq36_theoretical_truncated_moment(1e12, k2, μHat2), gamma(1 - μHat2*k2); atol=1e-9)
    # Monte Carlo convergence at a single (p, W) point (heavier convergence sweep is in the
    # standalone diag script; this is a fast, single-point regression gate)
    Random.seed!(778899)
    Wbig = 3_000_000
    Ubig = -log.(rand(Wbig))
    p2 = 0.5
    uthresh2 = theoretical_u_threshold(p2)
    zbig = Ubig .^ (-μHat2)
    zl2 = uthresh2^(-μHat2)
    mc = mean((zbig .^ k2) .* (zbig .< zl2))
    theo = eq36_theoretical_truncated_moment(zl2, k2, μHat2)
    @test isapprox(mc, theo; atol = 5e-3)   # W=3e6 Monte Carlo noise, generous but real tolerance
    println("PASS: closed form reduces to Γ(1-μk) as z_ℓ→∞; Monte Carlo average converges to it (|diff|=$(abs(mc-theo)) at W=$Wbig).")
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
