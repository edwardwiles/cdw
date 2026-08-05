# Pure moment tests, D=3/4/5/20, no KNITRO/inner-solve dependency -- validates
# cm_meanzc_moments.jl's construction/residual/pair-index machinery in
# isolation before it is wired into any inner solve. Adapted from the
# independently-audited prototype (archive/fullA-cm-mean-zc-prototype-2026-07-22,
# c40_test_meanzc_pure_moments.jl) to this integration's Ref-free ν-threading
# API (d_delta_dual_d_nu_vec/d_delta_dual_d_eta_nu_vec take νvec as an explicit
# argument, not aug.nu_ref[]), and generalized from K=1 (equal means [+ pairwise
# zero covariance]) to K_mean/K_pair power-level moments E[z_o^k]=ν_k,
# E[z_o^k z_p^k]=ν_k^2. K_mean=1,K_pair=0/1 reproduce the original two arms
# exactly (see "K=1 regression" testsets below); a K_mean=2,K_pair=2 testset
# validates the generalization itself.
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
using Test, Random, LinearAlgebra, Statistics

Random.seed!(4021)

# Fixed test exponent for the Frechet productivity transform z = U^(-mu); used everywhere below
# a ZC feature matrix is built from raw U so tests exercise the SAME z^k = U^(-mu*k) formula as
# production (frechet_power_feature), not the pre-fix U^k. 0 < mu_test*3 < 1 so k=1,2,3 all stay
# comfortably inside the Gamma(1-mu*k) finiteness region tested elsewhere.
const MU_TEST = 0.3

@testset "packed_pair_index / round-trip mapping" begin
    for D in (3, 4, 5, 20)
        pairs = packed_pair_index(D)
        @test length(pairs) == div(D * (D - 1), 2)
        @test length(unique(pairs)) == length(pairs)          # no duplicates
        @test all(o < p for (o, p) in pairs)                  # canonical order o<p, no (p,o) dupes
        for (k, (o, p)) in enumerate(pairs)
            @test pair_lin_to_oi(k, D) == (o, p)
            @test pair_oi_to_lin(o, p, D) == k
            @test pair_oi_to_lin(p, o, D) == k                 # order-independent lookup
        end
    end
end

@testset "n_meanzc_moments (K_mean, K_pair)" begin
    for D in (3, 4, 5, 20)
        @test n_meanzc_moments(D, 1, 0) == D                                    # old :cm_plus_equal_means
        @test n_meanzc_moments(D, 1, 1) == D + div(D * (D - 1), 2)              # old :cm_plus_equal_means_zero_covariance
        @test n_meanzc_moments(D, 2, 0) == 2D
        @test n_meanzc_moments(D, 2, 1) == 2D + div(D * (D - 1), 2)
        @test n_meanzc_moments(D, 2, 2) == 2D + 2 * div(D * (D - 1), 2)
    end
    @test_throws ErrorException n_meanzc_moments(4, 0, 0)      # K_mean must be >= 1
    @test_throws ErrorException n_meanzc_moments(4, 1, 2)      # K_pair must be <= K_mean
    @test_throws ErrorException n_meanzc_moments(4, 1, -1)     # K_pair must be >= 0
end

@testset "meanzc_extension_to_K sugar" begin
    @test meanzc_extension_to_K(:cm_plus_equal_means) == (1, 0)
    @test meanzc_extension_to_K(:cm_plus_equal_means_zero_covariance) == (1, 1)
    @test_throws ErrorException meanzc_extension_to_K(:cm_only)
    @test_throws ErrorException meanzc_extension_to_K(:bogus)
end

@testset "nu_feasible_interval" begin
    W, D = 5000, 4
    U = rand(W, D) .* 2.0 .+ [0.0 0.3 0.6 0.9]     # each origin's range [shift, shift+2) -- overlapping
    Z = frechet_power_feature(U, 1, MU_TEST)
    lo, hi = nu_feasible_interval(U, 1; μ = MU_TEST)
    @test lo < hi
    @test lo == maximum(minimum(Z, dims = 1))
    @test hi == minimum(maximum(Z, dims = 1))
    # degenerate: force an empty interval (origin 1 max < origin 2 min) -- z=U^(-mu) is strictly
    # decreasing in U for mu>0, so disjoint U-ranges stay disjoint (order-reversed) in z-space too.
    U2 = hcat(rand(W) .* 0.1, rand(W) .* 0.1 .+ 5.0)
    @test_throws ErrorException nu_feasible_interval(U2; μ = MU_TEST)
end

@testset "direct weighted calc vs moment-matrix implementation (D=4, arbitrary positive weights)" begin
    W, D = 2000, 4
    U = rand(W, D) .* 2 .+ 0.5
    m_raw = rand(W) .* 3 .+ 0.1
    m = m_raw .* (W / sum(m_raw))            # normalized so mean(m)=1 (sum(m)=W)
    ν = 1.3

    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; μ = MU_TEST, want_pair = true)
    Zexpect = frechet_power_feature(U, 1, MU_TEST)
    @test Zraw ≈ Zexpect
    @test !isapprox(Zraw, U)   # old U^k formula must NOT be what's returned (k=1 old: U; new: U^(-mu))
    @test size(Zpairraw, 2) == div(D * (D - 1), 2)
    pairs = packed_pair_index(D)
    for (k, (o, p)) in enumerate(pairs)
        @test Zpairraw[:, k] ≈ Zexpect[:, o] .* Zexpect[:, p]
    end

    Gmean = mean_columns_direct(Zraw, ν)
    Gpair = pair_columns(Zpairraw, ν)
    @test Gmean ≈ Zexpect .- ν
    for (k, (o, p)) in enumerate(pairs)
        @test Gpair[:, k] ≈ Zexpect[:, o] .* Zexpect[:, p] .- ν^2
    end

    resid_mean = recovered_mean_residuals(m, Zraw, ν)
    resid_pair = recovered_pair_residuals(m, Zpairraw, ν)
    for o in 1:D
        direct = sum(m[s] * (Zexpect[s, o] - ν) for s in 1:W) / W
        @test resid_mean[o] ≈ direct atol=1e-9
    end
    for (k, (o, p)) in enumerate(pairs)
        direct = sum(m[s] * (Zexpect[s, o] * Zexpect[s, p] - ν^2) for s in 1:W) / W
        @test resid_pair[k] ≈ direct atol=1e-9
    end
end

@testset "level-k (k=2) moment construction and build_raw_mean_pair_matrix_levels" begin
    W, D = 2000, 4
    U = rand(W, D) .* 2 .+ 0.5
    ν2 = 1.9

    Z1expect = frechet_power_feature(U, 1, MU_TEST)
    Z2expect = frechet_power_feature(U, 2, MU_TEST)

    Z2, Zpair2 = build_raw_mean_pair_matrices(U, 2; μ = MU_TEST, want_pair = true)
    @test Z2 ≈ Z2expect
    @test Z2 ≈ Z1expect .^ 2                          # z_o^2 == (z_o^1)^2, no double transform
    @test !isapprox(Z2, U .^ 2)                        # old (wrong) formula must not reappear
    pairs = packed_pair_index(D)
    for (k, (o, p)) in enumerate(pairs)
        @test Zpair2[:, k] ≈ Z2expect[:, o] .* Z2expect[:, p]
    end
    Gmean2 = mean_columns_direct(Z2, ν2)
    @test Gmean2 ≈ Z2expect .- ν2

    # build_raw_mean_pair_matrix_levels stacks K_mean/K_pair levels consistently with the
    # single-level constructor above
    Zraw_all, Zpairraw_all = build_raw_mean_pair_matrix_levels(U, 2, 2; μ = MU_TEST)
    @test length(Zraw_all) == 2 && length(Zpairraw_all) == 2
    @test Zraw_all[1] ≈ Z1expect
    @test Zraw_all[2] ≈ Z2
    @test Zpairraw_all[2] ≈ Zpair2

    Zraw_all_meanonly, Zpairraw_all_meanonly = build_raw_mean_pair_matrix_levels(U, 2, 0; μ = MU_TEST)
    @test length(Zpairraw_all_meanonly) == 0    # no pair matrices built at all when K_pair=0
end

@testset "mean-only arm never builds pair columns" begin
    W, D = 500, 4
    U = rand(W, D) .+ 0.5
    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; μ = MU_TEST, want_pair = false)
    @test Zpairraw === nothing
end

@testset "country means equal ν exactly when mean residuals are zero (D=4)" begin
    W, D = 3000, 4
    U = rand(W, D) .* 2 .+ 0.3
    ν = 1.1
    m = ones(W)
    resid = recovered_mean_residuals(m, U, ν)
    means = [dot(m, @view(U[:, o])) / W for o in 1:D]
    @test all(isapprox.(resid, means .- ν; atol = 1e-9))
    m2 = ones(W) .+ 0.01 .* randn(W)
    m2 .= max.(m2, 1e-6)
    resid2 = recovered_mean_residuals(m2, U, ν)
    means2 = [dot(m2, @view(U[:, o])) / W for o in 1:D]
    @test resid2 ≈ means2 .- ν atol=1e-9
    @test !all(isapprox.(resid2, 0.0; atol=1e-6))     # generic m2 does NOT satisfy the restriction
    Ueq = repeat(U[:, 1], 1, D)
    νeq = dot(m2, @view(Ueq[:, 1])) / W
    resideq = recovered_mean_residuals(m2, Ueq, νeq)
    @test all(isapprox.(resideq, 0.0; atol = 1e-9))
end

@testset "each unordered-pair residual equals the corresponding covariance when means hold (D=4)" begin
    W, D = 4000, 4
    U = rand(W, D) .* 1.5 .+ 0.2
    m = ones(W) .+ 0.02 .* randn(W)
    m .= max.(m, 1e-6)
    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; μ = MU_TEST, want_pair = true)
    pairs = packed_pair_index(D)

    ν_bad = 0.9
    resid_pair_bad = recovered_pair_residuals(m, Zpairraw, ν_bad)
    Σ = recovered_covariance_matrix(m, Zraw, D)
    for (k, (o, p)) in enumerate(pairs)
        @test !isapprox(resid_pair_bad[k], Σ[o, p]; atol = 1e-6)   # generically different
    end

    Ueq = repeat(U[:, 1], 1, D)
    Zeq = frechet_power_feature(Ueq, 1, MU_TEST)
    Zpair_eq = build_raw_mean_pair_matrices(Ueq; μ = MU_TEST, want_pair = true)[2]
    νeq = dot(m, @view(Zeq[:, 1])) / W
    resid_pair_eq = recovered_pair_residuals(m, Zpair_eq, νeq)
    Σeq = recovered_covariance_matrix(m, Zeq, D)
    for (k, (o, p)) in enumerate(pairs)
        @test resid_pair_eq[k] ≈ Σeq[o, p] atol = 1e-9
    end
end

@testset "direct vs anchored mean basis: equivalent zero-residual sets (D=4)" begin
    W, D = 3000, 4
    U = rand(W, D) .* 1.2 .+ 0.4
    refIndex1 = 2
    ν = 1.0
    d_direct = d_mean_dnu_direct(D)
    d_anchor = d_mean_dnu_anchored(D, refIndex1)
    @test d_direct == fill(-1.0, D)
    @test d_anchor[refIndex1] == -1.0
    @test all(d_anchor[o] == 0.0 for o in 1:D if o != refIndex1)

    Ueq = repeat(fill(ν, W), 1, D)
    Gd = mean_columns_direct(Ueq, ν)
    Ga = mean_columns_anchored(Ueq, ν, refIndex1)
    @test all(iszero, Gd)
    @test all(iszero, Ga)

    Upert = copy(Ueq)
    bad_origin = (refIndex1 % D) + 1
    Upert[:, bad_origin] .+= 0.37
    Gd2 = mean_columns_direct(Upert, ν)
    Ga2 = mean_columns_anchored(Upert, ν, refIndex1)
    nz_direct = [o for o in 1:D if !all(iszero, Gd2[:, o])]
    nz_anchor = [o for o in 1:D if !all(iszero, Ga2[:, o])]
    @test nz_direct == [bad_origin]
    @test nz_anchor == [bad_origin]
end

@testset "positive-variance diagnostic (zero covariance != zero correlation without it)" begin
    W, D = 3000, 4
    U = rand(W, D) .* 1.5 .+ 0.3
    m = ones(W)
    Σ = recovered_covariance_matrix(m, U, D)
    @test all(Σ[o, o] > 0 for o in 1:D)
    Uconst = copy(U)
    Uconst[:, 1] .= 2.0
    Σc = recovered_covariance_matrix(m, Uconst, D)
    @test Σc[1, 1] ≈ 0.0 atol = 1e-12
end

@testset "envelope derivative sign/formula sanity (mock aug, no KNITRO, ν explicit not Ref)" begin
    D = 4
    ncore_econ = 50   # arbitrary, only ordering matters here
    n_mean = D
    npair = div(D * (D - 1), 2)
    ν = 1.2
    λ = zeros(ncore_econ - 1 + n_mean + npair)
    λ_mean_vals = [0.3, -0.1, 0.2, 0.05]
    λ_pair_vals = fill(0.02, npair)
    λ[ncore_econ:ncore_econ+n_mean-1] .= λ_mean_vals
    λ[ncore_econ+n_mean:ncore_econ+n_mean+npair-1] .= λ_pair_vals

    mock_Zraw = [zeros(1, D)]   # only size(.,2)==D matters for d_delta_dual_d_nu_vec here
    aug_direct = (ncore_econ = ncore_econ, K_mean = 1, K_pair = 1, Zraw_all = mock_Zraw,
                  meanzc_basis = :direct, refIndex1 = 1)
    aug_anchor = (ncore_econ = ncore_econ, K_mean = 1, K_pair = 1, Zraw_all = mock_Zraw,
                  meanzc_basis = :anchored, refIndex1 = 1)
    aug_direct_meanonly = (ncore_econ = ncore_econ, K_mean = 1, K_pair = 0, Zraw_all = mock_Zraw,
                  meanzc_basis = :direct, refIndex1 = 1)

    mean_m = 1.0
    d_direct = d_delta_dual_d_nu_vec(λ, aug_direct, [ν]; mean_m = mean_m)
    expect_direct = -(sum(λ_mean_vals) + 2ν * sum(λ_pair_vals))
    @test only(d_direct) ≈ expect_direct atol = 1e-12

    d_anchor = d_delta_dual_d_nu_vec(λ, aug_anchor, [ν]; mean_m = mean_m)
    expect_anchor = -(λ_mean_vals[1] + 2ν * sum(λ_pair_vals))   # only refIndex1=1 mean term survives
    @test only(d_anchor) ≈ expect_anchor atol = 1e-12

    λ_meanonly = λ[1:ncore_econ-1+n_mean]
    d_meanonly = d_delta_dual_d_nu_vec(λ_meanonly, aug_direct_meanonly, [ν]; mean_m = mean_m)
    expect_meanonly = -sum(λ_mean_vals)
    @test only(d_meanonly) ≈ expect_meanonly atol = 1e-12   # no pair term at all in the mean-only arm

    @test only(d_delta_dual_d_eta_nu_vec(λ, aug_direct, [ν]; mean_m = mean_m)) ≈ ν * only(d_direct) atol = 1e-12
end

@testset "envelope derivative sign/formula sanity: K_mean=2, K_pair=2 generalization (mock aug)" begin
    D = 4
    ncore_econ = 50
    npair = div(D * (D - 1), 2)
    K_mean, K_pair = 2, 2
    ν = [1.2, 1.6]
    λlen = ncore_econ - 1 + K_mean * D + K_pair * npair
    λ = zeros(λlen)
    λ_mean_vals = [ [0.3, -0.1, 0.2, 0.05], [0.11, -0.07, 0.13, 0.02] ]     # per level k
    λ_pair_vals = [ fill(0.02, npair), fill(-0.01, npair) ]                # per level k
    mean_start = ncore_econ
    pair_start0 = ncore_econ + K_mean * D
    for k in 1:K_mean
        λ[mean_start+(k-1)*D : mean_start+k*D-1] .= λ_mean_vals[k]
    end
    for k in 1:K_pair
        λ[pair_start0+(k-1)*npair : pair_start0+k*npair-1] .= λ_pair_vals[k]
    end
    mock_Zraw = [zeros(1, D), zeros(1, D)]
    aug = (ncore_econ = ncore_econ, K_mean = K_mean, K_pair = K_pair, Zraw_all = mock_Zraw,
           meanzc_basis = :direct, refIndex1 = 1)

    mean_m = 1.0
    d = d_delta_dual_d_nu_vec(λ, aug, ν; mean_m = mean_m)
    @test length(d) == K_mean
    for k in 1:K_mean
        expect_k = -(sum(λ_mean_vals[k]) + 2ν[k] * sum(λ_pair_vals[k]))
        @test d[k] ≈ expect_k atol = 1e-12
    end
    # block-diagonal-in-k check: perturbing ν_1 alone must not change level 2's component, and
    # vice versa (each level's derivative depends only on its OWN ν_k, per the file header's
    # block-diagonal Jacobian argument)
    ν_perturbed_1 = [ν[1] + 0.3, ν[2]]
    d_p1 = d_delta_dual_d_nu_vec(λ, aug, ν_perturbed_1; mean_m = mean_m)
    @test d_p1[2] ≈ d[2] atol = 1e-12
    @test !isapprox(d_p1[1], d[1]; atol = 1e-12)

    d_eta = d_delta_dual_d_eta_nu_vec(λ, aug, ν; mean_m = mean_m)
    @test d_eta ≈ ν .* d atol = 1e-12
end

@testset "malformed extension/configuration rejection" begin
    @test_throws ErrorException n_meanzc_moments(4, 1, 2)
    @test_throws ErrorException meanzc_extension_to_K(:cm_only)
end

println("All pure-moment tests passed.")
