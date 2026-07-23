# Section 6.1 (task brief): pure moment tests, D=3 and D=4, no KNITRO/inner-solve
# dependency -- validates mean_zero_cov_moments.jl's construction/residual/pair-index
# machinery in isolation before it is wired into any inner solve.
include(joinpath(@__DIR__, "mean_zero_cov_moments.jl"))
using Test, Random, LinearAlgebra, Statistics

Random.seed!(4021)

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

@testset "nu_feasible_interval" begin
    W, D = 5000, 4
    U = rand(W, D) .* 2.0 .+ [0.0 0.3 0.6 0.9]     # each origin's range [shift, shift+2) -- overlapping
    lo, hi = nu_feasible_interval(U)
    @test lo < hi
    @test lo == maximum(minimum(U, dims = 1))
    @test hi == minimum(maximum(U, dims = 1))
    # degenerate: force an empty interval (origin 1 max < origin 2 min)
    U2 = hcat(rand(W) .* 0.1, rand(W) .* 0.1 .+ 5.0)
    @test_throws ErrorException nu_feasible_interval(U2)
end

@testset "direct weighted calc vs moment-matrix implementation (D=4, arbitrary positive weights)" begin
    W, D = 2000, 4
    U = rand(W, D) .* 2 .+ 0.5
    m_raw = rand(W) .* 3 .+ 0.1
    m = m_raw .* (W / sum(m_raw))            # normalized so mean(m)=1 (sum(m)=W) -- this
                                              # codebase's CC recovered-weight convention
                                              # (recovered_mean_residuals divides by W, not
                                              # sum(m), exactly because a genuine solution's m
                                              # already satisfies mean(m)=1; see mean_m_resid in
                                              # cm_production_bundle.jl's archC_verified_state)
    ν = 1.3

    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; want_pair = true)
    @test Zraw == U
    @test size(Zpairraw, 2) == div(D * (D - 1), 2)
    pairs = packed_pair_index(D)
    for (k, (o, p)) in enumerate(pairs)
        @test Zpairraw[:, k] ≈ U[:, o] .* U[:, p]
    end

    Gmean = mean_columns_direct(Zraw, ν)
    Gpair = pair_columns(Zpairraw, ν)
    @test Gmean ≈ U .- ν
    for (k, (o, p)) in enumerate(pairs)
        @test Gpair[:, k] ≈ U[:, o] .* U[:, p] .- ν^2
    end

    # weighted moment via the moment-matrix path (dot(m, column)/W) vs a fully independent
    # direct per-origin/per-pair loop computation (no shared code with recovered_*_residuals)
    resid_mean = recovered_mean_residuals(m, Zraw, ν)
    resid_pair = recovered_pair_residuals(m, Zpairraw, ν)
    for o in 1:D
        direct = sum(m[s] * (U[s, o] - ν) for s in 1:W) / W
        @test resid_mean[o] ≈ direct atol=1e-9
    end
    for (k, (o, p)) in enumerate(pairs)
        direct = sum(m[s] * (U[s, o] * U[s, p] - ν^2) for s in 1:W) / W
        @test resid_pair[k] ≈ direct atol=1e-9
    end
end

@testset "mean-only arm never builds pair columns" begin
    W, D = 500, 4
    U = rand(W, D) .+ 0.5
    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; want_pair = false)
    @test Zpairraw === nothing
end

@testset "country means equal ν exactly when mean residuals are zero (D=4)" begin
    W, D = 3000, 4
    U = rand(W, D) .* 2 .+ 0.3
    ν = 1.1
    # build weights m such that ALL D mean residuals are exactly zero:
    # simplest construction -- uniform weight m=1 exactly reproduces the SAMPLE means, so
    # set ν = pooled sample mean isn't enough (per-origin means differ); instead solve for a
    # weight vector via a tiny exponential-tilt toward equal means (enough for a numerical check,
    # not a real CC solve -- this test only checks the RESIDUAL<->EQUALITY identity, not solver behavior).
    # Use Newton on a low-dim tilt parameter per origin acting multiplicatively... simpler:
    # directly verify the CONVERSE identity instead (residuals zero => means equal), which is
    # definitionally what recovered_mean_residuals checks; construct m analytically for D=2 (solvable
    # in closed form) and confirm here, then assert the general identity symbolically for D=4.
    m = ones(W)
    resid = recovered_mean_residuals(m, U, ν)
    means = [dot(m, @view(U[:, o])) / W for o in 1:D]
    @test all(isapprox.(resid, means .- ν; atol = 1e-9))
    # Genuine "residuals zero <=> means equal" check: perturb m slightly per column so that means
    # are NOT all equal, then confirm resid is nonzero and equals means .- ν  (i.e. the residual IS
    # exactly the equality gap, not some other unrelated quantity -- the actual invariant this task cares about).
    m2 = ones(W) .+ 0.01 .* randn(W)
    m2 .= max.(m2, 1e-6)
    resid2 = recovered_mean_residuals(m2, U, ν)
    means2 = [dot(m2, @view(U[:, o])) / W for o in 1:D]
    @test resid2 ≈ means2 .- ν atol=1e-9
    @test !all(isapprox.(resid2, 0.0; atol=1e-6))     # generic m2 does NOT satisfy the restriction
    # Now force EXACT equality by construction: pick ν = mean of column 1 under m2, then rescale
    # columns 2:D's weight contribution multiplicatively is not enough in general (nonlinear); instead
    # just build a synthetic U~ where all columns are IDENTICAL (a degenerate but fully rigorous
    # equal-means-by-construction case) and confirm the residual is then exactly (mean-ν) for every
    # column, simultaneously, under ANY weight vector -- the D=1-effective-column sanity check.
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
    Zraw, Zpairraw = build_raw_mean_pair_matrices(U; want_pair = true)
    pairs = packed_pair_index(D)

    # Case A: mean restriction does NOT hold (means differ from an arbitrary nu) -- pair residual
    # should NOT equal Cov(o,p) in general (it equals E[z_o z_p] - nu^2, which differs from
    # Cov = E[z_o z_p] - E[z_o]E[z_p] unless E[z_o]=E[z_p]=nu).
    ν_bad = 0.9
    resid_pair_bad = recovered_pair_residuals(m, Zpairraw, ν_bad)
    Σ = recovered_covariance_matrix(m, Zraw, D)
    for (k, (o, p)) in enumerate(pairs)
        @test !isapprox(resid_pair_bad[k], Σ[o, p]; atol = 1e-6)   # generically different
    end

    # Case B: force the mean restriction to hold exactly (identical columns, as above) so that
    # E[z_o]=E[z_p]=ν holds by construction, THEN the pair residual reduces exactly to Cov(o,p).
    Ueq = repeat(U[:, 1], 1, D)
    Zpair_eq = build_raw_mean_pair_matrices(Ueq; want_pair = true)[2]
    νeq = dot(m, @view(Ueq[:, 1])) / W
    resid_pair_eq = recovered_pair_residuals(m, Zpair_eq, νeq)
    Σeq = recovered_covariance_matrix(m, Ueq, D)
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

    # Feasible-set equivalence: {z_o = ν ∀o} exactly implies BOTH bases' columns vanish, and
    # conversely all-anchored-columns-vanish implies all-direct-columns-vanish (same feasible F).
    Ueq = repeat(fill(ν, W), 1, D)     # every draw exactly at the common mean (degenerate but exact)
    Gd = mean_columns_direct(Ueq, ν)
    Ga = mean_columns_anchored(Ueq, ν, refIndex1)
    @test all(iszero, Gd)
    @test all(iszero, Ga)

    # Now perturb ONE origin only (o != refIndex1): direct basis shows exactly one nonzero column
    # (that origin), anchored basis shows exactly one nonzero column too (the SAME origin's
    # contrast column) -- both bases flag the identical violation, at the identical origin.
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
    @test all(Σ[o, o] > 0 for o in 1:D)     # positive variance on the diagonal -- required before
                                            # correlation = covariance/sqrt(var_o var_p) is well-defined
    # degenerate counterexample: a constant column has zero variance, so covariance-with-it is
    # zero REGARDLESS of dependence -- this is exactly why the positive-variance check must run
    # before ever calling a zero-covariance result "zero correlation."
    Uconst = copy(U)
    Uconst[:, 1] .= 2.0
    Σc = recovered_covariance_matrix(m, Uconst, D)
    @test Σc[1, 1] ≈ 0.0 atol = 1e-12
end

@testset "envelope derivative sign/formula sanity (mock aug, no KNITRO)" begin
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

    aug_direct = (ncore_econ = ncore_econ, n_mean = n_mean, n_pair = npair,
                  nu_ref = Ref(ν), meanzc_basis = :direct, refIndex1 = 1)
    aug_anchor = (ncore_econ = ncore_econ, n_mean = n_mean, n_pair = npair,
                  nu_ref = Ref(ν), meanzc_basis = :anchored, refIndex1 = 1)
    aug_direct_meanonly = (ncore_econ = ncore_econ, n_mean = n_mean, n_pair = 0,
                  nu_ref = Ref(ν), meanzc_basis = :direct, refIndex1 = 1)

    mean_m = 1.0
    d_direct = d_delta_dual_d_nu(λ, aug_direct; mean_m = mean_m)
    expect_direct = -(sum(λ_mean_vals) + 2ν * sum(λ_pair_vals))
    @test d_direct ≈ expect_direct atol = 1e-12

    d_anchor = d_delta_dual_d_nu(λ, aug_anchor; mean_m = mean_m)
    expect_anchor = -(λ_mean_vals[1] + 2ν * sum(λ_pair_vals))   # only refIndex1=1 mean term survives
    @test d_anchor ≈ expect_anchor atol = 1e-12

    λ_meanonly = λ[1:ncore_econ-1+n_mean]
    d_meanonly = d_delta_dual_d_nu(λ_meanonly, aug_direct_meanonly; mean_m = mean_m)
    expect_meanonly = -sum(λ_mean_vals)
    @test d_meanonly ≈ expect_meanonly atol = 1e-12   # no pair term at all in the mean-only arm

    @test d_delta_dual_d_eta_nu(λ, aug_direct, ν; mean_m = mean_m) ≈ ν * d_direct atol = 1e-12
end

println("All Section 6.1 pure-moment tests passed.")
