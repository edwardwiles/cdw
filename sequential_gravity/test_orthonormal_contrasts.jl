# Unit tests for the orthonormal country-contrast reparameterization of the common-marginals
# restriction (common_marginals_moments.jl). No KNITRO / model data needed -- pure linear algebra
# plus the precompute function itself (only needs a W x D draw matrix).
#
#   julia --project=. sequential_gravity/test_orthonormal_contrasts.jl

using LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "common_marginals_moments.jl"))

const TOL = 1e-9
nfail = Ref(0)
function check(name::String, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    cond || (nfail[] += 1)
    @printf("  [%s] %s\n", status, name)
end
function check_close(name::String, a, b; tol = TOL)
    d = maximum(abs.(a .- b))
    check(name * @sprintf("  (max abs diff = %.2e)", d), d < tol)
end

"Anchored contrast matrix B (n=D-1 x D): row o (for origin index o=2..D in 1-based `origins`
order) has -1 in column 1 (the reference) and +1 in column o's own position."
function anchored_B(D::Int)
    n = D - 1
    B = zeros(n, D)
    for i in 1:n
        B[i, 1] = -1.0
        B[i, i+1] = 1.0
    end
    return B
end

for D in (2, 4, 20)
    println("\n=== D = $D ===")
    n = D - 1
    B = anchored_B(D)
    R = orthonormal_contrast_matrix(D)
    Rinv = orthonormal_contrast_matrix_inverse(D)
    C = R * B

    # 1. Matrix identities
    check_close("R symmetric", R, R')
    check_close("R * Rinv = I_n", R * Rinv, Matrix(I, n, n))
    check_close("C * 1_D = 0", C * ones(D), zeros(n); tol = 1e-8)
    check_close("C * C' = I_n", C * C', Matrix(I, n, n))
    check("rank(C) = D-1", rank(C) == n)
    check_close("R*B*B'*R' = I_n  (RBB'R'=I)", R * (B * B') * R', Matrix(I, n, n))

    # 2. Same contrast space: B'(BB')^{-1}B = C'C = I_D - (1/D) 1_D 1_D'
    lhs = B' * ((B * B') \ B)
    rhs = C' * C
    proj = Matrix(I, D, D) .- (1 / D) .* ones(D, D)
    check_close("B'(BB')^{-1}B == C'C", lhs, rhs)
    check_close("C'C == I_D - (1/D)*1_D*1_D'", rhs, proj; tol = 1e-8)

    # 3. Same zero restrictions: Bv=0 <=> Cv=0, for random test vectors AND for v in row space of B'
    Random.seed!(20260716 + D)
    for _ in 1:5
        v = randn(D)
        # project v onto null(B) (i.e. v with all entries equal) to get a genuine Bv=0 example
        v_null = fill(mean(v), D)
        check_close("Bv=0 => Cv=0 (v in null(B))", C * v_null, zeros(n); tol = 1e-8)
        # a generic v is (almost surely) NOT in null(B); confirm Bv=0 <=> Cv=0 by checking both
        # sides are simultaneously (non-)zero, not that they're equal (B,C map to different bases)
        bv_zero = norm(B * v) < 1e-8
        cv_zero = norm(C * v) < 1e-8
        check("Bv=0 <=> Cv=0 (generic v, both nonzero here)", bv_zero == cv_zero)
    end

    # 4. Dual-index equivalence: (lambda_old)'h_old == (lambda_new)'h_new
    #    h_new = R h_old, lambda_new = R^{-T} lambda_old = R^{-1} lambda_old (R symmetric)
    for _ in 1:5
        h_old = randn(n)
        λ_old = randn(n)
        h_new = R * h_old
        λ_new = Rinv' * λ_old   # = Rinv * λ_old since R (hence Rinv) symmetric
        check_close("(λ_old)'h_old == (λ_new)'h_new", [dot(λ_old, h_old)], [dot(λ_new, h_new)])
    end

    # 5. Reference covariance diagnostic: simulate iid reference draws, compare empirical
    #    covariance of the anchored vs orthonormal CM block at one threshold.
    Wsim = 200_000
    Usim = rand(Wsim, D)  # Uniform(0,1) stand-in (only the CDF-indicator feature is exercised)
    L = 2   # L=1 hits a pre-existing edge case in the quantile-range formula (range(1,0,length=1)
            # is invalid regardless of contrasts choice) -- not this task's concern, just avoided here
    CM_anch, _, origins = precalc_common_marginals_cdf(Usim, 1, L; contrasts = :anchored)
    CM_orth, _, _        = precalc_common_marginals_cdf(Usim, 1, L; contrasts = :orthonormal)
    # isolate the FIRST threshold's block (columns 1:nO) for the covariance check -- mixing both
    # thresholds' columns would conflate genuine cross-threshold correlation (real, expected) with
    # the cross-COUNTRY correlation this test targets.
    Σ_anch = cov(CM_anch[:, 1:n])
    Σ_orth = cov(CM_orth[:, 1:n])
    # anchored: Sigma ~ c*(I + 11'); orthonormal: Sigma ~ c*I (same overall per-origin variance
    # scale c, since R is norm-preserving on average -- check off-diagonal shrinks close to 0 and
    # the diagonal stays close to the SAME scale, rather than asserting an exact proportionality
    # constant, which is noisy at finite Wsim).
    offdiag_anch = maximum(abs.(Σ_anch - Diagonal(Σ_anch)))
    offdiag_orth = maximum(abs.(Σ_orth - Diagonal(Σ_orth)))
    @printf("  [INFO] anchored  cov: diag mean=%.4f  max |offdiag|=%.4f\n", mean(diag(Σ_anch)), offdiag_anch)
    @printf("  [INFO] orthonorm cov: diag mean=%.4f  max |offdiag|=%.4f\n", mean(diag(Σ_orth)), offdiag_orth)
    if n == 1
        check("(D=2, n=1: no cross-country pair exists, off-diagonal check vacuous)", true)
    else
        check("orthonormal off-diagonal << anchored off-diagonal (decorrelated)", offdiag_orth < 0.15 * offdiag_anch)
    end
    check("orthonormal off-diagonal ~ 0 (relative to its own diagonal scale)", offdiag_orth < 0.05 * mean(diag(Σ_orth)))

    # 6. Same total moment count regardless of contrasts (n_cm_moments unaffected)
    check("n_cm_moments unaffected by contrasts choice",
          n_cm_moments(D, L) == n * L == size(CM_anch, 2) == size(CM_orth, 2))

    # 7. cm_block_to_anchored_residuals round-trips correctly: rotate a random anchored vector by
    #    R, then invert back, and it should reproduce the original.
    h_anchored = randn(n)
    h_rotated = R * h_anchored
    recovered = cm_block_to_anchored_residuals(h_rotated, D, 1, n; contrasts = :orthonormal)
    check_close("cm_block_to_anchored_residuals inverts R exactly", vec(recovered), h_anchored)
    recovered_id = cm_block_to_anchored_residuals(h_anchored, D, 1, n; contrasts = :anchored)
    check_close("cm_block_to_anchored_residuals is identity under :anchored", vec(recovered_id), h_anchored)
end

println("\n" * "="^60)
if nfail[] == 0
    println("ALL TESTS PASSED")
else
    println("$(nfail[]) TEST(S) FAILED")
    exit(1)
end
