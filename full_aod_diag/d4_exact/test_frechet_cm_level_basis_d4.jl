# D=4 basis-equivalence gate (task Part VI §16) for the CM-plus-level moment CONSTRUCTION layer
# (Part II). Checks, at real D=4 production draws:
#   1. C'q_l and u'q_l (computed via the CM+level dense construction) match the direct
#      country-by-country q_l = f_l(omega) - F*(H_l)*ones(D) exactly (basis-equivalence proof,
#      math doc §3, verified numerically here rather than only algebraically).
#   2. The bin-lookup fast path (fill_frechet_level_columns_from_bins!) matches the dense reference
#      (precalc_frechet_level_dense) to machine precision.
#   3. Total column count is exactly D*L (task §3's "exactly DL restrictions" claim).
#   4. M=[C u] (or [CR u]) reconstructs q_l exactly via its inverse (round-trip check).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "cm_frechet_level.jl"))
using LinearAlgebra
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
U = ctx.U
refIndex1 = ctx.γ.refIndex1
L = 10

npass = 0
nfail = 0
function check(name, cond; tol = nothing)
    global npass, nfail
    ok = cond
    if ok
        npass += 1
        println("  PASS  ", name)
    else
        nfail += 1
        println("  FAIL  ", name)
    end
end

for contrasts in (:anchored, :orthonormal)
    println("== contrasts = $contrasts ==")

    CM, z, origins = precalc_common_marginals_cdf(U, refIndex1, L; contrasts = contrasts)
    level_targets = frechet_level_targets(D, L)
    level_probs = frechet_level_probs(L)
    check("level_targets == sqrt(D)*probs", isapprox(level_targets, sqrt(D) .* level_probs; atol=1e-14))

    LEVEL = precalc_frechet_level_dense(U, z, D, level_targets)
    check("LEVEL size == (W,L)", size(LEVEL) == (size(U,1), L))

    # ---- direct country-by-country q_l reference ----
    W = size(U, 1)
    R = contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing
    C = zeros(D, D - 1)
    oi = 0
    for o in 1:D
        o == refIndex1 && continue
        oi += 1
        C[o, oi] = 1.0
        C[refIndex1, oi] = -1.0
    end
    Mmat = R === nothing ? hcat(C, ones(D) / sqrt(D)) : hcat(C * R, ones(D) / sqrt(D))
    check("M = [C(R) u] is D x D", size(Mmat) == (D, D))
    check("M is full rank (D)", rank(Mmat) == D)
    Minv = inv(Mmat)

    max_cm_err = 0.0
    max_level_err = 0.0
    max_roundtrip_err = 0.0
    for l in 1:L
        fl = Float64.(U .<= z[l])              # W x D indicator matrix f_l(omega)
        ql = fl .- level_probs[l]               # W x D, direct fixed-Frechet residual q_l(omega) = f_l - F*(H_l)*1
        cm_cols = (l - 1) * (D - 1) + 1 : l * (D - 1)
        cm_from_construction = @view CM[:, cm_cols]
        level_from_construction = @view LEVEL[:, l]
        # C(R)'q_l should equal the CM block for threshold l
        cm_direct = ql * (R === nothing ? C : C * R)
        max_cm_err = max(max_cm_err, maximum(abs.(cm_direct .- cm_from_construction)))
        # u'q_l should equal the level column for threshold l
        level_direct = ql * (ones(D) / sqrt(D))
        max_level_err = max(max_level_err, maximum(abs.(level_direct .- level_from_construction)))
        # round-trip: stacked[s,:] = M' * ql[s,:] (as a column), i.e. in row-vector form
        # stacked_row = ql_row * M  =>  ql_row = stacked_row * inv(M).
        stacked = hcat(cm_from_construction, level_from_construction)   # W x D
        reconstructed = stacked * Minv
        max_roundtrip_err = max(max_roundtrip_err, maximum(abs.(reconstructed .- ql)))
    end
    check("max|C'q_l - CM_construction| < 1e-10", max_cm_err < 1e-10)
    check("max|u'q_l - LEVEL_construction| < 1e-10", max_level_err < 1e-10)
    check("max|round-trip reconstruction - q_l| < 1e-8", max_roundtrip_err < 1e-8)
    @printf("  max_cm_err=%.3e max_level_err=%.3e max_roundtrip_err=%.3e\n", max_cm_err, max_level_err, max_roundtrip_err)

    # ---- bin-lookup fast path vs dense reference ----
    Bidx = Int.(compute_bin_indices(U, z))   # production convention (cm_production_bundle.jl): force Int, matches fill_cm_columns_from_bins!'s own signature
    LEVEL_bins = Matrix{Float64}(undef, W, L)
    fill_frechet_level_columns_from_bins!(LEVEL_bins, Bidx, D, L, level_targets)
    max_bin_err = maximum(abs.(LEVEL_bins .- LEVEL))
    check("bin-lookup LEVEL matches dense LEVEL to machine precision", max_bin_err < 1e-12)
    @printf("  max_bin_err=%.3e\n", max_bin_err)

    # ---- combined obj: total columns == D*L ----
    aug = build_cm_frechet_level_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    check("ncm == D*L", aug.ncm == D * L)
    check("ncm_cm == (D-1)*L", aug.ncm_cm == (D - 1) * L)
    check("ncm_level == L", aug.ncm_level == L)
    println()
end

println("==================================================")
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
