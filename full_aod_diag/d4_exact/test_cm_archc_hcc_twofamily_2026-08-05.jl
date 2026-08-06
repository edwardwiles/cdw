# 2026-08-05 truncated-power task, Architecture C extension: focused correctness gate for the
# NEW H_CC (CM-CM self-block) two-family bin-table machinery (Ttab12/Ttab22/CT12/CT22,
# fill_cm_HCC!'s H^{12}/H^{22} blocks) in cm_hessian_architectures.jl, checked against a fully
# independent brute-force direct sum -- BEFORE wiring the much larger winner-pair H_EC extension
# or touching hessian_cm_structured!'s own hard-refuse guard. Isolates H_CC math from everything
# else (H_EE, H_EC, KNITRO, the FG operator) so a bug here is cheap to catch and fix.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
Random.seed!(20260805)
L = 4

aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true)
@assert aug.n_families == 2
cctx = build_cm_bin_ctx(ctx, aug)
@assert cctx.Pow !== nothing "Pow must be populated for a two-family cctx"

W, Dcheck = size(ctx.U)
@assert Dcheck == D
M = W
NCORE = aug.ncore
nO = length(aug.origins)
origins = aug.origins
refIndex1 = aug.refIndex1
z = aug.z
Pow = cctx.Pow

# Random positive weights standing in for `w = ddPsi!(arg2, arg0)` -- H_CC's math is a pure
# bilinear form in these weights, so any fixed positive vector exercises the identical code path
# a real KNITRO callback would (this is a math/indexing gate, not a solve gate).
w = 0.1 .+ rand(W)

build_bin_tables!(cctx, nothing, w; fill_S = false)
prefix_sum_tables!(cctx; fill_S = false)
Hfull = cctx.Hfull
fill!(Hfull, 0.0)
fill_cm_HCC!(Hfull, cctx, M)

# Brute-force direct raw (R=:anchored, i.e. R===nothing) reference, independent of T/CT tables
# entirely: A_{o,l}[s] = 1{U[s,o]<=z[l]} - 1{U[s,refIndex1]<=z[l]}, B_{o,l}[s] = Pow[s,o]*
# 1{U[s,o]<=z[l]} - Pow[s,refIndex1]*1{U[s,refIndex1]<=z[l]}.
function raw_A(l::Int, o::Int)
    return (ctx.U[:, o] .<= z[l]) .- (ctx.U[:, refIndex1] .<= z[l])
end
function raw_B(l::Int, o::Int)
    # 2026-08-05 BUG FIX (user-caught): eq.36's own indicator is `1{U>z[l]}`, not `<=` -- see
    # common_marginals_moments.jl's own docstring for the full derivation/proof.
    return Pow[:, o] .* (ctx.U[:, o] .> z[l]) .- Pow[:, refIndex1] .* (ctx.U[:, refIndex1] .> z[l])
end

ncm_cdf = nO * L
@assert cctx.contrasts == :anchored "this brute-force check assumes R===nothing (:anchored); aug/cctx built without an explicit contrasts kwarg default to :anchored"

Random.seed!(1)
test_pairs = [(rand(1:L), rand(1:L)) for _ in 1:6]
push!(test_pairs, (1, 1), (L, L), (2, L))

max_err_11 = 0.0
max_err_12 = 0.0
max_err_21 = 0.0
max_err_22 = 0.0
for (l, lp) in test_pairs
    rows = NCORE + (l-1)*nO+1 : NCORE + l*nO
    cols = NCORE + (lp-1)*nO+1 : NCORE + lp*nO
    rows_pow = NCORE + ncm_cdf + (l-1)*nO+1 : NCORE + ncm_cdf + l*nO
    cols_pow = NCORE + ncm_cdf + (lp-1)*nO+1 : NCORE + ncm_cdf + lp*nO

    H11 = Hfull[rows, cols]
    H12 = Hfull[rows, cols_pow]
    H21 = Hfull[rows_pow, cols]
    H22 = Hfull[rows_pow, cols_pow]

    for (oi, o) in enumerate(origins), (pi, p) in enumerate(origins)
        A_ol = raw_A(l, o); A_plp = raw_A(lp, p)
        B_ol = raw_B(l, o); B_plp = raw_B(lp, p)
        brute11 = sum(w .* A_ol .* A_plp) / M
        brute12 = sum(w .* A_ol .* B_plp) / M
        brute21 = sum(w .* B_ol .* A_plp) / M
        brute22 = sum(w .* B_ol .* B_plp) / M
        global max_err_11 = max(max_err_11, abs(H11[oi, pi] - brute11))
        global max_err_12 = max(max_err_12, abs(H12[oi, pi] - brute12))
        global max_err_21 = max(max_err_21, abs(H21[oi, pi] - brute21))
        global max_err_22 = max(max_err_22, abs(H22[oi, pi] - brute22))
    end
end

println("max|H^11 - brute| = ", max_err_11)
println("max|H^12 - brute| = ", max_err_12)
println("max|H^21 - brute| = ", max_err_21)
println("max|H^22 - brute| = ", max_err_22)
check("H^11 (unchanged CDF-CDF block) matches brute force", max_err_11 < 1e-9)
check("H^12 (new CDF-POW cross block) matches brute force", max_err_12 < 1e-9)
check("H^21 (new POW-CDF mirror block) matches brute force", max_err_21 < 1e-9)
check("H^22 (new POW-POW block) matches brute force", max_err_22 < 1e-9)

println("="^80)
if isempty(FAILURES)
    println("ALL H_CC TWO-FAMILY GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
