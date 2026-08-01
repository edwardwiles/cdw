# Profiled economic block port (2026-08-01), commit 4: D4 gate for the NEW
# use_profiled_correction=true path in winner_pair_cross_hessian_zc_block! (H_EZ).
# Modeled on test_winner_pair_cross_hessian_zc_d4.jl's CM+ZC setup (real CompressedFactual, dense
# E'*diag(S)*Z reference). Same two-check structure as test_profiled_hec_correction_d4_2026-08-01.jl:
#   1. Regression safety: use_profiled_correction=false unchanged vs the pre-existing dense reference.
#   2. New-path correctness: use_profiled_correction=true vs an INDEPENDENT brute-force
#      T^Z_{d,x} = sum_w S_w*wval[w,d]*Z[w,x], built directly from raw data, not reusing TZ_buf/SnuWval.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"""
Fully independent brute-force reference for T^Z_{d,x} = sum_w S[w]*nu[w]*wval[w,d]*Z[w,x] -- built
directly from cf/Z, no reuse of SnuWval/TZ_buf.
"""
function brute_force_tz(cf::CompressedFactual, S::AbstractVector{Float64}, Z::AbstractMatrix{Float64}, d::Int, x::Int)
    W = cf.W
    nu = cf.SW
    acc = 0.0
    for w in 1:W
        acc += S[w] * nu[w] * cf.wval[w, d] * Z[w, x]
    end
    return acc
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

const L = 10
K_mean, K_pair, label = 1, 1, "K1"
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = :direct)
objA = aug.obj_cm
n = objA.outer_constr_index
θ_ext_calib = vcat(θ_full_calib, nu0vec(K_mean))
K = zeros(size(ctx.U, 1))
objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_ext_calib, objA.U, objA)
objA.H[:, 1] .= K
objA.H[:, 2] .= 1.0

cctx = build_cm_meanzc_bin_ctx(ctx, aug)
ncore = cctx.ncore_core; NCORE_ext = cctx.NCORE
n_restr = NCORE_ext - ncore

Random.seed!(9100 + K_mean)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
maxerr_old = 0.0
maxerr_new = 0.0
for (pi_, x) in enumerate(xs)
    _archC_prep_for_hessian!(objA, x)
    cf = cctx.core_cf_ref[]
    check("$label pt$pi_: cf is a real CompressedFactual", cf isa CompressedFactual)
    cf isa CompressedFactual || continue

    H = objA.H; M = objA.M
    ddPsi! = objA.ddPsi!; ddPsi!(objA.arg2, objA.arg0); S = objA.arg2
    E = @view H[:, 2:1+NCORE_ext]
    EC = @view E[:, 1:ncore]
    Z = @view E[:, ncore+1:NCORE_ext]
    H_EM_dense = ((EC .* S)' * Z) ./ M

    wctx = build_winner_pair_ctx(cf)
    check("$label pt$pi_: wctx.ncolI+1 == ncore", wctx.ncolI + 1 == ncore)
    ws_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
    ws = ensure_winner_zc_cross_scratch!(ws_ref, wctx.W, n_restr, wctx.Ddest)
    winner_pair_cross_hessian_zc_prep!(ws, wctx, S)

    # Check 1: regression safety
    H_EM_old = zeros(ncore, n_restr)
    winner_pair_cross_hessian_zc_block!(H_EM_old, wctx, ws, S, Z, M; use_profiled_correction = false)
    err_old = maximum(abs.(H_EM_dense .- H_EM_old))
    global maxerr_old = max(maxerr_old, err_old)

    # Check 2: profiled correction vs independent brute-force T^Z. `expected` is built entirely
    # from scratch (winner/wval/kappa0/pi_vec), not derived from H_EM_dense, so a shared bug
    # between this check and the code under test cannot hide.
    H_EM_new = zeros(ncore, n_restr)
    winner_pair_cross_hessian_zc_block!(H_EM_new, wctx, ws, S, Z, M; use_profiled_correction = true)
    nbilateral = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
    maxdiff = 0.0
    y = wctx.y; winner = wctx.winner
    nu = cf.SW
    for j in 1:nbilateral
        d = wctx.target_slot[j]
        o_row = div(j - d, wctx.Ddest) + 1
        for x in 1:n_restr
            keep = 0.0
            for w in 1:cf.W
                if cf.winner[w, d] == o_row
                    keep += S[w] * nu[w] * cf.wval[w, d] * wctx.kappa0[j] * Z[w, x]
                end
            end
            Tdx = brute_force_tz(cf, S, Z, d, x)
            expected = (keep - wctx.pi_vec[j] * Tdx) / M
            # row j+1 in HEZ's own convention: row 1 is the ones/zeta row, row j+1 is economic column j.
            maxdiff = max(maxdiff, abs(H_EM_new[j + 1, x] - expected))
        end
    end
    global maxerr_new = max(maxerr_new, maxdiff)
    @printf("  %-4s pt%d x-scale=%.3f  n_restr=%d  old max|Δ|=%.3e  new max|Δ|=%.3e\n",
            label, pi_, maximum(abs.(x)), n_restr, err_old, maxdiff)
end
check("$label: use_profiled_correction=false regression-safe (max|Δ| over points = $(maxerr_old))", maxerr_old < 1e-8)
check("$label: use_profiled_correction=true matches independent brute-force T^Z (max|Δ| over points = $(maxerr_new))", maxerr_new < 1e-6)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
