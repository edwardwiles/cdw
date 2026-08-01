# Profiled economic block port (2026-08-01), follow-up: D4 gate for
# winner_pair_cross_hessian_zc_block_threaded!'s new use_profiled_correction path.
#
# Design goal stated in threaded_cross_hessian.jl's own header: threaded output must be
# BIT-IDENTICAL to the serial kernel (output-row ownership, no atomics, same per-row accumulation
# order) -- not merely "agrees to floating-point tolerance". Since the serial
# use_profiled_correction=true path is already independently validated to machine precision against
# a brute-force reference (test_profiled_hez_correction_d4_2026-08-01.jl), the correct check here is
# bit-identical agreement between threaded and serial at use_profiled_correction=true, across
# several worker counts -- exactly what test_threaded_cross_hessian_d4.jl already established the
# precedent for at use_profiled_correction=false (implicitly, the only mode that existed before).
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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

const L = 10
K_mean, K_pair = 1, 1
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

nt = Threads.nthreads()
worker_counts = unique(filter(w -> w >= 1 && w <= nt, [1, 2, nt]))
println("Threads.nthreads()=$nt, testing worker_counts=$worker_counts")

Random.seed!(9100 + K_mean)
xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
maxdiff_overall = 0.0
for (pi_, x) in enumerate(xs)
    _archC_prep_for_hessian!(objA, x)
    cf = cctx.core_cf_ref[]
    check("pt$pi_: cf is a real CompressedFactual", cf isa CompressedFactual)
    cf isa CompressedFactual || continue

    H = objA.H; M = objA.M
    ddPsi! = objA.ddPsi!; ddPsi!(objA.arg2, objA.arg0); S = objA.arg2
    E = @view H[:, 2:1+NCORE_ext]
    Z = @view E[:, ncore+1:NCORE_ext]

    # bi_slot supplied (follow-up to the original gate): also exercises the France/cf row's
    # profiled-correction path threaded-vs-serial, not just the bilateral rows.
    wctx = build_winner_pair_ctx(cf; bi_slot = dest_slot(ctx, ctx.bi))
    ws_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
    ws = ensure_winner_zc_cross_scratch!(ws_ref, wctx.W, n_restr, wctx.Ddest)
    winner_pair_cross_hessian_zc_prep!(ws, wctx, S)

    H_serial = zeros(ncore, n_restr)
    winner_pair_cross_hessian_zc_block!(H_serial, wctx, ws, S, Z, M; use_profiled_correction = true)

    for workers in worker_counts
        H_threaded = zeros(ncore, n_restr)
        winner_pair_cross_hessian_zc_block_threaded!(H_threaded, wctx, ws, S, Z, M; workers = workers, use_profiled_correction = true)
        maxdiff = maximum(abs.(H_serial .- H_threaded))
        global maxdiff_overall = max(maxdiff_overall, maxdiff)
        check("pt$pi_ workers=$workers: threaded matches serial bit-for-bit under use_profiled_correction=true (max|Δ|=$(maxdiff))",
              maxdiff == 0.0)
        @printf("  pt%d workers=%d: max|Δ(threaded-serial)|=%.3e\n", pi_, workers, maxdiff)
    end

    # Also confirm the OLD path (use_profiled_correction=false) is still bit-identical threaded vs
    # serial -- regression safety for the pre-existing behavior this amendment must not disturb.
    H_serial_old = zeros(ncore, n_restr)
    winner_pair_cross_hessian_zc_block!(H_serial_old, wctx, ws, S, Z, M; use_profiled_correction = false)
    for workers in worker_counts
        H_threaded_old = zeros(ncore, n_restr)
        winner_pair_cross_hessian_zc_block_threaded!(H_threaded_old, wctx, ws, S, Z, M; workers = workers, use_profiled_correction = false)
        maxdiff_old = maximum(abs.(H_serial_old .- H_threaded_old))
        check("pt$pi_ workers=$workers: threaded matches serial bit-for-bit under use_profiled_correction=false (regression) (max|Δ|=$(maxdiff_old))",
              maxdiff_old == 0.0)
    end
end

println()
println("Overall max|Δ(threaded-serial)| under use_profiled_correction=true: ", maxdiff_overall)
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
