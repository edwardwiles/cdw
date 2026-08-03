# Genuine-cold ZC Hessian K=3 optimization task (2026-08-01): correctness gate for the WIRED
# production dispatch paths (cctx.hcz_prep_backend=:draw_chunk_reordered,
# cctx.zc_ez_backend=:drawmajor) -- as opposed to hcz_hez_candidates_gate_and_benchmark_k3_2026-08-01.jl,
# which calls the candidate kernels directly. This exercises the ACTUAL edited dispatch branches
# in cm_hessian_architectures.jl/hcz_drawchunk_candidate_2026-07-29.jl through the real production
# entry point hessian_cm_structured_v2!, comparing the COMPLETE packed Hessian against the
# all-reference-backend baseline.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random
lp(xs...) = (println(xs...); flush(stdout))

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end
function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

lp("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair, L = 3, 3, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
obj_cm = aug.obj_cm
W = size(ctx.U, 1)
n = cctx.NCORE + cctx.ncm
lp("W=$W, D=$(ctx.D), NCORE=$(cctx.NCORE), ncm=$(cctx.ncm), n=$n, nZ=$(n_restriction(cctx.hzz_zc_op))")

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
θ_ext_calib = vcat(θ_econ_calib, ones(K_mean))
Kbuf = Vector{Float64}(undef, W)
Gbuf = Matrix{Float64}(undef, W, n + 64)
obj_cm.moments!(Kbuf, Gbuf, θ_ext_calib, ctx.U, obj_cm)
cctx.nu_ref[] = ones(K_mean)
lp("obj_cm.moments! primed core_cf_ref[].")

tls = build_thread_local_scratch(cctx)

for trial in 1:3
    Random.seed!(50000 + trial)
    x_fake = 0.1 .* randn(n)

    href = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.hcz_prep_backend = :draw_chunk_thread_local
    cctx.zc_ez_backend = :winner_bin
    cctx.zc_gram_backend = :reference
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(href, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    Href = unpack_packed(href, n)

    hcand = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.hcz_prep_backend = :draw_chunk_reordered
    cctx.zc_ez_backend = :drawmajor
    cctx.zc_gram_backend = :blas_syrk
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(hcand, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    Hcand = unpack_packed(hcand, n)

    maxdiff = maximum(abs.(Hcand .- Href))
    relscale = max(1.0, maximum(abs.(Href)))
    ok = maxdiff < 1e-7 * relscale
    check("trial=$trial: ALL-THREE-optimized-backends complete packed Hessian vs all-reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
end
cctx.hcz_prep_backend = :draw_chunk_thread_local
cctx.zc_ez_backend = :winner_bin
cctx.zc_gram_backend = :reference

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
