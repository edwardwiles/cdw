# Genuine-cold ZC Hessian K=3 optimization task (2026-08-01), Section 15: D=20 fixed-state
# correctness gate for the COMPLETE packed Hessian (not just isolated H_ZZ) across every H_ZZ
# backend, at production K=3 width (nx=630), for BOTH cm_meanzc (hessian_cm_structured_v2!) and
# origin_zc (archA_partitioned_hess_cb_builder -- exercises this task's own new originZC_* labels).
# Direct construction, no KNITRO call at all (same pattern as
# hcz_wired_complete_hessian_gate_2026-07-29.jl) -- synthetic random dual points, S = ddPsi!(arg0)
# from the real ddPsi! (never a hand-picked S), so this is a genuine algebraic correctness check,
# not a claim about a converged economic state.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_config.jl",
          "cm_originzc_moments.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_originzc_production.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl"]
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

const BACKENDS = [:reference, :blas_syrk, :blas_gemm, :threaded_packed]
rows = NamedTuple[]

lp("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair, L = 3, 3, 50

lp("\n=== cm_meanzc, K_mean=K_pair=3 ===")
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
obj_cm = aug.obj_cm
W = size(ctx.U, 1)
NCORE = cctx.NCORE; ncm = cctx.ncm
n = NCORE + ncm
lp("W=$W, D=$(ctx.D), NCORE=$NCORE, ncm=$ncm, n=$n, nZ=$(n_restriction(cctx.hzz_zc_op))")

for trial in 1:3
    Random.seed!(3000 + trial)
    x_fake = 0.1 .* randn(n)
    tls = build_thread_local_scratch(cctx)
    href = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.zc_gram_backend = :reference
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(href, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    Href = unpack_packed(href, n)
    for backend in BACKENDS
        backend === :reference && continue
        h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        cctx.zc_gram_backend = backend
        _archC_prep_for_hessian!(obj_cm, x_fake)
        hessian_cm_structured_v2!(h, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
        Hb = unpack_packed(h, n)
        maxdiff = maximum(abs.(Hb .- Href))
        relscale = max(1.0, maximum(abs.(Href)))
        ok = maxdiff < 1e-7 * relscale
        check("cm_meanzc trial=$trial backend=$backend: complete packed Hessian vs reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
        push!(rows, (family = "cm_meanzc", trial = trial, backend = backend, nx = n_restriction(cctx.hzz_zc_op), n = n, maxdiff = maxdiff, relscale = relscale, pass = ok))
    end
    cctx.zc_gram_backend = :reference
end

lp("\n=== origin_zc, K_mean=K_pair=3 ===")
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = OriginByPowerLayout(ctx.D, K_mean, K_pair)
νfull0 = fill(1.0, ctx.D)
pcx_o = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :operator)
octx = pcx_o.octx
octx.core_hessian_backend = :exact_winner_pair_parallel
obj_o = pcx_o.ctx_cm.obj
n_o = octx.NCORE + octx.n_eta
lp("NCORE=$(octx.NCORE), n_eta=$(octx.n_eta), n=$n_o, nZ=$(n_restriction(octx.hzz_zc_op))")

lp("Priming state via one real (short) inner solve at the calibration point (archOZ_base_state)...")
base_o = archOZ_base_state(x_free_calib, νfull0, pcx_o.ctx_cm)
lp("  primed, nStatus=", base_o.inner_status)

function full_hessian_oz(octx, obj, x::AbstractVector, n::Int)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cb = archA_partitioned_hess_cb_builder(octx)
    fake_req = (x = x,)
    fake_res = (hess = h,)
    cb(nothing, nothing, fake_req, fake_res, obj)
    return unpack_packed(h, n)
end

for trial in 1:3
    Random.seed!(4000 + trial)
    x_fake = 0.05 .* randn(n_o)
    octx.zc_gram_backend = :reference
    Href = full_hessian_oz(octx, obj_o, x_fake, n_o)
    for backend in BACKENDS
        backend === :reference && continue
        octx.zc_gram_backend = backend
        Hb = full_hessian_oz(octx, obj_o, x_fake, n_o)
        maxdiff = maximum(abs.(Hb .- Href))
        relscale = max(1.0, maximum(abs.(Href)))
        ok = maxdiff < 1e-7 * relscale
        check("origin_zc trial=$trial backend=$backend: complete packed Hessian vs reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
        push!(rows, (family = "origin_zc", trial = trial, backend = backend, nx = n_restriction(octx.hzz_zc_op), n = n_o, maxdiff = maxdiff, relscale = relscale, pass = ok))
    end
    octx.zc_gram_backend = :reference
end

outpath = joinpath(D4X, "..", "..", "docs", "HZZ_LIVE_SOLVE_BACKEND_CORRECTNESS_K3_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,trial,backend,nx,n,maxdiff,relscale,pass")
    for r in rows
        println(io, "$(r.family),$(r.trial),$(r.backend),$(r.nx),$(r.n),$(r.maxdiff),$(r.relscale),$(r.pass)")
    end
end
lp("Wrote ", outpath)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
