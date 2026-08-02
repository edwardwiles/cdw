# Gate 2 (production integration, 2026-08-01): complete packed-Hessian reference-vs-NEW-DEFAULT
# gate for BOTH cm_meanzc and origin_zc, using the REAL public production context builders with NO
# explicit backend kwargs (so this exercises whatever ZC_GRAM_BACKEND_DEFAULT[]/
# HCZ_PREP_BACKEND_DEFAULT[]/ZC_EZ_BACKEND_DEFAULT[] currently resolve to -- the production
# defaults after this integration, not a hand-picked Symbol). Deliberately uses ONLY the include
# list every real production caller already loads (campaign_cm_family_runner.jl's own list) --
# no manual candidate-file includes -- reconfirming Gate 1's own finding in the same breath as the
# correctness check.
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl", "cross_hessian_live_stash_2026-07-28.jl"]
    include(joinpath(_D4E, f))
end
using Test, Printf, LinearAlgebra, Random, Statistics
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

lp("Production defaults as loaded: ZC_GRAM_BACKEND_DEFAULT[]=", ZC_GRAM_BACKEND_DEFAULT[],
   " HCZ_PREP_BACKEND_DEFAULT[]=", HCZ_PREP_BACKEND_DEFAULT[], " ZC_EZ_BACKEND_DEFAULT[]=", ZC_EZ_BACKEND_DEFAULT[])

rows = NamedTuple[]

lp("\n", "="^100, "\n=== cm_meanzc: default backends vs :reference/:draw_chunk_thread_local/:winner_bin ===")
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair, L = 3, 3, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug)   # NO explicit backend kwargs -- real production defaults
obj_cm = aug.obj_cm
n = cctx.NCORE + cctx.ncm
lp("W=100000, D=", ctx.D, ", NCORE=", cctx.NCORE, ", ncm=", cctx.ncm, ", n=", n, ", nZ=", n_restriction(cctx.hzz_zc_op))
lp("cctx defaults (no kwargs passed): zc_gram_backend=", cctx.zc_gram_backend, " hcz_prep_backend=", cctx.hcz_prep_backend, " zc_ez_backend=", cctx.zc_ez_backend)
check("cm_meanzc cctx defaults to blas_syrk/draw_chunk_reordered/drawmajor_v2 with zero kwargs",
      cctx.zc_gram_backend === :blas_syrk && cctx.hcz_prep_backend === :draw_chunk_reordered && cctx.zc_ez_backend === :drawmajor_v2)

# CRITICAL FIX (found live via Gate 3's real-KNITRO crash, then confirmed by inspection): this
# gate's FIRST draft never called obj_cm.moments! at a real economic point, so cctx.core_cf_ref[]
# was never populated with a real CompressedFactual -- `use_direct_hcz`/the analogous H_ZZ/H_EZ
# gates in cm_hessian_threaded.jl all require a real `cf`, so EVERY comparison silently fell back
# to the DENSE reference path for BOTH arms (a trivial dense-vs-dense pass, never exercising the
# bin/threaded backend dispatch this gate is supposed to test). Priming with a real moments! call
# at the calibration point (same idiom as hcz_hez_candidates_gate_and_benchmark_k3_2026-08-01.jl)
# fixes this -- confirmed below to actually reach the winner-pair/bin-structured path this time.
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
θ_ext_calib = vcat(θ_econ_calib, ones(K_mean))
Kbuf = Vector{Float64}(undef, size(ctx.U, 1))
Gbuf = Matrix{Float64}(undef, size(ctx.U, 1), n + 64)
obj_cm.moments!(Kbuf, Gbuf, θ_ext_calib, ctx.U, obj_cm)
cctx.nu_ref[] = ones(K_mean)
check("cm_meanzc cctx.core_cf_ref[] is a real CompressedFactual after priming (not dense fallback)",
      cctx.core_cf_ref[] isa CompressedFactual)

for trial in 1:3
    Random.seed!(3000 + trial)
    x_fake = 0.1 .* randn(n)
    tls = build_thread_local_scratch(cctx)
    href = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cctx.zc_gram_backend = :reference; cctx.hcz_prep_backend = :draw_chunk_thread_local; cctx.zc_ez_backend = :winner_bin
    _archC_prep_for_hessian!(obj_cm, x_fake)
    winner_calls_before = NO_DENSE_G_COUNTERS[].winner_cross_hessian_calls
    hessian_cm_structured_v2!(href, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    Href = unpack_packed(href, n)
    if trial == 1
        check("cm_meanzc reference call genuinely reaches the winner-pair/bin-structured path (not dense fallback)",
              NO_DENSE_G_COUNTERS[].winner_cross_hessian_calls > winner_calls_before)
    end

    cctx.zc_gram_backend = :blas_syrk; cctx.hcz_prep_backend = :draw_chunk_reordered; cctx.zc_ez_backend = :drawmajor_v2
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    _archC_prep_for_hessian!(obj_cm, x_fake)
    hessian_cm_structured_v2!(h, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
    Hb = unpack_packed(h, n)
    maxdiff = maximum(abs.(Hb .- Href))
    relscale = max(1.0, maximum(abs.(Href)))
    ok = maxdiff < 1e-7 * relscale
    check("cm_meanzc trial=$trial: complete packed Hessian, production defaults vs reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
    push!(rows, (family = "cm_meanzc", trial = trial, maxdiff = maxdiff, relscale = relscale, pass = ok))
end

lp("\n", "="^100, "\n=== origin_zc: default backends vs :reference/:winner_bin ===")
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = OriginByPowerLayout(ctx.D, K_mean, K_pair)
nu0 = Vector{Float64}(undef, n_eta(layout))
for k in 1:K_mean, o in 1:ctx.D
    nu0[target_index(layout, o, k)] = mean(@view (ctx.U .^ k)[:, o])
end
pcx_o = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :operator)
octx = pcx_o.octx
octx.core_hessian_backend = :exact_winner_pair_parallel
obj_o = pcx_o.ctx_cm.obj
n_o = octx.NCORE + octx.n_eta
lp("origin_zc octx defaults (no kwargs passed beyond fg_backend/moment_representation): zc_gram_backend=", octx.zc_gram_backend, " zc_ez_backend=", octx.zc_ez_backend)
check("origin_zc octx defaults to blas_syrk/drawmajor_v2 with zero explicit backend kwargs",
      octx.zc_gram_backend === :blas_syrk && octx.zc_ez_backend === :drawmajor_v2)

lp("Priming state via a real inner solve at the calibration point + moment-matched nu0 (archOZ_base_state)...")
base_o = archOZ_base_state(x_free_calib, nu0, pcx_o.ctx_cm)
lp("  primed, nStatus=", base_o.inner_status)
base_o.inner_status in (0, -100, -101, -103) || error("origin_zc priming solve did not reach an accepted status")

function full_hessian_oz(octx, obj, x::AbstractVector, n::Int)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cb = archA_partitioned_hess_cb_builder(octx)
    fake_req = (x = x,)
    fake_res = (hess = h,)
    cb(nothing, nothing, fake_req, fake_res, obj)
    return unpack_packed(h, n)
end

for trial in 1:3
    Random.seed!(5000 + trial)
    x_fake = 0.05 .* randn(n_o)
    octx.zc_gram_backend = :reference; octx.zc_ez_backend = :winner_bin
    Href = full_hessian_oz(octx, obj_o, x_fake, n_o)
    octx.zc_gram_backend = :blas_syrk; octx.zc_ez_backend = :drawmajor_v2
    Hb = full_hessian_oz(octx, obj_o, x_fake, n_o)
    maxdiff = maximum(abs.(Hb .- Href))
    relscale = max(1.0, maximum(abs.(Href)))
    ok = maxdiff < 1e-7 * relscale
    check("origin_zc trial=$trial: complete packed Hessian, production defaults vs reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
    push!(rows, (family = "origin_zc", trial = trial, maxdiff = maxdiff, relscale = relscale, pass = ok))
end

outpath = joinpath(_D4E, "..", "..", "docs", "GATE2_PRODUCTION_DEFAULTS_COMPLETE_HESSIAN_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,trial,maxdiff,relscale,pass")
    for r in rows
        println(io, "$(r.family),$(r.trial),$(r.maxdiff),$(r.relscale),$(r.pass)")
    end
end
lp("Wrote ", outpath)

println()
println(ALL_PASS[] ? "GATE 2: ALL PASS" : "GATE 2: SOME FAILURES")
ALL_PASS[] || exit(1)
