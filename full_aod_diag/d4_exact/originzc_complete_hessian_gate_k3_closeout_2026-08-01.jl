# Genuine-cold ZC Hessian K=3 CLOSEOUT task (2026-08-01), Section 7: origin-ZC complete-packed-
# Hessian correctness gate at real production K=3 width (K_mean=K_pair=3, nx=630), using the FIXED
# direct-construction recipe from ORIGIN_ZC_HARNESS_ROOT_CAUSE_2026-08-01.md (moment-matched nu0,
# NOT the naive nu=1 that crashed/failed in the prior session's harness). Gates:
#   reference vs blas_syrk        (H_ZZ only)
#   reference vs drawmajor_v2     (H_EZ only)
#   reference vs both together    (H_ZZ=blas_syrk + H_EZ=drawmajor_v2)
# H_CZ is inapplicable to origin_zc (no CM-grid block) -- correctly excluded per task brief Section 5.
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
          "cm_originzc_production.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "hez_drawmajor_candidate_2026-08-01.jl", "hez_drawmajor_v2_candidate_2026-08-01.jl"]
    include(joinpath(D4X, f))
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

rows = NamedTuple[]

lp("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair = 3, 3
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = OriginByPowerLayout(D, K_mean, K_pair)

# FIX (ORIGIN_ZC_HARNESS_ROOT_CAUSE_2026-08-01.md): moment-matched nu0, correct length n_eta(layout),
# NOT the naive fill(1.0, ctx.D) that crashed (wrong length) / failed nStatus=-300 (wrong value).
nu0 = Vector{Float64}(undef, n_eta(layout))
for k in 1:K_mean, o in 1:D
    nu0[target_index(layout, o, k)] = mean(@view (ctx.U .^ k)[:, o])
end

pcx_o = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, moment_representation = :operator)
octx = pcx_o.octx
octx.core_hessian_backend = :exact_winner_pair_parallel
obj_o = pcx_o.ctx_cm.obj
n_o = octx.NCORE + octx.n_eta
lp("NCORE=$(octx.NCORE), n_eta=$(octx.n_eta), n=$n_o, nZ=$(n_restriction(octx.hzz_zc_op))")

lp("Priming state via a real inner solve at the calibration point + moment-matched nu0 (archOZ_base_state)...")
t0 = time()
base_o = archOZ_base_state(x_free_calib, nu0, pcx_o.ctx_cm)
lp("  primed, nStatus=", base_o.inner_status, "  (", round(time() - t0, digits = 1), "s)")
base_o.inner_status in (0, -100, -101, -103) || error("origin_zc priming solve did not reach an accepted status -- see ORIGIN_ZC_HARNESS_ROOT_CAUSE_2026-08-01.md for the required nu0 recipe")

function full_hessian_oz(octx, obj, x::AbstractVector, n::Int)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    cb = archA_partitioned_hess_cb_builder(octx)
    fake_req = (x = x,)
    fake_res = (hess = h,)
    cb(nothing, nothing, fake_req, fake_res, obj)
    return unpack_packed(h, n)
end

CONFIGS = [
    (:reference, :winner_bin, "reference"),
    (:blas_syrk, :winner_bin, "blas_syrk_only"),
    (:reference, :drawmajor_v2, "drawmajor_v2_only"),
    (:blas_syrk, :drawmajor_v2, "both_optimized"),
]

for trial in 1:3
    Random.seed!(5000 + trial)
    x_fake = 0.05 .* randn(n_o)
    octx.zc_gram_backend = :reference
    octx.zc_ez_backend = :winner_bin
    Href = full_hessian_oz(octx, obj_o, x_fake, n_o)
    for (hzz_be, hez_be, tag) in CONFIGS
        tag == "reference" && continue
        octx.zc_gram_backend = hzz_be
        octx.zc_ez_backend = hez_be
        Hb = full_hessian_oz(octx, obj_o, x_fake, n_o)
        maxdiff = maximum(abs.(Hb .- Href))
        relscale = max(1.0, maximum(abs.(Href)))
        ok = maxdiff < 1e-7 * relscale
        check("origin_zc trial=$trial config=$tag (H_ZZ=$hzz_be, H_EZ=$hez_be): complete packed Hessian vs reference (max|Δ|=$(maxdiff), scale=$(relscale))", ok)
        push!(rows, (family = "origin_zc", trial = trial, config = tag, hzz_backend = hzz_be, hez_backend = hez_be,
            nx = n_restriction(octx.hzz_zc_op), n = n_o, maxdiff = maxdiff, relscale = relscale, pass = ok))
    end
    octx.zc_gram_backend = :reference
    octx.zc_ez_backend = :winner_bin
end

outpath = joinpath(D4X, "..", "..", "docs", "ORIGIN_ZC_COMPLETE_HESSIAN_GATE_CLOSEOUT_K3_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "family,trial,config,hzz_backend,hez_backend,nx,n,maxdiff,relscale,pass")
    for r in rows
        println(io, "$(r.family),$(r.trial),$(r.config),$(r.hzz_backend),$(r.hez_backend),$(r.nx),$(r.n),$(r.maxdiff),$(r.relscale),$(r.pass)")
    end
end
lp("Wrote ", outpath)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
