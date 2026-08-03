# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 6: H_EZ drawmajor-v2 benchmark.
# Compares :winner_bin (reference), :drawmajor (v1), :drawmajor_v2 (this closeout's fix for v1's
# identified redundant per-x strided winner/v re-read) at real cm_meanzc production K=3 width
# (nx=630, W=100,000), workers in {1,4,8,10}, xtile in {32,64,128} -- same method as
# HCZ_HEZ_CANDIDATE_BENCHMARK_K3_2026-08-01.csv (direct construction, no KNITRO; real production
# Hessian callback run once to populate live scratch state, then reference/candidates called
# directly against the SAME captured inputs).
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
          "cm_meanzc_production.jl", "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl"]
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

const NT = Threads.nthreads()
lp("Threads.nthreads() = ", NT)

lp("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair, L = 3, 3, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
obj_cm = aug.obj_cm
W = size(ctx.U, 1)
NCORE = cctx.NCORE; ncm = cctx.ncm
n = NCORE + ncm
nz = n_restriction(cctx.hzz_zc_op)
lp("W=$W, D=$(ctx.D), NCORE=$NCORE, ncm=$ncm, n=$n, nZ=$nz")

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
θ_ext_calib = vcat(θ_econ_calib, ones(K_mean))
Kbuf = Vector{Float64}(undef, W)
Gbuf = Matrix{Float64}(undef, W, n + 64)
obj_cm.moments!(Kbuf, Gbuf, θ_ext_calib, ctx.U, obj_cm)
cctx.nu_ref[] = ones(K_mean)
lp("obj_cm.moments! ran once at a valid economic theta -- cctx.core_cf_ref[] should now hold a real CompressedFactual.")

x_fake = 0.1 .* randn(MersenneTwister(20260801), n)
tls = build_thread_local_scratch(cctx)
h_scratch = Vector{Float64}(undef, n * (n + 1) ÷ 2)
_archC_prep_for_hessian!(obj_cm, x_fake)
hessian_cm_structured_v2!(h_scratch, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
lp("Real production Hessian callback ran once -- cctx scratch fields now populated with live state.")

S = copy(obj_cm.arg2); M = obj_cm.M
wctx = serial_ctx(cctx.core_ws)
Z = @view cctx.hzz_centered.Zc[:, 1:nz]
ncolI = wctx.ncolI
nbilateral = wctx.has_cf ? ncolI - 1 : ncolI

const REPS = 5
function timeit(f::Function; reps::Int = REPS)
    f()
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f()
        ts[i] = (time_ns() - t0) / 1e9
    end
    return (minimum(ts), sum(ts) / reps)
end

rows = NamedTuple[]
function record!(candidate, workers, tile, tmin, tmean; maxdiff = NaN, relscale = NaN)
    push!(rows, (candidate = candidate, julia_threads = NT, workers = workers, tile = tile,
        t_min_s = tmin, t_mean_s = tmean, maxdiff = maxdiff, relscale = relscale))
    @printf("  [%-16s] workers=%-3d tile=%-4d min=%.4fs mean=%.4fs maxdiff=%.3e relscale=%.3e\n",
        candidate, workers, tile, tmin, tmean, maxdiff, relscale)
    flush(stdout)
end

lp("\n", "="^100, "\n=== H_EZ: winner_bin (reference) vs drawmajor (v1) vs drawmajor_v2 ===")
for workers in [1, 4, 8, 10]
    HEZ_ref = Matrix{Float64}(undef, ncolI + 1, nz)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HEZ_ref, wctx, cctx.zc_cross_scratch, S, Z, M; workers = workers))
    record!("winner_bin(ref)", workers, 0, tmin, tmean; maxdiff = 0.0, relscale = maximum(abs.(HEZ_ref)))

    for xtile in [32, 64, 128]
        dm1 = ensure_winner_zc_drawmajor_scratch!(nothing, W, wctx.Ddest, nbilateral, nz, workers)
        HEZ_v1 = Matrix{Float64}(undef, ncolI + 1, nz)
        tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_drawmajor!(HEZ_v1, wctx, cctx.zc_cross_scratch, dm1, S, Z, M; workers = workers, xtile = xtile))
        maxdiff = maximum(abs.(HEZ_v1 .- HEZ_ref)); relscale = maximum(abs.(HEZ_ref))
        check("drawmajor_v1 workers=$workers xtile=$xtile agrees within tol (maxdiff=$maxdiff, scale=$relscale)", maxdiff < 1e-6 * relscale)
        record!("drawmajor_v1", workers, xtile, tmin, tmean; maxdiff = maxdiff, relscale = relscale)

        dm2 = ensure_winner_zc_drawmajor_v2_scratch!(nothing, W, wctx.Ddest, nbilateral, nz, workers)
        HEZ_v2 = Matrix{Float64}(undef, ncolI + 1, nz)
        tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_drawmajor_v2!(HEZ_v2, wctx, cctx.zc_cross_scratch, dm2, S, Z, M; workers = workers, xtile = xtile))
        maxdiff = maximum(abs.(HEZ_v2 .- HEZ_ref)); relscale = maximum(abs.(HEZ_ref))
        check("drawmajor_v2 workers=$workers xtile=$xtile agrees within tol (maxdiff=$maxdiff, scale=$relscale)", maxdiff < 1e-6 * relscale)
        record!("drawmajor_v2", workers, xtile, tmin, tmean; maxdiff = maxdiff, relscale = relscale)
    end
end

outpath = joinpath(D4X, "..", "..", "docs", "HEZ_DRAWMAJOR_V2_BENCHMARK_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "candidate,julia_threads,workers,tile,t_min_s,t_mean_s,maxdiff,relscale")
    for r in rows
        println(io, "$(r.candidate),$(r.julia_threads),$(r.workers),$(r.tile),$(r.t_min_s),$(r.t_mean_s),$(r.maxdiff),$(r.relscale)")
    end
end
lp("Wrote ", outpath)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
