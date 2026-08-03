# Genuine-cold ZC Hessian K=3 optimization task (2026-08-01), Sections 11-12: H_CZ and H_EZ
# surgical candidates -- correctness gate (D=20, real production context, K=3) + isolated-kernel
# timing, cm_meanzc only (the family with a known-working direct-construction harness this session,
# see feedback-archc-verified-state / originzc-archozbase-state-direct-call-crash-k-gt-1 memory --
# origin-ZC's own direct-construction path is not reused here for the same reason).
#
# Method: build the real cm_meanzc production context directly (no KNITRO), synthesize a fixed
# dual point, run the REAL production Hessian callback (hessian_cm_structured_v2!) ONCE to
# populate every scratch field this task's candidates need (cctx.Bidx, cctx.hzz_centered.ZcS,
# cctx.bin_zc_cross / cctx.bin_zc_drawchunk, cctx.zc_cross_scratch, wctx) with REAL, live values --
# then call the reference AND candidate kernels directly against those exact same captured inputs
# for both correctness comparison and isolated timing.
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

# H_EZ needs cctx.core_ws (hence wctx=serial_ctx(cctx.core_ws)) which is only built when
# cctx.core_cf_ref[] holds a real CompressedFactual -- that is a side effect of calling obj_cm's
# OWN `moments!` closure (wrap_moments_with_cm_meanzc, cm_meanzc_moments.jl) at a real ECONOMIC
# theta, NOT of the dual-space Hessian call alone. Calling moments! directly is PURE JULIA (no
# KNITRO) -- deliberately NOT calling archC_meanzc_base_state/verified_state or
# cm_meanzc_production_value_verified, which DO invoke a real inner KNITRO solve and are exactly
# the functions flagged as unreliable for direct-construction testing
# (feedback-archc-verified-state-direct-call-knitro-callback-err memory).
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
Bidx = cctx.Bidx
ZcS = cctx.hzz_centered.ZcS
D = cctx.D
workers10 = resolve_cross_hessian_workers_default()
lp("workers (production default) = ", workers10, "  Ddest=", wctx.Ddest, " ncolI=", wctx.ncolI)

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
function record!(block, candidate, workers, tmin, tmean; maxdiff = NaN, relscale = NaN)
    push!(rows, (block = block, candidate = candidate, workers = workers, t_min_s = tmin, t_mean_s = tmean,
        maxdiff = maxdiff, relscale = relscale))
    @printf("  [%-4s|%-24s] workers=%-3d min=%.4fs mean=%.4fs maxdiff=%.3e relscale=%.3e\n",
        block, candidate, workers, tmin, tmean, maxdiff, relscale)
    flush(stdout)
end

lp("\n", "="^100, "\n=== H_CZ: :draw_chunk_thread_local (reference) vs reordered candidate ===")
for workers in [1, 4, 8, 10]
    dc_ref = BinZCrossDrawChunkScratch(D, L, nz, workers)
    ws_ref = BinZCrossScratch(D, L, nz)
    tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill_drawchunk!(ws_ref, dc_ref, Bidx, ZcS; workers = workers))
    record!("H_CZ", "draw_chunk_thread_local(ref)", workers, tmin, tmean; maxdiff = 0.0, relscale = maximum(abs.(ws_ref.ZBinTab)))

    for jtile in [32, 64, 128]
        dc_cand = BinZCrossDrawChunkScratch(D, L, nz, workers)
        ws_cand = BinZCrossScratch(D, L, nz)
        tmin, tmean = timeit(() -> bin_zc_cross_hessian_fill_drawchunk_reordered!(ws_cand, dc_cand, Bidx, ZcS; workers = workers, jtile = jtile))
        maxdiff = maximum(abs.(ws_cand.ZBinTab .- ws_ref.ZBinTab))
        maxdiff_cscum = maximum(abs.(ws_cand.ZBinCScum .- ws_ref.ZBinCScum))
        relscale = maximum(abs.(ws_ref.ZBinTab))
        ok = maxdiff < HCZ_CANDIDATE_TOL * relscale && maxdiff_cscum < HCZ_CANDIDATE_TOL * relscale
        check("H_CZ reordered workers=$workers jtile=$jtile: ZBinTab/ZBinCScum agree within tol (maxdiff=$maxdiff, cscum=$maxdiff_cscum, scale=$relscale)", ok)
        record!("H_CZ", "reordered_jtile$jtile", workers, tmin, tmean; maxdiff = maxdiff, relscale = relscale)
    end
end

lp("\n", "="^100, "\n=== H_EZ: winner_bin destination-owned threaded (reference) vs draw-major candidate ===")
ncolI = wctx.ncolI
nbilateral = wctx.has_cf ? ncolI - 1 : ncolI
for workers in [1, 4, 8, 10]
    HEZ_ref = Matrix{Float64}(undef, ncolI + 1, nz)
    tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_threaded!(HEZ_ref, wctx, cctx.zc_cross_scratch, S, Z, M; workers = workers))
    record!("H_EZ", "winner_bin_threaded(ref)", workers, tmin, tmean; maxdiff = 0.0, relscale = maximum(abs.(HEZ_ref)))

    for xtile in [32, 64, 128]
        dm = ensure_winner_zc_drawmajor_scratch!(nothing, W, wctx.Ddest, nbilateral, nz, workers)
        HEZ_cand = Matrix{Float64}(undef, ncolI + 1, nz)
        tmin, tmean = timeit(() -> winner_pair_cross_hessian_zc_block_drawmajor!(HEZ_cand, wctx, cctx.zc_cross_scratch, dm, S, Z, M; workers = workers, xtile = xtile))
        maxdiff = maximum(abs.(HEZ_cand .- HEZ_ref))
        relscale = maximum(abs.(HEZ_ref))
        ok = maxdiff < 1e-8 * relscale
        check("H_EZ drawmajor workers=$workers xtile=$xtile: HEZ agrees within tol (maxdiff=$maxdiff, scale=$relscale)", ok)
        record!("H_EZ", "drawmajor_xtile$xtile", workers, tmin, tmean; maxdiff = maxdiff, relscale = relscale)
    end
end

outpath = joinpath(D4X, "..", "..", "docs", "HCZ_HEZ_CANDIDATE_BENCHMARK_K3_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "block,candidate,workers,t_min_s,t_mean_s,maxdiff,relscale")
    for r in rows
        println(io, "$(r.block),$(r.candidate),$(r.workers),$(r.t_min_s),$(r.t_mean_s),$(r.maxdiff),$(r.relscale)")
    end
end
lp("Wrote ", outpath)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
