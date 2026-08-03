# Genuine-cold ZC Hessian K=3 closeout task (2026-08-01), Section 5: H_CZ phase-level timing
# breakdown (fill/join, serial reduction, cumsum) + a surgical parallel-reduction variant, at real
# cm_meanzc production K=3 width (nx=630, W=100,000). Same direct-construction method as the other
# closeout benchmarks (real production Hessian callback run once to populate live Bidx/ZcS state).
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
          "hcz_phase_breakdown_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random
lp(xs...) = (println(xs...); flush(stdout))

const NT = Threads.nthreads()
lp("Threads.nthreads() = ", NT)

lp("Building real D=20 context (W=100000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
K_mean, K_pair, L = 3, 3, 50
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal, meanzc_basis = :direct)
cctx = build_cm_meanzc_bin_ctx(ctx, aug; cm_cross_hessian_backend = :winner_bin, zc_cross_hessian_backend = :winner_bin)
obj_cm = aug.obj_cm
W = size(ctx.U, 1)
nz = n_restriction(cctx.hzz_zc_op)
D = cctx.D
lp("W=$W, D=$D, nZ=$nz, L=$L")

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_econ_calib = CS.reconstruct_full(x_free_calib, ctx.m)
θ_ext_calib = vcat(θ_econ_calib, ones(K_mean))
Kbuf = Vector{Float64}(undef, W)
n = cctx.NCORE + cctx.ncm
Gbuf = Matrix{Float64}(undef, W, n + 64)
obj_cm.moments!(Kbuf, Gbuf, θ_ext_calib, ctx.U, obj_cm)
cctx.nu_ref[] = ones(K_mean)

x_fake = 0.1 .* randn(MersenneTwister(20260801), n)
tls = build_thread_local_scratch(cctx)
h_scratch = Vector{Float64}(undef, n * (n + 1) ÷ 2)
_archC_prep_for_hessian!(obj_cm, x_fake)
hessian_cm_structured_v2!(h_scratch, obj_cm, cctx; threaded_bins = cctx.use_threaded_bins, tls = tls, use_syrk = true)
lp("Real production Hessian callback ran once -- cctx.Bidx/hzz_centered.ZcS now populated with live state.")

Bidx = cctx.Bidx
ZcS = cctx.hzz_centered.ZcS

const REPS = 5
function timeit_phases(f::Function; reps::Int = REPS)
    f()
    fills = Float64[]; reduces = Float64[]; cums = Float64[]; totals = Float64[]
    for i in 1:reps
        _, pt = f()
        push!(fills, pt.fill_and_join_s); push!(reduces, pt.reduction_s); push!(cums, pt.cumsum_s); push!(totals, pt.total_s)
    end
    return (fill = sum(fills)/reps, reduce = sum(reduces)/reps, cumsum = sum(cums)/reps, total = sum(totals)/reps)
end

rows = NamedTuple[]
function record!(candidate, workers, jtile, reduce_workers, pt)
    push!(rows, (candidate = candidate, julia_threads = NT, workers = workers, jtile = jtile, reduce_workers = reduce_workers,
        fill_and_join_s = pt.fill, reduction_s = pt.reduce, cumsum_s = pt.cumsum, total_s = pt.total,
        reduction_pct_of_total = 100 * pt.reduce / pt.total))
    @printf("  [%-20s] workers=%-3d jtile=%-4d reduce_workers=%-3d fill=%.4fs reduce=%.4fs(%.1f%%) cumsum=%.4fs total=%.4fs\n",
        candidate, workers, jtile, reduce_workers, pt.fill, pt.reduce, 100*pt.reduce/pt.total, pt.cumsum, pt.total)
    flush(stdout)
end

lp("\n", "="^100, "\n=== H_CZ phase breakdown: serial reduction vs surgical parallel reduction ===")
for workers in [1, 4, 8, 10]
    dc = BinZCrossDrawChunkScratch(D, L, nz, workers)
    ws = BinZCrossScratch(D, L, nz)
    pt = timeit_phases(() -> bin_zc_cross_hessian_fill_drawchunk_reordered_timed!(ws, dc, Bidx, ZcS; workers = workers, jtile = 64))
    record!("reordered_serial_reduce", workers, 64, 1, pt)

    for reduce_workers in [4, 8, 10]
        dc2 = BinZCrossDrawChunkScratch(D, L, nz, workers)
        ws2 = BinZCrossScratch(D, L, nz)
        pt2 = timeit_phases(() -> bin_zc_cross_hessian_fill_drawchunk_reordered_parreduce!(ws2, dc2, Bidx, ZcS; workers = workers, jtile = 64, reduce_workers = reduce_workers))
        maxdiff = maximum(abs.(ws2.ZBinTab .- ws.ZBinTab))
        record!("reordered_parallel_reduce", workers, 64, reduce_workers, pt2)
        @printf("      (parallel-reduce maxdiff vs serial-reduce = %.3e)\n", maxdiff)
    end
end

outpath = joinpath(D4X, "..", "..", "docs", "HCZ_CLOSEOUT_BREAKDOWN_2026-08-01.csv")
mkpath(dirname(outpath))
open(outpath, "w") do io
    println(io, "candidate,julia_threads,workers,jtile,reduce_workers,fill_and_join_s,reduction_s,cumsum_s,total_s,reduction_pct_of_total")
    for r in rows
        println(io, "$(r.candidate),$(r.julia_threads),$(r.workers),$(r.jtile),$(r.reduce_workers),$(r.fill_and_join_s),$(r.reduction_s),$(r.cumsum_s),$(r.total_s),$(r.reduction_pct_of_total)")
    end
end
lp("Wrote ", outpath)
lp("DONE")
