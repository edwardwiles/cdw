# Gate + benchmark for the H_CZ bin-major accumulator candidate (2026-08-11).
#
# Builds the REAL CM+ZC-CROSS production context at the real production shape (D20 real data,
# W=100,000, K_mean=K_pair=3, L=50, two CM families) so that D / L / nz / the Bidx bin distribution
# are the real ones -- the whole candidate is a cache-behaviour change, so measuring it at a toy
# shape would measure nothing.
#
# GATE: :draw_chunk_btranspose must be BIT-IDENTICAL to the production default
# :draw_chunk_reordered. It is a pure memory-layout permutation with the same accumulation order
# into every logical cell, so anything short of bit-identity means the permutation is wrong.
# (Its siblings only claim HCZ_CANDIDATE_TOL agreement because they genuinely reorder summation.)
#
# Usage: julia --project=. -t 10 .../test_hcz_btranspose_2026-08-11.jl [W] [reps]
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_lookup_production.jl", "country_resolve.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl",
          "hcz_btranspose_candidate_2026-08-11.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Random, LinearAlgebra
lp(xs...) = (println(xs...); flush(stdout))

const W    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const REPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
# H_CZ prep is SHARED code: build_cm_meanzc_bin_ctx and its Hessian blocks are reused verbatim by
# the cross family, so hcz_prep_dispatch! serves BOTH. FAMILY selects which shape to gate at --
# they differ only in nz (diagonal K=3/3 -> 630; cross K=3/3 -> 1770).
const FAMILY = length(ARGS) >= 3 ? ARGS[3] : "cross"
FAMILY in ("cross", "diag") || error("family must be cross or diag")
const KK = 3; const CM_L = 50
ALL_PASS = Ref(true)
check(name, cond) = (ALL_PASS[] &= cond; lp(cond ? "PASS  " : "FAIL  ", name))

lp("="^100)
lp("H_CZ candidates: gate + benchmark   FAMILY=", FAMILY, "  W=", W, "  julia_threads=", Threads.nthreads())
lp("="^100)

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
pcx = FAMILY == "cross" ?
    build_cm_meanzc_cross_production_context(ctx, CS; L = CM_L, K_mean = KK, K_pair = KK,
        include_truncated_moment = true, contrasts = :orthonormal, meanzc_basis = :direct,
        probs = nested_grid_sequence([10, 20, 50])[CM_L]) :
    build_cm_meanzc_production_context(ctx, CS; L = CM_L, K_mean = KK, K_pair = KK,
        include_truncated_moment = true, contrasts = :orthonormal, meanzc_basis = :direct,
        probs = nested_grid_sequence([10, 20, 50])[CM_L], moment_representation = :operator)
cctx = pcx.cctx
D = cctx.D; L = cctx.L
nz = n_restriction(cctx.hzz_zc_op)
workers = cctx.cross_hessian_workers
fam2 = cctx.n_families == 2
lp("REAL production shape: D=", D, "  L=", L, "  nz=", nz, "  workers=", workers, "  n_families=", cctx.n_families)
lp("  b-stride in the OLD (x,j,b) layout = D*nz = ", D * nz, " elements = ",
   @sprintf("%.0f KB", D * nz * 8 / 1024), ";  ", L + 1, " live targets span ",
   @sprintf("%.1f MB", (L + 1) * D * nz * 8 / 1048576))
lp("  b-stride in the NEW (b,x,j) layout = 1 element; ", L + 1, " live targets span ",
   @sprintf("%.0f bytes", (L + 1) * 8), " (L1-resident)")
lp("  inner-loop accumulates = D*nz*W = ", @sprintf("%.2e", D * nz * float(W)))

# Real Bidx (real bin distribution); ZcS values are irrelevant to timing and to the identity gate,
# so a deterministic RNG fill is used rather than driving a full inner solve to populate it.
Bidx = cctx.Bidx
rng = MersenneTwister(20260811)
ZcS = randn(rng, W, max(nz, 1))
Pow = fam2 ? abs.(randn(rng, W, D)) : nothing
lp("  Bidx bin occupancy (origin 1): min=", minimum(count(==(b), @view Bidx[:, 1]) for b in 1:(L+1)),
   " max=", maximum(count(==(b), @view Bidx[:, 1]) for b in 1:(L+1)))

ws_ref = ensure_bin_zc_cross_scratch!(nothing, D, L, nz)
ws_new = ensure_bin_zc_cross_scratch!(nothing, D, L, nz)
dc     = ensure_bin_zc_drawchunk_scratch!(nothing, D, L, nz, workers)
dcb    = ensure_bin_zc_btranspose_scratch!(nothing, D, L, nz, workers)

lp("\n---- correctness gate (must be BIT-IDENTICAL, not merely within tolerance) ----")
bin_zc_cross_hessian_fill_drawchunk_reordered!(ws_ref, dc, Bidx, ZcS; workers = workers, Pow = Pow)
bin_zc_cross_hessian_fill_drawchunk_btranspose!(ws_new, dcb, Bidx, ZcS; workers = workers, Pow = Pow)
check("ZBinTab   bit-identical to :draw_chunk_reordered", ws_new.ZBinTab == ws_ref.ZBinTab)
check("ZBinCScum bit-identical to :draw_chunk_reordered", ws_new.ZBinCScum == ws_ref.ZBinCScum)
if fam2
    check("ZBinTab_pow   bit-identical", ws_new.ZBinTab_pow == ws_ref.ZBinTab_pow)
    check("ZBinCScum_pow bit-identical", ws_new.ZBinCScum_pow == ws_ref.ZBinCScum_pow)
end
mx = maximum(abs.(ws_new.ZBinTab .- ws_ref.ZBinTab))
lp("    max |diff| on ZBinTab = ", mx, "   (must be exactly 0.0)")
check("ZBinTab is non-trivial (guards against comparing two zero arrays)", maximum(abs.(ws_ref.ZBinTab)) > 0.0)

lp("\n---- j-parallel candidate: correctness (tolerance-level, NOT bit-identical by design) ----")
ws_jp = ensure_bin_zc_cross_scratch!(nothing, D, L, nz)
bin_zc_cross_hessian_fill_jparallel!(ws_jp, Bidx, ZcS; workers = workers, Pow = Pow)
relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())
re = relerr(ws_jp.ZBinTab, ws_ref.ZBinTab); re_c = relerr(ws_jp.ZBinCScum, ws_ref.ZBinCScum)
lp("    rel err ZBinTab = ", re, "   ZBinCScum = ", re_c, "   (tol = ", HCZ_CANDIDATE_TOL, ")")
check("j-parallel ZBinTab   within HCZ_CANDIDATE_TOL", re < HCZ_CANDIDATE_TOL)
check("j-parallel ZBinCScum within HCZ_CANDIDATE_TOL", re_c < HCZ_CANDIDATE_TOL)
if fam2
    rep = relerr(ws_jp.ZBinTab_pow, ws_ref.ZBinTab_pow)
    lp("    rel err ZBinTab_pow = ", rep)
    check("j-parallel ZBinTab_pow within HCZ_CANDIDATE_TOL", rep < HCZ_CANDIDATE_TOL)
end

lp("\n---- benchmark (min of ", REPS, " reps, threaded, real shape) ----")
# `nw` is an explicit argument: the first version of this helper closed over the module-level
# `workers`, so every row of the thread sweep silently re-measured the same worker count (and then
# tripped the scratch's own workers-mismatch guard, which is what caught it).
function bench(f!, ws, scratch, nw::Int = workers)
    f!(ws, scratch, Bidx, ZcS; workers = nw, Pow = Pow)          # warm/compile
    minimum(@elapsed(f!(ws, scratch, Bidx, ZcS; workers = nw, Pow = Pow)) for _ in 1:REPS)
end
t_reordered = bench(bin_zc_cross_hessian_fill_drawchunk_reordered!, ws_ref, dc)
t_btrans    = bench(bin_zc_cross_hessian_fill_drawchunk_btranspose!, ws_new, dcb)
t_threadloc = bench(bin_zc_cross_hessian_fill_drawchunk!, ws_ref, dc)   # the older default, for context
bench_jp(nw) = (bin_zc_cross_hessian_fill_jparallel!(ws_jp, Bidx, ZcS; workers = nw, Pow = Pow);
                minimum(@elapsed(bin_zc_cross_hessian_fill_jparallel!(ws_jp, Bidx, ZcS; workers = nw, Pow = Pow)) for _ in 1:REPS))
t_jpar = bench_jp(workers)

@printf("  :draw_chunk_thread_local (older default)  %8.4f s\n", t_threadloc)
@printf("  :draw_chunk_reordered    (CURRENT default)%8.4f s   <- baseline\n", t_reordered)
@printf("  :draw_chunk_btranspose   (CANDIDATE A)    %8.4f s   speedup vs current = %.2fx\n",
        t_btrans, t_reordered / t_btrans)
@printf("  j-parallel               (CANDIDATE B)    %8.4f s   speedup vs current = %.2fx\n",
        t_jpar, t_reordered / t_jpar)

lp("\n---- THREAD SCALING (workers sweep; nthreads=", Threads.nthreads(), ") ----")
@printf("  %8s  %12s  %12s  %12s   %10s\n", "workers", "reordered", "btranspose", "j-parallel", "scratch_MB")
for nw in filter(<=(Threads.nthreads()), [1, 2, 4, 8, 16, 32])
    dcn  = ensure_bin_zc_drawchunk_scratch!(nothing, D, L, nz, nw)
    dcbn = ensure_bin_zc_btranspose_scratch!(nothing, D, L, nz, nw)
    tr = bench(bin_zc_cross_hessian_fill_drawchunk_reordered!, ws_ref, dcn, nw)
    tb = bench(bin_zc_cross_hessian_fill_drawchunk_btranspose!, ws_new, dcbn, nw)
    tj = bench_jp(nw)
    @printf("  %8d  %10.4f s  %10.4f s  %10.4f s   %8.0f MB\n", nw, tr, tb, tj,
            (L + 1) * D * nz * nw * 8 * (fam2 ? 2 : 1) / 2^20)
    flush(stdout)
end

lp("\n---- projected effect on the CM+ZC-CROSS Hessian callback ----")
lp("Measured 2026-08-11 at the real incumbent: callback = 6.251 s/call, of which H_CZ_prep = 2.122 s.")
saved = 2.122 * (1 - t_btrans / t_reordered)
lp(@sprintf("  H_CZ_prep 2.122 s -> %.3f s   (saves %.3f s/call)", 2.122 * t_btrans / t_reordered, saved))
lp(@sprintf("  callback  6.251 s -> %.3f s   (%.1f%% faster per Hessian call)", 6.251 - saved, 100 * saved / 6.251))
lp(@sprintf("  at ~105 Hessian calls per outer eval: saves ~%.0f s per outer evaluation", 105 * saved))

lp("\n", ALL_PASS[] ? "ALL PASS" : "SOME CHECKS FAILED")
flush(stdout)
