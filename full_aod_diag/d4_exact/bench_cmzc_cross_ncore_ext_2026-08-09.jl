# NCORE_ext memory/time benchmark for CM+ZC-CROSS (2026-08-09) -- the single biggest feasibility
# risk identified in the CM+ZC-CROSS handover, measured rather than assumed BEFORE any campaign
# commitment.
#
# THE CONCERN, stated precisely: `NCORE_ext = ncore_econ + n_mean + n_pair`
# (cm_meanzc_production.jl:84) feeds several dense buffers in `build_cm_meanzc_bin_ctx`, the largest
# being a packed `(NCORE_ext + ncm)^2` Hessian (`:131`) and a `(W, NCORE_ext)` scratch. At D=20/K=3
# the CROSS pair block is 9x the diagonal family's (K_pair^2=9 vs K_pair=3 blocks of npair=190), so
# NCORE_ext grows from ~1032 to ~2172 and those buffers grow roughly quadratically / linearly
# respectively. Unlike OZC-CROSS this family ALSO carries the CM-grid block (ncm), which enters the
# packed Hessian's dimension additively -- hence the separate measurement.
#
# WHAT THIS REPORTS, for BASE (diagonal) vs CROSS at matched (K_mean,K_pair):
#   * NCORE_ext, ncm, n_pair, and the derived buffer sizes in MB (computed from the actual cctx
#     field dimensions, not from a formula re-derived here);
#   * peak RSS growth across the context build (Sys.maxrss delta);
#   * wall-clock per inner solve (2 solves: first includes any lazy scratch allocation, second is
#     the steady-state number a campaign would actually pay);
#   * Delta_dual + inner_status, so the cost is reported next to what it buys.
#
# Usage:
#   julia --project=. -t 8 full_aod_diag/d4_exact/bench_cmzc_cross_ncore_ext_2026-08-09.jl [d4|d20] [W]
# Defaults: d4. For D20 the standard production manifest is used (sigma=3, sobol_randomized,
# Brazil-Korea gravity exclusion, exclude_row), W defaults to 100_000 (trap 3: W=8000 does NOT work
# at D20 -- the calibration point is degenerate at that scale; use W >= 80,000).
const D4X = @__DIR__
cd(D4X)
const SCALE = length(ARGS) >= 1 ? ARGS[1] : "d4"
const WARG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000

_files = ["context.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "compressed_moments.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "compressed_factual_buffer_reuse.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "winner_pair_cross_hessian.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "threaded_cross_hessian.jl",
          "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl", "operator_hessian_weights.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "autarky_cf.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl"]
for f in _files
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

lp(xs...) = (println(xs...); flush(stdout))
mb(nbytes) = nbytes / 1024^2

if SCALE == "d4"
    ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
    L = 8
    KS = [(1, 1), (2, 2), (3, 3)]
else
    GRAV = default_gravity_exclude_cells_brazil_korea()
    ctx = d20_real_setup_design(W = WARG, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, WARG)
    L = 50
    KS = [(2, 2), (3, 3)]
end
D = ctx.D
probs = cm_equal_grid_probs(L)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_shared(K::Int) = [gamma(1 - ctx.μHat * k) for k in 1:K]

lp("="^110)
lp("CM+ZC-CROSS NCORE_ext benchmark   scale=", SCALE, "  D=", D, "  W=", size(ctx.U, 1), "  L=", L,
   "  threads=", Threads.nthreads())
lp("="^110)

results = NamedTuple[]

function run_one(kind::Symbol, K_mean::Int, K_pair::Int)
    νvec = nu0_shared(K_mean)
    GC.gc(); GC.gc()
    rss0 = Sys.maxrss()
    tb0 = time()
    pcx = kind === :cross ?
        build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
            include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs) :
        build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
            include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs,
            moment_representation = :operator)
    t_build = time() - tb0
    rss1 = Sys.maxrss()
    cctx = pcx.cctx
    NCORE_ext = cctx.NCORE
    ncm = cctx.ncm
    W = size(ctx.U, 1)
    # Buffer sizes read off the ACTUAL allocated cctx fields, not re-derived from a formula.
    packed_mb = mb(length(cctx.Hfull) * 8)   # (NCORE_ext+ncm)^2 packed Hessian scratch (CMBinHessCtx.Hfull)
    ews_mb = mb(length(cctx.Ews) * 8)

    t1 = time(); base, verify = archC_meanzc_verified_state(x_free_calib, νvec, pcx.ctx_cm, cctx); t_solve1 = time() - t1
    t2 = time(); base2, verify2 = archC_meanzc_verified_state(x_free_calib, νvec, pcx.ctx_cm, cctx); t_solve2 = time() - t2

    r = (kind = kind, K_mean = K_mean, K_pair = K_pair, NCORE_ext = NCORE_ext, ncm = ncm,
         n_mean = pcx.aug.n_mean, n_pair = pcx.aug.n_pair,
         packed_mb = packed_mb, ews_mb = ews_mb, rss_delta_mb = mb(rss1 - rss0),
         t_build = t_build, t_solve1 = t_solve1, t_solve2 = t_solve2,
         status = verify.inner_status, Delta = verify.Delta_dual)
    push!(results, r)
    @printf("  %-6s K=%d/%d  NCORE_ext=%5d  ncm=%5d  n_pair=%6d  packedH=%8.1f MB  Ews=%8.1f MB  rss_delta=%8.1f MB\n",
            String(kind), K_mean, K_pair, NCORE_ext, ncm, pcx.aug.n_pair, packed_mb, ews_mb, mb(rss1 - rss0))
    @printf("          t_build=%7.2fs  t_solve(1st)=%8.2fs  t_solve(2nd)=%8.2fs  status=%4d  Delta=%.10e\n",
            t_build, t_solve1, t_solve2, verify.inner_status, verify.Delta_dual)
    flush(stdout)
    return r
end

for (K_mean, K_pair) in KS
    lp("\n---- K_mean=$K_mean K_pair=$K_pair ----")
    rb = run_one(:base, K_mean, K_pair)
    rc = run_one(:cross, K_mean, K_pair)
    @printf("  RATIO cross/base:  NCORE_ext %.2fx   packedH %.2fx   Ews %.2fx   t_solve(2nd) %.2fx   Delta %.3fx\n",
            rc.NCORE_ext / rb.NCORE_ext, rc.packed_mb / rb.packed_mb, rc.ews_mb / rb.ews_mb,
            rc.t_solve2 / max(1e-9, rb.t_solve2), rc.Delta / rb.Delta)
    flush(stdout)
end

lp("\n", "="^110)
lp("SUMMARY CSV")
lp("kind,K_mean,K_pair,NCORE_ext,ncm,n_mean,n_pair,packedH_MB,Ews_MB,rss_delta_MB,t_build_s,t_solve1_s,t_solve2_s,inner_status,Delta_dual")
for r in results
    @printf("%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%d,%.12e\n",
            String(r.kind), r.K_mean, r.K_pair, r.NCORE_ext, r.ncm, r.n_mean, r.n_pair,
            r.packed_mb, r.ews_mb, r.rss_delta_mb, r.t_build, r.t_solve1, r.t_solve2, r.status, r.Delta)
end
flush(stdout)
