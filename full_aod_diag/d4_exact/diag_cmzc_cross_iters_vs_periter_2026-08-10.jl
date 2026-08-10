# WHY is the CROSS inner solve slow: MORE ITERATIONS, or SLOWER ITERATIONS? (2026-08-10)
#
# bench_cmzc_cross_ncore_ext_2026-08-09.jl established the WALL-CLOCK gap (D20/W=100k: CM+ZC cross
# vs base, 54.5s vs 1.89s at K=2/2 and 293.4s vs 2.92s at K=3/3) but recorded no iteration counts,
# so it cannot distinguish the two explanations. This script measures both directly:
#
#   iterations      CS.INNER_ITERS_TOTAL[] delta (KNITRO's own KN_get_number_iters)
#   n_fg            st.n_fg_calls -- combined objective+gradient callbacks
#   n_hess          Hessian callbacks actually issued
#   wall            seconds
#   derived         s/iteration, s/FG, s/Hessian
#
# and, with CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED on, the per-sub-block breakdown INSIDE the Hessian
# callback (H_EE_core / H_ER / H_ZZ / H_CZ ...), which is what would localise a per-iteration cost
# increase to a specific kernel.
#
# TWO REGIMES ARE REPORTED, and the distinction matters for interpreting a campaign:
#   COLD   -- first solve at a freshly built context (obj.x untouched)
#   WARM   -- immediate re-solve at the SAME point (obj.x holds the previous converged iterate)
# The WARM number is the flattering one and is NOT what a campaign pays: an outer loop re-solves at
# a DIFFERENT theta each eval. The real production smoke spaced its outer evals ~255s apart at
# K=2/2 cross, which tracks the COLD figure (187s), not the WARM one (54.5s).
#
# NOTE: this is a solver-behaviour diagnostic, NOT a correctness gate. It changes no production code.
#
# Usage: julia --project=. -t 8 .../diag_cmzc_cross_iters_vs_periter_2026-08-10.jl [K] [W]
const _D4E = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "winner_pair_cross_hessian.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "threaded_cross_hessian.jl", "zc_gram_blas_candidates.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "hcz_reordered_candidate_2026-08-01.jl",
          "hez_drawmajor_candidate_2026-08-01.jl", "hez_drawmajor_v2_candidate_2026-08-01.jl",
          "operator_hessian_weights.jl", "cm_hessian_subblock_profiling.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "autarky_cf.jl", "country_resolve.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra
using SpecialFunctions: gamma
lp(xs...) = (println(xs...); flush(stdout))

const KK = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const W  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100_000
const L  = 50

ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
D = ctx.D
probs = cm_equal_grid_probs(L)
x_free = ctx.θ0_up[ctx.free_idx]
νvec = [gamma(1 - ctx.μHat * k) for k in 1:KK]

lp("="^108)
lp("CROSS inner solve: ITERATIONS vs PER-ITERATION COST    D=", D, " W=", W, " L=", L,
   " K=", KK, "/", KK, "  threads=", Threads.nthreads())
lp("="^108)

"One measured solve through the SAME dispatch archC_meanzc_verified_state uses (_meanzc_fg_dispatch)."
function measure(pcx, tag)
    cctx = pcx.cctx; obj = pcx.ctx_cm.obj
    θ_econ = CS.reconstruct_full(x_free, pcx.ctx_cm.m)
    θ_ext = vcat(θ_econ, νvec)
    cctx.nu_ref[] = collect(νvec)          # archC_meanzc_base_state does exactly this before dispatch
    it0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    _, _, nStatus, n_fg, n_hess = _meanzc_fg_dispatch(cctx, obj, θ_ext)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - it0
    @printf("  %-22s status=%4d  iters=%5d  n_fg=%6d  n_hess=%5d  wall=%8.2fs | %8.4f s/iter  %8.5f s/FG  %8.4f s/hess\n",
            tag, nStatus, iters, n_fg, n_hess, wall,
            iters == 0 ? NaN : wall / iters, n_fg == 0 ? NaN : wall / n_fg, n_hess == 0 ? NaN : wall / n_hess)
    flush(stdout)
    return (tag = tag, status = nStatus, iters = iters, n_fg = n_fg, n_hess = n_hess, wall = wall)
end

results = NamedTuple[]
for kind in (:base, :cross)
    lp("\n---- ", uppercase(String(kind)), " ----")
    pcx = kind === :cross ?
        build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = KK, K_pair = KK,
            include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs) :
        build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = KK, K_pair = KK,
            include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs,
            moment_representation = :operator)
    nx = n_restriction(pcx.cctx.hzz_zc_op)
    lp("    NCORE_ext=", pcx.cctx.NCORE, "  ncm=", pcx.cctx.ncm, "  n_restriction=", nx)
    # Sub-block profiling must be ON for the COLD solve -- that is the only solve guaranteed to issue
    # Hessian callbacks. (First version of this script enabled it for a third, post-convergence
    # re-solve, which does ZERO iterations and therefore recorded nothing at all.)
    prof_reset!(); CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
    push!(results, merge(measure(pcx, "$(kind) COLD"), (kind = kind, phase = :cold, nx = nx)))
    CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = false
    push!(results, merge(measure(pcx, "$(kind) WARM"), (kind = kind, phase = :warm, nx = nx)))
    lp("    Hessian sub-block totals (s), largest first:")
    rows = [(lab, sum(v), length(v)) for (lab, v) in PROF_TIMES if !isempty(v)]
    sort!(rows, by = r -> -r[2])
    for (lab, tot, n) in first(rows, min(10, length(rows)))
        @printf("      %-34s total=%8.3fs  calls=%6d  mean=%9.5fs\n", lab, tot, n, tot / n)
    end
end

lp("\n", "="^108)
lp("SUMMARY: is it MORE iterations, or SLOWER iterations?")
lp("="^108)
for phase in (:cold, :warm)
    b = findfirst(r -> r.kind === :base && r.phase === phase, results)
    c = findfirst(r -> r.kind === :cross && r.phase === phase, results)
    (b === nothing || c === nothing) && continue
    rb = results[b]; rc = results[c]
    @printf("%s:  wall %6.2fx   iters %6.2fx   n_fg %6.2fx   n_hess %6.2fx   s/iter %6.2fx   n_restriction %5.2fx\n",
            uppercase(String(phase)),
            rc.wall / rb.wall,
            rb.iters == 0 ? NaN : rc.iters / rb.iters,
            rb.n_fg == 0 ? NaN : rc.n_fg / rb.n_fg,
            rb.n_hess == 0 ? NaN : rc.n_hess / rb.n_hess,
            (rb.iters == 0 || rc.iters == 0) ? NaN : (rc.wall / rc.iters) / (rb.wall / rb.iters),
            rc.nx / rb.nx)
end
lp("\nRead: if `iters ~ 1x` and `s/iter ~ wall`, the cost is PER-ITERATION (bigger kernels).")
lp("      If `iters ~ wall` and `s/iter ~ 1x`, the cost is ITERATION COUNT (harder problem).")
lp("      Both >1 means both contribute; the product should reconcile to the wall ratio.")
flush(stdout)
