# WHERE does an outer evaluation's wall-clock actually go, for the two cross families? (2026-08-11)
#
# The existing attribution (00_READ_FIRST_CORRECTION.md / key_results/15_iters_vs_per_iteration) is
# STALE for two reasons, and both bias it:
#   1. it was measured at 1 BLAS thread -- the 2026-08-10 gate fix gives H_ZZ 8 threads, which the
#      kernel benchmark shows is worth 4.28x on that block ALONE, so the balance between H_ZZ and
#      everything else has necessarily shifted;
#   2. it was measured at the CALIBRATION point (Delta* ~ 0.039), whereas the 8-hour runs spent
#      almost all of their wall-clock near the Delta ~ 1 constraint boundary, where the inner solve
#      needs far more iterations. Per-eval cost in the real runs rose from 207s (eval 1) to ~800s.
#
# So this profiles at each run's OWN FINAL INCUMBENT, read out of its checkpoint -- i.e. at the
# point where the time was actually spent -- and optionally at the calibration point for contrast.
#
# It runs through the REAL public driver (run_{cm,originzc}_upper_checkpointed) rather than calling
# the inner dispatch directly: memory feedback-archC-verified-state-direct-call-knitro-callback-err
# records a confirmed KN_RC_CALLBACK_ERR trap for direct low-level calls at real D=20. Both
# profiling switches are process-global and accumulate into the SAME PROF_TIMES store, so
# prof_summary() at the end sees the coarse (FG vs Hessian callback) split AND the inside-the-
# callback sub-block breakdown from one run.
#
# Usage: julia --project=. -t 10 .../profile_cross_outer_eval_2026-08-11.jl <ozc|cmzc> [budget_s] [incumbent|calibration]
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
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Serialization, LinearAlgebra
using SpecialFunctions: gamma
lp(xs...) = (println(xs...); flush(stdout))

const FAM    = length(ARGS) >= 1 ? ARGS[1] : error("pass <ozc|cmzc>")
const BUDGET = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1800.0
const WHERE  = length(ARGS) >= 3 ? ARGS[3] : "incumbent"
FAM in ("ozc", "cmzc") || error("family must be ozc or cmzc")
WHERE in ("incumbent", "calibration") || error("point must be incumbent or calibration")

const RES  = joinpath(_D4E, "..", "..", "results")
const CKPT = FAM == "ozc" ?
    joinpath(RES, "ozc_cross_production_smoke_2026-08-09", "K3_W100000_sobol_randomized_upper8h_2026-08-10", "ozc_cross_K3_W100000_latest.jls") :
    joinpath(RES, "cmzc_cross_production_smoke_2026-08-09", "K3_W100000_sobol_randomized_upper8h_2026-08-10", "cmzc_cross_K3_W100000_latest.jls")
const KK = 3; const KSTAR = 2; const W = 100_000; const DELTA = 1.0; const CM_L = 50
const DESIGN = :sobol_randomized

lp("="^108)
lp("OUTER-EVAL PROFILE  family=", uppercase(FAM), "  point=", WHERE, "  budget=", BUDGET, "s  julia_threads=", Threads.nthreads())
lp("="^108)

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = DESIGN, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx0.D; Ddest = ctx0.D_dest

# ---- w0: the run's own final incumbent (where the wall-clock actually went), or calibration ----
local w0::Vector{Float64}
if WHERE == "incumbent"
    ck = FAM == "ozc" ? load_cm_checkpoint_v10(CKPT) : load_cm_checkpoint(CKPT)
    bf = ck.best_feasible
    bf === nothing && error("checkpoint has no best_feasible -- cannot profile at the incumbent")
    w0 = collect(Float64, bf.w)
    lp("incumbent from checkpoint: gp=", bf.gp, "  Delta=", bf.Delta, "  found at eval ", bf.n_eval)
    lp("  (this is the Delta~1 boundary region where the 8h run spent nearly all its wall-clock)")
else
    pe0 = build_pivot_elimination(ctx0); th0 = cm_fixed_theta(ctx0); xy0 = precompute_cm_aspace_xy(ctx0)
    xfc = ctx0.θ0_up[ctx0.free_idx]
    wac = vcat(xfc[1], cm_a_from_z(pivot_reduce(log.(reshape(xfc[2:end], D, Ddest)), pe0), th0, xy0, pe0))
    if FAM == "ozc"
        lay = OriginByPowerCrossLayout(D, KK, KK); aml = ActiveMeanLayout(lay, ctx0.bi, KSTAR, D)
        nud = Vector{Float64}(undef, n_eta(lay))
        for k in 1:KK, o in 1:D; nud[target_index(lay, o, k)] = gamma(1 - ctx0.μHat * k); end
    else
        lay = SharedByPowerCrossLayout(KK, KK); aml = ActiveMeanLayout(lay, ctx0.bi, KSTAR, D)
        nud = [gamma(1 - ctx0.μHat * k) for k in 1:KK]
    end
    w0 = vcat(wac, [log(nud[d]) for d in 1:length(nud) if d != aml.dense_omit_idx])
    lp("calibration point, gp=", w0[1])
end
lp("w0 length = ", length(w0))

const OUT = joinpath(RES, "profile_cross_2026-08-11", "$(FAM)_$(WHERE)")
rm(OUT; force = true, recursive = true); mkpath(OUT)

# BOTH switches on: coarse (@prof: inner_dual_fg_callback*, inner_dual_hessian_callback_archC,
# inner_knitro_dual_solve_arch) and fine (@cmhess_prof: H_ZZ / H_ER / H_EE_core / H_CZ_prep / ...).
# 2026-08-11: optional H_CZ backend override, so the btranspose candidate can be validated
# END-TO-END inside a real Hessian callback (where cache pressure from H_ZZ's ~1.4GB working set
# could change what an isolated kernel benchmark measures), not just in isolation.
if haskey(ENV, "HCZ_BACKEND")
    isdefined(Main, :bin_zc_cross_hessian_fill_drawchunk_btranspose!) ||
        include(joinpath(_D4E, "hcz_btranspose_candidate_2026-08-11.jl"))
    HCZ_PREP_BACKEND_DEFAULT[] = Symbol(ENV["HCZ_BACKEND"])
    lp(">> HCZ_PREP_BACKEND_DEFAULT[] overridden to :", HCZ_PREP_BACKEND_DEFAULT[])
end
prof_reset!()
PROF_ENABLED[] = true
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = true
# Inner-solve counters: how many INNER KNITRO solves the outer loop actually issued, and how many
# inner iterations in total. Without these the Hessian-call count cannot be turned into
# "iterations per inner solve", and "one inner solve per outer eval" is only an assumption
# (cb_G! may or may not reuse cb_F!'s state depending on whether shared.w == w).
const S0 = CS.INNER_SOLVE_COUNT[]; const I0 = CS.INNER_ITERS_TOTAL[]; const F0 = CS.INNER_INFEAS_COUNT[]

t0 = time()
result = FAM == "ozc" ?
    run_originzc_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = DESIGN, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = KK, K_pair = KK,
        power_target_layout = :origin_by_power_cross, originzc_profiled_level = KSTAR,
        inner_lower_limit = -10.0, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
        ckpt_dir = OUT, run_id = "prof_$(FAM)", label = "prof_$(FAM)",
        checkpoint_interval_s = 1e9, maxtime_real = BUDGET, verbose = true) :
    run_cm_upper_checkpointed(w0; W = W, delta = DELTA, draw_design = DESIGN, draw_seed = 20260719,
        L = CM_L, contrasts = :orthonormal, probs = nested_grid_sequence([10, 20, 50])[CM_L],
        include_truncated_moment = true, cm_extension = :cm_plus_moments,
        meanzc_K_mean = KK, meanzc_K_pair = KK,
        # 2026-08-11: MEANZC_LAYOUT lets this same profiler gate the DIAGONAL CM+ZC family
        # (:shared_by_power) as well as the cross one -- both share the H_CZ code path being changed.
        meanzc_target_layout = Symbol(get(ENV, "MEANZC_LAYOUT", "shared_by_power_cross")),
        meanzc_profiled_level = KSTAR, A_coordinate_mode = :powered_aspace, inner_lower_limit = -10.0,
        destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
        ckpt_dir = OUT, run_id = "prof_$(FAM)", label = "prof_$(FAM)",
        checkpoint_interval_s = 1e9, maxtime_real = BUDGET, verbose = true)
wall = time() - t0
CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] = false

lp("\n", "="^108)
@printf("PROFILED RUN: wall=%.1fs  n_eval=%d  n_grad=%d  status=%s\n", wall, result.n_eval, result.n_grad, string(result.knitro_status))
@printf("  => %.1f s per outer evaluation (%d evals in %.1f s of driver wall)\n",
        result.n_eval == 0 ? NaN : wall / result.n_eval, result.n_eval, wall)
lp("="^108)

rows = [(lab, sum(v), length(v), sum(v) / length(v)) for (lab, v) in PROF_TIMES if !isempty(v)]
sort!(rows, by = r -> -r[2])
lp("\nALL PROFILER LABELS, by total wall (coarse @prof + fine @cmhess_prof in one table).")
lp("NOTE the labels are NESTED, not disjoint -- inner_dual_hessian_callback_archC CONTAINS H_EE,")
lp("which CONTAINS H_EE_core/H_ER/H_ER_prep/H_ZZ. Do NOT sum them as a partition.\n")
@printf("  %-42s %12s %8s %12s %8s\n", "label", "total_s", "calls", "mean_s", "%wall")
for (lab, tot, n, mu) in rows
    @printf("  %-42s %12.3f %8d %12.5f %7.1f%%\n", lab, tot, n, mu, 100 * tot / wall)
end

# ---- the decomposition that answers "where is the bottleneck" ----
gett(l) = haskey(PROF_TIMES, l) ? sum(PROF_TIMES[l]) : 0.0
getn(l) = haskey(PROF_TIMES, l) ? length(PROF_TIMES[l]) : 0
# Pick whichever Hessian/FG label this family actually emitted, rather than guessing one: the
# origin-ZC path uses archA_partitioned, the CM path archC(_v2). Guessing produced a table of NaNs
# on the first run of this script.
const HESS_LABELS = ["inner_dual_hessian_callback_archA_partitioned", "inner_dual_hessian_callback_archC_v2",
                     "inner_dual_hessian_callback_archC", "inner_dual_hessian_callback"]
const FG_LABELS   = ["originZC_FG_callback", "meanZC_FG_callback", "inner_dual_fg_callback_compressed_v2",
                     "inner_dual_fg_callback_compressed", "inner_dual_fg_callback"]
hess  = sum(gett.(HESS_LABELS)); hessn = sum(getn.(HESS_LABELS))
fg    = sum(gett.(FG_LABELS));   fgn   = sum(getn.(FG_LABELS))
zz = max(gett("H_ZZ"), gett("originZC_H_ZZ_gram") + gett("originZC_H_ZZ_weight"))
nsolve = CS.INNER_SOLVE_COUNT[] - S0; niter = CS.INNER_ITERS_TOTAL[] - I0; ninfeas = CS.INNER_INFEAS_COUNT[] - F0
solve = hess + fg   # measured inner-solve work; the rest of the inner solve is KNITRO's own linear algebra

lp("\n", "="^108)
lp("DECOMPOSITION")
lp("="^108)
# Denominator is the DRIVER wall (the run_*_upper_checkpointed call), which includes the driver's
# own one-time context build. We do not subtract an unmeasured "context build" estimate -- the
# residual line below names it explicitly instead.
solve_wall = wall
@printf("  driver wall (incl. its own context build)      %10.1f s   (100%%)  = %.1f s per outer eval\n",
        solve_wall, result.n_eval == 0 ? NaN : solve_wall/result.n_eval)
@printf("    Hessian callbacks                            %10.1f s   (%4.1f%%)  calls=%d  mean=%.4f s\n", hess, 100*hess/solve_wall, hessn, hessn==0 ? NaN : hess/hessn)
@printf("      of which H_ZZ restriction gram             %10.1f s   (%4.1f%%)            (%4.1f%% of the Hessian)\n", zz, 100*zz/solve_wall, hess==0 ? NaN : 100*zz/hess)
@printf("    FG callbacks                                 %10.1f s   (%4.1f%%)  calls=%d  mean=%.5f s\n", fg, 100*fg/solve_wall, fgn, fgn==0 ? NaN : fg/fgn)
@printf("    RESIDUAL (KNITRO's own LA + %2d outer grads   %10.1f s   (%4.1f%%)\n", result.n_grad, solve_wall-hess-fg, 100*(solve_wall-hess-fg)/solve_wall)
lp("      + verification + screens + checkpointing + the driver's own context build/JIT)")
lp()
@printf("  INNER SOLVES issued              : %6d   (%.2f per outer eval; n_eval=%d, n_grad=%d)\n",
        nsolve, result.n_eval == 0 ? NaN : nsolve/result.n_eval, result.n_eval, result.n_grad)
@printf("  inner KNITRO iterations (total)  : %6d   (%.1f per inner solve)\n", niter, nsolve == 0 ? NaN : niter/nsolve)
@printf("  inner solves flagged infeasible  : %6d\n", ninfeas)
@printf("  Hessian calls per inner solve    : %6.1f\n", nsolve == 0 ? NaN : hessn/nsolve)
@printf("  FG calls per inner solve         : %6.1f\n", nsolve == 0 ? NaN : fgn/nsolve)
@printf("  FG calls per Hessian call        : %6.2f\n", hessn == 0 ? NaN : fgn/hessn)
@printf("  Hessian+FG per outer eval        : %6.1f s of %.1f s  (%.0f%% accounted)\n",
        result.n_eval == 0 ? NaN : (hess+fg)/result.n_eval, result.n_eval == 0 ? NaN : solve_wall/result.n_eval,
        solve_wall == 0 ? NaN : 100*(hess+fg)/solve_wall)
lp("\ntrace:")
for r in result.trace
    @printf("    eval %3d t=%8.1fs gp=%.10f Delta=%.10f feasible=%s verified=%s\n", r.idx, r.t, r.gp, r.Delta, string(r.feasible), string(r.verified))
end
flush(stdout)
