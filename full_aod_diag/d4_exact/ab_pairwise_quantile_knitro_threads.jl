# ================================================================================================
# A/B: does giving KNITRO's own linear algebra 10 threads speed up the L=10 inner solve?
# (user question, 2026-08-11)
#
# The stock inner opt file (`full_aod_diag/ek_inner.opt`) runs KNITRO fully single-threaded:
# `par_numthreads 1`, `par_blasnumthreads 0`, `par_lsnumthreads 0` (0 = "derive from
# par_numthreads"). At D=20/L=10/W=100,000 the inner dual has n = 15,952 variables and KNITRO
# factorizes a DENSE ~15,952^2 KKT system per interior-point iteration -- roughly 1.4e12 flops --
# so a single thread is a plausible dominant cost. `ek_inner_pq_threads10.opt` raises ONLY the
# BLAS and linear-solver thread counts to 10 (see that file's header for why `par_numthreads` and
# `par_concurrent_evals` are deliberately left alone).
#
# ARM A: ek_inner.opt              (stock, 1 thread)
# ARM B: ek_inner_pq_threads10.opt (10 BLAS / 10 linear-solver threads)
#
# Both arms solve the SAME point, so the answer must not move: the gate is that `Delta_dual` agrees
# to solver reproducibility AND both verify. A speedup that changed the answer would not be a
# speedup. Run order is A then B in one process, and each arm's timing is taken on an already
# JIT-warm code path (the first arm pays compilation; see memory
# `feedback-solve-timing-jit-thread-warmstart-pitfalls-2026-08-01`) -- so arm A is additionally
# re-run at the end as a warm control, and it is the WARM A vs B comparison that is reported.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=16 julia --project=. \
#     full_aod_diag/d4_exact/ab_pairwise_quantile_knitro_threads.jl [W] [L] [cutoff_source]
# ================================================================================================
_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, SpecialFunctions

lp(xs...) = (println(xs...); flush(stdout))
const W_AB   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const L_AB   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const CUTSRC = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :frechet_theoretical
const GRAV   = default_gravity_exclude_cells_brazil_korea()
const OPT_A  = joinpath(@__DIR__, "..", "ek_inner.opt")
const OPT_B  = joinpath(@__DIR__, "..", "ek_inner_pq_threads10.opt")
# Arm C is opt-in via ARGS[4] because it raises par_numthreads, which lets KNITRO dispatch
# evaluation callbacks concurrently onto our SHARED scratch -- see that file's own header. The
# bit-identical-Delta gate below is what would expose a race.
const OPT_C  = joinpath(@__DIR__, "..", "ek_inner_pq_threads10_full.opt")
isfile(OPT_A) || error("missing $OPT_A")
isfile(OPT_B) || error("missing $OPT_B")
length(ARGS) < 4 || isfile(OPT_C) || error("missing $OPT_C")

lp("="^96)
lp("KNITRO LINEAR-ALGEBRA THREADING A/B: W=", W_AB, " L=", L_AB, " cutoff_source=:", CUTSRC)
lp("  ARM A: ", OPT_A, "   (par_numthreads 1, blas/ls threads derived = 1)")
lp("  ARM B: ", OPT_B, "   (par_blasnumthreads 10, par_lsnumthreads 10)")
lp("  JULIA_NUM_THREADS=", Threads.nthreads(), "  OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", "<unset>"))
lp("="^96)

# delta=50 so the early-abort threshold is Inf and cannot be mistaken for a failure or shorten
# either arm asymmetrically.
t0 = time()
ctx_raw = d20_real_setup_design(W = W_AB, δ = 50.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("context built in ", round(time() - t0, digits = 1), "s  (D=", ctx.D, " W=", ctx.W, ")")

layout = PairwiseQuantileMassLayout(ctx.D, L_AB)
pcx = build_pairwise_quantile_production_context(ctx, layout;
    cutoff_source = CUTSRC, min_bin_count = max(10, W_AB ÷ (2 * L_AB^2)))
ctx_cm = pcx.ctx_cm
geo = build_aspace_geometry(ctx)
w_cal = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], cm_fixed_theta(ctx),
    precompute_cm_aspace_xy(ctx), geo.pe)), geo.pe)
mass0 = uniform_mass_raw(layout)
lp("n = 1 + ncore + n_total_rows = ", ctx_cm.obj.outer_constr_index,
   "   (n_total_rows=", n_total_rows(ctx.D, L_AB), ")")

"One verified inner solve under `optfile`, returning (Delta_dual, wall, n_fg, n_hess, class)."
function timed_arm(label::AbstractString, optfile::AbstractString)
    ctx_cm.obj.inner_loop_opt = optfile
    lp("\n--- ARM ", label, ": ", basename(optfile), " ---")
    flush(stdout)
    t = time()
    base, v = archPQ_verified_state(xf, mass0, ctx_cm)
    wall = time() - t
    cls = classify_inner_result(v)
    @printf("  Delta_dual = %.17g   status=%d   class=%s   n_fg=%d   n_hess=%d   wall=%.1fs\n",
            v.Delta_dual, v.inner_status, string(cls), v.n_fg, v.n_hess, wall)
    lp("  block KKT: E=", v.kkt_resid_E, " marginal=", v.kkt_resid_marginalbin,
       " pair=", v.kkt_resid_pairindep)
    return (Delta = v.Delta_dual, wall = wall, n_fg = v.n_fg, n_hess = v.n_hess, cls = cls)
end

# ORDER MATTERS. The FIRST solve in a process pays all the JIT for the FG/Hessian callback chain,
# which at L=10 measured 1161.9 s against a warm 355.3 s -- a 3.3x difference for identical work and
# identical iteration counts. Every arm is therefore run TWICE and only the SECOND (warm) run of
# each is compared; the first pass exists solely to warm the process. Memory
# `feedback-solve-timing-jit-thread-warmstart-pitfalls-2026-08-01` is exactly about this trap, and
# the first version of this A/B fell into a milder form of it (reporting B against a cold A would
# have shown a fictitious 3.1x "speedup" from threading).
rA_cold = timed_arm("A pass 1 (JIT-warming, discarded)", OPT_A)
rB_cold = timed_arm("B pass 1 (discarded)", OPT_B)
rC_cold = length(ARGS) >= 4 ? timed_arm("C pass 1 (discarded)", OPT_C) : nothing
rA_warm = timed_arm("A pass 2 (WARM, reported)", OPT_A)
rB_warm = timed_arm("B pass 2 (WARM, reported)", OPT_B)
rC_warm = length(ARGS) >= 4 ? timed_arm("C pass 2 (WARM, reported)", OPT_C) : nothing

lp("\n", "="^96)
@printf("%-40s %18s %10s %8s %8s\n", "arm", "Delta_dual", "wall(s)", "n_fg", "n_hess")
rows = Any[("A pass 1 (cold, discarded)", rA_cold), ("B pass 1 (cold, discarded)", rB_cold),
           ("A pass 2 WARM  (1 thread)", rA_warm), ("B pass 2 WARM  (blas/ls=10)", rB_warm)]
rC_cold === nothing || push!(rows, ("C pass 1 (cold, discarded)", rC_cold))
rC_warm === nothing || push!(rows, ("C pass 2 WARM  (all threads=10)", rC_warm))
for (nm, r) in rows
    @printf("%-40s %18.16g %10.1f %8d %8d\n", nm, r.Delta, r.wall, r.n_fg, r.n_hess)
end
@printf("\nWARM A -> WARM B speedup: %.2fx  (%.1fs -> %.1fs)\n",
        rA_warm.wall / rB_warm.wall, rA_warm.wall, rB_warm.wall)
rC_warm === nothing || @printf("WARM A -> WARM C speedup: %.2fx  (%.1fs -> %.1fs)\n",
        rA_warm.wall / rC_warm.wall, rA_warm.wall, rC_warm.wall)
@printf("JIT tax on the first solve of a process: %.2fx  (%.1fs cold vs %.1fs warm, same arm)\n",
        rA_cold.wall / rA_warm.wall, rA_cold.wall, rA_warm.wall)
rB = rB_warm

ok = true
global ok
# The answer must not move. Both arms solve the same convex problem; a threading option changes
# only the linear algebra's execution, so anything beyond round-off here means the option is doing
# something it should not be.
if rC_warm !== nothing
    relC = abs(rC_warm.Delta - rA_warm.Delta) / max(abs(rA_warm.Delta), eps())
    @printf("relative |Delta_C - Delta_A_warm| = %.3e\n", relC)
    println(relC == 0.0 ? "PASS  arm C (par_numthreads=10) is BIT-IDENTICAL -- no sign of a callback race" :
                          "FAIL  arm C moved the answer -- callbacks are racing on shared scratch, do NOT adopt")
    global ok &= (relC == 0.0)
end
relΔ = abs(rB.Delta - rA_warm.Delta) / max(abs(rA_warm.Delta), eps())
@printf("relative |Delta_B - Delta_A_warm| = %.3e\n", relΔ)
global ok &= relΔ < 1e-9
println(relΔ < 1e-9 ? "PASS  threading does not change the answer" :
                      "FAIL  threading CHANGED the answer -- not a speedup, a bug")
global ok &= (rA_warm.cls == VerifiedSolved && rB.cls == VerifiedSolved)
println(rA_warm.cls == VerifiedSolved && rB.cls == VerifiedSolved ?
        "PASS  both arms VerifiedSolved" : "FAIL  an arm did not verify")
# Same iteration counts => the comparison is like-for-like rather than one arm taking a shorter path.
println(rA_warm.n_hess == rB.n_hess && rA_warm.n_fg == rB.n_fg ?
        "PASS  identical iteration counts (like-for-like wall-clock comparison)" :
        "NOTE  iteration counts differ (A: $(rA_warm.n_fg)/$(rA_warm.n_hess), B: $(rB.n_fg)/$(rB.n_hess)) -- compare per-iteration, not totals")
lp("="^96)
exit(ok ? 0 : 1)
