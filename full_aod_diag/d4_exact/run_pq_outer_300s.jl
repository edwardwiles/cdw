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
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl",
          "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, Serialization, SpecialFunctions
import Dates

using LinearAlgebra, Printf, Serialization, SpecialFunctions
import Dates

lp(xs...) = (println("[pq300 ", Dates.format(Dates.now(), "HH:MM:SS"), "] ", xs...); flush(stdout))

# ================================================================================================
# 300-second production-style OUTER loop, one L per invocation. Mirrors the production runner's
# stage A exactly -- same scientific constants read from paper_upper_v1.toml, same algorithm
# (pin_cg_lbfgs), same :min_gp objective, same :uniform mass start -- with the budget cut to 300 s
# and the delta grid pinned to the single cell the comparison asks for.
#   usage: julia --project=. run_pq_outer_300s.jl <L> <delta> <seconds> <outdir>
# ================================================================================================
length(ARGS) >= 4 || error("need <L> <delta> <seconds> <outdir>")
const PQ_L    = parse(Int, ARGS[1])
const DELTA   = parse(Float64, ARGS[2])
const SECONDS = parse(Float64, ARGS[3])
const OUTDIR  = abspath(ARGS[4])
mkpath(OUTDIR)

# ---- protocol [scientific], identical to run_pairwise_quantile_production.jl -------------------
const SIGMA, W_PROD              = 3.0, 100_000
const DRAW_DESIGN, DRAW_SEED     = :sobol_randomized, 20260719
const DESTINATION_SAMPLE         = :exclude_row
const EXCLUDE_DIAG_GRAV          = true
const INNER_LOWER_LIMIT          = -10.0
const A_COORD_MODE               = :powered_aspace
const FIND_SMALLEST              = true          # direction = "upper"
const Z_HALFWIDTH                = 30.0
const CUTOFF_SOURCE              = :frechet_theoretical
const MIN_BIN_COUNT              = max(10, W_PROD ÷ (2 * PQ_L^2))
const INNER_OPT                  = "ek_inner_pq_ma97_t10.opt"   # 10 solver threads, as requested
const GRAV = default_gravity_exclude_cells_brazil_korea()

lp("="^92)
lp("PQ 300s OUTER  L=", PQ_L, "  delta=", DELTA, "  budget=", SECONDS, "s  inner_opt=", INNER_OPT)
lp("="^92)

t0 = time()
ctx_raw = d20_real_setup_design(W = W_PROD, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, destination_sample = DESTINATION_SAMPLE,
    exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV, gravity_exclude_cells = GRAV,
    σHat = SIGMA, inner_lower_limit = INNER_LOWER_LIMIT)
ctx    = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
geo    = build_aspace_geometry(ctx)
w_cal_econ = cm_w0_from_calibration(ctx, geo.pe, A_COORD_MODE)
mass0  = uniform_mass_raw(layout)
W0     = vcat(w_cal_econ, mass0)
lp("context + w0 in ", round(time() - t0, digits = 1), "s;  n_total_rows=", n_total_rows(ctx.D, PQ_L),
   "  n_raw=", n_raw(layout), "  length(w0)=", length(W0))

t1 = time()
r = run_pairwise_quantile_upper_checkpointed(copy(W0);
    ckpt_dir = OUTDIR, label = "pq300_L$(PQ_L)_d$(replace(string(DELTA), "." => "p"))",
    delta = DELTA, maxtime_real = SECONDS, checkpoint_interval_s = 1e9,
    objective_mode = :min_gp, pin_outer_algorithm = true, verbose = true,
    find_smallest = FIND_SMALLEST, W = W_PROD, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    σHat = SIGMA, inner_lower_limit = INNER_LOWER_LIMIT, z_halfwidth = Z_HALFWIDTH,
    destination_sample = DESTINATION_SAMPLE, exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV,
    gravity_exclude_cells = GRAV, L = PQ_L, cutoff_source = CUTOFF_SOURCE,
    min_bin_count = MIN_BIN_COUNT, A_coordinate_mode = A_COORD_MODE,
    inner_opt_override = INNER_OPT)
el = time() - t1

lp("")
lp("RESULT300  L=", PQ_L,
   "  wall=", round(el, digits = 1), "s",
   "  n_eval=", r.n_eval, "  n_grad=", r.n_grad,
   "  s_per_eval=", r.n_eval > 0 ? round(el / r.n_eval, digits = 2) : NaN,
   "  status=", r.knitro_status,
   "  best_gp=", r.best === nothing ? "nothing" : r.best.gp,
   "  kappa=", r.kappa)
# `best_feasible` lives on the CHECKPOINT struct, not on this NamedTuple (whose fields are
# knitro_status/wall/n_eval/n_grad/best/kappa/xsol/trace/final_checkpoint/ckpt_path/...).
bf = hasproperty(r.final_checkpoint, :best_feasible) ? r.final_checkpoint.best_feasible : nothing
lp("RESULT300b L=", PQ_L,
   "  best_feasible_gp=", bf === nothing ? "nothing" : bf.gp,
   "  best_Delta=", r.best === nothing ? "nothing" : r.best.Delta,
   "  best_feasible_Delta=", bf === nothing ? "nothing" : bf.Delta)
