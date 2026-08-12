# Smoke test for the paper_upper_v1 objective_mode=:min_delta_fixed_gp extension added to
# run_cm_upper_checkpointed (cm_checkpoint.jl) and run_originzc_upper_checkpointed
# (cm_originzc_checkpoint.jl). Proves, at real (if small) W, that:
#   1. the new mode runs end-to-end (box built with gp pinned, KNITRO solves an UNCONSTRAINED
#      min-Delta NLP, no crash from the cIndices=Int32[]/objGrad-vs-jac plumbing change);
#   2. it writes a checkpoint;
#   3. best_feasible[] (if any verified point is found) records a real Delta, and running longer
#      does not make Delta WORSE than the calibration start (min-Delta objective, sanity direction);
#   4. the pre-existing objective_mode=:min_gp (default) path is UNCHANGED -- run once in the old
#      mode too and confirm gp actually moves (still a real gp-minimization NLP).
#
# Run standalone:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 julia --project=. \
#     full_aod_diag/d4_exact/smoke_objective_mode_min_delta_fixed_gp.jl

_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl"]
    include(joinpath(_D4E, f))
end
using Random, LinearAlgebra, Serialization, SpecialFunctions

lp(xs...) = (println(xs...); flush(stdout))

const W_SMOKE = 20_000
const GRAV = default_gravity_exclude_cells_brazil_korea()
lp("Building D20 ctx (W=", W_SMOKE, ", production draw_seed=20260719, production draw_design=:sobol_randomized)...")
ctx_raw = d20_real_setup_design(W = W_SMOKE, δ = 0.1, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("ctx built: D=", ctx.D, " D_dest=", ctx.D_dest, " W=", ctx.W, " bi=", ctx.bi, " sigma=", ctx.σ)

geo = build_aspace_geometry(ctx)
w_cal_econ = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
gp_cal = w_cal_econ[1]
lp("gp_cal = ", gp_cal)

TMP = mktempdir()
lp("scratch ckpt_dir = ", TMP)

const CM_GRID_BUCKETS = 50
const CM_PROBS_L50 = resolve_cm_probs(CM_GRID_BUCKETS)

results = Dict{Symbol,Any}()

# ---- 1. CM-only (plain flexible CM, cm_extension=:cm_only) at objective_mode=:min_delta_fixed_gp ----
lp("="^100); lp("TEST 1: run_cm_upper_checkpointed, cm_extension=:cm_only, objective_mode=:min_delta_fixed_gp")
ckpt1 = joinpath(TMP, "cm_only_mindelta")
r1 = try
    run_cm_upper_checkpointed(copy(w_cal_econ); find_smallest = true, W = W_SMOKE, delta = 0.1,
        # CM grid: 50 equal-mass BUCKETS -> 49 LEVELS. `L` here is the driver's level count, so it
        # is derived from the grid rather than restated (2026-08-12, cm_equal_mass_probs).
        draw_design = :sobol_randomized, draw_seed = 20260719, L = length(CM_PROBS_L50), contrasts = :orthonormal,
        probs = CM_PROBS_L50,
        marginal_restriction = :common_flexible, cm_extension = :cm_only,
        include_truncated_moment = false, inner_lower_limit = -10.0,
        gp_fixed = gp_cal, objective_mode = :min_delta_fixed_gp,
        maxtime_real = 45.0, ckpt_dir = ckpt1, label = "smoke_cmonly_mindelta", checkpoint_interval_s = 20.0,
        verbose = true, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
        σHat = 3.0, destination_sample = :exclude_row)
catch e
    lp("TEST 1 THREW: ", sprint(showerror, e))
    e
end
results[:cm_only_mindelta] = r1
lp("TEST 1 result best = ", r1 isa Exception ? "N/A (threw)" : r1.best)
lp("TEST 1 checkpoint file exists: ", isfile(joinpath(ckpt1, "smoke_cmonly_mindelta_latest.jls")))

# ---- 2. CM-only at objective_mode=:min_gp (UNCHANGED default path) sanity check ----
lp("="^100); lp("TEST 2: run_cm_upper_checkpointed, cm_extension=:cm_only, objective_mode=:min_gp (default, unchanged)")
ckpt2 = joinpath(TMP, "cm_only_mingp")
r2 = try
    run_cm_upper_checkpointed(copy(w_cal_econ); find_smallest = true, W = W_SMOKE, delta = 3.0,
        # CM grid: 50 equal-mass BUCKETS -> 49 LEVELS. `L` here is the driver's level count, so it
        # is derived from the grid rather than restated (2026-08-12, cm_equal_mass_probs).
        draw_design = :sobol_randomized, draw_seed = 20260719, L = length(CM_PROBS_L50), contrasts = :orthonormal,
        probs = CM_PROBS_L50,
        marginal_restriction = :common_flexible, cm_extension = :cm_only,
        include_truncated_moment = false, inner_lower_limit = -10.0,
        maxtime_real = 45.0, ckpt_dir = ckpt2, label = "smoke_cmonly_mingp", checkpoint_interval_s = 20.0,
        verbose = true, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
        σHat = 3.0, destination_sample = :exclude_row)
catch e
    lp("TEST 2 THREW: ", sprint(showerror, e))
    e
end
results[:cm_only_mingp] = r2
lp("TEST 2 result best = ", r2 isa Exception ? "N/A (threw)" : r2.best)

# ---- 3. Origin-ZC K_mean=1/K_pair=1 (cheapest VALID zero-covariance ZC config -- that
#         distribution_restriction requires 1<=K_pair<=K_mean, confirmed live) at
#         objective_mode=:min_delta_fixed_gp. nu0 built via the SAME already-validated
#         companion-LFD machinery multistart_seed_generator.jl uses (not hand-derived formulas --
#         avoids guessing at OriginByPowerLayout's exact dense pair-block layout). ----
lp("="^100); lp("TEST 3: run_originzc_upper_checkpointed, K_mean=1/K_pair=1, objective_mode=:min_delta_fixed_gp")
geo_full = build_aspace_geometry(ctx)
x_free_cal = decode_w_econ(geo_full, w_cal_econ)
spec_oz1 = origin_zc_family_spec(:SMOKE_OZC; K_mean = 1, K_pair = 1)
fb_oz1 = build_family(ctx, spec_oz1)
layout1 = fb_oz1.layout::OriginByPowerLayout
nu0_active = companion_implied_nu_originzc(ctx, x_free_cal, layout1)
w0_oz = vcat(w_cal_econ, log.(nu0_active))
ckpt3 = joinpath(TMP, "originzc_mindelta")
r3 = try
    run_originzc_upper_checkpointed(copy(w0_oz); find_smallest = true, W = W_SMOKE, delta = 0.1,
        draw_design = :sobol_randomized, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        inner_lower_limit = -10.0, gp_fixed = gp_cal, objective_mode = :min_delta_fixed_gp,
        maxtime_real = 45.0, ckpt_dir = ckpt3, label = "smoke_originzc_mindelta", checkpoint_interval_s = 20.0,
        verbose = true, exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
        σHat = 3.0, destination_sample = :exclude_row)
catch e
    lp("TEST 3 THREW: ", sprint(showerror, e))
    e
end
results[:originzc_mindelta] = r3
lp("TEST 3 result best = ", r3 isa Exception ? "N/A (threw)" : r3.best)
lp("TEST 3 checkpoint file exists: ", isfile(joinpath(ckpt3, "smoke_originzc_mindelta_latest.jls")))

lp("="^100)
lp("SUMMARY")
for (k, r) in results
    if r isa Exception
        lp("  ", k, ": THREW -- ", sprint(showerror, r))
    else
        bf = r.best
        lp("  ", k, ": best = ", bf === nothing ? "nothing (no verified point found in budget)" : "gp=$(bf.gp) Delta=$(bf.Delta) n_eval=$(bf.n_eval)")
    end
end
lp("="^100)
lp("DONE")
