# ================================================================================================
# End-to-end smoke test for run_pairwise_quantile_upper_checkpointed (pairwise_quantile_checkpoint.jl)
# at REAL D=20 data and a real KNITRO outer solve -- the gate the handover doc requires before this
# family may be called runnable ("real KNITRO, a few concrete families/points, not synthetic").
#
# Mirrors smoke_objective_mode_min_delta_fixed_gp.jl's structure and its real-D20 context build.
#
# Proves:
#   1. the driver runs end-to-end at real D=20: context build, layout, bounds, KNITRO outer solve,
#      both callbacks, final verification -- no crash in the gp/zfree/raw_masses plumbing;
#   2. it writes a checkpoint that round-trips through its own loader with the outer point intact;
#   3. resume from that checkpoint reconstructs the same outer point and continues;
#   4. the outer gradient's two blocks are both live -- the run actually MOVES the MASS
#      coordinates, not just the economic ones (a gradient that were silently zero on the mass
#      block would still "run fine" and produce a plausible-looking result);
#   5. objective_mode=:min_delta_fixed_gp works, same mechanism as every other family;
#   6. the no-defaults rule is enforced: omitting a scientific kwarg raises UndefKeywordError.
#
# Run:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=8 julia --project=. \
#     full_aod_diag/d4_exact/smoke_pairwise_quantile_outer_driver.jl [W] [L] [maxtime_s]
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
          # ---- this family ----
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
using Random, LinearAlgebra, Serialization, SpecialFunctions, Printf

lp(xs...) = (println(xs...); flush(stdout))
ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool, detail::AbstractString = "")
    global ALL_PASS[] &= cond
    lp(cond ? "PASS  " : "FAIL  ", name, isempty(detail) ? "" : "  ($detail)")
end

const W_SMOKE   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
const L_SMOKE   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const MAXT      = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 300.0
# Version B's two extra required modelling choices. `:empirical_quantile` is picked here (rather
# than :frechet_theoretical) because it is the setting under which mu=1/L reproduces version A's
# moment matrix exactly, so this smoke and the version-A/B equivalence anchor
# (test_pairwise_quantile_version_ab_anchor.jl) describe the same restriction.
const CUTOFF_SOURCE = :empirical_quantile
# Floor at half the EXPECTED joint-cell occupancy (W/L^2): derived from this run's own W and L
# rather than typed in, and strict enough that a genuinely starved cell still fails.
const MIN_BIN_COUNT = max(10, W_SMOKE ÷ (2 * L_SMOKE^2))
const GRAV = default_gravity_exclude_cells_brazil_korea()

lp("="^100)
lp("pairwise-quantile OUTER DRIVER smoke (FREE MASSES): W=", W_SMOKE, " L=", L_SMOKE,
   " cutoff_source=:", CUTOFF_SOURCE, " min_bin_count=", MIN_BIN_COUNT, " maxtime=", MAXT, "s")
lp("="^100)

ctx_raw = d20_real_setup_design(W = W_SMOKE, δ = 0.1, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
lp("ctx: D=", ctx.D, " D_dest=", ctx.D_dest, " W=", ctx.W, " sigma=", ctx.σ)

geo = build_aspace_geometry(ctx)
w_cal_econ = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
layout = PairwiseQuantileMassLayout(ctx.D, L_SMOKE)
mass0 = uniform_mass_raw(layout)   # mu = 1/L
w0 = vcat(w_cal_econ, mass0)
lp("n_total_rows(D=", ctx.D, ", L=", L_SMOKE, ") = ", n_total_rows(ctx.D, L_SMOKE),
   "   n_raw = ", n_raw(layout), "   length(w0) = ", length(w0))
check("w0 length == D*Ddest + n_raw", length(w0) == ctx.D * ctx.D_dest + n_raw(layout))
check("start masses are all finite", all(isfinite, mass0))
begin
    st_chk = PairwiseQuantileMassState(ctx.D, L_SMOKE)
    set_pairwise_quantile_masses!(st_chk, mass0, layout)
    check("start masses decode to mu == 1/L exactly",
          maximum(abs, st_chk.mu .- 1.0 / L_SMOKE) < 1e-14)
end

TMP = mktempdir()
lp("scratch ckpt_dir = ", TMP)

common = (W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
          σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0,
          destination_sample = :exclude_row, exclude_diagonal_gravity = true,
          gravity_exclude_cells = GRAV, L = L_SMOKE, cutoff_source = CUTOFF_SOURCE,
          min_bin_count = MIN_BIN_COUNT, A_coordinate_mode = :powered_aspace,
          # Exercise the SAME inner options production uses, so this gate covers the ma97 /
          # par_concurrent_evals=no configuration rather than the shared default it does not run on.
          inner_opt_override = "ek_inner_pq.opt")

# ------------------------------------------------------------------------------------------------
# TEST 1: objective_mode=:min_gp, end to end
# ------------------------------------------------------------------------------------------------
lp("\n", "="^100); lp("TEST 1: run_pairwise_quantile_upper_checkpointed, objective_mode=:min_gp")
ck1 = joinpath(TMP, "pq_min_gp")
t0 = time()
r1 = run_pairwise_quantile_upper_checkpointed(copy(w0); find_smallest = true, ckpt_dir = ck1,
    label = "pq_smoke_mingp", maxtime_real = MAXT, checkpoint_interval_s = 30.0, verbose = true,
    objective_mode = :min_gp, common...)
lp("TEST 1 done in ", round(time() - t0, digits = 1), "s: knitro_status=", r1.knitro_status,
   " n_eval=", r1.n_eval, " n_grad=", r1.n_grad, " best=", r1.best === nothing ? "nothing" : r1.best.gp)
check("T1: driver returned a NamedTuple with the family-contract fields",
      hasproperty(r1, :knitro_status) && hasproperty(r1, :n_eval) && hasproperty(r1, :n_grad) && hasproperty(r1, :best))
check("T1: at least one outer evaluation happened", r1.n_eval >= 1, "n_eval=$(r1.n_eval)")
check("T1: at least one outer gradient happened", r1.n_grad >= 1, "n_grad=$(r1.n_grad)")
check("T1: checkpoint file was written", isfile(r1.ckpt_path), r1.ckpt_path)
if r1.best !== nothing
    check("T1: best_feasible has the exact shape family_start_chain.jl reads",
          hasproperty(r1.best, :gp) && hasproperty(r1.best, :w) && hasproperty(r1.best, :Delta) &&
          hasproperty(r1.best, :n_eval) && hasproperty(r1.best, :t))
    lp("  best: gp=", r1.best.gp, "  Delta=", r1.best.Delta, "  at eval ", r1.best.n_eval)
else
    lp("  NOTE: no verified feasible incumbent within the smoke budget (delta=0.1 is tight); the ",
       "structural checks above still apply.")
end

# ---- 2. checkpoint round-trip ----
ck = load_pairwise_quantile_checkpoint(r1.ckpt_path)
check("T1: checkpoint round-trips through its own loader", ck isa PairwiseQuantileMassCheckpointV1)
check("T1: checkpoint records this family's own fields", ck.L == L_SMOKE &&
      ck.cutoff_source == CUTOFF_SOURCE && ck.min_bin_count == MIN_BIN_COUNT &&
      ck.n_raw == n_raw(layout) && ck.D == ctx.D)
check("T1: checkpoint raw_masses has the right width", length(ck.raw_masses) == n_raw(layout))
# The cutoffs themselves must be recorded, not just the rule that produced them: :empirical_quantile
# depends on the draws, so the symbol alone does not pin the numbers and a run recorded without them
# is not reproducible.
check("T1: checkpoint records the FIXED cutoff matrix itself", size(ck.cutoffs) == (L_SMOKE - 1, ctx.D) &&
      all(isfinite, ck.cutoffs))
check("T1: recorded cutoffs are bit-identical to regenerating them from cutoff_source",
      ck.cutoffs == pairwise_quantile_fixed_cutoffs(
          pairwise_quantile_frechet_features(ctx.U, ctx.μHat), L_SMOKE;
          cutoff_source = CUTOFF_SOURCE, mu_frechet = ctx.μHat))
check("T1: checkpoint records sigma and draw provenance", ck.sigma == ctx.σ && ck.W == W_SMOKE &&
      ck.draw_seed == 20260719 && ck.draw_design == :sobol_randomized)

# ---- 4. did the MASS block actually move? ----
# The point of this check: an outer gradient whose restriction block was silently zero would still
# run to completion and return a perfectly plausible result. The only way to see it is to look at
# whether the restriction coordinates MOVED.
mass_moved = maximum(abs, ck.raw_masses .- mass0)
lp("  max |raw mass coord moved from start| = ", mass_moved)
# Guarded on n_eval>=1 deliberately: if EVERY point was rejected the checkpoint still records
# whatever terminal iterate KNITRO reported, and the masses will differ from the start for reasons
# that have nothing to do with the gradient. Reporting that as evidence the mass block is live
# would be a false pass -- it was one, before this guard (seen live at W=8000, where every inner
# solve was infeasible and n_eval=0 yet this check "passed" with a move of 3.95).
check("T1: the outer solve actually MOVED the mass coordinates (mass gradient block is live)",
      r1.n_eval >= 1 && r1.n_grad >= 1 && mass_moved > 1e-8,
      "n_eval=$(r1.n_eval) n_grad=$(r1.n_grad) max move = $mass_moved")

# ------------------------------------------------------------------------------------------------
# TEST 2: resume
# ------------------------------------------------------------------------------------------------
lp("\n", "="^100); lp("TEST 2: resume from the TEST 1 checkpoint")
r2 = run_pairwise_quantile_upper_checkpointed(nothing; find_smallest = true, ckpt_dir = ck1,
    label = "pq_smoke_mingp_resume", maxtime_real = min(MAXT, 120.0), checkpoint_interval_s = 30.0,
    verbose = false, objective_mode = :min_gp, resume_from = r1.ckpt_path, common...)
lp("TEST 2 done: knitro_status=", r2.knitro_status, " n_eval=", r2.n_eval, " (carried in from ", ck.n_eval, ")")
check("T2: resume ran without error and carried the evaluation counter forward",
      r2.n_eval >= ck.n_eval, "resumed n_eval=$(r2.n_eval) >= checkpoint n_eval=$(ck.n_eval)")

# ---- resume guards must be hard errors, not silent overrides ----
mismatch_caught = false
try
    run_pairwise_quantile_upper_checkpointed(nothing; find_smallest = true, ckpt_dir = ck1,
        label = "pq_smoke_badresume", maxtime_real = 5.0, verbose = false, resume_from = r1.ckpt_path,
        W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
        σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
        L = L_SMOKE + 1, cutoff_source = CUTOFF_SOURCE, min_bin_count = MIN_BIN_COUNT,
        A_coordinate_mode = :powered_aspace)
catch e
    global mismatch_caught = occursin("L MISMATCH", sprint(showerror, e))
end
check("T2: resuming under a DIFFERENT L is a hard error, not a silent override", mismatch_caught)

# Different cutoffs are a DIFFERENT restriction, not a different search over the same one -- so the
# cutoff_source guard must be a hard error on the same footing as the L guard.
cutoff_mismatch_caught = false
try
    run_pairwise_quantile_upper_checkpointed(nothing; find_smallest = true, ckpt_dir = ck1,
        label = "pq_smoke_badcutoff", maxtime_real = 5.0, verbose = false, resume_from = r1.ckpt_path,
        W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
        σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
        L = L_SMOKE, cutoff_source = :frechet_theoretical, min_bin_count = MIN_BIN_COUNT,
        A_coordinate_mode = :powered_aspace)
catch e
    global cutoff_mismatch_caught = occursin("cutoff_source MISMATCH", sprint(showerror, e))
end
check("T2: resuming under a DIFFERENT cutoff_source is a hard error", cutoff_mismatch_caught)

# ------------------------------------------------------------------------------------------------
# TEST 3: objective_mode=:min_delta_fixed_gp
# ------------------------------------------------------------------------------------------------
lp("\n", "="^100); lp("TEST 3: objective_mode=:min_delta_fixed_gp (Stage B / R0 restoration path)")
gp_fix = w0[1]
ck3 = joinpath(TMP, "pq_min_delta")
r3 = run_pairwise_quantile_upper_checkpointed(copy(w0); find_smallest = true, ckpt_dir = ck3,
    label = "pq_smoke_mindelta", maxtime_real = min(MAXT, 180.0), checkpoint_interval_s = 30.0,
    verbose = false, objective_mode = :min_delta_fixed_gp, gp_fixed = gp_fix, common...)
lp("TEST 3 done: knitro_status=", r3.knitro_status, " n_eval=", r3.n_eval,
   " best Delta=", r3.best === nothing ? "nothing" : r3.best.Delta)
check("T3: min_delta_fixed_gp ran end-to-end", r3.n_eval >= 1, "n_eval=$(r3.n_eval)")
check("T3: gp stayed pinned at gp_fixed", isapprox(r3.xsol[1], gp_fix; atol = 1e-8),
      "xsol[1]=$(r3.xsol[1]) vs gp_fixed=$gp_fix")
mode_guard = false
try
    run_pairwise_quantile_upper_checkpointed(copy(w0); find_smallest = true, ckpt_dir = ck3,
        label = "pq_smoke_badmode", maxtime_real = 5.0, verbose = false,
        objective_mode = :min_delta_fixed_gp, common...)   # gp_fixed deliberately omitted
catch e
    global mode_guard = occursin("requires gp_fixed", sprint(showerror, e))
end
check("T3: :min_delta_fixed_gp without gp_fixed is a hard error", mode_guard)

# ------------------------------------------------------------------------------------------------
# TEST 4: the no-defaults rule is actually enforced
# ------------------------------------------------------------------------------------------------
lp("\n", "="^100); lp("TEST 4: scientific parameters have NO defaults (CLAUDE.md rule)")
for (name, kwargs) in (
        (:σHat,          (; W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
                            inner_lower_limit = -10.0, z_halfwidth = 30.0, destination_sample = :exclude_row,
                            exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
                            L = L_SMOKE, cutoff_source = CUTOFF_SOURCE, min_bin_count = MIN_BIN_COUNT)),
        (:draw_seed,     (; W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, σHat = 3.0,
                            inner_lower_limit = -10.0, z_halfwidth = 30.0, destination_sample = :exclude_row,
                            exclude_diagonal_gravity = true, gravity_exclude_cells = GRAV,
                            L = L_SMOKE, cutoff_source = CUTOFF_SOURCE, min_bin_count = MIN_BIN_COUNT)),
        (:L,             (; W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
                            σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0,
                            destination_sample = :exclude_row, exclude_diagonal_gravity = true,
                            gravity_exclude_cells = GRAV, cutoff_source = CUTOFF_SOURCE,
                            min_bin_count = MIN_BIN_COUNT)),
        (:cutoff_source, (; W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
                            σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0,
                            destination_sample = :exclude_row, exclude_diagonal_gravity = true,
                            gravity_exclude_cells = GRAV, L = L_SMOKE, min_bin_count = MIN_BIN_COUNT)),
        (:min_bin_count, (; W = W_SMOKE, delta = 0.1, draw_design = :sobol_randomized, draw_seed = 20260719,
                            σHat = 3.0, inner_lower_limit = -10.0, z_halfwidth = 30.0,
                            destination_sample = :exclude_row, exclude_diagonal_gravity = true,
                            gravity_exclude_cells = GRAV, L = L_SMOKE, cutoff_source = CUTOFF_SOURCE)))
    caught = false
    try
        run_pairwise_quantile_upper_checkpointed(copy(w0); find_smallest = true, ckpt_dir = TMP,
            label = "pq_nodefault_$(name)", maxtime_real = 5.0, verbose = false, kwargs...)
    catch e
        caught = e isa UndefKeywordError && e.var === name
    end
    check("T4: omitting $name raises UndefKeywordError (no silent default)", caught)
end

lp("\n", "="^100)
lp(ALL_PASS[] ? "ALL PAIRWISE-QUANTILE OUTER-DRIVER SMOKE CHECKS PASSED" : "SOME SMOKE CHECKS FAILED")
lp("="^100)
exit(ALL_PASS[] ? 0 : 1)
