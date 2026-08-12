# ================================================================================================
# CAMPAIGN SMOKE for the CM + pairwise-quantile family (family #7, 2026-08-12): the checkpointed
# outer driver, end to end, at real D=20.
#
# Per memory `feedback-run-campaign-runner-end-to-end-not-just-driver`, a driver smoke that only
# calls the inner layers misses exactly the failures that matter for a campaign -- an
# `UndefKeywordError` on an arm nobody exercised, a checkpoint that cannot be reloaded, a resume
# guard that does not fire. So this runs the REAL `run_cm_pairwise_quantile_upper_checkpointed`
# against real D=20 data, with a deliberately tiny wall budget, and then:
#
#   1. reloads the checkpoint it wrote and checks the restriction's identity round-trips;
#   2. RESUMES from it, which is the path that reconstructs w0 from canonical z-space;
#   3. confirms every resume MISMATCH guard fires (L, G, family count, contrasts, sigma) -- a guard
#      nobody has watched fail is not known to be a guard;
#   4. confirms a non-exact inner option file is a HARD ERROR rather than a silent quasi-Newton
#      downgrade, which at production n is the difference between nStatus=0 in twelve evaluations
#      and -400 after 16,734.
#
# The wall budget is small on purpose: this gates WIRING, not convergence. Convergence of the inner
# solve is gated by test_cm_pairwise_quantile_d20_hessian_solve.jl, and the outer gradient by
# test_cm_pairwise_quantile_outer_gradient_fd.jl.
#
# Usage: julia --project=. -t N full_aod_diag/d4_exact/test_cm_pairwise_quantile_campaign_smoke.jl \
#             <W> <L> <G> <n_families> <contrasts> <sigmaHat> <maxtime_real_s>
# ================================================================================================

const D4X = @__DIR__
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "incumbent_logic.jl", "cm_checkpoint.jl", "cm_originzc_target_layout.jl",
          "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl",
          "country_resolve.jl", "cross_delta_cache.jl", "compressed_moments.jl",
          "canonical_price_precompute_workspace.jl", "hard_score_b_cache.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "lfix_buffer_reuse.jl",
          "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl", "bandwidth_cache_policy.jl",
          "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl", "dual_bank_ab_harness.jl",
          "reusable_context.jl", "organic_failure_capture.jl", "knitro_status.jl",
          "knitro_version_check.jl", "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "multistart_seed_generator.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl", "cm_pairwise_quantile_cplus.jl",
          "cm_pairwise_quantile_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    flush(stdout)
    return ok
end
"try/catch in a HARD scope -- at top level `threw = true` inside a catch is soft-scoped and Julia
1.12 silently makes it a new local when a global of the same name exists (observed live in this
family's own outer-production gate)."
function threw_matching(f, fragment::AbstractString)
    try
        f(); return (false, "")
    catch e
        msg = sprint(showerror, e)
        return (occursin(fragment, msg), first(msg, 160))
    end
end

const USAGE = "usage: julia ... test_cm_pairwise_quantile_campaign_smoke.jl <W> <L> <G> <n_families> <contrasts> <sigmaHat> <maxtime_real_s>"
length(ARGS) == 7 || error(USAGE)
const W_ARG = parse(Int, ARGS[1]); const L_ARG = parse(Int, ARGS[2]); const G_ARG = parse(Int, ARGS[3])
const NFAM_ARG = parse(Int, ARGS[4]); const CONTR_ARG = Symbol(ARGS[5])
const SIGMA_ARG = parse(Float64, ARGS[6]); const MAXT_ARG = parse(Float64, ARGS[7])

@printf("=== CM + pairwise-quantile CAMPAIGN SMOKE ===\nW=%d L=%d G=%d families=%d contrasts=%s sigma=%.4g maxtime=%.0fs threads=%d\n",
        W_ARG, L_ARG, G_ARG, NFAM_ARG, String(CONTR_ARG), SIGMA_ARG, MAXT_ARG, Threads.nthreads())
flush(stdout)

const CKPT_DIR = mktempdir(; prefix = "cmpq_smoke_")
const GRAV = default_gravity_exclude_cells_brazil_korea()
const SCI = (W = W_ARG, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
             σHat = SIGMA_ARG, inner_lower_limit = -10.0, z_halfwidth = 2.0,
             destination_sample = :exclude_row, exclude_diagonal_gravity = true,
             gravity_exclude_cells = GRAV)
const FAMKW = (L = L_ARG, cm_grid_size = G_ARG, cm_moment_families = NFAM_ARG,
               contrasts = CONTR_ARG, min_bin_count = 1, mass_start = :uniform,
               inner_opt = "ek_inner_cmpq.opt")

# ---- w0 at the calibration point, economic block via the SAME helper the campaign chain uses ----
t0 = time()
ctx_raw = d20_real_setup_design(; W = W_ARG, δ = SCI.delta, find_smallest = true,
    draw_design = SCI.draw_design, draw_seed = SCI.draw_seed,
    destination_sample = SCI.destination_sample, exclude_diagonal_gravity = SCI.exclude_diagonal_gravity,
    gravity_exclude_cells = GRAV, σHat = SCI.σHat, inner_lower_limit = SCI.inner_lower_limit)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
geo = build_aspace_geometry(ctx)
w_econ = cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)
W0 = vcat(w_econ, cmpq_uniform_mass_raw(L_ARG))
@printf("context+w0 in %.1fs: econ=%d + mass=%d = %d\n", time() - t0, length(w_econ),
        n_cmpq_raw(L_ARG), length(W0)); flush(stdout)
check("w0 has the collapsed mass block (L-1), not the standalone family's D*(L-1)",
      length(W0) == length(w_econ) + (L_ARG - 1),
      "$(L_ARG - 1) mass coords vs $(ctx.D * (L_ARG - 1)) for standalone PQ")

# ------------------------------------------------------------------------------------------------
println("\n=== 1. the driver runs and writes a checkpoint ===")
# ------------------------------------------------------------------------------------------------
t0 = time()
r1 = run_cm_pairwise_quantile_upper_checkpointed(W0; find_smallest = true, SCI..., FAMKW...,
    ckpt_dir = CKPT_DIR, label = "cmpq_smoke", maxtime_real = MAXT_ARG,
    checkpoint_interval_s = 10.0, run_id = "smoke", verbose = true)
@printf("  knitro_status=%d  n_eval=%d  n_grad=%d  wall=%.1fs  best=%s\n",
        r1.knitro_status, r1.n_eval, r1.n_grad, time() - t0,
        r1.best === nothing ? "nothing" : @sprintf("gp=%.8g Delta=%.8g", r1.best.gp, r1.best.Delta))
flush(stdout)
check("the outer driver ran at least one evaluation", r1.n_eval >= 1, "n_eval=$(r1.n_eval)")
check("the outer driver ran at least one gradient", r1.n_grad >= 1, "n_grad=$(r1.n_grad)")
check("a checkpoint file exists", isfile(r1.ckpt_path), r1.ckpt_path)
check("the run reports this family's own restriction identity",
      r1.L == L_ARG && r1.G == G_ARG && r1.n_families == NFAM_ARG && r1.n_raw == L_ARG - 1)

# ------------------------------------------------------------------------------------------------
println("\n=== 2. the checkpoint round-trips ===")
# ------------------------------------------------------------------------------------------------
ck = load_cm_pairwise_quantile_checkpoint(r1.ckpt_path)
check("checkpoint is a CMPairwiseQuantileCheckpointV1", ck isa CMPairwiseQuantileCheckpointV1)
check("checkpoint carries the restriction identity",
      ck.L == L_ARG && ck.G == G_ARG && ck.n_families == NFAM_ARG && ck.contrasts == CONTR_ARG,
      "L=$(ck.L) G=$(ck.G) fam=$(ck.n_families) contrasts=:$(ck.contrasts)")
check("checkpoint records the DERIVED numbers, not just the request",
      length(ck.cm_thresholds) == G_ARG - 1 && size(ck.pq_cutoffs) == (L_ARG - 1, ctx.D),
      "$(length(ck.cm_thresholds)) CM thresholds, PQ cutoffs $(size(ck.pq_cutoffs))")
check("checkpoint mass block has the collapsed width", length(ck.raw_masses) == L_ARG - 1)
check("checkpoint records draw provenance", ck.W == W_ARG && ck.draw_seed == SCI.draw_seed &&
      !isempty(ck.draw_checksum_uniform))

# ------------------------------------------------------------------------------------------------
println("\n=== 3. RESUME from the checkpoint ===")
# ------------------------------------------------------------------------------------------------
t0 = time()
r2 = run_cm_pairwise_quantile_upper_checkpointed(nothing; find_smallest = true, SCI..., FAMKW...,
    ckpt_dir = CKPT_DIR, label = "cmpq_smoke_resume", maxtime_real = MAXT_ARG,
    checkpoint_interval_s = 10.0, run_id = "smoke_resume", resume_from = r1.ckpt_path, verbose = false)
@printf("  resumed: knitro_status=%d  n_eval=%d (started from %d)  wall=%.1fs\n",
        r2.knitro_status, r2.n_eval, ck.n_eval, time() - t0); flush(stdout)
check("resume ran without w0 (reconstructed from canonical z-space)", r2.n_eval > ck.n_eval,
      "n_eval $(ck.n_eval) -> $(r2.n_eval)")

# ------------------------------------------------------------------------------------------------
println("\n=== 4. every resume MISMATCH guard fires ===")
# ------------------------------------------------------------------------------------------------
# A guard nobody has watched fail is not known to be a guard. Each of these changes exactly ONE
# field of the restriction's identity and must be refused, not silently accepted.
for (nm, kwmod, frag) in (
        ("L", (L = (L_ARG == 2 ? 5 : 2),), "L MISMATCH"),
        ("cm_grid_size", (cm_grid_size = 2 * G_ARG,), "cm_grid_size MISMATCH"),
        ("cm_moment_families", (cm_moment_families = (NFAM_ARG == 1 ? 2 : 1),), "cm_moment_families MISMATCH"),
        ("contrasts", (contrasts = (CONTR_ARG === :anchored ? :orthonormal : :anchored),), "contrasts MISMATCH"),
        ("sigma", (), "sigma MISMATCH"))
    famkw = merge(FAMKW, kwmod)
    sci = nm == "sigma" ? merge(SCI, (σHat = SCI.σHat + 0.5,)) : SCI
    ok, msg = threw_matching(frag) do
        run_cm_pairwise_quantile_upper_checkpointed(nothing; find_smallest = true, sci..., famkw...,
            ckpt_dir = CKPT_DIR, label = "cmpq_smoke_guard", maxtime_real = 5.0,
            run_id = "guard", resume_from = r1.ckpt_path, verbose = false)
    end
    check("resume guard fires on a changed $nm", ok, msg)
end
# ...and the direction guard, which is the one that would silently produce an upper bound from a
# lower-bound checkpoint.
ok_dir, msg_dir = threw_matching("direction MISMATCH") do
    run_cm_pairwise_quantile_lower_checkpointed(nothing; SCI..., FAMKW..., ckpt_dir = CKPT_DIR,
        label = "cmpq_smoke_dir", maxtime_real = 5.0, run_id = "dir", resume_from = r1.ckpt_path,
        verbose = false)
end
check("resume guard fires on a changed direction (upper checkpoint, lower run)", ok_dir, msg_dir)

# ------------------------------------------------------------------------------------------------
println("\n=== 5. argument-validation and silent-downgrade guards ===")
# ------------------------------------------------------------------------------------------------
ok_div, msg_div = threw_matching("does not divide") do
    run_cm_pairwise_quantile_upper_checkpointed(W0; find_smallest = true, SCI...,
        merge(FAMKW, (L = 3,))..., ckpt_dir = CKPT_DIR, label = "cmpq_smoke_div",
        maxtime_real = 5.0, run_id = "div", verbose = false)
end
check("an L that does not divide G is rejected before any data is touched", ok_div, msg_div)

ok_fg, msg_fg = threw_matching("hessopt=exact") do
    run_cm_pairwise_quantile_upper_checkpointed(W0; find_smallest = true, SCI...,
        merge(FAMKW, (inner_opt = "ek_inner_cmpq_fgonly.opt",))..., ckpt_dir = CKPT_DIR,
        label = "cmpq_smoke_fgonly", maxtime_real = 5.0, run_id = "fgonly", verbose = false)
end
check("an FG-only option file is a HARD ERROR, not a silent quasi-Newton downgrade", ok_fg, msg_fg)

# Every scientific kwarg omitted must be an UndefKeywordError, not a silent substitution. Spot-check
# one from each group -- the shared economic set and this family's own.
for nm in (:σHat, :W, :destination_sample)
    sci_missing = NamedTuple(k => v for (k, v) in pairs(SCI) if k != nm)
    ok, msg = threw_matching("UndefKeywordError") do
        run_cm_pairwise_quantile_upper_checkpointed(W0; find_smallest = true, sci_missing..., FAMKW...,
            ckpt_dir = CKPT_DIR, label = "cmpq_smoke_undef", maxtime_real = 5.0, run_id = "undef")
    end
    check("omitting the scientific kwarg $nm raises UndefKeywordError", ok, msg)
end
for nm in (:L, :cm_grid_size, :cm_moment_families, :contrasts, :mass_start, :inner_opt)
    fam_missing = NamedTuple(k => v for (k, v) in pairs(FAMKW) if k != nm)
    ok, msg = threw_matching("UndefKeywordError") do
        run_cm_pairwise_quantile_upper_checkpointed(W0; find_smallest = true, SCI..., fam_missing...,
            ckpt_dir = CKPT_DIR, label = "cmpq_smoke_undef", maxtime_real = 5.0, run_id = "undef")
    end
    check("omitting this family's kwarg $nm raises UndefKeywordError", ok, msg)
end

rm(CKPT_DIR; recursive = true, force = true)
println("\n", "="^92)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^92)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_campaign_smoke: $(NFAIL[]) check(s) failed")
