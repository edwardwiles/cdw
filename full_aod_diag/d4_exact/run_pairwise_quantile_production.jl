# ================================================================================================
# PRODUCTION RUNNER for the pairwise-quantile-independence family (version B: fixed Fréchet-z
# cutoffs + free bin masses), 2026-08-10.
#
# Runs this family through the SAME per-delta stage structure, scientific settings, algorithms and
# time budgets the frozen five-family protocol uses, so its results are directly comparable to the
# other five families -- WITHOUT touching `protocols/paper_upper_v1.toml`, which is frozen by its own
# header and had a live campaign running while this was written. Every number below is READ from
# that TOML's `[scientific]` / `[algorithms]` / `[budgets]` blocks and restated here as an explicit
# constant with its provenance; nothing is invented and nothing is defaulted.
#
#   [scientific]  sigma=3.0, W=100000, draw_design=:sobol_randomized, draw_seed=20260719,
#                 destination_sample=:exclude_row, exclude_diagonal_gravity=true,
#                 gravity_exclude_cells=default_gravity_exclude_cells_brazil_korea,
#                 inner_lower_limit=-10.0, A_coordinate_mode=:powered_aspace, find_smallest=true,
#                 deltas=[0.1,0.5,1.0,2.0], z_halfwidth=30.0
#   [algorithms]  primary = pin_cg_lbfgs (pin_outer_algorithm=true)
#                 alternate = direct_sr1 (outer_direct_hessopt=:sr1)
#   [budgets.discovery_feasible_start]  stage A 75 min, stage B 30 min, stage C 75 min
#
# STAGE STRUCTURE per delta cell (protocol section 8, feasible-start path):
#   A  primary push        objective_mode=:min_gp,               pin_outer_algorithm
#   B  fixed-gp restoration objective_mode=:min_delta_fixed_gp,  gp pinned at A's incumbent
#   C  alternate push      objective_mode=:min_gp,               outer_direct_hessopt=:sr1
# Each stage resumes from the previous stage's checkpoint, so the whole cell is interruptible and
# restartable at any point; re-running the script skips cells whose `stage_C` checkpoint exists.
#
# THIS FAMILY'S OWN THREE REQUIRED CHOICES (no defaults anywhere -- CLAUDE.md):
#   L               number of Fréchet-z quantile bins per origin        (ARGS[1])
#   cutoff_source   :frechet_theoretical | :empirical_quantile          (ARGS[2])
#   min_bin_count   non-degeneracy floor per marginal bin AND joint cell (derived: W / (2*L^2),
#                   i.e. half the expected joint-cell occupancy -- data-derived from this run's own
#                   W and L, printed and recorded, not a typed-in level)
#
# Usage (always under `screen`, see launch_pairwise_quantile_production.sh):
#   julia --project=. full_aod_diag/d4_exact/run_pairwise_quantile_production.jl <L> <cutoff_source> <campaign_root> [deltas_csv]
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
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl",
          "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, Serialization, SpecialFunctions
import Dates

lp(xs...) = (println("[pqprod ", Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"), "] ", xs...); flush(stdout))

length(ARGS) >= 3 ||
    error("run_pairwise_quantile_production.jl: need <L> <cutoff_source> <campaign_root> [deltas_csv]")
const PQ_L           = parse(Int, ARGS[1])
const CUTOFF_SOURCE  = Symbol(ARGS[2])
const CAMPAIGN_ROOT  = abspath(ARGS[3])
const DELTAS         = length(ARGS) >= 4 ? parse.(Float64, split(ARGS[4], ",")) : [0.1, 0.5, 1.0, 2.0]

# ---- protocol [scientific] (paper_upper_v1.toml lines 55-71), restated explicitly -------------
const SIGMA               = 3.0
const W_PROD              = 100_000
const DRAW_DESIGN         = :sobol_randomized
const DRAW_SEED           = 20260719
const DESTINATION_SAMPLE  = :exclude_row
const EXCLUDE_DIAG_GRAV   = true
const INNER_LOWER_LIMIT   = -10.0
const A_COORD_MODE        = :powered_aspace
const FIND_SMALLEST       = true          # direction = "upper"
const Z_HALFWIDTH         = 30.0
# Inner-solve KNITRO options. NOT the shared ek_inner.opt: at L=6/W=100k, 96.6% of KN_solve wall is
# non-callback time in a dense O(n^3) KKT factorization, and `linsolver auto` was picking a SERIAL
# solver -- measured at 1.45 average cores under every BLAS thread knob. ek_inner_pq.opt sets
# linsolver=ma97 (parallel AND deterministic; ma86 is faster still but explicitly non-deterministic,
# so rejected) with 4 solver threads and par_concurrent_evals off, since we thread our own callbacks
# and must not let KNITRO evaluate concurrently on top of that. 1.52x, all solvers agreeing on
# Delta* to ~8e-15. See the file's own header for the full sweep.
const INNER_OPT           = "ek_inner_pq.opt"
# ---- protocol [budgets.discovery_feasible_start] (minutes) ------------------------------------
const STAGE_A_MIN = 75.0
const STAGE_B_MIN = 30.0
const STAGE_C_MIN = 75.0
# ---- this family's own required choices ------------------------------------------------------
# Half the EXPECTED joint-cell occupancy W/L^2 -- derived from this run's own W and L, so it moves
# correctly with both, and strict enough that a genuinely starved cell still hard-errors.
const MIN_BIN_COUNT = max(10, W_PROD ÷ (2 * PQ_L^2))
const MASS_START    = :uniform            # mu = 1/L; see pairwise_quantile_start_masses

CUTOFF_SOURCE in (:frechet_theoretical, :empirical_quantile) ||
    error("run_pairwise_quantile_production.jl: cutoff_source must be :frechet_theoretical|:empirical_quantile, got :$CUTOFF_SOURCE")
mkpath(CAMPAIGN_ROOT)

lp("="^96)
lp("PAIRWISE-QUANTILE (Frechet-z, free masses) PRODUCTION: L=", PQ_L, " cutoff_source=:", CUTOFF_SOURCE)
lp("W=", W_PROD, " sigma=", SIGMA, " deltas=", DELTAS, " min_bin_count=", MIN_BIN_COUNT,
   " mass_start=:", MASS_START)
lp("campaign root: ", CAMPAIGN_ROOT)
lp("budgets (min): A=", STAGE_A_MIN, " B=", STAGE_B_MIN, " C=", STAGE_C_MIN, "  (protocol section 8)")
lp("="^96)

const GRAV = default_gravity_exclude_cells_brazil_korea()

# One context build, reused to construct the family's starting outer point. The driver builds its
# own context per call (it must: it owns the augmented obj), so this one exists only for w0.
lp("building context for the starting outer point ...")
t0 = time()
ctx_raw = d20_real_setup_design(W = W_PROD, δ = first(DELTAS), find_smallest = FIND_SMALLEST,
    draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, destination_sample = DESTINATION_SAMPLE,
    exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV, gravity_exclude_cells = GRAV,
    σHat = SIGMA, inner_lower_limit = INNER_LOWER_LIMIT)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
geo = build_aspace_geometry(ctx)
w_cal_econ = cm_w0_from_calibration(ctx, geo.pe, A_COORD_MODE)
mass0 = uniform_mass_raw(layout)                     # MASS_START = :uniform
const W0 = vcat(w_cal_econ, mass0)
lp("context + w0 built in ", round(time() - t0, digits = 1), "s: D=", ctx.D, " D_dest=", ctx.D_dest,
   " muHat=", ctx.μHat, " n_raw=", n_raw(layout), " length(w0)=", length(W0),
   " n_total_rows=", n_total_rows(ctx.D, PQ_L))

common = (W = W_PROD, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, σHat = SIGMA,
          inner_lower_limit = INNER_LOWER_LIMIT, z_halfwidth = Z_HALFWIDTH,
          destination_sample = DESTINATION_SAMPLE, exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV,
          gravity_exclude_cells = GRAV, L = PQ_L, cutoff_source = CUTOFF_SOURCE,
          min_bin_count = MIN_BIN_COUNT, A_coordinate_mode = A_COORD_MODE,
          find_smallest = FIND_SMALLEST, inner_opt_override = INNER_OPT)

"Path of a stage's `<label>_latest.jls` checkpoint."
ckpt_of(dir, label) = joinpath(dir, "$(label)_latest.jls")

"""
Run one delta cell: stage A (primary push) -> stage B (fixed-gp restoration) -> stage C (alternate
push). Each stage resumes from the previous one's checkpoint; a stage whose checkpoint already
exists is SKIPPED, so re-running the script continues an interrupted campaign rather than
restarting it (protocol section 21's own discipline: never silently redo completed work).
"""
function run_delta_cell(delta::Float64)
    tag = replace(string(delta), "." => "p")
    dir = joinpath(CAMPAIGN_ROOT, "delta_$tag")
    mkpath(dir)
    lp("-"^90)
    lp("DELTA CELL delta=", delta, "  dir=", dir)

    results = Dict{Symbol,Any}()

    # ---- Stage A: primary push (pin_cg_lbfgs), objective_mode=:min_gp ----
    labA = "pq_L$(PQ_L)_d$(tag)_A"
    if isfile(ckpt_of(dir, labA))
        lp("  stage A: checkpoint exists, skipping (", ckpt_of(dir, labA), ")")
        results[:A] = load_pairwise_quantile_checkpoint(ckpt_of(dir, labA))
    else
        lp("  stage A: primary push (pin_cg_lbfgs), budget ", STAGE_A_MIN, " min")
        rA = run_pairwise_quantile_upper_checkpointed(copy(W0); ckpt_dir = dir, label = labA,
            delta = delta, maxtime_real = STAGE_A_MIN * 60, checkpoint_interval_s = 120.0,
            objective_mode = :min_gp, pin_outer_algorithm = true, verbose = true, common...)
        lp("  stage A done: status=", rA.knitro_status, " n_eval=", rA.n_eval, " n_grad=", rA.n_grad,
           " best gp=", rA.best === nothing ? "nothing" : rA.best.gp,
           " kappa=", rA.kappa)
        results[:A] = rA
    end

    # ---- Stage B: fixed-gp Delta restoration ----
    labB = "pq_L$(PQ_L)_d$(tag)_B"
    ckA = load_pairwise_quantile_checkpoint(ckpt_of(dir, labA))
    if ckA.best_feasible === nothing
        lp("  stage B: SKIPPED -- stage A produced no verified feasible incumbent to pin gp at.")
    elseif isfile(ckpt_of(dir, labB))
        lp("  stage B: checkpoint exists, skipping")
    else
        gp_fix = ckA.best_feasible.gp
        lp("  stage B: fixed-gp restoration at gp=", gp_fix, ", budget ", STAGE_B_MIN, " min")
        rB = run_pairwise_quantile_upper_checkpointed(copy(ckA.best_feasible.w); ckpt_dir = dir,
            label = labB, delta = delta, maxtime_real = STAGE_B_MIN * 60,
            checkpoint_interval_s = 120.0, objective_mode = :min_delta_fixed_gp, gp_fixed = gp_fix,
            verbose = false, common...)
        lp("  stage B done: status=", rB.knitro_status, " n_eval=", rB.n_eval,
           " best Delta=", rB.best === nothing ? "nothing" : rB.best.Delta)
        results[:B] = rB
    end

    # ---- Stage C: alternate push (direct_sr1) ----
    labC = "pq_L$(PQ_L)_d$(tag)_C"
    if isfile(ckpt_of(dir, labC))
        lp("  stage C: checkpoint exists, skipping")
    else
        resume_src = isfile(ckpt_of(dir, labB)) ? ckpt_of(dir, labB) : ckpt_of(dir, labA)
        lp("  stage C: alternate push (direct_sr1) resuming from ", basename(resume_src),
           ", budget ", STAGE_C_MIN, " min")
        rC = run_pairwise_quantile_upper_checkpointed(nothing; ckpt_dir = dir, label = labC,
            delta = delta, maxtime_real = STAGE_C_MIN * 60, checkpoint_interval_s = 120.0,
            objective_mode = :min_gp, outer_direct_hessopt = :sr1, resume_from = resume_src,
            verbose = true, common...)
        lp("  stage C done: status=", rC.knitro_status, " n_eval=", rC.n_eval,
           " best gp=", rC.best === nothing ? "nothing" : rC.best.gp, " kappa=", rC.kappa)
        results[:C] = rC
    end

    # ---- cell summary ----
    best_gp = nothing
    for lab in (labA, labB, labC)
        p = ckpt_of(dir, lab)
        isfile(p) || continue
        ck = load_pairwise_quantile_checkpoint(p)
        ck.best_feasible === nothing && continue
        if best_gp === nothing || is_better_polish(ck.best_feasible.gp, best_gp.gp, FIND_SMALLEST)
            best_gp = ck.best_feasible
        end
    end
    κ = best_gp === nothing ? NaN : 1 - best_gp.gp^(SIGMA / (SIGMA - 1))
    lp("DELTA CELL delta=", delta, " COMPLETE: best gp=",
       best_gp === nothing ? "nothing" : best_gp.gp, "  Delta=",
       best_gp === nothing ? "nothing" : best_gp.Delta, "  kappa=", κ)
    open(joinpath(CAMPAIGN_ROOT, "summary_L$(PQ_L).txt"), "a") do io
        println(io, "delta=", delta, "  gp=", best_gp === nothing ? "nothing" : best_gp.gp,
                "  Delta=", best_gp === nothing ? "nothing" : best_gp.Delta, "  kappa=", κ,
                "  cutoff_source=", CUTOFF_SOURCE, "  L=", PQ_L, "  t=", Dates.now())
    end
    return best_gp
end

for delta in DELTAS
    try
        run_delta_cell(delta)
    catch e
        lp("DELTA CELL delta=", delta, " FAILED: ", first(sprint(showerror, e), 600))
        lp("continuing to the next delta -- the failed cell's checkpoints (if any) are on disk and ",
           "re-running this script will retry it.")
        flush(stdout)
    end
end

lp("="^96)
lp("PAIRWISE-QUANTILE PRODUCTION RUN COMPLETE (L=", PQ_L, ", cutoff_source=:", CUTOFF_SOURCE, ")")
lp("summary: ", joinpath(CAMPAIGN_ROOT, "summary_L$(PQ_L).txt"))
lp("="^96)
