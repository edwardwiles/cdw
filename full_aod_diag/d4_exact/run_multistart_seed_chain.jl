_D4E = joinpath(@__DIR__)
const D4X = _D4E   # the include block below is shared verbatim with the CM+PQ smoke, which names it D4X
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
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_cplus.jl",
          "pairwise_quantile_checkpoint.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl", "cm_pairwise_quantile_cplus.jl",
          "cm_pairwise_quantile_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Printf, Serialization, SpecialFunctions
import Dates

using LinearAlgebra, Printf, Serialization, SpecialFunctions
import Dates

lp(xs...) = (println("[pqseed ", Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"), "] ", xs...); flush(stdout))

# ================================================================================================
# ONE multistart chain of the PQ-only upper-bound campaign, started from ONE of the seeds the
# reproducible five-family campaign (protocols/paper_upper_v1.toml) already generated.
#
# WHY THE SEEDS TRANSFER. `seeds/seeds/S<k>/economic_seed.jls` stores `economic_vector`, the
# FAMILY-AGNOSTIC economic block of the seed (the per-family subdirectories store only digests and
# qualification metadata, not a second vector). A PQ outer point is exactly
# `vcat(economic_block, mass_coords)`, so a campaign seed transfers by concatenating the SAME
# economic vector with this family's own mass start. Nothing is re-generated and no RNG is re-run.
#
# WHAT DOES NOT TRANSFER -- READ THIS. The campaign qualified each seed with
# `seed_requires_all_five_families = true` and `Delta* <= 0.98`, against the FIVE paper families.
# Pairwise-quantile was not one of them and did not exist when these seeds were generated. So a
# seed's Delta* under PQ is UNKNOWN a priori and is measured here, per seed, before any stage runs.
# It decides which protocol budget path applies:
#     Delta*(seed) <= delta_1        -> [budgets.discovery_feasible_start]    A=75 B=30 C=75
#     delta_1 < Delta*(seed) < 1.0   -> [budgets.discovery_infeasible_start]  R0=30 A=60 B=30 C=60
#     Delta*(seed) >= 1.0            -> seed does NOT qualify for PQ; chain is SKIPPED and recorded
# (protocol section 5 / addendum C). Do not assume the feasible path.
#
# TWO FAMILIES share this file, because everything except the driver call and the shape of the mass
# block is identical: the seed handling, the Delta*(seed) qualification probe, the protocol budget
# routing, and the R0/A/B/C stage machine. The families differ in exactly two ways:
#
#   PQ    (standalone)  mass block is PER-ORIGIN, D*(L-1) coordinates, driver
#                       run_pairwise_quantile_upper_checkpointed, needs cutoff_source.
#   CMPQ  (CM + PQ)     CM pins all marginals to a shared reference, so the mass block COLLAPSES to
#                       (L-1) coordinates on that reference; driver
#                       run_cm_pairwise_quantile_upper_checkpointed, needs cm_grid_size /
#                       cm_moment_families / contrasts / mass_start, and has no cutoff_source.
#                       L MUST divide cm_grid_size or the shared-mass argument is unsound.
#
#   usage: julia run_multistart_seed_chain.jl <family> <seed_dir> <out_dir> <L> <deltas_csv>
#          family = PQ | CMPQ
# ================================================================================================
length(ARGS) >= 5 || error("usage: run_multistart_seed_chain.jl <family:PQ|CMPQ> <seed_dir> <out_dir> <L> <deltas_csv>")
const FAMILY   = Symbol(uppercase(ARGS[1]))
FAMILY in (:PQ, :CMPQ) || error("family must be PQ or CMPQ, got $(ARGS[1])")
const SEED_DIR = abspath(ARGS[2])
const OUT_DIR  = abspath(ARGS[3])
const PQ_L     = parse(Int, ARGS[4])
const DELTAS   = parse.(Float64, split(ARGS[5], ","))
mkpath(OUT_DIR)

# ---- protocol [scientific] (paper_upper_v1.toml), restated; that file is FROZEN and only read ----
const SIGMA, W_PROD          = 3.0, 100_000
const DRAW_DESIGN, DRAW_SEED = :sobol_randomized, 20260719
const DESTINATION_SAMPLE     = :exclude_row
const EXCLUDE_DIAG_GRAV      = true
const INNER_LOWER_LIMIT      = -10.0
const A_COORD_MODE           = :powered_aspace
const FIND_SMALLEST          = true            # direction = "upper"
const Z_HALFWIDTH            = 30.0
const CUTOFF_SOURCE          = :frechet_theoretical
const MIN_BIN_COUNT          = max(10, W_PROD ÷ (2 * PQ_L^2))
const GRAV                   = default_gravity_exclude_cells_brazil_korea()
# linsolver ma97 rather than the campaign's `auto`. This is a SOLVER setting, not a scientific one
# -- every linear solver tried agrees on Delta* to ~7e-14 -- and it is 1.29x at L=5, 6.26x at L=10.
const INNER_OPT              = FAMILY === :CMPQ ? "ek_inner_cmpq.opt" :
                               (PQ_L >= 8 ? "ek_inner_pq_ma97_t10.opt" : "ek_inner_pq.opt")
# ---- CM settings, from protocol [families.COMMON_MARGINALS.kwargs] (CMPQ only) ----
# L = 50 there is CM's GRID SIZE (this family's `cm_grid_size`), not the PQ bin count. contrasts is
# orthonormal. include_truncated_moment=false means the single-family eq.35 variant, i.e.
# cm_moment_families = 1 -- the protocol records that the eq.35+eq.36 two-family variant is not
# reachable through the production driver without a dense Hessian, which this protocol never uses.
const CM_GRID_SIZE       = 50
const CM_MOMENT_FAMILIES = 1
const CM_CONTRASTS       = :orthonormal
const MASS_START         = :uniform
if FAMILY === :CMPQ && CM_GRID_SIZE % PQ_L != 0
    error("CMPQ requires L | cm_grid_size so the PQ cutoffs lie on CM's grid: L=$PQ_L does not divide $CM_GRID_SIZE. " *
          "Valid L for G=50: 2, 5, 10, 25.")
end

# ---- protocol [budgets], both paths ----
const FEAS_A, FEAS_B, FEAS_C           = 75.0, 30.0, 75.0
const INFEAS_R0, INFEAS_A, INFEAS_B, INFEAS_C = 30.0, 60.0, 30.0, 60.0
const SEED_DELTA_CEILING               = 1.0    # [seeds].seed_delta_max

seed_rec = deserialize(joinpath(SEED_DIR, "economic_seed.jls"))
lp("="^96)
lp(FAMILY, " MULTISTART CHAIN  seed=", seed_rec.seed_id, "  L=", PQ_L, "  deltas=", DELTAS)
lp("  seed gp=", seed_rec.gp, "  A_pert_rms=", seed_rec.A_perturbation_rms,
   "  economic_digest=", seed_rec.economic_digest)
lp("  seed W=", seed_rec.W, "  source_sha=", seed_rec.source_sha)
lp("  inner opt=", INNER_OPT)
lp("="^96)
seed_rec.W == W_PROD ||
    error("seed $(seed_rec.seed_id) was generated at W=$(seed_rec.W) but this run uses W=$W_PROD")

t0 = time()
ctx_raw = d20_real_setup_design(W = W_PROD, δ = first(DELTAS), find_smallest = FIND_SMALLEST,
    draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, destination_sample = DESTINATION_SAMPLE,
    exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV, gravity_exclude_cells = GRAV,
    σHat = SIGMA, inner_lower_limit = INNER_LOWER_LIMIT)
ctx    = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
layout = PairwiseQuantileMassLayout(ctx.D, PQ_L)
geo    = build_aspace_geometry(ctx)

# Length check against THIS family's own layout -- a silent length mismatch would otherwise be
# reinterpreted as mass coordinates and produce a plausible-looking wrong answer.
w_cal_econ = cm_w0_from_calibration(ctx, geo.pe, A_COORD_MODE)
econ = collect(seed_rec.economic_vector)
length(econ) == length(w_cal_econ) || error(
    "seed economic_vector length $(length(econ)) != this context's economic block $(length(w_cal_econ)); " *
    "the seed was generated under a different gravity mask / destination_sample / A_coordinate_mode")
# CM collapses the mass block: with all marginals pinned to one reference there is a single
# simplex, (L-1) coordinates, NOT the standalone family's D*(L-1).
mass0 = FAMILY === :CMPQ ? cmpq_uniform_mass_raw(PQ_L) : uniform_mass_raw(layout)
const N_RAW_EXPECT = FAMILY === :CMPQ ? PQ_L - 1 : n_raw(layout)
length(mass0) == N_RAW_EXPECT ||
    error("$FAMILY mass block is $(length(mass0)), expected $N_RAW_EXPECT")
W0 = vcat(econ, mass0)
lp("context in ", round(time() - t0, digits = 1), "s;  family=", FAMILY, "  econ=", length(econ),
   "  n_raw=", length(mass0), "  w0=", length(W0))

ckpt_of(dir, label) = joinpath(dir, "$(label)_latest.jls")
# Scientific kwargs common to both families, then each family's own required block. Both drivers
# enforce the repo's no-defaults rule, so an omission here is an UndefKeywordError, not a silent
# substitution.
const SCI = (W = W_PROD, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED, σHat = SIGMA,
             inner_lower_limit = INNER_LOWER_LIMIT, z_halfwidth = Z_HALFWIDTH,
             destination_sample = DESTINATION_SAMPLE, exclude_diagonal_gravity = EXCLUDE_DIAG_GRAV,
             gravity_exclude_cells = GRAV, L = PQ_L, min_bin_count = MIN_BIN_COUNT,
             A_coordinate_mode = A_COORD_MODE, find_smallest = FIND_SMALLEST)
const FAMKW = FAMILY === :CMPQ ?
    (cm_grid_size = CM_GRID_SIZE, cm_moment_families = CM_MOMENT_FAMILIES,
     contrasts = CM_CONTRASTS, mass_start = MASS_START, inner_opt = INNER_OPT) :
    (cutoff_source = CUTOFF_SOURCE, inner_opt_override = INNER_OPT)
common = merge(SCI, FAMKW)

"Family-dispatched stage driver. Every stage below calls THIS, never a family-specific name."
run_stage(w0; kwargs...) = FAMILY === :CMPQ ?
    run_cm_pairwise_quantile_upper_checkpointed(w0; kwargs...) :
    run_pairwise_quantile_upper_checkpointed(w0; kwargs...)

"Family-dispatched checkpoint loader."
load_ckpt(path) = FAMILY === :CMPQ ?
    load_cm_pairwise_quantile_checkpoint(path) : load_pairwise_quantile_checkpoint(path)

"""
Measure Delta*(seed) under THIS family at `delta`, by running the outer driver for a single
evaluation (maxtime_real is tiny, so it evaluates w0, verifies, and stops). Returns the checkpoint's
first incumbent, or `nothing` if the point does not verify.
"""
function seed_delta_star(delta::Float64)
    probe_dir = joinpath(OUT_DIR, "seed_probe")
    mkpath(probe_dir)
    r = run_stage(copy(W0); ckpt_dir = probe_dir,
        label = "$(FAMILY)_probe_d$(replace(string(delta), "." => "p"))", delta = delta,
        maxtime_real = 1.0, checkpoint_interval_s = 1e9, objective_mode = :min_gp,
        verbose = false, common...)
    return r.best === nothing ? nothing : (Delta = r.best.Delta, gp = r.best.gp)
end

delta_1 = minimum(DELTAS)
lp("measuring Delta*(seed) under PQ at delta=", delta_1, " (seed was qualified for the OTHER five families, not this one) ...")
probe = seed_delta_star(delta_1)
if probe === nothing
    lp("SEED_VERDICT family=", FAMILY, " seed=", seed_rec.seed_id, " UNVERIFIED at delta=", delta_1, " -- chain SKIPPED")
    open(joinpath(OUT_DIR, "seed_verdict.txt"), "w") do io
        println(io, "family=$FAMILY seed_id=$(seed_rec.seed_id) verdict=UNVERIFIED delta_probe=$delta_1")
    end
    exit(0)
end
lp("  Delta*(seed) = ", probe.Delta, "   gp(seed) = ", probe.gp)

path = probe.Delta >= SEED_DELTA_CEILING ? :unqualified :
       (probe.Delta <= delta_1 ? :feasible_start : :infeasible_start)
lp("SEED_VERDICT family=", FAMILY, " seed=", seed_rec.seed_id, " Delta_star=", probe.Delta, " path=", path)
open(joinpath(OUT_DIR, "seed_verdict.txt"), "w") do io
    println(io, "family=$FAMILY seed_id=$(seed_rec.seed_id) verdict=$path Delta_star=$(probe.Delta) gp=$(probe.gp) delta_probe=$delta_1")
end
if path === :unqualified
    lp("Delta*(seed)=", probe.Delta, " >= ", SEED_DELTA_CEILING,
       " ([seeds].seed_delta_max): this seed does NOT qualify for PQ. Chain SKIPPED, recorded, not run.")
    exit(0)
end

# PQ_PROBE_ONLY=1 stops here, after the verdict is measured and written. This is what the campaign's
# preflight uses to answer "how many seeds fall into each category under PQ" WITHOUT committing to
# the ~24h of stages -- one inner solve per seed rather than 4 delta cells x 180 min.
if get(ENV, "PQ_PROBE_ONLY", "0") != "0"
    lp("PQ_PROBE_ONLY set -- verdict recorded, stages NOT run.")
    exit(0)
end

"Run one delta cell: R0 (infeasible start, delta_1 only) -> A -> B -> C, each resuming from the last."
function run_delta_cell(delta::Float64, is_first::Bool)
    tag = replace(string(delta), "." => "p")
    dir = joinpath(OUT_DIR, "delta_$tag"); mkpath(dir)
    lp("-"^90); lp("DELTA CELL delta=", delta, "  dir=", dir)
    A_min, B_min, C_min = path === :feasible_start ? (FEAS_A, FEAS_B, FEAS_C) : (INFEAS_A, INFEAS_B, INFEAS_C)
    start_w, resume = copy(W0), nothing

    # Stage R0 -- infeasible-start restoration, delta_1 cell only (addendum C)
    labR = "$(FAMILY)_S_d$(tag)_R0"
    if path === :infeasible_start && is_first && !isfile(ckpt_of(dir, labR))
        lp("  stage R0: initial restoration, budget ", INFEAS_R0, " min")
        rR = run_stage(copy(W0); ckpt_dir = dir, label = labR,
            delta = delta, maxtime_real = INFEAS_R0 * 60, checkpoint_interval_s = 120.0,
            objective_mode = :min_delta_fixed_gp, gp_fixed = probe.gp, verbose = false, common...)
        lp("  stage R0 done: status=", rR.knitro_status, " n_eval=", rR.n_eval,
           " best Delta=", rR.best === nothing ? "nothing" : rR.best.Delta)
    end
    isfile(ckpt_of(dir, labR)) && (resume = ckpt_of(dir, labR))

    labA = "$(FAMILY)_S_d$(tag)_A"
    if isfile(ckpt_of(dir, labA))
        lp("  stage A: checkpoint exists, skipping")
    else
        lp("  stage A: primary push (pin_cg_lbfgs), budget ", A_min, " min")
        rA = run_stage(resume === nothing ? copy(start_w) : nothing;
            ckpt_dir = dir, label = labA, delta = delta, maxtime_real = A_min * 60,
            checkpoint_interval_s = 120.0, objective_mode = :min_gp, pin_outer_algorithm = true,
            resume_from = resume, verbose = true, common...)
        lp("  stage A done: status=", rA.knitro_status, " n_eval=", rA.n_eval, " n_grad=", rA.n_grad,
           " best gp=", rA.best === nothing ? "nothing" : rA.best.gp, " kappa=", rA.kappa)
    end

    ckA = load_ckpt(ckpt_of(dir, labA))
    labB = "$(FAMILY)_S_d$(tag)_B"
    if ckA.best_feasible === nothing
        lp("  stage B: SKIPPED -- stage A produced no verified feasible incumbent to pin gp at.")
    elseif isfile(ckpt_of(dir, labB))
        lp("  stage B: checkpoint exists, skipping")
    else
        lp("  stage B: fixed-gp restoration at gp=", ckA.best_feasible.gp, ", budget ", B_min, " min")
        rB = run_stage(copy(ckA.best_feasible.w); ckpt_dir = dir,
            label = labB, delta = delta, maxtime_real = B_min * 60, checkpoint_interval_s = 120.0,
            objective_mode = :min_delta_fixed_gp, gp_fixed = ckA.best_feasible.gp,
            verbose = false, common...)
        lp("  stage B done: status=", rB.knitro_status, " n_eval=", rB.n_eval,
           " best Delta=", rB.best === nothing ? "nothing" : rB.best.Delta)
    end

    labC = "$(FAMILY)_S_d$(tag)_C"
    if isfile(ckpt_of(dir, labC))
        lp("  stage C: checkpoint exists, skipping")
    else
        src = isfile(ckpt_of(dir, labB)) ? ckpt_of(dir, labB) : ckpt_of(dir, labA)
        lp("  stage C: alternate push (direct_sr1) from ", basename(src), ", budget ", C_min, " min")
        rC = run_stage(nothing; ckpt_dir = dir, label = labC,
            delta = delta, maxtime_real = C_min * 60, checkpoint_interval_s = 120.0,
            objective_mode = :min_gp, outer_direct_hessopt = :sr1, resume_from = src,
            verbose = true, common...)
        lp("  stage C done: status=", rC.knitro_status, " n_eval=", rC.n_eval,
           " best gp=", rC.best === nothing ? "nothing" : rC.best.gp, " kappa=", rC.kappa)
    end

    ck = load_ckpt(ckpt_of(dir, isfile(ckpt_of(dir, labC)) ? labC : labA))
    bf = ck.best_feasible
    lp("DELTA CELL delta=", delta, " COMPLETE: best gp=", bf === nothing ? "nothing" : bf.gp,
       "  Delta=", bf === nothing ? "nothing" : bf.Delta)
    return bf
end

results = Dict{Float64,Any}()
for (i, d) in enumerate(DELTAS)
    results[d] = run_delta_cell(d, i == 1)
end

open(joinpath(OUT_DIR, "chain_summary.txt"), "w") do io
    println(io, "family=$FAMILY seed_id=$(seed_rec.seed_id) L=$PQ_L path=$path Delta_star_seed=$(probe.Delta)")
    for d in DELTAS
        bf = results[d]
        println(io, "delta=$d gp=$(bf === nothing ? "nothing" : bf.gp) Delta=$(bf === nothing ? "nothing" : bf.Delta)")
    end
end
lp("CHAIN COMPLETE family=", FAMILY, " seed=", seed_rec.seed_id, " -> ", joinpath(OUT_DIR, "chain_summary.txt"))
