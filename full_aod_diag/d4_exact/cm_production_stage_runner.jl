# ============================================================================
# One delta-stage of the 2026-07-22 D=20 CM production campaign. Owns exactly ONE
# checkpoint/output directory (never shared across processes/chains -- the supervisor
# guarantees this by construction, one directory per chain per delta).
#
# Seeding modes (ARGS[4]):
#   "calibration"  -- ARGS[5] unused. Starts from the real-data calibration point
#                     (theta0_up's own free parameters), the same start every stage-0.1
#                     of every chain uses. Chain-to-chain variation (for the 3
#                     independent chains) comes from a small seeded perturbation
#                     (ARGS[6] = chain perturbation seed; 0 = unperturbed / chain 1).
#   "seed_w0"      -- ARGS[5] = path to a serialized NamedTuple written by
#                     cm_cold_verify.jl (fields: w, Delta_dual, ...). Starts a FRESH
#                     run (resume_from=nothing) at that cold-verified w vector. This is
#                     the path used going from one delta to the next -- per the brief,
#                     "each higher delta must initialize from the preceding stage's best
#                     cold-verified feasible incumbent, not its terminal iterate."
#   "resume"       -- ARGS[5] = path to an existing CMCheckpoint (this SAME stage's own
#                     "<label>_latest.jls"). Used only when the supervisor is restarting
#                     a stage that hung/was killed mid-run -- continues the SAME stage,
#                     same delta, with the remaining wall budget.
#
# Usage:
#   julia --project=. cm_production_stage_runner.jl <ckpt_dir> <delta> <budget_s> <mode> <seed_arg> [chain_perturb_seed]
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))   # Part B restoration (2026-07-23 release): wires pairwise/hard-winner/witness screens into the CM inner-solve path
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))   # overnight task 2026-07-22: CM-aware C+ backend, opt-in via cm_gradient_backend=:cplus
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))   # CM+moments(+ZC) production integration 2026-07-23
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const CKPT_DIR = abspath(ARGS[1])   # release fix 2026-07-23 section 4.1: resolve before any real-data setup runs
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const MODE = ARGS[4]
const SEED_ARG = length(ARGS) >= 5 ? ARGS[5] : nothing
const CHAIN_PERTURB_SEED = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0

# CM-C+ production integration 2026-07-23: env override, defaults to the production backend
# (:cplus) when unset. `CM_GRADIENT_BACKEND=reference` selects the documented fallback/
# validation backend instead. Unknown values fail immediately (run_cm_upper_checkpointed's own
# validation), not silently coerced.
const CM_GRADIENT_BACKEND = Symbol(get(ENV, "CM_GRADIENT_BACKEND", "cplus"))
# Only meaningful for MODE=="resume": explicit, audited override required to resume under a
# DIFFERENT cm_gradient_backend than the checkpoint this stage is resuming was written with.
const CM_ALLOW_BACKEND_SWITCH = get(ENV, "CM_ALLOW_BACKEND_SWITCH", "0") == "1"

# exclude-ROW-destination production release (2026-07-24): env-driven opt-in, same pattern as
# CM_GRADIENT_BACKEND above. Defaults to :exclude_row (PRODUCTION DEFAULT as of this release --
# true D_origin/D_dest dimension shrink, ROW dropped as a destination only). Combinable with
# cm_gradient_backend=:cplus (rectangularized + validated real D=20/W=80000, see
# lfix_cplus_exclude_row_validation.jl). :all_legacy (square D x D) remains an explicit,
# reproduction-only opt-out, byte-identical to every pre-existing CM production run.
const CM_DESTINATION_SAMPLE = Symbol(get(ENV, "CM_DESTINATION_SAMPLE", "exclude_row"))
CM_DESTINATION_SAMPLE in (:all_legacy, :exclude_row) ||
    error("cm_production_stage_runner: CM_DESTINATION_SAMPLE must be all_legacy|exclude_row, got $CM_DESTINATION_SAMPLE")

# CM+moments(+ZC) production integration (2026-07-23): env overrides, same pattern as
# CM_GRADIENT_BACKEND above. Defaults to :cm_only (production default, unchanged behavior).
const CM_EXTENSION = Symbol(get(ENV, "CM_EXTENSION", "cm_only"))
const MEANZC_K_MEAN = parse(Int, get(ENV, "MEANZC_K_MEAN", "0"))
const MEANZC_K_PAIR = parse(Int, get(ENV, "MEANZC_K_PAIR", "0"))
const MEANZC_BASIS = Symbol(get(ENV, "MEANZC_BASIS", "direct"))
const IS_MEANZC = CM_EXTENSION !== :cm_only
# Initial nu_k for a fresh "calibration" start only (irrelevant for seed_w0/resume, which carry
# their own eta_nu forward) -- defaults to E_F[z^k]=k! (the Exp(1) draws' own k-th raw moment),
# matching test_cm_meanzc_d4_gates.jl's nu0vec convention. Override via MEANZC_ETA_NU0 (comma-
# separated log-nu values, one per level) only for deliberate off-calibration smoke testing.
const MEANZC_ETA_NU0_OVERRIDE = get(ENV, "MEANZC_ETA_NU0", "")

const DRAW_SEED = 20260719   # fixed across chains/deltas -- this is the economic DGP's own
                              # draw seed, part of the problem instance, NOT an optimizer
                              # multistart seed. Chain diversity comes from CHAIN_PERTURB_SEED
                              # perturbing the STARTING POINT only (see below), matching how
                              # every other real-data script in this tree fixes draw_seed=20260719.
const W = 80_000
const L = 50
const DRAW_DESIGN = :pseudorandom
# Approved production contrast basis (2026-07-22 contrast-basis review, see
# docs/fullA_cm_conditioning_and_adaptive_grid_report.md and
# docs/fullA_cm_hessian_architecture_report.md): :orthonormal, not the old default :anchored.
# Conditioning evidence (Part 1 of the conditioning report) shows orthonormal strictly
# dominates anchored at every L in {10,20,50} and both points tested (2.6-3.4x lower
# cond(Hessian)), with the gap widening as L grows -- exactly the L=50 regime this campaign
# runs at -- and orthonormal is also far less reference-country-sensitive (0.8-1.5% spread vs
# 2.9-9.7% for anchored). The only documented cost is losing per-origin sparsity in the CM
# moment columns; that cost does not block production because the wired Hessian backend
# (cm_hessian_backend=:structured, Architecture C) was independently validated correct AND
# still gives a real 2.2x-4.5x speedup under orthonormal contrasts at L=50 (hessian
# architecture report, Section 6) -- the conditioning win is not traded away against a broken
# or unvalidated fast path. (The separately-measured :interval basis is NOT this axis: it is a
# different moment-construction family not wired into build_cm_production_context at all, out
# of scope for this decision, which is specifically anchored-vs-orthonormal within the
# existing `contrasts::Symbol` parameter.)
const CM_CONTRASTS = :orthonormal
const FOCAL_BASEINDEX = 2   # France -- fixed by build_ad_context_real_d20, not a free parameter
                             # of this campaign; recorded here purely so seed-provenance checks
                             # below have a literal to assert against.

mkpath(CKPT_DIR)
lp(">>> Julia threads: ", Threads.nthreads(), "  mode=", MODE, " delta=", DELTA, " budget=", BUDGET,
   "s ckpt_dir=", CKPT_DIR, " chain_perturb_seed=", CHAIN_PERTURB_SEED, " contrasts=", CM_CONTRASTS,
   " cm_gradient_backend=", CM_GRADIENT_BACKEND, CM_ALLOW_BACKEND_SWITCH ? " (allow_backend_switch=true)" : "",
   " cm_extension=", CM_EXTENSION, IS_MEANZC ? " K_mean=$(MEANZC_K_MEAN) K_pair=$(MEANZC_K_PAIR) meanzc_basis=$(MEANZC_BASIS)" : "",
   " destination_sample=", CM_DESTINATION_SAMPLE)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

w0 = nothing
resume_from = nothing

if MODE == "calibration"
    # Part A (2026-07-23): d20_real_setup's default flipped to destination_sample=:exclude_row.
    # CM's own moment/pivot-elimination layer (build_pivot_elimination et al) was rectangularized
    # in the follow-up Lfix-gradient-layer pass (2026-07-24) -- CM_DESTINATION_SAMPLE now threads
    # a real runtime choice through, default :all_legacy (unchanged behavior), :exclude_row opt-in.
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true, destination_sample = CM_DESTINATION_SAMPLE)
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D; Ddest = ctx0.D_dest
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))
    local rng_seed_used = nothing
    if CHAIN_PERTURB_SEED == 0
        w0 = w_calib
    else
        # Stable, version-independent seed derivation. Julia's generic `hash` is explicitly NOT
        # a persistent/public API across Julia versions or even processes for all types (Base
        # makes no stability guarantee), so it must not be used to derive a production RNG seed
        # -- a Julia upgrade could silently change every chain's starting point with no error and
        # no way to reproduce a prior run's exact trajectory. This is a fixed integer arithmetic
        # formula instead: deterministic across Julia versions/machines by construction, and the
        # exact mapping is printed/serialized below so any chain's start point can be reproduced
        # from (tag, chain_id, draw_seed) alone.
        rng_seed_used = 2026_0722_00 + CHAIN_PERTURB_SEED
        rng = MersenneTwister(rng_seed_used)
        # Small, bounded perturbation of the FREE z-coordinates only (gp left at the calibration
        # value) -- large enough to give genuinely different KNITRO trajectories across chains,
        # small enough to stay well inside the z +/- 30 box run_cm_upper_checkpointed itself sets.
        # NOTE (found live this session): an earlier version of this used scale=0.5, which
        # produced a start point KNITRO's own presolver could not evaluate at all
        # (knitro_status=-502 "Could not evaluate objective or constraints at the initial
        # point", n_eval=0 -- confirmed via a real chain_perturb_seed=1 run,
        # smoke_test_2026-07-22/interrupt/stage.log). 0.02 is conservative enough to stay in the
        # well-defined region while still giving each chain a genuinely different trajectory.
        w0 = copy(w_calib)
        w0[2:end] .+= 0.02 .* randn(rng, length(w0) - 1)
    end
    # CM+moments(+ZC) production integration (2026-07-23): append eta_nu_1..eta_nu_K_mean to the
    # economic w0, matching cm_meanzc_production.jl's own w_ext = [gp; zfree; eta_nu_1;...] convention.
    # Perturbing eta_nu too (when CHAIN_PERTURB_SEED != 0) is deliberate -- gives genuinely
    # different starting nu across chains, not just different (g,A_od).
    if IS_MEANZC
        eta_nu0 = if isempty(MEANZC_ETA_NU0_OVERRIDE)
            log.(Float64.(factorial.(1:MEANZC_K_MEAN)))
        else
            parse.(Float64, split(MEANZC_ETA_NU0_OVERRIDE, ","))
        end
        length(eta_nu0) == MEANZC_K_MEAN ||
            error("cm_production_stage_runner: MEANZC_ETA_NU0 has $(length(eta_nu0)) values, expected MEANZC_K_MEAN=$(MEANZC_K_MEAN)")
        w0 = vcat(w0, eta_nu0)
    end

    lp(">>> calibration start: g=", w0[1], " ||zfree||=", norm(w0[2:(IS_MEANZC ? end-MEANZC_K_MEAN : end)]),
       IS_MEANZC ? " eta_nu0=$(w0[end-MEANZC_K_MEAN+1:end])" : "",
       " chain_perturb_seed=", CHAIN_PERTURB_SEED, " rng_seed=", something(rng_seed_used, "n/a (unperturbed)"))
    lp(">>> calibration start w0 (full vector, reproducible from tag+chain_id+draw_seed): ", w0)
    serialize(joinpath(CKPT_DIR, "w0_used.jls"),
        (chain_perturb_seed = CHAIN_PERTURB_SEED, rng_seed = rng_seed_used, w0 = copy(w0),
         draw_seed = DRAW_SEED, W = W, contrasts = CM_CONTRASTS, delta = DELTA,
         cm_extension = CM_EXTENSION, meanzc_K_mean = MEANZC_K_MEAN, meanzc_K_pair = MEANZC_K_PAIR,
         meanzc_basis = MEANZC_BASIS, generated_at = string(now())))

    # Feasibility pre-check (found necessary live, see NOTE above): never hand KNITRO a start
    # point that cannot even be evaluated -- fail fast with an actionable message instead of
    # burning the stage's wall budget on an instant KN_solve presolve error.
    let D2_econ0 = length(w0) - (IS_MEANZC ? MEANZC_K_MEAN : 0),
        xf0 = x_free_from_w(w0[1:D2_econ0], pe0)
        try
            if IS_MEANZC
                νvec0 = exp.(w0[D2_econ0+1:end])
                pcx0 = build_cm_meanzc_production_context(ctx0, CS; L = L, K_mean = MEANZC_K_MEAN, K_pair = MEANZC_K_PAIR,
                    contrasts = CM_CONTRASTS, meanzc_basis = MEANZC_BASIS, probs = probs)
                _, _, verify0 = cm_meanzc_production_value_verified_screened(xf0, νvec0, pcx0)
                isfinite(verify0.Delta_dual) || error("Delta_dual is not finite at the (possibly perturbed) start point")
                lp(">>> start-point feasibility pre-check passed: Delta_dual=", verify0.Delta_dual)
            else
                pcx0 = build_cm_production_context(ctx0, CS; L = L, contrasts = CM_CONTRASTS, probs = probs)
                pcx0 = with_screen_counters(pcx0)
                _, _, verify0 = cm_production_value_verified_screened(xf0, pcx0; counters = pcx0.screen_counters)
                isfinite(verify0.Delta_dual) || error("Delta_dual is not finite at the (possibly perturbed) start point")
                lp(">>> start-point feasibility pre-check passed: Delta_dual=", verify0.Delta_dual)
            end
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            error("cm_production_stage_runner: perturbed calibration start (chain_perturb_seed=$(CHAIN_PERTURB_SEED)) " *
                  "is not evaluable ($e) -- refusing to launch KNITRO on a point its own presolver would reject. " *
                  "Reduce the perturbation scale or use a different chain_perturb_seed.")
        end
    end
elseif MODE == "seed_w0"
    seed = deserialize(SEED_ARG)
    seed.delta == DELTA || lp(">>> NOTE: seed vector was cold-verified at delta=", seed.delta,
                               " (this stage is delta=", DELTA, ") -- expected, this IS the",
                               " cross-delta warm-start step.")
    # Cross-delta seed-provenance validation (only `delta` may legitimately differ between the
    # seed's own stage and this one -- everything else describes the SAME fixed production
    # problem instance, and a mismatch here would mean silently splicing together two different
    # economies/draw sets/moment restrictions across a delta transition).
    seed.W == W ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.W=$(seed.W) != this stage's W=$(W)")
    !hasproperty(seed, :draw_design) || seed.draw_design == DRAW_DESIGN ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.draw_design=$(seed.draw_design) != $(DRAW_DESIGN)")
    seed.draw_seed == DRAW_SEED ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.draw_seed=$(seed.draw_seed) != $(DRAW_SEED)")
    seed.cm_L == L ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.cm_L=$(seed.cm_L) != $(L)")
    !hasproperty(seed, :contrasts) || seed.contrasts == CM_CONTRASTS ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.contrasts=$(seed.contrasts) != $(CM_CONTRASTS)")
    !hasproperty(seed, :bi) || seed.bi === missing || seed.bi == FOCAL_BASEINDEX ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.bi=$(seed.bi) != $(FOCAL_BASEINDEX)")
    !hasproperty(seed, :schema) || seed.schema == CM_CHECKPOINT_SCHEMA ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.schema=$(seed.schema) != $(CM_CHECKPOINT_SCHEMA)")
    # CM+moments(+ZC) production integration (2026-07-23): the moment-column layout must be
    # IDENTICAL across a delta transition (only delta itself may legitimately differ) -- a seed
    # written under a different (cm_extension,K_mean,K_pair,meanzc_basis) would silently splice
    # together two different restriction sets.
    !hasproperty(seed, :cm_extension) || seed.cm_extension == CM_EXTENSION ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.cm_extension=$(seed.cm_extension) != $(CM_EXTENSION)")
    !hasproperty(seed, :meanzc_K_mean) || seed.meanzc_K_mean == MEANZC_K_MEAN ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.meanzc_K_mean=$(seed.meanzc_K_mean) != $(MEANZC_K_MEAN)")
    !hasproperty(seed, :meanzc_K_pair) || seed.meanzc_K_pair == MEANZC_K_PAIR ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.meanzc_K_pair=$(seed.meanzc_K_pair) != $(MEANZC_K_PAIR)")
    !hasproperty(seed, :meanzc_basis) || !IS_MEANZC || seed.meanzc_basis == MEANZC_BASIS ||
        error("cm_production_stage_runner: seed-provenance mismatch: seed.meanzc_basis=$(seed.meanzc_basis) != $(MEANZC_BASIS)")
    w0 = seed.w
    lp(">>> seeded from cold-verified prior-stage incumbent: ", SEED_ARG,
       " (prior Delta_dual=", seed.Delta_dual, ") -- provenance validated (W/draw_design/draw_seed/",
       "cm_L/contrasts/bi/schema/cm_extension/K_mean/K_pair/meanzc_basis all match except delta, as expected)")
elseif MODE == "resume"
    resume_from = SEED_ARG
    isfile(resume_from) || error("cm_production_stage_runner: resume requested but no checkpoint at $resume_from")
    lp(">>> RESUMING this stage from ", resume_from)
else
    error("cm_production_stage_runner: unknown mode $MODE (expected calibration|seed_w0|resume)")
end

res = run_cm_upper_checkpointed(resume_from === nothing ? w0 : nothing;
    W = W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    L = L, contrasts = CM_CONTRASTS, probs = probs,
    cm_hessian_backend = :structured, cm_grid_rule = :nested_family,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "cm_campaign_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    label = "stage", checkpoint_interval_s = 30.0, resume_from = resume_from,
    heartbeat_interval_s = 30.0,
    cm_gradient_backend = CM_GRADIENT_BACKEND, allow_backend_switch = CM_ALLOW_BACKEND_SWITCH,
    cm_extension = CM_EXTENSION, meanzc_K_mean = MEANZC_K_MEAN, meanzc_K_pair = MEANZC_K_PAIR,
    meanzc_basis = MEANZC_BASIS, destination_sample = CM_DESTINATION_SAMPLE)

lp(">>> STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   " n_eval=", res.n_eval, " n_grad=", res.n_grad,
   " best=", res.best === nothing ? "nothing" : "gp=$(res.best.gp) Delta=$(res.best.Delta)",
   " kappa=", res.kappa)
lp(">>> checkpoint: ", res.ckpt_path)
print_core_hessian_counters()   # final-gate continuation 2026-07-25 (task §2): prove the shared H_EE backend actually ran, not just that it was requested
lp(">>> STAGE_DONE")   # sentinel line the supervisor greps for to distinguish "finished" from "hung/killed"
