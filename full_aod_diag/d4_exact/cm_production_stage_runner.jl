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
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
using Printf, Random, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const CKPT_DIR = ARGS[1]
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const MODE = ARGS[4]
const SEED_ARG = length(ARGS) >= 5 ? ARGS[5] : nothing
const CHAIN_PERTURB_SEED = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0

const DRAW_SEED = 20260719   # fixed across chains/deltas -- this is the economic DGP's own
                              # draw seed, part of the problem instance, NOT an optimizer
                              # multistart seed. Chain diversity comes from CHAIN_PERTURB_SEED
                              # perturbing the STARTING POINT only (see below), matching how
                              # every other real-data script in this tree fixes draw_seed=20260719.
const W = 80_000
const L = 50

mkpath(CKPT_DIR)
lp(">>> Julia threads: ", Threads.nthreads(), "  mode=", MODE, " delta=", DELTA, " budget=", BUDGET,
   "s ckpt_dir=", CKPT_DIR, " chain_perturb_seed=", CHAIN_PERTURB_SEED)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

w0 = nothing
resume_from = nothing

if MODE == "calibration"
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe0))
    if CHAIN_PERTURB_SEED == 0
        w0 = w_calib
    else
        rng = MersenneTwister(hash((:cm_campaign_chain_perturb, CHAIN_PERTURB_SEED)))
        # Small, bounded perturbation of the FREE z-coordinates only (gp left at the calibration
        # value) -- large enough to give genuinely different KNITRO trajectories across chains,
        # small enough to stay well inside the z +/- 30 box run_cm_upper_checkpointed itself sets.
        w0 = copy(w_calib)
        w0[2:end] .+= 0.5 .* randn(rng, length(w0) - 1)
    end
    lp(">>> calibration start: g=", w0[1], " ||zfree||=", norm(w0[2:end]))
elseif MODE == "seed_w0"
    seed = deserialize(SEED_ARG)
    seed.delta == DELTA || lp(">>> NOTE: seed vector was cold-verified at delta=", seed.delta,
                               " (this stage is delta=", DELTA, ") -- expected, this IS the",
                               " cross-delta warm-start step.")
    w0 = seed.w
    lp(">>> seeded from cold-verified prior-stage incumbent: ", SEED_ARG,
       " (prior Delta_dual=", seed.Delta_dual, ")")
elseif MODE == "resume"
    resume_from = SEED_ARG
    isfile(resume_from) || error("cm_production_stage_runner: resume requested but no checkpoint at $resume_from")
    lp(">>> RESUMING this stage from ", resume_from)
else
    error("cm_production_stage_runner: unknown mode $MODE (expected calibration|seed_w0|resume)")
end

res = run_cm_upper_checkpointed(resume_from === nothing ? w0 : nothing;
    W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = DRAW_SEED,
    L = L, contrasts = :anchored, probs = probs,
    cm_hessian_backend = :structured, cm_grid_rule = :nested_family,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "cm_campaign_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    label = "stage", checkpoint_interval_s = 30.0, resume_from = resume_from,
    heartbeat_interval_s = 30.0)

lp(">>> STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   " n_eval=", res.n_eval, " n_grad=", res.n_grad,
   " best=", res.best === nothing ? "nothing" : "gp=$(res.best.gp) Delta=$(res.best.Delta)",
   " kappa=", res.kappa)
lp(">>> checkpoint: ", res.ckpt_path)
lp(">>> STAGE_DONE")   # sentinel line the supervisor greps for to distinguish "finished" from "hung/killed"
