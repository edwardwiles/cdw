# ============================================================================
# One delta-stage of an origin-specific-ZC (no CM) D=20 production run.
# Structural analog of cm_production_stage_runner.jl (same ARGS convention,
# same calibration/seed_w0/resume modes, same STAGE_DONE sentinel the
# supervisor greps for) but calls run_originzc_upper_checkpointed -- the
# SEPARATE opt-in entry point for the origin-specific-moments(-zero-covariance)
# restriction family (cm_originzc_checkpoint.jl) -- never
# run_cm_upper_checkpointed, and never routed through CM_EXTENSION/meanzc.
# distribution_restriction=:unrestricted or CM-family runs continue to use
# cm_production_stage_runner.jl unchanged; this script is off by default and
# only reached when the supervisor is explicitly pointed at it
# (STAGE_RUNNER_SCRIPT=full_aod_diag/d4_exact/originzc_production_stage_runner.jl).
#
# Seeding modes (ARGS[4]) -- identical contract to cm_production_stage_runner.jl:
#   "calibration"  -- ARGS[5] unused. Starts from the real-data calibration point
#                     (gp0, zfree0 from the benchmark A*) plus the origin-specific
#                     eta_{o,k}^(0) = log(nu_{o,k}^(0)) initialization required by
#                     task brief Section 7: nu_{o,k}^(0) = (1/W) sum_s z_{so}^k,
#                     i.e. the empirical k-th raw moment of U (=exp(z)) at each
#                     origin o, NEVER a common/shared value across origins
#                     (tested pattern: d20_originzc_shakedown.jl).
#   "seed_w0"      -- ARGS[5] = path to a serialized NamedTuple written by
#                     originzc_cold_verify.jl. Starts a FRESH run at that
#                     cold-verified w vector (cross-delta warm start).
#   "resume"       -- ARGS[5] = path to an existing CMCheckpointV5 (this SAME
#                     stage's own "<label>_latest.jls"). Continues the SAME
#                     stage/delta with the remaining wall budget.
#
# Required env vars (no defaults -- explicit opt-in per task brief Section 12):
#   DISTRIBUTION_RESTRICTION   origin_specific_moments | origin_specific_moments_zero_covariance
#   ORIGIN_K_MEAN              >= 1
# Optional env vars:
#   ORIGIN_K_PAIR              default = ORIGIN_K_MEAN for the zero-covariance restriction,
#                               0 for the mean-only restriction (validated by OriginZCConfig)
#   POWER_TARGET_LAYOUT        default origin_by_power
#   MEANZC_BASIS               default direct
#   CM_GRADIENT_BACKEND        default cplus (production default; reference = fallback)
#   CM_ALLOW_BACKEND_SWITCH    default 0
#
# Usage:
#   julia --project=. originzc_production_stage_runner.jl <ckpt_dir> <delta> <budget_s> <mode> <seed_arg> [chain_perturb_seed]
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
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
using Printf, Random, Dates, Serialization, LinearAlgebra, Statistics

lp(xs...) = (println(xs...); flush(stdout))

# Release fix (2026-07-23, section 4.1): resolve BEFORE any real-data setup runs (see the
# identical fix and rationale in run_originzc_upper_checkpointed / run_cm_upper_checkpointed).
const CKPT_DIR = abspath(ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const MODE = ARGS[4]
const SEED_ARG = length(ARGS) >= 5 ? ARGS[5] : nothing
const CHAIN_PERTURB_SEED = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 0

const DISTRIBUTION_RESTRICTION = Symbol(ENV["DISTRIBUTION_RESTRICTION"])   # required, no default
const ORIGIN_K_MEAN = parse(Int, ENV["ORIGIN_K_MEAN"])                     # required, no default
const ORIGIN_K_PAIR = parse(Int, get(ENV, "ORIGIN_K_PAIR",
    DISTRIBUTION_RESTRICTION === :origin_specific_moments_zero_covariance ? string(ORIGIN_K_MEAN) : "0"))
const POWER_TARGET_LAYOUT = Symbol(get(ENV, "POWER_TARGET_LAYOUT", "origin_by_power"))
const MEANZC_BASIS = Symbol(get(ENV, "MEANZC_BASIS", "direct"))
const CM_GRADIENT_BACKEND = Symbol(get(ENV, "CM_GRADIENT_BACKEND", "cplus"))
const CM_ALLOW_BACKEND_SWITCH = get(ENV, "CM_ALLOW_BACKEND_SWITCH", "0") == "1"

const DRAW_SEED = 20260719   # fixed production draw seed -- see cm_production_stage_runner.jl's
                              # own comment; same problem-instance convention applies here.
const W = 80_000
const DRAW_DESIGN = :pseudorandom
const FOCAL_BASEINDEX = 2   # France

mkpath(CKPT_DIR)
lp(">>> Julia threads: ", Threads.nthreads(), "  mode=", MODE, " delta=", DELTA, " budget=", BUDGET,
   "s ckpt_dir=", CKPT_DIR, " chain_perturb_seed=", CHAIN_PERTURB_SEED,
   " distribution_restriction=", DISTRIBUTION_RESTRICTION, " K_mean=", ORIGIN_K_MEAN, " K_pair=", ORIGIN_K_PAIR,
   " power_target_layout=", POWER_TARGET_LAYOUT, " meanzc_basis=", MEANZC_BASIS,
   " cm_gradient_backend=", CM_GRADIENT_BACKEND, CM_ALLOW_BACKEND_SWITCH ? " (allow_backend_switch=true)" : "")

w0 = nothing
resume_from = nothing

if MODE == "calibration"
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe0))
    local rng_seed_used = nothing
    if CHAIN_PERTURB_SEED == 0
        w0 = w_calib
    else
        rng_seed_used = 2026_0722_00 + CHAIN_PERTURB_SEED   # same formula as cm_production_stage_runner.jl
        rng = MersenneTwister(rng_seed_used)
        w0 = copy(w_calib)
        w0[2:end] .+= 0.02 .* randn(rng, length(w0) - 1)
    end

    # Task brief Section 7: nu_{o,k}^(0) = (1/W) sum_s z_{so}^k (the k-th empirical raw moment
    # of U=exp(z) AT EACH ORIGIN, never a common/shared value) -- tested initialization pattern,
    # matches d20_originzc_shakedown.jl exactly.
    layout = OriginByPowerLayout(D, ORIGIN_K_MEAN, ORIGIN_K_PAIR)
    nu0 = Vector{Float64}(undef, n_eta(layout))
    nu0_matrix = Matrix{Float64}(undef, D, ORIGIN_K_MEAN)
    for k in 1:ORIGIN_K_MEAN
        Uk = ctx0.U .^ k
        for o in 1:D
            m = mean(@view Uk[:, o])
            nu0_matrix[o, k] = m
            nu0[target_index(layout, o, k)] = m
        end
    end
    eta0 = log.(nu0)
    w0 = vcat(w0, eta0)

    nu0_checksum = string(hash(round.(nu0_matrix, digits = 12)))
    lp(">>> calibration start: g=", w0[1], " ||zfree||=", norm(w0[2:end-length(eta0)]),
       " n_eta=", length(eta0), " chain_perturb_seed=", CHAIN_PERTURB_SEED,
       " rng_seed=", something(rng_seed_used, "n/a (unperturbed)"))
    lp(">>> initial nu matrix (D x K_mean), checksum=", nu0_checksum, ":")
    lp(nu0_matrix)
    serialize(joinpath(CKPT_DIR, "w0_used.jls"),
        (chain_perturb_seed = CHAIN_PERTURB_SEED, rng_seed = rng_seed_used, w0 = copy(w0),
         draw_seed = DRAW_SEED, W = W, delta = DELTA,
         distribution_restriction = DISTRIBUTION_RESTRICTION, origin_K_mean = ORIGIN_K_MEAN,
         origin_K_pair = ORIGIN_K_PAIR, power_target_layout = POWER_TARGET_LAYOUT,
         meanzc_basis = MEANZC_BASIS, origin_D = D,
         nu0_matrix = nu0_matrix, nu0_checksum = nu0_checksum, generated_at = string(now())))

    # Feasibility pre-check, same discipline as cm_production_stage_runner.jl.
    let D2_econ0 = length(w0) - length(eta0), xf0 = x_free_from_w(w0[1:D2_econ0], pe0)
        try
            pcx0 = build_originzc_production_context(ctx0, CS, layout)
            _, _, verify0 = cm_originzc_production_value_verified(xf0, exp.(eta0), pcx0)
            isfinite(verify0.Delta_dual) || error("Delta_dual is not finite at the (possibly perturbed) start point")
            lp(">>> start-point feasibility pre-check passed: Delta_dual=", verify0.Delta_dual)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            error("originzc_production_stage_runner: perturbed calibration start (chain_perturb_seed=$(CHAIN_PERTURB_SEED)) " *
                  "is not evaluable ($e) -- refusing to launch KNITRO on a point its own presolver would reject. " *
                  "Reduce the perturbation scale or use a different chain_perturb_seed.")
        end
    end
elseif MODE == "seed_w0"
    seed = deserialize(SEED_ARG)
    seed.delta == DELTA || lp(">>> NOTE: seed vector was cold-verified at delta=", seed.delta,
                               " (this stage is delta=", DELTA, ") -- expected cross-delta warm-start.")
    seed.W == W ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.W=$(seed.W) != this stage's W=$(W)")
    !hasproperty(seed, :draw_design) || seed.draw_design == DRAW_DESIGN ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.draw_design=$(seed.draw_design) != $(DRAW_DESIGN)")
    seed.draw_seed == DRAW_SEED ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.draw_seed=$(seed.draw_seed) != $(DRAW_SEED)")
    seed.distribution_restriction == DISTRIBUTION_RESTRICTION ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.distribution_restriction=$(seed.distribution_restriction) != $(DISTRIBUTION_RESTRICTION)")
    seed.origin_K_mean == ORIGIN_K_MEAN ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.origin_K_mean=$(seed.origin_K_mean) != $(ORIGIN_K_MEAN)")
    seed.origin_K_pair == ORIGIN_K_PAIR ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.origin_K_pair=$(seed.origin_K_pair) != $(ORIGIN_K_PAIR)")
    seed.power_target_layout == POWER_TARGET_LAYOUT ||
        error("originzc_production_stage_runner: seed-provenance mismatch: seed.power_target_layout=$(seed.power_target_layout) != $(POWER_TARGET_LAYOUT)")
    w0 = seed.w
    lp(">>> seeded from cold-verified prior-stage incumbent: ", SEED_ARG,
       " (prior Delta_dual=", seed.Delta_dual, ") -- provenance validated")
elseif MODE == "resume"
    resume_from = SEED_ARG
    isfile(resume_from) || error("originzc_production_stage_runner: resume requested but no checkpoint at $resume_from")
    lp(">>> RESUMING this stage from ", resume_from)
else
    error("originzc_production_stage_runner: unknown mode $MODE (expected calibration|seed_w0|resume)")
end

res = run_originzc_upper_checkpointed(resume_from === nothing ? w0 : nothing;
    W = W, delta = DELTA, draw_design = DRAW_DESIGN, draw_seed = DRAW_SEED,
    maxtime_real = BUDGET, ckpt_dir = CKPT_DIR, run_id = "originzc_campaign_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    label = "stage", checkpoint_interval_s = 30.0, resume_from = resume_from,
    cm_gradient_backend = CM_GRADIENT_BACKEND, allow_backend_switch = CM_ALLOW_BACKEND_SWITCH,
    distribution_restriction = DISTRIBUTION_RESTRICTION, K_mean = ORIGIN_K_MEAN, K_pair = ORIGIN_K_PAIR,
    power_target_layout = POWER_TARGET_LAYOUT, meanzc_basis = MEANZC_BASIS)

lp(">>> STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall, digits = 1),
   " n_eval=", res.n_eval, " n_grad=", res.n_grad,
   " best=", res.best === nothing ? "nothing" : "gp=$(res.best.gp) Delta=$(res.best.Delta)",
   " kappa=", res.kappa)
lp(">>> checkpoint: ", res.ckpt_path)
lp(">>> STAGE_DONE")   # sentinel line the supervisor greps for to distinguish "finished" from "hung/killed"
