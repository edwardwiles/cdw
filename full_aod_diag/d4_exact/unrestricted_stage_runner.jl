# ============================================================================
# Standalone ARGS-driven CLI entry point for the UNRESTRICTED (no CM/ZC restriction) D=20
# production driver.
#
# Remediation (production-audit task, 2026-07-26, Phase A): this script previously called the
# older `run_profile_checkpointed` (c10_d20_production_driver.jl), not the unified direct-bound
# driver `run_polish_checkpointed_unified` (c10_d20_production_driver_unified.jl) -- confirmed
# stale wiring, dated via `git log` to 2026-07-24 (this script) vs 2026-07-25 (the unified driver,
# which did not exist yet when this script was written). The old path has no transformed-A/
# flexible-theta coordinate architecture, is fixed to legacy z-space, and hardcodes KNITRO
# `algorithm=3` by default -- none of which match the other four families' own production
# defaults. New default campaigns now go through the unified driver. See
# docs/UNRESTRICTED_UNIFIED_DRIVER_RELEASE_2026-07-26.md for the equivalence-gate evidence
# (test_unrestricted_legacy_vs_unified_equivalence.jl: legacy-z decode/eval/gradient agree with
# the old driver's own decode to ~1e-9-1e-12 at the real D=20 calibration point, both directions,
# ALL PASS).
#
# Seeding modes (ARGS[4]):
#   "calibration"           -- ARGS[5] unused. Starts from the real-data calibration point,
#                              through the UNIFIED driver (new default path).
#   "resume"                -- ARGS[5] = path to an existing D20CheckpointUnified
#                              ("<label>_unified_latest.jls"). Refuses (with an explicit migration
#                              message) if the file is actually a legacy D20CheckpointV4 -- see
#                              "legacy_profile_resume" below for that case.
#   "legacy_profile_resume" -- ARGS[5] = path to an existing D20CheckpointV4 ("<label>_latest.jls").
#                              Runs the OLD run_profile_checkpointed driver, unchanged, SOLELY to
#                              finish a genuinely in-flight campaign that was launched before this
#                              fix. Does not build a unified-schema checkpoint. Do not use this mode
#                              for new campaigns.
#
# Usage:
#   julia --project=. unrestricted_stage_runner.jl <ckpt_dir> <delta> <budget_s> <mode> <seed_arg>
#
# Env overrides (same pattern as cm_production_stage_runner.jl):
#   DESTINATION_SAMPLE (exclude_row|all_legacy, default exclude_row)
#   PRICE_CACHE_BACKEND (default cplus)
#   FIND_SMALLEST (1|0, default 1)
#   A_COORDINATE_MODE (transformed_a|legacy_z, default transformed_a -- NEW production default,
#                       Phase F1. transformed_a maps to the driver's own :powered_aspace symbol.
#                       legacy_z remains a fully supported, explicit replication option.)
#   TRADE_ELASTICITY_MODE (fixed|flexible, default fixed -- flexible theta is a separate overlay,
#                       see PRODUCTION_FLEXIBLE_THETA_OVERLAY docs, not this script's default path)
#
# ****************************************************************************************
# **** SCOPE NOTE (exclude-ROW-destination UNRESTRICTED-CORE release, 2026-07-24, carried    ****
# **** forward unchanged by this fix): the unrestricted family defaults to :exclude_row,      ****
# **** matching every other production entry point (CM/CM+ZC/origin-ZC).                     ****
# ****************************************************************************************
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "c10_d20_production_driver_unified.jl"))
using Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const CKPT_DIR = abspath(ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const MODE = ARGS[4]
const SEED_ARG = length(ARGS) >= 5 ? ARGS[5] : nothing
mkpath(CKPT_DIR)

const DESTINATION_SAMPLE = Symbol(get(ENV, "DESTINATION_SAMPLE", "exclude_row"))
DESTINATION_SAMPLE in (:exclude_row, :all_legacy) ||
    error("unrestricted_stage_runner: DESTINATION_SAMPLE=:$DESTINATION_SAMPLE requested, must be " *
          ":exclude_row or :all_legacy.")
const PRICE_CACHE_BACKEND = Symbol(get(ENV, "PRICE_CACHE_BACKEND", "cplus"))
const FIND_SMALLEST = get(ENV, "FIND_SMALLEST", "1") == "1"
const A_COORDINATE_MODE_ARG = get(ENV, "A_COORDINATE_MODE", "transformed_a")
const A_COORDINATE_MODE = A_COORDINATE_MODE_ARG == "transformed_a" ? :powered_aspace :
                           A_COORDINATE_MODE_ARG == "legacy_z" ? :legacy_z :
                           error("unrestricted_stage_runner: A_COORDINATE_MODE must be transformed_a|legacy_z, got $A_COORDINATE_MODE_ARG")
const TRADE_ELASTICITY_MODE = Symbol(get(ENV, "TRADE_ELASTICITY_MODE", "fixed"))
TRADE_ELASTICITY_MODE in (:fixed, :flexible) ||
    error("unrestricted_stage_runner: TRADE_ELASTICITY_MODE must be fixed|flexible, got :$TRADE_ELASTICITY_MODE")
const W = 80_000
const DRAW_SEED = 20260719

"Startup diagnostics (production-audit task A2/F2): identify exactly what this launch will run."
function print_startup_diagnostics(; public_driver::String, outer_problem_type::String,
        a_mode::Symbol, trade_mode::Symbol, outer_algorithm::String, checkpoint_schema)
    lp(">>> DIAGNOSTICS unrestricted_public_driver=", public_driver)
    lp(">>> DIAGNOSTICS outer_problem_type=", outer_problem_type)
    lp(">>> DIAGNOSTICS A_coordinate_mode=", a_mode)
    lp(">>> DIAGNOSTICS trade_elasticity_mode=", trade_mode)
    lp(">>> DIAGNOSTICS outer_algorithm=", outer_algorithm)
    lp(">>> DIAGNOSTICS checkpoint_schema=", checkpoint_schema)
end

const OUTER_PROBLEM_TYPE = FIND_SMALLEST ?
    "direct_constrained_bound: minimize g_p s.t. Delta*(A,g_p) <= delta (find_smallest=true)" :
    "direct_constrained_bound: maximize g_p s.t. Delta*(A,g_p) <= delta (find_smallest=false)"

if MODE == "legacy_profile_resume"
    # ------------------------------------------------------------------------------------------
    # LEGACY PATH -- unchanged run_profile_checkpointed, retained solely to finish a genuinely
    # in-flight pre-fix campaign. Not the default; must be explicitly requested.
    # ------------------------------------------------------------------------------------------
    print_startup_diagnostics(public_driver = "run_profile_checkpointed (LEGACY -- explicit opt-in only)",
        outer_problem_type = OUTER_PROBLEM_TYPE, a_mode = :legacy_z, trade_mode = :fixed,
        outer_algorithm = "algorithm=3 (Active-Set/SLQP, hardcoded default of this legacy driver)",
        checkpoint_schema = CHECKPOINT_SCHEMA_UNRESTRICTED)
    resume_from = SEED_ARG
    isfile(resume_from) || error("unrestricted_stage_runner(legacy_profile_resume): no checkpoint at $resume_from")
    lp(">>> LEGACY RESUME from ", resume_from, " under run_profile_checkpointed (V4 schema)")
    res = run_profile_checkpointed("stage", 0.0, FIND_SMALLEST, Float64[];
        maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 15.0, resume_from = resume_from,
        price_cache_backend = PRICE_CACHE_BACKEND, destination_sample = DESTINATION_SAMPLE)
    lp(">>> LEGACY STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
       " n_eval=", res.n_eval)
    lp(">>> checkpoint: ", res.ckpt_path)
    print_core_hessian_counters()
    lp(">>> STAGE_DONE (legacy_profile_resume)")
    exit(0)
end

# ------------------------------------------------------------------------------------------------
# DEFAULT PATH -- unified direct-bound driver (run_polish_checkpointed_unified). New campaigns
# default to A_coordinate_mode=:powered_aspace ("transformed_a"), trade_elasticity_mode=:fixed.
# ------------------------------------------------------------------------------------------------
layout = make_layout(trade_elasticity_mode = TRADE_ELASTICITY_MODE, A_coordinate_mode = A_COORDINATE_MODE, gp_coordinate_mode = :raw)

print_startup_diagnostics(public_driver = "run_polish_checkpointed_unified",
    outer_problem_type = OUTER_PROBLEM_TYPE, a_mode = A_COORDINATE_MODE, trade_mode = TRADE_ELASTICITY_MODE,
    outer_algorithm = "auto (csw_outer_wallclock_sr1.opt default; pin_outer_algorithm=false) -- see Phase G for live-resolved value",
    checkpoint_schema = CHECKPOINT_SCHEMA_UNIFIED)

lp(">>> Julia threads: ", Threads.nthreads(), "  mode=", MODE, " delta=", DELTA, " budget=", BUDGET,
   "s ckpt_dir=", CKPT_DIR, " destination_sample=", DESTINATION_SAMPLE,
   " price_cache_backend=", PRICE_CACHE_BACKEND, " find_smallest=", FIND_SMALLEST,
   " A_coordinate_mode=", A_COORDINATE_MODE, " trade_elasticity_mode=", TRADE_ELASTICITY_MODE)

resume_from = nothing
local w_start, theta_lo_arg, theta_hi_arg

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = DESTINATION_SAMPLE)
print_active_layout_banner(ctx0, "unified_$(layout.trade_elasticity_mode)_$(layout.A_coordinate_mode)")
theta_star = 1.0 / ctx0.μHat
D = ctx0.D; Ddest = ctx0.D_dest
xy = precompute_aspace_XY(ctx0)

if TRADE_ELASTICITY_MODE == :flexible
    sigma = ctx0.σ
    theta_min = 2 * (sigma - 1) * 1.05
    theta_max = 3 * theta_star
    pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
    theta_lo_arg, theta_hi_arg = theta_min, theta_max
else
    pgc = build_pivot_elimination_cheap(ctx0; mu_probe1 = 1.0 / theta_star * 0.999, mu_probe2 = 1.0 / theta_star * 1.001)
    theta_lo_arg, theta_hi_arg = NaN, NaN
end

if MODE == "calibration"
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    gp0 = x_free_calib[1]
    logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
    w_start = reduce_to_w_unified(theta_star, gp0, logA_full0, pgc, xy, layout)
    lp(">>> calibration start: gp=", gp0, " ||w_start||=", norm(w_start))
elseif MODE == "resume"
    resume_from = SEED_ARG
    isfile(resume_from) || error("unrestricted_stage_runner: resume requested but no checkpoint at $resume_from")
    # w_start is REQUIRED (positional) but is a no-op placeholder whenever resume_from is set --
    # run_polish_checkpointed_unified reconstructs the real (g, zfree, logA_full) from the
    # checkpoint itself (confirmed via test_unified_checkpoint_resume.jl's own resume cases, which
    # pass the ORIGINAL w_start unchanged on resume and still get the checkpoint's own state back).
    w_start = zeros(outer_dim(layout, D, Ddest))
    lp(">>> RESUMING (unified schema) from ", resume_from)
else
    error("unrestricted_stage_runner: unknown mode $MODE (expected calibration|resume|legacy_profile_resume)")
end

# A2: refuse (with a precise migration message), do not silently reinterpret, a legacy schema-4
# checkpoint on resume. run_polish_checkpointed_unified's own resume path calls
# load_checkpoint_unified(resume_from) internally, which hard-errors on any schema/type mismatch;
# this wraps that call to add a clear, actionable message pointing at legacy_profile_resume.
# NOTE: `res` is pre-declared as a global here, then assigned via explicit `global res = ...`
# inside the try block below. This project's OWN documented top-level scoping gotcha
# (julia-toplevel-catch-scoping-gotcha) covers the `catch`-only-assignment variant; this is the
# sibling trap on the `try` side -- with a global `res` already in scope, a plain `res = ...`
# inside `try` is AMBIGUOUS soft-scope and Julia silently treats it as a NEW LOCAL shadowing the
# global (confirmed live: triggers `Assignment to res in soft scope is ambiguous` then a
# `FieldError: type Nothing has no field` below once the try block's local vanishes) -- `global`
# is required to actually update the outer binding.
res = nothing
try
    global res = run_polish_checkpointed_unified("stage", FIND_SMALLEST, w_start;
        layout = layout, theta_lo = theta_lo_arg, theta_hi = theta_hi_arg,
        maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
        ckpt_dir = CKPT_DIR, checkpoint_interval_s = 15.0, resume_from = resume_from,
        price_cache_backend = PRICE_CACHE_BACKEND, destination_sample = DESTINATION_SAMPLE)
catch e
    if resume_from !== nothing
        lp("!!! MIGRATION ERROR: ", resume_from, " could not be resumed as a D20CheckpointUnified.")
        lp("!!! This is very likely a LEGACY D20CheckpointV4 checkpoint (schema=", CHECKPOINT_SCHEMA_UNRESTRICTED,
           ") written by the OLD run_profile_checkpointed driver, before this fix.")
        lp("!!! Old and new checkpoints are NOT interchangeable -- this runner does not pad, truncate, or")
        lp("!!! silently convert between them. To finish this specific in-flight campaign under the old")
        lp("!!! scientific driver, rerun with MODE=legacy_profile_resume and the same checkpoint path.")
        lp("!!! To start a fresh campaign under the new unified driver, rerun with MODE=calibration.")
        lp("!!! Original error: ", sprint(showerror, e))
    end
    rethrow()
end

lp(">>> STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
   " n_eval=", res.n_eval, " kappa=", res.kappa,
   " best=", res.best_feasible === nothing ? "nothing" : "gp=$(res.best_feasible.gp) Delta=$(res.best_feasible.Delta)")
lp(">>> checkpoint: ", res.final_checkpoint)
print_core_hessian_counters()   # prove the shared H_EE backend actually ran, not just that it was requested
lp(">>> screens(pw/wt/wn/env/wr/sn/pass)=", res.screen_counts)
lp(">>> STAGE_DONE")
