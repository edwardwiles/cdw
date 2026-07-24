# ============================================================================
# Standalone ARGS-driven CLI entry point for the unrestricted (no CM/ZC restriction) D=20
# production driver (c10_d20_production_driver.jl's run_profile_checkpointed), added for the
# exclude-ROW-destination production release (2026-07-24) Gate C -- the unrestricted family had
# no standalone process-launchable CLI before this (only in-process/library calls), unlike the
# CM/origin-ZC families' own cm_production_stage_runner.jl/originzc_production_stage_runner.jl,
# so it could not be exercised through the real process-group supervisor mechanics
# (scripts/cm_production_supervisor.sh's setsid/pgid launch) the other three families use.
#
# Seeding modes (ARGS[4]), matching cm_production_stage_runner.jl's own convention:
#   "calibration"  -- ARGS[5] unused. Starts from the real-data calibration point.
#   "resume"       -- ARGS[5] = path to an existing D20CheckpointV4 (this SAME stage's own
#                     "<label>_latest.jls").
#
# Usage:
#   julia --project=. unrestricted_stage_runner.jl <ckpt_dir> <delta> <budget_s> <mode> <seed_arg>
#
# Env overrides (same pattern as cm_production_stage_runner.jl):
#   DESTINATION_SAMPLE (all_legacy|exclude_row, default all_legacy -- see SCOPE NOTE below)
#   PRICE_CACHE_BACKEND (default cplus)
#   FIND_SMALLEST (1|0, default 1)
#
# ****************************************************************************************
# **** SCOPE NOTE (exclude-ROW-destination production release, 2026-07-24): UNLIKE every  ****
# **** other production entry point (CM/CM+ZC/origin-ZC all default to and are fully      ****
# **** validated under destination_sample=:exclude_row), the UNRESTRICTED family here      ****
# **** defaults to :all_legacy and HARD-ERRORS on :exclude_row. Its real per-point          ****
# **** evaluation path (moment_representation=:compressed, compressed_moments.jl's          ****
# **** CompressedFactual + fast_range_screen.jl) is square-D-only throughout and was        ****
# **** never rectangularized -- this driver's own "Part A" was never touched by the         ****
# **** validated omit-ROW work at all. See run_profile_checkpointed's own guard             ****
# **** (c10_d20_production_driver.jl) and EXCLUDE_ROW_DESTINATION_PRODUCTION_RELEASE_       ****
# **** 2026-07-24.md for the full rationale. Documented, scoped gap -- not silent.          ****
# ****************************************************************************************
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

const CKPT_DIR = abspath(ARGS[1])
const DELTA = parse(Float64, ARGS[2])
const BUDGET = parse(Float64, ARGS[3])
const MODE = ARGS[4]
const SEED_ARG = length(ARGS) >= 5 ? ARGS[5] : nothing
mkpath(CKPT_DIR)

const DESTINATION_SAMPLE = Symbol(get(ENV, "DESTINATION_SAMPLE", "all_legacy"))
DESTINATION_SAMPLE === :all_legacy ||
    error("unrestricted_stage_runner: DESTINATION_SAMPLE=:$DESTINATION_SAMPLE requested, but the UNRESTRICTED " *
          "family only supports :all_legacy -- see this file's own SCOPE NOTE header and " *
          "run_profile_checkpointed's identical guard (c10_d20_production_driver.jl) for the full rationale.")
const PRICE_CACHE_BACKEND = Symbol(get(ENV, "PRICE_CACHE_BACKEND", "cplus"))
const FIND_SMALLEST = get(ENV, "FIND_SMALLEST", "1") == "1"
const W = 80_000
const DRAW_SEED = 20260719

lp(">>> Julia threads: ", Threads.nthreads(), "  mode=", MODE, " delta=", DELTA, " budget=", BUDGET,
   "s ckpt_dir=", CKPT_DIR, " destination_sample=", DESTINATION_SAMPLE,
   " (unrestricted family: :all_legacy only -- :exclude_row unsupported, see SCOPE NOTE)",
   " price_cache_backend=", PRICE_CACHE_BACKEND, " find_smallest=", FIND_SMALLEST)

resume_from = nothing
local g0, zfree0

if MODE == "calibration"
    ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = FIND_SMALLEST, destination_sample = DESTINATION_SAMPLE)
    print_active_layout_banner(ctx0, "unrestricted")
    pe0 = build_pivot_elimination(ctx0)
    D = ctx0.D; Ddest = ctx0.D_dest
    x_free_calib = ctx0.θ0_up[ctx0.free_idx]
    zfree0 = pivot_reduce(reshape(log.(x_free_calib[2:end]), D, Ddest), pe0)
    g0 = x_free_calib[1]
    lp(">>> calibration start: g=", g0, " ||zfree||=", norm(zfree0))
elseif MODE == "resume"
    resume_from = SEED_ARG
    isfile(resume_from) || error("unrestricted_stage_runner: resume requested but no checkpoint at $resume_from")
    lp(">>> RESUMING this stage from ", resume_from)
    g0 = 0.0; zfree0 = Float64[]   # unused when resume_from !== nothing (run_profile_checkpointed re-derives from the checkpoint)
else
    error("unrestricted_stage_runner: unknown mode $MODE (expected calibration|resume)")
end

res = run_profile_checkpointed("stage", g0, FIND_SMALLEST, zfree0;
    maxtime_real = BUDGET, W_in = W, delta_in = DELTA, draw_seed_in = DRAW_SEED,
    ckpt_dir = CKPT_DIR, checkpoint_interval_s = 15.0, resume_from = resume_from,
    price_cache_backend = PRICE_CACHE_BACKEND, destination_sample = DESTINATION_SAMPLE)

lp(">>> STAGE result: knitro_status=", res.knitro_status, " wall=", round(res.wall_ext, digits = 1),
   " n_eval=", res.n_eval,
   " best=", res.best === nothing ? "nothing" : "Delta_dual=$(res.best.Delta_dual) n_eval=$(res.best.n_eval)")
lp(">>> checkpoint: ", res.ckpt_path)
lp(">>> screens(pw/wt/wn/env/wr/sn/pass)=", res.screen_counts)
lp(">>> STAGE_DONE")
