#!/usr/bin/env julia
# Production-consolidation Phase 8 (2026-07-31): THE canonical public Melitz production
# launcher. Every real Melitz production/campaign script should `include` this file FIRST
# (before building any fixture, before touching KNITRO) and call `melitz_production_preflight!`
# with its own real_data_dir/option_files -- this is deliberately a thin, stable, rarely-
# changing entry point (the actual preflight logic lives in
# `src/melitz/production_launcher.jl`, included via `include_melitz.jl` like every other
# Melitz source file -- this script just demonstrates/documents the required call sequence
# and provides a runnable standalone smoke-check).
#
# USAGE (as a library, from a real campaign script):
#   include(joinpath(REPO, "misc", "doubleDiff.jl"))
#   include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
#   using KNITRO
#   manifest = melitz_production_preflight!(REPO; knitro_module=KNITRO,
#       real_data_dir=joinpath(REPO, "real_data", "noah_D20"),
#       option_files=[joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")],
#       write_manifest_to=joinpath(checkpoint_dir, "production_manifest.txt"))
#   # ... only after this returns successfully does the campaign build a fixture / call KNITRO.
#
# USAGE (standalone smoke-check, this file directly):
#   julia --project=. scripts/melitz_production_launcher.jl [--allow-dirty]
#
# Exits with code 0 on a passed preflight (prints the manifest), code 1 on
# `MelitzLaunchRefusal` (prints the refusal reason, does NOT stack-trace -- a refusal is an
# expected, informative outcome, not a crash), code 2 on any other unexpected error.

REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using KNITRO

allow_dirty = "--allow-dirty" in ARGS

try
    manifest = melitz_production_preflight!(REPO; allow_dirty=allow_dirty, knitro_module=KNITRO,
        real_data_dir=joinpath(REPO, "real_data", "noah_D20"),
        option_files=filter(isfile, [
            joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt"),
            joinpath(REPO, "melitz_inner_loop_options.opt"),
        ]))
    println()
    println(melitz_manifest_to_text(manifest))
    exit(0)
catch e
    if isa(e, MelitzLaunchRefusal)
        println(stderr, "\nLAUNCH REFUSED: ", e.reason)
        exit(1)
    else
        println(stderr, "\nUNEXPECTED ERROR during preflight (not a normal refusal):")
        showerror(stderr, e, catch_backtrace())
        println(stderr)
        exit(2)
    end
end
