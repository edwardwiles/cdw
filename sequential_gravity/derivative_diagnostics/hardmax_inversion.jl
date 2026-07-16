# `hardmax_invert_destination` and `verify_hardmax_point` now live in production
# (`sequential_gravity/hardmax_verify.jl`, wired into `run_profiled_production.jl`'s
# `run_one_bound`) rather than here, so there is a single source of truth. This file just
# re-exposes them for the diagnostics scripts in this directory that `include` it.
include(joinpath(@__DIR__, "..", "hardmax_verify.jl"))
