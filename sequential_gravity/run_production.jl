# ============================================================================
# Unified production entry point. METHOD env var selects which of the 4
# head-to-head-validated methods to run (see
# head_to_head/FINAL_REPORT_2026-07-16.md for the comparison that established
# LC as the default): LC (local-constrained/KNITRO) wins every tested target
# for kappa once the recover_lfd nStatus bug (see below) is fixed, and is
# dramatically cheaper than the global (BlackBoxOptim) methods per unit of
# result quality -- so LC is the default when METHOD is unset.
#
#   METHOD=LC (default) julia --project=. sequential_gravity/run_production.jl
#       -> runs run_profiled_production.jl's own batch loop: BOTH bounds
#          (lower/upper) across DELTA_GRID (env var, comma-separated,
#          e.g. "0.1,1.0,2.0,5.0"), single start (theta_r0/Astar), warm-start
#          CHAINED across delta within each bound direction. This is the
#          general-purpose, delta-grid-driven production driver.
#   METHOD=LU / GC / GU julia --project=. sequential_gravity/run_production.jl
#       -> delegates to head_to_head/run_lu.jl / run_gc.jl / run_gu.jl.
#          IMPORTANT ASYMMETRY: these are NOT (yet) generic delta-grid
#          drivers like LC's own batch loop -- they run the SPECIFIC 3-target
#          (T1/T2/T3), 3-start head-to-head design those files were built for
#          (see their own headers). DELTA_GRID/BOUND_ARG env vars below are
#          NOT read by these paths. Unifying them onto run_profiled_production
#          .jl's generic interface is real, not-yet-done work -- see the
#          2026-07-16 handoff doc for why this was deliberately deferred
#          rather than rushed.
#
# All 4 paths share the same core machinery (seq_gravcol, the recover_lfd
# nStatus fix, DUAL_WARM_MODE persist-by-default dual warm-starting, the
# Optim.jl NewtonTrustRegion destination-inversion solver, and -- for LC --
# hardmax post-verification) via their own `include(run_profiled_production
# .jl)` calls; this file does not duplicate any of that, it only dispatches.
# ============================================================================

const METHOD = uppercase(get(ENV, "METHOD", "LC"))

if METHOD == "LC"
    include(joinpath(@__DIR__, "run_profiled_production.jl"))
elseif METHOD == "LU"
    include(joinpath(@__DIR__, "head_to_head", "run_lu.jl"))
elseif METHOD == "GC"
    include(joinpath(@__DIR__, "head_to_head", "run_gc.jl"))
elseif METHOD == "GU"
    include(joinpath(@__DIR__, "head_to_head", "run_gu.jl"))
else
    error("Unknown METHOD=$METHOD. Valid options: LC (default, local-constrained/KNITRO), " *
          "LU (local-unconstrained), GC (global-constrained/BlackBoxOptim), GU (global-unconstrained).")
end
