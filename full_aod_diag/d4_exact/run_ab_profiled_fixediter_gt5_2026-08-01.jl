# ============================================================================
# Claude Code task 2026-08-01, follow-up: matched A/B at the GT=5% waypoint.
# PROFILED arm, seeded from the SAME economic point the continuation reached
# (converted from the full/reference pivot's zfree/gp into the profiled
# w_profiled=[gp;r_free] representation via pivot_expand + reduce_to_w_profiled
# -- both arms start from the economically identical point, task's own
# "same initial economic point" requirement). Uses the FIXED
# run_profiled_outer_search (gp genuinely held constant, 2026-08-01 bugfix)
# and the adaptive-bandwidth O(1)-incremental gradient.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "profiled_lfix_incremental_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_ab_harness_2026-08-01.jl"))
using Printf, Serialization

const W = 80_000
const DELTA = 1.0
const MAXIT = parse(Int, get(ENV, "AB_MAXIT", "60"))
const SAFETY_BUDGET = 3600.0

waypoint = deserialize(joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "continuation_gt", "waypoint_GT5.jls"))
println("Loaded waypoint: gp=", waypoint.gp, "  kappa=", waypoint.kappa); flush(stdout)

ctx = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
spec, gauge, pe_profiled = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
pe_full = build_pivot_elimination(ctx)
D = ctx.D; Ddest = ctx.D_dest

# Convert the FULL formulation's (gp, zfree) waypoint into the SAME economic point in
# profiled coordinates: zfree -> full z (D x Ddest) via pivot_expand, then reduce via
# reduce_to_w_profiled (using the profiled system's own anchor+pivot, NOT the full's).
z_full_waypoint = pivot_expand(waypoint.zfree, pe_full)
w_start = reduce_to_w_profiled(waypoint.gp, z_full_waypoint, pe_profiled)
println("Converted to profiled coordinates: gp=", w_start[1], " (expect exactly ", waypoint.gp, ")  n_total=", length(w_start))
@assert abs(w_start[1] - waypoint.gp) < 1e-12 "gp round-trip mismatch converting to profiled coordinates"

println("="^90); println("PROFILED ARM (fixed-iteration, GT=5% waypoint, FIXED gp + adaptive-bandwidth incremental gradient): maxit=$MAXIT"); println("="^90); flush(stdout)

TRACE_CSV = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_gt5_PROFILED_TRACE.csv")
res = run_profiled_outer_search("profiled_fixediter_gt5", w_start; ctx = ctx, spec = spec, pe = pe_profiled,
    maxtime_real = SAFETY_BUDGET, maxit_override = MAXIT, trace_csv = TRACE_CSV,
    gradient_fn = profiled_composite_gradient_at_incremental)

println("="^90)
@printf("RESULT profiled_fixediter_gt5: wall=%.1fs n_eval=%d n_grad_calls=%d\n", res.wall_ext, res.n_eval, res.n_grad_calls)
println("best=", res.best)
println("="^90)
