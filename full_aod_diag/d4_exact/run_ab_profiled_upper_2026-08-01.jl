# ============================================================================
# Claude Code task 2026-08-01, §14: matched A/B, PROFILED arm, upper bound,
# delta=1. Uses run_profiled_outer_search (profiled_outer_ab_harness_2026-08-01.jl)
# -- same outer-KNITRO configuration as run_profile_checkpointed
# (csw_outer_wallclock_sr1.opt, algorithm=3, z_halfwidth=30), same W/delta/
# start point (reduced from the SAME calibration point the full arm starts
# from). No screens at all (evaluate_profiled_point never screens).
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
include(joinpath(@__DIR__, "profiled_outer_ab_harness_2026-08-01.jl"))
using Printf

const W = 80_000
const DELTA = 1.0
const BUDGET = 1800.0   # 30 minutes

ctx = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
w_start = reduce_calibration_to_w_profiled(ctx, pe)

println("="^90); println("PROFILED ARM: upper bound, delta=1, W=$W, budget=$(BUDGET)s, n_total=$(length(w_start))"); println("="^90); flush(stdout)

TRACE_CSV = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_UPPER_DELTA1_2026-08-01_PROFILED_TRACE.csv")
res = run_profiled_outer_search("profiled_upper", w_start; ctx = ctx, spec = spec, pe = pe,
    maxtime_real = BUDGET, trace_csv = TRACE_CSV)

println("="^90)
@printf("RESULT profiled_upper: wall=%.1fs n_eval=%d n_grad_calls=%d\n", res.wall_ext, res.n_eval, res.n_grad_calls)
println("best=", res.best)
println("="^90)
