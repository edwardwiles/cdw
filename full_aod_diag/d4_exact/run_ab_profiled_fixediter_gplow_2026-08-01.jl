# ============================================================================
# Claude Code task 2026-08-01, follow-up: matched A/B, fixed outer-iteration
# budget, NON-TRIVIAL start (gp=0.99*calibration). PROFILED arm, using the
# adaptive-bandwidth O(1)-incremental gradient (profiled_composite_gradient_
# at_incremental, now matching production's select_bandwidth method exactly
# per live user request -- no fixed global h).
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
using Printf

const W = 80_000
const DELTA = 1.0
const MAXIT = parse(Int, get(ENV, "AB_MAXIT", "60"))
const SAFETY_BUDGET = 3600.0
const GP_FRAC = parse(Float64, get(ENV, "AB_GP_FRAC", "0.99"))
const TAG = replace(@sprintf("gp%.3f", GP_FRAC), "." => "p")

ctx = d20_real_setup(W = W, find_smallest = true, δ = DELTA, destination_sample = :exclude_row)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
w_start = reduce_calibration_to_w_profiled(ctx, pe)
w_start[1] *= GP_FRAC

println("="^90); println("PROFILED ARM (fixed-iteration, gp=$(GP_FRAC)*calib, tag=$TAG, ADAPTIVE-bandwidth incremental gradient): maxit=$MAXIT, gp_start=$(w_start[1])"); println("="^90); flush(stdout)

TRACE_CSV = joinpath(D4X_ROOT, "FULL_VS_PROFILED_OUTER_AB_FIXEDITER_2026-08-01_$(TAG)_PROFILED_TRACE.csv")
res = run_profiled_outer_search("profiled_fixediter_$TAG", w_start; ctx = ctx, spec = spec, pe = pe,
    maxtime_real = SAFETY_BUDGET, maxit_override = MAXIT, trace_csv = TRACE_CSV,
    gradient_fn = profiled_composite_gradient_at_incremental)

println("="^90)
@printf("RESULT profiled_fixediter_%s: wall=%.1fs n_eval=%d n_grad_calls=%d\n", TAG, res.wall_ext, res.n_eval, res.n_grad_calls)
println("best=", res.best)
println("="^90)
