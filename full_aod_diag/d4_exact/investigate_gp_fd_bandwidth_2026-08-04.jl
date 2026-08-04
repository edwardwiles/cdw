# 2026-08-04 continuation: investigate the gp-coordinate outer-gradient discrepancy found by
# test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl (unrestricted family, real D20/W80000).
# That gate's ground_truth_fd uses a FULL RE-SOLVE FD (re-optimizes zeta,lambda at each perturbed
# point via resolve_delta->evaluate_profiled_point) with a SINGLE FIXED h=0.01 for every coordinate,
# including gp. Hypothesis: gp's own natural scale/curvature makes h=0.01 far too large -- either
# a plain FD-bandwidth mismatch (feedback-fd-bandwidth-mismatch-looks-like-a-bug), or the h=0.01
# step is crossing a genuine winner-switch/support kink along gp specifically (this codebase's own
# documented non-smoothness pattern, e.g. feedback-melitz-tail-statistic-is-exact-step-function's
# analogue on the EK/Ricardo side). Sweep h to see whether the FD slope converges toward the
# analytic value (bandwidth artifact) or stays pinned near 5204 (real analytic-formula bug).
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
using LinearAlgebra, Printf

println("="^90); println("Building real D=20 :exclude_row context at W=80000 ..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, destination_sample = :exclude_row)
D = ctx.D; Ddest = ctx.D_dest
korea_idx = 14; brazil_idx = 3
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(korea_idx => brazil_idx))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)

function resolve_delta(w::AbstractVector{Float64})
    ev = evaluate_profiled_point(w, ctx, spec, pe)
    return ev.result.Delta_dual, ev
end

D0, ev0 = resolve_delta(w_calib)
g_prof, meta = profiled_composite_gradient_at(w_calib, ctx, spec, pe, ev0)
println("base Delta_dual=$D0  analytic gp gradient g_prof[1]=$(g_prof[1])")
flush(stdout)

println("\n", "="^90)
println("Sweeping FD step h for gp coordinate (full re-solve FD, central difference):")
println("="^90)
for h in [0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    wp = copy(w_calib); wp[1] += h
    wm = copy(w_calib); wm[1] -= h
    t0 = time()
    Dp, evp = resolve_delta(wp)
    Dm, evm = resolve_delta(wm)
    fd = (Dp - Dm) / (2h)
    @printf("  h=%.1e  Delta(+h)=%.10f (status=%d)  Delta(-h)=%.10f (status=%d)  FD=%+.6e  wall=%.1fs\n",
        h, Dp, evp.result.inner_status, Dm, evm.result.inner_status, fd, time() - t0)
    flush(stdout)
end
println("\nanalytic gp gradient (for comparison) = $(g_prof[1])")
