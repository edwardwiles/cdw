# Phase 9/production-dims gate (integration/profiled-all-five-production-closeout, 2026-08-02):
# GENUINE-COLD unrestricted solve at PRODUCTION dimensions (D=20, Ddest=19, W=100,000), via the
# zero-dense reduced/profiled operator inner-solve path (`evaluate_profiled_point`, the SAME
# machinery test_profiled_outer_gradient_gate_D20_W80000_2026-08-01.jl already gates at W=80,000).
# Unrestricted has no CM-grid/ZC restriction (economic moments only), so there is no L/K_mean/K_pair
# here. Must be launched as a FRESH `julia` process -- see run_coldsolve_flexcm_w100k_2026-08-02.jl's
# own header for why.
const D4X = @__DIR__
t0_total = time()
include(joinpath(D4X, "context.jl"))
include(joinpath(dirname(dirname(D4X)), "cc_algo", "active_layout.jl"))
include(joinpath(D4X, "compressed_moments.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "oracle_fast.jl"))
include(joinpath(D4X, "operator_psi_bundle.jl"))
include(joinpath(D4X, "compressed_live.jl"))
include(joinpath(D4X, "context_real_d20.jl"))
include(joinpath(D4X, "operator_verification.jl"))
include(joinpath(D4X, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(D4X, "gravity_elimination.jl"))
include(joinpath(D4X, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(D4X, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(D4X, "recover_full_a_2026-07-31.jl"))
include(joinpath(D4X, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(D4X, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(D4X, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(D4X, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(D4X, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(D4X, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(D4X, "profiled_outer_evaluator_2026-08-01.jl"))
using Printf, LinearAlgebra, Random
flush(stdout)
println("PID=", getpid(), "  include done at t=", round(time() - t0_total, digits = 1), "s"); flush(stdout)

const W_VAL = 100_000
const D_VAL, DDEST_VAL = 20, 19

println("="^90); println("GENUINE-COLD unrestricted solve, PRODUCTION DIMS: D=$D_VAL Ddest=$DDEST_VAL W=$W_VAL")
println("="^90); flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D; Ddest = ctx.D_dest
@assert D == D_VAL && Ddest == DDEST_VAL "expected D=$D_VAL/Ddest=$DDEST_VAL, got D=$D/Ddest=$Ddest"

korea_idx = 14; brazil_idx = 3
t_spec = @elapsed (spec, gauge, pe) = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(korea_idx => brazil_idx))
n_total = outer_dim_profiled(pe)
@printf("spec/pe build: %.2fs  n_total=%d  pivot_pos=%d\n", t_spec, n_total, pe.pivot_pos)
flush(stdout)
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
@assert length(w_calib) == n_total

reset_no_dense_g_counters!()
println("Starting COLD KNITRO inner solve ..."); flush(stdout)
t_solve = @elapsed ev = evaluate_profiled_point(w_calib, ctx, spec, pe)
r = ev.result
@printf("SOLVE: wall=%.2fs  nStatus=%d  Delta_dual=%.10g  Delta_primal=%.10g  n_fg=%d  n_hess=%d\n",
    t_solve, r.inner_status, r.Delta_dual, r.Delta_primal, r.n_fg_calls, r.n_hess_calls)
flush(stdout)

c_disp = NO_DENSE_G_COUNTERS[]
@printf("dense_economic_G_materializations=%d (0 expected)\n", c_disp.dense_economic_G_materializations)
flush(stdout)
println("full verify NamedTuple = ", r)

ok = r.inner_status == 0 && r.max_abs_moment_kkt_resid < 1e-4 && c_disp.dense_economic_G_materializations == 0
@printf("\nTOTAL WALL: %.2fs\n", time() - t0_total)
println("UNRESTRICTED_W100K_COLD_SOLVE_RESULT: ", ok ? "PASS" : "FAIL")
ok || exit(1)
