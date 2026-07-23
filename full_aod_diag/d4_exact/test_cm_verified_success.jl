# ============================================================================
# AUD-04 CM-checkpoint gap fix regression test.
#
# docs/fullA_independent_audit_remediation.md (AUD-04) and
# docs/fullA_postmerge_allocation_productionization.md (section 1, "Bugs found") both flagged
# that `cm_checkpoint.jl`'s `run_cm_upper_checkpointed` had NO equivalent of `oracle.jl`'s
# `classify_inner_result`/`is_verified_success` gate: `is_new_best` only checked
# `isfinite(Delta) && Delta <= delta + 1e-6`, and the final `:stage_complete` checkpoint was
# written unconditionally, both status-only checks with no independent residual/gap
# verification. `cm_production_value`/`archC_base_state` returned a bare `BaseDualState` with no
# `primal_dual_gap`/`weight_norm_resid`/`mean_m_resid`/`max_abs_moment_kkt_resid`/`m_min` fields
# for `classify_inner_result` to read.
#
# Fixed (this branch): `archC_verified_state`/`cm_production_value_verified`
# (cm_production_bundle.jl) compute those fields for the Architecture-C CM path, reusing
# `primal_divergence` (oracle.jl) and `kkt_residual_blas` (oracle_fast.jl) rather than
# re-deriving either formula. `cm_checkpoint.jl`'s `cb_F!` now gates `is_new_best` on
# `is_verified_success`, and the final checkpoint falls back to `:stage_complete_unverified`
# (not `:stage_complete`) when the terminal point fails verification -- same AUD-10 pattern
# `c10_d20_production_driver.jl` already uses.
#
# This test verifies, with REAL KNITRO/D=20/W=80000 solves (no synthetic/mocked inner solve):
#   1. `archC_verified_state`/`cm_production_value_verified` return a `verify` NamedTuple that
#      `classify_inner_result`/`is_verified_success` can consume, and that a genuinely converged
#      real-data calibration point classifies as `VerifiedSolved`.
#   2. `archC_verified_state`'s explicit-recompute path and `archC_base_state`'s
#      KN-solve-last-call-trust path agree on (zeta*, inner_status) at the SAME point -- the two
#      code paths must not silently diverge.
#   3. `run_cm_upper_checkpointed`'s gate is wired end-to-end: every trace entry carries a
#      `verified` flag, and any point that became `best_feasible` re-verifies as
#      `VerifiedSolved` when independently re-checked outside the callback.
# ============================================================================
using Test

# NOTE: earlier versions of test_cm_checkpoint_original.jl/test_cm_checkpoint_resume.jl
# hardcoded D4X to a DIFFERENT worktree's absolute path (gravity-fullA-alloc-cache-cleanup) --
# fixed there and here to @__DIR__ so these tests exercise the files actually present in THIS
# worktree/checkout (found while building this test; see
# docs/fullA_independent_audit_remediation.md AUD-04 CM follow-up).
const D4X = @__DIR__
include(joinpath(D4X, "draw_design.jl"))
include(joinpath(D4X, "winners.jl"))
include(joinpath(D4X, "oracle.jl"))
include(joinpath(D4X, "common_marginals_moments.jl"))
include(joinpath(D4X, "common_marginals_interval.jl"))
include(joinpath(D4X, "instrumentation.jl"))
include(joinpath(D4X, "oracle_fast.jl"))
include(joinpath(D4X, "gravity_elimination.jl"))
include(joinpath(D4X, "three_way_derivatives.jl"))
include(joinpath(D4X, "lfix_incremental.jl"))
include(joinpath(D4X, "composite_gradient.jl"))
include(joinpath(D4X, "composite_gradient_fast.jl"))
include(joinpath(D4X, "cm_lookup_kernels.jl"))
include(joinpath(D4X, "lfix_cm_aware.jl"))
include(joinpath(D4X, "cm_hessian_architectures.jl"))
include(joinpath(D4X, "cm_production_bundle.jl"))
include(joinpath(D4X, "lfix_cm_cplus.jl"))   # CM-C+ production integration 2026-07-23: cm_gradient_backend now defaults to :cplus in run_cm_upper_checkpointed, so this must always be on the include path
include(joinpath(D4X, "nested_quantile_grids.jl"))
include(joinpath(D4X, "cm_outer_driver.jl"))
include(joinpath(D4X, "cm_checkpoint.jl"))
using Printf

W = 80000
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
Aod_theta_natural = ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D^2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe0)
gp0 = ctx0.θ0_up[3+D]
# Same 1.01 nudge test_cm_checkpoint_original.jl uses for its own w0: strictly interior/feasible,
# not sitting exactly on a boundary.
w0 = vcat(gp0 * 1.01, zfree0)
x_free_calib = x_free_from_w(w0, pe0)

snaps10 = nested_grid_sequence([10])[10]
pcx = build_cm_production_context(ctx0, CS; L = 10, contrasts = :anchored, probs = snaps10)

@testset "AUD-04 CM verified-success gate" begin
    println("=== (1) archC_verified_state at the real-data calibration point ===")
    base, verify = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
    @printf("  inner_status=%d Delta_dual=%.6g primal_dual_gap=%.3e weight_norm_resid=%.3e mean_m_resid=%.3e max_abs_moment_kkt_resid=%.3e m_min=%.6g\n",
        verify.inner_status, verify.Delta_dual, verify.primal_dual_gap, verify.weight_norm_resid,
        verify.mean_m_resid, verify.max_abs_moment_kkt_resid, verify.m_min)
    cls = classify_inner_result(verify)
    println("  classify_inner_result => ", cls)
    @test cls == VerifiedSolved
    @test is_verified_success(verify)

    println("=== (2) archC_verified_state vs archC_base_state agreement at the SAME point ===")
    K_v, base_v, verify_v = cm_production_value_verified(x_free_calib, pcx)
    @test base_v.ζstar == base.ζstar
    @test base_v.inner_status == base.inner_status
    K_cheap, base_cheap = cm_production_value(x_free_calib, pcx)
    @test base_cheap.ζstar == base_v.ζstar
    @test base_cheap.inner_status == base_v.inner_status
    println("  base_cheap.ζstar=", base_cheap.ζstar, " base_v.ζstar=", base_v.ζstar, " (agree)")

    println("=== (3) run_cm_upper_checkpointed short forced-early-stop run, gate wired into cb_F! ===")
    CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_verified_gate_test")
    rm(CKPT_DIR; force = true, recursive = true); mkpath(CKPT_DIR)
    res = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
        L = 10, contrasts = :anchored, probs = snaps10, maxtime_real = 45.0,
        ckpt_dir = CKPT_DIR, run_id = "verified_gate_smoke", label = "cm_vgate", checkpoint_interval_s = 10.0)
    @printf("  knitro_status=%d n_eval=%d n_grad=%d wall=%.1fs kappa=%s\n",
        res.knitro_status, res.n_eval, res.n_grad, res.wall, string(res.kappa))
    @test !isempty(res.trace)
    @test all(t -> haskey(t, :verified), res.trace)
    verified_evals = [t for t in res.trace if t.verified]
    println("  n_eval=", length(res.trace), " n_verified=", length(verified_evals))
    @test !isempty(verified_evals)   # this real calibration-seeded run should find at least one verified point

    final_ckpt = load_cm_checkpoint(res.ckpt_path)
    println("  final checkpoint_reason=", final_ckpt.checkpoint_reason)
    @test final_ckpt.checkpoint_reason in (:stage_complete, :stage_complete_unverified, :new_best, :wall_interval)

    if res.best !== nothing
        # best_feasible was only ever set when verified=true inside cb_F! (AUD-04 gate) --
        # re-verify it independently here, outside the callback, as a genuinely separate check.
        xf_best = x_free_from_w(res.best.w, pe0)
        _, _, verify_best = cm_production_value_verified(xf_best, pcx)
        println("  best_feasible re-verify classify_inner_result => ", classify_inner_result(verify_best))
        @test is_verified_success(verify_best)
    end
end
println("All AUD-04 CM verified-success gate tests passed.")
