# ============================================================================
# Part 1/2 of the CM checkpoint schema resume test (allocation/cache cleanup task §11).
# Real D=20/W=80000/L=10 run_cm_upper_checkpointed, forced to stop early via a short
# maxtime_real (simulating an interruption), writing a CMCheckpoint. Run
# test_cm_checkpoint_resume.jl next, in a SEPARATE process, to verify the resume path.
# ============================================================================
# NOTE: was hardcoded to a DIFFERENT worktree's path (gravity-fullA-alloc-cache-cleanup) --
# fixed to @__DIR__ so this test exercises the files actually present in THIS worktree/checkout
# (found while verifying the AUD-04 cm_checkpoint.jl gate fix; see
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

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_smoke_test")
rm(CKPT_DIR; force = true, recursive = true); mkpath(CKPT_DIR)

W = 80000
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D
Aod_theta_natural = ctx0.θ0_up[ctx0.Aod_offset+1:ctx0.Aod_offset+D^2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe0)
gp0 = ctx0.θ0_up[3+D]
w0 = vcat(gp0 * 1.01, zfree0)

snaps10 = nested_grid_sequence([10])[10]

println("=== ORIGINAL RUN: run_cm_upper_checkpointed, short budget (forced early stop) ===")
res = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :sobol_randomized, draw_seed = 20260719,
    L = 10, contrasts = :anchored, probs = snaps10, maxtime_real = 45.0,
    ckpt_dir = CKPT_DIR, run_id = "smoke", label = "cm_smoke", checkpoint_interval_s = 10.0)
@printf("original: knitro_status=%d n_eval=%d n_grad=%d wall=%.1fs kappa=%s\n",
    res.knitro_status, res.n_eval, res.n_grad, res.wall, string(res.kappa))
println("ckpt_path=", res.ckpt_path)
