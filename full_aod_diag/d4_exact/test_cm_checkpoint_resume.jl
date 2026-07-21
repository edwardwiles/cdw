# ============================================================================
# Part 2/2 of the CM checkpoint schema resume test (allocation/cache cleanup task §11).
# Run test_cm_checkpoint_original.jl FIRST (writes the checkpoint this script resumes from),
# then run this script in a genuinely SEPARATE process (mirrors
# c10_prod_driver_smoke_resume.jl's own convention for the unrestricted path's schema-3
# checkpoint). Verifies: (1) the eval/grad counters continue from where the original run
# left off, not reset; (2) resume regenerates draws that checksum-match the checkpoint
# exactly; (3) the checkpoint's own recorded best-feasible Delta is reproduced by an
# independently-rebuilt context at the SAME point, to floating-point tolerance (an EXACT
# bit-match is not expected here -- unlike the unrestricted path's schema-3 test, this
# comparison crosses a cold-vs-warm-start KNITRO boundary between two separate solves of the
# inner CC dual problem; ~1e-14 relative agreement is the same order of noise this codebase's
# own threaded-vs-serial Hessian comparisons already treat as "not a real discrepancy" -- see
# docs/fullA_cm_parallel_production_handoff.md §4).
# ============================================================================
using Test

const D4X = "/bbkinghome/edav/gravity_robustness/gravity-fullA-alloc-cache-cleanup/full_aod_diag/d4_exact"
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
include(joinpath(D4X, "nested_quantile_grids.jl"))
include(joinpath(D4X, "cm_outer_driver.jl"))
include(joinpath(D4X, "cm_checkpoint.jl"))
using Printf

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_smoke_test")
latest_path = joinpath(CKPT_DIR, "cm_smoke_latest.jls")
@assert isfile(latest_path) "no checkpoint found at $latest_path -- run the original script first"

latest = load_cm_checkpoint(latest_path)
println("Loaded checkpoint: schema=", latest.schema, " n_eval=", latest.n_eval, " n_grad=", latest.n_grad,
        " cm_L=", latest.cm_L, " draw_design=", latest.draw_design, " draw_seed=", latest.draw_seed,
        " best_feasible=", latest.best_feasible === nothing ? "nothing" : latest.best_feasible.gp)
n_eval_before_resume = latest.n_eval

println("=== RESUME RUN (fresh process): run_cm_upper_checkpointed with resume_from ===")
res = run_cm_upper_checkpointed(; ckpt_dir = CKPT_DIR, run_id = "smoke", label = "cm_smoke",
    resume_from = latest_path, maxtime_real = 45.0, checkpoint_interval_s = 10.0)
@printf("resumed: knitro_status=%d n_eval=%d n_grad=%d wall=%.1fs kappa=%s\n",
    res.knitro_status, res.n_eval, res.n_grad, res.wall, string(res.kappa))
println("resumed best: ", res.best === nothing ? "nothing" : (gp = res.best.gp, Delta = res.best.Delta))

@testset "CM checkpoint schema-1 resume" begin
    @test res.n_eval > n_eval_before_resume   # eval counter CONTINUED, was not reset to 0/1
    @test res.knitro_status in (0, -401, -410)   # solved or a benign wall/iteration-limit stop, not a crash

    # Reproducibility check: regenerating the SAME context independently and re-verifying the
    # checkpoint's own best-feasible incumbent at THAT exact point.
    ctx_check = d20_real_setup_design(W = latest.W, δ = latest.delta, find_smallest = latest.find_smallest,
        draw_design = latest.draw_design, draw_seed = latest.draw_seed)
    checksum_ok = ctx_check.draw_meta.checksum_uniform == latest.draw_checksum_uniform &&
                  ctx_check.draw_meta.checksum_transformed == latest.draw_checksum_transformed
    @test checksum_ok   # exact string match -- draw regeneration must be bit-reproducible

    pcx_check = build_cm_production_context(ctx_check, CS; L = latest.cm_L, contrasts = latest.cm_contrasts, probs = latest.cm_probs)
    xf_check = x_free_from_w(latest.best_feasible.w, build_pivot_elimination(ctx_check))
    _, base_check = cm_production_value(xf_check, pcx_check)
    Delta_check = -base_check.ζstar
    @printf("  re-verified Delta at checkpoint's best_feasible.w: %.15f  (checkpoint recorded: %.15f, diff=%.2e)\n",
        Delta_check, latest.best_feasible.Delta, abs(Delta_check - latest.best_feasible.Delta))
    @test abs(Delta_check - latest.best_feasible.Delta) < 1e-9
end
println("All CM-checkpoint resume tests passed.")
