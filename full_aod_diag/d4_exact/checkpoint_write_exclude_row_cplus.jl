# ============================================================================
# CM checkpoint round trip, cm_gradient_backend=:cplus, destination_sample=:exclude_row.
# Write phase: run a short, real multi-iteration KNITRO campaign under the REAL
# production :cplus backend (not :reference), write a CMCheckpointV6.
# checkpoint_resume_exclude_row_cplus.jl resumes from this output in a genuinely
# separate process (mirrors checkpoint_write_exclude_row.jl's own :reference
# convention, now for :cplus after the square-only fix + validation this pass).
# ============================================================================
const D4X = "/bbkinghome/edav/gravity_robustness/release-fullA-omit-row-restore-screens-2026-07-23/full_aod_diag/d4_exact"
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Printf
println("=== includes OK ==="); flush(stdout)

CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_excluderow_cplus_smoke_test")
mkpath(CKPT_DIR)

snaps10 = nested_grid_sequence([10])[10]

ctx0 = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
w0 = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0))
println("calibration w0 built: D=", D, " D_dest=", Ddest, " length(w0)=", length(w0)); flush(stdout)

println("=== WRITE RUN: run_cm_upper_checkpointed(destination_sample=:exclude_row, cm_gradient_backend=:cplus) ===")
flush(stdout)
res = run_cm_upper_checkpointed(w0; W = 80000, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 10, contrasts = :anchored, probs = snaps10, cm_gradient_backend = :cplus,
    destination_sample = :exclude_row,
    ckpt_dir = CKPT_DIR, run_id = "excluderow_cplus_smoke", label = "cm_excl_cplus_smoke",
    maxtime_real = 90.0, checkpoint_interval_s = 15.0)

@printf("WRITE run: knitro_status=%d n_eval=%d n_grad=%d wall=%.1fs kappa=%s\n",
    res.knitro_status, res.n_eval, res.n_grad, res.wall, string(res.kappa))
println("best: ", res.best === nothing ? "nothing" : (gp = res.best.gp, Delta = res.best.Delta))
println("checkpoint written at: ", res.ckpt_path)
println("=== WRITE RUN COMPLETE (n_eval=", res.n_eval, " n_grad=", res.n_grad, ") ===")
