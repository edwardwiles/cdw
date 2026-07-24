# ============================================================================
# CM checkpoint round trip, cm_gradient_backend=:cplus, destination_sample=:exclude_row.
# Resume phase: resume from checkpoint_write_exclude_row_cplus.jl's output, in a
# genuinely SEPARATE julia process. Verifies: (1) schema/destination_sample/
# row_idx/D_dest recorded correctly; (2) eval/grad counters continue (not
# reset); (3) resume regenerates draws that checksum-match exactly; (4) the
# checkpoint's own recorded best-feasible incumbent is reproduced (bit-identical
# inner recovery) by an independently-rebuilt :exclude_row context at the SAME
# point, under the SAME :cplus backend.
# ============================================================================
using Test
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
latest_path = joinpath(CKPT_DIR, "cm_excl_cplus_smoke_latest.jls")
@assert isfile(latest_path) "no checkpoint found at $latest_path -- run checkpoint_write_exclude_row_cplus.jl first"

latest = load_cm_checkpoint(latest_path)
println("Loaded checkpoint: schema=", latest.schema, " destination_sample=", latest.destination_sample,
        " row_idx=", latest.row_idx, " D_dest=", latest.D_dest,
        " cm_gradient_backend=", latest.cm_gradient_backend,
        " n_eval=", latest.n_eval, " n_grad=", latest.n_grad, " draw_design=", latest.draw_design,
        " best_feasible=", latest.best_feasible === nothing ? "nothing" : latest.best_feasible.gp)
n_eval_before_resume = latest.n_eval

@testset "CMCheckpointV6 destination_sample provenance (:cplus)" begin
    @test latest.schema == CM_CHECKPOINT_SCHEMA
    @test latest.destination_sample == :exclude_row
    @test latest.row_idx == 20
    @test latest.D_dest == 19
    @test latest.cm_gradient_backend == :cplus
end

println("\n=== RESUME RUN (fresh process): run_cm_upper_checkpointed with resume_from, :cplus ===")
flush(stdout)
res = run_cm_upper_checkpointed(; ckpt_dir = CKPT_DIR, run_id = "excluderow_cplus_smoke_resume", label = "cm_excl_cplus_smoke",
    resume_from = latest_path, maxtime_real = 45.0, checkpoint_interval_s = 10.0,
    cm_gradient_backend = :cplus, destination_sample = :exclude_row)
@printf("resumed: knitro_status=%d n_eval=%d n_grad=%d wall=%.1fs kappa=%s\n",
    res.knitro_status, res.n_eval, res.n_grad, res.wall, string(res.kappa))
println("resumed best: ", res.best === nothing ? "nothing" : (gp = res.best.gp, Delta = res.best.Delta))

@testset "CM checkpoint (:exclude_row, :cplus) resume" begin
    @test res.n_eval > n_eval_before_resume   # eval counter CONTINUED, was not reset

    # Reproducibility check: regenerating the SAME :exclude_row context independently and
    # re-verifying the checkpoint's own best-feasible incumbent at THAT exact point.
    ctx_check = d20_real_setup_design(W = latest.W, δ = latest.delta, find_smallest = latest.find_smallest,
        draw_design = latest.draw_design, draw_seed = latest.draw_seed, destination_sample = latest.destination_sample)
    @test ctx_check.D_dest == latest.D_dest
    checksum_ok = ctx_check.draw_meta.checksum_uniform == latest.draw_checksum_uniform &&
                  ctx_check.draw_meta.checksum_transformed == latest.draw_checksum_transformed
    @test checksum_ok   # exact string match -- draw regeneration must be bit-reproducible

    if latest.best_feasible !== nothing
        pcx_check = build_cm_production_context(ctx_check, CS; L = latest.cm_L, contrasts = latest.cm_contrasts, probs = latest.cm_probs)
        pe_check = build_pivot_elimination(ctx_check)
        xf_check = x_free_from_w(latest.best_feasible.w, pe_check)
        _, base_check = cm_production_value(xf_check, pcx_check)
        Delta_check = delta_dual_from_base(pcx_check.ctx_cm.obj, base_check)
        @printf("  re-verified Delta at checkpoint's best_feasible.w: %.15f  (checkpoint recorded: %.15f, diff=%.2e)\n",
            Delta_check, latest.best_feasible.Delta, abs(Delta_check - latest.best_feasible.Delta))
        @test abs(Delta_check - latest.best_feasible.Delta) < 1e-9
    else
        println("  (no best_feasible incumbent recorded in the write-phase checkpoint -- skipping incumbent re-verification)")
    end
end
println("All CM-checkpoint (:exclude_row, :cplus) resume tests passed.")
