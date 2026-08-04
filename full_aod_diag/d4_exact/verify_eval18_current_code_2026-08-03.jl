# fix/profiled-functional-readiness-closeout-2026-08-03, section 4: independent re-confirmation,
# on THIS session's current code/HEAD, of the eval18 forensic verdict
# (dropbox:.../eval18_forensic_stage2_verdict_2026-08-02/EVAL18_FORENSIC_VERDICT_2026-08-02.md,
# read and independently assessed this session -- not blindly re-cited). That prior forensic
# program is genuinely rigorous (exhaustive FD gradient/Hessian at multiple W, callback-freshness
# tests, an extended-maxit=1000 KNITRO run showing the objective crosses lower_limit=-50 at
# iteration ~299 firing the correct native nStatus=-300, AND a fully independent HiGHS LP
# infeasibility certificate -- two independent proofs, not the same evidence twice) -- but per this
# task's own instruction, that classification must be independently re-derived against CURRENT
# code, not inherited. This script replays the EXACT captured point
# (dropbox:.../eval18_secondmode_and_threading_findings_2026-08-02/key_results/
# captured_point_eval18_2026-08-02.txt, gp=0.9553510115510775, confirmed by matching first
# coordinate) through the real, unmodified production evaluator
# (evaluate_profiled_flexcm_point/reduced_cm_base_state) at real D20/W=100,000, with (a) the
# production maxit=100 budget and (b) an extended maxit=1000 budget, to confirm the SAME
# nStatus=-400 (n_iters=100) -> nStatus=-300 (n_iters~299, objSol=-Inf) signature still holds today.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl",
          "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl", "cm_checkpoint.jl",
          "operator_hessian_weights.jl", "operator_psi_bundle.jl", "cm_lookup_live_knitro.jl", "cm_lookup_production.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "homogeneous_contraction_2026-07-31.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "reduced_homogeneous_contraction_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "reduced_operator_verification_2026-08-01.jl",
          "recover_full_a_2026-07-31.jl",
          "profiled_reduced_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_frechet_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "reduced_restricted_family_verification_2026-08-03.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf

t0 = time()
println("PID=", getpid()); flush(stdout)

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
D = ctx.D
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)

aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
fctx_cm = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx_reduced)

# Parse the exact captured eval18 point (flat Julia array literal, 361 Float64s: w[1]=gp, w[2:end]=z_free).
w0_str = read(joinpath(D4X, "eval18_captured_point_2026-08-02.txt"), String)
w0 = Vector{Float64}(eval(Meta.parse(w0_str)))
@printf("Parsed captured point: length=%d  w[1](gp)=%.16f\n", length(w0), w0[1]); flush(stdout)
gp_expected = 0.9553510115510775
@assert abs(w0[1] - gp_expected) < 1e-12 "captured point gp mismatch: got $(w0[1]), expected $gp_expected"

"""
reduced_cm_base_state (called inside evaluate_profiled_flexcm_point) THROWS CMExpectedSolveFailure
(carrying nStatus) rather than returning a value when nStatus is outside {0,-100,-101,-103} --
confirmed live 2026-08-03 (the first version of this script did not catch this, exited with an
uncaught exception whose OWN error message nonetheless already carried the decisive nStatus=-400
result). Catch it here so both arms run regardless.
"""
function try_evaluate(label::String, w0, fctx; maxit_override = nothing)
    t = @elapsed begin
        result = try
            ev = evaluate_profiled_flexcm_point(w0, fctx; maxit_override = maxit_override)
            (status = ev.result.inner_status, zeta = ev.result.zeta, n_fg = ev.result.n_fg_calls, n_hess = ev.result.n_hess_calls)
        catch e
            e isa CMExpectedSolveFailure || rethrow()
            (status = e.nStatus, zeta = NaN, n_fg = -1, n_hess = -1)
        end
    end
    @printf("[%s] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d  zeta=%.6g\n", label, t, result.status, result.n_fg, result.n_hess, result.zeta)
    flush(stdout)
    return result
end

println("="^90); println("Replay at PRODUCTION maxit=100 (default)"); println("="^90); flush(stdout)
r1 = try_evaluate("maxit=100", w0, fctx_cm)

println("="^90); println("Replay at EXTENDED maxit=1000"); println("="^90); flush(stdout)
r2 = try_evaluate("maxit=1000", w0, fctx_cm; maxit_override = 1000)

println()
println("="^90); println("VERDICT"); println("="^90)
@printf("production maxit=100:  nStatus=%d  (expect -400, matching the 2026-08-02 forensic verdict)\n", r1.status)
@printf("extended  maxit=1000:  nStatus=%d  zeta=%.6g  (expect -300 with zeta at -KN_INFINITY-scale, matching the same verdict)\n",
    r2.status, r2.zeta)
ok = r1.status == -400 && r2.status == -300
println("\nEVAL18_CURRENT_CODE_REPLAY_RESULT: ", ok ? "CONFIRMS 2026-08-02 forensic verdict (genuinely unbounded, clamp correct, budget-limited)" : "DIVERGES FROM 2026-08-02 verdict -- investigate")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
exit(ok ? 0 : 1)
