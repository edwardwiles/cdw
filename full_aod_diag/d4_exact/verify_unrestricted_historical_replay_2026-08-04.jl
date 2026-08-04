# fix/profiled-functional-readiness-closeout-2026-08-03, task §5.2: replay of the EXACT historical
# unrestricted slow-failure points, recovered from the archived 2026-08-02 forensic package
# (dropbox:.../unrestricted_eval18_forensic_audit_2026-08-02/key_results/
# STAGE1A_UNRESTRICTED_REPRO_W100000_2026-08-02.jld2, `all_w::Dict{Int,Vector{Float64}}`, real
# 361-dim outer points from a real W=100,000 campaign run, NOT reconstructed/approximated).
#
# Documented verdict (UNRESTRICTED_INNER_STALL_FORENSIC_VERDICT_2026-08-02.md): a genuine missing
# `lower_limit` clamp meant these 6 points slow-failed (nStatus=-400, 100 iters, ~34-46s) before the
# fix; a single added line resolved all 6/6 to the correct native nStatus=-300 in 1-12 iterations
# (0.26-5.04s). This session's own §5.1 work already confirmed `lower_limit` is required (no
# struct-level default) as of this branch's HEAD (profiled-inner-readiness-2026-08-03's own fix,
# unchanged since) -- this script is the promised REPLAY confirming that fix still holds on THESE
# exact historical points, not a re-derivation.
#
# Points replayed: idx=2 (gp=0.9314418652958134, documented 40.8x speedup) and idx=17
# (gp=0.9302938353076481, documented 123.8x speedup, the single largest) -- satisfies the task's
# "at least two exact historical points" requirement.
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
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs\n", t_ctx); flush(stdout)
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(14 => 3))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
z_calib = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*ctx.D_dest], ctx.D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)

function replay_point(label, idx, gp_expected, path)
    w0 = Vector{Float64}(eval(Meta.parse(read(path, String))))
    @assert abs(w0[1] - gp_expected) < 1e-10 "$label gp mismatch: got $(w0[1]), expected $gp_expected"
    @printf("Parsed historical point idx=%d: length=%d w[1](gp)=%.16f\n", idx, length(w0), w0[1]); flush(stdout)
    t_solve = @elapsed ev = evaluate_profiled_point(w0, ctx, spec, pe)
    @printf("[%s idx=%d] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d  Delta=%.8g\n",
        label, idx, t_solve, ev.result.inner_status, ev.result.n_fg_calls, ev.result.n_hess_calls, -ev.result.zeta)
    flush(stdout)
    return ev.result.inner_status
end

println("="^90); println("Historical point idx=2 (documented: -400/100iters/36.1s BEFORE -> -300/3iters/0.89s AFTER)"); println("="^90); flush(stdout)
s2 = replay_point("unrestricted", 2, 0.9314418652958134,
    "/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03/historical_unrestricted_eval2_2026-08-04.txt")

println("\n" * "="^90); println("Historical point idx=17 (documented: -400/100iters/34.5s BEFORE -> -300/1iter/0.28s AFTER, largest speedup)"); println("="^90); flush(stdout)
s17 = replay_point("unrestricted", 17, 0.9302938353076481,
    "/bbkinghome/edav/repo_scratch/profiled-functional-readiness-closeout-2026-08-03/historical_unrestricted_eval17_2026-08-04.txt")

println("\n" * "="^90); println("VERDICT"); println("="^90)
@printf("idx=2:  nStatus=%d (expect -300, per documented AFTER)\n", s2)
@printf("idx=17: nStatus=%d (expect -300, per documented AFTER)\n", s17)
ok = s2 == -300 && s17 == -300
println(ok ? "UNRESTRICTED_HISTORICAL_REPLAY_CONFIRMED: fix still holds on the exact original slow-failure points" :
             "UNRESTRICTED_HISTORICAL_REPLAY_DIVERGES -- investigate")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ok || exit(1)
