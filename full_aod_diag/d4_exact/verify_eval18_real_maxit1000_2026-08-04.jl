# fix/profiled-functional-readiness-closeout-2026-08-03, task §5.1 continuation: GENUINE maxit=1000
# replay of the captured eval18 point.
#
# Root cause found live 2026-08-04: verify_eval18_current_code_2026-08-03.jl's own `maxit_override`
# kwarg on `evaluate_profiled_flexcm_point` is DEAD CODE -- confirmed by reading that function's
# body (profiled_restricted_family_adapters_2026-08-02.jl:99-119): it accepts `maxit_override` but
# never forwards it anywhere. The real inner solve (`reduced_cm_base_state` ->
# `inner_loop_KNITRO_reduced_cmlookup`, profiled_reduced_lookup_kernels_2026-08-02.jl:240) sets
# KNITRO's maxit exclusively via `KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)` -- a Julia-level
# `maxit_override` argument has no mechanism to reach KNITRO through this evaluator at all. This is
# why the previous take (both maxit=100 and maxit=1000 "arms") returned the IDENTICAL nStatus=-400 --
# both silently ran at whatever maxit `ek_inner.opt` bakes in (100), never the intended 1000.
#
# The REAL lever is `inner_loop_opt`, a kwarg on `d20_real_setup_design` (draw_design.jl:131,
# default `full_aod_diag/ek_inner.opt`) that flows UNCHANGED through
# `build_reduced_base_obj_for_family` -> `build_cm_augmented_obj_archB` -> `build_cm_bin_ctx` -> the
# solved `obj.inner_loop_opt` field `KN_load_param_file` actually reads (confirmed live by grep at
# every hop: profiled_restricted_family_base_2026-08-01.jl:60/92 documents "copied from ctx.obj
# UNCHANGED" and the same pattern repeats at cm_production_bundle.jl:151/163). This script builds
# the context ONCE with `inner_loop_opt` pointed at a genuine maxit=1000 variant
# (`ek_inner_maxit1000_2026-08-04.opt`, byte-identical to `ek_inner.opt` except `maxit 1000` instead
# of `maxit 100`) and replays the SAME captured eval18 point through it -- this is the first time
# maxit has actually been varied for this point on current code.
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
opt_maxit1000 = joinpath(D4X, "..", "ek_inner_maxit1000_2026-08-04.opt")
isfile(opt_maxit1000) || error("maxit=1000 opt file not found: $opt_maxit1000")
@printf("Using inner_loop_opt = %s (genuine maxit=1000 override, not the dead maxit_override kwarg)\n", opt_maxit1000)
flush(stdout)

t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_loop_opt = opt_maxit1000)
@printf("context build (maxit=1000 variant): %.2fs\n", t_ctx); flush(stdout)
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
@printf("obj.inner_loop_opt actually wired to solve = %s (must equal the maxit=1000 file above)\n", cctx_reduced.obj.inner_loop_opt)
flush(stdout)
@assert cctx_reduced.obj.inner_loop_opt == opt_maxit1000 "inner_loop_opt did not propagate to the solved bundle -- override is not reaching KNITRO, investigate further before trusting this run"

w0_str = read(joinpath(D4X, "eval18_captured_point_2026-08-02.txt"), String)
w0 = Vector{Float64}(eval(Meta.parse(w0_str)))
gp_expected = 0.9553510115510775
@assert abs(w0[1] - gp_expected) < 1e-12 "captured point gp mismatch: got $(w0[1]), expected $gp_expected"
@printf("Parsed captured point: length=%d  w[1](gp)=%.16f\n", length(w0), w0[1]); flush(stdout)

println("="^90); println("Replay at GENUINE maxit=1000 (real KN_load_param_file override)"); println("="^90); flush(stdout)
t_solve = @elapsed result = try
    ev = evaluate_profiled_flexcm_point(w0, fctx_cm)
    (status = ev.result.inner_status, zeta = ev.result.zeta, n_fg = ev.result.n_fg_calls, n_hess = ev.result.n_hess_calls)
catch e
    e isa CMExpectedSolveFailure || rethrow()
    m = match(r"nStatus=(-?\d+)", e.msg)
    m === nothing && rethrow()
    (status = parse(Int, m.captures[1]), zeta = NaN, n_fg = -1, n_hess = -1)
end
@printf("[maxit=1000 REAL] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d  zeta=%.6g\n",
    t_solve, result.status, result.n_fg, result.n_hess, result.zeta)
flush(stdout)

println()
println("="^90); println("VERDICT"); println("="^90)
@printf("genuine maxit=1000: nStatus=%d  (2026-08-02 forensic verdict expected -300 around iteration ~299)\n", result.status)
if result.status == -300
    println("EVAL18_REAL_MAXIT1000_RESULT: CONFIRMS 2026-08-02 verdict -- genuinely unbounded, correct lower_limit clamp fires once given enough budget")
elseif result.status == -400
    println("EVAL18_REAL_MAXIT1000_RESULT: STILL -400 at maxit=1000 -- either the true crossing iteration is beyond 1000, or the 2026-08-02 verdict's own claimed iteration (~299) does not reproduce on current code/data -- genuinely open, needs the archived iteration trajectory re-examined")
else
    println("EVAL18_REAL_MAXIT1000_RESULT: UNEXPECTED status $(result.status) -- investigate")
end
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
