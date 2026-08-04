# User-directed investigation (2026-08-04, NOT a production default change): replay the eval18
# captured point with obj.lower_limit overridden to -10 (the value the user says they will
# realistically use in production), instead of the codebase's own -50 default, to see how much
# faster the fast-fail clamp fires.
#
# lower_limit is a plain mutable field on OperatorPsiBundle (operator_psi_bundle.jl:81, `mutable
# struct`, no struct-level default per profiled-inner-readiness-2026-08-03's own fix) -- overridden
# HERE, post-construction, on a bundle built via the UNCHANGED production path
# (d20_real_setup_design's own default -50 is untouched; this script does not modify
# context_real_d20.jl or any shared default). This is a one-off investigation, not a proposal to
# change the production default.
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

"Mirrors reduced_cm_base_state exactly, except obj.lower_limit is overridden BEFORE priming/solve."
function reduced_cm_base_state_lowerlimit(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, lower_limit_override::Float64; method::Symbol = :suffix)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_cm_operator_bundle(ctx, θ_full0, layout, cctx; method = method)
    @printf("  obj.lower_limit BEFORE override = %.4f\n", obj.lower_limit); flush(stdout)
    obj.lower_limit = lower_limit_override
    @printf("  obj.lower_limit AFTER  override = %.4f\n", obj.lower_limit); flush(stdout)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_cmlookup(obj, st, cctx)
    return (obj = obj, st = st, inner_status = nStatus, n_fg = n_fg, n_hess = n_hess, objSol = objSol)
end

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
@printf("context build: %.2fs  (obj.lower_limit at context build, unchanged default = %.4f)\n",
    t_ctx, ctx.obj.lower_limit); flush(stdout)
D = ctx.D
korea_idx, brazil_idx = 14, 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)

w0_str = read(joinpath(D4X, "eval18_captured_point_2026-08-02.txt"), String)
w0 = Vector{Float64}(eval(Meta.parse(w0_str)))
θ0 = ctx.θ0_up
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
decoded = decode_outer_profiled(w0, ctx, pe)

println("="^90); println("eval18 point, PRODUCTION maxit=100, lower_limit OVERRIDDEN to -10.0"); println("="^90); flush(stdout)
t_solve = @elapsed result = try
    r = reduced_cm_base_state_lowerlimit(decoded.xf, ctx, layout, cctx_reduced, -10.0)
    (inner_status = r.inner_status, n_fg = r.n_fg, n_hess = r.n_hess, objSol = r.objSol)
catch e
    e isa CMExpectedSolveFailure || rethrow()
    m = match(r"nStatus=(-?\d+)", e.msg)
    m === nothing && rethrow()
    (inner_status = parse(Int, m.captures[1]), n_fg = -1, n_hess = -1, objSol = NaN)
end
@printf("[lower_limit=-10, maxit=100] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d  objSol=%s\n",
    t_solve, result.inner_status, result.n_fg, result.n_hess, string(result.objSol))
flush(stdout)

println("\n" * "="^90); println("VERDICT")
@printf("nStatus=%d  (compare: -50 default needed maxit=1000/iteration~299/577.67s to reach -300; this is the SAME point, SAME W=100,000, SAME maxit=100 production budget, ONLY lower_limit changed)\n", result.inner_status)
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
