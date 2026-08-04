# fix/profiled-functional-readiness-closeout-2026-08-03, task §4.1: real per-family W=100,000
# warm-start gate for flexible_CM, origin_ZC, CM_plus_ZC.
#
# Mechanism (confirmed live 2026-08-04 by reading operator_psi_bundle.jl): OperatorPsiBundle.x
# defaults to `NaN .* ones(...)` on every fresh construction (genuinely cold every time), and
# CS.inner_loop_initial_values(obj) = `obj.use_cached_x && norm(obj.x)<1e6 ? obj.x : zeros(...)`.
# `use_cached_x=true` is already threaded from the base context through every augmented bundle
# ("copied from ctx.obj UNCHANGED"). So warm-starting is: build a bundle, set `obj.x` to a
# converged (ζ*,λ*) BEFORE the inner KNITRO solve call -- this file adds thin "_warm" wrapper
# functions that do exactly the cold base_state functions' own body, with ONE extra line
# (`obj.x = collect(x_warm)`) inserted between priming and the inner_loop_KNITRO_* call. ADDITIVE
# ONLY -- does not modify reduced_cm_base_state/reduced_originzc_base_state/reduced_meanzc_base_state.
#
# Per CLAUDE.md's own standing finding (feedback-user-knitro-convergence-not-start-dependent):
# warm/cold start must affect ONLY speed (iterations/FG calls/wall), never whether the solve
# converges or what Delta-star it finds -- this gate's PASS criterion is exact status/Delta-star
# agreement, not "close enough."
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
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
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
          "profiled_reduced_originzc_lookup_kernels_2026-08-02.jl",
          "profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl",
          "profiled_outer_gradient_layout_contract_2026-08-01.jl",
          "profiled_stable_layout_digest_2026-08-01.jl",
          "profiled_operator_bundle_2026-08-01.jl",
          "profiled_outer_evaluator_2026-08-01.jl",
          "profiled_lfix_incremental_2026-08-01.jl",
          "profiled_shared_economic_gradient_engine_2026-08-01.jl",
          "profiled_family_adapters_2026-08-01.jl",
          "profiled_restriction_contrib0_operators_2026-08-01.jl",
          "reduced_restricted_family_verification_2026-08-03.jl",
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Printf

t0 = time()
println("PID=", getpid()); flush(stdout)

# ---------------------------------------------------------------------------
# Warm variants -- ADDITIVE, mirror the cold base_state functions exactly except one injected line.
# ---------------------------------------------------------------------------
function reduced_cm_base_state_warm(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, x_warm::AbstractVector{Float64}; method::Symbol = :suffix)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_cm_operator_bundle(ctx, θ_full0, layout, cctx; method = method)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    obj.x = collect(Float64, x_warm)   # <-- ONLY difference from reduced_cm_base_state
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_cmlookup(obj, st, cctx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_cm_base_state_warm: inner solve failed, nStatus=$nStatus"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end

function reduced_originzc_base_state_warm(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        octx::OriginZCCoreHessCtx, νfull::AbstractVector{Float64}, x_warm::AbstractVector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_originzc_operator_bundle(ctx, θ_full0, layout, octx)
    refresh_zc_targets!(st.zc_ws, st.op, st.zc_layout, νfull)
    octx.nu_ref[] = collect(νfull)
    prime_operator!(obj, θ_full0, ctx, octx.core_cf_ref; restriction_state = octx)
    obj.x = collect(Float64, x_warm)   # <-- ONLY difference from reduced_originzc_base_state
    octx.profiled_theta_ref[] = copy(θ_full0)
    octx.fg_lookup_st = st
    octx.fg_backend = :operator
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_originzc(obj, st, octx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_originzc_base_state_warm: inner solve failed, nStatus=$nStatus"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end

function reduced_meanzc_base_state_warm(x_free0::AbstractVector, νvec::AbstractVector{Float64}, ctx,
        layout::ProfiledEconomicMomentLayout, cctx::CMBinHessCtx, x_warm::AbstractVector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_meanzc_operator_bundle(ctx, θ_full0, layout, cctx)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    reset_for_solve!(st, νvec)
    cctx.nu_ref[] = collect(νvec)
    obj.x = collect(Float64, x_warm)   # <-- ONLY difference from reduced_meanzc_base_state
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_meanzc(obj, st, cctx)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_meanzc_base_state_warm: inner solve failed, nStatus=$nStatus"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return (obj = obj, st = st, inner_status = nStatus, ζstar = ζstar, λstar = λstar, n_fg = n_fg, n_hess = n_hess)
end

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

W_VAL = 100_000
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0)
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
w_profiled_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)
decoded_calib = decode_outer_profiled(w_profiled_calib, ctx, pe)
x_free_calib = decoded_calib.xf
@printf("calibration point: outer_dim=%d\n", length(w_profiled_calib)); flush(stdout)

"Compare a cold/warm pair and report the gate result. Requires EXACT status agreement and
Delta-star agreement to numerical tolerance (1e-8) -- warm start changes speed, not the answer."
function gate_pair(label::String, cold, warm; tol = 1e-8)
    Delta_cold = -cold.ζstar; Delta_warm = -warm.ζstar
    status_match = cold.inner_status == warm.inner_status
    delta_match = abs(Delta_cold - Delta_warm) < tol * max(1.0, abs(Delta_cold))
    check("$label: warm status matches cold ($(cold.inner_status) == $(warm.inner_status))", status_match)
    check("$label: warm Delta-star matches cold (|Δ|=$(abs(Delta_cold-Delta_warm)))", delta_match)
    @printf("  %-14s cold: status=%d Delta=%.10f n_fg=%d n_hess=%d\n", label, cold.inner_status, Delta_cold, cold.n_fg, cold.n_hess)
    @printf("  %-14s warm: status=%d Delta=%.10f n_fg=%d n_hess=%d\n", label, warm.inner_status, Delta_warm, warm.n_fg, warm.n_hess)
    return status_match && delta_match
end

println("="^90); println("FAMILY flexible_CM: cold vs warm at real D20/W=100,000"); println("="^90); flush(stdout)
aug_reduced_cm = build_cm_augmented_obj_archB(ctx, CS; L = 50, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx_cold_cm = build_cm_bin_ctx(ctx, aug_reduced_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
t_cold_cm = @elapsed cold_cm = reduced_cm_base_state(x_free_calib, ctx, layout, cctx_cold_cm)
@printf("flexible_CM cold solve: %.2fs\n", t_cold_cm); flush(stdout)
x_warm_cm = vcat(cold_cm.ζstar, cold_cm.λstar)
cctx_warm_cm = build_cm_bin_ctx(ctx, aug_reduced_cm; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)   # FRESH cctx
t_warm_cm = @elapsed warm_cm = reduced_cm_base_state_warm(x_free_calib, ctx, layout, cctx_warm_cm, x_warm_cm)
@printf("flexible_CM warm solve: %.2fs\n", t_warm_cm); flush(stdout)
gate_pair("flexible_CM", cold_cm, warm_cm)

println("\n" * "="^90); println("FAMILY origin-ZC: cold vs warm at real D20/W=100,000"); println("="^90); flush(stdout)
layout_o = OriginByPowerLayout(D, 1, 0)
aug_reduced_oz = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
octx_cold_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
νvec0_oz = fill(1.0, D)
t_cold_oz = @elapsed cold_oz = reduced_originzc_base_state(x_free_calib, ctx, layout, octx_cold_oz, νvec0_oz)
@printf("origin-ZC cold solve: %.2fs\n", t_cold_oz); flush(stdout)
x_warm_oz = vcat(cold_oz.ζstar, cold_oz.λstar)
octx_warm_oz = build_originzc_core_hess_ctx(aug_reduced_oz, ctx; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)   # FRESH octx
t_warm_oz = @elapsed warm_oz = reduced_originzc_base_state_warm(x_free_calib, ctx, layout, octx_warm_oz, νvec0_oz, x_warm_oz)
@printf("origin-ZC warm solve: %.2fs\n", t_warm_oz); flush(stdout)
gate_pair("origin-ZC", cold_oz, warm_oz)

println("\n" * "="^90); println("FAMILY CM+ZC: cold vs warm at real D20/W=100,000"); println("="^90); flush(stdout)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
aug_reduced_cz = build_cm_meanzc_augmented_obj(ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = reduced_obj0, profiled_layout = layout)
cctx_cold_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference, profiled_layout = layout)
νvec0_cz = fill(1.0, K_MEAN)
t_cold_cz = @elapsed cold_cz = reduced_meanzc_base_state(x_free_calib, νvec0_cz, ctx, layout, cctx_cold_cz)
@printf("CM+ZC cold solve: %.2fs\n", t_cold_cz); flush(stdout)
x_warm_cz = vcat(cold_cz.ζstar, cold_cz.λstar)
cctx_warm_cz = build_cm_meanzc_bin_ctx(ctx, aug_reduced_cz; core_hessian_backend = :exact_winner_pair_parallel,
    zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference, profiled_layout = layout)   # FRESH cctx
t_warm_cz = @elapsed warm_cz = reduced_meanzc_base_state_warm(x_free_calib, νvec0_cz, ctx, layout, cctx_warm_cz, x_warm_cz)
@printf("CM+ZC warm solve: %.2fs\n", t_warm_cz); flush(stdout)
gate_pair("CM+ZC", cold_cz, warm_cz)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ALL_PASS[] || exit(1)
