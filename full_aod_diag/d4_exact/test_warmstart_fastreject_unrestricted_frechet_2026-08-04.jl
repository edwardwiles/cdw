# fix/profiled-functional-readiness-closeout-2026-08-03, task §4.1/§4.2: W=100,000 warm-start +
# fast-rejection gates for unrestricted and common_frechet -- the 2 remaining REDUCED families
# after flexible_CM/origin_ZC/CM_plus_ZC (this session's earlier work,
# test_warmstart_w100k_2026-08-04.jl / test_fastreject_2026-08-04.jl /
# test_fastreject_stage2_w100k_2026-08-04.jl). Same methodology: warm-start via `.x` injection on
# a FRESH bundle (OperatorPsiBundle.x defaults NaN on construction, use_cached_x=true already
# threaded from the base context); fast-reject via a systematically-constructed additive log-A
# shift from calibration, classified first at cheap W=20,000, promoted to real W=100,000 if
# infeasible.
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

# ---------------------------------------------------------------------------
# Warm variants (additive, mirror the cold functions exactly + one injected line)
# ---------------------------------------------------------------------------
function evaluate_profiled_point_warm(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, x_warm::AbstractVector{Float64}; ref_obj = ctx.obj)
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    obj_p, st_p = build_profiled_operator_bundle(ctx, θ_full, spec; ref_obj = ref_obj)
    obj_p.x = collect(Float64, x_warm)   # <-- ONLY difference from evaluate_profiled_point
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_profiled(obj_p, st_p)
    return (inner_status = nStatus, ζstar = x[1], λstar = collect(x[2:end]), n_fg = n_fg, n_hess = n_hess)
end

function reduced_frechet_base_state_warm(x_free0::AbstractVector, ctx, layout::ProfiledEconomicMomentLayout,
        cctx::CMBinHessCtx, level_targets::Vector{Float64}, x_warm::AbstractVector{Float64})
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    obj, st = build_reduced_frechet_operator_bundle(ctx, θ_full0, layout, cctx, level_targets)
    prime_operator!(obj, θ_full0, ctx, cctx.core_cf_ref; restriction_state = cctx)
    obj.x = collect(Float64, x_warm)   # <-- ONLY difference from reduced_frechet_base_state
    cctx.profiled_theta_ref[] = copy(θ_full0)
    cctx.cmlookup_st = st
    cctx.inner_fg_backend = :cm_lookup
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_reduced_frechet(obj, st, cctx, level_targets)
    nStatus ∈ (0, -100, -101, -103) || throw(CMExpectedSolveFailure("reduced_frechet_base_state_warm: inner solve failed, nStatus=$nStatus"))
    return (inner_status = nStatus, ζstar = x[1], λstar = collect(x[2:end]), n_fg = n_fg, n_hess = n_hess)
end

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function try_status(label::String, thunk)
    t = @elapsed result = try
        r = thunk()
        (status = r.inner_status, n_fg = r.n_fg, n_hess = r.n_hess)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        m = match(r"nStatus=(-?\d+)", e.msg)
        m === nothing && rethrow()
        (status = parse(Int, m.captures[1]), n_fg = -1, n_hess = -1)
    end
    @printf("  [%s] wall=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d\n", label, t, result.status, result.n_fg, result.n_hess)
    flush(stdout)
    return (label = label, wall = t, result...)
end

function build_common(W_VAL::Int)
    ctx = d20_real_setup_design(W = W_VAL, δ = 1.0, find_smallest = true,
        draw_design = :sobol_randomized, draw_seed = 20260719, destination_sample = :exclude_row,
        exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
        σHat = 3.0)
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
    w_calib = reduce_to_w_profiled(θ0[3+D], z_calib, pe)
    return (ctx = ctx, spec = spec, layout = layout, reduced_obj0 = reduced_obj0, pe = pe, w_calib = w_calib, D = D)
end

println("="^90); println("STAGE 1: unrestricted + common_frechet, real W=100,000 warm-start"); println("="^90); flush(stdout)
t_ctx = @elapsed common = build_common(100_000)
@printf("W=100,000 context build: %.2fs\n", t_ctx); flush(stdout)

println("\n--- unrestricted: cold vs warm ---"); flush(stdout)
decoded_calib = decode_outer_profiled(common.w_calib, common.ctx, common.pe)
t_cold_u = @elapsed cold_u = evaluate_profiled_point(common.w_calib, common.ctx, common.spec, common.pe)
@printf("unrestricted cold solve: %.2fs  status=%d  Delta=%.10f\n", t_cold_u, cold_u.result.inner_status, -cold_u.result.zeta); flush(stdout)
x_warm_u = vcat(cold_u.result.zeta, cold_u.result.beta)
t_warm_u = @elapsed warm_u = evaluate_profiled_point_warm(common.w_calib, common.ctx, common.spec, common.pe, x_warm_u)
@printf("unrestricted warm solve: %.2fs  status=%d  Delta=%.10f\n", t_warm_u, warm_u.inner_status, -warm_u.ζstar); flush(stdout)
check("unrestricted: warm status matches cold", cold_u.result.inner_status == warm_u.inner_status)
check("unrestricted: warm Delta-star matches cold", abs(-cold_u.result.zeta - (-warm_u.ζstar)) < 1e-8 * max(1.0, abs(cold_u.result.zeta)))

println("\n--- common_frechet: cold vs warm ---"); flush(stdout)
# level_targets is a FIELD on the frechet-specific augmented obj (aug_reduced_f.level_targets),
# not a function to compute -- matches test_frechet_threaded_profiled_d4_2026-08-03.jl's own
# construction exactly (build_cm_frechet_augmented_obj_archB, NOT flexible_CM's
# build_cm_augmented_obj_archB -- different builder, confirmed by reading cm_frechet_level.jl).
aug_cf_probe = build_cm_frechet_augmented_obj_archB(common.ctx, CS; L = 50, contrasts = :anchored, base_obj = common.reduced_obj0, profiled_layout = common.layout)
level_targets = aug_cf_probe.level_targets
cctx_cold_fr = build_cm_bin_ctx(common.ctx, aug_cf_probe; profiled_layout = common.layout, inner_fg_backend = :dense_reference, threaded_bins = true)
t_cold_fr = @elapsed cold_fr = reduced_frechet_base_state(decoded_calib.xf, common.ctx, common.layout, cctx_cold_fr, level_targets)
@printf("common_frechet cold solve: %.2fs  status=%d  Delta=%.10f\n", t_cold_fr, cold_fr.inner_status, -cold_fr.ζstar); flush(stdout)
x_warm_fr = vcat(cold_fr.ζstar, cold_fr.λstar)
cctx_warm_fr = build_cm_bin_ctx(common.ctx, aug_cf_probe; profiled_layout = common.layout, inner_fg_backend = :dense_reference, threaded_bins = true)
t_warm_fr = @elapsed warm_fr = reduced_frechet_base_state_warm(decoded_calib.xf, common.ctx, common.layout, cctx_warm_fr, level_targets, x_warm_fr)
@printf("common_frechet warm solve: %.2fs  status=%d  Delta=%.10f\n", t_warm_fr, warm_fr.inner_status, -warm_fr.ζstar); flush(stdout)
check("common_frechet: warm status matches cold", cold_fr.inner_status == warm_fr.inner_status)
check("common_frechet: warm Delta-star matches cold", abs(-cold_fr.ζstar - (-warm_fr.ζstar)) < 1e-8 * max(1.0, abs(cold_fr.ζstar)))

println("\n" * "="^90); println("STAGE 2: fast-rejection classification sweep at W=20,000"); println("="^90); flush(stdout)
t_ctx20 = @elapsed common20 = build_common(20_000)
@printf("W=20,000 context build: %.2fs\n", t_ctx20); flush(stdout)
decoded_calib20 = decode_outer_profiled(common20.w_calib, common20.ctx, common20.pe)
aug_cf_probe20 = build_cm_frechet_augmented_obj_archB(common20.ctx, CS; L = 50, contrasts = :anchored, base_obj = common20.reduced_obj0, profiled_layout = common20.layout)
level_targets20 = aug_cf_probe20.level_targets

println("\n--- unrestricted: additive log-A shift ---"); flush(stdout)
u_results20 = NamedTuple[]
for shift in (3.0, 6.0, 10.0)
    w_s = copy(common20.w_calib); w_s[2:end] .+= shift
    r = try_status("unrestricted shift=$shift", () -> begin
        ev = evaluate_profiled_point(w_s, common20.ctx, common20.spec, common20.pe)
        (inner_status = ev.result.inner_status, n_fg = ev.result.n_fg_calls, n_hess = ev.result.n_hess_calls)
    end)
    push!(u_results20, r)
end

println("\n--- common_frechet: additive log-A shift ---"); flush(stdout)
fr_results20 = NamedTuple[]
for shift in (3.0, 6.0, 10.0)
    w_s = copy(common20.w_calib); w_s[2:end] .+= shift
    decoded_s = decode_outer_profiled(w_s, common20.ctx, common20.pe)
    cctx_s = build_cm_bin_ctx(common20.ctx, aug_cf_probe20; profiled_layout = common20.layout, inner_fg_backend = :dense_reference, threaded_bins = true)
    r = try_status("common_frechet shift=$shift", () -> reduced_frechet_base_state(decoded_s.xf, common20.ctx, common20.layout, cctx_s, level_targets20))
    push!(fr_results20, r)
end

println("\n" * "="^90); println("STAGE 2 SUMMARY (W=20,000)"); println("="^90)
for r in vcat(u_results20, fr_results20)
    cls = r.status == 0 ? "FEASIBLE" : (r.status == -300 ? "INFEASIBLE" : "STALL/OTHER($(r.status))")
    @printf("%-28s status=%-5d wall=%6.2fs  -> %s\n", r.label, r.status, r.wall, cls)
end
flush(stdout)

println()
println(ALL_PASS[] ? "STAGE 1 (warm-start) ALL PASS" : "STAGE 1 SOME FAILURES")
@printf("TOTAL WALL: %.1fs\n", time() - t0)
flush(stdout)
ALL_PASS[] || exit(1)
