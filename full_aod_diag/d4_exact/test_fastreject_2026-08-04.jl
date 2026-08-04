# fix/profiled-functional-readiness-closeout-2026-08-03, task §4.2: real fast-rejection gate for
# flexible_CM, origin_ZC, CM+ZC.
#
# Systematic construction (not arbitrary perturbation, per the task's own explicit instruction):
#   - flexible_CM: interpolate along the calibration -> eval18-captured-point ray
#     (w_k = w_calib + k*(w_eval18 - w_calib)), since the eval18 point IS an already-documented,
#     independently-confirmed-genuinely-unbounded point (this continuation's own earlier maxit=1000
#     result: nStatus=-300 at iteration ~299 -- the "slow genuinely unbounded" category). Testing
#     smaller k values along the SAME ray asks whether a less extreme point on that ray is FAST-
#     infeasible, still feasible, or also slow -- classified from real observed behavior, not assumed.
#   - origin_ZC / CM+ZC: no analogous documented ray exists for these families. Constructed instead
#     as a large systematic additive shift to every retained log-A coordinate (`w[2:end] .+= shift`,
#     shift in log-A units -- e.g. shift=10 means every retained A_od implied by this point is
#     exp(10)~=22000x its calibration value), a directionally uniform "far from calibration" probe,
#     not a random/arbitrary one.
#
# Runs each candidate FIRST at cheap D20/W=20,000 to classify (feasible / fast-infeasible /
# slow-infeasible), matching the task's own "at W=20,000 ... locate a point that is independently
# classified infeasible; promote the same decoded point to W=100,000" instruction. Any point
# classified infeasible at W=20,000 is then re-run at the real W=100,000 to confirm + record
# iterations/time to rejection there.
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

function try_status(label::String, thunk)
    t = @elapsed result = try
        r = thunk()
        (status = r.inner_status, n_fg = r.n_fg, n_hess = r.n_hess, zeta = r.ζstar)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        m = match(r"nStatus=(-?\d+)", e.msg)
        m === nothing && rethrow()
        (status = parse(Int, m.captures[1]), n_fg = -1, n_hess = -1, zeta = NaN)
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

println("="^90); println("STAGE 1: W=20,000 classification sweep"); println("="^90); flush(stdout)
t_ctx20 = @elapsed common20 = build_common(20_000)
@printf("W=20,000 context build: %.2fs\n", t_ctx20); flush(stdout)

w0_str = read(joinpath(D4X, "eval18_captured_point_2026-08-02.txt"), String)
w_eval18 = Vector{Float64}(eval(Meta.parse(w0_str)))
@assert length(w_eval18) == length(common20.w_calib) "eval18 point length $(length(w_eval18)) != flexible_CM outer_dim $(length(common20.w_calib))"

println("\n--- flexible_CM: interpolating calib -> eval18 ---"); flush(stdout)
aug_cm20 = build_cm_augmented_obj_archB(common20.ctx, CS; L = 50, contrasts = :anchored, base_obj = common20.reduced_obj0, profiled_layout = common20.layout)
flexcm_results20 = NamedTuple[]
for k in (0.3, 0.6, 1.0)
    w_k = common20.w_calib .+ k .* (w_eval18 .- common20.w_calib)
    decoded_k = decode_outer_profiled(w_k, common20.ctx, common20.pe)
    cctx_k = build_cm_bin_ctx(common20.ctx, aug_cm20; profiled_layout = common20.layout, inner_fg_backend = :dense_reference, threaded_bins = true)
    r = try_status("flexible_CM k=$k", () -> reduced_cm_base_state(decoded_k.xf, common20.ctx, common20.layout, cctx_k))
    push!(flexcm_results20, r)
end

println("\n--- origin-ZC: additive log-A shift from calibration ---"); flush(stdout)
D20v = common20.D
layout_o20 = OriginByPowerLayout(D20v, 1, 0)
aug_oz20 = build_originzc_augmented_obj(common20.ctx, CS, layout_o20; base_obj = common20.reduced_obj0, profiled_layout = common20.layout)
νvec0_oz = fill(1.0, D20v)
oz_results20 = NamedTuple[]
for shift in (3.0, 6.0, 10.0)
    w_s = copy(common20.w_calib); w_s[2:end] .+= shift
    decoded_s = decode_outer_profiled(w_s, common20.ctx, common20.pe)
    octx_s = build_originzc_core_hess_ctx(aug_oz20, common20.ctx; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, profiled_layout = common20.layout)
    r = try_status("origin-ZC shift=$shift", () -> reduced_originzc_base_state(decoded_s.xf, common20.ctx, common20.layout, octx_s, νvec0_oz))
    push!(oz_results20, r)
end

println("\n--- CM+ZC: additive log-A shift from calibration ---"); flush(stdout)
const K_MEAN, K_PAIR, L_GRID = 1, 0, 3
aug_cz20 = build_cm_meanzc_augmented_obj(common20.ctx, CS; L = L_GRID, K_mean = K_MEAN, K_pair = K_PAIR,
    base_obj = common20.reduced_obj0, profiled_layout = common20.layout)
νvec0_cz = fill(1.0, K_MEAN)
cz_results20 = NamedTuple[]
for shift in (3.0, 6.0, 10.0)
    w_s = copy(common20.w_calib); w_s[2:end] .+= shift
    decoded_s = decode_outer_profiled(w_s, common20.ctx, common20.pe)
    cctx_s = build_cm_meanzc_bin_ctx(common20.ctx, aug_cz20; core_hessian_backend = :exact_winner_pair_parallel,
        zc_cross_hessian_backend = :winner_bin, threaded_bins = true, inner_fg_backend = :dense_reference, profiled_layout = common20.layout)
    r = try_status("CM+ZC shift=$shift", () -> reduced_meanzc_base_state(decoded_s.xf, νvec0_cz, common20.ctx, common20.layout, cctx_s))
    push!(cz_results20, r)
end

println("\n" * "="^90); println("STAGE 1 SUMMARY (W=20,000)"); println("="^90)
for r in vcat(flexcm_results20, oz_results20, cz_results20)
    cls = r.status == 0 ? "FEASIBLE" : (r.status in (-300,) ? "INFEASIBLE(fast-or-slow, see wall)" : "STALL/OTHER($(r.status))")
    @printf("%-22s status=%-5d wall=%6.2fs  -> %s\n", r.label, r.status, r.wall, cls)
end
flush(stdout)

@printf("\nTOTAL WALL (stage 1): %.1fs\n", time() - t0)
flush(stdout)
