# Task "fixed-state FULL-vs-REDUCED inner A/B", step 6.1: genuine cold solve, both arms, from the
# SAME sentinel point. Fresh ctx each call, no resume_from, no dual bank (neither formulation's
# real drivers have one wired for these families -- confirmed by the outer-completion session's
# own grep, see FamilyRegistry.jl's origin_zc row notes), canonical formulation-specific cold
# start (REDUCED: the sentinel's own native w; FULL: the SAME decoded state pushed through
# full_coordinate_bridge.jl into that family's REAL production A_coordinate_mode).
#
# Construction mirrors bin/run_profiled_model.jl's own `_run_reduced`/`_run_reduced_zc_free_nu`/
# `_run_full_unrestricted`/`_run_full_originzc`/generic-CM-FULL dispatch EXACTLY (same functions,
# same call shapes, same real Korea/Brazil anchor override) -- reused, not reinvented -- with the
# ONE deliberate difference this task requires: the starting point is the caller's sentinel state,
# not always re-derived from ctx.θ0_up's own calibration point.
#
# Never writes into results/canonical_runner/ (production result directories) -- every checkpoint/
# manifest this script writes goes under repo_scratch.

const D4X = joinpath(dirname(dirname(dirname(@__DIR__))), "full_aod_diag", "d4_exact")
const REPO_ROOT = dirname(dirname(dirname(@__DIR__)))

for f in ["context.jl", "c10_d20_production_driver.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "no_dense_g_counters.jl", "economic_operator.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl",
          "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_lookup_kernels.jl", "cm_originzc_lookup_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl", "production_bundle_api.jl", "country_resolve.jl",
          "cm_exact_cache_production.jl", "cm_checkpoint.jl", "lfix_cm_cplus.jl", "cm_originzc_checkpoint.jl",
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
          "profiled_restricted_family_adapters_2026-08-02.jl",
          "profiled_originzc_family_adapter_2026-08-02.jl",
          "profiled_cmzc_family_adapter_2026-08-02.jl",
          "profiled_zc_lane_point_evaluators_2026-08-02.jl",
          "profiled_reduced_basis_cache_bank_2026-08-02.jl",
          "profiled_screen_bridge_2026-08-02.jl",
          "dual_bank.jl", "cm_dual_bank_production.jl",
          "profiled_production_outer_runner_2026-08-01.jl",
          "profiled_ab_comparability_and_plumbing_2026-08-01.jl",
          "cm_aspace_coordinate.jl", "profiled_powered_relative_a_2026-08-04.jl",
          "profiled_coordinate_mode_dispatch_2026-08-04.jl",
          "profiled_production_outer_constrained_2026-08-02.jl",
          "profiled_zc_free_eta_2026-08-04.jl",
          "profiled_zc_free_nu_production_driver_2026-08-04.jl",
          # unrestricted FULL
          "flexible_theta.jl", "flexible_theta_aspace_production.jl", "outer_coordinate_layout.jl",
          "c10_d20_production_driver_unified.jl"]
    include(joinpath(D4X, f))
end

include(joinpath(@__DIR__, "frozen_manifest.jl"))
using .FixedStateInnerABFrozenManifest
include(joinpath(@__DIR__, "full_coordinate_bridge.jl"))

using Printf

const KOREA_IDX, BRAZIL_IDX = 14, 3
const CANONICAL_FAMILIES = (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc)

"""
    ReducedSetup

Everything `run_profiled_upper_constrained[_free_nu]` needs, built ONCE per (family, mode) --
mirrors `bin/run_profiled_model.jl`'s own `_run_reduced`/`_run_reduced_zc_free_nu` construction,
same functions, same order, same Korea/Brazil override.
"""
function build_reduced_setup(family::Symbol, sci)
    ctx = d20_real_setup_design(W = sci.W, δ = 1.0, find_smallest = true,
        draw_design = sci.draw_design, draw_seed = sci.draw_seed,
        destination_sample = sci.destination_sample,
        exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
        gravity_exclude_cells = sci.gravity_exclude_cells, σHat = sci.sigma)
    D = ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(KOREA_IDX => BRAZIL_IDX))
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*ctx.D_dest], D, ctx.D_dest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    gp0 = θ0[3+D]
    w0_calib = reduce_to_w_profiled(gp0, z_calib, pe)

    θ_full_calib = CS.reconstruct_full(ctx.θ0_up[ctx.free_idx], ctx.m)
    cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
    has_france = cf_probe.cf_col > 0
    layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
    reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

    if family in (:origin_zc, :cm_meanzc)
        D_ = ctx.D
        if family == :origin_zc
            layout_o = OriginByPowerLayout(D_, 1, 0)
            νvec0 = fill(1.0, D_)
            aug = build_originzc_augmented_obj(ctx, CS, layout_o; base_obj = reduced_obj0, profiled_layout = layout)
            octx = build_originzc_core_hess_ctx(aug, ctx; core_hessian_backend = :exact_winner_pair_parallel,
                zc_cross_hessian_backend = :winner_bin, profiled_layout = layout)
            fctx = build_originzc_family_ctx(ctx, spec, pe, layout, aug)
            pes = OriginZCPointEvalState(octx, νvec0)
            evaluate_fn_free_nu = (w, eta, ff, p) -> evaluate_profiled_originzc_point(w, eta, ff, p)
            gradient_fn_free_nu = (w, eta, c, ff, ev; threaded_gradient) -> reduced_originzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = threaded_gradient)
        else
            K_MEAN, K_PAIR = sci.K_mean, sci.K_pair
            νvec0 = fill(1.0, max(K_MEAN, 1))
            aug = build_cm_meanzc_augmented_obj(ctx, CS; L = sci.L, K_mean = K_MEAN, K_pair = K_PAIR,
                base_obj = reduced_obj0, profiled_layout = layout)
            cctx = build_cm_meanzc_bin_ctx(ctx, aug; core_hessian_backend = :exact_winner_pair_parallel,
                zc_cross_hessian_backend = :winner_bin, threaded_bins = false, inner_fg_backend = :dense_reference,
                profiled_layout = layout)
            bins_u32 = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
            fctx = build_cmzc_family_ctx(ctx, spec, pe, layout, aug, cctx, bins_u32)
            pes = CMZCPointEvalState(cctx, νvec0)
            evaluate_fn_free_nu = (w, eta, ff, p) -> evaluate_profiled_cmzc_point(w, eta, ff, p)
            gradient_fn_free_nu = (w, eta, c, ff, ev; threaded_gradient) -> reduced_cmzc_outer_gradient_with_eta(w, eta, c, ff, ev; threaded = threaded_gradient)
        end
        eta_bounds = originzc_default_nu_bounds(ctx, fctx.zc_layout)
        return (ctx = ctx, spec = spec, pe = pe, layout = layout, family = family,
            fctx = fctx, pes = pes, evaluate_fn_free_nu = evaluate_fn_free_nu,
            gradient_fn_free_nu = gradient_fn_free_nu, eta_bounds = eta_bounds, is_free_nu = true)
    else
        fctx, evaluate_fn = if family == :unrestricted
            ev0 = evaluate_profiled_point(w0_calib, ctx, spec, pe)
            f = build_unrestricted_family_ctx(ctx, spec, pe, ev0)
            (f, (w, ff) -> evaluate_profiled_point(w, ff.ctx, ff.spec, ff.pe))
        elseif family == :flexible_cm
            aug = build_cm_augmented_obj_archB(ctx, CS; L = sci.L, contrasts = :anchored,
                base_obj = reduced_obj0, profiled_layout = layout)
            cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = true)
            f = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx)
            (f, (w, ff) -> evaluate_profiled_flexcm_point(w, ff))
        else # :common_frechet
            aug = build_cm_frechet_augmented_obj_archB(ctx, CS; L = sci.L, contrasts = :anchored,
                base_obj = reduced_obj0, profiled_layout = layout)
            cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference,
                threaded_bins = false, core_hessian_backend = :dense_reference, cm_cross_hessian_backend = :winner_bin)
            f = build_frechet_family_ctx(ctx, spec, pe, layout, cctx, aug.level_targets)
            (f, (w, ff) -> evaluate_profiled_frechet_point(w, ff))
        end
        return (ctx = ctx, spec = spec, pe = pe, layout = layout, family = family,
            fctx = fctx, evaluate_fn = evaluate_fn, is_free_nu = false)
    end
end

"""
    reduced_cold_solve(setup, w0; eta_nu0=nothing, delta, maxtime_real, threaded_gradient, ckpt_dir, run_id)

Genuine cold solve (`resume_from = nothing`, fresh checkpoint path under `ckpt_dir` -- NEVER
`results/canonical_runner/`) via the REAL production driver, called exactly as
`bin/run_profiled_model.jl` calls it.
"""
function reduced_cold_solve(setup, w0::Vector{Float64}; eta_nu0::Union{Nothing,Vector{Float64}} = nothing,
        delta::Float64, maxtime_real::Float64, threaded_gradient::Bool, ckpt_dir::String, run_id::String)
    mkpath(ckpt_dir)
    ckpt_path = joinpath(ckpt_dir, "checkpoint.jls")
    if setup.is_free_nu
        eta0 = eta_nu0 === nothing ? zeros(length(setup.eta_bounds)) : eta_nu0
        gfn = (w, eta, c, ff, ev) -> setup.gradient_fn_free_nu(w, eta, c, ff, ev; threaded_gradient = threaded_gradient)
        result = run_profiled_upper_constrained_free_nu(run_id, w0, eta0;
            fctx = setup.fctx, evaluate_fn_free_nu = setup.evaluate_fn_free_nu, gradient_fn_free_nu = gfn,
            pes = setup.pes, ctx = setup.ctx, pe = setup.pe, eta_bounds = setup.eta_bounds,
            delta = delta, maxtime_real = maxtime_real, hessopt_tag = "sr1",
            a_coordinate_mode = :profiled_pivot_anchor_relative,
            checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = nothing, verbose = false)
    else
        result = run_profiled_upper_constrained(run_id, w0; fctx = setup.fctx, evaluate_fn = setup.evaluate_fn,
            ctx = setup.ctx, pe = setup.pe, delta = delta, maxtime_real = maxtime_real, hessopt_tag = "sr1",
            a_coordinate_mode = :profiled_pivot_anchor_relative,
            checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = nothing, verbose = false,
            threaded_gradient = threaded_gradient)
    end
    return result
end

"""
    full_cold_solve(family, gp, z_full, ctx; eta_nu=nothing, delta, maxtime_real, ckpt_dir, meanzc_K_mean)

Genuine cold solve on the FULL side, from the SAME decoded state, bridged into that family's REAL
production `A_coordinate_mode` (`full_coordinate_bridge.jl`), via the REAL production driver
(`run_polish_checkpointed_unified`/`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`),
called exactly as `bin/run_profiled_model.jl` calls it (same kwargs, same layout construction).
"""
function full_cold_solve(family::Symbol, gp::Float64, z_full::AbstractMatrix{Float64}, ctx;
        eta_nu::Union{Nothing,Vector{Float64}} = nothing, delta::Float64, maxtime_real::Float64,
        ckpt_dir::String, sci, meanzc_K_mean::Int = 1)
    mkpath(ckpt_dir)
    w0_econ = full_w0_from_state(family, gp, z_full, ctx)

    if family === :unrestricted
        layout = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace, gp_coordinate_mode = :raw)
        result = run_polish_checkpointed_unified("full_cold", true, w0_econ; layout = layout, ckpt_dir = ckpt_dir,
            W_in = sci.W, delta_in = delta, draw_design_in = sci.draw_design, draw_seed_in = sci.draw_seed,
            destination_sample = sci.destination_sample, exclude_diagonal_gravity = sci.exclude_diagonal_gravity,
            gravity_exclude_cells = sci.gravity_exclude_cells, σHat = sci.sigma,
            maxtime_real = maxtime_real, resume_from = nothing)
        return (knitro_status = result.knitro_status, n_eval = result.n_eval, n_grad = result.n_grad_calls,
            wall = result.wall_ext, best = result.best_feasible)
    elseif family === :origin_zc
        D = ctx.D
        w0 = eta_nu === nothing ? vcat(w0_econ, zeros(D)) : vcat(w0_econ, eta_nu)
        nu_bounds = originzc_default_nu_bounds(ctx, OriginByPowerLayout(D, 1, 0))
        result = run_originzc_upper_checkpointed(w0; find_smallest = true, W = sci.W, delta = delta,
            draw_design = sci.draw_design, draw_seed = sci.draw_seed, destination_sample = sci.destination_sample,
            exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
            σHat = sci.sigma, distribution_restriction = :origin_specific_moments, K_mean = 1, K_pair = 0,
            power_target_layout = :origin_by_power, nu_bounds = nu_bounds, A_coordinate_mode = :legacy_z,
            outer_direct_hessopt = :sr1, maxtime_real = maxtime_real, ckpt_dir = ckpt_dir, run_id = "full_cold",
            label = "full_cold", checkpoint_interval_s = 60.0, resume_from = nothing, cm_gradient_backend = :cplus, verbose = false)
        return (knitro_status = result.knitro_status, n_eval = result.n_eval, n_grad = result.n_grad,
            wall = result.wall, best = result.best)
    else
        marginal_restriction = family == :common_frechet ? :common_frechet : :common_flexible
        cm_extension = family == :cm_meanzc ? :cm_plus_moments : :cm_only
        mzc_K_mean = family == :cm_meanzc ? max(meanzc_K_mean, 1) : 0
        mzc_K_pair = family == :cm_meanzc ? sci.K_pair : 0
        probs = nested_grid_sequence([sci.L])[sci.L]
        w0 = family == :cm_meanzc ? vcat(w0_econ, eta_nu === nothing ? zeros(mzc_K_mean) : eta_nu) : w0_econ
        result = run_cm_upper_checkpointed(w0; find_smallest = true, W = sci.W, delta = delta,
            draw_design = sci.draw_design, draw_seed = sci.draw_seed, L = sci.L, contrasts = :anchored, probs = probs,
            marginal_restriction = marginal_restriction, cm_extension = cm_extension,
            meanzc_K_mean = mzc_K_mean, meanzc_K_pair = mzc_K_pair, maxtime_real = maxtime_real,
            ckpt_dir = ckpt_dir, label = "full_cold", resume_from = nothing, σHat = sci.sigma,
            exclude_diagonal_gravity = sci.exclude_diagonal_gravity, gravity_exclude_cells = sci.gravity_exclude_cells,
            destination_sample = sci.destination_sample)
        return (knitro_status = result.knitro_status, n_eval = result.n_eval, n_grad = result.n_grad,
            wall = result.wall, best = result.best)
    end
end
