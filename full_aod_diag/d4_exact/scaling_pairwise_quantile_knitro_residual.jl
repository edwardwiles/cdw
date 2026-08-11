# ================================================================================================
# WHERE IS THE NON-CALLBACK TIME INSIDE KN_solve GOING?  (2026-08-11)
#
# Profiling attributes 60-90% of an L=10 inner solve to "KNITRO itself", but that number is a
# RESIDUAL -- KN_solve wall minus the callback time we measure from inside. It establishes only
# that the time is NOT in our callbacks. Calling it "the dense KKT factorization" is an inference,
# and this file is the experiment that tests it instead of asserting it.
#
# THE DISCRIMINATOR. The inner dual dimension is n = 382 + (L-1)*20 + (L-1)^2*190, so sweeping L
# sweeps n over more than an order of magnitude. Two hypotheses predict very different exponents:
#
#   O(n^3)  a dense KKT factorization per interior-point iteration
#   O(n^2)  handling the dense packed Hessian we hand KNITRO every callback
#           (KN_DENSE_ROWMAJOR, n(n+1)/2 entries = ~1 GB at L=10)
#
# Over L=2..7, n spans 592 -> 7342 (12.4x), so n^2 and n^3 differ by ~154x vs ~1907x in predicted
# growth. That separates cleanly; a fitted exponent near 2 points at the hand-off, near 3 at the
# algorithm.
#
# METHOD. One process, one context, one WARM solve per L (a throwaway solve first, because the
# first solve of a process carries a 3.6x JIT tax -- measured, and the trap that made every earlier
# L=10 timing wrong). Callback time is measured from inside the real production callbacks; the
# residual is the difference. Residual is normalised per interior-point iteration, taken as the
# Hessian callback count (KNITRO evaluates the Hessian once per IP iteration under hessopt=exact).
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=16 julia --project=. \
#     full_aod_diag/d4_exact/scaling_pairwise_quantile_knitro_residual.jl [W] [Lmin] [Lmax]
# ================================================================================================
_D4E = joinpath(@__DIR__)
for f in ["draw_design.jl", "context_real_d20.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "compressed_factual_buffer_reuse.jl", "cm_screen_bridge.jl", "gradient_workspace.jl",
          "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_outer_driver.jl", "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "incumbent_logic.jl", "cm_checkpoint.jl",
          "cm_originzc_target_layout.jl", "zc_restriction_operator_ragged.jl", "cm_aspace_coordinate.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl", "cm_originzc_config.jl",
          "cm_originzc_checkpoint.jl", "direction_bounds.jl", "cm_frechet_hessian.jl", "cm_frechet_level.jl",
          "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl", "country_resolve.jl",
          "cross_delta_cache.jl", "compressed_moments.jl", "canonical_price_precompute_workspace.jl",
          "hard_score_b_cache.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "lfix_buffer_reuse.jl", "lfix_base_workspace_pooled.jl", "lfix_kbplus_workspace.jl",
          "bandwidth_cache_policy.jl", "fast_range_screen.jl", "dual_bank.jl", "negative_cache.jl",
          "dual_bank_ab_harness.jl", "reusable_context.jl", "organic_failure_capture.jl",
          "multistart_seed_generator.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_mass_gradient.jl",
          "pairwise_quantile_outer_production.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, SpecialFunctions, Statistics

lp(xs...) = (println(xs...); flush(stdout))
const W_S = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const LMIN = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const LMAX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 7
const GRAV = default_gravity_exclude_cells_brazil_korea()

lp("="^104)
lp("KNITRO NON-CALLBACK RESIDUAL: scaling in n, W=", W_S, ", L=", LMIN, "..", LMAX)
lp("="^104)

ctx_raw = d20_real_setup_design(W = W_S, δ = 50.0, find_smallest = true, draw_design = :sobol_randomized,
    draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
geo = build_aspace_geometry(ctx)
xf = x_free_from_w(vcat(cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)[1],
    cm_z_from_a(cm_w0_from_calibration(ctx, geo.pe, :powered_aspace)[2:end], cm_fixed_theta(ctx),
                precompute_cm_aspace_xy(ctx), geo.pe)), geo.pe)

mutable struct T; fg::Float64; fgn::Int; hs::Float64; hsn::Int; end

function timed_solve(ctx_cm, layout, mass0, tm::T)
    obj = ctx_cm.obj
    prime_operator!(obj, CS.reconstruct_full(xf, ctx_cm.m), ctx_cm.pq_econ_ctx, ctx_cm.pq_core_cf_ref)
    st = PairwiseQuantileOperatorState(obj, obj.outer_constr_index - 1 - n_total_rows(ctx.D, layout.L),
        ctx_cm.pq_op, ctx_cm.pq_mass_state, ctx_cm.pq_core_cf_ref)
    reset_for_solve!(st, mass0, layout)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))
        fg_raw = _callbackEvalFG_inner_pairwisequantile!
        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(
            (a,b,c,d,e) -> (t=time(); r=fg_raw(a,b,c,d,e); tm.fg += time()-t; tm.fgn += 1; r), health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)
        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hraw = pairwisequantile_hess_cb_builder(ctx_cm.pq_hess_ctx)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callback_health_guard(
                (a,b,c,d,e) -> (t=time(); r=hraw(a,b,c,d,e.obj); tm.hs += time()-t; tm.hsn += 1; r), health))
        end
        t = @elapsed KNITRO.KN_solve(kc)
        nStatus, _, x, _ = KNITRO.KN_get_solution(kc)
        KNITRO.KN_free(kc)
        return t, nStatus
    finally
        CS.guard_exit_inner_solve!()
    end
end

rows = NamedTuple[]
for L in LMIN:LMAX
    layout = PairwiseQuantileMassLayout(ctx.D, L)
    pcx = build_pairwise_quantile_production_context(ctx, layout;
        cutoff_source = :frechet_theoretical, min_bin_count = max(10, W_S ÷ (2 * L^2)))
    ctx_cm = pcx.ctx_cm
    mass0 = uniform_mass_raw(layout)
    n = ctx_cm.obj.outer_constr_index
    timed_solve(ctx_cm, layout, mass0, T(0,0,0,0))              # throwaway: JIT + first-touch
    tm = T(0,0,0,0)
    wall, st = timed_solve(ctx_cm, layout, mass0, tm)
    resid = wall - tm.fg - tm.hs
    push!(rows, (L=L, n=n, wall=wall, fg=tm.fg, hs=tm.hs, hsn=tm.hsn, resid=resid,
                 per_it = resid / max(tm.hsn,1)))
    @printf("L=%2d  n=%6d  wall=%8.2fs  FG=%6.2fs  Hess=%7.2fs (%d calls)  RESID=%8.2fs  resid/IPit=%7.3fs  status=%d\n",
            L, n, wall, tm.fg, tm.hs, tm.hsn, resid, resid/max(tm.hsn,1), st)
end

lp("\n", "="^104)
lp("FITTED EXPONENT of the non-callback residual per interior-point iteration, vs n")
ln = [log(r.n) for r in rows]; lt = [log(max(r.per_it, 1e-9)) for r in rows]
nbar = mean(ln); tbar = mean(lt)
slope = sum((ln .- nbar) .* (lt .- tbar)) / sum((ln .- nbar).^2)
@printf("  slope = %.2f      (O(n^2) hand-off => ~2.0 ;  O(n^3) dense factorization => ~3.0)\n", slope)
lp("")
@printf("%4s %8s %12s %12s %12s\n", "L", "n", "resid/IPit", "vs n^2 pred", "vs n^3 pred")
r0 = rows[1]
for r in rows
    @printf("%4d %8d %12.4f %12.4f %12.4f\n", r.L, r.n, r.per_it,
            r0.per_it * (r.n/r0.n)^2, r0.per_it * (r.n/r0.n)^3)
end
lp("="^104)
