# ================================================================================================
# WHERE THE TIME ACTUALLY GOES, at two levels (2026-08-11).
#
# PART 1 -- one INNER solve, split into:  FG callbacks | Hessian callbacks | KNITRO itself
# PART 2 -- one OUTER iteration, split into:  cb_F! (inner solve + verify) | cb_G! (outer gradient)
#           | everything else, with each of those broken down further.
#
# Everything is measured by timing the REAL production callbacks, registered against a real
# KN_solve. Nothing is re-implemented at top level -- that is exactly the mistake that produced the
# bogus 86 s "packed write" number on 2026-08-10 (a hand-inlined copy of a production loop reading
# non-const globals ran 45x slower than the production loop it was supposed to represent; see
# `microbench_packed_write.jl` and status doc section 9.2). The only thing this file adds around the
# production callbacks is an accumulator; the callbacks themselves are the production ones.
#
# KNITRO-internal time is obtained by DIFFERENCE (`KN_solve` wall minus the callback time measured
# inside it), which is the only way to get it -- KNITRO does not report it -- and is why the
# callbacks are timed from inside rather than estimated.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=16 julia --project=. \
#     full_aod_diag/d4_exact/profile_pairwise_quantile_budget.jl [W] [L] [cutoff_source] [inner_opt]
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
          "pairwise_quantile_outer_production.jl", "pairwise_quantile_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Printf, SpecialFunctions

lp(xs...) = (println(xs...); flush(stdout))
const W_P    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const L_P    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
const CUTSRC = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :frechet_theoretical
const INNER_OPT = length(ARGS) >= 4 ? ARGS[4] : ""
const GRAV   = default_gravity_exclude_cells_brazil_korea()

lp("="^100)
lp("PAIRWISE-QUANTILE TIME BUDGET: W=", W_P, " L=", L_P, " cutoff_source=:", CUTSRC,
   isempty(INNER_OPT) ? "" : "  inner_opt=$(basename(INNER_OPT))")
lp("JULIA_NUM_THREADS=", Threads.nthreads())
lp("="^100)

t_ctx = @elapsed begin
    ctx_raw = d20_real_setup_design(W = W_P, δ = 50.0, find_smallest = true, draw_design = :sobol_randomized,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = GRAV, σHat = 3.0, inner_lower_limit = -10.0)
    global ctx = attach_compressed_factual_workspace(ctx_raw, ctx_raw.D, ctx_raw.D_dest, ctx_raw.W)
end
lp("context built in ", round(t_ctx, digits = 1), "s")

layout = PairwiseQuantileMassLayout(ctx.D, L_P)
pcx = build_pairwise_quantile_production_context(ctx, layout;
    cutoff_source = CUTSRC, min_bin_count = max(10, W_P ÷ (2 * L_P^2)))
ctx_cm = pcx.ctx_cm
isempty(INNER_OPT) || (ctx_cm.obj.inner_loop_opt = INNER_OPT)
geo = build_aspace_geometry(ctx)
pe = geo.pe
w_cal = cm_w0_from_calibration(ctx, pe, :powered_aspace)
theta_cm = cm_fixed_theta(ctx); xy_cm = precompute_cm_aspace_xy(ctx)
xf = x_free_from_w(vcat(w_cal[1], cm_z_from_a(w_cal[2:end], theta_cm, xy_cm, pe)), pe)
mass0 = uniform_mass_raw(layout)
n_inner = ctx_cm.obj.outer_constr_index
lp("inner dual dimension n = ", n_inner, "   n_total_rows = ", n_total_rows(ctx.D, L_P))

# ================================================================================================
# PART 1 -- inner solve: FG | Hessian | KNITRO
# ================================================================================================
"Accumulator threaded through the wrapped production callbacks. Wall-clock only; the callbacks
themselves are the production ones, untouched."
mutable struct InnerTimers
    fg_t::Float64;   fg_n::Int
    hess_t::Float64; hess_n::Int
end

"""
Registers the SAME production FG functor and Hessian callback against a real `KN_solve`, wrapped in
timing accumulators. This mirrors `inner_loop_KNITRO_pairwisequantile_operator`
(pairwise_quantile_production.jl) line for line -- it exists only because that function builds and
registers its callbacks internally, leaving no seam to time them through.
"""
function timed_inner_solve(obj, st::PairwiseQuantileOperatorState, hess_ctx, tm::InnerTimers)
    CS.guard_enter_inner_solve!()
    health = CallbackHealthRecord()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        x_initial = CS.inner_loop_initial_values(obj)
        KNITRO.KN_set_var_primal_init_values_all(kc, x_initial)

        fg_raw = _callbackEvalFG_inner_pairwisequantile!
        fg_timed = (kc2, cb2, req, res, up) -> begin
            t = time(); r = fg_raw(kc2, cb2, req, res, up)
            tm.fg_t += time() - t; tm.fg_n += 1
            return r
        end
        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], callback_health_guard(fg_timed, health))
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            hess_raw = pairwisequantile_hess_cb_builder(hess_ctx)
            hess_timed = callback_health_guard((kc2, cb2, req, res, up) -> begin
                t = time(); r = hess_raw(kc2, cb2, req, res, up.obj)
                tm.hess_t += time() - t; tm.hess_n += 1
                return r
            end, health)
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_timed)
        end

        t_solve = @elapsed KNITRO.KN_solve(kc)
        nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        assert_no_fake_success!("timed_inner_solve", health, nStatus, st.n_fg_calls, x_initial, x)
        KNITRO.KN_free(kc)
        return nStatus, x, t_solve
    finally
        CS.guard_exit_inner_solve!()
    end
end

function one_inner_solve()
    obj = ctx_cm.obj
    θ = CS.reconstruct_full(xf, ctx_cm.m)
    t_prime = @elapsed prime_operator!(obj, θ, ctx_cm.pq_econ_ctx, ctx_cm.pq_core_cf_ref)
    st = PairwiseQuantileOperatorState(obj, obj.outer_constr_index - 1 - n_total_rows(ctx.D, L_P),
        ctx_cm.pq_op, ctx_cm.pq_mass_state, ctx_cm.pq_core_cf_ref)
    reset_for_solve!(st, mass0, layout)
    tm = InnerTimers(0.0, 0, 0.0, 0)
    nStatus, x, t_solve = timed_inner_solve(obj, st, ctx_cm.pq_hess_ctx, tm)
    return (nStatus = nStatus, x = x, t_solve = t_solve, t_prime = t_prime, tm = tm)
end

lp("\n", "="^100)
lp("PART 1: INNER SOLVE BREAKDOWN  (first run is JIT-warming and is discarded)")
lp("="^100)
one_inner_solve()                      # warm-up: compile everything, discard
r1 = one_inner_solve()
tm = r1.tm
t_knitro = r1.t_solve - tm.fg_t - tm.hess_t
@printf("\nKN_solve wall                       %10.2f s   (nStatus=%d)\n", r1.t_solve, r1.nStatus)
@printf("  FG callbacks       %5d calls   %10.2f s  (%5.1f%%)   %8.3f s/call\n",
        tm.fg_n, tm.fg_t, 100 * tm.fg_t / r1.t_solve, tm.fg_t / max(tm.fg_n, 1))
@printf("  Hessian callbacks  %5d calls   %10.2f s  (%5.1f%%)   %8.3f s/call\n",
        tm.hess_n, tm.hess_t, 100 * tm.hess_t / r1.t_solve, tm.hess_t / max(tm.hess_n, 1))
@printf("  KNITRO itself (by difference)      %10.2f s  (%5.1f%%)\n",
        t_knitro, 100 * t_knitro / r1.t_solve)
@printf("\nprime_operator! (outside KN_solve)  %10.2f s\n", r1.t_prime)

# ================================================================================================
# PART 2 -- outer iteration: cb_F! | cb_G! | rest
# ================================================================================================
lp("\n", "="^100)
lp("PART 2: OUTER ITERATION BREAKDOWN")
lp("="^100)

# ---- cb_F!: what the driver's objective callback does, split solve vs verify -------------------
t_F = @elapsed begin
    global baseF, verifyF = archPQ_verified_state(xf, mass0, ctx_cm)
end
# archPQ_verified_state = archPQ_base_state (the KNITRO solve, already measured) + the verifier.
t_verify = @elapsed begin
    op = ctx_cm.pq_op
    ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
    cf = ctx_cm.pq_core_cf_ref[]
    econ_ws_v = economic_operator_workspace(cf)
    verify_inner_solution_operator_pairwisequantile!(baseF.ζstar, baseF.λstar, cf, op,
        ctx_cm.pq_mass_state, op.W, economic_forward!, economic_transpose!, econ_ws_v,
        ctx_cm.obj.Psi!, ctx_cm.obj.dPsi!, ncore1)
end
@printf("\ncb_F!  (value: inner solve + verification)   %10.2f s\n", t_F)
@printf("   of which inner KN_solve (Part 1)          %10.2f s  (%5.1f%%)\n",
        r1.t_solve, 100 * r1.t_solve / t_F)
@printf("   of which independent verifier             %10.2f s  (%5.1f%%)\n",
        t_verify, 100 * t_verify / t_F)
@printf("   of which prime_operator! + rest           %10.2f s  (%5.1f%%)\n",
        t_F - r1.t_solve - t_verify, 100 * (t_F - r1.t_solve - t_verify) / t_F)

# ---- cb_G!: the outer gradient, component by component, FULLY attributed --------------------
# Every line below calls a PRODUCTION function. Nothing is re-implemented at top level (that is what
# produced the bogus 86 s packed-write number on 2026-08-10). Where a production function has no
# internal seam to time through, its dominant kernel is called directly -- and labelled as such.
econ_ws = get_or_build_econ_a_grad_ws(W_P)
D2_econ = ctx.D * ctx.D_dest

# (a) build_lfix_base_cache -- the SHARED economic cache. Identical call in
#     cm_originzc_production.jl:216, cm_meanzc_production.jl:454 and this family's
#     pairwise_quantile_outer_production.jl:342, so this cost is common to every restricted family.
#     It contains NO Threads.@threads and allocates price0 + pTsigma0 as W x D x Ddest each.
ensure_pq_masses!(ctx_cm, mass0)
build_lfix_base_cache(xf, ctx_cm, baseF; validate_dense = false)   # warm
t_lfix = @elapsed cache0 = build_lfix_base_cache(xf, ctx_cm, baseF; validate_dense = false)
a_lfix = @allocated build_lfix_base_cache(xf, ctx_cm, baseF; validate_dense = false)

# (a1) its dominant kernel, called directly: the W x D x Ddest price / p^(T sigma) fill.
function _time_price_fill(ctx_l, base_l, D_l, Ddest_l, W_l)
    price0 = Array{Float64}(undef, W_l, D_l, Ddest_l)
    pT0 = Array{Float64}(undef, W_l, D_l, Ddest_l)
    for d in 1:Ddest_l, o in 1:D_l
        price_and_pTsigma_cell!(@view(price0[:, o, d]), @view(pT0[:, o, d]), base_l.θ_full0, ctx_l, o, d)
    end
    return price0[1, 1, 1] + pT0[1, 1, 1]
end
_time_price_fill(ctx_cm, baseF, ctx.D, ctx.D_dest, W_P)   # warm
t_price = @elapsed _time_price_fill(ctx_cm, baseF, ctx.D, ctx.D_dest, W_P)

# (b) this restriction's q0 fold on top of the shared cache (operator forward + the exact check)
t_fold = @elapsed build_lfix_base_cache_pairwise_quantile(xf, ctx_cm, baseF; verify = verifyF)
t_fold_only = max(t_fold - t_lfix, 0.0)
cacheG = build_lfix_base_cache_pairwise_quantile(xf, ctx_cm, baseF; verify = verifyF)

# (c) economic_A_gradient! -- the SHARED (g, A_od) block. Timed BOTH with a cold bandwidth cache
#     (what the first gradient at a new outer point pays) and with a warm one (every later call),
#     because production carries `bandwidth_cache` across the whole run.
g_econ = zeros(D2_econ)
bwc_cold = Dict{Int,Float64}()
t_econ_cold = @elapsed economic_A_gradient!(g_econ, baseF, ctx_cm, pe, econ_ws; cache = cacheG,
    threaded = true, h_mode = :cached, bandwidth_cache = bwc_cold)
t_econ_warm = @elapsed economic_A_gradient!(g_econ, baseF, ctx_cm, pe, econ_ws; cache = cacheG,
    threaded = true, h_mode = :cached, bandwidth_cache = bwc_cold)
# and the same call with threading OFF, to show the threading is actually reaching it
g_econ2 = zeros(D2_econ)
t_econ_serial = @elapsed economic_A_gradient!(g_econ2, baseF, ctx_cm, pe, econ_ws; cache = cacheG,
    threaded = false, h_mode = :cached, bandwidth_cache = bwc_cold)
@assert g_econ == g_econ2 "threaded and serial economic_A_gradient! disagree"

# (d) this restriction's own mass gradient
pairwise_quantile_mass_gradient_vec(baseF, verifyF, ctx_cm, mass0)   # warm
t_mass = @elapsed pairwise_quantile_mass_gradient_vec(baseF, verifyF, ctx_cm, mass0)

# (e) THE AUTHORITATIVE MEASUREMENT: the driver's own call, instrumented from inside so the
#     decomposition SUMS TO THE TOTAL. Everything above is a diagnostic on individual pieces (and
#     the build_lfix_base_cache line above deliberately times the ALLOCATING variant, to show what
#     the persistent-workspace routing avoids); this is what production actually executes.
tmr = PQGradTimers()
pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe; base = baseF, verify = verifyF,
    econ_ws = econ_ws, threaded = true, h_mode = :cached, bandwidth_cache = bwc_cold)   # warm
tmr = PQGradTimers()
t_G = @elapsed pairwise_quantile_production_gradient(xf, mass0, pcx, ctx, pe; base = baseF,
    verify = verifyF, econ_ws = econ_ws, threaded = true, h_mode = :cached,
    bandwidth_cache = bwc_cold, timers = tmr)

@printf("\ncb_G!  (outer gradient, steady state: base/verify reused, bandwidth cache warm)  %8.2f s\n", t_G)
@printf("   LFix cache build + q0 fold + exact check   %8.3f s  (%5.1f%%)   [shared builder, IN PLACE]\n",
        tmr.t_cache, 100 * tmr.t_cache / tmr.t_total)
@printf("   SHARED economic_A_gradient! (threaded)     %8.3f s  (%5.1f%%)\n",
        tmr.t_econ, 100 * tmr.t_econ / tmr.t_total)
@printf("   this restriction's mass gradient           %8.4f s  (%5.1f%%)\n",
        tmr.t_mass, 100 * tmr.t_mass / tmr.t_total)
@printf("   inner re-solve (0 here: base/verify reused)%8.3f s\n", tmr.t_solve)
@printf("   accounted                                  %8.3f s  of  %8.3f s  (residual %.3f s)\n",
        tmr.t_cache + tmr.t_econ + tmr.t_mass + tmr.t_solve, tmr.t_total,
        tmr.t_total - (tmr.t_cache + tmr.t_econ + tmr.t_mass + tmr.t_solve))
lp("")
lp("   REFERENCE POINTS for the two shared blocks above:")
@printf("     build_lfix_base_cache, ALLOCATING variant  %8.3f s  (%.0f MB) -- what passing cache= used to cost\n",
        t_lfix, a_lfix / 2^20)
@printf("       of which the serial W x D x Ddest price/pTsigma fill  %8.3f s  (380 independent (o,d) cells, UNthreaded)\n", t_price)
@printf("     economic_A_gradient! with threaded=false   %8.3f s  (threading speedup %.1fx)\n",
        t_econ_serial, t_econ_serial / max(t_econ_warm, eps()))
@printf("     economic_A_gradient! with a COLD bandwidth cache %8.3f s (first gradient at a new point)\n",
        t_econ_cold)

# ---- what an outer iteration costs, and what dominates ----------------------------------------
lp("\n", "-"^100)
lp("PER-OUTER-ITERATION BUDGET (one cb_F! + one cb_G!, the pattern a gradient-based outer solve runs)")
@printf("  cb_F!  %10.2f s  (%5.1f%%)\n", t_F, 100 * t_F / (t_F + t_G))
@printf("  cb_G!  %10.2f s  (%5.1f%%)\n", t_G, 100 * t_G / (t_F + t_G))
@printf("  total  %10.2f s\n", t_F + t_G)
lp("")
lp("NOTE: a real outer solve does NOT alternate 1:1 -- KNITRO takes several cb_F! per cb_G! during")
lp("line search. The measured production ratio at L=5/delta=0.1 was n_eval/n_grad ~ 3, so a")
@printf("realistic outer iteration is ~3 x cb_F! + 1 x cb_G! = %.1f s, of which %.1f%% is inner solves.\n",
        3 * t_F + t_G, 100 * 3 * r1.t_solve / (3 * t_F + t_G))
lp("="^100)
