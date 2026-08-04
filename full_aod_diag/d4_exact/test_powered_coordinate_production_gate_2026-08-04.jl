# Task §6.2 (profiled-outer-ab-completion-2026-08-04): production-context gates for
# :profiled_powered_relative_A, wired through run_profiled_upper_constrained (task §6.1). Real D4
# KNITRO context, flexible_CM family -- powered mode is FIXED-THETA ONLY
# (POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md §3d), so this gate deliberately uses the
# real flexible_cm family construction (theta = cm_fixed_theta(ctx), the SAME fixed value the
# production CM drivers use), NOT :unrestricted (whose theta is jointly searched -- a first attempt
# at this gate mistakenly reused an UnrestrictedFamilyCtx as a convenient stand-in and got both a
# correct hard-block from validate_mode_family_compatibility AND a spurious FD mismatch, because
# unrestricted's own theta is not cm_fixed_theta(ctx) at all; fixed by using the real family this
# mode targets, matching the pattern in test_profiled_reduced_flexcm_frechet_adapters_2026-08-02.jl).
# Checks:
#   1. encode/decode round trip identity (native -> powered -> native)
#   2. identical reconstructed full log A / gravity residual at the shared calibration start point
#   3. identical fixed-state Delta-star at that same shared start point
#   4. native-gradient vs powered-gradient chain rule (-theta scalar rescale)
#   5. central finite-difference coordinate check in POWERED units directly
#   6. checkpoint mode-mismatch refusal (write under native, refuse resume under powered)
#   7. short constrained KNITRO run completes cleanly under powered mode (production callback path)
#   8. :unrestricted + powered mode is a hard error, not a silent fallback
D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
          "cm_exact_cache_production.jl", "cm_checkpoint.jl", "draw_design.jl",
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
          "profiled_production_outer_constrained_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using LinearAlgebra, Random

lp(xs...) = (println(xs...); flush(stdout))
results = Dict{Symbol,Bool}()

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w0 = reduce_calibration_to_w_profiled(ctx, pe)
theta = cm_fixed_theta(ctx)
xy = precompute_cm_aspace_xy(ctx)
lp("D=", ctx.D, " n_free=", length(w0) - 1, " theta(cm_fixed_theta)=", theta)

x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)
aug = build_cm_augmented_obj_archB(ctx, CS; L = 10, contrasts = :anchored, base_obj = reduced_obj0, profiled_layout = layout)
cctx = build_cm_bin_ctx(ctx, aug; profiled_layout = layout, inner_fg_backend = :dense_reference, threaded_bins = false)
fctx = build_flexcm_family_ctx(ctx, spec, pe, layout, cctx)

# ---------------------------------------------------------------------------
# 1-3. Round trip + identical decoded state at the shared calibration start point.
# ---------------------------------------------------------------------------
lp("\n=== 1-3. Round trip + identical decoded state ===")
a_free0 = encode_r_free_to_w_mode(w0[2:end], pe, :profiled_powered_relative_A, theta, xy)
r_free_back = decode_w_mode_to_r_free(a_free0, pe, :profiled_powered_relative_A, theta, xy)
roundtrip_err = maximum(abs.(r_free_back .- w0[2:end]))
lp("  encode->decode round trip max abs err = ", roundtrip_err)

z_native = decode_full_z_on_retained(w0[2:end], pe)
z_powered = decode_full_z_on_retained_powered(a_free0, pe, theta, xy)
z_err = maximum(abs.(z_native .- z_powered))
lp("  decoded full log-A (native basis vs powered basis) max abs diff = ", z_err)

grav_native = gravity_from_logz(z_native, ctx)
grav_powered = gravity_from_logz(z_powered, ctx)
lp("  gravity residual (native) = ", grav_native, "  (powered) = ", grav_powered, "  diff = ", abs(grav_native - grav_powered))

ev_native = evaluate_profiled_flexcm_point(w0, fctx)
w0_from_powered = vcat(w0[1], r_free_back)
ev_powered_decoded = evaluate_profiled_flexcm_point(w0_from_powered, fctx)
delta_err = abs(ev_native.result.Delta_dual - ev_powered_decoded.result.Delta_dual)
lp("  inner_status native=", ev_native.result.inner_status, "  decoded-from-powered=", ev_powered_decoded.result.inner_status)
lp("  Delta_dual (native start) = ", ev_native.result.Delta_dual, "  (decoded-from-powered start) = ", ev_powered_decoded.result.Delta_dual, "  diff = ", delta_err)
results[:roundtrip_and_decoded_state] = (roundtrip_err < 1e-9) && (z_err < 1e-9) && (abs(grav_native - grav_powered) < 1e-9) && (delta_err < 1e-9) &&
    (ev_native.result.inner_status == ev_powered_decoded.result.inner_status)

# ---------------------------------------------------------------------------
# 4. Native-gradient vs powered-gradient chain rule.
# ---------------------------------------------------------------------------
lp("\n=== 4. Gradient chain rule ===")
g_native, _ = shared_family_outer_gradient(w0, ctx, fctx, ev_native; threaded = false)
g_native_free = g_native[2:end]
g_powered_free_expected = rescale_gradient_for_mode(g_native_free, :profiled_powered_relative_A, theta)
manual_rescale = g_native_free .* (-theta)
chain_rule_err = maximum(abs.(g_powered_free_expected .- manual_rescale))
lp("  rescale_gradient_for_mode vs manual (-theta) rescale max abs diff = ", chain_rule_err)
results[:gradient_chain_rule] = chain_rule_err == 0.0

# ---------------------------------------------------------------------------
# 5. Central FIXED-DUAL FD directly in POWERED units (independent check). MUST be fixed-dual
# (same beta/zeta as the analytic gradient's own cache), not a fresh KNITRO re-solve at each
# perturbed point -- shared_family_outer_gradient computes a FIXED-DUAL directional derivative
# (profiled_lfix_incremental_at, holding the dual cache fixed), a genuinely different mathematical
# object from a full re-solve FD (which would also capture how the dual itself moves). A first
# attempt at this test used evaluate_profiled_flexcm_point (full re-solve) here and got a spurious
# ~1500% relative error -- not a wiring bug, a wrong-comparison bug in the test itself, fixed here
# by using the SAME cache-based profiled_lfix_incremental_at machinery the analytic gradient uses.
# ---------------------------------------------------------------------------
lp("\n=== 5. Central FIXED-DUAL FD in powered units (adaptive bandwidth) ===")
cache = build_shared_profiled_lfix_cache(w0, fctx, ctx, ev_native)

# A first attempt at this test used a naive FIXED h=1e-5 and got a ~1500% relative error in BOTH
# native and powered space identically -- a red herring initially blamed on the coordinate
# transform, but a diagnostic (direct native-space FD vs g_native_free, no powered coordinates
# involved at all) reproduced the SAME error, proving the bug was never in the powered wiring.
# Root cause: this repo's own standing pitfall (feedback-fd-bandwidth-mismatch-looks-like-a-bug)
# -- a naive fixed FD step can straddle a winner-switch kink in Delta_dual that the production
# analytic gradient's own ADAPTIVE bandwidth (profiled_select_bandwidth) is specifically chosen to
# avoid. Fixed by using that SAME adaptively-selected h (in native r_free units), then converting
# it to an equivalent powered-units step via the known constant slope da/dr = -1/theta.
n_free = length(a_free0)
fd_g_native = zeros(n_free)
fd_g_powered = zeros(n_free)
h_native_used = zeros(n_free)
for k in 1:n_free
    h_native, _m, _selmeta = profiled_select_bandwidth(cache, ctx, spec, pe, w0, k + 1)
    h_native_used[k] = h_native
    # Native-space FD (diagnostic: confirms g_native_free itself is correct with matched bandwidth).
    Lp_n = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k + 1, w0[k+1] + h_native)
    Lm_n = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k + 1, w0[k+1] - h_native)
    fd_g_native[k] = (Lp_n - Lm_n) / (2h_native)
    # Powered-space FD: SAME native step, expressed via the corresponding a_free perturbation
    # (h_a = h_native/theta, from dr/da = -theta) -- decode back to r_free and evaluate at the
    # SAME two native points profiled_lfix_incremental_at needs, so this is not a re-derivation,
    # only a re-expression of the identical step in the other coordinate's units.
    h_a = h_native / theta
    a_p = copy(a_free0); a_p[k] += h_a
    a_m = copy(a_free0); a_m[k] -= h_a
    r_p = decode_w_mode_to_r_free(a_p, pe, :profiled_powered_relative_A, theta, xy)
    r_m = decode_w_mode_to_r_free(a_m, pe, :profiled_powered_relative_A, theta, xy)
    Lp = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k + 1, r_p[k])
    Lm = profiled_lfix_incremental_at(cache, ctx, spec, pe, w0, k + 1, r_m[k])
    fd_g_powered[k] = (Lp - Lm) / (2h_a)
end
native_fd_err = maximum(abs.(fd_g_native .- g_native_free))
native_fd_rel = maximum(abs.(fd_g_native .- g_native_free) ./ max.(abs.(fd_g_native), 1e-8))
lp("  DIAGNOSTIC native-space FD (matched bandwidth) vs g_native_free: max_abs_err=", native_fd_err, "  max_rel_err=", native_fd_rel)
fd_err = maximum(abs.(fd_g_powered .- g_powered_free_expected))
fd_rel = maximum(abs.(fd_g_powered .- g_powered_free_expected) ./ max.(abs.(fd_g_powered), 1e-8))
lp("  analytic (powered, rescaled) vs central fixed-dual FD (powered units, matched bandwidth): max_abs_err=", fd_err, "  max_rel_err=", fd_rel)
results[:central_fd_powered] = fd_rel < 1e-4 && native_fd_rel < 1e-4

# ---------------------------------------------------------------------------
# 6. Checkpoint mode-mismatch refusal.
# ---------------------------------------------------------------------------
lp("\n=== 6. Checkpoint mode-mismatch refusal ===")
ckpt_path = tempname() * ".jls"
res_native = run_profiled_upper_constrained("gate_native", w0; fctx = fctx,
    evaluate_fn = (w, ff) -> evaluate_profiled_flexcm_point(w, ff),
    ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 8.0, hessopt_tag = "sr1",
    a_coordinate_mode = :profiled_pivot_anchor_relative,
    checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = nothing, verbose = false,
    threaded_gradient = false)
lp("  native short run: n_eval=", res_native.n_eval, " n_grad=", res_native.n_grad, " status=", res_native.knitro_status)
local mismatch_refused = false
try
    global mismatch_refused
    run_profiled_upper_constrained("gate_resume_mismatch", w0; fctx = fctx,
        evaluate_fn = (w, ff) -> evaluate_profiled_flexcm_point(w, ff),
        ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 2.0, hessopt_tag = "sr1",
        a_coordinate_mode = :profiled_powered_relative_A,
        checkpoint_path = ckpt_path, checkpoint_interval_s = 60.0, resume_from = ckpt_path, verbose = false,
        threaded_gradient = false)
catch e
    global mismatch_refused = occursin("namespace mismatch", string(e))
    lp("  resume under different mode correctly refused: ", string(e)[1:min(150, end)])
end
results[:checkpoint_mode_mismatch_refused] = mismatch_refused

# ---------------------------------------------------------------------------
# 7. Short constrained KNITRO run under powered mode (production callback path).
# ---------------------------------------------------------------------------
lp("\n=== 7. Short constrained run under powered mode ===")
ckpt_path2 = tempname() * ".jls"
res_powered = run_profiled_upper_constrained("gate_powered", w0; fctx = fctx,
    evaluate_fn = (w, ff) -> evaluate_profiled_flexcm_point(w, ff),
    ctx = ctx, pe = pe, delta = 1.0, maxtime_real = 8.0, hessopt_tag = "sr1",
    a_coordinate_mode = :profiled_powered_relative_A,
    checkpoint_path = ckpt_path2, checkpoint_interval_s = 60.0, resume_from = nothing, verbose = false,
    threaded_gradient = false)
lp("  powered short run: n_eval=", res_powered.n_eval, " n_grad=", res_powered.n_grad, " status=", res_powered.knitro_status)
results[:short_run_powered_completes] = res_powered.n_eval > 0 && res_powered.n_grad > 0

# ---------------------------------------------------------------------------
# 8. unrestricted + powered mode is a hard error, not a silent fallback.
# ---------------------------------------------------------------------------
lp("\n=== 8. unrestricted + powered mode hard error ===")
local unrestricted_blocked = false
try
    global unrestricted_blocked
    validate_mode_family_compatibility(:profiled_powered_relative_A, :unrestricted)
catch e
    global unrestricted_blocked = true
    lp("  correctly blocked: ", string(e)[1:min(150,end)])
end
results[:unrestricted_powered_blocked] = unrestricted_blocked

lp("\nALL_RESULTS: ", results)
all_pass = all(values(results))
lp("\nPOWERED_COORDINATE_PRODUCTION_GATE (D4, flexible_cm): ", all_pass ? "PASS" : "FAIL")
all_pass || error("test_powered_coordinate_production_gate_2026-08-04: one or more checks failed -- see ALL_RESULTS above")
