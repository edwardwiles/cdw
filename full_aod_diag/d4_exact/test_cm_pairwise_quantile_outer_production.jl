# ================================================================================================
# Gate for the CM + pairwise-quantile family's OUTER PRODUCTION LAYER (family #7, 2026-08-12):
# the verifier, the verified state, the mandatory q0 fold, and the combined outer gradient.
#
# The reoptimized-FD gate on the mass gradient already exists
# (`test_cm_pairwise_quantile_outer_gradient_fd.jl`, rel L2 1.69e-9) and is NOT repeated here. What
# this file gates is everything BETWEEN that closed form and a campaign:
#
#   1. the verifier reproduces the solve's own objective and drives every block's KKT residual to
#      ~0 from a FRESH recompute -- and `classify_inner_result` accepts it under the SHARED gate
#      (memory `feedback-lfd-ok-verification-gate-required`: FiniteSolved && within_budget is NOT
#      a verification gate);
#   2. the CM IMPLICATION, measured: every non-reference origin's cumulative marginal under the LFD
#      matches the shared `Pcum`. This is the claim the whole family rests on, and the verifier
#      reports it rather than assuming it;
#   3. the q0 fold, WITH ITS NEGATIVE CONTROL. This family must fold TWO restriction blocks into
#      `q0`, not one, and the failure mode is silent. The control folds only `-G_R*lambda_R` and
#      shows that the result is badly wrong -- so check 3a is genuinely testing the CM half;
#   4. the combined gradient's layout and the outer-dimension collapse it exists for;
#   5. the D=20 production count assertions.
#
# Usage: julia --project=. --threads=N full_aod_diag/d4_exact/test_cm_pairwise_quantile_outer_production.jl \
#             <L> <G> <n_families> <contrasts>
# ================================================================================================

const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "lfix_base_workspace.jl", "shared_a_gradient.jl", "operator_verification.jl", "cm_screen_bridge.jl",
          "pairwise_quantile_mass_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_production.jl",
          "cm_pairwise_quantile_config.jl", "cm_pairwise_quantile_moments.jl",
          "cm_pairwise_quantile_hessian.jl", "cm_pairwise_quantile_hessian_assembly.jl",
          "cm_pairwise_quantile_lookup_kernels.jl", "cm_pairwise_quantile_production.jl",
          "cm_pairwise_quantile_verification.jl", "cm_pairwise_quantile_outer_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

const NFAIL = Ref(0); const NPASS = Ref(0)
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    ok ? (NPASS[] += 1) : (NFAIL[] += 1)
    println(ok ? "  PASS  " : "  FAIL  ", name, isempty(detail) ? "" : "   [$detail]")
    flush(stdout)
    return ok
end

const USAGE = "usage: julia ... test_cm_pairwise_quantile_outer_production.jl <L> <G> <n_families> <contrasts>"
length(ARGS) == 4 || error(USAGE)
const L_ARG = parse(Int, ARGS[1]); const G_ARG = parse(Int, ARGS[2])
const NFAM_ARG = parse(Int, ARGS[3]); const CONTR_ARG = Symbol(ARGS[4])
CONTR_ARG in (:anchored, :orthonormal) || error(USAGE)

println("=== CM + pairwise-quantile OUTER PRODUCTION LAYER gate ===")
@printf("L=%d  G=%d  families=%d  contrasts=%s\n", L_ARG, G_ARG, NFAM_ARG, String(CONTR_ARG))
flush(stdout)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
pe = build_pivot_elimination(ctx)
const PROD_OPT = joinpath(dirname(D4X), "ek_inner_cmpq.opt")
cfg = CMPairwiseQuantileConfig(L = L_ARG, cm_grid_size = G_ARG, cm_moment_families = NFAM_ARG,
                               contrasts = CONTR_ARG, min_bin_count = 1, mass_start = :uniform)
pcx = build_cm_pairwise_quantile_production_context(ctx, cfg; inner_opt = PROD_OPT)
ctx_cm = pcx.ctx_cm
cmpq = pcx.cmpq
nc = L_ARG - 1
W = cmpq.op.W

# The enforced point: masses off uniform, so the shared-mu machinery is genuinely exercised.
mu_off = [0.30, 0.10, 0.25, 0.08, 0.12, 0.06, 0.04, 0.02, 0.015][1:nc]
mu_off ./= (sum(mu_off) / 0.9)
raw_off = zeros(nc); raw_from_origin_masses!(raw_off, mu_off)

# ------------------------------------------------------------------------------------------------
println("\n=== 1. verified state ===")
# ------------------------------------------------------------------------------------------------
t0 = time()
base, verify = archCMPQ_verified_state(x_free_calib, raw_off, ctx_cm)
@printf("  inner_status=%d  Delta_dual=%.12g  Delta_primal=%.12g  gap=%.3e  (%.2fs)\n",
        verify.inner_status, verify.Delta_dual, verify.Delta_primal, verify.primal_dual_gap,
        time() - t0)
@printf("  n_fg=%d  n_hess=%d  mean_m_resid=%.3e  max_abs_moment_kkt_resid=%.3e\n",
        verify.n_fg, verify.n_hess, verify.mean_m_resid, verify.max_abs_moment_kkt_resid)
@printf("  block KKT: E=%.3e  level=%.3e  pair=%.3e  CM=%.3e\n",
        verify.kkt_resid_E, verify.kkt_resid_level, verify.kkt_resid_pairindep, verify.kkt_resid_cm)
flush(stdout)
check("Delta_dual is finite and positive", isfinite(verify.Delta_dual) && verify.Delta_dual > 0,
      @sprintf("%.8g", verify.Delta_dual))
check("the exact-Hessian callback ran inside the verified solve", verify.n_hess > 0,
      "n_hess=$(verify.n_hess)")
check("verify carries r_current of length W", length(verify.r_current) == W)
check("verify carries the shared mu this solve ran at",
      length(verify.mu) == nc && maximum(abs, verify.mu .- mu_off) < 1e-12)
check("ALL FOUR block KKT residuals are ~0 from the INDEPENDENT recompute",
      max(verify.kkt_resid_E, verify.kkt_resid_level, verify.kkt_resid_pairindep,
          verify.kkt_resid_cm) < 1e-7,
      @sprintf("max %.3e", max(verify.kkt_resid_E, verify.kkt_resid_level,
                               verify.kkt_resid_pairindep, verify.kkt_resid_cm)))
cls = classify_inner_result(verify)
println("  classify_inner_result = ", cls,
        cls == VerifiedSolved ? "" : "  reasons=$(verification_rejection_reasons(verify))")
check("inner solve is VerifiedSolved under the SHARED acceptance gate", cls == VerifiedSolved)

# ------------------------------------------------------------------------------------------------
println("\n=== 2. the CM implication, measured ===")
# ------------------------------------------------------------------------------------------------
# The family drops the per-origin marginal rows because CM + the level rows imply them. The dense
# oracle proves that as a SPAN statement; this measures it at a real solved point, under the LFD.
@printf("  max |cumulative level residual| (the ENFORCED rows) = %.3e\n",
        verify.max_level_cumulative_residual)
@printf("  max |cumulative residual| over the DROPPED per-origin marginals = %.3e\n",
        verify.max_implied_cumulative_residual)
@printf("  max |pairwise factorization residual|               = %.3e\n",
        verify.max_cumulative_residual)
flush(stdout)
check("the ENFORCED level rows hold under the LFD", verify.max_level_cumulative_residual < 1e-7,
      @sprintf("%.3e", verify.max_level_cumulative_residual))
check("the DROPPED per-origin marginals hold too -- the redundancy claim, at a real solved point",
      verify.max_implied_cumulative_residual < 1e-6,
      @sprintf("%.3e", verify.max_implied_cumulative_residual))

# ------------------------------------------------------------------------------------------------
println("\n=== 3. the q0 fold, and its negative control ===")
# ------------------------------------------------------------------------------------------------
# 3a is a hard error inside build_lfix_base_cache_cmpq if it fails, so reaching this line at all is
# most of the check; the residual is reported so the margin is visible rather than implied.
ws = get_or_build_econ_a_grad_ws(W)
cache = build_lfix_base_cache_cmpq(x_free_calib, ctx_cm, base; verify = verify, econ_ws = ws)
err_full = maximum(abs, cache.q0 .- verify.r_current)
check("3a  q0 with BOTH restriction folds == the independently recomputed r", err_full < 1e-8,
      @sprintf("max|diff| %.3e", err_full))

# 3b NEGATIVE CONTROL: fold only the level+pair block, as the standalone family does, and show the
# result is badly wrong. Without this, 3a would pass even if the CM fold were a no-op.
cache_bare = build_lfix_base_cache(x_free_calib, ctx_cm, base; validate_dense = false)
op = cmpq.op
λ_L, λ_P, λ_CM = reshape_cmpq_duals_lambda(base.λstar, op, cmpq.ncore1)
contrib_R = zeros(W)
cm_pq_forward!(contrib_R, λ_L, λ_P, op, ctx_cm.cmpq_mass_state, cmpq.refIndex1)
err_partial = maximum(abs, (cache_bare.q0 .+ contrib_R) .- verify.r_current)
check("3b  NEGATIVE CONTROL: folding ONLY the level+pair block leaves q0 badly wrong",
      err_partial > 1e-3,
      @sprintf("max|diff| %.3e (vs %.3e with both folds) -- ratio %.3g", err_partial, err_full,
               err_partial / max(err_full, 1e-300)))
# 3c and the mirror image: folding NEITHER (the plain economic cache) must also be wrong, so 3b is
# not accidentally measuring "the CM block happens to be the only nonzero one".
err_none = maximum(abs, cache_bare.q0 .- verify.r_current)
check("3c  and folding NEITHER block is wrong too (both terms are genuinely present)",
      err_none > 1e-3, @sprintf("max|diff| %.3e", err_none))

# ------------------------------------------------------------------------------------------------
println("\n=== 4. the combined outer gradient ===")
# ------------------------------------------------------------------------------------------------
Ddest = hasproperty(ctx_cm, :D_dest) ? ctx_cm.D_dest : ctx_cm.D
timers = CMPQGradTimers()
t0 = time()
g_ext, meta = cm_pairwise_quantile_production_gradient(x_free_calib, raw_off, pcx, ctx, pe;
    base = base, verify = verify, econ_ws = ws, timers = timers)
@printf("  |g_ext|=%d = econ %d + mass %d   (%.2fs: cache %.2fs, econ %.2fs, mass %.4fs)\n",
        length(g_ext), ctx.D * Ddest, nc, time() - t0, timers.t_cache, timers.t_econ, timers.t_mass)
flush(stdout)
check("combined gradient has the right length", length(g_ext) == ctx.D * Ddest + nc,
      "$(length(g_ext)) == $(ctx.D * Ddest) + $nc")
check("combined gradient is all-finite", all(isfinite, g_ext))
# The mass tail must be exactly what the standalone closed-form call returns -- the gradient
# assembler must not be quietly transforming it.
g_mass_direct = cmpq_mass_gradient_vec(base, verify, ctx_cm, raw_off)
check("the mass tail of g_ext is BIT-IDENTICAL to cmpq_mass_gradient_vec",
      all(g_ext[ctx.D*Ddest+1:end] .=== g_mass_direct))
check("the outer collapse is real: mass coords == L-1, not D*(L-1)",
      nc == n_cmpq_raw(L_ARG) && nc * ctx.D == (L_ARG - 1) * ctx.D,
      "$nc here vs $(nc * ctx.D) for the standalone family")

# A cheap, genuinely independent confirmation that the mass block is right THROUGH THIS LAYER
# (the closed form itself is gated at 1.69e-9 in test_cm_pairwise_quantile_outer_gradient_fd.jl):
# one reoptimized FD coordinate, at the h that gate found best.
gfd, nprobed = d_delta_dual_d_cmpq_mass_fd(x_free_calib, raw_off, ctx_cm; h = 1e-5, coords = [1])
rel1 = abs(g_mass_direct[1] - gfd[1]) / max(abs(gfd[1]), 1e-300)
check("mass gradient coord 1 == reoptimized FD through the production layer", rel1 < 1e-5,
      @sprintf("analytic %.10e vs FD %.10e, rel %.3e (%d probe)", g_mass_direct[1], gfd[1], rel1, nprobed))

# ------------------------------------------------------------------------------------------------
println("\n=== 5. production count assertions ===")
# ------------------------------------------------------------------------------------------------
cnt = assert_cm_pairwise_quantile_d20_counts(20, 5, 50, 2)
@printf("  D=20/L=5/G=50/2 families: restriction rows=%d, CM moments=%d, inner rows=%d, outer mass=%d (standalone %d)\n",
        cnt.restriction_rows, cnt.cm_moments, cnt.inner_rows, cnt.outer_mass_params,
        cnt.standalone_outer_mass_params)
check("D=20 production counts assert clean", cnt.restriction_rows == 3044 && cnt.cm_moments == 1862 &&
      cnt.outer_mass_params == 4 && cnt.standalone_outer_mass_params == 80)
check("no dense G in production", cnt.dense_G_production == false)
# Wrapped in a FUNCTION deliberately: at top level, `threw = true` inside a `catch` lands in a soft
# scope and Julia 1.12 makes it a NEW LOCAL when a global of the same name exists, so the outer flag
# stays `false` and the control reports a spurious failure. (Observed here on the first run -- the
# same pattern works in the other gates only because it sits inside `run_case`.) A function body is
# a hard scope and has no such ambiguity.
function threw_on(f)
    try
        f()
        return false
    catch
        return true
    end
end
check("NEGATIVE CONTROL: an L that does not divide G is rejected by the count assertion",
      threw_on(() -> assert_cm_pairwise_quantile_d20_counts(20, 7, 50, 2)))
check("NEGATIVE CONTROL: a non-D=20 call is rejected too",
      threw_on(() -> assert_cm_pairwise_quantile_d20_counts(4, 5, 50, 2)))

println("\n", "="^92)
@printf("TOTAL: %d passed, %d FAILED\n", NPASS[], NFAIL[])
println("="^92)
NFAIL[] == 0 || error("test_cm_pairwise_quantile_outer_production: $(NFAIL[]) check(s) failed")
