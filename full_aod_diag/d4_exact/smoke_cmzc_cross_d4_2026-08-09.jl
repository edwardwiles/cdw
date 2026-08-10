# Phase 1 smoke test for CM+ZC-CROSS (2026-08-09 task): confirms the shared FG/Hessian machinery
# (CMMeanZCOperatorState, zc_restriction_gram!, bin_zc_cross_hessian_fill!,
# winner_pair_cross_hessian_zc_block!) is genuinely generic for the COMMON-MARGINALS family too --
# i.e. that feeding it a K_pair^2 cross-power ZCRestrictionOperator (instead of the base family's
# K_pair diagonal-level one) "just works": the real KNITRO inner solve converges, AND the recovered
# solution actually satisfies every new cross-power restriction (not just "didn't crash"), AND the
# CM-grid block's own KKT conditions are undisturbed by the widened pair block sitting in front of
# it in the column layout.
#
# Structural twin of smoke_ozc_cross_d4_2026-08-09.jl (the OZC-CROSS equivalent).
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/smoke_cmzc_cross_d4_2026-08-09.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "compressed_moments.jl", "structured_moment_build.jl",
          "compressed_cc_inner.jl", "compressed_live.jl", "compressed_factual_buffer_reuse.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_threaded.jl", "winner_pair_cross_hessian.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "threaded_cross_hessian.jl",
          "zc_gram_blas_candidates.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "hcz_reordered_candidate_2026-08-01.jl", "hez_drawmajor_candidate_2026-08-01.jl",
          "hez_drawmajor_v2_candidate_2026-08-01.jl", "operator_hessian_weights.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS
    ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 8
probs = cm_equal_grid_probs(L)

# Theoretical POPULATION mean of z_o(w)^k = U_o(w)^{-mu*k}, U_o~Exp(1): E[U^{-mu*k}] = Gamma(1-mu*k).
# NOT a sample average of the very draws the restriction is imposed on (handover trap 1), and NOT
# mean(U^k) (wrong transform). Shared across origins by construction under common marginals --
# which is exactly why ONE nu_k per level suffices for this family.
nu0_shared(K::Int) = [gamma(1 - ctx.μHat * k) for k in 1:K]

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D4 CM+ZC-CROSS K_mean=$K_mean K_pair=$K_pair (K_pair^2=$(K_pair^2) combos/origin-pair) ====")
    νvec0 = nu0_shared(K_mean)

    pcx = build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs)
    layout = pcx.aug.layout
    npair = div(D * (D - 1), 2)
    check("K=$K_mean/$K_pair: n_eta(layout) unchanged vs base (K_mean)", n_eta(layout) == K_mean)
    check("K=$K_mean/$K_pair: n_pair = K_pair^2 * npair", pcx.aug.n_pair == K_pair^2 * npair)
    check("K=$K_mean/$K_pair: n_mean = K_mean*D", pcx.aug.n_mean == K_mean * D)
    # OperatorPsiBundle has no `d` field (unlike PsiObjectiveBundleImplicit) -- `outer_constr_index`
    # is the equivalent total-column count for this bundle type, and is what the augmented-obj
    # builder's own @assert checks against.
    check("K=$K_mean/$K_pair: obj.outer_constr_index = ncore_econ + n_mean + n_pair + ncm",
          pcx.aug.obj_cm.outer_constr_index == pcx.aug.ncore_econ + pcx.aug.n_mean + pcx.aug.n_pair + pcx.aug.ncm)
    check("K=$K_mean/$K_pair: ZC operator sees K_pair^2 pair blocks",
          n_pair(pcx.cctx.hzz_zc_op) == K_pair^2 * npair)
    check("K=$K_mean/$K_pair: cctx layout is the CROSS layout (aug.layout picked up, not SharedByPowerLayout)",
          pcx.cctx.hzz_zc_layout isa SharedByPowerCrossLayout && pcx.cctx.meanzc_zc_layout isa SharedByPowerCrossLayout)

    base, verify = archC_meanzc_verified_state(x_free_calib, νvec0, pcx.ctx_cm, pcx.cctx)
    println("    inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)  m_mean=$(verify.m_mean)")
    check("K=$K_mean/$K_pair: inner solve converged (nStatus in (0,-100,-101,-103))", verify.inner_status in (0, -100, -101, -103))
    check("K=$K_mean/$K_pair: KKT moment residual ~0", verify.max_abs_moment_kkt_resid < 1e-6)

    # Direct, independent residual check against the NEW cross-power restrictions specifically.
    # recovered_mean_residuals_origin/recovered_pair_residuals_origin (cm_originzc_moments.jl) are
    # column-agnostic (they take an explicit per-column target vector) -- reused unchanged here, fed
    # the SHARED-nu targets via mean_targets/pair_targets' own dispatch on the cross layout.
    m_weights = base.m_star
    for k in 1:K_mean
        r = recovered_mean_residuals_origin(m_weights, pcx.aug.Zraw_all[k], mean_targets(layout, νvec0, k, D))
        check("K=$K_mean/$K_pair: mean level k=$k residual ~0 (max=$(maximum(abs.(r))))", maximum(abs.(r)) < 1e-6)
    end
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        r = recovered_pair_residuals_origin(m_weights, pcx.aug.Zpairraw_all[klin], pair_targets(layout, νvec0, klin, D))
        check("K=$K_mean/$K_pair: cross-pair (k1=$k1,k2=$k2) residual ~0 (max=$(maximum(abs.(r))))", maximum(abs.(r)) < 1e-6)
    end
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
