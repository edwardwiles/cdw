# Phase 1 smoke test for OZC-CROSS (2026-08-09 task): confirms the shared FG/Hessian machinery
# (OriginZCOperatorState, zc_restriction_gram!, winner_pair_cross_hessian_zc_block!) is genuinely
# generic -- i.e. that feeding it a K_pair^2 cross-power ZCRestrictionOperator (instead of the base
# family's K_pair diagonal-level one) "just works": the real KNITRO inner solve converges, AND the
# recovered solution actually satisfies every new cross-power restriction (not just "didn't crash").
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/smoke_ozc_cross_d4_2026-08-09.jl
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Statistics
using SpecialFunctions: gamma

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]   # the calibration ("pre-step") point for A_od/gp

# Theoretical population mean of z_o(w)^k = U_o(w)^{-mu*k}, U_o~Exp(1): E[U^{-mu*k}] = Gamma(1-mu*k)
# (2026-08-09: confirmed by user this is the right nu0 -- matches the SAME Gamma(1-mu*k) seeding
# already used in production, see memory zc-cmzc-exclude-row-k2-k3-production-2026-08-07). Shared
# across origins since every origin draws from the SAME population by construction.
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D4 OZC-CROSS K_mean=$K_mean K_pair=$K_pair (K_pair^2=$(K_pair^2) combos/origin-pair) ====")
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)

    pcx = build_originzc_cross_production_context(ctx, CS, layout)
    check("K=$K_mean/$K_pair: n_eta(layout) unchanged vs base (K_mean*D)", n_eta(layout) == K_mean * D)
    check("K=$K_mean/$K_pair: n_pair = K_pair^2 * npair", pcx.aug.n_pair == K_pair^2 * div(D * (D - 1), 2))

    base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    println("    inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)  m_mean=$(verify.m_mean)")
    check("K=$K_mean/$K_pair: inner solve converged (nStatus in (0,-100,-101,-103))", verify.inner_status in (0, -100, -101, -103))
    check("K=$K_mean/$K_pair: KKT moment residual ~0", verify.max_abs_moment_kkt_resid < 1e-6)

    # Direct, independent residual check against the NEW cross-power restrictions specifically
    # (recovered_mean_residuals_origin/recovered_pair_residuals_origin, cm_originzc_moments.jl, are
    # column-agnostic -- reused completely unchanged here).
    m_weights = base.m_star
    for k in 1:K_mean
        νo_k = mean_targets(layout, νfull0, k, D)
        r = recovered_mean_residuals_origin(m_weights, pcx.aug.Zraw_all[k], νo_k)
        check("K=$K_mean/$K_pair: mean level k=$k residual ~0 (max=$(maximum(abs.(r))))", maximum(abs.(r)) < 1e-6)
    end
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        νprod = pair_targets(layout, νfull0, klin, D)
        r = recovered_pair_residuals_origin(m_weights, pcx.aug.Zpairraw_all[klin], νprod)
        check("K=$K_mean/$K_pair: cross-pair (k1=$k1,k2=$k2) residual ~0 (max=$(maximum(abs.(r))))", maximum(abs.(r)) < 1e-6)
    end
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
