# CONTROL against the UNMODIFIED base diagonal CM+ZC family, before bug-hunting the CM+ZC-CROSS
# nu-gradient restructure (memory feedback-control-against-base-family-before-bug-hunting -- this
# exact discipline has resolved four separate apparent "bugs" instantly across these sessions).
#
# OBSERVATION THAT PROMPTED THIS: verify_cmzc_cross_gradient_d4_2026-08-09.jl's check (A)
# (analytic eta-gradient vs reoptimized central FD, h=1e-4) showed a ~3-5% RELATIVE gap at every
# K. But its own check (A2) showed that at K_pair=1 the cross formula reproduces the BASE family's
# `d_delta_dual_d_eta_nu_vec` BITWISE (max abs diff = 0.0) -- so if the restructure were the cause,
# (A2) could not pass. That points at the FD REFERENCE, not at the analytic formula.
#
# WHAT THIS SCRIPT DOES, at the SAME D4 calibration point and the SAME nu seeding:
#   1. Runs the identical analytic-vs-FD comparison on the UNMODIFIED base family
#      (build_cm_meanzc_production_context + d_delta_dual_d_eta_nu_vec, neither touched by this
#      task). If the base family shows a comparable gap, the gap is a property of the FD reference.
#   2. Sweeps h over several decades for BOTH families. A genuine analytic-formula error shows a
#      gap that is FLAT in h; a finite-difference truncation error shows a gap that SHRINKS with h
#      until roundoff takes over. These two signatures are not confusable.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/control_base_cmzc_etagrad_fd_d4_2026-08-09.jl
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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 8
probs = cm_equal_grid_probs(L)
nu0_shared(K::Int) = [gamma(1 - ctx.μHat * k) for k in 1:K]

function fd_eta(x_free0, νvec, ctx_cm, cctx, h)
    n = length(νvec); g = Vector{Float64}(undef, n); η = log.(νvec)
    for j in 1:n
        ηp = copy(η); ηp[j] += h
        ηm = copy(η); ηm[j] -= h
        _, vp = archC_meanzc_verified_state(x_free0, exp.(ηp), ctx_cm, cctx)
        _, vm = archC_meanzc_verified_state(x_free0, exp.(ηm), ctx_cm, cctx)
        g[j] = (vp.Delta_dual - vm.Delta_dual) / (2h)
    end
    return g
end

HS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    νvec0 = nu0_shared(K_mean)

    println("\n", "="^96)
    println("BASE (diagonal) CM+ZC   K_mean=$K_mean K_pair=$K_pair   -- unmodified production family, the CONTROL")
    println("="^96)
    pcx_b = build_cm_meanzc_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs,
        moment_representation = :operator)
    base_b, ver_b = archC_meanzc_verified_state(x_free_calib, νvec0, pcx_b.ctx_cm, pcx_b.cctx)
    g_an_b = d_delta_dual_d_eta_nu_vec(base_b.λstar, pcx_b.aug, νvec0; mean_m = ver_b.m_mean)
    @printf("  inner_status=%d  Delta_dual=%.12e\n", ver_b.inner_status, ver_b.Delta_dual)
    println("  analytic = ", g_an_b)
    for h in HS
        g_fd = fd_eta(x_free_calib, νvec0, pcx_b.ctx_cm, pcx_b.cctx, h)
        rel = maximum(abs.(g_an_b .- g_fd)) / max(1e-12, maximum(abs.(g_fd)))
        @printf("  h=%.0e   fd=%s   max_rel_gap=%.3e\n", h, string(round.(g_fd, sigdigits = 8)), rel)
    end

    println("\n", "-"^96)
    println("CROSS CM+ZC   K_mean=$K_mean K_pair=$K_pair   -- this task's new family")
    println("-"^96)
    pcx_c = build_cm_meanzc_cross_production_context(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair,
        include_truncated_moment = true, contrasts = :anchored, meanzc_basis = :direct, probs = probs)
    base_c, ver_c = archC_meanzc_verified_state(x_free_calib, νvec0, pcx_c.ctx_cm, pcx_c.cctx)
    g_an_c = d_delta_dual_d_eta_nu_cross_vec(base_c.λstar, pcx_c.aug, νvec0; mean_m = ver_c.m_mean)
    @printf("  inner_status=%d  Delta_dual=%.12e\n", ver_c.inner_status, ver_c.Delta_dual)
    println("  analytic = ", g_an_c)
    for h in HS
        g_fd = fd_eta(x_free_calib, νvec0, pcx_c.ctx_cm, pcx_c.cctx, h)
        rel = maximum(abs.(g_an_c .- g_fd)) / max(1e-12, maximum(abs.(g_fd)))
        @printf("  h=%.0e   fd=%s   max_rel_gap=%.3e\n", h, string(round.(g_fd, sigdigits = 8)), rel)
    end
end

println("\nDONE -- read the h-sweeps: a gap that SHRINKS with h is FD truncation error (analytic formula fine);")
println("a gap FLAT in h is a genuine analytic-formula error. Compare BASE vs CROSS signatures directly.")
