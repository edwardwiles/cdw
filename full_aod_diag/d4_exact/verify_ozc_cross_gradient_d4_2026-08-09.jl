# Phase 2 verification for OZC-CROSS (2026-08-09 task): the outer-gradient envelope formula
# (d_delta_dual_d_eta_origin_cross_vec) and the fixed-contribution fold
# (originzc_cross_fixed_contribution), both hand-derived extensions of the base family's own
# analytic formulas to the K_pair^2 grid. Two INDEPENDENT checks, neither of which can pass by
# construction/mirroring alone:
#
#   (A) d_delta_dual_d_eta_origin_cross_vec vs central finite differences of Delta_dual at
#       INDEPENDENTLY REOPTIMIZED perturbed eta points (d_delta_dual_d_eta_origin_fd, UNCHANGED,
#       reused verbatim from cm_originzc_production.jl -- it only calls archOZ_verified_state, which
#       works identically for any ctx_cm). This is the SAME validation pattern
#       (test_cm_originzc_pure_moments.jl) the base family's own d_delta_dual_d_eta_origin_vec was
#       validated with.
#   (B) originzc_cross_fixed_contribution: cache0.q0 .- contrib0 (closed-form winner-cache path)
#       compared against a FRESH, independent recompute of q at the exact solution via the raw
#       operator FG primitives (OriginZCOperatorState's dual_index!, zc_restriction_operator.jl) --
#       two completely different code paths that must agree if the fold is correct.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/verify_ozc_cross_gradient_d4_2026-08-09.jl
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
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 1), (2, 2), (3, 3)]
    println("\n==== D4 OZC-CROSS gradient verification K_mean=$K_mean K_pair=$K_pair ====")
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_cross_production_context(ctx, CS, layout)

    base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    check("K=$K_mean/$K_pair: inner solve converged", verify.inner_status in (0, -100, -101, -103))

    # ---- Check (B) FIRST: fixed-contribution fold vs independent operator-FG recompute ----
    # (must run before (A) -- d_delta_dual_d_eta_origin_fd below re-solves at many perturbed eta
    # points, mutating pcx.octx.fg_lookup_st's cached state (core_cf_ref/zc_ws targets) away from
    # the calibration point; running (B) after (A) compares base.λstar (correct, calibration point)
    # against a STALE st (left at the last FD probe) -- a test-ordering bug, not a bug in the
    # functions under test, confirmed 2026-08-09 via debug2_cross_q0_2026-08-09.jl.)
    cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
    contrib0 = originzc_cross_fixed_contribution(base, pcx.aug, νfull0)
    q0_via_fold = cache0.q0 .- contrib0

    st = pcx.octx.fg_lookup_st   # populated by archOZ_verified_state's own inner solve above
    q0_via_operator = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
    d_q0 = maximum(abs.(q0_via_fold .- q0_via_operator))
    println("    (B) fixed-contribution-fold q0 vs independent operator-FG q0: max abs diff=$d_q0")
    check("K=$K_mean/$K_pair: (B) originzc_cross_fixed_contribution matches independent operator recompute", d_q0 < 1e-8)

    # ---- Check (A): analytic eta-gradient vs independently reoptimized central FD ----
    g_analytic = d_delta_dual_d_eta_origin_cross_vec(base.λstar, pcx.aug, νfull0; mean_m = verify.m_mean)
    g_fd = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull0, pcx.ctx_cm; h = 1e-4)
    d_ag = maximum(abs.(g_analytic .- g_fd))
    rel = d_ag / max(1e-8, maximum(abs.(g_fd)))
    println("    (A) analytic eta-grad vs FD: max abs diff=$d_ag  max rel diff=$rel")
    println("        analytic=$g_analytic")
    println("        fd      =$g_fd")
    check("K=$K_mean/$K_pair: (A) analytic eta-gradient matches reoptimized FD", d_ag < 5e-3 && rel < 1e-2)
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
