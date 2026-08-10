# Variant D (focal k*=sigma-1 mean-row omission) verification for OZC-CROSS (2026-08-09 task,
# handover follow-up). Extends verify_ozc_cross_gradient_d4_2026-08-09.jl's two independent checks
# (A: analytic eta-gradient vs FD, B: fixed-contribution fold vs independent operator recompute) to
# the aml-active path exercised by the four new pieces:
#   1. d_delta_dual_d_eta_active_and_nustar_cross (cm_originzc_cross_production.jl)
#   2. originzc_cross_fixed_contribution's new aml-aware mean-block branch
#   3. cm_originzc_cross_production_gradient's new aml branch (chain-rule-into-gp/A_dd via
#      apply_focal_kstar_chain_rule!, reused UNCHANGED from the base family)
#   4. run_originzc_cross_upper's new aml plumbing (exercised via a short outer-search smoke test)
#
# Key simplification for check (A): d_delta_dual_d_eta_origin_fd (cm_originzc_production.jl,
# UNCHANGED, reused verbatim) already computes a central-FD gradient at EVERY dense nu_eff
# coordinate by perturbing eta=log(nu) there and reoptimizing -- including the coordinate at
# aml.dense_omit_idx. Since aml only removes a MEAN target column (mean_active_origins already
# excludes the focal origin at level kstar, so the row does not exist regardless of what number sits
# in the dense nu vector there) and the PAIR block is completely unaffected by aml, perturbing
# nu_eff at dense_omit_idx is a perfectly meaningful probe of d(Delta_dual)/d(nu_star) directly
# (chain rule eta=log(nu) at that slot). So gather_active_grad(aml, d_delta_dual_d_eta_origin_fd(...))
# gives BOTH the active-eta FD gradient AND the FD d(Delta_dual)/d(nu_star) in one call, in exactly
# the shape d_delta_dual_d_eta_active_and_nustar_cross returns -- no new FD helper needed.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/verify_ozc_cross_variant_d_d4_2026-08-09.jl
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
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl",
          "compressed_live.jl", "autarky_cf.jl", "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "originzc_cross_outer_driver_2026-08-09.jl"]
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
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

# kstar chosen purely to exercise the Variant D MECHANISM (row-omission math), not to match
# ctx.σ-1 economically -- D4's default σHat=2.5 (ad_benchmark/setup_context.jl) gives a
# non-integer sigma-1, so real production Variant D only ever runs at D20/σHat=3.0 (kstar=2).
# nu_star_value_and_dgrad/apply_focal_kstar_chain_rule! are pure functions of theta/sigma with no
# dependence on kstar's economic identity, so this is a valid mechanical/gradient-consistency test.
const KSTAR = 2

for (K_mean, K_pair) in [(2, 2), (3, 3)]
    println("\n==== D4 OZC-CROSS Variant D K_mean=$K_mean K_pair=$K_pair kstar=$KSTAR ====")
    layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
    aml = ActiveMeanLayout(layout, ctx.bi, KSTAR, D)
    check("K=$K_mean/$K_pair: aml.active", aml.active)
    println("    dense_omit_idx=$(aml.dense_omit_idx)  n_eta: $(n_eta(layout)) -> $(aml.n_eta_active)")

    pcx = build_originzc_cross_production_context(ctx, CS, layout; aml = aml)
    check("K=$K_mean/$K_pair: n_mean reflects row omission", pcx.aug.n_mean == n_eta(layout) - 1)
    check("K=$K_mean/$K_pair: n_pair unaffected by aml", pcx.aug.n_pair == K_pair^2 * div(D * (D - 1), 2))

    νfull0_dense_theory = nu0_origin(K_mean, D)
    nu_star_calib = originzc_profiled_nu_value(x_free_calib, ctx)
    println("    nu_star (derived, calib point) = $nu_star_calib  (theoretical dense value at omitted slot = $(νfull0_dense_theory[aml.dense_omit_idx]))")
    νfull_active0, _ = gather_active_grad(aml, νfull0_dense_theory)
    # D4's default σHat=2.5 (ad_benchmark/setup_context.jl) makes sigma-1=1.5 non-integer, so KSTAR=2
    # here does NOT correspond to the genuine autarky-redundant level -- confirmed live: scattering
    # the ACTUAL derived nu_star_calib into this slot makes the inner solve infeasible (nStatus=-300)
    # at K=2/2, AND the control run against the UNMODIFIED base family at the identical point/seed
    # gives the IDENTICAL -300 (see control_base_variant_d_d4.jl output) -- confirming this is a
    # genuine economic infeasibility of this (point, mismatched-kstar, derived-value) combination,
    # not a bug in any new OZC-CROSS code (feedback-control-against-base-family-before-bug-hunting).
    # nu_star_value_and_dgrad/apply_focal_kstar_chain_rule! (the formula that PRODUCES nu_star_calib)
    # are reused UNCHANGED from the base family and already validated there -- not what this task
    # needs to re-verify. So checks (A)/(A')/(B) below, which test the NEW row-omission accumulation
    # machinery itself (not the specific economic value at the omitted slot), use the THEORETICAL
    # homogeneous value instead -- this keeps the point feasible while still exercising every piece
    # of new code (ragged aml offsets, mean/pair-block wiring, fixed-contribution fold) exactly as
    # the derived value would. Check (C) below (the gp-chain-rule wiring, which specifically depends
    # on the REAL derived-nu_star formula) is run separately, gated on convergence, and the
    # economically-real case (kstar=2=sigma-1 at sigma=3.0) gets its full end-to-end test at D20
    # production scale later in this task -- that is where the derived value is actually correct.
    nu_star_mechanical = νfull0_dense_theory[aml.dense_omit_idx]
    νfull_dense0 = scatter_nu_eff(aml, νfull_active0, nu_star_mechanical)
    check("K=$K_mean/$K_pair: scatter_nu_eff round-trips active/derived values back to dense",
          νfull_dense0[aml.dense_omit_idx] == nu_star_mechanical &&
          all(νfull_dense0[d] == νfull0_dense_theory[d] for d in 1:length(νfull0_dense_theory) if d != aml.dense_omit_idx))

    base, verify = archOZ_verified_state(x_free_calib, νfull_dense0, pcx.ctx_cm)
    println("    inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)  m_mean=$(verify.m_mean)")
    check("K=$K_mean/$K_pair: inner solve converged (nStatus in (0,-100,-101,-103))", verify.inner_status in (0, -100, -101, -103))
    check("K=$K_mean/$K_pair: KKT moment residual ~0", verify.max_abs_moment_kkt_resid < 1e-6)

    # Direct residual check on the ACTIVE columns only (omitted origin has no mean restriction at
    # kstar at all -- recovered_mean_residuals_origin over Zk_active/targets_active, mirroring
    # originzc_cross_fixed_contribution's own subsetting).
    m_weights = base.m_star
    active_k = aml.mean_active_origins[KSTAR]
    νo_kstar_active = mean_targets(layout, νfull_dense0, KSTAR, D)[active_k]
    r_mean = recovered_mean_residuals_origin(m_weights, pcx.aug.Zraw_all[KSTAR][:, active_k], νo_kstar_active)
    check("K=$K_mean/$K_pair: active-column mean level k*=$KSTAR residual ~0 (max=$(maximum(abs.(r_mean))))", maximum(abs.(r_mean)) < 1e-6)
    levels = cross_pair_level_index(K_pair)
    for (klin, (k1, k2)) in enumerate(levels)
        νprod = pair_targets(layout, νfull_dense0, klin, D)
        r = recovered_pair_residuals_origin(m_weights, pcx.aug.Zpairraw_all[klin], νprod)
        check("K=$K_mean/$K_pair: cross-pair (k1=$k1,k2=$k2) residual ~0 (max=$(maximum(abs.(r))))", maximum(abs.(r)) < 1e-6)
    end

    # ---- Check (B) FIRST (test-ordering lesson from last session, MASTER.md 2026-08-09):
    # originzc_cross_fixed_contribution's NEW aml-aware mean branch vs independent operator-FG q0 ----
    cache0 = build_lfix_base_cache(x_free_calib, pcx.ctx_cm, base)
    contrib0 = originzc_cross_fixed_contribution(base, pcx.aug, νfull_dense0)
    q0_via_fold = cache0.q0 .- contrib0
    st = pcx.octx.fg_lookup_st
    q0_via_operator = copy(dual_index!(st, vcat(base.ζstar, base.λstar)))
    d_q0 = maximum(abs.(q0_via_fold .- q0_via_operator))
    println("    (B) fixed-contribution-fold q0 vs independent operator-FG q0: max abs diff=$d_q0")
    check("K=$K_mean/$K_pair: (B) originzc_cross_fixed_contribution (aml) matches independent operator recompute", d_q0 < 1e-8)

    # ---- Check (A): d_delta_dual_d_eta_active_and_nustar_cross vs reoptimized central FD ----
    # d_delta_dual_d_eta_origin_fd (UNCHANGED) probes EVERY dense coordinate, including
    # dense_omit_idx -- gather_active_grad splits that into (active-eta FD grad, FD d(Delta)/d(nu_star)).
    g_dense_fd = d_delta_dual_d_eta_origin_fd(x_free_calib, νfull_dense0, pcx.ctx_cm; h = 1e-4)
    g_active_fd, g_omit_fd_raw = gather_active_grad(aml, g_dense_fd)
    # g_omit_fd_raw is d(Delta)/d(eta_at_omitted_slot) = nu_star * d(Delta)/d(nu_star) (chain rule
    # eta=log(nu), same as my function's own eta_grad_dense = nu_eff.*d_nu convention) -- convert to
    # raw d(Delta)/d(nu_star) for comparison against d_delta_d_nu_star (which is un-nu-multiplied).
    d_delta_d_nu_star_fd = g_omit_fd_raw / nu_star_mechanical

    eta_grad_active, d_delta_d_nu_star = d_delta_dual_d_eta_active_and_nustar_cross(base.λstar, pcx.aug, aml, νfull_dense0; mean_m = verify.m_mean)
    d_ag = maximum(abs.(eta_grad_active .- g_active_fd))
    rel = d_ag / max(1e-8, maximum(abs.(g_active_fd)))
    println("    (A) analytic active eta-grad vs FD: max abs diff=$d_ag  max rel diff=$rel")
    println("        analytic=$eta_grad_active")
    println("        fd      =$g_active_fd")
    check("K=$K_mean/$K_pair: (A) analytic active eta-gradient matches reoptimized FD", d_ag < 5e-3 && rel < 1e-2)

    d_nustar = abs(d_delta_d_nu_star - d_delta_d_nu_star_fd)
    rel_nustar = d_nustar / max(1e-8, abs(d_delta_d_nu_star_fd))
    println("    (A') analytic d(Delta)/d(nu_star) vs FD: analytic=$d_delta_d_nu_star  fd=$d_delta_d_nu_star_fd  abs diff=$d_nustar  rel diff=$rel_nustar")
    check("K=$K_mean/$K_pair: (A') analytic d_delta_d_nu_star matches reoptimized FD", d_nustar < 5e-3 && rel_nustar < 1e-2)

    # ---- Full production gradient wiring check: cm_originzc_cross_production_gradient's aml
    # branch (chain-rule-into-gp via apply_focal_kstar_chain_rule!, REUSED UNCHANGED from the base
    # family) vs FD of Delta_dual w.r.t. gp at fixed (zfree, eta_active) -- the ONE piece of the
    # chain not already covered by (A)/(A'). MUST use the REAL derived nu_star (originzc_profiled_
    # nu_value) throughout, self-consistently, at both the baseline and every perturbed gp -- unlike
    # (A)/(A')/(B) above, this specifically tests the nu_star(gp) WIRING, not just the row-omission
    # accumulation math, so the mechanical (theoretical) substitute used above would test nothing
    # (its "nu_star" doesn't actually move with gp). Since kstar=$KSTAR isn't the genuine sigma-1
    # level at D4's sigma=$(ctx.σ) (see note above the scatter_nu_eff call), the REAL derived value
    # may make this exact point infeasible (-300, confirmed for K=2/2 against the control base-family
    # run at the identical point) -- gated with try/catch rather than asserted, since the genuine
    # end-to-end validation of this real-value path happens at D20/sigma=3.0 (kstar=2=sigma-1
    # genuinely) later in this task. ----
    νfull_active0_for_grad, _ = gather_active_grad(aml, νfull0_dense_theory)
    h_gp = 1e-5
    gp0 = x_free_calib[1]
    function delta_at_gp(gp_pert)
        xfp = vcat(gp_pert, x_free_calib[2:end])
        nu_star_p = originzc_profiled_nu_value(xfp, ctx)
        νp = scatter_nu_eff(aml, νfull_active0_for_grad, nu_star_p)
        _, vp = archOZ_verified_state(xfp, νp, pcx.ctx_cm)
        return vp.Delta_dual
    end
    try
        νfull_dense_real0 = scatter_nu_eff(aml, νfull_active0_for_grad, nu_star_calib)
        g_ext, _ = cm_originzc_cross_production_gradient(x_free_calib, νfull_dense_real0, pcx, ctx, pe; threaded = true)
        g_gp_fd = (delta_at_gp(gp0 + h_gp) - delta_at_gp(gp0 - h_gp)) / (2h_gp)
        d_gp = abs(g_ext[1] - g_gp_fd)
        rel_gp = d_gp / max(1e-8, abs(g_gp_fd))
        println("    (C) full g_econ[1] (d/dgp, includes chain-rule-into-nu_star term) vs FD: analytic=$(g_ext[1])  fd=$g_gp_fd  abs diff=$d_gp  rel diff=$rel_gp")
        check("K=$K_mean/$K_pair: (C) cm_originzc_cross_production_gradient's aml g_econ[1] matches reoptimized FD", d_gp < 5e-3 && rel_gp < 1e-2)
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        println("    (C) SKIPPED: real derived nu_star=$nu_star_calib is infeasible at this D4 point (kstar=$KSTAR != true sigma-1=$(ctx.σ-1) here) -- ",
                 "confirmed a property of the point/mismatched-kstar, not a code bug, via the base-family control run. ",
                 "Real end-to-end validation of this exact wiring happens at D20/sigma=3.0 (kstar=2=sigma-1 genuinely).")
    end
end

println("\n", ALL_PASS[] ? "ALL PASS" : "SOME FAILED")
exit(ALL_PASS[] ? 0 : 1)
