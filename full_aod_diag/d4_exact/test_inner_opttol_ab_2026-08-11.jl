# Is the inner solve chasing a tolerance it cannot reach? (2026-08-11)
#
# full_aod_diag/ek_inner.opt sets opttol = opttol_abs = 1e-12, but memory
# `knitro-minus100-is-opttol-1e-12-vs-achievable-floor` records that the ACHIEVABLE optimality floor
# on this problem is ~1e-11. That would explain the observed signature exactly: healthy inner solves
# return -100 (KN_RC_NEAR_OPT, "stopping tests satisfied within a factor of 100" -- i.e. it reached
# ~1e-10) after ~38 Hessian calls and ~130 s, instead of 0.
#
# A/B at a FIXED point (the 8h run's own incumbent), same context, same warm state, varying ONLY
# opttol/opttol_abs. Everything else in the .opt is untouched -- notably feastol stays 1e-12, so
# this changes when KNITRO is SATISFIED with optimality, not what it considers feasible.
#
# The number that decides it is not the speedup but Delta_dual: if loosening the tolerance changes
# the answer by more than the solver's own noise, the speed is not free. Reported as an explicit
# relative difference against the 1e-12 baseline.
#
# Usage: julia --project=. -t 10 .../test_inner_opttol_ab_2026-08-11.jl
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl",
          "no_dense_g_counters.jl", "zc_restriction_operator.jl", "zc_restriction_operator_ragged.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "shared_a_gradient.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_meanzc_lookup_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl", "cm_originzc_cross_cplus.jl",
          "cm_meanzc_cross_target_layout.jl", "cm_meanzc_cross_moments.jl", "cm_meanzc_cross_production.jl", "cm_meanzc_cross_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl", "country_resolve.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Serialization
lp(xs...) = (println(xs...); flush(stdout))

const KK = 3; const KSTAR = 2; const W = 100_000
const OPTS = ["ek_inner.opt" => "1e-12 (PRODUCTION)",
              "ek_inner_opttol_1e-10.opt" => "1e-10",
              "ek_inner_opttol_1e-08.opt" => "1e-08"]
const CKPT = joinpath(_D4E, "..", "..", "results", "ozc_cross_production_smoke_2026-08-09",
                      "K3_W100000_sobol_randomized_upper8h_2026-08-10", "ozc_cross_K3_W100000_latest.jls")

lp("="^100)
lp("INNER opttol A/B at a FIXED point   (OZC-CROSS, D20, W=", W, ", K=", KK, "/", KK, ")")
lp("="^100)

ck = load_cm_checkpoint_v10(CKPT); bf = ck.best_feasible
w0 = collect(Float64, bf.w)
lp("point = the 8h run's own incumbent: gp=", bf.gp, "  Delta=", bf.Delta)

results = NamedTuple[]
for (optfile, label) in OPTS
    ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized,
        draw_seed = 20260719, destination_sample = :exclude_row, exclude_diagonal_gravity = true,
        gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(), σHat = 3.0,
        inner_lower_limit = -10.0,
        inner_loop_opt = joinpath(_D4E, "..", optfile))
    ctx = attach_compressed_factual_workspace(ctx, ctx.D, ctx.D_dest, W)
    D = ctx.D; Ddest = ctx.D_dest
    layout = OriginByPowerCrossLayout(D, KK, KK)
    aml = ActiveMeanLayout(layout, ctx.bi, KSTAR, D)
    pcx = build_originzc_cross_production_context(ctx, CS, layout; aml = aml)

    # decode w0 exactly as the driver does (:powered_aspace), then scatter the Variant-D eta block
    pe = build_pivot_elimination(ctx); th = cm_fixed_theta(ctx); xy = precompute_cm_aspace_xy(ctx)
    D2 = length(w0) - aml.n_eta_active
    xf = x_free_from_w(vcat(w0[1], cm_z_from_a(w0[2:D2], th, xy, pe)), pe)
    nu_act = exp.(w0[D2+1:end])
    nu = scatter_nu_eff(aml, nu_act, originzc_profiled_nu_value(xf, ctx))

    theta_econ = CS.reconstruct_full(xf, pcx.ctx_cm.m)
    theta_ext = vcat(theta_econ, nu)
    pcx.ctx_cm.octx.nu_ref[] = collect(nu)
    it0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    K, x, nStatus, n_fg, n_hess = _originzc_fg_dispatch(pcx.ctx_cm, pcx.ctx_cm.obj, theta_ext)
    wall = time() - t0
    iters = CS.INNER_ITERS_TOTAL[] - it0
    Delta = -pcx.ctx_cm.obj.H_save * (-1.0)^pcx.ctx_cm.obj.find_smallest   # H_save carries the sign
    push!(results, (label = label, status = nStatus, iters = iters, n_fg = n_fg, n_hess = n_hess,
                    wall = wall, K = K))
    @printf("  opttol %-18s status=%5d iters=%4d n_fg=%5d n_hess=%4d wall=%8.2fs  K=%.16g\n",
            label, nStatus, iters, n_fg, n_hess, wall, K)
    flush(stdout)
end

lp("\n", "="^100)
lp("VERDICT")
lp("="^100)
base = results[1]
@printf("  %-20s %8s %7s %8s %10s %14s %14s\n", "opttol", "status", "iters", "n_hess", "wall", "speedup", "rel diff in K")
for r in results
    @printf("  %-20s %8d %7d %8d %9.2fs %13.2fx %14.3e\n", r.label, r.status, r.iters, r.n_hess, r.wall,
            base.wall / r.wall, abs(r.K - base.K) / max(abs(base.K), eps()))
end
lp("\n  K is the inner objective the driver reads (Delta_dual = -K up to the find_smallest sign).")
lp("  A loosened tolerance is FREE only if `rel diff in K` stays at the solver's own noise level")
lp("  (~1e-11 on this problem); a larger difference means the speed is bought with accuracy.")
flush(stdout)
