# Outer-gradient FD check at the D20 Variant D calibration point (2026-08-09, user request: "I
# simply don't believe that the calibration point gives such a high delta" -- compare the analytic
# gradient against an independent central-FD gradient AT the solution/calibration point, for both
# families). NOTE: this deliberately does NOT include cm_originzc_checkpoint.jl, which a concurrent
# background agent is actively editing this session (driver-integration task) -- instead reimplements
# the tiny `originzc_profiled_nu_value` formula locally (4 lines, cf_denom/cf_num from
# autarky_cf_scalars, already included via autarky_cf.jl) to avoid any file-conflict/read-race.
#
# Important framing point: Delta_dual AT a fixed calibration point is a pure function VALUE (the
# solved inner dual objective) -- it does not involve any GRADIENT or HESSIAN computation at all.
# A wrong outer gradient could not, by itself, make Delta_dual's reported VALUE at a fixed point
# wrong. What a wrong gradient COULD do is make an outer SEARCH from this point converge somewhere
# bad/wrong. This script checks gradient correctness directly (independent central FD of Delta_dual
# w.r.t. gp, at fixed zfree/eta_active); the driver-integration agent running concurrently is
# separately checking whether a REAL outer search from this point finds a materially different
# optimum than naively evaluating Delta_dual at the calibration point suggests.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl",
          "compressed_live.jl", "autarky_cf.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra, Printf
using SpecialFunctions: gamma

# Local reimplementation of originzc_profiled_nu_value (cm_originzc_checkpoint.jl lines 32-46),
# to avoid including a file the driver-integration agent is concurrently editing. Identical formula.
function local_profiled_nu_value(xf::AbstractVector{Float64}, ctx)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    AodPow = aod_pow_matrix(θ_full, ctx)
    γ_prime_bi = θ_full[3+ctx.D]
    cf_num, cf_denom, _ = autarky_cf_scalars(ctx.obj, AodPow, ctx.σ, γ_prime_bi)
    return cf_denom / cf_num
end

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi) W=$W")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]

const K_mean = 3
const K_pair = 3
const KSTAR = 2
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

# Sanity: confirm local_profiled_nu_value matches the value already recorded from the earlier D20
# run (1.2487738293117998) before trusting it for the FD check below.
nu_star_check = local_profiled_nu_value(x_free_calib, ctx)
@printf("local_profiled_nu_value at calib point = %.16f  (expected 1.2487738293117998 from earlier run)\n", nu_star_check)
flush(stdout)

function gp_fd_check(label, layout, aml)
    println("\n================ $label ================")
    flush(stdout)
    pcx = layout isa OriginByPowerCrossLayout ?
        build_originzc_cross_production_context(ctx, CS, layout; aml = aml) :
        build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    νfull_active0, _ = gather_active_grad(aml, νfull0_dense_theory)

    function nu_at(xf)
        nu_star = local_profiled_nu_value(xf, ctx)
        return scatter_nu_eff(aml, νfull_active0, nu_star)
    end

    νfull0 = nu_at(x_free_calib)
    t0 = @elapsed (base0, verify0) = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
    @printf("  baseline: %.1fs  inner_status=%d  Delta_dual=%.10f\n", t0, verify0.inner_status, verify0.Delta_dual)
    flush(stdout)

    pe = build_pivot_elimination(ctx)
    g_ext, _ = layout isa OriginByPowerCrossLayout ?
        cm_originzc_cross_production_gradient(x_free_calib, νfull0, pcx, ctx, pe; base = base0, verify = verify0, threaded = true) :
        cm_originzc_production_gradient(x_free_calib, νfull0, pcx, ctx, pe; base = base0, verify = verify0, threaded = true)
    g_analytic_gp = g_ext[1]
    @printf("  analytic d(Delta_dual)/d(gp) = %.10f\n", g_analytic_gp)
    flush(stdout)

    h = 1e-4
    gp0 = x_free_calib[1]
    function delta_at_gp(gp_pert)
        xfp = vcat(gp_pert, x_free_calib[2:end])
        νp = nu_at(xfp)
        t = @elapsed (_, vp) = archOZ_verified_state(xfp, νp, pcx.ctx_cm)
        @printf("    probe gp=%.8f -> Delta_dual=%.10f  inner_status=%d  (%.1fs)\n", gp_pert, vp.Delta_dual, vp.inner_status, t)
        flush(stdout)
        return vp.Delta_dual
    end
    d_plus = delta_at_gp(gp0 + h)
    d_minus = delta_at_gp(gp0 - h)
    g_fd_gp = (d_plus - d_minus) / (2h)
    d_abs = abs(g_analytic_gp - g_fd_gp)
    d_rel = d_abs / max(1e-8, abs(g_fd_gp))
    @printf("  FD d(Delta_dual)/d(gp) [h=%.0e] = %.10f\n", h, g_fd_gp)
    @printf("  |analytic - fd| = %.6e   rel = %.6e\n", d_abs, d_rel)
    flush(stdout)
    return (Delta0 = verify0.Delta_dual, g_analytic = g_analytic_gp, g_fd = g_fd_gp)
end

base_layout = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
base_aml = ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D)
cross_aml = ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D)

r_base = gp_fd_check("base family, Variant D (kstar=2)", base_layout, base_aml)
r_cross = gp_fd_check("OZC-CROSS, Variant D (kstar=2)", cross_layout, cross_aml)

println("\n==== SUMMARY ====")
println("  base:  Delta0=$(r_base.Delta0)  analytic_dgp=$(r_base.g_analytic)  fd_dgp=$(r_base.g_fd)")
println("  cross: Delta0=$(r_cross.Delta0)  analytic_dgp=$(r_cross.g_analytic)  fd_dgp=$(r_cross.g_fd)")
