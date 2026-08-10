# Variant D applied to OZC-CROSS: the REAL end-to-end validation this task exists for (2026-08-09
# handover follow-up). At D20 real Brazil-Korea data, W=100,000, sigma=3.0 (so kstar=2=sigma-1
# GENUINELY, unlike the D4 mechanical smoke test in verify_ozc_cross_variant_d_d4_2026-08-09.jl,
# where sigma=2.5 made kstar=2 an artificial mismatch), compares FOUR points at the SAME
# K_mean=K_pair=3 config and calibration point:
#   1. base family (OriginByPowerLayout), no aml       -- known from last session: Delta*~0.00949
#   2. base family, aml kstar=2 (Variant D)             -- NEW: does Variant D reduce this further?
#   3. OZC-CROSS (OriginByPowerCrossLayout), no aml     -- known from last session: Delta*~0.0667 (7x)
#   4. OZC-CROSS, aml kstar=2 (Variant D)                -- THE open item: does Delta* come back down
#                                                            toward a sane multiple of (2)?
# Same context-building convention as investigate_delta_jump_2026-08-09.jl (same session/task):
# x_free_calib = ctx.θ0_up[ctx.free_idx] directly (native free-parameter units) is the CORRECT,
# already-validated input for a direct archOZ_verified_state call -- the pivot-elimination
# (powered_aspace) encoding noted in the handover's lesson #2 is specifically for constructing w0
# for the OUTER KNITRO DRIVER's z-space search, not for this kind of direct verified-state probe.
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
          "compressed_live.jl", "autarky_cf.jl", "cm_checkpoint.jl", "cm_originzc_checkpoint.jl",
          "cm_originzc_cross_moments.jl", "cm_originzc_cross_target_layout.jl", "cm_originzc_cross_production.jl"]
    include(joinpath(D4X, f))
end
using Statistics, LinearAlgebra
using SpecialFunctions: gamma

W = 100_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D
println("ctx built. D=$D sigma=$(ctx.σ) muHat=$(ctx.μHat) bi=$(ctx.bi)")
flush(stdout)
x_free_calib = ctx.θ0_up[ctx.free_idx]

const K_mean = 3
const K_pair = 3
const KSTAR = 2   # = sigma - 1, genuinely, at sigma=3.0
νfull0_dense_theory = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K_mean]...)

results = Dict{String,Any}()

function run_point(label, layout, aml)
    println("\n---- $label ----")
    flush(stdout)
    if layout isa OriginByPowerCrossLayout
        pcx = build_originzc_cross_production_context(ctx, CS, layout; aml = aml)
    else
        pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator, aml = aml)
    end
    if aml === nothing
        νfull = νfull0_dense_theory
    else
        nu_star = originzc_profiled_nu_value(x_free_calib, ctx)
        println("  nu_star (derived) = $nu_star   (theoretical at omitted slot = $(νfull0_dense_theory[aml.dense_omit_idx]))")
        νfull_active, _ = gather_active_grad(aml, νfull0_dense_theory)
        νfull = scatter_nu_eff(aml, νfull_active, nu_star)
    end
    t = @elapsed begin
        base, verify = archOZ_verified_state(x_free_calib, νfull, pcx.ctx_cm)
    end
    println("  solve: $(round(t, digits=1))s  inner_status=$(verify.inner_status)  Delta_dual=$(verify.Delta_dual)  max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
    flush(stdout)
    results[label] = (Delta_dual = verify.Delta_dual, inner_status = verify.inner_status, wall = t)
    return base, verify, pcx
end

base_layout = OriginByPowerLayout(D, K_mean, K_pair)
cross_layout = OriginByPowerCrossLayout(D, K_mean, K_pair)
base_aml = ActiveMeanLayout(base_layout, ctx.bi, KSTAR, D)
cross_aml = ActiveMeanLayout(cross_layout, ctx.bi, KSTAR, D)

run_point("1. base family, no aml", base_layout, nothing)
run_point("2. base family, Variant D (kstar=2)", base_layout, base_aml)
run_point("3. OZC-CROSS, no aml", cross_layout, nothing)
run_point("4. OZC-CROSS, Variant D (kstar=2)", cross_layout, cross_aml)

println("\n==== SUMMARY ====")
for label in ["1. base family, no aml", "2. base family, Variant D (kstar=2)",
              "3. OZC-CROSS, no aml", "4. OZC-CROSS, Variant D (kstar=2)"]
    r = results[label]
    println("  $label: Delta_dual=$(r.Delta_dual)  inner_status=$(r.inner_status)  wall=$(round(r.wall,digits=1))s")
end
r1 = results["1. base family, no aml"].Delta_dual
r2 = results["2. base family, Variant D (kstar=2)"].Delta_dual
r3 = results["3. OZC-CROSS, no aml"].Delta_dual
r4 = results["4. OZC-CROSS, Variant D (kstar=2)"].Delta_dual
println("\n  ratio (3 no-aml)/(1 no-aml)   [known 7x jump]           = $(r3/r1)")
println("  ratio (4 Variant D)/(2 Variant D)  [does the jump shrink?] = $(r4/r2)")
println("  ratio (2 Variant D)/(1 no-aml)  [base: does Variant D itself change Delta*?] = $(r2/r1)")
println("  ratio (4 Variant D)/(3 no-aml)  [cross: does Variant D reduce Delta* vs its own no-aml baseline?] = $(r4/r3)")
