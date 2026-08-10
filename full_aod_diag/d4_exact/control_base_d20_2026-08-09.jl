# Control check (2026-08-09): does the UNMODIFIED base origin-ZC family also fail (nStatus=-300)
# at D20/:exclude_row/W=10000/K_mean=1/K_pair=1 with the same theoretical-mean nu0 and calibration
# x_free? If yes, this is a nu0/W-conditioning property of D20 real data at this W, not a bug in
# OZC-CROSS. K_mean=1/K_pair=1 diagonal-only IS mathematically identical to OZC-CROSS's K=1/1
# (single combo (1,1)) -- confirmed at D4 already.
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "country_resolve.jl"]
    include(joinpath(D4X, f))
end
using SpecialFunctions: gamma

W = 10_000
println("Building D20 real-data context (W=$W)...")
flush(stdout)
ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = 20260719, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
D = ctx.D; Ddest = ctx.D_dest
println("ctx built. D=$D D_dest=$Ddest μHat=$(ctx.μHat)")
flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
println("free_idx matches pivot-manifold reconstruction? (sanity)")
nu0_origin(K::Int, D::Int) = vcat([fill(gamma(1 - ctx.μHat * k), D) for k in 1:K]...)

for (K_mean, K_pair) in [(1, 0), (1, 1)]
    layout = OriginByPowerLayout(D, K_mean, K_pair)
    νfull0 = nu0_origin(K_mean, D)
    pcx = build_originzc_production_context(ctx, CS, layout; fg_backend = :operator)
    try
        base, verify = archOZ_verified_state(x_free_calib, νfull0, pcx.ctx_cm)
        println("BASE FAMILY K=$K_mean/$K_pair: CONVERGED inner_status=$(verify.inner_status) Delta_dual=$(verify.Delta_dual) max_abs_moment_kkt_resid=$(verify.max_abs_moment_kkt_resid)")
    catch e
        println("BASE FAMILY K=$K_mean/$K_pair: FAILED -- ", sprint(showerror, e))
    end
    flush(stdout)
end
