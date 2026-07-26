# Transformed-A restricted-family port, gate 1: cross-check cm_aspace_coordinate.jl's
# independently-re-derived a<->z conversion against the ALREADY-VALIDATED
# flexible_theta_aspace_production.jl::z_from_a/a_from_z/precompute_aspace_XY at a real D=20
# point -- reconstruct via BOTH paths and diff directly, per this project's own standing rule
# (never assume two differently-constructed "equivalent" points actually agree; see CLAUDE.md's
# A_od-calibration warning). Also gates the round-trip identity (a->z->a, z->a->z) and confirms
# cm_z_from_a/cm_a_from_z reconstruct the SAME logA_full pivot_expand(pe) would from a genuine
# calibrated z-space point.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl",
          "cm_screen_bridge.jl","gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl",
          "c10_d20_production_driver.jl","flexible_theta.jl","flexible_theta_aspace_production.jl",
          "outer_coordinate_layout.jl","cm_aspace_coordinate.jl","cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using LinearAlgebra, Random, Printf

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

println("="^90); println("Real D=20/W=80,000 context"); println("="^90)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe = build_pivot_elimination(ctx)
theta_pe_implicit = 1.0 / ctx.fixed_vals[1]
theta_cm = cm_fixed_theta(ctx)
check("cm_fixed_theta(ctx) matches build_pivot_elimination's own implicit theta EXACTLY", theta_cm == theta_pe_implicit)

xy_mine = precompute_cm_aspace_xy(ctx)
xy_ref = precompute_aspace_XY(ctx)   # flexible_theta_aspace_production.jl, ALREADY VALIDATED
d_logX = maximum(abs.(xy_mine.logX .- xy_ref.logX))
d_logY = maximum(abs.(xy_mine.logY .- xy_ref.logY))
@printf "max|logX_mine - logX_ref| = %.3e   max|logY_mine - logY_ref| = %.3e\n" d_logX d_logY
check("precompute_cm_aspace_xy matches flexible_theta_aspace_production's precompute_aspace_XY EXACTLY (same formula)", d_logX == 0.0 && d_logY == 0.0)

# genuine calibrated z-space point
x_free_calib = ctx.θ0_up[ctx.free_idx]
gp_calib = x_free_calib[1]
logA_full_calib = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
z_nonpivot_calib = pivot_reduce(logA_full_calib, pe)

println("="^90); println("Cross-check against flexible_theta_aspace_production's z_from_a/a_from_z"); println("="^90)
a_nonpivot_mine = cm_a_from_z(z_nonpivot_calib, theta_cm, xy_mine, pe)
a_full_ref = a_from_z(logA_full_calib, theta_cm, xy_ref)   # D x Ddest matrix, reference impl
a_nonpivot_ref = vec(a_full_ref)[pe.other_idx]
d_a = maximum(abs.(a_nonpivot_mine .- a_nonpivot_ref))
@printf "max|a_mine - a_ref| at genuine calibration point = %.3e\n" d_a
check("cm_a_from_z matches flexible_theta_aspace_production's a_from_z EXACTLY at calibration", d_a == 0.0)

z_full_reconstructed_mine = cm_z_from_a(a_nonpivot_mine, theta_cm, xy_mine, pe)
z_full_reconstructed_ref = vec(z_from_a(a_full_ref, theta_cm, xy_ref))[pe.other_idx]
d_z_roundtrip_mine = maximum(abs.(z_full_reconstructed_mine .- z_nonpivot_calib))
d_z_ref_cross = maximum(abs.(z_full_reconstructed_ref .- z_nonpivot_calib))
@printf "round-trip |cm_z_from_a(cm_a_from_z(z)) - z| = %.3e (mine)   |z_from_a(a_from_z(z)) - z| = %.3e (ref)\n" d_z_roundtrip_mine d_z_ref_cross
check("cm_z_from_a . cm_a_from_z round-trips to genuine z-space calibration point (< 1e-9)", d_z_roundtrip_mine < 1e-9)

println("="^90); println("Reconstruct the SAME logA_full both ways (the real correctness bar, not just a<->z consistency)"); println("="^90)
logA_full_via_z_direct = pivot_expand(z_nonpivot_calib, pe)   # EXISTING production path, byte-identical to before this port
logA_full_via_a_then_z = pivot_expand(cm_z_from_a(a_nonpivot_mine, theta_cm, xy_mine, pe), pe)   # NEW a-space decode path
d_reconstruct = maximum(abs.(logA_full_via_z_direct .- logA_full_via_a_then_z))
@printf "max|logA_full(direct z) - logA_full(via a then z)| = %.3e\n" d_reconstruct
check("a-space decode reconstructs the IDENTICAL logA_full as the direct z-space path at the SAME calibration point (< 1e-9)", d_reconstruct < 1e-9)

println("="^90); println("Gradient rescale sanity: scalar -theta, cross-check against gradient_transform_unified"); println("="^90)
fake_gz = vcat(0.37, randn(MersenneTwister(1), length(z_nonpivot_calib)))   # [d/dgp; d/dz_nonpivot], arbitrary but fixed
mine_rescaled = copy(fake_gz); mine_rescaled[2:end] .*= (-theta_cm)
layout_ref = make_layout(trade_elasticity_mode = :fixed, A_coordinate_mode = :powered_aspace)
ref_rescaled = gradient_transform_unified(fake_gz, theta_cm, 0.987, layout_ref, nothing)
d_grad = maximum(abs.(mine_rescaled .- ref_rescaled))
@printf "max|mine - gradient_transform_unified| = %.3e\n" d_grad
check("gradient rescale (-theta scalar) matches gradient_transform_unified EXACTLY", d_grad == 0.0)

println()
println("="^90)
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
