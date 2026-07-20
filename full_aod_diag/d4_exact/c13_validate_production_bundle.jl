# Continuation 13, Section 3A validation: Architecture B (moment construction) + Architecture C
# (structured Hessian) combined into ONE obj for the first time (previously validated separately:
# B against A's Hessian, C against A's moments) -- plus archC_base_state/cm_production_gradient
# reproducing the already-validated composite_gradient_at_fast_cm results via a faster inner solve.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Random, Statistics

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(1301)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.04 .* randn(length(x_free_perturbed) - 1))

println("="^100)
println("SECTION 1: production context builds cleanly, dense-reference agreement (value only)")
println("="^100)
for L in (10, 20, 50)
    aug_ref = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    ctx_ref = merge(ctx, (obj = aug_ref.obj_cm,))
    r_ref = evaluate_fullA(x_free_calib, ctx_ref; use_cache = false, warm = false)

    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored)
    K_prod, base_prod = cm_production_value(x_free_calib, pcx)

    @printf "[L=%2d] dense: nStatus=%d Delta_dual=%.8f | production(B+C): nStatus=%d Delta_dual=%.8f | diff=%.3e\n" L r_ref.inner_status (-r_ref.zeta) base_prod.inner_status (-base_prod.ζstar) abs((-r_ref.zeta) - (-base_prod.ζstar))
end
println()

println("="^100)
println("SECTION 2: production gradient (ArchC base + CM-aware Lfix) reproduces the already-")
println("validated dense-inner-solve CM-aware gradient (composite_gradient_at_fast_cm), calib+perturbed")
println("="^100)
for (label, xf) in [("calibration", x_free_calib), ("perturbed", x_free_perturbed)]
    local aug = build_cm_augmented_obj(ctx, CS; L = 10, contrasts = :anchored)
    local ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    local bins = cm_bin_indices_for(ctx, aug)
    local base_dense = solve_base_state(xf, ctx_cm)
    local cache_dense = build_lfix_base_cache_cm(xf, ctx_cm, base_dense, ctx, aug, bins)
    g_dense, _ = composite_gradient_at_fast_cm(xf, ctx_cm, pe, ctx, aug, bins; base = base_dense, cache = cache_dense)

    local pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored)
    g_prod, _ = cm_production_gradient(xf, pcx, ctx, pe)

    @printf "[%s] max|g_dense-g_prod|=%.3e  ||g_dense||=%.6e  ||g_prod||=%.6e  cosine=%.8f\n" label maximum(abs.(g_dense .- g_prod)) norm(g_dense) norm(g_prod) (dot(g_dense,g_prod)/(norm(g_dense)*norm(g_prod)))
end
println()

println("="^100)
println("SECTION 3: nested-grid probs plug into the production bundle too")
println("="^100)
snaps = nested_grid_sequence([10, 20, 50])
pcx10 = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = snaps[10])
K10, base10 = cm_production_value(x_free_calib, pcx10)
@printf "  nested-L10 production: nStatus=%d Delta_dual=%.8f\n" base10.inner_status (-base10.ζstar)
println()

println("="^100)
println("SECTION 4: structurally-infeasible point correctly rejected by the production bundle")
println("="^100)
x_infeasible = copy(x_free_calib)
bad_idx = ctx.Aod_offset + 1
x_infeasible[bad_idx] = 1e-6
pcx_inf = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored)
try
    K_inf, base_inf = cm_production_value(x_infeasible, pcx_inf)
    println("  UNEXPECTED: no error, nStatus=", base_inf.inner_status)
catch e
    println("  correctly rejected: ", sprint(showerror, e)[1:min(120,end)])
end
println("DONE")
