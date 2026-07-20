# Smoke test: nested_quantile_grids.jl's probs plug into precalc_common_marginals_cdf /
# build_cm_augmented_obj via the new `probs=` kwarg, default path (probs=nothing) unchanged,
# and the CM-aware Lfix gradient (lfix_cm_aware.jl) works against a probs-built aug too.
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
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
using Printf, LinearAlgebra, Statistics

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
x_free_calib = ctx.θ0_up[ctx.free_idx]

snaps = nested_grid_sequence([10, 20, 50])

println("=== default (k/L) vs explicit-probs path, both give a valid CM-augmented context ===")
aug_default = build_cm_augmented_obj(ctx, CS; L = 10, contrasts = :anchored)
aug_nested = build_cm_augmented_obj(ctx, CS; L = 10, contrasts = :anchored, probs = snaps[10])
@printf "  default z[1:3] = %s\n" string(round.(aug_default.z[1:3], digits=4))
@printf "  nested  z[1:3] = %s\n" string(round.(aug_nested.z[1:3], digits=4))
@printf "  z differ (expected, different grids): %s\n" (aug_default.z != aug_nested.z)

println()
println("=== nested-grid CM-aware inner solve + gradient at calibration, L=10 (nested probs) ===")
bins10 = cm_bin_indices_for(ctx, aug_nested)
ctx_cm10 = merge(ctx, (obj = aug_nested.obj_cm,))
base10 = solve_base_state(x_free_calib, ctx_cm10)
cache10 = build_lfix_base_cache_cm(x_free_calib, ctx_cm10, base10, ctx, aug_nested, bins10)
g10, _ = composite_gradient_at_fast_cm(x_free_calib, ctx_cm10, pe, ctx, aug_nested, bins10; base = base10, cache = cache10)
@printf "  nStatus=%d  Delta_dual=-zeta*=%.6f  ||g||=%.6e  gamma_component=%.6e\n" base10.inner_status (-base10.ζstar) norm(g10) g10[1]

println()
println("=== nesting sanity: L=10 nested set IS a subset of L=20 nested set's induced z's index positions ===")
aug20 = build_cm_augmented_obj(ctx, CS; L = 20, contrasts = :anchored, probs = snaps[20])
z10 = Set(round.(aug_nested.z, digits=10))
z20 = Set(round.(aug20.z, digits=10))
println("  z(Q10) subset of z(Q20): ", issubset(z10, z20))
println("DONE")
