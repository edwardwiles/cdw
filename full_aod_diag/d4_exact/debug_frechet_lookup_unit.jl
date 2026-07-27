# Standalone unit-style check: evaluate CMFrechetLookupState(x,g) directly against the DENSE
# obj(x,g) callable, at a FIXED x -- no KNITRO solve involved -- to isolate whether the kernel
# math itself is correct, independent of any production/KNITRO-wiring bug.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "nested_quantile_grids.jl", "draw_design.jl", "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 10
contrasts = :anchored

pcx_dense = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
    cm_hessian_backend = :structured, inner_fg_backend = :dense_reference)
pcx_lookup = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts,
    cm_hessian_backend = :structured, inner_fg_backend = :cm_frechet_lookup)

obj_dense = pcx_dense.ctx_cm.obj
obj_lookup = pcx_lookup.ctx_cm.obj
θ_full0 = CS.reconstruct_full(x_free_calib, pcx_dense.ctx_cm.m)

# Build obj.H for BOTH (moments! call), matching what inner_loop_internal_* does before any FG eval.
obj_dense.moments!(@view(obj_dense.H[:, 1]), CS.select_G_from_H(obj_dense, obj_dense.H), θ_full0, obj_dense.U, obj_dense)
obj_dense.H[:, 2] .= 1.0
obj_lookup.moments!(@view(obj_lookup.H[:, 1]), CS.select_G_from_H(obj_lookup, obj_lookup.H), θ_full0, obj_lookup.U, obj_lookup)
obj_lookup.H[:, 2] .= 1.0

nvar = CS.inner_loop_number_variables(obj_dense)
println("nvar=", nvar, "  obj_dense.d=", obj_dense.d, "  ncore=", pcx_dense.aug.ncore, "  ncm=", pcx_dense.aug.ncm,
        "  ncm_cm=", pcx_dense.aug.ncm_cm, "  ncm_level=", pcx_dense.aug.ncm_level, "  L=", L)
flush(stdout)

Random.seed!(777)
x0 = CS.inner_loop_initial_values(obj_dense)  # KNITRO's own starting point
xr = x0 .+ 0.01 .* randn(length(x0))          # a random nearby point (nonzero lambda too)

cctx_lookup = pcx_lookup.cctx
bins_u = cctx_lookup.Bidx isa Matrix{UInt32} ? cctx_lookup.Bidx : Matrix{UInt32}(cctx_lookup.Bidx)
ncm_cm = cctx_lookup.ncm - cctx_lookup.L
st = CMFrechetLookupState(obj_lookup, cctx_lookup.NCORE, ncm_cm, cctx_lookup.L, cctx_lookup.L, cctx_lookup.D,
    cctx_lookup.origins, cctx_lookup.refIndex1, bins_u, cctx_lookup.R, pcx_lookup.aug.level_targets)

for (label, xtest) in (("x0 (KNITRO init)", x0), ("xr (random nearby)", xr))
    println("="^30, " ", label, " ", "="^30)
    g_dense = zeros(nvar); g_lookup = zeros(nvar)
    f_dense = obj_dense(xtest, g_dense)
    f_lookup = st(xtest, g_lookup)
    println(@sprintf("f_dense=%.10f  f_lookup=%.10f  diff=%.3e", f_dense, f_lookup, abs(f_dense - f_lookup)))
    gdiff = maximum(abs.(g_dense .- g_lookup))
    println(@sprintf("max|g_dense - g_lookup| = %.3e", gdiff))
    if gdiff > 1e-8
        # locate worst offending coordinate(s)
        idx = sortperm(abs.(g_dense .- g_lookup), rev = true)[1:min(10, nvar)]
        for i in idx
            println(@sprintf("  coord %4d: g_dense=% .6e  g_lookup=% .6e  diff=% .3e", i, g_dense[i], g_lookup[i], g_dense[i]-g_lookup[i]))
        end
        ncore1 = pcx_dense.aug.ncore - 1
        println("  ncore1 (core lambda count, cols 2:1+ncore1) = ", ncore1)
        println("  CM block cols: ", 2+ncore1, ":", 1+ncore1+ncm_cm)
        println("  level block cols: ", 2+ncore1+ncm_cm, ":", 1+ncore1+ncm_cm+cctx_lookup.L)
    end
    flush(stdout)
end
