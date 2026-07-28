# ================================================================================================
# True no-H operator bundle task, Part A.3/A.7 proof block: structural + empirical allocation proof
# for flexible-CM's OperatorPsiBundle. Writes OPERATOR_BUNDLE_FIELD_AND_ALLOCATION_PROOF_2026-07-28.json
# (flexible-CM section; other 4 families are NOT covered by this run -- see master report for scope).
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl"]
    include(joinpath(_D4E, f))
end
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]

pcx_o = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true,
                                     threaded_bins = false, moment_representation = :operator)
pcx_o.cctx.core_hessian_backend = :exact_winner_pair_parallel
pcx_o.cctx.cm_cross_hessian_backend = :winner_bin
obj_o = pcx_o.ctx_cm.obj

fn = fieldnames(typeof(obj_o))
has_H = :H in fn
has_H_copy = Symbol("H_copy") in fn
has_moments = Symbol("moments!") in fn
has_K = :K in fn
has_ones = :ones in fn

# Warm up (JIT) with one solve, then measure a SECOND solve's allocation to avoid counting
# one-time compilation allocation.
base1 = archC_base_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx)
alloc_bytes = @allocated archC_base_state(x_free_calib, pcx_o.ctx_cm, pcx_o.cctx)
W = size(ctx.U, 1)
d_econ = pcx_o.cctx.NCORE   # economic column count (winner-form core), what a dense G column-block would need
h_sized_bytes_if_dense = W * d_econ * 8   # a single W x NCORE dense Float64 block, for scale comparison

println("bundle_type = ", nameof(typeof(obj_o)))
println("has_H_field = ", has_H)
println("has_H_copy_field = ", has_H_copy)
println("has_moments_field = ", has_moments)
println("has_K_field = ", has_K)
println("has_ones_field = ", has_ones)
println("inner_status_calib = ", base1.inner_status)
println("per_inner_solve_alloc_bytes = ", alloc_bytes)
println("W_x_NCORE_dense_block_bytes_for_scale = ", h_sized_bytes_if_dense)
println("DONE")

open(joinpath(_D4E, "..", "..", "OPERATOR_BUNDLE_FIELD_AND_ALLOCATION_PROOF_2026-07-28.json"), "w") do io
    println(io, "{")
    println(io, "  \"flexible_cm\": {")
    println(io, "    \"bundle_type\": \"", nameof(typeof(obj_o)), "\",")
    println(io, "    \"has_H_field\": ", has_H, ",")
    println(io, "    \"has_H_copy_field\": ", has_H_copy, ",")
    println(io, "    \"has_moments_field\": ", has_moments, ",")
    println(io, "    \"has_K_field\": ", has_K, ",")
    println(io, "    \"has_ones_field\": ", has_ones, ",")
    println(io, "    \"fieldnames\": \"", fn, "\",")
    println(io, "    \"inner_status_calib\": ", base1.inner_status, ",")
    println(io, "    \"per_inner_solve_alloc_bytes_warm\": ", alloc_bytes, ",")
    println(io, "    \"reference_W_x_NCORE_dense_block_bytes\": ", h_sized_bytes_if_dense, ",")
    println(io, "    \"note\": \"other 4 families not covered by this run -- see master report scope\"")
    println(io, "  },")
    println(io, "  \"other_families_not_yet_wired\": [\"unrestricted\", \"common_frechet\", \"cm_plus_zc\", \"zc_only\"]")
    println(io, "}")
end
