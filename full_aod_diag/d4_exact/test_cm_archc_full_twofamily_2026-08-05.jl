# 2026-08-05 truncated-power task, Architecture C extension: full-Hessian correctness gate for
# the two-family (eq.35+eq.36) spec, modeled directly on c13_validate_hessian_archs.jl's own
# "Architecture C vs Architecture A at a fixed point, no KNITRO solve" methodology (already-
# validated pattern, not reinvented). Covers H_EE+H_EC+H_CC together (the FULL assembled packed
# Hessian), at several random dual points, for BOTH the dense (CS_/CScum2) H_EC path AND the
# winner-bin (Pow-extended) H_EC path -- the genuine no-dense-H production path.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "oracle_fast.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end

function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(x_free_calib, ctx.m)
L = 4

println("="^100)
println("Section 1: dense-H path (CS_/CScum2) -- Architecture C vs Architecture A, two-family, D4")
println("="^100)
let
    aug = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true)
    @assert aug.n_families == 2
    objA = aug.obj_cm
    n = objA.outer_constr_index
    K = zeros(size(ctx.U, 1))
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0

    cctx = build_cm_bin_ctx(ctx, aug)
    @assert cctx.n_families == 2
    @assert cctx.Pow !== nothing

    Random.seed!(2026)
    xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
    for (pi_, x) in enumerate(xs)
        hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        objA(x, h = hA)
        HA = unpack_packed(hA, n)

        hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        _archC_prep_for_hessian!(objA, x)
        hessian_cm_structured!(hC, objA, cctx)
        HC = unpack_packed(hC, n)
        errC = maximum(abs.(HC .- HA))
        check("dense-H path, point $pi_: full packed Hessian matches Architecture A (max|Δ|=$errC)", errC < 1e-8)
    end
end

println("="^100)
println("Section 2: winner-bin path (Pow-extended) -- Architecture C vs Architecture A, two-family, D4")
println("(uses build_cm_augmented_obj_archB with use_compressed_core=true to get a real core_cf_ref,")
println(" the genuine no-dense-H production path for H_EE/H_EC)")
println("="^100)
let
    # Reference (Architecture A / dense H, plain build_cm_augmented_obj -- unchanged path)
    augA = build_cm_augmented_obj(ctx, CS; L = L, include_truncated_moment = true)
    objA = augA.obj_cm
    n = objA.outer_constr_index
    K = zeros(size(ctx.U, 1))
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_full, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0

    # Winner-bin-eligible context: archB's own bin-index-driven CDF-only moment reconstruction
    # (fill_cm_columns_from_bins!) is single-family only (documented, unrelated to this gate --
    # Architecture B was never extended and isn't used here); what THIS test actually needs from
    # archB is just a populated `core_cf_ref` (a real CompressedFactual for θ_full) so
    # `_cm_cross_hessian_wants_winner_bin` can go true. Build cctx directly against augA (the
    # two-family aug) but manually populate a CompressedFactual into a fresh core_cf_ref, mirroring
    # what wrap_moments_with_cm_archB's moments! closure would publish on a real inner-solve call.
    cctx = build_cm_bin_ctx(ctx, augA)
    @assert cctx.n_families == 2
    cf = cf_build(θ_full, ctx; check_ties = false)
    cctx.core_cf_ref[] = cf

    Random.seed!(2027)
    xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
    for (pi_, x) in enumerate(xs)
        hA = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        objA(x, h = hA)
        HA = unpack_packed(hA, n)

        hC = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        _archC_prep_for_hessian!(objA, x)
        hessian_cm_structured!(hC, objA, cctx)
        HC = unpack_packed(hC, n)
        errC = maximum(abs.(HC .- HA))
        used_winner_bin = _cm_cross_hessian_wants_winner_bin(cctx, cctx.core_cf_ref[])
        check("winner-bin path, point $pi_: used_winner_bin=$used_winner_bin, full packed Hessian matches Architecture A (max|Δ|=$errC)", errC < 1e-8 && used_winner_bin)
    end
end

println("="^80)
if isempty(FAILURES)
    println("ALL FULL-HESSIAN TWO-FAMILY GATES PASS")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
