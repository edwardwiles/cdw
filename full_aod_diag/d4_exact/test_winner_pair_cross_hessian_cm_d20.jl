# Phase B (final-operator-stack-release, 2026-07-27): real D=20/W=80,000 companion to
# test_winner_pair_cross_hessian_cm_d4.jl -- the winner-bin H_EC cross-Hessian against the dense
# reference, at the production default L=50, both contrasts.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

println("Building real D=20 context (W=80000, delta=1.0)..."); flush(stdout)
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

for contrasts in (:anchored, :orthonormal)
    println("Solving contrasts=$contrasts L=50 ..."); flush(stdout)
    pcx = build_cm_production_context(ctx, CS; L = 50, contrasts = contrasts, use_compressed_core = true)
    cctx = pcx.cctx
    obj = pcx.ctx_cm.obj
    base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
    check("contrasts=$contrasts: inner solve feasible (status=$(base.inner_status))", base.inner_status in (0, -100, -101, -103))

    NCORE = cctx.NCORE; ncm = cctx.ncm; nO = cctx.nO; L_ = cctx.L
    n = NCORE + ncm
    x = vcat(base.ζstar, base.λstar)
    _archC_prep_for_hessian!(obj, x)
    h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    @time hessian_cm_structured!(h, obj, cctx)
    Hd = unpack_packed(h, n)
    H_EC_dense = Hd[1:NCORE, NCORE+1:NCORE+ncm]

    cf = cctx.core_cf_ref[]
    check("contrasts=$contrasts: cf is a real CompressedFactual", cf isa CompressedFactual)
    wctx = build_winner_pair_ctx(cf)
    ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
    ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, L_)
    @time winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)

    Hraw_EC = zeros(NCORE, nO)
    H_EC_new = zeros(NCORE, ncm)
    M = obj.M
    for l in 1:L_
        winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, cctx.origins, cctx.refIndex1, M)
        cols = (l - 1) * nO + 1 : l * nO
        block = cctx.R === nothing ? Hraw_EC : Hraw_EC * cctx.R
        H_EC_new[:, cols] .= block
    end

    maxdiff = maximum(abs.(H_EC_dense .- H_EC_new))
    relscale = max(1.0, maximum(abs.(H_EC_dense)))
    ok = maxdiff < 1e-6 * relscale
    check("contrasts=$contrasts: winner-bin H_EC matches dense at real D=20/L=50 (max|Δ|=$(maxdiff))", ok)
    @printf("  contrasts=%s D20/L=50: max|ΔH_EC|=%.3e  scale=%.3e\n", contrasts, maxdiff, relscale)
    flush(stdout)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
