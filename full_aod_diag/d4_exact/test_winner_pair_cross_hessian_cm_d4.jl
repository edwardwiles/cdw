# Phase B (final-operator-stack-release, 2026-07-27): D=4 correctness gate for
# winner_pair_cross_hessian_cm_block! (the new winner-bin H_EC), against the pre-existing dense
# reference H_EC (hessian_cm_structured!'s own CScum-based fill), at several points, both
# contrast conventions.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

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

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)

for contrasts in (:anchored, :orthonormal), L in (10, 20)
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true)
    cctx = pcx.cctx
    obj = pcx.ctx_cm.obj
    base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
    check("contrasts=$contrasts L=$L: inner solve feasible", base.inner_status in (0, -100, -101, -103))

    for (label, x) in (("calib", vcat(base.ζstar, base.λstar)),
                        ("perturbed", vcat(base.ζstar, base.λstar) .+ vcat(0.01, 0.02 .* randn(length(base.λstar)))))
        NCORE = cctx.NCORE; ncm = cctx.ncm; nO = cctx.nO; L_ = cctx.L
        n = NCORE + ncm
        _archC_prep_for_hessian!(obj, x)
        h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        hessian_cm_structured!(h, obj, cctx)
        Hd = unpack_packed(h, n)
        H_EC_dense = Hd[1:NCORE, NCORE+1:NCORE+ncm]

        # Same obj.arg0 state (set by _archC_prep_for_hessian! above, untouched by hessian_cm_structured!'s
        # own ddPsi! recompute since arg0 itself is never mutated) -- build the winner-bin version now.
        cf = cctx.core_cf_ref[]
        check("contrasts=$contrasts L=$L $label: cf is a real CompressedFactual", cf isa CompressedFactual)
        wctx = build_winner_pair_ctx(cf)
        ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
        ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, L_)
        winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)

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
        ok = maxdiff < 1e-8 * relscale
        check("contrasts=$contrasts L=$L $label: winner-bin H_EC matches dense (max|Δ|=$(maxdiff))", ok)
        @printf("  contrasts=%s L=%d %s: max|ΔH_EC|=%.3e  scale=%.3e\n", contrasts, L, label, maxdiff, relscale)
    end
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
