# Winner-aware H_ER phase (2026-07-27), Sections 4/5: D=4 standalone-primitive gate for
# `winner_pair_cross_hessian_zc_block!` (the new winner-bin H_EZ, shared by CM+ZC and origin-ZC),
# against a fully independent dense `(1/M)*E'*diag(S)*Z` reference built the "obvious slow way"
# (mirrors test_winner_pair_cross_hessian_cm_d4.jl's own methodology for the CM-grid primitive).
#
# Two families exercised, each at K_mean=1,K_pair=1 (regression-sized) AND K_mean=2,K_pair=2 (a
# larger config, matching test_cm_meanzc_d4_gates.jl's own CONFIGS precedent):
#   - CM+ZC: mean/pair columns folded into the widened E block (cm_meanzc_production.jl).
#   - origin-ZC: mean/pair columns are a separate H_ER block, no CM-grid at all (cm_originzc_production.jl).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl", "cm_originzc_production.jl"]
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
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

const L = 10

println("="^100)
println("Standalone primitive gate: winner_pair_cross_hessian_zc_block! (CM+ZC family)")
println("="^100)
for (K_mean, K_pair, label) in [(1, 1, "K1"), (2, 2, "K2")]
    aug = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, meanzc_basis = :direct)
    objA = aug.obj_cm
    n = objA.outer_constr_index
    θ_ext_calib = vcat(θ_full_calib, nu0vec(K_mean))
    K = zeros(size(ctx.U, 1))
    objA.moments!(K, CS.select_G_from_H(objA, objA.H), θ_ext_calib, objA.U, objA)
    objA.H[:, 1] .= K
    objA.H[:, 2] .= 1.0

    cctx = build_cm_meanzc_bin_ctx(ctx, aug)
    ncore = cctx.ncore_core; NCORE_ext = cctx.NCORE
    n_restr = NCORE_ext - ncore

    Random.seed!(9100 + K_mean)
    xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
    maxerr = 0.0
    for x in xs
        _archC_prep_for_hessian!(objA, x)
        cf = cctx.core_cf_ref[]
        check("$label calib/pt: cf is a real CompressedFactual", cf isa CompressedFactual)
        cf isa CompressedFactual || continue

        # dense reference: E = H[:,2:1+NCORE_ext], H_EM_dense = (1/M)*E[:,1:ncore]'*diag(S)*E[:,ncore+1:NCORE_ext]
        H = objA.H; M = objA.M
        ddPsi! = objA.ddPsi!; ddPsi!(objA.arg2, objA.arg0); S = objA.arg2
        E = @view H[:, 2:1+NCORE_ext]
        EC = @view E[:, 1:ncore]
        Z = @view E[:, ncore+1:NCORE_ext]
        H_EM_dense = ((EC .* S)' * Z) ./ M

        wctx = build_winner_pair_ctx(cf)
        check("$label: wctx.ncolI+1 == ncore", wctx.ncolI + 1 == ncore)
        ws_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
        ws = ensure_winner_zc_cross_scratch!(ws_ref, wctx.W, n_restr)
        winner_pair_cross_hessian_zc_prep!(ws, wctx, S)
        H_EM_new = zeros(ncore, n_restr)
        winner_pair_cross_hessian_zc_block!(H_EM_new, wctx, ws, S, Z, M)

        err = maximum(abs.(H_EM_dense .- H_EM_new))
        maxerr = max(maxerr, err)
        @printf("  %-4s x-scale=%.3f  n_restr=%d  max|Δ|=%.3e\n", label, maximum(abs.(x)), n_restr, err)
    end
    check("$label: CM+ZC standalone primitive matches dense reference (max|Δ| over points = $(maxerr))", maxerr < 1e-8)
end
println()

println("="^100)
println("Standalone primitive gate: winner_pair_cross_hessian_zc_block! (origin-ZC family)")
println("="^100)
for (K_mean, K_pair, label) in [(1, 1, "K1"), (2, 2, "K2")]
    layout = SharedByPowerLayout(K_mean, K_pair)
    aug = build_originzc_augmented_obj(ctx, CS, layout)
    obj = aug.obj_cm
    NCORE = aug.ncore_econ
    n_restr = aug.n_mean + aug.n_pair
    n = NCORE + n_restr
    θ_ext_calib = vcat(θ_full_calib, nu0vec(K_mean))
    K = zeros(size(ctx.U, 1))
    obj.moments!(K, CS.select_G_from_H(obj, obj.H), θ_ext_calib, obj.U, obj)
    obj.H[:, 1] .= K
    obj.H[:, 2] .= 1.0

    Random.seed!(9200 + K_mean)
    xs = [vcat(0.0, zeros(n - 1)), 0.01 .* randn(n), 0.05 .* randn(n)]
    maxerr = 0.0
    for x in xs
        _archC_prep_for_hessian!(obj, x)
        cf = aug.core_cf_ref[]
        check("$label calib/pt: cf is a real CompressedFactual", cf isa CompressedFactual)
        cf isa CompressedFactual || continue

        H = obj.H; M = obj.M
        ddPsi! = obj.ddPsi!; ddPsi!(obj.arg2, obj.arg0); S = obj.arg2
        EC = @view H[:, 2:1+NCORE]
        Z = @view H[:, 2+NCORE:1+n]
        H_ER_dense = ((EC .* S)' * Z) ./ M

        wctx = build_winner_pair_ctx(cf)
        check("$label: wctx.ncolI+1 == NCORE", wctx.ncolI + 1 == NCORE)
        ws_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
        ws = ensure_winner_zc_cross_scratch!(ws_ref, wctx.W, n_restr)
        winner_pair_cross_hessian_zc_prep!(ws, wctx, S)
        H_ER_new = zeros(NCORE, n_restr)
        winner_pair_cross_hessian_zc_block!(H_ER_new, wctx, ws, S, Z, M)

        err = maximum(abs.(H_ER_dense .- H_ER_new))
        maxerr = max(maxerr, err)
        @printf("  %-4s x-scale=%.3f  n_restr=%d  max|Δ|=%.3e\n", label, maximum(abs.(x)), n_restr, err)
    end
    check("$label: origin-ZC standalone primitive matches dense reference (max|Δ| over points = $(maxerr))", maxerr < 1e-8)
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
