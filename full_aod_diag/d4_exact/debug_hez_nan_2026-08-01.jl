const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_threaded.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "zc_restriction_operator.jl", "cm_originzc_target_layout.jl", "cm_originzc_moments.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl", "cm_originzc_production.jl"]
    include(joinpath(D4X, f))
end
using Random, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
nu0vec(K::Int) = [Float64(factorial(k)) for k in 1:K]
Random.seed!(2026)

K_mean, K_pair = 1, 1
aug = build_cm_meanzc_augmented_obj(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, meanzc_basis = :direct)
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
x = vcat(0.0, zeros(n - 1))
_archC_prep_for_hessian!(objA, x)
cf = cctx.core_cf_ref[]
H = objA.H; M = objA.M
ddPsi! = objA.ddPsi!; ddPsi!(objA.arg2, objA.arg0); S = objA.arg2
E = @view H[:, 2:1+NCORE_ext]
Z = @view E[:, ncore+1:NCORE_ext]

wctx = build_winner_pair_ctx(cf)
println("wctx.Ddest=", wctx.Ddest, " wctx.D=", wctx.D, " wctx.ncolI=", wctx.ncolI, " n_restr=", n_restr)
ws_ref = Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing)
ws = ensure_winner_zc_cross_scratch!(ws_ref, wctx.W, n_restr, wctx.Ddest)
winner_pair_cross_hessian_zc_prep!(ws, wctx, S)

# direct brute force TZ[d,x]
nu = cf.SW
function bf_tz(d, x)
    acc = 0.0
    for w in 1:cf.W
        acc += S[w] * nu[w] * cf.wval[w, d] * Z[w, x]
    end
    return acc
end

println("ws.SnuWval size=", size(ws.SnuWval))
TZ_direct = ws.SnuWval' * Z
for d in 1:wctx.Ddest
    for x in 1:min(3, n_restr)
        println("d=", d, " x=", x, "  TZ_gemm(pre-call)=", TZ_direct[d,x], "  bf_tz=", bf_tz(d,x))
    end
end

HEZ_new = zeros(ncore, n_restr)
winner_pair_cross_hessian_zc_block!(HEZ_new, wctx, ws, S, Z, M; use_profiled_correction = true)
println("post-call ws.TZ_buf[1:Ddest,1:3]:")
display(ws.TZ_buf[1:wctx.Ddest, 1:min(3,n_restr)])
println()

nbilateral = wctx.has_cf ? wctx.ncolI - 1 : wctx.ncolI
for j in 1:min(3, nbilateral)
    d = wctx.target_slot[j]
    o_row = div(j - d, wctx.Ddest) + 1
    println("j=", j, " target_slot=", d, " o_row=", o_row, " kappa0=", wctx.kappa0[j], " pi_vec=", wctx.pi_vec[j])
    keep = 0.0
    for w in 1:cf.W
        if cf.winner[w, d] == o_row
            keep += S[w] * nu[w] * cf.wval[w, d] * wctx.kappa0[j] * Z[w, 1]
        end
    end
    Tdx = bf_tz(d, 1)
    expected = (keep - wctx.pi_vec[j] * Tdx) / M
    println("  keep=", keep, " Tdx=", Tdx, " expected=", expected, " got=", HEZ_new[j,1])
end
