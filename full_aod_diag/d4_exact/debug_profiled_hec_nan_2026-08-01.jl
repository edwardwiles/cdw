const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl"]
    include(joinpath(D4X, f))
end
using Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)

pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, use_compressed_core = true,
                                   moment_representation = :dense_reference)
cctx = pcx.cctx
obj = pcx.ctx_cm.obj
base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
x = vcat(base.ζstar, base.λstar)
_archC_prep_for_hessian!(obj, x)
cf = cctx.core_cf_ref[]

println("cf.W=", cf.W, " cf.D=", cf.D, " cf.D_dest=", cf.D_dest, " cf.oci=", cf.oci, " cf.cf_col=", cf.cf_col)
println("any NaN in cf.wval? ", any(isnan, cf.wval))
println("any Inf in cf.wval? ", any(isinf, cf.wval))
println("any NaN in cf.SW? ", any(isnan, cf.SW))
println("extrema cf.wval: ", extrema(cf.wval))

wctx = build_winner_pair_ctx(cf)
println("any NaN in wctx.wval? ", any(isnan, wctx.wval))
println("any NaN in wctx.pi_vec? ", any(isnan, wctx.pi_vec))
println("any NaN in wctx.kappa0? ", any(isnan, wctx.kappa0))
println("wctx.target_slot extrema: ", extrema(wctx.target_slot))
println("wctx.ncolI=", wctx.ncolI, " has_cf=", wctx.has_cf, " cf.cf_col=", cf.cf_col)

ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, cctx.L, wctx.Ddest)
winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)
println("any NaN in ws.MTab? ", any(isnan, ws.MTab))
println("any NaN in ws.MCScum? ", any(isnan, ws.MCScum))
println("obj.M = ", obj.M)
S = obj.arg2
println("any NaN in obj.arg2 (S)? ", any(isnan, S))

M = obj.M
Hraw_EC_new = zeros(cctx.NCORE, cctx.nO)
winner_pair_cross_hessian_cm_block!(Hraw_EC_new, wctx, ws, 1, cctx.origins, cctx.refIndex1, M; use_profiled_correction = true)
println("any NaN in Hraw_EC_new (l=1)? ", any(isnan, Hraw_EC_new))
if any(isnan, Hraw_EC_new)
    idx = findfirst(isnan, Hraw_EC_new)
    println("first NaN at ", idx, " -> j=", idx[1]-1, " oi=", idx[2])
    j = idx[1] - 1
    if j >= 1
        println("wctx.target_slot[j]=", wctx.target_slot[j], " wctx.pi_vec[j]=", wctx.pi_vec[j], " wctx.kappa0[j]=", wctx.kappa0[j])
    end
end
