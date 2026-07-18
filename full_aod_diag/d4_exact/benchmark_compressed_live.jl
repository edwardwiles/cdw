# ============================================================================
# Continuation 8, workstream 2: END-TO-END speedup measurement for the LIVE
# wiring (compressed_live.jl), complementing benchmark_compressed.jl's
# isolated-component numbers (which benchmark the standalone bundle, not the
# actual KNITRO inner solve). Measures:
#   1. Full evaluate_fullA_fast wall time, dense vs compressed, warm and cold,
#      at the upper incumbent and calibration point (D=4, W=8000).
#   2. Per-callback-phase breakdown via the existing @prof instrumentation
#      (inner_dual_fg_callback[_compressed], inner_dual_hessian_callback[_compressed],
#      inner_moment_build[_compressed]) -- isolates WHERE the time goes.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using Printf, Statistics

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

w_lfixcomposite_sr1 = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
xf_upper = x_free_from_w(w_lfixcomposite_sr1)
xf_calib = ctx.θ0_up[ctx.free_idx]

med(f, N) = (GC.gc(); [f() for _ in 1:2]; median([(GC.gc(false); @elapsed f()) for _ in 1:N]))

println("="^90)
println("END-TO-END: evaluate_fullA_fast wall time, dense vs compressed (median of N reps)")
println("="^90)

function bench_e2e(label, xf; warm, N = 15)
    obj = ctx.obj
    # warm both paths identically first so the "warm=true" comparison starts from the SAME cached obj.x
    if warm
        evaluate_fullA_fast(xf, ctx; warm = true, moment_representation = :dense)
    end
    t_dense = med(() -> begin
        !warm && (obj.x .= NaN)
        evaluate_fullA_fast(xf, ctx; warm = warm, moment_representation = :dense)
    end, N)
    if warm
        evaluate_fullA_fast(xf, ctx; warm = true, moment_representation = :compressed)
    end
    t_compr = med(() -> begin
        !warm && (obj.x .= NaN)
        evaluate_fullA_fast(xf, ctx; warm = warm, moment_representation = :compressed)
    end, N)
    ratio = t_dense / t_compr
    @printf("%-28s warm=%-5s  dense=%.4f ms  compressed=%.4f ms  speedup=%.2fx\n",
            label, warm, 1000t_dense, 1000t_compr, ratio)
    return (label = label, warm = warm, t_dense = t_dense, t_compr = t_compr, ratio = ratio)
end

rows = NamedTuple[]
push!(rows, bench_e2e("upper_incumbent", xf_upper; warm = true))
push!(rows, bench_e2e("upper_incumbent", xf_upper; warm = false))
push!(rows, bench_e2e("calibration", xf_calib; warm = true))
push!(rows, bench_e2e("calibration", xf_calib; warm = false))

println("\n" * "="^90)
println("PER-PHASE PROFILE BREAKDOWN (median ms, n calls) -- dense vs compressed labeled phases")
println("="^90)
prof_reset!()
obj = ctx.obj
obj.x .= NaN
for _ in 1:10
    evaluate_fullA_fast(xf_upper, ctx; warm = false, moment_representation = :dense)
end
obj.x .= NaN
for _ in 1:10
    evaluate_fullA_fast(xf_upper, ctx; warm = false, moment_representation = :compressed)
end
for row in prof_summary()
    @printf("  %-38s n=%-4d median=%.4f ms  mean=%.4f ms  total=%.4f ms\n",
            row.label, row.n, 1000row.median_s, 1000row.mean_s, 1000row.n*row.mean_s)
end

println("\n" * "="^90)
println("ISOLATED: per-FG-call cost, dense BLAS.gemv vs compressed_cc_value_grad (matched # calls)")
println("="^90)
θ_full = CS.reconstruct_full(xf_upper, ctx.m)
K = zeros(size(ctx.U,1)); G = zeros(size(ctx.U,1), ctx.obj.d)
ctx.obj.moments!(K, G, θ_full, ctx.U, ctx.obj)
cf = build_compressed_factual(θ_full, ctx; check_ties = false)
ncol = cf.oci - 1
ζ = randn(); λ = randn(ncol)
using LinearAlgebra: BLAS
arg0 = zeros(size(ctx.U,1)); arg1 = similar(arg0)
# FAIR comparison: dense_fg must do the SAME work as the real FG callback (obj(x,evalResult.objGrad),
# which computes objective AND gradient together since length(g)>0) -- forward gemv + Psi! (objective)
# + dPsi! + transpose gemv (gradient), NOT just the objective. An earlier version of this benchmark
# only measured dense's objective-only cost against compressed's objective+gradient cost, an unfair
# comparison caught by cross-checking against the (fair, both-compute-objective+gradient) live
# @prof "inner_dual_fg_callback"[_compressed] numbers below, which disagreed with this isolated number
# until fixed -- see docs/compressed_live_integration_report.md.
gvec_dense = zeros(1 + ncol)
dense_fg() = begin
    BLAS.gemv!('N', 1.0, @view(ctx.obj.H[:, 2:1+ctx.obj.outer_constr_index]), -vcat(ζ,λ), 0.0, arg0)
    ctx.obj.Psi!(arg1, arg0)
    ctx.obj.dPsi!(arg1, arg0)
    gvec_dense[1] = 1.0 - sum(arg1) / length(arg0)
    BLAS.gemv!('T', -1/length(arg0), @view(ctx.obj.H[:, 3:1+ctx.obj.outer_constr_index]), arg1, 0.0, @view(gvec_dense[2:end]))
end
compr_fg() = compressed_cc_value_grad(ζ, λ, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
ctx.obj.H[:, 3:2+ncol] .= G[:, 1:ncol]; ctx.obj.H[:, 2] .= 1.0
dense_fg(); compr_fg()
t_dense_fg = med(dense_fg, 200)
t_compr_fg = med(compr_fg, 200)
@printf("single FG-equivalent call: dense=%.5f ms  compressed=%.5f ms  ratio=%.2fx\n",
        1000t_dense_fg, 1000t_compr_fg, t_dense_fg/t_compr_fg)

println("\nDone.")
