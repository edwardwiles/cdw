# ================================================================================================
# Continuation 12b, Part 1: conditioning-preserving-transformation comparison.
#
# Compares 4 common-marginals moment bases (anchored CDF, orthonormal/Helmert-contrast CDF,
# interval, standardized interval) at D=4, L in {10,20,50}, at 2 fixed points (calibration,
# perturbed-feasible), across all 4 choices of reference country. For each of the
# 4 x 3 x 2 x 4 = 96 combinations, records: nStatus/ok, numerical rank of the full augmented
# moment matrix G (via svdvals), cond() of the exact dense inner-dual Hessian at the solved
# point (cc_algo/PsiObjectiveBundle.jl::hessian!), inner KNITRO iteration count (diffed from
# CS.INNER_ITERS_TOTAL), max|lambda| restricted to the CM block, and wall time.
#
# See docs/fullA_cm_conditioning_and_adaptive_grid_report.md for the write-up.
# ================================================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "c12b_interval_common_marginals_moments.jl"))
using LinearAlgebra: cond, svdvals
using Printf, Random, DelimitedFiles

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_calib = ctx.θ0_up[ctx.free_idx]

Random.seed!(4243)   # match c12_d4_fixed_param_battery.jl's convention exactly
x_perturbed = copy(x_calib)
x_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_perturbed) - 1))

points = [("calib", x_calib), ("perturbed", x_perturbed)]

bases = [
    ("anchored",      (U, ref, L) -> precalc_common_marginals_cdf(U, ref, L; contrasts = :anchored)),
    ("orthonormal",   (U, ref, L) -> precalc_common_marginals_cdf(U, ref, L; contrasts = :orthonormal)),
    ("interval",      (U, ref, L) -> precalc_common_marginals_interval(U, ref, L; standardize = false, contrasts = :anchored)),
    ("std_interval",  (U, ref, L) -> precalc_common_marginals_interval(U, ref, L; standardize = true,  contrasts = :anchored)),
]

Ls = (10, 20, 50)
refs = 1:D

function hessian_cond(obj_cm)
    n = obj_cm.outer_constr_index
    hbuf = zeros(div(n * (n + 1), 2))
    obj_cm(obj_cm.x, h = hbuf)
    Hfull = Matrix{Float64}(undef, n, n)
    k = 1
    for i in 1:n, j in i:n
        Hfull[i, j] = hbuf[k]; Hfull[j, i] = hbuf[k]
        k += 1
    end
    return cond(Hfull)
end

rows = Vector{Any}()
header = ["basis","L","refIndex","point","ok","nStatus","d_total","rank_G","rank_deficiency",
          "cond_hessian","inner_iters","max_abs_lambda_cm","max_abs_lambda_core","Delta_dual","wall_s"]
push!(rows, header)

t_script0 = time()
ncombo = length(bases) * length(Ls) * length(refs)
combo_i = Ref(0)
for (bname, bfun) in bases
    for L in Ls
        for ref in refs
            combo_i[] += 1
            CM, z, origins = bfun(ctx.U, ref, L)
            obj_cm = build_cm_augmented_obj_from_CM(ctx, CS, CM)
            ncm = size(CM, 2)
            ncore = ctx.obj.d
            @printf("[%3d/%3d] basis=%-12s L=%-3d ref=%d  (ncm=%d, d=%d)\n", combo_i[], ncombo, bname, L, ref, ncm, obj_cm.d)
            flush(stdout)
            for (pname, xf) in points
                iters0 = CS.INNER_ITERS_TOTAL[]
                t0 = time()
                r = evaluate_fullA(xf, merge(ctx, (obj = obj_cm,)); use_cache = false, warm = false)
                wall = time() - t0
                iters = CS.INNER_ITERS_TOTAL[] - iters0
                ok = r.inner_status in (0, -100, -101, -103)
                if !ok
                    push!(rows, [bname, L, ref, pname, ok, r.inner_status, obj_cm.d, missing, missing,
                                  missing, iters, missing, missing, missing, wall])
                    @printf("    %-10s FAILED nStatus=%d  t=%.2fs\n", pname, r.inner_status, wall)
                    flush(stdout)
                    continue
                end
                W = size(ctx.U, 1)
                G = zeros(W, obj_cm.d)
                obj_cm.moments!(zeros(W), G, r.θ_full, ctx.U, obj_cm)
                sv = svdvals(G)
                tol = maximum(size(G)) * eps(maximum(sv))
                rnk = count(>(tol), sv)
                cnd = hessian_cond(obj_cm)
                lam_cm = r.lambda[ncore:(ncore - 1 + ncm)]
                lam_core = r.lambda[1:(ncore - 1)]
                push!(rows, [bname, L, ref, pname, ok, r.inner_status, obj_cm.d, rnk, obj_cm.d - rnk,
                              cnd, iters, maximum(abs.(lam_cm)), maximum(abs.(lam_core)), r.Delta_dual, wall])
                @printf("    %-10s ok  rank=%d/%d  cond=%.3e  iters=%d  max|lam_cm|=%.3e  Delta=%.6f  t=%.2fs\n",
                        pname, rnk, obj_cm.d, cnd, iters, maximum(abs.(lam_cm)), r.Delta_dual, wall)
                flush(stdout)
            end
        end
    end
end
@printf("\nTotal wall time: %.1f s\n", time() - t_script0)

outpath = joinpath(@__DIR__, "c12b_conditioning_battery_results.csv")
open(outpath, "w") do io
    for row in rows
        println(io, join(row, ","))
    end
end
println("Wrote results to $outpath")
