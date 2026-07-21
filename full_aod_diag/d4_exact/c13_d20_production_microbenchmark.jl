# Continuation 13, Section 8: D20 production microbenchmark of the FULLY-INTEGRATED bundle
# (Architecture B moments + Architecture C Hessian + CM-aware Lfix gradient + nested grids),
# real data, W=80000. This goes beyond c15_d20_cm_smoke_test.jl (Continuation 12, fixed-point
# inner solve only, dense-vs-ArchC comparison) by also timing ONE full CM-aware outer-gradient
# call end to end -- the piece that did not exist until this continuation.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
using Printf, LinearAlgebra, Random, Statistics

W = 80000
DELTA = 1.0
println(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
@printf ">>> ctx built in %.1fs. D=%d W=%d\n" (time()-t0) D W

x_free_calib = ctx.θ0_up[ctx.free_idx]
w_from_xfree(xf) = vcat(xf[1], pivot_reduce(log.(reshape(xf[2:end], D, D)), pe))
w_calib = w_from_xfree(x_free_calib)

Random.seed!(3113)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.02 .* randn(length(x_free_perturbed) - 1))
w_perturbed = w_from_xfree(x_free_perturbed)

snaps = nested_grid_sequence([10, 20, 50])

println()
println("="^110)
println("PER-GRID MICROBENCHMARK (calibration point)")
println("="^110)
for L in (10, 20, 50)
    println("-- L=$L --")
    t_setup = @elapsed pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
    ncm = pcx.aug.ncm; ncore = pcx.aug.ncore; dtotal = pcx.ctx_cm.obj.d
    @printf "  cutpoints=%d  ncm=%d  ncore=%d  d_total=%d  outer_constr_index=%d  setup=%.2fs\n" length(snaps[L]) ncm ncore dtotal pcx.ctx_cm.obj.outer_constr_index t_setup

    t_cold = @elapsed (K1, base1) = cm_production_value(x_free_calib, pcx)
    t_warm = @elapsed (K2, base2) = cm_production_value(x_free_calib, pcx)
    @printf "  cold inner solve=%.3fs (nStatus=%d)  warm(re-solve, no actual warm-start wiring here)=%.3fs\n" t_cold base1.inner_status t_warm

    t_grad = @elapsed (gfull, meta) = cm_production_gradient(x_free_calib, pcx, ctx, pe; base = base1, threaded = true, h_mode = :adaptive)
    @printf "  ONE full CM-aware Lfix gradient call: %.3fs  ||g||=%.4e  gamma_component=%.4e\n" t_grad norm(gfull) gfull[1]

    # dense-reference agreement (Architecture A + generic dense obj, no ArchB/ArchC) at this same point
    aug_ref = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored, probs = snaps[L])
    ctx_ref = merge(ctx, (obj = aug_ref.obj_cm,))
    t_dense = @elapsed r_ref = evaluate_fullA(x_free_calib, ctx_ref; use_cache = false, warm = false)
    @printf "  dense reference: %.2fs  Delta_dual=%.8f | production: Delta_dual=%.8f | diff=%.2e | speedup=%.2fx\n" t_dense (-r_ref.zeta) (-base1.ζstar) abs((-r_ref.zeta)-(-base1.ζstar)) (t_dense/t_cold)
    GC.gc()
end

println()
println("="^110)
println("PERTURBED-POINT SPOT CHECK (L=50 only, confirms the bundle isn't calibration-only)")
println("="^110)
pcx50 = build_cm_production_context(ctx, CS; L = 50, contrasts = :anchored, probs = snaps[50])
t_p = @elapsed (Kp, basep) = cm_production_value(x_free_perturbed, pcx50)
t_gp = @elapsed (gp_full, _) = cm_production_gradient(x_free_perturbed, pcx50, ctx, pe; base = basep, threaded = true)
@printf "  perturbed: inner solve=%.3fs (nStatus=%d, Delta=%.6f)  gradient=%.3fs  ||g||=%.4e\n" t_p basep.inner_status (-basep.ζstar) t_gp norm(gp_full)

println()
println("="^110)
println("MEMORY: current process RSS")
println("="^110)
rss_kb = try
    parse(Int, split(read(`ps -o rss= -p $(getpid())`, String))[1])
catch
    -1
end
@printf "  RSS = %.2f GB\n" (rss_kb / 1024^2)
println("DONE")
