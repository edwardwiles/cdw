# Minimal pre-flight timing probe (task brief Section 8: "if one inner solve takes many minutes
# ... do not launch an outer trial. Stop after producing a bottleneck analysis"). ONE cold CM-only
# solve, then ONE cold CM+mean solve, at L=10, D=20 real data, W=80000, calibration point.
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
include(joinpath(@__DIR__, "mean_zero_cov_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production_bundle.jl"))
using Printf, Dates

println("Timing probe start: ", now(), "  nthreads=", Threads.nthreads()); flush(stdout)
t0 = time()
ctx = d20_real_setup(W = 80000, δ = 1.0)
@printf "d20_real_setup wall = %.1fs\n" (time() - t0); flush(stdout)

x_free_calib = ctx.θ0_up[ctx.free_idx]
L = 10

t0 = time()
aug_cm = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
ctx_cm_plain = merge(ctx, (obj = aug_cm.obj_cm,))
@printf "build_cm_augmented_obj (L=%d) wall = %.1fs\n" L (time() - t0); flush(stdout)

t0 = time()
r_cm = evaluate_fullA(x_free_calib, ctx_cm_plain; use_cache = false, warm = false)
@printf "CM-only cold inner solve wall = %.1fs  inner_status=%d  Delta_dual=%.6f\n" (time() - t0) r_cm.inner_status r_cm.Delta_dual; flush(stdout)

nu_ref = Ref(1.0)
t0 = time()
pcx = build_cm_meanzc_production_context(ctx, CS; L = L, cm_extension = :cm_plus_mean, nu_ref = nu_ref)
@printf "build_cm_meanzc_production_context (mean-only, L=%d) wall = %.1fs\n" L (time() - t0); flush(stdout)

t0 = time()
_, base, verify = cm_meanzc_production_value_verified(x_free_calib, pcx)
@printf "CM+mean cold inner solve wall = %.1fs  inner_status=%d  Delta_dual=%.6f\n" (time() - t0) base.inner_status verify.Delta_dual; flush(stdout)

println("Timing probe done: ", now())
