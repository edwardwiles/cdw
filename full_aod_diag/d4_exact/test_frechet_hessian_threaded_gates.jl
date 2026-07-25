# Correctness gate: threaded/syrk CDF-only fixed-Fréchet structured Hessian (cm_frechet_hessian_threaded.jl)
# vs the serial reference (cm_frechet_hessian.jl::hessian_cm_frechet_structured!).
# Task: FIXED_FRECHET_INNER_SOLVER production-feasibility 2026-07-24 (CDF-only addendum), gate for
# main-brief §11 "exact HVP/threaded agreement" applied to the threaded Hessian instead.
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian_threaded.jl"))
using Printf, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

println("Threads.nthreads() = ", Threads.nthreads())

println("="^100); println("D=4 SETUP"); println("="^100)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
const L = 8
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)

cfg_frec = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
targets = build_frechet_reference_targets(ctx, cfg_frec; L = L)

aug_cdf = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = :orthonormal)
n_cdf = aug_cdf.ncore + aug_cdf.ncm
nh_cdf = div(n_cdf*(n_cdf+1), 2)
println("ncore=$(aug_cdf.ncore) ncm=$(aug_cdf.ncm) n=$(n_cdf)")

K_cdf, x_cdf, ns_cdf, _, _ = inner_loop_internal_archgeneric(aug_cdf.obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
check("D=4 CDF-only dense solve feasible", ns_cdf in (0,-100,-101,-103))

fctx_cdf = build_cm_frechet_bin_ctx(ctx, aug_cdf)
tls_cdf = build_thread_local_scratch(fctx_cdf.cctx)

function eval_h(hess_fn!, args...)
    aug_cdf.obj_cm.moments!(@view(aug_cdf.obj_cm.H[:,1]), CS.select_G_from_H(aug_cdf.obj_cm, aug_cdf.obj_cm.H), θ_full0, aug_cdf.obj_cm.U, aug_cdf.obj_cm)
    aug_cdf.obj_cm.H[:,2] .= 1.0
    _archC_prep_for_hessian!(aug_cdf.obj_cm, x_cdf)
    h = zeros(nh_cdf)
    hess_fn!(h, aug_cdf.obj_cm, args...)
    return h
end

h_serial_ref = eval_h(hessian_cm_frechet_structured!, fctx_cdf)
h_v2_serial  = eval_h((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=false, use_syrk=false), fctx_cdf)
h_v2_syrk    = eval_h((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=false, use_syrk=true), fctx_cdf)
h_v2_thread  = eval_h((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=true, tls=tls_cdf, use_syrk=false), fctx_cdf)
h_v2_full    = eval_h((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=true, tls=tls_cdf, use_syrk=true), fctx_cdf)

for (name, h) in [("v2 serial/gemm", h_v2_serial), ("v2 serial/syrk", h_v2_syrk),
                   ("v2 threaded/gemm", h_v2_thread), ("v2 threaded/syrk (production candidate)", h_v2_full)]
    d = maximum(abs.(h_serial_ref .- h))
    println("D=4 max|H_serial_ref - H_$name| = $d")
    check("D=4 $name matches serial reference to 1e-10", d < 1e-10)
end

# End-to-end solve identity via the threaded/syrk callback
K_t, x_t, ns_t, _, _ = inner_loop_internal_archgeneric(aug_cdf.obj_cm, θ_full0;
    hess_cb_builder = _o -> archC_frechet_hess_cb_builder_v2(fctx_cdf; threaded_bins=true, tls=tls_cdf, use_syrk=true))
check("D=4 threaded/syrk end-to-end solve feasible", ns_t in (0,-100,-101,-103))
base_ref = BaseDualState(collect(x_free_calib), θ_full0, x_cdf[1], collect(x_cdf[2:end]), copy(aug_cdf.obj_cm.arg1), ns_cdf)
base_t   = BaseDualState(collect(x_free_calib), θ_full0, x_t[1], collect(x_t[2:end]), copy(aug_cdf.obj_cm.arg1), ns_t)
Delta_ref = delta_dual_from_base(aug_cdf.obj_cm, base_ref)
Delta_t   = delta_dual_from_base(aug_cdf.obj_cm, base_t)
println("Delta*_ref=$Delta_ref  Delta*_threaded=$Delta_t  diff=$(abs(Delta_ref-Delta_t))")
check("D=4 Delta* threaded/syrk solve == serial reference solve (1e-6)", isapprox(Delta_ref, Delta_t; atol=1e-6, rtol=1e-6))

println()
println("="^100)
println("RESULT: $n_pass passed, $n_fail failed")
println("="^100)
exit(n_fail == 0 ? 0 : 1)
