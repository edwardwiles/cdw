# Matched flexible-CM vs CDF-only fixed-Fréchet comparison (task addendum §7), SAME D=20/W=80,000/
# L=50/:exclude_row context, SAME threaded/syrk Architecture-C Hessian discipline on both sides
# (cm_hessian_threaded.jl's existing archC_hess_cb_builder_v2 for flexible CM,
# cm_frechet_hessian_threaded.jl's archC_frechet_hess_cb_builder_v2 for CDF-only fixed Fréchet) --
# an apples-to-apples comparison of the SAME optimization applied to both, not old-vs-new.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "draw_design.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle_threaded.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_cdf_only_gradient.jl"))
using Printf, LinearAlgebra, Dates

const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fixed_frechet_cdf_only_bench_2026-07-24")
mkpath(OUTDIR)
const LOGIO = open(joinpath(OUTDIR, "matched_cm_vs_frechet.txt"), "w")
function lp(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
lp("matched_cm_vs_frechet_cdf_only_bench.jl  nthreads=$(Threads.nthreads())  started=$(now())")
function vmhwm_gb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2]) / 1e6
        end
    catch
    end
    return -1.0
end

t0 = time()
const W = 80_000; const L = 50; const DELTA = 1.0
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp(@sprintf("[%.1fs] ctx built (SHARED)  D=%d D_dest=%d  VmHWM=%.2fGB", time()-t0, ctx.D, ctx.D_dest, vmhwm_gb()))
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ0 = CS.reconstruct_full(x_free_calib, ctx.m)

# ---- flexible CM ----
t1 = time()
pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :orthonormal)
t_pcx = time() - t1
lp(@sprintf("[%.1fs] flexible-CM pcx built in %.1fs  ncore=%d ncm=%d  n_inner=%d  VmHWM=%.2fGB",
    time()-t0, t_pcx, pcx.aug.ncore, pcx.aug.ncm, pcx.aug.ncore+pcx.aug.ncm, vmhwm_gb()))
tls_flex = build_thread_local_scratch(pcx.cctx)

# ---- CDF-only fixed Fréchet ----
t2 = time()
cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
t_fpcx = time() - t2
lp(@sprintf("[%.1fs] CDF-only fpcx built in %.1fs  ncore=%d ncm=%d  n_inner=%d  VmHWM=%.2fGB",
    time()-t0, t_fpcx, fpcx.aug.ncore, fpcx.aug.ncm, fpcx.aug.ncore+fpcx.aug.ncm, vmhwm_gb()))
tls_frec = build_thread_local_scratch(fpcx.fctx.cctx)

lp(""); lp("="^100)
lp(@sprintf("DIMENSION COMPARISON: flexible-CM n=%d (ncore=%d ncm=(D-1)*L=%d)  vs  CDF-only n=%d (ncore=%d ncm=D*L=%d)  ratio(n)=%.3f",
    pcx.aug.ncore+pcx.aug.ncm, pcx.aug.ncore, pcx.aug.ncm,
    fpcx.aug.ncore+fpcx.aug.ncm, fpcx.aug.ncore, fpcx.aug.ncm,
    (fpcx.aug.ncore+fpcx.aug.ncm)/(pcx.aug.ncore+pcx.aug.ncm)))
lp("="^100)

function bench_fg(obj, θ, name)
    obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)  # warmup
    t = @elapsed obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    a = @allocated obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    lp(@sprintf("%-30s FG callback: %.4fs  %d bytes (%.3fMB)", name, t, a, a/1e6))
    return (t=t, bytes=a)
end
bench_fg(pcx.ctx_cm.obj, θ0, "flexible-CM")
bench_fg(fpcx.ctx_cm.obj, θ0, "CDF-only fixed-Frechet")

function bench_hess(obj, hess_fn!, nh, name)
    h = zeros(nh)
    hess_fn!(h, obj)  # warmup
    t = @elapsed hess_fn!(h, obj)
    a = @allocated hess_fn!(h, obj)
    lp(@sprintf("%-30s Hessian callback (threaded/syrk, nt=%d): %.4fs  %d bytes (%.3fMB)", name, Threads.nthreads(), t, a, a/1e6))
    return (t=t, bytes=a)
end
n_flex = pcx.aug.ncore + pcx.aug.ncm; nh_flex = div(n_flex*(n_flex+1),2)
obj_flex = pcx.ctx_cm.obj
x_probe_flex = zeros(n_flex)
_archC_prep_for_hessian!(obj_flex, x_probe_flex)
bench_hess(obj_flex, (h,o)->hessian_cm_structured_v2!(h,o,pcx.cctx; threaded_bins=true, tls=tls_flex, use_syrk=true), nh_flex, "flexible-CM")

n_frec = fpcx.aug.ncore + fpcx.aug.ncm; nh_frec = div(n_frec*(n_frec+1),2)
obj_frec = fpcx.ctx_cm.obj
x_probe_frec = zeros(n_frec)
_archC_prep_for_hessian!(obj_frec, x_probe_frec)
bench_hess(obj_frec, (h,o)->hessian_cm_frechet_structured_v2!(h,o,fpcx.fctx; threaded_bins=true, tls=tls_frec, use_syrk=true), nh_frec, "CDF-only fixed-Frechet")

lp(""); lp("-"^100); lp("FULL INNER-SOLVE at the SAME theta0 (calibration point) -- KNITRO nt=1"); lp("-"^100)
opt1 = joinpath(@__DIR__, "frechet_bench_opts", "ek_inner_nt1.opt")
function run_solve(name, obj, θ, hess_cb_builder)
    obj.inner_loop_opt = opt1
    prof_reset!(); CS.INNER_ITERS_TOTAL[] = 0
    t = @elapsed (K, x_sol, nStatus, n_fg, n_hess) = inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = hess_cb_builder)
    knitro_total = haskey(PROF_TIMES, "inner_knitro_dual_solve_arch") ? sum(PROF_TIMES["inner_knitro_dual_solve_arch"]) : NaN
    lp(@sprintf("%-30s wall=%.2fs  knitro=%.2fs  nStatus=%d  n_fg=%d  n_hess=%d  iters=%d  VmHWM=%.2fGB",
        name, t, knitro_total, nStatus, n_fg, n_hess, CS.INNER_ITERS_TOTAL[], vmhwm_gb()))
    return t
end
run_solve("flexible-CM (threaded/syrk)", obj_flex, θ0, _o -> archC_hess_cb_builder_v2(pcx.cctx; threaded_bins=true, tls=tls_flex, use_syrk=true))
run_solve("CDF-only Frechet (threaded/syrk)", obj_frec, θ0, _o -> archC_frechet_hess_cb_builder_v2(fpcx.fctx; threaded_bins=true, tls=tls_frec, use_syrk=true))

lp(""); lp("SUMMARY total_wall=$(round(time()-t0,digits=1))s  VmHWM=$(round(vmhwm_gb(),digits=2))GB  finished=$(now())")
close(LOGIO)
