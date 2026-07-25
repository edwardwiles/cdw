# CDF-only fixed-Fréchet production-feasibility benchmark, D=20/W=80,000/L=50, real data,
# destination_sample=:exclude_row. Task: FIXED_FRECHET_INNER_SOLVER_ARCHITECTURE 2026-07-24
# (CDF-only addendum supersedes CDF+POWER as the production target).
#
# Usage: JULIA_NUM_THREADS=<N> julia --project=. frechet_cdf_only_bench.jl <label> [run_knitro_sweep]
#
# Measures, at three exactly-reproducible points (P0=calibration, P1=nearby finite point
# gp_target=(1-(kappa*+1e-4))^((sigma-1)/sigma), P2=further nearby point +2e-4):
#   - D=20 correctness: threaded/syrk structured Hessian vs serial reference (direct diff, no solve)
#   - Hessian-callback-ONLY wall time: serial vs threaded_bins, at Threads.nthreads() for this process
#   - Full inner-solve wall time: serial Hessian (baseline) vs threaded Hessian, KNITRO nt=1 fixed
#   - If run_knitro_sweep=true: threaded Hessian x KNITRO numthreads in {1,4,8,20}
#   - Allocations (@allocated) for one warmed FG callback and one warmed Hessian callback
#   - Process VmHWM
#
# NOTE on "warm" P2: this codebase's inner_loop_internal_archgeneric does NOT implement KNITRO
# dual warm-starting (each call is a fresh KN_new/KN_solve/KN_free cycle; see gravity-robustness
# warm-start-map memory: "inner CC dual solve NEVER warm-started in sequential/profiled pipeline").
# P2 is therefore measured as an independent cold solve at a point near P1, NOT a literal warm
# start -- disclosed here rather than fabricating warm-start machinery that doesn't exist.
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
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
using Printf, LinearAlgebra, Dates

label = length(ARGS) >= 1 ? ARGS[1] : "nt$(Threads.nthreads())"
run_knitro_sweep = length(ARGS) >= 2 && ARGS[2] == "true"

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "fixed_frechet_cdf_only_bench_2026-07-24")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "bench_$(label).txt")
const LOGIO = open(LOGPATH, "w")
function lp(xs...)
    println(xs...); println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
lp("frechet_cdf_only_bench.jl  label=$label  run_knitro_sweep=$run_knitro_sweep  commit=$COMMIT  nthreads=$(Threads.nthreads())  started=$(now())")

function vmhwm_gb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2]) / 1e6
        end
    catch
    end
    return -1.0
end

const OPTDIR = joinpath(@__DIR__, "frechet_bench_opts")
opt_for_nt(nt) = joinpath(OPTDIR, "ek_inner_nt$(nt).opt")

t0 = time()
const W = 80_000; const L = 50; const DELTA = 1.0
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
lp(@sprintf("[%.1fs] ctx built  D=%d D_dest=%d  VmHWM=%.2fGB", time()-t0, ctx.D, ctx.D_dest, vmhwm_gb()))
pe = build_pivot_elimination(ctx)

cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_only, frechet_basis = :cumulative)
t_fpcx = @elapsed fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
lp(@sprintf("[%.1fs] fpcx built in %.1fs  ncore=%d ncm=%d  n_inner=%d  VmHWM=%.2fGB",
    time()-t0, t_fpcx, fpcx.aug.ncore, fpcx.aug.ncm, fpcx.aug.ncore+fpcx.aug.ncm+1, vmhwm_gb()))

# ---- P0/P1/P2 construction (reproducible from ctx.θ0_up; P1 recipe matches
# diag_frechet_slow_solve_breakdown.jl's already-observed slow point) ----
x_free_calib = ctx.θ0_up[ctx.free_idx]
z_star = log.(reshape(x_free_calib[2:end], ctx.D, ctx.D_dest))
zfree_star = pivot_reduce(z_star, pe)
σ = ctx.σ
κ_star = 1 - x_free_calib[1]^(σ / (σ - 1))
gp_P1 = (1 - (κ_star + 1e-4))^((σ - 1) / σ)
gp_P2 = (1 - (κ_star + 2e-4))^((σ - 1) / σ)
x_free_P0 = x_free_calib
x_free_P1 = vcat(gp_P1, vec(exp.(pivot_expand(zfree_star, pe))))
x_free_P2 = vcat(gp_P2, vec(exp.(pivot_expand(zfree_star, pe))))
lp(@sprintf("[%.1fs] P0 gp=%.10f (calib)  P1 gp=%.10f (kappa*+1e-4)  P2 gp=%.10f (kappa*+2e-4)  kappa*=%.10f",
    time()-t0, x_free_P0[1], gp_P1, gp_P2, κ_star))
lp("fingerprint P0[1:5] = $(x_free_P0[1:5])")
lp("fingerprint P1[1:5] = $(x_free_P1[1:5])")
lp("fingerprint P2[1:5] = $(x_free_P2[1:5])")

θ_P0 = CS.reconstruct_full(x_free_P0, fpcx.ctx_cm.m)
θ_P1 = CS.reconstruct_full(x_free_P1, fpcx.ctx_cm.m)
θ_P2 = CS.reconstruct_full(x_free_P2, fpcx.ctx_cm.m)

fctx = fpcx.fctx
tls = build_thread_local_scratch(fctx.cctx)
obj = fpcx.ctx_cm.obj

# ============================================================================
lp(""); lp("="^100); lp("D=20 CORRECTNESS GATE: threaded/syrk vs serial structured Hessian (direct diff, no solve)"); lp("="^100)
# ============================================================================
obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ_P1, obj.U, obj)
obj.H[:,2] .= 1.0
# Use a plausible dual iterate (zeros are a valid KNITRO evalRequest.x during the FIRST callback;
# more representative is the P0-calibration weight-vector shape -- use zeros for a config-
# independent gate, matching this repo's own D=4 gate convention). NOTE: obj.outer_constr_index
# (== fpcx.aug.ncore + fpcx.aug.ncm for this archB-built obj, per build_cm_frechet_augmented_obj_archB's
# own @assert) is the length _archC_prep_for_hessian! expects -- NOT +1 for zeta (zeta is already
# folded into this codebase's own "ncore" convention here, confirmed by the DimensionMismatch this
# comment replaces: A had 1382 cols, my first x_probe attempt wrongly had length 1383).
n = fpcx.aug.ncore + fpcx.aug.ncm
@assert n == obj.outer_constr_index "expected n==obj.outer_constr_index; got n=$n obj.outer_constr_index=$(obj.outer_constr_index)"
x_probe = zeros(n)
_archC_prep_for_hessian!(obj, x_probe)
nh = div(n*(n+1), 2)
h_serial = zeros(nh); h_thread = zeros(nh)
hessian_cm_frechet_structured!(h_serial, obj, fctx)
hessian_cm_frechet_structured_v2!(h_thread, obj, fctx; threaded_bins = true, tls = tls, use_syrk = true)
maxerr = maximum(abs.(h_serial .- h_thread))
lp(@sprintf("D=20 max|H_serial - H_threaded/syrk| = %.3e  (n=%d nh=%d)", maxerr, n, nh))
lp(maxerr < 1e-6 ? "PASS: D=20 threaded/syrk Hessian matches serial reference" : "FAIL: D=20 Hessian mismatch exceeds 1e-6")

# ============================================================================
lp(""); lp("="^100); lp("HESSIAN-CALLBACK-ONLY MICROBENCHMARK (no KNITRO, direct repeated calls)"); lp("="^100)
# ============================================================================
function bench_hess(fn!, args...; nrep = 5)
    h = zeros(nh)
    fn!(h, obj, args...)  # warmup / compile
    ts = Float64[]
    allocs = 0
    for i in 1:nrep
        t = @elapsed fn!(h, obj, args...)
        push!(ts, t)
        i == nrep && (allocs = @allocated fn!(h, obj, args...))
    end
    return (median = sort(ts)[div(nrep+1,2)], min = minimum(ts), max = maximum(ts), allocs_bytes = allocs)
end
_archC_prep_for_hessian!(obj, x_probe)
r_serial = bench_hess(hessian_cm_frechet_structured!, fctx)
r_thread = bench_hess((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=true, tls=tls, use_syrk=true), fctx)
r_v2ser  = bench_hess((h,o,f)->hessian_cm_frechet_structured_v2!(h,o,f; threaded_bins=false, use_syrk=true), fctx)
lp(@sprintf("serial (legacy)      : median=%.4fs min=%.4fs max=%.4fs allocs=%d bytes", r_serial.median, r_serial.min, r_serial.max, r_serial.allocs_bytes))
lp(@sprintf("v2 serial/syrk       : median=%.4fs min=%.4fs max=%.4fs allocs=%d bytes", r_v2ser.median, r_v2ser.min, r_v2ser.max, r_v2ser.allocs_bytes))
lp(@sprintf("v2 threaded(%2d)/syrk : median=%.4fs min=%.4fs max=%.4fs allocs=%d bytes  speedup_vs_serial=%.2fx",
    Threads.nthreads(), r_thread.median, r_thread.min, r_thread.max, r_thread.allocs_bytes, r_serial.median/r_thread.median))

# ============================================================================
lp(""); lp("="^100); lp("FULL INNER-SOLVE BENCHMARK (through KNITRO)"); lp("="^100)
# ============================================================================
function run_solve(name, θ, hess_cb_builder, opt_file; force_cold::Bool = false)
    obj.inner_loop_opt = opt_file
    force_cold && (obj.x .= NaN)   # defeat inner_loop_initial_values' obj.use_cached_x warm-start (cc_algo/inner_loop_functions.jl:157-159)
    prof_reset!()
    CS.INNER_ITERS_TOTAL[] = 0
    t = @elapsed begin
        K, x_sol, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ; hess_cb_builder = hess_cb_builder)
    end
    knitro_total = haskey(PROF_TIMES, "inner_knitro_dual_solve_arch") ? sum(PROF_TIMES["inner_knitro_dual_solve_arch"]) : NaN
    outcome = frechet_solve_outcome(nStatus)
    lp(@sprintf("%-48s wall=%8.2fs  knitro=%8.2fs (%.1f%%)  nStatus=%-4d outcome=%-10s n_fg=%-4d n_hess=%-4d iters=%-4d avg_iter=%.3fs  VmHWM=%.2fGB",
        name, t, knitro_total, 100*knitro_total/t, nStatus, String(outcome), n_fg, n_hess,
        CS.INNER_ITERS_TOTAL[], knitro_total/max(CS.INNER_ITERS_TOTAL[],1), vmhwm_gb()))
    return (name=name, wall=t, knitro=knitro_total, nStatus=nStatus, outcome=outcome, n_fg=n_fg, n_hess=n_hess, iters=CS.INNER_ITERS_TOTAL[])
end

hcb_serial(_o) = archC_frechet_hess_cb_builder(fctx)
hcb_thread(_o) = archC_frechet_hess_cb_builder_v2(fctx; threaded_bins = true, tls = tls, use_syrk = true)

nt = Threads.nthreads()
results = NamedTuple[]

# IMPORTANT (found live in this session): cc_algo/inner_loop_functions.jl's inner_loop_initial_values
# uses obj.x (the PREVIOUS solve's converged point, regardless of which theta produced it) as
# KNITRO's initial point whenever obj.use_cached_x && norm(obj.x)<1e6 -- i.e. THIS obj instance
# auto-warm-starts across consecutive solves, exactly as the real outer driver's reused
# fpcx.ctx_cm.obj does in production. A solve immediately following another on the SAME obj is
# therefore NEVER a clean "cold" measurement unless obj.x is explicitly reset first. Both numbers
# are reported below: force_cold=true (genuinely cold, comparable to the ORIGINAL documented
# CDF+POWER ~660-1000s baseline, itself measured as the first-ever solve on a fresh obj) and the
# natural warm-from-previous-point number (representative of real outer-search usage, where the
# same obj IS reused across trial points).
push!(results, run_solve("P1 serial-Hess KNITRO-nt1 COLD (obj.x reset, true baseline)", θ_P1, hcb_serial, opt_for_nt(1); force_cold = true))

# P0: sanity (near-calibration) -- warm from the just-completed P1 cold solve, serial vs threaded
push!(results, run_solve("P0 serial-Hess  KNITRO-nt1 (warm from P1)", θ_P0, hcb_serial, opt_for_nt(1)))
push!(results, run_solve("P0 thread($nt)-Hess KNITRO-nt1 (warm, same theta as prev)", θ_P0, hcb_thread, opt_for_nt(1)))

# P1 again, warm-started from P0 (matches what a real outer search does: same obj reused across trial points)
push!(results, run_solve("P1 thread($nt)-Hess KNITRO-nt1 (warm from P0)", θ_P1, hcb_thread, opt_for_nt(1)))

# P2: warm nearby point -- genuinely warm-started from P1's just-converged solution via obj.x (this
# IS the real dual warm-start the task brief asked P2 to test, contrary to this file's original
# header note before this session's live investigation -- corrected here, not silently).
push!(results, run_solve("P2 thread($nt)-Hess KNITRO-nt1 (warm from P1)", θ_P2, hcb_thread, opt_for_nt(1)))

if run_knitro_sweep
    lp(""); lp("-"^100); lp("KNITRO-thread sweep at P1 (COLD each time -- obj.x reset before every cell, else"); lp("subsequent cells trivially reconverge in 0 iterations from the previous cell's solution at the")
    lp("SAME theta_P1, as first observed live in this session), Julia-threaded($nt) Hessian fixed"); lp("-"^100)
    for knt in (1, 4, 8, 20)
        push!(results, run_solve("P1 thread($nt)-Hess KNITRO-nt$knt COLD", θ_P1, hcb_thread, opt_for_nt(knt); force_cold = true))
    end
    lp(""); lp("-"^100); lp("KNITRO thread-activation confirmation (outlev=1 rerun at nt=20)"); lp("-"^100)
    verbose20 = joinpath(OPTDIR, "ek_inner_nt20_verbose.opt")
    open(verbose20, "w") do io
        for line in eachline(opt_for_nt(20))
            println(io, startswith(line, "outlev") ? "outlev 1" : line)
        end
    end
    run_solve("P1 thread($nt)-Hess KNITRO-nt20 (outlev=1 confirm) COLD", θ_P1, hcb_thread, verbose20; force_cold = true)
end

# ============================================================================
lp(""); lp("="^100); lp("ALLOCATION AUDIT (warmed callbacks, @allocated)"); lp("="^100)
# ============================================================================
_archC_prep_for_hessian!(obj, x_probe)
h_tmp = zeros(nh)
hessian_cm_frechet_structured_v2!(h_tmp, obj, fctx; threaded_bins=true, tls=tls, use_syrk=true)  # warmup
a_hess = @allocated hessian_cm_frechet_structured_v2!(h_tmp, obj, fctx; threaded_bins=true, tls=tls, use_syrk=true)
lp(@sprintf("Hessian callback (threaded/syrk, warmed): %d bytes = %.4f MB", a_hess, a_hess/1e6))

obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ_P0, obj.U, obj)  # warmup
a_fg = @allocated obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ_P0, obj.U, obj)
lp(@sprintf("FG/moments! callback (warmed): %d bytes = %.4f MB", a_fg, a_fg/1e6))

lp(""); lp("="^100); lp("SUMMARY  label=$label  nthreads=$nt  total_wall=$(round(time()-t0,digits=1))s  VmHWM=$(round(vmhwm_gb(),digits=2))GB  finished=$(now())"); lp("="^100)
close(LOGIO)
