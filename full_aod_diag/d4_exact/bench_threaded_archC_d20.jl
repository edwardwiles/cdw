# Production integration continuation, Section 10: Julia-thread scaling benchmark for the
# draw-chunk-threaded Architecture C Hessian (cm_hessian_architecture_threaded.jl), D20/L=50/
# W=80000. Run once per JULIA_NUM_THREADS setting (1/5/10/20) -- Julia's thread count is fixed at
# process start, so this script is invoked as a separate process per thread count, each run
# appending one row to a shared CSV. OPENBLAS_NUM_THREADS must be pinned to 1 by the caller (see
# docs/fullA_thread_scaling_d20.md) so BLAS's own threading doesn't confound the measurement.
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
include(joinpath(@__DIR__, "cm_hessian_architecture_interval.jl"))
include(joinpath(@__DIR__, "cm_hessian_architecture_threaded.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
using Printf, LinearAlgebra, Random, Statistics
using Base.Threads: nthreads

W = 80000
DELTA = 1.0
L = 50
N_REPS = 15   # Hessian-callback-only timing (cheap, ~seconds each)
N_REPS_SOLVE = 3   # full cold-inner-solve timing (expensive, multiple KNITRO iterations each)
NT = nthreads()
OUT_CSV = joinpath(@__DIR__, "..", "..", "docs", "fullA_thread_scaling_d20_L$(L)_W$(W).csv")

println(">>> JULIA_NUM_THREADS (nthreads())=", NT, "  OPENBLAS_NUM_THREADS=", get(ENV, "OPENBLAS_NUM_THREADS", "unset"))
println(">>> building D20 real-data context, W=$W ...")
t0 = time()
ctx = d20_real_setup(W = W, δ = DELTA, find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D
ctx_build_s = time() - t0
@printf ">>> ctx built in %.1fs. D=%d W=%d\n" ctx_build_s D W

x_free_calib = ctx.θ0_up[ctx.free_idx]

cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = L,
               cm_basis = :cumulative, cm_hessian_backend = :structured)
pcx = build_cm_production_context_v2(ctx, CS, cfg)
ctx_cm = pcx.ctx_cm
cctx = build_cm_bin_ctx(ctx, pcx.aug)
cctxt = build_cm_bin_ctx_threaded(ctx, pcx.aug)

println(">>> [cold inner solve] serial-Hessian archC, n_reps=$N_REPS_SOLVE ...")
cold_serial = Float64[]
for i in 1:N_REPS_SOLVE
    tc0 = time_ns()
    base = cm_base_state_v2(x_free_calib, pcx)
    push!(cold_serial, (time_ns() - tc0) / 1e9)
end
println(">>> [cold inner solve] threaded-Hessian archC, n_reps=$N_REPS_SOLVE ...")
# Build a pcx_threaded-equivalent value function inline (build_cm_production_context_v2 always
# wires the serial archC_hess_cb_builder for cm_basis=:cumulative -- not modified; this script
# swaps in the threaded builder directly for the A/B comparison, reusing the same ctx_cm/pcx.aug).
function cold_solve_threaded(x_free0)
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0;
        hess_cb_builder = _obj -> archC_threaded_hess_cb_builder(cctxt))
    nStatus in (0, -100, -101, -103) || error("cold_solve_threaded: nStatus=$nStatus")
    return nStatus
end
cold_threaded = Float64[]
for i in 1:N_REPS_SOLVE
    tc0 = time_ns()
    cold_solve_threaded(x_free_calib)
    push!(cold_threaded, (time_ns() - tc0) / 1e9)
end

# ---- Hessian-callback-only timing (isolates the piece Section 9 identified as 80.2% of cost) ----
base = cm_base_state_v2(x_free_calib, pcx)
_archC_prep_for_hessian!(ctx_cm.obj, vcat(base.ζstar, base.λstar))
n = pcx.aug.ncore + pcx.aug.ncm
h = Vector{Float64}(undef, n * (n + 1) ÷ 2)

hessian_cm_structured!(h, ctx_cm.obj, cctx)          # warmup (JIT), serial
hessian_cm_structured_threaded!(h, ctx_cm.obj, cctxt) # warmup (JIT), threaded

hess_serial = Float64[]
for i in 1:N_REPS
    tc0 = time_ns()
    hessian_cm_structured!(h, ctx_cm.obj, cctx)
    push!(hess_serial, (time_ns() - tc0) / 1e9)
end
hess_threaded = Float64[]
for i in 1:N_REPS
    tc0 = time_ns()
    hessian_cm_structured_threaded!(h, ctx_cm.obj, cctxt)
    push!(hess_threaded, (time_ns() - tc0) / 1e9)
end

med(v) = sort(v)[cld(length(v), 2)]

@printf ">>> RESULTS nthreads=%d:\n" NT
@printf "    hessian-only:  serial median=%.4fs  threaded median=%.4fs  speedup=%.2fx\n" med(hess_serial) med(hess_threaded) (med(hess_serial)/med(hess_threaded))
@printf "    cold inner solve: serial-Hessian median=%.4fs  threaded-Hessian median=%.4fs  speedup=%.2fx\n" med(cold_serial) med(cold_threaded) (med(cold_serial)/med(cold_threaded))

row = (nthreads = NT, hess_serial_median_s = med(hess_serial), hess_threaded_median_s = med(hess_threaded),
       hess_speedup = med(hess_serial) / med(hess_threaded),
       cold_solve_serial_median_s = med(cold_serial), cold_solve_threaded_median_s = med(cold_threaded),
       cold_solve_speedup = med(cold_serial) / med(cold_threaded))

mkpath(dirname(OUT_CSV))
header = join(string.(keys(row)), ",")
line = join(string.(values(row)), ",")
exists = isfile(OUT_CSV)
open(OUT_CSV, "a") do io
    exists || println(io, header)
    println(io, line)
end
println(">>> appended row to $OUT_CSV")
println("DONE")
