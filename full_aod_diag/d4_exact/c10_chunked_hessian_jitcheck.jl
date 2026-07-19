# ============================================================================
# Continuation 10, Part 1: JIT-order control check.
#
# c10_chunked_hessian_bench_d20.jl's main sweep ALWAYS ran the dense reference
# cold solve strictly FIRST in the process, then each chunk size's cold solve
# afterward -- so the reference's cold time absorbs ALL first-call JIT
# compilation (KN_new/KN_add_vars/callback registration/etc, much of which is
# SHARED with the chunked path), while every chunked variant's "cold" solve
# benefits from that already-paid JIT tax. The main sweep's own isolated
# Hessian-callback-only timing (both sides warmed up before their own 10-rep
# loop) showed near-1.0x (0.95x-1.06x) -- but the FULL cold-solve numbers
# showed a suspiciously uniform ~1.7-1.8x speedup for EVERY chunk size,
# including chunk_size=80000 (mathematically identical to the dense baseline,
# bit-identical Hessian) -- a strong signal that the "1.7x" is a JIT-ordering
# artifact, not a genuine perf difference (per this investigation's standing
# "verify before causal claims" discipline).
#
# This script controls for that directly: after a warm-up pass that exercises
# BOTH the reference AND every chunk size ONCE (paying all first-call JIT for
# both paths in this same process), it re-runs cold (`obj.x .= NaN`, no warm
# start) solves for reference and each chunk size AGAIN, this time with
# everything already JIT-compiled -- isolating the genuine numerical/algorithmic
# cold-solve cost from compilation-order noise.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "chunked_hessian.jl"))
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c10_chunked_hessian_bench_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "jitcheck_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_chunked_hessian_jitcheck.jl starting at ", now())

ctx = d20_real_setup(W = 80000)
obj = ctx.obj
xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)
chunk_sizes = [5000, 10000, 20000, 40000, 80000]

logprint("\n---- WARM-UP PASS (pays first-call JIT for every variant, in this order: chunks first, reference LAST -- reverse of the main sweep) ----")
for cs in chunk_sizes
    obj.x .= NaN
    t = @elapsed inner_loop_internal_chunked(obj, θ_full, cs)
    logprint(@sprintf("  warmup chunk_size=%d: %.3fs", cs, t))
end
obj.x .= NaN
t = @elapsed inner_loop_internal_profiled(obj, θ_full)
logprint(@sprintf("  warmup reference: %.3fs", t))

logprint("\n---- TIMED PASS (everything already JIT-compiled for both paths; reference run FIRST this time) ----")
function timed_cold(fn, args...)
    obj.x .= NaN
    return @elapsed fn(args...)
end

N_REPS = 3
ref_times = Float64[timed_cold(inner_loop_internal_profiled, obj, θ_full) for _ in 1:N_REPS]
logprint(@sprintf("  reference cold (JIT-warm): median=%.3fs  reps=%s", median(ref_times), round.(ref_times, digits=3)))

for cs in chunk_sizes
    times = Float64[timed_cold(inner_loop_internal_chunked, obj, θ_full, cs) for _ in 1:N_REPS]
    med = median(times)
    logprint(@sprintf("  chunk_size=%6d cold (JIT-warm): median=%.3fs  speedup=%.3fx  reps=%s",
              cs, med, median(ref_times) / med, round.(times, digits=3)))
end

logprint("\nc10_chunked_hessian_jitcheck.jl COMPLETE at ", now())
close(LOGIO)
