# ============================================================================
# Continuation 9, Phase 4: end-to-end (real multi-iteration KNITRO inner
# solve) comparison of the destination-major _v2 FG callback
# (compressed_live_v2.jl) vs the existing compressed baseline
# (compressed_live.jl), at D=20/W=80,000. Per this task's "verify empirically
# rather than assuming" discipline (mirroring Phase 3C's own finding that an
# isolated FLOP-count win did not survive contact with a real solve) -- the
# isolated per-call kernel win (c9_phase4_kernels_v2_d20_bench.jl,
# ~1.2-1.3x) is checked here against actual wall-clock solve time, not
# assumed to carry over.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_cc_kernels_v2.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "compressed_live_v2.jl"))
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase4_v2_e2e_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase4_v2_e2e_d20_bench.jl starting at ", now(), "  commit=", COMMIT)

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D
obj = ctx.obj
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

xf_nat = ctx.θ0_up[ctx.free_idx]
θ_full = CS.reconstruct_full(xf_nat, ctx.m)

# ---- warm-up (untimed) ----
obj.x .= NaN
inner_loop_internal_compressed(obj, θ_full, ctx)
obj.x .= NaN
inner_loop_internal_compressed_v2(obj, θ_full, ctx)

N = 5
function time_reps_cold(f, N)
    times = Float64[]
    for _ in 1:N
        obj.x .= NaN
        push!(times, @elapsed f())
    end
    return median(times), times
end
function time_reps_warm(f, N)
    times = Float64[]
    for _ in 1:N
        push!(times, @elapsed f())
    end
    return median(times), times
end

logprint("\n---- COLD inner solve, N=", N, " ----")
med_orig_cold, times_orig_cold = time_reps_cold(() -> inner_loop_internal_compressed(obj, θ_full, ctx), N)
med_v2_cold, times_v2_cold = time_reps_cold(() -> inner_loop_internal_compressed_v2(obj, θ_full, ctx), N)
logprint(@sprintf("  original: median=%.4fs  reps=%s", med_orig_cold, round.(times_orig_cold, digits=4)))
logprint(@sprintf("  v2:       median=%.4fs  reps=%s", med_v2_cold, round.(times_v2_cold, digits=4)))
logprint(@sprintf("  speedup(cold) = %.3fx", med_orig_cold / med_v2_cold))

logprint("\n---- WARM inner solve, N=", N, " ----")
obj.x .= NaN; inner_loop_internal_compressed(obj, θ_full, ctx)   # establish a warm x
med_orig_warm, times_orig_warm = time_reps_warm(() -> inner_loop_internal_compressed(obj, θ_full, ctx), N)
obj.x .= NaN; inner_loop_internal_compressed_v2(obj, θ_full, ctx)
med_v2_warm, times_v2_warm = time_reps_warm(() -> inner_loop_internal_compressed_v2(obj, θ_full, ctx), N)
logprint(@sprintf("  original: median=%.4fs  reps=%s", med_orig_warm, round.(times_orig_warm, digits=4)))
logprint(@sprintf("  v2:       median=%.4fs  reps=%s", med_v2_warm, round.(times_v2_warm, digits=4)))
logprint(@sprintf("  speedup(warm) = %.3fx", med_orig_warm / med_v2_warm))

# ---- correctness: final duals/K_hard must match ----
obj.x .= NaN
K_o, x_o, status_o, nfg_o, nhess_o, st_o = inner_loop_internal_compressed(obj, θ_full, ctx)
obj.x .= NaN
K_v, x_v, status_v, nfg_v, nhess_v, st_v = inner_loop_internal_compressed_v2(obj, θ_full, ctx)
dK = abs(K_o - K_v); dx = maximum(abs.(x_o .- x_v))
logprint("\n  correctness: status_orig=", status_o, " status_v2=", status_v, "  |dK_hard|=", dK, "  max|dx|=", dx,
         "  n_fg(orig)=", nfg_o, " n_fg(v2)=", nfg_v, " n_hess(orig)=", nhess_o, " n_hess(v2)=", nhess_v)

write_csv_rows(joinpath(OUTDIR, "v2_e2e_speedup.csv"),
    [(mode = "cold", orig_median_s = med_orig_cold, v2_median_s = med_v2_cold, speedup = med_orig_cold/med_v2_cold),
     (mode = "warm", orig_median_s = med_orig_warm, v2_median_s = med_v2_warm, speedup = med_orig_warm/med_v2_warm),
     (mode = "correctness", orig_median_s = dK, v2_median_s = dx, speedup = NaN)])

logprint("\nVmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase4_v2_e2e_d20_bench.jl COMPLETE at ", now())
close(LOGIO)
