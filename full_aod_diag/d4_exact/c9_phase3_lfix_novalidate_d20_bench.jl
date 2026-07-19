# ============================================================================
# Continuation 9, Phase 3.2: re-measure the full 400-coordinate L_fix gradient
# wall time at the real D=20 economy (W=80,000, calibration point) with
# build_lfix_base_cache's new validate_dense kwarg ON vs OFF, in the SAME
# warmed process (avoids cross-run noise in the A/B comparison; the W80k
# doc's already-published 6.435s (docs/fullA_D20_W80k_microbenchmark.md
# §3C, adaptive+threaded) is the historical validate_dense=true-equivalent
# number -- reproduced here directly as a same-process cross-check, not
# assumed).
#
# Same rep count/config as the W80k doc's Part 1C primary row
# (h_mode=:adaptive, threaded=true, multi_method=:top3, N=4).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Statistics, Printf, Dates, LinearAlgebra

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase3_lfix_novalidate_d20")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase3_lfix_novalidate_d20_bench.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

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
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

pe = build_pivot_elimination(ctx)
xf1 = ctx.θ0_up[ctx.free_idx]
base1 = solve_base_state(xf1, ctx)

# ---- warm-up both branches (untimed) ----
logprint("\nWarming up (untimed)...")
t0 = time()
composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = true, h_mode = :adaptive, multi_method = :top3, validate_dense = true)
composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = true, h_mode = :adaptive, multi_method = :top3, validate_dense = false)
logprint("Warm-up complete, wall = ", round(time() - t0, digits = 1), "s")

"Wrapped in a function so g_true/g_false escape Julia top-level scripts' soft-scope for-loop ambiguity cleanly (a bug in an earlier version of this script left g_true/g_false undefined after the loop -- caught by the UndefVarError below, not silently wrong)."
function timed_reps(xf1, ctx, pe, base1; validate_dense::Bool, N::Int = 4)
    times = Float64[]
    local g_last, meta_last
    for _ in 1:N
        t0 = time()
        g_last, meta_last = composite_gradient_at_fast(xf1, ctx, pe; base = base1, threaded = true, h_mode = :adaptive, multi_method = :top3, validate_dense = validate_dense)
        push!(times, time() - t0)
    end
    return times, g_last
end

# ---- validate_dense = true (the OLD unconditional behavior) ----
logprint("\n---- validate_dense=true (old unconditional-dense-rebuild behavior), N=4 ----")
times_true, g_true = timed_reps(xf1, ctx, pe, base1; validate_dense = true)
logprint(@sprintf("  median=%.3fs  reps=%s", median(times_true), round.(times_true, digits = 3)))

# ---- validate_dense = false (the NEW default) ----
logprint("\n---- validate_dense=false (NEW default), N=4 ----")
times_false, g_false = timed_reps(xf1, ctx, pe, base1; validate_dense = false)
logprint(@sprintf("  median=%.3fs  reps=%s", median(times_false), round.(times_false, digits = 3)))

# ---- correctness: same-process gradients must match ----
maxdiff = maximum(abs.(g_true .- g_false))
logprint("\nbit_identical(g_true, g_false) = ", g_true == g_false, "  max|diff| = ", maxdiff)

med_true = median(times_true); med_false = median(times_false)
speedup = med_true / med_false
logprint(@sprintf("\nHEADLINE: validate_dense=true median=%.3fs  validate_dense=false median=%.3fs  speedup=%.3fx",
          med_true, med_false, speedup))
logprint("For reference, docs/fullA_D20_W80k_microbenchmark.md's already-published Part 1C ",
          "(adaptive+threaded, effectively validate_dense=true since that was unconditional then) = 6.435s")

write_csv_rows(joinpath(OUTDIR, "validate_dense_ab.csv"),
    [(config = "validate_dense_true", median_s = med_true, reps = join(round.(times_true, digits=3), ";")),
     (config = "validate_dense_false", median_s = med_false, reps = join(round.(times_false, digits=3), ";")),
     (config = "speedup", median_s = speedup, reps = "")])

logprint("VmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase3_lfix_novalidate_d20_bench.jl COMPLETE at ", now())
close(LOGIO)
