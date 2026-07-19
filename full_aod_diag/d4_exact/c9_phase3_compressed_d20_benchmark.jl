# ============================================================================
# Continuation 9, Phase 3.1: dense-vs-compressed VALUE-CALLBACK timing at the
# real D=20 economy, W=80,000, calibration point -- directly comparable to
# docs/fullA_D20_W80k_microbenchmark.md's Part 1A/1B (same @prof-timer
# discipline reused from c8_perfprofile_harness.jl/instrumentation.jl, same
# warm-N=6/cold-N=3 rep counts, same component-table format). The dense-mode
# numbers reproduced here should closely match the already-published W80k
# doc's §3A table (same commit lineage, same context, same point) -- included
# as a same-process cross-check, not because dense needed re-measuring.
#
# NEW this session: :compressed mode was never tried against the real D=20
# context before c9_phase3_compressed_d20_sanity.jl (this directory, run
# immediately before this script) confirmed it works, is memory-safe
# (VmHWM 2.8GB), and agrees with dense to ~4e-19 relative Delta_dual --
# only the TIMED re-measurement is new here.
#
# ONE warmed process for the whole run, same double-include-avoidance
# discipline as c9_w80k_microbenchmark.jl (context_real_d20.jl includes
# context.jl exactly once).
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
using Statistics, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "c9_phase3_compressed_d20_benchmark")
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...)
    println(LOGIO, xs...)
    flush(stdout); flush(LOGIO)
end
logprint("c9_phase3_compressed_d20_benchmark.jl starting at ", now(), "  commit=", COMMIT, "  nthreads=", Threads.nthreads())

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end

function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted) / n
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            mean_s = μ, std_s = (n > 1 ? sqrt(sum((t - μ)^2 for t in sorted) / (n - 1)) : 0.0))
end

function print_component_table(tag, summ)
    logprint("\n---- ", tag, " : component breakdown, median ms (N reps) ----")
    logprint(@sprintf("  %-36s %10s %14s", "component", "N", "median(ms)"))
    rows = NamedTuple[]
    for r in sort(summ; by = rr -> -rr.median_s)
        logprint(@sprintf("  %-36s %10d %14.4f", r.label, r.n, r.median_s * 1000))
        push!(rows, (component = r.label, n = r.n, median_ms = r.median_s * 1000, mean_ms = r.mean_s * 1000,
                     min_ms = r.min_s * 1000, max_ms = r.max_s * 1000, std_ms = r.std_s * 1000))
    end
    write_csv_rows(joinpath(OUTDIR, "$(tag).csv"), rows)
    return rows
end

# ============================================================================
# PART 0: context/setup (once)
# ============================================================================
logprint("\n", "="^90); logprint("PART 0: context/setup, W=80000"); logprint("="^90)
t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
D = ctx.D
logprint("d20_real_setup(W=80000) wall = ", round(t_setup, digits = 2), "s")
logprint("VmHWM after setup = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
xf_nat = ctx.θ0_up[ctx.free_idx]

# ============================================================================
# WARM-UP (untimed): pay JIT for BOTH dense and compressed paths before any
# timed measurement.
# ============================================================================
logprint("\nWarming up (untimed, both modes)...")
t0 = time()
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
end
logprint("Warm-up complete, wall = ", round(time() - t0, digits = 1), "s  VmHWM=", round(vmhwm_kb() / 1e6, digits = 2), " GB")

# Confirm calibration point is feasible + record correctness cross-check once more, warm.
r_dense0, _ = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
r_comp0, meta_comp0 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
logprint("Point 1 (calibration) dense: inner_status=", r_dense0.inner_status, " Delta_dual=", r_dense0.Delta_dual)
logprint("Point 1 (calibration) compressed: inner_status=", r_comp0.inner_status, " Delta_dual=", r_comp0.Delta_dual,
         " fallback_count=", COMPRESSED_FALLBACK_COUNT[])
logprint("abs(Delta_dual diff) = ", abs(r_dense0.Delta_dual - r_comp0.Delta_dual))
logprint("winner_hash match: ", r_dense0.winner_hash == r_comp0.winner_hash)

# ============================================================================
# PART 1: dense, warm (N=6) and cold (N=3) -- reproduces W80k doc's Part 1A/1B
# in this same process, as an internal cross-check.
# ============================================================================
logprint("\n", "="^90); logprint("PART 1: DENSE mode, Point 1 (calibration)"); logprint("="^90)

logprint("\n---- dense, warm-started (N=6) ----")
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
end
prof_reset!()
total_times_dense_warm = Float64[]
for _ in 1:6
    t0 = time_ns()
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
    push!(total_times_dense_warm, (time_ns() - t0) / 1e9)
end
summ_dense_warm = prof_summary()
push!(summ_dense_warm, stats_row("TOTAL", total_times_dense_warm))
rows_dense_warm = print_component_table("dense_warm", summ_dense_warm)

logprint("\n---- dense, COLD (warm=false, N=3) ----")
prof_reset!()
total_times_dense_cold = Float64[]
for _ in 1:3
    t0 = time_ns()
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = false, moment_representation = :dense)
    push!(total_times_dense_cold, (time_ns() - t0) / 1e9)
end
summ_dense_cold = prof_summary()
push!(summ_dense_cold, stats_row("TOTAL", total_times_dense_cold))
rows_dense_cold = print_component_table("dense_cold", summ_dense_cold)

# ============================================================================
# PART 2: compressed, warm (N=6) and cold (N=3) -- the genuinely new numbers.
# ============================================================================
logprint("\n", "="^90); logprint("PART 2: COMPRESSED mode, Point 1 (calibration)"); logprint("="^90)

logprint("\n---- compressed, warm-started (N=6) ----")
for _ in 1:2
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
end
prof_reset!()
total_times_comp_warm = Float64[]
for _ in 1:6
    t0 = time_ns()
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
    push!(total_times_comp_warm, (time_ns() - t0) / 1e9)
end
summ_comp_warm = prof_summary()
push!(summ_comp_warm, stats_row("TOTAL", total_times_comp_warm))
rows_comp_warm = print_component_table("compressed_warm", summ_comp_warm)

logprint("\n---- compressed, COLD (warm=false, N=3) ----")
prof_reset!()
total_times_comp_cold = Float64[]
for _ in 1:3
    t0 = time_ns()
    evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = false, moment_representation = :compressed)
    push!(total_times_comp_cold, (time_ns() - t0) / 1e9)
end
summ_comp_cold = prof_summary()
push!(summ_comp_cold, stats_row("TOTAL", total_times_comp_cold))
rows_comp_cold = print_component_table("compressed_cold", summ_comp_cold)

logprint("\nCOMPRESSED_FALLBACK_COUNT at end of run = ", COMPRESSED_FALLBACK_COUNT[])

# ============================================================================
# PART 3: headline dense-vs-compressed speedup summary
# ============================================================================
logprint("\n", "="^90); logprint("PART 3: headline speedup summary"); logprint("="^90)
med_dense_warm = median(total_times_dense_warm)
med_comp_warm = median(total_times_comp_warm)
med_dense_cold = median(total_times_dense_cold)
med_comp_cold = median(total_times_comp_cold)
logprint(@sprintf("  warm TOTAL: dense=%.4fs  compressed=%.4fs  speedup(dense/compressed)=%.3fx",
          med_dense_warm, med_comp_warm, med_dense_warm / med_comp_warm))
logprint(@sprintf("  cold TOTAL: dense=%.4fs  compressed=%.4fs  speedup(dense/compressed)=%.3fx",
          med_dense_cold, med_comp_cold, med_dense_cold / med_comp_cold))

moment_build_dense = only(r.median_ms for r in rows_dense_warm if r.component == "inner_moment_build") / 1000
moment_build_comp = only(r.median_ms for r in rows_comp_warm if r.component == "inner_moment_build_compressed") / 1000
logprint(@sprintf("  warm inner_moment_build: dense=%.4fs  compressed=%.4fs  speedup=%.3fx",
          moment_build_dense, moment_build_comp, moment_build_dense / moment_build_comp))

write_csv_rows(joinpath(OUTDIR, "part3_headline_speedup.csv"),
    [(mode_pair = "warm_TOTAL", dense_s = med_dense_warm, compressed_s = med_comp_warm, speedup = med_dense_warm / med_comp_warm),
     (mode_pair = "cold_TOTAL", dense_s = med_dense_cold, compressed_s = med_comp_cold, speedup = med_dense_cold / med_comp_cold),
     (mode_pair = "warm_moment_build", dense_s = moment_build_dense, compressed_s = moment_build_comp, speedup = moment_build_dense / moment_build_comp)])

logprint("VmHWM at end of run = ", round(vmhwm_kb() / 1e6, digits = 2), " GB")
logprint("\nc9_phase3_compressed_d20_benchmark.jl COMPLETE at ", now())
close(LOGIO)
