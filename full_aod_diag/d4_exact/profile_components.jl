# ============================================================================
# Phase 1B: rigorous component-level benchmark. Warm-up (discarded, JIT) then
# N repetitions at a fixed point/draws/settings, reporting median/min/p90/p95/
# std wall time and allocation stats per internal stage (via oracle_profiled.jl).
# Run: julia --project=. full_aod_diag/d4_exact/profile_components.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_profiled.jl"))
using Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "profile_components")
mkpath(OUTDIR)
const N_REPS = 50

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)

println("="^78); println("PHASE 1B: component-level profiling, N=$N_REPS reps per condition"); println("="^78)
flush(stdout)

# ---- warm-up: force JIT compile of every code path BEFORE any timed rep ----
println("Warm-up (discarded, JIT compile)...")
for warm in (true, false)
    evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = warm)
end
prof_reset!()   # discard warm-up measurements

println("Timed repetitions...")
total_times_warm = Float64[]; total_times_cold = Float64[]
for rep in 1:N_REPS
    t0 = time_ns()
    evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = true)
    push!(total_times_warm, (time_ns() - t0) / 1e9)
end
for rep in 1:N_REPS
    t0 = time_ns()
    evaluate_fullA_profiled(x0, ctx; cache = nothing, warm = false)
    push!(total_times_cold, (time_ns() - t0) / 1e9)
end

rows = prof_summary()
# add the two whole-call totals as extra rows (not captured by internal @prof labels, which sum to
# LESS than total wall time -- the gap is un-instrumented glue code, KNITRO's C-level call overhead
# outside the timed inner_solve block, and instrumentation overhead itself)
function stats_row(label, times)
    sorted = sort(times); n = length(sorted)
    μ = sum(sorted)/n
    σ = n > 1 ? sqrt(sum((t-μ)^2 for t in sorted)/(n-1)) : 0.0
    return (label = label, n = n, median_s = sorted[n÷2+1], min_s = sorted[1], max_s = sorted[end],
            p90_s = sorted[clamp(ceil(Int, 0.9*n), 1, n)], p95_s = sorted[clamp(ceil(Int, 0.95*n), 1, n)],
            mean_s = μ, std_s = σ, mean_alloc_bytes = NaN, total_alloc_bytes = NaN, mean_gc_s = NaN, total_gc_s = NaN)
end
push!(rows, stats_row("TOTAL_evaluate_fullA_warm", total_times_warm))
push!(rows, stats_row("TOTAL_evaluate_fullA_cold", total_times_cold))

write_csv_rows(joinpath(OUTDIR, "profile_components.csv"), rows)

# allocation-sorted view
alloc_rows = sort(filter(r -> !isnan(r.total_alloc_bytes), rows), by = r -> -r.total_alloc_bytes)
write_csv_rows(joinpath(OUTDIR, "profile_allocations.csv"), alloc_rows)

println("\n" * "="^78); println("TOP 10 HOTSPOTS BY MEDIAN WALL TIME"); println("="^78)
for r in sort(rows, by = r -> -r.median_s)[1:min(10,length(rows))]
    @printf("  %-30s median=%.6fs  p95=%.6fs  n=%d\n", r.label, r.median_s, r.p95_s, r.n)
end
println("\n" * "="^78); println("TOP 10 HOTSPOTS BY TOTAL ALLOCATION"); println("="^78)
for r in alloc_rows[1:min(10,length(alloc_rows))]
    @printf("  %-30s total_alloc=%.3f MB  mean_alloc=%.1f KB  n=%d\n", r.label, r.total_alloc_bytes/1e6, r.mean_alloc_bytes/1e3, r.n)
end

println("\nWrote ", joinpath(OUTDIR, "profile_components.csv"), " and profile_allocations.csv")
