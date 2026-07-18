# ============================================================================
# Phase 1A: low-overhead, feature-flagged timing/allocation instrumentation.
#
# Purely additive -- no existing file is modified. `@prof` wraps an expression
# with a wall-time + allocation + GC-time measurement, recorded into a global
# per-label log, ONLY when `PROF_ENABLED[]` is true (a single Bool check when
# disabled -- near-zero overhead, matches the task's "low-overhead,
# feature-flagged" requirement). Uses only Julia builtins (`Base.gc_num`,
# `Base.GC_Diff`, `time_ns` -- the same primitives `@timed`/`@allocated` use
# internally), no new dependency, per the task's "prefer built-ins" guidance.
# ============================================================================

const PROF_ENABLED = Ref(true)
const PROF_TIMES  = Dict{String, Vector{Float64}}()   # seconds, per label
const PROF_ALLOCS = Dict{String, Vector{Int}}()        # bytes, per label
const PROF_GCTIME = Dict{String, Vector{Float64}}()    # seconds spent in GC, per label
const PROF_COUNTS = Dict{String, Int}()

"Record one measurement for `label`. Not exported as public API -- call via `@prof`."
function prof_record!(label::String, elapsed_s::Float64, allocated_bytes::Int, gc_s::Float64)
    push!(get!(() -> Float64[], PROF_TIMES, label), elapsed_s)
    push!(get!(() -> Int[], PROF_ALLOCS, label), allocated_bytes)
    push!(get!(() -> Float64[], PROF_GCTIME, label), gc_s)
    PROF_COUNTS[label] = get(PROF_COUNTS, label, 0) + 1
end

"""
    @prof "label" expr

Times `expr`, records wall time / allocated bytes / GC time under `"label"`
(only while `PROF_ENABLED[]`), and returns `expr`'s value unchanged. Safe to
nest (each `@prof` block's own overhead is excluded from its parent's timing
only if the parent is measured around the nested call, i.e. nested `@prof`
blocks double-count inclusive time by design -- this is standard for a
simple label-sum profiler and is called out explicitly in
docs/fullA_performance_profile.md rather than silently assumed away).
"""
macro prof(label, expr)
    quote
        if PROF_ENABLED[]
            local gcstats0 = Base.gc_num()
            local t0 = time_ns()
            local result = $(esc(expr))
            local t1 = time_ns()
            local gcdiff = Base.GC_Diff(Base.gc_num(), gcstats0)
            prof_record!($(esc(label)), (t1 - t0) / 1e9, gcdiff.allocd, gcdiff.total_time / 1e9)
            result
        else
            $(esc(expr))
        end
    end
end

"Clear all recorded profiling data (call before each fresh benchmark run)."
function prof_reset!()
    empty!(PROF_TIMES); empty!(PROF_ALLOCS); empty!(PROF_GCTIME); empty!(PROF_COUNTS)
end

function _quantile_sorted(sorted::Vector{Float64}, q::Float64)
    isempty(sorted) && return NaN
    n = length(sorted)
    n == 1 && return sorted[1]
    pos = q * (n - 1) + 1
    lo = floor(Int, pos); hi = ceil(Int, pos)
    lo == hi && return sorted[lo]
    frac = pos - lo
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
end

"""
    prof_summary() -> Vector{NamedTuple}

One row per label: n, median/min/p90/p95/std of wall time (seconds), mean
allocated bytes, total allocated bytes, mean/total GC time (seconds).
"""
function prof_summary()
    rows = NamedTuple[]
    for label in sort(collect(keys(PROF_TIMES)))
        times = sort(PROF_TIMES[label])
        allocs = PROF_ALLOCS[label]
        gct = PROF_GCTIME[label]
        n = length(times)
        μ = sum(times) / n
        σ = n > 1 ? sqrt(sum((t - μ)^2 for t in times) / (n - 1)) : 0.0
        push!(rows, (label = label, n = n,
            median_s = _quantile_sorted(times, 0.5), min_s = times[1], max_s = times[end],
            p90_s = _quantile_sorted(times, 0.9), p95_s = _quantile_sorted(times, 0.95),
            mean_s = μ, std_s = σ,
            mean_alloc_bytes = sum(allocs) / n, total_alloc_bytes = sum(allocs),
            mean_gc_s = sum(gct) / n, total_gc_s = sum(gct)))
    end
    return rows
end

"Write `rows` (a Vector{NamedTuple}, all with identical keys) to a CSV file with no external dependency."
function write_csv_rows(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && (open(path, "w") do io; println(io, "(no rows)"); end; return)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((r[c] for c in cols), ","))
        end
    end
end
