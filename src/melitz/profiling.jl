# Session prompt Section 7: exact, low-overhead profiling instrumentation for the
# production Melitz paths. Uses cumulative `time_ns()`-style timers/counters INSIDE the
# actual production call sites (outer-state construction, moment construction, the inner
# CC solve, the FC/GA outer callbacks) -- NOT a sampling profiler, and not isolated
# standalone microbenchmarks. Switchable off (`MELITZ_PROFILE[] = false`, the default) with
# negligible added cost when disabled: every instrumented site is wrapped in
# `@melitz_profile category expr`, which expands to a single global-Bool branch around
# `expr` with no timer call at all when profiling is off.
#
# Design: one global `MelitzProfiler` (`MELITZ_PROF`) mapping category::Symbol ->
# `MelitzProfileStat` (count, total_ns, max_ns, a bounded ring of raw samples for
# percentile reporting). `melitz_profile_report` prints, per category: call count, total
# wall time, mean, median, p90, max, and percentage of a caller-supplied trajectory total.
#
# This is intentionally NOT an exhaustive per-leaf breakdown of every item listed in the
# governing prompt's Section 7.1-7.8 (that would require rewriting moments.jl's inner
# loops into a dozen separately-timed micro-steps, adding real overhead and code churn
# disproportionate to what the timings would show at D=4/W<=80,000 problem sizes) --
# instead it instruments the categories that structurally dominate wall time
# (outer-state construction, moment-matrix construction split into its two economically
# distinct loops, the inner KNITRO solve, and the outer FC/GA callback's own sub-steps),
# which is what Section 8's actual profiling runs need to identify bottlenecks and the
# upper/lower asymmetry. Finer subdivision can be added later at any specific category
# this first pass finds to be a bottleneck, without changing the profiler machinery itself.

"Global on/off switch (Section 7's own requirement: negligible cost when disabled)."
const MELITZ_PROFILE = Ref(false)

const MELITZ_PROFILE_MAX_SAMPLES = 50_000

mutable struct MelitzProfileStat
    count::Int
    total_ns::Int64
    max_ns::Int64
    samples::Vector{Int64}   # bounded ring buffer (first MELITZ_PROFILE_MAX_SAMPLES only)
end
MelitzProfileStat() = MelitzProfileStat(0, 0, 0, Int64[])

mutable struct MelitzProfiler
    stats::Dict{Symbol,MelitzProfileStat}
end
MelitzProfiler() = MelitzProfiler(Dict{Symbol,MelitzProfileStat}())

"The single global profiler instance every `@melitz_profile` call site records into."
const MELITZ_PROF = MelitzProfiler()

"Clears all recorded stats (call before a fresh profiling run)."
function melitz_profile_reset!()
    empty!(MELITZ_PROF.stats)
    return nothing
end

@inline function melitz_record!(category::Symbol, elapsed_ns::Int64)
    st = get!(MELITZ_PROF.stats, category) do
        MelitzProfileStat()
    end
    st.count += 1
    st.total_ns += elapsed_ns
    elapsed_ns > st.max_ns && (st.max_ns = elapsed_ns)
    length(st.samples) < MELITZ_PROFILE_MAX_SAMPLES && push!(st.samples, elapsed_ns)
    return nothing
end

"""
    @melitz_profile category expr

Times `expr` with `time_ns()` and records the elapsed time under `category` (evaluated
once, must produce a `Symbol`) into the global `MELITZ_PROF` profiler -- ONLY when
`MELITZ_PROFILE[]` is `true`. When disabled (the default), expands to just `expr` guarded
by a single `Bool` check -- no `time_ns()` call, no dictionary lookup, no allocation.

**Exception-safe (Section 3.1, 2026-07-23 continuation session)**: uses `try/finally` so
elapsed time is recorded whether `expr` returns normally OR throws. On a normal return the
time is recorded under `category`; on a throw it is recorded under
`Symbol(category, :_error)` and the original exception is ALWAYS rethrown unmodified (this
macro never swallows an exception -- callers relying on KNITRO.jl's own callback-error
conversion, e.g. `finite_delta_outer.jl`'s `DomainError`-throwing convention, depend on the
exception propagating out exactly as thrown). Previously this macro only wrapped `expr` in
a bare `if`, so any call that ultimately threw (a genuine inner-solve failure surviving a
cold retry) contributed ZERO measured time to its category even though real wall-clock
work happened before the throw -- documented as the single largest instrumentation gap in
`docs/melitz_optimization_report_2026-07-23.md` Section B/H.
"""
macro melitz_profile(category, expr)
    quote
        if MELITZ_PROFILE[]
            local cat = $(esc(category))
            local t0 = time_ns()
            local threw = true
            try
                local r = $(esc(expr))
                threw = false
                r
            finally
                melitz_record!(threw ? Symbol(cat, :_error) : cat, Int64(time_ns() - t0))
            end
        else
            $(esc(expr))
        end
    end
end

"Manually record a pre-measured elapsed time (seconds) under `category` -- for call sites that already measure their own wall time (e.g. `evaluate_melitz_delta`'s existing `state_time`/`inner_time`) and would otherwise double-time the same region."
function melitz_record_seconds!(category::Symbol, elapsed_s::Real)
    MELITZ_PROFILE[] || return nothing
    melitz_record!(category, Int64(round(elapsed_s * 1e9)))
    return nothing
end

"""
    melitz_record_seconds_outcome!(category, outcome, elapsed_s)

Section 3.1's outcome-labeled recording: records under `Symbol(category, :_, outcome)`.
For manual `try/finally` instrumentation at call sites that need to distinguish outcomes
finer than plain success/throw (at minimum: `warm_success`, `warm_failure`, `cold_success`,
`cold_failure`, `callback_success`, `callback_eval_error`, `cache_hit`) -- `@melitz_profile`
itself only distinguishes normal-return vs threw, which is not enough to see e.g. a warm
inner solve that returned a NUMERICAL failure (`nStatus` outside the accepted set) without
throwing at all.
"""
function melitz_record_seconds_outcome!(category::Symbol, outcome::Symbol, elapsed_s::Real)
    MELITZ_PROFILE[] || return nothing
    melitz_record!(Symbol(category, :_, outcome), Int64(round(elapsed_s * 1e9)))
    return nothing
end

function _percentile(sorted_samples::Vector{Int64}, p::Real)
    isempty(sorted_samples) && return NaN
    n = length(sorted_samples)
    idx = clamp(ceil(Int, p * n), 1, n)
    return sorted_samples[idx]
end

"""
    melitz_profile_summary() -> Vector{NamedTuple}

One row per recorded category: `category`, `count`, `total_s`, `mean_ms`, `median_ms`,
`p90_ms`, `max_ms`. Sorted by `total_s` descending (largest contributor first).
"""
function melitz_profile_summary()
    rows = NamedTuple[]
    for (cat, st) in MELITZ_PROF.stats
        st.count == 0 && continue
        sorted = sort(st.samples)
        push!(rows, (category=cat, count=st.count, total_s=st.total_ns / 1e9,
            mean_ms=(st.total_ns / st.count) / 1e6,
            median_ms=_percentile(sorted, 0.5) / 1e6,
            p90_ms=_percentile(sorted, 0.9) / 1e6,
            max_ms=st.max_ns / 1e6))
    end
    sort!(rows; by=r -> r.total_s, rev=true)
    return rows
end

"""
    melitz_profile_report(io::IO=stdout; trajectory_total_s=nothing)

Prints the Section 7 table: category, call count, total wall time, mean, median, p90,
max, and (if `trajectory_total_s` given) percentage of that total. Pass the actual
end-to-end wall time of the profiled trajectory/evaluation as `trajectory_total_s` so the
percentage column is meaningful (categories can double-count nested regions, e.g.
`:inner_solve_total` is part of `:evaluate_melitz_delta_total`, so percentages need not
sum to 100%).
"""
function melitz_profile_report(io::IO=stdout; trajectory_total_s::Union{Nothing,Real}=nothing)
    rows = melitz_profile_summary()
    if isempty(rows)
        println(io, "melitz_profile_report: no categories recorded (MELITZ_PROFILE[] was off, or nothing ran)")
        return rows
    end
    pct_header = trajectory_total_s === nothing ? "" : "  pct_of_total"
    println(io, rpad("category", 34), rpad("count", 8), rpad("total_s", 10),
        rpad("mean_ms", 10), rpad("median_ms", 11), rpad("p90_ms", 10), rpad("max_ms", 10), pct_header)
    for r in rows
        pct_str = trajectory_total_s === nothing ? "" :
                  string(round(100 * r.total_s / trajectory_total_s; digits=1), "%")
        println(io, rpad(string(r.category), 34), rpad(string(r.count), 8),
            rpad(string(round(r.total_s; digits=4)), 10),
            rpad(string(round(r.mean_ms; digits=3)), 10),
            rpad(string(round(r.median_ms; digits=3)), 11),
            rpad(string(round(r.p90_ms; digits=3)), 10),
            rpad(string(round(r.max_ms; digits=3)), 10), pct_str)
    end
    if trajectory_total_s !== nothing
        melitz_profile_print_residual(io, rows, trajectory_total_s)
    end
    return rows
end

"""
    melitz_profile_print_residual(io, rows, trajectory_total_s)

Section 3.2's decomposition: `total outer KNITRO wall` (= `trajectory_total_s`) vs.
`complete FC/GA callback wall` (= the sum of every TOP-LEVEL `:fc_total_*`/`:ga_total_*`
category -- ALL outcomes, since the callback wrapper's `try/finally` now records elapsed
time regardless of success/eval-error) vs. `residual KNITRO-C/API wall` (the difference).
Only this residual, computed AFTER exception-safe callback timing, may be described as
KNITRO-internal overhead -- callback time that went unrecorded merely because a call threw
is no longer part of it (the prior instrumentation gap this session's Section 3 closes).
"""
function melitz_profile_print_residual(io::IO, rows::Vector{<:NamedTuple}, trajectory_total_s::Real)
    fc_total = sum((r.total_s for r in rows if startswith(string(r.category), "fc_total")), init=0.0)
    ga_total = sum((r.total_s for r in rows if startswith(string(r.category), "ga_total")), init=0.0)
    callback_wall = fc_total + ga_total
    residual = trajectory_total_s - callback_wall
    pct = trajectory_total_s > 0 ? round(100 * residual / trajectory_total_s; digits=1) : NaN
    println(io, "\n-- Section 3.2 wall-time decomposition --")
    println(io, "total outer KNITRO wall        = ", round(trajectory_total_s; digits=4), "s")
    println(io, "complete FC/GA callback wall    = ", round(callback_wall; digits=4), "s (fc_total*=",
        round(fc_total; digits=4), "s, ga_total*=", round(ga_total; digits=4), "s)")
    println(io, "residual KNITRO-C/API wall      = ", round(residual; digits=4), "s (", pct, "% of trajectory)")
    return (trajectory_total_s=trajectory_total_s, callback_wall=callback_wall, residual=residual)
end
