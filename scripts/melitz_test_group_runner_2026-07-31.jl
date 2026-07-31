#!/usr/bin/env julia
# Production-consolidation Phase 5 (2026-07-31): independent per-group Melitz test runner.
#
# PROBLEM: `test/melitz/runtests.jl` is a single flat sequence of top-level `@testset "..."
# begin ... end` blocks (plus some top-level `if KNITRO_AVAILABLE ... end` blocks wrapping one
# or more testsets, and shared setup code -- includes, FIXTURE/LAYOUT construction -- between
# them). Julia's `Test` stdlib tracks testset NESTING DEPTH, not lexical/control-flow nesting:
# a `@testset` that is not itself lexically inside another `@testset` is "top-level" in Test's
# own bookkeeping regardless of whether it happens to sit inside an `if` block. When such a
# top-level testset finishes with ANY failure or error recorded inside it, `Test.finish`
# throws a `Test.TestSetException` -- and since nothing at the file's own top level catches
# that, it propagates straight out of `include("test/melitz/runtests.jl")` and aborts the
# ENTIRE remaining file. This is exactly the behavior this consolidation's governing prompt
# describes ("one failing top-level testset does not prevent the rest from running" was NOT
# true of the file as originally structured) and exactly what both prior audit sessions'
# regression logs independently observed (each stopped at the first failing testset, never
# reaching the end of the file).
#
# FIX (this script, not a rewrite of runtests.jl itself): rather than manually wrapping
# every one of dozens of top-level blocks in the 7000+-line test file in try/catch (invasive,
# easy to get subtly wrong, and would have to be kept in sync with every future testset added
# to that file), this driver:
#   1. Reads runtests.jl's own source text.
#   2. Splits it into top-level Julia forms via `Meta.parse` (the same unit Julia's own
#      top-level evaluator would execute one at a time).
#   3. Evaluates each top-level form in Main, ONE AT A TIME, inside its own try/catch.
#   4. Records PASS / FAIL / ERROR for each form that is (or contains, per the `if
#      KNITRO_AVAILABLE ... end` caveat below) a `@testset` -- and, critically, continues to
#      the NEXT top-level form regardless of whether the current one raised a
#      `Test.TestSetException` or any other exception.
#   5. Prints a final group-by-group matrix and a machine-readable summary line.
#
# GRANULARITY CAVEAT (documented honestly, not glossed over): this operates at the
# granularity of TOP-LEVEL FORMS in the file, not at the granularity of every individual
# `@testset`. The large majority of top-level forms in runtests.jl are exactly one
# `@testset "name" begin...end` each (the common case, and the one this matters for). A
# handful of top-level forms are `if KNITRO_AVAILABLE ... end` blocks that lexically contain
# ONE OR MORE `@testset`s -- for those, if the first nested testset inside such an `if` block
# fails, any SIBLING testsets nested inside the SAME `if` block (after the failing one, before
# the `if`'s own `end`) will NOT run this pass (the exception aborts the whole top-level `if`
# form, same as it would have aborted the whole file before this fix) -- but every subsequent
# TOP-LEVEL form (i.e. every other testset in the file, which is the overwhelming majority)
# still runs. This is a real, disclosed limitation, not a claimed complete fix -- see the
# consolidation doc's own group matrix for which specific groups this affects.
#
# USAGE: julia --project=. scripts/melitz_test_group_runner_2026-07-31.jl
#   (respects the same -t thread count and env vars as a normal test run)

using Test
using Dates

const RUNTESTS_PATH = joinpath(dirname(dirname(@__FILE__)), "test", "melitz", "runtests.jl")
@assert isfile(RUNTESTS_PATH) "expected test/melitz/runtests.jl at $RUNTESTS_PATH"
const SRC = read(RUNTESTS_PATH, String)

function toplevel_forms(src::String)
    forms = Expr[]
    pos = 1
    n = lastindex(src)
    while pos <= n
        # `filename=RUNTESTS_PATH` is essential: this is what makes `@__FILE__`/`@__DIR__`
        # (used throughout runtests.jl to build paths like
        # `joinpath(@__DIR__, "..", "..", "src", "melitz")`) resolve to test/melitz/'s own
        # real location, not this runner script's location under scripts/ -- without it,
        # every relative-path `include`/`joinpath(@__DIR__, ...)` in the test file silently
        # resolves to the WRONG directory (confirmed live: first attempt at this runner
        # produced a wall of `SystemError: opening file ".../src/melitz/profiling.jl"` one
        # directory too shallow, because @__DIR__ pointed at scripts/ instead of test/melitz/).
        ex, newpos = Meta.parse(src, pos; raise=false, filename=RUNTESTS_PATH)
        if ex === nothing
            break
        end
        if isa(ex, Expr) && ex.head == :error
            # Something Meta.parse itself couldn't handle at this position (rare, e.g. an
            # incomplete trailing fragment) -- stop; whatever was already collected is used.
            @warn "Meta.parse stopped early" pos ex
            break
        end
        push!(forms, Expr(:block, ex))  # wrap so `nothing`/literals are handled uniformly
        pos = newpos
    end
    return forms
end

# Best-effort human label for a top-level form: the first `@testset` name found in it
# (there may be more than one, e.g. inside an `if` block -- reported name is the FIRST).
function form_label(ex::Expr)
    label = nothing
    function walk(e)
        label !== nothing && return
        if isa(e, Expr)
            if e.head == :macrocall && length(e.args) >= 1 && e.args[1] === Symbol("@testset")
                for a in e.args
                    if isa(a, String)
                        label = a
                        return
                    end
                end
            end
            for a in e.args
                walk(a)
            end
        end
    end
    walk(ex)
    return label
end

function contains_testset(ex::Expr)
    found = false
    function walk(e)
        found && return
        if isa(e, Expr)
            if e.head == :macrocall && length(e.args) >= 1 && e.args[1] === Symbol("@testset")
                found = true
                return
            end
            for a in e.args
                walk(a)
            end
        end
    end
    walk(ex)
    return found
end

struct GroupResult
    idx::Int
    label::String
    status::Symbol   # :pass, :fail, :error, :skipped_no_testset
    detail::String
    elapsed::Float64
end

forms = toplevel_forms(SRC)
println("Parsed $(length(forms)) top-level forms from $RUNTESTS_PATH")
flush(stdout)

results = GroupResult[]
group_counter = 0

for (i, form) in enumerate(forms)
    has_ts = contains_testset(form)
    label_raw = form_label(form)
    if !has_ts
        # Setup code (includes, FIXTURE/LAYOUT construction, helper function defs, etc.) --
        # must succeed for later groups to have valid shared state. Evaluated directly
        # (still under try/catch so a failure here is REPORTED, not a silent process crash),
        # but not counted as a "test group" in the matrix.
        t0 = time()
        try
            Core.eval(Main, form)
        catch e
            io = IOBuffer(); showerror(io, e); msg = String(take!(io))
            println("!!! SETUP FORM $i FAILED (non-testset top-level code): ", msg)
            push!(results, GroupResult(i, "(setup code, form $i)", :error, msg, time() - t0))
        end
        continue
    end
    global group_counter += 1
    label = something(label_raw, "(unnamed testset group, form $i)")
    t0 = time()
    status = :pass
    detail = ""
    try
        Core.eval(Main, form)
    catch e
        if isa(e, Test.TestSetException)
            status = :fail
            detail = "TestSetException: $(e.pass) pass / $(e.fail) fail / $(e.error) error / $(e.broken) broken"
        else
            status = :error
            io = IOBuffer(); showerror(io, e); detail = String(take!(io))
        end
    end
    elapsed = time() - t0
    push!(results, GroupResult(i, label, status, detail, elapsed))
    println(rpad("[$status]", 10), " (", round(elapsed; digits=1), "s)  ", label)
    flush(stdout)
end

println()
println("=" ^ 100)
println("MELITZ TEST GROUP MATRIX -- $(now())")
println("=" ^ 100)
npass = count(r -> r.status == :pass, results)
nfail = count(r -> r.status == :fail, results)
nerr = count(r -> r.status == :error, results)
for r in results
    println(rpad("[$(r.status)]", 10), " form ", lpad(r.idx, 5), "  ", r.label)
    if r.status != :pass
        for line in split(r.detail, '\n')[1:min(3, end)]
            println("            ", line)
        end
    end
end
println("-" ^ 100)
println("TOTAL testset-bearing top-level forms: $(npass + nfail + nerr)   PASS=$npass  FAIL=$nfail  ERROR=$nerr")
println("=" ^ 100)

# Machine-readable one-line summary for scripting/doc-generation.
open(joinpath(dirname(@__FILE__), "..", "docs", "key_results", "melitz_test_group_matrix_2026-07-31.txt"), "w") do io
    println(io, "generated: ", now())
    println(io, "julia: ", VERSION, "  nthreads: ", Threads.nthreads())
    for r in results
        println(io, r.status, "\t", r.idx, "\t", replace(r.label, "\t" => " "), "\t", round(r.elapsed; digits=2), "s", r.status == :pass ? "" : "\t" * first(split(r.detail, '\n')))
    end
    println(io, "SUMMARY\tpass=$npass\tfail=$nfail\terror=$nerr")
end

exit(nfail + nerr > 0 ? 1 : 0)
