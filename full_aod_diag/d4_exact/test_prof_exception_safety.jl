# Regression test for the @prof exception-safety fix (section 8 of
# PRODUCTION_ALLOCATION_FIXES_PORT_REPORT_2026-07-25.md). Before the fix, a label's own
# time/allocation was silently dropped whenever the wrapped expression threw (e.g. a KNITRO
# callback rejecting a trial point via `reject_point`/error), even though completed nested
# @prof sub-calls still recorded theirs. Pure Julia, no KNITRO/context needed.
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_prof_exception_safety.jl
include(joinpath(@__DIR__, "instrumentation.jl"))

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Section 1: normal (non-throwing) @prof still records + returns value")
println("="^78)
prof_reset!()
v = @prof "ok_label" begin
    1 + 1
end
check("returns expr value", v == 2)
check("records one sample", get(PROF_COUNTS, "ok_label", 0) == 1)

println("="^78)
println("Section 2: throwing expr still records its OWN label, then rethrows")
println("="^78)
prof_reset!()
threw = false
try
    @prof "throwing_label" begin
        error("simulated KNITRO callback rejection")
    end
catch e
    global threw = true
    check("original exception propagates", e isa ErrorException && occursin("simulated", e.msg))
end
check("exception was thrown out of the macro", threw)
check("throwing label recorded exactly once (finally ran)", get(PROF_COUNTS, "throwing_label", 0) == 1)
check("throwing label has finite recorded time", isfinite(PROF_TIMES["throwing_label"][1]))

println("="^78)
println("Section 3: nested @prof -- child completes, then parent throws")
println("="^78)
prof_reset!()
try
    @prof "parent_label" begin
        @prof "child_label" begin
            42
        end
        error("parent throws after child completed")
    end
catch
end
check("child recorded (completed before parent's throw)", get(PROF_COUNTS, "child_label", 0) == 1)
check("parent ALSO recorded despite throwing (the actual bug fixed here)", get(PROF_COUNTS, "parent_label", 0) == 1)

println("="^78)
println("Section 4: PROF_ENABLED[]=false path still propagates exceptions untouched")
println("="^78)
prof_reset!()
PROF_ENABLED[] = false
threw2 = false
try
    @prof "disabled_label" error("disabled-path exception")
catch
    global threw2 = true
end
check("exception still propagates when profiling disabled", threw2)
check("no record made when disabled", get(PROF_COUNTS, "disabled_label", 0) == 0)
PROF_ENABLED[] = true

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
