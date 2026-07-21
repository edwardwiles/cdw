# Diagnostic branch diag/fullA-inner-blas-threading, Addendum sec 2: explicit runtime
# mutual-exclusion guard between "an inner KNITRO solve is running" and "the coordinate probe
# pool is running" -- makes the intended sequencing (base inner solve completes FULLY, then and
# only then the Threads.@threads coordinate sweep starts, never overlapping) an ENFORCED
# invariant instead of an implicit assumption, per the addendum's explicit request.
#
# Near-zero overhead when GUARD_ENABLED[] is true (two Bool reads/writes per inner solve call,
# two more per coordinate-pool entry/exit) -- same cost profile as this repo's own
# instrumentation.jl PROF_ENABLED[] pattern, reused deliberately rather than inventing a new one.
#
# NOT a lock -- a single-process, single-outer-loop invariant checker. This repo's inner solves
# and coordinate probes are already single-threaded-outer/KNITRO-concurrent_evals=0 by design
# (docs/fullA_inner_parallelism_audit.md sec 2.4); this guard exists to CATCH a violation of that
# design (e.g. a future caller that tries to prefetch/parallelize across outer points, or a bug
# that calls an inner solve from inside a coordinate probe) rather than to coordinate real
# concurrent access.
const GUARD_ENABLED = Ref(true)
const INNER_SOLVE_ACTIVE = Ref(false)
const COORD_POOL_ACTIVE = Ref(false)
const GUARD_VIOLATIONS = Ref(0)

"Call at the very start of any real inner-KNITRO-solve entry point (inner_loop_KNITRO_archgeneric,
inner_loop_KNITRO, inner_loop_KNITRO_compressed, ...), before KN_new. Errors if the coordinate
probe pool is currently active -- that would mean a probe (or something running concurrently with
the pool) launched an inner solve, which the addendum explicitly says must never happen."
function guard_enter_inner_solve!()
    GUARD_ENABLED[] || return nothing
    if COORD_POOL_ACTIVE[]
        GUARD_VIOLATIONS[] += 1
        error("parallelism_guards: an inner KNITRO solve was launched while the coordinate probe pool was active -- this violates the required base-solve-then-probes sequencing (addendum sec 2). See GUARD_VIOLATIONS[] / docs/fullA_inner_blas_threading_report.md sec on gradient sequencing.")
    end
    INNER_SOLVE_ACTIVE[] = true
    return nothing
end

"Call after KN_free / at every return path of an inner-solve entry point (use try/finally at the
call site, not inside this function, so a thrown error from KN_solve itself still clears the flag)."
function guard_exit_inner_solve!()
    GUARD_ENABLED[] || return nothing
    INNER_SOLVE_ACTIVE[] = false
    return nothing
end

"Call on the calling (non-parallel) thread immediately before a Threads.@threads coordinate-probe
loop starts. Errors if an inner solve is currently marked active -- that would mean the 'finish the
base solve completely, THEN dispatch probes' ordering was violated (e.g. by an async/Task-based
caller that didn't actually wait for the base solve)."
function guard_enter_coord_pool!()
    GUARD_ENABLED[] || return nothing
    if INNER_SOLVE_ACTIVE[]
        GUARD_VIOLATIONS[] += 1
        error("parallelism_guards: the coordinate probe pool was dispatched while an inner KNITRO solve was still active -- the base inner solve must fully complete (KN_free returned) before any coordinate probe launches (addendum sec 2).")
    end
    COORD_POOL_ACTIVE[] = true
    return nothing
end

"Call immediately after a Threads.@threads coordinate-probe loop returns (the loop itself is a
barrier -- all spawned tasks have already joined by the time control returns here)."
function guard_exit_coord_pool!()
    GUARD_ENABLED[] || return nothing
    COORD_POOL_ACTIVE[] = false
    return nothing
end

"Reset counters/flags between independent test runs."
function guard_reset!()
    INNER_SOLVE_ACTIVE[] = false
    COORD_POOL_ACTIVE[] = false
    GUARD_VIOLATIONS[] = 0
    return nothing
end
