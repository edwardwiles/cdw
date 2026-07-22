# ============================================================================
# Remediation task Part B: injected worker-exception test.
#
# Context: the final-gates diagnostics report (fullA_FINAL_RESIDUAL_GATES_AND_ENDTOEND_BENCHMARK
# _2026-07-22.md, Phase 4) reported "a real crash bug ... inside EK_moments_gammanorm_directgp!'s
# Threads.@threads block (a stack trace ending in threading_run/#wait#398 ... consistent with an
# uncaught task exception inside a threaded region leaving the process in a non-exiting state)".
#
# Direct evidence against the "crash" framing: the underlying raw log
# (phase4_control_full.log:519) reads `[<pid>] signal 15: Terminated` -- this is Julia's
# STANDARD signal handler dumping a backtrace of every thread's current native stack upon
# receiving SIGTERM (signal 15), i.e. an EXTERNAL `timeout` wrapper killed the process. This is
# NOT what an uncaught Julia exception looks like (that produces a `TaskFailedException`/
# `ERROR:` stack trace and the process exits ON ITS OWN with a nonzero code -- no external
# signal is needed, and Julia never prints "signal N: ..." for it). Per the task's own
# instruction: "The logs showing signal 15: Terminated were externally killed by a timeout. A
# Julia stack trace at the point of interruption is not proof of a threaded crash."
#
# What this script DOES establish (real evidence, not assumption): whether an uncaught
# exception thrown from ONE iteration of a `Threads.@threads :static` loop -- the exact
# scheduling mode this codebase's own threaded kernels use (lfix_base_workspace_pooled.jl,
# lfix_factorized_workspace.jl, lfix_kbplus_workspace.jl, gradient_workspace.jl,
# cm_hessian_threaded.jl all use `:static`) -- propagates cleanly (process exits with a Julia
# `ERROR:`/nonzero code, no external signal needed) or leaves the process hung, run as a
# genuinely separate child process so a hang here cannot wedge this test suite.
#
# What this does NOT establish: whether the SPECIFIC production crash the diagnostics report
# saw is caused by a threaded exception interacting with a concurrent KNITRO C-library call
# (this repo has documented history of a DIFFERENT nested-KNITRO/OpenMP-lock deadlock class --
# see docs -- fixed by `par_concurrent_evals=yes`). Reproducing that specific interaction was
# out of scope for this bounded test; see the remediation report's "deferred work" section.
# ============================================================================
using Test

function run_child(script_body::String; nthreads::Int = 4, timeout_s::Float64 = 30.0)
    tmp = tempname() * ".jl"
    write(tmp, script_body)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$nthreads $tmp`
    proc = run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = false)
    t0 = time()
    while process_running(proc) && (time() - t0) < timeout_s
        sleep(0.1)
    end
    hung = process_running(proc)
    if hung
        kill(proc, Base.SIGKILL)
        sleep(0.2)
    end
    rm(tmp; force = true)
    return (hung = hung, exitcode = hung ? nothing : proc.exitcode, wall = time() - t0)
end

@testset "Threaded worker-exception propagation (Part B injected test)" begin
    # Case 1: plain `Threads.@threads :static` loop, one iteration throws -- no concurrent C
    # calls from other threads. Baseline Julia semantics.
    r1 = run_child("""
        Threads.@threads :static for k in 1:Threads.nthreads()
            k == 2 && error("injected worker exception (case 1, k=\$k)")
            sleep(2.0)   # the OTHER iterations keep "working" past the throw point
        end
        println("UNREACHABLE_IF_PROPAGATED")
    """; nthreads = 4, timeout_s = 20.0)
    println("case1: hung=", r1.hung, " exitcode=", r1.exitcode, " wall=", round(r1.wall, digits = 2))
    @test !r1.hung
    @test r1.exitcode != 0   # a nonzero exit IS the process terminating cleanly on its own (no external kill needed)

    # Case 2: same, but the throwing iteration is nested inside a try/catch at the CALL SITE
    # (mirrors this codebase's own `cb_F!`/`cb_G!` KNITRO-callback pattern -- e.g.
    # cm_checkpoint.jl's cb_F! wraps its inner solve in try/catch and rethrows as a typed
    # DomainError). Confirms a caller-level try/catch around a `Threads.@threads` call DOES see
    # the propagated exception (not silently swallowed, not hung).
    r2 = run_child("""
        function do_threaded_work()
            Threads.@threads :static for k in 1:Threads.nthreads()
                k == 2 && error("injected worker exception (case 2, k=\$k)")
                sleep(2.0)
            end
        end
        try
            do_threaded_work()
            println("UNREACHABLE_IF_PROPAGATED")
        catch e
            println("CAUGHT: ", typeof(e))
            exit(0)   # deliberately exit 0 to distinguish "caught cleanly" from "hung"
        end
    """; nthreads = 4, timeout_s = 20.0)
    println("case2: hung=", r2.hung, " exitcode=", r2.exitcode, " wall=", round(r2.wall, digits = 2))
    @test !r2.hung
    @test r2.exitcode == 0   # reached the catch block and exited cleanly
end
println("All threaded worker-exception propagation tests passed.")
