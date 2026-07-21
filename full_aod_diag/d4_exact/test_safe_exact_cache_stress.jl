# ============================================================================
# Synthetic concurrency stress test for SafeExactCache (oracle.jl), no KNITRO
# involved. Deliberately kept as a SEPARATE standalone script/process from
# test_safe_exact_cache.jl's real-KNITRO checks -- running this same stress
# loop AFTER several real KN_new()/KN_solve() calls in one process silently
# kills the Julia process (exit code 1, no error/stacktrace printed) on this
# machine, while it passes cleanly (100/100 trials, exit 0) as its own
# process. That silent kill is itself a real, worth-noting interaction
# between KNITRO's own internal threading/state and a later plain
# Threads.@threads use in the SAME process -- orthogonal to SafeExactCache's
# own correctness (which is exactly what this isolated script verifies), and
# exactly why diag/fullA-d20-warmstart-replay's own cache_threadsafety_test.jl
# already ran its raw-vs-safe comparison in separate processes ("a real Dict
# data race can hard-crash the Julia process (segfault), not just throw a
# catchable exception").
#
# Adapted from that same script's stress_same_key/stress_distinct_keys,
# calling through THIS branch's actual _cache_lookup/_cache_store! dispatch
# (oracle.jl) instead of a raw Dict, to verify the lock mechanism itself
# never corrupts data or crashes.
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_safe_exact_cache_stress.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
using Base.Threads

println("Threads.nthreads()=", Threads.nthreads())
Threads.nthreads() < 4 && @warn "Fewer than 4 threads available -- race conditions may not manifest reliably."

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

function stress_same_key(cache, key, value, n_threads::Int)
    bad = Threads.Atomic{Int}(0); crashed = Threads.Atomic{Int}(0)
    Threads.@threads for t in 1:n_threads
        try
            hit = _cache_lookup(cache, key)
            if hit === nothing
                _cache_store!(cache, key, value)
            else
                (hit.x_free == value.x_free && hit.tag == value.tag) || Threads.atomic_add!(bad, 1)
            end
        catch e
            Threads.atomic_add!(crashed, 1)
        end
    end
    return (n_bad = bad[], n_crashed = crashed[])
end
function stress_distinct_keys(cache, n_keys::Int)
    crashed = Threads.Atomic{Int}(0)
    keys_used = Vector{FullAEvalKey}(undef, n_keys); vals_used = Vector{NamedTuple}(undef, n_keys)
    Threads.@threads for i in 1:n_keys
        try
            k = FullAEvalKey(fill(Float64(i), 5), 5.0, false, "ek_inner.opt", :hard)
            v = (x_free = fill(Float64(i), 5), tag = "k$i", cache_hit = false, inner_status = 0)
            keys_used[i] = k; vals_used[i] = v
            _cache_store!(cache, k, v)
        catch e
            Threads.atomic_add!(crashed, 1)
        end
    end
    lost = 0; wrong = 0
    for i in 1:n_keys
        hit = _cache_lookup(cache, keys_used[i])
        hit === nothing ? (lost += 1) : (hit.tag != vals_used[i].tag && (wrong += 1))
    end
    return (final_len = length(cache), lost = lost, wrong = wrong, crashed = crashed[])
end

n_trials = 500
n_threads = min(Threads.nthreads(), 20)
println("Running $n_trials trials x $n_threads threads-worth-of-tasks per trial...")
same_bad_total = 0; same_crash_total = 0; lost_total = 0; wrong_total = 0; distinct_crash_total = 0
for trial in 1:n_trials
    global same_bad_total, same_crash_total, lost_total, wrong_total, distinct_crash_total
    c1 = SafeExactCache()
    key = FullAEvalKey([1.0, 2.0, 3.0], 5.0, false, "ek_inner.opt", :hard)
    value = (x_free = [1.0, 2.0, 3.0], tag = "same", cache_hit = false, inner_status = 0)
    r1 = stress_same_key(c1, key, value, n_threads)
    same_bad_total += r1.n_bad; same_crash_total += r1.n_crashed

    c2 = SafeExactCache()
    r2 = stress_distinct_keys(c2, n_threads)
    lost_total += r2.lost; wrong_total += r2.wrong; distinct_crash_total += r2.crashed
    if trial % 100 == 0
        println("  ...trial $trial/$n_trials done")
        flush(stdout)
    end
end
println("same-key stress ($n_trials trials x $n_threads threads): bad=", same_bad_total, " crashed=", same_crash_total)
println("distinct-key stress: lost=", lost_total, " wrong=", wrong_total, " crashed=", distinct_crash_total,
        " (of ", n_trials * n_threads, " total insertions)")
check("same-key stress: zero corrupted/mismatched hits across $n_trials trials", same_bad_total == 0)
check("same-key stress: zero crashes across $n_trials trials", same_crash_total == 0)
check("distinct-key stress: zero lost keys across $n_trials trials", lost_total == 0)
check("distinct-key stress: zero wrong values across $n_trials trials", wrong_total == 0)
check("distinct-key stress: zero crashes across $n_trials trials", distinct_crash_total == 0)

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")
