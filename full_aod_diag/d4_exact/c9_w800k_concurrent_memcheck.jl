# ============================================================================
# Continuation 9, Phase 9: two-branch (upper+lower) concurrency check at
# W=800,000. Confirms how many independent W=800,000 contexts can run
# concurrently given the single-context peak (~16-20GB per
# c9_w800k_memsafety_probe.jl / c9_w800k_microbenchmark.jl) -- per the
# standing safety discipline, this script is run ONLY after single-context
# numbers are known, and tests 2 concurrent contexts (not jumping straight to
# a larger number).
#
# Single-process script; run TWICE concurrently as two separate OS processes
# (see the launcher shell commands in the task write-up) so combined RSS is
# externally observable via /proc/<pid>/status for BOTH pids, not just
# self-reported VmHWM (which only sees its own process).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
using Dates

function vmhwm_kb()
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
        end
    catch
    end
    return -1
end
gb(kb) = round(kb / 1e6, digits = 2)

const TAG = get(ENV, "C9_CONCURRENT_TAG", "A")
println("[$TAG] c9_w800k_concurrent_memcheck.jl starting at ", now(), " pid=", getpid()); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 800000)
t_setup = time() - t0
println("[$TAG] setup wall=", round(t_setup, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
t0 = time()
r = evaluate_fullA(xf_nat, ctx; warm = false)
t_eval = time() - t0
println("[$TAG] cold eval wall=", round(t_eval, digits=2), "s  inner_status=", r.inner_status,
        "  Delta_dual=", r.Delta_dual, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

t0 = time()
r2 = evaluate_fullA(xf_nat, ctx; warm = true)
t_eval2 = time() - t0
println("[$TAG] warm eval wall=", round(t_eval2, digits=2), "s  inner_status=", r2.inner_status,
        "  Delta_dual=", r2.Delta_dual, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("[$TAG] FINAL VmHWM=", gb(vmhwm_kb()), "GB  pid=", getpid())
println("[$TAG] DONE at ", now())
