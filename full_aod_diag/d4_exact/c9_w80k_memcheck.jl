# Minimal post-fix sanity probe (Continuation 9, Phase 2): build the D=20/W=80000
# context with the FIXED needs_outer_moment_jacobian=false default (commit bd313a2)
# and do ONE cold evaluate_fullA call, printing VmHWM/gc_live_bytes at each stage --
# purely to confirm the 109GB jac_h bug is actually gone before relaunching the full
# 4-point benchmark + thread sweep. Deliberately minimal, not the full harness.
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
using Dates

function vmhwm_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Int, split(line)[2])
    end
    return -1
end
function vmrss_kb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmRSS:") && return parse(Int, split(line)[2])
    end
    return -1
end

println("["); flush(stdout)
println("memcheck starting at ", now()); flush(stdout)
println("VmRSS at start = ", vmrss_kb(), " KB"); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
println("d20_real_setup(W=80000) wall = ", round(t_setup, digits=2), "s"); flush(stdout)
println("VmRSS after setup = ", vmrss_kb(), " KB = ", round(vmrss_kb()/1e6, digits=3), " GB"); flush(stdout)
println("VmHWM after setup = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)
println("gc_live_bytes after setup = ", round(Base.gc_live_bytes()/1e6, digits=1), " MB"); flush(stdout)

D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
t0 = time()
r = evaluate_fullA(xf_nat, ctx; warm = false)
t_eval = time() - t0
println("evaluate_fullA (cold) wall = ", round(t_eval, digits=2), "s  inner_status=", r.inner_status,
        "  Delta_dual=", r.Delta_dual); flush(stdout)
println("VmRSS after 1 eval = ", vmrss_kb(), " KB = ", round(vmrss_kb()/1e6, digits=3), " GB"); flush(stdout)
println("VmHWM after 1 eval = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)
println("]")
ok = vmhwm_kb() < 5_000_000   # sanity threshold: under 5GB (vs the old 109GB bug)
println(ok ? "MEMCHECK: PASS (VmHWM under 5GB)" : "MEMCHECK: FAIL (VmHWM still very large -- DO NOT PROCEED)")
println("MEMCHECK COMPLETE at ", now())
