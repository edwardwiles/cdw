# Continuation 9, Phase 3.1: minimal sanity probe for the :compressed moment
# representation (compressed_live.jl / oracle_fast.jl's moment_representation
# kwarg) against the REAL D=20 context, before any timed benchmark. Never
# tried against real data before this session (D=4-10 compressed tests all
# used synthetic d_exact_setup_scaled contexts). Mirrors c9_w80k_memcheck.jl's
# own minimal-first-then-scale discipline per the task's safety warning:
# build context, run ONE cheap compressed call, check VmHWM, only THEN
# consider a full benchmark.
include(joinpath(@__DIR__, "context_real_d20.jl"))   # -> includes context.jl exactly once
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using Dates, Printf

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
println("c9_phase3_compressed_d20_sanity.jl starting at ", now()); flush(stdout)
println("VmRSS at start = ", vmrss_kb(), " KB"); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
println("d20_real_setup(W=80000) wall = ", round(t_setup, digits=2), "s"); flush(stdout)
println("VmRSS after setup = ", vmrss_kb(), " KB = ", round(vmrss_kb()/1e6, digits=3), " GB"); flush(stdout)
println("VmHWM after setup = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)

D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
println("ctx.γ.indicators.gravMoment = ", ctx.γ.indicators.gravMoment); flush(stdout)

# ---- ONE cheap dense call first (already known-safe, per the W80k microbenchmark) ----
t0 = time()
r_dense = evaluate_fullA(xf_nat, ctx; warm = false)
t_dense = time() - t0
println("evaluate_fullA (dense, cold) wall = ", round(t_dense, digits=2), "s  inner_status=", r_dense.inner_status,
        "  Delta_dual=", r_dense.Delta_dual); flush(stdout)
println("VmHWM after dense eval = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)

# ---- ONE cheap compressed call, the actually-new thing this script exists to check ----
t0 = time()
local r_comp, meta_comp
try
    global r_comp, meta_comp
    r_comp, meta_comp = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = false, moment_representation = :compressed)
catch e
    println("COMPRESSED CALL THREW: ", sprint(showerror, e)); flush(stdout)
    println("VmHWM at throw = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)
    println("]")
    rethrow()
end
t_comp = time() - t0
println("evaluate_fullA_fast (compressed, cold) wall = ", round(t_comp, digits=2), "s  inner_status=", r_comp.inner_status,
        "  Delta_dual=", r_comp.Delta_dual, "  fallback_count=", COMPRESSED_FALLBACK_COUNT[]); flush(stdout)
println("VmRSS after compressed eval = ", vmrss_kb(), " KB = ", round(vmrss_kb()/1e6, digits=3), " GB"); flush(stdout)
println("VmHWM after compressed eval = ", vmhwm_kb(), " KB = ", round(vmhwm_kb()/1e6, digits=3), " GB"); flush(stdout)

# ---- correctness cross-check against dense at the SAME point ----
@printf("dense Delta_dual   = %.10f\n", r_dense.Delta_dual)
@printf("compressed Delta_dual = %.10f\n", r_comp.Delta_dual)
@printf("abs diff = %.3e\n", abs(r_dense.Delta_dual - r_comp.Delta_dual))
println("winner_hash match: ", r_dense.winner_hash == r_comp.winner_hash)
println("gravity_raw diff: ", abs(r_dense.gravity_raw - r_comp.gravity_raw))
println("max_abs_moment_resid diff: ", abs(r_dense.max_abs_moment_resid - r_comp.max_abs_moment_resid))

println("]")
ok = vmhwm_kb() < 10_000_000   # sanity threshold: under 10GB
println(ok ? "MEMCHECK: PASS (VmHWM under 10GB)" : "MEMCHECK: FAIL (VmHWM too large -- DO NOT SCALE UP)")
println("c9_phase3_compressed_d20_sanity.jl COMPLETE at ", now())
