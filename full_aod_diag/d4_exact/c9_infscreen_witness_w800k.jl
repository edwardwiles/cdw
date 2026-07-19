# ============================================================================
# Continuation 9 (branch c9-infeasibility-screen): W=800,000 memory-safety
# probe for the extreme-draw witness structure (Section 3 of the spec) --
# and for the pairwise certificate / winner-scan screen itself.
# MEMORY/PREPROCESSING TIME ONLY, per the standing safety discipline (this
# investigation triggered a real server-wide memory alert earlier in
# Continuation 9 -- see docs/fullA_D20_W80k_microbenchmark.md sec 0). Every
# step below checks VmHWM before proceeding to the next, with an explicit
# self-imposed kill threshold well under this shared machine's capacity.
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
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
const KILL_GB = 40.0
function check_mem(label)
    v = vmhwm_kb() / 1e6
    println("[mem check] ", label, ": VmHWM = ", round(v, digits=2), " GB")
    flush(stdout)
    if v > KILL_GB
        error("SAFETY ABORT: VmHWM $(round(v,digits=2))GB exceeds self-imposed $(KILL_GB)GB kill threshold at step '$label'")
    end
    return v
end

println("c9_infscreen_witness_w800k.jl starting at ", now()); flush(stdout)
check_mem("startup")

t0 = time()
ctx = d20_real_setup(W = 800_000)
t_setup = time() - t0
check_mem("after d20_real_setup(W=800000)")
println("d20_real_setup(W=800000) wall=", round(t_setup, digits=2), "s"); flush(stdout)

D = ctx.D
xf_nat = ctx.θ0_up[ctx.free_idx]
Pmat = target_shares(ctx)

t0 = time()
pc = precompute_pairwise_M(ctx)
t_pc = time() - t0
check_mem("after precompute_pairwise_M")
println("precompute_pairwise_M(W=800000): wall=", round(t_pc, digits=2), "s"); flush(stdout)

θ_full = CS.reconstruct_full(xf_nat, ctx.m)
t0 = time()
a = compute_a_od(θ_full, ctx)
pres = pairwise_certificate(a, pc, Pmat)
t_pw = time() - t0
println("pairwise_certificate at calibration: infeasible=", pres.infeasible, "  worst_slack=", pres.worst_slack, "  wall=", round(t_pw*1000, digits=3), "ms"); flush(stdout)

order = order_destinations(pres, D)
t0 = time()
wres = screen_hard_winners(θ_full, ctx, Pmat; order = order)
t_ws = time() - t0
check_mem("after screen_hard_winners (feasible path, full D destinations)")
println("screen_hard_winners at calibration: feasible=", wres.feasible, "  wall=", round(t_ws, digits=2), "s"); flush(stdout)

# ---- extreme-draw witness structure: the actual memory-hungry piece ----
println("\nBuilding extreme_draw_witness at W=800000 (D*(D-1)=", D*(D-1), " sorted pairs of length W each)..."); flush(stdout)
mem_est_gb = D * (D - 1) * size(ctx.U, 1) * (4 + 8) / 1e9   # Int32 idx + Float64 val
println("theoretical structure memory estimate: ", round(mem_est_gb, digits=3), " GB"); flush(stdout)
if mem_est_gb > KILL_GB
    error("SAFETY ABORT: estimated witness structure memory $(round(mem_est_gb,digits=2))GB exceeds kill threshold BEFORE building -- not attempting")
end

t0 = time()
ew = build_extreme_draw_witness(ctx)
t_build = time() - t0
vmhwm_after = check_mem("after build_extreme_draw_witness(W=800000)")
println("build_extreme_draw_witness(W=800000): wall=", round(t_build, digits=2), "s"); flush(stdout)

# a handful of queries to confirm correctness + measure per-query time at this scale
Bmat = hard_score_B(ctx)
using Statistics
qtimes = Float64[]
n_probe = 200
probed = 0
for d in 1:D, o in 1:D
    Pmat[o,d] > 0 || continue
    global probed
    probed >= n_probe && break
    t0 = time()
    exists, s, ntested, csize = query_witness(o, d, a, Bmat, ew)
    push!(qtimes, time() - t0)
    probed += 1
end
println("\nquery_witness at W=800000: N=", length(qtimes), " probes, median=", round(median(qtimes)*1e6, digits=1),
        "us  mean=", round(mean(qtimes)*1e6, digits=1), "us  max=", round(maximum(qtimes)*1e6, digits=1), "us")
flush(stdout)

println("\n=== SUMMARY (W=800000) ===")
println("setup wall: ", round(t_setup,digits=1), "s")
println("pairwise_M build wall: ", round(t_pc,digits=2), "s")
println("witness build wall: ", round(t_build,digits=2), "s")
println("witness structure memory (measured VmHWM delta from before build): see above")
println("final VmHWM: ", round(vmhwm_after, digits=2), " GB")
println("c9_infscreen_witness_w800k.jl DONE at ", now())
