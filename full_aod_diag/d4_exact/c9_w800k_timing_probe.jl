# ============================================================================
# Continuation 9, Phase 2 (W=800,000 half, resumed): quick timing probe to
# scope the full W=800k microbenchmark's rep counts sensibly, mirroring
# c9_w80k_timing_probe.jl's own role for the W=80k benchmark. NEW vs that
# probe: also times :compressed mode (never timed at W=800,000 before) and
# tracks VmHWM after every step, per the standing safety discipline (small
# sanity check before every new code path at W=800,000).
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
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using Printf, Dates

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

println("["); flush(stdout)
println("probe starting at ", now(), " nthreads=", Threads.nthreads()); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 800000)
t_setup = time() - t0
println("d20_real_setup(W=800000) wall = ", round(t_setup, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]

println("\n-- evaluate_fullA (dense/exact), natural theta, first call (COLD/JIT) --"); flush(stdout)
t0 = time()
r1 = evaluate_fullA(xf_nat, ctx; warm = false)
t1 = time() - t0
println("  wall=", round(t1, digits=2), "s  inner_status=", r1.inner_status, "  Delta_dual=", r1.Delta_dual, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- evaluate_fullA, second call, warm=true (warm-start sanity) --"); flush(stdout)
t0 = time()
r2 = evaluate_fullA(xf_nat, ctx; warm = true)
t2 = time() - t0
println("  wall=", round(t2, digits=2), "s  inner_status=", r2.inner_status, "  Delta_dual=", r2.Delta_dual, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- evaluate_fullA_fast DENSE, first call (warm=true) --"); flush(stdout)
t0 = time()
rf1, metaf1 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
t3 = time() - t0
println("  wall=", round(t3, digits=2), "s  n_fg=", metaf1.n_fg_calls, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- evaluate_fullA_fast DENSE, second call (warm=true) --"); flush(stdout)
t0 = time()
rf2, metaf2 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :dense)
t4 = time() - t0
println("  wall=", round(t4, digits=2), "s  n_fg=", metaf2.n_fg_calls, "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- evaluate_fullA_fast COMPRESSED, first call (warm=true) -- never timed at W=800,000 before --"); flush(stdout)
t0 = time()
rc1, metac1 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
t3c = time() - t0
println("  wall=", round(t3c, digits=2), "s  fallback_count=", COMPRESSED_FALLBACK_COUNT[], "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)
println("  |Delta_dual diff dense vs compressed| = ", abs(rf2.Delta_dual - rc1.Delta_dual)); flush(stdout)

println("\n-- evaluate_fullA_fast COMPRESSED, second call (warm=true) --"); flush(stdout)
t0 = time()
rc2, metac2 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = true, moment_representation = :compressed)
t4c = time() - t0
println("  wall=", round(t4c, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- solve_base_state, first call --"); flush(stdout)
t0 = time()
base0 = solve_base_state(xf_nat, ctx)
t5 = time() - t0
println("  wall=", round(t5, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- build_pivot_elimination + composite_gradient_at_fast, ONE call (top3,threaded,adaptive) --"); flush(stdout)
t0 = time()
pe = build_pivot_elimination(ctx)
t_pe = time() - t0
println("  build_pivot_elimination wall=", round(t_pe, digits=2), "s"); flush(stdout)

t0 = time()
grad1 = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :adaptive, multi_method = :top3)
t6 = time() - t0
println("  composite_gradient_at_fast (adaptive) wall=", round(t6, digits=2), "s  length(grad)=", length(grad1), "  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\n-- composite_gradient_at_fast, ONE call (top3,threaded,CACHED via BandwidthCachePolicy) --"); flush(stdout)
policy = BandwidthCachePolicy()
z0 = log.(reshape(xf_nat[2:end], D, D)); w0 = vcat(xf_nat[1], pivot_reduce(z0, pe))
maybe_invalidate!(policy, w0)   # init anchor
t0 = time()
grad2 = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache, multi_method = :top3)
t7 = time() - t0
println("  composite_gradient_at_fast (cached, COLD dict) wall=", round(t7, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)
t0 = time()
grad3 = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :cached, bandwidth_cache = policy.cache, multi_method = :top3)
t8 = time() - t0
println("  composite_gradient_at_fast (cached, WARM dict) wall=", round(t8, digits=2), "s  VmHWM=", gb(vmhwm_kb()), "GB"); flush(stdout)

println("\nSUMMARY (s): setup=", round(t_setup,digits=2), " evalFullA_cold=", round(t1,digits=2),
        " evalFullA_warm=", round(t2,digits=2), " evalFastDense1=", round(t3,digits=2), " evalFastDense2=", round(t4,digits=2),
        " evalFastCompressed1=", round(t3c,digits=2), " evalFastCompressed2=", round(t4c,digits=2),
        " base_state=", round(t5,digits=2), " pivot_elim=", round(t_pe,digits=2),
        " full_grad_adaptive=", round(t6,digits=2), " full_grad_cached_cold=", round(t7,digits=2), " full_grad_cached_warm=", round(t8,digits=2))
println("FINAL VmHWM = ", gb(vmhwm_kb()), " GB")
println("]")
println("PROBE COMPLETE at ", now())
