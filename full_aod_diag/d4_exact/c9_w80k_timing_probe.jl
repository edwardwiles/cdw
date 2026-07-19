# Quick timing probe (NOT the full benchmark) -- how long does one
# evaluate_fullA / evaluate_fullA_fast / composite_gradient_at_fast call take
# at the real D=20/W=80000 point, so the full microbenchmark harness can be
# scoped sensibly (per task instruction: "find out empirically early").
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
using Printf, Dates

println("["); flush(stdout)
println("probe starting at ", now(), " nthreads=", Threads.nthreads()); flush(stdout)

t0 = time()
ctx = d20_real_setup(W = 80000)
t_setup = time() - t0
println("d20_real_setup(W=80000) wall = ", round(t_setup, digits=2), "s"); flush(stdout)

D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]

println("\n-- evaluate_fullA (dense/exact), natural theta, first call (COLD/JIT) --"); flush(stdout)
t0 = time()
r1 = evaluate_fullA(xf_nat, ctx; warm = false)
t1 = time() - t0
println("  wall=", round(t1, digits=2), "s  inner_status=", r1.inner_status, "  Delta_dual=", r1.Delta_dual); flush(stdout)

println("\n-- evaluate_fullA, second call (warmed JIT, still cold KNITRO state) --"); flush(stdout)
t0 = time()
r2 = evaluate_fullA(xf_nat, ctx; warm = false)
t2 = time() - t0
println("  wall=", round(t2, digits=2), "s  inner_status=", r2.inner_status, "  Delta_dual=", r2.Delta_dual); flush(stdout)

println("\n-- evaluate_fullA_fast dense, first call --"); flush(stdout)
t0 = time()
rf1, metaf1 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = false, moment_representation = :dense)
t3 = time() - t0
println("  wall=", round(t3, digits=2), "s  n_fg=", metaf1.n_fg_calls); flush(stdout)

println("\n-- evaluate_fullA_fast dense, second call --"); flush(stdout)
t0 = time()
rf2, metaf2 = evaluate_fullA_fast(xf_nat, ctx; cache = nothing, warm = false, moment_representation = :dense)
t4 = time() - t0
println("  wall=", round(t4, digits=2), "s  n_fg=", metaf2.n_fg_calls); flush(stdout)

println("\n-- solve_base_state, first call --"); flush(stdout)
t0 = time()
base0 = solve_base_state(xf_nat, ctx)
t5 = time() - t0
println("  wall=", round(t5, digits=2), "s"); flush(stdout)

println("\n-- build_pivot_elimination + composite_gradient_at_fast, ONE call (top3,threaded,adaptive) --"); flush(stdout)
t0 = time()
pe = build_pivot_elimination(ctx)
t_pe = time() - t0
println("  build_pivot_elimination wall=", round(t_pe, digits=2), "s"); flush(stdout)

t0 = time()
grad1 = composite_gradient_at_fast(xf_nat, ctx, pe; base = base0, threaded = true, h_mode = :adaptive, multi_method = :top3)
t6 = time() - t0
println("  composite_gradient_at_fast wall=", round(t6, digits=2), "s  length(grad)=", length(grad1)); flush(stdout)

println("\nSUMMARY (s): setup=", round(t_setup,digits=2), " evalFullA_cold1=", round(t1,digits=2),
        " evalFullA_cold2=", round(t2,digits=2), " evalFast1=", round(t3,digits=2), " evalFast2=", round(t4,digits=2),
        " base_state=", round(t5,digits=2), " pivot_elim=", round(t_pe,digits=2), " full_grad=", round(t6,digits=2))
println("]")
println("PROBE COMPLETE at ", now())
