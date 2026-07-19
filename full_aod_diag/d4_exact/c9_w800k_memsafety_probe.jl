include(joinpath(pwd(), "full_aod_diag/d4_exact/context_real_d20.jl"))
include(joinpath(pwd(), "full_aod_diag/d4_exact/winners.jl"))
include(joinpath(pwd(), "full_aod_diag/d4_exact/oracle.jl"))

println("Building real D=20 context at W=800000...")
t0 = time()
ctx = d20_real_setup(W = 800000)
println("  build wall: ", round(time() - t0, digits = 2), "s")
GC.gc()
println("  Julia live heap after build+gc: ", round(Base.gc_live_bytes()/1e9, digits=3), " GB")

D = ctx.D
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
println("Evaluating at natural theta (cold)...")
t1 = time()
r = evaluate_fullA(xf_nat, ctx; warm = false)
println("  eval wall: ", round(time() - t1, digits = 2), "s")
println("  inner_status=", r.inner_status, "  Delta_dual=", r.Delta_dual, "  gravity_raw=", r.gravity_raw)
GC.gc()
println("  Julia live heap after eval+gc: ", round(Base.gc_live_bytes()/1e9, digits=3), " GB")
println("DONE")
