# ============================================================================
# Smoke test (2026-07-16): after the recover_lfd nStatus fix, does seq_gravcol
# now correctly report ok=false at GC's known-bad T2/warm theta (previously
# silently accepted as feasible)?
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf

d = JLD2.load(joinpath(@__DIR__, "out_gc", "gc_T2_warm.jld2"))
θsol = Float64.(d["theta_star"])
@printf("Testing GC T2/warm theta (previously silently accepted: kappa=%.6f, div_p=%.6f)\n",
        d["kappa"], d["div_p"])

col, R, Rcol, umat, p, ok = seq_gravcol(θsol; δ=1.0, maxit=100, tol=5e-4)
@printf("\nPOST-FIX seq_gravcol(δ=1.0 budget): ok=%s\n", ok)
if ok
    println(">>> FIX DID NOT WORK: still reports ok=true. Needs further investigation.")
else
    println(">>> FIX CONFIRMED: seq_gravcol now correctly rejects this point as infeasible.")
end
println("\nSMOKE_TEST_RECOVER_LFD_FIX DONE")
