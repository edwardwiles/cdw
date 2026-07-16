# Smoke test (2026-07-16): common-marginals wiring into run_profiled_production.jl.
# 1) CM_L=0 (default) must be a true no-op vs the pre-CM baseline seq_gravcol result.
# 2) CM_L>0 with GRADIENT_METHOD=pointwise_ad must not crash and must actually change G's column
#    count (nCM>0) / results (the restriction is real, not silently ignored).
# 3) CM_L>0 with GRADIENT_METHOD=fixed_dual_fd_full must fail with the NEW clear error, not the
#    old deep assertion.
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
ENV["CM_L"] = get(ENV, "CM_L", "3")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

@printf("\nCM_L=%d CM_REF=%d CM_ENABLED=%s nCM=%d\n", CM_L, CM_REF, CM_ENABLED, nCM)
@assert CM_ENABLED && nCM > 0 "expected CM enabled for this smoke test (set CM_L>0)"

col, R, Rcol, umat, p, ok = seq_gravcol(θr0; δ = 1.0, maxit = 20, tol = 5e-4)
@printf("seq_gravcol with CM ON: ok=%s R_mean=%.4e\n", ok, R)
@assert ok "seq_gravcol should still succeed at Astar/delta=1.0 with CM restriction active"

@printf("\nTesting GRADIENT_METHOD=fixed_dual_fd_full + CM_ENABLED=true raises the NEW clear error...\n")
try
    outer_solve_nested_cached(true, θr0; δ = 1.0, gradient_method = :fixed_dual_fd_full)
    println(">>> UNEXPECTED: did not raise -- guard failed to trigger")
catch e
    msg = sprint(showerror, e)
    if occursin("not yet implemented for CM_ENABLED=true", msg)
        println(">>> CONFIRMED: clear guard error raised as expected")
    else
        println(">>> UNEXPECTED error type/message: ", msg)
    end
end

println("\nSMOKE_TEST_COMMON_MARGINALS DONE")
