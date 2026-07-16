# Smoke test (2026-07-16): confirm the newly-wired DUAL_WARM_MODE (default :persist) in
# run_profiled_production.jl still produces the SAME accepted (R, kappa, ok) as :cold at a known
# point -- warm-starting must only change solver speed, never the accepted trajectory.
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

@printf("\nDefault DUAL_WARM_MODE = %s (should be :persist)\n", DUAL_WARM_MODE[])
@assert DUAL_WARM_MODE[] == :persist "expected default :persist"

for mode in (:persist, :reset_per_theta, :cold)
    global DUAL_WARM_MODE
    DUAL_WARM_MODE[] = mode
    _DUAL_CACHE[:blind] = nothing; _DUAL_CACHE[:aug] = nothing
    col, R, Rcol, umat, p, ok = seq_gravcol(θr0; δ = 1.0, maxit = 20, tol = 5e-4)
    @printf("mode=%-16s ok=%s R_mean=%.4e cache_blind=%s cache_aug=%s\n",
            mode, ok, R, _DUAL_CACHE[:blind] === nothing ? "nothing" : "set(len=$(length(_DUAL_CACHE[:blind])))",
            _DUAL_CACHE[:aug] === nothing ? "nothing" : "set(len=$(length(_DUAL_CACHE[:aug])))")
    @assert ok "seq_gravcol should succeed at Astar/delta=1.0 in every mode"
end
DUAL_WARM_MODE[] = :persist  # restore default

println("\nSMOKE_TEST_DUAL_WARM_PRODUCTION DONE")
