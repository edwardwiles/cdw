ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using Printf

θfixed = copy(θr0); θfixed[3] = 0.892861
@printf("Testing fixed-A (A=A*) convergence at gamma'=%.6f with increasing maxit\n", θfixed[3])
for mi in (20, 50, 100, 200)
    col, R, Rcol, umat, p, ok = seq_gravcol(θfixed; δ=Inf, maxit=mi, tol=5e-4, verbose=false)
    @printf("maxit=%4d  R_mean=%.4e  gravity_ok=%s  div(p)=%.4f\n", mi, R, abs(R)<=5e-4, isempty(p) ? NaN : divergence_of(p))
end
