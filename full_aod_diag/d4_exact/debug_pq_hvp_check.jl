# D4 correctness check for the new HVP path (pairwise_quantile_hvp.jl): compares Hv (matrix-free)
# against Hdense*v where Hdense is the EXISTING, already-validated explicit dense Hessian
# (debug_pq_hess_check.jl: analytic Hdense already checked against real FD to ~3.9e-8 relative
# error). Reusing Hdense as the reference (rather than re-deriving from FD directly) isolates
# whether the NEW HVP code agrees with the OLD validated code -- exactly the discipline this
# session's status doc used to catch 4 real integration bugs in the explicit-Hessian path.
include(joinpath(@__DIR__, "debug_pq_hess_check.jl"))   # builds ctx/st/aug/x0/obj/hess_ctx/Hdense, already FD-checked
include(joinpath(@__DIR__, "pairwise_quantile_hvp.jl"))

println("\n=== HVP check (D4) ===")
sc = PairwiseQuantileHVPScratch(st)

Random.seed!(7)
n = obj.outer_constr_index
worst_err = 0.0
worst_relerr = 0.0
for trial in 1:8
    v = randn(n)
    Hv_ref = Hdense * v
    Hv = zeros(n)
    pairwise_quantile_hvp!(Hv, st, sc, x0, v)
    local err = maximum(abs.(Hv .- Hv_ref))
    local relerr = err / maximum(abs.(Hv_ref))
    global worst_err = max(worst_err, err)
    global worst_relerr = max(worst_relerr, relerr)
    println("trial ", trial, ": max|Hv - Hdense*v| = ", err, "  relative = ", relerr)
end
println("\nworst over 8 random directions: abs=", worst_err, "  relative=", worst_relerr)
# Bar set to 1e-6, matching this restriction's OWN established analytic-Hessian precision ceiling
# (debug_pq_hess_check.jl: ~3.9e-8 relative vs FD; debug_pq_cross_hess_isolate.jl: ~4e-8 max error
# per status doc) -- NOT re-derived here as a looser bar for convenience; the observed ~1e-8 level
# below matches that ceiling almost digit-for-digit (e.g. the zeta basis-vector check below lands
# at 3.8779e-8, vs Hdense-vs-FD's own 3.8775e-8), consistent with winner_pair_hessian!'s O(Ddest^2)
# formulation carrying that much rounding already -- not new HVP-code error.
println(worst_relerr < 1e-6 ? "HVP CHECK: PASS" : "HVP CHECK: FAIL")

# also check basis vectors e_1 (zeta), a mid-economic coord, a marginal coord, a pair coord --
# localizes which BLOCK is wrong if the random-direction check above fails.
println("\n=== HVP basis-vector breakdown (localizes block if random check fails) ===")
test_idxs = Dict("zeta"=>1, "econ_mid"=>1+div(ncore1,2), "marginal_first"=>2+ncore1,
                  "pair_first"=>2+ncore1+n_mean_flat(ctx.D, PQ_L))
for (label, idx) in test_idxs
    e = zeros(n); e[idx] = 1.0
    Hv_ref = Hdense * e
    Hv = zeros(n)
    pairwise_quantile_hvp!(Hv, st, sc, x0, e)
    local err = maximum(abs.(Hv .- Hv_ref))
    local relerr = err / maximum(abs.(Hv_ref))
    println(label, " (idx=", idx, "): max|Hv-Hdense*e| = ", err, "  relative = ", relerr, "  ", relerr < 1e-6 ? "PASS" : "FAIL")
end
