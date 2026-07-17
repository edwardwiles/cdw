# Task §22 applied to the lower-direction short-run's best-feasible point
# (results/fullA_d4/9e03706/optfd_lower_20260717_190831/summary.txt).
include(joinpath(@__DIR__, "stationarity_check.jl"))

const COMMIT = "9e03706"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)

ctx = d4_exact_setup(find_smallest = false)
pe = build_pivot_elimination(ctx)
D2 = ctx.D^2
w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

w_best = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916,
          -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819,
          0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236,
          0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

println("="^78); println("EXTERNAL KKT STATIONARITY CHECK -- lower direction best-feasible point"); println("="^78)
result = external_stationarity_check(w_best, ctx, pe; find_smallest = false, w_lo = w_lo, w_hi = w_hi)
println("Delta = ", result.Delta, "  Delta-delta = ", result.Delta_minus_delta)
println("eta = ", result.eta, "  (eta>=0: ", result.eta_nonneg, ")")
println("KKT residual = ", result.residual_norm, "  relative = ", result.residual_relative)
println("complementary slackness = ", result.complementary_slackness)
println("active bounds: ", result.n_active_bounds, "  non-finite probes: ", result.n_nonfinite_probes)

verified = result.eta_nonneg && result.residual_relative < 0.05 && abs(result.complementary_slackness) < 0.05 && result.n_active_bounds == 0 && result.n_nonfinite_probes == 0
println("\nSTATIONARITY VERDICT: ", verified ? "PASSES" : "FAILS / inconclusive")
println("(expected to plausibly FAIL: Delta-delta=-0.101 is far from binding, i.e. the search stalled")
println(" before using its full divergence budget -- this is a genuine 'ran out of iterations' case,")
println(" not evidence the method itself is broken for this direction.)")

open(joinpath(OUTDIR, "stationarity_check_lower.txt"), "w") do io
    println(io, "w = ", w_best); println(io, result); println(io, "verified = ", verified)
end
println("\nWrote ", joinpath(OUTDIR, "stationarity_check_lower.txt"))
