# Task §22 applied to the upper-direction short-run's best-feasible point
# (results/fullA_d4/9e03706/optfd_upper_20260717_182444/summary.txt).
include(joinpath(@__DIR__, "stationarity_check.jl"))

const COMMIT = "9e03706"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D2 = ctx.D^2
w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

# best_feasible_tracked w from the upper run's summary.txt (copied verbatim, not retyped from memory)
w_best = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069,
          0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011,
          1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663,
          0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]

println("="^78); println("EXTERNAL KKT STATIONARITY CHECK -- upper direction best-feasible point"); println("="^78)
result = external_stationarity_check(w_best, ctx, pe; find_smallest = true, w_lo = w_lo, w_hi = w_hi)
println("Delta = ", result.Delta, "  Delta-delta = ", result.Delta_minus_delta)
println("eta (constraint multiplier, from least squares) = ", result.eta, "  (eta>=0: ", result.eta_nonneg, ")")
println("KKT residual ||grad_f + eta*grad_Delta|| = ", result.residual_norm, "  (relative to ||grad_f||=1: ", result.residual_relative, ")")
println("complementary slackness eta*(Delta-delta) = ", result.complementary_slackness)
println("active bounds: ", result.n_active_bounds, "  non-finite FD probes: ", result.n_nonfinite_probes)
println("grad_Delta = ", result.grad_Delta)

verified = result.eta_nonneg && result.residual_relative < 0.05 && abs(result.complementary_slackness) < 0.05 && result.n_active_bounds == 0 && result.n_nonfinite_probes == 0
println("\nSTATIONARITY VERDICT: ", verified ? "PASSES external KKT check (relative residual < 5%)" : "FAILS / inconclusive external KKT check")

open(joinpath(OUTDIR, "stationarity_check_upper.txt"), "w") do io
    println(io, "w = ", w_best)
    println(io, result)
    println(io, "verified = ", verified)
end
println("\nWrote ", joinpath(OUTDIR, "stationarity_check_upper.txt"))
