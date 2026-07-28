# Quick standalone validation of the new touched-row gradient backend (Phase 4, outer-search
# session) against the existing sorted crossing-slice backend -- D=4 (every coordinate) and a
# few representative real-D20 directions. Not the full test suite (which is slow to iterate
# on) -- a fast sanity gate before relying on it.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

println("== D=4 validation ==")
data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true)
ctx4 = obj4.γ
n4 = length(theta0_4)
r0_4 = evaluate_melitz_delta(theta0_4, ctx4, obj4; cold=true, store_G=false)
x0_4 = r0_4.dual_x
println("Delta0=", r0_4.Delta, " nStatus=", r0_4.nStatus)

gsorted = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
gtouched = make_melitz_gradient_delta_direct_touched_row_serial(1e-4)
g1 = zeros(n4); g2 = zeros(n4)
gsorted(g1, theta0_4, ctx4, obj4, x0_4)
gtouched(g2, theta0_4, ctx4, obj4, x0_4)
maxrel = maximum(abs.(g1 .- g2) ./ max.(abs.(g1), 1.0))
println("D=4 max relative diff (sorted vs touched-row): ", maxrel)
@assert maxrel < 1e-10 "D=4 mismatch too large!"

# call again to catch any stale-generation-stamp bug
gtouched(g2, theta0_4, ctx4, obj4, x0_4)
maxrel2 = maximum(abs.(g1 .- g2) ./ max.(abs.(g1), 1.0))
println("D=4 second call max relative diff: ", maxrel2)
@assert maxrel2 < 1e-10 "second-call mismatch -- stale generation stamp bug!"

# perturbed point (away from calibration) to exercise different touched-row patterns
rng = MersenneTwister(7)
theta_pert = theta0_4 .+ 0.001 .* randn(rng, n4)
r_pert = evaluate_melitz_delta(theta_pert, ctx4, obj4; cold=true, store_G=false)
if r_pert.nStatus == 0
    x_pert = r_pert.dual_x
    g1p = zeros(n4); g2p = zeros(n4)
    gsorted(g1p, theta_pert, ctx4, obj4, x_pert)
    gtouched(g2p, theta_pert, ctx4, obj4, x_pert)
    maxrelp = maximum(abs.(g1p .- g2p) ./ max.(abs.(g1p), 1.0))
    println("D=4 perturbed-point max relative diff: ", maxrelp)
    @assert maxrelp < 1e-10 "perturbed-point mismatch!"
else
    println("perturbed point did not verify (nStatus=$(r_pert.nStatus)), skipping that check")
end

println("\n== real D=20 validation (representative directions only) ==")
real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
theta0_20 = melitz_reduce_theta(p20, ctx20)
n20 = length(theta0_20)
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj20 = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt, hessian_backend=:structured_serial)
r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
x0_20 = r0_20.dual_x
println("D=20 Delta0=", r0_20.Delta, " nStatus=", r0_20.nStatus, " n_theta=", n20)

g1_20 = zeros(n20); g2_20 = zeros(n20)
t_sorted = @elapsed gsorted(g1_20, theta0_20, ctx20, obj20, x0_20)
t_touched = @elapsed gtouched(g2_20, theta0_20, ctx20, obj20, x0_20)
maxrel20 = maximum(abs.(g1_20 .- g2_20) ./ max.(abs.(g1_20), 1.0))
println(@sprintf("D=20 complete-gradient max relative diff: %.3e", maxrel20))
@assert maxrel20 < 1e-8 "D=20 mismatch too large!"

# warm timings
gsorted(g1_20, theta0_20, ctx20, obj20, x0_20)
gtouched(g2_20, theta0_20, ctx20, obj20, x0_20)
t_sorted_warm = @elapsed gsorted(g1_20, theta0_20, ctx20, obj20, x0_20)
t_touched_warm = @elapsed gtouched(g2_20, theta0_20, ctx20, obj20, x0_20)
b_sorted = @allocated gsorted(g1_20, theta0_20, ctx20, obj20, x0_20)
b_touched = @allocated gtouched(g2_20, theta0_20, ctx20, obj20, x0_20)
println(@sprintf("D=20 warm: sorted=%.6fs (%d bytes), touched_row=%.6fs (%d bytes), speedup=%.2fx",
    t_sorted_warm, b_sorted, t_touched_warm, b_touched, t_sorted_warm / t_touched_warm))

println("\nALL CHECKS PASSED")
