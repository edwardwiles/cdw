# Correction script: the Phase 1/2 script's own "complete finite FC" measurement warmed up
# and timed the SAME theta (JIT-warmup convention), which means the TIMED call hit the FC's
# own exact-point cache (melitz_exact_cache_get) rather than paying for a fresh KNITRO inner
# solve -- confirmed live (inner_solve_cache_hit count=1 on the timed call). This script
# measures an honest FRESH-theta FC: JIT-warm at theta0, then time a call at a DIFFERENT
# (perturbed) theta that has never been visited, so the exact-point cache genuinely misses.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

mutable struct MockEvalRequestC
    x::Vector{Float64}
end
mutable struct MockEvalResultC
    obj::Vector{Float64}
    c::Vector{Float64}
    objGrad::Vector{Float64}
    jac::Vector{Float64}
end

real_dir = joinpath(dirname(@__DIR__), "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
inner_opt_capped = joinpath(dirname(@__DIR__), "melitz_inner_loop_options_capped_2026-07-24.opt")
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial,
    inner_loop_opt=inner_opt_capped)
theta0_20 = melitz_reduce_theta(p20, ctx20)
n20 = length(theta0_20)
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj20 = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt, hessian_backend=:structured_serial)
r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
println("Delta0=", r0_20.Delta, " nStatus=", r0_20.nStatus)

m20 = 1
delta_loose20 = max(r0_20.Delta * 5, 1e-3)
cbset = melitz_build_finite_delta_callbacks(obj20, ctx20, delta_loose20, true;
    gradient_backend=:B_direct_argument_sorted_serial, h=1e-4, cutoff_constraint_backend=:linear)

# JIT-warmup call at theta0 (compiles everything, populates the cache at theta0 -- irrelevant
# to the FRESH-theta calls below, which use DIFFERENT thetas each time).
evalW = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalW, nothing)

rng = MersenneTwister(99)
println("\n== Fresh-theta FC calls (each theta visited exactly once -- genuine cache misses) ==")
times = Float64[]
trial = 0
attempts = 0
while length(times) < 5 && attempts < 20
    global trial += 1
    global attempts += 1
    theta_fresh = theta0_20 .+ 1e-5 .* randn(rng, n20)   # small perturbation, distinct theta each trial
    evalF = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
    local t
    try
        t = @elapsed cbset.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta_fresh)), evalF, nothing)
    catch e
        # A genuine NumericalFailure (capped inner solve, no certificate) is real production
        # behavior for SOME random draws, not a bug in this measurement script -- skip and
        # draw a different fresh theta rather than treat it as a script error.
        println("  (attempt $attempts: NumericalFailure on this random draw, skipping -- ", sprint(showerror, e)[1:min(80,end)], ")")
        continue
    end
    push!(times, t)
    println(@sprintf("trial %d (attempt %d): wall=%.4fs  obj=%.6e", length(times), attempts, t, evalF.obj[1]))
end
println(@sprintf("\nmean fresh-theta FC wall: %.4fs  (min=%.4fs, max=%.4fs, over %d successful / %d attempted)",
    sum(times)/length(times), minimum(times), maximum(times), length(times), attempts))

println("\n== Re-visiting the FIRST fresh theta again (should now hit the exact-point cache) ==")
theta_repeat = theta0_20 .+ 1e-5 .* randn(MersenneTwister(99), n20)   # same as trial 1 above
evalR = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
t_cached = @elapsed cbset.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta_repeat)), evalR, nothing)
println(@sprintf("cached re-visit wall: %.4fs", t_cached))
