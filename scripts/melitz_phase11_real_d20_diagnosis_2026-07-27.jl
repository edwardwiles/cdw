# Governing prompt (outer-search session), Phase 11: short real-D20 outer diagnosis.
#
# real D=20 (noah_D20), W=80,000, canonical seed, delta=1, upper direction, evaluation
# cap=10.0, strict production-fast (matrix-free, forbid_dense_fallback=true), native :linear
# cutoff constraints, sorted PARALLEL outer gradient (20 Julia threads), near-boundary finite
# start (this session's own theta0 -- the calibration point, Delta0~4e-4, well below the
# outer budget delta=1 so genuinely near-boundary in outer-feasibility terms), active_set
# algorithm (Phase 8's own D=4 finding: robust to this scale set without needing a tuned
# delta, unlike the default Interior/Direct), maxtime_real=480s wall budget.
#
# This run is possible to interpret meaningfully ONLY because of this session's own Phase 2
# fix (register_live_candidate! no longer pays the ~99.7%-of-FC O(D^2*W) equilibrium-check
# cost on every trial) -- before that fix, ~480s would have bought at most ~80 finite FC
# calls; after it, orders of magnitude more.

using Pkg
Pkg.activate(dirname(@__DIR__))
using Random, DelimitedFiles, Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
println("Threads.nthreads() = ", Threads.nthreads(), "  BLAS threads = ", LinearAlgebra.BLAS.get_num_threads())

real_dir = joinpath(REPO, "real_data", "noah_D20")
lambdaData = readdlm(joinpath(real_dir, "pi.csv"), ',')
LData = vec(readdlm(joinpath(real_dir, "L.csv"), ',')) ./ 1e6
tauData = readdlm(joinpath(real_dir, "tau.csv"), ',')
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
focal = findfirst(==("fra"), countries)
observed = MelitzObservedData(; lambda=lambdaData, L=LData, tau=tauData, countries=countries, atol=2e-3)
calib = calibrate_melitz_pareto(observed; sigma=2.5, theta_star=:estimate, focal_country=focal, p_min=0.001, wage_tol=1e-8, gravity_tol=1e-6)
z_draws = pareto_draws(80_000, calib.D, calib.theta_star; seed=calib.seed)
p20, eq20, cf20, ctx20 = melitz_calibration_outer_ctx(calib; z_draws=z_draws, moment_backend=:sorted_tail_serial)
theta0 = melitz_reduce_theta(p20, ctx20)
n = length(theta0)
D = ctx20.D
nA = D^2 - 1
op20 = build_melitz_moment_operator(ctx20.sorted_tail_ctx, ctx20.moment_layout)
obj = build_melitz_cc_bundle(op20, ctx20; mode=:delta, U=z_draws,
    outer_constr_index=ctx20.moment_layout.num_moments + 1,
    lower_limit=-10.0,   # explicit evaluation-cap wire-up -- build_melitz_cc_bundle no longer has a dangerous default (see cc_bundle.jl docstring); matches this session's standard delta_evaluation_cap=10.0 convention so a poorly-conditioned point fails fast instead of grinding on an uncapped inner solve
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt,
    hessian_backend=:structured_parallel)
r0 = evaluate_melitz_delta(theta0, ctx20, obj; cold=true, store_G=false)
@assert r0.verified
println("theta0: Delta0=", r0.Delta, "  n_theta=", n)

# Fixed-A/f benchmark incumbent (this session's own theta0 itself, since eta is fixed at its
# calibrated value there) -- retained explicitly as the external comparison incumbent.
wm0 = melitz_welfare_metrics_from_g(theta0[1], ctx20)
println("Fixed-A/f (theta0) benchmark: g=", theta0[1], " kappa_ratio=", wm0.kappa_ratio, " GT=", wm0.gains_from_trade)

# Scale set: same reasoning as Phase 7/8 (this session), same order of magnitude as the
# independently-derived 2026-07-25 real-D20 session's own candidate.
s_g, s_A, s_f = 1e-4, 1e-5, 1e-5
var_scale = ones(n)
var_scale[1] = s_g
var_scale[2:1+nA] .= s_A
var_scale[2+nA:end] .= s_f
var_center = collect(Float64.(theta0))

inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
outer_opt = joinpath(REPO, "melitz_outer_finite_delta_phase11_activeset_2026-07-27.opt")

MELITZ_PROFILE[] = true
melitz_profile_reset!()

t_total = @elapsed res = solve_melitz_finite_delta_bound(ctx20, obj, theta0; delta=1.0, direction=:upper,
    delta_evaluation_cap=10.0, gradient_backend=:B_direct_argument_sorted_parallel, h=1e-4,
    theta_box=2.0, cutoff_constraint_backend=:linear,
    inner_loop_opt=inner_opt, outer_loop_opt=outer_opt,
    var_scale=var_scale, var_center=var_center,
    external_incumbent=theta0,
    backend=:matrix_free, forbid_dense_fallback=true)

println("\n=== Phase 11 result ===")
println("wall=", t_total, "  nStatus=", res.nStatus)
println("n_fc=", res.n_fc_calls, "  n_ga=", res.n_ga_calls)
println("inner_solve_count=", res.inner_solve_count, "  inner_infeas_count=", res.inner_infeas_count)
cv = res.cold_verified_incumbent
if cv !== nothing
    wm = melitz_welfare_metrics_from_g(cv.eval.theta_free[1], ctx20)
    println("cold_verified: g=", cv.eval.theta_free[1], " Delta=", cv.eval.Delta,
        " kappa_ratio=", wm.kappa_ratio, " GT=", wm.gains_from_trade)
end
best_live = res.best_live_incumbent
if best_live !== nothing
    println("best_live: g=", best_live.eval.theta_free[1], " Delta=", best_live.eval.Delta)
end
println("terminal: g=", res.terminal_theta[1])
dg_raw = res.terminal_theta[1] - theta0[1]
dA_raw = maximum(abs.(res.terminal_theta[2:1+nA] .- theta0[2:1+nA]))
df_raw = maximum(abs.(res.terminal_theta[2+nA:end] .- theta0[2+nA:end]))
println("raw movement: |dg|=", abs(dg_raw), " max|dA|=", dA_raw, " max|df|=", df_raw)

melitz_profile_report(stdout; trajectory_total_s=t_total)

outfile = joinpath(OUTDIR, "melitz_phase11_real_d20_trajectory_2026-07-27.csv")
open(outfile, "w") do io
    println(io, "metric,value")
    println(io, "wall,", t_total)
    println(io, "nStatus,", res.nStatus)
    println(io, "n_fc_calls,", res.n_fc_calls)
    println(io, "n_ga_calls,", res.n_ga_calls)
    println(io, "inner_solve_count,", res.inner_solve_count)
    println(io, "inner_infeas_count,", res.inner_infeas_count)
    println(io, "dg_raw,", dg_raw)
    println(io, "dA_raw_max,", dA_raw)
    println(io, "df_raw_max,", df_raw)
    println(io, "fixed_af_kappa_ratio,", wm0.kappa_ratio)
    println(io, "fixed_af_GT,", wm0.gains_from_trade)
    if cv !== nothing
        wm = melitz_welfare_metrics_from_g(cv.eval.theta_free[1], ctx20)
        println(io, "cold_verified_g,", cv.eval.theta_free[1])
        println(io, "cold_verified_Delta,", cv.eval.Delta)
        println(io, "cold_verified_kappa_ratio,", wm.kappa_ratio)
        println(io, "cold_verified_GT,", wm.gains_from_trade)
    end
end
println("Wrote ", outfile)

outfile2 = joinpath(OUTDIR, "melitz_phase11_real_d20_wallclock_decomposition_2026-07-27.csv")
rows_profile = melitz_profile_summary()
open(outfile2, "w") do io
    println(io, "category,count,total_s,mean_ms,median_ms,p90_ms,max_ms,pct_of_total")
    for r in rows_profile
        println(io, join([r.category, r.count, r.total_s, r.mean_ms, r.median_ms, r.p90_ms, r.max_ms,
            round(100 * r.total_s / t_total; digits=2)], ","))
    end
end
println("Wrote ", outfile2)
