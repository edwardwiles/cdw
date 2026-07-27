# Governing prompt continuation (2026-07-27 night session), Phase 12: closure benchmarks.
# Standalone script -- complete D=4/real-D20 production-fast outer gradients (wall time +
# allocation), one finite D=20 FC, one AboveEvaluationCap D=20 FC, one short nuisance-profile
# callback sequence. Reuses production entry points exclusively.

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

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
rows = Vector{NamedTuple}()

# ----------------------------------------------------------------------------------------
# D=4 fixture
# ----------------------------------------------------------------------------------------
data4 = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
obj4, theta0_4 = build_melitz_psi_bundle(data4; forbid_dense_fallback=true)
ctx4 = obj4.γ
n4 = length(theta0_4)
r0_4 = evaluate_melitz_delta(theta0_4, ctx4, obj4; cold=true, store_G=false)
x0_4 = r0_4.dual_x
println("D=4: Delta0=", r0_4.Delta, " nStatus=", r0_4.nStatus)

gfun4 = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
gbuf4 = zeros(n4)
gfun4(gbuf4, theta0_4, ctx4, obj4, x0_4)   # warmup
t4 = @elapsed gfun4(gbuf4, theta0_4, ctx4, obj4, x0_4)
b4 = @allocated gfun4(gbuf4, theta0_4, ctx4, obj4, x0_4)
push!(rows, (scale="D4_W20000", item="complete_outer_gradient_serial", seconds=t4, bytes=b4, n_theta=n4, dense_fallbacks=0))
println(@sprintf("D=4 complete outer gradient: %.6fs, %d bytes", t4, b4))

# ----------------------------------------------------------------------------------------
# Real D=20 fixture
# ----------------------------------------------------------------------------------------
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
    inner_loop_opt=ctx20.inner_loop_opt, outer_loop_opt=ctx20.outer_loop_opt,
    hessian_backend=:structured_serial)
r0_20 = evaluate_melitz_delta(theta0_20, ctx20, obj20; cold=true, store_G=false)
x0_20 = r0_20.dual_x
println("D=20: Delta0=", r0_20.Delta, " nStatus=", r0_20.nStatus, " n_theta=", n20)

gfun20 = make_melitz_gradient_delta_direct_sorted_serial(1e-4)
gbuf20 = zeros(n20)
gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)   # warmup
t20 = @elapsed gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)
b20 = @allocated gfun20(gbuf20, theta0_20, ctx20, obj20, x0_20)
push!(rows, (scale="realD20_W80000", item="complete_outer_gradient_serial", seconds=t20, bytes=b20, n_theta=n20, dense_fallbacks=0))
println(@sprintf("D=20 complete outer gradient: %.6fs, %d bytes", t20, b20))

if Threads.nthreads() > 1
    gfun20p = make_melitz_gradient_delta_direct_sorted_parallel(1e-4)
    gbuf20p = zeros(n20)
    gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    t20p = @elapsed gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    b20p = @allocated gfun20p(gbuf20p, theta0_20, ctx20, obj20, x0_20)
    push!(rows, (scale="realD20_W80000", item="complete_outer_gradient_parallel", seconds=t20p, bytes=b20p, n_theta=n20, dense_fallbacks=0))
    println(@sprintf("D=20 complete outer gradient (parallel, %d threads): %.6fs, %d bytes", Threads.nthreads(), t20p, b20p))
end

# ----------------------------------------------------------------------------------------
# One finite D=20 FC + one AboveEvaluationCap D=20 FC (production-fast, linear cutoff
# constraint backend -- the actual production default, not the ForwardDiff reference path).
# ----------------------------------------------------------------------------------------
m20 = 1 + ctx20.D + ctx20.D * (ctx20.D - 1)
delta_loose20 = max(r0_20.Delta * 5, 1e-3)
cbset_finite = melitz_build_finite_delta_callbacks(obj20, ctx20, delta_loose20, true;
    gradient_backend=:B_direct_argument_sorted_serial, h=1e-4, cutoff_constraint_backend=:linear)
evalFinite = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset_finite.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFinite, nothing)   # warmup
tfin = @elapsed cbset_finite.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFinite, nothing)
bfin = @allocated cbset_finite.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalFinite, nothing)
push!(rows, (scale="realD20_W80000", item="finite_FC_call", seconds=tfin, bytes=bfin, n_theta=n20,
    dense_fallbacks=MELITZ_DENSE_G_MATERIALIZATIONS[]))
println(@sprintf("D=20 finite FC: %.6fs, %d bytes, obj=%.6e", tfin, bfin, evalFinite.obj[1]))

delta_tight20 = r0_20.Delta / 100   # forces AboveEvaluationCap: the real Delta is 100x the cap
cbset_cap = melitz_build_finite_delta_callbacks(obj20, ctx20, delta_tight20, true;
    gradient_backend=:B_direct_argument_sorted_serial, h=1e-4, cutoff_constraint_backend=:linear)
evalCap = MockEvalResultC(zeros(1), zeros(m20), zeros(n20), zeros(n20 * m20))
cbset_cap.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalCap, nothing)   # warmup
tcap = @elapsed cbset_cap.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalCap, nothing)
bcap = @allocated cbset_cap.cb_F!(nothing, nothing, MockEvalRequestC(copy(theta0_20)), evalCap, nothing)
push!(rows, (scale="realD20_W80000", item="above_evaluation_cap_FC_call", seconds=tcap, bytes=bcap, n_theta=n20,
    dense_fallbacks=MELITZ_DENSE_G_MATERIALIZATIONS[]))
expected_sentinel = 10.0 / delta_tight20   # default delta_evaluation_cap=10.0
println(@sprintf("D=20 AboveEvaluationCap FC: %.6fs, %d bytes, constraint_sentinel=%.6e (expected %.6e)",
    tcap, bcap, evalCap.c[1], expected_sentinel))

# ----------------------------------------------------------------------------------------
# One short nuisance-profile callback sequence (D=4, small radius, few free coordinates,
# strict production-fast).
# ----------------------------------------------------------------------------------------
free_mask4 = falses(n4)
free_mask4[2:5] .= true   # a handful of technology coordinates free, everything else fixed
inner_cfg4 = MelitzInnerSolveConfig(:full_value)
t_nuis = @elapsed nuis_result = solve_melitz_nuisance_min_delta(ctx4, obj4, theta0_4;
    free_mask=free_mask4, radius=0.05, gradient_backend=:B_direct_argument_sorted_serial, h=1e-4,
    inner_loop_opt=ctx4.inner_loop_opt, inner_solve_config=inner_cfg4, forbid_dense_fallback=true)
push!(rows, (scale="D4_W20000", item="nuisance_profile_short_sequence", seconds=t_nuis, bytes=0, n_theta=n4,
    dense_fallbacks=0))
println(@sprintf("D=4 short nuisance-profile sequence: %.6fs, Delta_min found=%.6e", t_nuis, nuis_result.Delta_min))

outfile = joinpath(OUTDIR, "melitz_phase12_closure_benchmarks_2026-07-27.csv")
open(outfile, "w") do io
    cols = keys(rows[1])
    println(io, join(cols, ","))
    for r in rows
        println(io, join([r[c] for c in cols], ","))
    end
end
println("Wrote ", outfile)
