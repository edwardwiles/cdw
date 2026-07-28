# Phase 4 Part B only (Part A already succeeded and is NOT re-run -- its 12 solves are
# preserved as-is; re-running it would burn budget for no new information). Fixes a bug in
# the original combined script's `make_opt` (missing `lines` argument to `foreach`).

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)
d20 = build_realD20_fixture(; policy=policy)
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
function g_for_target(target)
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    return gs[k] + t * (gs[k+1] - gs[k])
end

println("\n" * "="^100); println("PART B: gamma_only nested-block re-run via solve_melitz_finite_delta_bound"); println("="^100)

g_start = g_for_target(0.5)
theta_start = copy(theta0); theta_start[1] = g_start
sqp_base = joinpath(REPO, "melitz_outer_finite_delta_alg_sqp_2026-07-27.opt")
inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")

function make_opt(base, tag; delta, maxit, maxtime_real)
    tmpdir = joinpath(OUTDIR, "tmp_opt_post_consolidation_2026-07-28"); mkpath(tmpdir)
    lines = readlines(base)
    lines = filter(l -> !occursin(r"^\s*(delta|maxit|maxtime_real)\s", l), lines)
    push!(lines, @sprintf("delta           %.6g", delta))
    push!(lines, @sprintf("maxit           %d", maxit))
    push!(lines, @sprintf("maxtime_real    %.1f", maxtime_real))
    path = joinpath(tmpdir, tag * ".opt")
    open(io -> foreach(l -> println(io, l), lines), path, "w")
    return path
end
function block_box(n, D, g_radius, A_radius, f_radius)
    nA = D^2 - 1; b = zeros(n); b[1] = g_radius; b[2:1+nA] .= A_radius; b[2+nA:end] .= f_radius; b
end

opt = make_opt(sqp_base, "phase4_gamma_only_recert_2026-07-28"; delta=0.05, maxit=1000, maxtime_real=180.0)
box = block_box(n, D, 0.5, 0.0, 0.0)   # gamma-only: A/f radius = 0
vs = ones(n); vs[1] = 1e-4

MELITZ_PROFILE[] = true; melitz_profile_reset!()
t0 = time()
res = solve_melitz_finite_delta_bound(ctx, obj, theta_start; delta=1.0, direction=:upper,
    policy=policy, gradient_backend=:auto, theta_box=box,
    cutoff_constraint_backend=:linear, inner_loop_opt=inner_opt, outer_loop_opt=opt,
    var_scale=vs, var_center=collect(Float64.(theta_start)),
    backend=:matrix_free, forbid_dense_fallback=true, objective_scale=:auto)
wall = time() - t0
MELITZ_PROFILE[] = false

cv = res.cold_verified_incumbent
if cv === nothing
    println("gamma_only re-run: NO cold_verified_incumbent (nStatus=$(res.nStatus)) -- DIVERGENT from historical (which produced a real incumbent)")
    row_b = (block="gamma_only", nStatus=res.nStatus, wall=wall, dg=NaN, DeltaStar=NaN)
else
    theta_final = cv.eval.theta_free
    dg = theta_final[1] - theta_start[1]
    row_b = (block="gamma_only", nStatus=res.nStatus, wall=wall, dg=dg, DeltaStar=cv.eval.Delta)
    println("gamma_only re-run: nStatus=$(res.nStatus)  wall=$(round(wall,digits=1))s  dg=$dg  DeltaStar=$(cv.eval.Delta)")
    println("  historical:       nStatus=0             wall=45.2s               dg=-0.010048147673314078  DeltaStar=0.8603593934221736")
end
flush(stdout)

open(joinpath(OUTDIR, "melitz_post_consolidation_phase4b_gamma_only_rerun_2026-07-28.csv"), "w") do io
    println(io, "block,nStatus,wall,dg,DeltaStar")
    println(io, "gamma_only,$(row_b.nStatus),$(row_b.wall),$(row_b.dg),$(row_b.DeltaStar)")
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase4b_gamma_only_rerun_2026-07-28.csv")
println("\nDONE Phase 4 Part B (post-consolidation).")
