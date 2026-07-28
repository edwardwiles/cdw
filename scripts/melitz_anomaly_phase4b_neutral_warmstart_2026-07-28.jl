using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
d20 = build_realD20_fixture()
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
target = 0.5
k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
g_pt = gs[k] + t * (gs[k+1] - gs[k])
theta_pt = copy(theta0); theta_pt[1] = g_pt
theta_start = copy(theta_pt)

n_sub = min(80, n - 1)
sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)
Wsub = 4000
Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
ei = zeros(n)
for (jj, kk) in enumerate(sub_coords)
    ei[kk] = 1.0
    dG = melitz_moment_directional_derivative(theta_start, ei, ctx, obj)
    Jsub[:, jj] .= vec(dG[1:Wsub, :])
    ei[kk] = 0.0
end
Usvd = svd(Jsub)
v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
svd_basis_near_null = v_near_null ./ norm(v_near_null)
theta_new = theta_pt .+ (-1.0) * 0.05 .* svd_basis_near_null

# Set up a FRESH obj.x state (never contaminated by the uncapped runaway solve) with the
# CORRECT cap wired from the start, then classify theta_new directly.
obj.lower_limit = -10.0
obj.use_cached_x = false
obj.x .= NaN
obj.threshold_crossed[] = false

bank = MelitzDualBank(8)
println("INNER SOLVE 5/6: capped (evaluation_cap=10), NEUTRAL/clean warm start, theta_new")
r = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=10.0, bank=bank,
    warm_start_source=:neutral)
println("  result: ", typeof(r))
if r isa AboveEvaluationCap
    println("  certified_lower_bound=", r.certified_lower_bound, "  source=", r.source, "  crossing_time_s=", r.crossing_time_s)
elseif r isa FiniteSolved
    println("  Delta=", r.Delta, " nStatus=", r.nStatus)
elseif r isa InfiniteDeltaCertified
    println("  column=", r.column, " kind=", r.kind)
elseif r isa NumericalFailure
    println("  nStatus=", r.nStatus)
end
println("obj.threshold_crossed[] = ", obj.threshold_crossed[])
flush(stdout)
println("DONE")
