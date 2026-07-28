using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")

println("="^100)
println("PHASE 0/1: fixture + environment + exact anomaly reconstruction")
println("="^100)
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
println("KNITRO.KN_INFINITY = ", KNITRO.KN_INFINITY)
flush(stdout)

d20 = build_realD20_fixture()
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D
println("obj type = ", typeof(obj))
println("obj.lower_limit (at fixture construction) = ", obj.lower_limit)
println("obj.lower_limit == -KN_INFINITY ? ", obj.lower_limit == -KNITRO.KN_INFINITY)
println("obj.use_cached_x (fresh) = ", obj.use_cached_x, "   obj.x (fresh, first few) = ", obj.x[1:min(5,length(obj.x))])
flush(stdout)

# --- reconstruct theta_pt for target=0.5, EXACTLY as phase10_main does ---
profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
target = 0.5
k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
g_pt = gs[k] + t * (gs[k+1] - gs[k])
theta_pt = copy(theta0); theta_pt[1] = g_pt
println("\nReconstructed g_pt = ", g_pt, "  (CSV row 'g' = -0.48546126592859407)")
println("Match: ", g_pt == -0.48546126592859407)
flush(stdout)

# --- reconstruct theta_start for phase9 (same target=0.5 interpolation off SAME CSV) ---
theta_start = copy(theta0); theta_start[1] = g_pt   # identical construction

# --- reconstruct the reduced empirical SVD basis, EXACTLY as main() does ---
println("\nBuilding reduced empirical SVD basis (deterministic, seeded, NOT an inner solve)...")
flush(stdout)
n_sub = min(80, n - 1)
sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)
Wsub = 4000
idxsub = 1:Wsub
Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
ei = zeros(n)
for (jj, kk) in enumerate(sub_coords)
    ei[kk] = 1.0
    dG = melitz_moment_directional_derivative(theta_start, ei, ctx, obj)
    Jsub[:, jj] .= vec(dG[idxsub, :])
    ei[kk] = 0.0
end
Usvd = svd(Jsub)
v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
svd_basis_near_null = v_near_null ./ norm(v_near_null)
println("Smallest singular values: ", Usvd.S[end-5:end])
flush(stdout)

# --- INNER SOLVE 1 of budgeted 6: r0 at theta_pt ---
bank = MelitzDualBank(8)
CAP = 10.0
println("\n" * "-"^100)
println("INNER SOLVE 1/6: r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=$CAP, bank)")
obj.threshold_crossed[] = false
t0 = time()
r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
t_r0 = time() - t0
println("  wall = ", t_r0, "s")
println("  result type = ", typeof(r0))
if r0 isa FiniteSolved
    println("  r0.Delta = ", r0.Delta, "   r0.nStatus = ", r0.nStatus, "   ||x|| = ", norm(r0.x))
end
println("  obj.threshold_crossed[] after call = ", obj.threshold_crossed[])
println("  CSV Delta0 = 0.4832764950468894   match: ", r0 isa FiniteSolved && r0.Delta == 0.4832764950468894)
flush(stdout)

# --- theta_new: theta_pt + sign(-1) * step_norm(0.05) * near_null direction ---
theta_new = theta_pt .+ (-1.0) * 0.05 .* svd_basis_near_null

println("\n" * "-"^100)
println("INNER SOLVE 2/6: r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=$CAP, bank)  [svd_near_null, sign=-1]")
obj.threshold_crossed[] = false
t0 = time()
r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank)
t_r_new = time() - t0
println("  wall = ", t_r_new, "s")
println("  result type = ", typeof(r_new))
println("  obj.lower_limit AT THIS SOLVE = ", obj.lower_limit)
println("  obj.threshold_crossed[] after call = ", obj.threshold_crossed[])
if r_new isa FiniteSolved
    println("  r_new.Delta = ", r_new.Delta)
    println("  r_new.nStatus = ", r_new.nStatus)
    println("  ||r_new.x|| = ", norm(r_new.x))
    println("  any(!isfinite, r_new.x) = ", any(!isfinite, r_new.x))
    println("  CSV Delta_after_step = 1.510118285974391e14")
    println("  EXACT MATCH: ", r_new.Delta == 1.510118285974391e14)
    println("  relative diff: ", abs(r_new.Delta - 1.510118285974391e14) / 1.510118285974391e14)
elseif r_new isa AboveEvaluationCap
    println("  certified_lower_bound = ", r_new.certified_lower_bound, "  source = ", r_new.source)
elseif r_new isa InfiniteDeltaCertified
    println("  column=", r_new.column, " lo=", r_new.lo, " hi=", r_new.hi, " kind=", r_new.kind)
elseif r_new isa NumericalFailure
    println("  nStatus = ", r_new.nStatus)
end
flush(stdout)

println("\nDONE Phase 0/1 reconstruction script.")
