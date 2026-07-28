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

bank = MelitzDualBank(8)
CAP = 10.0
r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
println("r0 = ", typeof(r0), "  Delta=", r0 isa FiniteSolved ? r0.Delta : "NA")
flush(stdout)

println("\n" * "="^100)
println("PHASE 2: pre-solve screens at theta_new (free, no KNITRO call)")
println("="^100)
G_new = melitz_bundle_prepare_at_theta!(obj, theta_new)
range_cert = obj isa MelitzCCBundle ? melitz_range_screen(obj.op) : melitz_range_screen(G_new)
println("range screen (matrix-free) at theta_new: ", range_cert === nothing ? "PASS (no single-column certificate)" : range_cert)
flush(stdout)

print("origin_block_screen at theta_new (HiGHS LP, joint D-block feasibility, NOT the routine default): ")
flush(stdout)
try
    ob_cert = melitz_origin_block_screen(theta_new, ctx, obj)
    println(ob_cert === nothing ? "PASS (joint feasibility LP finds no certificate)" : ob_cert)
catch e
    println("ERROR calling origin_block_screen: ", e)
end
flush(stdout)

println("\n" * "="^100)
println("INNER SOLVE 2/6 (repeat, uncapped bundle): r_new at theta_new")
println("="^100)
r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank)
println("result: ", typeof(r_new))
if r_new isa FiniteSolved
    println("  Delta=", r_new.Delta, "  nStatus=", r_new.nStatus, "  ||x||=", norm(r_new.x))
    localc2 = zeros(1)
    obj(r_new.x, constr=localc2)
    Delta_recomputed = localc2[1] / 1e10
    println("  independent re-evaluation of obj(r_new.x) => Delta_recomputed = ", Delta_recomputed)
    println("  matches r_new.Delta exactly: ", Delta_recomputed == r_new.Delta)
    println("  all(isfinite, r_new.x): ", all(isfinite, r_new.x))
    println("  isfinite(r_new.Delta): ", isfinite(r_new.Delta))
end
flush(stdout)

println("\n" * "="^100)
println("PHASE 4: authoritative capped evaluator, SAME theta_new, lower_limit properly wired to -10.0")
println("="^100)
println("obj.lower_limit BEFORE fix = ", obj.lower_limit)
obj.lower_limit = -10.0
println("obj.lower_limit AFTER fix  = ", obj.lower_limit)
flush(stdout)
bank_capped = MelitzDualBank(8)
obj.threshold_crossed[] = false
println("\nINNER SOLVE 3/6: capped (evaluation_cap=10) re-solve of theta_new")
r_capped10 = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank_capped)
println("  result: ", typeof(r_capped10))
if r_capped10 isa AboveEvaluationCap
    println("  certified_lower_bound=", r_capped10.certified_lower_bound, "  source=", r_capped10.source,
        "  crossing_time_s=", r_capped10.crossing_time_s)
elseif r_capped10 isa FiniteSolved
    println("  UNEXPECTED: FiniteSolved even under lower_limit=-10.0 -- Delta=", r_capped10.Delta, " nStatus=", r_capped10.nStatus)
end
flush(stdout)

println("\nINNER SOLVE 4/6: diagnostic variant, evaluation_cap=50")
obj.lower_limit = -50.0
obj.threshold_crossed[] = false
bank_capped50 = MelitzDualBank(8)
r_capped50 = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=50.0, bank=bank_capped50)
println("  result: ", typeof(r_capped50))
if r_capped50 isa AboveEvaluationCap
    println("  certified_lower_bound=", r_capped50.certified_lower_bound, "  source=", r_capped50.source,
        "  crossing_time_s=", r_capped50.crossing_time_s)
elseif r_capped50 isa FiniteSolved
    println("  Delta=", r_capped50.Delta, " nStatus=", r_capped50.nStatus)
end
flush(stdout)

obj.lower_limit = -1.7976931348623157e308

println("\nDONE Phase 2/3/4.")
