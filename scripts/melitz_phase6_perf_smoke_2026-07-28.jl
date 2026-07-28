# Post-consolidation validation, Phase 6: production-fast performance smoke test. NOT a
# performance study -- confirms consolidation did not disable the optimized kernels
# (20-thread parallel outer gradient, matrix-free callbacks, no dense G, active lower_limit).

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
melitz_thread_startup_report(; require=20, strict=false)
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)
d20 = build_realD20_fixture(; policy=policy)
ctx, obj, theta0 = d20.ctx, d20.obj, d20.theta0
n = length(theta0); D = ctx.D

println("\nobj.lower_limit = ", obj.lower_limit, "  (== -$CAP ? ", obj.lower_limit == -CAP, ")")
println("MELITZ_DENSE_MOMENT_CALLS[] (before) = ", MELITZ_DENSE_MOMENT_CALLS[])
println("MELITZ_DENSE_G_MATERIALIZATIONS[] (before) = ", MELITZ_DENSE_G_MATERIALIZATIONS[])
println("MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[] (before) = ",
        MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[])
cfg_auto = MelitzBackendConfig(; inner_backend=:matrix_free)
resolved_backend = melitz_resolve_gradient_backend(cfg_auto, D)
println("melitz_resolve_gradient_backend(:auto config, D=20, nthreads=$(Threads.nthreads())) = ", resolved_backend)
@assert occursin("parallel", string(resolved_backend)) "PARALLEL backend not resolved -- kernel regression!"
flush(stdout)

profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
order = sortperm(gs); gs, ds = gs[order], ds[order]
function g_for_target(target)
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    return gs[k] + t * (gs[k+1] - gs[k])
end
g_interior = g_for_target(0.5)
theta_interior = copy(theta0); theta_interior[1] = g_interior

# --- one finite FC wall (cold) ---
session = MelitzInnerSession(obj, ctx, policy)
t0 = time()
r_finite = solve_melitz_delta!(session, theta_interior, policy; warm_start_source=:previous)
wall_finite = time() - t0
println("\nOne finite FC wall (cold, interior Delta~0.5): $(round(wall_finite,digits=3))s  result=$(typeof(r_finite))" *
        (r_finite isa FiniteSolved ? "  Delta=$(r_finite.Delta)" : ""))
@assert r_finite isa FiniteSolved
flush(stdout)

# --- one InfiniteDeltaCertified/AboveEvaluationCap FC wall (reuse Phase 1's exact anomaly point) ---
n_sub = min(80, n - 1)
sub_coords = sort(randperm(MersenneTwister(4242), n - 1)[1:n_sub] .+ 1)
Wsub = 4000; idxsub = 1:Wsub
Jsub = zeros(Wsub * ctx.moment_layout.num_moments, n_sub)
ei = zeros(n)
for (jj, kk) in enumerate(sub_coords)
    ei[kk] = 1.0
    dG = melitz_moment_directional_derivative(theta_interior, ei, ctx, obj)
    Jsub[:, jj] .= vec(dG[idxsub, :])
    ei[kk] = 0.0
end
Usvd = svd(Jsub)
v_near_null = zeros(n); v_near_null[sub_coords] .= Usvd.V[:, end]
svd_basis_near_null = v_near_null ./ norm(v_near_null)
theta_anomaly = theta_interior .+ (-1.0) * 0.05 .* svd_basis_near_null
t0 = time()
r_infinite = solve_melitz_delta!(session, theta_anomaly, policy; origin_block_screen=true, warm_start_source=:previous)
wall_infinite = time() - t0
println("One InfiniteDeltaCertified/AboveEvaluationCap FC wall: $(round(wall_infinite,digits=3))s  result=$(typeof(r_infinite))")
@assert r_infinite isa Union{InfiniteDeltaCertified,AboveEvaluationCap,NumericalFailure}
flush(stdout)

# --- one complete 20-thread outer-gradient wall ---
h_fd = 1e-4
direct_gradient_fn = make_melitz_gradient_delta_direct_parallel(h_fd)
g_grad = zeros(n)
t0 = time()
direct_gradient_fn(g_grad, theta_interior, ctx, obj, r_finite.x)
wall_gradient = time() - t0
println("\nOne complete 20-thread outer-gradient wall: $(round(wall_gradient,digits=3))s  (n=$n coordinates, Threads.nthreads()=$(Threads.nthreads()))")
flush(stdout)

println("\n" * "="^100)
println("SMOKE-TEST SUMMARY")
println("="^100)
println("lower_limit active:                 ", obj.lower_limit == -CAP)
println("resolved gradient backend (D=20):    ", resolved_backend, "  (parallel: ", occursin("parallel", string(resolved_backend)), ")")
println("MELITZ_DENSE_MOMENT_CALLS[] (after):  ", MELITZ_DENSE_MOMENT_CALLS[])
println("MELITZ_DENSE_G_MATERIALIZATIONS[] (after): ", MELITZ_DENSE_G_MATERIALIZATIONS[])
println("MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[] (after): ",
        MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[])
println("no dense fallback triggered:          ", MELITZ_DENSE_MOMENT_CALLS[] == 0 && MELITZ_DENSE_G_MATERIALIZATIONS[] == 0)
println("finite FC wall:                       $(round(wall_finite,digits=3))s")
println("infinite/above-cap FC wall:            $(round(wall_infinite,digits=3))s")
println("outer-gradient wall (20 threads):      $(round(wall_gradient,digits=3))s")

open(joinpath(OUTDIR, "melitz_post_consolidation_phase6_perf_smoke_2026-07-28.csv"), "w") do io
    println(io, "quantity,value")
    println(io, "lower_limit_active,$(obj.lower_limit == -CAP)")
    println(io, "resolved_gradient_backend,$resolved_backend")
    println(io, "dense_moment_calls,$(MELITZ_DENSE_MOMENT_CALLS[])")
    println(io, "dense_G_materializations,$(MELITZ_DENSE_G_MATERIALIZATIONS[])")
    println(io, "explicit_serial_despite_parallel_count,$(MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT[])")
    println(io, "finite_FC_wall_s,$wall_finite")
    println(io, "infinite_or_abovecap_FC_wall_s,$wall_infinite")
    println(io, "outer_gradient_wall_s,$wall_gradient")
    println(io, "n_theta,$n")
    println(io, "julia_threads,$(Threads.nthreads())")
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase6_perf_smoke_2026-07-28.csv")
println("\nDONE Phase 6 (post-consolidation).")
