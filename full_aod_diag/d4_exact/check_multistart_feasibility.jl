# User question 2: does the inner delta* program even converge (find a feasible reweighted
# distribution) at random starting points, across a range of perturbation magnitudes? NOT running
# the outer solver -- just testing whether evaluate_fullA succeeds (inner_status good, Delta_dual
# finite) at random points, per the user's explicit framing: "literally just whether picking a few
# randomized starting points even gets started or immediately fails."
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

const COMMIT = "90d1951"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]

const RADII = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]   # SD of the log-A perturbation (in log(Aod_theta) units)
const N_SEEDS = 8

println("="^78); println("MULTISTART FEASIBILITY SCAN (inner delta* program only, no outer solve)"); println("="^78)
println("Perturbation: z_free = z_free0 + radius * randn(15) (gravity satisfied EXACTLY by")
println("construction via pivot_expand, so gravity itself is never the cause of failure here);")
println("gamma'_focal drawn uniformly within its theoretical bounds each trial.\n")

results = NamedTuple[]
for radius in RADII
    n_success = 0; n_fail = 0; statuses = Int[]
    rng = MersenneTwister(1000 + round(Int, radius * 1000))
    for seed in 1:N_SEEDS
        gp_trial = rand(rng) * (ctx.bounds.γp_hi - ctx.bounds.γp_lo) + ctx.bounds.γp_lo
        zfree_trial = zfree0 .+ radius .* randn(rng, D2 - 1)
        z_trial = pivot_expand(zfree_trial, pe)
        Aod_theta_trial = exp.(z_trial)
        # sanity: gravity really is exact at this random point (not assumed)
        grav = gravity_from_logz(z_trial, ctx)
        @assert abs(grav) < 1e-8 "gravity elimination failed at a random point -- should be impossible by construction"
        xf = vcat(gp_trial, vec(Aod_theta_trial))
        t0 = time()
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)   # COLD start -- worst case for a multistart point
        elapsed = time() - t0
        ok = r.inner_status in (0, -100, -101, -103) && isfinite(r.Delta_dual)
        ok ? (n_success += 1) : (n_fail += 1)
        push!(statuses, r.inner_status)
        push!(results, (radius = radius, seed = seed, gp = gp_trial, ok = ok, inner_status = r.inner_status,
                         Delta_dual = r.Delta_dual, elapsed = elapsed,
                         Aod_theta_min = minimum(Aod_theta_trial), Aod_theta_max = maximum(Aod_theta_trial)))
    end
    println("radius=", radius, "  successes=", n_success, "/", N_SEEDS,
            "  statuses=", statuses, "  (Aod_theta range at last trial: ",
            round(results[end].Aod_theta_min, digits=3), " to ", round(results[end].Aod_theta_max, digits=3), ")")
end

open(joinpath(OUTDIR, "multistart_feasibility_scan.csv"), "w") do io
    println(io, "radius,seed,gp,ok,inner_status,Delta_dual,elapsed,Aod_theta_min,Aod_theta_max")
    for r in results
        println(io, r.radius, ",", r.seed, ",", r.gp, ",", r.ok, ",", r.inner_status, ",", r.Delta_dual, ",", r.elapsed, ",", r.Aod_theta_min, ",", r.Aod_theta_max)
    end
end

println("\n" * "="^78); println("SUMMARY BY RADIUS"); println("="^78)
for radius in RADII
    rows = filter(r -> r.radius == radius, results)
    succ_rate = count(r -> r.ok, rows) / length(rows)
    println("radius=", radius, "  success_rate=", round(100*succ_rate, digits=1), "%",
            "  median wall time (success only)=", isempty(filter(r->r.ok, rows)) ? "n/a" : round(median([r.elapsed for r in rows if r.ok]), digits=4), "s")
end
println("\nWrote ", joinpath(OUTDIR, "multistart_feasibility_scan.csv"))
