# Follow-up: isolate whether A_od perturbation alone (gamma' held at the base value) is enough to
# cause the infeasibility seen in check_multistart_feasibility.jl, or whether randomizing gamma'
# within its own bounds is the main driver.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

const COMMIT = "90d1951"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]

const RADII = [0.05, 0.1, 0.2, 0.5]
const N_SEEDS = 8

println("="^78); println("ISOLATION: A_od perturbation ONLY, gamma'_focal HELD at base value gp0"); println("="^78)
results = NamedTuple[]
for radius in RADII
    n_success = 0; statuses = Int[]
    rng = MersenneTwister(2000 + round(Int, radius * 1000))
    for seed in 1:N_SEEDS
        zfree_trial = zfree0 .+ radius .* randn(rng, D2 - 1)
        Aod_theta_trial = exp.(pivot_expand(zfree_trial, pe))
        xf = vcat(gp0, vec(Aod_theta_trial))   # gamma' FIXED at base
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)
        ok = r.inner_status in (0, -100, -101, -103) && isfinite(r.Delta_dual)
        ok && (n_success += 1)
        push!(statuses, r.inner_status)
        push!(results, (radius = radius, seed = seed, ok = ok, inner_status = r.inner_status, Delta_dual = r.Delta_dual))
    end
    println("radius=", radius, "  successes=", n_success, "/", N_SEEDS, "  statuses=", statuses)
end

open(joinpath(OUTDIR, "multistart_isolate_Aod_only.csv"), "w") do io
    println(io, "radius,seed,ok,inner_status,Delta_dual")
    for r in results
        println(io, r.radius, ",", r.seed, ",", r.ok, ",", r.inner_status, ",", r.Delta_dual)
    end
end
println("\nWrote ", joinpath(OUTDIR, "multistart_isolate_Aod_only.csv"))
