# Follow-up: are the radius>=0.2 multistart failures a COLD-START artifact (bad initial guess for
# the inner KNITRO dual) or genuine primal infeasibility (no feasible reweighting exists at all)?
# Test: for each FAILED cold-start point, try again warm-started from a chain of intermediate
# points walking from the KNOWN-feasible base (A_od*) to the failed point, seeding obj.x at each
# step from the previous step's solution -- i.e. does a "path of warm starts" rescue points that
# fail outright from a naive cold start?
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))

const COMMIT = "b90e062"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]

function xfree_at(zfree, gp)
    Aod_theta = exp.(pivot_expand(zfree, pe))
    return vcat(gp, vec(Aod_theta))
end

println("="^78); println("WARM-START-PATH RESCUE TEST for radius=0.5/1.0 (0% cold-start success)"); println("="^78)
for radius in (0.5, 1.0)
    rng = MersenneTwister(3000 + round(Int, radius * 1000))
    for seed in 1:4
        zfree_target = zfree0 .+ radius .* randn(rng, D2 - 1)
        gp_target = rand(rng) * (ctx.bounds.γp_hi - ctx.bounds.γp_lo) + ctx.bounds.γp_lo

        # confirm cold-start failure first
        r_cold = evaluate_fullA(xfree_at(zfree_target, gp_target), ctx; cache = nothing, warm = false)
        cold_ok = r_cold.inner_status in (0, -100, -101, -103) && isfinite(r_cold.Delta_dual)

        # walk a path of N_STEPS from (zfree0, gp0) to (zfree_target, gp_target), warm-starting
        # each step from the PREVIOUS step's solved dual (obj.x carries over automatically since
        # evaluate_fullA with warm=true reuses ctx.obj.x when finite)
        N_STEPS = 20
        path_ok = true
        local r_path
        for k in 1:N_STEPS
            α = k / N_STEPS
            zfree_k = (1 - α) .* zfree0 .+ α .* zfree_target
            gp_k = (1 - α) * gp0 + α * gp_target
            r_path = evaluate_fullA(xfree_at(zfree_k, gp_k), ctx; cache = nothing, warm = true)
            ok_k = r_path.inner_status in (0, -100, -101, -103) && isfinite(r_path.Delta_dual)
            if !ok_k
                path_ok = false
                break
            end
        end

        println("radius=$radius seed=$seed: cold_start_ok=$cold_ok  warm_PATH_ok=$path_ok" *
                (path_ok ? "  (final Delta=$(r_path.Delta_dual))" : ""))
    end
end
println("\nInterpretation: if warm_PATH_ok=true for points where cold_start_ok=false, the failure is a")
println("COLD-START artifact (rescuable by better initialization / continuation), not fundamental")
println("economic infeasibility. If warm_PATH_ok is ALSO false, the target point is genuinely")
println("primal-infeasible regardless of initialization.")
