# Task §13 run. Run: julia --project=. full_aod_diag/d4_exact/test_h_sweep.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "h_sweep.jl"))

const COMMIT = "1bdb1cc"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
base = solve_base_state(x0, ctx)

rng = MersenneTwister(4242)   # SAME direction as the winner-switching/method-A diagnostics, for continuity
v = randn(rng, length(x0)); v ./= norm(v)

println("="^78); println("h-sweep at theta0_up, common random numbers across all +-h evaluations"); println("="^78)
rows = h_sweep_one_direction(x0, v, ctx, base)
open(joinpath(OUTDIR, "h_sweep.csv"), "w") do io
    println(io, "h,Q_left,Q_right,Q_central,L_left,L_right,L_central,D_left,D_right,D_central,n_switches_plus,n_switches_minus,switch_mass_plus,switch_mass_minus,elapsed")
    for r in rows
        println(io, join((r.h, r.Q_left, r.Q_right, r.Q_central, r.L_left, r.L_right, r.L_central,
                           r.D_left, r.D_right, r.D_central, r.n_switches_plus, r.n_switches_minus,
                           r.switch_mass_plus, r.switch_mass_minus, r.elapsed), ","))
        println("h=", r.h, "  D_central=", r.D_central, "  L_central=", r.L_central, "  Q_central=", r.Q_central,
                "  switches(+/-)=", r.n_switches_plus, "/", r.n_switches_minus)
    end
end
println("\nWrote ", joinpath(OUTDIR, "h_sweep.csv"))

println("\n--- h=0.1 benchmark check: is it near where D_central stabilizes, or still moving? ---")
h01_idx = findfirst(r -> r.h == 0.1, rows)
h005_idx = findfirst(r -> r.h == 0.05, rows)
if h01_idx !== nothing && h005_idx !== nothing
    rel_change = abs(rows[h01_idx].D_central - rows[h005_idx].D_central) / abs(rows[h005_idx].D_central)
    println("relative change in D_central from h=0.1 to h=0.05: ", round(100*rel_change, digits=2), "%")
    println(rel_change > 0.02 ? "NOT yet converged at h=0.1 (>2% change halving h) -- h=0.1 is not a safe default here without further tightening"
                               : "roughly stable by h=0.1 (<=2% change halving h) for this direction/point")
end

println("\n" * "="^78); println("adaptive-h candidate (threshold-crossing rule only)"); println("="^78)
h_adapt = adaptive_h_candidate(x0, v, ctx)
println("adaptive h candidate = ", h_adapt, "  (vs fixed grid values ", DEFAULT_H_GRID, ")")

println("\nH-SWEEP COMPLETE")
