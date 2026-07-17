# Task §11 validation. Run: julia --project=. full_aod_diag/d4_exact/test_winner_switching.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))

const COMMIT = "45ac6c6"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
θ0 = CS.reconstruct_full(x0, ctx.m)
base = solve_base_state(x0, ctx)

println("="^78); println("TEST 1: switch stats between base and a moderate (h=0.1) perturbation"); println("="^78)
rng = MersenneTwister(4242)
v = randn(rng, length(x0)); v ./= norm(v)
winner0, _, _ = compute_winners(θ0, ctx)
θ_pert = CS.reconstruct_full(x0 .+ 0.1 .* (x0 .* v), ctx.m)
winner_pert, _, _ = compute_winners(θ_pert, ctx)
r1 = evaluate_fullA(x0, ctx; cache = nothing)
stats = switch_stats(winner0, winner_pert; m_weights = fill(1.0, size(winner0,1)))  # base weights not yet recovered here; use uniform as a placeholder base_mass check
println("n_switches = ", stats.n_switches, " / ", length(winner0), "  (", round(100*stats.n_switches/length(winner0), digits=2), "%)")
println("by destination: ", stats.by_destination)
println("base_mass (fraction of draw-destination cells switched) = ", stats.base_mass)
@assert stats.n_switches > 0 "h=0.1 perturbation produced ZERO winner switches -- direction v may be gravity-degenerate or too small; test is uninformative"
println("PASS: h=0.1 perturbation produces a measurable, nonzero switching mass -- confirms the FD experiments at this h ARE probing the hard-winner boundary, not just smooth curvature")

println("\n" * "="^78); println("TEST 2: exact tie thresholds along a random direction"); println("="^78)
v_mat = v_free_to_Amat(v, ctx)
thresh, winner_check = exact_tie_thresholds(θ0, v_mat, ctx)
@assert winner_check == winner0 "exact_tie_thresholds' own winner recomputation disagrees with compute_winners -- bug"
finite_pos = sort(thresh[(thresh .> 1e-10) .& isfinite.(thresh)])
finite_neg = sort(thresh[(thresh .< -1e-10) .& isfinite.(thresh)], rev=true)
println("number of finite positive thresholds = ", length(finite_pos), " / ", length(thresh))
println("smallest positive threshold  = ", finite_pos[1])
println("smallest |negative| threshold = ", finite_neg[1])
println("min/max ALL finite thresholds = ", minimum(vcat(finite_pos,-finite_neg)), " / ", maximum(vcat(finite_pos,abs.(finite_neg))))

println("\n--- verify smallest positive threshold ACTUALLY produces a switch just past it ---")
t_check = finite_pos[1]
xfun = t -> x0 .* exp.(t .* v)   # EXACT exponential family, matching exact_tie_thresholds' own derivation
θ_before = CS.reconstruct_full(xfun(t_check - 1e-7), ctx.m)
θ_after  = CS.reconstruct_full(xfun(t_check + 1e-7), ctx.m)
w_before, _, _ = compute_winners(θ_before, ctx)
w_after, _, _  = compute_winners(θ_after, ctx)
n_switch_at_threshold = count(w_before .!= w_after)
println("winner-array diffs straddling t*=", t_check, " (+-1e-7): ", n_switch_at_threshold, " cell(s) switched")
@assert n_switch_at_threshold >= 1 "predicted smallest tie threshold does not correspond to an ACTUAL winner switch -- exact_tie_thresholds formula bug"
println("PASS: exact tie threshold formula correctly predicts a real winner switch")

println("\n" * "="^78); println("TEST 3: kink test -- does mean(G)/Q_adj jump at an exact single-draw tie?"); println("="^78)
kt = kink_test(x0, v, ctx, base; delta = 1e-9)
println("t_star (smallest positive threshold) = ", kt.t_star)
println("other thresholds within 2x t_star: ", kt.n_thresholds_below_tstar_x2, " (expect 1 -- just t_star itself; if >1, multiple draws tie almost simultaneously)")
println("G jump AT threshold (delta=1e-9 straddle)   = ", kt.G_jump_at_threshold)
println("G jump AT control point (same delta)         = ", kt.G_jump_at_control)
println("ratio (threshold/control)                    = ", kt.G_jump_at_threshold / max(kt.G_jump_at_control, 1e-300))
println("Q_adj jump AT threshold  = ", kt.Q_jump_at_threshold)
println("Q_adj jump AT control    = ", kt.Q_jump_at_control)
println("ratio (threshold/control) = ", kt.Q_jump_at_threshold / max(kt.Q_jump_at_control, 1e-300))
@assert kt.G_jump_at_threshold > 10 * kt.G_jump_at_control "no detectable excess jump in mean(G) at the exact tie threshold vs a control point -- either the kink is too small to detect at this delta, or genuinely smooth (report exactly which, don't assume)"
println("PASS: mean(G) (and hence every downstream scalar) has a MEASURABLE non-smooth kink exactly at the")
println("      predicted tie threshold, not merely smooth curvature -- confirms the finite-draw function is")
println("      genuinely kinked (task's explicit question), at least locally around single ties.")

open(joinpath(OUTDIR, "winner_switching_summary.csv"), "w") do io
    println(io, "metric,value")
    println(io, "n_switches_h01,", stats.n_switches)
    println(io, "t_star,", kt.t_star)
    println(io, "G_jump_at_threshold,", kt.G_jump_at_threshold)
    println(io, "G_jump_at_control,", kt.G_jump_at_control)
    println(io, "Q_jump_at_threshold,", kt.Q_jump_at_threshold)
    println(io, "Q_jump_at_control,", kt.Q_jump_at_control)
end
println("\nWrote ", joinpath(OUTDIR, "winner_switching_summary.csv"))
println("\nALL WINNER-SWITCHING TESTS PASSED")
