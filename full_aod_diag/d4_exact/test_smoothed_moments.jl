# Task §12E validation. Run: julia --project=. full_aod_diag/d4_exact/test_smoothed_moments.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "smoothed_moments.jl"))

const COMMIT = "9e03706"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
θ0 = CS.reconstruct_full(x0, ctx.m)
base = solve_base_state(x0, ctx)

println("="^78); println("TEST 1: smoothed G -> hard G as tuner -> -infinity"); println("="^78)
W = size(ctx.U, 1); K_hard = zeros(W); G_hard = zeros(W, ctx.nTotalMoments)
ctx.obj.moments!(K_hard, G_hard, θ0, ctx.U, ctx.obj)
for tuner in (-1.0, -10.0, -100.0, -1000.0, -1e5)
    G_smooth = smoothed_factual_G(θ0, ctx, tuner)
    err = maximum(abs.(G_smooth .- G_hard[:, 1:ctx.D^2]))
    println("tuner=", tuner, "  max|G_smooth - G_hard| = ", err)
end
println("(expect monotone decrease toward 0 as tuner -> -infinity; measurement, not a hard assert)")

println("\n" * "="^78); println("TEST 2: smoothed_frozen_adjoint_Q reduces to frozen_adjoint_Q as tuner->-inf"); println("="^78)
Q_hard = frozen_adjoint_Q(x0, ctx, base)
for tuner in (-10.0, -100.0, -1000.0, -1e5)
    Q_smooth = smoothed_frozen_adjoint_Q(x0, ctx, base, tuner)
    println("tuner=", tuner, "  Q_smooth=", Q_smooth, "  Q_hard=", Q_hard, "  diff=", abs(Q_smooth - Q_hard))
end
@assert abs(smoothed_frozen_adjoint_Q(x0, ctx, base, -1e6) - Q_hard) < 1e-4 "smoothed_frozen_adjoint_Q does not converge to the hard frozen_adjoint_Q as tuner -> -infinity -- construction bug"
println("PASS: smoothed collapses to hard at extreme tuner")

println("\n" * "="^78); println("TEST 3: smoothing genuinely removes the kink at the known tie threshold"); println("="^78)
rng = MersenneTwister(4242)   # SAME direction/threshold as winner_switching.jl's kink test
v = randn(rng, length(x0)); v ./= norm(v)
v_mat = v_free_to_Amat(v, ctx)
thresh, _ = exact_tie_thresholds(θ0, v_mat, ctx)
finite_pos = sort(thresh[(thresh .> 1e-8) .& isfinite.(thresh)])
t_star = finite_pos[1]
xfun = t -> x0 .* exp.(t .* v)
delta = 1e-9
Qp_hard = frozen_adjoint_Q(xfun(t_star + delta), ctx, base); Qm_hard = frozen_adjoint_Q(xfun(t_star - delta), ctx, base)
println("HARD jump at threshold: ", abs(Qp_hard - Qm_hard))
for tuner in (-100.0, -1000.0)
    Qp_s = smoothed_frozen_adjoint_Q(xfun(t_star + delta), ctx, base, tuner)
    Qm_s = smoothed_frozen_adjoint_Q(xfun(t_star - delta), ctx, base, tuner)
    println("SMOOTHED (tuner=", tuner, ") jump at threshold: ", abs(Qp_s - Qm_s))
end
# at a MUCH wider straddle (delta=1e-3, comparable to the smoothing's own natural bandwidth), the
# smoothed function should vary continuously/gradually rather than show a step
Qp_s_wide = smoothed_frozen_adjoint_Q(xfun(t_star + 1e-3), ctx, base, -100.0)
Qm_s_wide = smoothed_frozen_adjoint_Q(xfun(t_star - 1e-3), ctx, base, -100.0)
println("SMOOTHED (tuner=-100) wide-delta(1e-3) change across threshold: ", abs(Qp_s_wide - Qm_s_wide), " (expect comparable to a generic nearby interval, not a concentrated jump)")

open(joinpath(OUTDIR, "smoothing_check.csv"), "w") do io
    println(io, "quantity,value")
    println(io, "hard_jump_at_threshold,", abs(Qp_hard - Qm_hard))
    println(io, "smoothed_tuner100_jump_at_threshold,", abs(smoothed_frozen_adjoint_Q(xfun(t_star+delta),ctx,base,-100.0) - smoothed_frozen_adjoint_Q(xfun(t_star-delta),ctx,base,-100.0)))
end
println("\nWrote ", joinpath(OUTDIR, "smoothing_check.csv"))
println("\nMETHOD E (DERIVATIVE-ONLY SMOOTHING) DIAGNOSTIC COMPLETE")
println("NOTE: labeled heuristic/inexact per task sec 12E -- NOT proposed as the exact-hard outer solver's gradient.")
