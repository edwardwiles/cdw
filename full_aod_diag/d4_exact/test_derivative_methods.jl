# Task §12A validation. Run: julia --project=. full_aod_diag/d4_exact/test_derivative_methods.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "winner_switching.jl"))
include(joinpath(@__DIR__, "derivative_methods.jl"))

const COMMIT = "6b6ff4a"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT)
mkpath(OUTDIR)

ctx = d4_exact_setup()
x0 = CS.pack_free(ctx.θ0_up, ctx.m)
θ0 = CS.reconstruct_full(x0, ctx.m)
base = solve_base_state(x0, ctx)

println("="^78); println("METHOD A: pathwise AD vs directional FD, straddling a KNOWN exact threshold"); println("="^78)
rng = MersenneTwister(4242)   # SAME seed as test_winner_switching.jl -- same direction, same known t_star
v = randn(rng, length(x0)); v ./= norm(v)

g_ad = method_A_pathwise_ad(x0, ctx, base)
directional_ad = dot(g_ad, x0 .* v)   # matches the multiplicative-perturbation convention: d/dt at t=0 of f(x0.*exp(t v)) ~ dot(grad, x0.*v)
println("AD directional derivative (dot(grad_AD, x0.*v)) = ", directional_ad)

v_mat = v_free_to_Amat(v, ctx)
thresh, _ = exact_tie_thresholds(θ0, v_mat, ctx)
finite_pos = sort(thresh[(thresh .> 1e-8) .& isfinite.(thresh)])
t_star = finite_pos[1]
println("nearest exact tie threshold along this EXACT direction: t_star = ", t_star)

xfun = t -> x0 .* exp.(t .* v)
println("\n--- FD directional secants of frozen_adjoint_Q at increasing h, some BELOW some ABOVE t_star ---")
for h in (t_star / 100, t_star / 10, t_star / 2, t_star * 2, t_star * 20, 0.01, 0.1)
    Qp = frozen_adjoint_Q(xfun(h), ctx, base); Qm = frozen_adjoint_Q(xfun(-h), ctx, base)
    slope = (Qp - Qm) / (2h)
    n_thresh_crossed = count(x -> abs(x) < h, vcat(finite_pos, -sort(thresh[(thresh .< -1e-8) .& isfinite.(thresh)])))
    println("h=", round(h, sigdigits=4), "  (", h < t_star ? "BELOW" : "ABOVE", " t_star, crosses ", n_thresh_crossed, " known thresholds)  FD_slope=", slope,
            "  |FD - AD| = ", abs(slope - directional_ad))
end

println("\nInterpretation: AD should agree closely with FD slopes computed at h < t_star (no threshold")
println("crossed, both miss the same switch), and should show a GROWING gap vs FD slopes at h > t_star")
println("(FD captures the switch's contribution to the secant, AD structurally cannot, by construction --")
println("MinInd! has zero a.e. derivative). This is a MEASUREMENT, not an assertion -- see printed values.")

open(joinpath(OUTDIR, "method_A_pathwise_ad_check.csv"), "w") do io
    println(io, "h,region,fd_slope,ad_directional,abs_diff")
    for h in (t_star/100, t_star/10, t_star/2, t_star*2, t_star*20, 0.001, 0.01, 0.1)
        Qp = frozen_adjoint_Q(xfun(h), ctx, base); Qm = frozen_adjoint_Q(xfun(-h), ctx, base)
        slope = (Qp - Qm) / (2h)
        println(io, h, ",", (h < t_star ? "below" : "above"), ",", slope, ",", directional_ad, ",", abs(slope - directional_ad))
    end
end
println("\nWrote ", joinpath(OUTDIR, "method_A_pathwise_ad_check.csv"))
println("\nMETHOD A DIAGNOSTIC COMPLETE")
