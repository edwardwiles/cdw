# Validate winners.jl's replicated price/winner logic against hFunction!'s own G output.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))

ctx = d4_exact_setup()

for (label, θ) in [("theta0_up", ctx.θ0_up),
                    ("perturbed +5% on A[2,3]", (θp = copy(ctx.θ0_up); θp[ctx.Aod_offset + 2 + (3-1)*ctx.D] *= 1.05; θp))]
    maxerr, ok = validate_winners_against_hFunction(θ, ctx)
    println(label, ": max|predicted G - actual G| over winner entries = ", maxerr, "  ", ok ? "PASS" : "FAIL")
    @assert ok
end

winner, price, gap = compute_winners(ctx.θ0_up, ctx)
W = size(winner, 1)
println("\nwinner array size: ", size(winner))
println("destination 1 winner distribution: ", [count(==(o), winner[:,1]) for o in 1:ctx.D], " / W=", W)
println("min gap (destination 1): ", minimum(gap[:,1]), "   fraction with gap < 1e-6: ", count(<(1e-6), gap[:,1])/W)
println("\nALL WINNER VALIDATION TESTS PASSED")
