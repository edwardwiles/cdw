# Fast correctness smoketest for the LP-duality hard-max inversion, on a small synthetic
# problem where we KNOW the true u (generate log_x, pick u_true, compute the EXACT hard-max
# shares under u_true as the target lambda_hat -- by construction lambda_hat IS achievable
# with zero split-mass since it came from a genuine winner-take-all assignment). The LP
# should recover u* with share(u*) matching lambda_hat to numerical tolerance, and u* should
# equal u_true up to the gauge + any degenerate-tie slack.
using JuMP, HiGHS, Printf, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "profiled_gravity.jl"))
using .ProfiledGravity

Random.seed!(42)
S, D = 500, 6
ref = 1
log_x = randn(S, D) .* 2.0
p = rand(S); p ./= sum(p)
u_true = randn(D); u_true[ref] = 0.0

logp = log.(p)
λ̂, _ = dest_share(log_x, logp, u_true; ρ = 0.0)
@printf("Synthetic target shares (from u_true): %s\n", string(round.(λ̂, sigdigits=4)))

function lp_hardmax_inversion(log_x::AbstractMatrix, p::AbstractVector, λ̂::AbstractVector; ref::Int=1)
    S, D = size(log_x)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, π[1:S, 1:D] >= 0)
    @objective(m, Max, sum(π[s, o] * log_x[s, o] for s in 1:S, o in 1:D))
    @constraint(m, row[s in 1:S], sum(π[s, o] for o in 1:D) == p[s])
    @constraint(m, col[o in 1:D], sum(π[s, o] for s in 1:S) == λ̂[o])
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL "LP status: $(termination_status(m))"
    u_dual = [dual(col[o]) for o in 1:D]
    u_dual .-= u_dual[ref]
    return u_dual
end

t0 = time()
u_lp = lp_hardmax_inversion(log_x, p, λ̂; ref = ref)
@printf("LP wall=%.2fs\n", time() - t0)
@printf("u_true = %s\n", string(round.(u_true, sigdigits=5)))
@printf("u_lp   = %s\n", string(round.(u_lp, sigdigits=5)))

hard_shares, _ = dest_share(log_x, logp, u_lp; ρ = 0.0)
@printf("hard-max share error at u_lp: max=%.3e\n", maximum(abs.(hard_shares .- λ̂)))
@printf("||u_lp - u_true||=%.3e (note: need not be small if some origins have ~0 share -- u is unidentified there)\n",
    norm(u_lp .- u_true))

println("SMOKETEST DONE")
