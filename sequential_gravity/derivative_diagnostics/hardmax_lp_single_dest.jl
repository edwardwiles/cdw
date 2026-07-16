# Step 2: hard-max destination inversion via LP duality (transportation problem).
#
# Theory: share(u) at rho=0 assigns each draw s (mass p[s]) entirely to
# argmax_o(u[o]+log_x[s,o]). Finding u with share(u)=lambda_hat (exactly, allowing
# infinitesimal splitting of draws that are EXACTLY tied) is precisely the dual of the
# max-weight transportation LP:
#   max_{pi>=0}  sum_{s,o} pi[s,o]*log_x[s,o]
#   s.t.         sum_o pi[s,o] = p[s]      (per-draw mass fully assigned)
#                sum_s pi[s,o] = lambda_hat[o]   (target share achieved EXACTLY)
# The dual variable on the second (per-origin) constraint IS u[o] (up to the additive
# gauge fixed by pinning u[ref]=0), and by strong LP duality this ALWAYS has a solution
# (the primal is always feasible -- e.g. the independence coupling pi[s,o]=p[s]*lambda_hat[o]
# -- and bounded, since log_x is finite), so a hard-max-achieving u EXISTS for ANY
# lambda_hat with lambda_hat>=0, sum=1, contrary to the "generically unachievable" worry in
# the handoff prompt -- as long as fractional splitting of tied draws is allowed. At most
# D-1 draws need to be split in a basic optimal solution (standard transportation-polytope
# fact), so evaluating share(u*) under the STRICT (no-split, winner-take-all) hard-argmax
# rule will differ from lambda_hat by, at most, the total probability mass of those <=D-1
# split draws -- this script measures that gap directly.
#
#   julia --project=. sequential_gravity/derivative_diagnostics/hardmax_lp_single_dest.jl
using JuMP, HiGHS, JLD2, Printf, LinearAlgebra
include(joinpath(@__DIR__, "..", "profiled_gravity.jl"))
using .ProfiledGravity

recon = JLD2.load(joinpath(@__DIR__, "hardmax_point1_recon.jld2"))
log_x = recon["log_x"]; p = recon["p"]; λ̂ = recon["lambda_dtest"]; dtest = recon["dtest"]
u_rho = recon["u_rho_solution"]
S, D = size(log_x)
ref = 1
@printf("S=%d D=%d dtest=%d\n", S, D, dtest)

function lp_hardmax_inversion(log_x::AbstractMatrix, p::AbstractVector, λ̂::AbstractVector; ref::Int=1)
    S, D = size(log_x)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, π[1:S, 1:D] >= 0)
    @objective(m, Max, sum(π[s, o] * log_x[s, o] for s in 1:S, o in 1:D))
    @constraint(m, row[s in 1:S], sum(π[s, o] for o in 1:D) == p[s])
    @constraint(m, col[o in 1:D], sum(π[s, o] for s in 1:S) == λ̂[o])
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL "LP did not solve to optimality: $(termination_status(m))"
    u_dual = [dual(col[o]) for o in 1:D]
    u_dual .-= u_dual[ref]
    # count how many draws are "split" (>1 origin with strictly positive pi) at the optimum --
    # a basic optimal solution has at most S+D-1 nonzero pi entries, i.e. at most D-1 draws split
    πval = value.(π)
    n_active_per_draw = [count(>(1e-9), πval[s, :]) for s in 1:S]
    n_split = count(>(1), n_active_per_draw)
    split_mass = sum(p[s] for s in 1:S if n_active_per_draw[s] > 1)
    return (u = u_dual, n_split = n_split, split_mass = split_mass, obj = objective_value(m),
            status = termination_status(m))
end

t0 = time()
res = lp_hardmax_inversion(log_x, p, λ̂; ref = ref)
twall = time() - t0
@printf("LP solve: status=%s wall=%.1fs  n_split_draws=%d  split_mass=%.3e\n",
    res.status, twall, res.n_split, res.split_mass)

logp = log.(p)
hard_shares, _ = dest_share(log_x, logp, res.u; ρ = 0.0)
hard_err = maximum(abs.(hard_shares .- λ̂))
@printf("LP hard-max u*: max|model-empirical| (STRICT hard-argmax, no split) = %.3e\n", hard_err)
@printf("  (compare: split_mass=%.3e -- these should be the same order)\n", res.split_mass)

# Compare to the existing rho=2e-3 solution
u_diff = norm(res.u .- u_rho)
u_diff_rel = u_diff / max(norm(u_rho), 1e-12)
@printf("\n||u_LP - u_rho2e-3||_2 = %.4e  (relative = %.4e)\n", u_diff, u_diff_rel)
@printf("u_LP[1:5]  = %s\n", string(round.(res.u[1:5], sigdigits=6)))
@printf("u_rho[1:5] = %s\n", string(round.(u_rho[1:5], sigdigits=6)))

JLD2.save(joinpath(@__DIR__, "hardmax_lp_point1_dtest.jld2"),
    "u_lp", res.u, "n_split", res.n_split, "split_mass", res.split_mass,
    "hard_err_lp", hard_err, "u_diff_from_rho", u_diff)

println("\nLP SINGLE-DEST DONE")
