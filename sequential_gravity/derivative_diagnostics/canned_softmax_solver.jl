# Prototype: replace invert_destination's hand-rolled LM-damped Newton (rho>0 branch) with
# NLsolve.jl's trust-region Newton, reusing the EXISTING analytic residual (share(u)-lambda_hat)
# and Jacobian (share_jacobian_smoothed) -- no new math, just a canned driver instead of the
# hand-rolled damping/line-search loop. Trust-region Newton is the standard canned analogue of
# what the custom LM damping does: bound the step within a region where the local quadratic
# model (built from a Hessian that can be badly conditioned/unreliable far from the optimum) is
# trusted, shrinking/growing that region based on how well the model predicted the actual
# improvement -- same purpose as the custom code's damping-escalate-on-reject/relax-on-accept
# logic, just via a well-tested, general implementation instead of hand-rolled bookkeeping.
using JLD2, Printf, LinearAlgebra, NLsolve
include(joinpath(@__DIR__, "..", "profiled_gravity.jl")); using .ProfiledGravity

function insert_ref(u_free::AbstractVector{T}, ref::Int, D::Int) where {T}
    u = Vector{T}(undef, D)
    j = 1
    for o in 1:D
        u[o] = (o == ref) ? zero(T) : u_free[j]
        o == ref || (j += 1)
    end
    return u
end

function invert_destination_nlsolve(log_x, p, λ̂; ref::Int=1, ρ::Real=2e-3,
        xtol=1e-12, ftol=1e-10, iterations=200, u_init=nothing, method=:trust_region)
    S, D = size(log_x)
    logp = log.(p)
    fi = free_idx(ref, D)
    u0 = u_init === nothing ? zeros(D - 1) : Float64.(u_init[fi])
    function f!(F, u_free)
        u = insert_ref(u_free, ref, D)
        share, _ = dest_share(log_x, logp, u; ρ = ρ)
        @inbounds for (k, o) in enumerate(fi); F[k] = share[o] - λ̂[o]; end
    end
    function j!(J, u_free)
        u = insert_ref(u_free, ref, D)
        st = dest_stats(log_x, logp, u; ρ = ρ)
        J .= share_jacobian_smoothed(st.rweight, st.W, st.share, ρ; ref = ref)
    end
    result = nlsolve(f!, j!, u0; method = method, xtol = xtol, ftol = ftol,
        iterations = iterations, show_trace = false)
    u = insert_ref(result.zero, ref, D)
    share, _ = dest_share(log_x, logp, u; ρ = ρ)
    err = maximum(abs.(share .- λ̂))
    return (u = u, converged = converged(result), iterations = result.iterations,
        f_calls = result.f_calls, g_calls = result.g_calls, share_err = err, result = result)
end

recon = JLD2.load(joinpath(@__DIR__, "hardmax_point1_recon.jld2"))
log_x = recon["log_x"]; p = recon["p"]; λ̂ = recon["lambda_dtest"]
S, D = size(log_x); ref = 1
@printf("S=%d D=%d rho=2e-3\n\n", S, D)

# warm up JIT
invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3)
invert_destination(log_x, p, λ̂; ref = ref, ρ = 2e-3, tol = 1e-6, maxit = 150, ls_iters = 100)

N_REP = 5
println("=== :trust_region (the canned analogue of the custom LM damping) ===")
t_tr = zeros(N_REP); r_tr = nothing
for i in 1:N_REP
    t0 = time(); global r_tr = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :trust_region)
    t_tr[i] = time() - t0
end
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e  mean_wall=%.4fs  [%s]\n",
    r_tr.converged, r_tr.iterations, r_tr.f_calls, r_tr.share_err, sum(t_tr)/N_REP, string(round.(t_tr,digits=4)))

println("\n=== :newton (plain Newton, no damping -- expected to be fragile/may fail) ===")
t_nt = zeros(N_REP); r_nt = nothing; nt_ok = true
for i in 1:N_REP
    t0 = time()
    try
        global r_nt = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :newton)
    catch e
        global nt_ok = false
        println("  FAILED: ", e)
        break
    end
    t_nt[i] = time() - t0
end
if nt_ok
    @printf("converged=%s iters=%d f_calls=%d share_err=%.3e  mean_wall=%.4fs\n",
        r_nt.converged, r_nt.iterations, r_nt.f_calls, r_nt.share_err, sum(t_nt)/N_REP)
end

println("\n=== existing hand-rolled invert_destination (baseline) ===")
t_old = zeros(N_REP); inv_old = nothing
for i in 1:N_REP
    t0 = time()
    global inv_old = invert_destination(log_x, p, λ̂; ref = ref, ρ = 2e-3, tol = 1e-6, maxit = 150, ls_iters = 100)
    t_old[i] = time() - t0
end
@printf("converged=%s iters=%d share_err=%.3e  mean_wall=%.4fs  [%s]\n",
    inv_old.converged, inv_old.iterations, inv_old.max_abs_share_error, sum(t_old)/N_REP, string(round.(t_old,digits=4)))

@printf("\n||u_trustregion - u_handrolled|| = %.3e\n", norm(r_tr.u .- inv_old.u_full))
println("\nCANNED_SOFTMAX_SOLVER DONE")
