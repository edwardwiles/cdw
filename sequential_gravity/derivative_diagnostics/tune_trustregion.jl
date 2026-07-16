# Tuning sweep for Optim.jl NewtonTrustRegion as a candidate replacement for
# invert_destination's hand-rolled LM-damped Newton (rho>0 branch only). Tests across SEVERAL
# destinations (not just one) to avoid overfitting conclusions to a single case, and sweeps
# tolerance to see whether 1e-8-level precision costs meaningfully more than 1e-6/1e-7.
#
# No KNITRO needed: reuses Point 1's already-recovered p (theta/point-specific, NOT
# destination-specific) from hardmax_point1_recon.jld2, and gets lambdaData (destination
# targets, NOT recover_lfd-dependent -- comes straight from calibrated GE data) via the normal
# (KNITRO-free unless recover_lfd is explicitly called, which this script never does) include.
ENV["SKIP_BATCH_LOOP"] = "true"
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using JLD2, Printf, LinearAlgebra, Optim

recon = JLD2.load(joinpath(@__DIR__, "hardmax_point1_recon.jld2"))
p = recon["p"]; log_x = recon["log_x"]
@assert log_x == build_log_x(Uσ, recon["theta"][1])
logp = log.(p)
fi = free_idx(ref, D)

function insert_ref(u_free::AbstractVector{T}, ref::Int, D::Int) where {T}
    u = Vector{T}(undef, D)
    j = 1
    for o in 1:D
        u[o] = (o == ref) ? zero(T) : u_free[j]
        o == ref || (j += 1)
    end
    return u
end

function solve_trustregion(dtarget::AbstractVector; x_abstol=1e-10, f_abstol=1e-14,
        initial_delta=1.0, delta_hat=100.0, u_init=nothing, iterations=200)
    obj(u_free) = begin
        u = insert_ref(u_free, ref, D)
        st = dest_stats(log_x, logp, u; ρ = ρ)
        st.logdenom - dot(dtarget, u)
    end
    grad!(G, u_free) = begin
        u = insert_ref(u_free, ref, D)
        st = dest_stats(log_x, logp, u; ρ = ρ)
        @inbounds for (k, o) in enumerate(fi); G[k] = st.share[o] - dtarget[o]; end
    end
    hess!(H, u_free) = begin
        u = insert_ref(u_free, ref, D)
        st = dest_stats(log_x, logp, u; ρ = ρ)
        H .= share_jacobian_smoothed(st.rweight, st.W, st.share, ρ; ref = ref)
    end
    u0 = u_init === nothing ? zeros(D - 1) : u_init[fi]
    method = Optim.NewtonTrustRegion(initial_delta = initial_delta, delta_hat = delta_hat)
    res = optimize(obj, grad!, hess!, u0, method,
        Optim.Options(iterations = iterations, x_abstol = x_abstol, f_abstol = f_abstol))
    u = insert_ref(Optim.minimizer(res), ref, D)
    share, _ = dest_share(log_x, logp, u; ρ = ρ)
    err = maximum(abs.(share .- dtarget))
    return (u = u, converged = Optim.converged(res), iters = Optim.iterations(res),
        f_calls = Optim.f_calls(res), share_err = err)
end

TEST_DESTS = [omitted[1], omitted[5], omitted[10], omitted[15], omitted[19]]
@printf("Test destinations: %s\n\n", string(TEST_DESTS))

println("="^100)
println(">>> PART A: baseline comparison (hand-rolled vs Optim NewtonTrustRegion default), cold start, tol~1e-10")
println("="^100)
@printf("%4s | %-24s %6s %8s %10s %8s | %-24s %6s %8s %10s %8s\n",
    "dest", "handrolled", "iters", "wall", "share_err", "conv", "trustregion(default)", "iters", "wall", "share_err", "conv")
for d in TEST_DESTS
    λ̂ = λData[:, d]
    t0 = time()
    inv = invert_destination(log_x, p, λ̂; ref = ref, ρ = ρ, tol = 1e-8, maxit = 150, ls_iters = 100)
    t_hr = time() - t0
    t0 = time()
    r = solve_trustregion(λ̂; x_abstol=1e-10, f_abstol=1e-14)
    t_tr = time() - t0
    @printf("%4d | %-24s %6d %8.3f %10.3e %8s | %-24s %6d %8.3f %10.3e %8s\n",
        d, "", inv.iterations, t_hr, inv.max_abs_share_error, inv.converged,
        "", r.iters, t_tr, r.share_err, r.converged)
end

println("\n" * "="^100)
println(">>> PART B: tolerance sweep on Optim NewtonTrustRegion (does 1e-8 cost much more than 1e-6/1e-7?)")
println("="^100)
for d in TEST_DESTS
    λ̂ = λData[:, d]
    @printf("dest=%d:\n", d)
    for (x_abstol, f_abstol, label) in [(1e-6,1e-10,"loose(1e-6)"), (1e-7,1e-12,"med(1e-7)"), (1e-10,1e-14,"tight(1e-10)")]
        t0 = time()
        r = solve_trustregion(λ̂; x_abstol=x_abstol, f_abstol=f_abstol)
        t = time() - t0
        @printf("    %-14s iters=%3d f_calls=%3d wall=%.3fs share_err=%.3e converged=%s\n",
            label, r.iters, r.f_calls, t, r.share_err, r.converged)
    end
end

println("\n" * "="^100)
println(">>> PART C: initial_delta sensitivity (cold start)")
println("="^100)
for d in TEST_DESTS[1:3]
    λ̂ = λData[:, d]
    @printf("dest=%d:\n", d)
    for idelta in [0.1, 1.0, 3.0, 10.0]
        t0 = time()
        r = solve_trustregion(λ̂; initial_delta=idelta)
        t = time() - t0
        @printf("    initial_delta=%-6.1f iters=%3d wall=%.3fs share_err=%.3e converged=%s\n",
            idelta, r.iters, t, r.share_err, r.converged)
    end
end

println("\nTUNE_TRUSTREGION DONE")
