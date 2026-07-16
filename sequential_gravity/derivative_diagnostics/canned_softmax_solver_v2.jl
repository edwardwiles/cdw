using JLD2, Printf, LinearAlgebra, NLsolve, LineSearches
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
        xtol=1e-12, ftol=1e-10, iterations=200, u_init=nothing, method=:trust_region, linesearch=nothing, kwargs...)
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
    result = linesearch === nothing ?
        nlsolve(f!, j!, u0; method = method, xtol = xtol, ftol = ftol, iterations = iterations, show_trace = false, kwargs...) :
        nlsolve(f!, j!, u0; method = method, linesearch = linesearch, xtol = xtol, ftol = ftol, iterations = iterations, show_trace = false, kwargs...)
    u = insert_ref(result.zero, ref, D)
    share, _ = dest_share(log_x, logp, u; ρ = ρ)
    err = maximum(abs.(share .- λ̂))
    return (u = u, converged = converged(result), iterations = result.iterations,
        f_calls = result.f_calls, share_err = err)
end

recon = JLD2.load(joinpath(@__DIR__, "hardmax_point1_recon.jld2"))
log_x = recon["log_x"]; p = recon["p"]; λ̂ = recon["lambda_dtest"]
u_seed = recon["u_rho_solution"]
S, D = size(log_x); ref = 1

invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3)  # warmup

println("=== :newton + BackTracking linesearch (canned globalized Newton, no trust region) ===")
t0 = time()
r = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :newton, linesearch = BackTracking())
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e wall=%.3fs\n", r.converged, r.iterations, r.f_calls, r.share_err, time()-t0)

println("\n=== :newton + StrongWolfe linesearch ===")
t0 = time()
r2 = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :newton, linesearch = StrongWolfe())
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e wall=%.3fs\n", r2.converged, r2.iterations, r2.f_calls, r2.share_err, time()-t0)

println("\n=== :trust_region, warm-started from the hand-rolled rho=2e-3 solution ===")
t0 = time()
r3 = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :trust_region, u_init = u_seed)
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e wall=%.3fs\n", r3.converged, r3.iterations, r3.f_calls, r3.share_err, time()-t0)

println("\n=== :trust_region, larger initial radius (factor=10.0) ===")
t0 = time()
r4 = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :trust_region, factor = 10.0)
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e wall=%.3fs\n", r4.converged, r4.iterations, r4.f_calls, r4.share_err, time()-t0)

println("\n=== :trust_region, smaller initial radius (factor=0.1) ===")
t0 = time()
r5 = invert_destination_nlsolve(log_x, p, λ̂; ref = ref, ρ = 2e-3, method = :trust_region, factor = 0.1)
@printf("converged=%s iters=%d f_calls=%d share_err=%.3e wall=%.3fs\n", r5.converged, r5.iterations, r5.f_calls, r5.share_err, time()-t0)

println("\nV2 DONE")
