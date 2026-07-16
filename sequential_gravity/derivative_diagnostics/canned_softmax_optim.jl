using JLD2, Printf, LinearAlgebra, Optim
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

recon = JLD2.load(joinpath(@__DIR__, "hardmax_point1_recon.jld2"))
log_x = recon["log_x"]; p = recon["p"]; λ̂ = recon["lambda_dtest"]
S, D = size(log_x); ref = 1; ρ = 2e-3
logp = log.(p)
fi = free_idx(ref, D)

function obj(u_free)
    u = insert_ref(u_free, ref, D)
    st = dest_stats(log_x, logp, u; ρ = ρ)
    return st.logdenom - dot(λ̂, u)
end
function grad!(G, u_free)
    u = insert_ref(u_free, ref, D)
    st = dest_stats(log_x, logp, u; ρ = ρ)
    @inbounds for (k, o) in enumerate(fi); G[k] = st.share[o] - λ̂[o]; end
end
function hess!(H, u_free)
    u = insert_ref(u_free, ref, D)
    st = dest_stats(log_x, logp, u; ρ = ρ)
    H .= share_jacobian_smoothed(st.rweight, st.W, st.share, ρ; ref = ref)
end

u0 = zeros(D - 1)
function report(name, res, t)
    umin = insert_ref(Optim.minimizer(res), ref, D)
    share, _ = dest_share(log_x, logp, umin; ρ = ρ)
    err = maximum(abs.(share .- λ̂))
    @printf("%-28s converged=%s iters=%d f_calls=%d g_calls=%d share_err=%.3e wall=%.3fs\n",
        name, Optim.converged(res), Optim.iterations(res), Optim.f_calls(res), Optim.g_calls(res), err, t)
end

t0 = time()
res1 = optimize(obj, grad!, hess!, u0, Optim.NewtonTrustRegion(), Optim.Options(iterations=200, x_abstol=1e-12, f_abstol=1e-14))
report(":NewtonTrustRegion (Optim.jl, cold)", res1, time() - t0)

t0 = time()
res2 = optimize(obj, grad!, hess!, u0, Optim.Newton(), Optim.Options(iterations=200, x_abstol=1e-12, f_abstol=1e-14))
report(":Newton (Optim.jl, plain, cold)", res2, time() - t0)

u_seed = recon["u_rho_solution"][fi]
t0 = time()
res3 = optimize(obj, grad!, hess!, u_seed, Optim.NewtonTrustRegion(), Optim.Options(iterations=200, x_abstol=1e-12, f_abstol=1e-14))
report(":NewtonTrustRegion (warm)", res3, time() - t0)

println("\nCANNED_SOFTMAX_OPTIM DONE")
