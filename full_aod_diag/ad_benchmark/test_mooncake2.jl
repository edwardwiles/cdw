# Retest Mooncake against the NON-threaded moments variant (the earlier test accidentally
# used the original threaded EK_moments_gammanorm_directgp!, not the simplified _ts one).
include("setup_context.jl")
include("derivative_core.jl")
include("newGravityMoment_typestable.jl")
include("moments_gammanorm_typestable.jl")
using Mooncake
using DifferentiationInterface
const DI = DifferentiationInterface

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]
d = data["nTotalMoments"]; oci = data["outer_constr_index"]

function envelope_scalar_div_ts_ctx(θ::AbstractVector, ctx)
    N = size(ctx.U, 1); T = eltype(θ)
    H = zeros(T, N, ctx.d + 2)
    K = @view H[:, 1]; G = @view H[:, 3:end]
    EK_moments_gammanorm_directgp_ts!(K, G, θ, ctx.U, (γ = ctx.γobj,))
    Gj = @view H[:, 3:1+ctx.outer_constr_index]
    s = zero(T)
    @inbounds for draw in 1:N
        acc = zero(T)
        for j in 1:ctx.outer_constr_index-1
            acc += ctx.λ[j] * Gj[draw, j]
        end
        s += ctx.arg1[draw] * acc
    end
    return (1e10 / N) * s
end

println("=== Mooncake on NON-THREADED envelope_scalar_div_ts_ctx ===")
backend = DI.AutoMooncake(; config=nothing)
results = Dict{Symbol,Any}()
for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = d, outer_constr_index = oci)
    gB = ForwardDiff.gradient(θθ -> envelope_scalar_div_ts_ctx(θθ, ctx), θ)
    try
        f = θθ -> envelope_scalar_div_ts_ctx(θθ, ctx)
        t_prep = @elapsed prep = DI.prepare_gradient(f, backend, θ)
        t_grad = @elapsed gM = DI.gradient(f, prep, backend, θ)
        err = maximum(abs.(gM .- gB)) / maximum(abs.(gB))
        nnan = count(isnan, gM)
        println("  point $name: Mooncake SUCCESS  relerr=$err  nan=$nnan  prep=$(round(t_prep;digits=3))s  grad=$(round(t_grad;digits=4))s")
        results[name] = (ok=true, err=err, nnan=nnan, t_prep=t_prep, t_grad=t_grad)
    catch e
        println("  point $name: Mooncake FAILED: ", sprint(showerror, e)[1:min(400,end)])
        results[name] = (ok=false,)
    end
end
@save joinpath(@__DIR__, "mooncake_results2.jld2") results
println("DONE")
