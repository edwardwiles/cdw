include("setup_context.jl")
include("derivative_core.jl")
include("newGravityMoment_typestable.jl")
include("moments_gammanorm_noreshapevcat.jl")
include("enzyme_gamma_rule.jl")
import Enzyme

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]
d = data["nTotalMoments"]; oci = data["outer_constr_index"]

function envelope_scalar_div_nrv_ctx(θ::AbstractVector, ctx)
    N = size(ctx.U, 1); T = eltype(θ)
    H = zeros(T, N, ctx.d + 2)
    K = @view H[:, 1]; G = @view H[:, 3:end]
    EK_moments_gammanorm_directgp_noreshapevcat!(K, G, θ, ctx.U, (γ = ctx.γobj,))
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

# primal equality first
pt = points[:B]
ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = d, outer_constr_index = oci)
s_orig = envelope_scalar_div_ctx(pt.θ, ctx)
s_nrv = envelope_scalar_div_nrv_ctx(pt.θ, ctx)
println("primal check: orig=$s_orig  no-reshape-vcat=$s_nrv  match=", isapprox(s_orig, s_nrv; rtol=1e-12))

println("\n=== Enzyme with NO reshape(vcat(...)) construction ===")
Enzyme.API.strictAliasing!(false)
for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = d, outer_constr_index = oci)
    gB = ForwardDiff.gradient(θθ -> envelope_scalar_div_nrv_ctx(θθ, ctx), θ)
    dθ = zeros(length(θ))
    try
        t = @elapsed Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), envelope_scalar_div_nrv_ctx,
            Enzyme.Active, Enzyme.Duplicated(copy(θ), dθ), Enzyme.Const(ctx))
        nnan = count(isnan, dθ)
        if nnan == 0
            err = maximum(abs.(dθ .- gB)) / maximum(abs.(gB))
            println("  point $name: NO NaN!  relerr vs ForwardDiff = $err  time=$(round(t;digits=3))s")
        else
            println("  point $name: still $nnan NaN entries. indices=", findall(isnan,dθ))
        end
    catch e
        println("  point $name: FAILED: ", sprint(showerror,e)[1:min(200,end)])
    end
end
println("DONE")
