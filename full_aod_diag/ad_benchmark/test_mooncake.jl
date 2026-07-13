# Test Mooncake reverse-mode AD on the ORIGINAL (unpatched) envelope_scalar_div_ctx first
# (does it even need the type-stability fix Enzyme needed?), then with the fixes if not.
include("setup_context.jl")
include("derivative_core.jl")
using Mooncake
using DifferentiationInterface
const DI = DifferentiationInterface

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]

println("=== Mooncake on ORIGINAL (unpatched) envelope_scalar_div_ctx ===")
results = Dict{Symbol,Any}()
backend = DI.AutoMooncake(; config=nothing)
for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = data["nTotalMoments"], outer_constr_index = data["outer_constr_index"])
    gB = ForwardDiff.gradient(θθ -> envelope_scalar_div_ctx(θθ, ctx), θ)
    try
        f = θθ -> envelope_scalar_div_ctx(θθ, ctx)
        t_prep = @elapsed prep = DI.prepare_gradient(f, backend, θ)
        t_grad = @elapsed gM = DI.gradient(f, prep, backend, θ)
        err = maximum(abs.(gM .- gB)) / maximum(abs.(gB))
        nnan = count(isnan, gM)
        println("  point $name: Mooncake SUCCESS  relerr=$err  nan_count=$nnan  prep=$(round(t_prep;digits=3))s  grad=$(round(t_grad;digits=4))s")
        results[name] = (ok=true, err=err, nnan=nnan, t_prep=t_prep, t_grad=t_grad)
    catch e
        println("  point $name: Mooncake FAILED: ", sprint(showerror, e)[1:min(400,end)])
        results[name] = (ok=false,)
    end
end
@save joinpath(@__DIR__, "mooncake_results.jld2") results
println("MOONCAKE TEST DONE")
