# Full end-to-end Method C (Enzyme reverse) test, combining BOTH fixes:
#  1. enzyme_gamma_rule.jl — custom @easy_rule for SpecialFunctions.gamma (bypasses the
#     broken digamma JIT symbol, Enzyme.jl issue #2890)
#  2. moments_gammanorm_typestable.jl — the one-token type-stability fix for the dead
#     UoModel==1 branch of newGravityMoment! (IllegalTypeAnalysisException otherwise)
# against the REAL benchmark points from this session's derivative audit.
include("setup_context.jl")
include("derivative_core.jl")
include("newGravityMoment_typestable.jl")
include("moments_gammanorm_typestable.jl")
include("enzyme_gamma_rule.jl")
import Enzyme
using DifferentiationInterface
const DI = DifferentiationInterface

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]

function envelope_scalar_div_ts_ctx(θ::AbstractVector, ctx)
    N = size(ctx.U, 1)
    T = eltype(θ)
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

println("=== primal equality check (ts variant vs original) ===")
for name in (:A, :B, :C, :D)
    pt = points[name]
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = data["nTotalMoments"], outer_constr_index = data["outer_constr_index"])
    s_orig = envelope_scalar_div_ctx(pt.θ, ctx)
    s_ts = envelope_scalar_div_ts_ctx(pt.θ, ctx)
    println("  point $name: orig=$s_orig  ts=$s_ts  match=", isapprox(s_orig, s_ts; rtol=1e-12))
end

println("\n=== Method C: Enzyme reverse (with gamma rule + type-stable gravity) ===")
results = Dict{Symbol,Any}()
for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ
    ctx = (U = pp.U, γobj = pp.γ, λ = pt.λ, arg1 = pt.arg1, d = data["nTotalMoments"], outer_constr_index = data["outer_constr_index"])

    gB = ForwardDiff.gradient(θθ -> envelope_scalar_div_ts_ctx(θθ, ctx), θ)

    try
        Enzyme.API.strictAliasing!(false)
        dθ = zeros(length(θ))
        t_enzyme = @elapsed Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), envelope_scalar_div_ts_ctx,
            Enzyme.Active, Enzyme.Duplicated(copy(θ), dθ), Enzyme.Const(ctx))
        err = maximum(abs.(dθ .- gB)) / maximum(abs.(gB))
        println("  point $name: Enzyme SUCCESS!  relerr vs ForwardDiff = $err  time=$(round(t_enzyme;digits=3))s")
        results[name] = (ok=true, err=err, time=t_enzyme, grad=dθ)
    catch e
        println("  point $name: Enzyme FAILED: ", sprint(showerror, e)[1:min(400,end)])
        results[name] = (ok=false, err=NaN, time=NaN)
    end
end
@save joinpath(@__DIR__, "enzyme_fixed_results.jld2") results
println("\nTEST ENZYME FULL DONE")
