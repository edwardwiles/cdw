# Follow-up to step 5: Enzyme's OWN error message on the Const-mutable-scratch pattern
# suggested a specific workaround (`set_runtime_activity`). Test whether it (a) resolves
# the compile-time error, and (b) if so, whether the resulting gradient is then CORRECT
# or silently NaN/wrong -- i.e. does turning on runtime activity turn a loud error into
# the same kind of silent wrong-answer that the real moments!.jl pipeline produces?
using ForwardDiff, LinearAlgebra, Random
import Enzyme

mutable struct ScratchCtx
    U::Matrix{Float64}
    scratch::Matrix{Float64}
end

function f_scratch(theta, ctx::ScratchCtx)
    mu = theta[1]
    extra = theta[1:1] .* 1.0
    if eltype(extra) === Float64 && size(ctx.scratch) == size(ctx.U)
        UPow = ctx.scratch
    else
        UPow = zeros(eltype(extra), size(ctx.U))
    end
    @. UPow = ctx.U ^ (-mu)
    return sum(UPow)
end

Random.seed!(7)
U = rand(20, 5) .+ 0.5
ctx = ScratchCtx(U, zeros(size(U)))
theta0 = [0.2]

g_fd = ForwardDiff.gradient(θ -> f_scratch(θ, ctx), theta0)
println(">>> ForwardDiff: ", g_fd)

g_enz = zeros(1)
try
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), theta -> f_scratch(theta, ctx),
                     Enzyme.Active, Enzyme.Duplicated(theta0, g_enz))
    nnan = count(isnan, g_enz)
    println(">>> Enzyme with set_runtime_activity(Reverse): ", g_enz, "  finite=", all(isfinite, g_enz), "  nan_count=", nnan)
    println(">>> matches ForwardDiff: ", isapprox(g_enz, g_fd; rtol=1e-8))
catch e
    println(">>> STILL FAILED even with set_runtime_activity: ", sprint(showerror, e)[1:min(500,end)])
end
