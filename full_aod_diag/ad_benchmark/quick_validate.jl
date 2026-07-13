# Quick standalone check: does envelope_scalar_div_ctx's ForwardDiff gradient
# match production's ACTUAL ∂c_∂θ[1,:] (divergence-budget constraint column),
# computed via the real calculate_jac_θ! + contraction path (the object's own
# callback, exactly as callbackEval_and_ConsG_outer! calls it)?
include("setup_context.jl")
include("derivative_core.jl")

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
points = data["points"]

for name in (:A, :B, :C, :D)
    pt = points[name]
    θ = pt.θ; λ = pt.λ; arg1 = pt.arg1

    obj = make_ad_obj(pp, so)
    # production path: real callback with g,θ,jac all populated (Method A)
    CS.inner_loop_internal(obj, θ)          # sets obj.H, obj.x
    obj(pt.x, Float64[], Float64[]; constr = zeros(obj.d - obj.outer_constr_index + 2))  # sync arg0/arg1 (should match pt.arg1)
    @assert isapprox(obj.arg1, arg1; rtol=1e-10) "arg1 mismatch for $name — inner solve not reproducible?"

    g = zeros(obj.l)
    jac = zeros((obj.d - obj.outer_constr_index + 2) * obj.l)
    obj(pt.x, g, θ; jac = jac)
    ∂c_∂θ = reshape(jac, obj.l, obj.d - obj.outer_constr_index + 2)'   # (2, l)
    prod_div_grad = ∂c_∂θ[1, :]   # divergence-budget constraint row

    ctx = (U = pp.U, γobj = pp.γ, λ = λ, arg1 = arg1, d = obj.d, outer_constr_index = obj.outer_constr_index)
    g_env = ForwardDiff.gradient(θ -> envelope_scalar_div_ctx(θ, ctx), θ)

    err = maximum(abs.(g_env .- prod_div_grad))
    relerr = err / max(maximum(abs.(prod_div_grad)), 1e-12)
    println("point $name: max abs err = $err   max rel err = $relerr")
end
println("QUICK VALIDATE DONE")
