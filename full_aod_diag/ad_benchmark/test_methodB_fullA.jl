# Validate PsiObjectiveBundleImplicitMethodBFullA against production PsiObjectiveBundleImplicit
# at D=4 (fast): same objective gradient, same divergence-constraint gradient, same
# gravity-constraint gradient, same inner-solve behavior.
include("setup_context.jl")
include("derivative_core.jl")
CS.include(joinpath(dirname(@__DIR__), "PsiObjectiveBundleImplicitMethodB_fullA.jl"))

so, pp = build_ad_context()
data = JLD2.load(joinpath(@__DIR__, "benchmark_points.jld2"))
pt = data["points"][:B]
θ = pt.θ
D = data["D"]

INNEROPT = joinpath(dirname(@__DIR__), "ek_inner.opt")
OUTEROPT = joinpath(dirname(@__DIR__), "csw_outer_25.opt")

obj_std = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=pp.γ, (moments!)=EK_moments_gammanorm_directgp!,
    moments_jacobian! = error, d=data["nTotalMoments"], outer_constr_index=data["outer_constr_index"],
    inequality_index=Int64[], complement_index=[0 0], l=length(θ), U=pp.U, N=8000, lower_limit=-50.0,
    use_cached_x=false, outer_loop_opt=OUTEROPT, inner_loop_opt=INNEROPT)

ggrav = CS.make_gravity_grad(pp.γ, D)
obj_B = CS.PsiObjectiveBundleImplicitMethodBFullA(δ=1.0, find_smallest=true, γ=pp.γ, (moments!)=EK_moments_gammanorm_directgp!,
    moments_jacobian! = error, d=data["nTotalMoments"], outer_constr_index=data["outer_constr_index"],
    inequality_index=Int64[], complement_index=[0 0], l=length(θ), U=pp.U, N=8000, lower_limit=-50.0,
    use_cached_x=false, outer_loop_opt=OUTEROPT, inner_loop_opt=INNEROPT, gravity_grad=ggrav)

objSol_std, x_std, st_std = CS.inner_loop_internal(obj_std, θ)
objSol_B, x_B, st_B = CS.inner_loop_internal(obj_B, θ)
println("inner solve: std status=$st_std  B status=$st_B  objSol match=", isapprox(objSol_std, objSol_B; rtol=1e-10))
println("x match: ", isapprox(x_std, x_B; rtol=1e-8))

l = length(θ)
g_std = zeros(l); jac_std = zeros(2*l)
obj_std(x_std, g_std, θ; jac = jac_std)

g_B = zeros(l); jac_B = zeros(2*l)
obj_B(x_B, g_B, θ; jac = jac_B)

println("objective gradient match: ", isapprox(g_std, g_B; rtol=1e-8))
relerr = maximum(abs.(jac_std .- jac_B)) / maximum(abs.(jac_std))
println("constraint jac (divergence+gravity, both rows) relerr = ", relerr)
@assert relerr < 1e-8 "MethodBFullA does NOT match production!"
println("METHOD B FULL-A STRUCT VALIDATED OK")
