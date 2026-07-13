# Standalone test: (1) does moments! accept ForwardDiff Duals? (A1)
#                  (2) autodiff moment-Jacobian vs analytic EK_moments_Jacobian! (A2 cross-check)

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra
using Plots, JLD2

include("setup/include_setup.jl")
include("prestep/include_prestep.jl")
include("prepare_cc/include_prepare_cc.jl")
include("moments/include_moments.jl")
include("cc_algo/include_cc_algo.jl")
include("lfd/include_lfd.jl")
include("misc/include_misc.jl")
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity

params = (
    server=1, user=2, fakeData=1, DFake=4, seedFakeData=889,
    counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=8000,
    seedU=888, importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0,
    IndMomentOrder=5, θConstant=0, gravMoment=0, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05,
    δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=1,
    calc_δ_star_initial=1, Jac_W=8000, theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

setup_output = master_setup(params)
useParams = (; params..., D=setup_output.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(useParams)
prestep_output = master_prestep(setup_output.data, setup_output.counters, useParams)
prep_output = master_prepare_cc(setup_output.data, setup_output.counters, prestep_output, useParams)

@unpack θ_initial, U, γ, numMoments, δ_grid, outer_constr_index, inequality_index, nTotalMoments, complement_index = prep_output
θ = copy(θ_initial)
N = useParams.Jac_W
println("XX l (θ dim) = ", length(θ), "   nTotalMoments d = ", nTotalMoments, "   N = ", N)

function build_obj(jacfun)
    PsiObjectiveBundleImplicit(
        δ = 1.0, find_smallest = true, γ = γ,
        (moments!) = EK_moments!, moments_jacobian! = jacfun,
        d = nTotalMoments, outer_constr_index = outer_constr_index,
        inequality_index = inequality_index, complement_index = complement_index,
        l = length(θ), U = U, N = N, lower_limit = -50,
        outer_loop_opt = "csw_outer_loop_settings_cluster.opt",
        inner_loop_opt = "ek_inner_loop_options.opt",
    )
end

obj_ana = build_obj(EK_moments_Jacobian!)
obj_ad  = build_obj(error)

# ---------- (1) grad_k: analytic vs autodiff (also exercises Duals through moments!) ----------
println("XX\nXX === A1/A2: grad_k (dk/dtheta) analytic vs autodiff ===")
g_ana = zeros(length(θ)); g_ad = zeros(length(θ))
CS.calculate_grad_k!(g_ana, obj_ana, θ)                 # analytic
t_ana_gk = @elapsed CS.calculate_grad_k!(g_ana, obj_ana, θ)
dual_ok = false
try
    CS.calculate_grad_k_autodiff!(g_ad, obj_ad, θ)      # autodiff (Duals through moments!)
    global dual_ok = true
catch e
    println("XX grad_k autodiff FAILED under Duals:")
    showerror(stderr, e, catch_backtrace()); println()
end
if dual_ok
    t_ad_gk = @elapsed CS.calculate_grad_k_autodiff!(g_ad, obj_ad, θ)
    println("XX A1 Dual smoke test: PASSED (moments! ran under ForwardDiff)")
    println("XX time analytic grad_k = ", round(t_ana_gk*1e3; digits=3), " ms   autodiff = ", round(t_ad_gk*1e3; digits=3), " ms")
    println("XX max abs diff grad_k  = ", maximum(abs.(g_ana .- g_ad)))
    println("XX analytic grad_k = ", round.(g_ana; digits=8))
    println("XX autodiff grad_k = ", round.(g_ad;  digits=8))
end

# ---------- (2) moment Jacobian jac_h analytic vs autodiff ----------
if dual_ok
    println("XX\nXX === A2: full moment Jacobian jac_h analytic vs autodiff ===")
    obj_ana.moments!(@view(obj_ana.H[:,1]), CS.select_G_from_H(obj_ana, obj_ana.H), θ, obj_ana.U, obj_ana)
    obj_ad.moments!(@view(obj_ad.H[:,1]),   CS.select_G_from_H(obj_ad,  obj_ad.H),  θ, obj_ad.U,  obj_ad)
    obj_ana.H_copy .= obj_ana.H
    obj_ad.H_copy  .= obj_ad.H

    CS.calculate_jac_θ!(obj_ana, θ); CS.calculate_jac_θ!(obj_ad, θ)
    t_ana_j = @elapsed CS.calculate_jac_θ!(obj_ana, θ)
    t_ad_j  = @elapsed CS.calculate_jac_θ!(obj_ad,  θ)
    println("XX time analytic jac = ", round(t_ana_j*1e3; digits=3), " ms   autodiff jac = ", round(t_ad_j*1e3; digits=3), " ms")

    Ja = obj_ana.jac_h[1:N, :, :]; Jd = obj_ad.jac_h[1:N, :, :]
    println("XX overall max abs diff jac_h   = ", maximum(abs.(Ja .- Jd)))
    println("XX   col1 (dk/dtheta) max diff   = ", maximum(abs.(Ja[:,1,:] .- Jd[:,1,:])))
    println("XX   cols3:end (dg/dtheta) maxdif= ", maximum(abs.(Ja[:,3:end,:] .- Jd[:,3:end,:])))
    dg = abs.(Ja[:,3:end,:] .- Jd[:,3:end,:])
    println("XX   per g-moment max diff (>1e-8):")
    for m in 1:size(dg,2)
        mx = maximum(dg[:,m,:])
        mx > 1e-8 && println("XX     g-moment $m : max diff = ", round(mx; sigdigits=6), "   (mean|ana|=", round(mean(abs.(Ja[:,2+m,:])); sigdigits=4), ")")
    end
    dgt = dropdims(maximum(dg; dims=(1,2)); dims=(1,2))
    println("XX   per theta-col max diff (>1e-8): ", [(i, round(dgt[i]; sigdigits=4)) for i in 1:length(dgt) if dgt[i] > 1e-8])
    println("XX mean abs analytic jac_h = ", round(mean(abs.(Ja)); sigdigits=5), "   mean abs autodiff = ", round(mean(abs.(Jd)); sigdigits=5))
end
println("XX DONE.")
