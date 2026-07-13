# B1 profiling: time + allocations of the inner-loop hot paths, plus one full inner solve
# and the outer θ-Jacobian path. Uses @elapsed / @allocated and stdlib Profile (no deps).

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra
using Plots, JLD2
import Profile

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
    counterType=1, counterExplicit=0, θHat=0, σHat=2.5, baseIndex=2, W=8000,
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
@unpack θ_initial, U, γ, δ_grid, outer_constr_index, inequality_index, nTotalMoments, complement_index = prep_output
θ = copy(θ_initial); N = params.Jac_W

obj = PsiObjectiveBundleImplicit(
    δ = 1.0, find_smallest = true, γ = γ, (moments!) = EK_moments!,
    moments_jacobian! = EK_moments_Jacobian!, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = length(θ), U = U, N = N, lower_limit = -50,
    outer_loop_opt = "csw_outer_loop_settings_cluster.opt", inner_loop_opt = "ek_inner_loop_options.opt")

bench(f, n) = (f(); GC.gc(); t = @elapsed (for _ in 1:n; f(); end); a = @allocated f(); (t/n, a))

println("PB ==================== B1 PROFILE ====================")
println("PB M(draws)=", obj.M, " d(moments)=", nTotalMoments, " l(theta)=", length(θ), " x(inner vars)=", outer_constr_index)

# ---- moment assembly EK_moments! (fills H) ----
Hbuf = copy(obj.H)
f_mom = () -> obj.moments!(@view(Hbuf[:,1]), CS.select_G_from_H(obj, Hbuf), θ, obj.U, obj)
t,a = bench(f_mom, 50); println("PB EK_moments!            : ", round(t*1e3;digits=3), " ms/call   alloc=", a, " B")

# fill H once for downstream
obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj); obj.H[:,2] .= 1.0

# ---- Psi!/dPsi!/ddPsi! (over M draws) ----
a0 = randn(obj.M).*0.3; a1 = similar(a0)
tP,aP = bench(()->CS.Psi!(a1,a0), 500);  println("PB Psi!  (M=",obj.M,")       : ", round(tP*1e6;digits=2), " us/call  alloc=",aP)
tdP,adP = bench(()->CS.dPsi!(a1,a0), 500);println("PB dPsi! (M=",obj.M,")       : ", round(tdP*1e6;digits=2), " us/call  alloc=",adP)
tddP,addP = bench(()->CS.ddPsi!(a1,a0),500);println("PB ddPsi!(M=",obj.M,")       : ", round(tddP*1e6;digits=2), " us/call  alloc=",addP)

# ---- inner objective+grad callable Q(x,g) (BLAS gemv + Psi!) ----
xin = zeros(outer_constr_index); gin = zeros(outer_constr_index)
tQ,aQ = bench(()->obj(xin, gin), 500); println("PB inner Q(x,g) obj+grad  : ", round(tQ*1e6;digits=2), " us/call  alloc=",aQ)

# ---- one full inner KNITRO solve ----
CS.INNER_SOLVE_COUNT[]=0; CS.INNER_INFEAS_COUNT[]=0; CS.INNER_ITERS_TOTAL[]=0
tsolve = @elapsed CS.inner_loop_internal(obj, θ)
println("PB inner_loop_internal(1) : ", round(tsolve*1e3;digits=2), " ms  (solves=",CS.INNER_SOLVE_COUNT[]," iters=",CS.INNER_ITERS_TOTAL[],")")

# ---- outer θ-Jacobian path: analytic calculate_jac_θ! + grad_k ----
obj.H_copy .= obj.H
tj,aj = bench(()->CS.calculate_jac_θ!(obj, θ), 5);   println("PB calculate_jac_θ! (ana) : ", round(tj*1e3;digits=2), " ms/call  alloc=",aj)
gg = zeros(length(θ))
tgk,agk = bench(()->CS.calculate_grad_k!(gg, obj, θ), 50); println("PB calculate_grad_k! (ana): ", round(tgk*1e3;digits=3), " ms/call  alloc=",agk)

# ---- sampled profile of a representative outer FC+grad evaluation (inner solve + jac + ift) ----
println("PB\nPB ---- sampled @profile of 20 x [inner solve + bundle θ-grad/jac] ----")
Profile.clear(); Profile.init(n=Int(1e7), delay=0.0005)
function outer_eval_once(obj, θ)
    objSol, x, ns = CS.inner_loop_internal(obj, θ)
    g = zeros(length(θ)); jac = zeros((nTotalMoments - outer_constr_index + 2)*length(θ)); c = zeros(nTotalMoments - outer_constr_index + 2)
    obj(x, g, θ, constr = c, jac = jac)
    return objSol
end
outer_eval_once(obj, θ)  # warmup
Profile.@profile (for _ in 1:20; outer_eval_once(obj, θ); end)
open("scratch_exp/results/profile_inner_flat.txt","w") do io
    Profile.print(IO=io, format=:flat, sortedby=:count, mincount=5)
end
println("PB wrote scratch_exp/results/profile_inner_flat.txt")
println("PB DONE.")
