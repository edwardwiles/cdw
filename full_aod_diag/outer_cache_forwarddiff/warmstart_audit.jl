# ============================================================================
# Warm-start audit (mega-prompt §8). Distinguishes a WARM START (seed a new,
# genuinely different theta's inner solve with the PREVIOUS theta's optimal
# (ζ,λ) as initial value, still a full re-solve) from a CACHE HIT (reuse the
# result without re-solving at all, only valid at the EXACT same theta).
#
# Benchmarks, at 8 successive small steps along a fixed random direction from
# the D=4 audit's starting point:
#   1. cold inner solves      (obj.use_cached_x=false -> initial value = zeros)
#   2. warm-started solves    (obj.use_cached_x=true  -> initial value = obj.x
#                               from the immediately preceding solve)
#   3. repeat-theta solves    (solve TWICE at the exact same theta, back to
#                               back, with warm start on -> this is what an
#                               exact-point CACHE would avoid doing at all)
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/warmstart_audit.jl
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ROOT = dirname(dirname(@__DIR__))
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(dirname(@__DIR__), "moments_gammanorm.jl"))

const OUT = @__DIR__
const INNEROPT = joinpath(dirname(@__DIR__), "ek_inner.opt")
const OUTEROPT = joinpath(@__DIR__, "ek_outer_loop_options_audit.opt")  # only used for obj construction; not solved

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so = master_setup(params); up = (; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps = master_prestep(so.data, so.counters, up); pp = master_prepare_cc(so.data, so.counters, ps, up)
@unpack θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D = so.D; bi = params.baseIndex; σ = params.σHat; μHat = γ.μHat
θ0 = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
bounds = theoretical_gammaprime_bounds(γ, σ)
θ0[3+D] = clamp(θ0[3+D], bounds.γp_lo, bounds.γp_hi)

Random.seed!(4242)
dir = randn(length(θ0)); dir[2] = 0.0; dir[3:2+D] .= 0.0   # only perturb free coords (mu, A_od, gamma'_focal)
dir ./= norm(dir)
steps = 0:7
thetas = [θ0 .+ 0.01 * s .* dir for s in steps]

mkobj() = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = length(θ0), U = U, N = params.Jac_W,
    lower_limit = -50, use_cached_x = true,
    outer_loop_opt = OUTEROPT, inner_loop_opt = INNEROPT)

function run_series(mode::Symbol)
    obj = mkobj()
    times = Float64[]; iters = Int[]; statuses = Int[]
    CS.INNER_ITERS_TOTAL[] = 0
    for (i, θ) in enumerate(thetas)
        if mode == :cold
            obj.use_cached_x = false
            obj.x .= NaN
        elseif mode == :warm
            obj.use_cached_x = (i > 1)
        elseif mode == :repeat
            obj.use_cached_x = (i > 1)
        end
        t0 = time()
        before = CS.INNER_ITERS_TOTAL[]
        objSol, x, nStatus = CS.inner_loop_internal(obj, θ)
        push!(times, time() - t0)
        push!(iters, CS.INNER_ITERS_TOTAL[] - before)
        push!(statuses, nStatus)
        if mode == :repeat
            # immediately re-solve at the IDENTICAL theta, warm-started from the solution just found
            t0b = time(); beforeb = CS.INNER_ITERS_TOTAL[]
            CS.inner_loop_internal(obj, θ)
            push!(times, time() - t0b); push!(iters, CS.INNER_ITERS_TOTAL[] - beforeb); push!(statuses, nStatus)
        end
    end
    return times, iters, statuses
end

# warm up JIT with a throwaway call before any timed series
mkobj().use_cached_x = true
CS.inner_loop_internal(mkobj(), θ0)

t_cold, i_cold, s_cold   = run_series(:cold)
t_warm, i_warm, s_warm   = run_series(:warm)
t_rep,  i_rep,  s_rep    = run_series(:repeat)

println(">>> COLD   (use_cached_x=false every step): mean_time=", round(sum(t_cold)/length(t_cold),digits=4),
        "s mean_iters=", round(sum(i_cold)/length(i_cold),digits=1), " statuses=", s_cold)
println(">>> WARM   (use_cached_x=true after step1) : mean_time=", round(sum(t_warm)/length(t_warm),digits=4),
        "s mean_iters=", round(sum(i_warm)/length(i_warm),digits=1), " statuses=", s_warm)
println(">>> REPEAT (each theta solved twice, warm) : first-solve mean_time=",
        round(sum(t_rep[1:2:end])/length(t_rep[1:2:end]),digits=4),
        "s  REPEAT-solve mean_time=", round(sum(t_rep[2:2:end])/length(t_rep[2:2:end]),digits=4),
        "s  REPEAT-solve mean_iters=", round(sum(i_rep[2:2:end])/length(i_rep[2:2:end]),digits=2))
flush(stdout)

open(joinpath(OUT, "warmstart_benchmark.csv"), "w") do io
    println(io, "mode,step,time_s,iters,status")
    for (mode, ts, its, sts) in (("cold", t_cold, i_cold, s_cold), ("warm", t_warm, i_warm, s_warm))
        for i in 1:length(ts)
            println(io, join((mode, i, round(ts[i],digits=5), its[i], sts[i]), ","))
        end
    end
    for i in 1:length(t_rep)
        step = ceil(Int, i/2); tag = isodd(i) ? "repeat_first" : "repeat_duplicate"
        println(io, join((tag, step, round(t_rep[i],digits=5), i_rep[i], s_rep[i]), ","))
    end
end

println("\n================ SUMMARY ================")
println("warm start reduces mean inner iterations vs cold by: ",
        round(100*(1 - sum(i_warm[2:end])/sum(i_cold[2:end])), digits=1), "%  (excluding each series' first/cold step)")
println("a duplicate solve at an IDENTICAL theta (what caching avoids) still costs full re-solve time: ",
        round(sum(t_rep[2:2:end])/length(t_rep[2:2:end]), digits=4), "s and ",
        round(sum(i_rep[2:2:end])/length(i_rep[2:2:end]), digits=1), " iters per call -- warm-starting from the",
        " IDENTICAL point does NOT make it free; only an exact-point CACHE (skip solving entirely) does.")
println("Wrote: ", joinpath(OUT, "warmstart_benchmark.csv"))
