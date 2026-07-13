# ============================================================================
# Outer-solver integration test (mega-prompt §15), configs A vs B only:
#   A. current callback/caching behavior (production's real eval_fcga=no,
#      uninstrumented cost -- i.e. what ships today)
#   B. improved exact-point cache, SAME derivative method (production's dense
#      ForwardDiff Jacobian via moments_jacobian!=error -> calculate_jac_θ_autodiff!)
#
# Config C from the spec (cache + optimized scalar-ForwardDiff Method B) is
# NOT run here: wiring Method B into the actual outer KNITRO callback (not
# just a diagnostics-only side-by-side comparison, which ../ad_benchmark/
# already did) would mean building a cached variant of
# PsiObjectiveBundleImplicitMethodB_fullA specifically for the gravity-
# constraint-free case this D=4 config uses -- real additional engineering,
# not a rerun of existing pieces. Flagged as a scoped-out follow-on in the
# final report rather than attempted under this budget; §7 of the final
# report gives the expected combined effect analytically from the two
# already-measured pieces (cache: this file; Method B vs dense Jacobian:
# forwarddiff_benchmark.csv + ../ad_benchmark/benchmark_results.csv).
#
# Uses the SAME θ_init/bounds/draws/model as audit_outer_callbacks.jl but at
# maxit=25 (this session's other standard diagnostics use this budget, e.g.
# ../csw_outer_25.opt) rather than maxit=15, to see whether the ~44%
# duplicate-solve fraction found at maxit=15 holds at a more realistic budget.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/outer_solver_comparison.jl
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
CS.include(joinpath(@__DIR__, "cached_outer_loop.jl"))

const OUT = @__DIR__
const OPT25    = joinpath(@__DIR__, "ek_outer_loop_options_audit25.opt")   # production eval_fcga=no, maxit=25
const WARMUP_OPT = joinpath(@__DIR__, "ek_outer_loop_options_warmup.opt")
const INNEROPT = joinpath(dirname(@__DIR__), "ek_inner.opt")

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so = master_setup(params); up = (; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps = master_prestep(so.data, so.counters, up); pp = master_prepare_cc(so.data, so.counters, ps, up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D = so.D; bi = params.baseIndex; σ = params.σHat; μHat = γ.μHat

θ_lower_G = (θ_initial .* 0.0001)[:]; θ_upper_G = (θ_initial .* 10000)[:]
θ_lower_G[2] = θ_initial[2]; θ_upper_G[2] = θ_initial[2]
θ_upper_G[1] = 1/(σ-1) - 0.001; θ_lower_G[1] = 0.001; θ_upper_G[1] = min(θ_upper_G[1], 1/(σ-1))
for d in 1:D
    idx = 2 + d
    θ_upper_G[idx] = θ_initial[idx]; θ_lower_G[idx] = θ_initial[idx]
end
bounds = theoretical_gammaprime_bounds(γ, σ)
θ_lower_G[3+D] = bounds.γp_lo; θ_upper_G[3+D] = bounds.γp_hi
for i in 1:length(θ_initial)
    if θ_initial[i] < 0
        θ_upper_G[i] = -10000*θ_initial[i]; θ_lower_G[i] = 10000*θ_initial[i]
    end
end
θ_initial_up_G = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
θ_initial_up_G[3+D] = clamp(θ_initial_up_G[3+D], bounds.γp_lo, bounds.γp_hi)

mkobj(outer_opt) = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = length(θ_initial_up_G), U = U, N = params.Jac_W,
    lower_limit = -50, use_cached_x = true,
    outer_loop_opt = outer_opt, inner_loop_opt = INNEROPT)

println(">>> WARMUP (both paths)"); flush(stdout)
CS.outer_loop_instrumented(mkobj(WARMUP_OPT), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = false)
CS.outer_loop_instrumented(mkobj(WARMUP_OPT), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = true)

println(">>> CONFIG A: current behavior (no cache), maxit=25"); flush(stdout)
tA = time(); rA = CS.outer_loop_instrumented(mkobj(OPT25), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = false); tA = time() - tA
CS.summarize(rA.cache; label = "CONFIG A (maxit=25, no cache)")

println(">>> CONFIG B: exact-point cache, maxit=25"); flush(stdout)
tB = time(); rB = CS.outer_loop_instrumented(mkobj(OPT25), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = true); tB = time() - tB
CS.summarize(rB.cache; label = "CONFIG B (maxit=25, cache)")

open(joinpath(OUT, "outer_solver_comparison.csv"), "w") do io
    println(io, "config,unique_theta,total_callbacks,inner_solves,grad_computations,cache_hits,wall_time_s,outer_iters,outer_fc,opt_err,kappa,status")
    for (label, r, wall) in (("A_current_nocache", rA, tA), ("B_exact_point_cache", rB, tB))
        c = r.cache
        nuniq = length(Set(rr.theta_hash for rr in c.trace))
        println(io, join((label, nuniq, c.n_callback, c.n_inner_solve, c.n_grad_compute, c.n_cache_hit,
                           round(wall, digits=3), r.outer_iters, r.outer_fc, r.opt_err, r.κ_min, r.nStatus), ","))
    end
end

println("\n================ INTEGRATION TEST SUMMARY (maxit=25) ================")
println("kappa A=", rA.κ_min, "  kappa B=", rB.κ_min, "  identical=", isapprox(rA.κ_min, rB.κ_min; atol=1e-10))
println("opt_err A=", rA.opt_err, "  opt_err B=", rB.opt_err, "  identical=", isapprox(rA.opt_err, rB.opt_err; atol=1e-10))
println("status A=", rA.nStatus, "  status B=", rB.nStatus)
nuniqA = length(Set(rr.theta_hash for rr in rA.cache.trace))
println("unique theta points: ", nuniqA, "   inner solves A=", rA.cache.n_inner_solve, "  B=", rB.cache.n_inner_solve)
println("duplicate inner solves eliminated: ", rA.cache.n_inner_solve - rB.cache.n_inner_solve, " of ", rA.cache.n_inner_solve,
        " (", round(100*(rA.cache.n_inner_solve - rB.cache.n_inner_solve)/rA.cache.n_inner_solve, digits=1), "%)")
println("wall time: A=", round(tA,digits=2), "s  B=", round(tB,digits=2), "s  speedup=", round(tA/tB, digits=3), "x")
println("Wrote: ", joinpath(OUT, "outer_solver_comparison.csv"))
