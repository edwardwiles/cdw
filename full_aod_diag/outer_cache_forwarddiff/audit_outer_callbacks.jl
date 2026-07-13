# ============================================================================
# Outer-loop callback/caching audit (mega-prompt §5-8, §15).
#
# Empirically answers: does production's ACTUAL outer-loop configuration
# (ek_outer_loop_options.opt, eval_fcga=no — separate F and G callbacks; used
# by every production driver: cc_algo/ccOuter.jl, sequential_gravity/
# run_sequential.jl, lfd/LFD.jl, prepare_cc/PMM.jl) call inner_loop_internal
# MORE THAN ONCE at the exact same theta, and does an exact-point cache fix it?
#
# NOTE: every prior full_aod_diag/ad_benchmark timing comparison this session
# used csw_outer_25.opt (eval_fcga=YES, the combined F+G callback), which
# structurally cannot exhibit this duplication. Production's real default is
# different. This script is the first one in the session to test the actual
# production callback-splitting configuration.
#
# Runs D=4, γ_d≡1 + direct-γ' config (same as ad_benchmark/compare_directgp.jl),
# maxit bounded to 15 via a diagnostics-only copy of ek_outer_loop_options.opt
# (ek_outer_loop_options_audit.opt — identical except maxit/maxtime_real, so
# the run finishes in reasonable wall time; eval_fcga stays "no", matching
# production exactly). Two runs: instrumented WITHOUT cache (baseline, =
# production behavior today) and WITH cache (the proposed fix). Writes
# callback_trace.csv (uncached run) and cache_benchmark.csv (both runs
# compared), and prints the §7 report numbers.
#
# Run: julia --project=. full_aod_diag/outer_cache_forwarddiff/audit_outer_callbacks.jl
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
const AUDIT_OPT  = joinpath(@__DIR__, "ek_outer_loop_options_audit.opt")   # production eval_fcga=no, maxit bounded
const WARMUP_OPT = joinpath(@__DIR__, "ek_outer_loop_options_warmup.opt")  # same, maxit=1, JIT warm-up only
const INNEROPT   = joinpath(dirname(@__DIR__), "ek_inner.opt")

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so = master_setup(params); up = (; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps = master_prestep(so.data, so.counters, up); pp = master_prepare_cc(so.data, so.counters, ps, up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D = so.D; bi = params.baseIndex; σ = params.σHat; μHat = γ.μHat
Aod_offset = 3 + D

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
for v in (θ_initial_up_G,)
    v[3+D] = clamp(v[3+D], bounds.γp_lo, bounds.γp_hi)
end
nfree = count(i -> θ_lower_G[i] != θ_upper_G[i], 1:length(θ_lower_G))
println(">>> AUDIT: D=$D free_outer_params=$nfree (D^2=$(D^2) A_od + gamma'_focal + mu; sigma and gamma_theta pinned)")
flush(stdout)

mkobj(outer_opt, use_cached_x) = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=true, γ=γ,
    (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
    outer_constr_index = outer_constr_index, inequality_index = inequality_index,
    complement_index = complement_index, l = length(θ_initial_up_G), U = U, N = params.Jac_W,
    lower_limit = -50, use_cached_x = use_cached_x,
    outer_loop_opt = outer_opt, inner_loop_opt = INNEROPT)

# JIT/compilation is paid once per (use_cache) code path here so the timed runs below are not
# contaminated by the FIRST call's compilation cost (both cb_F!/cb_G!/ensure_inner!/ensure_grad!
# and KNITRO's own first-solve overhead) — each config gets its OWN warm-up, maxit=1, discarded.
println(">>> WARMUP: paying JIT cost for use_cache=false path (maxit=1, discarded)"); flush(stdout)
CS.outer_loop_instrumented(mkobj(WARMUP_OPT, true), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = false)
println(">>> WARMUP: paying JIT cost for use_cache=true path (maxit=1, discarded)"); flush(stdout)
CS.outer_loop_instrumented(mkobj(WARMUP_OPT, true), θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = true)

# ---- Run 1: instrumented, NO cache (= production's actual behavior today) ----
println(">>> RUN 1: instrumented baseline, use_cache=false, matching production eval_fcga=no"); flush(stdout)
obj1 = mkobj(AUDIT_OPT, true)
t0 = time()
r1 = CS.outer_loop_instrumented(obj1, θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = false)
t1 = time() - t0
CS.summarize(r1.cache; label = "RUN 1 (no cache, production eval_fcga=no)")
CS.write_trace_csv(joinpath(OUT, "callback_trace.csv"), r1.cache)
println("RUN 1 wall=", round(t1, digits=2), "s  status=", r1.nStatus, "  opt_err=", r1.opt_err,
        "  outer_iters=", r1.outer_iters, "  outer_fc=", r1.outer_fc, "  kappa=", r1.κ_min)
flush(stdout)

# ---- Run 2: instrumented, WITH cache (proposed fix), same start/bounds/opt file ----
println(">>> RUN 2: instrumented, use_cache=true, otherwise identical config"); flush(stdout)
obj2 = mkobj(AUDIT_OPT, true)
t0 = time()
r2 = CS.outer_loop_instrumented(obj2, θ_lower_G, θ_upper_G, θ_initial_up_G; use_cache = true)
t2 = time() - t0
CS.summarize(r2.cache; label = "RUN 2 (exact-point cache, production eval_fcga=no)")
println("RUN 2 wall=", round(t2, digits=2), "s  status=", r2.nStatus, "  opt_err=", r2.opt_err,
        "  outer_iters=", r2.outer_iters, "  outer_fc=", r2.outer_fc, "  kappa=", r2.κ_min)
flush(stdout)

# ---- Comparison CSV ----
open(joinpath(OUT, "cache_benchmark.csv"), "w") do io
    println(io, "run,use_cache,total_callbacks,unique_theta,inner_solves,grad_computations,cache_hits,inner_solve_time_s,grad_time_s,wall_time_s,outer_iters,outer_fc,opt_err,kappa,status")
    for (label, uc, r, wall) in ((("run1_nocache", false, r1, t1)), (("run2_cache", true, r2, t2)))
        c = r.cache
        nuniq = length(Set(rr.theta_hash for rr in c.trace))
        println(io, join((label, uc, c.n_callback, nuniq, c.n_inner_solve, c.n_grad_compute, c.n_cache_hit,
                           round(c.t_inner, digits=4), round(c.t_grad, digits=4), round(wall, digits=3),
                           r.outer_iters, r.outer_fc, r.opt_err, r.κ_min, r.nStatus), ","))
    end
end

println("\n================ SUMMARY ================")
same_result = isapprox(r1.κ_min, r2.κ_min; atol=1e-10)
println("kappa identical between cached/uncached runs: ", same_result, "  (", r1.κ_min, " vs ", r2.κ_min, ")")
println("duplicate inner solves in production config (run1): ", r1.cache.n_inner_solve - length(Set(rr.theta_hash for rr in r1.cache.trace)))
println("inner solves avoided by caching: ", r1.cache.n_inner_solve - r2.cache.n_inner_solve, " of ", r1.cache.n_inner_solve)
println("wall time: run1=", round(t1,digits=2), "s  run2=", round(t2,digits=2), "s  speedup=", round(t1/t2, digits=3), "x")
println("Wrote: ", joinpath(OUT,"callback_trace.csv"), " and ", joinpath(OUT,"cache_benchmark.csv"))
