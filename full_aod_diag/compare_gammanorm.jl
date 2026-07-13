# ================================================================================
# 25-iter comparison: A[1,d]=1 normalization (current default, "full" variant
# from solve_variant.jl) vs γ_d≡1-for-all-d normalization (moments_gammanorm.jl),
# both with the FULL A_od matrix otherwise free. Same underlying economy/data/
# draws for both (shared master_setup/prestep/prepare_cc call) — only θ bounds,
# θ_initial, and the moments! function differ. Additive/isolated: uses the
# frozen full_aod_diag/*.opt copies (maxit=25 override) so it never touches the
# shared production config.
#
# Run: julia --project=. full_aod_diag/compare_gammanorm.jl   (needs KNITRO env)
# ================================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "moments_gammanorm.jl"))

const OUT = joinpath(@__DIR__,"out"); isdir(OUT) || mkpath(OUT)
const MAXIT_OPT = joinpath(@__DIR__, "csw_outer_25.opt")
const INNEROPT = joinpath(@__DIR__, "ek_inner.opt")

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,θ_initial_low,θ_initial_up,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D=so.D; bi=params.baseIndex; σ=params.σHat; μHat=γ.μHat
Aod_offset = 3 + D

function run_variant(TAG::String, moments_fn, θ_lower, θ_upper, θ_init_up, θ_init_low)
    nfree = count(i -> θ_lower[i] != θ_upper[i], 1:length(θ_lower))
    nfreeA = count(o -> θ_lower[Aod_offset+o[1]+(o[2]-1)*D] != θ_upper[Aod_offset+o[1]+(o[2]-1)*D],
                    [(o,d) for o in 1:D for d in 1:D])
    println(">>> VARIANT=$TAG  free params=$nfree  freeA=$nfreeA"); flush(stdout)

    mkobj(find_smallest, θi) = PsiObjectiveBundleImplicit(δ=1.0, find_smallest=find_smallest, γ=γ,
        (moments!)=moments_fn, moments_jacobian! = error, d=nTotalMoments,
        outer_constr_index=outer_constr_index, inequality_index=inequality_index,
        complement_index=complement_index, l=length(θi), U=U, N=params.Jac_W,
        lower_limit=-50, use_cached_x=true,
        outer_loop_opt=MAXIT_OPT, inner_loop_opt=INNEROPT)

    t0=time()
    println(">>> $TAG UPPER (maxit=25)"); flush(stdout)
    κ_up, θ_up, st_up, _ = outer_loop(mkobj(false, θ_init_up), θ_lower, θ_upper, θ_init_up)
    println(">>> $TAG LOWER (maxit=25)"); flush(stdout)
    κ_lo, θ_lo, st_lo, _ = outer_loop(mkobj(true, θ_init_low), θ_lower, θ_upper, θ_init_low)
    wall=time()-t0

    open(joinpath(OUT,"compare_$(TAG)_summary.txt"),"w") do f
        for ln in (
            "variant=$TAG  D=$D  baseIndex=$bi  free=$nfree  freeA=$nfreeA  maxit=25",
            @sprintf("kappa_lower = %.6f   (status=%d)", κ_lo, st_lo),
            @sprintf("kappa_upper = %.6f   (status=%d)", κ_up, st_up),
            @sprintf("width       = %.6f", κ_up-κ_lo),
            @sprintf("total wall  = %.1fs", wall))
            println(f, ln); println(ln)
        end
    end
    @save joinpath(OUT,"compare_$(TAG).jld2") κ_up κ_lo θ_up θ_lo st_up st_lo
    return (TAG=TAG, κ_up=κ_up, κ_lo=κ_lo, st_up=st_up, st_lo=st_lo, wall=wall, nfree=nfree)
end

# ---------------- Variant 1: A[1,d]=1 normalization (current default) ----------------
θ_lower_A = (θ_initial.*0.0001)[:]; θ_upper_A = (θ_initial.*10000)[:]
θ_lower_A[2]=θ_initial[2]; θ_upper_A[2]=θ_initial[2]
θ_upper_A[1]=1/(σ-1)-0.001; θ_lower_A[1]=0.001; θ_upper_A[1]=min(θ_upper_A[1],1/(σ-1))
for i in 1:D  # A[1,d]=1
    idx = Aod_offset+1+D*(i-1)
    θ_upper_A[idx]=θ_initial[idx]; θ_lower_A[idx]=θ_initial[idx]
end
for i in 1:length(θ_initial)
    if θ_initial[i] < 0
        θ_upper_A[i] = -10000*θ_initial[i]; θ_lower_A[i] = 10000*θ_initial[i]
    end
end

# ---------------- Variant 2: γ_d≡1 for all d (new), all A_od free ----------------
θ_lower_G = (θ_initial.*0.0001)[:]; θ_upper_G = (θ_initial.*10000)[:]
θ_lower_G[2]=θ_initial[2]; θ_upper_G[2]=θ_initial[2]
θ_upper_G[1]=1/(σ-1)-0.001; θ_lower_G[1]=0.001; θ_upper_G[1]=min(θ_upper_G[1],1/(σ-1))
for d in 1:D   # old γ_θ slots: now INERT, pin (any value; ignored by EK_moments_gammanorm!)
    idx = 2+d
    θ_upper_G[idx]=θ_initial[idx]; θ_lower_G[idx]=θ_initial[idx]
end
bounds = theoretical_gammaprime_bounds(γ, σ)
println(">>> theoretical γ'_focal bounds: [$(bounds.γp_lo), $(bounds.γp_hi)]  (κ_max=$(bounds.κ_max))")
θ_lower_G[3+D] = bounds.γp_lo; θ_upper_G[3+D] = bounds.γp_hi   # γ'_focal DIRECT, theoretical range
# A_od: ALL free (no pins at all)
for i in 1:length(θ_initial)
    if θ_initial[i] < 0
        θ_upper_G[i] = -10000*θ_initial[i]; θ_lower_G[i] = 10000*θ_initial[i]
    end
end

θ_initial_G = build_theta_gammanorm(θ_initial, D, bi, μHat, σ)
θ_initial_up_G = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
θ_initial_low_G = build_theta_gammanorm(θ_initial_low, D, bi, μHat, σ)
# clamp the derived γ'_focal init into the theoretical bounds (numerical safety)
for v in (θ_initial_G, θ_initial_up_G, θ_initial_low_G)
    v[3+D] = clamp(v[3+D], bounds.γp_lo, bounds.γp_hi)
end
println(">>> γ'_focal initial (adjusted, new gauge) = ", θ_initial_G[3+D])

# ---------------- run both ----------------
res_A = run_variant("A1d_norm", EK_moments!, θ_lower_A, θ_upper_A, θ_initial_up, θ_initial_low)
res_G = run_variant("gammad_norm", EK_moments_gammanorm!, θ_lower_G, θ_upper_G, θ_initial_up_G, θ_initial_low_G)

println("\n=== COMPARISON (maxit=25) ===")
for r in (res_A, res_G)
    @printf("%-14s free=%3d  kappa=[%.6f, %.6f] width=%.6f  status=(%d,%d)  wall=%.1fs\n",
        r.TAG, r.nfree, r.κ_lo, r.κ_up, r.κ_up-r.κ_lo, r.st_lo, r.st_up, r.wall)
end
println("DONE compare_gammanorm")
