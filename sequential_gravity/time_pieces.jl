# Head-to-head timing of the sequential-loop primitives on the REAL 4-country pipeline:
#   (1) one destination inversion (cold / warm) — the new smooth-regime Armijo Newton
#   (2) one min-divergence CC solve (recover the LFD) — for comparison
#   julia --project=. sequential_gravity/time_pieces.jl        (needs KNITRO env)

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity, Printf

params = (server=1, user=2, fakeData=1, DFake=4, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=8000, seedU=888, importanceSampling=0, importanceSamplingFactor=2,
    stratifiedSampling=0, IndMomentOrder=5, θConstant=0, gravMoment=1, localGravityMoment=0,
    localGravityCrossMoment=0, GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0, NormalizeMoments=0,
    useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1, refIndex1=1, OuterLoop=1, UoModel=1,
    use_Jacobian=0, calc_δ_star_initial=1, Jac_W=8000, theta_init=0, runLFD=1, runLFDCounterFactual=1)

setup_output = master_setup(params); @unpack data, counters = setup_output; D = setup_output.D
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)
γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat
θr = build_focal_theta(prep.θ_initial, D, focal)
μ = θr[1]; ρ = 2e-3; ref = 1
log_x = build_log_x(γ.Uσ, μ); λData = Matrix(reshape(γ.P, (D, D))')
p = fill(1/W, W)

@printf("\n=== timing on real pipeline (D=%d W=%d ρ=%g) ===\n", D, W, ρ)

# --- (1) inversion ---
inv1 = invert_destination(log_x, p, λData[:, 1]; ref=ref, ρ=ρ, tol=1e-9, maxit=100)  # compile
@printf("inversion converged=%s iters=%d err=%.1e\n", inv1.converged, inv1.iterations, inv1.max_abs_share_error)
t = @elapsed for _ in 1:20; invert_destination(log_x, p, λData[:,1]; ref=ref, ρ=ρ, tol=1e-9, maxit=100); end
@printf("  COLD inversion:  %.2f ms/call  (avg of 20)\n", 1000*t/20)
u0 = inv1.u_full
t = @elapsed for _ in 1:20; invert_destination(log_x, p, λData[:,1]; ref=ref, ρ=ρ, tol=1e-9, maxit=100, u_init=u0); end
@printf("  WARM inversion:  %.2f ms/call  (avg of 20, warm-started at solution)\n", 1000*t/20)
# warm at a *nearby* distribution (what the loop actually sees)
pt = p .* (1 .+ 0.02 .* tanh.(randn(MersenneTwister(3), W))); pt ./= sum(pt)
t = @elapsed for _ in 1:20; invert_destination(log_x, pt, λData[:,1]; ref=ref, ρ=ρ, tol=1e-9, maxit=100, u_init=u0); end
@printf("  WARM inversion (nearby p): %.2f ms/call\n", 1000*t/20)

# --- (2) one min-divergence CC solve (recover LFD), reduced focal moments ---
function one_mindiv_solve()
    d = D + 1; oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = EK_moments_focal!, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θr), U = U, N = params.Jac_W, lower_limit = -5000,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    inner_loop(obj, θr)
end
one_mindiv_solve()  # compile
t = @elapsed for _ in 1:20; one_mindiv_solve(); end
@printf("\nmin-divergence CC solve (reduced, %d moments): %.2f ms/call (avg of 20)\n", D+1, 1000*t/20)

@printf("\n=> a sequential iteration ≈ 1 min-div solve + ~%d inversions (D-1 omitted dests, ×~2 for verify)\n", 2*(D-1))
