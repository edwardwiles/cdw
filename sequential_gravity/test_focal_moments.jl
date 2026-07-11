# Unit test: the reduced focal-only moments must EXACTLY reproduce the focal columns of the full
# EK_moments! (at θ_initial and at a few perturbed θ with matching focal A column). Run:
#   julia --project=. sequential_gravity/test_focal_moments.jl
# (needs KNITRO env for master_prepare_cc's δ*-initial solve; see SETUP_AND_FINDINGS.md)

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
using Printf

params = (
    server=1, user=2, fakeData=1, DFake=4, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=8000, seedU=888,
    importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0, IndMomentOrder=5,
    θConstant=0, gravMoment=1, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1,
    refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=8000,
    theta_init=0, runLFD=1, runLFDCounterFactual=1,
)

setup_output = master_setup(params)
@unpack data, counters = setup_output
D = setup_output.D
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

γ = prep.γ; θ_initial = prep.θ_initial; U = prep.U
W = params.W; focal = params.baseIndex
nTot = prep.nTotalMoments
obj_like = (γ = γ,)

# full model at θ_initial
Kf = zeros(W); Gf = zeros(W, nTot)
EK_moments!(Kf, Gf, θ_initial, U, obj_like)

# reduced model at the matching reduced θ
θr = build_focal_theta(θ_initial, D, focal)
Kr = zeros(W); Gr = zeros(W, n_focal_moments(D))
EK_moments_focal!(Kr, Gr, θr, U, obj_like)

# full focal trade-share columns are at d1 = focal + (o-1)*D; counterfactual col = D^2+1
focal_cols = [focal + (o - 1) * D for o in 1:D]
ts_err = maximum(abs.(Gf[:, focal_cols] .- Gr[:, 1:D]))
cf_err = maximum(abs.(Gf[:, D^2 + 1] .- Gr[:, D + 1]))
k_err  = maximum(abs.(Kf .- Kr))
@printf("\n=== reduced-vs-full focal moments at θ_initial (D=%d, W=%d) ===\n", D, W)
@printf("  κ (K) match:               max|Δ| = %.2e\n", k_err)
@printf("  focal trade-share moments: max|Δ| = %.2e\n", ts_err)
@printf("  focal counterfactual:      max|Δ| = %.2e\n", cf_err)

# perturbed θ: vary μ, γ_focal, γ'_focal, and the focal A column; keep other-column A at initial.
Random.seed!(11)
ok = (k_err < 1e-10 && ts_err < 1e-10 && cf_err < 1e-10)
worst = max(k_err, ts_err, cf_err)
for trial in 1:5
    θf = copy(θ_initial)
    θf[1] *= (1 + 0.2 * randn())                              # μ
    θf[1] = clamp(θf[1], 0.02, 1 / (params.σHat - 1) - 0.01)
    θf[2 + focal] *= (1 + 0.1 * randn())                      # γ_focal
    θf[3 + D] *= (1 + 0.1 * randn())                          # γ'_focal
    for o in 2:D                                             # A[o,focal], o≠1 (A[1,focal]=1 pinned)
        θf[(3 + D) + (focal - 1) * D + o] *= (1 + 0.15 * randn())
    end
    Kf2 = zeros(W); Gf2 = zeros(W, nTot); EK_moments!(Kf2, Gf2, θf, U, obj_like)
    θr2 = build_focal_theta(θf, D, focal)
    Kr2 = zeros(W); Gr2 = zeros(W, n_focal_moments(D)); EK_moments_focal!(Kr2, Gr2, θr2, U, obj_like)
    e = max(maximum(abs.(Gf2[:, focal_cols] .- Gr2[:, 1:D])),
            maximum(abs.(Gf2[:, D^2 + 1] .- Gr2[:, D + 1])),
            maximum(abs.(Kf2 .- Kr2)))
    global worst = max(worst, e)
    global ok &= (e < 1e-9)
    @printf("  trial %d perturbed θ: max|Δ focal moments| = %.2e\n", trial, e)
end

@printf("\n%s  (worst |Δ| = %.2e)\n", ok ? "PASS: reduced focal moments == full focal columns" :
                                          "FAIL: reduced focal moments differ from full", worst)
exit(ok ? 0 : 1)
