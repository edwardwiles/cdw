# Phase 2a-ii: end-to-end REDUCED focal-only CC solve.
# Wires EK_moments_focal! into the existing CC outer/inner solver with the reduced θ
# (μ, σ, γ_focal, γ'_focal, A[:,focal]) and reduced moments (D focal shares + 1 counterfactual;
# no gravity yet). Produces the focal-only κ bounds — the :focal_only mode baseline.
#   julia --project=. sequential_gravity/run_focal_bounds.jl
# (needs KNITRO env; see SETUP_AND_FINDINGS.md)

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

γ = prep.γ; U = prep.U; focal = params.baseIndex; σ = params.σHat
θr = build_focal_theta(prep.θ_initial, D, focal)
l = length(θr)                 # D+4
d = n_focal_moments(D)         # D+1 inner moments, no outer constraints
oci = d + 1                    # outer_constr_index: only the divergence budget is an outer constraint
δ = params.δ_ref

@printf("\n=== reduced focal-only CC solve (D=%d, W=%d, δ=%g) ===\n", D, params.W, δ)
@printf("reduced θ length = %d (full was %d);  inner moments = %d (full was %d)\n",
        l, length(prep.θ_initial), d, prep.nTotalMoments)

# reduced θ bounds (mirror ccOuter): fix σ and A[1,focal]; μ∈(0,1/(σ-1)); others wide.
function focal_bounds(θr, σ)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2];  hi[2] = θr[2]                 # σ fixed
    lo[1] = 0.001;  hi[1] = 1 / (σ - 1) - 0.001    # μ
    lo[5] = θr[5];  hi[5] = θr[5]                 # A[1,focal] pinned (=1)
    for i in 1:length(θr)
        if θr[i] < 0; hi[i] = -1e4 * θr[i]; lo[i] = 1e4 * θr[i]; end
    end
    return lo, hi
end
θ_lo, θ_hi = focal_bounds(θr, σ)

make_obj(find_smallest) = PsiObjectiveBundleImplicit(
    δ = δ, find_smallest = find_smallest, γ = γ,
    (moments!) = EK_moments_focal!, moments_jacobian! = error,
    d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
    l = l, U = U, N = params.Jac_W, lower_limit = -50, use_cached_x = true,
    outer_loop_opt = "csw_outer_loop_settings_cluster.opt",
    inner_loop_opt = "ek_inner_loop_options.opt",
)

# point estimate at θ_initial (F*): κ from K
Kchk = zeros(params.W); Gchk = zeros(params.W, d)
EK_moments_focal!(Kchk, Gchk, θr, U, (γ = γ,))
@printf("point estimate κ(F*) = %.6f\n\n", Kchk[1])

println(">>> UPPER bound solve"); flush(stdout)
obj_up = make_obj(false)
κ_upper, θ_up, st_up, _ = outer_loop(obj_up, θ_lo, θ_hi, copy(θr))

println("\n>>> LOWER bound solve"); flush(stdout)
obj_lo = make_obj(true)
κ_lower, θ_low, st_lo, _ = outer_loop(obj_lo, θ_lo, θ_hi, copy(θr))

@printf("\n=== FOCAL-ONLY BOUNDS (δ=%g) ===\n", δ)
@printf("  κ_lower = %.6f  (status %d)\n", κ_lower, st_lo)
@printf("  point   = %.6f\n", Kchk[1])
@printf("  κ_upper = %.6f  (status %d)\n", κ_upper, st_up)
@printf("  bracket point? %s\n", (κ_lower <= Kchk[1] <= κ_upper) ? "yes" : "NO")
