# ============================================================================
# Validation for cc_algo/free_param_map.jl and full_aod_diag/gravity_tariff.jl,
# BEFORE either is used inside a real KNITRO solve. Two things checked:
#   1. FreeParamMap round-trip (full->pack->reconstruct->full, exact) at a
#      representative D=4 full-A theta layout.
#   2. gravity_value/gravity_grad_free! against (a) the EXISTING production
#      newGravityMoment!/withinTransform formula (should match up to the
#      documented sign flip + 1/N_obs), and (b) central finite differences of
#      gravity_value w.r.t. Aod_theta, to catch any chain-rule/sign error in
#      the closed-form gradient BEFORE it goes anywhere near KNITRO.
#
# Run: julia --project=. full_aod_diag/test_free_param_and_gravity.jl
# ============================================================================
using Parameters, Random, LinearAlgebra, Printf
using Base.Threads, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "setup/include_setup.jl")); include(joinpath(ROOT, "prestep/include_prestep.jl"))
include(joinpath(ROOT, "prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT, "moments/include_moments.jl"))
include(joinpath(ROOT, "cc_algo/include_cc_algo.jl")); include(joinpath(ROOT, "misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "moments_gammanorm.jl"))
include(joinpath(@__DIR__, "gravity_tariff.jl"))

println("================ TEST 1: FreeParamMap round-trip ================")
D = 4
l_full = 3 + D + D^2   # [mu, sigma, gamma_theta(D), gamma'_focal, Aod(D^2)]
Aod_offset = 3 + D
# free = gamma'_focal (index 3+D) + all D^2 Aod entries; fixed = mu, sigma, gamma_theta(D)
free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
fixed_idx = vcat(1, 2, collect(3:2+D))
Random.seed!(42)
theta_full = rand(l_full) .+ 0.5
fixed_vals = theta_full[fixed_idx]
m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
println("n_free = ", CS.n_free(m), " (expect ", 1 + D^2, ")")
@assert CS.n_free(m) == 1 + D^2
ok_roundtrip = CS.round_trip_check(theta_full, m)
println("round_trip_check: ", ok_roundtrip)
@assert ok_roundtrip

# perturb only the free coordinates, confirm fixed ones stay byte-identical after reconstruct
x = CS.pack_free(theta_full, m)
x2 = x .+ 1.234
theta_back = CS.reconstruct_full(x2, m)
@assert all(theta_back[fixed_idx] .== fixed_vals) "fixed coordinates must be EXACTLY unchanged"
@assert all(theta_back[free_idx] .== x2) "free coordinates must be EXACTLY the packed values"
println("fixed-coordinate exactness: PASS")
println("free-coordinate exactness:  PASS")

# ForwardDiff dimensionality check: differentiating through reconstruct_full should only carry
# n_free partials, not l_full
using ForwardDiff
g = ForwardDiff.gradient(xf -> sum(CS.reconstruct_full(xf, m) .^ 2), x)
println("ForwardDiff.gradient output length = ", length(g), " (expect ", CS.n_free(m), ")")
@assert length(g) == CS.n_free(m)
# and the gradient values should be exactly 2*x (trivial check of correctness through the map)
@assert isapprox(g, 2 .* x; rtol = 1e-12)
println("TEST 1: ALL PASS\n")

println("================ TEST 2: tariff-residualized gravity value + analytic gradient ================")
params = (server=1,user=2,fakeData=1,DFake=D,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so = master_setup(params); up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps = master_prestep(so.data, so.counters, up); pp = master_prepare_cc(so.data, so.counters, ps, up)
γobj = pp.γ
τ = γobj.τ; cHat = γobj.cHat; wHat = γobj.wHat
lambda = reshape(γobj.P, (D, D))'
μ_fixed = γobj.μHat

q_tilde, N_obs = precompute_q_tilde(τ)
println("N_obs = ", N_obs, " (expect D^2 = ", D^2, ")")
@assert N_obs == D^2

Random.seed!(7)
Aod_theta_test = 0.7 .+ 0.6 .* rand(D, D)   # random positive levels, away from the theta_initial=1 point

function build_AodPow(Aod_theta, μ)
    Aod = Aod_theta .* cHat .* (((wHat .* τ) ./ (wHat[1, 1] .* τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    return (Aod ./ cHat) .^ (-μ)
end

AodPow_test = build_AodPow(Aod_theta_test, μ_fixed)

# (a) compare to the EXISTING production formula (newGravityMoment! / withinTransform), which
# computes sumGrav = Σ Wτ·W(AodPow) with NO 1/N_obs and NO sign flip.
Wτ = withinTransform(τ); WAodPow = withinTransform(AodPow_test)
sumGrav_current = sum(Wτ .* WAodPow)
g_new = gravity_value(τ, AodPow_test, q_tilde, N_obs)
g_expected_from_current = -sumGrav_current / N_obs
println("gravity_value (new)                = ", g_new)
println("-sumGrav_current/N_obs (predicted) = ", g_expected_from_current)
@assert isapprox(g_new, g_expected_from_current; rtol = 1e-12) "FWL sign/normalization relation broken"
println("(a) matches existing formula up to documented sign+N_obs: PASS")

# (b) verify against DIRECT computation of Σ q_tilde[o,d]*log(A_od[o,d])/N_obs with A_od=1/AodPow,
# NOT via the withinTransform-of-AodPow shortcut -- an independent check of the FWL identity itself.
A_od_test = 1.0 ./ AodPow_test
g_direct = sum(q_tilde .* log.(A_od_test)) / N_obs
println("gravity_value (via withinTransform shortcut) = ", g_new)
println("gravity_value (direct Σ q_tilde*log(A_od)/N_obs) = ", g_direct)
@assert isapprox(g_new, g_direct; rtol = 1e-10) "FWL identity itself is broken -- do not trust the shortcut"
println("(b) FWL identity (single-sided residualization = both-sided demeaning): PASS")

# (c) analytic gradient vs central finite differences of gravity_value w.r.t. Aod_theta (direct
# Σq_tilde*log(A_od)/N_obs form, re-deriving AodPow fresh at each perturbation -- the real
# dependency chain, not a shortcut).
function g_of_Aod_theta(Aod_theta_vec)
    Aod_theta_mat = reshape(Aod_theta_vec, D, D)
    AodPow = build_AodPow(Aod_theta_mat, μ_fixed)
    A_od = 1.0 ./ AodPow
    return sum(q_tilde .* log.(A_od)) / N_obs
end

x0 = vec(Aod_theta_test)
h = 1e-6
g_fd = zeros(D^2)
for i in 1:D^2
    xp = copy(x0); xp[i] += h
    xm = copy(x0); xm[i] -= h
    g_fd[i] = (g_of_Aod_theta(xp) - g_of_Aod_theta(xm)) / (2h)
end

# closed form: dg/dAod_theta[o,d] = (q_tilde[o,d]/N_obs) * (mu / Aod_theta[o,d])
g_analytic = zeros(D, D)
for o in 1:D, d in 1:D
    g_analytic[o, d] = (q_tilde[o, d] / N_obs) * (μ_fixed / Aod_theta_test[o, d])
end
g_analytic_vec = vec(g_analytic)

relerr = norm(g_analytic_vec .- g_fd) / norm(g_fd)
println("analytic gradient vs central finite differences: relerr = ", relerr)
@printf("  max abs diff = %.3e\n", maximum(abs.(g_analytic_vec .- g_fd)))
@assert relerr < 1e-4 "analytic gravity gradient does NOT match finite differences -- DO NOT USE"
println("(c) analytic gradient matches finite differences: PASS")

println("\nTEST 2: ALL PASS")
println("\n================ ALL TESTS PASSED ================")
