# Explicit re-verification of run_profiled_D10_methodB.jl's reported solutions: at the
# saved theta*, re-run the sequential loop to get (p, umat), then check ALL THREE things
# directly: (1) divergence_of(p) <= delta, (2) exact gravity R_mean ~ 0, (3) the FOCAL
# trade-share moments themselves are matched under p (not just "the inner solve didn't
# fail" -- compute the actual E_p[G] residuals).
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "focal_moments_directgp.jl"))
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2

include(joinpath(@__DIR__, "..", "setup", "include_setup.jl"))
include(joinpath(@__DIR__, "..", "prestep", "include_prestep.jl"))
include(joinpath(@__DIR__, "..", "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(@__DIR__, "..", "moments", "include_moments.jl"))
include(joinpath(@__DIR__, "..", "cc_algo", "include_cc_algo.jl"))
include(joinpath(@__DIR__, "..", "lfd", "include_lfd.jl"))
include(joinpath(@__DIR__, "..", "misc", "include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity

const D10 = 10
params = (server=1,user=2,fakeData=1,DFake=D10,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params)
useParams = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(so.data, so.counters, useParams)
prep = master_prepare_cc(so.data, so.counters, prestep_output, useParams)
D = so.D; γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1; ρ = 2e-3; δ = params.δ_ref
logτ = log.(τ); logw = log.(wHat)
focal_u(θ) = begin
    μ = θ[1]; Acol = θ[4:3+D]
    AodPow = [ (Acol[o]*((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ)*(λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
    (σ - 1) .* (log.(1 ./ AodPow) .- logw .- logτ[:, focal])
end
function divergence_of(p::AbstractVector)
    e = exp(1); acc = 0.0
    @inbounds for s in eachindex(p)
        m = p[s] * W
        if !(m > 0) || !isfinite(m); return Inf
        elseif m <= e; acc += m * log(m) - m + 1
        else; acc += m^2 / (2e) - e / 2 + 1
        end
    end
    return acc / W
end
function recover_lfd(θ, moments_fn, d)
    oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = params.Jac_W, lower_limit = -5000,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    val, x, nStatus = inner_loop(obj, θ)
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    # BUG FIX (2026-07-16, see sequential_gravity/run_profiled_production.jl's recover_lfd for the
    # full writeup): inner_loop_internal(::PsiObjectiveBundleDelta,...) NaNs its own cache field on
    # a rejected nStatus (e.g. -300=KN_RC_UNBOUNDED) but NOT the x it returns to this caller --
    # checking only isfinite(x) silently accepts a genuinely failed/unbounded solve.
    nStatus ∈ (0, -100, -101, -103) || return fill(1.0 / W, W), false, G
    all(isfinite, x) || return fill(1.0 / W, W), false, G
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false, G
    return LFD ./ s, true, G
end

d = JLD2.load(joinpath(@__DIR__, "..", "full_aod_diag", "ad_benchmark", "run1_profiled_D10_results.jld2"))
all_results = d["all_results"]; δval = first(keys(all_results))
r = all_results[δval]

for (name, res) in ((:lower, r[:lower]), (:upper, r[:upper]))
    θstar = res.best_θ
    println("\n=== $name bound: verifying best-feasible θ* ===")
    p, ok, Gfocal = recover_lfd(θstar, EK_moments_focal_norm_directgp!, D + 1)
    println("  recover_lfd ok=$ok")
    divp = divergence_of(p)
    println("  divergence_of(p) = $divp   (budget δ=$δval)   δ_ok = ", divp <= δval*(1+1e-6)+1e-10)
    # moment residuals: E_p[G] for each of the D focal shares + 1 counterfactual price index
    resid = [dot(p, Gfocal[:, j]) for j in 1:D+1]
    println("  focal moment residuals E_p[G] (should be ~0): max|resid| = ", maximum(abs.(resid)))
    println("    per-moment: ", round.(resid; sigdigits=3))
    # exact gravity check
    log_x = build_log_x(Uσ, θstar[1]); uf = focal_u(θstar)
    um = zeros(D, D); um[:, focal] .= uf
    for dd in omitted
        inv = invert_destination(log_x, p, λData[:, dd]; ref=ref, ρ=ρ, tol=1e-8, maxit=150, ls_iters=50)
        um[:, dd] .= inv.u_full
    end
    R = gravity_residual(um, logτ, logw, σ).R_mean
    println("  exact gravity R_mean = $R  (tol used in search = 5e-4)")
    kappa = 1 - res.best_gp^(σ/(σ-1))
    println("  kappa = $kappa  (gamma'_focal = $(res.best_gp))")
end
println("\nVERIFY DONE")
