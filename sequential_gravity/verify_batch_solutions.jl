# ============================================================================
# Rigorous cross-check for "is the sequential method's wider bound meaningful,
# or an artifact of dropping non-focal trade-share moments?" (spec section
# 16D/16E: both methods must match the SAME full bilateral trade data --
# full-A explicitly via inner moments, sequential implicitly via exact
# omitted-column inversion). Adapts sequential_gravity/verify_D10_solution.jl
# (which only checked focal moments + gravity) to ALSO extract the
# max_abs_share_error the destination-inversion step already computes
# internally but the production driver discards -- this is the ONE thing not
# already checked: does the sequential solution's IMPLIED non-focal trade
# shares actually match the real data, or does the inversion silently fail to
# match some destination while still reporting "gravity-feasible"?
#
# For every saved sequential_gravity/batch_out/seq_*.jld2 checkpoint: reloads
# BOTH the KNITRO-own theta_star AND best_feasible_theta (when different),
# re-inverts all omitted destinations at the saved LFD weights, and reports:
#   - divergence_of(p) <= delta (primal budget check)
#   - focal moment residuals E_p[G] (should be ~0)
#   - max_abs_share_error PER omitted destination (should be ~0 if the
#     inversion actually matched that destination's real trade-share data)
#   - exact gravity R_mean
#
# Run: julia --project=. sequential_gravity/verify_batch_solutions.jl
# ============================================================================
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

const D10 = parse(Int, get(ENV, "DVAL", "10"))
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))
const FAKEDATA = parse(Int, get(ENV, "FAKEDATA", "1"))
params = (server=1,user=2,fakeData=FAKEDATA,DFake=D10,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=WVAL,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=WVAL,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params)
useParams = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(so.data, so.counters, useParams)
prep = master_prepare_cc(so.data, so.counters, prestep_output, useParams)
D = so.D; γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1; ρ = 2e-3
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
    all(isfinite, x) || return fill(1.0 / W, W), false, G
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false, G
    return LFD ./ s, true, G
end

function verify_one(θstar, δval, label)
    println("\n--- $label ---")
    if θstar === nothing
        println("  (no theta -- skipped)")
        return
    end
    p, ok, Gfocal = recover_lfd(θstar, EK_moments_focal_norm_directgp!, D + 1)
    println("  recover_lfd ok=$ok")
    divp = divergence_of(p)
    δ_ok = divp <= δval * (1 + 1e-6) + 1e-10
    @printf("  divergence_of(p) = %.6f  (budget delta=%.4g)  delta_ok=%s\n", divp, δval, δ_ok)
    resid = [dot(p, Gfocal[:, j]) for j in 1:D+1]
    @printf("  FOCAL moment residuals E_p[G]: max|resid| = %.4e\n", maximum(abs.(resid)))
    log_x = build_log_x(Uσ, θstar[1]); uf = focal_u(θstar)
    um = zeros(D, D); um[:, focal] .= uf
    share_errors = Dict{Int,Float64}()
    for dd in omitted
        inv = invert_destination(log_x, p, λData[:, dd]; ref=ref, ρ=ρ, tol=1e-8, maxit=150, ls_iters=50)
        um[:, dd] .= inv.u_full
        share_errors[dd] = inv.max_abs_share_error
        if !inv.converged
            @printf("    !! destination %d inversion NOT converged (iters=%d, share_err=%.4e)\n", dd, inv.iterations, inv.max_abs_share_error)
        end
    end
    max_nonfocal_share_err = maximum(values(share_errors))
    @printf("  NON-FOCAL (omitted) max|share_error| across %d destinations = %.4e\n", length(omitted), max_nonfocal_share_err)
    R = gravity_residual(um, logτ, logw, σ).R_mean
    @printf("  exact gravity R_mean = %.4e\n", R)
    κ = 1 - θstar[3]^(σ/(σ-1))
    @printf("  kappa = %.6f  (gamma'_focal = %.6f)\n", κ, θstar[3])
    all_share_ok = max_nonfocal_share_err < 1e-4 && maximum(abs.(resid)) < 1e-3
    @printf("  VERDICT: full D^2 trade-share data matched (focal+non-focal) = %s,  gravity ~0 = %s,  delta-feasible = %s\n",
            all_share_ok, abs(R) < 5e-4, δ_ok)
    return (divp=divp, δ_ok=δ_ok, focal_resid=maximum(abs.(resid)), nonfocal_share_err=max_nonfocal_share_err, R=R, κ=κ)
end

BATCH_DIR = get(ENV, "VERIFY_BATCH_DIR", joinpath(@__DIR__, "batch_out"))
for f in sort(readdir(BATCH_DIR))
    endswith(f, ".jld2") || continue
    path = joinpath(BATCH_DIR, f)
    d = JLD2.load(path)
    get(d, "done", false) === true || continue
    println("\n================ ", f, " ================")
    δval = d["delta"]
    verify_one(d["theta_star"], δval, "$(f): KNITRO-own theta_star")
    bt = get(d, "best_feasible_theta", nothing)
    if bt !== nothing && bt != d["theta_star"]
        verify_one(bt, δval, "$(f): best_feasible_theta")
    end
end
println("\nVERIFY_BATCH DONE")
