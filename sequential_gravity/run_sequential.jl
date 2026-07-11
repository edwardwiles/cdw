# Phase 2b: the sequentially-linearized profiled full-gravity loop, at a FIXED outer θ.
# Demonstrates that adding the linearized gravity moment E_F[ψ̄]=−R_k to the reduced CC problem
# and iterating (recover LFD → invert omitted → build ψ̄ → re-solve → re-invert → damp) drives the
# exact origin+dest-FE gravity residual toward zero — letting gravity influence which distribution
# CC selects, without adding the omitted A columns/moments to the problem.
#   julia --project=. sequential_gravity/run_sequential.jl
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
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
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

γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat; JacW = params.Jac_W
θr = build_focal_theta(prep.θ_initial, D, focal)
ρ = 2e-3                      # smoothing for inversion + influence (ρ→0 = hard max residual)
gravity_tol = 5e-4
δ = params.δ_ref

# ---- LFD recovery at a fixed θ for a given moment function (min-divergence dual; cf lfd/LFD.jl) --
function recover_lfd(θ, moments_fn, d)
    oci = d + 1                                   # all moments inner
    obj = PsiObjectiveBundleDelta(
        γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    val, x, nStatus = inner_loop(obj, θ)
    G = zeros(W, d); K = zeros(W)
    moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W
        arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x)))
    end
    LFD = zeros(W); dPsi!(LFD, arg0)
    p = LFD ./ sum(LFD)
    return p, val, nStatus
end

# ---- fixed objects for the residual/influence ------------------------------------------------
μ = θr[1]
Uσ = γ.Uσ
log_x = build_log_x(Uσ, μ)
λData = Matrix(reshape(γ.P, (D, D))')      # λData[o,d]
wHat = γ.wHat; τ = γ.τ
Acol = θr[5:4+D]
AodPow_focal = [ (Acol[o] * ((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ) *
                  (λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
A_focal = 1 ./ AodPow_focal
u_focal = (σ - 1) .* (log.(A_focal) .- log.(wHat) .- log.(τ[:, focal]))
omitted = [d for d in 1:D if d != focal]
ref = 1

# invert every omitted destination under weights p (warm-started), return the full u matrix
function invert_omitted(p; warm = nothing)
    um = zeros(D, D); um[:, focal] .= u_focal
    for d in omitted
        ui = warm === nothing ? nothing : warm[:, d]
        um[:, d] .= invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = ρ,
                                       tol = 1e-11, maxit = 400, u_init = ui).u_full
    end
    return um
end
exact_R(um) = gravity_residual(um, log.(τ), log.(wHat), σ).R_beta

@printf("\n=== sequential profiled full-gravity loop at fixed θ_initial (D=%d, W=%d, ρ=%g) ===\n", D, W, ρ)

# ---- initialization: reduced focal-only LFD --------------------------------------------------
p, val0, st0 = recover_lfd(θr, EK_moments_focal!, D + 1)
umat = invert_omitted(p)
R = exact_R(umat)
@printf("init: focal-only LFD (status %d)  R_beta=%.6e\n", st0, R)

# ---- sequential iterations -------------------------------------------------------------------
for k in 1:12
    global p, umat, R
    if abs(R) <= gravity_tol
        @printf("CONVERGED at iter %d: |R_beta|=%.2e <= %.1e\n", k-1, abs(R), gravity_tol)
        break
    end
    # linearize: influence function ψ̄ of R at the current LFD
    infl = influence_function(log_x, p, umat, λData, omitted, log.(τ), log.(wHat), σ;
                              ref = ref, ρ = ρ, scale = :R_beta)
    ψbar = infl.ψ_bar
    # augmented moments: focal (D+1) + one linearized-gravity column = ψ̄ + R  (target E_F[·]=0)
    Rk = infl.R_beta
    moments_aug! = (K, G, θ, Uarg, obj) -> begin
        EK_moments_focal!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        @. G[:, D+2] = ψbar + Rk
    end
    p_cand, valc, stc = recover_lfd(θr, moments_aug!, D + 2)
    # exact re-inversion + damping: accept largest α with |R_α|<|R_k|
    α = 1.0; accepted = false; R_new = R; p_new = p_cand; um_new = umat
    for _ in 1:12
        p_try = (1 - α) .* p .+ α .* p_cand
        um_try = invert_omitted(p_try; warm = umat)
        R_try = exact_R(um_try)
        if abs(R_try) < abs(R)
            accepted = true; R_new = R_try; p_new = p_try; um_new = um_try; break
        end
        α *= 0.5
    end
    div_new = sum(p_new .* log.(max.(p_new .* W, 1e-300))) / W    # KL(F‖F*) proxy on the tilt
    @printf("iter %2d: pred E_F[ψ̄]=%.3e (=-R_k)  cand LFD st=%d  α=%.4f  R_beta: %.4e -> %.4e  KL≈%.3e\n",
            k, -Rk, stc, accepted ? α : 0.0, R, R_new, div_new)
    if !accepted
        @printf("  no α reduced |R| — linearization stalled at iter %d\n", k); break
    end
    p = p_new; umat = um_new; R = R_new
end

@printf("\nfinal exact gravity residual R_beta = %.6e  (started 7.1e-3-ish)\n", R)
@printf("max omitted share error at final LFD: %.2e\n",
        maximum(maximum(abs.(dest_stats(log_x, log.(p), umat[:, d]; ρ = ρ).share .- λData[:, d])) for d in omitted))
