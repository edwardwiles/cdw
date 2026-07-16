# ============================================================================
# D=10, μ FIXED, profiled/sequential-linearization method ("our best current
# approach" per the user), enhanced with THIS SESSION's improvements:
#   - γ_focal≡1 normalization (already existed: focal_moments.jl::EK_moments_focal_norm!)
#   - direct-γ' objective, no σ/(σ-1) power transform (NEW: focal_moments_directgp.jl)
#   - Method B outer gradient: PsiObjectiveBundleImplicitMethodB replaces the
#     dense-Jacobian-then-contract path with a direct scalar ForwardDiff
#     gradient of the divergence-budget envelope scalar. Valid here because
#     this method's gravity moment is INNER-matched (d=D+2, outer_constr_index
#     =d+1 ⇒ ZERO extra outer constraints beyond the divergence budget), so
#     there's no ift! term to reproduce — a complete, not partial, dense-
#     Jacobian replacement (unlike the full-A method, see run_fullA_D10.jl).
#
# Based on run_profiled_bounds_norm.jl (byte-identical sequential-loop/
# inversion machinery; profiled_gravity.jl untouched). D changed 4->10,
# EK_moments_focal_norm! -> EK_moments_focal_norm_directgp! throughout,
# FREEZE_MU hardcoded true, PsiObjectiveBundleImplicit -> ...MethodB.
#
#   julia --project=. sequential_gravity/run_profiled_D10_methodB.jl   (needs KNITRO env)
# ============================================================================

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
const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "focal_moments.jl"))
include(joinpath(@__DIR__, "focal_moments_directgp.jl"))
include(joinpath(@__DIR__, "profiled_gravity.jl"))
using .ProfiledGravity
using Printf
# Inject the Method-B objective bundle INTO the CounterfactualSensitivity module (so its
# internal generic calls like `inner_loop_internal(obj,θ)` correctly dispatch to our new
# methods) without editing any production file.
CS.include(joinpath(@__DIR__, "PsiObjectiveBundleImplicitMethodB.jl"))

const D10 = 10
params = (
    server=1, user=2, fakeData=1, DFake=D10, seedFakeData=889, counterType=1, counterExplicit=0,
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
@assert D == D10
useParams = (; params..., D = D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
prestep_output = master_prestep(data, counters, useParams)
prep = master_prepare_cc(data, counters, prestep_output, useParams)

γ = prep.γ; U = prep.U; W = params.W; focal = params.baseIndex; σ = params.σHat; JacW = params.Jac_W
ρ = 2e-3; δ = params.δ_ref
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1
logτ = log.(τ); logw = log.(wHat)

θr0_orig = build_focal_theta(prep.θ_initial, D, focal)
let γf0 = θr0_orig[3], μ0 = θr0_orig[1]
    global θr0 = vcat(μ0, σ, θr0_orig[4] / γf0, fill(γf0^(-σ / (μ0 * (σ - 1))), D))
end
const KBOUNDS = theoretical_kappa_bounds(γ, σ)

focal_u(θ) = begin
    μ = θ[1]; Acol = θ[4:3+D]
    AodPow = [ (Acol[o]*((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ)*(λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
    (σ - 1) .* (log.(1 ./ AodPow) .- logw .- logτ[:, focal])
end

function recover_lfd(θ, moments_fn, d)
    oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    val, x, nStatus = inner_loop(obj, θ)
    # BUG FIX (2026-07-16, see sequential_gravity/run_profiled_production.jl's recover_lfd for the
    # full writeup): inner_loop_internal(::PsiObjectiveBundleDelta,...) NaNs its own cache field on
    # a rejected nStatus (e.g. -300=KN_RC_UNBOUNDED) but NOT the x it returns to this caller --
    # checking only isfinite(x) silently accepts a genuinely failed/unbounded solve.
    nStatus ∈ (0, -100, -101, -103) || return fill(1.0 / W, W), false
    all(isfinite, x) || return fill(1.0 / W, W), false
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false
    return LFD ./ s, true
end

function divergence_of(p::AbstractVector)
    e = exp(1); acc = 0.0
    @inbounds for s in eachindex(p)
        m = p[s] * W
        if !(m > 0) || !isfinite(m)
            return Inf
        elseif m <= e
            acc += m * log(m) - m + 1
        else
            acc += m^2 / (2e) - e / 2 + 1
        end
    end
    return acc / W
end

function seq_gravcol(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    invert_omitted(p; warm = nothing) = begin
        um = zeros(D, D); um[:, focal] .= uf
        for d in omitted
            ui = warm === nothing ? nothing : warm[:, d]
            inv = invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-8,
                                     maxit = 150, ls_iters = 50, u_init = ui)
            um[:, d] .= inv.u_full
            if verbose && !inv.converged
                @printf("      dest %d NOT converged: iters=%d share_err=%.2e ‖u‖=%.2e\n",
                        d, inv.iterations, inv.max_abs_share_error, maximum(abs, inv.u_full))
            end
        end
        um
    end
    p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    if !ok
        verbose && println("    [seq] initial recover_lfd (focal-only) FAILED")
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R
    try
        umat = invert_omitted(p; warm = warm)
        R = gravity_residual(umat, logτ, logw, σ).R_mean
    catch e
        verbose && println("    [seq] initial invert_omitted threw: ", e)
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    isfinite(R) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    verbose && @printf("    [seq] init: R0=%.4e\n", R)
    col = zeros(W); Rcol = 0.0
    for k in 1:maxit
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        col = infl.ψ_bar .+ infl.R_beta
        Rcol = infl.R_beta
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
        end
        p_cand, okc = recover_lfd(θ, moments_aug!, D + 2)
        if !okc
            verbose && println("    [seq] iter $k: augmented recover_lfd FAILED (linearized moment likely unmatchable)")
            break
        end
        α = 1.0; acc = false
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try
            try
                um_try = invert_omitted(p_try; warm = umat); R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if isfinite(R_try) && abs(R_try) < abs(R); p = p_try; umat = um_try; R = R_try; acc = true; break; end
            α *= 0.5
        end
        verbose && @printf("    [seq] iter %d: R_mean -> %.4e  accepted=%s  α=%.4f\n", k, R, acc, acc ? α : 0.0)
        acc || break
    end
    div_p = divergence_of(p)
    gravity_ok = abs(R) <= tol
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10
    if verbose
        @printf("    [seq] FINAL: R_mean=%.4e gravity_ok=%s  divergence(p)=%.4e (budget δ=%.4g) δ_ok=%s\n",
                R, gravity_ok, div_p, δ, δ_ok)
    end
    return col, R, Rcol, umat, p, gravity_ok && δ_ok
end

function grad_R_theta(θ, umat, p)
    l = length(θ); dRdθ = zeros(l)
    gr = gravity_residual(umat, logτ, logw, σ)
    fi = free_idx(ref, D)
    Jfocal = ForwardDiff.jacobian(focal_u, θ)
    c_focal = gr.Qt[:, focal] ./ (σ - 1) ./ gr.S_Q
    dRdθ .+= Jfocal' * c_focal
    logp = log.(p); μ0 = θ[1]
    for d in omitted
        st = dest_stats(build_log_x(Uσ, μ0), logp, umat[:, d]; ρ = ρ)
        Hd = free_hessian(st, ρ; ref = ref)
        dsh = ForwardDiff.derivative(μ -> dest_share(build_log_x(Uσ, μ), logp, umat[:, d]; ρ = ρ)[1][fi], μ0)
        dud_dmu = -(Hd \ dsh)
        c_d = gr.Qt[fi, d] ./ (σ - 1) ./ gr.S_Q
        dRdθ[1] += dot(c_d, dud_dmu)
    end
    return dRdθ
end

# μ FIXED, per user request (both this run and the full-A comparison)
const FREEZE_MU = true
function focal_bounds(θr)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2]; hi[2] = θr[2]
    lo[1] = 0.001; hi[1] = 1/(σ-1) - 0.001
    lo[3] = KBOUNDS.γp_lo; hi[3] = KBOUNDS.γp_hi
    if FREEZE_MU
        lo[1] = θr[1]; hi[1] = θr[1]
    end
    lo, hi
end
θ_lo, θ_hi = focal_bounds(θr0)

const INFCOL = 1.0 .+ 0.1 .* sin.(1:W)

function make_stateful_moments(; use_exact_grad::Bool = true, find_smallest::Bool = false, δ::Real = δ)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastRmean = Ref(NaN)
    lastRcol  = Ref(NaN)
    lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)   # tracks the EXTREMAL γ'_focal (this run's K), not κ
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol(θf; δ = δ, warm = warm[])
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; nfeas[] += 1
                    dRdθ[] = use_exact_grad ? grad_R_theta(θf, um, p) : zeros(length(θf))
                    EK_moments_focal_norm_directgp!(Ktmp, Gtmp, θf, view(U, 1:1, :), (γ = γ,))
                    κθ = Ktmp[1]
                    if (find_smallest && κθ < best_κ[]) || (!find_smallest && κθ > best_κ[])
                        best_κ[] = κθ; best_θ[] = copy(θf); best_warm[] = copy(um)
                    end
                else
                    gcol[] = INFCOL
                    dRdθ[] = zeros(length(θf))
                end
                neval[] += 1
                if neval[] % 10 == 0
                    @printf("    [θ-eval %d, gravity-feasible %d] seqR_mean=%.2e ok=%s seq_time=%.2fs\n",
                            neval[], nfeas[], lastRmean[], ok, time()-t0); flush(stdout)
                end
                lastθ[] = copy(θf)
            end
        end
        EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        nrow = size(G, 1)
        if eltype(θ) <: ForwardDiff.Dual && lastok[]
            Rlin = lastRcol[] + dot(dRdθ[], θ .- lastθ[])
            @inbounds @views @. G[:, D+2] = (gcol[][1:nrow] - lastRcol[]) + Rlin
        else
            @inbounds @views @. G[:, D+2] = gcol[][1:nrow]
        end
    end
    return m!, gcol, lastRmean, best_θ, best_κ, best_warm
end

function outer_solve_nested(find_smallest, θinit; use_exact_grad::Bool = true, δ::Real = δ)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm = make_stateful_moments(; use_exact_grad = use_exact_grad, find_smallest = find_smallest, δ = δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,
        outer_loop_opt = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "..", "full_aod_diag", "csw_outer_25.opt")),
        inner_loop_opt = joinpath(@__DIR__, "..", "full_aod_diag", "ek_inner.opt"))
    κ, θstar, st, _ = outer_loop(obj, θ_lo, θ_hi, copy(θinit))
    κ, θstar, st, best_θ[], best_κ[], best_warm[]
end

Kchk = zeros(W); Gchk = zeros(W, D + 1); EK_moments_focal_norm_directgp!(Kchk, Gchk, θr0, U, (γ = γ,))
GP_POINT_EST = Kchk[1]
KAPPA_POINT_EST = 1 - GP_POINT_EST^(σ/(σ-1))
@printf("point estimate γ'_focal(F*) = %.6f  ->  kappa point estimate = %.6f\n", GP_POINT_EST, KAPPA_POINT_EST)

gp2kappa(gp) = 1 - gp^(σ/(σ-1))

function run_at_delta(δval::Real)
    @printf("\n=== [RUN 1: profiled/sequential + normalization + direct-gp + MethodB] D=%d W=%d δ=%g ρ=%g MU_FIXED=%s ===\n",
            D, W, δval, ρ, FREEZE_MU)
    @printf("point estimate kappa = %.6f\n", KAPPA_POINT_EST)

    results = Dict{Symbol,Any}()
    # kappa DECREASING in gamma'_focal: kappa_lower <-> gamma' MAXIMIZED (find_smallest=false);
    # kappa_upper <-> gamma' MINIMIZED (find_smallest=true) -- SWAPPED vs the old kappa-objective
    # convention (see compare_directgp.jl for the same swap in the full-A model).
    for (name, fs) in ((:lower, false), (:upper, true))
        @printf("\n----- %s bound: gamma'_focal %s, delta=%g -----\n", name, fs ? "MINIMIZED" : "MAXIMIZED", δval); flush(stdout)
        t0 = time()
        gp, θstar, st, bθ, b_gp, bwarm = outer_solve_nested(fs, θr0; use_exact_grad = true, δ = δval)
        κ = gp2kappa(gp)
        _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ = δval)
        @printf("  KNITRO:        gamma'_%s = %.6f -> kappa = %.6f  (status %d)  exact R_mean(θ*) = %.3e  gravity-feasible=%s  wall %.1fs\n",
                name, gp, κ, st, Rθ, okθ, time() - t0)
        if bθ === nothing
            @printf("  best-feasible: NONE FOUND\n")
            results[name] = (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = NaN, best_gp = NaN, best_θ = nothing, best_ok = false)
        else
            _, Rb, _, _, _, okb = seq_gravcol(bθ; δ = δval, warm = bwarm)
            bκ = gp2kappa(b_gp)
            @printf("  best-feasible: gamma'_%s = %.6f -> kappa = %.6f  exact R_mean = %.3e  gravity-feasible=%s\n", name, b_gp, bκ, Rb, okb)
            results[name] = (κ = κ, gp = gp, R = Rθ, ok = okθ, best_κ = bκ, best_gp = b_gp, best_θ = bθ, best_ok = okb)
        end
    end

    @printf("\n=== RUN 1 RESULTS (delta=%g) ===\n", δval)
    @printf("  kappa_lower : KNITRO=%.6f (feasible=%s)   best-feasible=%.6f (feasible=%s)\n",
            results[:lower].κ, results[:lower].ok, results[:lower].best_κ, results[:lower].best_ok)
    @printf("  point       = %.6f\n", KAPPA_POINT_EST)
    @printf("  kappa_upper : KNITRO=%.6f (feasible=%s)   best-feasible=%.6f (feasible=%s)\n",
            results[:upper].κ, results[:upper].ok, results[:upper].best_κ, results[:upper].best_ok)
    return results
end

DELTA_GRID = let s = get(ENV, "DELTA_GRID", "")
    isempty(s) ? [δ] : parse.(Float64, split(s, ","))
end

all_results = Dict{Float64,Any}()
for δval in DELTA_GRID
    all_results[δval] = run_at_delta(δval)
end
@save joinpath(@__DIR__, "..", "full_aod_diag", "ad_benchmark", "run1_profiled_D10_results.jld2") all_results KAPPA_POINT_EST D
println("RUN1 DONE")
