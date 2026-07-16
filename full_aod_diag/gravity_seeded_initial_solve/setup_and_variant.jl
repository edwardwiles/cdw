# ============================================================================
# Experiment: does seeding the FIRST (focal-only) CC solve with a linearized
# gravity moment from the previous F reduce drift, vs the current method's
# first solve which ignores gravity entirely?
#
# Current method (seq_gravcol, sequential_gravity/run_profiled_production.jl):
#   1. p = recover_lfd(theta, focal-only moments, D+1)         <- NO gravity moment at all
#   2. umat = invert_omitted(p)                                 <- exact shares given p
#   3. R = gravity_residual(umat)                               <- first look at gravity, AFTER p is fixed
#   4. loop: linearize gravity at (p, umat), re-solve CC with the augmented moment, damp/accept
#
# Proposed variant (seq_gravcol_gravityseed): if a previous (p_prev, umat_prev) is available (the
# previous outer-loop theta's converged F, or the same theta's own prior F), replace step 1 with
# a solve that ALSO includes the linearized gravity moment evaluated at (p_prev, umat_prev) --
# i.e. seed the very first CC solve with an approximate gravity constraint instead of ignoring
# gravity until step 4's first loop iteration. Falls back to the current blind behavior when no
# prior F is available (first-ever evaluation).
#
# This file defines BOTH functions (byte-identical pipeline setup/preamble to
# sequential_gravity/run_profiled_production.jl, D=10) so they can be run head-to-head in the same
# process on identical (log_x, lambda_data, p_cand solves, ...).
#
#   DVAL=10 julia --project=. full_aod_diag/gravity_seeded_initial_solve/setup_and_variant.jl
# ============================================================================

using Parameters, Base.Threads, Random, Dates, DelimitedFiles
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2, Printf

const ROOT = joinpath(@__DIR__, "..", "..")
include(joinpath(ROOT, "setup", "include_setup.jl"))
include(joinpath(ROOT, "prestep", "include_prestep.jl"))
include(joinpath(ROOT, "prepare_cc", "include_prepare_cc.jl"))
include(joinpath(ROOT, "moments", "include_moments.jl"))
include(joinpath(ROOT, "cc_algo", "include_cc_algo.jl"))
include(joinpath(ROOT, "lfd", "include_lfd.jl"))
include(joinpath(ROOT, "misc", "include_misc.jl"))
using .CounterfactualSensitivity
const CS = CounterfactualSensitivity
include(joinpath(ROOT, "sequential_gravity", "focal_moments.jl"))
include(joinpath(ROOT, "sequential_gravity", "focal_moments_directgp.jl"))
include(joinpath(ROOT, "sequential_gravity", "profiled_gravity.jl"))
using .ProfiledGravity

const DVAL = parse(Int, get(ENV, "DVAL", "10"))
params = (
    server=1, user=2, fakeData=1, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
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
@assert D == DVAL
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

function invert_omitted_serial(log_x, p, uf; warm = nothing)
    um = zeros(D, D); um[:, focal] .= uf
    stats = Vector{Any}(undef, D)
    for d in omitted
        ui = warm === nothing ? nothing : warm[:, d]
        inv = invert_destination(log_x, p, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-8,
                                 maxit = 150, ls_iters = 50, u_init = ui)
        um[:, d] .= inv.u_full
        stats[d] = inv.stats
    end
    um, stats
end

# ----------------------------------------------------------------------------------------------
# CURRENT method (byte-identical logic to sequential_gravity/run_profiled_production.jl::seq_gravcol,
# minus the parallel/interp/stats-reuse changes -- kept as the literal "baseline" comparator here
# so this experiment isolates ONLY the "gravity-blind first solve" question, not conflated with
# the other three production changes)
# ----------------------------------------------------------------------------------------------
function seq_gravcol_baseline(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return (R=Inf, R0=Inf, umat=nothing, p=fill(1.0/W,W), ok=false, n_iters=0, div_p=Inf)
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    ok || return (R=Inf, R0=Inf, umat=nothing, p=fill(1.0/W,W), ok=false, n_iters=0, div_p=Inf)
    umat, _ = invert_omitted_serial(log_x, p, uf; warm = warm)
    R0 = gravity_residual(umat, logτ, logw, σ).R_mean   # <- residual right after the FIRST (blind) solve
    R = R0
    n_iters = 0
    for k in 1:maxit
        n_iters = k
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
        end
        p_cand, okc = recover_lfd(θ, moments_aug!, D + 2)
        okc || break
        α = 1.0; acc = false
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try
            try
                um_try, _ = invert_omitted_serial(log_x, p_try, uf; warm = umat)
                R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if isfinite(R_try) && abs(R_try) < abs(R); p = p_try; umat = um_try; R = R_try; acc = true; break; end
            α *= 0.5
        end
        acc || break
    end
    div_p = divergence_of(p)
    gravity_ok = abs(R) <= tol
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10
    return (R = R, R0 = R0, umat = umat, p = p, ok = gravity_ok && δ_ok, n_iters = n_iters, div_p = div_p)
end

# ----------------------------------------------------------------------------------------------
# GRAVITY-SEEDED variant: if (p_prev, umat_prev) available (previous theta's, or same theta's
# prior, converged F), the FIRST CC solve ALSO includes the linearized gravity moment evaluated
# at that prior F -- i.e. step 1 becomes a D+2 augmented solve instead of a D+1 focal-only solve.
# Falls back to the current blind D+1 solve when no prior F is available.
# ----------------------------------------------------------------------------------------------
function seq_gravcol_gravityseed(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing,
                                 warm_p::Union{Nothing,AbstractVector} = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return (R=Inf, R0=Inf, umat=nothing, p=fill(1.0/W,W), ok=false, n_iters=0, div_p=Inf)
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    local p, ok
    if warm !== nothing && warm_p !== nothing
        infl_seed = influence_function(log_x, warm_p, warm, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        moments_aug_seed! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj)
            @. G[:, D+2] = infl_seed.ψ_bar + infl_seed.R_beta
        end
        p, ok = recover_lfd(θ, moments_aug_seed!, D + 2)
    else
        p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    end
    ok || return (R=Inf, R0=Inf, umat=nothing, p=fill(1.0/W,W), ok=false, n_iters=0, div_p=Inf)
    umat, _ = invert_omitted_serial(log_x, p, uf; warm = warm)
    R0 = gravity_residual(umat, logτ, logw, σ).R_mean   # <- residual right after the FIRST (seeded) solve
    R = R0
    n_iters = 0
    for k in 1:maxit
        n_iters = k
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
        end
        p_cand, okc = recover_lfd(θ, moments_aug!, D + 2)
        okc || break
        α = 1.0; acc = false
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try
            try
                um_try, _ = invert_omitted_serial(log_x, p_try, uf; warm = umat)
                R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if isfinite(R_try) && abs(R_try) < abs(R); p = p_try; umat = um_try; R = R_try; acc = true; break; end
            α *= 0.5
        end
        acc || break
    end
    div_p = divergence_of(p)
    gravity_ok = abs(R) <= tol
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10
    return (R = R, R0 = R0, umat = umat, p = p, ok = gravity_ok && δ_ok, n_iters = n_iters, div_p = div_p)
end

println("[gravity_seeded_initial_solve setup] D=$D W=$W ρ=$ρ done.")
