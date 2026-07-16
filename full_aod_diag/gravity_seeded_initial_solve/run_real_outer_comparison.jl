# ============================================================================
# Real-outer-loop comparison: current method (blind first CC solve) vs the
# gravity-seeded variant (first CC solve augmented with the linearized
# gravity moment from the PREVIOUS THETA's converged F), each driven by an
# ACTUAL KNITRO outer solve (outer_solve_nested_cached), not a synthetic
# theta perturbation. This directly answers "does seeding reduce drift as the
# real outer search explores" using the real search trajectory.
#
# Matches the delta=10 condition that produced genuinely degenerate KNITRO-own
# points in trade_robustness_modular's 60ca63d ("Sequential lower delta=10:
# ... STILL flagged infeasible"; "Sequential upper delta=10: KNITRO-own point
# is ALSO degenerate (R_mean=1.1e+107)"). Uses a reduced-maxit outer opt file
# for tractability (the reported issue was NOT a maxit artifact -- same
# degenerate point recurred at maxit=100 and maxit=300 in that commit, so a
# shorter run should still surface it if it's real).
#
# Byte-identical pipeline preamble/seq_gravcol/outer-loop machinery to
# sequential_gravity/run_profiled_production.jl (this repo's own convention
# for diagnostic copies), with exactly two changes:
#   (a) seq_gravcol takes an extra `warm_p` argument and, when both `warm`
#       (umat) and `warm_p` (p) are available, seeds the first CC solve with
#       the previous F's linearized gravity moment instead of solving blind
#       -- gated by `GRAVITY_SEED::Bool`, so the SAME code path runs both
#       the baseline (GRAVITY_SEED=false) and the variant (GRAVITY_SEED=true).
#   (b) make_stateful_moments carries a `warm_p::Ref` alongside `warm`,
#       updated on every feasible solve exactly like `warm` already is.
#
#   DVAL=10 GRAVITY_SEED=false DELTA=10 BOUND=upper OUTER_MAXIT=15 julia --project=. full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl
#   DVAL=10 GRAVITY_SEED=true  DELTA=10 BOUND=upper OUTER_MAXIT=15 julia --project=. full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl
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
CS.include(joinpath(ROOT, "sequential_gravity", "PsiObjectiveBundleImplicitMethodB.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "10"))
const GRAVITY_SEED = lowercase(get(ENV, "GRAVITY_SEED", "false")) in ("1", "true", "yes")
const DELTA_VAL = parse(Float64, get(ENV, "DELTA", "10"))
const BOUND_NAME = get(ENV, "BOUND", "upper")   # "upper" or "lower"
const OUTER_MAXIT = parse(Int, get(ENV, "OUTER_MAXIT", "15"))
const OUT_TAG = GRAVITY_SEED ? "gravityseed" : "baseline"

# Build a reduced-maxit outer opt file (byte copy of production's csw_outer_25.opt with only
# `maxit` changed) so this comparison is tractable; the reported degeneracy was NOT a maxit
# artifact (recurred at maxit=100 and 300), so a shorter run is expected to still surface it.
const BASE_OUTER_OPT = joinpath(ROOT, "full_aod_diag", "csw_outer_25.opt")
const OUTER_OPT_FILE = joinpath(@__DIR__, "csw_outer_$(OUTER_MAXIT)_driftcheck.opt")
if !isfile(OUTER_OPT_FILE)
    txt = read(BASE_OUTER_OPT, String)
    write(OUTER_OPT_FILE, replace(txt, r"^maxit\s+\d+"m => "maxit $OUTER_MAXIT"))
end
const INNER_OPT_FILE = joinpath(ROOT, "full_aod_diag", "ek_inner.opt")
const WVAL = parse(Int, get(ENV, "WVAL", "8000"))

params = (
    server=1, user=2, fakeData=1, DFake=DVAL, seedFakeData=889, counterType=1, counterExplicit=0,
    θHat=0, σHat=2.5, baseIndex=2, W=WVAL, seedU=888,
    importanceSampling=0, importanceSamplingFactor=2, stratifiedSampling=0, IndMomentOrder=5,
    θConstant=0, gravMoment=1, localGravityMoment=0, localGravityCrossMoment=0,
    GravityMomentFirstApproach=0, sameMarginalsMoment=0, NoScalingforSameMartingale=1,
    useCDFforMarginalMatching=0, independenceMoment=0, momentOrder=5, momentOrderForBaseIndex=50,
    ForceFrechetMarginal=0, OuterScaling=1, useParallel=0, usePMM=0, PMMGammaOnly=0,
    NormalizeMoments=0, useConfidenceIntervals=0, ConfidenceLevel=0.05, δGridType=0, δ_ref=1,
    refIndex1=1, OuterLoop=1, UoModel=1, use_Jacobian=0, calc_δ_star_initial=1, Jac_W=WVAL,
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
ρ = 2e-3; δ = DELTA_VAL
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

# seq_gravcol: identical to production except (a) takes `warm_p`, (b) when GRAVITY_SEED is true
# AND both warm/warm_p are available, seeds the first CC solve with the linearized gravity moment
# from the PREVIOUS theta's converged (p, umat) instead of solving blind. Everything downstream
# (invert_omitted, the influence-function refinement loop, damping) is untouched.
function seq_gravcol(θ; δ::Real = δ, maxit = 20, tol = 5e-4, warm = nothing,
                     warm_p::Union{Nothing,AbstractVector} = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    invert_omitted(p_arg; warm = nothing) = begin
        um = zeros(D, D); um[:, focal] .= uf
        all_converged = true
        for d in omitted
            ui = warm === nothing ? nothing : warm[:, d]
            inv = invert_destination(log_x, p_arg, λData[:, d]; ref = ref, ρ = ρ, tol = 1e-8,
                                     maxit = 150, ls_iters = 50, u_init = ui)
            um[:, d] .= inv.u_full
            if !inv.converged
                all_converged = false
                λd = λData[:, d]
                H = free_hessian(inv.stats, ρ; ref = ref)
                @printf("      [DIVERGENCE] dest %d NOT converged: iters=%d share_err=%.3e ‖u‖_max=%.3e  λ̂ range=[%.3e,%.3e]  achieved_share range=[%.3e,%.3e]  cond(H)=%.3e\n",
                        d, inv.iterations, inv.max_abs_share_error, maximum(abs, inv.u_full),
                        minimum(λd), maximum(λd), minimum(inv.model_shares), maximum(inv.model_shares), cond(Matrix(H)))
                flush(stdout)
            end
        end
        um, all_converged
    end
    local p, ok
    if GRAVITY_SEED && warm !== nothing && warm_p !== nothing
        infl_seed = influence_function(log_x, warm_p, warm, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_beta)
        moments_aug_seed! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal_norm_directgp!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj)
            @. G[:, D+2] = infl_seed.ψ_bar + infl_seed.R_beta
        end
        p, ok = recover_lfd(θ, moments_aug_seed!, D + 2)
    else
        p, ok = recover_lfd(θ, EK_moments_focal_norm_directgp!, D + 1)
    end
    if !ok
        verbose && println("    [seq] initial recover_lfd FAILED")
        return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R
    try
        umat, all_ok_init = invert_omitted(p; warm = warm)
        if !all_ok_init
            verbose && println("    [seq] initial invert_omitted: at least one destination did NOT converge -- treating as infeasible")
            return zeros(W), Inf, Inf, nothing, fill(1.0/W, W), false
        end
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
            verbose && println("    [seq] iter $k: augmented recover_lfd FAILED")
            break
        end
        α = 1.0; acc = false
        for _ in 1:12
            p_try = (1 - α) .* p .+ α .* p_cand
            local um_try, R_try, all_ok_try
            try
                um_try, all_ok_try = invert_omitted(p_try; warm = umat); R_try = gravity_residual(um_try, logτ, logw, σ).R_mean
            catch
                α *= 0.5; continue
            end
            if all_ok_try && isfinite(R_try) && abs(R_try) < abs(R); p = p_try; umat = um_try; R = R_try; acc = true; break; end
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

# Same as make_stateful_moments in run_profiled_production.jl, PLUS a warm_p::Ref threading the
# previous theta's converged p alongside warm (umat), and a live counter of how many gravity-
# INFEASIBLE (ok=false) callbacks are hit -- our primary "drift" signal.
function make_stateful_moments(; use_exact_grad::Bool = true, find_smallest::Bool = false, δ::Real = δ)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastRmean = Ref(NaN); lastRcol = Ref(NaN); lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0); ninfeas_bad = Ref(0)
    max_absR_seen = Ref(0.0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    warm_p = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol(θf; δ = δ, warm = warm[], warm_p = warm_p[])
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if isfinite(Rmean); max_absR_seen[] = max(max_absR_seen[], abs(Rmean)); end
                if ok
                    gcol[] = c; warm[] = um; warm_p[] = p; nfeas[] += 1
                    dRdθ[] = use_exact_grad ? grad_R_theta(θf, um, p) : zeros(length(θf))
                    EK_moments_focal_norm_directgp!(Ktmp, Gtmp, θf, view(U, 1:1, :), (γ = γ,))
                    κθ = Ktmp[1]
                    if (find_smallest && κθ < best_κ[]) || (!find_smallest && κθ > best_κ[])
                        best_κ[] = κθ; best_θ[] = copy(θf); best_warm[] = copy(um)
                    end
                else
                    gcol[] = INFCOL
                    dRdθ[] = zeros(length(θf))
                    ninfeas_bad[] += 1
                end
                neval[] += 1
                @printf("    [%s θ-eval %d, feasible %d, infeasible %d] R_mean=%.3e ok=%s t=%.2fs\n",
                        OUT_TAG, neval[], nfeas[], ninfeas_bad[], Rmean, ok, time()-t0); flush(stdout)
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
    return m!, gcol, lastRmean, best_θ, best_κ, best_warm, neval, nfeas, ninfeas_bad, max_absR_seen
end

function make_seq_div_grad_fn!(obj, fpmap)
    d = obj.d; oci = obj.outer_constr_index
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        obj(inner_x, Float64[], Float64[]; constr = zeros(1))
        λ = collect(@view inner_x[2:end])
        Usub = obj.U[1:obj.N, :]
        f = x -> CS._methodB_envelope_scalar(reconstruct_full(x, fpmap), obj.moments!, obj.γ, Usub, λ, obj.arg1, d, oci)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
        end
        ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
        return g_free
    end
end

function outer_solve_nested_cached(find_smallest, θinit; use_exact_grad::Bool = true, δ::Real = δ)
    d = D + 2; oci = d + 1
    CS.check_methodB_valid(d, oci)
    m!, gcol, lastRmean, best_θ, best_κ, best_warm, neval, nfeas, ninfeas_bad, max_absR_seen =
        make_stateful_moments(; use_exact_grad = use_exact_grad, find_smallest = find_smallest, δ = δ)
    obj = CS.PsiObjectiveBundleImplicitMethodB(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,
        outer_loop_opt = OUTER_OPT_FILE, inner_loop_opt = INNER_OPT_FILE)

    l_full = length(θinit)
    free_idx_ = vcat(3, collect(4:3+D))
    fixed_idx = [1, 2]
    fixed_vals = θinit[fixed_idx]
    fpmap = CS.FreeParamMap(l_full, free_idx_, fixed_idx, fixed_vals)
    @assert CS.n_free(fpmap) == D + 1

    div_grad_fn! = make_seq_div_grad_fn!(obj, fpmap)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest
    end

    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        has_gravity = false, use_cache = true, outer_loop_opt = OUTER_OPT_FILE)

    gp = r.θ_min_full[3]
    gp, r.θ_min_full, r.nStatus, best_θ[], best_κ[], best_warm[], r.cache, neval[], nfeas[], ninfeas_bad[], max_absR_seen[]
end

gp2kappa(gp) = 1 - gp^(σ/(σ-1))
const find_smallest = BOUND_NAME == "upper"

@printf("\n=== DRIFT COMPARISON: %s method, D=%d, delta=%g, bound=%s, outer maxit=%d ===\n",
        OUT_TAG, D, DELTA_VAL, BOUND_NAME, OUTER_MAXIT)
t0 = time()
gp, θstar, nstatus, bθ, b_gp, bwarm, cache, neval, nfeas, ninfeas_bad, max_absR_seen =
    outer_solve_nested_cached(find_smallest, copy(θr0); use_exact_grad = true, δ = DELTA_VAL)
wall = time() - t0
κ = gp2kappa(gp)
_, Rstar, _, _, _, okstar = seq_gravcol(θstar; δ = DELTA_VAL)
CS.summarize(cache; label = "$OUT_TAG bound cache stats")
@printf("\nRESULT [%s]: gamma_p=%.6f kappa=%.6f nStatus=%d  KNITRO-own exact_R_mean=%.4e gravity_ok=%s\n",
        OUT_TAG, gp, κ, nstatus, Rstar, okstar)
@printf("  theta-evals=%d  feasible=%d  infeasible=%d  max|R_mean| seen across all evals=%.4e  wall=%.1fs\n",
        neval, nfeas, ninfeas_bad, max_absR_seen, wall)
if bθ === nothing
    @printf("  best-feasible: NONE FOUND\n")
else
    _, Rb, _, _, _, okb = seq_gravcol(bθ; δ = DELTA_VAL, warm = bwarm)
    @printf("  best-feasible: gamma_p=%.6f kappa=%.6f exact_R_mean=%.4e gravity_ok=%s\n", b_gp, gp2kappa(b_gp), Rb, okb)
end

open(joinpath(@__DIR__, "real_outer_comparison_results.csv"), "a") do io
    println(io, "$OUT_TAG,$BOUND_NAME,$DELTA_VAL,$OUTER_MAXIT,$gp,$κ,$nstatus,$Rstar,$okstar,$neval,$nfeas,$ninfeas_bad,$max_absR_seen,$wall")
end
println("\nDONE [$OUT_TAG]")
