# Phase 2c: profiled full-gravity BOUNDS via an outer fixed-point over the linearized gravity moment.
#
# The per-θ nested sequential loop (run_sequential.jl) drives the exact gravity residual to 0 at a
# fixed θ. Integrating it into the outer optimizer with exact θ-gradients-through-the-loop is heavy;
# instead we use the faithful OUTER analog of that linearization:
#   1. Outer CC solve (existing solver, unchanged) with a FROZEN gravity column appended to the
#      focal moments  →  θ*.
#   2. At θ*, run the inner sequential loop to re-linearize the gravity column (ψ̄ + R).
#   3. Repeat until θ* and the column stabilize (and the exact R at θ* is small).
# At the fixed point θ* extremizes κ s.t. the min-divergence-with-gravity ≤ δ, and R(θ*)≈0 — the
# profiled-gravity solution. Each outer solve reuses the validated CC machinery; the gravity column
# is θ-independent data within a solve. (The remaining refinement is the per-θ nested loop with
# exact-value/approx-gradient; this outer fixed point is the robust first cut.)
#
#   julia --project=. sequential_gravity/run_profiled_bounds.jl      (needs KNITRO env)

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
θr0 = build_focal_theta(prep.θ_initial, D, focal)
ρ = 2e-3; δ = params.δ_ref
Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
omitted = [d for d in 1:D if d != focal]; ref = 1
logτ = log.(τ); logw = log.(wHat)

focal_u(θ) = begin
    μ = θ[1]; Acol = θ[5:4+D]
    AodPow = [ (Acol[o]*((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ)*(λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
    (σ - 1) .* (log.(1 ./ AodPow) .- logw .- logτ[:, focal])
end

# returns (p, ok): ok=false if the inner solve failed / produced non-finite LFD weights
function recover_lfd(θ, moments_fn, d)
    oci = d + 1
    obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
        d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
        l = length(θ), U = U, N = JacW, lower_limit = -5000,
        outer_loop_opt = "ek_outer_loop_options.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    val, x, nStatus = inner_loop(obj, θ)
    all(isfinite, x) || return fill(1.0 / W, W), false
    G = zeros(W, d); K = zeros(W); moments_fn(K, G, θ, U, (γ = γ,))
    arg0 = zeros(W)
    @inbounds for ω in 1:W; arg0[ω] = -x[1] - dot(view(G, ω, 1:oci-1), view(x, 2:length(x))); end
    LFD = zeros(W); dPsi!(LFD, arg0)
    s = sum(LFD)
    (isfinite(s) && s > 0 && all(isfinite, LFD) && all(≥(0), LFD)) || return fill(1.0 / W, W), false
    return LFD ./ s, true
end

# inner sequential loop at fixed θ.  Returns (col, R, umat, p, ok) where ok=true ONLY if the loop
# genuinely drove the exact residual to |R|≤tol (gravity is satisfiable at this θ). If the LFD /
# inversion is non-finite, or the loop cannot reach |R|≤tol, ok=false ⇒ θ is gravity-infeasible.
function seq_gravcol(θ; maxit = 20, tol = 5e-4, warm = nothing, verbose = false)
    μ = θ[1]
    (isfinite(μ) && μ > 0) || return zeros(W), Inf, nothing, fill(1.0/W, W), false
    log_x = build_log_x(Uσ, μ); uf = focal_u(θ)
    (all(isfinite, uf) && all(isfinite, log_x)) || return zeros(W), Inf, nothing, fill(1.0/W, W), false
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
    p, ok = recover_lfd(θ, EK_moments_focal!, D + 1)
    if !ok
        verbose && println("    [seq] initial recover_lfd (focal-only) FAILED")
        return zeros(W), Inf, nothing, fill(1.0/W, W), false
    end
    local umat, R
    try
        umat = invert_omitted(p; warm = warm)
        R = gravity_residual(umat, logτ, logw, σ).R_mean
    catch e
        verbose && println("    [seq] initial invert_omitted threw: ", e)
        return zeros(W), Inf, nothing, fill(1.0/W, W), false
    end
    isfinite(R) || return zeros(W), Inf, nothing, fill(1.0/W, W), false
    verbose && @printf("    [seq] init: R0=%.4e\n", R)
    col = zeros(W)
    for k in 1:maxit
        infl = influence_function(log_x, p, umat, λData, omitted, logτ, logw, σ; ref = ref, ρ = ρ, scale = :R_mean)
        col = infl.ψ_bar .+ infl.R_mean                      # E_F[col]=0 ⟺ E_F[ψ̄]=-R
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_mean
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
        verbose && @printf("    [seq] iter %d: R -> %.4e  accepted=%s  α=%.4f\n", k, R, acc, acc ? α : 0.0)
        acc || break
    end
    verbose && @printf("    [seq] FINAL: R=%.4e  |R|<=tol? %s  (maxit=%d)\n", R, abs(R) <= tol, maxit)
    return col, R, umat, p, abs(R) <= tol
end

# ------------------------------------------------------------------------------------------------
# EXACT gradient of R_mean w.r.t. θ, holding F (the LFD p) fixed — the envelope-theorem piece the
# frozen column was dropping. No autodiff through KNITRO or through the inversion's Newton loop:
#   • ∂R_sum/∂u[o,d] = Q̃[o,d]/(σ-1) for every (o,d) (closed form: R_sum = S_Q + (1/(σ-1))ΣQ̃·u,
#     since two_way_demean is a linear, idempotent, self-adjoint projection and Q̃ is already
#     demeaned — so no chain rule through the demean is needed); ∂R_mean/∂u = (∂R_sum/∂u)/D², the
#     literal sample-moment average E[ΔΔlogA·ΔΔlogτ]=0 with NO variance normalization (per review:
#     this is the correct scaling for the GMM moment condition itself, not a regression coefficient).
#   • the focal column u_focal(θ) is an explicit function of θ ⇒ ∂u_focal/∂θ via ForwardDiff.jacobian
#     on that plain function (cheap: D×l).
#   • each omitted column's ∂u_d/∂μ via the implicit function theorem on the inversion's fixed point
#     share_d(μ,u_d(μ))≡λ̂_d: ∂share_d/∂μ|_u + H_d·∂u_d/∂μ = 0 ⇒ ∂u_d/∂μ = -H_d⁻¹·∂share_d/∂μ|_u.
#     ∂share_d/∂μ|_u is ONE ForwardDiff.derivative of dest_share w.r.t. μ (u, p fixed) — an O(S)
#     functional pass, not a Newton solve. Only μ (θ[1]) enters via the omitted columns.
function grad_R_theta(θ, umat, p)
    l = length(θ); dRdθ = zeros(l)
    gr = gravity_residual(umat, logτ, logw, σ)
    fi = free_idx(ref, D)
    Jfocal = ForwardDiff.jacobian(focal_u, θ)            # D × l
    c_focal = gr.Qt[:, focal] ./ (σ - 1) ./ D^2
    dRdθ .+= Jfocal' * c_focal
    logp = log.(p); μ0 = θ[1]
    for d in omitted
        st = dest_stats(build_log_x(Uσ, μ0), logp, umat[:, d]; ρ = ρ)
        Hd = free_hessian(st, ρ; ref = ref)
        dsh = ForwardDiff.derivative(μ -> dest_share(build_log_x(Uσ, μ), logp, umat[:, d]; ρ = ρ)[1][fi], μ0)
        dud_dmu = -(Hd \ dsh)
        c_d = gr.Qt[fi, d] ./ (σ - 1) ./ D^2
        dRdθ[1] += dot(c_d, dud_dmu)
    end
    return dRdθ
end

# reduced θ bounds
function focal_bounds(θr)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2]; hi[2] = θr[2]; lo[1] = 0.001; hi[1] = 1/(σ-1) - 0.001; lo[5] = θr[5]; hi[5] = θr[5]
    lo, hi
end
θ_lo, θ_hi = focal_bounds(θr0)

# §18 nested integration: a STATEFUL moment function that, at each Float64 θ the outer optimizer
# visits, runs the sequential loop to (re)compute the converged gravity column, then fills the
# augmented moments. So Δ_sequential(θ) — the min-divergence-with-gravity the outer solver
# constrains ≤ δ — is recomputed exactly at every θ (not frozen). The gradient path (Dual θ) reuses
# the just-computed column as constant data, giving the exact constraint VALUE at every θ with an
# analytic focal gradient + a θ-independent gravity term (exact-value / approximate-gradient).
# A strictly-positive, non-constant column: E_F[INFCOL]=0 is IMPOSSIBLE for any distribution (the
# mean of positive numbers is positive), so appending it as an inner moment makes the CC min-div
# problem infeasible ⇒ the outer optimizer rejects that θ. Used when gravity can't be enforced.
const INFCOL = 1.0 .+ 0.1 .* sin.(1:W)

function make_stateful_moments(; use_exact_grad::Bool = true)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastR = Ref(NaN)
    lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)          # value path: (re)linearize gravity at this θ
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rθ, um, p, ok = seq_gravcol(θf; warm = warm[])
                lastR[] = Rθ; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; nfeas[] += 1
                    dRdθ[] = use_exact_grad ? grad_R_theta(θf, um, p) : zeros(length(θf))
                else
                    gcol[] = INFCOL                    # gravity-infeasible θ ⇒ reject
                    dRdθ[] = zeros(length(θf))
                end
                neval[] += 1
                if neval[] % 10 == 0
                    @printf("    [θ-eval %d, gravity-feasible %d] seqR=%.2e ok=%s seq_time=%.2fs\n",
                            neval[], nfeas[], lastR[], ok, time()-t0); flush(stdout)
                end
                lastθ[] = copy(θf)
            end
        end
        EK_moments_focal!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        nrow = size(G, 1)
        if eltype(θ) <: ForwardDiff.Dual && lastok[]
            # affine reconstruction col(θ) ≈ ψ̄_frozen[s] + (R0 + dRdθ·(θ-θ0)): correct value at θ0
            # (matches the Float64 pass) with the EXACT envelope-theorem gradient of R attached, so
            # the outer KNITRO gradient sees how A_focal/μ trade off against gravity satisfaction.
            Rlin = lastR[] + dot(dRdθ[], θ .- lastθ[])
            @inbounds @views @. G[:, D+2] = (gcol[][1:nrow] - lastR[]) + Rlin
        else
            @inbounds @views @. G[:, D+2] = gcol[][1:nrow]
        end
    end
    return m!, gcol, lastR
end

function outer_solve_nested(find_smallest, θinit; use_exact_grad::Bool = true)
    d = D + 2; oci = d + 1
    m!, gcol, lastR = make_stateful_moments(; use_exact_grad = use_exact_grad)
    obj = PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,   # gravity column changes per θ ⇒ stale warm-start invalid
        outer_loop_opt = "csw_outer_loop_settings_cluster.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    κ, θstar, st, _ = outer_loop(obj, θ_lo, θ_hi, copy(θinit))
    κ, θstar, st
end

USE_EXACT_GRAD = get(ENV, "PROF_EXACT_GRAD", "1") == "1"

Kchk = zeros(W); Gchk = zeros(W, D + 1); EK_moments_focal!(Kchk, Gchk, θr0, U, (γ = γ,))
@printf("\n=== profiled full-gravity bounds (§18 nested, exact_grad=%s), D=%d W=%d δ=%g ρ=%g ===\n",
        USE_EXACT_GRAD, D, W, δ, ρ)
@printf("point estimate κ(F*) = %.6f\n", Kchk[1])

results = Dict{Symbol,Any}()
for (name, fs) in ((:upper, false), (:lower, true))
    @printf("\n----- %s bound (sequential loop recomputed at every θ) -----\n", name); flush(stdout)
    t0 = time()
    κ, θstar, st = outer_solve_nested(fs, θr0; use_exact_grad = USE_EXACT_GRAD)
    _, Rθ, _, _, okθ = seq_gravcol(θstar)      # exact residual at the bound-achieving θ*
    @printf("  κ_%s = %.6f  (status %d)  exact R_mean(θ*) = %.3e  gravity-feasible=%s  wall %.1fs\n",
            name, κ, st, Rθ, okθ, time() - t0)
    results[name] = (κ = κ, R = Rθ, ok = okθ)
end

@printf("\n=== PROFILED FULL-GRAVITY BOUNDS (δ=%g) ===\n", δ)
@printf("  κ_lower = %.6f   (exact R_mean %.2e)\n", results[:lower].κ, results[:lower].R)
@printf("  point   = %.6f\n", Kchk[1])
@printf("  κ_upper = %.6f   (exact R_mean %.2e)\n", results[:upper].κ, results[:upper].R)
@printf("\nreference: focal-only (no gravity) [0.00055, 0.2326];  legacy all-A gravity [0.0102, 0.1595]\n")
