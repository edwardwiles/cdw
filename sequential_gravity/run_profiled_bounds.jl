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

# Primal CDW/CC hybrid-divergence functional φ — the Legendre dual of Psi (cc_algo/Psi.jl) — applied
# DIRECTLY to an explicit candidate distribution p (no optimization). Needed because the accepted
# `p` inside the sequential loop can be a DAMPED convex combination (1-α)p_k + α·p_candidate, which
# is a valid distributional iterate but is NOT itself the argmin of any single min-divergence
# problem (spec §15) — so its divergence from F* must be evaluated directly, not read off a solver's
# internal `val`. Derivation: Psi(a) = exp(a)-1 for a≤1, 0.5·e·(a²+1)-1 for a>1; Psi'=dPsi (matches
# dPsi! exactly); by Legendre duality with m=Psi'(a) ⟺ a=φ'(m), φ(m)=am-Psi(a) at that a:
#   φ(m) = m·log(m) - m + 1        for 0 < m ≤ e   (a=log m ≤ 1)
#   φ(m) = m²/(2e) - e/2 + 1        for m > e        (a=m/e > 1)
# divergence(p) = E_F*[φ(dF/dF*)] = (1/W) Σ_s φ(p[s]·W), since F* is uniform (π*[s]=1/W).
# Sanity check: divergence_of(fill(1/W,W)) = 0 (p=F* exactly ⇒ m≡1 ⇒ φ(1)=0).
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

# S_Q = Σ Q̃² depends ONLY on the tariff data (two_way_demean(logτ)), never on θ/F/A — a true global
# constant. So R_beta = R_sum/S_Q and R_mean = R_sum/D² are related by the FIXED, θ-independent
# positive constant D²/S_Q: R_beta ≡ (D²/S_Q) · R_mean, exactly, always (not an approximation). We
# use this to separate two different jobs cleanly:
#   • the IDENTIFICATION CONDITION / convergence check (|R_mean|≤tol, per review — the literal GMM
#     sample-moment average, no variance normalization) — unaffected by anything below;
#   • the NUMERICAL SCALE of the moment/gradient handed to the CC/KNITRO solvers, which we keep at
#     R_beta's scale (empirically well-conditioned; R_mean's ~34× smaller magnitude in this example
#     made KNITRO's outer search take oversized steps into extreme-θ territory it couldn't recover
#     from — diagnosed via the θ-eval trace of a maxit-100 run). Since accept/reject comparisons
#     (`abs(R_try)<abs(R)`) are invariant to a common positive rescaling, using R_mean throughout for
#     THOSE and R_beta's scale only for the col/gradient content is exact, not a compromise.

# inner sequential loop at fixed θ. Returns (col, R_mean, R_col, umat, p, ok) where `col` and `R_col`
# are R_beta-scaled (solver-facing) and `R_mean` is the UNSCALED identification-condition value used
# for the |R|≤tol / gravity-feasibility decision. ok=true ONLY if BOTH (a) the loop genuinely drove
# |R_mean|≤tol (gravity is satisfiable) AND (b) the recovered distribution p's actual divergence from
# F* is ≤ δ (checked via divergence_of, the exact primal CDW/CC functional — NOT assumed just because
# a solver converged). (b) was a real gap: gravity-feasibility alone does not imply the δ-neighborhood
# budget is respected, and this previously let the best-feasible-θ tracker report economically
# impossible points (negative GT, which is provably ≥0 — see the paper draft) that were gravity-
# consistent but corresponded to a distribution far outside the stated δ. If the LFD/inversion is
# non-finite, or the loop cannot reach tol, or divergence_of(p)>δ, ok=false ⇒ θ is infeasible.
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
    p, ok = recover_lfd(θ, EK_moments_focal!, D + 1)
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
        col = infl.ψ_bar .+ infl.R_beta                      # solver-facing (R_beta scale)
        Rcol = infl.R_beta
        abs(R) <= tol && break
        moments_aug! = (K, G, θθ, Uarg, obj) -> begin
            EK_moments_focal!(K, @view(G[:, 1:D+1]), θθ, Uarg, obj); @. G[:, D+2] = infl.ψ_bar + infl.R_beta
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
    δ_ok = div_p <= δ * (1 + 1e-6) + 1e-10   # tiny numerical slack on the boundary
    if verbose
        @printf("    [seq] FINAL: R_mean=%.4e gravity_ok=%s  divergence(p)=%.4e (budget δ=%.4g) δ_ok=%s\n",
                R, gravity_ok, div_p, δ, δ_ok)
    end
    return col, R, Rcol, umat, p, gravity_ok && δ_ok
end

# ------------------------------------------------------------------------------------------------
# EXACT gradient of R_beta w.r.t. θ, holding F (the LFD p) fixed — the envelope-theorem piece the
# frozen column was dropping, computed at R_beta's numerical scale to match `col`/`Rcol` (see the
# note above `seq_gravcol`: R_beta ≡ (D²/S_Q)·R_mean exactly, so this is the SAME gradient as
# R_mean's up to that fixed constant — only the solver-facing SCALE differs, not the underlying
# derivative). No autodiff through KNITRO or through the inversion's Newton loop:
#   • ∂R_sum/∂u[o,d] = Q̃[o,d]/(σ-1) for every (o,d) (closed form: R_sum = S_Q + (1/(σ-1))ΣQ̃·u,
#     since two_way_demean is a linear, idempotent, self-adjoint projection and Q̃ is already
#     demeaned — so no chain rule through the demean is needed); ∂R_beta/∂u = (∂R_sum/∂u)/S_Q.
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

# reduced θ bounds
FREEZE_MU = get(ENV, "FREEZE_MU", "0") == "1"
function focal_bounds(θr)
    lo = θr .* 1e-4; hi = θr .* 1e4
    lo[2] = θr[2]; hi[2] = θr[2]; lo[1] = 0.001; hi[1] = 1/(σ-1) - 0.001; lo[5] = θr[5]; hi[5] = θr[5]
    if FREEZE_MU
        lo[1] = θr[1]; hi[1] = θr[1]
    end
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

function make_stateful_moments(; use_exact_grad::Bool = true, find_smallest::Bool = false, δ::Real = δ)
    lastθ = Ref(fill(NaN, length(θr0)))
    gcol  = Ref(zeros(W))
    lastRmean = Ref(NaN)     # UNSCALED identification-condition value (reporting/logging only)
    lastRcol  = Ref(NaN)     # R_beta-scaled value actually baked into gcol[] (affine-reconstruction math)
    lastok = Ref(true)
    dRdθ  = Ref(zeros(length(θr0)))
    neval = Ref(0); nfeas = Ref(0)
    warm  = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    # best-feasible-θ tracker: independent of what KNITRO's own terminal iterate ends up being, the
    # BEST (extremal κ, subject to gravity-feasible |R_mean|≤tol) θ seen during the ENTIRE search.
    # Also snapshot the umat that ACHIEVED that feasibility: the sequential loop is warm-start
    # dependent (it's a nested iterative procedure, not a pure function of θ alone), so re-verifying
    # this θ later with a COLD start can genuinely fail even though it was truly feasible via the
    # warm-started trajectory reached during the search — re-verify warm, not cold.
    best_θ = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    best_κ = Ref(find_smallest ? Inf : -Inf)
    best_warm = Ref{Union{Nothing,Matrix{Float64}}}(nothing)
    Ktmp = zeros(1); Gtmp = zeros(1, D + 1)
    function m!(K, G, θ, Uarg, obj)
        if !(eltype(θ) <: ForwardDiff.Dual)          # value path: (re)linearize gravity at this θ
            θf = Float64.(θ)
            if θf != lastθ[]
                t0 = time()
                c, Rmean, Rcol, um, p, ok = seq_gravcol(θf; δ = δ, warm = warm[])
                lastRmean[] = Rmean; lastRcol[] = Rcol; lastok[] = ok
                if ok
                    gcol[] = c; warm[] = um; nfeas[] += 1
                    dRdθ[] = use_exact_grad ? grad_R_theta(θf, um, p) : zeros(length(θf))
                    EK_moments_focal!(Ktmp, Gtmp, θf, view(U, 1:1, :), (γ = γ,))
                    κθ = Ktmp[1]
                    if (find_smallest && κθ < best_κ[]) || (!find_smallest && κθ > best_κ[])
                        best_κ[] = κθ; best_θ[] = copy(θf); best_warm[] = copy(um)
                    end
                else
                    gcol[] = INFCOL                    # gravity-infeasible θ ⇒ reject
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
        EK_moments_focal!(K, @view(G[:, 1:D+1]), θ, Uarg, obj)
        nrow = size(G, 1)
        if eltype(θ) <: ForwardDiff.Dual && lastok[]
            # affine reconstruction col(θ) ≈ ψ̄_frozen[s] + (R_beta,0 + dRdθ·(θ-θ0)): correct value at
            # θ0 (matches the Float64 pass, R_beta scale) with the EXACT envelope-theorem gradient of
            # R_beta attached, so the outer KNITRO gradient sees how A_focal/μ trade off against
            # gravity satisfaction, at a numerical scale that's empirically well-conditioned for the
            # solver (unlike R_mean's ~34× smaller magnitude — see the note above seq_gravcol).
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
    m!, gcol, lastRmean, best_θ, best_κ, best_warm = make_stateful_moments(; use_exact_grad = use_exact_grad, find_smallest = find_smallest, δ = δ)
    obj = PsiObjectiveBundleImplicit(δ = δ, find_smallest = find_smallest, γ = γ,
        (moments!) = m!, moments_jacobian! = error, d = d, outer_constr_index = oci,
        inequality_index = Int64[], complement_index = [0 0], l = length(θinit), U = U, N = JacW,
        lower_limit = -50, use_cached_x = false,   # gravity column changes per θ ⇒ stale warm-start invalid
        outer_loop_opt = "csw_outer_loop_settings_cluster.opt", inner_loop_opt = "ek_inner_loop_options.opt")
    κ, θstar, st, _ = outer_loop(obj, θ_lo, θ_hi, copy(θinit))
    κ, θstar, st, best_θ[], best_κ[], best_warm[]
end

USE_EXACT_GRAD = get(ENV, "PROF_EXACT_GRAD", "1") == "1"

Kchk = zeros(W); Gchk = zeros(W, D + 1); EK_moments_focal!(Kchk, Gchk, θr0, U, (γ = γ,))
POINT_EST = Kchk[1]

function run_at_delta(δval::Real)
    @printf("\n=== profiled full-gravity bounds (§18 nested, exact_grad=%s), D=%d W=%d δ=%g ρ=%g ===\n",
            USE_EXACT_GRAD, D, W, δval, ρ)
    @printf("point estimate κ(F*) = %.6f\n", POINT_EST)

    results = Dict{Symbol,Any}()
    for (name, fs) in ((:upper, false), (:lower, true))
        @printf("\n----- %s bound (sequential loop recomputed at every θ), δ=%g -----\n", name, δval); flush(stdout)
        t0 = time()
        κ, θstar, st, bθ, bκ, bwarm = outer_solve_nested(fs, θr0; use_exact_grad = USE_EXACT_GRAD, δ = δval)
        _, Rθ, _, _, _, okθ = seq_gravcol(θstar; δ = δval)      # exact residual (R_mean) at KNITRO's own θ*
        @printf("  KNITRO:        κ_%s = %.6f  (status %d)  exact R_mean(θ*) = %.3e  gravity-feasible=%s  wall %.1fs\n",
                name, κ, st, Rθ, okθ, time() - t0)
        if bθ === nothing
            @printf("  best-feasible: NONE FOUND (no gravity-feasible θ visited during the search)\n")
            results[name] = (κ = κ, R = Rθ, ok = okθ, best_κ = NaN, best_θ = nothing, best_ok = false)
        else
            # re-verify WARM (from the umat that achieved feasibility during the search) — the sequential
            # loop is warm-start dependent, so a cold re-check at the same θ can spuriously fail.
            _, Rb, _, _, _, okb = seq_gravcol(bθ; δ = δval, warm = bwarm)
            @printf("  best-feasible: κ_%s = %.6f  exact R_mean = %.3e  gravity-feasible=%s\n", name, bκ, Rb, okb)
            results[name] = (κ = κ, R = Rθ, ok = okθ, best_κ = bκ, best_θ = bθ, best_ok = okb)
        end
    end

    @printf("\n=== PROFILED FULL-GRAVITY BOUNDS (δ=%g) ===\n", δval)
    @printf("  κ_lower : KNITRO=%.6f (R_mean %.2e, feasible=%s)   best-feasible=%.6f (feasible=%s)\n",
            results[:lower].κ, results[:lower].R, results[:lower].ok, results[:lower].best_κ, results[:lower].best_ok)
    @printf("  point   = %.6f\n", POINT_EST)
    @printf("  κ_upper : KNITRO=%.6f (R_mean %.2e, feasible=%s)   best-feasible=%.6f (feasible=%s)\n",
            results[:upper].κ, results[:upper].R, results[:upper].ok, results[:upper].best_κ, results[:upper].best_ok)
    return results
end

DELTA_GRID = let s = get(ENV, "DELTA_GRID", "")
    isempty(s) ? [δ] : parse.(Float64, split(s, ","))
end

all_results = Dict{Float64,Any}()
for δval in DELTA_GRID
    all_results[δval] = run_at_delta(δval)
end

@printf("\n\n=== SUMMARY ACROSS δ (point estimate κ = %.6f) ===\n", POINT_EST)
@printf("%8s | %22s | %22s\n", "δ", "κ_lower (KNITRO/best)", "κ_upper (KNITRO/best)")
for δval in DELTA_GRID
    r = all_results[δval]
    @printf("%8.4g | %10.6f / %10.6f | %10.6f / %10.6f\n",
            δval, r[:lower].κ, r[:lower].best_κ, r[:upper].κ, r[:upper].best_κ)
end
