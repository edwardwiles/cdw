# ============================================================================
# Phase 5: reconstruct the full-A competitiveness matrix / free-parameter
# vector from a converged sequential/profiled production run (upper OR lower
# bound), convert it into the full-A gamma_d≡1 gauge, and evaluate it in the
# EXACT full-A oracle (evaluate_fullA, oracle.jl) to certify genuine
# feasibility -- NOT an approximate/close point, an EXACTLY feasible one,
# since it is the same underlying (u, A, p) solution re-parameterized, not a
# new/different solution.
#
# ---- Why no conversion is needed at all for the focal column ------------
# sequential_gravity/focal_moments_directgp.jl::EK_moments_focal_norm_directgp!
# builds AodPow[o] = (Acol[o] * ((wHat[o]*tau[o,focal])/(wHat[1]*tau[1,focal]))^(1/mu)
#                      * (lambda[o,focal]/lambda[1,focal]))^(-mu)
# and full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp! builds
#   AodPow[o,d] = (Aod_theta[o,d] * ((wHat[o]*tau[o,d])/(wHat[1]*tau[1,d]))^(1/mu)
#                   * (lambda[o,d]/lambda[1,d]))^(-mu)          [cHat cancels]
# These are IDENTICAL formulas with Acol[o] playing the role of Aod_theta[o,focal]
# -- i.e. the sequential script's theta[4:3+D] (its "A[.,focal]" free block) IS,
# bit for bit, the same free parameter as full-A's Aod_theta[:,focal] under the
# gamma_d≡1 gauge. No conversion needed for the focal column; verified below by
# an independent round-trip (focal_u -> AodPow -> Aod_theta) equivalence check,
# not just asserted from the algebra.
#
# ---- Why the omitted columns DO need a conversion ------------------------
# The sequential method's omitted destinations (d != focal) are recovered via
# sequential_gravity/profiled_gravity.jl::invert_destination, which returns the
# competitiveness index u[.,d] in the DESTINATION-INVERSION gauge (u[ref,d]=0,
# ref=1) -- sequential_methodology.tex section "Gauge". This is a genuinely
# different gauge convention from full-A's gamma_d≡1/Aod_theta parameterization.
# The conversion path used here (methodology section "The gravity identification
# restriction", already-validated code path, not re-derived): first go from u to
# the LEVEL log A_od = log(w_o) + log(tau_od) + u_od/(sigma-1) [methodology's own
# formula, identical to what profiled_gravity.jl::gravity_residual uses
# internally to build the gravity moment], then invert full-A's own Aod_theta
# formula (same formula quoted above) for Aod_theta[o,d] given AodPow[o,d] =
# A_od^{-1} = exp(-log A_od):
#   Aod_theta[o,d] = AodPow[o,d]^(-1/mu) / ( ((wHat[o]*tau[o,d])/(wHat[1]*tau[1,d]))^(1/mu)
#                                              * (lambda[o,d]/lambda[1,d]) )
# ============================================================================

include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "oracle.jl"))

const SEQ_ROOT = joinpath(D4X_ROOT, "sequential_gravity")
include(joinpath(SEQ_ROOT, "focal_moments.jl"))            # build_focal_theta, theoretical_kappa_bounds
include(joinpath(SEQ_ROOT, "focal_moments_directgp.jl"))   # EK_moments_focal_norm_directgp!
include(joinpath(SEQ_ROOT, "profiled_gravity.jl"))          # ProfiledGravity module
using .ProfiledGravity
using JLD2, LinearAlgebra, Printf

"""
    build_reconstruction_context(; δ=1.0, find_smallest=true)

Builds the full-A ctx (via d4_exact_setup, EXACT same economy/free-param
layout used throughout this investigation) and re-hosts the sequential
method's own helper functions (recover_lfd, seq_gravcol, divergence_of --
copied VERBATIM from sequential_gravity/run_profiled_production.jl lines
74-183, byte-identical logic, not re-derived) to operate on ctx.γ/ctx.U
instead of re-running master_setup/master_prestep/master_prepare_cc a second
time -- ctx.γ IS the same synthetic economy (identical AD_PARAMS /
seedFakeData=889 / seedU=888 / D=4 / W=8000 / baseIndex=2 / σHat=2.5 to
run_profiled_production.jl's own `params`, confirmed by direct comparison of
the two param tuples), so this is a legitimate re-hosting, not a new/different
draw set.
"""
function build_reconstruction_context(; δ::Float64 = 1.0, find_smallest::Bool = true)
    ctx = d4_exact_setup(; δ = δ, find_smallest = find_smallest)
    γ = ctx.γ; D = ctx.D; U = ctx.U; W = size(U, 1); focal = ctx.bi; σ = ctx.σ
    ρ = 2e-3
    Uσ = γ.Uσ; λData = Matrix(reshape(γ.P, (D, D))'); wHat = γ.wHat; τ = γ.τ
    omitted = [d for d in 1:D if d != focal]; ref = 1
    logτ = log.(τ); logw = log.(wHat)

    focal_u(θ) = begin
        μ = θ[1]; Acol = θ[4:3+D]
        AodPow = [ (Acol[o]*((wHat[o]*τ[o,focal])/(wHat[1]*τ[1,focal]))^(1/μ)*(λData[o,focal]/λData[1,focal]))^(-μ) for o in 1:D ]
        (σ - 1) .* (log.(1 ./ AodPow) .- logw .- logτ[:, focal])
    end

    function recover_lfd(θ, moments_fn, d)
        oci = d + 1
        obj = PsiObjectiveBundleDelta(γ = γ, (moments!) = moments_fn, moments_jacobian! = error,
            d = d, outer_constr_index = oci, inequality_index = Int64[], complement_index = [0 0],
            l = length(θ), U = U, N = W, lower_limit = -5000,
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

    return (ctx = ctx, γ = γ, D = D, U = U, W = W, focal = focal, σ = σ, ρ = ρ,
            Uσ = Uσ, λData = λData, wHat = wHat, τ = τ, omitted = omitted, ref = ref,
            logτ = logτ, logw = logw, focal_u = focal_u, recover_lfd = recover_lfd,
            divergence_of = divergence_of, seq_gravcol = seq_gravcol)
end

"""
    aod_theta_from_AodPow(AodPow, o, d, μ, wHat, τ, λData)

Full-A's Aod_theta[o,d] as a function of AodPow[o,d] (= A_od^{-1} in the
economic model), inverting EK_moments_gammanorm_directgp!'s own formula
(full_aod_diag/moments_gammanorm.jl lines ~92-102 / winners.jl::factual_prices
lines 26-28) -- cHat cancels exactly (shown in the file header derivation), so
it is NOT needed here.
"""
function aod_theta_from_AodPow(AodPow_od::Real, o::Int, d::Int, μ::Real,
                                wHat::AbstractVector, τ::AbstractMatrix, λData::AbstractMatrix)
    baseline = ((wHat[o]*τ[o,d])/(wHat[1]*τ[1,d]))^(1/μ) * (λData[o,d]/λData[1,d])
    return AodPow_od^(-1/μ) / baseline
end

"""
    reconstruct_fullA_point(rc, θseq; δ=1.0) -> (x_free, diagnostics::NamedTuple)

`rc` from `build_reconstruction_context`. `θseq` = the sequential run's
converged/best-feasible theta = [μ, σ, γ'_focal, A[1,focal],...,A[D,focal]]
(length D+3). Returns the full-A free vector `x_free` (length 1+D^2) ready
for `evaluate_fullA(x_free, rc.ctx.obj-consistent ctx)`, plus a diagnostics
NamedTuple recording the fresh seq_gravcol() re-solve's own feasibility
verdict (gravity_ok, δ_ok, R_mean, divergence(p)) and the focal-column
round-trip equivalence check.
"""
function reconstruct_fullA_point(rc, θseq::AbstractVector; δ::Real = 1.0)
    D = rc.D; σ = rc.σ; μ = θseq[1]; focal = rc.focal
    @assert length(θseq) == D + 3 "expected sequential theta of length D+3=$(D+3), got $(length(θseq))"

    # 1. Fresh COLD re-solve of the sequential loop at this exact theta (no warm start passed --
    #    the saved JLD2 does not persist the destination-inversion warm state, only theta itself;
    #    a cold re-solve is the honest, from-scratch feasibility re-certification the task asks for).
    col, R, Rcol, umat, p, ok = rc.seq_gravcol(θseq; δ = δ, verbose = true)
    div_p = rc.divergence_of(p)

    # 2. Focal-column round-trip equivalence check: Acol (theta_seq[4:3+D]) should equal
    #    aod_theta_from_AodPow(AodPow_from_focal_u(...)) to near machine precision, since they are
    #    algebraically the SAME quantity (see file header). This is a genuine numerical check, not
    #    an assumed identity.
    Acol = θseq[4:3+D]
    uf = rc.focal_u(θseq)
    AodPow_focal_from_u = [exp(-(uf[o]/(σ-1) + rc.logw[o] + rc.logτ[o,focal])) for o in 1:D]
    Aod_theta_focal_roundtrip = [aod_theta_from_AodPow(AodPow_focal_from_u[o], o, focal, μ, rc.wHat, rc.τ, rc.λData) for o in 1:D]
    focal_roundtrip_max_abs_err = maximum(abs.(Aod_theta_focal_roundtrip .- Acol))

    # 3. Build the full D x D Aod_theta matrix: focal column directly (theta_seq's own Acol, exact,
    #    no conversion -- see file header); every omitted column converted from umat[:,d] via the
    #    AodPow round-trip.
    Aod_theta = zeros(D, D)
    Aod_theta[:, focal] .= Acol
    for d in rc.omitted
        for o in 1:D
            AodPow_od = exp(-(umat[o, d]/(σ-1) + rc.logw[o] + rc.logτ[o, d]))
            Aod_theta[o, d] = aod_theta_from_AodPow(AodPow_od, o, d, μ, rc.wHat, rc.τ, rc.λData)
        end
    end

    # 4. Assemble full-A theta_full and pack to x_free via the EXISTING FreeParamMap (not manual
    #    indexing) -- guarantees exact agreement with ctx's own layout.
    ctx = rc.ctx
    θ_full = zeros(ctx.l_full)
    θ_full[1] = μ; θ_full[2] = σ
    θ_full[3:2+D] .= 1.0                      # gamma_d slots, INERT under gammanorm (ignored by moments!, per file header)
    θ_full[3+D] = θseq[3]                     # gamma'_focal DIRECT -- identical convention in both scripts
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .= vec(Aod_theta)
    x_free = CS.pack_free(θ_full, ctx.m)

    diagnostics = (seq_R_mean = R, seq_gravity_ok = ok, seq_divergence_p = div_p, seq_delta_ok = div_p <= δ*(1+1e-6)+1e-10,
                   focal_roundtrip_max_abs_err = focal_roundtrip_max_abs_err,
                   Aod_theta = Aod_theta, umat = umat, p = p, θ_full = θ_full)
    return x_free, diagnostics
end
