# ============================================================================
# Part 2: conditional winner-boundary derivative for the focal-moments method.
#
# Verified equivalence (documented here, not assumed -- see
# derivative_methods_report.md "winner criterion" section for the full
# derivation): production's ACTUAL winner rule is argmin LEVEL PRICE
# (price[o] = wHat[o]*AodPow[o]*tau[o,focal]*U[o]^mu). The user's Part 2.1
# setup defines the winner via argmax Phi_od = b_od*X_o. These are the SAME
# event here (not in general): writing price[o] = c_o/z_o (z_o=U_o^{-mu}) and
# Phi_od = b_o*z_o^{sigma-1}, algebra gives Phi_od = K_o * price[o]^{-(sigma-1)}
# with K_o = b_o * c_o^{sigma-1}. Substituting the code's actual definitions
# (constConsσ[o] = b_o = (w_o*tau_o/A_o)^{sigma-1} = c_o^{1-sigma}, since
# c_o = w_o*tau_o/A_o) gives K_o = c_o^{1-sigma}*c_o^{sigma-1} = 1 for EVERY o.
# So Phi_od = price[o]^{-(sigma-1)} EXACTLY (no origin-specific constant) --
# argmin(price) = argmax(price^{-(sigma-1)}) = argmax(Phi) since -(sigma-1)<0
# reverses the order. Confirmed, not assumed.
#
# X_o = z_o^{sigma-1} = U_o^{-mu(sigma-1)} = U_o^{-1/beta}. Since U_o~Exp(1),
# X_o is a monotone power transform of an Exponential(1) variable; standard
# change-of-variables gives X_o ~ Frechet(beta) EXACTLY:
#   f*_X(x) = beta * x^(-beta-1) * exp(-x^(-beta)),   x>0.
# (Derivation: U=x^{-beta} (from x=u^{-1/beta}), du/dx=-beta*x^{-beta-1},
# f_X(x)=f_U(u(x))|du/dx|=exp(-x^{-beta})*beta*x^{-beta-1}.) This matches
# beta=theta*/(sigma-1) throughout this project: a Frechet(theta*) productivity
# draw raised to the (sigma-1) power is Frechet(theta*/(sigma-1)) -- a standard
# property of Frechet variables under power transforms.
#
# Threshold algebra: j wins destination "focal" iff b_j·X_j ≥ M_{-j} :=
# max_{o≠j} b_o·X_o, i.e. iff X_j ≥ X_j* := M_{-j}/b_j. Since b_j=exp(a_j),
# dX_j*/da_j = -X_j* (increasing a_j shrinks the threshold at rate X_j*). AT
# the threshold, b_j·X_j* = M_{-j} EXACTLY -- i.e. the "S*" needed for the
# boundary jump (share_mag at the tie) equals M_{-j} directly, with NO extra
# recomputation: S*(ω) = M_{-j}(ω).
# ============================================================================

"Frechet(β) density: f*(x) = β x^(-β-1) exp(-x^(-β)), x>0."
frechet_density(x::Real, β::Real) = β * x^(-β - 1) * exp(-x^(-β))

"""
    phi_and_runnerup(θ, γobj, U, D)

Φ[ω,o] = constConsσ[o]*U[ω,o]^(μ(1-σ)) = b_{o,focal}·X_o(ω) for every draw and
origin (the SAME quantity the moments code calls `share_mag`). Also returns,
per draw, the winner (max1_idx), its Φ value (max1_val), and the RUNNER-UP's
identity and value (max2_idx, max2_val) -- one O(W·D) pass, reused for every
coordinate j (Part 1.3-style caching: M_{-j}(ω) and its associated competitor
k(ω), for ANY j, are recoverable from these five cached vectors with no
further scan over origins).
"""
function phi_and_runnerup(θ::Vector{Float64}, γobj, U::Matrix{Float64}, D::Int)
    focal = γobj.baseIndex
    wHat = γobj.wHat; τ = γobj.τ; λData = reshape(γobj.P, (D, D))'
    μ = θ[1]; σ = θ[2]
    Acol = θ[4:3+D]
    W = size(U, 1)

    AodPow = Vector{Float64}(undef, D)
    constConsσ = Vector{Float64}(undef, D)
    wPow = wHat .^ (1 - σ)
    @inbounds for o in 1:D
        base = Acol[o] * ((wHat[o] * τ[o, focal]) / (wHat[1] * τ[1, focal]))^(1 / μ) * (λData[o, focal] / λData[1, focal])
        AodPow[o] = base^(-μ)
        constConsσ[o] = wPow[o] * (AodPow[o] * τ[o, focal])^(1 - σ)
    end

    Φ = Matrix{Float64}(undef, W, D)
    @inbounds for o in 1:D, ω in 1:W
        Φ[ω, o] = constConsσ[o] * U[ω, o]^(μ * (1 - σ))
    end

    max1_idx = Vector{Int}(undef, W); max1_val = Vector{Float64}(undef, W)
    max2_idx = Vector{Int}(undef, W); max2_val = Vector{Float64}(undef, W)
    @inbounds for ω in 1:W
        m1 = -Inf; m1i = 1; m2 = -Inf; m2i = 1
        for o in 1:D
            v = Φ[ω, o]
            if v > m1
                m2, m2i = m1, m1i
                m1, m1i = v, o
            elseif v > m2
                m2, m2i = v, o
            end
        end
        max1_idx[ω] = m1i; max1_val[ω] = m1; max2_idx[ω] = m2i; max2_val[ω] = m2
    end
    return (Φ=Φ, constConsσ=constConsσ, max1_idx=max1_idx, max1_val=max1_val, max2_idx=max2_idx, max2_val=max2_val)
end

"For coordinate j: (M_{-j}(ω), competitor k(ω)) for every draw, recovered from the cached top-2 (no rescan)."
function m_minus_j_and_competitor(j::Int, cache)
    W = length(cache.max1_idx)
    Mmj = Vector{Float64}(undef, W); k = Vector{Int}(undef, W)
    @inbounds for ω in 1:W
        if cache.max1_idx[ω] == j
            Mmj[ω] = cache.max2_val[ω]; k[ω] = cache.max2_idx[ω]
        else
            Mmj[ω] = cache.max1_val[ω]; k[ω] = cache.max1_idx[ω]
        end
    end
    return Mmj, k
end

# ============================================================================
# Part 2.4 (FIRST, per the user's explicit "do not proceed to production use
# until this test passes"): verify the boundary formula against the
# analytical Frechet RAW-MOMENT Jacobian, before touching the full dual
# integrand. Reference: J_d = β·diag(λ) + (1-β)λλ' (population, alpha=log(b)
# units) -- SAME formula validated for the full-A method earlier this
# project; the focal-reduced model's per-destination moment structure is
# identical (same economic model), so it applies unchanged here.
# ============================================================================

"""
Population INTENSIVE raw-moment Jacobian block (alpha-units): diag(λ_od), NO
β factor. Exact tautology: ∂(Φ_o·1{o wins})/∂α_o = 1{o wins}·Φ_o (product
rule; the indicator's own AD partial is exactly zero, and ∂Φ_o/∂α_o=Φ_o
exactly since Φ_o=e^{α_o}·X_o). E[1{o wins}·Φ_o] = λ_o·denomf by the
Frechet-calibration identity (this is E_F[G_o]=0 at the benchmark, rearranged)
-- λ_o alone, β does NOT appear in the smooth/AD part. (β only enters via the
boundary term below; the FULL formula β·λ+(1-β)λ² only equals smooth+boundary
after algebraic recombination -- its own leading "β·λ" term is NOT itself the
smooth part, a decomposition mistake worth flagging since it's easy to make.)
"""
raw_intensive_jacobian(λfocal::Vector{Float64}, β::Real) = Diagonal(λfocal) |> Matrix

"""
    raw_boundary_jacobian_mc(θ, γobj, U, D, β)

Monte Carlo estimate of the BOUNDARY raw-moment Jacobian block (alpha-units),
∂E[G_o]/∂α_j for o,j=1:D: for each draw ω and coordinate j, find M_{-j}(ω)
and competitor k(ω) (from the shared top-2 cache), then accumulate
`X_j*·f*(X_j*)·M_{-j}(ω)·(𝟙{o=j} − 𝟙{o=k(ω)})`, X_j* = M_{-j}(ω)/constConsσ[j].
O(D) for the shared top-2 pass + O(W·D) for the D coordinate columns (O(1)
extra work per (j,ω) pair) -- no re-scan of all D origins per j.

UNITS: this is in α=log(b_j) units (matching J_ref below, which is ALSO the
raw "β·diag(λ)+(1-β)λλ'" formula in α-units, not the earlier log(Acol)-unit
version). Both sides of the Part 2.4 comparison are self-consistently in
α-units, which is fine for THAT internal check, but do not reuse this
function's output directly against a log(Acol)-unit quantity (e.g. the
fixed-dual FD gradient) without an extra 1/β chain-rule factor -- see
`boundary_envelope_gradient` below, where forgetting exactly this factor was
a real, caught bug.
"""
function raw_boundary_jacobian_mc(θ::Vector{Float64}, γobj, U::Matrix{Float64}, D::Int, β::Real)
    cache = phi_and_runnerup(θ, γobj, U, D)
    W = size(U, 1)
    J = zeros(D, D)
    for j in 1:D
        bj = cache.constConsσ[j]
        Mmj, k = m_minus_j_and_competitor(j, cache)
        acc = zeros(D)
        @inbounds for ω in 1:W
            Xstar = Mmj[ω] / bj
            wgt = Xstar * frechet_density(Xstar, β) * Mmj[ω]
            acc[j] += wgt
            acc[k[ω]] -= wgt
        end
        J[:, j] .= acc ./ W
    end
    return J
end

"Scalar hybrid-divergence conjugate Ψ(a), matching cc_algo/Psi.jl::Psi! exactly (a<=1: exp(a); else 0.5e(a²+1); then -1)."
psi_scalar(a::Real) = (a <= 1.0 ? exp(a) : 0.5 * exp(1) * (a^2 + 1.0)) - 1.0

# ============================================================================
# Part 2.2/2.3: the full fixed-dual-integrand conditional winner-boundary
# gradient. INTENSIVE term derived by differentiating arg0[ω] within the
# CURRENT winner regime (holds bo(ω) fixed -- this is exactly what pointwise
# AD computes, verified against it in Part 1's sanity check and Part 2.4's
# raw-moment gate); BOUNDARY term is the conditional-MC estimator of the
# winner-switching Dirac contribution, evaluating the COMPLETE Ψ(arg0)
# integrand under both winner assignments (NOT a linearized moment times a
# fixed dual weight -- Ψ is nonlinear and evaluated exactly at both points).
# ============================================================================

"""
    boundary_envelope_gradient(θ, obj, x_fixed, γobj, U, D, β; Acol_offset=3)

Returns (grad, diag) where `grad[j]` = d(dual_criterion_fixed_x)/d(log Acol[j])
via intensive+boundary (j=1:D), and `diag` carries the intensive/boundary
split plus the shared Φ/top-2 cache for inspection. Requires `obj.H`/`obj.arg1`
already populated at (θ,x_fixed) -- call `obj.moments!` + `obj(x_fixed,
zeros(length(x_fixed)))` first (same pattern as Part 1's sanity check).
"""
function boundary_envelope_gradient(θ::Vector{Float64}, obj::PsiObjectiveBundleDelta, x_fixed::Vector{Float64},
        γobj, U::Matrix{Float64}, D::Int, β::Real; Acol_offset::Int=3)
    focal = γobj.baseIndex
    λData = reshape(γobj.P, (D, D))'
    μ = θ[1]; σ = θ[2]
    Γ = gamma(μ * (1 - σ) + 1)
    denomf = γobj.wHat[focal] * γobj.L[focal]
    W = size(U, 1)
    ζ = x_fixed[1]
    λ = @view x_fixed[2:end]          # length D+1: λ[1:D] for the D focal-share moments, λ[D+1] for the price-index moment
    sign_conv = obj.find_smallest ? -1.0 : 1.0

    cache = phi_and_runnerup(θ, γobj, U, D)

    # arg0_base(ω): arg0 with ALL D trade-share origins treated as losers AND the G[D+1] price-index
    # term excluded (added back separately below -- see the j==focal note).
    CONST_LOSERS = denomf / Γ * sum(λ[o] * λData[o, focal] for o in 1:D)
    arg0_base = fill(-ζ + CONST_LOSERS, W)
    G_Dplus1_actual = @view obj.H[:, 2+D+1]   # H columns: 1=K,2=const,3:2+D=G[1:D],2+D+1=G[D+1]

    # cc_prime/denom_prime: the SAME price-index-moment constants EK_moments_focal_norm_directgp!
    # builds (focal_moments_directgp.jl:42-43). Needed because when j==focal, the boundary hypothesis
    # sets X_focal TO THE THRESHOLD value -- and G[D+1] uses the SAME underlying draw
    # U[.,focal]^(mu(1-sigma)) = X_focal (NOT an independent draw), so it must be re-evaluated at the
    # threshold too, not left at its actual value. This was a real bug, caught because it is the ONLY
    # thing that structurally differs about j==focal, and only j==focal showed a residual mismatch
    # against the fixed-dual FD gradient that did NOT shrink with more draws (128k -> 800k, ruling out
    # plain MC noise as the explanation).
    Acol = θ[Acol_offset+1:Acol_offset+D]
    wHat = γobj.wHat; τ = γobj.τ; τPrime = γobj.τPrime
    base_focal = Acol[focal] * ((wHat[focal] * τ[focal, focal]) / (wHat[1] * τ[1, focal]))^(1 / μ) * (λData[focal, focal] / λData[1, focal])
    AodPow_focal = base_focal^(-μ)
    cc_prime = (AodPow_focal * τPrime[focal, focal])^(1 - σ)
    denom_prime = θ[3]^σ * γobj.LPrime[focal]

    grad = zeros(D)
    intensive_parts = zeros(D)
    boundary_parts = zeros(D)
    for j in 1:D
        bj = cache.constConsσ[j]
        Mmj, k = m_minus_j_and_competitor(j, cache)

        # ---- intensive: differentiate within the CURRENT winner regime ----
        acc_int = 0.0
        @inbounds for ω in 1:W
            if cache.max1_idx[ω] == j
                acc_int += obj.arg1[ω] * (-λ[j] / (Γ * β)) * cache.Φ[ω, j]
            end
        end
        intensive_j = acc_int / W
        if j == focal
            # extra smooth term: G[D+1] depends on Acol[focal] too (own price-index moment), same 1/beta chain rule.
            acc_extra = 0.0
            @inbounds for ω in 1:W
                acc_extra += obj.arg1[ω] * (-λ[D+1] / (Γ * β)) * cc_prime * U[ω, focal]^(μ * (1 - σ))
            end
            intensive_j += acc_extra / W
        end

        # ---- boundary: conditional MC over the winner-switching threshold ----
        # NOTE: the threshold/density algebra (dX_j*/dalpha_j=-X_j*, Frechet(beta) density) is derived
        # w.r.t. alpha_j=log(b_j), exactly like raw_boundary_jacobian_mc/verify_raw_moment_jacobian
        # above (which are therefore in ALPHA units, not log(Acol) units -- their Part 2.4 comparison
        # was self-consistent, both sides in alpha units, but does NOT by itself carry the chain rule
        # needed here). Converting to the code's actual free coordinate a_j=log(Acol[j]) needs the
        # SAME 1/beta factor already applied to the intensive term (dalpha_j/da_j=1/beta) -- missing
        # this here was a real bug, caught by comparing against the validated fixed-dual FD gradient
        # (Part 1): omitting it made the boundary term exactly beta=4x too large.
        acc_bnd = 0.0
        @inbounds for ω in 1:W
            Xstar = Mmj[ω] / bj
            G_Dp1 = (j == focal) ? (cc_prime * Xstar - denom_prime) / Γ : G_Dplus1_actual[ω]
            common = arg0_base[ω] - λ[D+1] * G_Dp1
            arg0_plus = common - λ[j] * Mmj[ω] / Γ
            arg0_minus = common - λ[k[ω]] * Mmj[ω] / Γ
            Hp = psi_scalar(arg0_plus)
            Hm = psi_scalar(arg0_minus)
            acc_bnd += Xstar * frechet_density(Xstar, β) * (Hp - Hm)
        end
        boundary_j = (acc_bnd / W) / β

        intensive_parts[j] = intensive_j
        boundary_parts[j] = boundary_j
        grad[j] = sign_conv * (intensive_j + boundary_j)
    end
    return grad, (intensive=intensive_parts, boundary=boundary_parts, cache=cache)
end

"""
    verify_raw_moment_jacobian(θ, γobj, U, D, β; gamma_norm=1.0)

Part 2.4's gate: builds intensive-only, intensive+boundary, and the reference
analytical J_d = (β·diag(λ)+(1-β)λλ')/gamma_norm, all in α=log(b) units, and
returns error reports for (a) intensive-only vs the reference's DIAGONAL
(should match closely -- this is "intensive alone reproduces old AD"), and
(b) intensive+boundary vs the FULL reference (diagonal AND off-diagonal).
"""
function verify_raw_moment_jacobian(θ::Vector{Float64}, γobj, U::Matrix{Float64}, D::Int, β::Real; gamma_norm::Real=1.0)
    focal = γobj.baseIndex
    λfocal = reshape(γobj.P, (D, D))'[:, focal]
    # NOTE: this file's J's are built directly from Phi=share_mag (E_F[share_mag]=lambda*denomf, NOT
    # lambda itself -- see moments_gammanorm-style deviation-object docstring in ../../diagnostics/
    # context.jl), so the population reference formula needs the SAME denomf scaling: G = (share_mag -
    # lambda*denomf)/Gamma, so dG/dalpha_j = denomf/Gamma * (classic per-SHARE formula), not
    # 1/Gamma * (...). denomf is a scalar constant (no alpha-dependence), so it does not change which
    # entries are diagonal/off-diagonal -- only an overall scale, but must be included for numerical match.
    denomf = γobj.wHat[focal] * γobj.L[focal]
    J_ref = denomf .* (β .* Diagonal(λfocal) .+ (1 - β) .* (λfocal * λfocal')) ./ gamma_norm |> Matrix
    J_intensive = denomf .* raw_intensive_jacobian(λfocal, β) ./ gamma_norm
    J_boundary = raw_boundary_jacobian_mc(θ, γobj, U, D, β) ./ gamma_norm
    J_total = J_intensive .+ J_boundary
    return (J_ref=J_ref, J_intensive=J_intensive, J_boundary=J_boundary, J_total=J_total)
end
