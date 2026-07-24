# ============================================================================
# Tariff-residualized exact gravity moment + closed-form analytic gradient.
#
# Spec (§9 of the production-refactor task):
#   g_gravity(A) = (1/N_obs) * Σ_{(o,d)} q_tilde[o,d] * log(A[o,d])
#   q[o,d] = log(1 + tariff[o,d]);  q_tilde = q residualized once with origin +
#            destination FE on the same sample;  N_obs = D^2 for the full grid.
#   ∂g_gravity/∂logA[o,d]   = q_tilde[o,d]/N_obs           (if outer var = logA)
#   ∂g_gravity/∂A[o,d]      = q_tilde[o,d]/(N_obs*A[o,d])  (if outer var = A level)
#
# This fake-economy pipeline has no separate tariff series distinct from the
# bilateral trade cost τ (confirmed: sequential_gravity/run_profiled_bounds.jl's
# own comment already treats "the tariff data" as `τ`; there is no
# `tariff[o,d]` field anywhere in prep/data). So q := log(τ) here, i.e.
# tariff := τ-1 in level terms -- the natural reading of "tariff" as the sole
# available bilateral-cost proxy in this dataset. q_tilde = withinTransform(τ)
# (misc/doubleDiff.jl already takes log(.) internally), precomputed ONCE per
# economy (data-only, never re-demeaned per callback -- §9's explicit
# requirement).
#
# WHICH "A"? EXPERIMENTS_FINDINGS.md's own audit ("Which A object? AodPow is
# the raw structural A") established AodPow enters the price directly and the
# STRUCTURAL comparative-advantage term is A_od = 1/AodPow. That is the "A"
# this formula means (matches the economic object the gravity condition is
# actually a statement about: log A_od orthogonal to log tau, not log of the
# price-multiplier AodPow).
#
# DERIVATION of the exact analytic gradient w.r.t. the FREE outer variable
# Aod_theta[o,d] (θ's actual A-block entries, LEVELS not logs):
#   By FWL/orthogonality (q_tilde ⊥ the origin+destination FE space by
#   construction): Σ q_tilde[o,d]*log(A[o,d]) = Σ q_tilde[o,d]*(within log A)[o,d]
#   for ANY A -- i.e. this "single-sided-residualized" form is EXACTLY equal to
#   the two-way-demeaned-both-sides form already used elsewhere in this
#   codebase (Σ withinTransform(τ)·withinTransform(AodPow)), up to sign
#   (A=1/AodPow ⟹ logA=-log(AodPow)) and the N_obs normalization. Proof:
#   q_tilde'x = (Mq)'x = q'M'x = q'Mx = q'M(Mx)=(Mq)'(Mx)=q_tilde'x_tilde for
#   the idempotent-symmetric FE-annihilator M, any x. So no new demeaning
#   machinery is needed -- g_gravity(θ) = -sumGrav_current(θ)/N_obs exactly,
#   where sumGrav_current is newGravityMoment!'s existing UoModel==1 formula.
#
#   For the GRADIENT: with μ held FIXED (never differentiated, per spec §1),
#   the map Aod_theta[o,d] -> Aod[o,d] -> AodPow[o,d] -> A_od[o,d] is
#   PURELY ELEMENTWISE (each entry depends only on its own Aod_theta[o,d],
#   data, and the fixed μ -- confirmed from moments!.jl's own construction:
#   `Aod = Aod_θ .* cHat .* (((wHat.*τ)./(wHat[1,1].*τ[1,:]'))^(1/μ)) .*
#   (lambda./lambda[1,:]')` is elementwise in Aod_θ for fixed μ). Composing:
#     log(AodPow[o,d]) = -μ*(log(Aod[o,d]) - log(cHat[o,d]))
#     log(Aod[o,d])    = log(Aod_theta[o,d]) + log(c[o,d])   (c = the fixed,
#                         data/μ-dependent elementwise multiplier above)
#     log(A_od[o,d])   = -log(AodPow[o,d]) = μ*log(Aod_theta[o,d]) + const[o,d]
#   ⟹ ∂log(A_od[o,d])/∂Aod_theta[o,d] = μ / Aod_theta[o,d]
#   ⟹ ∂g_gravity/∂Aod_theta[o,d] = (q_tilde[o,d]/N_obs) * (μ/Aod_theta[o,d])
#   -- a fully closed-form scalar per entry, NO ForwardDiff, NO draws loop.
# ============================================================================

"""
    precompute_q_tilde(τ) -> (q_tilde, N_obs)

Precompute ONCE per economy (data-only): the FE-residualized log-cost
regressor and the observation count for the FULL D×D grid.
"""
function precompute_q_tilde(τ::AbstractMatrix)
    D, Ddest = size(τ)   # Ddest==D unless τ is already destination-restricted (row_idx, Part A 2026-07-23)
    q_tilde = within_transform_rect(τ)    # = within(log(τ)); τ stands in for (1+tariff) in this dataset
    N_obs = D * Ddest
    return q_tilde, N_obs
end

"""
    gravity_value(Aod_θ, μ, q_tilde, N_obs) -> Float64

g_gravity(θ) = (1/N_obs) Σ q_tilde[o,d]·log(A_od[o,d]), A_od = 1/AodPow.
Computed via the FWL identity as -sumGrav_current/N_obs (see module docstring)
to avoid re-deriving the Aod_θ->Aod->AodPow chain here; matches
`newGravityMoment!`'s existing UoModel==1 value bit-for-bit up to the
documented sign flip + N_obs normalization. `Aod_θ`, `μ` enter only through
the caller-supplied `AodPow` (kept as an explicit argument for clarity /
testability against the existing formula).
"""
function gravity_value(τ::AbstractMatrix, AodPow::AbstractMatrix, q_tilde::AbstractMatrix, N_obs::Int)
    Wτ = within_transform_rect(τ)
    WAodPow = within_transform_rect(AodPow)
    sumGrav = zero(eltype(WAodPow))
    D, Ddest = size(τ)
    @inbounds for o in 1:D, d in 1:Ddest
        sumGrav += Wτ[o, d] * WAodPow[o, d]
    end
    return -sumGrav / N_obs
end

"""
    gravity_grad_free!(g_free, x_free, m::FreeParamMap, Aod_free_idx_local, μ, q_tilde, N_obs)

Fills `g_free` (length n_free) with the closed-form analytic gravity gradient.
`Aod_free_idx_local` gives, for each (o,d) FREE A-block entry (in θ's
row-major reshape order matching `reshape(θ[Aod_offset+1:Aod_offset+D^2],(D,D))`),
its position `k` in `x_free` (i.e. `x_free[k] == θ_full[Aod_offset + od_linear]`)
-- all other entries of `g_free` (μ, σ, γ'_focal, γ_θ if present) are exactly
zero, since g_gravity depends on Aod_θ ONLY.
"""
function gravity_grad_free!(g_free::AbstractVector, x_free::AbstractVector, D::Int,
        Aod_free_pos::AbstractMatrix{Int}, μ::Real, q_tilde::AbstractMatrix, N_obs::Int)
    fill!(g_free, 0.0)
    Ddest = size(Aod_free_pos, 2)   # Ddest==D unless row_idx excludes ROW (Part A, 2026-07-23); derived from Aod_free_pos's own shape rather than a new arg so existing callers (D4/D10 legacy production) need no changes
    @inbounds for o in 1:D, d in 1:Ddest
        k = Aod_free_pos[o, d]
        Aod_theta_od = x_free[k]
        g_free[k] = (q_tilde[o, d] / N_obs) * (μ / Aod_theta_od)
    end
    return g_free
end
