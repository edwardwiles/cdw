# ============================================================================
# Task §10: exact gravity elimination. Currently (docs/fullA_d4_code_audit.md
# sec 5) gravity is a SECOND explicit KNITRO equality constraint, not
# eliminated -- this file builds both alternatives the task brief asks for,
# working in log(Aod_theta) coordinates where gravity is EXACTLY LINEAR
# (verified below, not assumed): from gravity_tariff.jl's own closed-form
# gradient `d g_gravity/d Aod_theta[o,d] = (q_tilde[o,d]/N_obs)*(mu/Aod_theta[o,d])`,
# the chain rule gives `d g_gravity/d log(Aod_theta[o,d]) = mu*q_tilde[o,d]/N_obs`
# -- CONSTANT, independent of Aod_theta's value, i.e. g_gravity is exactly
# affine in z:=log(Aod_theta).
# ============================================================================
using LinearAlgebra: nullspace, qr

"gravity coefficient vector c (D x D) in log(Aod_theta) coordinates: g_gravity(z) = sum(c.*z) + g0."
function gravity_linear_coeffs(ctx)
    μ = ctx.fixed_vals[1]
    return (μ .* ctx.q_tilde) ./ ctx.N_obs
end

"D_dest (destination count) for a context ctx -- Ddest==ctx.D unless row_idx excludes ROW as a destination (Part A/CM+ZC extension, 2026-07-23)."
_ctx_ddest(ctx) = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

"g_gravity evaluated directly from a log(Aod_theta) matrix z (D x Ddest), reusing gravity_value unchanged."
function gravity_from_logz(z::AbstractMatrix, ctx)
    Ddest = _ctx_ddest(ctx)
    Aod_θ = exp.(z)
    θ_full = copy(ctx.θ0_up)
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*Ddest] .= vec(Aod_θ)
    μ = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (Ddest, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ)
    return gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
end

"g0 = g_gravity at Aod_theta==1 (z==0) -- the affine offset."
gravity_offset(ctx) = gravity_from_logz(zeros(ctx.D, _ctx_ddest(ctx)), ctx)

struct PivotGravityElim
    D::Int                  # origin count
    Ddest::Int               # destination count (Ddest==D unless row_idx excludes ROW)
    pivot_lin::Int          # linear index (column-major) of the pivot entry within the D x Ddest A-block
    c::Vector{Float64}      # length D*Ddest, gravity_linear_coeffs flattened
    g0::Float64
    other_idx::Vector{Int}  # the D*Ddest-1 non-pivot linear indices, in order
end

"""
    build_pivot_elimination(ctx) -> PivotGravityElim

Chooses the A-block entry with the LARGEST |gravity coefficient| as the pivot
(task §10.A: "not near zero"), and returns the map z_free (D*Ddest-1 free
log-A entries, all except the pivot) -> full z (D*Ddest, pivot solved so
g_gravity(z)==0 exactly). D*Ddest==D^2 unless row_idx excludes ROW as a
destination (CM+ZC true-shrink extension, 2026-07-23).
"""
function build_pivot_elimination(ctx)
    Ddest = _ctx_ddest(ctx)
    c = vec(gravity_linear_coeffs(ctx))
    g0 = gravity_offset(ctx)
    pivot = argmax(abs.(c))
    other = setdiff(1:ctx.D*Ddest, pivot)
    return PivotGravityElim(ctx.D, Ddest, pivot, c, g0, other)
end

"z_free (length D*Ddest-1) -> full z (D x Ddest matrix), gravity-feasible EXACTLY."
function pivot_expand(z_free::AbstractVector{T}, pe::PivotGravityElim) where {T}
    z = zeros(T, pe.D * pe.Ddest)
    @inbounds for (k, i) in enumerate(pe.other_idx)
        z[i] = z_free[k]
    end
    rhs = -pe.g0 - sum(pe.c[pe.other_idx[k]] * z_free[k] for k in eachindex(z_free))
    z[pe.pivot_lin] = rhs / pe.c[pe.pivot_lin]
    return reshape(z, pe.D, pe.Ddest)
end

"full z (D x Ddest) -> z_free (length D*Ddest-1), dropping the pivot coordinate."
pivot_reduce(z::AbstractMatrix, pe::PivotGravityElim) = vec(z)[pe.other_idx]

struct NullspaceGravityElim
    D::Int
    Ddest::Int
    Z::Matrix{Float64}        # (D*Ddest) x (D*Ddest-1) orthonormal basis for {v : c'v == 0}
    z_anchor::Vector{Float64} # D*Ddest, one particular gravity-feasible point (minimum-norm)
    c::Vector{Float64}
end

"""
    build_nullspace_elimination(ctx) -> NullspaceGravityElim

Orthonormal nullspace parameterization: z = z_anchor + Z*zeta, zeta in
R^{D*Ddest-1}, Z an orthonormal basis for the 1-dimensional-constraint nullspace
{v : c'v = 0}. z_anchor is the MINIMUM-NORM solution to c'z_anchor = -g0
(z_anchor = -g0*c/||c||^2), which is automatically orthogonal to every
column of Z. D*Ddest==D^2 unless row_idx excludes ROW as a destination.
"""
function build_nullspace_elimination(ctx)
    Ddest = _ctx_ddest(ctx)
    c = vec(gravity_linear_coeffs(ctx))
    g0 = gravity_offset(ctx)
    D2 = ctx.D * Ddest
    Z = nullspace(reshape(c, 1, D2))   # (D*Ddest) x (D*Ddest-1), orthonormal (LinearAlgebra guarantees this)
    @assert size(Z, 2) == D2 - 1 "nullspace rank != D*Ddest-1 -- unexpected degeneracy in the gravity coefficient vector"
    z_anchor = (-g0 / dot(c, c)) .* c
    return NullspaceGravityElim(ctx.D, Ddest, Z, z_anchor, c)
end

function nullspace_expand(ζ::AbstractVector{T}, ne::NullspaceGravityElim) where {T}
    z = ne.z_anchor .+ ne.Z * ζ
    return reshape(z, ne.D, ne.Ddest)
end
nullspace_reduce(z::AbstractMatrix, ne::NullspaceGravityElim) = ne.Z' * (vec(z) .- ne.z_anchor)
