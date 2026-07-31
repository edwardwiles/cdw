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

"""
gravity coefficient vector c (D x Ddest) in log(Aod_theta) coordinates: g_gravity(z) = sum(c.*z) + g0.

`μ` defaults to `ctx.fixed_vals[1]` (today's fixed-theta production behavior, byte-identical to
before this keyword existed). Flexible-theta mode (port 2026-07-25, see
docs/FLEXIBLE_THETA_ASPACE_MATHEMATICAL_PARAMETERIZATION_2026-07-25.md) must pass the CURRENT
base-point's `μ` explicitly, since `ctx.fixed_vals[1]` no longer corresponds to absolute index 1
of theta_full once mu is a free (not fixed) outer coordinate (make_flexible_theta moves it into
free_idx). c(μ) = μ .* q_tilde ./ N_obs is EXACTLY linear in μ -- see build_pivot_elimination_cheap
below for the theta-invariance consequences this enables.
"""
function gravity_linear_coeffs(ctx; μ::Float64 = ctx.fixed_vals[1])
    return (μ .* ctx.q_tilde) ./ ctx.N_obs
end

"D_dest (destination count) for a context ctx -- Ddest==ctx.D unless row_idx excludes ROW as a destination (Part A/CM+ZC extension, 2026-07-23)."
_ctx_ddest(ctx) = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D

"""
g_gravity evaluated directly from a log(Aod_theta) matrix z (D x Ddest), reusing gravity_value unchanged.
`μ` defaults to `θ_full[1]` (the calibration-time value baked into `ctx.θ0_up`) -- pass explicitly in
flexible-theta mode, since θ0_up[1] no longer holds mu once make_flexible_theta has repurposed
absolute-index-1's slot for the eta_theta outer-box bookkeeping.
"""
function gravity_from_logz(z::AbstractMatrix, ctx; μ::Union{Nothing,Float64} = nothing)
    Ddest = _ctx_ddest(ctx)
    Aod_θ = exp.(z)
    θ_full = copy(ctx.θ0_up)
    θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*Ddest] .= vec(Aod_θ)
    μ_use = μ === nothing ? θ_full[1] : μ
    lambda_g = reshape(ctx.γ.P, (Ddest, ctx.D))'
    Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_use)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_use)
    return gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs; exclude_diagonal=get(ctx, :exclude_diagonal_gravity, false))
end

"g0 = g_gravity at Aod_theta==1 (z==0) -- the affine offset. `μ` explicit in flexible-theta mode (see gravity_from_logz)."
gravity_offset(ctx; μ::Union{Nothing,Float64} = nothing) = gravity_from_logz(zeros(ctx.D, _ctx_ddest(ctx)), ctx; μ = μ)

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
function build_pivot_elimination(ctx; μ::Union{Nothing,Float64} = nothing)
    Ddest = _ctx_ddest(ctx)
    c = vec(μ === nothing ? gravity_linear_coeffs(ctx) : gravity_linear_coeffs(ctx; μ = μ))
    g0 = gravity_offset(ctx; μ = μ)
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

# ============================================================================
# Flexible-theta production port (2026-07-25): cheap, theta-invariant pivot cache.
#
# c(μ) = μ .* q_tilde ./ N_obs is EXACTLY linear in μ. For μ>0 (the theory-safe domain always has
# μ=1/theta>0), a positive scalar multiplier never changes an argmax, so
# pivot = argmax|c(μ)| = argmax|c0| (c0 := q_tilde./N_obs, pure data) is THETA-INVARIANT.
# Likewise `other_idx` and the pivot-reconstruction SLOPE (-c0[j]/c0[pivot], the μ cancels exactly)
# are theta-invariant. Only the affine OFFSET g0(μ) is theta-dependent, and -- verified directly
# from gravity_from_logz's own price reconstruction, not assumed -- g0 is EXACTLY affine in μ at a
# fixed gp/gamma base state (the (1/μ) exponent on the wHat*τ ratio and the -μ exponent on
# Aod_lvl/cHat combine through gravity_value's own bilinear structure to a pure affine map in μ;
# see docs/FLEXIBLE_THETA_RECTANGULAR_GRAVITY_AUDIT_2026-07-25.md for the full re-derivation on the
# ACTIVE, POST-OMIT-ROW sample -- this was NOT assumed from any pre-omit-ROW/z-space claim). This
# section computes the theta-invariant pieces ONCE and fits g0(μ)=a+b*μ from two probes, so no
# outer-loop iterate that only moves theta ever needs to rebuild the pivot choice, other_idx, or
# re-derive the affine offset from scratch -- an O(1) evaluation per theta probe instead of the
# O(D*Ddest) gravity_from_logz reconstruction build_pivot_elimination(ctx; μ=...) would otherwise
# pay at every theta probe (still cheap in absolute terms at D=20, but this avoids redundant work
# on every cb_G! theta secant call, which evaluates two probes per outer gradient).
#
# Generalizes gravity_elimination.jl's original (pre-flexible-theta, square-only) PivotGravityElimCache
# to the rectangular D x Ddest omit-ROW active layout -- NOT copied verbatim from the square D=D
# experimental prototype (see the mathematical-parameterization doc's rectangular audit section).
# ============================================================================

struct PivotGravityElimCache
    D::Int
    Ddest::Int
    pivot_lin::Int
    other_idx::Vector{Int}     # length D*Ddest-1, theta-invariant
    c0::Vector{Float64}        # q_tilde/N_obs, length D*Ddest, theta-invariant (pure data)
    slope::Vector{Float64}     # -c0[other_idx[k]]/c0[pivot_lin], length D*Ddest-1, theta-invariant
    a::Float64                 # g0(μ) = a + b*μ, fit at the gp/gamma state this cache was built from
    b::Float64
end

"""
    build_pivot_elimination_cheap(ctx; mu_probe1, mu_probe2) -> PivotGravityElimCache

Computes the theta-invariant pivot/other_idx/slope ONCE from data alone (`c0`, on the ACTIVE
D x Ddest sample), then fits the affine offset `g0(μ)=a+b*μ` from exactly two `gravity_offset`
evaluations at the CURRENT gp/γ base state (any two distinct μ values in the theory-safe domain
work -- the relation is exact, not approximate, so the fit is not sensitive to the probe choice;
this is verified, not merely asserted, by test_flexible_theta_aspace_d4.jl's pivot-affine-fit
gate and its D=20 counterpart). Call this once per outer base point (i.e. whenever gp changes);
do NOT rebuild it merely because theta moved -- pivot_expand_cheap below handles that in O(1).
"""
function build_pivot_elimination_cheap(ctx; mu_probe1::Float64, mu_probe2::Float64)
    @assert mu_probe1 != mu_probe2 "need two distinct mu probes to fit the affine offset"
    Ddest = _ctx_ddest(ctx)
    c0 = vec(ctx.q_tilde) ./ ctx.N_obs
    pivot = argmax(abs.(c0))
    other = setdiff(1:ctx.D*Ddest, pivot)
    slope = [-c0[j] / c0[pivot] for j in other]
    g0_1 = gravity_offset(ctx; μ = mu_probe1)
    g0_2 = gravity_offset(ctx; μ = mu_probe2)
    b = (g0_2 - g0_1) / (mu_probe2 - mu_probe1)
    a = g0_1 - b * mu_probe1
    return PivotGravityElimCache(ctx.D, Ddest, pivot, other, c0, slope, a, b)
end

"""
    assert_pivot_layout(z_free, pgc::PivotGravityElimCache)

Throws `DimensionMismatch` if `z_free` is not a pivot-reduced layout vector (length `D*Ddest-1`)
for `pgc`'s `(D,Ddest)`. Must run before the `@inbounds` scatter loop in `pivot_expand_cheap`: that
loop indexes `z_free[k]` for `k` up to `length(pgc.other_idx)`, so a too-short `z_free` is silent
undefined behavior (reads adjacent heap memory), and a too-long one is silently truncated rather
than rejected -- this exact bug class was caught live during the original z-space port
(theta_fixed_dual_delta_pivot / flexible_theta_production.jl).
"""
function assert_pivot_layout(z_free::AbstractVector, pgc::PivotGravityElimCache)
    expected = pgc.D * pgc.Ddest - 1
    length(z_free) == expected || throw(DimensionMismatch(
        "expected pivot-layout vector of length D*Ddest-1=$expected (D=$(pgc.D), Ddest=$(pgc.Ddest)), got length $(length(z_free))"))
    return nothing
end

"""
z_free (length D*Ddest-1), current μ -> full z (D x Ddest matrix), gravity-feasible EXACTLY,
O(D*Ddest) cost with NO re-derivation of pivot/other_idx/slope (those came from `pgc`, built once).
"""
function pivot_expand_cheap(z_free::AbstractVector{T}, pgc::PivotGravityElimCache, μ::Float64) where {T}
    assert_pivot_layout(z_free, pgc)
    z = zeros(T, pgc.D * pgc.Ddest)
    @inbounds for (k, i) in enumerate(pgc.other_idx)
        z[i] = z_free[k]
    end
    g0_mu = pgc.a + pgc.b * μ
    intercept = -g0_mu / (μ * pgc.c0[pgc.pivot_lin])
    acc = intercept
    @inbounds for k in eachindex(z_free)
        acc += pgc.slope[k] * z_free[k]
    end
    z[pgc.pivot_lin] = acc
    return reshape(z, pgc.D, pgc.Ddest)
end

"""
full z (D x Ddest) -> z_free (length D*Ddest-1), dropping the pivot coordinate. Identical to
`pivot_reduce` (kept as a separate name for symmetry with `pivot_expand_cheap`; the reduce
direction never needs μ at all, since dropping a coordinate doesn't touch the affine offset).
"""
function pivot_reduce_cheap(z::AbstractMatrix, pgc::PivotGravityElimCache)
    size(z) == (pgc.D, pgc.Ddest) || throw(DimensionMismatch(
        "expected full-layout $(pgc.D)x$(pgc.Ddest) matrix, got size $(size(z)) -- pivot_reduce_cheap " *
        "expects the full gravity-manifold matrix, not a pivot-reduced vector"))
    return vec(z)[pgc.other_idx]
end

"""
    pivot_elim_from_cache(pgc, μ) -> PivotGravityElim

Fills the ORIGINAL `PivotGravityElim` struct (needed by consumers that expect that exact type,
e.g. the D=20 production C+/buffered gradient backends) from the precomputed, never-rebuilt
`PivotGravityElimCache`, at O(D*Ddest) cost with NO argmax/setdiff/gravity_offset call: pivot_lin/
other_idx are copied straight from `pgc` (theta-invariant, computed once), `c = μ.*pgc.c0` is one
elementwise multiply, and `g0 = pgc.a + pgc.b*μ` is O(1) (the whole point of the analytic affine
fit -- no price-level reconstruction needed at all). Numerically identical to
`build_pivot_elimination(ctx; μ=μ)` (both give the same `(pivot_lin, other_idx, c, g0)`), just
without re-deriving anything.
"""
function pivot_elim_from_cache(pgc::PivotGravityElimCache, μ::Float64)
    c = μ .* pgc.c0
    g0 = pgc.a + pgc.b * μ
    return PivotGravityElim(pgc.D, pgc.Ddest, pgc.pivot_lin, c, g0, pgc.other_idx)
end
