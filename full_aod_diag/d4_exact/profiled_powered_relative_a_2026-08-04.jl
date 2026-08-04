# task §4.2: :profiled_powered_relative_A, an OPTIONAL new REDUCED A-coordinate mode. ADDITIVE
# ONLY -- does not modify relative_a_coordinate_2026-07-31.jl, gravity_pivot_on_retained_2026-07-31.jl,
# or cm_aspace_coordinate.jl, all reused unchanged. Requires all three included first. Derivation:
# docs/audits/profiled-outer-ab-readiness-2026-08-04/POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md
# (§3-4: pure per-coordinate affine reparametrization of REDUCED's existing native `r_free`
# outer coordinate, using the SAME (logX,logY,theta) constants FULL's own :powered_aspace mode
# already uses -- proven a bijection there, not assumed here).
#
# :profiled_pivot_anchor_relative (native) REMAINS the default for every REDUCED family -- this
# file only registers :profiled_powered_relative_A as an available, opt-in mode (task §4.2's
# explicit "do not make the new mode the default before testing").

isdefined(Main, :PivotGravityElimOnRetained) ||
    error("profiled_powered_relative_a_2026-08-04.jl requires gravity_pivot_on_retained_2026-07-31.jl to be included first.")
isdefined(Main, :decode_relative_A) ||
    error("profiled_powered_relative_a_2026-08-04.jl requires relative_a_coordinate_2026-07-31.jl to be included first.")
isdefined(Main, :CMAPivotXY) ||
    error("profiled_powered_relative_a_2026-08-04.jl requires cm_aspace_coordinate.jl to be included first (CMAPivotXY/precompute_cm_aspace_xy/cm_fixed_theta).")

"""
    _powered_relative_free_indices(pe::PivotGravityElimOnRetained) -> Vector{Int}

Linear indices (into the flattened D x Ddest array, column-major, matching `_lin(o,d,D)=o+(d-1)*D`)
of `pe`'s own free (`r_free`) coordinates, in `r_free` order.
"""
_powered_relative_free_indices(pe::PivotGravityElimOnRetained) = retained_linear_indices(pe.spec)[pe.other_pos]

"""
    encode_powered_relative_A(r_free, pe, theta, xy::CMAPivotXY) -> a_free

Derivation doc §3 encode: REDUCED's native `r_free` (length `n_retained(pe.spec)-1`) -> powered
`a_free`, SAME length, pointwise affine per coordinate using FULL's own `(logX,logY)` constants
(`xy = precompute_cm_aspace_xy(ctx)`, ctx-only, byte-identical whether built from a FULL or
REDUCED `ctx` since both share `d20_real_setup_design`'s own `γ` object).
"""
function encode_powered_relative_A(r_free::AbstractVector{T}, pe::PivotGravityElimOnRetained,
        theta::Float64, xy::CMAPivotXY) where {T}
    length(r_free) == length(pe.other_pos) ||
        throw(DimensionMismatch("encode_powered_relative_A: expected length $(length(pe.other_pos)), got $(length(r_free))"))
    idx = _powered_relative_free_indices(pe)
    D = pe.spec.D
    a_free = zeros(T, length(r_free))
    @inbounds for k in eachindex(r_free)
        i = idx[k]
        d = div(i - 1, D) + 1
        a_free[k] = -(r_free[k] + pe.gauge[d] + xy.logY[i]) / theta - xy.logX[i]
    end
    return a_free
end

"""
    decode_powered_relative_A(a_free, pe, theta, xy::CMAPivotXY) -> r_free

Inverse of `encode_powered_relative_A` (derivation doc §3 decode).
"""
function decode_powered_relative_A(a_free::AbstractVector{T}, pe::PivotGravityElimOnRetained,
        theta::Float64, xy::CMAPivotXY) where {T}
    length(a_free) == length(pe.other_pos) ||
        throw(DimensionMismatch("decode_powered_relative_A: expected length $(length(pe.other_pos)), got $(length(a_free))"))
    idx = _powered_relative_free_indices(pe)
    D = pe.spec.D
    r_free = zeros(T, length(a_free))
    @inbounds for k in eachindex(a_free)
        i = idx[k]
        d = div(i - 1, D) + 1
        r_free[k] = -theta * (a_free[k] + xy.logX[i]) - pe.gauge[d] - xy.logY[i]
    end
    return r_free
end

"""
    decode_full_z_on_retained_powered(a_free, pe, theta, xy) -> Matrix{Float64}

Full composition under `:profiled_powered_relative_A`: `a_free -> r_free`
(`decode_powered_relative_A`, new) `-> r` (`pivot_expand_on_retained`, EXISTING unchanged) `-> z`
(`decode_relative_A`, EXISTING unchanged). Mirrors native mode's own
`decode_full_z_on_retained` exactly except for the first step -- same anchor/gravity-pivot
recovery, no duplicated logic (task §4.1's "anchor recovery"/"gravity-pivot recovery" are
therefore identical to native mode, not re-derived).
"""
function decode_full_z_on_retained_powered(a_free::AbstractVector{T}, pe::PivotGravityElimOnRetained,
        theta::Float64, xy::CMAPivotXY) where {T}
    r_free = decode_powered_relative_A(a_free, pe, theta, xy)
    r = pivot_expand_on_retained(r_free, pe)
    return decode_relative_A(r, pe.spec, pe.gauge)
end

"""
    encode_full_z_on_retained_powered(z, pe, theta, xy) -> Vector{Float64}

Inverse direction: full `z` (D x Ddest) -> `a_free`. `z -> r` (`encode_relative_A`, EXISTING
unchanged) `-> r_free` (`pivot_reduce_on_retained`, EXISTING unchanged) `-> a_free`
(`encode_powered_relative_A`, new).
"""
function encode_full_z_on_retained_powered(z::AbstractMatrix{T}, pe::PivotGravityElimOnRetained,
        theta::Float64, xy::CMAPivotXY) where {T}
    r = encode_relative_A(z, pe.spec, pe.gauge)
    r_free = pivot_reduce_on_retained(r, pe)
    return encode_powered_relative_A(r_free, pe, theta, xy)
end

"""
    powered_relative_gradient_rescale(g_r_free, theta) -> Vector{Float64}

Derivation doc §3d: `d(Δ)/d(a_free) = d(Δ)/d(r_free) * (-θ)`, a scalar rescale of REDUCED's
EXISTING native-coordinate analytic outer gradient (`shared_family_outer_gradient`'s own A-block
output, unchanged) -- mirrors `outer_coordinate_layout.jl::gradient_transform_unified`'s own
`g[2:end] .*= (-theta)` line for FULL's `:powered_aspace` mode verbatim. No new gradient
computation; this is the entire practical payoff of the affine-composition proof in the
derivation doc.
"""
powered_relative_gradient_rescale(g_r_free::AbstractVector{Float64}, theta::Float64) = g_r_free .* (-theta)

"""
    powered_relative_bounds(r_lo, r_hi, pe, theta, xy) -> (a_lo, a_hi)

Derivation doc §3f: since `a_free[k] = f_k(r_free[k])` is affine with NEGATIVE slope `-1/θ` (θ>0
always, a trade elasticity), an interval bound on `r_free` maps to an interval bound on `a_free`
by applying `f_k` to both endpoints and swapping order (the negative slope flips which endpoint is
the lower/upper bound) -- mirrors how FULL's own drivers already convert a `z_halfwidth`-style
bound under `:powered_aspace`.
"""
function powered_relative_bounds(r_lo::AbstractVector{Float64}, r_hi::AbstractVector{Float64},
        pe::PivotGravityElimOnRetained, theta::Float64, xy::CMAPivotXY)
    length(r_lo) == length(r_hi) == length(pe.other_pos) ||
        throw(DimensionMismatch("powered_relative_bounds: r_lo/r_hi must both have length $(length(pe.other_pos))"))
    a_at_lo = encode_powered_relative_A(r_lo, pe, theta, xy)
    a_at_hi = encode_powered_relative_A(r_hi, pe, theta, xy)
    # slope is negative (-1/theta), so encode(r_lo) > encode(r_hi) coordinatewise -- swap.
    return (min.(a_at_lo, a_at_hi), max.(a_at_lo, a_at_hi))
end
