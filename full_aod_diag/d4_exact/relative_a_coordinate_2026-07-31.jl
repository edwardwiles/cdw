# ============================================================================
# Task §6: relative-A coordinate layer for the profiled-destination-scales
# reparameterization. ADDITIVE ONLY, per this repo's established convention
# (infeasibility_screen.jl/fast_range_screen.jl/autarky_cf.jl all follow the
# same "new function, verify vs trusted path, then wire in" pattern) -- does
# NOT modify context.jl, gravity_elimination.jl, outer_coordinate_layout.jl,
# or any of the >15 files FULL_TO_PROFILED_CHANGE_MATRIX_2026-07-31.csv
# classifies as "shared formula needs new inputs" for the Topic-2 gauge
# reconstruction.
#
# DESIGN NOTE (see FULL_TO_PROFILED_PIPELINE_CALL_GRAPH_2026-07-31.md topic 2
# and PROFILED_DESTINATION_SCALES_MASTER_2026-07-31.md section 3a(ii)): the
# existing Aod_theta -> Aod_lvl reconstruction formula already contains a
# DIFFERENT per-destination gauge device (dividing by origin-1's factual
# share, `lambda./lambda[1,:]'`). This file's anchor-relative layer is
# deliberately independent of that: it operates ENTIRELY in
# z = log(Aod_theta) space (the SAME coordinate gravity_elimination.jl's
# pivot already operates in), never touching the level-A reconstruction. This
# is exactly what theory doc PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md
# section 2.1 proves is valid: a common additive shift of z[:,d] is EXACTLY a
# common multiplicative rescale of the true level-A column, independent of
# how B(o,d) (the origin-1-gauge-dependent per-cell constant) varies by o.
# So there is no interaction to manage between the two gauge devices -- this
# layer can be built and tested without touching or duplicating the
# Topic-2 formula at all.
# ============================================================================

"""
    AnchorSpec

One anchor origin per active destination. `anchor_origin[d]` is the GLOBAL
origin index (1..D) of destination `d`'s anchor cell. Structurally guarantees
task §2.4's "exactly one anchor per destination" requirement: `anchor_origin`
is a `Vector{Int}` of length `Ddest`, so representing zero or two anchors for
the same destination is not expressible at the type level, not merely
checked at runtime.
"""
struct AnchorSpec
    D::Int
    Ddest::Int
    anchor_origin::Vector{Int}   # length Ddest

    function AnchorSpec(D::Int, Ddest::Int, anchor_origin::Vector{Int})
        length(anchor_origin) == Ddest ||
            throw(DimensionMismatch("AnchorSpec: anchor_origin has length $(length(anchor_origin)), expected Ddest=$Ddest"))
        all(1 .<= anchor_origin .<= D) ||
            throw(ArgumentError("AnchorSpec: anchor_origin entries must be in 1:$D, got $anchor_origin"))
        return new(D, Ddest, anchor_origin)
    end
end

"""
    default_anchor_spec(D, Ddest; overrides=Dict{Int,Int}()) -> AnchorSpec

Own-cell anchor for every destination (`anchor_origin[d] = d`), except
`overrides[d] = o` for any destination that needs a non-own anchor (e.g. the
task's Korea->Brazil rule: `overrides[kor_idx] = bra_idx`). Matches
`DESTINATION_SCALE_ANCHOR_MANIFEST_2026-07-31.json`'s rule set exactly when
called with the manifest's real-D20 indices; used directly with D=4 synthetic
indices for this file's own tests.
"""
function default_anchor_spec(D::Int, Ddest::Int; overrides::Dict{Int,Int} = Dict{Int,Int}())
    Ddest <= D || throw(ArgumentError("default_anchor_spec: Ddest=$Ddest must be <= D=$D (own-cell default requires destination d to be a valid origin index)"))
    anchor_origin = [get(overrides, d, d) for d in 1:Ddest]
    return AnchorSpec(D, Ddest, anchor_origin)
end

"n_retained(spec) -- number of free relative-A coordinates after removing one anchor per destination."
n_retained(spec::AnchorSpec) = spec.D * spec.Ddest - spec.Ddest

"Column-major linear index (matching gravity_elimination.jl's `i = o + (d-1)*D` convention)."
_lin(o::Int, d::Int, D::Int) = o + (d - 1) * D

"""
    anchor_linear_indices(spec) -> Vector{Int}

The `Ddest` linear indices (into a flattened `D x Ddest` column-major array)
that are anchor cells, one per destination, in destination order.
"""
anchor_linear_indices(spec::AnchorSpec) = [_lin(spec.anchor_origin[d], d, spec.D) for d in 1:spec.Ddest]

"""
    retained_linear_indices(spec) -> Vector{Int}

The `D*Ddest - Ddest` linear indices that are NOT anchor cells, in ascending
order (same convention as `gravity_elimination.jl`'s `other_idx`).
"""
retained_linear_indices(spec::AnchorSpec) = setdiff(1:(spec.D * spec.Ddest), anchor_linear_indices(spec))

"""
    build_anchor_gauge(z_calib::AbstractMatrix, spec::AnchorSpec) -> Vector{Float64}

Extracts the fixed gauge value `gauge[d] = z_calib[anchor_origin[d], d]` for
every destination, from a GENUINE calibration z-matrix (`log(Aod_theta)` at
`ctx.θ0_up`, e.g.) -- never from the gravity-elimination pivot's `zfree=0`
reference point (see this repo's standing CLAUDE.md warning). Matches the
task's stated preference `Ã_{j_d,d} = A*_{j_d,d}` (theory doc section 6/24:
"prefer the calibration-anchor gauge unless evidence favors another").
"""
function build_anchor_gauge(z_calib::AbstractMatrix{Float64}, spec::AnchorSpec)
    size(z_calib) == (spec.D, spec.Ddest) ||
        throw(DimensionMismatch("build_anchor_gauge: z_calib size $(size(z_calib)) != (D,Ddest)=($(spec.D),$(spec.Ddest))"))
    return [z_calib[spec.anchor_origin[d], d] for d in 1:spec.Ddest]
end

"""
    decode_relative_A(r::AbstractVector, spec::AnchorSpec, gauge::AbstractVector) -> Matrix{Float64}

`r` (length `n_retained(spec)`) -> full `z` (D x Ddest), where `z[o,d] =
gauge[d]` for the anchor origin and `z[o,d] = r[k] + gauge[d]` for every
other origin (`r[k]` is destination-`d`'s relative-A coordinate for origin
`o`, i.e. `r[k] = log(Aod_theta[o,d]) - log(Aod_theta[anchor_origin[d],d])`
at whatever point `r` describes).
"""
function decode_relative_A(r::AbstractVector{T}, spec::AnchorSpec, gauge::AbstractVector{Float64}) where {T}
    length(r) == n_retained(spec) ||
        throw(DimensionMismatch("decode_relative_A: expected length $(n_retained(spec)), got $(length(r))"))
    length(gauge) == spec.Ddest ||
        throw(DimensionMismatch("decode_relative_A: gauge length $(length(gauge)) != Ddest=$(spec.Ddest)"))
    z = zeros(T, spec.D, spec.Ddest)
    ridx = retained_linear_indices(spec)
    @inbounds for (k, i) in enumerate(ridx)
        d = div(i - 1, spec.D) + 1
        z[i] = r[k] + gauge[d]
    end
    @inbounds for d in 1:spec.Ddest
        z[spec.anchor_origin[d], d] = gauge[d]
    end
    return z
end

"""
    encode_relative_A(z::AbstractMatrix, spec::AnchorSpec, gauge::AbstractVector) -> Vector{Float64}

Inverse direction: full `z` (D x Ddest) -> `r` (length `n_retained(spec)`),
dropping the anchor cells. Does NOT require `z`'s anchor cells to already
equal `gauge` (they are simply dropped, not checked) -- `decode_relative_A
∘ encode_relative_A` is the identity everywhere EXCEPT at anchor cells, which
get forced to `gauge[d]` regardless of `z`'s original anchor-cell values
(exactly the intended coordinate reduction, not a bug).
"""
function encode_relative_A(z::AbstractMatrix{T}, spec::AnchorSpec, gauge::AbstractVector{Float64}) where {T}
    size(z) == (spec.D, spec.Ddest) ||
        throw(DimensionMismatch("encode_relative_A: z size $(size(z)) != (D,Ddest)=($(spec.D),$(spec.Ddest))"))
    ridx = retained_linear_indices(spec)
    r = zeros(T, length(ridx))
    @inbounds for (k, i) in enumerate(ridx)
        d = div(i - 1, spec.D) + 1
        r[k] = z[i] - gauge[d]
    end
    return r
end
