# ============================================================================
# Task §7: gravity pivot composed with the relative-A anchor reduction.
# ADDITIVE ONLY -- does not modify gravity_elimination.jl or
# relative_a_coordinate_2026-07-31.jl, both reused unchanged. Requires both
# included first.
#
# Composition (PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md section 1 and
# section 2.1(c)'s closing note): the anchor reduction removes Ddest
# coordinates (one per destination), producing the retained vector `r`
# (length n_retained(spec) = D*Ddest-Ddest). The EXISTING gravity pivot then
# removes exactly ONE MORE coordinate, but must select its pivot cell from
# among the RETAINED cells only -- theory doc section 2.1(c) proves any
# anchor cell has EXACTLY ZERO gravity coefficient contribution when shifted,
# so selecting an anchor cell as the gravity pivot would be degenerate
# (dividing by a coefficient that, while not necessarily zero itself at a
# SINGLE cell, is the wrong object to eliminate against once that cell is no
# longer a free coordinate at all -- the pivot must be chosen from cells that
# still vary).
#
# Because gravity is exactly affine in z (gravity_elimination.jl's own
# verified premise) and z is exactly affine in r (decode_relative_A is
# affine: z[i] = r[pos]+gauge[d] on retained cells, z[i]=gauge[d] fixed on
# anchor cells), gravity is exactly affine in r too:
#   gravity(r) = offset_r0 + cr'r,   cr[pos] := c_full[retained_linear_indices(spec)[pos]]
#   offset_r0 := gravity_from_logz(decode_relative_A(zeros(n_retained), spec, gauge), ctx)
# (offset_r0 is gravity's value at r≡0, i.e. z≡gauge broadcast to every
# retained cell -- NOT gravity_elimination.jl's own g0, which is gravity at
# z≡0 everywhere including anchor cells; a different, but equally
# well-defined and equally reusable-via-gravity_from_logz, reference point).
# ============================================================================

isdefined(Main, :gravity_from_logz) || error("gravity_pivot_on_retained_2026-07-31.jl requires gravity_elimination.jl to be included first.")
isdefined(Main, :decode_relative_A) || error("gravity_pivot_on_retained_2026-07-31.jl requires relative_a_coordinate_2026-07-31.jl to be included first.")

struct PivotGravityElimOnRetained
    spec::AnchorSpec
    gauge::Vector{Float64}
    cr::Vector{Float64}          # length n_retained(spec): gravity coefficient in r-space
    offset_r0::Float64           # gravity value at r==0
    pivot_pos::Int                # position within r (1..n_retained(spec)) chosen as pivot
    other_pos::Vector{Int}        # n_retained(spec)-1 positions, the rest, in order
end

"""
    build_pivot_elimination_on_retained(ctx, spec, gauge; μ=nothing) -> PivotGravityElimOnRetained

Selects the gravity pivot from among `spec`'s retained (non-anchor) cells
only, `argmax|cr|` over the retained-coordinate gravity-coefficient vector
(mirrors `gravity_elimination.jl::build_pivot_elimination`'s `argmax|c|`,
restricted to eligible cells).
"""
function build_pivot_elimination_on_retained(ctx, spec::AnchorSpec, gauge::Vector{Float64}; μ::Union{Nothing,Float64} = nothing)
    Ddest = _ctx_ddest(ctx)
    (ctx.D, Ddest) == (spec.D, spec.Ddest) ||
        throw(DimensionMismatch("build_pivot_elimination_on_retained: ctx (D=$(ctx.D),Ddest=$Ddest) != spec (D=$(spec.D),Ddest=$(spec.Ddest))"))
    c_full = vec(μ === nothing ? gravity_linear_coeffs(ctx) : gravity_linear_coeffs(ctx; μ = μ))
    ridx = retained_linear_indices(spec)
    cr = c_full[ridx]
    offset_r0 = gravity_from_logz(decode_relative_A(zeros(length(ridx)), spec, gauge), ctx; μ = μ)
    pivot_pos = argmax(abs.(cr))
    other_pos = setdiff(1:length(ridx), pivot_pos)
    return PivotGravityElimOnRetained(spec, gauge, cr, offset_r0, pivot_pos, other_pos)
end

"r_free (length n_retained(spec)-1) -> full r (length n_retained(spec)), gravity-feasible EXACTLY."
function pivot_expand_on_retained(r_free::AbstractVector{T}, pe::PivotGravityElimOnRetained) where {T}
    length(r_free) == length(pe.other_pos) ||
        throw(DimensionMismatch("pivot_expand_on_retained: expected length $(length(pe.other_pos)), got $(length(r_free))"))
    r = zeros(T, length(pe.cr))
    @inbounds for (k, pos) in enumerate(pe.other_pos)
        r[pos] = r_free[k]
    end
    rhs = -pe.offset_r0 - sum(pe.cr[pe.other_pos[k]] * r_free[k] for k in eachindex(r_free))
    r[pe.pivot_pos] = rhs / pe.cr[pe.pivot_pos]
    return r
end

"full r (length n_retained(spec)) -> r_free (length n_retained(spec)-1), dropping the pivot coordinate."
function pivot_reduce_on_retained(r::AbstractVector, pe::PivotGravityElimOnRetained)
    length(r) == length(pe.cr) ||
        throw(DimensionMismatch("pivot_reduce_on_retained: expected length $(length(pe.cr)), got $(length(r))"))
    return r[pe.other_pos]
end

"""
    decode_full_z_on_retained(r_free, pe, ctx) -> Matrix{Float64}

Full composition: `r_free` (length `n_retained(spec)-1`, the final free-A
outer coordinate under the profiled reparameterization) -> full `z` (D x
Ddest), both anchor-fixed AND gravity-feasible exactly.
"""
function decode_full_z_on_retained(r_free::AbstractVector{T}, pe::PivotGravityElimOnRetained) where {T}
    r = pivot_expand_on_retained(r_free, pe)
    return decode_relative_A(r, pe.spec, pe.gauge)
end
