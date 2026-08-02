# ============================================================================
# Production outer bridge task (2026-08-01), §3: separate a cheap in-process
# structural check from a PERSISTED, cross-process-stable compatibility
# fingerprint.
#
# `structural_checksum` (profiled_outer_gradient_layout_contract_2026-08-01.jl)
# uses Julia's generic `hash(...)`, which is explicitly NOT guaranteed stable
# across Julia processes, Julia versions, or hash seeds (Base's own docs:
# "The hash value may change... between different runs of Julia"). That is
# fine for `validate_family_layout_contract`'s own in-process "did two objects
# built from the same point drift apart" assertion -- it is not fine as
# something written into a checkpoint or campaign manifest and compared after
# a process restart, which is exactly what a production outer runner needs
# (task §15: "stable layout digest ... a full-formulation checkpoint cannot
# be loaded into a profiled context").
#
# This file adds a second, independent fingerprint alongside the existing one
# -- it does NOT replace or edit `structural_checksum`/
# `validate_family_layout_contract` (those stay exactly as the outer-gradient
# branch built them; ADDITIVE ONLY):
#
#   runtime_structural_check(fctx)  -- thin, explicitly-named wrapper around
#     the existing validate_family_layout_contract, for callers that want the
#     "cheap in-process validation" concept under the name this task's spec
#     uses. Same throws-on-mismatch behavior, zero new logic.
#
#   stable_layout_digest(fctx; restriction_outer_param_names=Symbol[])
#     -> String (64-char lowercase hex SHA256)
#     A canonical, deterministic byte serialization of every field the task
#     spec lists (family, D, Ddest, destination IDs, anchor origins, retained
#     economic-row map, France ratio index, gravity pivot position/map,
#     economic dual range, restriction dual ranges and names, outer
#     coordinate names/order, restriction outer-parameter names/order,
#     normalization convention), hashed with SHA256 (stdlib `SHA`, not
#     `Base.hash`). Reproducible byte-for-byte across fresh Julia processes,
#     versions, and hash seeds, because it never touches `hash()`, object
#     identity, `Dict` iteration order, or anything else process-seeded --
#     only fixed-order writes of primitive values from already-ordered
#     Vector/UnitRange fields.
# ============================================================================

using SHA

isdefined(Main, :validate_family_layout_contract) ||
    error("profiled_stable_layout_digest_2026-08-01.jl requires profiled_outer_gradient_layout_contract_2026-08-01.jl to be included first.")

"""
    runtime_structural_check(fctx) -> NamedTuple

Explicit name for the "cheap in-process validation" the task spec asks for.
Pure wrapper -- identical behavior to `validate_family_layout_contract(fctx)`
(same throws, same return value), added only so call sites can say what they
mean ("this is the fast per-call check", not "this is the persisted
fingerprint") without duplicating any logic.
"""
runtime_structural_check(fctx) = validate_family_layout_contract(fctx)

"""
    NORMALIZATION_CONVENTION_TAG

Fixed string describing the sign/scale convention every stable digest bakes
in, so a digest computed under a future, DIFFERENT convention cannot collide
with one computed under this convention even if every other field matches.
Bump this string (and nothing else) if the convention itself ever changes:
  - dual vector x = [zeta; beta], zeta = x[1], beta = x[2:end]
  - fixed-dual functional q[w] = -zeta - t_economic[w] - t_restriction[w]
  - outer coordinates: log-relative-A on retained cells (anchor cell
    excluded), gauge-fixed gravity pivot eliminated via
    PivotGravityElimOnRetained (task's own r-space / pivot_pos convention)
  - restriction dual ranges are 1-based indices into beta (NOT into the full
    x vector), disjoint from and strictly after economic_dual_range, in
    COLUMN ORDER as returned by restriction_dual_ranges(fctx)
"""
const NORMALIZATION_CONVENTION_TAG = "profiled-v1:zeta=x[1]|beta=x[2:end]|q=-zeta-t_econ-t_restr|logrelA-retained-anchor-excluded|gravity-pivot-r-space"

_wr(io::IO, x::Integer) = write(io, Int64(x))
_wr(io::IO, x::AbstractString) = (write(io, Int64(sizeof(x))); write(io, x))
_wr(io::IO, x::Symbol) = _wr(io, String(x))
function _wr(io::IO, xs::AbstractVector)
    write(io, Int64(length(xs)))
    for x in xs
        _wr(io, x)
    end
end
_wr(io::IO, r::UnitRange{Int}) = (_wr(io, first(r)); _wr(io, last(r)))

"""
    stable_layout_digest(fctx; restriction_outer_param_names::Vector{Symbol}=Symbol[]) -> String

Canonical SHA256 hex digest of every field task §3 requires. `restriction_
outer_param_names` defaults to empty (correct for the unrestricted family,
and for any restricted family until §9's typed combined outer-coordinate
layout supplies real outer-parameter names for that family) -- pass the
family's `restriction_outer_ranges(fctx)` key order (§9) once that exists, so
the digest also detects an outer-parameter reordering, not just an economic
one.

Field order (fixed, part of the format -- do not reorder without bumping
NORMALIZATION_CONVENTION_TAG):
  family_kind, D, Ddest, destination_ids, anchor_origin_by_slot,
  retained_full_factual_j (economic-row map), france_ratio_reduced_j,
  gravity pivot_pos, gravity other_pos (pivot map), economic_dual_range,
  restriction_dual_ranges (name+range per entry, in order), outer coordinate
  names/order (pe.other_pos length + pivot_pos, i.e. the same r-space
  ordering `profiled_outer_coordinate_layout` fixes -- there is no separate
  named-coordinate list at the economic-A level, "names" there ARE the
  (origin,destination-slot) pairs in `layout.retained_origin`/
  `layout.retained_slot` order), restriction_outer_param_names,
  NORMALIZATION_CONVENTION_TAG.
"""
function stable_layout_digest(fctx; restriction_outer_param_names::Vector{Symbol} = Symbol[])
    v = runtime_structural_check(fctx)   # throws first if fctx is internally inconsistent
    layout, spec, pe = v.layout, v.spec, v.pe

    io = IOBuffer()
    _wr(io, String(family_kind(fctx)))
    _wr(io, layout.D)
    _wr(io, layout.Ddest)
    _wr(io, layout.destination_ids)
    _wr(io, layout.anchor_origin_by_slot)
    _wr(io, layout.retained_full_factual_j)
    _wr(io, layout.retained_origin)
    _wr(io, layout.retained_slot)
    _wr(io, layout.france_ratio_reduced_j)
    _wr(io, pe.pivot_pos)
    _wr(io, pe.other_pos)
    _wr(io, v.economic_dual_range)
    _wr(io, length(v.restriction_dual_ranges))
    for r in v.restriction_dual_ranges
        _wr(io, r.name)
        _wr(io, r.range)
    end
    _wr(io, restriction_outer_param_names)
    _wr(io, NORMALIZATION_CONVENTION_TAG)

    return bytes2hex(sha256(take!(io)))
end
