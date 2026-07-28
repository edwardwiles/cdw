# ================================================================================================
# No-moments/no-composite-G task (2026-07-28).
#
# Root cause (see docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md for the original flexible-CM/
# common-Fréchet investigation this generalizes to all 4 restricted families): every restricted
# family's production Hessian callback calls `_archC_prep_for_hessian!(obj, x)`
# (cm_hessian_architectures.jl), which reconstructs the per-draw dual index `r` into `obj.arg0` via
# a DENSE `BLAS.gemv!` against `obj.H[:, 2:1+outer_constr_index]` -- every moment column, economic
# AND restriction alike -- purely so the Hessian block-builders (`hessian_cm_structured!`/`_v2!`,
# `hessian_cm_frechet_structured!`/`_v2!`, `archA_partitioned_hess_cb_builder`) can call their OWN
# `ddPsi!(arg2, arg0)` immediately after. It also calls `Psi!(arg1, arg0)`, whose output (`arg1`) is
# never read again by any of those consumers -- confirmed dead weight, dropped here.
#
# Every restricted family's operator-mode FG callback (`CMLookupState`, `CMFrechetLookupState`,
# `CMMeanZCOperatorState`, `OriginZCOperatorState`) ALREADY computes this exact `r` with zero dense-G
# reads (via the shared `economic_forward!`/`economic_transpose!` for the economic block, plus a
# family-specific restriction piece), and already publishes it into `obj.arg0` at the end of every
# FG call ("keep obj in sync for a subsequent dense Hessian callback"). `_archC_prep_for_hessian!`
# then OVERWRITES that already-correct value with a redundant (and, under any column-fill skip, an
# actually-WRONG) dense recompute.
#
# This file provides the shared replacement:
#   - `dual_index!(st, x)` -- one function name, one method per family (defined in that family's own
#     kernel file, cm_lookup_kernels.jl / cm_frechet_lookup_kernels.jl / cm_meanzc_lookup_kernels.jl /
#     cm_originzc_lookup_kernels.jl), extracted VERBATIM from that family's existing FG functor's own
#     "compute r" block -- no new mathematics, a pure refactor. Each state's FG functor now calls
#     this SAME function instead of inlining the logic, so FG and Hessian-prep are provably running
#     the identical code path, not two independently-maintained copies.
#   - `HessianWeightCache` -- a strict (exact-match, no tolerance) same-point cache: a Hessian call
#     immediately following an FG call at the same `x` reuses the FG callback's own `obj.arg0` at
#     zero extra cost; any other case (different point, first call, KNITRO Hessian call not
#     immediately preceded by FG at that point) recomputes `r` fresh via `dual_index!`, still with
#     zero dense-G reads. Both paths are numerically identical (same `dual_index!` call, just
#     skipped on a hit) -- verified in the D=4/D=20 gates (test_no_moments_hessian_weights_*.jl),
#     which force each path explicitly.
#   - `operator_prep_for_hessian!(st, x)` -- the actual drop-in replacement for
#     `_archC_prep_for_hessian!(obj, x)` at every production Hessian callback call site. Populates
#     `st.obj.arg0` only; the caller's own subsequent `ddPsi!(arg2, arg0)` call (already present,
#     unchanged, in every Hessian block-builder) is untouched -- this file does not call `ddPsi!`
#     itself, by design, to keep the diff to exactly the one legacy step being replaced.
# ================================================================================================

isdefined(Main, :NO_DENSE_G_COUNTERS) || include(joinpath(@__DIR__, "no_dense_g_counters.jl"))

"""
    HessianWeightCache

Per-state (one per `CMLookupState`/`CMFrechetLookupState`/`CMMeanZCOperatorState`/
`OriginZCOperatorState` instance, i.e. per `cctx`/`octx`, campaign-lifetime, built once) same-point
cache for the Hessian-weight dual index `r`.

- `generation`: `objectid(st.core_cf_ref[])` at the time `x`/`r` were last computed -- the compressed
  economic factual (`CompressedFactual`) is rebuilt every outer point (winner assignments change),
  so a change in its identity invalidates the cache even if `x` happened to coincide numerically
  (never happens in practice, but this is the "context identity/generation" half of the required
  strict check, not just an `x`-match).
- `x`: an OWNED copy (not a view) of the exact inner vector last used, for elementwise `==` (no
  tolerance) comparison -- per this task's explicit requirement, never a tolerance-based match.
- `valid`: false until the first successful compute; also left `false` on construction so a
  cold-start Hessian call (no preceding FG call at this context yet) always takes the recompute path.
"""
mutable struct HessianWeightCache
    generation::UInt64
    x::Vector{Float64}
    valid::Bool
end

HessianWeightCache(n::Int) = HessianWeightCache(UInt64(0), fill(NaN, n), false)

"""
    _cf_identity(st) / _r_buffer(st)

Two small dispatched accessors that let ALL FIVE families (not just the 4 restricted ones) share
the exact same `HessianWeightCache`/`operator_prep_for_hessian!` machinery, per this task's "one
function, used everywhere it conceptually applies" design -- there is nothing restriction-specific
about "cache or recompute the per-draw dual index"; the ONLY thing that varies by family is which
buffer holds the current `r` and which object identifies the current outer-point "generation".

Default (covers `CMLookupState`/`CMFrechetLookupState`/`CMMeanZCOperatorState`/
`OriginZCOperatorState`, all of which hold `core_cf_ref::Ref{Any}` and compute into `st.arg0`):
"""
_cf_identity(st) = st.core_cf_ref[]
_r_buffer(st) = st.arg0

"""
    _publish_dual_index_cache!(st, x)

Call from the END of a family's FG functor (right after its own `obj.arg0 .= _r_buffer(st)` sync
line) to record that `_r_buffer(st)`/`obj.arg0` now hold the correct `r` for this exact `x`, at the
current `_cf_identity(st)` -- enabling a same-point Hessian call immediately afterward to hit the
cache instead of recomputing.
"""
function _publish_dual_index_cache!(st, x::AbstractVector{Float64})
    cache = st.hw_cache
    cache.generation = objectid(_cf_identity(st))
    copyto!(cache.x, x)
    cache.valid = true
    return nothing
end

"""
    operator_prep_for_hessian!(st, x) -> nothing

Drop-in replacement for `_archC_prep_for_hessian!(obj, x)` at every production Hessian callback call
site, for a family whose operator-mode FG state is `st` (`cctx.cmlookup_st`/`octx.fg_lookup_st`).
Ensures `st.obj.arg0` holds the exact dual index `r` at `x`, via:

    fast path (cache hit):  reuse the exact-same-point `r` the FG callback already published
    safe fallback (miss):   recompute `r` fresh via `dual_index!(st, x)` (the family's own operator
                             forward kernel -- NOT a dense `obj.H` read), then publish it

Both paths are dense-G-free and numerically identical (the miss path calls the exact same
`dual_index!` method the FG functor itself calls). Does NOT call `Psi!`/`ddPsi!` -- the caller's own
subsequent `ddPsi!(arg2, arg0)` (already present in every Hessian block-builder) is untouched.

Per this task's explicit requirement: the cache match is exact (`generation === ` and `x == `, no
tolerance), and a cache miss recomputes directly from the given `x` with no assumption that KNITRO
called FG immediately before Hessian at this point.
"""
function operator_prep_for_hessian!(st, x::AbstractVector{Float64})
    cache = st.hw_cache
    cur_gen = objectid(_cf_identity(st))
    if cache.valid && cache.generation === cur_gen && cache.x == x
        record_hessian_weight_cache_hit!()
    else
        record_hessian_weight_cache_miss!()
        dual_index!(st, x)                 # writes the family's r buffer; dispatches on st's concrete type
        st.obj.arg0 .= _r_buffer(st)
        _publish_dual_index_cache!(st, x)
        record_hessian_weight_operator_recompute!()
    end
    return nothing
end
