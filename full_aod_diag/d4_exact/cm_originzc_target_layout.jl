# ============================================================================
# Immutable target-layout abstraction for the mean/pairwise-ZC moment family.
# See docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md for the full derivation.
#
# `cm_meanzc_moments.jl` hard-codes ONE nu_k shared across all D origins
# (correct under common marginals). This file introduces the abstraction
# needed to ALSO support one nu_{o,k} per origin (no common marginals) --
# `MeanZCTargetLayout`, `SharedByPowerLayout`, `OriginByPowerLayout` -- without
# touching any existing cm_meanzc_*.jl code. Every subtype is a plain
# immutable struct; no mutable Ref, no closed-over state. Every function that
# consumes a layout takes the complete eta/nu vector as an explicit argument.
# ============================================================================

abstract type MeanZCTargetLayout end

"""
    SharedByPowerLayout(K_mean, K_pair)

One nu_k shared by every origin, per power level k=1:K_mean (the existing
production CM+meanzc behavior, `cm_meanzc_moments.jl`, reproduced bit-for-bit
when this layout is used). `n_eta = K_mean`.
"""
struct SharedByPowerLayout <: MeanZCTargetLayout
    K_mean::Int
    K_pair::Int
    function SharedByPowerLayout(K_mean::Int, K_pair::Int)
        K_mean >= 1 || error("SharedByPowerLayout: K_mean must be >= 1, got $K_mean")
        0 <= K_pair <= K_mean || error("SharedByPowerLayout: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
        return new(K_mean, K_pair)
    end
end

"""
    OriginByPowerLayout(D, K_mean, K_pair)

One nu_{o,k} per origin o=1:D, per power level k=1:K_mean -- the new
no-common-marginals restriction (docs/ORIGIN_SPECIFIC_ZC_MATH_NOTE_2026-07-23.md).
`n_eta = K_mean*D`. Coordinate ordering is level-major, origin-minor:
`eta_{1,1},...,eta_{D,1}, eta_{1,2},...,eta_{D,K_mean}` (matches the task
brief's own listing and `target_index` below).
"""
struct OriginByPowerLayout <: MeanZCTargetLayout
    D::Int
    K_mean::Int
    K_pair::Int
    function OriginByPowerLayout(D::Int, K_mean::Int, K_pair::Int)
        D >= 2 || error("OriginByPowerLayout: D must be >= 2, got $D")
        K_mean >= 1 || error("OriginByPowerLayout: K_mean must be >= 1, got $K_mean")
        0 <= K_pair <= K_mean || error("OriginByPowerLayout: K_pair must satisfy 0 <= K_pair <= K_mean, got K_pair=$K_pair, K_mean=$K_mean")
        return new(D, K_mean, K_pair)
    end
end

"""
    n_eta(layout) -> Int

Number and (implicit, via `target_index`) ordering of outer eta coordinates.
"""
n_eta(layout::SharedByPowerLayout) = layout.K_mean
n_eta(layout::OriginByPowerLayout) = layout.K_mean * layout.D

"""
    target_index(layout, o, k) -> Int

Maps (origin `o`, power `k`) to the coordinate index into the length-`n_eta`
eta/nu vector. `SharedByPowerLayout` ignores `o` (every origin shares index
`k`); `OriginByPowerLayout` returns the level-major, origin-minor index
`(k-1)*D + o`.
"""
target_index(layout::SharedByPowerLayout, o::Int, k::Int) = k
target_index(layout::OriginByPowerLayout, o::Int, k::Int) = (k - 1) * layout.D + o

"""
    nu_at_level(layout, νfull, k) -> AbstractVector{Float64}  (length D)

Per-origin target vector at level `k`, sliced from the full length-`n_eta`
nu vector (already exponentiated from eta -- callers pass `νfull = exp.(η)`).
`SharedByPowerLayout` broadcasts the single shared scalar to a length-D
vector; `OriginByPowerLayout` returns a `@view` into the packed layout (no
copy).
"""
nu_at_level(layout::SharedByPowerLayout, νfull::AbstractVector{Float64}, k::Int, D::Int) =
    fill(νfull[k], D)
nu_at_level(layout::OriginByPowerLayout, νfull::AbstractVector{Float64}, k::Int, D::Int) =
    @view νfull[(k-1)*layout.D+1 : k*layout.D]

"""
    layout_name(layout) -> Symbol

`:shared_by_power` | `:origin_by_power` -- the `power_target_layout` config
value and checkpoint-serialization tag (task brief Section 5/12).
"""
layout_name(::SharedByPowerLayout) = :shared_by_power
layout_name(::OriginByPowerLayout) = :origin_by_power

"""
    layout_D(layout) -> Int

Number of origins the layout is defined over (`0` for `SharedByPowerLayout`,
which is D-agnostic by construction -- one target regardless of D; the
CALLER's D still governs column counts elsewhere). Used only for checkpoint/
context-fingerprint metadata, never for indexing math (see `target_index`).
"""
layout_D(::SharedByPowerLayout) = 0
layout_D(layout::OriginByPowerLayout) = layout.D

"""
    make_target_layout(layout_name::Symbol, D::Int, K_mean::Int, K_pair::Int) -> MeanZCTargetLayout

Config-facing constructor: `:shared_by_power` -> `SharedByPowerLayout(K_mean,K_pair)`
(the `D` argument is accepted but ignored, matching `layout_D`'s D-agnostic
convention -- kept in the signature so callers do not need to branch before
calling); `:origin_by_power` -> `OriginByPowerLayout(D,K_mean,K_pair)`.
"""
function make_target_layout(layout_name::Symbol, D::Int, K_mean::Int, K_pair::Int)
    layout_name === :shared_by_power && return SharedByPowerLayout(K_mean, K_pair)
    layout_name === :origin_by_power && return OriginByPowerLayout(D, K_mean, K_pair)
    error("make_target_layout: layout_name must be :shared_by_power or :origin_by_power, got $layout_name")
end

"""
    layout_checkpoint_meta(layout) -> NamedTuple

Serialization metadata for the checkpoint schema (task brief Section 12):
`(target_layout, D, K_mean, K_pair)`. `D=0` for `SharedByPowerLayout` (see
`layout_D`). Round-tripping through `make_target_layout(meta.target_layout,
meta.D, meta.K_mean, meta.K_pair)` reconstructs an equal layout.
"""
layout_checkpoint_meta(layout::MeanZCTargetLayout) =
    (target_layout = layout_name(layout), D = layout_D(layout), K_mean = layout.K_mean, K_pair = layout.K_pair)

"""
    layout_fingerprint(layout) -> String

Context-fingerprint metadata: a short deterministic string distinguishing
layouts for cache-key / context-mismatch checks (task brief Section 4/12),
e.g. `"origin_by_power:D=20:K_mean=2:K_pair=2"`.
"""
function layout_fingerprint(layout::MeanZCTargetLayout)
    meta = layout_checkpoint_meta(layout)
    return "$(meta.target_layout):D=$(meta.D):K_mean=$(meta.K_mean):K_pair=$(meta.K_pair)"
end
