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
#
# Include order: this file must be included AFTER cm_meanzc_moments.jl
# (`pair_targets` below calls that file's `packed_pair_index`).
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
    mean_targets(layout, νfull, k, D) -> Vector{Float64}   (length D)

Mean-column targets: `nu_{o,k}` for every origin `o=1:D` at level `k`,
dispatched entirely through `target_index` -- one implementation for BOTH
layouts (`SharedByPowerLayout` returns the same shared value `D` times since
`target_index` ignores `o`; `OriginByPowerLayout` returns each origin's own
value).
"""
mean_targets(layout::MeanZCTargetLayout, νfull::AbstractVector{Float64}, k::Int, D::Int) =
    [νfull[target_index(layout, o, k)] for o in 1:D]

"""
    pair_targets(layout, νfull, k, D) -> Vector{Float64}   (length D*(D-1)/2)

Pair-column targets: `nu_{o,k} * nu_{p,k}` for every unordered pair `o<p`
(`packed_pair_index` ordering), dispatched through `target_index` -- again
one implementation for both layouts (reduces to `nu_k^2` under
`SharedByPowerLayout`). Requires `packed_pair_index` (`cm_meanzc_moments.jl`,
included before this file).
"""
function pair_targets(layout::MeanZCTargetLayout, νfull::AbstractVector{Float64}, k::Int, D::Int)
    pairs = packed_pair_index(D)
    return [νfull[target_index(layout, o, k)] * νfull[target_index(layout, p, k)] for (o, p) in pairs]
end

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

# ============================================================================
# fix/zc-profile-focal-sigmaminus1-mean-2026-08-07: focal k=(sigma-1) mean-row
# omission ("Variant D" -- derive nu_{focal,k*}=b(psi)/c(psi) exactly, per
# autarky_cf.jl, and OMIT the now-exactly-redundant focal mean row/eta coordinate
# entirely, rather than the 2026-08-05 merge's Variant C, which derives the same
# value but leaves the row/eta coordinate in place -- see
# docs/audits/zc-profile-focal-sigmaminus1-mean-2026-08-07/MASTER.md for the
# exact-redundancy proof (autarky moment and focal k* mean moment are affinely
# related in the draw omega, for EVERY nu value, so once nu is pinned at its
# consistent value the two rows become exactly proportional -- a genuine KKT
# rank deficiency, not merely an off-manifold-search hazard).
#
# `ActiveMeanLayout` is a THIN WRAPPER around an existing `layout` -- it does
# NOT replace `target_index`/`mean_targets`/`pair_targets` above (those keep
# operating on a "dense" nu vector of length `n_eta(layout)`, UNCHANGED). The
# active layout instead describes (a) which of the `n_eta(layout)` dense
# coordinates has no outer eta anymore, (b) how to reconstruct a full dense nu
# vector (`nu_eff`) from the shorter active outer eta vector plus the one
# derived scalar, for consumption by the UNCHANGED dense `target_index`-based
# functions, and (c) which origin-column subset of each mean level is ACTIVE
# (present in `ZCRestrictionOperator.Zraw_all[k]`, `zc_restriction_operator.jl`).
# ============================================================================

"""
    ActiveMeanLayout(base, focal_origin, kstar)

`base` is the EXISTING, unmodified `SharedByPowerLayout`/`OriginByPowerLayout`.
`active = (1 <= kstar <= base.K_mean)` (task Section 4: profiling only occurs
if the requested power set actually contains k*=sigma-1 -- for any other
sigma, `active=false` and this struct is a pure passthrough, zero behavior
change). `dense_omit_idx = target_index(base, focal_origin, kstar)` (meaningless
if `!active`). `mean_active_origins[k]` lists, in column order, which of `1:D`
origins occupy `ZCRestrictionOperator.Zraw_all[k]`'s columns after compaction
-- `1:D` for every level except `kstar` (if active), which is `1:D` with
`focal_origin` removed. `n_eta_active = n_eta(base) - (active ? 1 : 0)`.
"""
struct ActiveMeanLayout{L<:MeanZCTargetLayout}
    base::L
    focal_origin::Int
    kstar::Int
    active::Bool
    dense_omit_idx::Int
    n_eta_active::Int
    mean_active_origins::Vector{Vector{Int}}
end

function ActiveMeanLayout(base::MeanZCTargetLayout, focal_origin::Int, kstar::Int, D::Int)
    1 <= focal_origin <= D || error("ActiveMeanLayout: focal_origin=$focal_origin out of range 1:$D")
    active = 1 <= kstar <= base.K_mean
    dense_omit_idx = active ? target_index(base, focal_origin, kstar) : 0
    n_eta_active = n_eta(base) - (active ? 1 : 0)
    mean_active_origins = Vector{Vector{Int}}(undef, base.K_mean)
    for k in 1:base.K_mean
        mean_active_origins[k] = (active && k == kstar) ? setdiff(1:D, focal_origin) : collect(1:D)
    end
    return ActiveMeanLayout(base, focal_origin, kstar, active, dense_omit_idx, n_eta_active, mean_active_origins)
end

"""
    scatter_nu_eff(aml, νfull_active, nu_star) -> nu_eff   (dense, length n_eta(aml.base))

Reconstructs the FULL dense nu vector consumed UNCHANGED by `target_index`/
`mean_targets`/`pair_targets` above, by inserting the derived `nu_star` at the
one omitted dense coordinate (if `aml.active`) and every other outer eta value
at its (shifted) dense slot, in order. Identity map if `!aml.active`
(`νfull_active` is already dense length `n_eta(aml.base)`; `nu_star` unused).
"""
function scatter_nu_eff(aml::ActiveMeanLayout, νfull_active::AbstractVector{Float64}, nu_star::Float64)
    aml.active || return νfull_active
    n_dense = n_eta(aml.base)
    length(νfull_active) == n_dense - 1 ||
        error("scatter_nu_eff: length(νfull_active)=$(length(νfull_active)) != n_eta(base)-1=$(n_dense-1)")
    nu_eff = Vector{Float64}(undef, n_dense)
    j = 1
    @inbounds for d in 1:n_dense
        if d == aml.dense_omit_idx
            nu_eff[d] = nu_star
        else
            nu_eff[d] = νfull_active[j]
            j += 1
        end
    end
    return nu_eff
end

"""
    gather_active_grad(aml, g_dense) -> (g_active, g_omit)

Inverse-shaped helper for the outer-gradient side: splits a length-`n_eta(base)`
dense gradient vector (e.g. from `d_delta_dual_d_eta_origin_vec`-style envelope
math evaluated as if every dense coordinate still had an eta) into the active
`n_eta_active`-length gradient (in the SAME order the active outer eta vector
uses) and the single scalar gradient component at the omitted coordinate
(needed by the `dnu_star/dx` chain-rule term, Section 11/12). Identity
(`g_dense`, `0.0`) if `!aml.active`.
"""
function gather_active_grad(aml::ActiveMeanLayout, g_dense::AbstractVector{Float64})
    aml.active || return g_dense, 0.0
    n_dense = length(g_dense)
    g_active = Vector{Float64}(undef, n_dense - 1)
    g_omit = 0.0
    j = 1
    @inbounds for d in 1:n_dense
        if d == aml.dense_omit_idx
            g_omit = g_dense[d]
        else
            g_active[j] = g_dense[d]
            j += 1
        end
    end
    return g_active, g_omit
end

"""
    active_mean_layout_fingerprint(aml) -> String

Checkpoint/manifest metadata (task Section 18): distinguishes an
active/row-omitted layout from the base dense layout it wraps, for hard
resume-refusal against pre-2026-08-07 checkpoints that have no such field.
"""
function active_mean_layout_fingerprint(aml::ActiveMeanLayout)
    return "$(layout_fingerprint(aml.base)):focal=$(aml.focal_origin):kstar=$(aml.kstar):active=$(aml.active)"
end
