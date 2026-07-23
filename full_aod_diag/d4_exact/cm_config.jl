# ============================================================================
# Production-integration continuation: explicit common-marginals configuration
# surface (brief Section 2/3), sitting on top of the already-validated
# Continuation 12/13 machinery (cm_production_bundle.jl, common_marginals_moments.jl,
# common_marginals_interval.jl, cm_hessian_architectures.jl,
# cm_hessian_architecture_interval.jl, nested_quantile_grids.jl). Additive only
# -- no existing function's default behavior changes.
#
# `common_marginals = false` is the production default and means: don't touch
# this file's machinery at all. The unrestricted driver (c10_d20_production_driver.jl)
# never includes this file's dispatch and is byte-for-byte unmodified -- CM is
# wired in as a SEPARATE opt-in outer-driver entry point (cm_outer_driver.jl /
# the D20 CM continuation driver), not a runtime branch inside the single
# unrestricted KNITRO callback path. This is what makes "CM off exactly
# reproduces unrestricted production results" true by construction rather than
# by testing alone.
# ============================================================================

"""
    CMConfig(; common_marginals=false, cm_grid_size=50, cm_grid_rule=:equal,
               cm_grid_sizes=[10,20,50], cm_basis=:cumulative,
               cm_hessian_backend=:structured, contrasts=:anchored)

Production configuration for the common-marginals restriction.

- `common_marginals::Bool`: opt-in switch. `false` (default) means run the
  plain unrestricted problem; nothing below is consulted.
- `cm_grid_size::Int`: number of cutpoints `L` for a STANDALONE run
  (`cm_grid_rule = :equal`).
- `cm_grid_rule::Symbol`: `:equal` (single deterministic equal-probability
  grid at `cm_grid_size`) or `:nested_family` (genuinely nested grids at every
  size in `cm_grid_sizes`, `G_10 ⊂ G_20 ⊂ G_50 ⊂ ...` by construction).
- `cm_grid_sizes::Vector{Int}`: only consulted when `cm_grid_rule = :nested_family`.
- `cm_basis::Symbol`: `:cumulative` (production default) or `:interval`.
- `cm_hessian_backend::Symbol`: `:structured` (production default -- bin-index
  lookup construction, no persistent dense `W x ncm` matrix) or
  `:dense_reference` (Architecture A's generic dense Hessian, used as the
  trusted reference / equivalence-testing baseline, not for production speed).
"""
Base.@kwdef struct CMConfig
    common_marginals::Bool = false
    cm_grid_size::Int = 50
    cm_grid_rule::Symbol = :equal
    cm_grid_sizes::Vector{Int} = [10, 20, 50]
    cm_basis::Symbol = :cumulative
    cm_hessian_backend::Symbol = :structured
    contrasts::Symbol = :anchored
end

function _cm_validate(cfg::CMConfig)
    cfg.cm_grid_rule in (:equal, :nested_family) ||
        error("CMConfig: cm_grid_rule must be :equal or :nested_family, got $(cfg.cm_grid_rule)")
    cfg.cm_basis in (:cumulative, :interval) ||
        error("CMConfig: cm_basis must be :cumulative or :interval, got $(cfg.cm_basis)")
    cfg.cm_hessian_backend in (:structured, :dense_reference) ||
        error("CMConfig: cm_hessian_backend must be :structured or :dense_reference, got $(cfg.cm_hessian_backend)")
    if cfg.cm_grid_rule === :equal
        cfg.cm_grid_size >= 1 || error("CMConfig: cm_grid_size must be >= 1")
    else
        !isempty(cfg.cm_grid_sizes) || error("CMConfig: cm_grid_sizes must be non-empty for :nested_family")
        issorted(cfg.cm_grid_sizes) || error("CMConfig: cm_grid_sizes must be ascending, got $(cfg.cm_grid_sizes)")
    end
    return nothing
end

"""
    cm_equal_grid_probs(L) -> Vector{Float64}

THE documented `:equal` convention (brief Section 3): `L` deterministic,
approximately-equal-probability cutpoints, `range(1/L, (L-1)/L, length=L)`.
Identical to `precalc_common_marginals_cdf`'s own pre-existing `probs===nothing`
default (verified: this function is only ever used to make that convention an
explicit, callable, documented artifact -- not a new convention).
"""
cm_equal_grid_probs(L::Int) = collect(range(1 / L, (L - 1) / L, length = L))

"""
    cm_nested_family_probs(sizes) -> Dict{Int,Vector{Float64}}

THE documented `:nested_family` convention (brief Section 3): one
largest-gap-bisection sequence out to `maximum(sizes)`, snapshotted at every
requested size, so `G_{sizes[1]} ⊂ G_{sizes[2]} ⊂ ...` holds by construction
(`nested_quantile_grids.jl::nested_grid_sequence`, Continuation 13 Section 6 --
reused verbatim, not reimplemented).
"""
cm_nested_family_probs(sizes::Vector{Int}) = nested_grid_sequence(sizes)

"""
    cm_resolve_probs(cfg::CMConfig) -> Vector{Float64} | Dict{Int,Vector{Float64}}

Resolves `cfg`'s grid rule to actual probability cutpoints. For `:equal`,
returns a single `Vector{Float64}` of length `cfg.cm_grid_size`. For
`:nested_family`, returns the full `Dict{Int,Vector{Float64}}` (one entry per
`cfg.cm_grid_sizes` element) -- a continuation driver iterates this dict's
keys in ascending order.
"""
function cm_resolve_probs(cfg::CMConfig)
    _cm_validate(cfg)
    return cfg.cm_grid_rule === :equal ? cm_equal_grid_probs(cfg.cm_grid_size) :
                                          cm_nested_family_probs(cfg.cm_grid_sizes)
end

"""
    cm_resolve_probs_for_L(cfg::CMConfig, L::Int) -> Vector{Float64}

Convenience accessor for a single grid size `L`, valid for either rule
(`:equal` ignores `L` beyond an equality assertion against `cfg.cm_grid_size`;
`:nested_family` looks `L` up in the resolved family).
"""
function cm_resolve_probs_for_L(cfg::CMConfig, L::Int)
    probs = cm_resolve_probs(cfg)
    if probs isa Dict
        haskey(probs, L) || error("cm_resolve_probs_for_L: L=$L not in cm_grid_sizes=$(cfg.cm_grid_sizes)")
        return probs[L]
    end
    L == cfg.cm_grid_size || error("cm_resolve_probs_for_L: L=$L != cfg.cm_grid_size=$(cfg.cm_grid_size) under :equal")
    return probs
end

# ============================================================================
# Unified production context: dispatches cm_basis x cm_hessian_backend onto
# the already-validated Continuation 12/13 builders. Generalizes
# `cm_production_bundle.jl::build_cm_production_context`/`archC_base_state`
# (which hardcode cumulative+structured) to all 4 (basis, backend) combinations
# WITHOUT modifying those functions -- they remain the fast-path default
# call sites everywhere they're already used.
# ============================================================================

"CMConfig-aware production context: (ctx_cm, aug, hess_cb_builder, cfg). `hess_cb_builder` is a unary function of `obj` (KNITRO's own calling convention, see inner_loop_internal_archgeneric), ready to pass straight through."
function build_cm_production_context_v2(ctx, CS, cfg::CMConfig; L::Int = cfg.cm_grid_size)
    _cm_validate(cfg)
    probs = cm_resolve_probs_for_L(cfg, L)

    if cfg.cm_basis === :cumulative
        pcx = build_cm_production_context(ctx, CS; L = L, contrasts = cfg.contrasts, probs = probs,
                                           use_archB_moments = cfg.cm_hessian_backend === :structured)
        hess_cb_builder = cfg.cm_hessian_backend === :structured ?
            (_obj -> archC_hess_cb_builder(pcx.cctx)) : archA_hess_cb_builder
        return (ctx_cm = pcx.ctx_cm, aug = pcx.aug, hess_cb_builder = hess_cb_builder, cfg = cfg, L = L)
    else # :interval
        # No Architecture-B-accelerated moment path exists for the interval basis
        # (fill_cm_columns_from_bins! is cumulative-specific, "<=l"; an interval
        # analogue would need a new "==l" fill kernel -- not built, since interval
        # is a documented non-default follow-up candidate, not production-critical
        # per Section 8's D20 conditioning-reversal finding). Moments always go
        # through the dense build_cm_augmented_obj_interval path regardless of
        # cm_hessian_backend; only the HESSIAN callback (the expensive per-call
        # piece) actually varies with cm_hessian_backend for this basis.
        aug = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = cfg.contrasts, probs = probs)
        ctx_cm = merge(ctx, (obj = aug.obj_cm,))
        hess_cb_builder = archA_hess_cb_builder
        if cfg.cm_hessian_backend === :structured
            cctx_i = build_cm_bin_ctx_interval(ctx, aug)
            hess_cb_builder = _obj -> archC_interval_hess_cb_builder(cctx_i)
        end
        return (ctx_cm = ctx_cm, aug = aug, hess_cb_builder = hess_cb_builder, cfg = cfg, L = L)
    end
end

"CMConfig-generalized `archC_base_state`: same contract, any (basis, backend) combination via `pcx.hess_cb_builder`."
function cm_base_state_v2(x_free0::AbstractVector, pcx)
    obj = pcx.ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, pcx.ctx_cm.m)
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder = pcx.hess_cb_builder)
    nStatus in (0, -100, -101, -103) || error("cm_base_state_v2: inner solve failed, nStatus=$nStatus (x_free0=$x_free0, basis=$(pcx.cfg.cm_basis), backend=$(pcx.cfg.cm_hessian_backend))")
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    CMEvalKey

Exact-point cache key for the CM production path -- `FullAEvalKey` (oracle.jl) plus the
`cm_cache_key(cfg, L, draw_checksum)` fragment, per brief Section 4/11: never let a CM-on point
collide with an unrestricted (or differently-configured CM) point in the same cache. Uses
`SafeExactCache{CMEvalKey}` (oracle.jl's genericized cache), NOT `SafeExactCache{FullAEvalKey}` --
the two must never share a cache instance, since nothing here type-checks that a caller passed
the wrong dict; keep unrestricted and CM caches as physically separate `SafeExactCache` objects.

AUD-08 fix: carries ctx_fingerprint (context_fingerprint(ctx), oracle.jl) for the same reason
FullAEvalKey does -- see that struct's docstring.
"""
struct CMEvalKey
    x_free::Vector{Float64}
    δ::Float64
    find_smallest::Bool
    inner_loop_opt::String
    cm::NamedTuple
    ctx_fingerprint::String
end
Base.:(==)(a::CMEvalKey, b::CMEvalKey) = a.x_free == b.x_free && a.δ == b.δ &&
    a.find_smallest == b.find_smallest && a.inner_loop_opt == b.inner_loop_opt && a.cm == b.cm &&
    a.ctx_fingerprint == b.ctx_fingerprint
Base.hash(k::CMEvalKey, h::UInt) = hash((k.x_free, k.δ, k.find_smallest, k.inner_loop_opt, k.cm, k.ctx_fingerprint), h)

"Fresh, empty CM exact-point cache -- mirrors oracle.jl's `oracle_cache_for`, scoped to CMEvalKey."
cm_oracle_cache_for(pcx) = SafeExactCache{CMEvalKey}()

"""
    cm_production_value_v2(x_free0, pcx; cache=nothing, use_cache=true, draw_checksum=nothing, tag="") -> (K, base)

CMConfig-generalized `cm_production_value`. `cache` is a `SafeExactCache{CMEvalKey}`
(`cm_oracle_cache_for`); when supplied and `use_cache=true`, an exact repeat of
`(x_free0, δ, find_smallest, inner_loop_opt, cfg, L, draw_checksum)` returns without invoking
`inner_loop_KNITRO_archgeneric` at all. `cm_base_state_v2` already errors on any non-feasible
`nStatus` before returning, so every stored entry is a genuine feasible solve (same
cacheability contract as `oracle.jl`'s `is_cacheable_result`, just enforced upstream by the
`error()` instead of a post-hoc status check).
"""
function cm_production_value_v2(x_free0::AbstractVector, pcx; cache = nothing, use_cache::Bool = true,
                                 draw_checksum = nothing, tag::String = "")
    obj = pcx.ctx_cm.obj
    key = nothing
    if cache !== nothing && use_cache
        key = CMEvalKey(collect(x_free0), obj.δ, obj.find_smallest, obj.inner_loop_opt,
                         cm_cache_key(pcx.cfg, pcx.L, draw_checksum), context_fingerprint(pcx.ctx_cm))
        hit = _cache_lookup(cache, key)
        if hit !== nothing
            return hit.K, hit.base
        end
    end

    base = cm_base_state_v2(x_free0, pcx)
    K = pcx.ctx_cm.obj.H_save

    if key !== nothing
        _cache_store!(cache, key, (K = K, base = base))
    end
    return K, base
end

# ============================================================================
# Cache-keying helper (brief Section 4/11): "A cached result or dual solution
# must be keyed by common-marginal on/off status, exact grid cutpoints, basis,
# moment ordering, draw checksum. Never reuse unrestricted duals or cached
# inner results in a CM context."
#
# The exact-point cache / successful-dual cache themselves live on the
# consolidation branch (integration/fullA-d20-runtime-delta5, still WIP/dirty
# as of this port -- see docs/fullA_common_marginals_production_integration.md
# Section "Required rebase point"). This helper exists now so wiring CM into
# that cache, once it lands, only needs to thread `cm_cache_key(cfg,...)` into
# whatever hash/tuple key that cache already uses -- not invent the concept
# from scratch under time pressure during the final rebase.
# ============================================================================

"""
    cm_cache_key(cfg::CMConfig, L::Int, draw_checksum) -> NamedTuple

A hashable, comparable key fragment identifying a CM configuration for cache
partitioning. `draw_checksum` should be a cheap deterministic hash of `ctx.U`
(e.g. `hash(ctx.U)` or a stored `draw_seed`, whichever the consolidation
branch's exact-point cache already threads through context construction) --
NOT recomputed here, since this file has no opinion on which draw-identity
convention the cache adopts. `common_marginals=false` collapses to a single
canonical key (`cm_basis`/`cm_hessian_backend`/`L`/draw_checksum` all ignored)
so unrestricted results are never partitioned by CM-irrelevant fields.
"""
function cm_cache_key(cfg::CMConfig, L::Int, draw_checksum)
    cfg.common_marginals || return (common_marginals = false,)
    probs = cm_resolve_probs_for_L(cfg, L)
    return (common_marginals = true, cm_basis = cfg.cm_basis, cm_hessian_backend = cfg.cm_hessian_backend,
            L = L, cutpoints = Tuple(probs), contrasts = cfg.contrasts, draw_checksum = draw_checksum)
end
