# 2026-07-27 addendum (governing prompt Phase 6): bundles the two orthogonal outer-coordinate
# axes -- PARTICIPATION (`log_cutoff_param.jl`, `ctx.outer_parameterization`, `:logf`/
# `:logcutoff`) and TECHNOLOGY (`technology_coordinate.jl`, `ctx.technology_coordinate`,
# `:logA`/`:theta_logA`/`:sigma_minus_one_logA`) -- into ONE immutable, Melitz-owned
# configuration object, giving the full 3x2=6-combination factorial a single named identity
# for production run summaries, cache fingerprints (already wired,
# `bounded_cache.jl`'s `melitz_context_fingerprint` v2), warm-start compatibility checks, and
# checkpoint metadata (this file's own `melitz_parameterization_compatible` below).

"""
    MelitzOuterParameterizationConfig(technology_coordinate::Symbol, participation_coordinate::Symbol)

Immutable pairing of the two orthogonal outer-coordinate axes. `participation_coordinate`
maps directly onto `ctx.outer_parameterization` (pre-existing name, kept as-is at the `ctx`
field level for backward compatibility -- `log_cutoff_param.jl`'s own dispatchers already key
off it); `technology_coordinate` maps onto `ctx.technology_coordinate`
(`technology_coordinate.jl`). Construct via `melitz_apply_parameterization(ctx, config)` to
get a new `ctx` with both fields set consistently, rather than setting the two `ctx` fields
separately by hand.
"""
struct MelitzOuterParameterizationConfig
    technology_coordinate::Symbol
    participation_coordinate::Symbol
    function MelitzOuterParameterizationConfig(technology_coordinate::Symbol, participation_coordinate::Symbol)
        technology_coordinate in MELITZ_TECHNOLOGY_COORDINATES || throw(ArgumentError(
            "MelitzOuterParameterizationConfig: technology_coordinate must be one of " *
            "$MELITZ_TECHNOLOGY_COORDINATES, got $technology_coordinate"))
        participation_coordinate in (:logf, :logcutoff) || throw(ArgumentError(
            "MelitzOuterParameterizationConfig: participation_coordinate must be :logf or " *
            ":logcutoff, got $participation_coordinate"))
        return new(technology_coordinate, participation_coordinate)
    end
end

"""
    MELITZ_DEFAULT_PARAMETERIZATION

This codebase's pre-existing (pre-addendum) behavior, expressed as a config: plain `log(A)`,
`:logf` participation. Every existing caller that never mentions either axis gets exactly
this.
"""
const MELITZ_DEFAULT_PARAMETERIZATION = MelitzOuterParameterizationConfig(:logA, :logf)

"""
    MELITZ_ALL_PARAMETERIZATIONS

The full 3x2=6-combination factorial (governing prompt Section D/6), in a fixed, reproducible
order (technology outer loop, participation inner loop) -- used by the staged D=4 comparison
and the roundtrip/chain-rule test sweeps so every "all six combinations" claim iterates the
SAME set in the SAME order.
"""
const MELITZ_ALL_PARAMETERIZATIONS = Tuple(
    MelitzOuterParameterizationConfig(tc, pc)
    for tc in MELITZ_TECHNOLOGY_COORDINATES for pc in (:logf, :logcutoff)
)

"""
    melitz_parameterization_label(config::MelitzOuterParameterizationConfig) -> String

Short, filesystem/log-safe label, e.g. `"theta_logA__logcutoff"` -- used in run summaries,
result file names, and printed at run start (Phase 13's own "print it at run start"
requirement).
"""
melitz_parameterization_label(config::MelitzOuterParameterizationConfig) =
    string(config.technology_coordinate, "__", config.participation_coordinate)

"""
    melitz_apply_parameterization(ctx, config::MelitzOuterParameterizationConfig) -> new_ctx

Returns a new `ctx` (NamedTuple `merge`, `ctx` itself never mutated -- NamedTuples are
immutable in Julia anyway, but this makes the "never mutates its argument" contract explicit)
with `outer_parameterization`/`technology_coordinate` set to `config`'s own two fields,
every other field UNCHANGED. This is the ONE place a caller should set both axes at once,
rather than two separate `merge(ctx, (outer_parameterization=..., technology_coordinate=...))`
calls that could drift out of sync.
"""
function melitz_apply_parameterization(ctx, config::MelitzOuterParameterizationConfig)
    return merge(ctx, (outer_parameterization=config.participation_coordinate,
                        technology_coordinate=config.technology_coordinate))
end

"""
    melitz_ctx_parameterization(ctx) -> MelitzOuterParameterizationConfig

Reads the current parameterization OFF a `ctx` (defaulting absent fields exactly as
`melitz_expand_theta`/`melitz_reduce_theta`/`melitz_context_fingerprint` already do:
`outer_parameterization` -> `:logf`, `technology_coordinate` -> `:logA`) -- the inverse of
`melitz_apply_parameterization`, for reporting/logging a `ctx`'s own current config without
requiring the caller to have kept the original `MelitzOuterParameterizationConfig` object
around.
"""
melitz_ctx_parameterization(ctx) = MelitzOuterParameterizationConfig(
    get(ctx, :technology_coordinate, :logA), get(ctx, :outer_parameterization, :logf))

"""
    melitz_parameterization_compatible(a, b) -> Bool

`true` iff two `MelitzOuterParameterizationConfig`s (or two `ctx`s, or one of each --
whichever combination a caller has on hand, resolved via `melitz_ctx_parameterization` first)
represent the SAME outer coordinate. Warm-start/checkpoint-metadata consumers should call
this BEFORE reusing a stored `theta_free`/dual-bank/exact-point-cache entry across a
parameterization boundary -- a `theta_free` vector's own NUMBERS are only meaningful under
the SAME parameterization that produced them (Phase 6's own "no old warm start or cache entry
crosses parameterizations" requirement); `melitz_context_fingerprint`'s v2 hash already
enforces this automatically for the exact-point cache specifically, this function is the
explicit, parameterization-level check for callers (warm-start banks, checkpoints) that key
on something coarser than a full content fingerprint.
"""
melitz_parameterization_compatible(a::MelitzOuterParameterizationConfig, b::MelitzOuterParameterizationConfig) =
    a.technology_coordinate == b.technology_coordinate && a.participation_coordinate == b.participation_coordinate
melitz_parameterization_compatible(a, b) = melitz_parameterization_compatible(
    a isa MelitzOuterParameterizationConfig ? a : melitz_ctx_parameterization(a),
    b isa MelitzOuterParameterizationConfig ? b : melitz_ctx_parameterization(b))
