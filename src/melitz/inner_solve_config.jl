using KNITRO

# 2026-07-25 local-geometry/continuation session (governing prompt Phase 1): "make the
# evaluation cap impossible to omit."
#
# Root problem, confirmed live twice in this repo's own history before this file existed:
#
#   1. `build_melitz_implicit_bundle` (finite_delta_outer.jl) has always accepted
#      `lower_limit_guard::Union{Nothing,Real}=nothing` -- a SILENT default that leaves
#      `PsiObjectiveBundleImplicit.lower_limit == -KNITRO.KN_INFINITY` (permanently
#      disabled) unless a caller remembers to pass a real guard. This is exactly what
#      happened in the 2026-07-24 outer-benchmark session (Section 7.0): the first full
#      campaign attempt hung 55+ minutes because the guard was never wired up.
#   2. `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration` (delta_star.jl/
#      pareto_calibration.jl) construct a `PsiObjectiveBundleDelta` with NO cap mechanism
#      AT ALL -- there has never been a `lower_limit`-related keyword on either function.
#      Every caller that reuses the resulting `obj_inner` for REPEATED nested inner solves
#      (`nuisance_profile.jl`'s own `solve_melitz_nuisance_min_delta`, or any future
#      continuation/predictor-corrector driver built on the same object) inherits an
#      unguarded object by construction. Confirmed live, same session
#      (`docs/melitz_real_d20_outer_correction_2026-07-24.md` Section 10.4):
#      `obj_inner.lower_limit` printed as `-1.797693e+308` (i.e. `-KNITRO.KN_INFINITY`)
#      before a one-off diagnostic script patched it in by hand -- the fix was applied to
#      ONE script, not to the construction/use path itself, so the same omission remains
#      possible for the next caller.
#
# This file is the SINGLE place that computes a Melitz inner-solve `lower_limit` from an
# explicit, named mode. It does not replace `build_melitz_implicit_bundle`'s existing
# `lower_limit_guard`/`delta_evaluation_cap` kwargs (changing that signature would force
# every one of the ~90 existing D=4 unit-test call sites in `test/melitz/runtests.jl` --
# most of which test gradient backends/cache mechanics at a small, bounded, single-shot
# solve where an unguarded inner solve was never the failure mode -- to specify a mode they
# do not need, pure churn with no safety benefit). Instead:
#
#   - `build_melitz_implicit_bundle`'s internal guard/cap arithmetic now ROUTES THROUGH
#     `melitz_configure_lower_limit` (one implementation, not two) and additionally accepts
#     an `inner_solve_config::MelitzInnerSolveConfig` kwarg as the PREFERRED way to specify
#     the cap going forward (see finite_delta_outer.jl).
#   - `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration` gain the SAME
#     optional `inner_solve_config` kwarg (default `nothing`, preserving every existing
#     call site's exact current behavior -- an explicitly uncapped `PsiObjectiveBundleDelta`,
#     appropriate for a single bounded D=4/D=20 one-shot solve).
#   - `solve_melitz_nuisance_min_delta` (nuisance_profile.jl) -- the ACTUAL site of the
#     confirmed-live incident, and the one function in this codebase whose entire purpose is
#     REPEATED nested inner solves inside an outer KNITRO trajectory over a caller-supplied
#     `obj_inner` -- now REQUIRES `inner_solve_config::MelitzInnerSolveConfig` with NO
#     default at all, and applies it to `obj_inner.lower_limit` unconditionally at entry
#     (save/restore around the call, `melitz_without_lower_limit_bailout`'s own convention),
#     independent of how `obj_inner` happened to be constructed. This has zero existing
#     committed call sites (`solve_melitz_nuisance_min_delta` appears in zero place in
#     `test/melitz/runtests.jl` as of this session's start), so tightening its signature
#     breaks nothing already committed -- and it is exactly the function this session's own
#     Phases 7/8 (predictor-corrector, nuisance-profile experiments) build directly on top
#     of, so guarding it now protects every subsequent use in this session too.

"""
    MELITZ_INNER_SOLVE_MODES

The three modes `melitz_configure_lower_limit`/`MelitzInnerSolveConfig` accept, per the
governing prompt's Phase 1.1:

  - `:full_value`     -- deliberately uncapped (`lower_limit = -KNITRO.KN_INFINITY`). Valid
                         only when the caller explicitly wants the TRUE, fully-optimized
                         value regardless of magnitude (e.g. a cold end-of-run
                         reverification, or a single bounded D=4 unit-test fixture where
                         runaway divergence has never been observed) -- never the default
                         you fall into by omission; constructing this mode is itself an
                         explicit, named choice, not silence.
  - `:evaluation_cap` -- the production routine-inner-solve cap (governing prompt Section
                         1/2): requires a finite `delta_evaluation_cap`, sets
                         `lower_limit = -(delta_evaluation_cap + guard)`.
  - `:diagnostic`     -- mathematically identical to `:evaluation_cap` (same formula, same
                         finite-cap requirement) but tagged separately so a one-off
                         diagnostic/canary script's intent ("I am exploring, not running the
                         production trajectory") is visible in the config object itself
                         (`MelitzInnerSolveConfig.mode`), without weakening any invariant --
                         the assertions in `melitz_assert_evaluation_cap_active` (Phase 1.3)
                         apply identically to both `:evaluation_cap` and `:diagnostic`.
"""
const MELITZ_INNER_SOLVE_MODES = (:full_value, :evaluation_cap, :diagnostic)

"""
    MelitzInnerSolveConfig

Immutable record of a fully-resolved inner-solve cap decision -- `mode` names WHY
`lower_limit` has the value it does, so a downstream consumer (a report, a test, a runtime
assertion) never has to re-derive intent from a bare `Float64`. Construct via
`MelitzInnerSolveConfig(mode; delta_evaluation_cap=..., guard=..., outer_delta=...)`, never
via the raw positional constructor (which performs no validation) -- `Base.show` prints a
one-line summary for logging.
"""
struct MelitzInnerSolveConfig
    mode::Symbol
    lower_limit::Float64
    delta_evaluation_cap::Union{Nothing,Float64}
    guard::Float64
end

function Base.show(io::IO, cfg::MelitzInnerSolveConfig)
    if cfg.mode == :full_value
        print(io, "MelitzInnerSolveConfig(:full_value, lower_limit=-Inf [uncapped])")
    else
        print(io, "MelitzInnerSolveConfig($(cfg.mode), delta_evaluation_cap=$(cfg.delta_evaluation_cap), ",
                  "guard=$(cfg.guard), lower_limit=$(cfg.lower_limit))")
    end
end

"""
    melitz_configure_lower_limit(mode; delta_evaluation_cap=nothing, guard=1e-6,
        outer_delta=nothing) -> Float64

The one authoritative computation of a Melitz `PsiObjectiveBundleDelta`/
`PsiObjectiveBundleImplicit`'s KNITRO-native `lower_limit` early-bailout threshold (the
SAME shared `cc_algo/PsiObjectiveBundle.jl` mechanism the Ricardian model's own
`lower_limit=-50` convention uses -- this function is Melitz's own analogue, never a
reimplementation of `cc_algo` itself).

Fails fast (an `ArgumentError`, before ever touching KNITRO) rather than silently returning
an uncapped/disabled threshold for any input the governing prompt's Phase 1.1 flags as
invalid: an unrecognized `mode`, a missing `delta_evaluation_cap` under `:evaluation_cap`/
`:diagnostic`, a non-finite or non-positive `delta_evaluation_cap`, a negative `guard`, or a
stray `delta_evaluation_cap` supplied under `:full_value` (a caller passing both is almost
certainly confused about which mode they want).

`outer_delta`, if given, is compared against `delta_evaluation_cap` (Phase 1.3's "the cap
differs from the outer budget unless explicitly configured otherwise") -- logs a warning,
not an error, since `delta_evaluation_cap == outer_delta` is a legitimate (if unusual)
explicit choice (e.g. the 2026-07-24 evaluation-cap-correction session's own live
reproduction of the OLD pre-correction bug, Section 3 of that report, deliberately sets
`delta_evaluation_cap = delta` to demonstrate the original coupling for comparison).
"""
function melitz_configure_lower_limit(mode::Symbol; delta_evaluation_cap::Union{Nothing,Real}=nothing,
                                       guard::Real=1e-6, outer_delta::Union{Nothing,Real}=nothing)
    mode in MELITZ_INNER_SOLVE_MODES || throw(ArgumentError(
        "melitz_configure_lower_limit: mode must be one of $(MELITZ_INNER_SOLVE_MODES), got $mode"))
    guard >= 0 || throw(ArgumentError("melitz_configure_lower_limit: guard must be >= 0, got $guard"))

    if mode == :full_value
        delta_evaluation_cap === nothing || throw(ArgumentError(
            "melitz_configure_lower_limit: mode=:full_value must not be given a " *
            "delta_evaluation_cap ($delta_evaluation_cap supplied) -- pass mode=:evaluation_cap " *
            "or mode=:diagnostic if a cap is actually wanted"))
        return -KNITRO.KN_INFINITY
    end

    # :evaluation_cap or :diagnostic -- identical arithmetic, see this file's header.
    delta_evaluation_cap === nothing && throw(ArgumentError(
        "melitz_configure_lower_limit: mode=$mode requires an explicit delta_evaluation_cap::Real " *
        "-- there is no silent default that disables the cap under this mode. Pass " *
        "mode=:full_value if an uncapped solve is genuinely intended."))
    dec = Float64(delta_evaluation_cap)
    isfinite(dec) || throw(ArgumentError(
        "melitz_configure_lower_limit: delta_evaluation_cap must be finite, got $dec"))
    dec > 0 || throw(ArgumentError(
        "melitz_configure_lower_limit: delta_evaluation_cap must be strictly positive " *
        "(it is a divergence-magnitude threshold), got $dec"))
    if outer_delta !== nothing && Float64(outer_delta) == dec
        @warn "melitz_configure_lower_limit: delta_evaluation_cap == outer_delta ($dec) -- " *
              "this reproduces the pre-2026-07-24 conflation of the evaluation cap with the " *
              "outer budget (docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md). " *
              "Only intentional if you are deliberately reproducing that OLD behavior for comparison."
    end
    return -(dec + Float64(guard))
end

"""
    MelitzInnerSolveConfig(mode; delta_evaluation_cap=nothing, guard=1e-6, outer_delta=nothing)

Validating outer constructor -- see `melitz_configure_lower_limit` for the full argument
semantics/error conditions. This is the ONLY supported way to build a
`MelitzInnerSolveConfig`.
"""
function MelitzInnerSolveConfig(mode::Symbol; delta_evaluation_cap::Union{Nothing,Real}=nothing,
                                 guard::Real=1e-6, outer_delta::Union{Nothing,Real}=nothing)
    ll = melitz_configure_lower_limit(mode; delta_evaluation_cap=delta_evaluation_cap, guard=guard,
                                       outer_delta=outer_delta)
    dec = delta_evaluation_cap === nothing ? nothing : Float64(delta_evaluation_cap)
    return MelitzInnerSolveConfig(mode, ll, dec, Float64(guard))
end

"""
    melitz_assert_evaluation_cap_active(cfg::MelitzInnerSolveConfig)

Phase 1.3's runtime assertions, callable on ANY resolved config (not just ones this session
constructs) -- a defensive check a production driver can call right after building/applying
a config, so a future refactor that reintroduces a silent uncapped path fails loudly at
runtime, not just in a test file that might not be re-run. Throws `AssertionError` (never
silently returns `false`) if `cfg.mode in (:evaluation_cap, :diagnostic)` but any of:

  - `cfg.delta_evaluation_cap` is not finite;
  - `cfg.lower_limit` is not finite (i.e. still `-KNITRO.KN_INFINITY`, the disabled value);
  - `cfg.lower_limit` is not strictly below `0` with the correct sign
    (`cfg.lower_limit == -(delta_evaluation_cap + guard)`, i.e. more negative than
    `-delta_evaluation_cap` alone -- the guard must be doing SOMETHING).

No-op (returns `true`) for `mode==:full_value` -- that mode's entire point is an
intentionally inactive cap, not a bug to flag.
"""
function melitz_assert_evaluation_cap_active(cfg::MelitzInnerSolveConfig)
    cfg.mode == :full_value && return true
    @assert cfg.delta_evaluation_cap !== nothing && isfinite(cfg.delta_evaluation_cap) (
        "melitz_assert_evaluation_cap_active: mode=$(cfg.mode) but delta_evaluation_cap is not finite")
    @assert isfinite(cfg.lower_limit) (
        "melitz_assert_evaluation_cap_active: mode=$(cfg.mode) but lower_limit=$(cfg.lower_limit) " *
        "is not finite -- the evaluation cap is NOT active (this is exactly the silent-omission " *
        "bug this file exists to prevent)")
    expected = -(cfg.delta_evaluation_cap + cfg.guard)
    @assert cfg.lower_limit == expected (
        "melitz_assert_evaluation_cap_active: lower_limit=$(cfg.lower_limit) does not match " *
        "-(delta_evaluation_cap+guard)=$expected -- wrong sign or stale config")
    return true
end
