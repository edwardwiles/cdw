# ============================================================================
# Continuation 8, workstream 4: convenience wiring for the low-risk moment-
# build specializations (`enable_pow_cache!`, `enable_autarky_cf_v2!`) at the
# SCRIPT level, not inside the shared context builder.
#
# `context.jl`/`d4_exact_setup()` are owned by workstream 2 and are NOT touched
# here or anywhere in this file's own history. This file is purely additive:
# it `include`s the (also-unmodified) files that define the opt-in helpers and
# adds one more convenience wrapper, `enable_live_defaults!`, that a script can
# call right after `ctx = d4_exact_setup(...)` to opt into either/both
# specializations with one call. Nothing here changes behaviour unless a
# caller explicitly invokes it -- `d4_exact_setup()` itself remains completely
# unaffected, exactly like `enable_pow_cache!`/`enable_autarky_cf!`/
# `enable_autarky_cf_v2!` themselves.
#
# Equivalence status (re-confirmed this session against HEAD, see
# docs/lowrisk_specialization_live_wiring.md):
#   - enable_pow_cache!      -- bit-identical through the full live oracle path
#                                (verify_pow_cache_wiring.jl, re-run clean).
#   - enable_autarky_cf_v2!  -- tight (<=1 ULP) on the raw moment column, but
#                                BIT-IDENTICAL Delta_dual through the full live
#                                inner solve (per docs/autarky_cf_v2_cached_base.md
#                                Target-C1/live-wiring benchmark; not re-derived
#                                here, just relied upon).
#
# Usage (typical, from a script in this directory):
#
#   include(joinpath(@__DIR__, "context.jl"))
#   include(joinpath(@__DIR__, "live_defaults.jl"))
#   ctx = d4_exact_setup(find_smallest = true)
#   enable_live_defaults!(ctx; pow_cache = true, autarky_cf_v2 = false)
#
# `autarky_cf_v2` defaults to `false` here deliberately -- see
# docs/lowrisk_specialization_live_wiring.md for the measured verdict on
# whether it is worth enabling in `gamma_profile.jl`'s specific access
# pattern (full per-eval moment builds via evaluate_fullA, NOT a CF-only
# sweep from raw Uσ -- the scenario where v2's 35.7x isolated gain was
# measured). `pow_cache` defaults to `true` because it is unconditionally
# bit-identical and a modest but real allocation/GC win with no scenario
# where it can be a net loss.
# ============================================================================
include(joinpath(@__DIR__, "moments_fast.jl"))     # enable_pow_cache!
include(joinpath(@__DIR__, "autarky_cf.jl"))        # enable_autarky_cf! (+ autarky_cf_scalars, needed by v2)
include(joinpath(@__DIR__, "autarky_cf_v2.jl"))     # enable_autarky_cf_v2!

"""
    enable_live_defaults!(ctx; pow_cache=true, autarky_cf_v2=false) -> NamedTuple

Opt-in convenience wrapper composing the low-risk moment-build specializations
in one call, for scripts that build their own `ctx` via `d4_exact_setup()` and
want the documented-safe defaults without repeating the wiring boilerplate.

- `pow_cache=true`  -> calls `enable_pow_cache!(ctx)` (fixed-mu,sigma power cache;
  bit-identical everywhere; recommended default for any live oracle use in this
  directory).
- `autarky_cf_v2=true` -> calls `enable_autarky_cf_v2!(ctx; pow_cache=...)`
  (cached-base focal-autarky CF column; tight/<=1ULP on the moment, bit-identical
  Delta_dual). Composes with the pow cache automatically when both are requested.
  Off by default -- see the module docstring above and
  docs/lowrisk_specialization_live_wiring.md for when this is (and is not)
  worth turning on.

Returns a NamedTuple `(pow_cache=<MuSigmaPowCache or nothing>, autarky_cf_v2=<AutarkyCFBase or nothing>)`
so callers can inspect `.n_recompute`/`.n_reuse` / `.n_build`/`.n_reuse` if desired.
"""
function enable_live_defaults!(ctx; pow_cache::Bool = true, autarky_cf_v2::Bool = false)
    pc = pow_cache ? enable_pow_cache!(ctx) : nothing
    base = autarky_cf_v2 ? enable_autarky_cf_v2!(ctx; pow_cache = pc) : nothing
    return (pow_cache = pc, autarky_cf_v2 = base)
end
