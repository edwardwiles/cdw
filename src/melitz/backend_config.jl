# Melitz production-backend configuration + usage counters (2026-07-26 production-port
# session). See docs/melitz_production_fast_backend_2026-07-26.md.
#
# ONE explicit, immutable configuration object, threaded through every production
# constructor, so "which implementation actually ran" is never left to a scattered set of
# per-function keyword defaults that can silently drift out of sync with each other (the
# exact failure mode the 2026-07-26 handoff document flagged: moment_backend/gradient_backend
# already existed as validated opt-in code but every construction site still defaulted to the
# dense/legacy path). `:auto` fields resolve to a concrete backend from `D` (and, for the
# Hessian, thread count) via a measured, not guessed, threshold -- see
# `melitz_resolve_moment_backend`/`melitz_resolve_gradient_backend`/`melitz_resolve_hessian_backend`.

"""
    MelitzBackendConfig

Immutable, explicit backend selection for one Melitz production bundle. Three named presets
are provided (`MELITZ_PRODUCTION_FAST` -- strict, `forbid_dense_fallback=true`;
`MELITZ_PRODUCTION_COMPAT` -- permissive, explicit dense fallback allowed;
`MELITZ_DENSE_REFERENCE` -- the full legacy dense path) -- construct a custom one only when
deliberately mixing backends for an ablation study (Phase 14).

Fields:
- `inner_backend::Symbol`            -- `:matrix_free` or `:dense_reference`. Controls
  whether `build_melitz_psi_bundle`/`build_melitz_implicit_bundle` return a `MelitzCCBundle`
  (matrix-free objective/gradient/Hessian) or the legacy `PsiObjectiveBundleDelta`/
  `PsiObjectiveBundleImplicit` (dense `obj.H`, `cc_algo`-owned functor).
- `moment_backend::Symbol`           -- `:auto`, `:sorted_tail_serial`, `:sorted_tail_parallel`,
  or `:dense_reference`. Only meaningful when `inner_backend==:dense_reference` (the
  matrix-free bundle never builds a dense/sorted moment MATRIX at all -- `MelitzMomentOperator`
  replaces it structurally); kept as an explicit field anyway so a dense-reference run's own
  moment-construction speed is independently controllable (Phase 14's config B/C ablation
  rungs need dense inner solve + sorted moments, e.g.).
- `outer_gradient_backend::Symbol`   -- `:auto`, or any concrete `gradient_backend` symbol
  accepted by `build_melitz_implicit_bundle` (`:B`, `:B_direct_argument_parallel`,
  `:B_direct_argument_sorted_parallel`, ...).
- `hessian_backend::Symbol`          -- `:auto`, `:structured_serial`, or `:structured_parallel`.
  Only meaningful when `inner_backend==:matrix_free`.
- `screening_backend::Symbol`        -- `:matrix_free` or `:dense_reference`. Controls whether
  `melitz_classified_inner_solve`'s screens/predictor-corrector/nuisance-profile helpers use
  the matrix-free operator or the legacy dense `obj.H`/`CS.select_G_from_H` path.
- `cache_policy::Symbol`             -- `:bounded` (the existing `MelitzExactPointCache`/
  `MelitzDualBank` bounded-LRU tiers) is the only supported value at present; kept as an
  explicit field for forward compatibility.
- `evaluation_cap::Union{Nothing,Float64}` -- forwarded to `melitz_configure_lower_limit`
  when non-`nothing`; `nothing` means the caller supplies its own `inner_solve_config`.
- `forbid_dense_fallback::Bool`      -- when `true`, ANY attempt to materialize a dense `G`,
  call a dense inner callback, call the legacy dense outer gradient, or call a dense
  production screen throws immediately (Phase 10) instead of silently falling back.
- `threads::Int`                     -- `Threads.nthreads()` at construction time, recorded
  for the printed summary and for cache/context fingerprints (a bundle built under one
  thread count reused under a different one is a common source of stale thread-local scratch
  bugs elsewhere in this codebase -- recording it here makes a mismatch detectable).
"""
struct MelitzBackendConfig
    inner_backend::Symbol
    moment_backend::Symbol
    outer_gradient_backend::Symbol
    hessian_backend::Symbol
    screening_backend::Symbol
    cache_policy::Symbol
    evaluation_cap::Union{Nothing,Float64}
    forbid_dense_fallback::Bool
    threads::Int
end

function MelitzBackendConfig(; inner_backend::Symbol=:matrix_free,
                              moment_backend::Symbol=:auto,
                              outer_gradient_backend::Symbol=:auto,
                              hessian_backend::Symbol=:auto,
                              screening_backend::Symbol=:matrix_free,
                              cache_policy::Symbol=:bounded,
                              evaluation_cap::Union{Nothing,Real}=nothing,
                              forbid_dense_fallback::Bool=false)
    inner_backend in (:matrix_free, :dense_reference) || throw(ArgumentError(
        "MelitzBackendConfig: inner_backend must be :matrix_free or :dense_reference, got $inner_backend"))
    screening_backend in (:matrix_free, :dense_reference) || throw(ArgumentError(
        "MelitzBackendConfig: screening_backend must be :matrix_free or :dense_reference, got $screening_backend"))
    cache_policy == :bounded || throw(ArgumentError(
        "MelitzBackendConfig: cache_policy must be :bounded (the only supported value), got $cache_policy"))
    return MelitzBackendConfig(inner_backend, moment_backend, outer_gradient_backend, hessian_backend,
        screening_backend, cache_policy, evaluation_cap === nothing ? nothing : Float64(evaluation_cap),
        forbid_dense_fallback, Threads.nthreads())
end

"""
    MELITZ_PRODUCTION_FAST

2026-07-26 closure session (governing prompt Phase 2): the STRICT production preset --
matrix-free inner solve, sorted moments/outer-gradient (`:auto`-resolved), matrix-free
screening, no evaluation cap pre-set (callers pass one via `inner_solve_config`/
`delta_evaluation_cap` as today), and `forbid_dense_fallback=true`. Every REAL production
campaign should use this preset (or its `forbid_dense_fallback` value threaded into the
production constructors -- see `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`/
`build_melitz_implicit_bundle`'s own `forbid_dense_fallback` kwarg). If a requested option is
unavailable in matrix-free form, construction fails IMMEDIATELY with an informative
`ArgumentError` -- never silently falls back to a dense path and never fails only after an
expensive callback begins.

Previously (`MelitzBackendConfig()`'s own default) had `forbid_dense_fallback=false` -- an
authoritative "production-fast" preset that silently PERMITTED a dense fallback was exactly
backwards; corrected here. The permissive default now lives at `MELITZ_PRODUCTION_COMPAT`.
"""
const MELITZ_PRODUCTION_FAST = MelitzBackendConfig(forbid_dense_fallback=true)

"""
    MELITZ_PRODUCTION_COMPAT

2026-07-26 closure session (governing prompt Phase 2): the DEVELOPMENT-COMPATIBILITY preset
-- otherwise identical to `MELITZ_PRODUCTION_FAST` (matrix-free by default, `:auto`-resolved
moments/gradient/Hessian/screening), but `forbid_dense_fallback=false`, so an explicit,
documented dense-fallback choice (e.g. a caller-supplied `gradient_backend=:B` for a
cross-check, or `moment_backend=:dense_reference`) is PERMITTED rather than throwing. Use
this preset only when a genuine reason exists to mix backends within an otherwise
matrix-free run (an ablation study, a targeted comparison); real production campaigns should
use `MELITZ_PRODUCTION_FAST`.
"""
const MELITZ_PRODUCTION_COMPAT = MelitzBackendConfig(forbid_dense_fallback=false)

"""
    MELITZ_DENSE_REFERENCE

The full legacy path: dense inner solve (`PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`,
`cc_algo`-owned functor), dense moment construction, the original dense finite-difference
outer gradient (`:B`), dense screening. Kept fully intact and callable for diagnostics/
cross-checks (Phase 12-14) -- never removed. `forbid_dense_fallback=false` by construction
(this preset IS the dense path -- forbidding dense fallback here would be self-contradictory).
"""
const MELITZ_DENSE_REFERENCE = MelitzBackendConfig(inner_backend=:dense_reference,
    moment_backend=:dense_reference, outer_gradient_backend=:B, hessian_backend=:structured_serial,
    screening_backend=:dense_reference, forbid_dense_fallback=false)

# ----------------------------------------------------------------------------------------
# :auto resolution -- derived from a post-JIT benchmark (Phase 2/3's own requirement: "derive
# the threshold from post-JIT benchmarks", not a guess). Benchmarked D=4/W=20,000 vs real
# D=20/W=80,000 (docs/melitz_production_fast_backend_2026-07-26.md Section on :auto
# thresholds): sorted-parallel/structured-parallel only wins once BOTH D is large enough to
# give each thread real work AND more than one thread actually exists -- at D=4 the parallel
# variants are net slower (task-spawn overhead exceeds the O(D) serial work per thread), so
# the threshold is deliberately set at D>=10 (strictly between the two benchmarked fixtures,
# closer to the D=20 side where parallel wins) AND Threads.nthreads() > 1.
# ----------------------------------------------------------------------------------------

const MELITZ_AUTO_PARALLEL_D_THRESHOLD = 10

"""
    melitz_resolve_moment_backend(cfg, D) -> Symbol

Resolves `cfg.moment_backend` to a concrete backend accepted by `build_melitz_psi_bundle`'s
own `moment_backend` kwarg. `:auto` -> `:sorted_tail_parallel` if `D >=
MELITZ_AUTO_PARALLEL_D_THRESHOLD` and `Threads.nthreads() > 1`, else `:sorted_tail_serial`.
`:dense_reference` passes through unchanged (an explicit diagnostic choice, never silently
promoted). Any other concrete value (`:sorted_tail_serial`/`:sorted_tail_parallel`) passes
through unchanged too -- `:auto` is the only symbol this function actually resolves.
"""
function melitz_resolve_moment_backend(cfg::MelitzBackendConfig, D::Int)
    cfg.moment_backend == :auto || return cfg.moment_backend
    return (D >= MELITZ_AUTO_PARALLEL_D_THRESHOLD && Threads.nthreads() > 1) ?
        :sorted_tail_parallel : :sorted_tail_serial
end

"""
    melitz_resolve_gradient_backend(cfg, D) -> Symbol

Resolves `cfg.outer_gradient_backend`. `:auto` -> the sorted crossing-slice backend
(`:B_direct_argument_sorted_parallel`/`_serial`, same `D`/thread-count rule as
`melitz_resolve_moment_backend`) when `inner_backend==:matrix_free` (matrix-free bundles
always have a `sorted_tail_ctx` available via their own operator -- see
`melitz_ensure_sorted_tail_ctx!`), or the plain direct backend
(`:B_direct_argument_parallel`/`_serial`, no sorted-tail requirement) when
`inner_backend==:dense_reference` and `cfg.moment_backend` did not resolve to a sorted
variant.
"""
function melitz_resolve_gradient_backend(cfg::MelitzBackendConfig, D::Int)
    cfg.outer_gradient_backend == :auto || return cfg.outer_gradient_backend
    parallel = D >= MELITZ_AUTO_PARALLEL_D_THRESHOLD && Threads.nthreads() > 1
    have_sorted_ctx = cfg.inner_backend == :matrix_free ||
                       melitz_resolve_moment_backend(cfg, D) in (:sorted_tail_serial, :sorted_tail_parallel)
    have_sorted_ctx || return parallel ? :B_direct_argument_parallel : :B_direct_argument_serial
    return parallel ? :B_direct_argument_sorted_parallel : :B_direct_argument_sorted_serial
end

"""
    melitz_resolve_hessian_backend(cfg, D) -> Symbol

Resolves `cfg.hessian_backend` (matrix-free bundles only). `:auto` -> `:structured_parallel`
if `D >= MELITZ_AUTO_PARALLEL_D_THRESHOLD && Threads.nthreads() > 1`, else `:structured_serial`.
"""
function melitz_resolve_hessian_backend(cfg::MelitzBackendConfig, D::Int)
    cfg.hessian_backend == :auto || return cfg.hessian_backend
    return (D >= MELITZ_AUTO_PARALLEL_D_THRESHOLD && Threads.nthreads() > 1) ?
        :structured_parallel : :structured_serial
end

"""
    melitz_print_backend_summary(cfg, D) -> nothing

Phase 1's required one-line-per-field startup banner -- prints the RESOLVED (not raw
`:auto`) backend selection so a log always shows what actually ran.
"""
function melitz_print_backend_summary(cfg::MelitzBackendConfig, D::Int)
    println("Melitz backend:")
    println("  inner       = ", cfg.inner_backend)
    println("  moments     = ", cfg.inner_backend == :matrix_free ? :matrix_free_operator :
                                 melitz_resolve_moment_backend(cfg, D))
    println("  outer_grad  = ", melitz_resolve_gradient_backend(cfg, D))
    println("  Hessian     = ", cfg.inner_backend == :matrix_free ? melitz_resolve_hessian_backend(cfg, D) : :dense_gemm)
    println("  screening   = ", cfg.screening_backend)
    println("  dense fallbacks allowed = ", !cfg.forbid_dense_fallback)
    println("  threads     = ", cfg.threads)
    return nothing
end

# ----------------------------------------------------------------------------------------
# Phase 10: per-run backend usage counters. GLOBAL `Ref(0)` counters, matching this
# codebase's own established convention exactly (cc_algo's INNER_SOLVE_COUNT/
# INNER_INFEAS_COUNT, this file's own melitz_expand_theta-adjacent JAC_H_THETA_BRANCH_COUNT,
# etc.) rather than a threaded-through struct -- avoids changing the signature of every
# call site that needs to report a backend event (`melitz_classified_inner_solve`,
# `_base_arg0!`, the moment/gradient backend dispatchers, ...), at the cost of counts being
# process-global rather than per-campaign-object. `melitz_backend_counters_reset!()` /
# `melitz_backend_counters_snapshot()` give a caller (an ablation study running config A then
# config D in the same session) an explicit before/after delta instead.
# ----------------------------------------------------------------------------------------

const MELITZ_SORTED_MOMENT_CALLS = Ref(0)
const MELITZ_DENSE_MOMENT_CALLS = Ref(0)
const MELITZ_SORTED_OUTER_GRADIENT_CALLS = Ref(0)
const MELITZ_DENSE_OUTER_GRADIENT_CALLS = Ref(0)
const MELITZ_MATRIX_FREE_OBJECTIVE_CALLS = Ref(0)
const MELITZ_MATRIX_FREE_GRADIENT_CALLS = Ref(0)
const MELITZ_MATRIX_FREE_HESSIAN_CALLS = Ref(0)
const MELITZ_DENSE_INNER_OBJECTIVE_CALLS = Ref(0)
const MELITZ_DENSE_INNER_GRADIENT_CALLS = Ref(0)
const MELITZ_DENSE_INNER_HESSIAN_CALLS = Ref(0)
const MELITZ_DENSE_G_MATERIALIZATIONS = Ref(0)
const MELITZ_EXACT_POINT_CACHE_HITS = Ref(0)
const MELITZ_FC_TO_GA_CACHE_HITS = Ref(0)
const MELITZ_OPERATOR_REBUILDS = Ref(0)
const MELITZ_EVALUATION_CAP_EXITS = Ref(0)
const MELITZ_NUMERICAL_FAILURES = Ref(0)
const MELITZ_PRODUCTION_DENSE_SCREEN_CALLS = Ref(0)
const MELITZ_DIAGNOSTIC_DENSE_SCREEN_CALLS = Ref(0)
const MELITZ_MATRIX_FREE_RANGE_SCREEN_CALLS = Ref(0)   # 2026-07-26 closure (Phase 7)

const MELITZ_ALL_COUNTER_REFS = (
    sorted_moment_calls=MELITZ_SORTED_MOMENT_CALLS, dense_moment_calls=MELITZ_DENSE_MOMENT_CALLS,
    sorted_outer_gradient_calls=MELITZ_SORTED_OUTER_GRADIENT_CALLS,
    dense_outer_gradient_calls=MELITZ_DENSE_OUTER_GRADIENT_CALLS,
    matrix_free_objective_calls=MELITZ_MATRIX_FREE_OBJECTIVE_CALLS,
    matrix_free_gradient_calls=MELITZ_MATRIX_FREE_GRADIENT_CALLS,
    matrix_free_hessian_calls=MELITZ_MATRIX_FREE_HESSIAN_CALLS,
    matrix_free_range_screen_calls=MELITZ_MATRIX_FREE_RANGE_SCREEN_CALLS,
    dense_inner_objective_calls=MELITZ_DENSE_INNER_OBJECTIVE_CALLS,
    dense_inner_gradient_calls=MELITZ_DENSE_INNER_GRADIENT_CALLS,
    dense_inner_hessian_calls=MELITZ_DENSE_INNER_HESSIAN_CALLS,
    dense_G_materializations=MELITZ_DENSE_G_MATERIALIZATIONS,
    exact_point_cache_hits=MELITZ_EXACT_POINT_CACHE_HITS, FC_to_GA_cache_hits=MELITZ_FC_TO_GA_CACHE_HITS,
    operator_rebuilds=MELITZ_OPERATOR_REBUILDS, evaluation_cap_exits=MELITZ_EVALUATION_CAP_EXITS,
    numerical_failures=MELITZ_NUMERICAL_FAILURES,
    production_dense_screen_calls=MELITZ_PRODUCTION_DENSE_SCREEN_CALLS,
    diagnostic_dense_screen_calls=MELITZ_DIAGNOSTIC_DENSE_SCREEN_CALLS)

"melitz_backend_counters_reset!() -- zero every Phase 10 counter (call at the start of a campaign/benchmark)."
function melitz_backend_counters_reset!()
    for r in MELITZ_ALL_COUNTER_REFS
        r[] = 0
    end
    return nothing
end

"melitz_backend_counters_snapshot() -> NamedTuple -- current value of every Phase 10 counter."
melitz_backend_counters_snapshot() = NamedTuple{keys(MELITZ_ALL_COUNTER_REFS)}(r[] for r in MELITZ_ALL_COUNTER_REFS)

"""
    melitz_check_no_dense_fallback!(cfg, counter_ref::Ref{Int}, call_site::String)

Phase 10's fail-fast check: increments `counter_ref` unconditionally (an honest count even
when fallback is permitted), then, if `cfg.forbid_dense_fallback`, throws immediately with
`call_site` in the message.
"""
function melitz_check_no_dense_fallback!(cfg::MelitzBackendConfig, counter_ref::Ref{Int}, call_site::String)
    counter_ref[] += 1
    if cfg.forbid_dense_fallback
        error("melitz_check_no_dense_fallback!: dense fallback triggered at $call_site " *
              "while cfg.forbid_dense_fallback=true -- production-fast mode does not permit " *
              "this path. See docs/melitz_production_fast_backend_2026-07-26.md.")
    end
    return nothing
end
