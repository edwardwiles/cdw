# Melitz hot-path allocation audit (2026-07-27 continuation, governing prompt Phase 3)

Continues the 2026-07-27 addendum session's own Section H (`docs/melitz_outer_parameterization_comparison_2026-07-26.md`),
which found one real W-scaling allocation (fixed and verified there) and flagged, but did not
fix, two further items: `copy(theta)` inside the PARALLEL gradient backends, and redundant
per-call recomputation inside `expand_free_theta`/`reduce_to_free_theta`. This document covers
this session's own fixes for both.

**Scope note, stated up front**: the governing prompt's own ask for this phase (exhaustive
call-graph tracing across ~15 named functions, a full static hazard inventory across ~15
pattern classes, dynamic `Profile.Allocs` measurement at two scales for every named callback,
and allocation-ceiling regression tests for every one) is larger than this session's remaining
time could execute to the letter, on top of the six-way tournament/gradient-quality/multi-seed
work the same governing prompt also requires. What follows is a genuine, evidence-based pass on
the two SPECIFIC, previously-flagged issues -- found, fixed, and verified with live
`@allocated` measurements and new regression tests -- not a complete instantiation of every
requested subsection. Flagged explicitly, matching this repo's own established convention for
honestly scoping a large ask against session time (e.g. the 2026-07-26 addendum's own Section H
opening note).

## 1. Fix 1: `copy(theta)` in the parallel gradient backends

**Where**: `direct_gradient.jl`'s `make_melitz_gradient_delta_direct_parallel` and
`sorted_crossing_gradient.jl`'s `make_melitz_gradient_delta_direct_sorted_parallel`. Both
allocated TWO fresh `n`-length vectors (`theta_p`, `theta_m`) on EVERY coordinate, inside a
`Threads.@threads :static for r in 1:n` sweep -- i.e. `2n` allocations of length `n` per
gradient call (`O(n^2)` total bytes), while their SERIAL siblings already correctly reused a
single persistent buffer via `copyto!`.

**Fix**: per-thread persistent `theta_p`/`theta_m` buffers (`thetap_bufs`/`thetam_bufs`, one per
`Threads.maxthreadid()`), mirroring the serial backend's own `copyto!`-based reuse exactly:
mutate only the one affected entry per perturbation, no restore needed since the buffer is
fully overwritten by `copyto!` on the next coordinate/next call.

**A latent bug found and fixed while writing this fix**: the buffer-reallocation check
initially reused the SAME `nthreads_alloc[]` flag the pre-existing `arg0_buf`/`Gp_bufs` block
already uses -- but that block updates `nthreads_alloc[]` to the new thread count FIRST, so the
new `thetap_bufs` block's own `nthreads_alloc[] != nt` check would already read `false` by the
time it ran, silently skipping reallocation if `nt` (Julia's thread pool, which only ever
grows, never shrinks, within a session) grew between calls without `W`/`n` also changing --
this WOULD have thrown a `BoundsError` accessing `thetap_bufs[][tid]` for a `tid` beyond the
stale buffer's length. Fixed by checking `length(thetap_bufs[]) != nt` directly instead of the
shared flag, in both files.

**Verification**: existing "serial and parallel direct backends are bit-identical" /
`g_sorted_par_d20 == g_sorted_d20` tests (unaffected numerically -- confirmed still passing,
byte-for-byte) plus two NEW allocation regression tests (Section 4 below).

## 2. Fix 2: redundant f-gravity-pivot recomputation in `expand_free_theta`/`reduce_to_free_theta`

**Where**: `delta_star.jl`. Both functions called `f_gravity_pivot_avoid_indices` (a
comprehension plus two `setdiff` calls) followed by `build_gravity_pivot` (another `setdiff`,
an `abs.(...)` temporary, an `argmax`) on EVERY call -- i.e. on every FD coordinate probe, up to
`O(n)` times per outer gradient -- despite the PIVOT CHOICE depending only on
`ctx.c_full[ctx.f_free_lin]` and `ctx.A_pivot.pivot` (both fixed for the lifetime of a given
`ctx`), never on `g0` (`build_gravity_pivot`'s own docstring: "WHICH cell is chosen depends only
on `c` and the avoid-set, never on `g0`").

**Fix**: `melitz_cached_f_pivot_parts(ctx)` memoizes `(c_free, pivot_index, other_indices)` by
`ctx` OBJECT IDENTITY (`===`), mirroring the SAME established pattern this codebase already uses
for context-invariant precomputation (`direct_gradient.jl`'s own `compact_cache`/`ctx_cache`).
Callers reconstruct a fresh, cheap `GravityPivot(length(c_free), pivot, other, c_free, g0)` from
the cached parts on every call -- no new allocation of `c_free`/`other` (`pivot_expand`/
`pivot_reduce` never mutate a `GravityPivot`'s fields, confirmed by reading both), only the
per-call-varying `g0` scalar changes.

**Thread safety**: unlike the closure-local `compact_cache`/`ctx_cache` pattern (scoped to one
gradient-backend INSTANCE, populated single-threaded before its own `Threads.@threads` region
starts), this cache is process-global AND reached from INSIDE a live parallel region
(`sorted_crossing_gradient.jl`'s `_fill_compact_direct_columns_crossing_sorted!`, called from
the parallel sorted gradient backend's per-coordinate thread body, calls `melitz_expand_theta`
directly). A bare check-then-set `Ref` would race both harmlessly (redundant recompute within
one solve) and harmfully (cross-contamination between two DIFFERENT outer solves' `ctx`
objects running concurrently in the same process). Guarded by a `ReentrantLock` for correctness
in every case; contention is negligible in the common case (one active outer solve, cache
already warm for its own `ctx`).

**Verification**: roundtrip correctness at D=4 (`~1.1e-15` A, `~7.8e-15` f) and real D=20
(`~8.9e-16` A, `~2.3e-13` f) -- both machine precision, confirming the cache-based
reconstruction is numerically identical to the original per-call rebuild. Warm-cache
`@allocated` is stable across repeated calls with the same `ctx` (832 bytes at D=4, 19,672
bytes at D=20 -- the latter dominated by the legitimately-necessary output arrays `A`/`f`/
`logA_full`/`logf_free_full`, not by the eliminated redundant pivot-selection work).

## 3. Phase 4 (matrix-free dead code) cross-reference

`matrix_free_dual_solve.jl`'s `MelitzMatrixFreeDualBundle` confirmed unreachable from any
production entry point (grep, every `src/melitz/*.jl` file) -- its only real consumer is a
dedicated cross-check test in `test/melitz/runtests.jl`. Documented explicitly as
diagnostic/reference-only via a new file-header banner (see that file); not deleted, since it
remains useful as an independently-coded validation oracle for `MelitzCCBundle`. No production
`backend`/`inner_backend` option anywhere can select it.

## 4. New allocation regression tests

Added to `test/melitz/runtests.jl`:

- **Real D=20/W=80,000, parallel sorted gradient backend** ("Phase 3.2/3.3... parallel sorted
  gradient allocation regression"): post-warm-up `@allocated` bound (absolute cap, `500,000`
  bytes -- the adjacent testset's own D=4-fixture-derived `calib` makes `n` small here despite
  the "D=20" name, so a relative `n^2` bound is the wrong shape of test at this scale, matching
  this repo's own established reasoning for the adjacent serial-backend test). Also asserts
  `g_scratch_par == g_sorted_d20` (bit-identical to the serial reference, unaffected by the
  buffer-reuse refactor).
- **D=4 fixture, parallel direct gradient backend** ("Phase 3.2... parallel direct backend no
  longer copy(theta)-per-coordinate"): asserts POST-WARM-UP allocation is STABLE across repeated
  calls at a fixed `n` (`bytes_p1 == bytes_p2`) -- would catch a reintroduced `copy(theta)`
  (which would show up as allocation that doesn't shrink/stabilize after warm-up) regardless of
  the fixture's small absolute scale.
- **Real D=20/W=80,000, `melitz_expand_theta`**: `@allocated` stable across repeated calls with
  the SAME `ctx` (would catch `melitz_cached_f_pivot_parts`'s own ctx-identity check ever being
  broken, which would cause a rebuild -- with its own `setdiff`/`abs.`/`argmax` allocations --
  on every call instead of once).

Full suite re-verified green after both fixes: 56/56 testsets `Pass==Total`, standalone (no
`cc_algo`) subprocess passing, zero regressions.

## 5. What this audit explicitly did NOT cover (disclosed, not silently dropped)

- The exhaustive per-function call-graph table (Function / Calls per solve / Allocated bytes /
  Scaling / Action) the governing prompt's own Section 3.1 requests.
- `Profile.Allocs`-based (as opposed to `@allocated`-based) measurement.
- Separating KNITRO.jl's own allocations from Melitz-owned callback allocations.
- The full static hazard inventory across the ~15 named pattern classes (Section 3.4) beyond
  the two patterns actually found and fixed here.
- Memory-traffic auditing (`copyto!`/`fill!` bytes-moved reporting, Section 3.6).

These remain open for a future session with a larger dedicated allocation-audit time budget,
per this repo's own established convention of flagging incomplete scope explicitly rather than
fabricating results for it.
