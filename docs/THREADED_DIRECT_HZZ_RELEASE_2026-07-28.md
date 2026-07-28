# Threaded direct H_ZZ — release notes (2026-07-28)

**Superseded/refined by the user's same-day addendum** — see
`ZC_GRAM_BLAS_EXPERIMENT_DESIGN_2026-07-28.md` for the full, addendum-scoped candidate menu
(`:reference` / `:blas_syrk` / `:blas_gemm` / `:threaded_packed`). This document records the
original Section 7 threaded-Gram design (`zc_gram_threaded_packed!`) for completeness; the addendum
additionally requires two BLAS candidates and an explicit "do not rematerialize centered Z"
requirement that changed the workspace design (raw immutable `Phi` instead of a per-callback
centered copy) — see the BLAS doc for the actual production candidate menu and decision.

## `zc_gram_threaded_packed!` (Candidate C / non-BLAS)

Column-block ownership over the Gram's `nx x nx` output (`nx = n_restriction(op)`, typically small
— 20 to a few hundred at production `K_mean`/`K_pair` widths). Each worker owns a disjoint,
contiguous range of Gram COLUMNS and computes the FULL upper-triangle entries for its own columns
directly: `HZZraw[j1,j2] = sum_w Phi[w,j1]*Y[w,j2]` for `j1<=j2`, `Y = S.*Phi` precomputed once
(serial, `O(W*nx)`, cheap relative to the `O(W*nx^2)` main loop for any `nx` beyond a handful).

This supersedes the original task brief's "thread-local packed accumulator + reduce" design
(Candidate A there) — the addendum itself suggested this simplification ("benchmark disjoint
upper-column ownership if it avoids the thread-local reduction"), adopted here since column
ownership needs no reduce step and no thread-local `nx x nx` scratch per worker (memory scales with
`workers`, not `workers * nx^2`).

No atomics: each output entry `HZZraw[j1,j2]` is written by exactly one worker (the one owning
column `j2`), never contended.

## Correction step

Per the addendum's algebraic identity (`Z'SZ = Phi'SPhi - u*t' - t*u' + s0*t*t'`), the raw Gram
computed above is UNCENTERED — a small `O(nx^2)` serial correction (not worth threading at these
sizes) is applied afterward, upper-triangle only, then mirrored once for the packed-Hessian
consumer. See `ZC_GRAM_BLAS_EXPERIMENT_DESIGN_2026-07-28.md` for the full derivation.

## Shared by

CM+ZC's `HMM` and origin-ZC's `HRR` — ONE dispatcher (`zc_gram_dispatch!`,
`zc_gram_blas_candidates.jl`), never two independent implementations, satisfying the task brief's
own Section 12/13 "shared by CM+ZC; ZC-only... do not maintain separate H_ZZ implementations by
family" requirement.

## Correctness gates

`test_threaded_cross_hessian_d4.jl` — `threaded_packed` backend checked against `:reference` for
both families, several K configs, calibration + perturbed points (`maxdiff < 1e-9`, the slightly
looser tolerance vs the pure-threading kernels reflecting the algebraic-identity re-derivation, not
floating-point reordering within an identical formula).
