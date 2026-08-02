# CM+ZC widened-core finding (2026-08-01) — Phase 8 deferred, by user decision

## What CM+ZC actually is in this codebase (not what the task prompt assumed)

The task's own Phase 8 instructions describe CM+ZC as "profiled H_EE, corrected H_EC, corrected
H_EZ, unchanged H_CC, unchanged H_CZ, unchanged H_ZZ" — implying the same shape as the other three
families: a separate H_E,R off-diagonal block per restriction type, appended after the economic
columns. That is **not** how CM+ZC (`cm_meanzc_moments.jl`/`cm_meanzc_production.jl`) is actually
built.

CM+ZC widens `NCORE` itself to include the mean/pair-ZC restriction columns **inside** the core
BLAS block:

```text
NCORE_ext = aug.ncore_econ + aug.n_mean + aug.n_pair    (build_cm_meanzc_bin_ctx, cm_meanzc_production.jl)
```

`H_EZ` (task's own name) shows up as `H_EM` — a cross-term **within** the widened `NCORE×NCORE`
block (`HEE[1:ncore_core, ncore_core+1:NCORE]`, `_fill_cm_HEE!`'s own `ncore_core < NCORE` branch),
not as a separate off-diagonal block the way H_EC (flexible CM) or H_EF (common Fréchet) are. Only
the CM-grid restriction (`C`, this family's own `ncm` = the finite-grid CDF contrasts, unrelated to
the mean/pair-ZC columns) remains a genuine separate block beyond `NCORE`.

This is confirmed **pre-existing**, not a gap this session introduced: `_fill_cm_HEE!`'s own
profiled branch already has an explicit guard —

```julia
ncore == NCORE || error("_fill_cm_HEE!: profiled_layout set but ncore_core=$ncore != NCORE=$NCORE "
    * "-- CM+ZC/mean-pair widening is not ported for the reduced economic layout yet.")
```

— written by a prior session, deliberately declining to handle this case rather than risk a wrong
Hessian.

## What real work this actually requires

A genuinely new reduced kernel, not a gather branch mirroring the other three families:

1. A **reduced+widened H_EE** kernel: the reduced economic sub-block (`1+layout.total_reduced_
   economic_moments` wide, exactly like the other families) **plus** the `H_EM` cross-term between
   those reduced economic rows and the (unwidened, unchanged) mean/pair-ZC columns, **plus** the
   Z-restriction Gram (`H_MM`) — all three pieces consistently sized against the SAME widened total.
2. A gather branch for the true-economic **H_EC** rows only (the CM-grid restriction block, which
   stays a separate block beyond the widened `NCORE`).
3. Re-verification that `H_CZ`/`H_ZZ` (task's own "unchanged" blocks) are genuinely unaffected —
   plausible by analogy to the other three families' own H_CF/H_FF/H_ZZ findings, but not yet
   confirmed by direct read for CM+ZC's own widened-width bookkeeping.
4. A new D4 gate mirroring the other three families' own (Part 1 gathered-G check, Part 2 real
   KNITRO solve).

## Decision

Presented to the user as an explicit fork (2026-08-01): attempt this new design now vs. defer and
move to other unblocked work. **User chose to defer.** CM+ZC is left exactly as it was at the start
of this session — still blocked by the same pre-existing `_fill_cm_HEE!` guard, no new bugs
introduced, no half-finished attempt left in the tree.

## Status of the other three restricted families (for contrast — these really were the simpler shape)

Flexible CM, common Fréchet, and ZC-only all match the task's assumed shape (separate H_E,R block(s)
appended after the reduced economic columns) and are complete — see
`PROFILED_RESTRICTED_ENDTOEND_SOURCE_SNAPSHOT_2026-08-01.md` and this branch's own commit history
(`c8143cc` common Fréchet, `6677f9d` ZC-only, flexible CM pre-existing at `f1f969b`).
