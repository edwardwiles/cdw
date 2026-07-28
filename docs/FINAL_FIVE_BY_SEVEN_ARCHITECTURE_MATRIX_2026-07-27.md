# Final 5×7 Architecture Matrix — 2026-07-27

Machine-readable version: `FINAL_FIVE_BY_SEVEN_ARCHITECTURE_MATRIX_2026-07-27.csv` (same data, one
row per family×column cell). This document is the narrative companion.

Columns: (1) economic moment construction, (2) outer A-gradient, (3) economic FG, (4) restriction
FG, (5) H_EE, (6) H_ER, (7) H_RR. Rows: unrestricted, flexible CM, common Fréchet, CM+ZC, ZC-only.

## Summary against the task's expected architecture

```
economic moment construction:  shared in-place builder (build_economic_moment_state!), all five  -- CONFIRMED
outer A-gradient:               shared economic_A_gradient!, all five                              -- CONFIRMED (unrestricted flipped this session)
economic FG:                    shared economic operator, all five                                 -- CONFIRMED (common-Frechet flipped this session)
restriction FG:                 family-specific operator, all restricted families                  -- CONFIRMED
H_EE:                           shared exact winner-pair, all five                                  -- CONFIRMED
H_ER:                           shared winner-side primitive + family-specific restriction accumulator -- CONFIRMED
H_RR:                           family-specific structured/direct method, never composite-G-dependent -- CONFIRMED (CM+ZC/ZC-only's H_ZZ now direct, this session)
```

**All seven expected-architecture properties hold for all five families as of this session's
merged HEAD.** Two cells were flipped/completed THIS session (marked in the CSV's
`default_status` column):

- common-Fréchet's economic FG + restriction FG (Goal 3, `CM_FRECHET_INNER_FG_BACKEND_DEFAULT`
  `:dense_reference` → `:cm_frechet_lookup`).
- unrestricted's outer A-gradient (Goal 4, `resolve_price_cache_backend`'s no-kwarg default
  `:cplus` → `:shared`, i.e. `economic_A_gradient!`).

And the CM+ZC/ZC-only H_RR (H_ZZ) cell moved from "obj.H-column dense read" to "raw ZC feature
state, shared `zc_restriction_gram!`" (Goal 7/8), with CM+ZC's H_ER cell gaining a genuinely new
`H_CZ` primitive completing the `[E|C|Z]` partition (Goal 7).

## Composite-G / dense-E read column, by family

Per the CSV's own `reads_composite_G_or_dense_E` column: **every cell across all five families
reads `no`** in ordinary production configuration. The two families whose Hessian internals this
session touched directly (CM+ZC, ZC-only) have this confirmed by a REAL, isolated,
counter-instrumented production-only run (not a static code read) — see
`CM_MEANZC_BLOCK_PARTITION_AND_HCZ_RELEASE_2026-07-27.md`'s "Production-only invariant check"
section: `dense_cross_hessian_calls=0` at CM+ZC's real production default, K_mean=1/K_pair=1, real
D=20/W=80,000.

## Known, disclosed exceptions (not silently dropped)

- **`skip_cm_fill_ref`** (flexible-CM, common-Fréchet): still exists as of this document's
  writing, mid-investigation/refactor by a concurrently-dispatched agent (Goal 10) — see
  `docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md` for the outcome once that lands. Does not
  affect the composite-G/dense-E read invariant above (it controls a CM-column FILL, not a read of
  composite `obj.H` outside the family's own restriction block), but is a pre-existing mutable-Ref
  pattern the task explicitly wants gone.
- **Common-Fréchet's H_RR (level-anchor self-term)**: analytically simple (the level anchor
  contributes a near-trivial diagonal/near-zero cross structure to `H_CC`'s own block per the
  existing `hessian_cm_structured!` implementation) — not given a SEPARATE named routine distinct
  from flexible-CM's `H_CC`, since the same function already serves both families correctly (this
  is the "shared structured method" the architecture table expects, not a gap).
- **`H_CC`/`H_EE`'s own self-blocks** (flexible-CM, common-Fréchet, CM+ZC): unconditionally dense
  BLAS/structured accumulation by design (small, family-owned state — CM bin thresholds, winner
  contrasts) — this is the EXPECTED terminal state per the architecture table ("family-specific
  structured/direct restriction method"), not something further to "fix."

## Gate status column

All `d20_gate` entries marked `PASS` in the CSV reflect REAL runs from this session or the
immediately-prior winner-aware-H_ER phase (whose numbers this session re-confirmed still hold via
its own D=4/D=20 5-family integration smokes, not re-derived from scratch) — see each family's own
linked release doc for the underlying numbers (max|ΔH|, status match, KKT/moment residuals).
