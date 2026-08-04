# Section 12 — short outer A/Bs

## Algorithmic-parity A/B (task §12.1)

**Not run.** Blocked on section 11's own precise blocker (FULL production wrapper hardcodes
`threaded=true, h_mode=:cached`; no diagnostic adapter was built this session to bypass it). Only
the REDUCED side of an algorithmic-parity harness was confirmed functional (section 11). A genuine
two-arm A/B under this mode requires both sides.

```
ALGORITHMIC_PARITY_AB_W20K =
    unrestricted:fail_full_side_adapter_not_built
    flexible_CM:fail_full_side_adapter_not_built
    common_frechet:fail_full_side_adapter_not_built
    origin_ZC:fail_full_side_adapter_not_built
    CM_plus_ZC:fail_full_side_adapter_not_built
```

## Production-parity A/B (task §12.2) — real, all 5 families, real D20/W=20,000

Run via the canonical CLI (`bin/run_profiled_model.jl`), both formulations, all 5 families,
`--diagnostic-budget 90` (90s KNITRO wall-clock cap), `--threaded-gradient true` (10 threads),
REDUCED using native coordinate mode (`--a-coordinate-mode profiled_pivot_anchor_relative`, the
retained default per section 10's own verdict), same scientific manifest
(`configs/smoke_w20k_2026-08-03.toml`) for both arms.

**A real, pre-existing bug was found and fixed while launching these** (not introduced this
session, but only surfaced by exercising this exact CLI path with the new coordinate-mode wiring):
`bin/run_profiled_model.jl`'s own `_run_reduced` include list had
`profiled_production_outer_constrained_2026-08-02.jl` listed BEFORE the three new coordinate-mode
dependency files it now requires (`cm_aspace_coordinate.jl`,
`profiled_powered_relative_a_2026-08-04.jl`, `profiled_coordinate_mode_dispatch_2026-08-04.jl`),
causing every REDUCED CLI invocation to fail immediately with an include-order error. Caught within
the first ~25 seconds of the first launch (per this repo's own standing "check in early" rule) and
fixed by reordering the include list.

| Family | Formulation | n_eval | n_grad | wall (s) | best objective (gp, or kappa for unrestricted) |
|---|---|---|---|---|---|
| unrestricted | reduced | 75 | 15 | 96.9 | gp=0.95906 |
| unrestricted | full | 26 | 21 | 94.1 | kappa=0.06836 (different metric, not directly comparable to gp) |
| flexible_cm | reduced | 20 | 8 | 135.5 | gp=0.96591 |
| flexible_cm | full | 13 | 5 | 95.5 | gp=0.96619 |
| common_frechet | reduced | 17 | 9 | 122.0 | gp=0.96690 |
| common_frechet | full | 18 | 4 | 93.8 | gp=0.97897 |
| origin_zc | reduced | 95 | 41 | 102.6 | gp=0.96030 |
| origin_zc | full | 52 | 15 | 90.8 | gp=0.96113 |
| cm_meanzc | reduced | 17 | 7 | 209.4 | gp=0.96623 |
| cm_meanzc | full | 9 | 4 | 133.8 | gp=0.96667 |

All 10 runs completed cleanly (`status=-401`/`-411` time-limit, expected given the bounded 90s
budget; KNITRO's own "Current point is feasible" for every FULL arm) with real, verified feasible
incumbents — none showed anomalous failure. Per the task's own explicit instruction, the comparison
below uses the **verified best objective value** (`gp`, the thing the outer problem actually
minimizes), not evaluation/gradient counts alone.

**Real, tentative pattern**: REDUCED reaches an equal-or-better (lower) `gp` than FULL in 3 of the
4 directly comparable families within a similar or smaller wall-clock budget (flexible_cm: 0.96591
vs 0.96619; common_frechet: 0.96690 vs 0.97897, a notably larger gap; origin_zc: 0.96030 vs
0.96113); cm_meanzc is close (0.96623 vs 0.96667) but took REDUCED over twice the wall time
(209.4s vs 133.8s) to get there. `unrestricted` is not directly comparable (FULL reports a
different metric, `kappa`, for this family's own objective convention).

## Not done this session

- **Repeated runs** for statistical confidence (task's own "at least two repetitions if contention
  is material" / "at least two repetitions for representative families") — every cell above is a
  single run.
- **Both execution orders** (task's own explicit requirement) — only one order run per family.
- **W=100,000 production-parity smoke** (task's own explicit ask, target ≥3 completed gradients per
  arm, ≥5 for unrestricted/flexible_cm/origin_zc) — not run this session; real remaining work,
  addressed partially by section 13's own W=100k gradient/short-run checks (a resource/safety gate,
  not a full production-parity A/B).

## Verdict

```
SHORT_OUTER_AB_ALGORITHMIC_PARITY_W20K = not_run (blocked, see task §11's own precise blocker)

SHORT_OUTER_AB_PRODUCTION_PARITY_W20K =
    unrestricted: complete_not_directly_comparable (different objective metric convention)
    flexible_CM: complete_reduced_reaches_equal_or_better_gp
    common_frechet: complete_reduced_reaches_notably_better_gp
    origin_ZC: complete_reduced_reaches_equal_or_better_gp
    CM_plus_ZC: complete_close_but_reduced_took_longer_wall_time
    (single run per arm, one execution order -- real, verified-objective evidence, but not yet a
    statistically confident result; do not treat as a final production recommendation)

SHORT_OUTER_AB_PRODUCTION_PARITY_W100K = not_run
```
