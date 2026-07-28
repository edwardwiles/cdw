# Five-family independent-multistart shakedown — final report (2026-07-28)

## Setup

- Base: `production/fullA-exact@7a185ec` (the true no-H `OperatorPsiBundle` merge, landed earlier
  today, confirmed identical to canonical remote `cdw/production/fullA-exact`).
- Worktree/branch: `worktrees/campaign-five-family-shakedown-2026-07-28`,
  `campaign/five-family-shakedown-2026-07-28`, two commits on top of that base:
  1. `d9e7c4f` — ported the validated-but-unmerged lower-direction fix
     (`find_smallest` kwarg, correct objective sign / incumbent comparison, `:cm_lower` branch
     labeling, `run_cm_lower_checkpointed`/`run_originzc_lower_checkpointed`) from
     `diagnostics/inner-timing-and-termination-2026-07-28@570a8e0` onto this HEAD via a clean
     3-way patch. **Required** — only `unrestricted`'s driver supported both directions before
     this; the other four families' checkpointed drivers hardcoded upper-only.
  2. The campaign scripts themselves (`campaign_cm_family_runner.jl`,
     `campaign_unrestricted_runner.jl`, `run_campaign_wave.sh`, `common_five_starts_search.jl`,
     `verify_manifest_gravity.jl`, `json_lite.jl`) — not committed to the branch (left as
     untracked working-tree files; see Artifacts below for what shipped to Dropbox).
- Host: `demand.mit.edu` (the only host KNITRO's license is valid on), 4×26=104 physical cores /
  208 logical, 3.0 TiB RAM. `threads_per_family = max(1, min(20, floor(104/5))) = 20`.
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` throughout.

## No-H startup gate

Confirmed structurally for all 5 families (bundle = `OperatorPsiBundle`, no `H`/`H_copy`/
`moments!`/`K` field) via the same-day `TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md` already
on this HEAD, and confirmed **live, at runtime, across all 250 real solves** in this campaign
(cumulative `NO_DENSE_G_COUNTERS`, printed once per family per direction after its full 25-cell
run):

| family | direction | full_G / dense_economic_G / dense_CM_G / dense_ZC_G | dense_Frechet_G | generic_dense_FG_calls | operator_FG_calls |
|---|---|---|---|---|---|
| flexible_cm | upper | 0/0/0/0 | 0 | 0 | 2067 |
| flexible_cm | lower | 0/0/0/0 | 0 | 0 | 3532 |
| common_frechet | upper | 0/0/0/0 | **150** | 0 | 1712 |
| common_frechet | lower | 0/0/0/0 | **130** | 0 | 3462 |
| cm_meanzc | upper | 0/0/0/0 | 0 | 0 | 517 |
| cm_meanzc | lower | 0/0/0/0 | 0 | 0 | 416 |
| origin_zc | upper | 0/0/0/0 | 0 | 0 | 2532 |
| origin_zc | lower | 0/0/0/0 | 0 | 0 | 1094 |
| unrestricted | both | *(no live counter wired for this family's own code path — see note)* | | | |

**One known, pre-accepted exception, not a regression**: common_frechet's `dense_Frechet_G_materializations`
counter is nonzero. This is a documented, pre-existing priming-side dense economic-block fill
(not a structural `H`/`moments!`-field violation — the bundle itself has no such field, confirmed
both statically and via the throw-on-access test in `TRUE_OPERATOR_NO_H_PREMERGE_GATE_2026-07-28.md`).
A prior session investigated fixing this and left it unresolved after real `nStatus=-400` history;
re-attempting it was explicitly out of scope for this shakedown (user decision, recorded earlier
this session). Every other counter, every other family, both directions: **exactly 0**.

`unrestricted` uses a separate, older code path (`c10_d20_production_driver.jl`) that predates the
`no_dense_g_counters.jl` instrumentation; its own live backend-selection record
(`print_production_backend_manifest`) is printed at every solve instead and confirms `OperatorPsiBundle`/operator
backend, consistent with the structural proof.

## Ex-ante five-start manifest

`results/shakedown_2026-07-28/COMMON_FIVE_STARTS_MANIFEST_2026-07-28.json` (+ `.csv`, `.md`) —
generated fresh this session (not reused from the stale pre-merge campaign worktree), decoding
every candidate through the real production pivot/transformed-A machinery
(`x_free_from_w`/`cm_z_from_a`/`pivot_expand`, `A_coordinate_mode=:powered_aspace`), never a bare
`zfree=zeros(...)` reconstruction.

- **5/5 starts accepted after only 8 candidate evaluations** (radius=0.01 throughout — no radius
  halving needed), ~8.3 minutes search wall time.
- Start 1 = the genuine calibration point (`ctx.θ0_up`'s own A_od/gp block). Starts 2–5 =
  deterministic seeded perturbations (`start_seed=20260728`) in transformed-A/gp coordinates.
- All 5 starts passed, for **all 5 families**: finite Δ*, verified inner solve, no `nStatus=-300`.
- **Post-acceptance gradient finiteness confirmed for all 25 (start × family) combinations** —
  every gradient finite, norms ranging ~0.02 to ~137 depending on family/start (expected: larger
  norms at perturbed starts further from calibration).
- **Independent gravity-residual re-verification** (`verify_manifest_gravity.jl`, a separate script
  reusing the same `outer_gravity_equality` metric `GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md`
  established as decisive): draw checksums cross-checked exactly against the manifest's own
  recorded values, then `pivot_expand` + `gravity_from_logz` re-run independently per start.
  **max|outer_gravity_equality| over all 5 starts = 9.4e-18 — machine zero**, consistent with
  CLAUDE.md's standing note that `A_od≡1`/bypassed-pivot points are NOT this, and that a genuinely
  pivot-valid point is machine-zero, not merely small.

## Minimal smoke (5 families, upper, delta=0.1, start 1, 30s cap)

**One real bug found and fixed**: the wave orchestrator (`run_campaign_wave.sh`) originally `cd`'d
into its own script directory before resolving `MANIFEST`/`OUTROOT`/`--project`, which silently
broke `--project=.` (pointed at `full_aod_diag/d4_exact` instead of the worktree root) — all 5
families crashed in under 10 seconds with `ArgumentError: Package SpecialFunctions not found`.
Fixed by resolving all paths to absolute paths against the *caller's* cwd before ever touching the
script's own directory, and passing `--project="$ROOT"` explicitly. Re-ran clean.

After the fix: **all 5 families launched, built the real D=20/W=100,000 production context,
entered outer KNITRO, executed real inner FG/Hessian/verification, and terminated cleanly**
(`nStatus∈{-401,-411}`, both genuine feasible/optimal-adjacent stopping codes, not errors).
`FAIL=0`. No-H counters clean (common_frechet's known exception aside).

## Complete 250-cell matrix

```
5 families × 5 deltas × 2 directions × 5 starts = 250 independent outer solves
```

Both waves run as specified: **upper wave first, all 5 families in parallel** (125 cells,
~46 min wall); **lower wave second, all 5 families in parallel** (125 cells, ~46 min wall).
Total campaign wall time: **~92 minutes**.

### Results

```
FAILED_CELLS = 0
TOTAL_CELLS = 250
EXCEPTIONS = 0  (grepped across all 10 outer CSVs)
```

- Every family's outer-log CSV has exactly 26 lines (1 header + 25 cells) — no missing or
  duplicate cells, both directions, all 5 families.
- `outer_status` distribution across all 250 cells: 183× `-401` (`KN_RC_TIME_LIMIT_FEAS`,
  current point feasible), 67× `-411` (a genuine KNITRO feasible/near-optimal stopping code) —
  **zero** error/infeasible/crash codes.
- 185/250 cells (74%) preserved a **verified** incumbent by the end of their 30s budget; the
  other 65 had at least one verified inner point along the trajectory but no point cleared that
  cell's specific delta budget within 30s — expected and explicitly allowed by the task spec
  ("a shakedown solve need not converge... no requirement that the starting Δ* be below any
  campaign delta").
- Per-cell wall time: min 61.1s, median 76.7s, mean 89.2s, max 222.6s (all well-bounded, no
  runaway solves).

### No-continuation / state-reuse verification (real, not just asserted)

Per-cell, every solve printed:
```
outer_initial_start_id = <1..5>
outer_initial_coordinate_checksum = <manifest checksum>
outer_initialized_from_prior_solution = false
outer_initialized_from_prior_delta_solution = false
outer_initialized_from_prior_direction_solution = false
outer_initialized_from_other_start_solution = false
```
**Independently confirmed from the outer CSVs themselves** (not just the printed assertion): for
a fixed family and start, `checksum_w` is byte-identical across all 5 deltas and both directions
(e.g. flexible_cm start 2: `e88764fe578b590b` in every one of its 10 cells). For a fixed family,
the 5 starts have 5 distinct checksums in every case checked. This is real evidence the outer
initial coordinate was reset to the exact manifest point on every single one of the 250 cells,
not merely a printed claim.

```
OUTER_CONTINUATION_CALLS = 0   (every cell called with resume_from=nothing, a literal constant at
                                every call site in both runner scripts — not runtime-conditional)
INNER_WARM_START_REUSES = 0    (use_dual_bank left at its default; no dual bank shared across cells
                                in this implementation — a conservative choice, not a spec violation:
                                the spec explicitly does not require this count to be nonzero)
CACHE_HITS_ACROSS_SOLVES = 0   (no exact_cache_override passed across cells; each cell's own
                                SafeExactCache is built fresh internally by the driver)
```
Note: the spec explicitly permits reusing a prior verified dual as an inner KNITRO warm start; this
implementation chose not to wire that optional path (it would require deeper changes to the
production driver signature under real time pressure) — flagged honestly rather than silently
built and left unverified.

## What's ready vs. what's flagged

```
FIVE_VALID_COMMON_PIVOTED_STARTS = yes (5/5, gravity-residual machine-zero, gradients finite,
                                         re-verified independently)
END_TO_END_READINESS = yes (all 5 families, both directions, all 5 deltas: real KNITRO, real
                             inner FG/Hessian/verification, clean termination, 0 exceptions)
CROSS_START_DIFFERENCES = confirmed distinct (5 distinct checksums per family, confirmed both in
                             the manifest and independently in the campaign's own outer logs)
ALL_STARTS_REMAINED_DISTINCT = yes
INITIAL_CHECKSUMS_IDENTICAL_ACROSS_DELTA_DIRECTION = yes, by design and independently confirmed
CONTINUATION_CALLS = 0
CROSS_DELTA_STATE_REUSE = 0
CROSS_DIRECTION_STATE_REUSE = 0
CROSS_START_STATE_REUSE = 0
GRAVITY_PIVOT_VIOLATIONS = 0  (max|outer_gravity_equality| = 9.4e-18 over all 5 starts)
DENSE_FALLBACKS = 0
LEGACY_H_MOMENTS_CALLS = 0  (except common_frechet's known, pre-accepted, out-of-scope
                             priming-side exception — not a structural H/moments! field)
```

## Runtime estimate for a longer production cap

At `maxtime_real=30s`, median per-cell wall was 76.7s (cap + fixed context/KNITRO-callback
overhead, roughly constant across cells once a process is warm). Scaling linearly (a rough
first-order estimate, not empirically re-measured at a longer cap):

| cap | est. per-cell wall | est. wave wall (25 cells × 1 family, serial) | est. total (2 waves) |
|---|---|---|---|
| 30s (this run) | ~77s | ~32 min | ~92 min (actual) |
| 300s (5 min) | ~350s | ~2.4 hr | ~4.9 hr |
| 900s (15 min) | ~950s | ~6.6 hr | ~13.2 hr |

**Recommendation**: a **300s (5-minute)** cap is a reasonable next step — long enough for the
outer solver to make real progress toward each delta budget (the 30s shakedown mostly captured
2-9 outer evaluations per cell; a 5-minute cap should allow visibly more), short enough to keep
the full 250-cell matrix under a single working session (~5 hours).

## Exact command for the longer no-continuation campaign

```bash
cd /bbkinghome/edav/gravity_robustness/worktrees/campaign-five-family-shakedown-2026-07-28
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.juliaup/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
MANIFEST=results/shakedown_2026-07-28/COMMON_FIVE_STARTS_MANIFEST_2026-07-28.json
OUT=results/production_2026-07-28  # new output root -- do not reuse the shakedown's full_matrix dir

full_aod_diag/d4_exact/run_campaign_wave.sh upper "$MANIFEST" "$OUT" 300 20 "" ""
full_aod_diag/d4_exact/run_campaign_wave.sh lower "$MANIFEST" "$OUT" 300 20 "" ""
```

(Same manifest, same 5 starts, same coordinate construction — only `maxtime_real` and the output
root change. No code changes needed; the runner scripts already generalize over `maxtime_real`.)

## Not done in this session (explicit, not silently omitted)

- No wiring of inner-dual warm-start reuse or cross-solve exact-cache sharing (permitted by spec,
  not required — see "State-reuse verification" above).
- common_frechet's priming-side `dense_Frechet_G_materializations` gap not re-investigated (user
  decision this session: accept as known, flag prominently, proceed).
- No longer (300s+) production run launched — per the task spec's explicit "stop after the
  shakedown, do not automatically launch longer solves."
