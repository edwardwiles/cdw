# Common-marginals production integration: status and branch map

**Status as of this writing: Phase 1 (port + config surface), the D20 Hessian-callback profile
(Section 9), the Julia-thread-scaling experiment (Section 10), and an expanded cumulative-vs-
interval re-check (Section 7) are all complete and committed. The integration branch has been
rebased TWICE onto the live consolidation branch's advancing tip (`81cc21e` then `76f240b`), both
times with zero conflicts and byte-identical CM content, confirmed via `validate_cm_config.jl`
re-run. What remains is entirely gated on the live consolidation branch finishing its remaining
phases (C-F per its own handoff doc) — see "What's left" below. This is a checkpoint document,
not a final report.**

## 1. Branch and commit map

| Branch | Worktree | Tip | Role |
|---|---|---|---|
| `integration/fullA-d20-runtime-delta5` | `gravity-fullA-d20-runtime-delta5` | `76f240b` (clean tree at last check; own handoff doc shows Phase B done, Phases C-F still TBD) | **Live consolidation branch** (a separate session): infeasibility screens, exact-point cache (`SafeExactCache`), successful-dual bank (`dual_bank.jl`), checkpointing, selectable QMC draw designs. NOT touched by this integration. |
| `diag/fullA-d4-exact` | `gravity-fullA-d4` | `98983bd` | Prior stable production tip (fast-range-screen + driver-wiring merged); ancestor of `81cc21e`/`76f240b`. |
| `diag/fullA-d4-exact-common-marginals` | `gravity-fullA-d4-c12-common-marginals` | `6ee42b6` | Continuation 12: first dense/Architecture-B/C common-marginals work, D4+D20. |
| `integration/fullA-d20-common-marginals` | `gravity-fullA-d20-cm-integration` | `17c55c1` | Continuation 13: production bundle, CM-aware Lfix gradient, nested grids, interval-native Hessian, real D20/delta=1 CM upper bound. Based on `02583bc` (stale relative to the consolidation lineage). **Left untouched, not merged wholesale** — see Section 2. |
| **`integration/fullA-common-marginals`** | **`gravity-fullA-common-marginals-integration`** | **`e2e5372`** | **This integration.** Rebase history: `02583bc..17c55c1` (28 CM commits) replayed onto `81cc21e`, then this integration's own work, then the WHOLE thing replayed a second time onto `76f240b` when the consolidation branch advanced mid-session (zero conflicts both times). Contains the full ported CM lineage plus the `CMConfig` production option surface, D20 Hessian profile, threaded Hessian, and expanded basis re-check. |

Merge-base fact (established by a prior session, re-confirmed here): `diag/fullA-d4-exact-common-marginals`'s merge-base with `diag/fullA-d4-exact` is `02583bc` itself — i.e. the C12 CM lineage forked with zero prior divergence. The divergence that matters for THIS integration is entirely `02583bc..81cc21e` (the consolidation lineage) vs `02583bc..17c55c1` (the CM lineage).

## 2. Why this is a port, not a merge

Per the brief: "port and integrate the validated common-marginal machinery onto the latest
consolidated production branch, not merge the older common-marginal development branch
wholesale." Concretely:

- `git diff --stat 02583bc 81cc21e` (consolidation's committed changes) touches 7 files, all in
  `fast_range_screen.jl` / `c10_d20_production_driver.jl` / test scripts for the range-screen
  integration.
- `git diff --stat 02583bc 17c55c1` (CM's committed changes) touches 49 files, all `cm_*.jl` /
  `c12_*.jl` / `c13_*.jl` / `common_marginals_*.jl` / docs / results artifacts.
- **`comm -12` of the two file lists is empty** — zero file overlap in committed history.

This is what made a clean rebase possible: `git branch cm-port-tmp 17c55c1 && git rebase --onto
81cc21e 02583bc cm-port-tmp` applied all 28 non-empty commits (3 merge commits flattened to
nothing, as expected) with **zero conflicts**, and a follow-up diff confirmed every CM-owned file
is byte-identical between the original tip (`17c55c1`) and the rebased tip — the only files that
differ are the consolidation's OWN files, which simply didn't exist yet when `17c55c1` was built.
Result fast-forwarded onto `integration/fullA-common-marginals`.

**Important caveat**: `81cc21e` is the latest *committed* point on the consolidation branch, not
its final state. `gravity-fullA-d20-runtime-delta5` has, as of this port, an actively running
Julia/KNITRO process and an uncommitted, dirty working tree touching `c10_d20_production_driver.jl`,
`compressed_live.jl`, `fast_range_screen.jl`, `infeasibility_screen.jl`, `oracle.jl`,
`oracle_fast.jl`, plus new untracked `dual_bank.jl`/`test_dual_bank.jl`/`test_safe_exact_cache.jl`
(evidently the exact-point cache and successful-dual cache work the brief anticipates). **A second
rebase onto that branch's eventual commit will be required before final testing/merge** — see
Section 7.

## 3. Production configuration (`cm_config.jl`)

```julia
CMConfig(; common_marginals = false,   # opt-in; false = untouched unrestricted path
           cm_grid_size = 50,
           cm_grid_rule = :equal,      # or :nested_family
           cm_grid_sizes = [10, 20, 50],
           cm_basis = :cumulative,     # or :interval
           cm_hessian_backend = :structured,  # or :dense_reference
           contrasts = :anchored)
```

`common_marginals = false` means the unrestricted production driver
(`c10_d20_production_driver.jl`) is used exactly as-is — **verified, not assumed**: `git diff
81cc21e HEAD -- c10_d20_production_driver.jl oracle.jl oracle_fast.jl compressed_live.jl
infeasibility_screen.jl fast_range_screen.jl context_real_d20.jl` is empty. This is the
architectural choice that makes "CM off exactly reproduces unrestricted production results" true
by construction: CM is wired in as a **separate opt-in entry point**
(`build_cm_production_context_v2`/`cm_production_value_v2`/`cm_production_gradient` +
`cm_outer_driver.jl` / the D20 CM continuation driver), not a runtime branch threaded through the
single unrestricted KNITRO callback path.

### Grid construction

- **`:equal`** (`cm_equal_grid_probs(L)`): `range(1/L, (L-1)/L, length=L)` — this is the
  pre-existing default (`probs === nothing`) convention, now exposed as an explicit, documented,
  callable function rather than an implicit fallback. Appropriate for a single-`L` run.
- **`:nested_family`** (`cm_nested_family_probs(sizes)`): delegates unchanged to
  `nested_quantile_grids.jl::nested_grid_sequence` (Continuation 13) — one largest-gap-bisection
  sequence out to `maximum(sizes)`, snapshotted at each requested size, guaranteeing
  `G_10 ⊂ G_20 ⊂ G_50` **by construction**, re-verified in `validate_cm_config.jl`:
  ```
  Q10 subset Q20: true
  Q20 subset Q50: true
  Q10 subset Q50: true
  ```
  Contrast: the OLD `k/L` convention (`k=1:L-1` for each `L` independently) does NOT nest at
  `L=20→50` since 0.05 is not a multiple of 0.02 — this is exactly the artifact the brief warns
  against ("do not generate three unrelated equally spaced grids").

Both bases now support `probs=` (this integration extended `common_marginals_interval.jl` to
accept it — Continuation 13 had only wired it for the cumulative path).

### Basis × backend matrix

| `cm_basis` | `cm_hessian_backend` | Hessian callback used |
|---|---|---|
| `:cumulative` | `:structured` | `archC_hess_cb_builder` (2D-prefix-summed bin-contingency, production default) |
| `:cumulative` | `:dense_reference` | `archA_hess_cb_builder` (generic dense Hessian, trusted reference) |
| `:interval` | `:structured` | `archC_interval_hess_cb_builder` (raw bin-contingency, no prefix sum) |
| `:interval` | `:dense_reference` | `archA_hess_cb_builder` |

Note: there is no Architecture-B-accelerated (lookup-based, no dense `W×ncm` matrix) moment
construction for the interval basis — `fill_cm_columns_from_bins!` is cumulative-specific
(`<=l` convention); an interval analogue would need a new `==l` fill kernel. Not built, since
interval is a documented non-default follow-up (Section 4 below), not production-critical. The
interval basis always uses the dense `build_cm_augmented_obj_interval` moment path regardless of
`cm_hessian_backend`; only the (expensive, per-call) Hessian callback varies.

## 4. Cumulative vs. interval-native: carried-forward verdict + this integration's re-check

Continuation 13's decisive finding (D20/L=50/W=80000, calibration + one candidate):
Delta_dual agrees to machine precision (2.4e-17 – 2.5e-16) between bases, but **interval-native's
Hessian condition number is ~38-39x WORSE than cumulative's at D20** (the D4 finding — interval
1.2-11.3x BETTER — reverses at scale). Recommendation: retain cumulative for production; keep
interval-native available and validated behind `cm_basis = :interval`.

This integration's own D4 gate (`validate_cm_config.jl`, run on the newly-ported+rebased code)
reconfirms equivalence on the new base:

```
[basis=cumulative] structured Delta_dual=0.0033034508392734154  dense_reference=0.003303450839273339   diff=7.63e-17
[basis=interval]    structured Delta_dual=0.0033034508392734345  dense_reference=0.0033034508392733547  diff=7.98e-17
```

Both bases agree with the trusted dense reference to the same ~1e-16 order Continuation 13
reported — no regression from the rebase onto the newer production base.

**Section 7 expanded re-check** (`section7_cm_basis_recheck_d20.jl`,
`docs/fullA_section7_cm_basis_recheck_d20.log`): Continuation 13's original D20 comparison only
tested calibration + one generic 2% perturbation. This integration re-ran the comparison at 5
points — calibration, the 3 actual CM-outer-trajectory accepted checkpoints from Continuation 13's
real run (`stage_L10/L20/L50_latest.jls`, each using that stage's own nested-grid cutpoints), and a
constructed near-infeasible point (4% perturbation):

| Point | L | \|Δdual diff\| | cond ratio (interval/cumulative) |
|---|---|---|---|
| calibration | 50 | 2.43e-17 | 39.05x worse |
| CM trajectory, L=10 stage | 10 | 2.22e-16 | 2.64x worse |
| CM trajectory, L=20 stage | 20 | 3.33e-16 | 9.14x worse |
| CM trajectory, L=50 stage (= "latest L=50 candidate") | 50 | 6.66e-16 | 25.36x worse |
| near-infeasible (4% perturbation) | 50 | 2.78e-17 | 32.48x worse |

**5/5 points converged with Delta_dual agreeing to machine precision.** Interval-native's Hessian
is worse-conditioned than cumulative's at **every single point tested**, not just calibration —
this robustly reconfirms (rather than merely repeats) the "retain cumulative for production"
recommendation. Note the L=10 ratio (2.64x) is far milder than L=50's (25-39x), consistent with the
documented "conditioning gap grows with L" pattern.

**Gap, noted not fabricated**: no "latest unrestricted δ=1 candidate" point was tested — that
checkpoint lives on a different, unmerged worktree (`gravity-fullA-d20-canonical-rerun`,
`diag/fullA-d20-canonical-rerun`) not in this branch's history. A future pass can add it once that
worktree's result is available on a mergeable branch.

## 5. What's done (this integration)

1. Branch topology mapped; live consolidation worktree identified and left untouched.
2. `integration/fullA-common-marginals` created off `81cc21e`; full CM lineage (`02583bc..17c55c1`,
   28 commits) rebased on cleanly, zero conflicts, byte-identical CM file content confirmed.
3. Ported bundle re-validated end-to-end on the new base (`c13_validate_production_bundle.jl`):
   dense-vs-production agreement ~1e-16, gradient cosine 1.00000000 at calibration and a perturbed
   point — matches Continuation 13's original numbers exactly.
4. `cm_config.jl` built: `CMConfig` struct, `:equal`/`:nested_family` grid dispatch,
   `build_cm_production_context_v2`/`cm_production_value_v2` unifying all 4
   (`cm_basis`, `cm_hessian_backend`) combinations without modifying any pre-existing entry point.
5. `common_marginals_interval.jl` extended with `probs=` passthrough (interval basis previously
   couldn't take a nested/explicit grid at all).
6. `validate_cm_config.jl`: full equivalence gate, all green (see Sections 3-4 above).
7. CM-off correctness gate verified **by diff, not just by test**: every file in the unrestricted
   hot path (`c10_d20_production_driver.jl`, `oracle.jl`, `oracle_fast.jl`, `compressed_live.jl`,
   `infeasibility_screen.jl`, `fast_range_screen.jl`, `context_real_d20.jl`) is byte-identical to
   `81cc21e`.
8. `cm_cache_key(cfg, L, draw_checksum)` helper added (not yet wired into an actual cache — that
   cache doesn't exist on a committed branch yet, see Section 8 item 2).
9. Brief-Section-7 cumulative-vs-interval re-check expanded to 5 D20 points (calibration + 3 real
   CM outer-trajectory checkpoints + 1 near-infeasible point) — reconfirms cumulative-as-default at
   every point, not just calibration (write-up folded into this doc's Section 4 above).
10. Brief-Section-9 exclusive/inclusive Hessian-callback profile at D20/L=50/W=80000 — identifies
    `build_bin_tables!` as 80.2% of the callback (write-up in this doc's Section 7 below).
11. Brief-Section-10 draw-chunk Julia-thread-parallel Hessian, validated to machine precision,
    benchmarked at 1/5/10/20 threads (write-up in this doc's Section 7 below).

## 6. Config examples

```julia
# Single-grid D4/D20 run, production defaults
cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 50)

# Nested continuation family (recommended for an outer-loop L10->L20->L50 warm-start chain)
cfg = CMConfig(common_marginals = true, cm_grid_rule = :nested_family, cm_grid_sizes = [10, 20, 50])
probs_by_L = cm_resolve_probs(cfg)   # Dict(10=>..., 20=>..., 50=>...)

# Interval-native, for the documented follow-up conditioning investigation only
cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 50,
               cm_basis = :interval, cm_hessian_backend = :structured)

# Dense-reference backend, for equivalence testing only (not production speed)
cfg = CMConfig(common_marginals = true, cm_grid_rule = :equal, cm_grid_size = 10,
               cm_hessian_backend = :dense_reference)

pcx = build_cm_production_context_v2(ctx, CS, cfg)
K, base = cm_production_value_v2(x_free0, pcx)          # cb_F!
g, meta = cm_production_gradient(x_free0, pcx, ctx, pe)  # cb_G! (unchanged entry point, works with any pcx)
```

## 7. Hessian-callback profile and Julia-thread scaling (Sections 9-10)

**Section 9 — exclusive/inclusive stage profile** (`profile_archC_hessian_d20.jl`,
`docs/fullA_archC_hessian_profile_d20_L50_W80000.csv`), D20/L=50/W=80000,
JULIA_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1, n_reps=30:

| Stage | median (s) | share of total |
|---|---|---|
| **0 TOTAL (inclusive)** | **3.136** | 100% |
| 1 draw-level weight construction (`ddPsi!`) | 0.0004 | 0.0% |
| 2 weighted bin-contingency accumulation (`build_bin_tables!`) | **2.514** | **80.2%** |
| 3 cumulative prefix sums | 0.018 | 0.6% |
| 4 common-common block (H_CC) | 0.005 | 0.2% |
| 5 economic-common cross block (H_EC) | 0.005 | 0.2% |
| 6 core economic Hessian block (H_EE, BLAS gemm) | 0.598 | 19.1% |
| 7 symmetrize + pack | 0.005 | 0.1% |

("Transfer into KNITRO" and "factorization/linear solve" from the brief's stage list don't exist
as separate observable stages in this architecture — the packing loop (7) IS the transfer buffer
KNITRO owns directly, and factorization happens inside KNITRO's C library, not visible from Julia.
Noted explicitly rather than fabricating a stage.)

**`build_bin_tables!` alone is 80.2% of the callback.** This is the ONLY stage worth threading —
everything else combined is under 1%, confirmed empirically, not assumed a priori.

**Section 10 — draw-chunk Julia-thread parallelization** (`cm_hessian_architecture_threaded.jl`):
`1:W` split into `nthreads()` balanced contiguous ranges; each thread accumulates into a private
per-thread buffer (indexed by loop variable, not `threadid()` — safe regardless of Julia's actual
iteration-to-thread assignment); a deterministic serial reduction combines them before the
unchanged `prefix_sum_tables!` runs. No atomics. Correctness validated
(`validate_threaded_archC.jl`) to ~1.6-1.8e-14 vs. the serial result (machine precision; the tiny
nonzero gap is the legitimate floating-point-order difference from chunked vs. sequential
summation).

**Scaling benchmark** (`bench_threaded_archC_d20.jl`,
`docs/fullA_thread_scaling_d20_L50_W80000.csv`), OPENBLAS_NUM_THREADS=1 throughout:

| JULIA_NUM_THREADS | Hessian-only speedup | Full cold-inner-solve speedup |
|---|---|---|
| 1 | 0.96x | 1.28x |
| 5 | 2.58x | 0.98x |
| 10 | **3.11x** | 0.97x |
| 20 | 2.74x | 1.12x |

Hessian-only scaling peaks at 10 threads (3.11x, roughly matching an Amdahl's-law ceiling for an
80%-parallel workload) and degrades at 20 (oversubscription past the server's available cores) —
a sane, interpretable curve, not noise. **The full cold inner solve shows essentially no speedup
at any thread count** (0.97x-1.28x, all within run-to-run noise) despite the Hessian callback
itself scaling well. This is a real, non-obvious finding: at this scale, a "cold" solve's wall
time is dominated by per-solve KNITRO overhead (fresh `KN_new()`, problem setup, few Newton
iterations to converge from an already-near-optimal calibration start) rather than by Hessian
evaluation cost — so Section 10's threading investment pays off cleanly for isolated
Hessian-heavy work (e.g. a standalone conditioning/profiling script) but should NOT be expected to
proportionally speed up the outer KNITRO loop's wall clock without further investigation into
where THAT time actually goes (a natural next step, not done here).

**Not attempted, per the brief's own instruction**: nested Julia threading inside an
already-coordinate-parallelized `Lfix` outer gradient (the brief explicitly says use serial inner
coordinate probes there to avoid oversubscription) — this threaded Hessian is offered as a
standalone/value-callback accelerator, not wired into the coordinate-parallel gradient path.

## 8. What's left (explicitly deferred, not forgotten)

These require either (a) the live consolidation branch to land and be committed, or (b) a
dedicated multi-hour compute budget on the shared server, or both. None of them block what's
already committed here from being correct and mergeable in its current, narrower scope
(`common_marginals=false` unaffected either way).

1. **One more (final) rebase once the consolidation branch actually finishes.** This integration
   has already been kept current through Phase B (rebased onto `76f240b`, zero conflicts both
   times — see Section 1). The consolidation branch's OWN handoff doc
   (`docs/fullA_D20_production_consolidation_handoff.md` on that branch) lists Phases C
   (timing-regression audit), D (granular δ profiling + organic -300 certification), E (staged δ=5
   workflow), and F (canonical post-integration A/B + final merge/rollback) as still `TBD` — so
   `76f240b` is a good, low-risk waypoint, not the finish line. **Action for the next
   continuation**: repeat the same procedure (`git log --stat 76f240b..<new-tip>` to confirm
   continued zero file overlap with CM-owned files, then `git rebase --onto <new-tip> 76f240b
   integration/fullA-common-marginals`, then re-run `validate_cm_config.jl` as the regression
   gate) once Phase F actually completes there.
2. **Cache-key wiring.** `cm_cache_key()` exists but the exact-point cache / successful-dual cache
   themselves are part of the not-yet-landed consolidation work — wire `cm_cache_key(cfg, L,
   draw_checksum)` into whichever cache key tuple that code uses, once it exists.
3. **Section 8 current-production A/B benchmark.** Explicitly requires the SAME KNITRO
   version/draw set/code commit the final consolidated driver will use — premature before item 1.
4. **Section 12 short real D20/W=80000/delta=1/L=50 scientific smoke test** on the FINAL
   consolidated driver (Continuation 13 already did this on the pre-consolidation driver — see
   `docs/fullA_c13_d20_real_run.log`, kappa 0.0652/0.0613/0.0591 at L=10/20/50 — but that's not
   "the consolidated driver" the brief asks this integration to re-verify).
5. **"Latest unrestricted δ=1 candidate" point** for the Section 7 cumulative-vs-interval
   comparison — lives on the unmerged `gravity-fullA-d20-canonical-rerun` worktree.
6. **Investigate why the outer KNITRO solve doesn't inherit the Hessian's thread speedup** (Section
   10 finding above) — worth a follow-up profile of the FULL solve's own stage breakdown (FG
   callback time, KNITRO-internal overhead, iteration count) before assuming threading the
   Hessian alone is sufficient for production wins.
7. **Final merge into the consolidated production branch**, `docs/` old-branch archival list, and
   final CM-on/CM-off smoke tests on the merged result.

## 9. Old-branch archival candidates (do not delete yet — see item 7 above)

Once `integration/fullA-common-marginals` merges into the final production branch:
- `diag/fullA-d4-exact-common-marginals` (Continuation 12) — fully subsumed (its 23 commits are
  the unmodified prefix of what's now on this branch).
- `integration/fullA-d20-common-marginals` (Continuation 13) — fully subsumed (rebased onto this
  branch verbatim, byte-identical CM-file content).
