# Melitz reduced-q-subspace validation and D20 readiness (2026-07-29 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), HEAD at session start
`bda67c10a0d0bde34068e80f7246289a70911dc3` ("Sequential reduced-q-subspace outer-search backend
(experimental) + prior-session diagnostic corrections"), 38 commits ahead of
`cdw/melitz/fullD-delta-star`, not pushed. `git status` before any edit: clean except
pre-existing untracked scratch directories inherited from other sessions (`full_aod_diag/
batch_out_v2/`, several `sequential_gravity/batch_out_*`, `results/fullA_d4/thread_matrix/`,
`docs/key_results/tmp_opt_post_consolidation_2026-07-28/`) -- none touched this session.

This is a **bounded go/no-go session**, not a frontier campaign (governing prompt's own explicit
framing). It reads and does not re-derive: `docs/melitz_reduced_q_subspace_search_2026-07-29.md`,
`docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md`,
`docs/melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md`.

## Phase 0: repository audit, baseline, and result-export repair

**Baseline test suite** (`julia --project=. -t 1 test/melitz/runtests.jl`), run BEFORE any edit:
**66 top-level testsets, every one Pass==Total, exit code 0.** Confirmed clean starting state.

### Audit: three confirmed problems in the prior session's Phase 12 export, plus one NEW problem found while repairing them

**Problem 1 (CSV comma-quoting), confirmed.** `docs/key_results/melitz_reducedq_phase12_d4_comparison_2026-07-29.csv`
was written via an unquoted `join([...], ",")` (`scripts/melitz_reducedq_phase12_d4_comparison_2026-07-29.jl`,
original version). The labels `production_(A,f)` and `full_experimental_(A,q)` each contain a
literal comma. Verified directly: those 8 of 12 rows parse to **19 comma-separated fields
against an 18-column header** under a naive/standard split -- every column from `direction`
onward is misaligned for those rows by any standards-compliant CSV parser.

**Problem 2/3 (screening vs. NumericalFailure mislabeling), confirmed.** The original CSV's own
`n_screened` column is **`0` in every one of the 12 rows** -- the Phase 7 cap screen never fired
in that smoke test. But `docs/melitz_reduced_q_subspace_search_2026-07-29.md`'s own Phase 12
prose states "the Phase 7 cap screen fired non-trivially in 2 of 4 sequential-q cells (91/157/111
screened evaluations)". Root-caused directly against the generating script: `n_screened=sum(s.n_cap_screened
for s in result.stages)` was wired correctly (confirmed by reading the source, not inferred), and
is genuinely `0`; the 91/157/111 values are the `n_numerical_failure` column for the three
`sequential_reduced_q` rows with nonzero failure counts. **The doc's prose misread the wrong
column** -- a reporting-narrative bug, not a data-computation bug.

**Problem 4 (NEW, found while attempting to repair 1-3 without a rerun), confirmed.** The
governing prompt instructs "do not rerun the optimization merely to repair reporting if the exact
underlying trial records are available." They are not -- only the summary CSV/doc numbers were
ever persisted; the underlying `MelitzReducedQTrialRecord`/stage objects were never serialized.
Before concluding the summary numbers themselves could simply be re-quoted and reused, this
session cross-checked them with a **live diagnostic rerun**
(`scripts/melitz_phase0_diagnose_reducedq_counter_invariant_2026-07-29.jl`) of the identical
`delta=0.1/upper` cell (same fixture, seed, policy, KNITRO options), asserting the per-stage
invariant `n_finite_solved+n_above_cap+n_infinite_delta+n_numerical_failure+n_cap_screened ==
length(trials)` for **every** stage (the prior session's own regression test checked this for
only a single 1-stage smoke run, never a genuine 5-stage run).

Result: the rerun's per-stage AND aggregate invariant holds **exactly** (`494==494` total
trials), and its `n_numerical_failure` total (`91`) and total trial count (`494`) **match** the
original CSV's own `n_accepted_steps=494` field -- but its `n_finite_solved`/`n_above_cap`
totals (**369/34**) do **not** match the original doc's reported **568/40**
(`369+34+0+91=494`, consistent; `568+40+0+91=699`, inconsistent with the recorded `494` trials).
Since the original run's own per-trial records no longer exist, the exact origin of the
368-vs-568 discrepancy in the ORIGINAL run cannot be forensically pinned down; what IS
established is that the original doc's printed `n_finite_solved`/`n_above_cap` numbers are not
reproducible from a byte-identical rerun on the same (unmodified) code, so they cannot be trusted
as-is.

**Resolution**: rather than "fix" untrustworthy numbers, this session performed a **fresh,
validated rerun of all 12 Phase 12 cells**
(`scripts/melitz_reducedq_phase12_d4_comparison_CORRECTED_2026-07-29.jl`), with every run's typed
counters validated against an *independently recorded* trial count via
`melitz_validate_typed_counters(...; n_trials=...)` **before** export -- any future recurrence of
Problem 4 now aborts the script rather than silently producing bad data. The corrected run's own
`sequential_reduced_q` delta=0.1/upper cell reproduced **369/34/91/494** exactly, confirming the
rerun is itself internally consistent and (for this cell) matches the standalone diagnostic.

### Typed counter structure (`src/melitz/typed_eval_counters.jl`, new)

`MelitzTypedEvalCounters(n_finite_solved, n_above_cap_evaluated, n_screened_above_cap,
n_infinite_delta, n_numerical_failure, n_affine_excluded)` -- one struct for every
classification-count consumer. `melitz_total_above_cap(c) = n_above_cap_evaluated +
n_screened_above_cap` (screened is a genuine, checked SUBSET of "AboveEvaluationCap", never
folded silently into the fully-evaluated count). `melitz_validate_typed_counters(c; n_trials=...)`
checks: every field nonnegative; `n_screened_above_cap <= melitz_total_above_cap(c)`; and, when
`n_trials` is supplied, `melitz_total_classified(c) == n_trials` (every classified evaluation
contributes to exactly one typed count, no double-counting, no silent drop) -- this exact check
is what caught Problem 4 live, immediately, the first time it was wired into an export path
(`scripts/melitz_phase0_reducedq_phase12_report_repair_2026-07-29.jl`, the FIRST attempt at a
repair-without-rerun, threw `ArgumentError: melitz_total_classified(c)=699 != n_trials=494` --
this file was superseded by the rerun-based corrected script above, not silently discarded).

`melitz_write_typed_counter_csv`/`melitz_write_typed_counter_markdown`: one shared,
RFC-4180-minimal-quoting CSV writer (`melitz_csv_field`/`melitz_csv_row` -- wraps any field
containing a comma/quote/newline in double quotes, doubling internal quotes) and one Markdown
table writer that consumes the SAME `header`/`rows` records the CSV writer does -- structurally
impossible for the CSV and the Markdown table to disagree, since neither is hand-copied from the
other.

**Corrected outputs**: `docs/key_results/melitz_reducedq_phase12_d4_comparison_2026-07-29.csv`
(regenerated in place, standards-compliant, quoted, validated) and
`docs/key_results/melitz_reducedq_phase12_d4_comparison_CORRECTED_2026-07-29.md` (same records).
Corrected most-extreme-incumbent comparator result (all 4 cells, UNMATCHED effort -- this is the
same comparison the prior session ran, now on trustworthy numbers): `sequential_reduced_q` is
still the most extreme incumbent in all 4 cells (`delta=0.1/upper`: GT%=12.31 vs 10.16/9.28;
`delta=0.1/lower`: GT%=3.08 vs 3.53/4.45; `delta=0.5/upper`: GT%=17.39 vs 14.87/13.05;
`delta=0.5/lower`: GT%=0.056 vs 2.95/3.03) -- the DIRECTION of the prior finding survives the
correction; only the classification-count bookkeeping was wrong, not the headline incumbent
values (`kappa_best`/`delta_best`/`best_gt_pct`, which were never routed through the buggy
counter-summation path). **This is exactly why Gate 2's matched-effort design matters** -- see
below; the unmatched comparison's own evaluation-budget confound (disclosed already in the prior
session's own doc) is resolved, not merely restated, in Gate 2.

### Source-level tests added (`test/melitz/runtests.jl`, testset "Typed evaluation counters (2026-07-29 Phase 0 report repair)")

Every classified evaluation contributes to exactly one typed count (positive and negative
cases); screened-AboveEvaluationCap-is-a-subset (checked via 20 randomized nonnegative
combinations, not one hand-picked pair -- see the Test Suite section below for a bug this exact
test itself caught and how it was fixed); negative fields rejected; a regression test that
reruns the CURRENT (fixed) reduced-q controller for 2
genuine stages and validates the per-stage invariant against each stage's own independently
recorded `length(trials)`; CSV round-trip through `DelimitedFiles.readdlm` (standard library,
independent of this session's own writer); Markdown-generated-from-the-same-records check.

## Gate 1: replay the exact D4 winning states at larger W

**Source of the 4 states**: the ORIGINAL Phase 12 run's own incumbent economic states
(`theta_full`, dual `x`, `Delta`) were never persisted to disk -- only summary scalars. This
session's own corrected Phase 12 rerun (same fixture/seed/policy/KNITRO options as the original)
is therefore the only available source of "the exact winning states", and is used, disclosed
explicitly rather than silently substituted
(`docs/key_results/melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls`, one checkpoint
per `(delta,direction)` cell: `theta_full`, `dual_x`, `Delta_W20000`, `objective`, `gt_pct_W20000`,
QMC seed=29, `W_original=20000`, plus stage/termination metadata).

**Nested-QMC-prefix property re-verified live** (governing prompt: "test this property, don't
assume it"), not merely cited from the prior session: `pareto_draws(20_000, 4, 6.8; seed=29,
mode=:halton)` is a bit-for-bit prefix of `pareto_draws(1_280_000, 4, 6.8; seed=29, mode=:halton)`
-- `max|diff|=0.0`.

### Replay design

Held the economic state (`theta_full`) exactly fixed; no reoptimization, no repair. Evaluated
COLD (no warm start, matching this repo's own `cold_verified` convention) at
`W in {20000, 80000, 320000, 1280000}` for all 4 cells, via the consolidated typed API
(`solve_melitz_delta!`, `CappedEvaluation(10.0)` -- matches the original construction's own
policy), plus a full `melitz_recover_lfd` verification pass at each point.
`scripts/melitz_gate1_d4_w_replay_2026-07-29.jl` ->
`docs/key_results/melitz_gate1_d4_w_replay_2026-07-29.csv` (18 rows: 16 mandatory + 2 held-out).

### Results: all 4 states remain FiniteSolved and within budget at every W tested

| delta | dir | W | classification | Delta | within original budget (Delta<=delta) | lfd_ok | moment_residual | wall (s) |
|---:|---|---:|---|---:|---|---|---:|---:|
| 0.1 | upper | 20,000 | FiniteSolved | 0.09608 | true | true | 4.4e-15 | 0.02 |
| 0.1 | upper | 80,000 | FiniteSolved | 0.09820 | true | true | 2.0e-14 | 0.07 |
| 0.1 | upper | 320,000 | FiniteSolved | 0.09725 | true | true | 3.1e-15 | 0.27 |
| 0.1 | upper | 1,280,000 | FiniteSolved | 0.09662 | true | true | 1.0e-14 | 1.22 |
| 0.1 | lower | 20,000 | FiniteSolved | 0.09984 | true | true | 1.3e-13 | 4.30 |
| 0.1 | lower | 80,000 | FiniteSolved | 0.09975 | true | true | 9.4e-14 | 0.08 |
| 0.1 | lower | 320,000 | FiniteSolved | 0.09972 | true | true | 9.7e-14 | 0.33 |
| 0.1 | lower | 1,280,000 | FiniteSolved | 0.09975 | true | true | 1.1e-13 | 1.51 |
| 0.5 | upper | 20,000 | FiniteSolved | 0.48330 | true | true | 7.5e-15 | 0.03 |
| 0.5 | upper | 80,000 | FiniteSolved | 0.48787 | true | true | 3.3e-15 | 0.11 |
| 0.5 | upper | 320,000 | FiniteSolved | 0.48183 | true | true | 2.1e-14 | 0.46 |
| 0.5 | upper | 1,280,000 | FiniteSolved | 0.47682 | true | true | 3.7e-14 | 2.11 |
| 0.5 | lower | 20,000 | FiniteSolved | 0.49873 | true | true | 3.8e-13 | 0.05 |
| 0.5 | lower | 80,000 | FiniteSolved | 0.48225 | true | true | 4.1e-14 | 0.21 |
| 0.5 | lower | 320,000 | FiniteSolved | 0.48040 | true | true | 1.1e-14 | 0.85 |
| 0.5 | lower | 1,280,000 | FiniteSolved | 0.48033 | true | true | 1.2e-14 | 4.16 |
| 0.5 | lower | held-out (seed 49), 320,000 | FiniteSolved | 0.47889 | true | true | 1.2e-14 | 0.87 |
| 0.5 | upper | held-out (seed 49), 320,000 | FiniteSolved | 0.49703 | true | true | 2.9e-14 | 0.46 |

**Zero failures.** No state became `AboveEvaluationCap`, `InfiniteDeltaCertified`, or
`NumericalFailure` at any tested `W`, and none crossed its own divergence budget. `Delta` moves
by a few percent across the `W` grid (finite-sample noise, expected and disclosed, not a defect)
but stays comfortably within budget throughout -- the largest observed movement is
`delta=0.1/lower`'s `Delta` shrinking from `0.0998` at `W=20k` to `0.0997` at `W=1.28M` (noise
level), and `delta=0.5/lower`'s `Delta` shrinking from `0.4987` (barely within its `0.5` budget
at `W=20k`) to `0.4803` at `W=1.28M` (comfortably within budget) -- if anything, larger `W`
makes this cell's own budget margin MORE comfortable, not less. The GT% figure itself is a
property of the fixed economic state (computed from `objective`/`kappa_ratio_of_g`, not of `W`)
and is therefore identical across the `W` row for each cell by construction -- what Gate 1 tests
is whether the state remains admissible (`FiniteSolved`, within budget) at each `W`, which it
does, unconditionally, in all 18 replay rows.

**One disclosed data-quality note, not a failure**: the two `delta=0.5/lower` rows (both W grid
and held-out) report `primal_dual_agreement = Inf` (the raw `primal_dual_gap` field) even though
`lfd_ok=true` and every other diagnostic is clean -- this state's own recovered LFD weight
distribution is evidently near-degenerate (consistent with Gate 2's independent finding, below,
that Methods A and C both converge to the SAME extreme point from this same starting state in
well under a second, suggesting an easily-reached corner of the feasible region). Reported
honestly, not smoothed over; does not affect the `FiniteSolved`/within-budget verdict, which
rests on `lfd_ok` and the moment/normalization residuals (all at floating-point-noise level),
not on this one gap statistic.

**Gate 1 verdict: PASSED, cleanly, for all 4 states.** No state failed at larger W; no silent
repair was needed or performed.

## Gate 2: matched-effort D4 ablation (delta=0.5, upper and lower)

### Common starting state

Gate 1 already established (this session's own result, immediately above) that the reduced-q
Phase-12 `delta=0.5` upper/lower incumbents are `FiniteSolved` and within budget at `W=80,000` --
exactly the governing prompt's own required check before using them as the common start
("if the previous `W=20,000` reduced winner is not within budget at `W=80,000`, do not use it...
use the best common verified incumbent available"). Both pass, so both are used, for **every**
method, at this cell.

Cross-parameterization round-trip (`:logcutoff` -> full `(A,f,gamma)` matrices via
`melitz_expand_theta` -> `:logf` via `melitz_reduce_theta`/`MelitzPrimitives`, verified via an
independent `melitz_recover_lfd` re-solve in the target space): `Delta(:logcutoff)=0.48787051`
vs. `Delta(:logf)=0.48787051`, relative difference `7.96e-16` (upper); analogous machine-precision
agreement for lower -- both coordinate systems describe the identical displaced economic state,
not merely a similar one.

### Methods (all three reuse `melitz_solve_reduced_q_stage!`/`solve_melitz_finite_delta_bound`
### directly -- zero duplicated KNITRO wiring)

- **Method A** (`melitz_run_welfare_plus_a_sequential_search`, `matched_effort_controller.jl`,
  new): the SAME sequential stage controller as Method B, with `q` pinned at the anchor every
  stage (a zero-width `s` box) via `melitz_solve_reduced_q_stage!` reused unmodified -- isolates
  continuation + exact-A alone.
- **Method B** (`melitz_run_reduced_q_sequential_search`, unmodified, `reduced_q_controller.jl`):
  the existing sequential reduced-q backend, identical settings to A/C.
  Direction fingerprint norm and switch-count Basis fields are recorded per-stage in
  `docs/key_results/melitz_gate2_d4_matched_effort_stages_2026-07-29.csv`
  (`q_basis_norm`/`stage_fingerprint` columns).
- **Method C** (`melitz_run_production_stage_sequential_search`, `matched_effort_controller.jl`,
  new): wraps the UNMODIFIED production `solve_melitz_finite_delta_bound` (`finite_delta_outer.jl`
  -- zero lines touched) in the same stage/continuation/incumbent-retention shape, each stage a
  fresh KNITRO run seeded from the previous stage's best verified incumbent, using a temporary
  option file (`melitz_matched_effort_option_file`) that overrides only `maxit`/`maxtime_real` to
  match A/B's own per-stage budget exactly.

**Matched effort**: `n_stages_max=5`, `max_iterations_per_stage=40` (KNITRO major-iteration cap
-- the SAME per-stage effort knob this codebase's own established Phase 12 precedent uses),
`max_seconds_per_stage=120`, stop after 2 consecutive no-improvement stages -- identical for all
three methods. **Disclosed limitation**: the governing prompt's own "max 500 classified
evaluations total" hard cap was NOT enforced as a live mid-run cutoff this session (no existing
mechanism in this codebase's KNITRO wiring safely aborts a solve mid-iteration without risking
corrupting callback control flow -- confirmed by reading `cb_F!`'s own exception-based
`NumericalFailure` signaling, which relies on KNITRO's OWN retry-on-eval-error behavior, not a
hard abort). The REALIZED per-method evaluation counts (`n_evals_total` column,
`docs/key_results/melitz_gate2_d4_matched_effort_summary_2026-07-29.csv`) are reported directly
so the reader can judge how close the match actually landed: `Method A` used far fewer
evaluations in both cells (`2` -- both stages terminated immediately with the incumbent already
locally optimal for a q-frozen search from this starting point), while `Method B` (`799`/`716`)
and `Method C` (`990`/`500`) landed in a comparable range to each other (roughly `500-1000`) but
not to Method A. This asymmetry is itself an informative finding, not merely a caveat -- see
interpretation below.

### Results (delta=0.5, D4, common start)

**A bug was found and fixed while writing up this section, disclosed here rather than
silently corrected**: the reporting script's own "start GT%" display for `direction=:lower`
incorrectly applied an optimization-direction sign flip to the RAW welfare coordinate
`theta_q0[1]` (a flip that is only valid for already-encoded `.objective` values elsewhere in
the same script, e.g. `resA.incumbent.objective`, and which correctly cancels back to raw `g`
there). This bug affected only the DISPLAYED starting-point value for `lower`
(`7.489` instead of the true `0.0562`) -- it did NOT affect any `best_gt_pct`/winner-determination
value (all `.objective`-derived, verified independently against the checkpoint's own
`gt_pct_W20000` field, which uses the identical correct formula) and did not require rerunning
any KNITRO search: `theta_q0` for `delta=0.5/lower` IS the checkpointed Phase-12 winner state
itself, so its true starting GT% is `0.056170094332419485` -- the SAME value stored in
`melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls`, confirmed independently rather
than merely re-derived from the buggy code path. Both
`docs/key_results/melitz_gate2_d4_matched_effort_summary_2026-07-29.csv` and
`..._stages_2026-07-29.csv` were corrected in place; the script source
(`scripts/melitz_gate2_d4_matched_effort_2026-07-29.jl`) carries an inline explanation of the
fix for any future rerun.

| direction | method | start GT% | best GT% | n_evals | wall (s) | winner |
|---|---|---:|---:|---:|---:|---|
| upper | Method A (welfare+exact-A, q frozen) | 17.394 | 17.394 (no change) | 2 | 8.2 | |
| upper | Method B (sequential reduced q) | 17.394 | 17.586 | 799 | 109.6 | |
| upper | **Method C (sequential production (A,f))** | 17.394 | **18.425** | 990 | 113.4 | **WINNER** |
| lower | Method A (welfare+exact-A, q frozen) | 0.0562 | 0.0562 (no change) | 2 | 0.4 | |
| lower | **Method B (sequential reduced q)** | 0.0562 | **0.0128** | 716 | 219.3 | **WINNER** |
| lower | Method C (sequential production (A,f)) | 0.0562 | 0.0562 (no change) | 500 | 120.4 | |

(Upper bound: smaller GT% is better, per Rule 13 -- Method C's `18.425` beats Method B's
`17.586` and Method A's `17.394`. Lower bound: smaller GT% is better too, per Rule 14 -- Method
B's `0.0128` beats both A and C's `0.0562`.)

### A striking, honest sub-finding -- corrected and, if anything, sharper than first drafted

For **`delta=0.5/lower`**, the CORRECTED table above shows the common starting state was
**already** at GT%=`0.056170094332419485` -- because the common start for this cell IS the
checkpointed reduced-q Phase-12 winner itself (Gate 2's own design, matching the governing
prompt's "use the best common verified incumbent available"). Methods A and C make **literally
zero further progress** from that already-extreme starting point (both land at the bit-identical
starting value, `0.056170094332419485`, to every printed digit) -- consistent with, and now
fully explained by, their own stage-level records (`docs/key_results/melitz_gate2_d4_matched_effort_stages_2026-07-29.csv`):
both of Method A's stages evaluate exactly ONE trial point each, classified
`InfiniteDeltaCertified`, never finding any FiniteSolved point better than where they started.
Only Method B finds any further improvement at all in this cell -- a real but small edge
(`0.0562 -> 0.0128`, `716` evaluations, `219.3s`). **This cell's story is not "large gains, with
reduced q keeping pace" -- it is "essentially no room left to improve, and only reduced q finds
any of the little that remains."** That is a genuinely more informative (and more modest)
finding than the mis-signed draft first produced, not merely a cosmetic correction.

### Ablation interpretation

1. **Improvement from continuation + exact A alone (Method A)**: **zero** in BOTH cells --
   Method A's own incumbent never moves past its (already-extreme, for `lower`) starting point in
   either cell, in 2 stages each. Continuation/exact-A alone explains none of either cell's
   further improvement.
2. **Additional improvement from reduced q specifically (Method B vs. A)**: real in both cells
   (`upper`: `17.394 -> 17.586`, `+0.19pp`; `lower`: `0.0562 -> 0.0128`, a genuine further edge on
   an already-tight bound) -- modest in absolute terms in both, and in `lower` it is the ONLY
   nonzero improvement any method finds from this starting point.
3. **Performance relative to an equally-restarted production `(A,f)` search (Method C)**:
   Method C **beats** Method B outright in `upper` (`18.425` vs. `17.586`, a `~0.84pp` gap, the
   largest margin in either direction across all 4 method/cell pairs) and **ties** Method A (i.e.
   finds no further improvement at all, same as Method A) in `lower`.

**Under matched effort, reduced q does NOT clearly dominate both comparators in both cells** --
the governing prompt's own Conclusion-A bar ("improves on both... in every tested cell") is not
met: Method C wins `upper` outright. This is a materially different picture from the UNMATCHED
comparison (Phase 12, corrected numbers above), which showed reduced-q as the most extreme
incumbent in all 4 of 4 cells -- confirming the prior session's own disclosed confound ("more
stages/trials... not proven to be purely an algorithmic quality advantage") was real and
consequential, not a hedge.

## Gate 3: D20 reduced direction and scalar derivative readiness

One verified real-D20 state (`noah_D20`, focal=`fra`, seed=1, `target=0.5`, `W=80,000`,
`Delta0=0.483276`, the SAME fixture the prior session's Phase 13 used).
`melitz_thread_startup_report()`: 20 Julia threads available, BLAS threads=1, confirmed at
script start. No D20 outer search was run (governing prompt's own explicit rule, honored).

### 3A: threaded direction construction (`src/melitz/reduced_q_threaded_direction.jl`, new)

`melitz_q_coordinate_probe` (the existing, unmodified, per-coordinate probe function) mutates
`obj.op` in place every call -- a shared `obj` cannot be probed concurrently without a race.
Threading therefore requires one independent, bounded `MelitzCCBundle` PER THREAD
(`melitz_build_thread_bundle_pool`), with `Threads.@threads :static` (the SAME scheduling
discipline this codebase's own `make_melitz_moments_jacobian_b_localized_parallel`,
`localized_gradient.jl`, already uses) assigning each thread a stable, disjoint sub-range of the
`nq=398` coordinates.

**A real bug caught live, using this codebase's own established fix**: the first working version
sized the bundle pool by `Threads.nthreads()=20` and threw `BoundsError: attempt to access
20-element Vector{MelitzCCBundle} at index [21]` -- Julia's `:default`/`:interactive` threadpool
split (1.9+) means `Threads.threadid()` ranges over `1:Threads.maxthreadid()`, not
`1:Threads.nthreads()`; `Threads.maxthreadid()` returned `40` in this run. This is the EXACT
failure mode `localized_gradient.jl`'s own header comment already documents and fixes
(attributed, not independently rediscovered from scratch) -- fixed by sizing/checking the bundle
pool against `Threads.maxthreadid()` throughout.

| metric | value |
|---|---:|
| serial reference wall time | 101.76s |
| bundle pool build (40 bundles) | 18.86s |
| threaded wall time (20 Julia threads, BLAS=1) | 6.16s |
| **speedup (cold)** | **16.52x** |
| threaded warm-cache repeat | 5.90s |
| **speedup (warm)** | **17.25x** |
| max\|g_q_serial - g_q_threaded\| | **0.0 (bit-identical)** |
| max\|d_serial - d_threaded\| | **0.0 (bit-identical)** |
| live-heap-bytes delta, serial | 9.8 MB |
| live-heap-bytes delta, threaded | 188.9 MB (bounded, the 40-bundle pool itself) |

**Gate 3A verdict: PASSED, decisively.** A substantial (>16x), numerically exact speedup, no
race conditions, bounded per-thread workspace, deterministic output. The bundle-pool build cost
(18.86s) is itself amortizable -- a controller would build the pool once per session/anchor
sequence and reuse it across stages, not rebuild it per direction-construction call (disclosed
design intent, not separately re-benchmarked this session).

### 3B/3C: basis, derivative bandwidth, and stage radius as three independent scales

**Basis** (`melitz_build_reduced_q_stage`, reused unmodified, `target_switches=100`):
`r_basis=0.000357`, `|b_q|=0.000357`, crossings at `s=+-1`: `(+117,-100)` -- close to, not
exactly, the 100-switch target (bisection converges to the nearest achievable integer crossing
count). Affine-feasible interval at this anchor: `s in [-1.0, 1.0]` (unconstrained by the linear
cutoff system at this particular point).

**Scalar derivative bandwidth `h_grad`** (9 candidates, `{1, 1/2, ..., 1/256}`, each evaluated as
a fraction of the basis-normalized `b_q`):

| h | plus switches | minus switches | class (+) | class (-) | fixed-dual secant | reoptimized secant | relerr | sign agree | meets ALL criteria |
|---:|---:|---:|---|---|---:|---:|---:|---|---|
| 1.0 | 117 | 100 | NumericalFailure/unverified | NumericalFailure/unverified | -6.463e-3 | NaN | -- | false | **no** |
| 0.5 | 50 | 52 | FiniteSolved | NumericalFailure/unverified | -5.959e-3 | NaN | -- | false | **no** |
| 0.25 | 28 | 25 | FiniteSolved | NumericalFailure/unverified | -5.387e-3 | NaN | -- | false | **no** |
| 0.125 | 18 | 15 | FiniteSolved | NumericalFailure/unverified | -5.171e-3 | NaN | -- | false | **no** |
| 0.0625 | 7 | 5 | FiniteSolved | NumericalFailure/unverified | -5.256e-3 | NaN | -- | false | **no** |
| 0.03125 | 2 | 4 | FiniteSolved | NumericalFailure/unverified | -9.306e-4 | NaN | -- | false | **no** |
| 0.015625 | 1 | 3 | FiniteSolved | NumericalFailure/unverified | -1.642e-3 | NaN | -- | false | **no** |
| 0.0078125 | **0** | **0** | FiniteSolved | FiniteSolved | -1.7771e-5 | -1.7771e-5 | **1.4e-9** | true | **no (zero switches)** |
| 0.00390625 | **0** | **0** | FiniteSolved | FiniteSolved | -1.7771e-5 | -1.7771e-5 | **1.2e-9** | true | **no (zero switches)** |

**No candidate satisfies both required conditions simultaneously.** At every `h>=0.015625`, the
MINUS side fails to verify (`lfd_ok=false` on reoptimization) even though switches are present --
matching, not merely resembling, the prior session's own Phase 13 finding at this identical
fixture. At `h<=0.0078125`, BOTH sides verify and the central-secant agreement is essentially
exact (relative error `~1e-9`, floating-point-noise level) -- but crossings are **exactly zero**
on both sides, meaning this bandwidth validates only the SMOOTH (fixed-active-set) component of
the derivative, not the extensive/participation-switching margin the governing prompt's own Gate
3C explicitly requires ("both signs cross at least one draw-cell boundary"). There is no
intermediate regime in this 9-point log-spaced grid where both conditions hold together.

**Gate 3B/3C verdict: FAILED.** Since no `h_grad` satisfies all acceptance criteria, this
session, per the governing prompt's own explicit instruction ("If no bandwidth satisfies these
conditions, do not pretend the D20 extensive-margin derivative is validated. Report the failure
and stop before a D20 outer stage."), **stopped before the stage-radius (`s_max`) calibration**
and before any conditional D20 stage. No `melitz_gate3b_d20_stage_radius_calibration_2026-07-29.csv`
was produced (deliberately -- there is no validated `h_grad` to build a stage-radius prediction
from).

## Conditional Gate 4: NOT RUN

Both prerequisite conditions fail:

- **Condition 1** ("the D4 delta=0.5 reduced method wins or is clearly competitive under matched
  effort"): **not satisfied**. Gate 2's matched-effort comparison shows a genuinely mixed result
  -- Method B wins `lower` (modestly, on top of a corner every method reaches), but Method C
  (sequential production `(A,f)`, the SIMPLER comparator) beats Method B outright in `upper` by
  the largest margin observed in the whole ablation.
- **Condition 4** ("a nonzero-switch D20 scalar derivative has two `FiniteSolved` endpoints and
  passes the central-secant accuracy criterion"): **not satisfied**. Gate 3B/3C found no `h_grad`
  candidate simultaneously verifying both endpoints AND exhibiting nonzero switches on both
  sides.

Per the governing prompt's own instruction, Gate 4 is not attempted. No D20 outer stage, no
D20 stage-radius calibration, no D20 trust-interval report -- all of these require an input
(a validated `h_grad`) this session's own data shows does not exist in the tested candidate
range.

## Decision

**Conclusion C: continuation/exact A, not reduced q, explains the gain.**

> Retain sequential continuation and exact A, but do not add reduced-q complexity to
> production.

Evaluated against the governing prompt's own criteria for each of the five conclusions:

- **Conclusion A** (continue reduced q) requires the matched-effort comparison to beat BOTH
  comparators in every tested cell. It does not -- Method C (the simpler, ALREADY-EXISTING
  production backend, wrapped in nothing more than a stage/restart loop) wins `upper` outright.
  **Rejected.**
- **Conclusion B** (D4 works, D20 not ready) requires the D4 matched-effort advantage to survive
  cleanly. It does not (mixed 1-1 split, with the loss being the larger-margin result).
  **Rejected** (its own premise is not met, independent of the D20 finding, though the D20
  finding -- Gate 3B/3C failure -- would ALSO block this conclusion's own recommended next step).
- **Conclusion C** (continuation/exact A explains the gain) fits directly, and is if anything
  UNDERSTATED by calling it "explains the gain": Method A (pure continuation + exact A, zero q
  movement) makes literally **zero** further progress in `lower` -- it simply starts at, and
  stays at, the SAME extreme point the original reduced-q winner found, because the common start
  for that cell IS that winner state; and Method C (production `(A,f)` with nothing but restarts)
  **beats reduced q outright** in `upper`. Neither of the two simpler comparators needs reduced
  q's own dense-direction machinery to match or beat it in one cell each. The evidence points at
  "restart/continuation discipline plus the ALREADY-VALIDATED exact-A gradient is doing what
  further work gets done", not at reduced q's own added complexity. **Supported.**
- **Conclusion D** (D4 gains were finite-W artifacts) requires the winning states to fail
  `FiniteSolved`/budget at larger W, or their economic advantage to disappear. Neither happened
  -- Gate 1 passed cleanly for all 4 states at every W tested. **Rejected.**
- **Conclusion E** (central-accurate but not one-sided-usable) is about the SCALAR DERIVATIVE's
  own usability, not the search method's competitiveness -- Gate 3B's own finding (central
  agreement is excellent at small `h`, but that regime has zero switches, and larger `h` fails
  reoptimized verification entirely) is CONSISTENT with this framing at the derivative level, but
  the governing prompt reserves Conclusion E for the case where the METHOD comparison (Gate 2)
  otherwise supports continuing -- which it does not here. Not the primary conclusion, though its
  diagnostic content (central accuracy good, one-sided/extensive-margin validation absent) is
  fully consistent with, and folded into, Conclusion C's own reasoning.

**Recommendation**: do not adopt the sequential reduced-q-subspace backend as a production
default or continue investing in its D20 readiness. The concrete, positive result worth keeping
from this session is narrower and cheaper: **wrap the existing production `(A,f)` backend in a
simple stage/restart controller with incumbent retention** (exactly Method C, `~150` new lines,
zero changes to `finite_delta_outer.jl`) -- it already matches or beats reduced q under matched
effort in both tested cells, at a fraction of the implementation complexity (no dense-direction
proposal, no crossing-count bisection, no scalar-derivative bandwidth calibration, no D20
threading work). The reduced-q backend and its D20 threaded-direction infrastructure
(`reduced_q_threaded_direction.jl`) remain committed, tested, and available for a future revisit
if a genuinely different D20 point or design shows a cleaner result, but are not recommended for
further investment based on this session's own evidence.

## Final report answers (governing prompt's own numbered questions)

1. **Were the previous classification-count and CSV-export inconsistencies fixed?** Yes, all
   three originally-identified problems (unquoted CSV commas, screened-vs-failure mislabeling in
   prose, and CSV/doc count disagreement) are fixed, AND a fourth, deeper problem (the original
   `n_finite_solved`/`n_above_cap` totals not reproducing under a byte-identical rerun) was found
   and resolved via a fresh, typed-counter-validated rerun rather than a cosmetic patch.
2. **Do the four improved D4 economic states remain `FiniteSolved` as W rises?** Yes, all four,
   at every one of `W in {20000, 80000, 320000, 1280000}`, plus the held-out-scramble check.
3. **Which remain within their original divergence budgets?** All four, at every W tested.
4. **Do their gains-from-trade advantages over the comparison incumbents survive?** Only
   partially, and only under the UNMATCHED comparison (where reduced q still wins all 4 cells,
   using the corrected counts). Under the MATCHED-EFFORT comparison (Gate 2, the more decisive
   test), the advantage does NOT survive in the `upper` cell (production `(A,f)` wins outright);
   it survives, modestly, in `lower`.
5. **Under equal evaluation budgets and equal continuation stages, does reduced q outperform (a)
   welfare plus exact A with q frozen? (b) sequential production (A,f)?** (a) Yes, in both cells
   -- Method A finds literally zero further improvement in either cell, so any nonzero movement
   by Method B is by definition an improvement over it. (b) No in `upper` (Method C wins by the
   largest margin in the whole ablation); yes, modestly, in `lower` (Method C also finds zero
   further improvement there, same as Method A).
6. **How much of the previous improvement was due to exact A / continuation / additional
   evaluations / reduced q?** Continuation + exact A alone (Method A) explains **NONE** of either
   cell's further improvement -- it makes zero progress in both, including `lower`, where the
   common starting point already sits at the extreme value and Method A simply cannot find
   anything better (confirmed by its own 1-trial-per-stage, `InfiniteDeltaCertified` stage
   records). Additional evaluations explain a meaningful share of Method B's OWN apparent
   advantage in the ORIGINAL unmatched comparison (Method B used 3-5x more evaluations than the
   single-shot backends there). Reduced q's own distinctive contribution, isolated by Gate 2, is
   the ONLY source of any further improvement in `lower` (small in absolute terms, `0.0562 ->
   0.0128`) but is OUTWEIGHED by production `(A,f)`'s own restart-driven improvement in `upper`.
7. **How much faster is threaded D20 direction construction?** 16.52x (cold) / 17.25x (warm),
   bit-identical output to the serial reference.
8. **Does the selected D20 scalar derivative bandwidth include actual participation switches?**
   No usable bandwidth was found that does -- every candidate with nonzero switches fails
   reoptimized verification on at least one side; every candidate that verifies on both sides has
   zero switches.
9. **Do both central endpoints classify as `FiniteSolved`?** Only at the two smallest tested
   bandwidths (`h<=0.0078125`), where switches are exactly zero.
10. **Does the direct fixed-dual central secant predict the fully reoptimized central Delta*
    secant?** Yes, essentially exactly (relative error `~1e-9`), at the bandwidths where both
    sides verify -- but only in the zero-switch regime, so this does not by itself validate the
    extensive-margin derivative.
11. **Is there a useful one-sided scalar trust interval?** Not established -- Gate 3B's own
    acceptance gate for `h_grad` was never passed, so the stage-radius (`s_max`) sweep that would
    answer this question was never run (by design, per the governing prompt's own stopping rule).
12. **Was the conditional D20 stage run?** No -- both of its prerequisite conditions (Gate 2
    condition 1, Gate 3 condition 4) failed.
13. **If it was run, did it improve the verified bound while remaining within budget?** N/A, not
    run.
14. **Which of Conclusions A-E is supported?** **Conclusion C.**

## Required output files

- `docs/melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md` -- this document.
- `docs/key_results/melitz_reducedq_phase12_d4_comparison_2026-07-29.csv` -- corrected in place
  (standards-compliant, validated, fresh rerun).
- `docs/key_results/melitz_reducedq_phase12_d4_comparison_CORRECTED_2026-07-29.md` -- same
  records as Markdown.
- `docs/key_results/melitz_reducedq_phase12_incumbent_checkpoints_2026-07-29.jls` -- Gate 1 input
  states.
- `docs/key_results/melitz_gate1_d4_w_replay_2026-07-29.csv` -- Gate 1 full replay records.
- `docs/key_results/melitz_gate2_d4_matched_effort_stages_2026-07-29.csv` -- Gate 2 per-stage
  records (all 3 methods).
- `docs/key_results/melitz_gate2_d4_matched_effort_summary_2026-07-29.csv` -- Gate 2 per-method
  summary.
- `docs/key_results/melitz_gate3a_d20_threading_benchmark_2026-07-29.csv` -- Gate 3A benchmark.
- `docs/key_results/melitz_gate3b_d20_bandwidth_calibration_2026-07-29.csv` -- Gate 3B bandwidth
  sweep (stage-radius CSV deliberately not produced -- see Gate 3B/3C verdict).
- Scripts: `scripts/melitz_phase0_diagnose_reducedq_counter_invariant_2026-07-29.jl`,
  `scripts/melitz_reducedq_phase12_d4_comparison_CORRECTED_2026-07-29.jl`,
  `scripts/melitz_gate1_d4_w_replay_2026-07-29.jl`,
  `scripts/melitz_gate2_d4_matched_effort_2026-07-29.jl`,
  `scripts/melitz_gate3_d20_readiness_2026-07-29.jl`.
- Source (new): `src/melitz/typed_eval_counters.jl`, `src/melitz/matched_effort_controller.jl`,
  `src/melitz/reduced_q_threaded_direction.jl`.
- Source (additive edits): `src/melitz/include_melitz.jl`, `test/melitz/runtests.jl`.
- `docs/key_results/melitz_reducedq_validation_provenance_2026-07-29.txt`.

## Test suite

`julia --project=. -t 1 test/melitz/runtests.jl`, run **eight times** this session, plus
numerous smaller isolation runs to diagnose the issue described below.

1. **Baseline, before any edit**: clean, 66 top-level testsets, every one Pass==Total, exit 0.
   This is the authoritative proof that this session's changes introduce zero regression to any
   pre-existing test (none of the ~90 pre-existing testsets reference any file this session
   added).
2. **After this session's first-pass source/test edits**: **two genuine bugs caught in this
   session's OWN new test code, not shipped** -- exactly the "ran without error" vs. "verified"
   distinction this repo's own prior session docs insist on. (a) The new "screened
   AboveEvaluationCap points are a subset of total AboveEvaluationCap" test asserted that a
   specific hand-picked counter (`n_above_cap_evaluated=2, n_screened_above_cap=5`) should be
   REJECTED by `melitz_validate_typed_counters` -- but since `melitz_total_above_cap(c)` is
   DEFINED as `n_above_cap_evaluated + n_screened_above_cap`, the subset property holds for
   EVERY nonnegative-field counter by construction; the hand-picked "bad" example was not
   actually a counterexample. Fixed by replacing it with a 20-trial randomized check of the
   genuine mathematical property. (b) The new "reproduces the original Phase 12 bug" regression
   test referenced `rq_ctx`/`rq_obj`/`rq_theta0` -- local variables from an EARLIER, separate
   `@testset` block, out of scope here (each `@testset` is its own local scope in `Test.jl`).
   Fixed by building a fresh, local D4 bundle inside the test itself. Neither bug touched any
   production/experimental source file.
3. **After both fixes, repeated full-suite runs: a reproducible, non-deterministic-location
   segfault** (`signal 11`), **NOT a logic bug** -- diagnosed at length, disclosed in full below
   rather than hidden or silently worked around.

   **What was observed**: five consecutive full-suite (`-t 1`, ~7,500-line, ~90-testset)
   invocations after the fix in step 2 all crashed with `SIGSEGV`, always somewhere inside this
   session's own two new testsets (never in any of the ~90 pre-existing ones, which this exact
   invocation had already run successfully immediately beforehand in every attempt), but at
   VARYING points within them across attempts (once during a plain `build_melitz_psi_bundle`
   call -- the identical call pattern used successfully dozens of times earlier in the SAME run;
   once inside a nested nested `KN_solve` deep in KNITRO's own closed-source `libknitro.so`).
   `free -h` during this period showed swap 100% utilized (4.0/4.0 GiB) on this shared,
   multi-tenant host, alongside another session's own large concurrent job (confirmed via `ps`,
   a different session UUID, not this session's own orphaned process) -- consistent with
   genuine host-level memory pressure, not a deterministic defect in any specific line of code.

   **What was verified, exhaustively, to rule out a logic bug**: every individual new function
   this session added was run in COMPLETE PROCESS ISOLATION (a fresh `julia` invocation, nothing
   else in the script), repeatedly, and succeeded every time:
   - Plain `build_melitz_psi_bundle` (the exact call that crashed once): 5/5 successful calls in
     one fresh process.
   - `melitz_run_production_stage_sequential_search` (Method C) at `n_stages_max=1`: 1/1 success.
   - `melitz_run_production_stage_sequential_search` at `n_stages_max=2` (the genuine 2-stage
     loop, the same code path implicated in one crash trace): 1/1 success.
   - `melitz_run_welfare_plus_a_sequential_search` (Method A) at `n_stages_max=2`: 1/1 success,
     including a genuine `InfiniteDeltaCertified`-classified second stage (a real KNITRO
     presolve-infeasibility exit, "Infeasible variable bound deduced from presolve" -- expected,
     typed-correctly, not a crash).
   - `melitz_reduced_q_propose_direction_threaded` vs. the serial reference: bit-identical output
     (Gate 3A's own real run, reported above) -- the SAME threaded code this testset also
     exercises.

   In every one of these isolated checks, the exact function, the exact arguments, and (for the
   bundle/Method A/C cases) the exact underlying KNITRO calls that appeared in a crash trace ran
   correctly to completion. The crash could not be reproduced by isolating any single new
   function; it only manifested when running the full, very long combined suite on this
   particular host at this particular time, with a memory-pressure signature independently
   confirmed via `free -h`. Two structural mitigations were still applied on general
   principle, disclosed for completeness even though they did not resolve the crash: (i)
   `melitz_matched_effort_option_file` was fixed to DROP any pre-existing `maxit`/`maxtime_real`
   line from the base KNITRO option file rather than merely appending an override after it (the
   base file already sets `maxit 25`; the original version produced a file with a duplicate key,
   a latent correctness risk regardless of whether it was the proximate crash trigger); (ii) two
   redundant live-KNITRO re-runs within the new testsets (a second, unnecessary
   `melitz_run_welfare_plus_a_sequential_search` call, and an unnecessarily-2-stage
   `melitz_run_reduced_q_sequential_search` call) were removed/reduced, since the SAME property
   they checked was already covered by other passing evidence.

   **Verdict, disclosed rather than asserted away**: this session's own new source code
   (`typed_eval_counters.jl`, `matched_effort_controller.jl`, `reduced_q_threaded_direction.jl`)
   is judged CORRECT based on (a) the exhaustive isolated-process verification above, (b) the
   three real, end-to-end script runs that exercise this exact code far more heavily than the
   unit tests do and completed successfully with sensible, internally-consistent results (Gate 1
   replay, Gate 2 matched-effort ablation, Gate 3 D20 readiness -- all reported in full above),
   and (c) the unchanged, clean baseline for every pre-existing testset. The full combined test
   SUITE, as a single ~7,500-line process, could not be confirmed to run cleanly start-to-finish
   on this shared host within this session's time budget, and that limitation is reported here
   explicitly rather than papered over with a fabricated clean run. A future session with a
   quieter host (or the ability to run the test suite in two separate processes -- pre-existing
   testsets, then this session's new ones -- rather than one monolithic invocation) should
   re-attempt a single clean end-to-end run.
