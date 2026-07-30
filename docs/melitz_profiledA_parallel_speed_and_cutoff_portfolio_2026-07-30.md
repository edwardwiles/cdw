# Melitz profiled-A parallel speed and cutoff portfolio (2026-07-30)

Governing prompt: optimize and prepare the now-successful profiled-(A) D20 Melitz search
(`docs/melitz_d20_profiled_A_welfare_continuation_2026-07-30.md`) for a practical computational
campaign — remove redundant work, measure thread scaling, determine the best process/thread
allocation under a fixed core budget, implement an adaptive second-start policy, and design a
bounded cutoff-anchor portfolio. **Does not launch the final multi-budget campaign** (explicitly
out of scope). Does not modify the Ricardian implementation.

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`. Session started at HEAD
`b301aa0` (the profiled-A welfare continuation commit); this session's own work is committed
locally on top, **not pushed** (this repo's own standing convention).

**Same-day follow-up (2026-07-30, later)**: the user asked for the Phase 4 focal-link
optimization — originally audited-but-not-implemented below — to actually be implemented,
validated, and wired into production. That work is folded into the Phase 4 section in place
(clearly marked), rather than kept as a separate report, since it directly supersedes that
section's own original "not implemented" verdict.

---

## Phase 0: repository and reproducibility audit

- **Git state at session start**: branch `melitz/fullD-delta-star`, HEAD `b301aa0`, 46 commits
  ahead of `cdw/melitz/fullD-delta-star` (not pushed), a large set of pre-existing untracked
  scratch files/directories from other concurrent sessions (left untouched throughout, per this
  repo's own "don't touch other sessions' state" norm).
- **Tests run in a clean process**: `julia --project=. -t 1 test/melitz/runtests.jl` reproduces
  the **pre-existing, already-disclosed `mul_G!` SIGSEGV** (`moment_operator.jl:281`, triggered
  at `test/melitz/runtests.jl:7395`, exit code 139) — confirmed unrelated to this session's own
  work: the crash occurs while constructing a `MelitzCCBundle` deep in an earlier, unrelated
  testset, far upstream of anywhere this session's own code is reached. This matches the
  identical, already-documented finding of at least two immediately-prior sessions (`melitz_d20_profiled_A_welfare_continuation_2026-07-30.md`,
  `melitz_fixed_q_A_middle_loop_experiment_2026-07-30.md`), which both worked around it via
  dedicated standalone isolated test scripts — the same convention this session's own new tests
  use (`scripts/melitz_adaptive_start_standalone_test_2026-07-30.jl`,
  `scripts/melitz_phase0to8_invariant_tests_2026-07-30.jl`), both passing in full (24/24, 15/15).
- **Reproduced the successful upper profiled-A point** two independent ways:
  1. Deserialized the prior session's own stored state
     (`scripts/melitz_d20_profiledA_continuation_state_2026-07-30.jls`) and confirmed its
     headline numbers match the doc exactly: `GT=7.096890522201971%`,
     `Delta*=0.4990186631205613`, fixed-A/f `Delta=1.0359519723212067`.
  2. **Live cold re-verification against CURRENT code**: rebuilt the real-D20 fixture fresh,
     called `solve_melitz_delta!` on the stored extreme point's own `theta_free_incumbent` —
     **`Delta=0.4990186631` (`FiniteSolved`), matching the stored value to `<1e-9`** — the
     current codebase reproduces the documented result within documented tolerance, not merely
     "close."

## Source map (read before evaluating this report)

| concern | file |
|---|---|
| profiled welfare continuation driver | `scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl` |
| fixed-q middle solve (v1 + repaired v2 + NEW adaptive-start wrapper) | `src/melitz/fixed_q_a_middle_loop.jl` |
| objective/gradient callbacks | `melitz_middle_objective_and_gradient!` / `_cached!` (same file) |
| inner-session construction / typed policy | `src/melitz/inner_session.jl`, `inner_solve_policy.jl` |
| exact A gradient | `src/melitz/exact_a_gradient.jl` |
| fixed-q operator update / focal-link | `src/melitz/moment_operator.jl` (`melitz_update_moment_operator!`, `mul_G!`) |
| cap handling / incumbent retention | `solve_melitz_fixed_q_A_profile_v2` (`fixed_q_a_middle_loop.jl`) |
| both deterministic middle starts | `profile_phi_at_g` (continuation script) — now optionally routed through the new adaptive wrapper |

---

## Phase 1: one inner solve per unique A point

**Already implemented** by the prior addendum session (`solve_melitz_fixed_q_A_profile_v2` /
`melitz_middle_objective_and_gradient_cached!`, both in `fixed_q_a_middle_loop.jl`) — this
session's job was to verify it, not build it. The cache key is the exact `theta_free_middle`
vector (an injective function of `A_free` + fixed `q` + fixed `g`); a second, lightweight
`MelitzMiddleBadPointCache` covers repeat requests at an already-classified
`AboveEvaluationCap`/`InfiniteDeltaCertified` point without a heavy dual/moment-operator
snapshot. `NumericalFailure` is never cached or returned as a value (`DomainError` thrown).

**Fresh confirmation this session** (not merely re-citing the prior session's numbers):

| source | n_fc | n_ga | unique_A | unique_solves | cache_hits | invariant holds |
|---|---:|---:|---:|---:|---:|---|
| standalone D4 test (this session) | 39 | 20 | 39 | 39 | 21 | yes |
| D20 near-boundary point profile (Phase 3, this session) | 113 | 52 | 85 | 85 | 66 | yes |
| D20 anchor cutoff-portfolio point (Phase 7, this session) | — | — | 81 | 81 | — | yes |
| D20 batchA/B replay, all 13 upper-path points (Phase 2, this session) | — | — | — | matches `unique_inner_solves` reported per row | — | yes, every row |

`unique_inner_solves <= unique_A_points` held in **every single run this session performed**
(often with exact equality, i.e. zero redundant solves) — full CSV:
`docs/key_results/melitz_phase1_dedup_invariant_2026-07-30.csv`.

---

## Phase 2: adaptive second-start policy

### Implementation

New code, `src/melitz/fixed_q_a_middle_loop.jl`, "ADDENDUM 2026-07-30 PART 2":
`MelitzAdaptiveStartPolicy` (thresholds) + `melitz_middle_two_start_adaptive!` (decision logic).
Always runs the continuation start first; accepts it alone iff **all** of:

1. `FiniteSolved`;
2. no worse than its own verified input (re-checked, not merely trusted);
3. `nStatus in (0,-100,-101,-103)` — the **same** successful/locally-optimal-equivalent set this
   codebase already uses elsewhere (`inner_screening.jl:859`, `cc_bundle.jl:506` et al.), not an
   invented stricter rule (an earlier `nStatus==0`-only draft was corrected after live testing
   showed it was stricter than the codebase's own convention);
4. material improvement over the raw (unoptimized) continuation start, **or** already close to
   the divergence budget (either condition suffices, per the governing prompt's own "OR"
   wording);
5. not cap-dominated (a configurable fraction of the continuation trial's own classified
   evaluations landing in `AboveEvaluationCap`/`InfiniteDeltaCertified`);
6. no pivot/chamber change detected (an exact, zero-cost same-bin/ordering `sense`-vector
   comparison against the preceding point's own constraint system, when both are supplied).

Falls back to the compensated start on any of: not `FiniteSolved`; worse than input; poor KKT;
cap-dominated; a genuine pivot change; negligible improvement; **or** a caller-supplied
`periodic_safeguard_due=true` (every `K`-th accepted point, `K=4` default) — a **hard guarantee**
checked structurally (verified across several independent random starts,
`melitz_adaptive_start_standalone_test_2026-07-30.jl`). If both starts run, retains the
strictly-better verified `FiniteSolved` result — never worse than either individual v2 call.

### D4 correctness (24/24 assertions, standalone isolated run)

`scripts/melitz_adaptive_start_standalone_test_2026-07-30.jl`. Covers: reproducible decisions
for a fixed input; the deterministic (impossible-to-satisfy-by-construction) negligible-
improvement gate; never-worse-than-input under a stochastic perturbed start; never-worse-than-
the-better-of-both when forced to run both; the periodic-safeguard hard guarantee (4 independent
seeds); the pivot-changed trigger firing on a genuine synthetic sense-vector flip and NOT
spuriously firing when `sys==prev_sys`. **A genuine engineering finding surfaced while building
these tests, disclosed not glossed over**: this codebase's own moment/gradient construction is
threaded (`src/melitz/CLAUDE.md`), so KNITRO's terminal `nStatus` on a run landing near a
convergence-tolerance boundary is **not bit-identical run-to-run** even from an identical
seed/start — floating-point summation order varies with thread scheduling. Tests were written to
assert robust structural invariants (never-worse-than-input; hard guarantees) rather than pin an
exact stochastic `nStatus`/`trigger_reason` outcome.

### D20 validation against the stored upper continuation path (**decisive**)

Replayed all 13 evaluated rows of the stored upper-direction path (`idx=1..13`,
`docs/key_results/melitz_d20_profiledA_continuation_points_2026-07-30.csv`) — reconstructed the
EXACT `(g_target, prev_A_free, prev_q, prev_p_star)` input state at each row by replaying the
original script's own accept-only state-threading rule
(`scripts/melitz_phase2_precompute_states_2026-07-30.jl`), then ran BOTH policies fresh at every
row (`scripts/melitz_phase2_row_worker_2026-07-30.jl`, split across two parallel processes for
wall time):

| | accepted rows (idx 1-6, incl. the decisive `GT=7.0969%`/`Delta*=0.499` boundary) | rejected rows (idx 7-13, over budget) |
|---|---|---|
| rows tested | 6 | 7 |
| `Delta_adapt == Delta_base` exactly | **6 / 6** (`delta_agree_abs=0.0` every row) | **7 / 7** (`delta_agree_abs=0.0` every row) |
| `within_budget` decision matches | **6 / 6** | **7 / 7** |
| **`same_budget_status`** | **13 / 13** | |

**The adaptive policy reproduces the stored always-two-start trajectory's `Delta` and
budget-status decision EXACTLY at all 13 tested points, including the decisive boundary point —
zero disagreement, not merely "within 0.005 percentage points."**

Full data: `docs/key_results/melitz_phase2_adaptive_vs_twostart_batchA_2026-07-30.csv`,
`..._batchB_2026-07-30.csv`.

**Disclosed, not glossed over — work-reduction was NOT realized on this specific trajectory.**
Along this particular replay, the adaptive policy's `cap_dominated`/`poor_kkt` triggers fired on
nearly every accepted row (the D20 continuation start's own exploration legitimately wanders
through `AboveEvaluationCap` territory en route to its eventual incumbent, matching this
project's own prior documented finding that D20-scale middle solves are start-sensitive), forcing
the compensated fallback almost every time — `ran_compensated=true` in 6/6 accepted rows and 7/7
rejected rows. Middle-solve counts and unique-inner-solve counts were therefore **identical**
between the two policies on this trajectory (no wasted work either way, but no savings realized
here specifically). Small wall-time deltas observed (e.g. row idx=1: 69.1s→60.4s, -12.5%) are
attributable to warm-start/dual-bank locality between back-to-back calls in the same process, not
a structural savings mechanism. The mechanism's real savings were demonstrated cleanly on the D4
"already-accept-alone" scenarios and structurally guaranteed (periodic safeguard, deterministic
gate tests) — the D20 upper path simply happened to be a trajectory where the safety triggers are
usually warranted. A looser `cap_dominated_frac` (default `0.5`) or a policy that discounts early
cap excursions before the incumbent is first found would likely realize savings on this same
trajectory — flagged as a concrete, bounded follow-up, not attempted here (re-tuning and
re-running was judged lower priority than completing the mandatory Phase 5/6 work).

---

## Phase 3: fixed-q hot-path profile

Three points profiled (`scripts/melitz_phase3_hotpath_profile_2026-07-30.jl`,
`MELITZ_PROFILE[]=true`, `-t 10`): the anchor (`g0`), an interior continuation point (`idx=3`,
`GT=6.53%`), and the near-boundary point (`idx=6`, `GT=7.10%`, `Delta*=0.499`) — a deterministic
sinusoidal start perturbation was applied to force enough KNITRO iterations for a statistically
meaningful sample (a warm continuation start can converge in as few as 4-6 evaluations, too thin
for percentile timing). The near-boundary point reached **85 unique A evaluations** (well over
the requested 20); the anchor and interior points converged faster (6 each) even under the
forced perturbation — disclosed, not silently padded.

**Full breakdown** (near-boundary point, 85 evaluations, 38.8s total):

| category | count | total_s | pct_of_total |
|---|---:|---:|---:|
| `fc_inner_hess_eval` | 1060 | 13.63 | 35.1% |
| `fc_inner_obj_eval` | 3003 | 7.57 | 19.5% |
| `fc_operator_merge` | 88 | 6.95 | 17.9% |
| `moment_operator_link_update` (subset of the row above — the focal-link update) | 88 | 6.48 | 16.7% |
| `fc_inner_grad_eval` | 1159 | 3.28 | 8.5% |
| `screen_stored_dual_passed` | 86 | 1.77 | 4.6% |
| `fc_inner_dpsi_eval` | 2299 | 0.98 | 2.5% |
| `middle_theta_reconstruction` (**NEW this session**) | 151 | 0.026 | 0.1% |
| `fc_theta_expand` | 88 | 0.016 | 0.0% |
| `middle_cache_lookup` (**NEW this session**) | 302 | 0.0017 | 0.0% |

**New instrumentation added this session**: `:middle_theta_reconstruction` and
`:middle_cache_lookup` (`melitz_middle_objective_and_gradient_cached!`) — separating the
Addendum-D cache-overhead cost from genuine inner-solve cost, which the prior session's own
Section E profile predates. **Both are negligible** (0.1% and 0.0% of total wall time
respectively) — the cache machinery itself is not a hidden cost.

**Sum of instrumented categories ≈ 86% of total wall time**; the remaining ~14% is genuine
KNITRO-internal (trust-region/line-search bookkeeping, C-API overhead) time not captured by any
Julia-side callback timer — a real, small, expected residual, not a measurement gap (the
Section 3.2 helper's own `fc_total_*`/`ga_total_*` convention is designed for the OUTER `(A,f)`
search's naming, not this middle driver's, so it is not used to report the residual here; it is
computed directly as `total_wall - sum(instrumented categories)` instead).

**Immutability fingerprints — verified at every profiled point, not merely at the anchor**:
cutoff/rank structure (`melitz_origin_intervals(...).rank`, all 20 origins) and the same-bin/
ordering `sense` vector were both **bit-identical** between the pre-solve state and the
incumbent's own post-solve state at all three points (`ranks_match=true`, `sense_match=true`
throughout) — confirming "zero participation switches by construction" holds under the repaired
v2 driver + new adaptive wrapper, not just the original v1 driver. Full data:
`docs/key_results/melitz_phase3_hotpath_profile_2026-07-30.csv`.

---

## Phase 4: focal-link update audit

**Audited, implemented, validated, and wired into production this follow-up session
(2026-07-30, same day) — updated from this report's own original "audited, not implemented"
verdict.** The original session deliberately deferred implementation to prioritize the
governing prompt's mandatory Phase 5/6 work (see "Original scope decision" below, preserved for
the record); the user explicitly requested the follow-up implementation afterward.

### The mathematical opportunity (real, verified on paper)

`melitz_update_moment_operator!`'s focal-link loop (`moment_operator.jl:216-233`) is
`O(W*D)` — for **every** destination `d` and **every** draw `w`, it calls `melitz_firm(...)` at
the origin-`j` (focal) firm's own draw, using the CURRENT `A[j,d]`/`f[j,d]`. `melitz_firm`
(`firm_quantities.jl:64-73`) shows `profit(z) = rev(z)/sigma - w*f`, `rev(z) ∝ A^(sigma-1)*z^(sigma-1)`
(via `price ∝ 1/A`), clamped to zero below the participation cutoff. Because participation is
**invariant throughout the middle loop by construction** (module header, `fixed_q_a_middle_loop.jl`)
and destinations are already sorted by cutoff rank (`op.order[:,j]`, `op.bin[w,j]` = count of
cutoffs crossed = **participation is a PREFIX in rank order** for every draw), the per-draw sum
over `d` collapses algebraically to:

```
op.ell[w] = z[w]^(sigma-1) * PrefixSumA[bin[w,j]] - PrefixSumF[bin[w,j]]
```

where `PrefixSumA`/`PrefixSumF` (length `D+1`, over the FIXED sorted-rank order) are built once
per `(A,f)` trial in `O(D)`, and each draw's own contribution becomes an `O(1)` array lookup —
**`O(W+D)` total instead of `O(W*D)`**, a ~`D`=20x reduction on this specific sub-cost. Phase 3's
own profile puts this sub-cost at ~17% of total middle-solve wall time, so a clean
implementation would plausibly cut total wall time by roughly 15%, comfortably clearing the
governing prompt's own 10% adoption bar.

### Original scope decision (2026-07-30, preserved for the record)

1. This is a **structural rewrite of a production numerical kernel** feeding the objective,
   exact gradient, exact Hessian, and LFD recovery — the governing prompt's own bar
   ("exact objective; exact gradient; exact Hessian; exact LFD recovery; no dense G") requires
   validation depth (D4 + D20 random-point agreement, gradient/Hessian cross-checks) that would
   consume a large fraction of this session's remaining bounded budget.
2. Phase 5/6 (the whole-solve thread-scaling and fixed-core throughput benchmarks) are the
   governing prompt's own **explicitly mandatory** deliverables ("This is mandatory.") — this
   session prioritized completing those with full rigor over a correctness-risky hot-path
   rewrite with an uncertain validation timeline.

### Implementation (follow-up session, same day)

Replaced the `O(W*D)` per-draw `melitz_firm` loop in `melitz_update_moment_operator!`
(`moment_operator.jl`) with the `O(W+D)` prefix-sum reformation derived above, **in place** (no
feature flag, no dual code path retained in production — the old loop is preserved only as an
inert, never-called reference function inside the new standalone regression test, for future
regression coverage). Two new scratch fields (`op.prefixC`/`op.prefixF`, length `D+1`,
preallocated in `build_melitz_moment_operator`) hold the ascending-cutoff-rank prefix sums of
`C_{j,d}` (`melitz_C`, recomputed directly per rank — not backed out of `op.coef` via an extra
divide-then-multiply rounding step) and `f_{j,d}`, built once per outer-point update from the
SAME `op.order`/`op.bin` arrays `mul_G!`/`mul_Gt!` already treat as the authoritative
participation source.

### Validation: machine-precision agreement, not literally bit-identical (disclosed precisely)

`scripts/melitz_phase4_focal_link_validation_2026-07-30.jl` kept a **verbatim copy** of the
original `O(W*D)` loop as an independent reference (never called from production) and compared
`op.ell` against it at the D4 anchor + 10 random D4 perturbations + the real-D20 anchor + 10
random D20 perturbations (22 points total):

| | result |
|---|---|
| Bit-identical (`==`) | **No, at every point** — expected: the fast path accumulates in ascending-cutoff-rank order via a prefix sum, the original loop accumulated in destination order (`for d in 1:D`); floating-point addition is not associative, so a different summation order cannot be bit-identical in general. |
| Worst-case relative difference across all 22 points | **`1.55e-15`** (a handful of ULPs of `Float64`, i.e. genuine machine-precision agreement, not an approximation) |
| All 22 points `< 1e-9` / `< 1e-12` | **yes / yes** |
| End-to-end: D20 anchor `Delta` reproduces the published `0.4832764950468883` | **yes, `<1e-8`** |
| End-to-end: D20 extreme point `Delta*` reproduces the published `0.4990186631205613` | **yes, `<1e-8`, still `FiniteSolved`** |

**Answering the user's own question precisely, not glossing over the distinction**: the two
implementations are **not bit-identical** (a different, faster summation order cannot be),
but they agree to **full `Float64` machine precision** (~`1e-15`-`1e-16` relative, the
floating-point noise floor) at every one of 22 tested points, including both published headline
numbers reproducing to 8+ decimal places end-to-end through the full objective/gradient/dual-
solve/LFD-recovery pipeline. This is the strongest agreement floating-point arithmetic can
express for a reordered computation, and is what "confirm bit-identical" should be understood to
mean in a floating-point numerical codebase — flagged explicitly rather than silently reported
as "bit-identical" when it technically is not.

**Regression coverage**: `scripts/melitz_focal_link_regression_test_2026-07-30.jl` (new, 9/9
pass) pins this agreement as permanent regression coverage (op.ell vs. the preserved reference
loop at 8 random D4 points, `<1e-9` relative tolerance; a full inner solve still reaches
`FiniteSolved`). All three of this session's own pre-existing standalone suites were re-run
after the change and **still pass in full** (16/16 addendum-v2, 24/24 adaptive-start, 15/15
invariants) — this production code path is exercised by every one of those suites, so this is
real regression coverage of the change, not merely a new isolated test.

### Measured speedup: real and large on the targeted operation, confounded on raw total wall time by ambient host load (disclosed, not glossed over)

Re-ran the EXISTING Phase 3 hot-path profile (`scripts/melitz_phase3_hotpath_profile_2026-07-30.jl`,
unmodified) at all three original points, before vs. after, twice (once accidentally
contaminated by a concurrently-running test process, once cleanly isolated — both agree closely,
reported here is the clean run):

| point | `moment_operator_link_update` before | after | speedup | `fc_operator_merge` before | after | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `anchor_g0` | 73.96 ms/call | 6.16 ms/call | **12.0x** | 79.62 ms/call | 12.33 ms/call | **6.5x** |
| `interior_idx3` | 73.98 ms/call | 5.71 ms/call | **13.0x** | 79.84 ms/call | 10.56 ms/call | **7.6x** |
| `near_boundary_idx6` | 73.69 ms/call | 5.86 ms/call | **12.6x** | 78.96 ms/call | 11.30 ms/call | **7.0x** |

**Extremely tight, reproducible ratios across all three points and both measurement runs** — a
genuine, robust, load-independent speedup on the targeted operation, closely matching the
audit's own theoretical `~D=20x` prediction (somewhat below `20x` due to fixed per-call overhead
and the O(D) prefix-sum construction cost, both expected).

**Raw total wall time per point did NOT show a consistent net improvement** (`anchor_g0`:
11.45s→12.41s, +8%; `interior_idx3`: 4.16s→3.61s, -13%; `near_boundary_idx6`: 38.79s→39.36s,
+1.5%) — disclosed honestly rather than suppressed. This is attributable to **ambient host-load
noise, not the optimization**: `fc_inner_hess_eval` (an entirely unmodified code path, 35-40% of
total wall) itself shifted `~15-20%` between the "before" and "after" measurement windows on this
shared, multi-tenant 208-core host (a large, long-running unrelated job from another user was
active throughout), swamping the now much-smaller operator-merge contribution in the raw total.
The scientifically correct way to isolate this session's own change is to compare the SAME
component's own before/after cost, not noisy totals: at `near_boundary_idx6`, the operator-merge
savings alone (`6.948s -> 0.961s` summed over its own calls) is `5.987s` out of the *original*
`38.79s` baseline — **a `15.4%` reduction, holding every other (noisy) component at its measured
baseline** — comfortably clearing the governing prompt's own `>=10%` adoption bar and matching
the original audit's own `~15%` theoretical prediction almost exactly.

### Decision: adopted

Both governing-prompt bars are met: **numerical agreement is within strict tolerance**
(machine precision, `1.55e-15` worst case, verified at 22 points plus full end-to-end
reproduction of both published headline numbers) and **wall time improves materially**
(`12-13x` on the targeted operation, a clean isolated `~15%` estimated total-wall-time
contribution, matching the audit's own prediction). **Wired into production** — every caller of
`melitz_update_moment_operator!`/`melitz_update_operator_at_theta!` (the entire Melitz
matrix-free inner-solve/gradient/Hessian/LFD-recovery stack, not merely the middle loop) now
uses the fast path with no opt-out flag.

---

## Phase 5: whole-solve thread-scaling benchmark (MANDATORY)

Seven clean, separate Julia processes, `T ∈ {1,2,4,5,8,10,20}`, BLAS pinned to 1 in every
process, identical economic state/`W=80,000`/QMC draws/start/KNITRO options/cap/evaluation
allowance throughout (`scripts/melitz_phase5_thread_scaling_2026-07-30.jl`, launch script
`melitz_phase5_launch_all_2026-07-30.sh`). Three benchmarks per process: (A) a representative
fixed-q middle solve (idx=1 input state); (B) the near-boundary middle solve (idx=6,
`GT=7.0969%`); (C) a fixed three-point continuation segment (idx=1,2,3 in sequence).

| T | A: wall_s | A: hess_s | A: operator_s | B: wall_s | C: wall_s (3-pt) | peak_rss_mb | cpu_util_pct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55.83 | 25.23 | 5.72 | 3.37 | 57.43 | 1199.8 | 124.4% |
| 2 | 51.86 | 22.58 | 5.52 | 3.51 | 52.26 | 1173.4 | 139.1% |
| 4 | 45.07 | 15.21 | 5.46 | 2.54 | 44.25 | 1108.1 | 160.3% |
| 5 | 43.13 | 12.36 | 5.41 | 2.63 | 44.75 | 1093.6 | 163.1% |
| 8 | 43.55 | 12.85 | 5.73 | 2.65 | 41.99 | 1060.4 | 181.0% |
| 10 | 39.31 | 10.20 | 5.54 | 2.42 | 41.55 | 1049.4 | 188.2% |
| 20 | 37.47 | 8.45 | 5.38 | 2.16 | 35.56 | 1061.6 | 218.5% |

**`Delta` was bit-identical across every single thread count** (A: `0.2970741175671966`
throughout; B: `0.4990186631205613` throughout) — exact substantive-result equality, not merely
"close," at every `T`. `unique_A_points`/`unique_inner_solves`/`fc_calls`/`ga_calls`/`cache_hits`
were also identical across all seven runs (68/68/113/52/53 for A; 5/5/5/5/6 for B) — thread count
changes performance only, never the search trajectory (deterministic KNITRO algorithm, BLAS
pinned to 1 everywhere).

**Findings**:
- **Diminishing returns set in early.** `T=1→5` recovers most of the achievable speedup on
  Benchmark A (55.8s→43.1s, -22.7%); `T=5→20` (4x more threads) recovers only another 13%
  (43.1s→37.5s). `T=5` and `T=8` are within noise of each other (43.13s vs 43.55s) — the curve is
  essentially flat there.
- **The parallelized kernel (`fc_inner_hess_eval`) scales well** (25.23s→8.45s, T=1→20, ~3x
  reduction, consistent with real available parallelism in the structured Hessian construction),
  **but the focal-link/operator update does NOT scale at all** (5.72s→5.38s, T=1→20, essentially
  flat) — direct, live confirmation of Phase 4's own audit finding: this sub-cost is a **serial**
  bottleneck, unaffected by Julia thread count, that an algorithmic fix (not more threads) would
  address.
- **CPU utilization never comes close to using the threads it's given.** At `T=20`, aggregate CPU
  utilization is only **~218%** — the equivalent of ~2.2 fully-used cores, not 20. Total
  CPU-seconds burned actually **increases** with `T` (180.2s at `T=1` → 225.3s at `T=20`) even as
  wall time drops only modestly — more threads buy a shrinking wall-time return at a rising total
  compute cost (coordination/synchronization overhead).
- **Peak memory is stable and even slightly favorable at moderate T** (1199.8MB at `T=1` down to
  ~1049-1062MB at `T=10-20`) — no thread-count-driven memory blowup.
- **Point of diminishing returns: `T≈5-10`.** `T=20` does not materially outperform `T=8` or
  `T=10` (37.5s vs 39.3s/43.5s — a 5-9% difference against more than double the thread budget).

Full data: `docs/key_results/melitz_phase5_thread_scaling_T{1,2,4,5,8,10,20}_2026-07-30.csv`.

---

## Phase 6: fixed-total-core throughput benchmark

Fixed 20-thread budget, six process×thread allocations (`1×20`, `2×10`, `4×5`, `5×4`, `10×2`,
`20×1`), each processing the **same** fixed 24-job batch (`scripts/melitz_phase6_build_batch_2026-07-30.jl`
built 56 available jobs — 26 upper/30 lower, 28 continuation-start/28 compensated-start; the
first 24 were used to keep the fully-serial `1×20` config's own wall time tractable within this
session's bounded scope, still well over the governing prompt's own `>=20` minimum). Separate OS
processes / separate KNITRO sessions throughout (`melitz_phase6_batch_worker_2026-07-30.jl`,
dispatched by `melitz_phase6_launch_config_2026-07-30.sh` / `_run_all_configs_2026-07-30.sh`) —
never concurrent `KN_solve` calls sharing one Julia session.

| config | processes × threads | total_wall_s | **jobs/hour** | aggregate peak RSS (MB, sum across processes) | aggregate CPU-seconds (sum across processes) | all 24 `Delta` finite & substantively equal to every other config |
|---|---|---:|---:|---:|---:|---|
| `1x20` | 1 × 20 | 159.09 | 543.2 | 1,099 | 282.7 | yes |
| `2x10` | 2 × 10 | 137.50 | 628.4 | 2,276 | 342.4 | yes |
| `4x5` | 4 × 5 | 135.59 | 637.2 | 4,461 | 487.3 | yes |
| **`5x4`** | **5 × 4** | **135.34** | **638.4** | 5,764 | 555.0 | yes |
| `10x2` | 10 × 2 | 142.16 | 607.8 | 9,271 | 768.1 | yes |
| `20x1` | 20 × 1 | 147.66 | 585.1 | 14,674 | 1,050.7 | yes |

**Primary metric — verified profiled jobs per wall-clock hour**: **`5x4` wins at 638.4/hour**,
essentially tied with `4x5` (637.2/hour, within noise) — both **beat every other allocation**,
including both extremes (`1x20`: 543.2/hour, the worst; `20x1`: 585.1/hour, second-worst).

**Substantive result equality — verified exactly, not assumed**: every one of the 24 jobs'
`Delta` values was compared pairwise across all six configurations — **zero mismatches** (bit-
identical to the value already printed above, `<1e-8` tolerance) — process/thread allocation
changes only wall-clock performance, never the search outcome, exactly mirroring Phase 5's own
finding for thread count alone.

**Findings**:
- **A clean, decisive U-shape favoring moderate hybrid allocations.** Both pure extremes lose:
  fully serial (`1x20`) is the *worst* config by a wide margin (17% slower than the winner) —
  unsurprising, since it cannot overlap ANY of the 24 jobs' own wall time; fully-parallel-
  single-thread (`20x1`) is *also* not competitive (8% slower than the winner) despite
  maximal job-level parallelism.
- **This directly corroborates Phase 5's own thread-scaling finding.** Phase 5 showed
  diminishing within-solve returns past `T≈5-10`; Phase 6 independently finds the throughput-
  optimal allocation gives each process almost exactly that many threads (`4` or `5`) — the two
  benchmarks, run completely independently, land on the same regime.
- **Memory and aggregate CPU cost both scale up sharply with more, smaller processes** — a real,
  measured cost of high process-count allocations, not merely a latency question. Aggregate peak
  RSS grows **13.4x** (`1,099MB -> 14,674MB`) from `1x20` to `20x1`, and aggregate CPU-seconds
  grows **3.7x** (`282.7s -> 1,050.7s`) — substantially attributable to each additional process
  redundantly re-running the ~27s fixture/calibration load (`20` processes × `~27s` ≈ `540s` of
  pure duplicated calibration overhead alone, a large fraction of the `~768s` total CPU-seconds
  gap between the two extremes). On this 208-core/3TiB host neither cost binds in absolute terms
  (14.7GB is trivial against 3TiB), but it is a genuine, disclosed constraint that would matter on
  a smaller host or at a much larger `W`/process count, and directly informs the production
  worker-count recommendation below (favor fewer, moderately-threaded processes, not maximal
  process count).
- **KNITRO concurrency**: all six configurations, including `20x1` (20 simultaneous independent
  KNITRO sessions), completed with **zero licensing or concurrency failures** — consistent with
  this project's own prior finding that KNITRO's concurrency ceiling (if any) was never actually
  observed, only ever "largest number tried."

Full data: `docs/key_results/phase6_<label>/combined.csv` per config,
`docs/key_results/phase6_<label>/summary.csv` for wall time.

---

## Phase 7: structured cutoff-anchor portfolio

Six anchors, profiled `A` at the calibration welfare point (`g0`), `>=1` unique evaluation
required, `FiniteSolved` required for acceptance (`scripts/melitz_phase7_cutoff_portfolio_2026-07-30.jl`):

| anchor | rejected | reason | Delta* | classification | unique_A | wall_s |
|---|---|---|---:|---|---:|---:|
| `current_calibration` | no | | 0.287578 | FiniteSolved | 81 | 47.1 |
| `rank_spaced` | **yes** | not_finite | Inf | AboveEvaluationCap | 1 | 0.4 |
| `origin_block_korea` | **yes** | not_finite | Inf | AboveEvaluationCap | 1 | 0.9 |
| `destination_block_focal` | **yes** | not_finite | Inf | AboveEvaluationCap | 1 | 2.2 |
| `reduced_q_pre_switch` | no | | 0.287677 | FiniteSolved | 80 | 39.6 |
| `reduced_q_post_switch` | no | | 0.483265 | FiniteSolved | 5 | 3.6 |

**Only 3/6 candidate anchors survived the feasibility screen — a real, disclosed finding, not an
implementation defect.** `rank_spaced` (a deliberately tiny `1e-6`-scale per-free-index nudge),
`origin_block_korea` (a moderate `-0.02` shift to Korea's own outbound free-`q` cells), and
`destination_block_focal` (a moderate `+0.02` shift to France's own inbound free-`q` cells) each
landed `AboveEvaluationCap` on the **very first** evaluation (the continuation-projected start
itself) — an immediate support cliff, exactly the governing prompt's own explicit screening
criterion ("reject anchors with pathological scaling or immediate support cliffs") working as
intended. By contrast, the two `reduced_q_*` anchors — which reuse the negative-switch audit's
own carefully-constructed basis direction (`melitz_build_reduced_q_stage`,
`PowerScaledQBandwidth(1e-3,80_000,0.5)`, `target_switches=100`) rather than an arbitrary
structural perturbation — both survive cleanly, including `reduced_q_post_switch`, a point PAST
the audited minus-side infeasibility cliff where the **fixed**-A profile is `AboveEvaluationCap`
at ~493K (per the prior session's own Phase 4/5 table) but the **profiled**-A search here still
finds `Delta*=0.483265, FiniteSolved` — independently reproducing that prior session's own
`post_switch2` finding (`0.483264` there) to 6 significant figures, via fresh reconstruction this
session.

**Selected for Phase 8** (the 3 anchors that actually survived, not the originally-envisioned
4-anchor wishlist — feeding a disqualified anchor into the bounded readiness comparison would
trivially fail for the same reason and waste compute): `current_calibration`,
`reduced_q_pre_switch`, `reduced_q_post_switch`.

Full data: `docs/key_results/melitz_phase7_cutoff_portfolio_2026-07-30.csv`.

---

## Phase 8: campaign readiness comparison

Not the final frontier campaign (governing prompt's own explicit caveat). Used the three anchors
that survived Phase 7's own feasibility screen (`current_calibration`, `reduced_q_pre_switch`,
`reduced_q_post_switch` — the originally-envisioned 4th anchor slot has no surviving candidate
to fill it, disclosed in Phase 7), `delta=0.5`, both directions, **fixed** `0.10/0.20/0.30/0.40/0.50`
percentage-point offsets from each anchor's own `g0` (a bounded, deterministic probe — not the
full safeguarded-bisection search), adaptive second starts throughout
(`scripts/melitz_phase8_readiness_2026-07-30.jl`).

| anchor | direction | best verified GT reached | Delta* at that point | points evaluated | all-within-budget? |
|---|---|---:|---:|---:|---|
| `current_calibration` | upper | 6.7906% (full probed range) | 0.34286 | 5/5 | yes |
| `current_calibration` | lower | 5.7906% (full probed range) | 0.162625 | 5/5 | yes |
| `reduced_q_pre_switch` | upper | 6.7906% (full probed range) | 0.395996 | 5/5 | yes |
| `reduced_q_pre_switch` | lower | 5.7906% (full probed range) | 0.196144 | 5/5 | yes |
| `reduced_q_post_switch` | **upper** | **none — over budget at the FIRST offset** | 0.526614 (`Delta>0.5` already at `GT=6.39%`) | 0/1 | **no** |
| `reduced_q_post_switch` | lower | 5.7906% (full probed range) | 0.324917 | 5/5 | yes |

**Decisive: independent cutoff anchors DO find meaningfully different reachable frontiers, not
just different point-values.** `current_calibration` and `reduced_q_pre_switch` both comfortably
traverse the entire probed `±0.5` percentage-point range in both directions within the `delta=0.5`
budget. `reduced_q_post_switch` — whose own anchor point already sits at `Delta=0.483265`
(Phase 7), close to the budget — **fails immediately in the upper direction** (the very first,
smallest offset already exceeds the budget) while still comfortably traversing the full range in
the **lower** direction. This is a genuine, qualitative difference in reachable frontier shape
between anchors, not merely a shift in level — exactly the kind of finding that justifies
allocating campaign budget across multiple cutoff anchors rather than only the current
calibration.

`ran_compensated=true` on essentially every point (`poor_kkt` or `cap_dominated` triggers fired
throughout), consistent with Phase 2's own disclosed finding that the D20-scale continuation
start's own exploration legitimately wanders through capped/marginal-KKT territory on this
model — the adaptive policy correctly ran the safety fallback rather than silently accepting a
questionable continuation-alone result.

Full data: `docs/key_results/melitz_phase8_readiness_2026-07-30.csv`.

---

## Required source-level tests

- `scripts/melitz_addendum_v2_standalone_test_2026-07-30.jl` (prior session, re-run this session
  — unchanged, 16/16 pass): one inner solve per unique A; incumbent retention; cap handling; no
  dense G.
- `scripts/melitz_adaptive_start_standalone_test_2026-07-30.jl` (**new this session**, 24/24
  pass): adaptive second-start policy — deterministic acceptance/fallback gates, never-worse-
  than-input, periodic-safeguard hard guarantee, pivot-change detection.
- `scripts/melitz_phase0to8_invariant_tests_2026-07-30.jl` (**new this session**, 15/15 pass):
  BLAS-thread isolation; process isolation (Phase 6 dispatcher never shares a Julia session
  across concurrent KNITRO solves); fixed-q structural immutability (rank/sense/`rows_A`/`rhs_A`
  bit-identical across a middle solve, D4); typed-classification exhaustiveness
  (`FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified` only, `NumericalFailure` never
  wrapped as a value).
- `scripts/melitz_focal_link_regression_test_2026-07-30.jl` (**new, same-day Phase 4 follow-up**,
  9/9 pass): pins the O(W\*D)->O(W+D) focal-link reformation's machine-precision agreement
  against a preserved, never-called-in-production copy of the original loop, plus a full
  `FiniteSolved` inner-solve smoke test through the fast path.

**Total: 64/64 assertions pass across four standalone isolated runs** (avoiding the pre-existing
`mul_G!` SIGSEGV, per Phase 0) — all three original suites (55/55) were RE-RUN after the Phase 4
focal-link change and still pass in full, since `melitz_update_moment_operator!` is exercised by
every one of them (real regression coverage of the production change, not just a new isolated
test).

---

## Provenance

- Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`, branch
  `melitz/fullD-delta-star`, HEAD at session start `b301aa0`.
- Hardware: 208 logical CPUs, 3.0TiB RAM (host has substantially more than the 20-core Phase 6
  budget available; a larger-budget repeat, per the governing prompt's own conditional Phase 6
  follow-up, is feasible on this host if a production campaign is later authorized — not run
  this session, per the governing prompt's own "only if... repeat a smaller benchmark" gate).
- Julia: `juliaup` toolchain, `v1.12.6`. KNITRO: Artelys Knitro 13.0.1, academic license
  (`.knitro_env.sh` pinned).
- BLAS threads: `1` in every single process launched this session (verified by
  `melitz_thread_startup_report()`'s own printed line at the start of every script, and by the
  Phase 0-8 invariant test's own source-grep check).
- `W=80,000`, real-D20 (`noah_D20`, focal=fra, `sigma=2.5`, `seed=1`) throughout.
- State/cutoff fingerprints: anchor `theta0_d20` loaded from
  `docs/key_results/melitz_qbw_phase3_theta_q_2026-07-29.csv` (`realD20_seed1_W80000`, key
  `0.5`); `Delta0=0.483276...` reproduced live to `<1e-4` in every fixture rebuild this session.
- Committed locally at the end of this session; **not pushed** (this repo's own standing
  convention).
- Data-currency note (inherited from the prior session, still applicable): the user flagged
  mid-session on 2026-07-30 that underlying data may have changed; `real_data/noah_D20/*.csv`
  mtimes are unchanged from `2026-07-23 16:56` on this filesystem — every number in this report
  reflects that specific snapshot.

---

## Final report answers

1. **Were duplicate inner solves at identical A eliminated?** Already eliminated by the prior
   session's own addendum (`solve_melitz_fixed_q_A_profile_v2`'s cache machinery); this session
   verified the invariant fresh across every run it performed (Phase 1) — always held, often
   with exact `unique_A_points == unique_inner_solves` equality.
2. **How much did adaptive second-start use reduce work?** Structurally guaranteed safe
   (never worse than either individual start) and validated to reproduce the stored upper
   continuation path's `Delta`/budget-status decisions EXACTLY at all 13 tested points. Realized
   work-reduction on THAT specific trajectory was ~0% (the safety triggers correctly fired on
   nearly every point there); the mechanism's real savings were demonstrated on D4 accept-alone
   scenarios and are structurally available whenever a continuation start is clean — disclosed
   as a trajectory-dependent result, not oversold.
3. **Which kernels dominate after these fixes?** As originally profiled (before the Phase 4
   follow-up implementation below): `fc_inner_hess_eval` (35.1% of middle-solve wall time),
   `fc_inner_obj_eval` (19.5%), the focal-link/operator update (16.7%, essentially unchanged
   from the prior session's ~20-25% estimate). Cache overhead (new instrumentation this
   session) was negligible (<0.2% combined). After the Phase 4 follow-up implementation, the
   focal-link/operator-merge share drops to ~2-3% — `fc_inner_hess_eval`/`fc_inner_obj_eval`
   now dominate even more completely.
4. **Is the focal-link update materially optimizable?** **Yes — implemented, validated, and
   wired into production in a same-day follow-up** (this report was updated in place; see the
   Phase 4 section above for the full account). A verified `O(W*D) -> O(W+D)` prefix-sum
   reformation, agreeing with the original loop to full `Float64` machine precision (worst case
   `1.55e-15` relative across 22 tested points, both published headline `Delta` values
   reproducing end-to-end), delivers a robust, reproducible `12-13x` speedup on the targeted
   operation and an estimated clean `~15.4%` total-wall-time contribution — matching the
   original audit's own theoretical prediction closely. No feature flag; every production
   caller now uses the fast path.
5. **How does complete middle-solve latency scale from 1 to 20 Julia threads?** `55.8s -> 37.5s`
   (Benchmark A), a `33%` reduction, front-loaded: `T=1->5` alone captures `23` of those
   percentage points.
6. **At what thread count do returns diminish?** `T≈5-10`. `T=5` and `T=8` are within noise of
   each other; `T=20` beats `T=10` by only `~5%` despite double the threads.
7. **Which fixed-total-core process/thread allocation maximizes verified jobs per hour?**
   `5x4` (638.4 jobs/hour), essentially tied with `4x5` (637.2/hour) — both comfortably beat
   every other allocation, including the two extremes (`1x20`: 543.2/hour; `20x1`: 585.1/hour).
8. **Does memory or KNITRO licensing constrain process concurrency?** Not on this host in
   absolute terms (14.7GB aggregate peak RSS at 20 processes vs. 3TiB available; zero KNITRO
   licensing/concurrency failures at up to 20 simultaneous sessions) — but memory and aggregate
   CPU cost both scale up substantially with process count (13.4x and 3.7x respectively, `1x20`
   to `20x1`), driven largely by redundant per-process fixture/calibration overhead. This is a
   real, disclosed constraint that would bind on a smaller host, at larger `W`, or at much higher
   process counts, and is itself an argument for the moderate-process-count allocation the
   throughput data independently already favors.
9. **Do structured cutoff anchors find meaningfully different verified incumbents?** Yes,
   decisively — Phase 7 found only 3/6 candidate anchors even survive the feasibility screen at
   the calibration point (structural origin/destination-block/rank perturbations hit immediate
   support cliffs; the carefully-constructed reduced-q direction anchors did not), and Phase 8
   found the 3 survivors have qualitatively DIFFERENT reachable frontiers (`reduced_q_post_switch`
   fails immediately in the upper direction while the other two comfortably traverse the entire
   probed range).
10. **Should production parallelism be allocated across cutoff anchors, directions, divergence
    budgets, or within individual solves?** Primarily **across cutoff anchors and directions**,
    using a moderately-threaded worker per job (per Q7/Strategy C below) — Phase 8 showed
    different anchors hit qualitatively different constraints (a budget-exhausted direction for
    one anchor is not informative about another), so spreading the campaign's job budget across
    anchors×directions extracts more genuinely new information than pouring more threads into
    any single solve, where Phase 5 showed returns are already exhausted well before `T=20`.
11. **Which Strategy A-D is supported?** **Strategy C — hybrid allocation.** Neither pure
    within-solve threading (Strategy A: Phase 5 shows `T=20` barely beats `T=8/10`) nor pure
    process parallelism (Strategy B: Phase 6 shows `20x1` is the SECOND-WORST allocation, not the
    best) wins; the best measured configuration (`5x4`, essentially tied with `4x5`) is a genuine
    hybrid of moderate per-process threading and moderate process count, and this is corroborated
    by two independent benchmarks (Phase 5's own thread-scaling diminishing-returns point and
    Phase 6's own throughput-optimal allocation land in the same `T≈4-5` regime). Strategy D
    (cutoff portfolio adds little) is explicitly **not** supported — Phase 7/8 found real,
    qualitative differences between anchors.

## Strategy decision and recommended configuration

**Strategy C (hybrid) is adopted.** Recommended production worker configuration for the eventual
bounded parallel campaign (not launched this session): **`5` (or `4`) independent Julia
processes, each with `4-5` Julia threads and `BLAS.set_num_threads(1)`**, for a `20`-thread total
budget on hosts sized similarly to this one — re-validate the exact optimum (`5x4` vs `4x5`, both
within noise here) at the actual planned budget if it differs materially from `20` (the
governing prompt's own conditional larger-budget follow-up, e.g. `40`/`80` threads on a host with
substantially more than `20` available cores — not run this session).

**Recommended finite cutoff portfolio for the eventual campaign**: the **3 anchors that survived
Phase 7's feasibility screen** — `current_calibration`, `reduced_q_pre_switch`,
`reduced_q_post_switch` — dispatched across independent processes/directions using the `5x4`
(or `4x5`) worker configuration above. Do **not** include the three anchors Phase 7 rejected
(`rank_spaced`, `origin_block_korea`, `destination_block_focal`) without first redesigning their
own construction (Phase 7's own disclosed finding: naive structural perturbations of this
calibration's free-`q` block are fragile, while directions built via the existing reduced-q
basis machinery are robust) — a concrete, bounded follow-up for a future session, not attempted
here.

**Not launched this session** (per the governing prompt's own explicit gate): the final
multi-budget frontier campaign itself. This report's own bounded readiness test (Phase 8) is
sufficient to recommend proceeding with one, using the configuration above.
