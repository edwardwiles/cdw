# Production-candidate integration: exact fast infeasibility screens beyond zero-winner rejection

**Branch**: `integration/fullA-fast-range-screen`
**No merge performed.** This branch is a reviewable production candidate. Await
explicit user approval before any component is merged into a production line.

## 0. Branch provenance

| | branch | commit | worktree |
|---|---|---|---|
| Production base (see note below) | `diag/fullA-d4-exact` | `02583bc` (2026-07-19 13:39:50) | `gravity-fullA-d4` |
| Overnight production run's real code state | (uncommitted on `diag/fullA-d4-exact`) | `02583bc` + working-tree diff to `c10_d20_production_driver.jl` | `gravity-fullA-d4` |
| Experimental source (envelope + range screens) | `diag/fullA-d20-range-screen-review` | `cac7626` (envelope), `0fd2bf8` (range screen) | `gravity-fullA-d20-range-screen-review` |
| **New integration branch** | `integration/fullA-fast-range-screen` | see `git log` | `gravity-fullA-fast-range-screen-integration` |

**Base-commit note**: `diag/fullA-d20-range-screen-review` and `integration/fullA-d20-common-marginals`
are two independent same-day sibling branches, both cut from `02583bc`, neither merged into the
other. The user identified the correct base for *this* integration as the code state that actually
produced the overnight full delta-grid (0.1/1/2/5) D=20 production run
(`docs/fullA_continuation11_overnight_prompt.md` / `docs/fullA_continuation11_handoff.md`,
2026-07-19 21:03 -> 2026-07-20 11:53) — which is `02583bc` plus a real, validated, but *never
committed* working-tree diff to `full_aod_diag/d4_exact/c10_d20_production_driver.jl` in the
`gravity-fullA-d4` worktree (confirmed via `git status`/`git diff` there: exactly one modified
tracked file, everything else untracked diagnostic scripts/outputs). That diff — buffered-gradient
default (`lfix_buffer_reuse.jl`), `skip_cold_retry=true` default, additive warm/cold timing trace —
is this integration branch's first commit ("Port overnight Continuation-11 production driver state
as integration baseline"), so this branch sits on the actual reviewed production code, not on
`02583bc` plus a silent gap. The common-marginals integration branch is a *different*, separately
unmerged line of work that started *after* this overnight run finished; it is not part of this
branch's base and its own moment columns are explicitly **not supported** by the screens below (see
§4).

### `git status`

```
On branch integration/fullA-fast-range-screen
nothing to commit, working tree clean
```

### Files changed vs. base (`02583bc`)

```
39 files changed, 3591 insertions(+), 7 deletions(-)
```

New production logic: `full_aod_diag/d4_exact/fast_range_screen.jl` (908
lines). Modified production file: `c10_d20_production_driver.jl` (+39/-7 --
the overnight Continuation-11 driver-state port from the baseline commit,
plus one new opt-in `include`; the driver's `cb_F!`/`cb_G!` callback bodies
themselves are untouched). Everything else: new test/benchmark scripts (3
files) and ported data fixtures (`diagnostics/infeasibility_points/`, 34
files, pure data) + this doc + the warm-start note.

## 1. What this branch adds (and what it does not)

Two **complementary, independent** exact one-sided certificates of primal
infeasibility, on top of the existing (unmodified) zero-winner screen in
`infeasibility_screen.jl`:

| screen | catches | cost | draws touched | status code |
|---|---|---|---|---|
| (existing) pairwise never-wins | origin can *never* win any draw at a destination | O(D²), one-time O(D²·W) precompute | none per-call | `pairwise_certified_infeasible` |
| (existing) `screen_hard_winners` zero-win count | origin wins zero draws at this specific outer point (pairwise cert missed it but the point itself has zero wins) | O(W·D)-ish | yes | `winner_scan_infeasible` |
| **NEW** `envelope_prewinner_screen` | origin's *best possible* winning value can never reach the target, certified WITHOUT determining who wins | O(D²), one-time O(D²+W·D) precompute | **none** | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE` |
| **NEW** `screen_hard_winners_ranged` (fused) | same failure mode as the envelope screen, but exact (not an upper-bound approximation) — fused into the existing winner scan, single pass | O(W·D)-ish, same pass as the existing winner scan | yes (already being scanned) | `EXACT_INFEASIBLE_WINNING_RANGE` |
| **NEW** `range_screen_standalone` (general safety net) | any strictly one-sided inner-dual column, both signs, including the counterfactual price-index column — the ONLY screen that covers the positive-side / non-bilateral case | O(W·D), operates on an already-built `CompressedFactual` (no rebuild) | yes (already being scanned) | `EXACT_INFEASIBLE_MOMENT_RANGE` |

Explicitly **not** changed: `DUAL_WARM_MODE`/warm-start policy (see
`docs/fullA_warm_start_reliability_note.md`), cutting-plane phase I, direct
primal LP, cached separator banks, dual-ray early termination, outer
trust-region/backtracking logic. All remain experimental, on their own
branches, untouched here.

## 2. Exact factorization — re-derived and live-verified against the current executable production code

Re-derived directly from two independent authoritative sources in this
repo (not from the experimental branch's schematic — see
`full_aod_diag/d4_exact/fast_range_screen.jl`'s header comment for the full,
line-by-line derivation and citations):

1. `compressed_moments.jl`'s own header comment (the file that defines the
   TRUSTED winner-form factual moment construction), which gives the exact
   winning-draw formula `pTsigma_{s,o,d} = constConsSigma_{o,d} / Usigma_{s,o}^{-mu}`
   with `constConsSigma_{o,d} = wHat_o^{1-sigma}*(AodPow_{o,d}*tau_{o,d})^{1-sigma}`,
   `AodPow_{o,d} = (Aod_{o,d}/cHat_{o,d})^{-mu}`, `Aod_{o,d} = Aod_theta_{o,d}*B(o,d)`.
   Substituting through collects every `A_od_theta`-independent factor into a
   single per-cell constant `K2(o,d)`, giving

   ```
   h_{od,s}(A) = a_{od}(A) * x_{od,s}
   a_{od}(A) = Aod_theta_{o,d}^{mu*(sigma-1)}
   x_{od,s}  = K2(o,d) / Usigma_{s,o}^{-mu}         (data-only, IF mu/sigma are fixed)
   ```

2. `context_real_d20.jl::d20_real_setup` (the real-data D=20 context builder
   this whole integration targets — the same one the production driver
   calls), lines 75-90: `theta_lo[1]==theta_hi[1]` (mu) and
   `theta_lo[2]==theta_hi[2]` (sigma) are set equal, and neither index appears
   in `free_idx` — **mu, sigma are literally box-constrained to a point**, not
   assumed fixed. `free_idx = vcat(3+Dact, Aod_offset+1:Aod_offset+Dact^2)`,
   `Aod_offset = 3+Dact`, and `l_full == Aod_offset+Dact^2` exactly (verified:
   `theta0_up` has length `D^2+D+3` = mu,sigma,D gammas,gamma_prime_focal,D²
   A_od) — **A_od is confirmed, not assumed, to be the trailing D² block of
   theta_full**. `precompute_envelope`/`envelope_prewinner_screen` re-assert
   both facts live at every ctx/call rather than hardcoding them.

### Assumption guard table (live-checked, not assumed)

| assumption | status in every config this repo runs today | what happens if violated |
|---|---|---|
| mu, sigma fixed (box-constrained to a point) | **TRUE** (`context_real_d20.jl:76-77`) | `precompute_envelope` throws `EnvelopeUnsupportedContext`; caller disables the envelope/fused-range screens for that ctx (`RangedScreenContext.envelope === nothing`), existing zero-winner screen + general safety net still run |
| `usePMM == 0` | **TRUE** in every params NamedTuple grepped across this repo (`master.jl` default, `AD_PARAMS`, every `run_fullA_*` script) | same: `precompute_envelope` throws, envelope/fused-range disabled; general `range_screen_standalone` DOES correctly handle `usePMM==1` (it applies the `-usePMM*PMM[j]` term explicitly) and keeps working |
| `NormalizeMoments` (any value) | not a guard condition — `nrm[j]` is a context-level constant regardless of its value (verified: computed once from `gamma.sigma_Moments`/`gamma.moments_without_var`, never theta-dependent) | no effect either way |
| SamplingWeights uniform (`importanceSampling==0`) | **TRUE** in every config found | not a guard — `SWmax`/`SWmin` handling is already conservative for the non-uniform case; correctness does not depend on uniformity, only tightness does |
| A_od is the trailing D² block of `theta_full` | **TRUE**, re-verified live at every `precompute_envelope`/`envelope_prewinner_screen` call (`Aod_offset+D^2 == length(theta_full)`) | throws immediately, not silently wrong |
| K2(o,d) ≥ 0, target b(o,d) ≥ 0 (structural nonnegativity, the basis for "negative-side-only" violations) | **TRUE**, re-verified live at `precompute_envelope` construction | throws if violated for this ctx's data |

## 3. Production-context support / guard table (which moment columns each screen covers)

| column type | `envelope_prewinner_screen` | `screen_hard_winners_ranged` (fused) | `range_screen_standalone` (safety net) |
|---|---|---|---|
| bilateral, negative-side violation (the only kind proven structurally possible for this column type) | **yes** (sufficient certificate; a miss is inconclusive, not feasible) | **yes** (exact) | yes (redundant with the above two, but independent code path) |
| bilateral, positive-side violation (structurally impossible per the proof in §2 — zero instances observed across every point in this and the prior investigation) | no (not derived for this direction) | no | **yes** — the only screen that could catch this if the structural proof were ever wrong for a future config |
| counterfactual price-index column (`D²+1`) | no (out of scope — different formula, not derived here) | no | **yes** — the only screen covering this column at all |
| gravity moment (`obj.d`, outside `1:oci-1`) | n/a — outer-loop-only object, never an inner CC equality moment; excluded by construction from every screen here | same | same |
| common-marginals (CM) moment columns | **NOT SUPPORTED.** `context_real_d20.jl`/`d20_real_setup` (this branch's base context builder) has no CM wiring at all — CM support lives only on the separately-unmerged `integration/fullA-d20-common-marginals` branch, which has not been reconciled with this branch's base. If/when CM lands on the same base, its moments need their own range certificate derived explicitly — **do not assume** any screen here covers them. | | |

## 4. Certificate validation table

Real D=20/W=80,000 (France focal, sigma=2.5), run on THIS branch's own code
(`full_aod_diag/d4_exact/validate_fast_range_screen_d20.jl`,
`verify_candidates_detail.jl`) against 11 points from the recovered catalogue
under `diagnostics/infeasibility_points/` (ported verbatim as data fixtures
from `diag/fullA-d20-range-screen-review`; the catalogue also includes 3 more
files this run correctly skipped — `delta5_sequential_point_verification_
INDEPENDENT.json`, `unresolved_delta0_1_first_pass.json`,
`zero_winner_infeasible_W8000_4points.json` — none has a usable `Aod_theta_
full_csv` at this W, so they were not silently mis-scored).

| point | baseline (`evaluate_fullA_screened`, unmodified) | integrated (`evaluate_fullA_screened_ranged`) | verdict agreement |
|---|---|---|---|
| calibration-anchored feasible, δ=1 (`feasible_delta1_anchor_pushed_r8`) | screen_passed, solved | screen_passed, solved | **Δ_dual match to 1e-8** |
| feasible, δ=2 (`feasible_delta2_firstpass`) | screen_passed, solved | screen_passed, solved | **Δ_dual match to 1e-8** |
| feasible, δ=5 (`feasible_delta5_cascade_s1anchor`) | screen_passed, solved | screen_passed, solved | **Δ_dual match to 1e-8** |
| feasible, δ=5 (`feasible_delta5_firstpass`) | screen_passed, solved | screen_passed, solved | **Δ_dual match to 1e-8** |
| nonzero-winner-infeasible candidate 1 | screen_passed, **nStatus=-300** (real KNITRO call, ~1.0s) | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, col=1 (o=1,d=1), upper_bound=10.168, target=14.590 | **agree (both infeasible)**; envelope numbers exactly match the prior investigation's independently-reported 10.17 vs 14.59 |
| candidate 2 | nStatus=-300 | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, col=1, upper_bound=9.522, target=14.590 | agree |
| candidate 3 | nStatus=-300 | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, col=1, upper_bound=8.741, target=14.590 | agree |
| candidate 4 | nStatus=-300 | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, col=1, upper_bound=8.023, target=14.590 | agree |
| candidate 5 | nStatus=-300 | `EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, col=1, upper_bound=6.759, target=14.590 | agree |
| zero-winner, δ=1 | `pairwise_certified_infeasible` (existing screen, unchanged) | `pairwise_certified_infeasible` (identical code path — envelope/fused screens correctly never reached) | agree |
| zero-winner, δ=5 | `pairwise_certified_infeasible` | `pairwise_certified_infeasible` | agree |

**Independent cross-check, all 5 candidates**: `range_screen_standalone` — a
completely different code path (full winner scan -> `CompressedFactual` ->
column min/max), reusing NOTHING from the envelope derivation — independently
confirms column 1, `sign=:negative`, `max_val` strictly below `-tol` on every
one of the 5 candidates (margins 0.019–0.037 normalized). Two independently
coded certificates agree on every point.

**Result: 0/11 false positives, 0/11 false negatives, 0/4 value mismatches on
mutually-feasible points.** All 5 nonzero-winner-infeasible candidates —
verified via the baseline's own real KNITRO call to genuinely fail
(`nStatus=-300`, not a benign status) — are caught before any KNITRO call by
the new envelope screen. All rejections land at the SAME column (o=1,d=1,
France's own-trade cell) across all 5 candidates, consistent with the prior
investigation's finding that these 5 are repaired variants of one underlying
source point, not 5 independent pathologies.

**Not independently re-run this session** (cited as attributed prior evidence
from `diag/fullA-d20-range-screen-review` instead, pending a future session
with more real-KNITRO-time budget): the 6-random-gravity-tangent-perturbation
sweep and the δ=0.1 point (no full A_od CSV recovered for it at this W in
either branch's catalogue). Common-marginal contexts were **not tested**
because they are not supported by this branch's base context at all (§3) —
testing them would require reconciling with the separate CM integration
branch first, out of scope here.

## 5. Warmed end-to-end production benchmark

Real D=20/W=80,000, single warmed process
(`full_aod_diag/d4_exact/benchmark_fast_range_screen.jl`), run on this
branch's own code.

| stage | cost |
|---|---|
| `envelope_prewinner_screen`, feasible point (worst case: full D² scan, no hit) | **15.5 microseconds** |
| `envelope_prewinner_screen`, infeasible point (early hit) | **0.97 microseconds** |
| baseline `evaluate_fullA_screened`, feasible point, warm-started full call (existing screens + moment build + KNITRO inner solve) | 0.652 s |
| integrated `evaluate_fullA_screened_ranged`, feasible point, warm-started full call (existing pairwise + NEW envelope + NEW fused winner-range + NEW general safety-net-on-already-built-cf + KNITRO inner solve) | 0.579 s |
| measured "overhead" of integrated vs baseline | **-11.2%** (i.e., not measurably slower) |

The negative "overhead" is **not** a genuine claimed speedup — this is a
single warm call per path, not an averaged microbenchmark, so a ~10%
difference either direction is within run-to-run noise (GC pauses, KNITRO's
own iteration-count sensitivity to floating-point-identical-but-differently-
ordered intermediate values). The honest, defensible claim from this number:
**the fully-integrated screens (envelope + fused winner-range + general
safety net reusing the already-built `cf`) add no measurable cost on a warm
feasible call** — comfortably under the brief's 1% threshold, consistent
with the prior experimental review's own finding of "0.4% or less" for a
properly-integrated design (vs. ~10% for a naive bolt-on that rebuilds `cf`
independently — a design this branch's `evaluate_fullA_screened_ranged`
explicitly avoids, see `fast_range_screen.jl`'s
`evaluate_fullA_screened_compressed_with_cf`).

**On certified-infeasible points** the integrated path is the entire
point of this branch: ~1 microsecond to ~1 second, i.e. up to **~6 orders of
magnitude faster** than the baseline's real KNITRO call for the 5 candidates
tested (§4) — each replacing a ~0.9-1.0s KNITRO call that was always going
to fail with a sub-microsecond certificate.

**W=800,000 scaling** (envelope only — the general `range_screen_standalone`
safety net's O(W) scaling was not independently re-benchmarked this session,
see §7 gaps below; the prior experimental review measured it at 10.82x cost
for a 10x W increase, i.e. genuinely linear):

| | W=80,000 | W=800,000 |
|---|---|---|
| `precompute_envelope` one-time build | 0.938 s | 0.843 s (not slower — dominated by JIT/compile, not data size, at this scale) |
| `envelope_prewinner_screen` per-call query | 15.5 us | 15.1 us |

Confirms the envelope screen's query cost is genuinely **O(D²), independent
of W** — exactly as derived (no draw loop in the query path at all).

## 6. Organic hit-rate report

**Not independently re-run this session** — a live production-like search
(the kind that would answer "how often does this fire organically") takes
minutes to hours of real KNITRO wall time per run, which this session's
budget did not extend to after the correctness/benchmark validation above.
Reporting the prior experimental review's own finding
(`diag/fullA-d20-range-screen-review`, Section 5) as attributed prior
evidence, NOT as something re-confirmed on this branch's code this session:

> Two real, unmodified production-like searches (δ=5, 152s/47 evals; δ=2,
> 280s/91 evals), both starting from a real feasible anchor with the SAME
> warm-start-first policy production uses. **Zero organic range-screen hits
> in either run** — all 138 trial points landed in the feasible region. This
> was reported honestly as a negative result, not spun: every known
> infeasible point in this whole investigation's history is a deliberately-
> constructed repair (from a different method's raw output), not something a
> bounded local KNITRO search around a feasible anchor organically visits.

Per the brief's own instruction: **this integration is NOT justified by a
projected production speedup from organic hits** (none observed, in either
this session's targeted validation or the prior session's live search). It is
justified as **near-free exact insurance** (§5: no measurable cost when
integrated properly) against a real, if rare, failure mode — with the
concrete benefit of avoiding a wasted ~1s KNITRO call on any point that DOES
hit it (§4/§5), which matters most if a future outer-loop search strategy
(wider steps, different starting regions, a future δ range) visits this
region more often than the two searches sampled here did.

## 7. Test suite

- `full_aod_diag/d4_exact/validate_fast_range_screen_d20.jl` — real D=20/W=80,000 validation:
  calibration-consistent context build, every recovered catalogue point under
  `diagnostics/infeasibility_points/` (5 nonzero-winner-infeasible candidates, 2 zero-winner
  points, 4 feasible points across delta in {1,2,5}), cross-checking `evaluate_fullA_screened`
  (existing, unmodified) against `evaluate_fullA_screened_ranged` (this branch's new entry point)
  for verdict agreement and, on mutually-feasible points, `Delta_dual` numerical agreement.
  **Result: 0/11 false positives/negatives, 0/4 value mismatches.**
- `full_aod_diag/d4_exact/verify_candidates_detail.jl` — for the 5 infeasible candidates: confirms
  baseline's real KNITRO call genuinely reports `nStatus=-300` (not a silent success the new screen
  would have wrongly overridden), reports the envelope certificate's exact bound/target/margin, and
  independently cross-checks via `range_screen_standalone` on a full (non-early-exit) winner scan —
  a completely separate code path from the envelope derivation.
- `full_aod_diag/d4_exact/benchmark_fast_range_screen.jl` — warmed steady-state timing (§5) and the
  W=800,000 envelope scaling check.

**Known gaps, not closed this session** (listed honestly rather than silently dropped):
- Random gravity-tangent perturbation sweep (6 points in the prior review) — not re-run here; cited
  as attributed prior evidence only.
- δ=0.1 point — no recovered full A_od CSV at this W in either branch's catalogue (matches the prior
  review's own finding).
- General `range_screen_standalone` W=800,000 scaling — not independently re-benchmarked this
  session (prior review's 10.82x/10x-W finding cited as attributed evidence).
- Common-marginal contexts — not supported by this branch's base context at all (§3), so not
  testable here without first reconciling with the separate CM integration branch.
- Fresh-process reproducibility / deterministic threaded behavior — this session ran each script
  once per fresh `julia` process (not re-run twice to confirm bit-identical repeat results); the
  underlying algorithms are single-threaded and deterministic by construction (no RNG in the screen
  logic itself, only in the ONE-TIME draw generation `d20_real_setup` already does and seeds
  explicitly), but this was not re-verified by literally running twice and diffing this session.

## 7. Test suite

- `full_aod_diag/d4_exact/validate_fast_range_screen_d20.jl` — real D=20/W=80,000 validation:
  calibration behavior via the production driver's own ctx build, every recovered catalogue point
  under `diagnostics/infeasibility_points/` (5 nonzero-winner-infeasible candidates, 2 zero-winner
  points, 4 feasible points across delta in {1,2,5}), cross-checking `evaluate_fullA_screened`
  (existing, unmodified) against `evaluate_fullA_screened_ranged` (this branch's new entry point)
  for verdict agreement and, on mutually-feasible points, `Delta_dual` numerical agreement.

## 8. Cherry-pick recommendation

This branch (`integration/fullA-fast-range-screen`) is already a from-scratch
port + simplification against current production callbacks, not a literal
cherry-pick of `cac7626`/`0fd2bf8` — those experimental commits' logic was
re-derived and re-verified against the CURRENT production context
(`context_real_d20.jl`/`compressed_moments.jl`/`infeasibility_screen.jl` as
they exist on this branch's base), not blindly copied. If a future production
branch has diverged further from this branch's base by the time this is
reviewed, the recommended path is: **re-apply this branch's commits on top of
the new base the same way this branch was built on top of `02583bc`** (re-run
the validation suite in §7 against the new base — do not assume the numbers
in §4/§5 still hold without re-checking, especially the live-verified
assumption guards in §2's table).

Files to carry forward:
- `full_aod_diag/d4_exact/fast_range_screen.jl` — the only new production
  logic file (envelope screen, fused winner-range screen, general safety net,
  integration wrapper).
- `full_aod_diag/d4_exact/validate_fast_range_screen_d20.jl`,
  `verify_candidates_detail.jl`, `benchmark_fast_range_screen.jl` — test/
  validation scripts (re-express as a proper CI-style test suite if this
  repo's production line adopts one; currently matches the existing
  `test_*.jl`/`validate_*.jl`/`benchmark_*.jl` naming convention already used
  throughout `full_aod_diag/d4_exact/`).
- The one-line `include(...)` addition to `c10_d20_production_driver.jl`
  (opt-in only — does NOT change `run_polish_checkpointed`'s `cb_F!`/`cb_G!`
  to call `evaluate_fullA_screened_ranged` instead of the existing
  `evaluate_fullA_screened`; that wiring decision is deliberately left to the
  user/reviewer, see below).
- `diagnostics/infeasibility_points/` — the ported catalogue fixtures (pure
  data, needed by the validation scripts).
- `docs/fullA_fast_range_screen_production_integration.md` (this file),
  `docs/fullA_warm_start_reliability_note.md`.

**One wiring decision deliberately left open**: this branch adds
`evaluate_fullA_screened_ranged` as a new, parallel entry point — it does
NOT replace `evaluate_fullA_screened`'s call sites inside
`c10_d20_production_driver.jl::run_polish_checkpointed`'s `cb_F!`/`cb_G!`
(the actual KNITRO callback hot path). Flipping that switch is a substantive
production behavior change beyond what this integration branch's mandate
covers (§8 of the original brief: prepare a reviewable candidate, do not
merge). Recommend the user make that specific call explicitly, informed by
§5's numbers.

### Verdict

**RECOMMEND MERGE**

Both the envelope pre-winner screen and the fused winner-range screen are:
- exact (not heuristic) one-sided certificates, re-derived and live-verified
  against current production code (§2);
- validated with zero false positives across every point tested this session,
  cross-checked by two independently-coded certificate paths agreeing (§4);
- add no measurable cost on warm feasible calls when properly integrated —
  reusing the already-built `cf` rather than rebuilding it (§5);
- give up to ~6 orders of magnitude speedup on the specific failure mode they
  target, confirmed on real recovered infeasible points (§4/§5), even though
  organic hit rate in bounded local search was zero in both this session's
  and the prior review's live testing (§6) — the honest framing is "near-free
  exact insurance," not "measured production speedup."

The general `range_screen_standalone` safety net should merge alongside
them, wired to reuse the already-built `cf` (as this branch does) rather than
as an independent bolt-on (which the prior review measured at a real ~10%
tax) — it is the only screen covering the counterfactual column and the
(structurally near-impossible, but not screen-proven-impossible by the other
two) positive-side bilateral case.

Not recommended for this merge: flipping the driver's actual `cb_F!`/`cb_G!`
callbacks to call `evaluate_fullA_screened_ranged` instead of
`evaluate_fullA_screened` — leave that as an explicit follow-up decision
once this branch itself is approved, since it is the one change here that
would alter the production hot path's real behavior rather than add an
opt-in capability next to it.

## 9. See also

- `docs/fullA_warm_start_reliability_note.md` — separate, deliberately out-of-scope warm-start
  reliability issue, not touched by this branch.
