# Exact early infeasibility screening for hard-winner outer points

Continuation 9, branch `c9-infeasibility-screen` (forked from `diag/fullA-d4-exact` @
`b200eda`). Implements the user's verbatim specification for a draw-free/cheap-first
exact rejection screen that catches outer points where the primal moment problem is
structurally infeasible (some positive-target-share origin-destination pair has zero
possible winners on the fixed simulation draws), before constructing moments or
invoking the inner CC dual solver (KNITRO). Measured on `demand.mit.edu`,
`JULIA_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

**Non-negotiable correctness requirement, confirmed met**: every rejection this screen
issues is an EXACT sufficient condition for infeasibility, never a heuristic. Across
every validation battery run in this report — D=4: 7 registered candidates + 300
random-perturbation trials (60 small + 200 large + 40 early-exit-vs-full-scan) + 592
extreme-draw-witness (o,d) cross-checks + 9 integration checks; D=6/8/10: 12 checks;
D=20/W=80,000: 4 known points + 150 perturbation trials + 5-class workload benchmark;
D=20/W=8,000: 4 historical points — **zero false positives were found**: the screen
never rejected a point the exact winner-construction ground truth (or the real
dense/compressed oracle) would have called feasible.

---

## 1. Code-audit mapping table (spec math -> this codebase)

Built BEFORE any implementation, per the task's explicit instruction, and empirically
validated (not assumed) — see §2. Read `full_aod_diag/d4_exact/winners.jl`,
`compressed_moments.jl`, `oracle.jl`, `oracle_fast.jl`, `compressed_live.jl` first if
extending this further; the header of the new file
`full_aod_diag/d4_exact/infeasibility_screen.jl` reproduces this table verbatim.

| Spec symbol | This codebase | Notes |
|---|---|---|
| hard score `S_sod` | NOT a pre-existing named quantity anywhere in this codebase; derived here as `S_sod := -log(price[s,o,d])/mu` from `winners.jl::factual_prices`'s price formula. Strictly increasing in `-log(price)` since `mu>0`, so `argmax_o S_sod == argmin_o price[s,o,d] ==` `winners.jl::compute_winners`'s `winner[s,d]`. | Confirmed bit-for-bit against `compute_winners` AND `compressed_moments.jl::build_compressed_factual`'s own winner arrays (0/32000 mismatches, two independent D=4 points). |
| `B_so` (fixed across optimization) | `-log.(ctx.U[s,o])` — **pure data**, zero `theta`-dependence at all (not merely "fixed at the current outer point" but literally invariant across the ENTIRE outer loop, since `ctx.U` is baked in at context construction). | `hard_score_B(ctx)` in `infeasibility_screen.jl`. |
| `a_od` (almost certainly `log(A_od)`) | `log(Aod_theta[o,d]) + log(lambda[o,d]) - log(lambda[1,d]) - (1/mu)*log(wHat[1,1]*tau[1,d])`, where `Aod_theta = reshape(theta_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)` is exactly the free A-block parameter this whole investigation optimizes over. | The guess was right up to fixed, `theta`-independent data terms (`lambda`, `wHat`, `tau`) that must be folded in somewhere for the additive `B+a` split to hold exactly — `mu`, `lambda`, `wHat`, `tau` are all FIXED at every outer point in this investigation's outer loop (which varies only `(gamma'_focal, A_od)`), so `a_od` is recomputed fresh every outer point exactly as `A_od` itself is. `compute_a_od(theta_full, ctx)`. |
| target trade share | `Pmat[o,d] = ctx.gamma.P[d+(o-1)*D]` — the observed bilateral expenditure-share matrix (lambda/pi). For D=20 real data this is the `pi.csv`-derived matrix loaded via `d20_real_setup`'s `fakeData==3` path; for D=4/6/8/10 synthetic it's `createFakeData`'s synthetic lambda. **Same field, no D=20-special-case code needed.** Confirmed bit-identical to the `lambda` matrix `compressed_moments.jl::build_compressed_factual` itself uses. | `target_shares(ctx)`. |
| `M_ok = max_s(B_so - B_sk)` | `precompute_pairwise_M(ctx) -> PairwiseCertificate.M` (D x D, `O(D^2 W)` once, draw-free at every SUBSEQUENT outer point). | |
| `a_kd - a_od > M_ok` rejection test | `pairwise_certificate(a, pc, Pmat)`. | |
| slack `m_od` | `pairwise_certificate(...).m_od`, `= min_{k!=o}[M_ok-(a_kd-a_od)]` for `Pmat[o,d]>0`, else `+Inf`. | |
| hard winner construction | `screen_hard_winners(theta_full, ctx, Pmat; order, full_scan)` — a **direct, attributed reuse** of `compressed_moments.jl::build_compressed_factual`'s winner-finding inner loop (`constCons`, `constConsσ`, `UPow`, `UσPow` formulas copied verbatim, not re-derived), NEW: destination-major processing in caller order with immediate per-destination zero-win rejection, and a **tie-safe** win-count pass (see §3). | |

### 1.1 Empirical validation of the mapping (before any screen code was built)

Script (scratch, not committed — reproduced here for the record):
`compute_winners`/`build_compressed_factual`'s winner arrays vs
`argmax_o(B[s,o]+a[o,d])` at two independent D=4 points (calibration, one random A
perturbation): **0/32000 mismatches** at each point, and `Pmat == lambda` (the matrix
`compute_a_od` uses) bit-for-bit. Only after this passed was any rejection logic built.

---

## 2. What was implemented

New file `full_aod_diag/d4_exact/infeasibility_screen.jl` (fully additive — zero lines
touched in `winners.jl`, `winners_v2.jl`, `compressed_moments.jl`,
`compressed_cc_inner.jl`, `compressed_live.jl`, `lfix_incremental.jl`, `oracle.jl`,
`oracle_fast.jl`):

1. **Draw-free pairwise certificate** (spec §2): `precompute_pairwise_M`,
   `compute_a_od`, `pairwise_certificate`, tolerance `tol=1e-9` applied ONLY on the
   rejecting side (so it can only make the certificate MORE conservative, never
   introduce a false rejection).
2. **Destination-by-destination hard-winner construction with immediate zero-count
   rejection** (spec §1): `screen_hard_winners`, deterministic vulnerability ordering
   `order_destinations` (ascending analytical slack `m_od`, the spec's third listed
   heuristic) plus an alternative `order_destinations_by_margin` (the spec's
   "previous winner margins" heuristic, provided for completeness, not the default).
3. **Extreme-draw witness query** (spec §3, optional/additive):
   `build_extreme_draw_witness`, `query_witness` — see §6 for the adopt/don't-adopt
   verdict.
4. **Integration** (spec §4): `evaluate_fullA_screened` (dense fall-through, zero
   change to `evaluate_fullA_fast`) and `evaluate_fullA_screened_compressed` /
   `compressed_factual_from_screen` — a screen-aware compressed path that reuses the
   screen's own already-computed winner/wval arrays instead of paying
   `build_compressed_factual`'s own redundant `O(W*D^2)` winner rescan (a genuine,
   demonstrable production win beyond what the spec strictly required — see §7).
   Returns a structured result with `Delta_dual=Delta_primal=Inf` and an
   `inner_status` sentinel (`-9001`/`-9002`/`-9003`, one per certificate stage) that
   is provably outside the set of real KNITRO status codes ({0,-100,-101,-103}
   solved; {-300,-400,...} numerical failure) — distinct from a numerical
   inner-solver failure by construction, not just convention. Infeasible points are
   cached in the SAME `FullAEvalKey`-keyed dict feasible points use.

New test file `full_aod_diag/d4_exact/test_infeasibility_screen.jl` (9 sections, 72
checks, all pass) and 4 new benchmark/validation drivers (`c9_infscreen_d20_validate.jl`,
`c9_infscreen_d20_workloads.jl`, `c9_infscreen_d10_check.jl`,
`c9_infscreen_witness_w800k.jl`).

---

## 3. A real correctness subtlety found and fixed: tie-safe win counting

`compute_winners`/`build_compressed_factual` record a SINGLE `winner[s,d]` index per
draw (first-index-wins under an exact tie, matching Julia's `findmin`/strict-`<`
convention). But the actual dense production winner rule
(`hFunction.jl`'s inline `MinInd!`) **splits mass across every origin tied at the
row-minimum price**, not just the first-indexed one. A win-COUNT that credited only
`winner[s,d]`'s first-index choice would UNDER-count wins at an exact tie — an origin
whose only route to winning a destination is via a tie could be (incorrectly) flagged
zero-win, a genuine false-positive risk. `screen_hard_winners` therefore runs a
SECOND, tie-safe pass per draw crediting every origin at the exact row-minimum
(mirroring `lfix_incremental.jl::detect_price_ties`'s own `<=`-based tie-detection
convention), at a modest extra cost (roughly doubling the origin-scan per draw,
matching `build_compressed_factual`'s own `check_ties=true` cost).

**Validated directly** (test §6): a deliberate exact tie was constructed (forcing
`Aod_theta[1,1] == Aod_theta[2,1]` bit-for-bit) and confirmed the tie-safe count
credits BOTH tied origins with `win_counts > 0`, where a naive first-index-only count
would have incorrectly zeroed one of them. Ties are a probability-zero event for
continuous Frechet draws in real data (never observed anywhere in this investigation
per `docs/fullA_fully_compressed_inner_report.md`), so this is a genuine
belt-and-suspenders exactness fix, not something expected to matter in practice — but
required given the "zero false positives, no approximation" standing instruction.

---

## 4. Validation results

### 4.1 D=4 (`test_infeasibility_screen.jl`, all 7 registered candidates + random sweeps)

| Check | Result |
|---|---|
| 7 registered candidates (calibration, upper_maxit15/40, lower_stalled, lower_v2, lower_lfixcomposite_fast_sr1_300s, upper_lfixcomposite_sr1_60s) | **0/7 false positives** (neither pairwise nor winner-scan ever rejects) |
| 60 small perturbations (\|step\| in [0.01,0.3], the outer-search's typical local step scale) | 0 pairwise-infeasible, 0 winner-infeasible — no false positives possible |
| 200 LARGE perturbations (\|step\| in [2,10]) | 13/200 pairwise-certified infeasible, 19/200 exact-winner-infeasible, **0 disagreements (0 false positives)**; pairwise catches 13/19 = **68.4%** of the genuine infeasible points found here (the rest are real false negatives, expected/acceptable) |
| Order-independence (natural vs `randperm` order, feasible point) | feasible flag, winner matrix, wval matrix, win_counts all **bit-identical** |
| Early-exit vs full-scan agreement, 40 large-perturbation trials | **0 mismatches** |
| Tie-safety (deliberate exact tie) | confirmed BOTH tied origins retain `win_counts>0` |
| Extreme-draw witness vs exact ground truth | 112 (o,d) pairs (7 feasible points) + 480 pairs (30 large-perturbation points) = **592 total, 0 mismatches** |
| `evaluate_fullA_screened` (dense) vs `evaluate_fullA_fast` at 7 feasible points | inner_status, Delta_dual (or NaN==NaN, see note below), winner_hash all match |
| `evaluate_fullA_screened` (`:compressed`) vs dense direct | Delta_dual matches to 1e-8, winner_hash matches |
| Infeasible-point integration (structured status) | `Delta_dual==Inf`, `inner_status` sentinel `<-9000`, **zero new KNITRO solves performed** |

**Total: 72/72 checks pass.** (One process note: the D=4 `d4_exact_setup()`
calibration point's plain oracle call itself returns `inner_status=-300,
Delta_dual=NaN` — a pre-existing property of this specific synthetic setup, confirmed
identical between the direct and screened call paths; `NaN != NaN` under Julia's
`isapprox` required a `dd_match` helper in the test, not a screen behavior change.)

### 4.2 D=6/8/10 (`c9_infscreen_d10_check.jl`, synthetic `d_exact_setup_scaled`)

Calibration + 3 gravity-tangent perturbations at each of D=6, 8, 10 (12 checks total):
**0 false positives**, and the calibration point's `evaluate_fullA_screened` result
matches `evaluate_fullA_fast` exactly at every D. *(Pragmatic substitute noted
honestly: reproducing the LITERAL archived D=10 upper-gate optimizer output would
require re-running a full KNITRO outer-loop optimization, out of this task's time
budget; each dimension's own calibration point is the same "known-feasible reference
class" Continuation 8's D=6/8/10 gated pilots themselves used and reported feasible
"on the first try.")*

### 4.3 D=20 real data, W=80,000 (`c9_infscreen_d20_validate.jl` Section A)

The 4 known-feasible outer points from `docs/fullA_D20_W80k_microbenchmark.md`
(calibration, gravity-tangent, upper-branch offset=0.01, lower-branch offset=0.01),
reconstructed identically:

| point | pairwise.infeasible | winner_scan.feasible | worst_slack |
|---|---|---|---|
| calibration | false | true | 2.156 |
| gravity_tangent | false | true | 2.1575 |
| upper_branch_offset01 | false | true | 2.156 |
| lower_branch_offset01 | false | true | 2.156 |

**0/4 false positives.**

### 4.4 D=20 real data, random A-perturbation sweep, W=80,000 (Section D, pure-screen cost, 150 trials)

Step sizes 0.5/1/2/4/8/16 (unit-norm random direction in pivot-reduced z-space), 25
trials each:

| step | N | pairwise-infeasible | winner-infeasible |
|---|---|---|---|
| 0.5 | 25 | 0 | 0 |
| 1.0 | 25 | 0 | 0 |
| 2.0 | 25 | 0 | 0 |
| 4.0 | 25 | 0 | 0 |
| 8.0 | 25 | 0 | 0 |
| 16.0 | 25 | **2** | **2** |

**Zero false positives confirmed** (the script `error()`s immediately if any are
found — none were, across all 150 trials). Interesting D=20-specific finding: unlike
D=4 (where pairwise catches ~68% of genuine infeasible points, §4.1), at D=20 **the
pairwise certificate caught 2/2 = 100%** of the genuine zero-winner points found in
this sweep — Section E's `class3_src` search (points winner-infeasible but NOT
pairwise-certified) came up **empty** (0 candidates found among 150 trials). This is a
real, D-scale-dependent finding, not assumed: at D=20 the pairwise certificate's
sufficient condition is evidently tight enough to catch every genuine failure this
particular perturbation family produces — plausibly because breaking the winner set
for one destination among 400 candidate coordinates, via a single global random
direction, tends to produce a "clean" single-rival domination (exactly what the
pairwise test is built to catch) rather than the more delicate multi-rival
intersection failure the pairwise test can miss (§4.1's D=4 false negatives). This
should NOT be read as "pairwise is exact at D=20" (it provably is not, in general —
the extreme-draw witness's own exact intersection logic exists precisely because
pairwise is only a sufficient, not necessary, condition) — only that THIS sweep's
perturbation family didn't happen to construct the harder case at D=20. Also notable:
D=20 needed a step size (16, in unit-norm pivot-reduced z-space over 399
coordinates) an order of magnitude larger than D=4 needed (2-10) before ANY
infeasibility appeared at all — consistent with D=20's much larger redundant
coverage (20 origins × 400 bilateral A-entries vs D=4's 4×15) making it structurally
harder to break any single destination's winner set with a single global
perturbation.

Early-exit savings: at step=16, the 2 genuine infeasible points were rejected at
**stage 1 of 20 destinations** (median and mean both 1.0/20) — the deterministic
vulnerability ordering (`order_destinations`) found the failing destination on the
very first try both times.

---

## 5. The retroactive W=8,000 finding (directly answers the task's flagged question)

`docs/fullA_D20_production_path_audit.md` / the Continuation 9 interim handoff flagged
that **W=8,000 was found uniformly infeasible/unbounded (`inner_status=-300`) at
every tested point regardless of delta (tested to delta=1e6) at D=20 real data**, with
the mechanism never diagnosed exactly — plausibly (not confirmed) the exact zero-winner
structural infeasibility this screen targets.

**This screen retroactively confirms the hypothesis.** Testing the SAME 4 points
(calibration, gravity-tangent, upper-branch, lower-branch) reconstructed at W=8,000
instead of W=80,000:

| point (W=8,000) | pairwise.infeasible | winner_scan.feasible | zero-win positive-share pairs | real dense `inner_status` |
|---|---|---|---|---|
| calibration_W8000 | **true** | **false** | 5 / 400 | **-300** |
| gravity_tangent_W8000 | **true** | **false** | 5 / 400 | **-300** |
| upper_branch_W8000 | **true** | **false** | 5 / 400 | **-300** |
| lower_branch_W8000 | **true** | **false** | 5 / 400 | **-300** |

**All 4 points are pairwise-certified EXACTLY infeasible** — 5 of the 400
positive-target-share bilateral pairs have literally zero winning draws among the
8,000 available at that origin/destination — and all 4 real dense oracle calls
independently return `inner_status=-300`, exactly consistent. **This is a clean,
directly-relevant confirmation**: the historical W=8,000 "-300 everywhere" phenomenon
IS the zero-winner structural infeasibility this screen is built to detect, not a
generic "too-few-draws-for-numerical-conditioning" phenomenon as the alternative
hypothesis framed it. Practical implication: this screen would have caught and
explained this operational finding for free, in milliseconds, the first time it was
encountered, rather than requiring the multi-hour W-sweep investigation that
originally resolved it operationally (by moving to W>=80,000) without ever pinning
down why.

---

## 6. Extreme-draw witness (spec §3): adopted?

Benchmarked at D=20, calibration point, W=80,000 (`c9_infscreen_d20_witness_only.jl`)
and W=800,000 memory/preprocessing-only (`c9_infscreen_witness_w800k.jl`, per the
spec's own W=800,000 scope limit).

### 6.1 W=80,000

| metric | value |
|---|---|
| preprocessing (`build_extreme_draw_witness`) wall | **2.47s** |
| structure memory (theoretical, `D*(D-1)*W*(4+8)` bytes) | **0.365 GB** |
| VmHWM before -> after build | 2.76 GB -> 2.76 GB (no measurable delta — within GC/measurement noise of the ~0.36GB structure against a multi-GB process baseline) |
| queries (all 400 positive-share `(o,d)` pairs at calibration) | **N=400**, all exact-match ground truth (0 mismatches) |
| query time | median **15.0us**, mean 138.6us (skewed by one 49.3ms outlier — a query whose smallest-rival candidate set happened to be unusually large), max 49.3ms |
| candidate-set size (best rival's count) | median **293.5**, mean 4430.3, out of W=80,000 — **100% of queries resolved touching fewer than W draws** (median candidate set is 0.37% of W) |
| full destination-major scan (`screen_hard_winners`, feasible path) | 302.9ms/outer-point |
| witness-based cost, querying ALL 400 positive-share pairs | **6.01ms/outer-point** (**50.4x cheaper** than the full scan, once built) |
| break-even (one-time 2.47s build vs the 296.9ms/point saving) | **~8.3 outer-point evaluations** |

**Verdict at W=80,000: the witness structure is a clear, real win** for any workload
that evaluates more than a handful of outer points against the SAME context (i.e. any
actual outer-loop optimization run, which evaluates hundreds to thousands of points) —
50x cheaper per feasible-path screen check than the direct destination-major scan,
amortizing its one-time build cost within under 10 evaluations.

### 6.2 W=800,000

Memory/preprocessing-only, per the spec's explicit scope limit and this
investigation's standing memory-safety discipline (`c9_infscreen_witness_w800k.jl`,
every step checked against a self-imposed 40GB kill threshold before proceeding —
never approached: peak observed was **16.93 GB**, matching the earlier
`docs/fullA_D20_W800k_microbenchmark.md` finding of ~16.55GB for the plain context
build almost exactly, confirming this task added no new memory risk).

| metric | value |
|---|---|
| `d20_real_setup(W=800000)` wall | 179.2s |
| VmHWM after setup | 16.93 GB |
| `precompute_pairwise_M` wall | 0.94s (VmHWM unchanged) |
| pairwise certificate at calibration | `infeasible=false` (worst_slack=3.55) — **confirms zero false positive at W=800,000 too** |
| `screen_hard_winners` (feasible path, full D=20 destinations) wall | 3.78s |
| witness structure theoretical memory estimate | **3.648 GB** |
| `build_extreme_draw_witness` wall | **32.52s** |
| VmHWM after witness build | **16.93 GB — unchanged from before the build** (the 3.65GB structure fit inside already-allocated headroom / GC-reclaimed transient setup allocations; no growth in the process's own high-water mark) |
| query time (N=200 probes) | median **31.0us**, mean 272.2us, max 47.9ms |

**No memory concern at W=800,000**: the witness structure's own footprint (3.65GB
theoretical) never even registered as a NEW peak against this process's existing
16.93GB baseline — nowhere near the self-imposed 40GB threshold, let alone this
machine's actual capacity (2.6TB free at the time of this run, confirmed via `free -h`
before launching).

### 6.3 Adoption verdict

**Adopted as an available, opt-in mechanism at both W=80,000 and W=800,000.** The
spec's own stated bar ("do not adopt at W=800,000 unless measured savings justify its
memory") is met: memory cost is negligible relative to the context's own footprint at
both scales, preprocessing cost (2.47s / 32.5s) is a small fraction of a single
`evaluate_fullA_fast` cold solve (4.6-13.4s at W=80,000 per §7), and the per-query
cost stays in the tens-of-microseconds range at both W (15.0us median at W=80,000,
31.0us median at W=800,000 — growing sub-linearly with W, consistent with the
`O(log W)` binary-search-dominated cost model). Given §6.1's ~8-outer-point break-even
against the direct full-scan and the fact that a real outer-loop optimization run
evaluates hundreds to thousands of points against the SAME fixed draw set, this is a
clear net win for any workload that re-screens many points at one context — the
natural use case this whole task is built for. **Not wired into the default
`evaluate_fullA_screened` path in this pass** (which uses the plain
`screen_hard_winners` full-scan by default) — offered as an available, separately
invocable primitive (`build_extreme_draw_witness`/`query_witness`) a caller can
opt into for a long-running outer-loop driver, matching the spec's "as an additive
optional diagnostic" framing rather than promoting it to a load-bearing default
sight unseen in an actual outer-loop driver.

---

## 7. Realistic infeasible-workload benchmark (spec §5)

`c9_infscreen_d20_workloads.jl` Section E, D=20 real data, W=80,000. Five classes
requested by the spec; class 4 required an explicit search (moderate-magnitude
perturbations, screen-passing, probed against the REAL dense KNITRO solve).

| class | N | rejected at pairwise | rejected at winner-scan | t_screen (median) | real solve outcome |
|---|---|---|---|---|---|
| 1. feasible (calibration, gravity-tangent, 3 small perturbations) | 5 | 0 | 0 | ~240-250ms | all `inner_status=0`, real wall 4.9-13.4s |
| 2. pairwise-certified zero-winner | 2 | **2** | n/a (never reached) | **0.02-0.04ms** | not attempted — screen rejects before any KNITRO call |
| 3. zero-winner NOT caught by pairwise | 0 found | — | — | — | none found in this sweep — see §4.4's honest discussion of why |
| 4. positive-winner-count but moment-infeasible | 2 found (of 6 probed) | 0 | 0 (screen passes both) | ~240ms (passes) | real dense solve: `inner_status=-300` despite exact structural feasibility, wall 3.76-5.48s |
| 5. feasible outside the divergence budget (`δ=0.001` context) | 1 | 0 | 0 | ~268ms | `inner_status=0`, `Delta_dual=0.00259 > δ=0.001` — genuinely outside budget but exact-feasible and numerically well-posed |

**Class 2 headline number**: rejecting a pairwise-certified point costs **0.02-0.04ms**
versus a feasible point's real solve costing **4.9-13.4 seconds** (dense, cold/warm
mixed) — a **>100,000x** time-to-rejection speedup for this class, and (by
construction) 100% of the moment-construction and KNITRO work is avoided, not merely
sped up.

**Class 4 is the important limiting case, reported honestly**: 2 of 6 moderately
perturbed points that PASS the exact zero-winner screen (every positive-share pair has
>=1 possible winner) still return `inner_status=-300` from the real KNITRO inner
solve. **This is expected and correct, not a screen failure** — the spec is explicit
that this screen targets ONLY the zero-winner structural infeasibility, not the
broader class of numerical/conditioning failures the CC dual solve can hit for other
reasons. These 2 points cost the full ~4-5.5s wall time regardless of screening
(the screen cannot and should not reject them, since they are not exactly infeasible
in the sense this task defines) — they establish the ceiling of what an exact
zero-winner screen can save: real class-4 failures are NOT preventable by this screen,
only class-2/3 (genuine zero-winner) failures are.

**Class 5** confirms the screen correctly does NOT conflate "exactly feasible" with
"within the divergence budget" — an exact-feasible point with `Delta_dual` above a
tight `δ` still passes the screen and is evaluated normally (as it should: this is not
a structural infeasibility, and the spec is explicit that "do not change the economic
feasible set" applies here too).

---

## 8. What got implemented vs investigated-but-not-adopted

- **Adopted, live, opt-in, wired into the default `evaluate_fullA_screened` path**:
  pairwise certificate, tie-safe destination-by-destination winner scan with
  immediate rejection + deterministic ordering, structured exact-infeasibility status
  + shared cache, screen-aware compressed integration (avoids the compressed path's
  redundant winner rescan on BOTH the infeasible AND feasible paths).
- **Adopted as an available, opt-in primitive, NOT wired into `evaluate_fullA_screened`'s
  own default path**: extreme-draw witness query (§3 of the spec) — real, measured net
  win at BOTH W=80,000 (50x cheaper per point, ~8-point break-even) and W=800,000
  (memory-safe, 32.5s one-time build, ~31us/query) — see §6 for the full numbers and
  the reasoning for keeping it a separately-invocable primitive rather than a default.
- **Not implemented** (explicitly out of scope per the spec / time budget): a
  genuinely restructured destination-batched moment kernel sharing base-score
  computation across coordinates (would touch `lfix_incremental.jl`'s internals, out
  of this task's additive-only scope); wiring the screen into any production driver's
  DEFAULT settings (this task delivers an opt-in `evaluate_fullA_screened`, callers
  must choose to use it, matching the spec's "keep changes behind an opt-in flag").

---

## 9. Files

New, all under `full_aod_diag/d4_exact/`:
- `infeasibility_screen.jl` — the screen itself (pairwise certificate, winner-scan,
  witness, integration).
- `test_infeasibility_screen.jl` — 72-check correctness battery (D=4).
- `c9_infscreen_d10_check.jl` — D=6/8/10 zero-false-positive check.
- `c9_infscreen_d20_validate.jl` — D=20/W=80,000 known-point check + W=8,000
  retroactive check + basic timing.
- `c9_infscreen_d20_workloads.jl` — D=20/W=80,000 perturbation sweep (Section D) +
  5-class workload benchmark (Section E) + witness benchmark (Section F, superseded
  by the standalone rerun below after a dead-code bug in the original Section F was
  found and fixed post-hoc).
- `c9_infscreen_d20_witness_only.jl` — standalone rerun of Section F alone (W=80,000
  witness benchmark), used instead of re-running the expensive D+E sections a second
  time after the bug fix.
- `c9_infscreen_witness_w800k.jl` — W=800,000 memory/preprocessing-only probe.

Modified: **none** (fully additive, per the spec's "small, additive, reviewable
commits ... new files preferred" instruction).

Raw logs/CSVs: `results/fullA_d4/<commit>/c9_infscreen_d20/`,
`results/fullA_d4/<commit>/c9_infscreen_d20_workloads/` (`sectionD_perturbation_sweep.csv`,
`sectionE_workload_benchmark.csv`, `sectionF_witness_benchmark.csv`).
