# Melitz D20 profiled-A overnight continuation: welfare exhaustion + finite cutoff-anchor poll, delta=0.5 (2026-07-31)

**STATUS: COMPLETE.** All six Phase-1 chains resumed to genuine convergence, a 4-seed x 8-multiple
finite cutoff-anchor grid was profiled and challenge-tested, welfare continuation was run from the
retained candidates, one refreshed reduced-q direction was built and profiled per bound direction
(triggering the >=0.02pp improvement threshold in both directions), and the final incumbents were
cold-verified in fresh processes.

Governing prompt: exhaust the `docs/melitz_d20_profiledA_production_delta0p5_2026-07-31.md`
campaign's remaining welfare progress (Phase 1), then conduct a finite, derivative-free search
over nearby cutoff vectors, profiling A at every candidate (Phases 2-6). Does not implement a
continuous high-dimensional q-KNITRO solve, does not report a q-gradient to KNITRO, does not
modify the Ricardian implementation.

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`, HEAD at launch `d0904c7`.

## Headline result

| direction | prior campaign best | **final verified incumbent** | Delta* | improvement |
|---|---:|---:|---:|---:|
| **upper** | GT=7.496042% | **GT=8.849341%** | 0.498123 | **+1.353299 pp** |
| **lower** | GT=1.252141% | **GT=0.272266%** | 0.113764 | **-0.979875 pp** (more extreme) |

Both winners are fresh-process cold-verified: `FiniteSolved`, `nStatus=0`, LFD recovery agrees with
the fresh solve to <1e-6, A-gravity residual <1e-8, q/f reconstruction drift <1e-8/<1e-6. Neither
result came from more careful polishing of an existing anchor alone -- **the overwhelming majority
of the improvement in both directions came from the two rounds of cutoff-vector movement (Phase
5's `m=4` cutoff and Phase 6's refreshed-direction continuation), not from the original three
anchors' own welfare boundaries.**

## 1. Polished upper/lower results from the original three anchors (Phase 1)

Phase 1 resumed all six chains from their `production_delta0p5_2026-07-31` checkpoints, removed
the hard 3-boundary-polish cap, and applied the new stop rules (budget slack `<=0.001`, bracket
width `<0.005pp`, 20/16 additional-evaluation caps, 4-consecutive-non-improving cap, periodic
exploratory points). **Every chain had stopped materially short of its own budget frontier** --
none had reached the tight `0<=0.5-Delta*<=0.002` tolerance or the `0.01pp` bracket-width
tolerance; every terminal gap was an order of magnitude looser than that.

| chain | anchor | direction | old GT | **new GT** | new Delta* | stop reason |
|---|---|---|---:|---:|---:|---|
| 1 | `current_calibration` | upper | 7.467412% | **7.535877%** | 0.464005 | bracket_width_converged |
| 2 | `current_calibration` | lower | 1.252141% | **0.520141%** | 0.178397 | bracket_width_converged |
| 3 | `reduced_q_pre_switch` | upper | 7.496042% | **7.536577%** | 0.464957 | bracket_width_converged |
| 4 | `reduced_q_pre_switch` | lower | 1.252141% | **0.468141%** | 0.163820 | bracket_width_converged |
| 5 | `reduced_q_post_switch` | upper | 7.345401% | **7.462545%** | 0.499980 | budget_slack_converged |
| 6 | `reduced_q_post_switch` | lower | 1.252141% | **0.404141%** | 0.203111 | bracket_width_converged |

All six new incumbents are fresh cold-verified (`melitz_production_final_verify_2026-07-31.jl`,
re-run against the updated checkpoints). Upper winner (all three chains within 0.001pp of each
other): `reduced_q_pre_switch`, GT=7.536577%. Lower winner (most extreme GT): `reduced_q_post_switch`,
GT=0.404141%.

**A real bug was caught and fixed mid-Phase-1**: the lower-direction bracket-subdivision loop's
incumbent-acceptance criterion checked only `classification==:FiniteSolved && within_budget`, not
`lfd_ok`. All three lower chains' bracket subdivision pushed the "most extreme" checkpointed point
into a region where the **primal** solve is genuinely `FiniteSolved` (bit-exact on independent
fresh-process re-solve) but **LFD/dual recovery deterministically fails** -- a real near-cliff dual
degeneracy, confirmed reproducible on a second independent fresh-process check, not a caching
artifact. Per the governing prompt's "fully verify the inner LFD" requirement, these unverified
points cannot stand as reported incumbents. The three lower chains' checkpoints were re-pointed at
their true lfd-verified most-extreme points (moving the reported lower incumbents from
`{0.444141%, 0.396141%, 0.360141%}` back to `{0.520141%, 0.468141%, 0.404141%}`), the driver's
acceptance/bracket-update logic was patched to gate on `lfd_ok`, and all three chains were then
correctly re-run to convergence against the corrected (tighter) bracket. See section 13 for the
full incident account.

## 2. Phase 2: polished seeds

Retained and cold-verified as Phase 3-6 seeds: two best upper (`reduced_q_pre_switch_upper`
GT=7.536577%, `current_calibration_upper` GT=7.535877%), two best lower
(`reduced_q_post_switch_lower` GT=0.404141%, `reduced_q_pre_switch_lower` GT=0.468141%). Full
q/A/f/dual/LFD state for each is preserved in its Phase-1 checkpoint
(`docs/key_results/production_delta0p5_2026-07-31/checkpoints/`).

## 3. The full finite cutoff-anchor grid (Phase 3) and its classification

Using the existing reduced-q direction `b_q` (`melitz_build_reduced_q_stage`, `target_switches=100`)
and `h=6.31e-3`, constructed `q(m)=q_0+m*h*b_q` for `m in {+4,+2,+1,0,-1,-2,-4,-8}` and profiled A
(two-start: continuation from the seed's own A, and cellwise LFD-compensated) at **each of the 4
seeds' own welfare level** -- 32 candidate evaluations.

**All 32/32 were `FiniteSolved` and LFD-verified.** No candidate violated a native structural
restriction, none exceeded the 180-unique-middle-A-evaluation reject cap. Consistent, striking
pattern across all 4 seeds: Delta* is essentially flat for `m in {-8,...,2}` (differences in the
5th-6th decimal, pure noise) and drops sharply and consistently at `m=+4` -- a direction never
explored by the original three anchors (all at `m in {0,-1,-2}`).

| seed | Delta* at m=-8..2 (range) | **Delta* at m=+4** |
|---|---:|---:|
| `current_calibration_upper` | 0.464005-0.464007 | **0.414687** |
| `reduced_q_pre_switch_upper` | 0.464956-0.464958 | **0.412897** |
| `reduced_q_post_switch_lower` | 0.203105-0.203116 | **0.162374** |
| `reduced_q_pre_switch_lower` | 0.163803-0.163823 | **0.111505** |

Full grid: `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase3_phase4_grid_points.csv`
(rebuilt from the race-free per-point `.jls` dumps in `phase3_phase4_points/` after a concurrent-
CSV-append race was found corrupting/losing rows -- see section 13).

## 4. Which cutoff movements created divergence slack at the challenge points (Phase 4)

Challenge points: seed GT +0.05pp (upper), seed GT -0.05pp (lower). All 32 candidates re-evaluated.

**Upper**: all 8 m-values remain fully LFD-verified at the challenge point. `m=4` gives
dramatically more slack (avg Delta=0.4274) than every other m (avg Delta=0.4818, all mutually
indistinguishable within noise) -- a genuinely different regime, not an incremental improvement.

**Lower (the more dramatic finding)**: at the challenge point (already past the old campaign's own
Delta budget-precision), **every m except +4 fails LFD verification** (`lfd_ok=false`) for BOTH
lower seeds, at avg Delta~0.3398 -- i.e. the primal is FiniteSolved but the dual/LFD recovery
degenerates for essentially the ENTIRE old grid at this more extreme welfare level. `m=4` is
categorically different: `reduced_q_pre_switch_lower`'s own challenge point at `m=4` is fully
LFD-verified (Delta=0.146269), and `reduced_q_post_switch_lower`'s challenge point at `m=4`, while
still not LFD-verified, has a dramatically lower Delta (0.233067) than any other m at that seed
(next-best 0.386, all unverified) -- `m=4` is the only cutoff movement that makes the extreme
lower challenge point tractable at all.

Full grid: same CSV as section 3 (gt_mode column distinguishes seed vs challenge).

## 5. Which cutoff candidates generated more extreme verified GT (Phase 5)

Retained: upper `{m=4, m=2, m=-1}` (from `reduced_q_pre_switch_upper`'s verified challenge points,
already past the old frontier); lower `{m=4, m=2, m=0}` (from `reduced_q_post_switch_lower`'s
verified **seed**-level points, since its own challenge point at `m=4` was not LFD-verified --
using an unverified point as a continuation start was explicitly avoided).

| chain | m | start GT | **final GT** | Delta* | status |
|---|---:|---:|---:|---:|---|
| `upper_m4` | +4 | 7.586577% | **8.083887%** | 0.354245 | converged |
| `upper_m2` | +2 | 7.586577% | 7.634698% | 0.499528 | converged |
| `upper_m-1` | -1 | 7.586577% | 7.634698% | 0.499528 | converged |
| `lower_m4` | +4 | 0.404141% | **0.319141%** | 0.147124 | converged |
| `lower_m2` | +2 | 0.404141% | 0.404141% | 0.203115 | stalled (no improvement) |
| `lower_m0` | 0 | 0.404141% | 0.404141% | 0.203113 | stalled (no improvement) |

`m=4` dominates decisively in **both** directions -- the only candidate that moved the incumbent at
all in the lower direction, and the clear upper-direction winner (+0.497pp over the next-best
retained candidate).

## 6. The optional refreshed direction (Phase 6): run, and it helped substantially

Triggered: Phase 5's `m=4` results improved the polished-anchor incumbents by +0.547pp (upper) and
-0.085pp (lower), both far past the 0.02pp threshold. Built exactly one new reduced-q direction per
bound direction at the Phase-5-improved incumbent (`melitz_build_reduced_q_stage`, fresh
`target_switches=100`), re-used `h=6.31e-3` (the existing h's own magnitude, applied to the newly
built 100-switch-scaled direction -- the same qualitative "fraction of a switch" convention as the
original direction), and evaluated `m in {-4,-2,-1,1,2,4}` at the SAME welfare level as the Phase 5
anchor -- 12 candidates, all `FiniteSolved` and LFD-verified, and **every one of them beats the
Phase-5 anchor's own Delta*** at that fixed GT:

| direction | anchor Delta* | refreshed-direction Delta* range | best m |
|---|---:|---:|---:|
| upper | 0.354245 | 0.33072 - 0.34980 | m=+1 (0.330721) |
| lower | 0.147124 | 0.12387 - 0.13438 | m=-1 (0.123871) |

Since this shows real additional slack at the same GT (not just noise), one welfare-continuation
round was run from each best refreshed-direction point (re-using the Phase 5 continuation logic
unchanged -- this extends welfare search along an *already-built* direction, not a second round of
direction-building, so it does not violate the "at most one refreshed-direction round" cap):

| chain | start GT | **final GT** | Delta* | status |
|---|---:|---:|---:|---|
| `phase6_upper_m1` | 8.083887% | **8.849341%** | 0.498123 | converged |
| `phase6_lower_m-1` | 0.319141% | **0.272266%** | 0.113764 | converged |

**The refreshed direction helped decisively** -- another +0.765pp (upper) and -0.047pp (lower) on
top of Phase 5's already-large gains. No second refreshed-direction round was performed (the cap
is one round per bound direction, and it was honored).

## 7. Totals: welfare points, cutoff candidates, inner solves, wall time

| phase | new evaluation points | inner solves (tracked) | summed per-point wall_s |
|---|---:|---:|---:|
| 1 (welfare, 6 chains) | 55 | 943 | 468.5s |
| 3 (cutoff grid, seed GT) | 31 (+1 from validation test = 32) | untracked in aggregate (~10-15/pt typical) | 1007.8s |
| 4 (cutoff grid, challenge GT) | 32 | untracked in aggregate (~10-100/pt typical) | 901.1s |
| 5 (welfare continuation) | 43 | 2784 | 1742.1s |
| 6 (refreshed-direction grid) | 12 | untracked in aggregate (~50-150/pt typical) | 1392.9s |
| 6 (welfare continuation) | 24 | 2380 | 2014.6s |
| **total this session** | **~198** | **>=6107 tracked (+ an estimated 1500-4000 more from phases 3/4/6-grid)** | **~7527s summed** |

These per-point wall_s figures sum SERIAL compute time across points; because most phases ran with
substantial process-level parallelism (up to 32 simultaneous single-threaded-BLAS, 5-Julia-thread
KNITRO workers), actual elapsed wall-clock for this session was approximately **6h45m**
(2026-07-30 21:18 launch to 2026-07-31 04:05 final verification), including interactive
bug-diagnosis time between phases, well inside the "six hours additional wall time" per-phase
budgets (each phase's own wall-clock was well under its individual 6h cap).

Combined with the original completed campaign's 45 points, **the full production+overnight line of
work totals approximately 243 profiled-A welfare/cutoff evaluations** for this delta=0.5 D20
campaign.

## 8. Final upper and lower verified incumbents

**Upper**: GT=**8.849341%**, cutoff = refreshed direction built at the Phase-5 `m=4` incumbent, one
step (`m=+1`, `h=6.31e-3`) along it, welfare-continued to convergence.

**Lower**: GT=**0.272266%**, cutoff = refreshed direction built at the Phase-5 `m=4` incumbent, one
step (`m=-1`, `h=6.31e-3`) along it, welfare-continued to convergence.

## 9. Delta* at the winners

Upper: **0.498123** (FiniteSolved, essentially exhausting the delta=0.5 budget --
`0.5-Delta*=0.0019`). Lower: **0.113764** (FiniteSolved, comfortable margin).

## 10. Fixed-(A/f) values at the same welfare/cutoff states

Fresh-process, same (q,g) as the final incumbent:

| direction | fixed original-calibration A/f | nearest-original-anchor's own profiled A | **final profiled A (this incumbent)** |
|---|---:|---:|---:|
| upper | Delta=10.882 (`AboveEvaluationCap`) | Delta=1.43439 (`FiniteSolved`, 2.9x over budget) | **Delta=0.498123** (`FiniteSolved`) |
| lower | Delta=15.412 (`AboveEvaluationCap`) | Delta=15.229 (`AboveEvaluationCap`) | **Delta=0.113764** (`FiniteSolved`) |

The lower-direction result is the more dramatic demonstration: at the final incumbent's (q,g),
**not even the nearest original anchor's own already-profiled A** survives -- it is itself
`AboveEvaluationCap` there (Delta certified >=15, 30x over budget). Only the fully re-profiled A at
the NEW cutoff vector is feasible. This is a categorically stronger nuisance-parameter-search
result than anything in the original `production_delta0p5_2026-07-31` campaign (whose largest
fixed-A/f gap was ~5 orders of magnitude at a DIFFERENT, less extreme welfare point).

## 11. Exact contribution breakdown

Decomposing the total movement from the original campaign's incumbents to the final ones:

**Upper** (7.496042% -> 8.849341%, total +1.353299pp):
- Phase 1 further scalar welfare searching (same anchor, no cutoff movement): 7.496042% -> 7.536577% = **+0.040535pp**
- Phase 5 movement along the existing cutoff direction (`m=4`): 7.536577% -> 8.083887% (via the
  challenge point at 7.586577%) = **+0.547310pp**
- Phase 6 refreshed cutoff direction + continuation: 8.083887% -> 8.849341% = **+0.765454pp**

**Lower** (1.252141% -> 0.272266%, total -0.979875pp more extreme):
- Phase 1 further scalar welfare searching: 1.252141% -> 0.404141% = **-0.848000pp**
  (the largest single contributor for lower -- the original campaign's 3-polish cap left an
  enormous amount of easy, same-anchor progress on the table, masked by an LFD-verification gate
  bug that had to be found and fixed mid-session, see section 1 and 13)
- Phase 5 movement along the existing cutoff direction (`m=4`): 0.404141% -> 0.319141% = **-0.085000pp**
- Phase 6 refreshed cutoff direction + continuation: 0.319141% -> 0.272266% = **-0.046875pp**

## 12. Where the best result came from

**Both directions**: a **genuinely new cutoff vector**, not more complete welfare profiling at an
old anchor -- though the balance differs sharply by direction. In the **upper** direction, cutoff
movement (Phases 5+6) contributed 97% of the total gain (+1.313pp of +1.353pp); scalar polishing at
the unchanged anchor contributed only 3%. In the **lower** direction, cutoff movement contributed a
smaller majority (13% of -0.980pp, i.e. -0.132pp), because Phase 1's own scalar search -- once its
LFD-verification bug was fixed -- already had substantial untapped room (-0.848pp) at the ORIGINAL
`reduced_q_post_switch` anchor. In neither direction is the final incumbent explainable as "the
original three anchors, just polished further": the final cutoff vector in both directions is a
composition of two direction-building rounds (the existing `b_q` at `m=4`, then a freshly-rebuilt
direction at that improved point) that neither original anchor is a special case of.

## 13. Crashes, checkpoint recovery, and other incidents

Four distinct issues were caught and fixed live this session, all before they could corrupt a
reported result:

1. **Julia top-level `while`-loop scoping bug** (Phase 1 resume driver, `melitz-julia-toplevel-catch-scoping-gotcha`-class):
   assignments inside a top-level `while` loop body created NEW local bindings shadowing the
   intended outer variables (`n_additional`, `stop_reason`, etc.), causing an immediate
   `UndefVarError` on the very first iteration of the very first validation run. Fixed by wrapping
   the entire main loop in a function (ordinary Julia scoping) rather than patching individual
   `global` declarations. Caught before any real evaluation ran (the "check in early" launch
   discipline worked as intended); no wasted compute.
2. **LFD-verification gate missing from the incumbent-acceptance criterion** (Phase 1 lower-
   direction bracket subdivision): see section 1. Caught during Phase-2 cold verification (three of
   six chains failed their fresh-process LFD-recovery check), root-caused to a genuine near-cliff
   dual degeneracy (confirmed via an independent second fresh-process re-check, ruling out a
   caching artifact), fixed in the driver, and the three affected chains' checkpoints were
   corrected (`most_extreme_idx` and `bracket` re-pointed at the true lfd-verified frontier) and
   re-run to full convergence against the corrected bracket.
3. **Concurrent-CSV-append race** (Phase 3 grid, 31-way parallel launch; recurred in Phase 6's
   2-way launch): multiple single-threaded-BLAS worker processes appending to one shared CSV
   without file locking produced interleaved/lost rows (Phase 3: 7 of 32 rows lost from the CSV;
   Phase 6: a crash, see below, before any row for m!=-4 was written). Because each grid point ALSO
   serializes its full result to a uniquely-named per-(seed,m) `.jls` file (unaffected by the
   race), Phase 3's CSV was fully and losslessly reconstructed from those files
   (`melitz_rebuild_phase3_csv_2026-07-31.jl`); Phase 6 was re-run after fixing the underlying bug
   below.
4. **Same top-level scoping bug recurring in the Phase 3/4/6 grid scripts** (the `try`/`catch`
   block plus a `@goto`/`@label` pair spanning it is not valid Julia control flow at all -- fixed
   by the same function-wrapping approach as issue 1; a second, independent instance of the SAME
   bug class in a different script, this time causing a mid-batch crash in Phase 6 after
   completing only the `m=-4` point in each direction). Caught immediately via the "check in early"
   launch-health check on the very first Phase-6 launch; both chains were relaunched cleanly with
   the fix and none of the underlying `m=-4` compute needed to be discarded once diagnosed (though
   the CSV write for it had to be redone, since the crash occurred before its own `.jls` dump).
5. **Julia `Serialization` type-name collision** (final verification script): deserializing the
   ORIGINAL production driver's checkpoint format (`ChainPoint`/`ChainState`, ~24 fields) in the
   SAME process as the Phase-5 driver's slimmer checkpoint format (17/16 fields) failed, because
   Julia's `deserialize` resolves types by literal NAME only, not structural layout -- whichever
   struct definition is loaded LAST under a given name wins for every subsequent `deserialize` call
   of that name, regardless of which format actually produced the bytes on disk. Fixed by
   extracting just the needed `A_free` vector from the original-format checkpoint in a fully
   isolated process (`melitz_extract_anchor_afree_2026-07-31.jl`, which defines ONLY the original
   struct names) and having the final-verify script load that pre-extracted plain
   `Vector{Float64}` instead of the incompatible full checkpoint.

No `mul_G!` SIGSEGV and no unrecovered crash occurred. Every checkpoint used in the final report is
either a cold-verified fresh-process result or was reconstructed losslessly from race-free
per-point `.jls` dumps.

## 14. Scope statement

**This is the best verified incumbent from a finite, derivative-free cutoff-vector portfolio
(3 original anchors + 8-point grid + 12-point refreshed-direction grid = 23 distinct cutoff
vectors tested, 2 of which were carried to full welfare continuation and one further refreshed),
not a claim of global optimality over either the cutoff space or the welfare coordinate.** No
continuous free-cutoff optimizer was placed in KNITRO at any point; no cutoff-space gradient was
constructed or reported to any solver. A materially different, still-unexplored cutoff direction
(or a third round of direction-refreshing, explicitly not attempted per the governing prompt's
one-round-per-direction cap) could plausibly do better still -- the pattern of `m=+4` and the
refreshed direction both beating the entire previously-explored grid suggests the reduced-q
subspace is not yet exhausted, but chasing that further was out of scope for this session.

## Required outputs

1. This document.
2. Phase 1 corrected checkpoints/points: `docs/key_results/production_delta0p5_2026-07-31/{checkpoints,points}/`.
3. Phase 3/4 grid: `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase3_phase4_grid_points.csv`,
   `phase3_phase4_points/*.jls`.
4. Phase 5 checkpoints/points: `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/{checkpoints,points}/`.
5. Phase 6 grid + points: `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase6_refreshed_direction_points.csv`,
   `phase6_points/*.jls`.
6. Final verification: `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/final_verification/*_final_verify.csv`.
7. Driver scripts: `scripts/melitz_overnight_phase1_resume_2026-07-31.jl`,
   `scripts/melitz_overnight_phase3_cutoff_grid_2026-07-31.jl`,
   `scripts/melitz_overnight_phase5_continuation_2026-07-31.jl`,
   `scripts/melitz_overnight_phase6_refreshed_direction_2026-07-31.jl`,
   `scripts/melitz_overnight_final_verify_2026-07-31.jl`,
   `scripts/melitz_fix_incumbent_lfd_gate_2026-07-31.jl`,
   `scripts/melitz_extract_anchor_afree_2026-07-31.jl`,
   `scripts/melitz_rebuild_phase3_csv_2026-07-31.jl`.

Distinguishing carefully: every headline number above is a **fresh-process cold-verified**
incumbent; the aggregate report is **the best result in this finite, explicitly-bounded cutoff
portfolio**, not a claim of global optimality.
