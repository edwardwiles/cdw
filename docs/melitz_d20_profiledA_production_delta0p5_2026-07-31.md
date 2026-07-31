# Melitz D20 profiled-A production campaign, delta=0.5 (2026-07-31)

**STATUS: COMPLETE.** All six chains ran to a terminal status, were cold-verified in fresh
processes, and are reported below.

Governing prompt: launch a restartable production search of the successful profiled-(A) D20
Melitz method (`docs/melitz_d20_profiled_A_welfare_continuation_2026-07-30.md`,
`docs/melitz_profiledA_parallel_speed_and_cutoff_portfolio_2026-07-30.md`) and produce the best
verified incumbents obtainable within a fixed overnight budget. Does not test whether profiling
works (already established), does not build another cutoff-gradient method/chamber
graph/parameterization/direct high-dimensional welfare optimizer, does not modify the Ricardian
implementation, does not continuously optimize the free cutoff coordinates.

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`, HEAD at launch `169f77d`.

## Headline result

Across 6 chains (3 cutoff anchors x 2 directions), **45 welfare points evaluated, 3536 unique
inner solves, 1795.7s (~30 minutes) total wall time** (well under the 6h-per-chain / 40-point
budgets -- every chain terminated by exhausting its 3-boundary-polish-evaluation cap, not by
running out of time or points; see "Why the chains finished in minutes, not hours" below).

| direction | winning anchor | best verified GT | Delta* | fixed-A/f Delta at same GT | fixed-A/f classification |
|---|---|---:|---:|---:|---|
| **upper** | `reduced_q_pre_switch` | **7.496042%** | 0.451698 | 1.644155 | FiniteSolved |
| **lower** | `current_calibration` (tiebreak, see below) | **1.252141%** | 0.013967 | 0.026902 | FiniteSolved |

**The `reduced_q_post_switch` anchor is the more dramatic finding, even though it did not win
either direction on GT alone**: at both its upper (`GT=7.345401%`) and lower (`GT=1.252141%`)
extreme points, the **fixed**-A/f profile is catastrophically `AboveEvaluationCap`
(`Delta=240249.1` and `Delta=955156.8` respectively -- five orders of magnitude over the
`delta=0.5` budget), while the **profiled**-A search finds `Delta*=0.459837` and
`Delta*=0.018315` respectively, both comfortably `FiniteSolved` and within budget. This is a
categorically stronger demonstration of productive nuisance-parameter search than either other
anchor (where the fixed-A/f baseline is merely 2-4x over budget, not five orders of magnitude).

## Production scope (as executed)

- `D=20`, `W=80,000`, `delta=0.5`, real-D20 (`noah_D20`, focal=fra, `sigma=2.5`, `seed=1`).
- Three free-cutoff anchors (the 3/6 that survived the prior session's own Phase 7 feasibility
  screen): `current_calibration`, `reduced_q_pre_switch`, `reduced_q_post_switch`.
- Both gains-from-trade directions (upper, lower) per anchor -- six independent production
  chains, each a separate OS process (5 Julia threads, BLAS=1, one independent KNITRO session,
  no shared mutable state).
- Method per chain: hold the anchor's free-`q` coordinates fixed; reconstruct the full `q`
  vector through the existing q-gravity map at every welfare point; genuinely optimize `A`
  (fixed-q A-middle loop, `solve_melitz_fixed_q_A_profile_v2`); reconstruct `f`; fully reoptimize
  and verify the inner LFD; adaptive two-start policy (`melitz_middle_two_start_adaptive!`) at
  every point.
- Welfare search: wide sequential expansion (initial step `0.5` GT-percentage-points, growth up
  to `1.5x` on acceptance, capped at `2.0` percentage points) until a local boundary bracket is
  found, then safeguarded linear interpolation (clamped to the interior 20-80% of the bracket,
  falling back to an ordinary midpoint when interpolation fails or the local profile is
  nonmonotone), **at most 3 boundary-polish evaluations** -- this cap, not the 40-point/6h
  ceiling, is what actually bound every chain (see below).

Full launch mechanics, exact commands, hardware/software versions, and a live bug caught and
fixed at launch: [`docs/key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md`](key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md),
[`docs/key_results/production_delta0p5_2026-07-31/process_registry_2026-07-31.md`](key_results/production_delta0p5_2026-07-31/process_registry_2026-07-31.md).

## Why the chains finished in minutes, not hours

Every chain reached its terminal status (`n_polish=3`, the hard cap) well inside the 40-point
and 6-hour ceilings: `n_evaluated` ranged `6-9` per chain (1 anchor + 2-5 wide-expansion points +
exactly 3 polish points), total wall time per chain `70-435s`. This is a direct, intended
consequence of the governing prompt's own explicit design, not a shortcut taken this session:
wide `0.5`-`2.0`pp steps find a local bracket within 2-5 expansion points, and the polish phase
is capped at exactly 3 evaluations regardless of whether the tight `0<=delta-Delta*<=0.002`
tolerance or the `0.01`pp bracket-width tolerance is actually reached (none of the 6 chains
reached either tight tolerance within 3 polish points -- every chain's terminal gap to the
`delta=0.5` budget is materially larger than `0.002`, see the per-chain tables below). **This
was surfaced to the user explicitly** (the chains finishing in ~1-7 minutes each was flagged as
worth a decision point, not silently reported as "converged"); the user confirmed accepting the
literal governing-prompt limits (3 polish evaluations is a hard cap, not a target) rather than
relaxing it and re-running with a larger polish budget. The tradeoff, stated plainly: this
campaign trades boundary precision for portfolio breadth -- 6 verified points across 3 anchors x
2 directions in under 30 minutes total, versus the prior single-anchor session's much slower
13-point bisection that converged to within `0.003`pp of the true `delta=0.5` boundary.

## A live bug caught and fixed during launch

The two `reduced_q_post_switch` chains crashed deterministically on their first evaluation
(root cause: the anchor-establishment step built the fixed-q constraint system from the
calibration's theta rather than the anchor's own theta -- immaterial for `current_calibration`
itself, since these coincide there, but wrong for both `reduced_q_*` anchors). Caught live via
the "always check in early" launch-health check, fixed, verified against the Phase 7 session's
own documented reference value (`Delta*=0.483265` at the `reduced_q_post_switch` anchor,
reproduced exactly), and the two chains relaunched cleanly. **A second, related gap was found
while diagnosing this**: the crash-aware wrapper's own "did the same welfare-point target crash
twice" dedup never actually triggered, because the pending-target marker was cleared before the
assertion that caused the crash could fire (every attempt logged `pending target: none_pending`)
-- recovery in this specific incident was via direct diagnosis and manual intervention, not the
automated wrapper's own repeat-crash detection. Both issues are fixed in the committed driver
source (marker now stays live through the relevant assertions); full account:
[`docs/key_results/production_delta0p5_2026-07-31/process_registry_2026-07-31.md`](key_results/production_delta0p5_2026-07-31/process_registry_2026-07-31.md#live-bug-caught-and-fixed-at-launch-2026-07-31-2036-2041-local).
The other four chains (`current_calibration` x2, `reduced_q_pre_switch` x2) were never affected.

## Source map (reused, not modified)

| concern | file |
|---|---|
| repaired fixed-q A-middle-loop driver (cap handling, incumbent retention, FC/GA dedup) | `src/melitz/fixed_q_a_middle_loop.jl` (`solve_melitz_fixed_q_A_profile_v2`) |
| adaptive two-start policy | `src/melitz/fixed_q_a_middle_loop.jl` (`melitz_middle_two_start_adaptive!`) |
| exact A gradient | `src/melitz/exact_a_gradient.jl` |
| fast O(W+D) focal-link update | `src/melitz/moment_operator.jl` (`melitz_update_moment_operator!`) |
| cutoff-anchor construction (reduced-q basis) | `src/melitz/reduced_q_controller.jl`, `reduced_q_subspace.jl` |
| this campaign's production driver (per anchor x direction chain) | `scripts/melitz_production_chain_2026-07-31.jl` |
| crash-aware launch/resume wrapper | `scripts/melitz_production_chain_launch_2026-07-31.sh` |
| 6-way orchestrator (manual/reproducible use) | `scripts/melitz_production_launch_all_2026-07-31.sh` |
| final cold verification (fresh process, per chain) | `scripts/melitz_production_final_verify_2026-07-31.jl` |
| preflight | `scripts/melitz_production_preflight_2026-07-31.jl` |

## Per-chain results (all fresh-process cold-verified: FiniteSolved, `nStatus=0`, `lfd_ok=true`,
## A-gravity residual <1e-14, q/f reconstruction drift = 0 exactly)

| chain | anchor | direction | points (anchor+wide+polish) | unique inner solves | wall_s | best GT | Delta* | fixed-A/f Delta | fixed-A/f class | improvement |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| 1 | `current_calibration` | upper | 1+2+3=6 | 594 | 329.7 | 7.467412% | 0.441569 | 1.585140 | FiniteSolved | 1.143571 |
| 2 | `current_calibration` | lower | 1+5+3=9 | 850 | 318.8 | 1.252141% | 0.013967 | 0.026902 | FiniteSolved | 0.012935 |
| 3 | `reduced_q_pre_switch` | upper | 1+2+3=6 | 592 | 325.4 | 7.496042% | 0.451698 | 1.644155 | FiniteSolved | 1.192457 |
| 4 | `reduced_q_pre_switch` | lower | 1+5+3=9 | 886 | 434.8 | 1.252141% | 0.015239 | 0.026902 | FiniteSolved | 0.011663 |
| 5 | `reduced_q_post_switch` | upper | 1+2+3=6 | 537 | 317.3 | 7.345401% | 0.459837 | **240249.093** | **AboveEvaluationCap** | 240248.634 |
| 6 | `reduced_q_post_switch` | lower | 1+5+3=9 | 77 | 69.7 | 1.252141% | 0.018315 | **955156.756** | **AboveEvaluationCap** | 955156.737 |

Machine-readable: `docs/key_results/production_delta0p5_2026-07-31/points/<chain>_points.csv`
(full per-point history), `docs/key_results/production_delta0p5_2026-07-31/final_verification/<chain>_final_verify.csv`
(fresh cold-verification results).

### Why the three lower-direction chains land on the exact same GT

Not a bug: the lower-direction step schedule (`GT0 - 0.5, GT0-1.25, GT0-2.375, ...`) is
anchor-independent (only `A`/`f` differ by anchor, not `g0`/`wage_ratio`/`sigma`), and in every
one of the 3 chains the wide-expansion phase accepted the identical 4 steps then rejected at the
identical 5th (`GT=0.228141%`, `AboveEvaluationCap`, a genuine support cliff -- `Delta` jumps
from `~0.001-0.004` at `GT=2.228%` to a certified lower bound of `28.0-33.0` just `2` GT-points
lower). Because that certified lower bound (`28-33`) is so far above the `delta=0.5` budget
relative to the feasible edge's tiny `Delta` (`~0.001-0.004`), the safeguarded interpolation's
raw proposal in every one of the 9 total polish evaluations (3 per chain) computed a fraction
`t_raw` deep below `0.2` and was clamped to the interior-20% floor -- **anchor-independent by
construction of the clamp**, since the clamp bound itself doesn't depend on the anchor's
specific `Delta` value once `t_raw` is that far outside `[0.2,0.8]`. This produced bit-identical
`GT_target` sequences across all three anchors' lower-direction chains, purely as an emergent
consequence of the safeguard, not a shared-state bug (independently confirmed: each chain's own
`A_free`/`Delta`/`f` at each shared `GT` differ, only the `GT` targets coincide).

## Aggregate selection and tiebreak

**Upper**: `reduced_q_pre_switch` wins outright on GT (`7.496042%` vs `7.467412%` vs
`7.345401%`).

**Lower**: all three anchors tie exactly on GT (`1.252141%`, see above). Since GT alone does not
discriminate, the tiebreak is the smallest `Delta*` (the most robust margin under budget, not
requiring a re-run to break the tie): `current_calibration` (`Delta*=0.013967`), ahead of
`reduced_q_pre_switch` (`0.015239`) and `reduced_q_post_switch` (`0.018315`).

**Not claimed to be globally optimal** -- these are the best verified incumbents in a finite,
3-anchor portfolio under a bounded (3-polish-evaluation) local search, not an exhaustive search
of either the cutoff space or the welfare coordinate.

## Final questions

1. **Best verified upper-bound gains-from-trade incumbent?** `GT=7.496042%`,
   `Delta*=0.451698`, `FiniteSolved`, cold-verified fresh (`reduced_q_pre_switch` anchor).

2. **Best verified lower-bound incumbent?** `GT=1.252141%`, `Delta*=0.013967`, `FiniteSolved`,
   cold-verified fresh (`current_calibration` anchor, tiebreak winner among 3 exactly-tied GTs).

3. **Which cutoff anchor won in each direction?** Upper: `reduced_q_pre_switch`. Lower: 3-way
   exact tie on GT, broken by `Delta*` in favor of `current_calibration`.

4. **`Delta*` at each winner?** Upper: `0.451698`. Lower: `0.013967`.

5. **How much worse was the fixed-(A/f) profile at the same welfare point?** Modest for
   `current_calibration`/`reduced_q_pre_switch` (fixed-A/f `1.14-1.19` points worse, upper;
   `0.012-0.013` points worse, lower -- both still `FiniteSolved`, just over/near budget).
   **Catastrophic for `reduced_q_post_switch`**: fixed-A/f is `AboveEvaluationCap` at
   `Delta~240,249` (upper) and `Delta~955,157` (lower) -- five orders of magnitude over the
   `0.5` budget, vs. the profiled search's `0.4598`/`0.0183`.

6. **How many wide exploration points and boundary-polish points were used?** 45 total welfare
   points across 6 chains: 6 anchor points + 21 wide-expansion points (2-5 per chain,
   direction-dependent) + 18 boundary-polish points (exactly 3 per chain, every chain hit this
   hard cap).

7. **Did safeguarded interpolation reduce the need for bisection?** **Direction-dependent,
   verified by directly recomputing each proposal's interpolation fraction `t`, not assumed**:
   in the **upper** direction, all 3 anchors used genuine UNCLAMPED linear interpolation at
   every one of their 3 polish points (`t` ranged `~0.46-0.64`, i.e. real information from the
   observed `(GT,Delta)` pairs, not a naive midpoint or a safeguard fallback) -- interpolation
   genuinely helped there. In the **lower** direction, every single polish evaluation across all
   3 anchors (9 total) computed a raw `t` far below the `0.2` floor (because the infeasible
   bracket edge was an `AboveEvaluationCap` certificate `28-33`x the budget, not a comparably-
   scaled `FiniteSolved` value) and was clamped to exactly `t=0.2` -- the safeguard's floor did
   the work, not genuine interpolation, in that direction.

8. **Did any chain display materially nonmonotone profiled behavior?** **Yes, decisively, and
   consistently across all 3 upper-direction chains**: in every one, `Delta*(GT)` DECREASED
   between polish points 2 and 3 (e.g. `current_calibration`: `GT=7.271%,Delta=0.4327` ->
   `GT=7.394%,Delta=0.4191` -- Delta fell even though GT rose) before rising again at the final
   point -- a genuine local non-monotonicity, not noise (verified: same qualitative pattern in
   all 3 anchors' upper chains). The lower direction shows a sharp support CLIFF instead
   (`Delta` jumps `~10,000x` over a `2`-percentage-point GT move, `GT=2.228%->0.228%`) rather
   than smooth non-monotonicity.

9. **Did any process crash, and was checkpoint recovery successful?** Yes -- both
   `reduced_q_post_switch` chains crashed deterministically (a real code bug in the anchor-
   establishment step, not the anticipated `mul_G!` SIGSEGV) on their first evaluation, 2-3
   times each, before being caught, root-caused, fixed, and relaunched cleanly this session.
   **Recovery was via direct manual diagnosis, not fully automatic**: the wrapper's own "same
   target crashed twice" dedup did not fire, because a design gap (now fixed) cleared the
   pending-target marker before the assertion that actually crashed the process could fire, so
   every attempt was misclassified as a "different/first" crash. No `mul_G!` SIGSEGV was
   observed on any of the 6 chains' 45 welfare-point evaluations.

10. **How much variation was there across cutoff anchors?** In the profiled-A results
    themselves, modest: `Delta*` spans `0.4416-0.4598` (upper, a `0.018` range) and
    `0.0140-0.0183` (lower, a `0.004` range) -- all three anchors find comparably good,
    comfortably-within-budget solutions. In the fixed-A/f BASELINE at those same points,
    enormous: `1.59-1.64` (`current_calibration`/`reduced_q_pre_switch`, upper) vs.
    `~240,000` (`reduced_q_post_switch`, upper) -- a five-orders-of-magnitude difference driven
    entirely by which anchor is used, not by the welfare point itself.

11. **Is there sufficient evidence that the outer procedure performs productive nuisance-
    parameter search beyond calibration?** **Yes, and the `reduced_q_post_switch` results make
    this the strongest evidence produced by any session in this line of work to date**: at a
    cutoff configuration where the calibration's OWN fixed technology matrix is catastrophically
    infeasible (`AboveEvaluationCap` at 5+ orders of magnitude over budget), genuinely
    reoptimizing `A` (holding `q` fixed) finds a comfortably within-budget `FiniteSolved` point
    at the SAME welfare level -- this cannot be explained as "the search barely moved off the
    calibration" (Q10's answer already rules that out) or as recovering something already
    implicit in the data (the fixed-A/f alternative at that exact point is not just worse, it is
    not even a valid weak-duality-bounded finite point in the same regime).

## Required outputs

1. This document.
2. Per-chain machine-readable point history: `docs/key_results/production_delta0p5_2026-07-31/points/<anchor>_<direction>_points.csv`.
3. Per-chain checkpoint directory: `docs/key_results/production_delta0p5_2026-07-31/checkpoints/`.
4. Aggregate upper/lower incumbent table: see "Per-chain results" above.
5. Fixed-(A/f) comparison results: `docs/key_results/production_delta0p5_2026-07-31/final_verification/*_final_verify.csv`.
6. Launch and resume scripts: `scripts/melitz_production_chain_launch_2026-07-31.sh`,
   `scripts/melitz_production_launch_all_2026-07-31.sh` (same script resumes -- idempotent; a
   terminal chain exits immediately when re-launched).
7. Provenance manifest: `docs/key_results/production_delta0p5_2026-07-31/provenance_2026-07-31.md`.

Distinguishing carefully, as required: every incumbent above is a **verified feasible
incumbent** (fresh cold `FiniteSolved`, full residual verification, in this session's own final
process, not merely the campaign process's own trajectory value); the aggregate table reports
**the best result in this finite 3-anchor portfolio**, not a claim of global optimality over
either the cutoff space or the welfare coordinate.
