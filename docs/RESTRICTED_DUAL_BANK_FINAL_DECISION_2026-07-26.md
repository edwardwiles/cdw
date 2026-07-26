# Restricted Dual-Bank Final Decision — 2026-07-26

## Decision: KEEP_OPT_IN (default reverted to `false`) — real trajectory evidence now exists, and it does not support promoting to default

Task §2 required real short-outer-loop trajectory evidence (not the inherited session's tiny D=4
two-point sequence) before considering `use_dual_bank=true` a default for the four restricted
families. This session ran `phase2_dual_bank_benchmark.jl`: real D=20/W=80,000 campaigns, bank OFF
vs bank ON, same calibration start, same `maxtime_real=90.0`s budget, through the actual public
drivers (`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`).

## Real results (this session's own bug-fixed final run — see below for the fix)

| Family | Bank | Wall (s) | n_eval | best Δ | bank queries | hits | warm_inner_solves | **warm_start_failures** | mean selected distance |
|---|---|---|---|---|---|---|---|---|---|
| Flexible CM | off | 126.0 | 8 | 0.0518695 | 0 | 0 | 0 | 0 | n/a |
| Flexible CM | on | 115.6 | 8 | 0.0518695 (identical) | 10 | 9 | 9 | **2** | 4.43e8 |
| Common Fréchet | off | 189.7 | 1 | 0.0202939 | 0 | 0 | 0 | 0 | n/a |
| Common Fréchet | on | 178.3 | 1 | 0.0202939 (identical) | 5 | 4 | 4 | **4 (ALL warm solves failed)** | 1.95e8 |
| CM+ZC | off | 140.7 | 6 | 0.0443240 | 0 | 0 | 0 | 0 | n/a |
| CM+ZC | on | 149.5 | 6 | 0.0443240 (identical) | 8 | 7 | 7 | **2** | 9.91e7 |
| Origin-ZC | off | 195.8 | 10 | 0.0492383 | 0 | 0 | 0 | 0 | n/a |
| Origin-ZC | on | 161.7 | 10 | 0.0492383 (identical) | 13 | 12 | 12 | **3** | (not separately logged) |

Origin-ZC required its own focused re-run (`phase2_originzc_dual_bank_only.jl`) after this
session's first two full-relaunch attempts hit two of its own script bugs (a missing include for
`cm_originzc_config.jl`/`cm_originzc_target_layout.jl` ordering, an invalid
`distribution_restriction` symbol, and — the one that produced a real, silent-looking failure — a
missing `cm_originzc_cplus.jl` include that made every origin-ZC KNITRO solve in this benchmark
die with a genuine gradient-callback error (`nStatus=-500`) rather than actually running; see
`TRANSFORMED_A_ALL_FAMILIES_RELEASE_2026-07-26.md` for the parallel discovery in the Phase 8 smoke
test). Fixed, and the corrected run above is the trustworthy one.

## Interpretation

1. **Outer progress is identical bank-on vs bank-off in every case measured, all four families** —
   `best_Δ` and `n_eval` match exactly between arms. This matches the theoretical expectation (a
   warm start changes how KNITRO reaches a converged point, not what it converges to for a
   well-posed convex inner problem) and is the same discipline the inherited session's own D=4
   gate already established (`diff=0.0` at both points).
2. **Wall-clock is NOT a reliable directional signal from this data** — flexible CM and common
   Fréchet improved modestly with the bank on in one full run (-8.3%/-6.0%) but a repeat full run
   showed flexible CM's bank-on arm 4% *slower*; CM+ZC's bank-on arm was 6.3% slower in the
   confirmed final run; origin-ZC's bank-on arm was 17.4% faster. This run-to-run variance (JIT
   warm-up, system load from concurrent background jobs this session ran, KNITRO's own
   barrier-method path-dependence) is larger than the effect being measured — a single 90-second
   short-trajectory run per arm is not enough repetitions to draw a wall-clock conclusion either
   way, and this document does not claim one.
3. **Real warm-start failures occurred in every family** — flexible CM 2/9 (or 2/9 in the earlier
   run), common Fréchet **4/4 (100%)**, CM+ZC 2/7, origin-ZC 3/12. `warm_start_failures` counts a
   warm-started solve that came back non-feasible (`nStatus ∉ FEASIBLE_CODES`) — the driver's own
   architecture presumably falls back/retries (net outer progress was unaffected, per point 1), but
   this is exactly the concrete risk task §2 asked to measure, and it is real and consistent across
   every family tested, not an isolated fluke.
4. **The selected distances are enormous** (4.4e8, 1.95e8, 9.9e7) relative to the outer point's own
   scale (`||w0|| ≈ 292`) — consistent with `select_warm_start_restricted`'s own documented
   behavior: unscaled Euclidean distance is used until 3+ history points accumulate
   (`cm_dual_bank_production.jl:90-95`), and this project's own CLAUDE.md records that the real
   calibrated `A_od` block spans **~11 orders of magnitude** — exactly the regime where an unscaled
   distance metric over economic-core coordinates is close to meaningless until enough history
   exists to estimate a per-coordinate scale. With only a handful of solves in a 90-second budget,
   the bank rarely if ever reaches that 3-point threshold in this short-trajectory test.

## Verdict

`use_dual_bank` stays **opt-in** (`false` default) for all four restricted families. The real
trajectory evidence this session collected does not support promoting it to default: real,
non-trivial warm-start failure rates (up to 100% of attempted warm starts in the common-Fréchet
arm, and non-zero in all four families) were observed, even though net outer progress happened not
to be harmed in this short window. A longer campaign, or a harder outer point, could plausibly let
a failed warm start cost real wall time (extra KNITRO retries) rather than being absorbed for free.
The measured wall-clock effect is inconsistent in direction and smaller than run-to-run noise (see
point 2 above) — there is no reliable speed benefit demonstrated to weigh against the demonstrated,
consistent, non-trivial warm-start failure rate.

Per the task's own suggestion, a **restriction-aware bank scorer** (evaluating the candidate dual
against the complete family moment operator, not distance alone) was considered but not built —
correctly gated behind Phase 5's operator-FG work (not yet real), since a KKT/residual-based score
needs the operator machinery that work would provide.

## A note on getting this evidence right

The first attempt at this benchmark crashed three times on script bugs before producing real data:
missing `w0` (this driver requires an explicit calibration start, unlike the unrestricted family's
own stage runner), a missing `probs` cutpoint argument, an incorrect `distribution_restriction`
symbol for origin-ZC, and an include-order dependency between `cm_originzc_target_layout.jl` and
`cm_originzc_config.jl`. All were genuine bugs in this session's own new benchmark script, not in
production code — fixed and re-verified before any number above was recorded, consistent with this
project's own standing discipline of not trusting a plausible-looking result without first
confirming the harness that produced it is actually correct.
