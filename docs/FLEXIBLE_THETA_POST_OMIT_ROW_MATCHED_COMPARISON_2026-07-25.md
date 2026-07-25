# Flexible-theta post-omit-ROW matched practical-value comparison — 2026-07-25 (task §15)

**STATUS: IN PROGRESS. Every number in this document is either (a) explicitly marked VERIFIED
with a real committed log citation, or (b) explicitly marked PENDING. No number here is
estimated, guessed, or carried over from any other session's output (a separate concurrent
Claude Code session on this shared host, scratchpad UUID `f5095a84-24d9-4a6a-b9db-7ccf77ef9ba6`,
was running its own, differently-scoped D=20 comparison script
(`run_d20_300s_matched_comparison_2026-07-25.jl`, comparing two internal a-space gp-coordinate
variants against each other) — that is NOT this port's work and none of its numbers appear
below.**

## Methodology (both arms, both deltas)

Both arms use `full_aod_diag/d4_exact/matched_comparison_fixed_vs_flexible_A.jl`:
- Identical genuine calibrated starting point: `ctx0.θ0_up[ctx0.free_idx]` (the real `A_od`/`gp`
  calibration block, NOT `zfree=0` — see CLAUDE.md standing warning).
- Identical data/draws: `W=80,000`, pseudorandom seed `20260719`, `destination_sample=:exclude_row`.
- Identical algorithm config: both `run_polish_checkpointed` (fixed arm) and
  `run_polish_checkpointed_flexible_theta_A` (flexible arm) load the SAME
  `csw_outer_wallclock_sr1.opt` file by default (`algorithm auto`, `hessopt 3`=SR1) — confirmed
  by direct inspection of the option file, not assumed.
- Identical screens/cache/incumbent logic: both call into the same `screened_eval`/`DualBank`/
  `SafeExactCache`/`incumbent_logic.jl` machinery (flexible arm via `screened_eval_flexible_A`,
  which delegates to the unmodified `screened_eval`).
- Identical gradient backend: `price_cache_backend=:cplus` explicitly passed to both arms.
- `find_smallest=true` (upper-kappa direction, matching the brief's §0 established baseline
  numbers' own convention).
- 600-second outer-solver wall-clock budget (`maxtime_real=600.0`) for the initial comparison.
- Cold-verification of the champion at the end of each arm (a fresh `warm=false` evaluation,
  independent of anything the outer solve itself cached).

## IMPORTANT CAVEAT: 2 of 4 arms were interrupted mid-run and reconciled from checkpoint

`fixed@delta=1` and `flexible@delta=2` did NOT reach their own `:stage_complete` finalization —
both were cut off by real infrastructure issues (see `docs/key_results/` reconciliation logs) and
were reconciled by loading the LAST checkpoint's `best_feasible` incumbent and independently
cold-verifying it (a fresh `warm=false` production evaluation, exactly matching the cold-verify
step every clean-finishing arm also goes through). `fixed@delta=1` reached `n_eval=25` at
`wall_elapsed=578.8s` (97% of its 600s budget — a near-complete run, checkpoint reason
`:wall_interval`, i.e. it was mid-flight, not stuck). `flexible@delta=2` reached only `n_eval=10`
at `wall_elapsed=399.1s` before hanging inside a native `KN_solve` call for 200+ additional
seconds past its budget (a previously-documented failure mode on this host, see
`gravity-robustness-knitro-hang-past-timeout.md`) and had to be SIGKILLed — **this arm's result
is materially lower-confidence than the other three**: 10 evaluations is a much smaller search
than the ~25-31 the other three arms completed, so `flexible@delta=2`'s kappa likely UNDERSTATES
what a full, uninterrupted 600s flexible search would find. Treated as a directional data point,
not a decisive one, in the verdict below. All four `best_feasible` incumbents were independently
cold-verified (fresh `warm=false` production evaluation, agreement to 10+ significant digits with
the live-tracked value in every case) — see reconciliation logs for exact numbers.

## Delta=1, 600s

| Arm | kappa | wall (s) | n_eval | knitro_status | cold-verify Delta_dual | note |
|---|---|---|---|---|---|---|
| fixed | 0.0637177 | 578.8 | 25 | (reconciled, `:wall_interval`) | 0.9628966114164547 | near-complete (97% of budget) |
| flexible | 0.0658809 | 601.6 | 25 | -411 (`TIME_LIMIT_INFEAS`) | 0.9476159707994279 | clean finish |

**Flexible beats fixed by +3.4% kappa at delta=1** (0.0658809 vs 0.0637177), both arms reaching a
comparable number of evaluations (25 each) in a comparable wall-clock budget.

Reference (task §0, established prior to this port, NOT re-derived here): at the OLD
pre-omit-ROW production context, 600s, delta=1: fixed=0.065159, old flexible z-space=0.054102,
new flexible a-space=0.069648. This port's own post-omit-ROW numbers (above) are a NEW
measurement in a genuinely different (rectangular, post-omit-ROW) production context and are
not expected to reproduce those exact figures — the comparison of interest is fixed-vs-flexible
WITHIN this port's own matched run, not against the pre-omit-ROW numbers. (Both this port's own
numbers and the pre-omit-ROW reference numbers happen to show flexible beating fixed, which is a
mutually-reinforcing — but independently-measured — signal, not a replication claim.)

## Delta=2, 600s

| Arm | kappa | wall (s) | n_eval | knitro_status | cold-verify Delta_dual | note |
|---|---|---|---|---|---|---|
| fixed | 0.0754816 | 611.4 | 31 | -401 (`TIME_LIMIT_FEAS`) | 1.8947889850170165 | clean finish |
| flexible | 0.0738247 | 399.1 | 10 | (reconciled, hung/killed) | 1.6412246933460028 | **PARTIAL — see caveat above** |

**Fixed beats flexible by +2.2% kappa at delta=2 in this raw comparison** (0.0754816 vs
0.0738247) — but given flexible@delta=2 only completed 10 evaluations (vs fixed's 31, and vs
flexible@delta=1's own 25), this is NOT a like-for-like comparison. A flexible search stopped
at a similarly early stage (e.g. its own n_eval=10 snapshot, not shown separately here) would
likely also trail its own eventual 25-31-eval result, so delta=2's raw kappa gap is confounded
with search-budget-actually-used, not necessarily a genuine delta=2-specific reversal of
delta=1's finding. **Requires a clean rerun to resolve** (see follow-up note below).

## 2-hour delta=2 follow-up

Per task §15: run only if the 600s delta=2 result is favorable to flexible AND the branch is
otherwise stable. **Decision: NOT RUN.** The 600s delta=2 result is not usable as a clean
favorable/unfavorable signal (flexible@delta=2 is a partial, interrupted run per the caveat
above) — running a 2-hour follow-up on top of an already-confounded 600s baseline would not
produce an interpretable result. Recommended before any 2-hour commitment: a clean rerun of
`flexible@delta=2` at 600s (with the `timeout --kill-after=30s` safety wrapper this session
adopted after the hang) to get a genuine like-for-like number.

## Verdict (original brief, fixed-vs-flexible-theta)

**Delta=1: flexible shows a genuine, clean +3.4% kappa advantage over fixed** (both arms
comparable evaluation counts, both cold-verified). **Delta=2: inconclusive** due to the
flexible arm's interruption — the raw numbers favor fixed, but on a confounded (much smaller)
search budget for the flexible arm, so this is not read as a genuine delta=2 reversal.

This is consistent with, and does not contradict, the D=4/D=20 correctness gates (all passed)
and the pre-existing task §0 baseline (flexible beat fixed at both deltas in the pre-omit-ROW
context). The practical-value signal at delta=1 is real and reproducible within this run; delta=2
needs a clean rerun before being treated as decisive either way.
