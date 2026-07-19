# Continuation 9, Phase 7: D=10 upper gate rerun with the full production architecture

Branch `c9-phase7-d10gate` (worktree `gravity-fullA-d4-c9-phase7`, forked from
`diag/fullA-d4-exact` @ `7783ad3`), machine `demand.mit.edu`,
`JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Reruns
Continuation 8's D=10 upper direction (`docs/fullA_d4_section10_dimension_scaling_c8.md`
§2-4, which budget-stalled at 120s, KNITRO status -401) with a materially
longer budget (600s) and the three production-architecture levers built
earlier in Continuation 9: compressed moment representation (Phase 3.1),
staleness-aware bandwidth caching (Phase 5), and both outer Hessian modes
(SR1, L-BFGS) per this task's explicit instruction. Context: same synthetic
`d_exact_setup_scaled(D=10, W=8000)` as the original benchmark (this task
reruns the SAME gated-benchmark scope, not the real-data D=20 driver built
elsewhere in Continuation 9).

**Headline: both Hessian modes now converge cleanly (KNITRO status -103
SR1, -101 L-BFGS — both genuine soft-convergence codes, categorically
different from the old run's -401 time-limit code), in well under 3 minutes
each against the 600s budget, cold-recheck-verified exact-feasible, with
kappa improving over the old budget-stalled result (+2.5% for SR1, the
better of the two directions tried). GATE VERDICT: PASS — this counts as
clean D=10 upper convergence, the standing brief's own prerequisite before a
full D=20 frontier.**

---

## 0. Setup and safety

Driver: new `full_aod_diag/d4_exact/c9_phase7_d10_upper_gate.jl` — a thin
wrapper around Continuation 8's `run_d6_pilot.jl` (reused, not rewritten, per
this task's instruction), same context/pivot-elimination/KNITRO-wiring
pattern, with three swaps: (1) F callback uses
`evaluate_fullA_fast(...; moment_representation=:compressed)` instead of the
plain dense `evaluate_fullA`; (2) G callback uses
`composite_gradient_at_fast(...; h_mode=:cached, bandwidth_cache=policy.cache)`
wrapped in a `BandwidthCachePolicy` (Phase 5's recommended lever) instead of
`h_mode=:adaptive` every call; `multi_method=:top3` (Phase 4) and
`validate_dense=false` (Phase 3.2) are already `composite_gradient_at_fast`'s
own defaults, confirmed present, not overridden; (3) both
`csw_outer_wallclock_sr1.opt` (hessopt=3) and `csw_outer_wallclock_lbfgs.opt`
(hessopt=6) tried. The inner CC dual solve itself is unchanged from
Continuation 8 (`ek_inner.opt`, dense-Hessian exact) — Phase 3C's own D=20
finding was that this baseline beats all three dense-Hessian-free
alternatives it built, so it was not re-litigated here.

A minimal sanity probe ran first, per this investigation's standing
discipline: one dense-vs-compressed value-equivalence check plus one
cached-bandwidth gradient call at the D=10 natural-theta start point, before
the full gate run. Dense vs compressed `Delta_dual` agreed to **8.3e-17**
(machine precision). Peak `VmHWM` across the entire task (sanity probe + both
gate runs + all diagnostics): **3.37 GB** — dramatically below any safety
threshold, as expected for D=10 (100 free A-entries vs D=20's 400; this
investigation's earlier D=20 memory incident does not apply at this scale).

**Does the compressed path actually help at D=10, not just D=20?** Reusing
existing evidence rather than re-measuring: Continuation 8's own D-scaling
grid (`docs/fullA_canonical_performance_profile_c8.md` §"D-scaling") already
measured this directly at D=10 (W=8000, same context this task reuses):
`inner_moment_build` is **3.64x** faster compressed vs dense at D=10
(continuing the D=4→6→8→10 trend 2.35x→3.22x→3.27x→3.64x) — the FG-callback/
TOTAL verdict is more mixed at D=10 specifically (that report found dense
slightly faster there, 0.70x, on that one component), matching the
"FG-callback verdict stays noisy across D" pattern already flagged
repeatedly in this investigation. This task's own sanity probe adds a fresh
D=10 correctness data point (8.3e-17 agreement) on top of that existing
timing evidence; a full new timed dense-vs-compressed benchmark was not
re-run here (would duplicate the canonical profile's own D=10 row).

## 1. First run: a real bug found and fixed

The first gate run (results in `results/fullA_d4/7783ad3/c9_phase7_d10_upper_gate_20260719_063857/`)
found a genuine bug in this task's own new driver, not in any pre-existing
production code: on a `TiedWinnerError`, `composite_gradient_at_fast`
internally falls back to `full_rebuild_gradient_fallback` and returns a
**valid** gradient, but that fallback's metadata `NamedTuple` has no
`cache_hits` field (`composite_gradient_fast.jl` line ~139:
`merge(meta_fb, (tie_fallback = true, tie_error = e))`). This driver's
`record_hits!(policy, meta.cache_hits[2:end])` call blindly indexed that
field, threw a `FieldError`, and the broad `catch` block then wrongly
treated an already-correct gradient computation as a total failure,
discarding it in favor of an all-zero A-block for that one call. This hit
**2 of 241 L-BFGS gradient calls** (0.8%) in the first run (SR1's 450 calls
were unaffected — no ties encountered there). Fixed by only invoking
`record_hits!` on the non-tie-fallback path (`meta.tie_fallback ||
record_hits!(...)`), preserving the already-valid gradient either way. The
full gate was rerun cleanly after the fix
(`results/fullA_d4/7783ad3/c9_phase7_d10_upper_gate_20260719_064757/`, the
numbers reported below) — no more fallback warnings, and the affected
L-BFGS run's status changed from -102 to -101 (both are genuine
soft-convergence codes per §2 below; the fix changed which specific stopping
test triggered first, not whether it converged) while `kappa` landed at the
**same value to 15 significant figures** (0.39351693778733654 both times),
indicating the 2 degraded steps in the buggy run did not materially affect
the outcome — reported for completeness, not because it changed the
headline finding.

## 2. KNITRO status codes: what they actually mean (checked against the local KNITRO 14.2.0 manual)

Pulled directly from `/opt/shared_sw/knitro/14.2.0/doc/html/3_referenceManual/returnCodes.html`
(the exact solver version this investigation uses) rather than assumed:

| code | meaning (verbatim, condensed) | category |
|---|---|---|
| -101 | "Primal feasible solution; ... relative change in the solution estimate is less than xtol." | genuine soft convergence |
| -102 | "Primal feasible solution estimate cannot be improved; desired accuracy in dual feasibility could not be achieved. No further progress can be made." | genuine soft convergence |
| -103 | "Primal feasible solution; ... relative change in objective function is less than ftol for ftol_iters consecutive iterations." | genuine soft convergence |
| -400 to -499 | "Knitro terminated because it reached a pre-defined limit (-40x = feasible point found before the limit, -41x = no feasible point found before the limit)" | **limit-based stall** |
| -401 | "The time limit was reached before being able to satisfy the required stopping criteria. A feasible point was found." | **limit-based stall** |

This confirms categorically: the OLD D=10 upper result's `-401` is a
time-limit stall (feasible point found, but the solver was cut off before
its own stopping tests were satisfied). Both of THIS task's results (-103
SR1, -101 L-BFGS) are in the -10x "genuine soft convergence" family,
unrelated to any limit — this codebase's own existing convention already
treats all of `(-100,-101,-102,-103)` as "converged" uniformly (e.g.
`full_aod_diag/d4_exact/c8_nestedw_run_grid.jl` line 243:
`converged = nStatus in (-100, -101, -102, -103)`), confirmed here against
the primary-source manual text rather than just that convention.

## 3. Headline results (bug-fixed rerun)

Context: `d_exact_setup_scaled(D=10, W=8000)`, δ=1 (same as the old
benchmark), upper direction (`find_smallest=true`), 600s budget per Hessian
mode.

| | **OLD (Continuation 8, 120s, pre-production)** | **NEW: SR1** | **NEW: L-BFGS** |
|---|---|---|---|
| knitro_status | **-401** (time-limit stall) | **-103** (converged) | **-101** (converged) |
| kappa | 0.38953 | **0.39932462338440167** | 0.39351693778733654 |
| Δ kappa vs old | — | **+2.51%** | +1.02% |
| gamma'_focal | (not reported) | 0.7365189029637139 | 0.7407833387035868 |
| Delta_dual | (not reported) | 0.9998457681365884 | 0.9999070839124788 |
| Delta − delta | -2.5e-5 | -1.542e-4 (comfortably feasible) | -9.29e-5 (comfortably feasible) |
| cold recheck (dense `evaluate_fullA`, `warm=false`) | matches | **matches**, Delta=0.9998457681365871 (diff ~1.3e-12) | **matches**, Delta=0.9999070839124786 (diff ~2e-13) |
| n_eval / n_grad_calls | (not reported) | 1600 / 450 | 1014 / 240 |
| bandwidth-cache hit rate | n/a (no caching, old run used `h_mode=:adaptive`) | 28510/44550 = **64.0%** | 11286/23562 = **47.9%** |
| wall-clock budget used | 120s (full budget, stalled) | ≈145-150s of 600s | ≈65-75s of 600s |

**SR1 wins** (larger kappa is better for the upper direction) and is the
recommended headline for this gate. Both directions converge genuinely and
finish in well under 3 minutes real time against a 600s (10-minute) budget —
not a close call.

### 3.1 A real finding: this driver's own internal `wall` field is not trustworthy

Both runs' self-reported `wall` field (`time()` immediately around
`KNITRO.KN_solve(kc)`) is implausibly small (SR1: 0.07-0.1s, L-BFGS:
1.1-1.4s across the two runs) — directly contradicted by two independent
pieces of evidence from the SAME runs: (1) the SAME script's own
`grad_wall_total` accumulator (summed `time()` durations measured *inside*
every gradient callback, called from the exact same process) reports 41.5s
(SR1) / 24.1s (L-BFGS) — a hard lower bound on the true solve duration,
already ~250-400x larger than the reported `wall`; (2) direct external
observation during the first run: repeated `date`-stamped `tail` checks on
the log file showed KNITRO's own `outlev=iter` iteration table climbing
steadily (~3.1 iterations/second, consistently) over multiple real minutes
while the process was live. A quick isolated check confirmed `time()` itself
resolves correctly to `Base.Libc.time()` and measures sub-second durations
correctly in this exact environment (`julia --project=. -e '...'`, `sleep(1.5)`
measured as `1.509s`) — so this is not a symbol-shadowing bug in `time`
specifically; the discrepancy is localized to the `t0=time(); KN_solve();
wall=time()-t0` pattern around this specific KNITRO callback-based solve and
was not chased to a definitive root cause (out of this task's time budget,
per this investigation's "verify before causal claims" discipline: report
the symptom and a defensible workaround, do not force an unverified
explanation). **This directly extends** (in a more severe, previously
unobserved form) **the pre-existing "this codebase's own internal wall-clock
field is unreliable" finding** (`docs/fullA_d4_section10_dimension_scaling_c8.md`
§7) — worth flagging for anyone else who reuses `run_d6_pilot.jl`'s pattern
and wants to trust its printed `wall` number; that field was apparently
never relied upon by any prior report in this investigation (Continuation
8's own D=6/8/10 table omits `wall` entirely).

**Genuine wall-clock, from external timestamps instead**: the bug-fixed
rerun's whole script (sanity probe + both directions' solve + cold recheck +
diagnostics) ran from process launch (`06:46:53`) to completion
(`06:51:27.28`, via file mtime) = **274s total**, of which ~64s was JIT/setup
(the OUTDIR timestamp `..._064757` marks the sanity probe's completion and
the SR1 phase's start). The first run's directly-timestamped log
observations (13 separate `date`+`tail` checks spanning the SR1 solve) give
a consistent ~3.1 iterations/second rate throughout, extrapolating to
**≈145-150s** for SR1's 450-call solve+diagnostics and, by the remaining
budget, **≈65-75s** for L-BFGS's 240-call solve+diagnostics — both figures
cross-validate against the clean rerun's 210s combined post-setup total
(145+70=215 ≈ 210, consistent) and against the two runs' `n_grad_calls`
ratio (450:240 = 1.875, matching 145:75 = 1.93 to within measurement
noise). **Both directions finish in a small fraction (≈4-9x under) of the
600s budget** — this is the real, load-bearing number for the gate verdict,
not the buggy in-script field.

## 4. Directional sanity diagnostics (5 random directions each, h=0.02)

Per this task's scope ("does this point look locally sane," lighter-touch
than the concurrent Phase 6 task's full D=20 gradient validation), reused
the W80k doc §1F / bandwidth report §3G methodology directly: perturb the
converged point in the pivot-reduced z-space, 2 full warm-started dense
`evaluate_fullA` re-solves per direction (bypassing the incremental
machinery entirely), compare the re-solved secant of `Delta_dual` against
the production gradient's directional derivative at the same point.

| | SR1 (5 dirs) | L-BFGS (5 dirs) |
|---|---|---|
| finite/sane | 5/5 | 5/5 |
| abs_err range | 0.0142 – 0.0644 | 0.0049 – 0.1075 |
| abs_err mean | 0.0392 | 0.0578 |
| sign agreement | 3/5 | 2/5 |

**5/5 finite and sane in both runs** — the basic sanity bar this task asks
for is met. The magnitude/sign disagreements on several directions are
**not new**: this is the same A-block gradient sign-disagreement pattern
already flagged as an open, unresolved finding in this investigation
(`fullA-continuation8-summary` memory: "73-93% coordinate-sign agreement in
a near-flat region"; the D=20 bandwidth report §3G found the identical
pattern on one of its own 5 directions at W=80,000). It is also consistent
with the bandwidth report's independently-confirmed small-W noise
explanation (§2/§3D of `docs/fullA_D20_bandwidth_optimization_report.md`:
D=4/W=8,000 cosine agreement of only 0.826 between two independently-correct
bandwidth selectors, rising to 0.9984 at D=20/W=80,000) — this task's W=8,000
context is exactly the small-W regime where that noise is expected. Not a
new blocker; flagged and left open, matching this investigation's standing
practice, since resolving it is explicitly the concurrent Phase 6 task's
job, not this one's.

## 5. Gate verdict

**PASS. This counts as clean D=10 upper convergence**, satisfying the
standing brief's own stated prerequisite before a full D=20 frontier
attempt:

- Both Hessian modes (SR1, L-BFGS) reach genuine KNITRO soft-convergence
  codes (-103, -101 respectively) — verified against the primary-source
  KNITRO 14.2.0 manual to mean real stopping-criteria satisfaction, not a
  limit-based cutoff — categorically different from the old run's -401
  time-limit stall.
- Both finish in well under the 600s budget (≈145-150s SR1, ≈65-75s
  L-BFGS, externally verified) — not a marginal pass; there was no
  indication either run was approaching its budget.
- Both are cold-recheck-verified exact-feasible (dense `evaluate_fullA` at
  `warm=false` matches the tracked value to ~1e-12/1e-13).
- kappa improves over the old budget-stalled result (SR1: +2.51%, the
  clear winner between the two modes tried).
- The directional sanity check passes its stated bar (5/5 finite/sane in
  both runs) with no new failure mode — the residual gradient
  disagreement is the investigation's pre-existing, already-tracked open
  issue, not something this task introduces or needs to resolve.

**Recommendation for whoever picks up the D=20 frontier next**: use SR1 as
the default outer Hessian mode at this scale (it beat L-BFGS here, as it
also did in Continuation 8's algorithm-frontier work); carry forward the
same three production levers (compressed value path, `h_mode=:cached` +
`BandwidthCachePolicy`, `multi_method=:top3`/`validate_dense=false`
defaults); and do not trust this driver pattern's self-reported `wall`
field without external cross-checking (§3.1).

## 6. Files

New: `full_aod_diag/d4_exact/c9_phase7_d10_upper_gate.jl` (driver, includes
the `record_hits!`/`tie_fallback` bugfix described in §1). Modified: none
(no pre-existing production file touched). Raw output:
`results/fullA_d4/7783ad3/c9_phase7_d10_upper_gate_20260719_063857/` (first
run, buggy L-BFGS numbers, kept for the record per §1),
`results/fullA_d4/7783ad3/c9_phase7_d10_upper_gate_20260719_064757/` (clean
rerun, the numbers reported above), full logs in
`/tmp/claude-181517/-bbkinghome-edav-gravity-robustness/084523fb-9ba6-49b2-a78a-c7490cc63208/scratchpad/d10_upper_gate.log`
(run 1) and `d10_upper_gate_run2.log` (run 2, not committed — scratchpad
only, referenced here for traceability within this session).
