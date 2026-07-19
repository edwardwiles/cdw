# Continuation 8 handoff — read this FIRST if picking up this investigation

Written at end of session. Branch `diag/fullA-d4-exact`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, **HEAD `b2ed3e8`** (clean tree). This session
picked up Continuation 7's handoff and executed the full standing 10-section continuation-8 brief
(compressed-solver integration, winner accelerators, performance profiling, algorithm-frontier
rerun, gamma-profile completion, final candidate verification, nested-W continuation, and gated
dimension scaling), plus several mid-session user-directed corrections and diagnostics. All work is
merged into `diag/fullA-d4-exact`.

## Canonical headline — CHANGED this session

- **Upper incumbent** (unchanged): `upper_lfixcomposite_sr1_60s`, κ=0.17245688540655113,
  γ'=0.8926359584642946, Δ=0.9999924058.
- **Lower incumbent (NEW, replaces the old one)**: `lower_v2`, **κ=0.004387827651021192**,
  γ'=0.9973649883022927, Δ=0.9941306706106116 (Δ−δ=−5.87e-3, comfortably feasible). Registered in
  `candidate_registry.jl`, independently re-verified through the exact oracle before committing.
  Beats the old registered lower incumbent (κ=0.005428799948779983, essentially on-the-boundary at
  Δ−δ=+8.5e-7) by **19.2%**. Found because the two-branch gamma-profile workstream (§ below)
  discovered the old incumbent sat on a locally-KKT-stationary but NOT constrained-minimal A-basin.

## What this session did, section by section

### Section 1 — gamma-profile interpretation corrected
The U-shaped `profile_Δ(g)` (Continuation 6/7 finding) is reframed: the benchmark Fréchet value
g_F≈0.960965 sits almost exactly at the profile's interior minimum (Δ≈3.3e-5 there), which is
expected/structural (a correctly-specified benchmark should have near-zero divergence at its own
calibration point), not a residual mystery. Report:
`results/fullA_d4/bb74649/gamma_profile_nonmonotonicity_report.md` (addendum at top).

### Section 2 — compressed winner-form moments wired into the live solver
New mode flag `evaluate_fullA_fast(x_free, ctx; moment_representation=:dense|:compressed)` in
`oracle_fast.jl`/new `compressed_live.jl`. Default stays `:dense`. Automatic dense fallback on
`TiedWinnerError`, logged via `COMPRESSED_FALLBACK_COUNT`. Equivalence to ≤1.7e-11 across a large
suite (calibration, both incumbents, profile points, random points, all coordinates × both FD
signs × 3 h-values, warm/cold). **Honest finding**: at D=4, compressed's FG-callback is actually
~2x SLOWER than dense's BLAS `gemv!` (only the one-time moment build is reliably faster, ~2x) —
corrects Continuation 7's standalone estimate. D-scaling grid (D=6/8/10, done in §5 below) shows the
moment-build advantage growing with D but the FG-callback verdict staying noisy/inconclusive.
Report: `docs/compressed_live_integration_report.md`.

### Section 3 — winner accelerators wired
Coordinate-specialized top-3 update closes the last generic O(D) fallback
(`count_winner_flips_multi` → `count_winner_flips_multi_top3`), exact (240/240 synthetic cases
match), but noise-level (0.90-1.17x) at the full-15-coordinate level at D=4 — real 7-13% per-
coordinate win only visible when isolated. Winner-margin certificate wired as a persistent-cache
value evaluator (`lfix_value_certified`, `PersistentWinnerCache`), giving 6-6.5x on simulated
line-search sequences at ~98% certified fraction. Report: `docs/winner_accelerator_live_wiring.md`.

### Section 4 — low-risk specializations
Pow-cache defaulted on in `gamma_profile.jl` via new `live_defaults.jl` (bit-identical, confirmed).
Autarky-CF-v2 measured in its real repeated-sweep context and found to be a clean **negative
result** (~4-5% slower there) — correctly left off by default. `delta_star_schedule.jl` confirmed
out of scope (doesn't exist in this diagnostic directory). Report:
`docs/lowrisk_specialization_live_wiring.md`.

### Section 5 — canonical performance profile
D=4/W=8000 and D=4/W=80000 dense-vs-compressed breakdown, plus a D=6/8/10 scaling grid (all reached
feasibility on the first try). Compressed's moment-build advantage grows monotonically with D
(2.35x→3.64x, D=4→10); the FG-callback verdict stays inconclusive across D (flips direction at
different D, reported honestly rather than forced into a trend). Report:
`docs/fullA_canonical_performance_profile_c8.md`.

### Section 6 — algorithm frontier rerun
50-run comparison (5 configs × 5 wall-time checkpoints × 2 directions), full grid, nothing trimmed.
Compressed+SR1 essentially matches the upper incumbent (+0.02%); compressed+L-BFGS found a lower-
direction local optimum (κ=0.004634) that looked like a big win against the THEN-current registry
but was later corrected (see below) once `lower_v2` (κ=0.004388) superseded it — the frontier's find
is actually 5.6% worse than the final registered number. `smoothed_ad` (single-rho, not the full
homotopy) underperforms both directions and never fully converges. Report:
`docs/fullA_algorithm_frontier_c8.md` (includes an explicit correction note added after merge).

### Section 7 — both branches of the gamma profile completed
Low-g crossing refined to g\*=0.892635790±2e-6 (negligible shift from the incumbent). High-g branch:
classified as a **regular Δ=δ crossing** at g≈0.997031 (bracket width 4.9e-7), with a separate,
non-binding LP-certified infeasibility wall at g=1.0 exactly. Critically, this workstream found the
OLD registered lower incumbent's A was NOT the true constrained minimizer at its own g — this
directly motivated Section 8's `lower_v2`. Report: `docs/fullA_gamma_profile_two_branches_c8.md`.

### Section 8 — final D=4 candidate verification, NEW headline lower incumbent
Built `lower_v2` via a genuine production outer-loop run (same driver class as the original
incumbent) warm-started from Section 7's high-g crossing — converged to κ=0.004387827651021192,
beating the old incumbent by 19.2%, independently re-verified through the exact oracle and
**registered directly in `candidate_registry.jl`** (the completing agent recommended this; the
coordinating session applied it). Full verification battery (cold solve, secants, gravity-tangent
directions, poll, unscaled KKT residual) run on all three candidates (upper, old lower, new
lower_v2); all classified "bandwidth-KKT candidate" (exact-feasible, not certified globally robust —
no candidate in this investigation ever is). Two real bugs caught and fixed in the verification
scripts themselves (a `warm=true` state-corruption bug, an out-of-bounds FD probe). Report:
`docs/fullA_d4_final_candidate_verification_c8.md`.

### Section 9 — nested-W continuation (8000/20000/80000)
Built a genuinely-nested draw pool (fresh seed 91234, deliberately distinct from the discovery seed
888, because `drawU()`'s column-major RNG fill does NOT nest across different W — verified from
source, documented in `c8_nestedw_context.jl`). Finding: both candidates lose feasibility under a
FRESH same-size resample at W=8000/20000 (finite-sample fragility, not a bug); re-optimizing (letting
A re-adapt) recovers full feasibility at every W for both candidates, with κ staying close to the
registered values — no evidence of a better candidate lurking at higher W. Gradient cross-check:
full-vector fast-vs-slow gradient agreement is excellent (cosine ≥0.9998); the A-block-only
sub-gradient shows real, unresolved sign disagreement (73-93% agreement) in what's likely a
near-flat region — flagged honestly, not smoothed over. Report:
`docs/fullA_nested_w_continuation_c8.md`.

### Section 10 — gated dimension scaling
D=6 and D=8 free-A pilots fully converge (cold-verified); D=10 upper stalls at its 120s budget but
is still cold-verified feasible (consistent with D=10's lighter "gated benchmark" scope). D=20 NOT
launched, per the brief. **User-requested addition**: a fixed-A\* sanity check (search only γ'_focal,
A held at calibration values) at every D — confirms the free-A outer loop does real, substantial work
(+20% to +33% κ improvement over fixed-A at every D tested, growing with D, not shrinking). Report:
`docs/fullA_d4_section10_dimension_scaling_c8.md`.

## Two additional user-directed fixes, both worth carrying forward

1. **BLAS threading**: `JULIA_NUM_THREADS` does NOT bound OpenBLAS's own thread pool. Two D=8/D=10
   jobs were found consuming 60+ cores each (`nlwp`≈144) with only `JULIA_NUM_THREADS=20` set;
   killed and fixed. An initial conservative `OPENBLAS_NUM_THREADS=1` fix was correctly challenged
   as possibly leaving performance on the table — tested empirically (A/B, externally timed) rather
   than argued from intuition: `OPENBLAS_NUM_THREADS=20` gives ~0-5.6% wall-clock benefit for
   3.9-4.3x more aggregate CPU on this shared machine. **`OPENBLAS_NUM_THREADS=1` (and
   `MKL_NUM_THREADS=1`) should be set explicitly alongside `JULIA_NUM_THREADS=20` on every future
   launch in this investigation** — full data in
   `docs/fullA_d4_section10_dimension_scaling_c8.md` §7.
2. **Deprecated KNITRO options**: `mip_integral_gap_{abs,rel}` renamed to `mip_opt_gap_{abs,rel}`
   (pure rename, same parameter) across all 33 active `.opt` files in this investigation — silences
   the warning printed on every KNITRO solve. Commit `d547142`.

## Session mechanics — a recurring failure pattern, worth fixing before it recurs again

**Subagents that background a long Julia/KNITRO run and then fail to wake themselves back up to
report and commit — this happened THREE separate times this session** (the performance-profile
agent, the algorithm-frontier agent, and the nested-W agent, the last one twice). In every case the
actual compute had finished successfully — sometimes up to an hour earlier — and the agent's own
final message was a passive "I'll wait for a notification" or similar, when subagents do NOT receive
the main session's background-task notifications the way the coordinating session does. Each time,
the fix was either (a) resuming the agent with an explicit instruction to poll/check directly rather
than wait passively, or (b) — on the second stall from the same agent — the coordinating session just
finished the write-up and commit directly from the agent's already-completed raw output, since
re-prompting a third time was not a good use of time. **If this pattern recurs, prefer (b) sooner** —
once an agent's compute artifacts exist on disk and are readable, the coordinating session can
usually finish the report/commit faster than another round-trip.

The coordinating session made closely related mistakes of its own this session, all caught and
fixed: wrapping a command in `nohup ... & disown` INSIDE a call that already used the harness's own
`run_in_background: true` (double-backgrounds the work, orphaning it from tracking — the harness
reported "completed" for the wrapper shell, not the actual Julia process); the persistent shell's
cwd drifting between background-launched commands (background commands' `cd` doesn't propagate back,
and even foreground cwd was observed to reset unexpectedly at least once) — fixed by prefixing every
launch with an explicit `cd`; and the BLAS-threading oversight above.

## Worktrees — cleanup needed

Nine per-workstream worktrees were created this session under
`gravity-fullA-d4/gravity-fullA-d4-c8-*/`, all merged into `diag/fullA-d4-exact` and safe to remove
(`git worktree remove`) plus their now-merged branches. Left in place at the time of this handoff for
final verification; should be cleaned up as this session's last step (or by whoever picks this up
next, if not already done).

## What's left — in priority order

1. **The A-block sub-gradient sign-disagreement flagged in Section 9** (fast `lfix_composite` vs.
   slow `delta_fd`, 73-93% coordinate-sign agreement in a near-flat region) — real, unresolved,
   hasn't visibly broken any outer solve so far but worth understanding better before leaning on it
   for anything sensitive to the A-block gradient's exact direction.
2. **Section 6's `smoothed_ad` config used a single fixed rho, not the original 5-stage homotopy** —
   flagged as a real limitation in that report; a fair smoothed-vs-hard comparison would need the
   full homotopy wired into the same frontier harness.
3. D=8/D=10 as fully-supervised (not just gated-benchmark) pilots, if this investigation wants to
   push dimension scaling further — D=10's upper direction budget-stalled at 120s this session; a
   longer budget or a warm start might resolve it cleanly.
4. Nothing else from Continuation 7's own list remains flagged as blocking; this session closed out
   the full standing 10-section brief.

## Operational notes (carried forward, BLAS note is NEW)

- Must run on `demand.mit.edu` (KNITRO license, demand-side only).
- `source .knitro_env.sh` before any Julia invocation touching KNITRO, as a plain `source` (never
  inside a pipe — this drops the exported env vars into a subshell).
- Set **both** `JULIA_NUM_THREADS=20` **and** `OPENBLAS_NUM_THREADS=1`/`MKL_NUM_THREADS=1` explicitly
  on every launch (new this session, see above).
- Push new docs/handoffs to Dropbox at
  `dropbox:Gravity robustness/Analysis/Server Output/fullA_d4_continuation8_2026-07-18/`.
