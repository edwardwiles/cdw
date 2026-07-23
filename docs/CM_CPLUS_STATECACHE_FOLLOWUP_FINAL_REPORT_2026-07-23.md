# CM-C+ / complete-state cache follow-up — final report — 2026-07-23

Branch: `perf/fullA-cm-cplus-statecache-overnight-2026-07-22`
Worktree: `/bbkinghome/edav/gravity_robustness/gravity-perf-fullA-cm-cplus-statecache-overnight`
Base commit for this follow-up: `b9b8462e976e431c8aa6df059ec9ffa73bb829b0` (the prior overnight
session's own final report commit, itself based on `22683016b5a927d4952e52049145ad8d1f5a2b87`,
`production/fullA-exact` @ tag `cm-production-ready-2026-07-22-r3`)
Final commit this session: `59756acece960054dccdefd343079257736178ad`

**Not merged into `production/fullA-exact` or `production/sequential-linearized`.** Neither
canonical trunk was touched. No production tag was created.

## Commits this session

| Commit | Summary |
|---|---|
| `0eeaf03` | Part I directional/sign audit — root-cause the fixed-dual vs reoptimized opposite-sign observation |
| `ea7767a` | Part II.1 — expanded D=4 CM-C+ equivalence battery (60/60 pass) |
| `2db2058` | Part II.2 — real D=20/W=80000/L=50 multi-point CM-C+ gate (5/5 pass) |
| `849c592` | Part II.3 — matched short real KNITRO trajectories, Reference vs C+ |
| `faf5893` | Part II.4 — checkpoint backend provenance: schema bump, validated on resume |
| `eb08254` | Part III.1 — complete-state cache process-lifecycle and hit-opportunity audit |
| `59756ac` | Part III.2 — shadow-mode measurement: zero measured cache hit opportunity |

Part IV (conditional cache wiring/persistence) was **not attempted**, correctly, per the brief's
own instruction — see below.

---

## Direct answers to the brief's closing questions

### 1. Was the fixed-dual vs reoptimized opposite-sign observation a diagnostic artifact, nonsmoothness, or a genuine gradient problem?

**Diagnostic artifact.** An extra unary minus in the PRIOR session's own follow-up test script
(`overnight_cm_cplus_d4_gate.jl` Section 4: `true_secant = -(Dp-Dm)/(2h)` instead of
`(Dp-Dm)/(2h)`) inverted the sign of the comparison at every coordinate where the fixed-dual
approximation and the true reoptimized derivative actually agree — which is the normal case.
Independently re-derived the sign convention from the production call path (not merely quoted from
the prior algebra-trace doc), added a base-point sanity check the prior work lacked, then ran a
28-row bandwidth/winner-stability battery and a 32-row independent directional battery: **60/60
correct-sign results** once the harness bug is fixed, including deep into winner-switch-heavy
territory (up to 60 switches in one test). The exact numeric example the brief quoted
(`true_reoptimized=+2.28e-2`, `fixed_dual=-2.02e-2`) now prints with MATCHING signs after the fix.
No blocker. See `docs/CM_FIXED_DUAL_VS_REOPTIMIZED_DIRECTIONAL_AUDIT_2026-07-23.md`.

### 2. Is CM-C+ promotion-ready? Should it remain default off or become default on after review?

**Classification: `READY_FOR_REVIEW_NOT_PROMOTION_READY`**, materially strengthened from the prior
session's identical classification but not upgraded to a promotion-ready tier, because a small,
well-defined set of the brief's own explicit II.1 sub-items were not exercised this session (see
"What remains blocked or deferred" below). Everything that WAS tested passed at machine precision:

- D=4: 60/60 (every free coordinate individually, 8 bandwidths, top3-vs-generic-fallback
  isolation, forced exact ties, swept near-ties, a constructed incumbent-swap point, nonfinite-
  probe/retry discipline).
- D=20 (real production points, not synthetic): 5/5 (calibration-equivalent + δ=0.1/1.0/2.0 across
  3 independent chains from the just-completed real campaign), cosine=1.0 to 10 decimals, 0/399
  sign mismatches, max|Δg| 3.9e-15 to 2.2e-13, C+ speedup 4.9x-6.8x, ~85x less allocation.
- Matched real trajectories: bit-identical downstream KNITRO behavior when only one gradient call
  occurs (100s budget); genuinely divergent but NOT degraded trajectories over a longer budget
  (600s: C+'s best cold-verified incumbent was at least as good as Reference's in this one-run
  comparison — reported honestly as n=1, not oversold).
- Checkpoint backend provenance: now schema-safe (persisted `cm_gradient_backend`, validated on
  resume, explicit audited override policy), 17/17 tests against real short KNITRO runs.

If a future session closes the specific gaps below, this should become
`PROMOTION_READY_DEFAULT_OFF` directly — the evidence quality is already there for everything that
was actually run. **Recommendation: keep default off, review-ready.**

### 3. Does the complete-state cache have meaningful real hit opportunities under the actual supervisor/process architecture?

**No, not measurably.** The process-lifecycle audit (Part III.1, traced from
`cm_production_supervisor.sh`/`cm_production_stage_runner.jl`/`cm_cold_verify.jl`) established that
every delta stage, every supervisor restart, and every next-delta seed launches a brand-new OS
process — an in-memory, unserialized cache cannot survive any of those boundaries, contrary to the
prior session's own design doc, which named the prototype the "complete-state exact/cross-delta
cache" and listed "a later stage/restart revisits a point already solved" as a target use case.
Shadow-mode measurement (Part III.2) on three real trajectories (δ=0.1, 1.0, 2.0, 300s each, same
problem instance as the real production campaign) found **zero exact-point revisits across all 19
`cb_F!` calls measured** — every point solved in every run was solved exactly once. This is
consistent with (not contradicted by) the structural argument: KNITRO's interior-point/barrier
algorithm (confirmed from real trajectory output) does not naturally revisit exact prior iterates
the way a trust-region reject/retry algorithm would.

### 4. Was it wired? Was persistence required?

**Not wired, and this is the correct outcome, not a shortfall.** The brief's own §3.2 instruction
is explicit: "If the potential hit rate is negligible or the avoidable solve wall is trivial, stop
and leave the cache as a prototype. Do not wire it merely because it is correct." The measured hit
rate is not merely low — it is exactly zero in every real trajectory tested. Persistence (disk-
backed cache surviving process boundaries) would be REQUIRED to reach any of the theoretically
higher-value cross-process hit classes at all, but Part III.1's own analysis shows that even WITH
persistence, the two classes the design doc emphasized most (next-delta seed, cross-chain) are
structurally unlikely to hit, because the next delta starts from a materially different point
(the previous delta's cold-verified incumbent, not a revisited iterate) and different chains are
deliberately seeded to diverge. **Classification: `PROTOTYPE_ONLY_NEGLIGIBLE_VALUE`.**

### 5. What end-to-end wall savings were measured — not inferred — from each feature separately and jointly?

- **CM-C+ alone**: measured, not inferred, at 5 real D=20/W=80000/L=50 production points (median
  of 3 timed reps each, post-warmup): 4.9x-6.8x wall-clock speedup per gradient callback (14.85s→
  ~1s scale), ~85x less allocation (13.9GB→162MB per call). Matched-trajectory measurement (real
  KNITRO runs, not isolated callback timing) shows the speedup does not obviously translate into a
  proportional increase in outer iterations within a fixed wall budget in the ONE longer (600s) run
  measured (Reference: 11 iters; C+: 2 iters) — the D=20/W=80000 inner solve's own cost, common to
  both backends, and KNITRO's own algorithmic response to (very slightly) different gradient values
  both plausibly explain this; not fully disentangled this session (see gaps below).
- **Complete-state cache alone**: zero measured wall savings (zero hits observed; see above).
- **Jointly**: not tested — the cache was never wired into either gradient backend's production
  call path, so there is nothing to combine (same as the prior session's own honest disclosure).

### 6. Which commits are safe candidates for a later production cherry-pick/fast-forward?

All 7 commits this session are additive, opt-in, default-off, and were built on top of the prior
session's own already-reviewed commits. In rough order of standalone value and lowest review
burden:
- `0eeaf03` (Part I) — pure bugfix to a follow-up TEST script + a new, reusable diagnostic harness;
  touches no production code path at all. Safest possible cherry-pick.
- `eb08254`, `59756ac` (Part III.1/III.2) — a doc + an opt-in `shadow_stats` kwarg on
  `run_cm_upper_checkpointed` that is exactly zero-cost when omitted (every existing call site).
  Very low risk; genuinely useful if anyone wants to re-measure hit opportunity later without
  re-deriving the instrumentation.
- `faf5893` (Part II.4) — the checkpoint schema bump. Slightly more consequential (changes
  `cm_checkpoint.jl`, a file every real production run touches) but thoroughly tested (17/17
  against real KNITRO runs) and carefully backward-compatible (verified against a real copy of the
  actual completed campaign's own schema-2 checkpoint). Recommend this land BEFORE any future CM
  production campaign that might want to resume across a backend change.
  Should be reviewed carefully, but is architecturally sound.
- `ea7767a`, `2db2058`, `849c592` (Part II.1-3) — test/harness/results additions only; zero
  production code changes. Safe to cherry-pick independently of anything else, at any time, purely
  as accumulated evidence.

### 7. What remains blocked or deferred?

Explicitly, not silently:
- **II.1 gaps**: runner-up-origin-change isolated as its OWN sub-test (only incumbent-origin-change
  was explicitly constructed); CM-columns-with-zero-economic-multiplier and economic-columns-with-
  zero-CM-multiplier (none occurred naturally at the tested points; constructing one deliberately
  was judged out of proportion for this session, disclosed rather than fabricated); a full 399-
  coordinate FD sweep specifically AT a winner-switch-heavy point for CM-C+ (Part I's I.3 tested
  winner-switch-heavy DIRECTIONS with sign checks, not a full CM-C+-vs-Reference coordinate sweep
  at such a point); `:anchored` contrasts were not re-tested in the expanded D=4 battery (only
  `:orthonormal`, matching production's actual decision, and matching the ORIGINAL D=4 gate test's
  own scope, which DID cover both contrasts for its narrower checks).
- **II.2 gap**: `:anchored` contrasts skipped at D=20 (same production-decision rationale).
- **II.3 gap**: a literal "record every state along one trajectory, replay both backends at each
  exact state" exercise was not built as a separate artifact — Part II.2's 5-point real-checkpoint
  comparison and Part II.3's matched-trajectory work substantially cover the intent (identical
  states, both backends, real production points) but not the letter of "one continuous trajectory,
  every visited state." The 600s trajectory's iteration-count disparity (11 vs 2) is flagged but
  not root-caused to completion.
- **II.4 gap**: the actual bash supervisor's own restart path was not exercised end-to-end (only a
  Julia-level interruption+resume simulation, T6) — a full `cm_production_supervisor.sh`-level
  restart drill was judged unnecessary given the underlying `run_cm_upper_checkpointed` resume path
  IS what the supervisor calls, and that path was directly tested.
- **Part IV**: correctly not attempted (see above) — this is a disposition, not a gap.

## Process/resource safety log

- Live campaign (`production_runs/cm_campaign_2026-07-22`, 3 chains) was confirmed **fully
  completed and all processes exited** (via `ps`, supervisor `.out` logs showing "all deltas
  finished and cold-verified") before ANY read of its checkpoint files; only `stage_latest.jls`/
  `cold_verified_seed.jls` files were read (`deserialize`, read-only), never written to, moved, or
  deleted. mtimes of the specific files read were re-checked at the end of the session and match
  their original completion times.
- Partway through this session, a SEPARATE, NOT-self-initiated campaign extension
  (`sensitivity_w100k_sobol`, `sensitivity_w100k_sobol_unrestricted`) was discovered actively
  running in the SAME `production_runs/cm_campaign_2026-07-22` directory, on cores 20-39 and
  60-79, launched by a different process/session around 01:56-02:03 EDT. This was NOT started by
  this session. All of this session's own work was confirmed (via `ps`/`taskset` immediately
  before each launch) to run on disjoint cores (100-187 range used across the session); no
  resource contention occurred. `production_runs/` was never written to by this session at any
  point — only specific, named files were read via `deserialize`/`load_cm_checkpoint`.
- System load and memory were checked before every real-KNITRO launch this session (`ps`,
  `/proc/loadavg`, `free -g`); the machine had comfortable headroom (~39-40/208 cores loaded,
  &gt;800GB free memory) throughout.
- Existing regression tests (`test_cm_checkpoint_original.jl` + `test_cm_checkpoint_resume.jl`, the
  pre-existing schema-2 checkpoint/resume suite) were re-run after this session's
  `cm_checkpoint.jl` edits and pass 4/4 — confirming the schema bump did not regress the
  already-reviewed checkpoint/resume behavior.
- Neither canonical production trunk (`production/fullA-exact`, `production/sequential-linearized`)
  was touched. No merge, no fast-forward, no production tag.

## Notable methodological finding (worth its own line, not just embedded in a commit message)

While constructing the Part II.1 near-tie battery, a large (~6e-2) spurious CM-Reference-vs-C+
discrepancy appeared that was NOT a backend defect — it traced to `ctx.γ.Uσ` (a value precomputed
once from `ctx.U` at context-setup time) going stale after a naive `merge(ctx, (U=Unew,))`
perturbation. The tell was that the discrepancy did not shrink as the constructed tie separation
shrank toward zero (flat from eps=1e-2 down to 1e-8), which is inconsistent with genuine near-tie
floating-point sensitivity. Documented as a durable memory
(`ctx-U-perturbation-stale-Usigma-pitfall`) for any future test that perturbs draws directly.

## Morning recommendation

1. Review and land `0eeaf03` (Part I bugfix) and the Part III commits promptly — near-zero risk,
   real value (a corrected test harness, a reusable diagnostic).
2. Treat CM-C+ as `READY_FOR_REVIEW_NOT_PROMOTION_READY` — strong enough evidence to schedule a
   real review, not yet strong enough (by the brief's own strict "all gates pass" bar) to flip the
   production default. The remaining gaps are narrow and well-scoped for a short follow-up.
3. Retire the complete-state cache to prototype-only status in the production roadmap — it is
   correct but has no measured value in the current architecture; do not revisit unless the
   production algorithm class changes (e.g., a switch away from interior-point/barrier) or a
   genuine cross-process persistence use case is separately motivated.
4. Push this branch (clean, all commits intentional) to `cdw` only, per the brief's own
   authorization — done as the final action of this session.
