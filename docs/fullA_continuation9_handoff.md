# Continuation 9 handoff — real-data D=20 production readiness, all phases complete

Written at end of session. Branch `diag/fullA-d4-exact`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, **HEAD `1d4f0c0`**
(clean tree). Machine `demand.mit.edu`, KNITRO 14.2.0, Julia 1.12.6,
`JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
throughout. Supersedes `docs/fullA_continuation9_interim_handoff.md` (written
mid-session, after Phases 1-5 only).

This session picked up the standing continuation-9 brief — "learn the
production bottlenecks and memory constraints for the real full-A_od D=20
problem at W=80,000 and W=800,000, remove the remaining dense-moment
dependencies, then run a short supervised pilot" — and completed all nine
phases plus one user-directed addendum, ending with the **first-ever
genuinely converged full-A_od D=20 real-data optimization results** in this
investigation's history.

## Headline results

- **D=20 real-data infrastructure now exists** (it did not at the start of
  this session): `context_real_d20.jl::d20_real_setup(; W)`, France focal,
  built by porting real-data loading from a sibling worktree into this one.
- **A server-wide memory incident occurred and was fully resolved at the
  source**, not routed around: a mis-defaulted flag allocated a dense
  tensor that hit ~109GB at W=80,000 and would have exceeded 1TB at
  W=800,000. Fixed; W=800,000 now runs safely at ~17-21GB peak.
- **Three production speedup phases (3, 4, 5) all landed**, each validated
  against real D=20 data for the first time (all prior validation was D=4
  synthetic): compressed moments (1.7-1.8x warm value / 5-5.7x moment-build),
  winner-margin certificate (~21x on line-search sweeps), staleness-aware
  bandwidth caching (~1.7-1.9x), plus a real pre-existing thread-safety bug
  found and fixed.
- **A user-directed addendum** (exact early infeasibility screening) landed
  cleanly, with zero false positives across an extensive validation battery,
  and **retroactively resolved an open question from Phase 1**: confirmed
  exactly why W=8,000 was uniformly infeasible at D=20 (5/400 positive-share
  bilateral pairs have zero winners on that draw support).
- **Both gates required before a D=20 pilot passed**: Phase 6 (A-block
  gradient validation) returned GO, Phase 7 (D=10 upper gate rerun) returned
  PASS (clean `-101` convergence, was `-401` budget-stall before this
  session's architecture).
- **Phase 8's pilot produced two genuinely converged, cold-verified-feasible
  full-A_od D=20 points** — the first in this investigation's history — after
  finding and fixing a real bug (box bounds copied from D≤10's O(1)-scale
  convention, silently clipping KNITRO's start point at real D=20's much
  larger natural-theta scale).
- **A user-requested fixed-A\* sanity check confirms the free-A search is
  doing genuine work** at D=20: +37.1% (upper) / 52.5%-tighter (lower) vs. a
  naive fixed-A gamma-only search — continuing, and exceeding, the D=4→D=10
  growth trend from Continuation 8.

## Session narrative, phase by phase

### Phase 1 — production-path audit, and a premise correction

`docs/fullA_D20_production_path_audit.md`. The standing brief assumed a real
D=20 full-A_od driver existed and needed benchmarking. It did not: real D=20
data + a D-generic driver existed only for the *sequential* method on a
sibling worktree; the modern full-A machinery (compressed moments,
`lfix_composite`, winner certificates) existed only on this worktree, with
zero real-data loading code. Resolved by porting real-data loading in
(commit `5c1baca`), per the coordinating session's judgment call after
presenting the gap to the user. New `context_real_d20.jl::d20_real_setup`
validated: κ_ACR matches the known real-data point estimate (0.020314) to 5
significant figures; D=20 confirmed feasible at W≥80,000, uniformly
infeasible at W=8,000 (later exactly explained by the infeasibility-screen
addendum).

### A live server-wide memory incident (commit `bd313a2`)

Caught by the user ("some job... using 23.5% and growing"), not by any
internal safeguard. Root cause: `d20_real_setup` inherited a diagnostic-only
default (`needs_outer_moment_jacobian=true`) that allocates a dense
`W × (nTotalMoments+2) × l_full` tensor — ~109GB at D=20/W=80,000, would have
exceeded 1TB at W=800,000. Two runaway processes killed; default fixed to
match the real production drivers' own convention. Verified: W=80,000 peak
dropped from 109GB to 4.5GB; W=800,000 peak dropped from a killed
>780GB-and-climbing trajectory to 16.6GB.

### Phase 2 — microbenchmarks at W=80,000 and W=800,000

`docs/fullA_D20_W80k_microbenchmark.md`, `docs/fullA_D20_W800k_microbenchmark.md`.
At W=80,000: dense `inner_moment_build` is 65% of a warm value call — the
clear Phase 3 target. At W=800,000 (measured against the *finished*
production architecture, not the original slow path, per the user's own
resequencing decision mid-session): compressed mode now wins cold too
(1.175x, reversing W=80k's cold loss), gradient threading speedup grows with
W (7.33x at 20 threads vs W80k's 5.47x). Peak memory 21.07GB across the full
workload at W=800,000.

### Phases 3, 4, 5 — production speedups (all real-D20-validated for the first time)

Dispatched as parallel agent workstreams once the user confirmed this
resequencing (do the speed work before the expensive benchmarks/pilot, not
after) was the right call.

- **Phase 3** (`docs/fullA_fully_compressed_inner_report.md`): compressed
  moments ported to real D=20 with zero code changes (1.73x warm value,
  5.74x moment-build — continues the D=4→10 trend of 2.35x→3.64x cleanly to
  5.74x). `build_lfix_base_cache`'s dense self-validation made opt-in
  (~1.3x gradient speedup). Three dense-Hessian-free inner CC solvers built
  and honestly rejected — all lose to the existing dense-Hessian-compressed
  baseline at D=20 despite fewer FLOPs (many small compressed calls losing
  to one BLAS `gemm!`). **The dense-Hessian baseline remains the production
  default.**
- **Phase 4** (`docs/fullA_D20_winner_kernel_optimization_report.md`):
  winner-margin certificate 20.9-21.2x on line-search sweeps (vs D=4's
  ~6.3x — confirms the win scales with D). Top-3 coordinate update now a
  real 1.16-1.38x win (was noise at D=4, confirming Continuation 7/8's
  "grows with D" prediction). Destination-batching: honest null. New
  destination-major kernels: mixed (warm-only modest win), not wired in.
- **Phase 5** (`docs/fullA_D20_bandwidth_optimization_report.md`):
  staleness-aware bandwidth caching (~1.7-1.9x) is the real production
  lever; the closed-form quantile selector alone was weak (1.08x). Found (not
  fixed by that agent) a real pre-existing thread-safety bug — **fixed
  directly by the coordinating session** (commit `a6ed25e`): a plain `Dict`
  written concurrently from `Threads.@threads`, corrupting ~1.5% of trials;
  wrapped in a `ReentrantLock`, verified 0/200 corrupted post-fix,
  bit-identical gradient output.

### User addendum — exact early infeasibility screening

`docs/fullA_D20_infeasibility_screening_report.md`. Implemented exactly per
the user's mathematical specification: a draw-free pairwise impossibility
certificate (`S_sod = B_so + a_od` decomposition, `M_ok` precomputed once)
plus a tie-safe winner scan with immediate zero-count rejection, plus an
optional extreme-draw witness query. Zero false positives across D=4
(300+ trials), D=6/8/10, and D=20 real data at both W values. Pairwise
rejection ~100,000x cheaper than a real solve. **Retroactively confirmed**
this session's own open Phase 1 finding: all 4 historical D=20/W=8,000
"-300" points are exactly infeasible by this certificate.

### Phase 6 — A-block gradient validation: GO

`docs/fullA_D20_gradient_validation.md`. 20 gravity-tangent A-only
directions × 3 points × 2 bandwidths at real D=20/W=80,000. A-only agreement
poor at the near-flat calibration point (cosine 0.41-0.73, corroborating
Continuation 8's D=4 finding) but good at both branch points (cosine
0.94-0.996). Critically, the actual steepest-descent direction a real
outer-loop line search uses is directionally correct at all 3 points
(6/6) — **GO for Phase 8**, with the caveat that individual A-block
coordinate signs near near-degenerate points shouldn't be over-trusted.

### Phase 7 — D=10 upper gate rerun: PASS

`docs/fullA_D10_upper_gate.md`. Old result (Continuation 8, pre-this-session
architecture): `-401` budget-stall at 120s, κ=0.38953. New result (full
Phase 3-5 production architecture, 600s budget): SR1 converges cleanly at
`-103` in ~145-150s, κ=0.39932 (+2.51%); L-BFGS also converges (`-101`) but
SR1 wins. Both cold-recheck-verified. **PASS — the brief's own recommended
gate before a D=20 frontier attempt.**

### Phase 8 — first real full-A_od D=20 pilot

`docs/fullA_D20_short_pilot.md`. Both gates passed; cleared to proceed. Found
and fixed a real bug live: box bounds `[-8,8]` were copied from the
synthetic D≤10 drivers' O(1)-A_od convention, but real D=20's natural-theta
`A_od` ranges up to 4.86e11 (`z=log(A)` up to 26.9, pivot-reduced z-norm
315.5) — KNITRO's presolve silently clipped the start point to a box corner,
causing every earlier attempt to crash or falsely "converge" instantly with
a bogus zero gradient. Fixed by centering bounds on the actual start point.
Result: both profile runs (fixed-g, A-only search) converged cleanly
(`-101`); both joint constrained-polish runs are honest time-limit stalls
but already tight against the δ boundary and cold-recheck-verified. Matches
the brief's own stated default expectation: profile+continuation for
mapping, then joint polish near δ.

**Label correction (coordinating session, commit `1d4f0c0`)**: Phase 8's own
driver used the opposite `find_smallest`↔direction convention from the rest
of this investigation (confirmed by direct source comparison against Phase 7
and the canonical D=4 registry) — its "upper" branch is actually the
small-κ branch and its "lower" branch is actually the large-κ branch. All
numerical results are correct; only the English labels were swapped. The
pilot doc now carries an explicit correction table.

### User-requested fixed-A\* comparison

Appended to `docs/fullA_D20_short_pilot.md` §8.2. Direct adaptation of
Continuation 8's D=4/6/8/10 sanity check to real D=20/W=80,000 (script:
`c9_fixedA_pilot_d20_real.jl`). Using the corrected labels: free-A beats
fixed-A by **+37.1%** on the upper (large-κ) branch and finds a **52.5%
tighter** κ on the lower (small-κ) branch — continuing Continuation 8's
D=4→D=10 growth trend (+19.8%→+21.9%→+23.2%→+32.5%) cleanly to D=20's
+37.1%, and the largest lower-side gain found at any dimension tested. The
free-A outer loop is doing real, substantial search work, not idling near
its start.

## Repository state

- Branch `diag/fullA-d4-exact`, HEAD `1d4f0c0`, clean working tree.
- Six per-workstream worktrees created this session, all merged and removed
  (`git worktree remove`); six matching branches deleted.
- Untracked leftover result directories from Continuation 8
  (`results/fullA_d4/{c4243c1,d547142}/d{8,10}_pilot_*`) remain from before
  this session started — still not touched, still flagged for whoever picks
  this up next.
- All docs pushed to Dropbox at
  `dropbox:Gravity robustness/Analysis/Server Output/fullA_d4_continuation9_2026-07-18/`
  and `.../fullA_d4_continuation9_2026-07-19/` (some sub-agents pushed to
  their own per-workstream subfolders too — see each doc's own commit
  message for the exact path if a specific one is needed).

## Final go/no-go classification

### D=20, W=80,000

**GO for production points, with the production architecture now
established**: compressed moments + opt-in-only dense validation +
dense-Hessian-baseline inner solve + winner-margin certificate + top-3
coordinate update + staleness-aware bandwidth caching + exact infeasibility
pre-screening. Both required gates (Phase 6, Phase 7) passed. Phase 8 proved
the full pipeline produces genuine, cold-verified-feasible converged points
in well under the 900s budget used for profile runs; joint polish near δ may
need a somewhat longer budget than 450s for a fully clean (non-`-401`)
convergence, though the current budget-limited numbers are already real and
usable. Recommended production recipe: profile+continuation for
A-block mapping at each target γ', then joint constrained polish near δ,
exactly as Phase 8 validated.

### D=20, W=800,000

**READY as a fallback, with one gap to close first**: memory is confirmed
safe (21GB peak across a full production-speed workload, vs. a killed
>780GB pre-fix trajectory), the full 4-point microbenchmark is done, and
Phase 9's cost projection (~49.4 CPU-hours / ~24.7h wall-clock for all ten
frontier points on a 2-branch-concurrent plan) is grounded in real per-call
data. The gap: Phase 9 flagged that the existing checkpoint/resume pattern
from the D=4/D=10 production drivers is confirmed *applicable* to
D=20/W=800,000 but the compressed/cached machinery is not yet wired into
that specific resumable-batch driver — worth closing before a genuine
unattended multi-hour W=800,000 run, not before a supervised one.

## Answers to the standing brief's seven closing questions

1. **Real production bottleneck at D=20, each W**: at both W values, dense
   `inner_moment_build` dominated a warm value call before Phase 3
   (65% at W=80k) — now resolved by compressed mode. Post-Phase-3-5, the
   full-gradient's `fd_and_bandwidth` component (per-coordinate bandwidth
   search + FD probes) is the largest remaining piece, addressed but not
   eliminated by Phase 5's caching.
2. **Fastest, most memory-safe inner method**: the existing dense-Hessian
   compressed-FG baseline (Phase 3C tested and rejected two genuinely
   compressed alternatives — both cost more wall-clock despite fewer FLOPs).
3. **Fully compressed path savings**: 1.7-1.8x warm value / 5-5.7x
   moment-build at both W values; ~1.3x additional from skipping the
   `build_lfix_base_cache` dense self-validation.
4. **A-block `L_fix` derivative reliability**: GO, with a specific caveat —
   individual coordinate signs are unreliable near near-degenerate (small-Δ)
   points, but the aggregate steepest-descent direction (what a real
   line search actually uses) is reliable everywhere tested (Phase 6).
5. **Best thread/process allocation**: `JULIA_NUM_THREADS=20`,
   `OPENBLAS_NUM_THREADS=1` throughout (re-confirmed, not re-litigated, per
   Continuation 8's own empirical A/B test) — no session this continuation
   found reason to revisit this.
6. **Profile-plus-constrained-polish vs. either alone**: yes, preferable —
   Phase 8's direct pilot evidence shows profile reliably converges cleanly
   and fast, and joint polish (even budget-limited) reliably lands on a
   tight, cold-verified feasible point once warm-started from a profile
   point — matching the brief's own stated default expectation.
7. **Projected wall time/memory, five upper + five lower points**: per
   Phase 9's cost projection, ~49.4 CPU-hours / ~24.7h wall-clock on a
   2-branch-concurrent plan at W=800,000 (the harder, more conservative
   case) — built from real per-call costs plus D=4/D=10 production
   gradient-call-count data, with D-scaling extrapolation flagged as the
   single biggest remaining uncertainty in that number. Memory per process:
   ~17-21GB at W=800,000, confirmed safe for at least 2 concurrent contexts
   (36GB combined, negligible slowdown) — likely many more, not yet tested
   beyond 2.

## What's left, in priority order

1. Wire the compressed/cached machinery into the existing D=4/D=10
   checkpoint/resume production driver pattern before a genuine unattended
   multi-hour W=800,000 batch run (Phase 9's one flagged gap).
2. Extend Phase 8's joint-polish budget beyond 450s (both runs were still
   improving at time-limit) if a fully clean `-101`/`-103` convergence
   (rather than a tight but budget-limited `-401`) is wanted for the
   production frontier points.
3. The A-block sub-gradient sign-disagreement near near-degenerate points
   (Phase 6, Continuation 8 before it) remains real and unresolved — still
   not a demonstrated correctness blocker, still worth understanding better
   before leaning on individual A-block coordinate signs for anything else.
4. Nothing else from this session's own scope remains flagged as blocking;
   the standing continuation-9 brief's nine phases plus the user's
   infeasibility-screening addendum are all complete.

## Session mechanics — a recurring pattern, again

Every single background agent dispatched this session ended its first turn
passively waiting for a notification ("I'll wait for the Monitor...", "I'll
resume once...") instead of actively polling its own backgrounded run — the
exact failure pattern Continuation 8's own handoff already flagged. Every
one was resumed with an explicit correction and then completed its task
properly on the second attempt. **This is now a very well-established
pattern across two full continuations' worth of sessions** — a future
session's initial task prompt to any subagent should probably state the
active-polling requirement even more forcefully/early, or the harness-level
fix (if one becomes available) should be preferred once it exists.
