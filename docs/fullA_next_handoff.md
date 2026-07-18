# Full-A next-continuation handoff

Written at the end of the "performance profiling, D=4 completion, and staged scaling" continuation.
Read this first if picking up this investigation again.

## CONTINUATION 5 UPDATE — read this section FIRST (supersedes Continuation 4's headline numbers)

Branch `diag/fullA-d4-exact` unchanged, worktree unchanged. Commits this continuation: `56877ff`
(Priority 0: candidate canonicalization + external revalidation), `0ed1944` (Priority 0: stale-doc
pointers), `d1db154` (Priority 1C: smoothed-gradient timing contradiction resolved), `f559b01`
(Priority 1A: warmed breakdown + F/G redundancy audit), `d84ba73` (Priority 2: threaded/cached
composite gradient, equivalence-tested), `91fe104` (moment-construction audit: mu/sigma value cache,
a user-requested addendum), `53d5331` (Priority 2 wiring + Priority 3: live `lfix_composite_fast`
results + fair hard-vs-smoothed comparison), `292fd89` (doc updates), `486a414` (Priority 4: lower
direction completion + gamma-profile partial + a new bug found), `298019e` (Priority 5: D=6 pilot +
D-scaling with the live gradient). Priorities 0-3 are **complete**. Priority 4 is **partially
complete**: lower direction is fully done (a major improvement), the gamma profile is only reliable in
a narrow window near the incumbent (blocked on a newly-discovered bug, see below), upper polish and
full crossing-refinement were not reached. Priority 5 is **partially complete**: the D=6 pilot (both
directions) and a D-scaling profile using the live gradient are done; nested-W (5a) was not attempted.
See §"CONTINUATION 5 PART 2" below for the Priority 4/5 detail.

### Headline: new best-validated upper candidate, and it is now demonstrably fast to reach

**`upper_lfixcomposite_sr1_60s`, κ=0.17245688540655113`** (already the Continuation 4 incumbent, NOT a
new number) is now canonicalized in `candidate_registry.jl` and externally revalidated
(`phaseA_lfixcomposite_sr1_revalidation.jl`): `EXACT_FEASIBLE_CANDIDATE` (8/8 cold/warm rechecks at 2
tolerances) + `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)`, but **NOT** `ROBUST_LOCAL_CANDIDATE` — same tier as
the old `upper_maxit40` incumbent, carried forward honestly (h=0.02 is degenerate, 31/32 probes cross
a winner boundary; h≤0.01 is stable but KKT residual grows slightly as h shrinks, consistent with the
run having stopped at KNITRO's own `opt_err=0.0021` rather than exact stationarity). **What changed
this continuation**: a new `D4X_GRADIENT_METHOD=lfix_composite_fast` KNITRO driver option
(`run_d4_optimized_fd.jl`) reaches this EXACT SAME point (bit-identical `w`) in **26.9s instead of
42.7s** (1.59x faster wall-clock), with the gradient callback itself 2.57x cheaper (183ms→71ms/call)
and consuming **zero** inner CC-dual solves per call (was 1) — see `docs/fullA_p2_p3_fast_gradient_and_comparison.md`.

### What Priority 1 found (docs/fullA_p1_warmed_profile.md)

1. **The archived "3.147s" smoothed-gradient timing was a JIT-compilation artifact, not a real cost.**
   Isolated directly: same closure, unwarmed first call = 3.85s, warmed median (N=30) = 44ms (86.7x
   difference). The smoothed gradient's TRUE warmed cost is ~44ms/call at D=4 — cite this number, not
   3.147s, in any future comparison.
2. **The live composite-gradient callback's documented 0.14-0.18s cost is now precisely localized**:
   `composite_gradient_at` alone (unshared base) = 141.0ms, of which only ~10-12ms is its own
   (shareable) inner solve — the other ~130ms is the SERIAL per-coordinate bandwidth-selection+FD
   loop. Sharing the base state with `eval_F` saves only 10.9ms (6.8%) — real but NOT the primary
   lever (Priority 2's threading fix is).

### What Priority 2 built (docs/fullA_p2_p3_fast_gradient_and_comparison.md, composite_gradient_fast.jl)

`composite_gradient_at_fast` adds three composable levers (threaded A-block loop; `h_mode=:fixed`
skipping the bisection+diagnostic; `h_mode=:cached` reusing a caller-owned per-coordinate bandwidth
across outer iterates) plus a shared-base-state wiring in `run_d4_optimized_fd.jl`. Only
`h_mode=:adaptive` (byte-identical to the original) + `threaded=true` + shared base was validated
live end-to-end this continuation (see live results above); `h_mode=:fixed`/`:cached` are
equivalence-tested (`test_composite_gradient_fast.jl`, 4/4 PASS) but **not yet run live in KNITRO** —
flagged as the natural next lever if `:adaptive`'s ~27s-to-converge is still a bottleneck for a future
continuation's larger-D work.

### User-addendum: moment-construction audit (docs/fullA_moment_construction_audit.md)

Found and fixed ONE genuinely redundant computation: `EK_moments_gammanorm_directgp!` recomputed
`UPow=U.^(-mu)`/`UσPow=Uσ.^(-mu)` via a full `O(W*D)` broadcast on EVERY call despite `mu`/`sigma`
being PROVABLY fixed for a ctx's entire lifetime in this investigation (never in `free_idx`). Fixed via
`moments_fast.jl::MuSigmaPowCache` (bit-identical output, 8/8 equivalence points at exactly 0.0 diff),
giving a real 1.06x-1.41x warmed speedup across D=4/6/8/10 (W=8000) and D=4/W=80000. Also confirmed
draw-chunk threading (`Threads.@threads` over `Th=nthreads()` draw chunks) already exists in
production and is genuinely active (not dead code) — benchmarked 1/2/4/8/16 threads, plateaus past 2
threads at this problem size. **NOT YET WIRED** into the live `oracle_fast.jl`/`evaluate_fullA` path or
`run_d4_optimized_fd.jl` (additive/standalone as of this continuation) — low-priority next step given
its ~2ms-per-call impact is small relative to Priority 2's ~130ms A-block lever, but free and stacks
correctly with everything else once wired.

### Priority 3: the fair comparison's bottom line

At a matched ~60s-class wall-clock budget, the fast hard route (`lfix_composite_fast`+SR1) reaches a
**higher** exact-hard κ (0.172457) in **less** wall time (26.9s) than the smoothed route's own
previously-reported total (κ=0.171972 after ~80.5s homotopy+polish, numbers REUSED from the unchanged
prior experiment, not re-run under identical conditions — flagged, see the doc's own caveats section).
The hard route also has a structural transparency advantage: every `eval_F` call IS a valid exact-hard
evaluation, while the smoothed route only produces ONE trustworthy exact-hard checkpoint, at the very
end of its schedule, by design. This reverses the brief edge the smoothed-then-polished candidate held
at the end of the prior continuation, once Priority 2's engineering fixes are applied to the hard side.

### CONTINUATION 5 PART 2: Priority 4/5 (lower direction, gamma profile, D=6 pilot, D-scaling)

Full detail: `docs/fullA_priority4_gamma_profile_and_lower.md`, `docs/fullA_priority5_d6_pilot_and_scaling.md`.

**Lower direction: DONE, a major improvement.** `run_d4_optimized_fd.jl direction=lower`,
`lfix_composite_fast`+SR1, 300s budget (converged at 22.6s, status -102): new canonical
`lower_lfixcomposite_fast_sr1_300s`, **κ=0.005428799948779983**, `Delta-delta=+8.5e-7` — essentially
EXACTLY on the divergence boundary, replacing the old `lower_stalled` candidate (κ=0.0107,
`Delta-delta=-0.101`, far inside — simply ran out of iterations). Externally revalidated: same
qualitative tier as the upper candidate (`EXACT_FEASIBLE_CANDIDATE`, degenerate-at-h=0.01-but-clean-
at-h≤0.0025 KKT check, poll "improvements" are ~1e-7 in γ', negligible).

**Gamma profile: a NEW BUG FOUND, only a narrow window validated.** `gamma_profile.jl` (new: fixes γ',
minimizes Δ over the A-block only via KNITRO, sharing the same Priority 2 levers) hit a genuine,
reproducible failure in `build_lfix_base_cache`'s self-validation (`max|q0_true-q0_cache|` up to ~13,
not a tolerance issue) when evaluated at (γ',A) combinations DECOUPLED from any jointly-optimized
trajectory — e.g. the incumbent's own A-block paired with a DIFFERENT γ'. Diagnosed this session (not
root-caused): the winner computation matches exactly (0/32000 mismatches), the counterfactual term
matches exactly (0.0 diff, ruling out γ' itself — the factual formulas never read it), but the factual
`contrib0` reconstruction is wrong specifically for these z_free values. **This is a real, open bug —
priority item for whoever continues gamma-profile or wide-parameter-space work.** A `try/catch`
fallback (return a zero A-block gradient rather than crash) is now standard practice in any driver that
might hit this (`gamma_profile.jl`, `run_d6_pilot.jl` both have it) — but only genuinely re-optimized
points should be trusted; check `n_eval`/`n_grad_calls` aren't trivially 1 before trusting a
`profile_Delta(g)` row. The 3 points that DID optimize successfully (g=0.8793/0.8859/0.8926) show
`profile_Delta(g)` strictly decreasing toward the incumbent — real, if narrow, structural confirmation
the incumbent sits close to the true crossing.

**D=6 pilot: WORKS in both directions.** Calibration is cold-infeasible at D≥6 (confirmed, matches
prior D=6 finding, now also D=8/10) — the fix is `ctx.θ0_up`'s own "natural" pre-gammanorm Aod values
(NOT `ones(D,D)`), found by direct query (`inner_status=0` immediately, no search needed) — **use this
for any future D≥6 work**, not calibration or random search (60 random trials + a 60s unconstrained
Delta-min both failed first). `run_d6_pilot.jl`: lower converges genuinely (13.7s, κ=0.007458,
`Delta-delta=-9.7e-5`); upper makes real progress within a 60s budget (κ=0.19758, 0.03% slack, would
likely converge with more time). D=4→D=6 live gradient-call cost grows only ~1.49x (71ms→106ms) for a
2.25x increase in free coordinates — sub-linear, decisively better than the old D^3.5-3.8 projection
(only 2 data points, directional not a precise fit).

**Not attempted**: multistart from the smoothed candidate/poll points/fixed-A (Priority 4 item 2),
`profile_Delta(g)=delta` crossing refinement, nested-W (Priority 5a — a distinct engineering task, not
reached), D=8/10 live pilots (conditional on D=6, now cleared, but not run).

### Immediate next steps for Continuation 6

1. **Root-cause and fix the `build_lfix_base_cache` self-validation bug** (Priority 4's finding above)
   — the single highest-value remaining item; blocks a genuinely wide gamma profile and any future
   work that decouples (γ,A) from a joint optimization trajectory. Start from: factual `contrib0`
   piece is wrong despite winner/counterfactual both matching exactly — check `pTσ0`'s VALUE (not just
   its use in winner-finding) against `hFunction!`'s own internal `pricesTempσ[winner]` directly at
   the specific draws where the discrepancy concentrates (not fully isolated this session).
2. Once fixed, re-run `gamma_profile.jl` across the FULL theoretical γ' interval, refine the
   `profile_Delta(g)=delta` crossings, and attempt upper polish from the smoothed candidate / poll
   points / fixed-A starts (Priority 4 items 2-3, not reached).
3. **Wire `moments_fast.jl`'s cache into the live oracle path** — small, low-risk, still not done.
4. **Try `h_mode=:cached` live in KNITRO** — equivalence-tested but not yet run against a real outer
   loop.
5. **Nested-W (Priority 5a)** and **D=8/10 live pilots** (gate now cleared by the D=6 pilot) — both
   fully open.
6. A genuinely apples-to-apples hard-vs-smoothed timing run (same process conditions, back-to-back) if
   a precise ratio is needed — Priority 3's comparison reused older smoothed-route numbers.

## CONTINUATION 4 UPDATE — read this section next (superseded above by Continuation 5 for headline numbers)

Branch `diag/fullA-d4-exact`, worktree `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`. Commits
this continuation: `5917add` (Phase 3a: composite gradient), `ed64fb7` (Phase 3b: live driver
extension + refresh-policy bug fixes), plus three merge commits pulling in the three background-agent
branches dispatched at the start of this continuation (`86d3612` smoothed-consistent, `126c52c`
jach-audit, `e4f7036` phase5-sequential — see §0 below), plus this doc update and
`docs/fullA_algorithm_frontier_v2.md`.

### 0. Background-agent workstreams (§3 of the handover prompt) — all three completed and merged

All three agents dispatched from `b5c109d` finished and were reviewed + merged into
`diag/fullA-d4-exact` this continuation (none were blindly merged — each doc was read, each
agent's own validation claims spot-checked against its stated method before merging):

1. **jac_h audit** (`docs/fullA_jach_audit.md`, merged `126c52c`) — confirmed via runtime counters
   (not static analysis) that the dense `jac_h` tensor is allocated once per bundle construction but
   NEVER populated or read in the live full-A cached/Method-B path (only in a legacy branch nothing
   here reaches). Added an opt-in `needs_outer_moment_jacobian=false` mode (default `true`, fully
   backward compatible), validated at D=4/6/8 (12/12 checks PASS + a real end-to-end production run
   matching the prior documented baseline numbers exactly), wired into both named production drivers.
   Saves ~14ms (D=4) to ~220ms (D=8) of one-time construction cost; **not** an active-computation
   removal (nothing was actively running). Touches shared `cc_algo/` files, additively/guarded.
2. **Smoothed-consistent full-A** (`docs/fullA_smoothed_consistent_experiment.md`, merged `86d3612`)
   — a genuinely consistent (same-temperature values+allocation) smoothed solve, temperature homotopy,
   then exact-hard polish via `lfix_incremental.jl`'s `:incremental_o1` tier. **Headline result: a NEW
   best upper candidate, κ=0.17197168927740825** (exact-feasible, `inner_status=0`,
   `max_abs_moment_kkt_resid=1.07e-16`), beating the existing `upper_maxit40` incumbent
   (κ=0.17176461388430053) by +0.12% relative. The polish only ran 2 rounds (1 accepted step) before
   stalling in its single tried direction — flagged as unfinished headroom, not a ceiling. **This is
   now the best validated upper candidate and should be the starting point for any further upper-
   direction work (Phase 6), not `upper_maxit40`.**
3. **Sequential/profiled reconstruction** (`docs/fullA_sequential_exact_comparison.md`, merged
   `e4f7036`) — re-ran the production sequential method correctly (maxit 25→150, `eval_fcga=no`
   genuinely honoring `hessopt=4`, real 5-start multistart per `sequential_methodology.tex`).
   Best-of-5 upper κ=0.157907 — clears the fixed-A floor (0.1440) confirming genuine convergence
   (unlike the retracted 0.0779 run), but **falls short of the full-A incumbent by ~8% relative**.
   The reconstructed full-A point is NOT exactly feasible in the hard-max oracle (KNITRO -300,
   independently LP-certified infeasible) — root-caused to an economically extreme implied
   `Aod_theta[4,4]` (≈41-122× calibration) that amplifies small (1-2 percentage point) share-matching
   gaps into a real, nonzero moment residual, not a smoothing/numerical artifact. **Answers the
   handover's question 6 directly: a properly-run sequential reconstruction does NOT beat the current
   full-A incumbent** — no warm-start action required on that basis (though it remains a reasonable
   diversity-of-starts candidate on weaker grounds).

### 1. Phase 3 (composite hybrid gradient) — DONE, validated, with an important caveat found

`full_aod_diag/d4_exact/composite_gradient.jl`: gamma'_focal component is an EXACT closed-form
derivative of `L_fix` (uses `base.m_star`, no re-evaluation of `Psi` needed); A-block is central FD
over `lfix_incremental.jl`'s `:incremental_o1` tier with a NEW adaptive-bandwidth-per-coordinate
selector (switching-mass target + floor/ceiling + h-vs-h/2 diagnostic), replacing the flagged
`FIXED_H=0.01` gap. Validated (`test_composite_gradient.jl`) at `upper_maxit40`/`lower_stalled`
(calibration/fixed_A are documented cold-infeasible, skipped): gamma matches the true tangent to
~1e-6 relative; A-block cosine 0.973-0.9997 against a MATCHED-bandwidth `Delta_FD` reference.

**Important, re-usable finding**: `Delta_dual`'s own finite-difference estimate is itself severely
h-dependent near these candidates (confirmed: the gamma FD slope moves from -76.2 at h=0.01 to -44.6
at h=1e-5 at `upper_maxit40`, while `L_fix`'s FD barely moves over the same range) — consistent with
the historical `results/fullA_d4/1bdb1cc/h_sweep.csv` finding that raw `Delta_dual` FD doesn't
stabilize even at h=0.00625. **Any future comparison against a `Delta_FD` "ground truth" MUST use a
matched (or small) h, or the comparison is measuring Delta_dual's own curvature bias, not the cheap
method's error** — this cost real time to discover this continuation (see composite_gradient.jl's
Phase 3a commit message) and should not need re-discovering.

Two real bugs were found and fixed in `HybridGradientPolicy` before trusting it (full detail in the
Phase 3b commit message): (1) the winner-jump trigger compared a whole-matrix hash for exact equality,
which fires on virtually every step at W=8000 draws — fixed to a genuine fractional-change threshold;
(2) the disagreement trigger was fed the gamma component, which is exact in the cheap method but
severely biased in the h=0.01 expensive method (see above), so it disagreed almost every call for
reasons unrelated to staleness — fixed by comparing `||A-block gradient||` instead. **NOT wired**
(documented gap): the "rejected-step" and "failed random-directional check" triggers — would need a
`KN_set_newpoint_callback` to detect KNITRO's own accept/reject decision, not implemented given time.

**A separate, more serious finding**: `hybrid` mode's gradient-SOURCE switching (cheap vs expensive
across outer iterates) causes KNITRO to terminate prematurely (status -102 after only ~2 outer
iterations) when paired with a quasi-Newton Hessian mode (SR1/BFGS/L-BFGS) — plausibly because those
methods' curvature updates assume a consistent gradient source across secant pairs, and switching
breaks that assumption. `lfix_composite` (always cheap, no switching) does not show this. **Treat
`hybrid` as not-yet-safe-to-use for a real run** until this is either fixed (e.g. reset/skip the
Hessian update on a source switch, if KNITRO's API allows signaling that) or `hybrid` is restricted to
`productfd`-style Hessian modes that don't accumulate cross-iterate curvature.

`run_d4_optimized_fd.jl` extended (backward compatible, default behavior unchanged, verified) with
`D4X_GRADIENT_METHOD` (`delta_fd`|`lfix_composite`|`hybrid`), `D4X_HESSOPT`
(`auto`|`sr1`|`lbfgs`|`productfd`), `D4X_MAXTIME_REAL` (wall-clock override via
`KN_set_param_by_name`, avoids one `.opt` file per budget), `D4X_REFRESH_EVERY`/`D4X_GAP_TOL`.

### 2. Phase 4 (wall-clock frontier) — minimum bar MET, decisively

See `docs/fullA_algorithm_frontier_v2.md` for the full table/interpretation. One wall-clock budget
(60s), one direction (upper), 7 configs. **`lfix_composite` (the always-cheap composite gradient, no
source-switching) beats the historical `delta_fd`+`productfd` control by +10.2% relative κ
(0.172457 vs. 0.156552) using 13-16x fewer inner CC-dual solves (114 vs. 1792), and modestly beats
even the best-tuned `delta_fd` variant found here (`deltafd_lbfgs`, 0.171774) too** — the clearest,
most decisive finding of this continuation. `hybrid` (both Hessian modes) underperforms
`lfix_composite` here (κ 0.1686-0.1696) because the refresh policy, even after the two bug fixes
above, still triggers an expensive refresh on the large majority of calls (7/47, 2/42 cheap) —
flagged as a real tuning gap for a future continuation, not hidden. Multiple budgets
(30/60/180/~324s per the task's full spec) and the lower direction were not run this continuation —
flagged as the natural next step given the time this continuation spent on Phase 3's derivation and
the two `HybridGradientPolicy` bugs. **Given `lfix_composite` alone already clears the bar
decisively, a future continuation should consider whether `hybrid`'s extra complexity is worth
pursuing further, or whether `lfix_composite` should simply become the new default.**

### 3. Phases 6-8 — NOT STARTED this continuation

Given the time spent on Phase 3 (composite gradient derivation + two real bugs found in the refresh
policy + the hybrid/quasi-Newton incompatibility discovery) and Phase 4's minimum bar, Phases 6
(gamma profile / upper polish / lower completion), 7 (nested-W stability), and 8 (D=6 pilot) were not
started. **Phase 6's starting point should now be the smoothed-consistent candidate
(κ=0.17197168927740825, §0.2 above), not `upper_maxit40`** — it is validated, exact-feasible, and
already ahead of the old incumbent before any of this continuation's own polishing is applied to it.

## CONTINUATION 3 UPDATE (in progress, mid-session) — read this section first

Branch/worktree unchanged. Commits so far this continuation: `f500490` (Phase 0: resume audit +
candidate registry), `6613e40` (Phase 1: dedupe moments + fast winners + corrected inner-solve
profiling), `0ddc364`/`bbc0e47` (Phase 2: incremental `L_fix` — block-local/incremental/O(1)-winner
tiers, all equivalence-tested at machine precision + separately profiled). Full detail:
`docs/fullA_continuation3_resume_audit.md`, `docs/fullA_performance_profile_v2.md`,
`docs/fullA_block_local_performance.md`.

**Phase 0-2 status: COMPLETE, all equivalence tests pass.** Headline: `incremental_o1` (Tier 3, O(1)
winner update) is 7.34x faster than a full-rebuild `L_fix` gradient single-threaded, 44.5x at
`JULIA_NUM_THREADS=16`, at D=4 — all cross-checked against `fixed_dual_L`/`evaluate_fullA` to machine
precision. Corrected the profiling INTERPRETATION per explicit user request: continuation 2's
"inner_solve=53%" conflated a moment-matrix BUILD (unavoidable, O(D²W)) with the actual (cheap) CC
dual optimization — nested timers now separate these; see `docs/fullA_performance_profile_v2.md`.

**Not yet done this continuation** (queued, not started or in progress when this was last saved):
Phase 3 (composite hybrid gradient + live outer solver — the natural next step, `lfix_incremental.jl`'s
`incremental_o1` tier is the A-block piece; still need the gamma-coordinate analytic/AD piece and
wiring into a KNITRO custom-gradient callback, following `run_d4_optimized_fd.jl`'s existing
`cb_G!`/`KN_set_cb_grad` pattern), Phase 4 (wall-clock-matched algorithm frontier), Phase 5 (reliable
sequential benchmark), Phase 6 (gamma profile / upper polish / lower completion), Phase 7 (nested-W
stability), Phase 8 (gated D=6+ pilot — blocked partly on a D=6 base-infeasibility issue found while
attempting a quick generalization check, see `docs/fullA_block_local_performance.md` §7).
**Two additional user-requested investigations were queued mid-session, not yet started**: (a) audit
and eliminate the dense `jac_h` (draw×moment×outer-param) tensor allocation in the cached/Method-B
full-A path where it's unused by the `L_fix`/hybrid gradient method — establish actual runtime
behavior via counters, not just grep, before changing anything; (b) a genuinely CONSISTENT smoothed
full-A solve (log-sum-exp values AND softmax allocation probabilities at the SAME temperature,
throughout — not just smoothing the min) as a controlled comparison against the exact hard-value
method, first fixing the known-nondeterministic `smoothed_frozen_adjoint_Q` diagnostic. Both are
substantial, separately-scoped investigations — see the conversation this session for the full
requirement text if picking this up without the original prompt.

## 1. Where everything is

- Repo: `git@github.com:habibiscoding/Trade-Model-Robustness.git`, branch `diag/fullA-d4-exact`,
  branched from production `53ffb58` (branch `sequential-profiled-gravity` in
  `trade_robustness_modular`).
- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`.
- **Current HEAD at the end of this continuation**: run `git log --oneline -1` in the worktree —
  the last commit of this session is titled "Required deliverables: docs/fullA_d4_W_stability.md..."
  or later if more commits followed. Pushed to origin throughout; verify with `git fetch origin
  diag/fullA-d4-exact && git log origin/diag/fullA-d4-exact -1`.
- New code: `full_aod_diag/d4_exact/` (unchanged discipline from prior sessions) plus, this
  continuation, one read-only-adjacent artifact in the **production** worktree
  (`trade_robustness_modular/sequential_gravity/batch_out_d4x_continuation/` — untracked output from
  running the existing `run_profiled_production.jl` script with `OUT_DIR` overridden; no production
  file was modified, only a new output directory written).
- Results: `results/fullA_d4/<commit>/<run_id>/`, same convention as before.

## 2. What this continuation completed (commits `1b2a3a0`..`46b461c` and later)

1. **Phase 0**: state verification (`docs/fullA_next_resume_audit.md`) — HEAD matched the task's
   expected `8ae56d0` exactly, no drift.
2. **Phase 1 (mandatory, complete)**: instrumentation (`instrumentation.jl`, `oracle_profiled.jl`,
   equivalence-tested), component profiling (`profile_components.jl` →
   `docs/fullA_performance_profile.md`), gradient-method cost with corrected counters
   (`profile_gradient_methods.jl` — resolves the "n_inner_solves=34" mislabel from the prior
   continuation's Phase D: `Q_adj_FD`/`L_fix_FD` do **zero** real inner solves, confirmed via
   `CS.INNER_SOLVE_COUNT[]` diffs), and a complete D/W baseline scaling matrix
   (`profile_D_W_scaling.jl` → `docs/fullA_scaling_projection.md`, D∈{4,6,8,10} at W=8000, D=4 at
   W∈{8000,20000,80000}).
3. **Phase 6**: blockwise gradient re-check (`phase6_blockwise_gradient_check.jl`) — the single most
   important correction this continuation made: `L_fix_FD` is the only cheap method that survives
   scrutiny in the gravity-tangent A-block; hard pathwise AD's A-block gradient is anti-correlated
   with truth near the optimum despite a deceptively good full-vector cosine.
4. **Phase 5 (mandatory external validity check)**: located and ran a fresh sequential/profiled
   production solve on the identical synthetic economy (confirmed via `kappa_point_estimate` match,
   not filename trust — two candidate result files found first were actually stale D=10 runs).
   Full-A's upper candidate (κ=0.1718) beats the sequential method's own genuinely-feasible number
   (κ=0.0779) by >2x; caveated (one start, not fully converged).
5. **Phase 3 (partial)**: W-stability cost scaling complete; one candidate feasibility check at
   W=20,000 (independent, not nested, draws) — reassuring, not exhaustive.
6. **Documentation**: all 9 required deliverables exist (`docs/fullA_next_resume_audit.md`,
   `docs/fullA_performance_profile.md`, `docs/fullA_d4_W_stability.md` (partial),
   `docs/fullA_d4_profile_and_bounds.md` (scoped, not attempted),
   `docs/fullA_algorithm_frontier.md` (scoped, not attempted),
   `docs/fullA_scaling_projection.md`, updated `docs/fullA_d4_final_report.md` §9 and
   `docs/fullA_d4_recommendation.md` (with the required final decision labels), this handoff).

## 2a. CORRECTION (post-hoc, flagged by the sequential method's domain owner): the Phase 5
    full-A-vs-sequential comparison was wrong and is retracted

The original version of this handoff (and §9.3 of `docs/fullA_d4_final_report.md`) reported the
sequential run's terminal κ_upper=0.0779 as a valid comparison point and concluded full-A "beats"
sequential by >2x. **This was an error**: 0.0779 is below this investigation's own fixed-A benchmark
(κ_fixedA=0.1440, `results/fullA_d4/a377fff/movement_and_fixedA_check.txt`), which a properly-run
sequential search should never fall below (it optimizes a strict superset of what "fixed-A" allows).
Root cause, confirmed not speculated: the run used the *default* opt file
(`full_aod_diag/csw_outer_25.opt`, `maxit=25`, `eval_fcga=yes`+`hessopt=4`→silently downgraded to
L-BFGS) and one start, not the production 5-start multistart `sequential_methodology.tex` §9 requires.
**To get a reliable comparison next time**: re-run `run_profiled_production.jl` with a much larger
`maxit`/`maxtime_real`, `eval_fcga=no`, and genuine multistart (5 starts per the production spec) —
none of which the retracted run used. Both `docs/fullA_d4_final_report.md` and
`docs/fullA_d4_recommendation.md` have been corrected; treat any other document or memory referencing
"full-A beats sequential" or "κ=0.0779" from this investigation as stale until a proper comparison
exists.

## 3. What was NOT attempted, and why (do not silently assume it's done)

- **Phase 2** (block-locality optimization, incremental moment updates, parallel FD, allocation
  reduction): profiling identified two concrete targets (redundant `moments_recompute` — ~34% of
  wall time; `winner_compute`'s per-column `sort()` allocation pattern) but neither was implemented.
  This is the highest-value next step given it's now evidence-backed, not speculative.
- **Phase 4** (gamma-profile, upper polish, lower-direction completion): not started at all. Scoped
  in `docs/fullA_d4_profile_and_bounds.md` with exactly what exists to build on.
- **Phase 7** (wall-clock-matched algorithm frontier): not started. The existing Phase B data from
  the PRIOR continuation is iteration-matched and explicitly flagged as not substituting.
- **Phase 8** (staged D pilots beyond the Phase 1C cost benchmark): no actual D=6/8/10 optimization
  pilot was run, only per-evaluation/per-gradient cost benchmarking. `docs/fullA_scaling_projection.md`
  gives the cost basis for deciding whether/when to attempt one.
- **W-stability breadth** (Phase 3): only the maxit=40 upper candidate was checked, only at
  W=20,000, only with independent (not nested) draws, only via a same-theta recheck (no
  re-optimization). The lower candidate, poll-improved points, W=80,000, and nested draws are all
  outstanding.
- **Full A-matrix reconstruction of the sequential solution** (Phase 5's fuller requirement): only
  the gauge-invariant kappa comparison was done. Reconstructing the full competitiveness matrix
  requires re-running the sequential method's destination-share inversion for the non-focal columns
  and converting between the two gauges (`sequential_methodology.tex` §2.3) — not attempted.

## 4. Operational notes for whoever continues

- **Background jobs in this environment took much longer than expected but generally DID complete
  successfully** — several jobs that appeared to have "died" (no process found via `ps`, log file not
  growing for several minutes) turned out to still be running and completed correctly 10-20+ minutes
  later with a delayed completion notification. Do not conclude a background job has failed just
  because `ps` shows nothing and the log looks stale after a few minutes — wait for the actual
  completion notification before relaunching, or you will get duplicate/overlapping runs (which
  happened this session with the sequential production run — harmlessly, because
  `run_profiled_production.jl` has its own resume-skip logic keyed on result files already present in
  `OUT_DIR`, but do not rely on every script having that safety net).
- `profile_D_W_scaling.jl` now supports a `CONFIGS_SUBSET` env var (e.g. `CONFIGS_SUBSET=D8` or
  `CONFIGS_SUBSET=D10,W80000`) for resuming one config at a time — use this if a full-matrix run is
  interrupted again, rather than re-running everything.
- The sequential production run lives in the **production worktree**
  (`trade_robustness_modular/sequential_gravity/run_profiled_production.jl`), invoked with
  `DVAL=<D> DELTA_GRID=<comma-list> BOUND=<upper|lower|both> OUT_DIR=<path> julia --project=.
  sequential_gravity/run_profiled_production.jl`. Always set a **fresh, dedicated `OUT_DIR`** — the
  default (`sequential_gravity/batch_out`) has already been overwritten at least once across
  different D values by earlier sessions, which is exactly how this continuation nearly used a stale
  D=10 result under a D=4-looking filename (caught by checking the `D` field in the loaded JLD2, not
  the filename — always do this).

## 5. Next command to run

Given the priority ordering established in `docs/fullA_next_resume_audit.md` and reaffirmed by this
continuation's findings (Phase 1's redundant-moments finding is now the highest-value, lowest-risk
target):

```
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-d4
julia --project=. full_aod_diag/d4_exact/test_oracle_profiled.jl   # confirm nothing has drifted
```

then start Phase 2 by implementing the `moments_recompute` redundancy fix (cache `inner_loop_internal`'s
own internal `obj.moments!` output rather than recomputing in `oracle.jl`'s `evaluate_fullA`) as a new,
additive, equivalence-tested variant — following exactly the pattern `oracle_profiled.jl` already
established (mirror, don't modify, equivalence-test before trusting).
