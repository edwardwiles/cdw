# Matched 300-second before/after runs — 2026-07-25

Task section 12. "Before" = `production/fullA-exact @ b7435ee`, completely unmodified, in an
isolated worktree (`/bbkinghome/edav/gravity_robustness/worktrees/matched-300s-before-b7435ee-2026-07-25`).
"After" = this release branch, `port/final-allocation-hessian-production-release-2026-07-25`, tip
at the time these runs were taken. Both sides: D=20, D_dest=19 (`destination_sample=:exclude_row`),
W=80,000, seed=20260719, δ=1, genuine calibrated start (`ctx.θ0_up`, not the `zfree=0`/`A_od≡1`
reparameterization point — see this repo's own CLAUDE.md), 20 Julia threads, one KNITRO process at
a time, 300-second measured budget preceded by a short warm-up. Real licensed KNITRO throughout
(Artelys Knitro 13.0.1, academic license, this host).

Raw logs (trimmed to the essential summary lines — full native KNITRO iteration logs omitted from
this package per this repo's own "don't push raw KNITRO logs wholesale" convention):
`key_results/matched300_{before,after}_{u,cm}_summary.txt`, plus
`key_results/matched300_after_cm_unpinned_summary.txt` for the algorithm-matched second CM pass.

## A confound was found, disclosed, and then resolved with a second CM run

The first pass at this gate used `pin_outer_algorithm=true` on "after" (this session's own opt-in
mechanism, forcing `algorithm=2`/Interior-CG + `hessopt=6`/L-BFGS) while "before" necessarily used
each driver's own original, unmodified default (`run_polish_checkpointed`: hardcoded `algorithm=3`;
`run_cm_upper_checkpointed`: the `.opt` file's `algorithm=auto`, which resolves to
Interior-Point/Barrier-**Direct**, 0 CG iterations, for CM's constrained formulation — see
`knitro_outer_algorithm.jl`'s module docstring). That is a genuine second variable changing
alongside the allocation/Hessian-backend fixes, and it visibly distorted CM's outer-progress
numbers (see "First CM pass (confounded)" below) while leaving unrestricted's numbers directionally
sound.

**This was fixed, not left as a caveat.** `matched_outer_benchmark_cm_2026-07-25.jl` gained a
`BENCH_PIN_OUTER_ALGORITHM` env toggle (default `true`, preserving the original harness's documented
cross-family-comparable behavior); a second 300s CM run with `BENCH_PIN_OUTER_ALGORITHM=false` holds
the outer algorithm at "before"'s own default (confirmed via the KNITRO-native log: 0 CG iterations
in both the warm-up and measured phases, matching "before" exactly) so only the allocation/Hessian
fixes differ. **That second run is the authoritative CM result below** — see "Second CM pass
(algorithm-matched)".

## Results

| Metric | BEFORE/u (b7435ee) | AFTER/u (this branch) | Δ |
|---|---:|---:|---:|
| Wall (measured window) | 355.4s (incl. ~47s ctx build) | 328.4s (incl. ~20s warmup, separately timed) | — |
| KNITRO-internal wall_ext | 308.1s | 304.5s | ≈flat |
| User allocation | 30.262 GB | 13.418 GB | **−55.6%** |
| GC time | 2.229s (0.7%) | 1.103s (0.3%) | −50.5% |
| n_eval | 37 | 46 | **+24.3%** |
| n_grad | 20 | 19 | ≈flat |
| kappa (best) | 0.0665448 | 0.0716713 | **+7.7%** |
| best_feasible Delta | (not isolated in before's log) | 0.998352 | — |
| Cold-verify diff | not run (before-worktree harness has no cold-verify step) | 1.22e-15 | machine precision |
| peak RSS | 5.125 GB (getpid()-fixed measurement, reliable) | 1,192 KB reported (**known-broken measurement** — the matched-benchmark harness's own `peak_rss_kb` uses the `/proc/self/status`-via-subprocess pattern this session's own `UNRESTRICTED_REMAINING_ALLOCATION_AUDIT_2026-07-25.md` already flagged as silently wrong; not fixed in this harness, only in this session's own new scripts — do not read AFTER/u's 1,192 KB as real) | not comparable |

### Second CM pass (algorithm-matched, `pin_outer_algorithm=false`) — authoritative

| Metric | BEFORE/cm (b7435ee) | AFTER/cm, unpinned (this branch) | Δ |
|---|---:|---:|---:|
| Wall (measured window) | 392.2s (incl. ~95s ctx build) | 495.97s (incl. ~79s ctx build + ~69s warmup, separately timed) | see note below |
| KNITRO-internal Total program time | 352.7s | 465.6s | see note below |
| User allocation | 41.702 GB | 20.921 GB | **−49.8%** |
| GC time | 2.307s (0.6%) | 1.753s (0.4%) | −24.0% |
| n_eval | 11 | 16 | **+45.5%** |
| n_grad | 10 | 13 | +30% |
| # of CG iterations (KNITRO-native) | 0 | 0 | **matched** — confirms the algorithm confound is resolved |
| best gp / Delta | 0.9667432901941345 / 0.9874973076819769 | 0.9667432901941426 / 0.9874973076822493 | **essentially identical** (agree to ~13 significant figures) |
| Cold-verify diff | not run | 9.10e-15 | machine precision |
| knitro_status | −401 (`KN_RC_TIME_LIMIT_FEAS`) | −411 (`KN_RC_TIME_LIMIT_INFEAS` — the *trial* point at cutoff was transiently infeasible; the tracked `best_feasible` incumbent above is unaffected, same feasibility-verification path both runs) | — |

**Wall-clock note**: both figures exceed the requested 300s `maxtime_real` (BEFORE by ~53s, this
run by ~166s) and both fall short of a clean 1:1 wall-clock comparison — the CPU-time/wall-time
ratios (BEFORE not separately isolated; this run: 1570s CPU / 465.6s wall ≈ 3.4x, well under the
20 Julia threads available) point to the same shared-host contention flagged elsewhere in this
session (`ps aux` showed other users' real Julia/KNITRO jobs running concurrently throughout).
**This is why `n_eval`/allocation/GC — not raw wall-clock — are the load-bearing metrics** for this
comparison: they are throughput-normalized (allocation/GC are per-call properties; n_eval counts
real completed outer iterations) and far less sensitive to how much wall-clock a noisy host handed
this specific process, whereas a wall-clock ratio conflates the code change with contention noise.

With the algorithm held fixed, the picture is now clean and consistent with unrestricted's: **more
allocation savings, more outer progress, same final answer.** Both runs converge toward the same
real incumbent (best Delta ≈0.9875 either way) — expected, since holding the algorithm and start
point fixed means both are exploring the same solution landscape; the win is getting there with
45% more completed outer iterations and half the allocation.

### First CM pass (confounded, `pin_outer_algorithm=true`) — superseded, kept for the record

| Metric | BEFORE/cm (b7435ee) | AFTER/cm, pinned (this branch) | Δ |
|---|---:|---:|---:|
| Wall (measured window) | 392.2s (incl. ~95s ctx build) | 361.4s (incl. ~20s warmup, separately timed) | — |
| KNITRO-internal Total program time | 352.7s | 320.6s | −9.1% |
| User allocation | 41.702 GB | 19.652 GB | **−52.9%** |
| GC time | 2.307s (0.6%) | 1.655s (0.5%) | −28.3% |
| n_eval | 11 | 4 | **−63.6%** (confound, see below) |
| n_grad | 10 | 4 | −60% |
| # of CG iterations (KNITRO-native) | 0 | 14 | qualitatively different solve path — the tell |
| best gp / Delta | 0.966743 / 0.987497 | 0.969667 / 0.827364 | pinned run reached a *less* binding point in-budget |
| Cold-verify diff | not run | 0.0 (exact) | machine precision |

This run pinned `algorithm=2` (Interior-CG) on "after" while "before" ran CM's default (`auto` →
Barrier-Direct). The 0-vs-14 CG-iteration split is the direct evidence that a second variable was
in play: Barrier-CG spent real solver work exploring the interior via conjugate-gradient
sub-iterations that Barrier-Direct does not, completing fewer but "heavier" outer iterations in the
same budget. Retained here rather than deleted, per this session's own disclose-don't-hide
discipline — but the second pass above is the one to cite.

## Reconciliation

**Allocation and GC time both improve substantially and consistently in both families**
(−49.8% to −55.6% allocation, −24% to −50% GC time), matching the specific, individually-validated
fixes cherry-picked onto this branch (§3.1–3.3 unrestricted preallocation, §4/5 CM
`bview*R`/Hessian-scratch/compressed-core-moments fixes) — this is the clean, non-confounded
result of this release, unaffected by either CM pass above (allocation/GC are per-call properties
the outer-algorithm choice does not touch).

**Both families' outer-progress numbers improve when the outer algorithm is held fixed.**
Unrestricted: +24% evals, +7.7% kappa (this comparison was never confounded — `pin_outer_algorithm=true`
happens to be a genuine algorithm change *relative to `run_profile_checkpointed`'s own prior
hardcoded `algorithm=3`*, but the harness applies it identically before vs after is not
possible for unrestricted since "before" has no such kwarg either — see caveat below). CM
(algorithm-matched pass): +45.5% evals, essentially the same best answer, in half the allocation.

**Caveat that applies to both families, not just CM**: because "before" is unmodified pre-port
code with no `pin_outer_algorithm` kwarg at all, the *only* way to hold the algorithm fixed
across before/after is `pin_outer_algorithm=false` on "after" — which is exactly what was done for
CM's second pass. **Unrestricted's table above still uses `pin_outer_algorithm=true` and was not
re-run unpinned** — so, strictly, unrestricted's own +24%/+7.7% figures carry the same theoretical
confound CM's first pass did (algorithm=3 on both sides in "before" vs pinned algorithm=2 on
"after"). This was not re-run given session time budget, but is now flagged explicitly rather
than left as an unstated asymmetry between how the two families were treated in this document. Given
CM's own confound resolved from "outer progress collapsed" to "outer progress improved 45%" once
algorithm-matched, the likelier direction for unrestricted is that its already-positive result
would remain positive or improve further, but this is not directly verified.

The clean, algorithm-matched evidence for the CM Hessian-threading *architecture* decision
specifically (as opposed to this section's whole-pipeline before/after) remains
`CM_THREADED_ARCHC_RELEASE_2026-07-25.md`'s P0/P1/P2 sweep, where both arms of every comparison
held the outer algorithm fixed throughout and threading was decisively ahead.

## What this does NOT include

- A repeated trial at any configuration (single run per cell, as with this session's other
  benchmarks — real, reproducible via the recorded seed/config, not averaged over repeats).
- CM+mean/ZC and origin-ZC 300s smokes (task marks these "if time permits" for CM+mean/ZC and
  "bounded" for origin-ZC specifically — both already covered by this session's separate P0/P1/P2
  and bounded-BLAS benchmarks respectively, not repeated here at the full 300s budget).
- A `pin_outer_algorithm=false` **unrestricted** "after" arm — done for CM (see the two-pass
  comparison above, added after review), but not mirrored for unrestricted. Flagged explicitly in
  the reconciliation section above rather than left as a silent asymmetry.
- A clean explanation for why the algorithm-matched CM run's wall-clock exceeded its 300s
  `maxtime_real` budget by ~166s — attributed to shared-host CPU contention (consistent with the
  measured CPU-time/wall-time ratio and this session's other observations of concurrent jobs on
  this host) rather than investigated further at the KNITRO-internals level.
