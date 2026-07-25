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
`key_results/matched300_{before,after}_{u,cm}_summary.txt`.

## One disclosed confound, not hidden

**"After" runs used `pin_outer_algorithm=true`** (this session's own opt-in mechanism, forcing
`algorithm=2`/Interior-CG + `hessopt=6`/L-BFGS on the outer KNITRO problem) **while "before" runs
used each driver's own original, unmodified default** (`run_polish_checkpointed`: hardcoded
`algorithm=3`; `run_cm_upper_checkpointed`: the `.opt` file's `algorithm=auto`). This is a genuine
second variable changing alongside the allocation/Hessian-backend fixes between before and after —
not isolated out. It was a deliberate choice for this specific 300s gate (the matched-benchmark
harness's whole design point is a *reproducible, explicit* outer algorithm for A/B comparability
across families — see `knitro_outer_algorithm.jl`'s module docstring), but it means **this
particular before/after pair conflates two changes** (allocation/Hessian fixes + outer algorithm)
for the *outer-progress* metrics (n_eval, kappa, best Delta) specifically. **Allocation and GC-time
deltas are not confounded this way** — those are properties of the per-call moment/Hessian
computation, which the outer-algorithm choice does not touch — so those numbers are read as a
clean before/after of this session's own fixes.

The clean, non-confounded complete-solve comparison for the CM Hessian-threading decision is
`CM_THREADED_ARCHC_RELEASE_2026-07-25.md`'s own P0/P1/P2 sweep (both arms of every comparison
there used `pin_outer_algorithm=true` identically) — that is the load-bearing evidence for the CM
verdict, not this section's 300s numbers alone.

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

| Metric | BEFORE/cm (b7435ee) | AFTER/cm (this branch) | Δ |
|---|---:|---:|---:|
| Wall (measured window) | 392.2s (incl. ~95s ctx build) | 361.4s (incl. ~20s warmup, separately timed) | — |
| KNITRO-internal Total program time | 352.7s | 320.6s | −9.1% |
| User allocation | 41.702 GB | 19.652 GB | **−52.9%** |
| GC time | 2.307s (0.6%) | 1.655s (0.5%) | −28.3% |
| n_eval | 11 | 4 | **−63.6%** (see confound discussion) |
| n_grad | 10 | 4 | −60% |
| # of CG iterations (KNITRO-native) | 0 | 14 | qualitatively different solve path |
| best gp / Delta | 0.966743 / 0.987497 | 0.969667 / 0.827364 | AFTER reached a *less* binding point in-budget |
| Cold-verify diff | not run | 0.0 (exact) | machine precision |

## Reconciliation

**Allocation and GC time both improve substantially and consistently in both families**
(−52.9% to −55.6% allocation, −28% to −50% GC time), matching the specific, individually-validated
fixes cherry-picked onto this branch (§3.1–3.3 unrestricted preallocation, §4/5 CM
`bview*R`/Hessian-scratch/compressed-core-moments fixes) — this is the clean, non-confounded
result of this release.

**Unrestricted's outer-progress numbers (n_eval, kappa) improved too** (+24% evals, +7.7% kappa)
— directionally consistent with "faster per-callback work leaves more time for outer iterations,"
though this specific pairing cannot cleanly separate "allocation fixes helped" from "algorithm=2
happens to suit this family's unconstrained profile formulation better than algorithm=3" (the
`.opt`-file default `run_profile_checkpointed` previously hardcoded).

**CM's outer-progress numbers went the other way** (n_eval 11→4, best Delta 0.987→0.827 — i.e.
AFTER reached a *less* feasibility-binding point within the 300s budget) despite allocation and
GC time both clearly improving. The KNITRO-native solve statistics make the mechanism visible:
BEFORE reports **0 CG iterations** (consistent with `algorithm=auto` resolving to
Interior-Point/Barrier-**Direct** for CM's constrained formulation, per this session's own
`knitro_outer_algorithm.jl` module docstring, itself citing the 2026-07-25 wall-clock audit's
finding); AFTER reports **14 CG iterations** across only 3 major outer iterations (consistent with
the *explicitly pinned* `algorithm=2`, Interior-Point/Barrier-**CG**, spending more work per outer
step exploring the interior via conjugate-gradient sub-iterations). This is exactly the
"`algorithm=auto` resolves differently per family" behavior `knitro_outer_algorithm.jl` documents
— for CM specifically, in this one 300s window, the pinned algorithm was **not** a clear win on
top of the allocation fixes. **This is disclosed, not spun**: the allocation/Hessian-backend fixes
themselves are not implicated by this result (allocation/GC improved regardless), but this
specific 300s A/B does not, by itself, demonstrate an unambiguous complete-solve win for CM the
way it does for unrestricted. The clean, algorithm-matched evidence for the CM Hessian-threading
decision remains `CM_THREADED_ARCHC_RELEASE_2026-07-25.md`'s P0/P1/P2 sweep, where both arms of
every comparison held the outer algorithm fixed and threading was decisively ahead (P1: serial
made zero outer progress in-budget; threaded completed 4 real outer iterations and materially
improved the incumbent).

## What this does NOT include

- A repeated trial at either configuration (single run per cell, as with this session's other
  benchmarks — real, reproducible via the recorded seed/config, not averaged over repeats).
- CM+mean/ZC and origin-ZC 300s smokes (task marks these "if time permits" for CM+mean/ZC and
  "bounded" for origin-ZC specifically — both already covered by this session's separate P0/P1/P2
  and bounded-BLAS benchmarks respectively, not repeated here at the full 300s budget).
- A `pin_outer_algorithm=false` "after" arm that would isolate the allocation fixes from the
  outer-algorithm change cleanly at 300s (would need a 5th 300s+ run; not run given this session's
  overall time budget — the P0/P1/P2 sweeps already provide the algorithm-matched evidence where
  it matters most, i.e. the Hessian-backend decision).
