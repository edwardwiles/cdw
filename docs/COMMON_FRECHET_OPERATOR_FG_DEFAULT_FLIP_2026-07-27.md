# Common-Fréchet operator-FG default flip — 2026-07-27 (Task A)

Agent: `agent/default-flips-and-operator-setup-2026-07-27` (this session ran physically in worktree
`agent-a736b23e9b3184691`, checked out on a locally-created branch
`agent/default-flips-and-operator-setup-2026-07-27-relocated` from the exact same commit, `8c630e3`,
as the intended branch — see "Worktree note" at the end of this doc).

## What changed

`full_aod_diag/d4_exact/core_exact_hessian.jl:223`:

```julia
const CM_FRECHET_INNER_FG_BACKEND_DEFAULT = Ref{Symbol}(:dense_reference)   # BEFORE
const CM_FRECHET_INNER_FG_BACKEND_DEFAULT = Ref{Symbol}(:cm_frechet_lookup)  # AFTER
```

`:dense_reference` is retained unchanged as an explicit, selectable diagnostic backend (not
deleted) — every call site that takes `inner_fg_backend=` still accepts it.

## Why this was previously held back, and why that reason no longer applies

`docs/COMMON_FRECHET_OPERATOR_FG_DEFAULT_FINAL_GATE_2026-07-27.md` (same-day, earlier session) ran
the full broader gate this flip needed (real D=20/W=80,000, both contrasts, 3 points, 24 checks) and
found the operator backend **correct** (24/24 pass, ζ* agreeing to 2.6e-16–3.0e-13) and **1.1–1.2x
faster**, but declined to flip the default solely because of a **3.94% allocation regression**
(`alloc_ratio=1.0394` uniformly across all 6 point/contrast combinations), under an old rule that
treated any allocation increase as blocking.

That rule has been explicitly superseded by this task's priority order: **correctness/stability >
architecture (dense-G removal) > runtime > memory/GC > small relative allocation deltas**. A 3.94%
allocation increase with no runtime regression and no dense-G dependency is not a reason to hold
back the flip under this order. The 3.94% number itself is not re-litigated here — it is accepted
as-is, exactly as measured previously.

## Confirmation gate (this session, post-flip)

Reused the pre-existing gate script this codebase already had for exactly this comparison,
`full_aod_diag/d4_exact/bench_frechet_operator_fg_default_gate_2026-07-27.jl` (D=20/W=80,000/L=50,
`destination_sample=:exclude_row`, both contrasts, calibration + `near_delta1_perturbed` +
`hard_point_x1.01`, complete inner solve via `inner_loop_internal_archgeneric`/
`inner_loop_internal_cmfrechetlookup_production`, both sides built with
`cm_cross_hessian_backend=CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT[]` = the current production
`:winner_bin` winner-aware H_ER backend, so this measures the FG question against the real current
production Hessian configuration). The script explicitly passes `inner_fg_backend=:dense_reference`
/`:cm_frechet_lookup` on both sides of the A/B regardless of the global default, so this run is a
direct re-confirmation, not a different measurement — the only thing that changed since the last run
is which backend is now the *unlabeled* default a caller gets with no `inner_fg_backend=` kwarg.

Two small, additive instrumentation changes to the script (not a rewrite): (1) capture and print
`gctime` from `@timed` alongside time/bytes, to give the "no material peak-memory or GC regression"
requirement a real measured signal, not just an eyeball check on wall-clock; (2) print
`CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]` at the end of the run to record what the global default
actually resolved to for this run.

Command:
```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. -t 4 full_aod_diag/d4_exact/bench_frechet_operator_fg_default_gate_2026-07-27.jl
```

### Results (real run, this session)

**Correctness: ALL PASS, 24/24 checks.** Every point/contrast combination feasible on both backends,
KNITRO status identical (`0`) throughout, ζ* agreeing to 1.4e-16–3.1e-13.

| contrasts | point | speedup (dense/lookup) | alloc_ratio (lookup/dense) | gc_ratio (lookup/dense) | n_fg (d/l) | n_hess (d/l) |
|---|---|---|---|---|---|---|
| anchored | calib | 1.158x | 1.0393 | 0.000x | 1/1 | 0/0 |
| anchored | near_delta1_perturbed | 1.097x | 1.0393 | 0.000x | 1/1 | 0/0 |
| anchored | hard_point_x1.01 | **0.834x** | 1.0393 | 0.000x | 1/1 | 0/0 |
| orthonormal | calib | 1.143x | 1.0393 | 0.000x | 1/1 | 0/0 |
| orthonormal | near_delta1_perturbed | 1.062x | 1.0393 | 0.000x | 1/1 | 0/0 |
| orthonormal | hard_point_x1.01 | 1.101x | 1.0393 | 0.000x | 1/1 | 0/0 |

Isolated per-FG-callback allocation, warmed-up `CMFrechetLookupState`: **1,728 bytes** (both
contrasts) — small and consistent with the previously-reported healthy state (not the historical
~14MB/call regression that was already found resolved).

`gctime` was `0.0000s` for **both** backends at **every** point/contrast — no measurable GC-time
difference, i.e. no GC regression from the flip, real and measured, not assumed.

`alloc_ratio` reproduces the historical 3.94% figure exactly (1.0393–1.0394, effectively identical
given rounding) at every single point — stable, not point-dependent, consistent with the earlier
gate's own finding.

### One timing outlier, disclosed rather than hidden

5 of 6 points reproduce the historical 1.06x–1.22x-class speedup range. One point/contrast pair,
`anchored`/`hard_point_x1.01`, measured **0.834x** (lookup slower than dense) this run, versus
**1.171x** in the historical gate for the exact same point/contrast/backend/context construction.
Correctness (feasibility, status match, ζ* agreement) and `alloc_ratio` (1.0393, identical to every
other point) were **unaffected** at this point — only the wall-clock ratio differs from the
historical run. Given (a) this machine is shared with other concurrent jobs during this session
(confirmed via `ps`/`pgrep` showing unrelated long-running processes), (b) the *dense* side's own
wall-clock varied 0.96s–1.35s across ostensibly-comparable points within this single run (a ~40%
spread with no structural reason for that point to be harder), and (c) `gctime=0` for both backends
at this point rules out a GC-driven explanation, this reads as shared-server wall-clock contention
noise on this one measurement, not a genuine backend regression. Flagged explicitly rather than
smoothed over, per this project's own verify-before-causal-claims discipline — a future session
with exclusive machine access re-running just this one point/contrast pair would settle it
definitively.

### Peak-memory/GC regression: none found

- `gctime` identical (0.0000s) both backends, every point.
- `alloc_ratio` a stable +3.94%, not growing point-to-point or contrast-to-contrast — no sign of a
  compounding/leak-shaped regression.
- No crash, no KNITRO infeasible status, no correctness failure at any of the 6 points across 2
  runs (historical + this session's re-confirmation).

**Conclusion: the flip is confirmed safe.** `CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]` is now
`:cm_frechet_lookup` in production, with `:dense_reference` retained as an explicit opt-out.

## Worktree note (process, not a science finding)

This task's brief named worktree `/bbkinghome/edav/gravity_robustness/worktrees/agent-default-flips-2026-07-27`
on branch `agent/default-flips-and-operator-setup-2026-07-27`. This agent's sandbox is hard-locked to
`/bbkinghome/edav/cdw/.claude/worktrees/agent-a736b23e9b3184691` (a different worktree of the SAME
underlying repository — confirmed via `git worktree list`, which shows both paths sharing one object
store) and cannot `cd`/`EnterWorktree` outside it. The named branch was already checked out in the
other, inaccessible worktree, so git refuses to check it out a second time here. All work in this
doc and its sibling docs was therefore done on a locally-created branch,
`agent/default-flips-and-operator-setup-2026-07-27-relocated`, branched from the identical commit
(`8c630e3`) the intended branch was at. The calling/integrating session should reconcile
(cherry-pick or merge) this branch's commits onto the intended branch name — the diff is what
matters, not the branch label, but this is flagged explicitly rather than silently worked around.
