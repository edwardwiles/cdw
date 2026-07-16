# Handoff: recover_lfd bug fix + head-to-head comparison cleanup (2026-07-16)

Paste this whole file's content, or point a new Claude at this file, to continue. Previous
session hit context limits mid-investigation; this captures everything learned so a new
session can pick up WITHOUT re-deriving it.

## Where this lives

Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf` (Julia project, branch
`feature/sequential-inversion-perf`, **not a git repo with commits made this session** -- all
changes below are uncommitted working-tree edits). KNITRO only licenses on `demand.mit.edu`:

    export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
    export KNITRODIR=/opt/shared_sw/knitro/14.2.0
    export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
    export PATH="$HOME/.juliaup/bin:$PATH"
    cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
    julia -t 19 --project=. <script>   # -t 19 + PARALLEL_INVERSION=true for D=20 real data

D=20 real-data env: `FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true`.

**Note**: the machine is heavily loaded by other users (1165+ julia processes system-wide
observed this session; one unrelated python3 process got OOM-killed by the kernel). If a
launched script disappears from `ps` with no error/crash trace in its own log, this is the
likely explanation (see "Incomplete work" below) -- not necessarily a bug in the script itself.
Check `dmesg | tail` for OOM-kill lines mentioning your PID before assuming a code bug.

## THE BIG FINDING: `recover_lfd` silently accepted non-converged inner solves

### Context: what was being done

A 4-method head-to-head comparison (LC=local-constrained/KNITRO, LU=local-unconstrained/KNITRO,
GC=global-constrained/BlackBoxOptim, GU=global-unconstrained/BlackBoxOptim) at D=20 real data,
3 divergence targets (T1=0.1, T2=1.0, T3=2.0 budget), 3 starts each (Astar/rand1/rand2-or-warm)
= 9 solves/method, all under `sequential_gravity/head_to_head/`. See
`sequential_gravity/derivative_diagnostics/HEAD_TO_HEAD_PROMPT.md` for the original task spec
(now superseded/extended by everything below) and `sequential_gravity/head_to_head/*.jl` for
all the driver scripts (`run_lc.jl`, `run_lu.jl`, `run_gc.jl`, `run_gu.jl`,
`run_lu_multistart50.jl`).

**All 4 official methods' runs are DONE** (`ls sequential_gravity/head_to_head/*_ALLDONE`):
`lc_ALLDONE`, `lu_ALLDONE`, `lu_multistart50_ALLDONE` exist. GC finished all 9 (`out_gc/*.jld2`
has 9 files) but has **no `gc_ALLDONE` file** because the user asked to kill it mid-checkpoint-
write of its own summary after the T2/warm corruption was found (see below) -- its 9 solve
files are all on disk and valid to read, just the run script itself never got to print
`GC_HEAD_TO_HEAD DONE`. GU was manually terminated by user request at 8/9 (T3/warm never run,
`out_gu/` has 8 files, no `gu_ALLDONE`).

### The bug

During review, one GC result (T2/warm) looked suspiciously good: `kappa=0.1055` (nominal
delta=1.0 budget), which is far better than LC's own carefully-validated result at the same
budget (kappa=0.0815) and even beats T3's own looser-budget reference (kappa=0.0915). This
looked wrong to the user and was investigated.

**Root cause, fully confirmed** (not a hypothesis -- traced through the code and empirically
verified via direct re-runs):

1. `inner_loop_internal(obj::PsiObjectiveBundleDelta, θ)` in `cc_algo/inner_loop_functions.jl`
   (lines ~222-238) already does the right thing internally:
   ```julia
   nStatus, objSol, x, lambda_ = inner_loop_KNITRO(obj)
   if nStatus ∈ [0, -100, -101, -103]
       obj.x .= x
       return objSol, x, nStatus
   else
       obj.x .= NaN          # only this CACHE field gets NaN'd
       return -1e10, x, nStatus   # the RETURNED x is NOT NaN'd -- still raw KNITRO garbage
   end
   ```
2. `recover_lfd` (the function that turns an inner KNITRO dual solve into an LFD reweighting
   `p`) calls `inner_loop` and then only checks `all(isfinite, x))` -- it never looks at
   `nStatus` or the `-1e10` sentinel in the returned objective value. Since the *returned* `x`
   is not NaN'd on failure (only the object's cache field is), `isfinite(x)` frequently still
   passes even when the solve genuinely failed.
3. Directly confirmed at GC's T2/warm theta: re-running `recover_lfd`'s own `inner_loop` call
   gives **`nStatus = -300`** (`KN_RC_UNBOUNDED` in KNITRO's own status code table --
   confirmed via `libknitro.jl:2545: const KN_RC_UNBOUNDED = -300`), with `all(isfinite, x) ==
   true`. So `recover_lfd` silently derived an LFD `p` from a dual solve KNITRO itself was
   reporting as UNBOUNDED, i.e. not a real solution at all.
4. Consequence: everything computed *from* that bogus `p` (gravity residual `R`, divergence
   `div_p`) is internally self-consistent (looks fine) because it's derived from the SAME bad
   `p` -- but the actual trade shares (which depend on `p` matching real economic moments) are
   badly wrong. Confirmed directly: at GC's T2/warm point, trade-share error was **6.2%** (vs a
   1e-4 pass threshold; genuine points show ~1e-6 to 1e-7) and the focal CC first-order
   condition `E_p[G_focal]=0` was violated by **8.45** (vs ~1e-14 to 1e-16 at genuine points).
5. **This is not a redundant-solve issue.** The user asked whether the outer loop already
   computes δ* separately and recover_lfd is wastefully re-solving it -- traced through and
   confirmed NO: `recover_lfd`'s own single `inner_loop` call already returns BOTH `val` (δ*-like
   objective) AND `x` (dual multipliers) together in one KNITRO solve. The bug is narrower and
   dumber than a redundant-solve problem: a real failure signal existed and was computed, it
   just wasn't being propagated into the return value that mattered, and the caller wasn't
   checking the signal that WAS available (`nStatus`).

### The fix (APPLIED to 11 files, all parse-checked OK)

Pattern applied everywhere: right after `val, x, nStatus = inner_loop(obj, θ)`, added
`nStatus ∈ (0, -100, -101, -103) || return fill(1.0 / W, W), false` (or the appropriate
tuple-arity variant for functions returning `(p, ok, G)` instead of `(p, ok)`), matching the
KNITRO-acceptable-status convention already used elsewhere in this codebase (e.g.
`outer_solve_nested_cached`'s own probe-solve check).

**Files fixed** (searched via `grep -rln "function recover_lfd" --include="*.jl" .` --
**this file list is the ONE place a fresh grep should be re-run to confirm no other copies
exist that were missed, especially if any code was added/copied since this session**):

1. `sequential_gravity/run_profiled_production.jl` -- **the primary one**, used by the whole
   head_to_head/ comparison. Has the fullest comment explaining the bug.
2. `full_aod_diag/gravity_seeded_initial_solve/run_real_outer_comparison.jl`
3. `full_aod_diag/gravity_seeded_initial_solve/setup_and_variant.jl`
4. `sequential_gravity/compare_D4_cached_vs_reference.jl`
5. `sequential_gravity/verify_batch_solutions.jl` (3-tuple return variant: `p, ok, G`)
6. `sequential_gravity/run_profiled_bounds_norm.jl`
7. `sequential_gravity/run_profiled_bounds.jl`
8. `sequential_gravity/run_sequential.jl` -- **structurally different**: this file's own
   `recover_lfd` already returns `nStatus` to ITS caller (doesn't hide it internally) but the
   caller printed it for diagnostics and never gated on it. Fixed by adding explicit `∈ (0,-100,
   -101,-103)` checks at both call sites (initial blind solve -> `error(...)` if bad; augmented
   iterative solve -> `break` out of the refinement loop if bad). This file is an early "Phase
   2b" demonstration script per its own header, likely lower-priority/not part of the active
   head_to_head pipeline, but was fixed for completeness per user instruction ("ensure production
   code does not have this bug").
9. `sequential_gravity/rerun_seq_upper_delta20_from_bestfeasible.jl`
10. `sequential_gravity/verify_D10_solution.jl` (3-tuple return variant)
11. `sequential_gravity/run_profiled_D10_methodB.jl`

**Fix confirmed working empirically**: re-ran `seq_gravcol` at GC's T2/warm theta post-fix --
now correctly returns `ok=false` (previously `true`). Script:
`sequential_gravity/head_to_head/smoke_test_recover_lfd_fix.jl`.

### Confirmed-corrupted points (via post-fix re-verification, NOT hypothesis)

Three points across the whole comparison are confirmed to have been silently accepted despite
a failed (status -300, UNBOUNDED) inner solve:

- **`out_lc/lc_T2_rand1.jld2`** (was LC's reported T2 "winner")
- **`out_lc/lc_T3_warm.jld2`** (was LC's reported T3 "winner")
- **`out_gc/gc_T2_warm.jld2`** (was GC's reported T2 "winner", the one that triggered this
  whole investigation)

All three: gravity residual and divergence looked fine (self-consistent with the bad `p`), but
trade shares were off by percent-level amounts and the CC first-order condition was violated by
order-1 amounts. **Since these were each their target's reported "winner," the ORIGINAL
head-to-head comparison table (if one was drafted before this session ended) is WRONG for LC's
T2 and T3 rows and GC's T2 row** -- the true winner for each of those target/method cells is
whichever of that cell's OTHER 2 starts is genuine (see re-audit results below for what's
already confirmed clean).

## Re-audit progress -- INCOMPLETE, needs to be finished

A systematic re-check of every saved point (post-fix `seq_gravcol` + a direct trade-share
check, no linearized-gravity-moment reconstruction needed -- checking gravity DIRECTLY via
`gravity_residual(umat, logτ, logw, σ)` on the fixed A_od is sufficient and simpler, per user's
own explicit guidance) was run via
`sequential_gravity/head_to_head/comprehensive_reaudit.jl`.

**Two false-positive bugs were found and fixed IN THIS RE-AUDIT SCRIPT ITSELF** (not in
production code) before trusting its output -- if re-running or writing a new version, avoid
reintroducing these:
1. First version applied the same `delta`/budget value to ALL 4 methods' `seq_gravcol` calls.
   LU/GU are **unconstrained** (fix gp, minimize δ* -- no budget to violate by construction), so
   using e.g. `δ=0.1` for a T1 LU point wrongly rejected LU's own legitimate achieved δ*
   (0.1147, which exceeds the "budget" only because LU was never subject to one). Fixed: use
   `δ=Inf` for method ∈ {lu, gu}, keep the real target budget only for {lc, gc}.
2. Points where LU/GU's OWN original run already reported "NO FEASIBLE RESULT" (e.g. LU's own
   T3, all 3 starts) still have a `best_feasible_Aod`/`Acol_best` key in their JLD2 (it's just
   the unchanged starting point, since `best_x[]` is initialized to `copy(x0)` and only updated
   on genuine improvement, with `best_feasible_delta_star`/`delta_star_best` left at `Inf`).
   Re-checking these produces a fast (~0.4s) rejection that is **NOT new corruption** -- it's
   the script re-confirming a result that was never claimed as feasible in the first place. When
   interpreting/finishing this audit, **check `isfinite(d["best_feasible_delta_star"])` (LU) /
   `d["delta_star_best"] < 1e3` (GU) before treating a "REJECTED" status as a new finding** --
   only points that were ORIGINALLY reported as feasible/genuine and NOW fail are actual new
   corruption.

**Results so far (fixed version, `comprehensive_reaudit2.log` in scratchpad -- see note below
on scratchpad persistence)**:

| Method | Target | Checked | Result |
|---|---|---|---|
| LC | T1 | Astar, rand1, rand2 | all 3 GENUINE |
| LC | T2 | Astar, rand1 (skip: already-confirmed-bad), warm | Astar+warm GENUINE, rand1 confirmed bad |
| LC | T3 | Astar, rand1, warm (skip: already-confirmed-bad) | Astar+rand1 GENUINE, warm confirmed bad |
| LU | T1 | Astar, rand1, rand2 | all 3 GENUINE |
| LU | T2 | Astar, rand1, warm | all 3 GENUINE |
| LU | T3 | Astar, rand1, warm | all 3 "rejected" but these are the **expected non-claims** described above (LU's own run found 0/3 feasible at T3) -- NOT new corruption |
| GC | T1 | Astar, rand1, rand2 | all 3 GENUINE |
| GC | T2 | Astar, rand1, warm (skip: already-confirmed-bad) | Astar+rand1 GENUINE, warm already known bad |
| GC | T3 | Astar, rand1 | both GENUINE (T3/warm never completed a full run -- see below) |
| GU | T1 | Astar, rand1, | GENUINE (both checked so far) |
| GU | T1 | rand2 | **NOT YET CHECKED -- script was interrupted here** |
| GU | T2, T3 | -- | **NOT YET CHECKED AT ALL** |

**The re-audit script (PID gone, not in `ps aux`) stopped/crashed silently right after printing
`[GU T1/rand2] checking...`** with no error trace in its own log. Given the machine-load/OOM
context noted above, this is plausibly an environmental kill, not a script bug -- but re-run
`sequential_gravity/head_to_head/comprehensive_reaudit.jl` (already has both fixes applied,
should just work) to finish checking GU's remaining 6 points (rand2, all of T2, all of T3) and
get the final complete picture. **Estimated ~15s/point, so ~90s** to finish what's left, or
just re-run the whole thing fresh (~34 points x ~15s ≈ 8-9 min) for a clean, complete log.

Given results so far show **zero new corruption beyond the original 3 points** (everything else
checked -- 23 of ~31 non-skip points -- came back GENUINE), it's plausible the corruption really
was isolated to those 3 extreme (large relΔA, unbounded-dual-solve) points, but this should be
CONFIRMED by finishing GU's remaining 6 points, not assumed.

**Separately, the 150-point `lu_multistart50` results were NOT re-audited at all** -- lower
priority (was reported to the user with real, plausible-looking findings: T1 48/50 feasible
best δ*=0.1065, T2 38/50 feasible best δ*=1.037, T3 2/50 feasible best δ*=2.318) but given the
scale of this bug, spot-checking a sample (especially the T3 2 "feasible" points, and T1/T2's
best-found points, which are the ones anyone would actually cite) would be prudent before fully
trusting those numbers either. Script pattern to adapt: same as
`comprehensive_reaudit.jl`'s LU branch, pointed at `out_lu_multistart50/lu_ms_*.jld2` instead.

## The 3 things the user explicitly wants from the NEXT session

Quoting the user directly:

> 1) A table or figure that compares all of the genuine results, including time taken.
> 2) Assurance that the production code has the LFD issue fixed everywhere.
> 3) An explanation of the current warm start procedures at various places in the code. Does
>    the inner CC delta^* solve use a warm start for the dual multipliers ever? Does the
>    inversion solver use a warm start?

### On (1): building the genuine-results table

Once the re-audit is finished/confirmed complete (see above), build the final comparison table
using ONLY genuine points. For each (method, target) cell, the "winner" should be recomputed
from just the genuine points in that cell (using each method's own winner criterion: LC/GC
maximize kappa among feasible; LU/GU minimize achieved delta* since gp is fixed for those). Pull
`wall` (per-solve time, already saved in every JLD2 file) for the "time taken" column. Note:
LC's `wall` for its bad points (T2/rand1, T3/warm) reflects wasted compute on a corrupted run --
worth reporting total wall time honestly (including the bad solves) alongside a separate
"time to genuine result" framing if useful.

For **LU-multistart50**, if spot-checked/re-audited and found trustworthy, the T3 finding (2/50
feasible, vs the official 9-solve LU run's 0/3) is a genuinely interesting, reportable result
about the value of broader multistart specifically for hard targets -- worth including as its
own section/callout, not folded into the main 4-method table (different experimental design,
150 solves not 3).

### On (2): confirming the fix is everywhere

Already done to the best of this session's ability: `grep -rln "function recover_lfd"
--include="*.jl" .` from the repo root found exactly 11 files, all 11 patched, all 11
parse-checked clean. **Re-run that exact grep command fresh** as the first step of the new
session to confirm (a) the count is still 11 (no new copies appeared), and (b) each one
actually contains the `nStatus ∈ (0, -100, -101, -103)` guard (grep for that string too). Also
worth a final semantic smoke test: re-run `sequential_gravity/head_to_head/smoke_test_recover_lfd_fix.jl`
once more to reconfirm GC's T2/warm point is still correctly rejected (nothing should have
reverted it, but cheap to double-check).

### On (3): warm-start procedures -- PARTIALLY investigated, needs a full pass

What's confirmed so far this session (from reading `cc_algo/inner_loop_functions.jl` and
`sequential_gravity/profiled_gravity.jl`):

- **`invert_destination`** (the SMOOTHED, ρ=2e-3 destination-share inversion, in
  `sequential_gravity/profiled_gravity.jl`) **DOES support warm-starting** via its own `u_init`
  keyword argument, and `seq_gravcol`'s own `invert_all`/`invert_omitted` closures (in
  `sequential_gravity/run_profiled_production.jl`) DO pass a warm start (`u_init_fn(d)` sourced
  from the previous iteration's `umat[:,d]`, or `warm[:,d]` from a prior outer-loop theta) when
  available. This warm-starting is validated/documented (see `full_aod_diag/
  sequential_inversion_performance/warm_start_report.md` if it exists, referenced in code
  comments) as giving a real speedup, not just a nicety.

- **The inner CC δ* dual solve** (`inner_loop`/`inner_loop_internal(obj::PsiObjectiveBundleDelta,
  ...)`, called via `recover_lfd` and via `exact_inner_divergence_at`/`build_fixed_dual_bundle`):
  the underlying `PsiObjectiveBundleDelta` struct (`cc_algo/PsiObjectiveBundle.jl:254`) HAS the
  infrastructure for this -- a `use_cached_x::Bool = false` field and an `x::Array{Float64,1} =
  NaN .* ones(outer_constr_index)` cache field, and `inner_loop_initial_values(obj::
  PsiObjectiveBundleDelta) = obj.use_cached_x && norm(obj.x) < 1e6 ? obj.x : zeros(...)`
  (`cc_algo/inner_loop_functions.jl:140`) -- i.e. warm-starting IS supported by the machinery.
  **But**: `recover_lfd` (`build_fixed_dual_bundle` too) constructs a **FRESH**
  `PsiObjectiveBundleDelta` object on every single call, with `use_cached_x` left at its default
  `false` -- so **as currently called, recover_lfd/exact_inner_divergence_at do NOT warm-start
  the dual multipliers across calls, even though the struct could support it.** This was
  directly demonstrated empirically this session: a warm-start test
  (`sequential_gravity/head_to_head/test_warmstart_audit.jl`) fed a good, converged dual vector
  from a nearby "easy" point as the initial `x` for a known-failing point, and it STILL failed
  with the same `nStatus=-300` (UNBOUNDED) -- so even if warm-starting were wired up here, it
  would not by itself have fixed the specific corruption found this session (that needed the
  nStatus check, not a better starting point). This is worth stating precisely in the final
  writeup: warm-starting the dual solve is NOT currently done at this level, and a quick test
  suggests it wouldn't have masked the bug either way (the underlying problem was genuinely
  unbounded from at least two different starting points tried).

- **NOT YET INVESTIGATED**: whether the OUTER KNITRO loop itself (`outer_loop_cached` in
  `cc_algo/outer_loop_cached.jl`, used by `outer_solve_nested_cached` in
  `run_profiled_production.jl` for the LC method specifically) warm-starts ITS OWN internal
  inner-solve across the OUTER loop's own sequence of trial points -- there's an `OuterEvalCache`
  mentioned in that file's own header comment ("solve the inner CC problem at most once per
  unique free-outer point") which suggests SOME caching/reuse discipline exists at that level,
  but whether it uses `use_cached_x`/warm-started duals, or just avoids re-solving IDENTICAL
  points (a cache, not a warm start), was not determined this session. **Read
  `cc_algo/outer_loop_cached.jl` in full plus its use of `obj.x`/`use_cached_x` (if any) to
  answer this precisely** -- the file was read for ~100 lines this session (its header/setup)
  but not to the point of tracing the actual inner-solve invocation inside KNITRO's own callback
  loop.
- Also not investigated: does GC/GU's own population search (`bbo_common.jl`'s `make_fitness`/
  `eval_candidate`, calling `seq_gravcol` fresh per candidate) get ANY benefit from warm-starting
  across POPULATION MEMBERS (e.g. does BlackBoxOptim itself, or this codebase's wrapper around
  it, pass a `warm=` state between consecutive fitness evaluations of nearby candidates)? A
  quick grep of `sequential_gravity/global_opt/bbo_common.jl`'s `make_fitness`/`eval_candidate`
  shows `seq_gravcol(θ; δ=Inf, maxit=maxit, tol=tol)` called with NO `warm=`/`warm_p=` argument
  at all -- so GC/GU's own per-candidate `seq_gravcol` calls are COLD every time (no warm start
  from a previous population member), unlike the LC/LU local search's own within-run warm
  starting. Worth confirming this reading is complete and stating it plainly in the final
  writeup, since it's directly relevant to explaining GC/GU's relative slowness per-eval.

## Separate, unrelated deliverable already complete

`sequential_gravity/derivative_diagnostics/HARDMAX_INVERSION_HANDOFF_PROMPT.md` -- a fully
self-contained handoff prompt for a DIFFERENT investigation (validating whether the
smoothed-inversion solutions this whole codebase produces also satisfy the TRUE hard-max
economic model, and building a working hard-max destination-inversion solver). This is
unrelated to the recover_lfd bug and does not need to be redone -- it's ready to hand to a
separate session whenever the user wants to pursue it. It does reference 3 example converged
points by file path; if those files get regenerated/overwritten by future re-runs, the prompt
already says to re-pull fresh values rather than trust the hardcoded numbers in the prompt.

## Persistent memory -- NOT YET UPDATED

This session's key findings (the recover_lfd bug + fix, the head-to-head comparison
infrastructure, the 4 driver-script bugs found during initial smoke-testing) have **not been
saved to the project's persistent memory system** (`/bbkinghome/edav/.claude/projects/
-bbkinghome-edav-gravity-robustness/memory/`) yet. The new session should do this once the
re-audit/table/warm-start writeup is complete -- worth one consolidated memory entry (or two:
one for "head-to-head comparison infrastructure + 4 driver bugs," one for "recover_lfd
nStatus bug, root cause + fix + which points were corrupted") rather than saving mid-stream
before the full picture is confirmed.

## Quick-start commands for the new session

```bash
# 1. Confirm the fix is complete (task 2)
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf
grep -rln "function recover_lfd" --include="*.jl" .
grep -rl "nStatus ∈ (0, -100, -101, -103)" --include="*.jl" .   # should match the same 11 (10 use this exact string; run_sequential.jl's 2 call sites use it too)

# 2. Finish the re-audit (needed before task 1)
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt KNITRODIR=/opt/shared_sw/knitro/14.2.0 \
  LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH PATH="$HOME/.juliaup/bin:$PATH"
FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
  julia -t 19 --project=. sequential_gravity/head_to_head/comprehensive_reaudit.jl
# (writes sequential_gravity/head_to_head/comprehensive_reaudit_results.csv when done)

# 3. Investigate warm-starting (task 3) -- start here:
grep -n "use_cached_x\|obj.x" cc_algo/outer_loop_cached.jl
grep -n "warm" sequential_gravity/global_opt/bbo_common.jl
```
