# Head-to-head comparison: final report (2026-07-16)

D=20 real data (`FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true`), 4 methods
(LC=local-constrained/KNITRO, LU=local-unconstrained/KNITRO, GC=global-constrained/BlackBoxOptim,
GU=global-unconstrained/BlackBoxOptim), 3 divergence targets (T1/T2/T3, nominal budgets
0.1/1.0/2.0), 3 starts each. Supersedes any table drafted before this session — several
originally-reported "winners" were spurious (see below).

## 1. The bug that was found and fixed

`recover_lfd` (the function that turns an inner KNITRO dual solve into an LFD reweighting `p`)
only checked `all(isfinite, x))` on the returned dual vector, never the KNITRO solve-status code
(`nStatus`). `inner_loop_internal` NaNs its own *cached* `obj.x` field on a rejected status (e.g.
`-300` = `KN_RC_UNBOUNDED`) but does **not** NaN the `x` it actually returns to the caller — so a
genuinely failed/unbounded dual solve could still pass the `isfinite` check and get silently
accepted as if converged. Everything computed *from* that bogus `p` (gravity residual,
divergence) looked internally consistent (small residual, in-budget divergence) because it was
derived from the same bad `p` — but the real trade shares were off by percent-level amounts
(confirmed directly: 6.2% share error and an 8.45 first-order-condition violation at one
corrupted point, vs. ~1e-6-1e-7 and ~1e-15 at genuine points).

**Fixed in all 11 files that define `recover_lfd`** across the repo (confirmed via
`grep -rln "function recover_lfd" --include="*.jl" .`, all 11 re-confirmed patched this
session): the primary production driver (`sequential_gravity/run_profiled_production.jl`) plus
10 other diagnostic/verification scripts. Pattern: reject unless
`nStatus ∈ (0, -100, -101, -103)` (the same KNITRO-acceptable-status convention already used by
`outer_solve_nested_cached`'s own probe-solve check), immediately after the `inner_loop` call.

**Fix reconfirmed working this session** via a fresh independent re-run of
`smoke_test_recover_lfd_fix.jl`: GC's known-bad T2/warm point still correctly returns `ok=false`
post-fix.

## 2. Full re-audit: no corruption beyond the 3 originally found

Every one of the 32 points from the official 4-method comparison (plus the 4 most-cited
`lu_multistart50` points) was independently re-verified this session using the fixed
`seq_gravcol` plus a **direct** trade-share check (`gravity_residual` / `dest_share` computed
straight off the recovered `(A_od, p, umat)`, no linearized-moment reconstruction needed).

**Result: exactly 3 points are corrupted, no others.** All three were each their cell's
originally-reported "winner":

| Point | Was reported as | Status |
|---|---|---|
| `out_lc/lc_T2_rand1.jld2` | LC's T2 winner | **SPURIOUS** — confirmed via re-run: fixed `recover_lfd` now rejects it |
| `out_lc/lc_T3_warm.jld2` | LC's T3 winner | **SPURIOUS** |
| `out_gc/gc_T2_warm.jld2` | GC's T2 winner (kappa=0.1055, the one that triggered this investigation) | **SPURIOUS** |

Every other checked point (29 of 32 official points, all 4 spot-checked `lu_multistart50`
points) came back **GENUINE** (worst per-destination trade-share error 5e-7 to 1e-6, several
orders of magnitude inside the 1e-4 pass threshold), with one exception that is *not* new
corruption: LU's official T3 run (all 3 starts) is correctly re-rejected by the fixed checker,
but all 3 of those points already reported `best_feasible_delta_star=Inf` / no feasible solution
in their *original* run — i.e. LU never claimed a T3 answer in the first place, so there is
nothing to invalidate there.

## 3. Final comparison table (genuine points only)

Winner criterion per method's own definition: LC/GC maximize kappa among feasible points; LU/GU
(unconstrained — gp is fixed, only delta* is minimized) take the lowest achieved delta*.
"Wall" = the *winning* point's own solve time. "Cell total" includes the discarded/spurious
starts' wall time too, for an honest accounting of total compute spent per cell.

| Target | Method | Winner start | Result | Winner wall | Cell total wall (n genuine/n run) |
|---|---|---|---|---|---|
| **T1** (budget 0.1) | LC | Astar | kappa=0.04481 | 73.6 min | 164.5 min (2/2) |
| | GC | rand2 | kappa=0.03154 | 90.3 min | 271.8 min (3/3) |
| | LU | Astar | delta*=0.11474 | 0.5 min | 1.2 min (3/3) |
| | GU | Astar | delta*=0.11474 | 90.6 min | 271.5 min (3/3) |
| **T2** (budget 1.0) | LC | warm | kappa=0.08208 | 90.8 min | 166.5 min (2/3) |
| | GC | Astar | kappa=0.06499 | 90.5 min | 271.1 min (2/3) |
| | LU | Astar | delta*=1.08286 | 7.2 min | 21.4 min (3/3) |
| | GU | warm | delta*=1.06556 | 90.4 min | 271.4 min (3/3) |
| **T3** (budget 2.0) | LC | rand1 | kappa=0.09157 | 44.6 min | 176.7 min (2/3) |
| | GC | Astar | kappa=0.08729 | 90.8 min | 181.3 min (2/2) |
| | LU | — | **NO FEASIBLE RESULT** (3/3 starts found none — pre-existing, not new) | — | 1.3 min (3/3) |
| | GU | Astar | delta*=2.72381 | 90.6 min | 90.6 min (1/1, T3/warm never run) |

### Headline findings

- **LC (local-constrained) wins every target for kappa**, now that its corrupted T2/T3 "winners"
  are corrected to their genuine best-feasible values. Previously GC's spurious T2/warm point
  (kappa=0.1055) looked like it beat LC — post-fix, LC's genuine T2 kappa (0.0821) actually beats
  GC's genuine T2 best (0.0650).
- **LU is dramatically cheaper than every other method for the same answer where comparable**: at
  T1, LU's 3 cheap local KNITRO solves (71s total) land on the *exact same* delta*=0.11474 as
  GU's 271-minute global BlackBoxOptim search. LU is ~230x cheaper for an identical result at T1.
- **GC/GU (BlackBoxOptim) cost ~90 min per solve regardless of difficulty** (population-based,
  capped at the 5400s wall cap in several cases) vs. LC/LU's cost scaling with target difficulty
  (LC: 45-91 min; LU: 0.5-7 min). The global methods' main value-add is at T2, where GU finds a
  slightly lower delta* (1.066 vs LU's 1.083) — a real but modest win for ~90 min vs ~7 min.
- **T3 is hard for the official 3-start budget of every method**: LC/GC both lose a start to
  corruption AND need their remaining 2 starts to find anything; LU's official 3 starts find
  nothing at all. Only GU (1 completed start, expensive) and the separate 50-start LU sweep
  (below) find real T3 answers.

## 4. LU-multistart50: T3 callout (own section, per user's framing — different experimental design)

150 independent solves (50 shared graduated-noise starting points x 3 targets, no warm-starting
between targets), reusing LU's own cheap KNITRO machinery. Re-verified this session for the 4
points anyone would actually cite (all **GENUINE**, matching originally-reported numbers exactly):

| Target | Feasible / 50 | Best delta* | Point | Note |
|---|---|---|---|---|
| T1 | 48/50 | 0.10653 | pt07 (sigma=0.10) | matches official LU/GU's 0.11474 closely |
| T2 | 38/50 | 1.03696 | pt20 (sigma=0.30) | slightly better than official LU (1.08286), close to GU (1.06556) |
| T3 | **2/50** | **2.31818** | pt40 (sigma=1.00) | **beats GU's official (expensive, 90min) T3 result of 2.72381** |

The T3 result is the most interesting finding of the whole comparison: broad multistart with the
*cheap* local method (50 x ~20-30s ≈ tens of minutes total, dominated by pt45's one expensive
2844s outlier solve) finds a **better** T3 answer than the single *expensive* global search (GU,
90 min) did. This suggests T3's difficulty is more about needing the right starting basin than
about needing a genuinely global (non-local) search method — a materially different diagnosis
than "T3 requires BlackBoxOptim," worth flagging explicitly if this comparison informs future
method choice.

## 5. Warm-starting: what's actually wired up

**(a) The inner CC delta\* dual solve (`inner_loop`/`inner_loop_internal`, called via
`recover_lfd`) is NEVER warm-started anywhere in this comparison's production pipeline**, despite
the underlying machinery supporting it (`PsiObjectiveBundleDelta` has a `use_cached_x::Bool` field
and `inner_loop_initial_values` reads cached `obj.x` when `use_cached_x=true`):

- `recover_lfd` (`sequential_gravity/run_profiled_production.jl:147`, the function this whole
  bug lived in) constructs a **fresh** `PsiObjectiveBundleDelta` on every single call, with
  `use_cached_x` left at its struct default of `false`. Every LC/LU/GC/GU solve ultimately routes
  through `seq_gravcol` → `recover_lfd`, so this is true for all 4 methods.
- The LC method's own outer KNITRO loop (`outer_solve_nested_cached`,
  `run_profiled_production.jl:508`) explicitly passes `use_cached_x = false` to the
  `PsiObjectiveBundleImplicitMethodB` it constructs — so even the OUTER loop's own sequence of
  trial points does not warm-start the inner dual across outer iterations. (Sibling files
  `run_profiled_bounds.jl`/`run_profiled_bounds_norm.jl` document why: "gravity column changes
  per θ ⇒ stale warm-start invalid" — for the sequential/profiled formulation, gravity is
  enforced *inside* the inner solve, so a prior θ's converged dual is not a valid starting point
  for a different θ's inner problem.)
- A direct empirical test this session (`test_warmstart_audit.jl`) fed a converged dual from a
  nearby easy point as the starting `x` for a known-failing point — it still failed with the same
  `nStatus=-300` (UNBOUNDED). So even if this warm-start were wired up, it would not have masked
  the bug found in this session (the underlying failure was genuinely unbounded from multiple
  starting points, not merely slow to converge).
- **Contrast**: `use_cached_x=true` for the dual solve *is* used elsewhere in this repo — but only
  in the unrelated **full-A_od** formulation (`full_aod_diag/`, `run_fullA_D10_production.jl`,
  `delta_star_schedule.jl`, `run_focal_bounds.jl`), a different method not used by this
  head-to-head comparison, where gravity is a *separate* outer constraint (not embedded in the
  inner solve) so a stale dual warm-start is valid there.

**(b) The destination-share inversion (`invert_destination`, the smoothed ρ=2e-3 Newton solver in
`profiled_gravity.jl`) DOES warm-start, at two levels**:

- Within one `seq_gravcol` call's iterative refinement loop: each trial α=1 solve warm-starts
  from the pre-iteration `umat`; subsequent smaller-α line-search trials warm-start by
  interpolating between the two known endpoint solutions rather than restarting cold (validated
  ~1.7x fewer total Newton iterations, bit-exact same accepted trajectory either way —
  `full_aod_diag/sequential_inversion_performance/warm_start_report.md`).
- Across separate `seq_gravcol` calls: the stateful moments closure threads `warm=`/`warm_p=`
  (the previous accepted iterate's `umat`/`p`) into the next call via `invert_all`'s `u_init_fn`.

**(c) GC/GU's population search gets no warm-start benefit at all**: `bbo_common.jl`'s
`make_fitness`/`eval_candidate` calls `seq_gravcol(θ; δ=Inf, maxit=maxit, tol=tol)` with no
`warm=`/`warm_p=` argument — every population candidate's inner solve (both the CC dual AND the
destination inversion) starts cold, unlike LC/LU's own within-run warm-starting. This at least
partly explains GC/GU's much higher per-eval cost relative to LC/LU.

## Files produced this session

- `sequential_gravity/head_to_head/comprehensive_reaudit_results.csv` — full 32-point re-audit
  results (machine-readable)
- `sequential_gravity/head_to_head/spotcheck_lu_multistart50.jl` — spot-check script (4 points)
- This report
