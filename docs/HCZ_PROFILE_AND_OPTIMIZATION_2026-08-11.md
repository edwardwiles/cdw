# Where the cross-family wall-clock actually goes, and a 2.8× H_CZ kernel

2026-08-11. Branch `feature/ozc-cross-2026-08-09`, commit `eb2f65f`. Not pushed to any remote.
Companion to `CROSS_8H_UPPER_BOUND_RUNS_2026-08-10.md` (the δ=1 runs this profiles).

## 0. Summary

| | before | after | |
|---|---|---|---|
| `H_CZ_prep` (CM+ZC, nz=1770, 32 workers) | 1.185 s | **0.422 s** | **2.81×** |
| `H_CZ_prep` (CM+ZC diagonal, nz=630, 32 workers) | 0.488 s | **0.175 s** | **2.78×** |
| `canonical_price_precompute` | 0.1091 s/call | **~0** | mu-invariant rebuild removed |

`HCZ_PREP_BACKEND_DEFAULT[]` is now `:j_parallel`, covering **both** CM+ZC families (one shared code
path). Two end-to-end `Delta_dual` gates passed at the solver's own tolerance floor.

**The headline finding is that the standing optimisation guidance was stale.** After the 2026-08-10
BLAS-thread fix, `H_ZZ` is no longer the bottleneck for CM+ZC — `H_CZ_prep` is.

## 1. Profiling method, and why the previous attribution was misleading

The inherited attribution (`00_READ_FIRST_CORRECTION.md`) was measured (a) at **1 BLAS thread** and
(b) at the **calibration point**. Both bias it, in the same direction:

* the 2026-08-10 gate fix gives `H_ZZ` 8 threads, which the kernel benchmark shows is worth 4.28× on
  that block *alone*, so the balance between `H_ZZ` and everything else necessarily shifted;
* the 8-hour runs spent almost all their wall-clock near the Δ≈1 boundary, not at calibration.

So `profile_cross_outer_eval_2026-08-11.jl` profiles at **each run's own final incumbent, read out of
its checkpoint**. It runs through the real public driver (memory
`feedback-archC-verified-state-direct-call-knitro-callback-err` records a confirmed
`KN_RC_CALLBACK_ERR` trap for direct low-level calls at D=20).

Two instrumentation gaps had to be closed first, both of which had made earlier numbers *residuals
with a story attached* rather than measurements:

* **Neither inner FG callback had any timing label.** Added `originZC_FG_callback` /
  `meanZC_FG_callback` via `@cmhess_prof` (default OFF, one `Ref` check — these fire once per inner
  iteration, so production must pay nothing).
* **The inner-solve counters were never read.** `CS.INNER_SOLVE_COUNT` / `INNER_ITERS_TOTAL` /
  `INNER_INFEAS_COUNT` now reported, which is what corrected §3 below.

## 2. Where the time goes

### OZC-CROSS, at its own incumbent (90% of the outer solve accounted)

| | s | % of outer solve |
|---|---|---|
| Hessian callbacks (343 calls, mean 2.72 s) | 932.7 | **77.7%** |
| ├ `H_ZZ` restriction gram | 632.8 | 52.7% |
| ├ `H_EZ_fill` | 252.6 | 21.0% |
| └ `H_EE_core` — **the true economic Hessian** | 10.7 | **0.9%** |
| FG callbacks (1227 calls, mean 0.118 s) | 145.0 | 12.1% |
| residual (KNITRO's own LA, 3 outer gradients, screens, checkpointing) | 122.3 | 10.2% |

### CM+ZC-CROSS, at its own incumbent — one Hessian callback = 6.251 s

| block | s | % of callback |
|---|---|---|
| **`H_CZ_prep`** (CM-grid × restriction) | **2.122** | **33.9%** |
| `H_ZZ` (restriction gram) | 1.745 | 27.9% |
| `H_ER` | 0.708 | 11.3% |
| `bintables_prep` | 0.620 | 9.9% |
| `H_ER_prep` | 0.463 | 7.4% |
| `H_EC_asm` + `H_EC_prep` + `H_CC` + packing + misc | 0.559 | 8.9% |
| **`H_EE_core`** — the true economic Hessian | **0.024** | **0.4%** |

99.9% accounted. Against the stale 1-thread attribution: `H_ZZ` 7.524 → 1.745 s (**4.31×**, versus
the kernel benchmark's 4.28× prediction — an independent confirmation the thread fix does what it
claims), while `H_CZ_prep` 2.650 → 2.122 s (1.25×, not BLAS-bound). **`H_ZZ` was 2.84× larger than
`H_CZ_prep`; it is now 0.82×. The bottleneck changed hands.**

The two families now have *different* bottlenecks — OZC-CROSS is still `H_ZZ`-dominated (68%),
CM+ZC-CROSS is `H_CZ_prep`-dominated — and need different targets. Common to both: the genuine
economic Hessian is 0.4–0.9%. **Essentially all Hessian cost is restriction-block bookkeeping.**

### Per-call cost is point-independent; only the iteration count varies

| | at calibration | at the Δ≈1 incumbent |
|---|---|---|
| Hessian mean per call | 2.79 s | 2.59 s |
| `H_ZZ` share of the Hessian | 67.9% | 68.8% |
| `H_EZ_fill` share | 26.8% | 26.8% |

So the growth in per-eval cost within a run (77 s → 337 s → 537 s) is **entirely** the number of
inner iterations, not any kernel getting slower.

## 3. A correction: there is no "5 inner solves per θ"

An earlier reading of these counters claimed 5.25 inner solves per outer evaluation and called it an
unexplained 5× multiplier. **That was wrong**, and the correct explanation was supplied by the user:
they are failed KNITRO trial points.

`reject_point` is `throw(DomainError(...))` (`oracle.jl:32`), thrown from `cb_F!` *before*
`n_eval[] += 1`. So a failed inner solve consumes a solve but never increments the evaluation
counter: **21 solves = 4 that succeeded + 17 that failed**, confirmed by `INNER_INFEAS_COUNT = 17`
and by the screen counter matching the solve count exactly. `archOZ_verified_state` issues exactly
one solve per call, and verification is operator-based post-processing with no second solve — so it
is **1 inner solve per θ evaluated**, as it should be.

The 5.25 figure was also unrepresentative: profiling *from* the incumbent, where nearly every
neighbouring point is infeasible. In the real 8-hour run the ratio is **2.46** (246 attempts, 146
rejected, 100 evaluated).

What survives is narrower but real: **the pairwise certificate screen had a 0% hit rate** in the 8h
run (`pairwise_hits=0, inner_solves_avoided=0` over 246 calls, costing 134 s for no benefit), so all
146 rejections were paid at full inner-solve price. That screen demonstrably *can* fire — it caught
11 of 13 in the nesting experiment — it just does not discriminate near this family's own δ=1
boundary. Whether that is worth attacking depends on how expensive a *failed* solve is versus a
successful one, which the present counters cannot separate.

## 4. The H_CZ optimisation

`bin_zc_cross_hessian_fill_drawchunk_reordered!` had already fixed the **read** side (contiguous
`ZcS[:,j]` columns). Two problems remained, and its own docstring flagged the first:

1. **Write-side scatter.** The accumulator is `[x, j, b]`, so consecutive bins are `D*nz = 35,400`
   elements apart — **277 KB per step, 51 live targets spanning 13.8 MB** — L3-class, so nearly every
   one of the 3.54e9 accumulates can miss L2.
2. **`ZcS` re-streamed.** With `x` outer, each 781 KB column is read once per `(x,j)` *pair*:
   **26.4 GB per call**.

Two candidates, both purely additive (`hcz_btranspose_candidate_2026-08-11.jl`):

* **`:draw_chunk_btranspose`** — accumulator permuted to `(L+1, D, nz, workers)`, so the 51 targets
  become 408 contiguous bytes, L1-resident for the whole inner loop. **Bit-identical** to
  `:draw_chunk_reordered` (same accumulation order per cell); 1.63× isolated, **1.55× in situ** in a
  real Hessian callback — so the cache win survives alongside `H_ZZ`'s 1.4 GB working set.
* **`:j_parallel`** — partitions over restriction **columns**. Different `j` write to disjoint output
  cells, so there is **no per-worker accumulator** (882 MB at 32 workers → ~0) and **no cross-worker
  reduction**; its `j`-outer/`x`-inner order reads each `ZcS` column once (26.4 GB → 1.3 GB).

### Thread scaling — this is where the win is

| workers | `reordered` | `btranspose` | **`j_parallel`** |
|---|---|---|---|
| 1 | 10.150 s | 8.908 s | 9.343 s |
| 8 | 2.555 | 1.482 | 1.147 |
| 16 | 1.543 | 1.091 | **0.668** |
| 32 | 1.185 | 1.117 *(plateaus)* | **0.422** |
| **1→32** | 8.6× (27% eff.) | 8.0× (25%) | **22.1× (69%)** |

At one thread all three are within 14% — this is almost entirely a *parallel-scaling* win.
`:draw_chunk_btranspose` plateaus after 16 workers as its scratch and serial reduction grow;
`:j_parallel` keeps scaling because it has neither.

### It is one code path, so the diagonal family gets it for free

`build_cm_meanzc_bin_ctx` and its Hessian blocks are reused verbatim by the cross family, so there is
a single `hcz_prep_dispatch!`. Measured at the diagonal shape (`nz=630`) rather than assumed:
0.488 → 0.175 s at 32 workers (**2.78×**), scaling 21.0× vs 8.7×.

### Correctness

* `:draw_chunk_btranspose` — **bit-identical**, `max|diff| = 0.0` on all four outputs, with a guard
  against the trap of comparing two all-zero arrays.
* `:j_parallel` — tolerance-level by design (each cell summed over all W in one pass rather than as
  per-worker partials): **3.7e-15** (diagonal) / **5.0e-15** (cross) relative, against
  `HCZ_CANDIDATE_TOL = 1e-9`.
* **End-to-end `Delta_dual` through the real driver**, calibration point: diagonal A/B **1.47e-11**
  relative; cross vs the 8h run's reference **1.04e-10**. Both at or below the **2.734e-10** of the
  already-accepted 2026-08-10 BLAS-thread flip — the solver's own tolerance floor.

Consequence to be explicit about: runs after this flip are **not bit-reproducible** against earlier
results. The two kernels compute the same mathematical quantity and differ only by floating-point
reassociation; `:draw_chunk_btranspose` remains selectable and *is* bit-identical if reproducibility
against pre-flip results is ever needed.

The kernel also carries a coverage guard: it *assigns* rather than accumulates (that is what lets it
skip the zero-fill and reduction), so incomplete `j` coverage would silently leave stale values
rather than error.

## 5. `canonical_price_precompute`: a mu-invariant rebuild

`winner_certificate.jl` rebuilt `mulU` / `UPow` / `UσPow` on **every** call — 2 × W × D = **4e6
non-integer power evaluations** at real D=20/W=100,000 — for arrays that depend only on `mu`, which
is not a free outer coordinate (`context_real_d20.jl`: `free_idx = vcat(3 + Dact, Aod_offset+1:…)`
covers `gp` and the `A_od` block). Now refilled only when `mu` actually moves, via a `mu_filled`
field. **0.1091 s → ~0 per call**, and 38 MB of allocation avoided.

Keyed on `mu` *changing* rather than on an assumption that it is fixed, so layouts that do vary `mu`
stay correct — and it rests on exactly the assumption the workspace's existing `logU` cache already
relies on (if `U` could change under a live workspace, `logU` would already be wrong).

**Honest scope:** `prime_operator!` fires once per *inner solve*, not per FG call, so this path is
reached ~once per inner solve plus once per screen — about 42 calls in a 1200 s profile, **≈0.4% of
outer-solve wall**. Free, correct, and it benefits every family, but it is not a headline win here.
It would matter far more wherever that call sits on an inner loop.

## 6. Checked and found already correct: the dense-vs-factorized outer gradient

A dense `build_lfix_base_cache` path (two 304 MB tensors rebuilt per gradient call) was reported as a
trap that new restricted families fall into. **The cross families do not have it**, verified five
ways rather than by file existence: both `_cplus.jl` adapters exist; they call
`build_lfix_base_cache_C!` (factorized), not dense `build_lfix_base_cache`; they call
`composite_gradient_at_Cplus_from_cache`, not `economic_A_gradient!`; the driver defaults to
`cm_gradient_backend = :cplus`; and the 8-hour runs logged `cm_gradient_backend=cplus`. The
2026-08-09 work built both adapters on the `cm_originzc_cplus.jl` template.

## 7. Remaining levers, ranked

1. **`H_EZ_fill` for OZC-CROSS** — 21% of its outer solve, same segmented-reduction character as
   `H_CZ_prep` was, and invisible in the old attribution. The most likely next 2×.
2. **Give H_CZ more Julia threads.** The campaign runs `-t 10`; `:j_parallel` is still scaling at 32,
   and the memory objection that made high worker counts expensive is gone.
3. **The 0% screen hit rate** (§3) — needs the cost of a failed vs successful inner solve separated
   first.
4. **`H_ZZ`** — still 53% of OZC's outer solve, but already 4.31× improved and near its knee; 16 BLAS
   threads would add ~18% of that block, and that 8-vs-16 question remains unsettled.

Not worth pursuing: the economic Hessian (0.4–0.9%) and the FG callbacks (4–12%).
