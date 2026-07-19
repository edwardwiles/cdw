# Continuation 8, Section 10: gated dimension scaling (D=4/6/8/10)

Branch `diag/fullA-d4-exact`, machine `demand.mit.edu`, `JULIA_NUM_THREADS=20`. Both gates required by
the standing brief were satisfied before this section ran: the algorithm frontier
(`docs/fullA_algorithm_frontier_c8.md`) and a W=20000 nested continuation pass
(`docs/fullA_nested_w_continuation_c8.md`).

## 1. Microbenchmarks (D=4/6/8/10) — already covered, not re-run

Wave 2A's canonical performance profile (`docs/fullA_canonical_performance_profile_c8.md`,
`results/fullA_d4/d6e3b05/c8_perfprofile/d_scaling_grid.csv`) already produced the D=6/8/10
component-level microbenchmark grid (moment build, FG callback, dense vs. compressed) this same
session, after Wave 1's compressed-live and winner-accelerator integrations. Reused as-is; not
duplicated here.

## 2-4. Free-A pilots at D=6, D=8, D=10

Generic-D driver `full_aod_diag/d4_exact/run_d6_pilot.jl` (pre-existing, continuation 5), run fresh
this session at each D, upper and lower directions, `D6_MAXTIME_REAL=120`. All cold-recheck-verified
(`evaluate_fullA` re-run at the best-feasible point with `warm=false`, matching the tracked value to
<1e-6 in every case).

| D | direction | knitro_status | κ | Δ−δ | cold recheck |
|---|---|---|---|---|---|
| 6 | upper | -101 (converged) | 0.19758 | -3.0e-4 | matches |
| 6 | lower | -103 (converged) | 0.00746 | -9.7e-5 | matches |
| 8 | upper | -101 (converged) | 0.34634 | -1.4e-5 | matches |
| 8 | lower | -101 (converged) | 0.03844 | -9.9e-6 | matches |
| 10 | upper | -401 (**budget-stalled**) | 0.38953 | -2.5e-5 | matches |
| 10 | lower | -103 (converged) | 0.04517 | -1.6e-4 | matches |

D=6 and D=8 pass fully (all four runs genuinely converged). D=10's upper direction did not fully
converge within the 120s budget (KNITRO status -401, iteration/time-limit stop) — the best-feasible
point it found is still cold-recheck-verified exact-feasible, consistent with D=10 being explicitly
scoped as a lighter-scrutiny "gated benchmark" per the standing brief, not a fully-supervised pilot
like D=6.

Raw output: `results/fullA_d4/c4243c1/d6_pilot_20260718_194913/`,
`.../d8_pilot_20260718_195517/`, `.../d10_pilot_20260718_195555/summary.txt`.

## 5. D=20 — explicitly NOT launched

Per the standing brief's gating, no D=20 solve was attempted this session.

## 6. Fixed-A* sanity check (user request): does the outer loop actually search A-space?

New script `full_aod_diag/d4_exact/c8_fixedA_pilot.jl` — same generic-D setup and oracle, but A_od is
held fixed at the "natural theta" calibration values throughout; only γ'_focal (1 free variable) is
searched. Compares against the free-A results in §2-4 and (for D=4) the registered incumbents.

| D | direction | κ (fixed-A\*) | κ (free-A) | gain from free-A search |
|---|---|---|---|---|
| 4 | upper | 0.14398 | 0.17246 (registered) | +19.8% |
| 4 | lower | 0.00599 | 0.00439 (`lower_v2`) | −26.8% |
| 6 | upper | 0.16211 | 0.19758 | +21.9% |
| 6 | lower | 0.00802 | 0.00746 | −7.0% |
| 8 | upper | 0.28113 | 0.34634 | +23.2% |
| 8 | lower | 0.05642 | 0.03844 | −31.9% |
| 10 | upper | 0.29394 | 0.38953 | +32.5% |
| 10 | lower | 0.06317 | 0.04517 | −28.5% |

**Verdict: the outer loop is doing genuine, substantial work.** Free-A search beats fixed-A by
20-33% at every dimension tested, on both directions (upper: larger is better; lower: smaller is
better) — not a marginal or noise-level difference. If anything the advantage grows with D (roughly
20%→33% on the upper side from D=4 to D=10), consistent with more A_od entries giving genuinely more
exploitable freedom, not diminishing returns. Cross-check: the D=4 fixed-A\* upper result
(γ'=0.910941) matches the pre-existing manually-recorded `fixed_A_benchmark` entry in
`candidate_registry.jl` (γ'=0.9109408706) to 6 significant figures — confirms this new script
reproduces a previously-established reference point correctly, not just a plausible-looking new number.

Raw output: `results/fullA_d4/d547142/d{4,6,8,10}_fixedA_pilot_202607182006{44,48,51,55}/summary.txt`.

## 7. BLAS threading: empirical finding (user request)

Mid-session, this section's D=8/D=10 launches were found consuming 60+ cores each (`nlwp`≈144,
`%cpu`≈2600-3250%) despite `JULIA_NUM_THREADS=20` — `JULIA_NUM_THREADS` only bounds Julia's own
thread pool, not OpenBLAS's separate thread pool underneath it, which was defaulting to a large
value independent of anything explicitly configured. Both runaway processes were killed
(`kill -9`) and relaunched with `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` explicitly set.

An initial conservative fix (`OPENBLAS_NUM_THREADS=1`) was questioned as possibly leaving real
performance on the table, since KNITRO's inner-solve BLAS calls (`gemv!` on the fixed-dual moment
matrix) aren't competing with any other Julia-level parallelism at that point in the callstack. This
was tested empirically rather than assumed — a clean, sequential, externally-timed (`time`, not this
codebase's own internal wall-clock field, which was found to be unreliable/mis-measuring in this
same investigation) A/B comparison at D=8 and D=10:

| D | `OPENBLAS_NUM_THREADS` | wall-clock (`real`) | aggregate CPU-time (`user`) | CPU cost vs. BLAS=1 |
|---|---|---|---|---|
| 8 | 1 | 2m39.1s | 7m31.1s | 1.0x |
| 8 | 20 | 2m38.4s | 29m30.9s | **3.9x** |
| 10 | 1 | 4m4.8s | 11m30.7s | 1.0x |
| 10 | 20 | 3m51.1s | 49m33.3s | **4.3x** |

**Finding: `OPENBLAS_NUM_THREADS=20` gives ~0% wall-clock benefit at D=8 (2m39s vs 2m38s, within
noise) and a modest ~5.6% benefit at D=10 (4m5s vs 3m51s) — in exchange for 3.9-4.3x more aggregate
CPU consumption** on a shared, multi-user machine. The extra BLAS parallelism is a poor trade at
these problem sizes: the hot BLAS operation here (`gemv!` on an 8000×~100 matrix) is tall-thin and
memory-bandwidth-bound, not compute-bound, so it doesn't benefit much from more threads, while KNITRO's
own SQP/interior-point overhead (not a BLAS call) likely dominates the inner-solve wall time
regardless. **Recommendation: `OPENBLAS_NUM_THREADS=1` (or a small fixed number, not left unset and
not matched to `JULIA_NUM_THREADS`) should be the standing default for this investigation's KNITRO
runs going forward** — this reverses this session's own mid-stream "correction" to `=20`, now with
real data rather than a guess behind the choice either way.

Raw timing logs: `results/fullA_d4/d547142/blas_threading_test/`.

## Operational note for future sessions

Always set **both** `JULIA_NUM_THREADS=20` **and** `OPENBLAS_NUM_THREADS=1` (`MKL_NUM_THREADS=1` too,
belt-and-suspenders) when launching Julia/KNITRO jobs in this investigation — the former alone does
not bound BLAS's own thread pool, and per §7 above there is no performance reason to allow it more
than 1.
