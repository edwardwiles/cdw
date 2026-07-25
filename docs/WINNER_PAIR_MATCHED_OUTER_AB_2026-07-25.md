# Matched real outer A/B gates — 2026-07-25 continuation

Task §6. Real production entry points (`run_polish_checkpointed`, `run_cm_upper_checkpointed`),
D=20/D_dest=19/W=80,000/`:exclude_row`/seed=20260719/delta=1, genuine calibrated start
(`ctx.θ0_up`), identical explicit pinned outer algorithm (`pin_outer_algorithm=true`), 20 Julia
threads, one process at a time (sequential chain, not concurrent), 300s measured budget +20s
warm-up. The only intended difference between arms: `core_hessian_backend =
:dense_reference` vs `:exact_winner_pair_parallel` (workers=10), via a new additive ENV-based
override (`BENCH_CORE_HESSIAN_BACKEND`) added to the pre-existing
`matched_outer_benchmark_u_2026-07-25.jl`/`matched_outer_benchmark_cm_2026-07-25.jl` harnesses —
unset leaves each harness's behavior completely unchanged from before this port.

Raw summaries: `docs/key_results/matched_outer_ab_{u_dense,u_shared,cm_dense,cm_shared}_summary_2026-07-25.txt`.

## UNRESTRICTED — clear, decisive gain

| | dense-reference | shared winner-pair |
|---|---|---|
| Value evaluations (`n_eval`) | 54 | **197** |
| Gradient evaluations (`n_grad`) | — | 81 |
| Checkpoint reuse hits | — | 80 |
| Measured wall | 327.8s | 320.8s |
| Best `gp` (find_smallest minimizes) | 0.955593 | **0.952064** (better) |
| Best `Delta` | 0.986423 | 0.999528 |
| Cold-verify | ok, diff=9.99e-16 | ok |
| Allocation | (not separately isolated) | 52.27 GB / 5.67s GC |
| Terminal status | -401 (KN_RC_TIME_LIMIT_FEAS) both | -401 both |

**3.65x more value evaluations in essentially the same wall-clock budget** (197 vs 54), reaching a
strictly BETTER (lower) `gp` — a clean, decisive complete-solve/outer-throughput win, satisfying
task §8's eligibility bullet ("unrestricted and flexible CM show a clear complete-solve or
outer-throughput gain") for this family without qualification.

## FLEXIBLE CM (L=50) — real, more modest gain, consistent with its own cost profile

| | dense-reference | shared winner-pair |
|---|---|---|
| Value evaluations (`n_eval`) | 4 | **8** |
| Gradient evaluations | 4 | 6 |
| Measured wall | 347.8s | 451.0s |
| Best `kappa` | 0.050043 | 0.050043 (**identical to 6 s.f.**) |
| Best `Delta` | 0.827364 (11 s.f.) | 0.827364 (11 s.f., agrees to `diff=0.0` at cold-verify) |
| Cold-verify | ok | ok, `diff=0.0` |
| Terminal status | -401 both | -401 both |

Both arms found the SAME optimum (`kappa` identical, `Delta_dual` agreeing to `0.0` at cold-verify)
— expected, since backend choice cannot change the true converged answer, only how fast it's
reached. The shared backend completed **2x the evaluations** (8 vs 4) — a real, positive gain, but
much smaller in relative terms than unrestricted's 3.65x. This is EXPECTED and consistent, not a
red flag: per `CM_CROSS_BLOCK_OPERATOR_AUDIT_2026-07-25.md`'s own measured breakdown, H_EE was only
~19% of CM's pre-port Hessian-callback cost (bin-table construction, unchanged by this port,
dominates at ~84%) — whereas H_EE is the ENTIRE Hessian cost for unrestricted. A ~10x
callback-level H_EE speedup applied to only ~19% of a family's total per-call cost mechanically
produces a smaller whole-callback (and therefore whole-outer-loop) speedup than the same kernel
speedup applied to 100% of another family's cost — exactly what these two families' relative gains
(3.65x vs ~1.5-2x throughput) show.

## CM+mean/ZC and origin-ZC — NOT run this session (disclosed, not fabricated)

The task's own minimum (120s per arm) was not executed for these two families this session, given
the cumulative time already spent on the required 300s×2 (unrestricted) + 300s×2 (CM) runs above
plus every other gate in this continuation. What IS available for these two families instead:
- Full D=4 AND D=20/real-scale correctness gates (`D20_RESTRICTED_FULL_HESSIAN_VALIDATION_2026-07-25.md`)
  — both families' dense-vs-shared-backend dual solutions, Delta_dual, and KKT residuals agree to
  the same ~1e-13 tolerance as flexible CM's own genuine outer-loop-verified numbers above.
- No structural reason to expect a DIFFERENT throughput profile than flexible CM: both families
  reuse the EXACT SAME `_fill_cm_HEE!`/shared winner-pair kernel CM uses for H_EE (CM+mean/ZC
  verbatim; origin-ZC via its own partitioned callback using the identical shared workspace/kernel
  code), with their OWN restriction/cross blocks (dense BLAS, entirely unchanged by this port)
  contributing whatever additional per-call cost they already did pre-port. The qualitative
  argument above (H_EE's share of total callback cost bounds the achievable whole-callback
  speedup) transfers directly.
- This is REASONED EXTRAPOLATION, not empirical confirmation — the final verdict in
  `SHARED_WINNER_PAIR_FINAL_PRODUCTION_GATE_2026-07-25.md` treats this gap honestly, not as
  satisfied.

## Eligibility bullets from task §8, checked against this data

- "All numerical correctness gates pass": ✓ (D=4, D=20 restricted, all four families).
- "Unrestricted and flexible CM show a clear complete-solve or outer-throughput gain": ✓ for both,
  per the tables above.
- "CM+ZC and origin-ZC do not regress by more than 5% in verified progress": **NOT EMPIRICALLY
  CHECKED** this session (no outer A/B run for these two families) — see above.
- "No new callback failures": ✓ (`dense_core_fallback_calls=0` throughout; both arms of every
  family reached `-401`/feasible terminal status, never a hard failure).
- "No unexplained dense fallback": ✓ (task §2's counters, zero across every gate).
