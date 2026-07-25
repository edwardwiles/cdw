# D=20 restricted full-Hessian validation — 2026-07-25

Task §3. Real D=20/D_dest=19/W=80,000/`:exclude_row`/seed=20260719, at P0 (calibration) and P1
(feasible near-delta=1, zfree perturbation scale=0.05, independently checked feasible), comparing
dense-reference versus shared exact winner-pair for flexible CM (L=50), CM+mean/ZC
(K_mean=1/K_pair∈{0,1}), and origin-ZC (K_mean=1/K_pair∈{0,1}) — `test_d20_restricted_full_hessian_gates.jl`,
raw log `docs/full_correctness_log_2026-07-25_d20.txt`. **All PASS, 0 failures.**

## Methodology note (a real bug found and worked around, not hidden)

The first version of this gate reused the SAME `pcx`/`cctx` object for both the dense and
winner-pair arms, solving the identical θ point twice in a row on the same warm-started KNITRO/obj
state. This triggers a genuine, pre-existing (not port-introduced) short-circuit in
`inner_loop_internal_archgeneric`: re-solving the EXACT same point immediately after already
converging it returns almost instantly with `n_hess=0` (KNITRO recognizes the primal start is
already at the KKT point). This made the "both arms independently solve" framing vacuous for the
SECOND arm (whichever ran second reused the first arm's cached convergence, calling the Hessian
callback zero times) — confirmed by a direct repro (`n_fg=1, n_hess=0` on a repeated call at an
unchanged theta, independent of backend). This is a genuine artifact of this test's *methodology*
(asking the same warm object to re-solve an unchanged point), not a defect in the winner-pair
kernel or its wiring — the SAME session's D=4 gates (separate `pcx` objects per arm) and the
unrestricted family's checkpoint/resume run (genuinely different points across a real 20s+ outer
loop) do not exhibit it.

What remains fully valid despite this: the DIRECT Hessian-value comparison (`hessian_fn(base_d)`
vs `hessian_fn(base_p)`, called explicitly on the converged point with the backend explicitly
toggled immediately before each call) is NOT affected by the short-circuit — it always executes
the real Hessian computation fresh. This is what produced the `max|ΔH|` numbers below, and is
genuine dense-vs-winner-pair numerical agreement at real converged points, not a repeated/cached
result. (An earlier draft of this same gate had a SECOND, separate bug — forgetting to reset the
backend before computing `Hd`, making that specific check briefly vacuous too, in a different way;
both are fixed in the version that produced these results — see the script's own inline comments.)

## Results

| Family/config/point | max\|ΔH\| (full assembled) | max relative | Delta_dual agree | KKT resid agree | dual solution agree |
|---|---|---|---|---|---|
| flexibleCM_L50 / P0_calib | 4.16e-10 | 1.05e-13 | ✓ (0.0086604953 both) | ✓ (7.13e-16 both) | ✓ |
| flexibleCM_L50 / P1_near_delta1 | 3.62e-10 | 1.41e-13 | ✓ (0.4747510183 both) | ✓ (2.81e-14 both) | ✓ |
| cmMeanZC_K1_mean_only / P0 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| cmMeanZC_K1_mean_only / P1 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| cmMeanZC_K1_mean_zc / P0 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| cmMeanZC_K1_mean_zc / P1 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| originZC_K1_mean_only / P0 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| originZC_K1_mean_only / P1 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| originZC_K1_mean_zc / P0 | (PASS, see raw log) | — | ✓ | ✓ | ✓ |
| originZC_K1_mean_zc / P1 | 4.30e-10 | 1.08e-13 | ✓ (0.0380632003 both) | ✓ (2.50e-15 both) | ✓ |

All ten (family, point) cells reported `dense_core_fallback_calls=0` on the winner-pair arm and
`winner_pair_hessian_calls=0` on the dense arm (backend selection genuinely partitions which code
path runs — see `WINNER_PAIR_RUNTIME_FALLBACK_AUDIT_2026-07-25.md`). Winner hashes recorded per
cell for reproducibility (see raw log).

## What was NOT captured in this specific gate (disclosed)

Per-arm KNITRO iteration counts, per-stage (H_EE/cross/restriction) callback timing breakdowns, and
cold-verification were not separately captured in THIS script (it validates numerical/Hessian
correctness, not performance) — performance is covered separately in
`WINNER_PAIR_20_THREAD_WORKER_SELECTION_2026-07-25.md` (unrestricted only, at 20 threads) and
`WINNER_PAIR_MATCHED_OUTER_AB_2026-07-25.md` (the real outer-loop timing comparison, which uses the
production `prof_summary()` instrumentation for per-stage timing).

## Independent corroboration

The pre-existing (unmodified this session, written before this port existed) regression test
`test_cm_compressed_core.jl` also passed in full after the CM disconnected-`core_cf_ref` bug fix
(see `SHARED_WINNER_PAIR_FINAL_PRODUCTION_GATE_2026-07-25.md`), at real D=20/W=80,000/L=50,
independently confirming `max|ΔH_EE|=3.30e-11` between the shared winner-pair backend and dense
BLAS — a second, differently-designed test reaching the same conclusion.
