# Is the profile_Delta(g) dip-then-rise real, or a warm-start artifact?

**Branch** `diag/fullA-d4-exact`, worktree commit `bb74649` (content-identical to `gravity-fullA-d4`).
**Date** 2026-07-18. **Machine** demand.mit.edu, KNITRO 14.2.0, `JULIA_NUM_THREADS=20`.
**Driver** `full_aod_diag/d4_exact/gamma_profile_multistart.jl` (reuses `gamma_profile.jl`'s exact
per-point A-block minimizer + the validated `lfix_composite` gradient, `h_mode=:adaptive`).
**Run dir** `results/fullA_d4/bb74649/gamma_profile_multistart_20260718_113923/`.

## Verdict

**The dip-then-rise is REAL, not a warm-start / continuation artifact.** `profile_Delta(g) = min_A
Delta(g, A)` is a smooth, essentially unimodal **U-shaped** curve with a single interior minimum near
**g\* ≈ 0.96** where Δ ≈ 9.4e-5 (the model very nearly reproduces the factual bilateral estimand
there), rising monotonically on both sides. Nine independent starts per g — including two structurally
distant *cold* starts (incumbent-A from g≈0.8926, and the calibration A) that share no path with the
continuation — all converge to the **same** Δ at each g, so the rise cannot be an artifact of the
single continuation path landing in a worse basin.

**Implication for the methodology:** the non-monotonicity does **not** threaten the robustness bound.
The feasible set `{g : profile_Delta(g) ≤ δ=1}` is a single connected **interval** whose *lower*
endpoint is a **unique** crossing at g ≈ 0.8926 (the current upper incumbent) — the profile is strictly
monotone decreasing from that crossing down to g\*, so `find_smallest` is well-posed. What is *false*
is any assumption that profile_Delta is monotone **decreasing** in g (which would make feasibility a
half-line `{g ≥ g_lo}`); it is two-sided. But since the bound takes the smallest feasible g, the answer
is unchanged.

## Refined multistart profile (0.94 ≤ g ≤ 0.99, plus the two coarse anchors)

`min_A Delta` = min over the feasible starts; `max` = max over feasible starts (spread ≈ optimizer
tolerance + flat-valley wandering, see basins below). `δ = 1.0` throughout, so every interior point is
feasible with large slack. Values are the authoritative in-memory numbers (`multistart_summary.csv`).

| g | multistart-min Δ | max Δ (feasible) | n feasible / 9 | Δ − δ |
|---|---|---|---|---|
| 0.9400 | 6.473e-02 | 6.512e-02 | 3 | −0.935 |
| 0.9450 | 3.599e-02 | 3.704e-02 | 3 | −0.964 |
| 0.9500 | 1.679e-02 | 1.695e-02 | 3 | −0.983 |
| 0.9550 | 4.994e-03 | 5.439e-03 | 3 | −0.995 |
| 0.9570544 (anchor) | 2.020e-03 | 3.908e-03 | 4 | −0.998 |
| **0.9600** | **9.354e-05** | 1.018e-04 | 3 | −1.000 |
| 0.9650 | 3.189e-03 | 3.298e-03 | 3 | −0.997 |
| 0.9700 | 1.654e-02 | 1.701e-02 | 3 | −0.983 |
| 0.9750 | 4.314e-02 | 4.610e-02 | 3 | −0.957 |
| 0.9785272 (anchor) | 7.272e-02 | 7.566e-02 | 4 | −0.927 |
| 0.9800 | 8.899e-02 | 9.397e-02 | 3 | −0.911 |
| 0.9850 | 1.658e-01 | 1.683e-01 | 3 | −0.834 |
| 0.9900 | 2.991e-01 | 3.065e-01 | 3 | −0.701 |

ASCII view of `log10(min_A Delta)` vs g (interior minimum at g≈0.96):

```
 g       log10(Δ)   |----------------------------------------|
0.940    -1.19      |                    ##                  |
0.945    -1.44      |                 ###                    |
0.950    -1.77      |              ###                       |
0.955    -2.30      |          ###                          |
0.957    -2.69      |        ###                            |
0.960    -4.03      |#  (min, Δ≈1e-4)                       |
0.965    -2.50      |         ###                           |
0.970    -1.78      |              ###                       |
0.975    -1.37      |                 ###                    |
0.9785   -1.14      |                    ##                  |
0.980    -1.05      |                     ##                 |
0.985    -0.78      |                        ###             |
0.990    -0.52      |                           ###          |
```

## Why this is not a continuation artifact (the direct test)

The task's concern was that `gamma_profile.jl`'s single continuation path might land in different
basins at different g and report whichever it finds first. This run breaks the continuation dependence
by running, at every g, three structurally-independent starts plus six perturbations:

- **continuation** (previous g's best) — the only path-dependent start;
- **incumbent** — fixed A from the upper incumbent (g≈0.8926), same at every g, no path;
- **calib** — the calibration A (`pivot_reduce(log Aod0)`), same at every g, no path;
- incumbent±{0.5,1.5}·randn, calib±{0.5,1.5}·randn, two uniform[−2,2] draws.

At the **rise** point g=0.9785 the three independent starts return Δ = 0.0727 (cont), 0.0757 (incumbent),
0.0741 (calib) — all near 0.073, none near the g=0.96 minimum of 1e-4. A continuation artifact would
require the *fixed* incumbent/calib starts to find a much lower value at 0.9785; they do not. The same
tight agreement holds at every g in the table. (The randn/uniform perturbations mostly start
inner-**infeasible** — KNITRO returns the sentinel on eval 1, status 0, n_eval 1 — because most of
A-space violates the inner CC feasibility manifold; this is the known feasibility-corner behavior, not
a solver failure. Where a perturbation *did* stay feasible, e.g. calib+0.5r at g=0.9570 → 3.9e-3, it
landed in the same Δ neighborhood.)

Cross-check against the pre-fix coarse single path (`results/fullA_d4/4f696b7/…111444`): coarse had
g=0.9785 → Δ=0.0733; multistart-min here is 0.0727. The rise reproduces to 3 significant figures.

## Basin structure: one Δ-valued basin with a flat A-valley (not competing basins)

The per-point clustering flags "2–4 basins", but that counts distinct **A_od** vectors, not distinct
**Δ** values. The three independent feasible starts converge to nearly-identical Δ while sitting at
A_od solutions that differ measurably:

| g | Δ(cont / inc / calib) | relL2(A) cont–inc | cont–calib | inc–calib |
|---|---|---|---|---|
| 0.960 | 9.79e-5 / 1.02e-4 / 9.35e-5 | 2.79e-2 | 1.06e-2 | 1.95e-2 |
| 0.9785 | 7.27e-2 / 7.57e-2 / 7.41e-2 | 5.98e-2 | 2.14e-2 | 6.08e-2 |

So at fixed g the minimizer is a **flat / near-degenerate valley in A** (many A_od give essentially the
same Δ — expected, since gravity elimination and the winner-share moments leave gauge/flat directions),
*not* multiple basins at different Δ. Consequently `profile_Delta(g)` is a well-defined single value at
each g (the few-percent min↔max spread is optimizer tolerance plus flat-valley wandering), and the
U-shape it traces is unambiguous.

## What this says about `profile_Delta(g) = δ` crossings

- **Low-g side (the one that matters):** from the coarse run, Δ(0.8846)=1.477 (>δ), Δ(0.8926)=1.0000
  (≈δ), then strictly decreasing through this whole refined interior. Exactly **one** crossing of Δ=δ,
  at g ≈ 0.8926 = the current upper incumbent. `find_smallest` (smallest feasible g) is well-posed and
  its answer is unaffected by the interior dip/rise.
- **High-g side:** the profile rises off g\*≈0.96 but only reaches 0.299 at g=0.99, and the calibration
  corner g=1.0 is inner-**infeasible** (Δ=NaN). On this grid the upper feasibility boundary is set by
  inner-infeasibility near g→1, not by a second Δ=δ crossing. Either way it is far above the incumbent
  and irrelevant to the smallest-g bound.
- **New substantive finding:** g\* ≈ 0.96 (Δ≈1e-4, i.e. near-exact reproduction of the factual
  estimand) is a distinguished "best-fit γ'" that is *not* the robustness-boundary γ' (≈0.8926). The
  profile genuinely has interior structure; worth a note in the canonical writeup, but it does not move
  the reported κ.

## Reproduce

```bash
source .knitro_env.sh
JULIA_NUM_THREADS=20 MS_MAXTIME_PER_START=18 \
  julia --project=. full_aod_diag/d4_exact/gamma_profile_multistart.jl
# analysis (reads multistart_summary.csv, the authoritative in-memory numbers):
julia --project=. full_aod_diag/d4_exact/analyze_multistart.jl <run_dir>
```

## Caveats / provenance

- `multistart_summary.csv` and the run stdout (`ms_fullrun.log`) are computed in-memory and are the
  authoritative source for all numbers above. The per-start dump `multistart_per_start.csv` from THIS
  run has a cosmetic column-shift on the two `uniform[-2,2]#N` rows per g (an unquoted CSV field with a
  literal comma); those two rows per g were infeasible sentinels anyway. The source has been fixed
  (kinds renamed `uniform_pm2_a/b`) so a re-run produces a clean per-start CSV; the aod columns for the
  seven comma-free start kinds (incl. all feasible ones used above) are correctly aligned.
- Multistart is strong evidence, not a global-optimality proof: three mutually distant starts agreeing
  to a few percent at every g, plus reproduction of the coarse anchor, is what "not an artifact" rests
  on.
- Inner CC dual warm-state (`ctx.obj.arg1`, DUAL_WARM_MODE=persist) persists across starts by design;
  the inner solve is convex so this affects inner speed only, never which outer A-basin KNITRO descends
  into — kept identical to the coarse run for apples-to-apples comparison.
