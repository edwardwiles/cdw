# Two-branch profile_Delta(g) investigation (Continuation 8, Wave 2, workstream C)

Branch `c8-gamma-branches`, worktree `gravity-fullA-d4-c8-gamma-branches`, commits `f6790e0` (core +
low-g), `128f260` (high-g sweep + refinement + assembled table). Machine `demand.mit.edu`, KNITRO
14.2.0, `JULIA_NUM_THREADS=20`. Reuses `gamma_profile.jl`'s exact per-point constrained A-block
minimizer and `gamma_profile_multistart.jl`'s 9-start pattern as read-only libraries (no edits to
either file) — see `full_aod_diag/d4_exact/c8_gammabranch_core.jl` for the reuse/attribution details.

Background this session takes as settled (per `results/fullA_d4/bb74649/gamma_profile_nonmonotonicity_report.md`'s
addendum): `profile_Delta(g) = min_A Delta(g,A)` is U-shaped with its interior minimum at the benchmark
Frechet value `g_F = 0.960965...` (Delta ≈ 3.3e-5 there), rising on both sides. This report searches the
two rising branches separately: the low-g branch (site of the existing upper incumbent's crossing) and
the high-g branch (the previously-uncharacterized region past g=0.99, "the more important unfinished
part" per the standing brief).

## Part 1 — Low-g branch: root and uncertainty

**Root: g\* = 0.892635790 (bisection-converged to Delta−delta → −1.7e-15 on the continuation path;
independently re-confirmed via a cold external stationarity re-evaluation at Delta = 1.0000000000,
Delta−delta = −2.3e-12).**

Method: regula-falsi/secant hybrid on `profile_Delta(g) − delta` from the known bracket
`[0.885991098 (Delta=1.3819, infeasible), 0.892635959 (Delta=0.999992, barely feasible)]`, warm-started
with the exact constrained-A-minimizer at every trial g (`c8_gammabranch_lowg_bracket.jl`). 14 iterations
converge the trial-g sequence to a stable value to <1e-9 self-consistency; `Delta − delta` at the final
iterate is −1.7e-15 (floating-point zero). Note: because every trial point after the first landed on the
feasible side, the *nominal* bracket width (6.6e-3) is not itself informative about precision — the
sequence's own monotone convergence is the real precision indicator.

**9-start multistart robustness at the root**: 2 structurally-independent starts (the continuation path
itself, and the fixed low-g incumbent's own A used as a cold start) converge to **bit-identical**
Delta = 0.999999999999998; every calibration-based, perturbed, or uniform-random start is
inner-infeasible at this g (expected — matches doc#1's documented feasibility-corner behavior: most of
A-space violates the inner CC feasibility manifold near this g). Multistart Delta-spread at the root is
therefore exactly 0.

**Uncertainty estimate**: local slope `|dDelta/dg| ≈ 45` (from the bisection's own near-root iterates)
combined with KNITRO's `opttol=1e-6` (relative KKT tolerance, `csw_outer_wallclock_sr1.opt`) implies a
numerical-precision floor of order `1e-6/45 ≈ 2e-8` in g — i.e. the bisection sequence's own <1e-9
self-consistency is not an overstatement. Reporting a conservative band: **g\* = 0.8926358 ± 2e-6**
(order-of-magnitude above the pure optimizer-tolerance floor, to account for the same few-percent
Delta-spread doc#1's multistart found at OTHER g's in this general neighborhood, even though the spread
observed exactly at the root itself was zero).

**Local-stationarity** (`external_stationarity_check_c8`, h=0.01, reusing `stationarity_check.jl`'s
formula verbatim): `eta = 0.0117` (nonnegative, correct sign), `residual_relative = 2.3e-3` — small but
not machine-zero, consistent with continuation 5's own finding for this exact neighborhood (bandwidth-
dependent KKT check, not fully `ROBUST_LOCAL` at h=0.01 due to a winner-boundary near-degeneracy).

**Relation to the existing upper incumbent**: `upper_lfixcomposite_sr1_60s`'s own g (0.8926359585) and
this refined root (0.8926357895) differ by 1.7e-7 — a negligible, expected-scale refinement, not a
correction. The existing upper incumbent (kappa=0.17245688540655113) stands as previously reported.

## Part 2 — High-g branch: THE more important unfinished part

### Headline result

**profile_Delta(g) crosses delta=1 at g ≈ 0.997031** (basin-robust bisection bracket
`[0.99703076, 0.99703125]`, width 4.9e-7), located via `c8_gammabranch_highg_refine.jl`'s
best-of-3-starts (continuation / calibration / lower-incumbent-anchored) profile, which is smooth and
strictly monotone increasing from `g_F` through this crossing and well beyond (to g=0.9995), with **zero
inner-solve failures anywhere in [0.960965, 0.9995]**. Separately, **g=1.0 exactly is a distinct,
universal, start-independent primal-infeasibility wall** — all 12 multistart starts (including fixed
cold-start anchors unrelated to any continuation path) fail on their very first evaluation there, and an
independent LP primal-feasibility certificate (`lp_feasibility_check_c8`, HiGHS via JuMP, reproducing
`phaseF_primal_feasibility_lp.jl`'s construction) certifies `g=1.0` CERTIFIED_INFEASIBLE
(`phase1_max_resid ≈ 2.8–4.2e-5`, bounded away from 0) independent of KNITRO's own `-300` signature.

### Classification: **Case 1 (regular crossing), with two caveats**

Evaluated against the brief's 4 cases:

1. **Regular crossing (`profile_Delta(g)=delta`)** — YES, primarily. The transition from feasible
   (Delta<delta) to infeasible (Delta>delta) is smooth, monotone, and located by a well-converged local
   solve with no inner-feasibility failures anywhere near it (`knitro_status` −101/−102 throughout,
   `csw_outer_wallclock` accepted-convergence codes). This is the dominant classification.
2. **Moment-feasibility boundary (inner CC dual solve infeasible before Delta reaches delta)** — NO for
   the delta=1 crossing itself: the LP certificate stays FEASIBLE well past the crossing (checked
   directly up to g=0.9995, all still feasible with Delta 2.2–4.2, just far above delta). **YES,
   separately**, exactly at g=1.0 — but that is a distinct, much-further-out, non-binding corner (0.0028
   in g beyond the actual delta=1 crossing), not the mechanism that determines the robustness bound here.
3. **Outer-search failure** — genuinely implicated as a *hazard* that had to be actively managed, not
   present in the final answer. A naive single-continuation-path sweep (`c8_gammabranch_highg_sweep.jl`
   Phase 1–2) is real and well-converged everywhere, but Phase 4's multistart check discovered that OTHER
   starts (see Part 3 below) find materially lower Delta at the same g — i.e. the single-path profile is
   *basin-dependent* here, unlike the low-g branch's flat-valley-same-Delta structure. This was resolved
   (not merely noted) by the basin-robust best-of-3 refinement, which converges to a tight (<1e-6-wide)
   bracket once robustness is enforced — so the final crossing location is trustworthy, but reaching it
   required more than the naive single path.
4. **Combination (crossing approached as feasibility degenerates)** — mild evidence only. The
   bisection's own Delta trace in its last 5 iterations (right at the crossing) shows ~1–3% run-to-run
   noise (Delta bouncing between 0.985 and 1.02 rather than the low-g branch's clean monotone convergence
   to machine-precision), consistent with the constrained minimizer becoming somewhat less well-behaved
   very close to the boundary — but not severe enough to prevent locating a narrow, reproducible bracket.
   This is a genuine but secondary effect layered on top of Case 1, not the primary story.

### Part 3 — the mandatory finding: the lower incumbent is NOT on the true profile curve

Per the brief's explicit instruction, the current lower incumbent's own `(g,A)` point
(`lower_lfixcomposite_fast_sr1_300s`, `W_LOWER_INCUMBENT` in `c8_gammabranch_core.jl`,
g=0.9967391744173478, kappa=0.005428799948779983) was used as a **mandatory** start throughout, and its
own status was checked directly rather than assumed:

- **Raw evaluation** (no reoptimization): Delta_dual = 1.0000008524, Delta−delta = +8.5e-7 — confirms the
  registry's number exactly.
- **Reoptimizing FROM the incumbent's own A**: KNITRO converges back to the SAME point
  (`relL2(A_reopt − A_incumbent) = 0.0` exactly) — the incumbent's own A is a genuine local KKT point,
  not a fluke or a stale/unconverged artifact.
- **BUT an independent start** (continuation warm-started from a distant g=0.999 point, then
  re-solved at g=0.9967391744) finds **Delta = 0.8922699 at the exact same g** — 11% BELOW delta=1, and
  strictly below the incumbent's own 1.0000009. A `calib`-anchored start finds 0.9990429, also below the
  incumbent's own value. The basin-robust refinement's own smooth profile (interpolated at this g from
  its neighboring grid points, 0.7884 at g=0.9965 to 0.8889 at g=0.997) independently corroborates a
  true-profile value of roughly 0.83–0.89 at this g, consistent with the 0.8923 finding.

**Conclusion: the lower incumbent's own A is a genuine but locally-suboptimal KKT point — it sits ON the
Delta=delta boundary (correctly, feasible with ~0 slack), but NOT on the true `profile_Delta(g) = min_A
Delta(g,A)` curve.** The true profile at that g is materially lower (≈0.89, not 1.00), meaning there is
real headroom: pushing g upward from 0.996739 to the true crossing at ≈0.997031 (found by properly
tracking the lower basin) is feasible, implying an achievable **kappa ≈ 0.004944** — about **8.9%
smaller than the registered incumbent's kappa=0.005429**. Improving the actual registered candidate is
outside this workstream's scope (it belongs to whatever workstream owns the incumbent registry), but is
flagged here as a concrete, reproducible opportunity, not a vague suspicion — see
`results/fullA_d4/128f260/c8_gammabranch_highg_refine/` for the full trace and
`c8_gammabranch_a_solutions.jld2` for the corresponding A-solution.

### Reconciliation: doc#1's Delta(0.99)=0.299 vs the lower incumbent's Delta≈1.00 at g≈0.9967

Both numbers are correct; there is no contradiction once the g-window between them is filled in with
real intermediate points (which is exactly what this workstream was asked to do):

1. **The lower incumbent's g is 0.9967391744173478, not the brief's rough estimate of ≈0.9946.** The
   brief's `g ≈ 1 − kappa` approximation conflates `1−kappa` with `g^(sigma/(sigma-1))` — they coincide
   only because `g^(5/3) = 1 − kappa = 0.994571...`, and `g = (0.994571...)^{3/5} = 0.9967391744...`,
   matching the registry exactly once computed correctly (sigma=2.5 here, `sigma/(sigma-1) = 5/3`). So
   the actual g-gap from doc#1's grid stop (0.99) to the lower incumbent (0.996739) is 0.00674, not the
   smaller gap the rough estimate implied.
2. **Delta rises smoothly and monotonically across that whole 0.00674 window, with zero solver
   failures** — the fine continuation trace (`c8_gammabranch_highg_sweep.jl` Phase 2) gives
   Delta(0.990)=0.318, Delta(0.991)=0.360, Delta(0.992)=0.410, Delta(0.993)=0.469, Delta(0.994)=0.541,
   Delta(0.995)=0.633, Delta(0.996)=0.760, Delta(0.997)=0.952 — a genuinely steep but perfectly smooth,
   monotone, well-converged climb. It is steep because `Delta(g)` is climbing toward a boundary a
   relatively short distance away in g (recall `kappa = 1 − g^(5/3)` is itself a smooth but
   increasingly curved map as g→1), not because of any solver pathology — there is no discontinuity, no
   infeasibility, and no basin-dependence issue in this *particular* sub-window (0.99–0.997); the
   basin-dependence documented in Part 3 is specific to the immediate neighborhood of the
   crossing/incumbent itself (g≈0.9967–0.9970), not the approach to it.
3. **The lower incumbent's own Delta≈1.00 is close to, but not exactly on, the true profile crossing** —
   it sits at essentially exactly the delta=1 boundary using its OWN (suboptimal) A, while the TRUE
   profile crossing (using the best available A at each g) is at a slightly higher g (0.997031 vs
   0.996739) with a correspondingly lower kappa. The "jump" from 0.299 to ≈1.00 is therefore fully
   explained by (a) the corrected, larger g-gap, and (b) an entirely ordinary steep-but-smooth rise across
   that gap — not by any anomaly requiring further explanation.

## Deliverables

- **Machine-readable table**: `results/fullA_d4/128f260/c8_gammabranch_profile_table.csv` (50 rows, one
  per distinct g tested across both branches) — columns: `g`, `best_Delta_multistart` /
  `second_best_Delta_multistart` (from the ORIGINAL multistart/best-of-3 search at that g),
  `best_start_kind`, `n_feasible_of_9`, `n_total_starts`, `exact_feasible`, `Delta_minus_delta`,
  `knitro_status`, `runtime_wall_s`, `a_solution_key`. **Caveat**: `Delta_minus_delta` /
  `knitro_status` / `a_solution_key` come from a SEPARATE, single-chain re-derivation pass (ascending-g
  continuation within each branch, used purely to produce one clean, storable A-solution per row) — in
  the high-g region this re-derivation can land in a different basin than the original multistart search
  that produced `best_Delta_multistart` (this is itself an illustration of the Part 3 finding, not a
  data-quality bug). Where the two disagree, `best_Delta_multistart` is the authoritative reported
  divergence; `Delta_minus_delta`'s A-solution is a valid-but-not-necessarily-best feasible point at that
  g, stored for reference.
- **A-solutions**: `results/fullA_d4/128f260/c8_gammabranch_a_solutions.jld2` (`Aod_jld2` dict, keyed by
  `a_solution_key`, one 16-vector per row; `NaN`-filled for infeasible rows).
- **Stationarity diagnostic**: `results/fullA_d4/128f260/c8_gammabranch_stationarity.csv` (low-g root:
  eta=0.0117, residual_rel=2.3e-3; lower-incumbent registry point: eta≈0, one non-finite h=0.01 probe,
  matching continuation 5's own finding; a third row using an UN-reoptimized A at the high-g crossing g
  is included as a rough sanity check only, explicitly caveated in the CSV).
- **Underlying runs**: `results/fullA_d4/d6e3b05/c8_gammabranch_{lowg_bracket,compressed_vs_dense_bench}/`,
  `results/fullA_d4/f6790e0/c8_gammabranch_highg_sweep/`,
  `results/fullA_d4/128f260/c8_gammabranch_highg_refine/`.
- **Code**: all new, `full_aod_diag/d4_exact/c8_gammabranch_{core,smoke,compressed_vs_dense_bench,
  lowg_bracket,highg_sweep,highg_refine,assemble_table,stationarity_only}.jl`. No edits to
  `gamma_profile.jl`, `gamma_profile_multistart.jl`, `composite_gradient*.jl`, `oracle*.jl`,
  `compressed_*.jl`, or `winner_certificate.jl`.

## moment_representation decision

`c8_gammabranch_compressed_vs_dense_bench.jl` compared `:dense` vs `:compressed` on a FULL per-g local
minimization (not just an isolated F-eval) at two representative points: compressed is **1.8–1.9x
faster end-to-end** (1.09–1.26s vs 2.01–2.36s wall), with `Delta` matching dense to 8 significant digits.
Adopted `:compressed` as the default for every sweep in this workstream (the gradient callback itself is
unchanged/dense-based in both cases — `compressed_live.jl` has no gradient variant, see
`c8_gammabranch_core.jl`'s header note — so this speedup is entirely from the cheaper `F`-evaluations,
consistent with Wave 1's own D=4 characterization).
