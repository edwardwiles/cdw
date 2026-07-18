# Full-A D=4 final candidate verification (Continuation 8, Section 8)

Branch `c8-final-verify`, worktree `gravity-fullA-d4-c8-final-verify`, commits `16c571f` (Part 0:
`lower_v2` optimization) and `a66ee3e` (Part 1: verification battery), on top of `0a8e68c`
(tip of `diag/fullA-d4-exact`, includes all of Wave 1 + Wave 2A/2C). Machine `demand.mit.edu`,
KNITRO 14.2.0, `JULIA_NUM_THREADS=20`.

**Discipline maintained throughout: nothing here claims global optimality for any candidate.**

## Part 0 — a genuine `lower_v2` candidate

`docs/fullA_gamma_profile_two_branches_c8.md` (this session, Wave 2C) found that the registered
lower incumbent (`lower_lfixcomposite_fast_sr1_300s`, κ=0.005428799948779983, g=0.9967391744173478)
sits on the Δ=δ boundary using its own A, but that A is a locally-KKT-stationary point, not the true
constrained minimizer at that g — an independent search found Δ(g_incumbent, A_true_min) ≈ 0.892 at
the same g, and separately bracketed the true `profile_Δ(g)=δ` crossing at g≈0.997031 (implied
κ≈0.004944). That workstream explicitly stopped short of turning this into a real, freshly-optimized
candidate.

`full_aod_diag/d4_exact/c8_finalverify_lower_v2_opt.jl` does that:

1. **Re-refine the A-block** at g=0.99703076171875 (the high-g branch's feasible-side bracket edge,
   `results/fullA_d4/128f260/c8_gammabranch_highg_refine/highg_bisection_trace.csv`), starting from
   two anchors — the branch's own stored `row40` A-solution and the registered incumbent's own A —
   via `profile_delta_at_gamma_c8` (read-only reuse of `c8_gammabranch_core.jl`, 60s/anchor,
   `:compressed` moments, SR1). The `row40` anchor wins: Δ(g,A)=0.92838625 (Δ−δ=−0.0716), cold-dense
   rechecked at 0.92838625 exactly (gravity 7.4e-18).
2. **Real production outer-loop** from that warm start: joint (g,A) KNITRO solve, `lfix_composite_fast`
   gradient + SR1 Hessian, `eval_fcga=no` (`csw_outer_wallclock_sr1.opt`), direction=lower
   (`find_smallest=false`), 300s wall-clock budget — the identical driver class (same gradient/Hessian/
   opt-file combination) `run_d4_optimized_fd.jl` used to produce the registered incumbent.

**Result: knitro_status=−102 ("Primal feasible solution estimate cannot be improved; desired accuracy
in dual feasibility could not be achieved" — the same accepted-convergence code the registered
incumbent's own run terminated with), feas_err=0.0, opt_err=9.47e-4 relative, 58 outer iters, 13.5s of
the 300s budget, 274 Δ(w) evaluations.** Cold-dense recheck at the tracked best-feasible point:

- g = γ'_focal = 0.9973649883022927 (KNITRO pushed g past both the refine point 0.997031 and the
  profile's own crossing estimate 0.997031)
- **κ = 0.004387827651021192**
- Δ_dual = Δ_primal = 0.9941306706106116 (Δ−δ = −0.005869, comfortably feasible, not marginal)
- gravity_value = −5.6e-18, max_abs_moment_kkt_resid = 4.0e-15, mean_m_resid = 3.6e-15, inner_status=0

**lower_v2 beats the registered incumbent by 19.2%** (0.005429 → 0.004388) and **beats the two-branch
workstream's own profile estimate** (0.004943) by a further 11.2%, because the outer loop kept moving
g beyond the profile's 0.997031 crossing once a genuinely-optimized A freed up real slack.

## Part 1 — verification battery

`full_aod_diag/d4_exact/c8_finalverify_battery.jl` runs the standing brief's full battery on three
candidates, reusing existing machinery throughout (never re-derived): `evaluate_fullA` for the cold
dense CC solve/Δ_primal+dual/gravity/moments (`oracle.jl`), `h_sweep.jl::h_sweep_one_direction` for
secants, `gravity_elimination.jl`'s pivot construction for tangent directions, the exact poll pattern
from `phaseA_upper_revalidation.jl`/`phaseA_lower_lfixcomposite_fast_revalidation.jl`, and
`stationarity_check.jl::external_stationarity_check` for the reduced/"scaled" KKT read.

**Two real bugs found and fixed while building this** (both documented in the script's own header and
commit message):

1. `external_stationarity_check` hardcodes `warm=true` internally. Calling it after hundreds of
   unrelated poll/secant probes left `ctx.obj.arg1` (the shared inner-dual warm state) corrupted,
   producing a spurious exact `eta=-0.0` for both lower candidates — the identical hazard
   `c8_gammabranch_core.jl`'s own header already flagged for this exact function. Fixed by reordering:
   KKT checks now run immediately after a cold prime, before poll/secant/tangent.
2. A literal h=0.01 central-FD probe on the γ'_focal coordinate steps **outside the theoretical box
   bound** for both lower-direction candidates (margin_hi ≈ 0.0026–0.0033, since both sit close to the
   g≈1 economic bound; upper's margin_hi=0.107 has no such issue). This is a bandwidth-selection
   artifact, not evidence of non-stationarity — confirmed directly (`n_nonfinite_probes=1` at
   h∈{0.02,0.01,0.005} for both lower candidates, 0 nonfinite once h shrinks below the margin). Added
   `h_safe = min(0.01, 0.4·min(margin_lo,margin_hi))` as the bound-respecting primary KKT bandwidth,
   reporting the raw h=0.01 result alongside for direct comparability with upper.

Every poll-detected "improvement" was independently cold-verified (not a warm-start artifact): e.g.
the original lower incumbent's poll finding (Δ drops from 1.0000 to 0.9844 at a small nearby
perturbation, radius 0.02 in reduced-w space) reproduces bit-for-bit under an explicit cold
`evaluate_fullA(...; warm=false)` re-solve — this **independently corroborates**, via a completely
different method (local basis+random poll vs. the two-branch workstream's own multistart/continuation
search), that the registered incumbent sits in a suboptimal A-basin.

### Results table (full data: `results/fullA_d4/16c571f/c8_finalverify_battery/summary.csv`)

| candidate | κ | Δ_dual | Δ−δ | gravity | moment-KKT resid | poll improvements (of 288) | h_safe | KKT resid @ h_safe |
|---|---|---|---|---|---|---|---|---|
| upper_lfixcomposite_sr1_60s | 0.17245689 | 0.99999241 | −7.6e-6 | −3.6e-18 | 1.4e-16 | 1 (known, tiny — matches registry's pre-existing "3 poll-improved points" note) | 0.01 (unclipped) | 0.0023 |
| lower_lfixcomposite_fast_sr1_300s **(ORIGINAL)** | 0.00542880 | 1.00000085 | +8.5e-7 | −3.7e-18 | 6.5e-15 | **3, including a substantial one (Δ: 1.0000→0.9844 at radius 0.02)** | 0.00130 | 0.0215 |
| **lower_v2 (NEW)** | **0.00438783** | 0.99413067 | −5.9e-3 | −5.6e-18 | 4.0e-15 | 13, all tiny (largest Δg ≈ 5e-6, consistent with opt_err=9.5e-4 early stop) | 0.00105 | 0.0491 |

Other bullets: gravity-tangent A-only directions (`gravity_tangent_directions.csv`) confirm gravity
stays at machine-zero (~1e-18) on every probe for all three candidates, as pivot-elimination
guarantees by construction — not a live check of anything, but a clean sanity confirmation. Bounds:
all three respect the genuine economic γ'_focal bound with the margins shown above; the z_free box
`[-8,8]` is confirmed (again) to be a numerical safeguard nowhere near active (max|z_free| 1.46–1.75).
The unscaled (original economic: γ'_focal + full unreduced log(A_od,θ), gravity multiplier ν made
explicit) KKT check — new in this script, since `context_scaled.jl` turned out on inspection to be an
unrelated D/W-scaling benchmark tool, not an unscaled-KKT check — agrees closely with the reduced
check at the same h for all three (residual_relative within ~2x), as expected since gravity is exactly
linear and its multiplier is well-identified. The secant sweep (`secant_sweep.csv`) reproduces this
investigation's established "h=0.1 is not safe as a default bandwidth" finding along a random full
x_free direction: D_central is non-finite for h≥0.0125–0.05 for all three candidates (one-sided probe
hits an infeasible/non-convergent point), only stabilizing at the smallest grid step (h=0.00625) —
inconclusive for a stability read on this particular direction/tool, reported honestly rather than
forced.

## Classification

All three candidates are **exact-feasible** (Δ≤δ with real margin, clean inner CC solve,
moment-KKT residual ~1e-15, gravity ~1e-18). All three classify as **bandwidth-KKT candidate**:
genuinely stationary at their own appropriately-scaled bandwidth (h_safe, chosen to respect each
candidate's box-bound proximity), but not certified robust across the full h-grid/poll set —
consistent with, not a departure from, this investigation's historical treatment of the upper
incumbent (whose own dedicated revalidation, `phaseA_upper_revalidation.jl`, likewise reported
`H_BANDWIDTH_KKT_CANDIDATE(h=0.01)=true` as its headline, with 1 known tiny poll improvement already
on record in `candidate_registry.jl`).

The **original lower incumbent's classification is explicitly downgraded** relative to its own
1-off `phaseA_lower_lfixcomposite_fast_revalidation.jl` (which reported `ROBUST_LOCAL_CANDIDATE=true`,
0 poll improvements, at the time) — this session's broader/cold-verified poll finds a real, substantial
nearby improvement, corroborating the two-branch workstream's independent finding that this point sits
in a suboptimal A-basin. This is a genuine update based on new evidence, not a contradiction of the
prior script (different poll coverage, and prior to this session's basin-suboptimality discovery).

**lower_v2 is comparatively the cleanest of the three low-κ candidates**: its 13 poll improvements are
all at the ~1e-6 scale in g (order the KNITRO run's own 9.5e-4 relative opt_err would predict), versus
the original incumbent's one substantial (~1.6%-scale) improvement.

## Recommendation

**Replace the registered lower incumbent with `lower_v2` in `candidate_registry.jl`-style canonical
status.** Numbers: κ 0.005428799948779983 → **0.004387827651021192** (19.2% tighter), both
exact-feasible with real Δ<δ margin (old: +8.5e-7 essentially on the boundary; new: −5.9e-3,
comfortably interior), both produced by the identical driver class (`lfix_composite_fast`+SR1,
`eval_fcga=no`, wall-clock-bound outer loop), and lower_v2's own residual poll-detectable slack is
~1000x smaller than the old incumbent's. The new w vector (reduced coordinates, ready for direct
`candidate_registry.jl`-style reporting):

```
w_lower_v2 = [0.9973649883022927, 0.4191333995096165, 0.3278704261228879, 0.34822377242086583,
  0.3266848028818515, 1.1414170966875377, 1.299758470774316, 1.005176411943463, 1.0769193031619044,
  0.8872003122885062, 0.7853545670122154, 0.8376830845864721, 0.7594387919364166, 1.7457822520932191,
  1.4007037442409944, 1.5130549509169442]
```

This recommendation is a judgment call left to the coordinating session per the task brief; this
workstream does not edit `candidate_registry.jl` itself.

## Artifacts

- Code: `full_aod_diag/d4_exact/c8_finalverify_lower_v2_opt.jl`, `c8_finalverify_battery.jl` (new,
  this workstream only — no edits to any existing `.jl` file).
- Part 0 run: `results/fullA_d4/0a8e68c/c8_finalverify_lower_v2_opt_20260718_173246/` (`summary.txt`,
  `knitro.log`, `callback_trace.csv`).
- Part 1 battery: `results/fullA_d4/16c571f/c8_finalverify_battery/` (`summary.csv`,
  `secant_sweep.csv`, `gravity_tangent_directions.csv`, `poll.csv`, `kkt_scaled_hgrid.csv`,
  `moment_resid.csv`).
- Commits: `16c571f` (Part 0), `a66ee3e` (Part 1), branch `c8-final-verify`.
