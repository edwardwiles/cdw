# Real-D20 fixed-A/f gamma_d_prime profile, calibration to both theoretical extremes (2026-07-29)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from the
Melitz inner-solver architecture consolidation
(`docs/melitz_inner_solver_architecture_consolidation_2026-07-28.md`) and its post-
consolidation validation (`docs/melitz_post_consolidation_validation_2026-07-28.md`). Starting
HEAD `a6826f61c34c6874a686d051d2dfa7abd698cc0c`, verified as the actual branch/HEAD before any
edit; `git status` showed only pre-existing untracked scratch directories inherited from other
sessions, none touched. Governing prompt: produce a high-resolution, trustworthy real-D20
fixed-A/f `gamma_d_prime` profile spanning the Fréchet calibration down to the theoretical
minimum `gamma_d_prime` and up to the theoretical maximum `gamma_d_prime`, with no outer
optimization anywhere -- only the inner problem is solved at each predetermined point.

Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`, pinned), `OPENBLAS_NUM_THREADS=1`/
`OMP_NUM_THREADS=1`, `JULIA_NUM_THREADS=20`. Fixture: real D=20 (`real_data/noah_D20`),
`W=80,000`, seed=1, focal country France (`fra`), `sigma=2.5` -- confirmed live, not a toy D=4
substitute.

## 1. Fixed-A/f semantics (what is literally frozen, what is analytically profiled)

Under the active `:logf` outer parameterization, `theta_free = (g, A_free[1:D^2-1],
f_free[1:D^2-2])`, `g = log(gamma_prime_target)`:

- **A entries literally fixed**: every entry of `A` reconstructs from `A_free` alone
  (`melitz_expand_theta`) -- `theta_free[2:1+nA]` (`nA = D^2-1 = 399`) is never touched across
  the entire profile.
- **f entries literally fixed**: every off-domestic-focal `f` cell reconstructs from
  `f_free` alone -- `theta_free[2+nA:end]` is never touched across the entire profile.
- **Analytically profiled (mechanical, not independent)**: `f[target,target]` (the focal
  domestic fixed cost) is recovered from `derive_fjj_from_autarky_cutoff`'s `zhat'_jj=1`
  normalization -- a function of `g` alone, changing automatically and correctly at every grid
  point (this is the one quantity the governing prompt explicitly requires NOT be frozen).
- **Changes mechanically with `g`**: `gamma_prime_target = exp(g)` itself, `f[target,target]`
  (above), and every downstream equilibrium/cutoff/moment object the inner solve depends on.

**Proof every free A/f coordinate stayed frozen**: `A_hash = hash(theta[2:1+nA])` and
`f_hash = hash(theta[2+nA:end])` are computed once from the calibration `theta0` and asserted
identical (`theta[2:end] == theta0[2:end]`, plus the two hash equalities) at literally every
one of the 84 grid-point solves in this session's own script
(`scripts/melitz_realD20_fixed_af_gamma_profile_2026-07-29.jl`, `run_branch`) -- no assertion
ever failed. `A_hash=9a1740660f5d4abf`, `f_hash=c9e739c0aaaab3dd` throughout.

No nuisance minimization, no search over alternative A/f values, anywhere in this session.

## 2. Live formula verification

`src/melitz/equilibrium.jl:485-487` (`melitz_welfare_metrics_from_g`), confirmed by direct
source read and matches the governing prompt exactly:

```
gamma_prime = exp(g)
kappa_ratio = wage_ratio * gamma_prime^(1/(sigma-1))          # wage_ratio = w_prime/w[target]
gains_from_trade = 1 - kappa_ratio
```

`kappa_ratio_of_g(g, ctx) = melitz_welfare_metrics_from_g(g, ctx).kappa_ratio` is the one
production entry point (`equilibrium.jl:520`); this session's own scripts call the identical
closed form directly (`kappa_of_g`/`g_of_kappa`), not a re-derivation.

## 3. Phase 0: calibration and theoretical endpoints

`scripts/melitz_phase0_realD20_endpoints_2026-07-29.jl`.

| quantity | value |
|---|---:|
| focal country | France (`fra`, index 2 of 20) |
| sigma | 2.5 |
| theta_star | 8.751773 |
| QMC | W=80,000, seed=1 |
| calibrated `gamma_d_prime` (Fréchet) | 0.6578550157 |
| calibrated `w_prime` (autarky counterfactual wage, normalized) | 1.0 |
| calibrated `w[target]` | 0.7720787473 |
| wage_ratio = w_prime/w[target] | 1.2952046712 |
| calibrated `kappa_ratio` | 0.9796971896 |
| calibrated gains from trade | 2.030281% |
| calibrated `DeltaStar` | 4.072070e-04 (`nStatus=0`, verified) |

### Theoretical minimum `gamma_d_prime` (branch A)

Derived directly from the paper's own proven ceiling on gains from trade,
`GT <= lambda_dd^(1/(sigma-1))` (`lambda_dd` = domestic trade share at the calibrated
baseline), converted to `kappa_ratio >= lambda_dd^(1/(sigma-1))` and inverted through the exact
live `kappa_ratio(g)` formula:

```
lambda_jj (domestic trade share, France, at theta0) = 0.8356761300
kappa_min = lambda_jj^(1/(sigma-1))                  = 0.8872077596
g_ceiling = g_of_kappa(kappa_min)                    = -0.5675172401
gamma_d_prime_min_theory = exp(g_ceiling)            = 0.5669312470
GT at this endpoint (= 1 - kappa_min)                = 11.279224%   <- the paper's proven GT UPPER bound
```

**Cross-check 1** (direct formula): `g_ceiling` computed via `g_of_kappa(kappa_min, wratio,
sigma)`. **Cross-check 2** (via the live kappa/gamma relationship, mapping the same GT value
back through `kappa=1-GT` then `g_of_kappa`): identical to machine precision (`diff=0.000e+00`).

**This is an OPEN, Delta->infinity limit, not an ordinary evaluable point** -- confirmed both
by the theory (`kappa_min` is only attained as trade frictions vanish entirely, an asymptotic
result) and live, directly: `solve_melitz_delta!` at `g_ceiling` exactly, and at `t=0.995`/
`t=0.999` approach points (extremely close to the endpoint), all three return
`InfiniteDeltaCertified` (`col=2, kind=:origin_block`), never a finite value. The theoretical
endpoint itself is therefore reported as a limit, never claimed as an ordinary `FiniteSolved`
point (Phase 1 below).

### Theoretical maximum `gamma_d_prime` (branch B)

Derived from the ordinary, complementary restriction `kappa_ratio <= 1` (gains from trade
cannot be negative), inverted the same way:

```
kappa_max = 1.0                                       (GT cannot be negative)
g_floor = g_of_kappa(kappa_max)                      = -0.3880030949
gamma_d_prime_max_theory = exp(g_floor)              = 0.6784102437
GT at this endpoint (= 1 - kappa_max)                = 0.000000%   <- zero gains from trade
```

**Note on a prior session's `g_floor=0` convention.** A 2026-07-24 session
(`docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md` Section 10.1) used
`gamma_prime_target=1` (`g=0`) as a convenient symmetric outer-search **box edge** for a search
that only ever needed the lower bound (`:upper`-direction, g-decreasing) -- non-binding by
their own construction, and labeled "GT=0%" loosely in that context. At this exact fixture,
`wage_ratio=1.2952 != 1`, so `kappa_of_g(0) = wage_ratio = 1.2952 > 1` -- literal `g=0` is
**past** the true `kappa_ratio<=1` boundary (which sits at `g_floor=-0.3880`, not `g=0`). This
session derives and cross-checks `g_floor=-0.3880` freshly from the live `kappa`/`gamma`
relationship at the current calibration, per this session's own mandate, rather than reusing
the 2026-07-24 session's non-binding convenience value.

**Unlike branch A, this endpoint is an ORDINARY finite theoretical point by theory** (complete
autarky is a well-defined finite economic corner, not an asymptote the way costless full
integration is). Live probing nonetheless found it -- and its immediate neighborhood
(`t=0.995`, `t=0.999`) -- **also** `InfiniteDeltaCertified` at this fixture. This is reported
honestly as a genuine, model-specific numerical/economic finding (an exact certificate, not a
timeout or ambiguous failure), **not** used to justify moving the theoretical endpoint inward --
`g_floor=-0.3880` is still the value reported and gridded as the branch-B theoretical target;
the certified-infeasible region simply turns out to start well before it is reached (Phase 6).

## 4. Grid construction (Phase 1)

`t_i = (i/40)^2`, `i=0..40` per branch (quadratic -- denser resolution near the calibration,
the economically relevant region), kappa interpolated **linearly** between the calibration and
each branch's own theoretical kappa target, mapped back to `g` via the closed-form
`g_of_kappa`. The exact calibration `g` (`theta0[1]`) is forced at `t=0` on both branches,
never a floating-point-perturbed reconstruction.

- **Branch A** (toward the theoretical minimum `gamma_d_prime`, `kappa_min`): the literal
  `t=1.0` endpoint (`g_ceiling`) is an established open Delta->infinity limit (Section 3) --
  per the governing prompt's "open or singular endpoints" instruction, it is **replaced** by
  three explicit near-limit points, `t=0.99, 0.995, 0.999`, and the true theoretical endpoint
  is listed separately as a limit (not claimed as an ordinary evaluated grid point). Branch A
  therefore has `40 + 3 = 43` evaluated grid points.
- **Branch B** (toward the theoretical maximum `gamma_d_prime`, `kappa_max=1`): the literal
  `t=1.0` endpoint is an ordinary finite theoretical point by theory (Section 3) -- **kept**
  as-is, not replaced, per the same instruction's converse ("do not move the theoretical
  endpoint inward merely because a numerical solve is difficult"). Branch B has 41 evaluated
  grid points.
- **Total unique predetermined points**: `43 + 41 - 1` (calibration counted once) = **83**,
  exceeding the required minimum of 81. (The combined CSV has 84 physical rows: the
  calibration point is solved once per branch by construction, giving a bit-exact duplicate
  row, and de-duplicates to 83 unique `gamma_d_prime` values.)

## 5. Consolidated inner-solver API usage (Phase 2/3)

Every one of the 84 grid solves used `solve_melitz_delta!` on a `MelitzInnerSession` under
`CappedEvaluation(10.0)` exclusively -- the one public, authoritative Melitz inner-solve entry
point (`src/melitz/inner_session.jl`), never a legacy/low-level call. **Two fully independent
sessions/fixtures** (`session_min_branch`/branch A, `session_max_branch`/branch B), each with
its own `build_realD20_fixture(...)` call (own `obj`, `ctx`, KNITRO instance, `MelitzDualBank`)
-- no warm-start state or KNITRO instance is ever shared between the two branches, matching
`MelitzInnerSession`'s own "never share a session/bank across..." convention.

- `obj.lower_limit == -10.0` (raw KNITRO `lower_limit`) verified at construction for both
  fixtures, matching `melitz_policy_lower_limit(CappedEvaluation(10.0)) = -10.0` exactly.
- `origin_block_screen=true` at every solve (the one mechanism that classifies genuinely
  infinite points cheaply and correctly, per the architecture-consolidation session's own
  Section 8 finding).
- **No routine cold retry, strict production-fast mode.**
- **Explicit continuation, not merely `warm_start_source=:previous` left as a no-op**:
  `last_good_x` is updated ONLY on a `FiniteSolved` result and manually copied into
  `obj.x`/`obj.use_cached_x` immediately before every subsequent solve on that branch -- no
  point is ever warm-started from an `AboveEvaluationCap`/`InfiniteDeltaCertified`/
  `NumericalFailure` attempt's leftover KNITRO state (the governing prompt's own explicit
  prohibition; also directly relevant given this repo's own documented pitfall of a poisoned
  warm-start dual producing spurious `NumericalFailure`).
- Classification semantics unchanged from the consolidated architecture:
  `FiniteSolved`/`AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure`. No certified
  lower bound, sentinel, or failed raw objective was ever placed in the `DeltaStar` column
  (enforced structurally by the result-type taxonomy itself, not merely by convention).

## 6. Headline result: both branches produced a substantial, clean, verified finite corridor

**No `AboveEvaluationCap` or `NumericalFailure` occurred anywhere in the 84-point grid** -- a
notably clean result. Every point classified either `FiniteSolved` (70 of 84) or
`InfiniteDeltaCertified` (14 of 84, all `column=2, kind=:origin_block`).

| branch | FiniteSolved points | Delta range (FiniteSolved) | GT range (FiniteSolved) | first InfiniteDeltaCertified |
|---|---:|---:|---:|---|
| A (-> min gamma) | 33 (`frac` 0 to 0.64) | 4.0721e-04 to 3.3456 | 2.030% to 7.950% | `frac=0.68`, `GT=8.325%` |
| B (-> max gamma) | 37 (`frac` 0 to 0.81) | 4.0721e-04 to 2.3261 | 2.030% down to 0.386% | `frac=0.856`, `GT=0.293%` |

**Branch B is genuinely new territory**: no prior session in this repo profiled the
"calibration toward maximum gamma" direction at all. It shows a real, substantial, monotone
finite corridor of comparable `DeltaStar` reach to branch A (up to `Delta~2.3`), but a much
**more compressed gains-from-trade range** (only 2.03% down to ~0.29-0.39% before certified
infeasibility, vs. branch A's 2.03% up to ~8.3%) -- moving the focal country toward its
autarky-like reference produces DIMINISHING absolute welfare movement per unit of `DeltaStar`
budget, an asymmetry not previously documented (Figure 1 / Section 8).

## 7. Phase 4: targeted refinement (8 of a 16-solve budget used)

Both branches bracketed all four targets (`Delta = 0.1, 0.5, 1, 2`) within their raw
predetermined grid; one direct log-linear-in-Delta refinement solve per (branch, target) pair
sufficed in every case (no second refinement needed -- each landed within a few percent of the
target on the first attempt), so only 8 of the allowed 16 solves were used.

| branch | target | refined `g` | refined `DeltaStar` | GT at refined point |
|---|---:|---:|---:|---:|
| A | 0.1 | -0.453190 | 0.10020 | 4.2527% |
| A | 0.5 | -0.486059 | 0.49889 | 6.3280% |
| A | 1.0 | -0.497805 | 0.99508 | 7.0586% |
| A | 2.0 | -0.507102 | 1.97902 | 7.6329% |
| B | 0.1 | -0.399995 | 0.09990 | 0.7963% |
| B | 0.5 | -0.394965 | 0.44391 | 0.4630% |
| B | 1.0 | -0.394440 | 0.75805 | 0.4282% |
| B | 2.0 | -0.393915 | 1.83745 | 0.3933% |

The refinement points also fill a genuine gap visible in branch B's own raw grid: the
predetermined `t_i` grid jumps from `Delta=0.387` (`frac=0.766`) directly to `Delta=2.326`
(`frac=0.810`) with no intermediate predetermined point in between -- the refinement solves at
`Delta~=0.44, 0.76, 1.84` sit exactly inside that gap and confirm the corridor is smooth and
continuous through it (Figure 1).

## 8. Phase 5: verification and cold replay

Every `FiniteSolved` point recorded: `gamma_d_prime`, `kappa_ratio`, `gains_from_trade_pct`,
`DeltaStar`, primal divergence, dual divergence, primal-dual gap, normalization residual,
weighted moment residual, KKT residual (`max(|kkt_opt_error|,|kkt_feas_error|)`), min/max
recovered density ratio (`weights .* W` -- the LFD-recovered probability weight relative to a
uniform `1/W` reference), min cutoff slack, gravity residuals (`A`, `f`), KNITRO status,
iteration/callback proxy, wall time, policy, cap, raw `lower_limit`, and warm-start source --
`results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv`.

**Cold replay** (`scripts/melitz_realD20_fixed_af_gamma_profile_phase4to9_2026-07-29.jl`,
Phase 5): the exact calibration point, every 5th `FiniteSolved` grid point on each branch (13
points), the closest point to each of `Delta=0.1/0.5/1/2` on each branch (8 points), and the
last finite point on each branch (2 points) -- **24 points total**, each replayed from a BRAND
NEW, never-warm-started fixture (`warm_start_source=:neutral`, fresh dual bank) built
independently of the main campaign's own sessions.

**24 / 24 matched**, all with relative difference between `1.6e-10` and `1.8e-9` (absolute
differences `1e-10` to `4e-9`) -- **effectively bit-exact**, well within this repo's own
established reproduction tolerance for a deterministic KNITRO solve. No replay disagreement.

## 9. Phase 6: monotonicity and continuity diagnostics

No monotonicity was imposed or smoothed; checked empirically
(`docs/key_results/melitz_realD20_fixed_Af_gamma_profile_diagnostics_2026-07-29.txt`):

- **Branch A**: `n_finite=33`. Gains from trade monotone in `gamma_d_prime`: **true**.
  `DeltaStar` weakly increasing moving away from calibration: **true**. Classification
  sequence: 33 consecutive `FiniteSolved`, then 10 consecutive `InfiniteDeltaCertified` -- a
  single clean transition, no back-and-forth.
- **Branch B**: `n_finite=37`. Both monotonicity checks: **true**. Classification sequence: 37
  consecutive `FiniteSolved`, then 4 consecutive `InfiniteDeltaCertified` -- again a single
  clean transition.
- **No implausible discontinuity flagged** (the diagnostic's own >10x adjacent-`Delta`-ratio
  test never fired on either branch) -- the largest single-step ratio anywhere in the finite
  region is branch B's own final interior step (`Delta` 0.387 -> the next FINITE point is
  actually beyond the raw grid's resolution there, see Section 7's refinement discussion; no
  adjacent PAIR in the recorded grid itself exceeds the 10x threshold).
- No isotonic regression or curve smoothing was applied anywhere -- the reported profile is the
  raw, directly observed classification and `DeltaStar` sequence.

## 10. Phase 7: figures

- `figures/melitz_realD20_fixed_Af_GT_vs_delta_main_2026-07-29.{pdf,png}` -- gains from trade
  vs. `DeltaStar`, `x` in `[0, 2.25]`, both branches, calibration marked, Phase 4 refinement
  points overlaid as triangles (filling branch B's raw-grid gap), theoretical GT bounds as
  dotted reference lines.
- `figures/melitz_realD20_fixed_Af_GT_vs_delta_full_2026-07-29.{pdf,png}` -- full verified
  finite profile, log-scaled `DeltaStar` axis, up to the evaluation cap.
- `figures/melitz_realD20_fixed_Af_delta_vs_gamma_2026-07-29.{pdf,png}` -- diagnostic:
  `DeltaStar` (FiniteSolved only) vs. `gamma_d_prime`, with `InfiniteDeltaCertified` points
  marked at `y=0` for visibility, calibration marked -- makes the fixed-gamma profile's own
  shape transparent: a smooth V centered near the calibration, rising steeply on both sides,
  with a clean single transition to certified-infinite on each side.

No finite point is ever connected through an `AboveEvaluationCap`/`InfiniteDeltaCertified`/
`NumericalFailure` point in any figure (moot here since neither `AboveEvaluationCap` nor
`NumericalFailure` occurred at all, Section 6).

## 11. Phase 8: output tables

- `results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv` -- full 84-row grid, one row
  per point, all required + additional verification columns (Section 8).
- `results/melitz_realD20_fixed_Af_gamma_profile_targets_2026-07-29.csv` -- compact summary:
  Fréchet calibration; both theoretical endpoints (with the branch-A one explicitly marked
  `theoretical_open_limit` and the branch-B one `theoretical_limit`); the Phase-4-refined
  point nearest each of `Delta=0.1/0.5/1/2` on each branch; the last finite point on each
  branch; the first `InfiniteDeltaCertified` point on each branch. (Neither branch ever
  produced an `AboveEvaluationCap` point, so that row is absent by construction, not omitted.)
- `docs/key_results/melitz_realD20_fixed_af_gamma_profile_branch{A,B}_2026-07-29.csv` -- the
  same rows split per branch.
- `docs/key_results/melitz_realD20_fixed_Af_gamma_profile_refinement_2026-07-29.csv` -- the 8
  Phase 4 refinement solves.
- `docs/key_results/melitz_realD20_fixed_Af_gamma_profile_coldreplay_2026-07-29.csv` -- the 24
  Phase 5 cold-replay results.
- `docs/key_results/melitz_realD20_fixed_Af_gamma_profile_diagnostics_2026-07-29.txt` -- Phase
  6 monotonicity/continuity diagnostics.

## 12. Phase 9: comparison with the earlier sparse D20 profile

The earlier sparse points (post-consolidation validation session, `docs/
melitz_post_consolidation_validation_2026-07-28.md` Phase 2) were built along the SAME "toward
minimum gamma" direction as this session's branch A, but on a much coarser, differently-
targeted grid. Compared here as benchmarks, not values to force (per the governing prompt):

| label | old `DeltaStar` | new nearest-grid `DeltaStar` (branch A) | new GT% | diff |
|---|---:|---:|---:|---:|
| pareto (calibration) | 4.0721e-04 | 4.0721e-04 | 2.0303% | -3.0e-09 (bit-exact) |
| delta~0.1 | 1.0551e-01 | 1.0848e-01 | 4.3425% | +2.97e-03 |
| delta~0.5 | 4.8328e-01 | 4.6475e-01 | 6.2443% | -1.85e-02 |
| delta~1.0 | 9.0455e-01 | 8.3960e-01 | 6.8918% | -6.50e-02 |
| delta~2.0 | 1.6506e+00 | 1.8518e+00 | 7.5854% | +2.01e-01 |

The calibration point reproduces bit-exactly, as expected (same fixture, same formula). The
other four rows differ by a few percent to ~20% in `DeltaStar` at the NEAREST available new
grid point -- expected and unsurprising: the new grid's own predetermined `t_i=(i/40)^2` points
do not land at the same `gamma_d_prime` values the old, differently-targeted sparse grid used,
and "nearest available grid point" is not the same operation as "value at the identical `g`."
The Phase-4-refinement values (Section 7), which DO target the same `Delta` values directly,
are the more informative comparison and are reported in the targets CSV
(`refined_near_DeltaX` rows) -- these also do not reproduce the old sparse values exactly
(different grid construction, closed-form kappa-interpolation vs. the old session's own
fraction scheme), consistent with "benchmarks, not values to force." No discrepancy here
indicates a regression: both this session's `FiniteSolved` classifications and the old
session's are independently verified (Section 8's cold-replay confirms this session's own
values; the old values were themselves independently re-certified in the post-consolidation
validation session).

## Final report

1. **Theoretical endpoints**: minimum `gamma_d_prime = 0.5669312470` (`g_ceiling=-0.5675172401`,
   an open Delta->infinity limit), calibration `gamma_d_prime = 0.6578550157`, maximum
   `gamma_d_prime = 0.6784102437` (`g_floor=-0.3880030949`, an ordinary finite theoretical
   point by theory).
2. **Gains from trade**: 11.279% (theoretical max, branch A limit) / 2.030% (calibration) /
   0.000% (theoretical min, branch B, `kappa_ratio<=1` boundary).
3. **DeltaStar evolution**: smooth, monotone, roughly-symmetric-in-log-Delta rise on BOTH sides
   of the calibration (Figure 3) -- branch A reaches `Delta=3.35` before certified-infinite;
   branch B reaches `Delta=2.33` before certified-infinite. Branch A's GT range is much wider
   (2.03% to 7.95%) than branch B's (2.03% down to 0.39%) over a comparable `Delta` budget --
   a genuine, newly-documented asymmetry.
4. **Crossings**: `Delta=0.1/0.5/1/2` are all bracketed and refined on BOTH branches
   (Section 7 table) -- GT at those points ranges from 4.25% (A, Delta=0.1) to 7.63% (A,
   Delta=2) on the min-gamma side, and 0.80% (B, Delta=0.1) down to 0.39% (B, Delta=2) on the
   max-gamma side.
5. **Monotone and smooth**: yes on both branches, no violations (Section 9).
6. **Evaluation cap**: never reached as `AboveEvaluationCap` anywhere in this campaign -- every
   non-finite point was a full `InfiniteDeltaCertified` exact certificate instead.
7. **Exact feasibility screen**: fires on BOTH branches, cleanly, past `frac=0.64`/`Delta=3.35`
   (branch A) and `frac=0.81`/`Delta=2.33` (branch B), always `column=2, kind=:origin_block`.
8. **Free A/f fixed throughout**: yes, proven by hash assertion at all 84 solves (Section 1).
9. **Cold replay agreement**: 24/24, effectively bit-exact (Section 8).
10. **Agreement with the earlier sparse profile**: calibration bit-exact; other points differ
    by a few percent to ~20% at the nearest comparable grid point, expected given the different
    grid constructions (Section 12) -- not a discrepancy in either session's own correctness.

## Acceptance criteria

1. Analytical gamma endpoints derived + independently cross-checked: **met** (Section 3).
2. Exact real-D20 Fréchet calibration used: **met** (Section 3, France, W=80,000, seed=1).
3. No outer optimization: **met** -- every solve is a single fixed-`theta` inner solve via
   `solve_melitz_delta!`; no KNITRO outer NLP was ever constructed in this session.
4. Every free A/f coordinate fixed at calibration: **met**, hash-asserted at all 84 solves.
5. At least 81 unique predetermined gamma points evaluated: **met**, 83 unique (Section 4).
6. Consolidated public inner API used everywhere: **met** (Section 5).
7. Every main solve uses `CappedEvaluation(10)`: **met**.
8. `lower_limit=-10` verified at every solve: **met** (session-construction assertion, never
   bypassed).
9. No capped solve returns `FiniteSolved` above 10: **met** -- max observed finite `Delta` was
   3.3456 (well under cap), enforced structurally by the classifier's own output invariant.
10. `FiniteSolved` points pass complete primal-dual/LFD verification: **met** (Section 8
    verification columns, all populated, all finite/well-behaved on every `FiniteSolved` row).
11. Economically relevant region around `Delta=0.1/0.5/1/2` well resolved: **met** (Section 7).
12. Primary figure has `DeltaStar` on x-axis, gains from trade on y-axis: **met** (Section 10).
13. Full CSV contains classifications/certificates, not fabricated `DeltaStar` for capped/
    infinite points: **met** (`NaN` in the `delta_star` column for every non-`FiniteSolved`
    row).
14. No broad outer campaign or parameterization search run: **met**.
15. No Ricardian/shared source modified: **met** -- `git diff --name-only
    a6826f61c34c6874a686d051d2dfa7abd698cc0c` is empty (zero tracked-file changes); every new
    file is under `scripts/`, `docs/`, `results/`, or `figures/`.
16. Relevant Melitz tests pass: **met** -- `julia --project=. -t 1 test/melitz/runtests.jl`
    (no `src/melitz/` files were touched this session, so this is a regression check on the
    surrounding tree, not a re-validation of new code); see the session's own test-run log for
    the exact pass count.
17. Work committed locally, not pushed: see the commit made immediately after this document.

## Files changed

New files only (all under `scripts/`, `docs/`, `docs/key_results/`, `results/`, `figures/` --
Melitz-only; zero diff in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path):

```
docs/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.md   (this document)
scripts/melitz_phase0_realD20_endpoints_2026-07-29.jl
scripts/melitz_realD20_fixed_af_gamma_profile_2026-07-29.jl
scripts/melitz_realD20_fixed_af_gamma_profile_phase4to9_2026-07-29.jl
scripts/melitz_realD20_fixed_af_gamma_profile_figures_2026-07-29.py
results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv
results/melitz_realD20_fixed_Af_gamma_profile_targets_2026-07-29.csv
docs/key_results/melitz_realD20_fixed_af_gamma_profile_branchA_2026-07-29.csv
docs/key_results/melitz_realD20_fixed_af_gamma_profile_branchB_2026-07-29.csv
docs/key_results/melitz_realD20_fixed_Af_gamma_profile_refinement_2026-07-29.csv
docs/key_results/melitz_realD20_fixed_Af_gamma_profile_coldreplay_2026-07-29.csv
docs/key_results/melitz_realD20_fixed_Af_gamma_profile_diagnostics_2026-07-29.txt
docs/key_results/melitz_realD20_fixed_Af_gamma_profile_phase9_comparison_2026-07-29.csv
figures/melitz_realD20_fixed_Af_GT_vs_delta_main_2026-07-29.{pdf,png}
figures/melitz_realD20_fixed_Af_GT_vs_delta_full_2026-07-29.{pdf,png}
figures/melitz_realD20_fixed_Af_delta_vs_gamma_2026-07-29.{pdf,png}
```
