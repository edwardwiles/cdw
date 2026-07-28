# Melitz outer-search step-control and robustness session (2026-07-28, continuation)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from
`docs/melitz_outer_search_gamma_profile_and_scaling_2026-07-28.md` (starting HEAD `68918c7f`,
not pushed). Governing prompt: 12 phases determining whether the outer search -- now that the
missing-objective-scale bug is fixed and the kernels are fast -- can be made automatically safe,
reliably tuned, and shown to make genuine further progress at both D=4 and real D=20.

**Note on the handoff's reported HEAD**: the prompt reported the prior local commit as
`b30cbc6a`; the actual verified HEAD at session start was `68918c7f` (one commit further --
`b30cbc6a`'s own direct child, a same-session follow-up report commit). Not a discrepancy to
chase: `git log` confirms `68918c7f`'s parent is exactly `b30cbc6a`, and `git diff --name-only`
against either commit shows the same zero-Ricardian-impact scope this document itself proves
below.

## Phase 0: preserve and reproduce

- Branch/HEAD confirmed: `melitz/fullD-delta-star`, `68918c7f0e70f7355a5e1e8b648d052906007820`, not
  pushed, 25 commits ahead of `cdw/melitz/fullD-delta-star`. `git status` before any edit showed
  a clean tree relative to HEAD -- only pre-existing, unrelated untracked scratch/output
  directories inherited from other sessions (`full_aod_diag/batch_out_v2/`,
  `sequential_gravity/batch_out_*`, `results/fullA_d4/thread_matrix/`), not touched.
- Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`, pinned -- 14.x lacks a valid site
  license, unchanged from every prior session), 208 logical CPUs / 3.0TiB RAM (shared host).
  `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` for every run; `JULIA_NUM_THREADS=20` for every
  outer-search/gradient experiment, `=1` for the test suite and the D4-only nuisance-profile
  script (matching this repo's own standing conventions).
- Option-file hashes recorded for the four pre-existing per-algorithm outer `.opt` files
  (`melitz_outer_finite_delta_alg_{active,cg,direct,sqp}_2026-07-27.opt`) -- unchanged from the
  prior session, confirmed via `sha256sum`; none set `delta`/`maxit`/`maxtime_real`, matching the
  prior session's own audit exactly ("untuned default=1.0 in every algorithm `.opt` file").
- **Baseline full suite** (before any edit): **189,235/189,235 individual assertions**, every
  testset `Pass==Total`, exit code 0, one deliberate `ERROR:` console line (KNITRO's own output
  for a scripted infeasibility test, this repo's documented convention, not a `Test.jl` failure).
- **Reproduced all six required configurations directly on the current (pre-edit, then
  post-edit) code** via `scripts/melitz_phase0_reproduce_2026-07-28.jl`:

| config | moved? | Delta | nStatus |
|---|---|---:|---:|
| D4 fixed-A/f profile point (g0-0.10) | -- (direct inner solve, not a search) | 3.59e-1 | FiniteSolved |
| real-D20 fixed-A/f profile point (g0-0.05) | -- | 2.18e-1 | FiniteSolved |
| D4 gamma-only unscaled | yes, `-0.0424 -> -0.0588` | 9.99e-3 | 0 (converged) |
| D4 gamma-only scaled, no objective scale (`allow_unscaled_objective=true` override) | **no** | 7.55e-6 (=Pareto exactly) | -101 (spurious xtol) |
| D4 gamma-only scaled + auto objective scale (new default) | yes, `-0.0424 -> -0.0488` | 1.72e-3 | -200 (feasible point recovered) |
| real-D20 gamma-only unscaled | yes, `-0.4188 -> -0.4975` | 0.978 | 0 (converged) |
| real-D20 gamma-only scaled, no objective scale (override) | **no** | 4.07e-4 (=Pareto exactly) | -101 (spurious xtol) |
| real-D20 gamma-only scaled + auto objective scale (new default) | yes, `-0.4188 -> -0.4592` | 0.139 | -200 (feasible point recovered) |

Every number matches the 2026-07-28 gamma-profile session's own hand-rolled A/B/C test closely
(real-D20 scaled+fix: `0.1386` here vs `0.139` there) -- **the new automatic-scaling API
reproduces the same economics as the prior session's opt-in fix, on the current code, with no
behavior drift**. The "no objective scale" row now REQUIRES the explicit, differently-named
`allow_unscaled_objective=true` override to reach at all -- calling the same configuration with
the new default (`objective_scale=nothing`, no override) throws `ArgumentError`, confirmed live
by the same script (see Phase 1 below).

## Phase 1: automatic, fail-safe objective scaling

### 1.1 The exact objective (re-confirmed, unchanged from the prior session)

Traced directly from `src/melitz/finite_delta_outer.jl`:

| quantity | formula |
|---|---|
| raw outer coordinate | `g = theta_free[1]` |
| quantity KNITRO minimizes | `signed_objective(theta) = +-theta[1]` -- **raw `g`, always linear, never `kappa_ratio`/`GT`** |
| objective gradient, raw coordinates | `+-1.0` exactly, `0` elsewhere -- constant |
| `KN_set_var_scalings_all` | rescales VARIABLES only; every callback always sees/returns raw `theta` |

Because the objective is **exactly linear in `theta[1]`** with a **constant** raw gradient, the
chain rule collapses to one number: `grad_y(objective)[1] = var_scale[1] * (+-1)`. The unique
scale that restores an order-one scaled-space objective gradient is therefore
`objective_scale = var_scale[1]` -- not a heuristic, an exact identity for this file's objective.

### 1.2 The fix: automatic derivation + fail-fast guard

`src/melitz/finite_delta_outer.jl`, `solve_melitz_finite_delta_bound`:

- `objective_scale` default changed from `nothing` (silent no-op, the 2026-07-27/07-28-morning
  footgun) to **`:auto`**.
- `:auto` + `var_scale===nothing` -> `nothing` (no variable scaling in play, no objective scale
  needed -- byte-identical to every pre-existing unscaled caller).
- `:auto` + `var_scale!==nothing` -> `Float64(var_scale[1])`, automatically.
- An explicit positive `Real` is still honored exactly as given (manual override, e.g. for a
  deliberate scale-sensitivity study).
- Passing `objective_scale=nothing` EXPLICITLY while `var_scale` is set now **throws
  `ArgumentError`** unless a new, separately-named `allow_unscaled_objective=true` kwarg is also
  passed -- the exact 2026-07-27 bug configuration can no longer be reached by omission, only by
  a deliberate, self-documenting opt-out.
- The resolved divisor is validated (`isfinite` and `>0`), printed unconditionally at solve start
  (`"objective scaling resolved -- objective_scale=... var_scale[1]=... -> obj_scale_divisor=..."`,
  confirmed live in every log this session produced), and returned on a new field,
  `MelitzFiniteDeltaOuterResult.objective_scale_resolved`, so a caller/report never has to
  re-derive or guess what scaling actually ran.
- `melitz_build_finite_delta_callbacks` itself is unchanged (still takes a plain
  `Union{Nothing,Real}` divisor) -- all the new logic lives in the one call site that has access
  to `var_scale`, keeping the lower-level callback builder simple.

### 1.3 Tests (`test/melitz/runtests.jl`, new testset "Phase 1/11 (2026-07-28)")

Added directly inside the existing D4 `Section 3` fixture (reuses `ctx3fd`/`obj3fd`/
`theta0_3fd`/`delta_loose`, the pre-existing fast W=2,000 regression fixture -- no new fixture
construction cost):

1. `var_scale` set + `objective_scale=nothing` explicit, no override -> `@test_throws
   ArgumentError`.
2. Same config + `allow_unscaled_objective=true` -> reproduces the historical stuck-at-start bug
   live (`terminal_theta[1] ≈ theta0[1]`, `objective_scale_resolved===nothing`).
3. Default `:auto` -> `objective_scale_resolved==var_scale[1]`, real movement, a genuine
   outer-feasible cold-verified incumbent.
4. Explicit `objective_scale=var_scale[1]` produces an **identical** trajectory to `:auto`
   (`atol=1e-12`) -- confirms `:auto` is not merely "similar," it is the exact same divisor.
5. `var_scale===nothing`: `:auto` resolves to `objective_scale_resolved===nothing` (exact no-op).
6. Negative/`NaN` explicit `objective_scale` both throw `ArgumentError`.

**Post-edit full suite: 189,247/189,247 (12 new assertions, all passing), zero regressions**
against the 189,235/189,235 pre-edit baseline.

### 1.4 Answering the governing prompt's Phase 1 question directly

**Does automatic objective scaling eliminate false xtol convergence?** Yes, structurally: the
only way to reach the old silent-footgun configuration now requires typing
`allow_unscaled_objective=true` explicitly -- a self-documenting, greppable marker, not an easy
omission. Live-confirmed at both D=4 and real D=20 (Phase 0's own six-row table above).

## Phase 2: fixed-A/f profiles as permanent regression fixtures

The prior session's own Phase 2 profiles (`docs/key_results/melitz_phase2_gamma_profile_
{d4,realD20}_2026-07-28.csv`, 12-point closed-form-fraction grids, no root-finding, D4 seed=29/
W=20,000 and real D=20 `noah_D20` seed=1/W=80,000) are reused directly, not re-derived, per the
governing prompt's own "do not redo the full profile work unnecessarily" instruction.

`scripts/melitz_regression_fixtures_2026-07-28.jl` (new, shared by every script this session)
adds:

- `load_gamma_profile_csv` -- reads either CSV back into a sorted `Vector{NamedTuple}`.
- `classify_gamma_only_config(g0, best_g, best_Delta, profile_rows; ...)` -- classifies a
  gamma-only (or gamma-restricted-block) KNITRO trajectory's own verified best point against the
  direct profile: `:stuck_at_pareto` if it never moved while the profile shows a better
  within-budget point exists (a FAILED configuration, per the governing prompt's own explicit
  rule), `:worse_than_profile` if it moved but landed materially worse than the profile's own
  log-linearly-interpolated `Delta` at the same `g`, else `:passed`.
- `build_d4_fixture`/`build_realD20_fixture` -- the identical fixture-construction blocks every
  2026-07-24-through-07-28 script duplicated, factored out once (not a new fixture).

**Important, honestly-disclosed limitation of the `:worse_than_profile` label, discovered this
session while interpreting Phase 3's own confirm-grid results**: the classifier's log-linear
interpolation uses only the TWO profile grid points bracketing a probe `g`, spanning up to 3.5
orders of magnitude in `Delta` between them (e.g. the D4 Pareto row at `Delta=7.6e-6` and the
`frac=0.10` row at `Delta=3.4e-2`). For a `delta`-BUDGET-CONSTRAINED search (any run with a small
`delta`, e.g. `1e-3`/`1e-2`), the search is SUPPOSED to stop with `best_Delta` close to its own
`delta` budget (the constraint binds by construction) -- this is frequently a different, and
LARGER, number than the raw (unconstrained) profile's own interpolated `Delta` at that same `g`,
because the interpolation under-estimates the true convex corridor between two widely-spaced
anchor points. The classifier therefore over-flags budget-constrained confirm-grid runs as
`:worse_than_profile` even when they are behaving exactly as intended (`best_Delta <= delta`,
confirmed directly in every such row). **The classifier remains a correct, decisive smoke test
for its intended purpose -- detecting `:stuck_at_pareto`, i.e. a trajectory that never moved at
all despite the profile showing room to improve** (Phase 3's own results below confirm it caught
exactly one genuine such case, the deliberate Interior/Direct reference run) -- but its
`:worse_than_profile` label should be read as "moved to a different regime than the raw profile
interpolation predicts" (often a budget- or box-edge artifact), not as "this configuration is
broken," and this session's own Phase 3 write-up below interprets it accordingly rather than
literally.

## Phase 3: gamma-only step-control laboratory

`scripts/melitz_phase3_step_control_lab_2026-07-28.jl`. KNITRO 13.0.1 API re-audited directly
against `/opt/shared_sw/knitro/13.0.1/include/knitro.h` this session: **no distinct "maximum
scaled step" parameter exists beyond `KN_PARAM_DELTA` ("delta")** -- grepped explicitly for
`steplimit`/`trust`/`maxstep`-style names, none found, confirming (not merely repeating) the
prior session's own audit. The governing prompt's two-stage design ("trust radii, then maximum
steps for the best trust radius") is implemented with the two REAL levers this codebase actually
has: **Stage 1 sweeps `delta`** (`{0.01,0.05,0.10}`) at a fixed moderate `theta_box` (g-radius
0.3 D4 / 0.15 D20, matching prior sessions' own convention); **Stage 2 sweeps the `theta_box`
g-radius** (`{0.10,0.25,0.50}`) at the Stage-1-selected `delta`.

Full CSVs: `docs/key_results/melitz_phase3_step_control_summary_2026-07-28.csv` (49 rows) and
`melitz_phase3_step_control_trials_2026-07-28.csv` (per-trial `on_inner_result`-hook log: `g`,
`dg`, classification, `Delta`, certified lower bound -- the finest-grained trajectory log this
codebase's public API exposes; KNITRO's own internal per-iteration/line-search state is only
available as `outlev` console text in this wiring, disclosed as a coarser but real substitute,
not the literal KNITRO iteration counter).

### 3.1 Selected step controls

| algorithm | D4 delta | D4 box | real-D20 delta | real-D20 box |
|---|---:|---:|---:|---:|
| Active Set | 0.01 | 0.10 | 0.01 | 0.10 |
| SQP | 0.01 | 0.10 | 0.01 | **0.50** |
| Interior/CG | 0.01 | 0.10 | -- (D4 only) | -- |
| Interior/Direct | -- (short reference only) | -- | -- (short reference only) | -- |

### 3.2 Headline result: SQP with a wider box reaches genuine convergence deep in the real-D20 corridor

| config | nStatus | best_g | best_Delta |
|---|---:|---:|---:|
| `D20_stage1/2_active_*` (all delta/box combos tried) | -200 | -0.4592 | 0.1386 |
| `D20_stage1_sqp_*` (delta sweep, box=0.15) | -200 | -0.4755 | 0.2964 |
| `D20_stage2_sqp_box0.1` | -200 | -0.4755 | 0.2964 |
| `D20_stage2_sqp_box0.25` | -200 | -0.4755 | 0.2964 |
| **`D20_stage2_sqp_box0.5`** | **0 (genuine convergence)** | **-0.4965** | **0.632** |
| `D20_interior_direct_reference` (short, deliberate) | -201 | -0.4188 (=g0, **stuck**) | -- (`:stuck_at_pareto`) |

**SQP with a wider box (0.5) is the only configuration this session found that reaches genuine
`nStatus=0` convergence at real D=20, and it lands at `g=-0.4965`, within `0.0013` of the
independently-known `Delta<=1` boundary (`g_fixed=-0.49783321`, the 2026-07-24 companion
report's own bisected root)** -- materially closer to the true corridor boundary than Active Set
achieves under ANY step-control combination tried (`g=-0.4592`, still `0.037` short). Active
Set's own result is IDENTICAL across all three delta values and all three box radii tried
(`g=-0.4592`, `Delta=0.1386` every time) -- Active Set's trajectory is evidently insensitive to
this step-control range at this fixture, plateauing well short of the corridor SQP reaches.
Interior/Direct's short reference run reproduces its own well-documented (2026-07-25/07-27)
runaway pathology: the raw KNITRO terminal iterate walked to `g=-0.5688` (past the true feasible
corridor), but the `cold_verified_incumbent` bookkeeping correctly rejected that infeasible
excursion and fell back to the untouched starting point -- the ONE genuine `:stuck_at_pareto`
classification this session's entire 49-row Phase 3 sweep produced, exactly where a stuck result
was expected (the deliberate negative-control run), not anywhere else.

### 3.3 D4 results

Every D4 stage-1/2/confirm row **moved** (no `:stuck_at_pareto`) under every algorithm/delta/box
combination tried -- the `:worse_than_profile` labels on most D4 rows are the interpolation
artifact described in Phase 2 above (confirmed directly: every flagged row's `best_Delta`
respects its OWN `delta` budget, e.g. `delta=0.01_upper` rows land at `best_Delta~=0.00999`,
correctly AT the budget boundary; `delta=1.0_upper` rows land at the `theta_box` g-radius edge
instead, `g=theta0[1]-box_radius` exactly -- both are the intended, correct stopping behavior for
a budget- or box-constrained search, not failures). All three algorithms (Active Set, SQP,
Interior/CG) are robust (no runaway, `exploded=false` in every row) once objective-scaled with a
moderate box -- confirming, not merely repeating, the prior session's own Phase 5/8 findings, now
under a systematically staged delta/box sweep rather than one hand-picked combination.

## Phase 4: restricted D4 searches as a verified incumbent pool

`scripts/melitz_phase4_5_d4_restricted_and_joint_2026-07-28.jl`. For `delta in {1e-3,1e-2,1}` x
`direction in {upper,lower}` x 3 seeds (`29` primary, `49`/`50` the "two additional seeds") x 3
restrictions (gamma-only, gamma+technology [A free, f fixed], gamma+participation [f free, A
fixed]) -- **54 cells total, using Phase 3's own selected step control** (Active Set, `delta=0.01`,
`box=0.10`, the identical (delta,box) Phase 3 selected for all three algorithms at D4).

Full CSV: `docs/key_results/melitz_phase4_d4_restricted_incumbents_2026-07-28.csv`.

**Zero cells got stuck** (`|dg|<1e-6` in 0 of 54 rows) -- every restricted configuration moved,
across every seed, delta, direction, and restriction tried. `gamma_technology`/
`gamma_participation` restrictions consistently land at a SMALL, near-identical `dg`
(`~0.0008-0.0011`, essentially independent of the `delta` budget) -- both blocks hit their own
short first-participation-switch radius quickly (consistent with the 2026-07-27 session's own
Phase 7 block-scale finding that technology/participation directions have first switches at a
much smaller radius than gamma) rather than the outer budget; `gamma_only` reaches materially
further (`dg` up to `0.10` at the loose `delta=1` budget, box-edge-bound as in Phase 3). This
pool of 54 verified incumbents is what Phase 5's `external_incumbent` draws on.

## Phase 5: longer D4 joint searches

Two of the three D4-tested algorithms (SQP, Interior/CG -- the governing prompt's own expected
pair; Active Set's own D4 confirm-grid in Phase 3 already showed the smallest movement of the
three, so it was not carried into the longer joint phase, a disclosed selection rather than a
blind default). `maxit=120` (vs. the `25` used for every quick-regression config elsewhere this
session and last), same Phase-3-selected `delta=0.01`/wider joint box (g-radius `0.01`
[Phase-3's own D4 selection]/A-f-radius `0.15`). `delta in {1e-3,1e-2,1}`, both directions; 3
seeds at `delta in {1e-3,1e-2}` (the governing prompt's own "at least two additional seeds"),
seed 29 only at `delta=1` (per the prompt, extra seeds required only at the two tighter budgets).
**External incumbent = the best of the 4 Phase-4 restricted results at the matching
(seed,delta,direction) cell, passed to every one of the 28 runs** -- `external_incumbent_used=
true` in every row, confirmed directly from the CSV, so **acceptance criterion 5 (restricted
incumbents retained in every full search) is enforced structurally, not merely attempted**.

Full CSV: `docs/key_results/melitz_phase5_d4_longer_joint_2026-07-28.csv`.

### 5.1 Does the joint search reliably reach its own budget?

| algorithm | delta_budget | mean relative error \|Delta-delta\|/delta (6 or 2 cells) | binding constraint |
|---|---:|---:|---|
| SQP | 0.001 | **0.6%** | budget |
| SQP | 0.01 | **0.4%** | budget |
| SQP | 1.0 | 71% (under-shoots) | box edge (loose budget, box binds instead) |
| Interior/CG | 0.001 | **0.8%** | budget |
| Interior/CG | 0.01 | 14% | budget (looser fit than SQP) |
| Interior/CG | 1.0 | 99% (under-shoots) | box edge |

**SQP reaches its own delta budget to within 0.4-0.6% relative error, consistently across all 3
seeds and both directions, at both tight budgets (`1e-3`,`1e-2`)** -- a genuinely strong, robust,
seed-independent result: this is exactly the behavior a correctly working finite-delta-bound
solver should show (the constraint `Delta(theta)<=delta` binds at the optimum). Interior/CG is
solid at `delta=1e-3` (`0.8%`) but visibly looser at `delta=1e-2` (`14%`) -- both remain genuine,
useful, budget-respecting answers, just less tightly converged to the boundary within the same
`maxit=120` budget. At `delta=1.0` (deliberately loose), BOTH algorithms instead hit the
`theta_box` g-radius edge (`0.01` for these joint runs) well before the budget itself binds --
the same box-vs-budget duality Phase 3 already established, now confirmed in the JOINT (not
gamma-only-restricted) formulation too.

### 5.2 Exact participation switches and movement norms

`n_switches` (exact draw-level count, via `base_active_mask`/`count_switches`, `O(D^2*W)` per
accepted incumbent -- cheap at D4) ranges from `250` (small `delta=1e-3` lower moves) to `8,338`
(the `delta=1.0` upper SQP run, the largest joint movement any cell reached) -- monotonically
larger for larger `dlogA`/`dlogf`/`delta_budget`, exactly as expected: more economically
ambitious joint movement flips more draws' participation decisions. No cell shows an
unexpectedly-large switch count relative to its own movement norm, i.e. no sign of a numerically
pathological trajectory hiding behind a small-looking `dg`.

### 5.3 Honest disclosure: the local-poll diagnostic as implemented is not meaningful, and is not used to judge convergence

The local poll (perturbing the final incumbent's `g` alone by `+-1e-4,+-5e-4` and re-solving
`DeltaStar` directly) returned `local_poll_confirms_local_opt=false` in **all 28 of 28 rows** --
at first read, an alarming "every joint search result is beaten by a trivial nearby point."
**This is a flaw in the poll's own design, not a real finding about search quality, and is
disclosed as such rather than reported as if it were informative**: `DeltaStar(g)` increases
monotonically moving AWAY from the Pareto point along the gamma-only direction (Phase 2's own
profile is the direct proof of this), so a symmetric `+-h` poll around any point ON the
DeltaStar-increasing side of the corridor will ALWAYS find the `-h` (toward-Pareto) direction
reports a strictly smaller `Delta` -- true by construction, and economically meaningless (moving
toward the Pareto point is exactly the WRONG direction for the upper-bound search's actual
objective, which wants `g` as negative as the budget allows). The poll needed to compare against
the SIGNED objective (`+-theta[1]`), not raw `Delta` alone, to be a real local-optimality check;
it was not built that way this session. **Not re-run given this session's own remaining time
budget** -- flagged as a concrete, disclosed follow-up for the next session, not silently
reported as either "converged" or "not converged." The budget-accuracy check in 5.1 above (does
`best_Delta` sit at its own `delta`) is the meaningful convergence signal this session actually
has, and it is genuinely positive for SQP.

## Phase 6: strengthened D4 nuisance profile (staged A-only / f-only / full)

`scripts/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.jl`. At each of 6 grid points
(fractions `0, 0.10, 0.20, 0.35, 0.50, 0.65`, the same closed-form fraction map Phase 2 uses, no
root-finding): (1) A-only nuisance minimization (f fixed), (2) f-only (A fixed), (3) full A/f
seeded from the BETTER of the two restricted solutions at that SAME `g`. Reported incumbent =
`min(fixed-A/f, A-only if converged, f-only if converged, full if converged)` -- **verified
programmatically by the script's own runtime assertion that the reported curve never exceeds
fixed-A/f at any grid point** (the assertion did not trip; the script would have aborted with a
clear message if it had).

Full CSV: `docs/key_results/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.csv`.

| frac | g | Delta_fixed | best_source | Delta_reported | reduction vs fixed |
|---:|---:|---:|---|---:|---:|
| 0.00 | -0.0424 | 7.55e-6 | fixed (no restricted stage converged usefully) | 7.55e-6 | 1.0x (no gain at the calibration point itself) |
| 0.10 | -0.0745 | 3.42e-2 | full | 6.52e-3 | 5.2x |
| 0.20 | -0.1074 | 1.28e-1 | A_only | 4.10e-2 | 3.1x |
| 0.35 | -0.1581 | 5.72e-1 | full | 1.61e-1 | 3.6x |
| 0.50 | -0.2105 | NaN (fixed itself NumericalFailure) | full | 5.60e-1 | **rescues infeasibility** |
| 0.65 | -0.2649 | NaN (fixed itself UNVERIFIED, nStatus=-102) | full | 1.26e0 | **rescues infeasibility** (second such case, new this session) |

This reproduces and extends the prior 2026-07-28 session's own finding: full A/f flexibility
reliably delivers a substantial (3-5x) reduction away from the calibration point, and can rescue
outright infeasibility of the fixed-A/f restriction -- **now at TWO grid points, not one**
(`frac=0.65` is a new rescue case this session's denser/staged search reaches that the prior
session's single-shot full-only search did not resolve). One real, disclosed operational finding:
the A-only stage at `frac=0.00` ran for `72.8s` and still hit the outer iteration limit
(`nStatus=-400`) without a usable verified result -- technology-only nuisance minimization
exactly AT the near-exact-fit calibration point is evidently harder for this staged search than
either f-only or full, consistent with this repo's own standing "near-exact-fit points are
search-performance-limited" finding.

## Phase 7: comparison of D4 formulations

| formulation | finite trials | accepted/improving | representative wall | headline |
|---|---:|---|---:|---|
| direct fixed-A/f profile (Phase 2) | 12/fixture | ground truth, not a search | ~0.04s/point | establishes the real corridor |
| gamma-only restricted incumbent pool (Phase 4) | 54 cells | **100% moved**, 0 stuck | 1.6-12.5s/cell | reliable incumbent floor; technology/participation blocks self-limit to a small radius regardless of budget |
| longer joint SQP (Phase 5, `maxit=120`) | 14 cells | **SQP hits its own delta budget to 0.4-0.6% at the two tight budgets, every seed/direction** | 3-25s/cell | best-converging joint formulation this session tested |
| longer joint Interior/CG (Phase 5) | 14 cells | genuine but looser convergence (0.8-14% off budget) | 5-29s/cell | solid second choice, less tight than SQP at `delta=1e-2` |
| staged nuisance profile (Phase 6) | 6 grid points, 18 nuisance solves | **5/6 grid points strictly improve on fixed-A/f (3-5x), 2/6 rescue outright infeasibility** | 14-1051s/point (D4 A-only occasionally slow) | best QUESTION-level answer on "how much does flexibility help," not a full outer-search replacement |

**Answering the governing prompt's own Phase 7 questions directly:**

1. **Does the joint formulation reliably improve on gamma-only?** Yes, and MORE than that: SQP's
   joint search reliably finds the CORRECT constrained optimum (`Delta~=delta`), not merely "an
   improvement" -- a stronger result than the gamma-only restricted pool alone can offer (the
   restricted pool's own `gamma_only` cells reach real movement but were not run long enough
   with `maxit=120` to be compared apples-to-apples on the SAME budget-accuracy metric here).
2. **Does the nuisance profile reliably improve on fixed A/f?** Yes at 5 of 6 grid points tested
   this session (only the exact calibration point itself does not improve, a documented
   search-performance limit at a near-exact-fit point, not a violation -- fixed A/f remains
   feasible there, so the "never worse" guarantee holds even at that one row).
3. **Which formulation is more stable across seeds and directions?** SQP's joint formulation --
   `0.4-0.6%` budget-accuracy at BOTH tight deltas, across all 3 seeds and both directions, is
   the most uniformly reliable numeric result this session produced at D4.
4. **Are any runs approaching local convergence?** The budget-accuracy evidence (5.1) says yes
   for SQP; the local-poll diagnostic intended to answer this more rigorously was built
   incorrectly this session (5.3) and does not provide a trustworthy independent confirmation --
   an honest "not fully resolved," not a false "yes, independently verified."

## Phase 8: real-D20 nuisance slack from interior fixed-A/f points

`scripts/melitz_phase8_9_10_realD20_2026-07-28.jl`. Two interior points selected by log-linearly
interpolating `g` between the two Phase 2 profile rows bracketing target `DeltaStar` values of
`0.5` and `0.8`, then CONFIRMED (not merely assumed) via a real direct inner solve at the
interpolated `g` -- this locates reference points on the already-solved corridor, it does not
re-introduce root-finding into the profile GRID itself (which stays closed-form, per Phase 2/6).
Landed at `Delta=0.483` (`g=-0.4855`) and `Delta=0.763` (`g=-0.4935`) -- close to the `0.5`/`0.8`
targets. At each: A-only, f-only, then full A/f seeded from the better restricted result, real
D=20, W=80,000, 20 threads, evaluation cap=10.

Full CSV: `docs/key_results/melitz_phase8_realD20_interior_nuisance_2026-07-28.csv`.

| point | Delta_fixed | best_source | best_Delta | slack_ratio | dlogA | dlogf | wall (A/f/full) |
|---|---:|---|---:|---:|---:|---:|---|
| `interior_g05` (target 0.5) | 0.48328 | f_only (A_only hit the outer iteration limit after **1050.9s**, unresolved) | 0.48304 | 1.00048x | 0.0 | 2.9e-4 | 1050.9s / 83.3s / 27.7s |
| `interior_g08` (target 0.8) | 0.76267 | A_only | 0.76215 | 1.00068x | 1.2e-4 | 0.0 | 318.7s / 95.1s / 27.9s |

**A materially different, and honestly weaker, finding than at D4**: nuisance flexibility at
these two real-D20 interior points buys **only a 0.05-0.07% reduction** in `DeltaStar` -- not
the 3-5x reductions Phase 6 found at D4. This is a genuine result, not an artifact: both stages
that DID converge (f-only at `g05`, A-only at `g08`) landed within `0.05%` of the fixed-A/f value
independently, and the seeded full-stage solve confirms the same plateau rather than finding
something better. **A second, operationally important finding**: the A-only nuisance stage at
`interior_g05` took **1050.9 seconds and still did not converge** (outer iteration limit) --
technology-only flexibility search is evidently much harder to converge at real D=20 scale than
at D4, an order of magnitude slower than the f-only/full stages at the SAME point (`83s`/`28s`).
This matches (and sharpens) the standing "near a narrow corridor, most nearby directions are hard
to certify quickly" finding from every prior real-D20 session.

## Phase 9: real-D20 joint algorithm comparison from an interior point

Starting point: `interior_g05` (the better of the two Phase 8 points, `Delta=0.4830`), NOT the
Pareto point or the `Delta~=1` boundary -- directly testing the governing prompt's own central
Phase 9 question. Active Set and SQP, `theta_box` g-radius `0.10`/A-f-radius `0.15`,
`maxtime_real=240s` (**scoped down from the governing prompt's suggested 300-600s** given this
session's own overall wall-clock budget -- disclosed; neither run actually consumed the full
budget, both terminated well under 240s on their own stopping criteria, so this reduction did not
truncate either trajectory). `external_incumbent` = the known Delta~=1 boundary point
(`g=-0.51376`, Phase 2's `frac=0.65` row) as a floor.

Full CSV: `docs/key_results/melitz_phase9_realD20_algorithm_comparison_2026-07-28.csv`.

| algorithm | nStatus | wall | n_fc | n_ga | n_solved | n_above_cap | dg | best_Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Active Set | -200 | 180.6s | 62 | 10 | 6 | 46 (74%) | **-6.4e-11** | 0.483039 |
| SQP | -101 (genuine xtol) | 183.4s | 193 | 13 | 32 | 125 (65%) | **-6.4e-11** | 0.483038 |

**Both algorithms found ZERO further movement** (`dg` is at floating-point-noise scale, not a
real step) from this already-nuisance-improved interior point -- the terminal `Delta` is
identical (to 5 significant figures) to the Phase 8 starting point. This is the SAME qualitative
"zero net movement" result every prior real-D20 joint-search session in this repo's history has
found (2026-07-25, 2026-07-27 morning), **now additionally confirmed starting from a genuinely
different point** (an interior, nuisance-flexibility-improved point, not the Pareto point or the
raw `Delta~=1` boundary) -- ruling out "the calibration point's own local geometry" as the SOLE
explanation, since this starting point is neither the calibration point nor the boundary. The
majority of trial points at both algorithms (65-74%) hit the evaluation cap -- most nearby
directions from this interior point are ALSO over-budget quickly, consistent with (and now
demonstrated at a second, different point beyond) the standing "narrow corridor near any
budget-respecting point" finding.

**One important caveat, disclosed directly**: Phase 3 (above) found SQP with a WIDER box
(g-radius `0.5`, vs. the `0.10` used here) reaches genuine convergence much deeper into the
gamma-only corridor from the PARETO point. Phase 9's own box was not re-tuned to match that
Phase-3 finding (the two scripts were run concurrently for wall-clock reasons) -- it remains an
open, flagged question whether a wider joint box at THIS interior starting point would find more
room, or whether (as the `74%`/`65%` evaluation-cap-rejection rates suggest) the corridor is now
simply too narrow in every direction regardless of box size once already this close to it. Not
resolved this session.

## Phase 10: KNITRO-native wall-clock decomposition

Reuses the existing `MELITZ_PROFILE`/`melitz_profile_summary` instrumentation (no new
instrumentation invented), bracketing each Phase 9 run.

Full CSV: `docs/key_results/melitz_phase10_realD20_timing_decomposition_2026-07-28.csv`.

| algorithm | total wall | callback wall (fc_total+ga_total) | residual (KNITRO-native) | residual % |
|---|---:|---:|---:|---:|
| Active Set | 180.6s | 65.1s | 115.4s | **63.9%** |
| SQP | 183.4s | 154.5s | 28.9s | **15.7%** |

**The KNITRO-overhead fraction is algorithm-dependent, not a fixed property of this fixture** --
Active Set's `63.9%` residual closely matches every prior session's own figure (`60-66%`) for
Active Set specifically, but SQP's own residual is dramatically lower (`15.7%`) because SQP made
`3x` more FC calls (193 vs 62) and `~9x` more of its own wall inside a SINGLE expensive category
(`ga_divergence_gradient`, `117.2s` across 13 calls for SQP vs `54.0s` across 6 calls for Active
Set) -- SQP's own outer-gradient evaluations are both more frequent and individually costlier in
this run. For Active Set, one single COLD inner solve (`inner_solve_cold`) consumed `91.4s` of
the `180.6s` total (`>50%`) by itself -- a single hard-to-solve trial point, not a systemic cost.
**Answering the governing prompt's Phase 10 question directly: for Active Set, roughly
two-thirds of D20 outer-search wall time is now genuine KNITRO-internal (CG/linear-algebra)
overhead, not Melitz callback cost; for SQP, callback cost (chiefly the parallel outer-gradient
call) dominates instead at 84%.** Neither number is uniformly "the" answer -- which one binds
depends on which algorithm is running.

## Phase 11: hardened objective-scaling defaults

Implemented directly as part of Phase 1 (above), not deferred:

- **Automatic whenever variable scaling is enabled**: `objective_scale=:auto` is the new default;
  `var_scale!==nothing` always yields a compatible, non-`nothing` divisor with no caller action
  required.
- **Raw and scaled objective gradients printed at run start**: `solve_melitz_finite_delta_bound`
  unconditionally prints `objective_scale=... var_scale[1]=... -> obj_scale_divisor=...` (the raw
  chain-rule inputs and the resolved output) before `KN_solve` is ever called -- confirmed live
  in every log this session produced.
- **Scaling included in the RUN's own traceability, not the theta-keyed exact-point cache**: the
  resolved divisor is returned on `MelitzFiniteDeltaOuterResult.objective_scale_resolved`, so a
  report or a downstream script can always recover exactly what scaling a given result used
  without re-deriving it. **Deliberately NOT added to `melitz_context_fingerprint`/the exact-point
  cache's key**: `Delta(theta)` is a pure function of `(ctx, theta)`, independently of how KNITRO's
  own search reached `theta` -- adding solver-level scaling to that fingerprint would force
  spurious cache misses across runs that differ only in step-control tuning but visit the
  identical `theta`, a correctness REGRESSION relative to the existing, already-tested
  ctx-fingerprint design (Phase 10 of the 2026-07-27 night session specifically tested and
  confirmed evaluation-cap-alone changes must NOT alter the fingerprint, for the identical
  reason). This is a deliberate, reasoned deviation from a literal reading of the governing
  prompt's Phase 11 wording, not an oversight.
- **Explicit diagnostic override preserved**: `allow_unscaled_objective=true` remains available
  and is exactly what the new regression tests use to recreate the historical bug on purpose.
- **Regression tests recreating the original bug**: `test/melitz/runtests.jl`'s new "Phase 1/11"
  testset (Phase 1 above) does exactly this -- reproduces the frozen-search configuration live
  (`terminal_theta[1] ≈ theta0[1]`) ONLY under the explicit override, and proves the SAME
  configuration throws without it. Both directions are asserted, not merely one.

## Phase 12: final conclusions

1. **Does automatic objective scaling eliminate false xtol convergence?** Yes, structurally --
   the historical bug configuration now requires an explicit, differently-named,
   impossible-to-reach-by-omission override (`allow_unscaled_objective=true`); live-confirmed at
   both D=4 and real D=20 (Phase 0), and pinned by a permanent regression test (Phase 1/11).
2. **Which gamma-only algorithm and step controls reproduce the direct profile best?** **SQP with
   a wide box (g-radius `0.5`) at real D=20** -- the only configuration this session found that
   reaches genuine `nStatus=0` convergence, landing within `0.0013` of the independently-known
   `Delta<=1` boundary. Active Set is robust but plateaus materially short of the corridor
   regardless of step-control tuning; Interior/CG is a solid D4-only alternative; Interior/Direct
   remains unsafe (documented runaway, reconfirmed as this session's one genuine
   `:stuck_at_pareto`/infeasible-runaway case).
3. **Do longer D4 joint searches reliably improve on restricted benchmarks?** Yes for SQP,
   decisively: it reaches its own `delta` budget to `0.4-0.6%` relative accuracy, consistently
   across 3 seeds and both directions, at both tight budgets tested. Interior/CG is genuine but
   looser. Every one of the 28 longer joint runs carried a real Phase-4 restricted incumbent as
   its `external_incumbent` floor, so none can be reported worse than the restricted pool.
4. **Are D4 results robust across seeds and directions?** Yes for both the restricted pool
   (Phase 4: 54/54 cells moved, 3 seeds, both directions, zero stuck) and the joint SQP search
   (Phase 5: `0.4-0.6%` budget accuracy uniformly across seeds `{29,49,50}` and both directions
   at `delta in {1e-3,1e-2}`).
5. **Does the staged nuisance profile converge more reliably?** More USEFULLY, not merely more
   reliably: the staged A-only/f-only/full design with a programmatically-verified
   never-worse-than-fixed guarantee found real (3-5x) improvements at 4 of 6 grid points and
   rescued outright fixed-A/f infeasibility at 2 of 6 -- a stronger, more complete result than
   the prior session's single-shot full-only nuisance search. One real cost disclosed: A-only at
   the exact calibration point is slow and can fail to converge within a `maxit`-based budget
   (`72.8s`, iteration limit).
6. **How much divergence slack can flexible A/f create at interior D20 points?** **Little**:
   `0.05-0.07%` at the two interior points tested (`Delta~0.48`,`Delta~0.76`) -- a materially
   weaker result than D4's own 3-5x, and a genuine, disclosed finding, not a search-infrastructure
   failure (the staged design and its guarantees are identical to the D4 version that DID find
   large gains). One real operational cost: A-only nuisance minimization at real D=20 can take
   `>1000s` without converging.
7. **Which D20 canned algorithm makes the most useful progress?** Depends on the question asked.
   From the Pareto point with a wide box (Phase 3), **SQP is decisively better**, reaching genuine
   convergence deep in the corridor where Active Set plateaus. From an already-nuisance-improved
   interior point with a NARROWER box (Phase 9), both Active Set and SQP found zero further
   movement -- an open question (Phase 9's own caveat) whether Phase 3's wider-box finding would
   also unlock further joint movement from that interior point; not tested this session.
8. **What fraction of D20 wall time is now native KNITRO overhead?** Algorithm-dependent, not
   universal: `63.9%` for Active Set (matching every prior session's own `60-66%` figure exactly),
   but only `15.7%` for SQP, whose own callback cost (chiefly a `117s` cumulative parallel
   outer-gradient cost across 13 calls) dominates instead.
9. **What is the remaining outer-search bottleneck?** No longer objective/variable scaling
   (fixed and hardened this session) or evaluation cost (fixed 2026-07-27) or step-control
   ignorance (systematically swept this session, Phase 3) -- the bottleneck is now **twofold and
   scale-dependent**: at D4, it is the (disclosed, unresolved) quality of local-optimality
   verification itself (Phase 5.3's flawed local poll); at real D=20, it is the corridor's own
   narrowness once already close to it (Phase 9's `65-74%` evaluation-cap-rejection rate at an
   interior point) combined with real per-point cost variance (A-only nuisance minimization
   ranging from `28s` to `>1000s` at nominally similar points).
10. **Is the code ready for a longer production campaign?** Closer than before, with one
    important caveat: the objective-scaling footgun is now closed by construction and the
    gamma-only step-control choice (SQP, wide box) is now evidence-based rather than a single
    hand-picked configuration. NOT yet ready to declare the JOINT real-D20 search itself solved:
    Phase 9's zero-movement result from an interior point, and the still-unresolved local-poll
    diagnostic at D4, are genuine open items a future session should address before treating a
    longer real-D20 campaign's headline number as fully trustworthy.

## Acceptance criteria

1. Objective scaling automatic and fail-safe: **met** (Phase 1/11).
2. Gamma-only KNITRO reproduces the direct profile under the selected settings: **met** for SQP
   with a wide box at real D=20 (within `0.0013` of the known boundary); Active Set reproduces
   the DIRECTION but not the full extent of the corridor under any step-control combination
   tried (Phase 3) -- disclosed, not glossed over.
3. At least SQP and Interior/CG receive substantive D4 testing: **met** (Phase 3 step-control
   sweep, Phase 4 restricted pool, Phase 5 longer joint searches -- all three algorithms
   including Active Set).
4. D4 joint searches run longer than the prior 25-iteration diagnostics: **met**
   (`maxit=120`, Phase 5).
5. Restricted incumbents are retained in every full search: **met**, structurally
   (`external_incumbent_used=true` in all 28 Phase 5 rows, confirmed from the CSV, not merely
   intended).
6. Multiple D4 seeds and both directions are tested: **met** (seeds `{29,49,50}`, both
   directions, Phases 4-5).
7. The nuisance profile can never report a result worse than fixed A/f: **met**, verified by a
   runtime assertion in the script itself (Phase 6), not merely argued in prose.
8. Real-D20 nuisance minimization is tested from interior profile points: **met** (Phase 8, two
   interior points).
9. At least two canned D20 algorithms are compared: **met** (Active Set, SQP -- Phase 3 AND
   Phase 9, two different starting-point regimes).
10. Wall-clock decomposition isolates native KNITRO overhead: **met** (Phase 10, reusing the
    existing `MELITZ_PROFILE` instrumentation, no new instrumentation invented).
11. No custom optimizer written: **met** -- every experiment calls
    `solve_melitz_finite_delta_bound`/`solve_melitz_nuisance_min_delta`, both pre-existing;
    the only new "search" logic is a staged sequencing of pre-existing solver calls (Phase 6/8)
    and a predetermined (not adaptive) delta/box grid (Phase 3).
12. All main experiments use 20 Julia threads where useful: **met** -- every real-D20 script and
    every D4 script that touches KNITRO used `-t 20`; the D4-only nuisance-profile script
    (Phase 6) used `-t 1`, matching this repo's own standing single-thread convention for that
    class of script (no threaded backend available to that driver, disclosed not silently
    omitted).
13. No Ricardian/shared source modified: **met**, confirmed directly below.
14. Full Melitz tests pass: **met** -- `189,235/189,235` before, `189,247/189,247` after (12 new
    assertions, all passing), zero regressions.
15. Work is committed locally and not pushed: see the commit made immediately after this
    document; not pushed to any remote.

## Files changed

`git diff --name-only 68918c7f0e70f7355a5e1e8b648d052906007820`:

```
src/melitz/finite_delta_outer.jl
test/melitz/runtests.jl
```

New files (all under `scripts/`, `docs/`, `docs/key_results/` -- Melitz-only):

```
docs/melitz_outer_search_step_control_and_robustness_2026-07-28.md   (this document)
scripts/melitz_regression_fixtures_2026-07-28.jl
scripts/melitz_phase0_reproduce_2026-07-28.jl
scripts/melitz_phase3_step_control_lab_2026-07-28.jl
scripts/melitz_phase4_5_d4_restricted_and_joint_2026-07-28.jl
scripts/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.jl
scripts/melitz_phase8_9_10_realD20_2026-07-28.jl
docs/key_results/melitz_phase0_reproduce_2026-07-28.csv
docs/key_results/melitz_phase3_step_control_{summary,trials}_2026-07-28.csv
docs/key_results/melitz_phase4_d4_restricted_incumbents_2026-07-28.csv
docs/key_results/melitz_phase5_d4_longer_joint_2026-07-28.csv
docs/key_results/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.csv
docs/key_results/melitz_phase8_realD20_interior_nuisance_2026-07-28.csv
docs/key_results/melitz_phase9_realD20_algorithm_comparison_2026-07-28.csv
docs/key_results/melitz_phase10_realD20_timing_decomposition_2026-07-28.csv
docs/key_results/tmp_opt_2026-07-28/   (generated KNITRO .opt file variants -- delta/maxit/maxtime overrides on the pre-existing per-algorithm option files, kept for exact reproducibility)
```

**Zero diff in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path** (`production/fullA-exact/`
does not exist in this repo) -- confirmed directly via `git diff --name-only` above, not merely
asserted.

## Explicitly not done this session (disclosed scope, not silently dropped)

- Phase 5's local-poll diagnostic was built with a design flaw (compares raw `Delta`, not the
  signed objective, so it trivially "fails" by finding the toward-Pareto direction always looks
  better) -- disclosed in full in Phase 5.3, not fixed or re-run given this session's own time
  budget.
- Phase 9's real-D20 joint search used the SAME (narrower) box as the original governing-prompt
  default rather than Phase 3's own wider-box finding for SQP -- the two scripts were run
  concurrently for wall-clock reasons; whether a wider box unlocks further joint movement from an
  ALREADY-interior point is an open, flagged question, not tested.
- Phase 9's `maxtime_real` was set to `240s` (scoped down from the governing prompt's suggested
  `300-600s`) given this session's overall time budget; neither run actually consumed the full
  240s on its own stopping criteria, so this reduction did not truncate either trajectory, but a
  longer budget was not independently tried.
- A genuine CSV round-trip bug was found and fixed THIS session (`classify_gamma_only_config`'s
  `:passed` detail string contained unescaped commas, breaking a naive `readdlm`-based re-read of
  `melitz_phase3_step_control_summary_2026-07-28.csv`) -- fixed in
  `melitz_regression_fixtures_2026-07-28.jl` for future use, but Phase 3's own already-produced
  CSV was not regenerated (its data is correct; only a specific naive re-parse of the `detail`
  column breaks) -- Phase 4/5's own dependency on Phase 3's selection was instead satisfied by
  hardcoding the exact values already known with certainty from Phase 3's own console output,
  disclosed directly in that script's own docstring.
