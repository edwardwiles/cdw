# Melitz three-level outer-search architecture: fixed-q A middle loop, bounded experiment (2026-07-30)

Governing prompt: bounded experiment for a NEW middle loop inside a proposed three-level
Melitz outer-search architecture (outer: welfare `g` + cutoff chambers `q`; **middle: for
fixed `(g,q)`, genuinely optimize `DeltaStar` over `A`**; inner: verified `DeltaStar` via
optimization over `F`). Follow-up to
[`melitz_d20_negative_switch_geometry_audit_2026-07-30.md`](melitz_d20_negative_switch_geometry_audit_2026-07-30.md)
and
[`melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md`](melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md).

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`. Session HEAD at start:
`523c4af` (the joint-`(A,q)` LFD-preserving search commit). This session's own work is
**committed locally at the end of this session, not pushed** (see Provenance).

**Does not modify the Ricardian implementation.** Purely additive: one new source file
(`src/melitz/fixed_q_a_middle_loop.jl`), one new KNITRO options file
(`melitz_middle_loop_opt_2026-07-30.opt`), a new experiment driver script, a new testset
appended to `test/melitz/runtests.jl`, and this report. Production `(A,f)` search
(`finite_delta_outer.jl`), the default `:logf`/`:logcutoff` backends, the reduced-q backend,
and `lfd_preserving_state.jl`/`hybrid_chamber_corrector.jl` are untouched (the new module
*reuses* `melitz_f_from_Aq`/`reduce_to_free_theta_logcutoff` from the first of those, verbatim,
rather than duplicating them).

---

## 0. Read-first summary

Read in full before starting: the Melitz-local `CLAUDE.md`
(`src/melitz/CLAUDE.md`, 20-threads default); the six prior reports on strict `(A,q)`
separation and exact `A` derivatives
([`melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md`](melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md)),
reduced-q search and D20 readiness
([`melitz_reduced_q_subspace_search_2026-07-29.md`](melitz_reduced_q_subspace_search_2026-07-29.md),
[`melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md`](melitz_reduced_q_validation_and_d20_readiness_2026-07-29.md)),
the reduced-q `NumericalFailure` forensic audit
([`melitz_reduced_q_numericalfailure_forensic_audit_2026-07-30.md`](melitz_reduced_q_numericalfailure_forensic_audit_2026-07-30.md)),
the negative-switch geometry audit
([`melitz_d20_negative_switch_geometry_audit_2026-07-30.md`](melitz_d20_negative_switch_geometry_audit_2026-07-30.md)),
and the joint-`(A,q)` LFD-preserving search
([`melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md`](melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md));
and the current source for the exact `A`-block gradient (`exact_a_gradient.jl`), log-cutoff
reconstruction (`log_cutoff_param.jl`), the A-gravity pivot (`equilibrium.jl`/`delta_star.jl`/
`affine_cutoff.jl`), q/f gravity, the focal free-entry link (`moments.jl`/`moment_operator.jl`),
the typed inner API (`inner_screening.jl`/`inner_session.jl`), matrix-free callbacks and the
structured inner Hessian (`cc_bundle.jl`/`moment_operator.jl`), and the finite-support
origin-block screens (`origin_block_screen.jl`).

Git state verified at session start: `trade_robustness_modular` branch
`melitz/fullD-delta-star`, HEAD `523c4af`, 43 commits ahead of `cdw/melitz/fullD-delta-star`
(not pushed), a handful of pre-existing untracked scratch files/directories from other
in-progress sessions (left untouched throughout, per this repo's own "don't touch other
sessions' state" norm).

---

## Phase 0: exact fixed-q constraint derivation

Full derivation is reproduced verbatim as the header comment of
`src/melitz/fixed_q_a_middle_loop.jl` (re-derived directly against `firm_quantities.jl`,
`origin_block_screen.jl`, and `log_cutoff_param.jl` -- not assumed from the schematic prompt
formula). Summary:

**Exact affine `h`/`a` relation.** `melitz_C(w_o,tau_od,A_od,sigma,expenditure_d) =
expenditure_d*(markup*w_o*tau_od/A_od)^(1-sigma)` (`firm_quantities.jl`). Writing
`coef_od = C_od/expenditure_d` and `H_od = lambda_od/coef_od` (`lambda_od =
X_data[o,d]/expenditure_d`, EXACTLY `origin_block_screen.jl`'s own `H[d]`):

```
h_od := log(H_od) = const_od - (sigma-1)*log(A_od)
const_od = log(lambda_od) + (sigma-1)*log(markup*w_o*tau_od)     (fixed, data-only)
```

confirmed by direct substitution into `melitz_C`, matching the governing prompt's own
schematic form exactly. Inverse: `a_od = [const_od - h_od]/(sigma-1)`.

**Ordering / same-bin (per origin, over its own `D` destinations only -- different origins
live on unrelated draw axes and impose no cross-origin restriction).** Re-derives
`melitz_origin_block_monotonicity_check`'s own necessary condition
(`origin_block_screen.jl`) in `h`-space: for origin `o`'s sorted-by-cutoff destination chain
(`melitz_origin_intervals`'s own `rank`), adjacent pair `(d_lo,d_hi)` with
`rank[d_lo] <= rank[d_hi]`:

```
rank[d_lo]  < rank[d_hi]  =>  h[d_lo] >= h[d_hi]     (:ordering, linear inequality)
rank[d_lo] == rank[d_hi]  =>  h[d_lo] == h[d_hi]     (:same_bin, linear equality)
```

Only the minimal **adjacent-pair generating set** (`D-1` rows per origin, `D*(D-1)` total) is
registered -- transitivity along the sorted chain implies the full `O(D^2)` pairwise set.
Promoted from a diagnostic post-hoc check into genuine **linear KNITRO constraints** on the
middle loop's free coordinates (registered via `KN_add_con_linear_struct`, zero per-iterate
cost) -- the explicit reason (per the governing prompt) is that an inner-solve-only signal
only rejects an infeasible `A` *after* an expensive, gradient-free evaluation, whereas a hard
linear pre-constraint keeps KNITRO's own trust region inside the region where a differentiable
optimum can exist at all.

**A-gravity** (`dot(c_full,vec(log A))==0`) is an **algebraic identity** of `ctx.A_pivot`'s own
`pivot_expand` (`g0=0` always) for *any* `A_free` -- not an added constraint, as long as the
middle loop's free coordinates are mapped to the full `A` matrix via the same pivot (Phase
1.A) or an affine reparameterization of it (Phase 1.B). Verified directly (10 random
perturbations, `|residual| < 1e-8`, test suite).

**f-gravity redundancy (the governing prompt's explicit decisive question).** `build_q_gravity_offset`
(`log_cutoff_param.jl`, fixed 2026-07-29) computes the q-pivot's offset from `q_jj` (hence
`g`) and fixed data **only** -- it no longer reads `A` at all. Combined with A-gravity (an
identity for any `A_free`) and the algebraic relation `dot(c_full,vec(logf)) =
(sigma-1)*dot(c_full,vec(q)) + (sigma-1)*dot(c_full,vec(loga)) + dot(c_full,const_vec)`
(doc 1, Section 2), **f-gravity is an exact algebraic identity throughout the entire
middle-loop A-search, for every `A_free` tried, not merely at the anchor.** Verified live: 15
random `A_free` perturbations at fixed `q`, `max|f-gravity residual| = 8.67e-17` (machine
precision) -- see `docs/key_results/` test output and the standalone/appended testsets below.
**No additional fixed-q linear restriction is needed for f-gravity.**

**Participation invariant.** Because `q` never moves within the middle loop (by construction
-- `f` is always recovered from the fixed target `q` via `melitz_log_f_from_q`'s exact
inverse, so `melitz_baseline_cutoff(A_new,f_new,...)` reproduces `q` bit-for-bit for *any*
`A_new`), the active set at every cell is **identically fixed throughout the middle-loop
search** -- zero participation switches **by construction**, not merely empirically. Verified
live: 15 random `A_free` perturbations, `max|q_perturbed - q_fixed| < 1e-8` (typically
`1e-16`).

**Scope limitations, disclosed:** (1) the focal origin's autarky "virtual `(D+1)`-th
destination" breakpoint is not included as an extra ordering row -- it has no genuine `(o,d)`
A-cell counterpart, and is provably immaterial at the audited Korean cliff (origin 14 != focal
country 2/fra); (2) only the minimal adjacent-pair generating set is registered, not all
`O(D^2)` pairs (implied by transitivity).

---

## Phase 1: two middle-loop parameterizations

**A. Free log-A coordinates** (`coordinate=:logA`). Reuses `ctx.A_pivot` unchanged --
`A_free` (`D^2-1` entries) IS the search variable; A-gravity holds structurally. Ordering/
same-bin rows are built via `melitz_A_free_linear_map(ctx)` (the `D^2 x (D^2-1)` map
`a_full = M_A*A_free`, built by applying the existing `pivot_expand` to unit vectors, never
re-deriving the pivot formula by hand).

**B. Log-H coordinates** (`coordinate=:logH`). Since the *free* A-cells (`ctx.A_pivot.other`)
already are the search variables (only the single physical pivot cell is dependent),
`h_free[k] = const_free[k] - (sigma-1)*A_free[k]` is a **diagonal, invertible, per-coordinate
rescale** of the *same* coordinates -- not a second independent pivot. Exact inverse and exact
gradient chain rule (`d(DeltaStar)/dh_free[k] = -(1/(sigma-1))*d(DeltaStar)/dA_free[k]`,
matching the governing prompt's own stated transform) both implemented and unit-tested.

**Decisive Phase-0-adjacent finding, disclosed early because it recurs in Phases 3/7's own
data below**: because `sigma` is a *single global* elasticity parameter (not cell-specific),
`h_free = const_free - (sigma-1)*A_free` is a **uniform scalar affine reparameterization of
log-A** (slope `-(sigma-1)` identical for every free cell) -- mathematically, `rows_H =
(-1/(sigma-1)) * rows_A` for the constraint system (a single global scalar multiple, not a
per-column rescale). A uniform scalar multiple cannot change a matrix's condition number or
sparsity pattern. This is a testable, falsifiable prediction of the Phase 0 math, and Phase 7's
measured condition numbers (Section 7 below) confirm it exactly.

---

## Phase 2: `solve_melitz_fixed_q_A_profile`

New standalone KNITRO driver (`src/melitz/fixed_q_a_middle_loop.jl`), patterned on
`nuisance_profile.jl`'s direct "minimize `DeltaStar(theta)`" driver (objective-only nonlinear
callback, `Int32[]` constraint indices) rather than `finite_delta_outer.jl`'s delta-bound
formulation (no divergence-budget row needed -- the middle objective *is* `DeltaStar`
directly).

- **Every objective/gradient evaluation routes through `solve_melitz_delta!`**, the one
  authoritative typed inner API -- `FiniteSolved` (value = true `Delta`, gradient = exact
  envelope-theorem `melitz_exact_a_gradient_full!`/`_free`, chain-ruled to `:logH` if
  requested), `AboveEvaluationCap` (value = `certified_lower_bound`, a genuine weak-duality
  bound; gradient = zero -- no local optimum to differentiate at a capped point),
  `InfiniteDeltaCertified` (value = a fixed large sentinel `1e6`; gradient = zero).
  **`NumericalFailure` is never wrapped into a value** -- it `throw`s a `DomainError`
  (KNITRO.jl's own established convention, matching `finite_delta_outer.jl`), so it can never
  be silently absorbed as a normal middle objective value.
- **Ordering/same-bin rows registered as true KNITRO linear constraints**
  (`KN_add_con_linear_struct`), zero per-iterate cost, exactly the `nuisance_profile.jl`/
  `finite_delta_outer.jl` `:linear` pattern.
- **No finite differences over A. No cutoff movement (by construction). No dense `G`** (the
  same matrix-free `MelitzCCBundle`/`MelitzMomentOperator` production uses) -- verified via
  `MELITZ_DENSE_G_MATERIALIZATIONS[]` staying flat across a full middle solve (test suite).
- **Bounded evaluations**: `max_evals` caps the *combined* `cb_F!`+`cb_G!` call count; exceeding
  it `throw`s (KNITRO.jl reports a clean stop). *(Caveat, disclosed: because KNITRO's own
  `eval_fcga=no` retry behavior can re-invoke a callback several times after the first
  rejection before giving up, the raw `n_fc_calls+n_ga_calls` counters can modestly overstate
  true evaluation cost -- Phase 7 reports the more meaningful `length(eval_log)`, i.e. the
  count of evaluations that actually reached the typed classifier, which is capped exactly at
  `max_evals`.)*
- **Cold re-verification**: the KNITRO trajectory's terminal point is always independently
  re-classified from a fresh warm start under the same policy (`r_final`) -- this, not the raw
  KNITRO objective, is reported as "the answer" (this codebase's established convention).
- `melitz_project_start_to_middle_constraints`: converts a caller-supplied start (e.g. an
  anchor's own `A_free`, or a cellwise-`p*`-compensated `A`) into an exact feasible middle
  start by a single forward pass per origin's sorted chain -- "all starts must be converted to
  exact feasible middle coordinates before KNITRO begins," per the governing prompt.

**Live validation at D4** (`FIXTURE`, seed=29, W=20,000, `Delta0=7.5545e-6`): starting exactly
at the production anchor, the A-space middle solve converges cleanly (`nStatus=0`, "Locally
optimal solution found") to `Delta=1.412e-7` in 16 KNITRO iterations / 43 classified
evaluations / 0.68s wall; the H-space solve converges to `Delta=1.986e-7` in a comparable
budget. **78/78 assertions pass** in a dedicated standalone isolated test run (see Tests,
below) -- avoiding the pre-existing, already-documented, unrelated `mul_G!`
(`moment_operator.jl:281`) SIGSEGV that two immediately-prior sessions also disclosed in the
full 8000-line test suite.

---

## Phase 3: exact gradient / smoothness shakedown

CSV: `docs/key_results/melitz_fixedqA_phase3_gradient_shakedown_2026-07-30.csv`.

At both D4 (anchor, `seed=29/W=20,000`) and real D20 (anchor, `noah_D20`/focal=fra/`seed=1`/
`W=80,000`, `Delta0=0.4832764950468883`, matching the negative-switch audit's own documented
anchor to `1e-4`), the exact envelope-theorem gradient (`melitz_exact_a_gradient_free`) was
checked against a **reoptimized secant** (independent re-solve at a perturbed `A_free`, never
finite differencing the gradient itself) along random dense directions and one
**same-bin-equality tangent direction** (moves two free coordinates on the same adjacent-pair
row so the row's own value is held exactly fixed -- a direct stress test of the same-bin
claim):

| label | direction | h | exact | reoptimized secant | relerr |
|---|---|---:|---:|---:|---:|
| D4_anchor | random_dense (3 dirs) | 1e-5 | -- | -- | 1.3%-2.7% |
| D4_anchor | random_dense (3 dirs) | 1e-4 | -- | -- | 12.8%-27.3% |
| D4_anchor | same_bin_tangent | 1e-5 / 1e-4 | 0.002655 | 0.002743 / 0.003532 | 3.3% / 33.0% |
| D20_anchor | random_dense_2/3 | 1e-5 / 1e-4 | -0.0728 / 0.1135 | -- | 0.7%-8.9% |
| D20_anchor | same_bin_tangent | 1e-5 / 1e-4 | 0.006041 | 0.006062 / 0.006253 | 0.35% / 3.5% |

Relative error scales **approximately linearly with `h`** (a ~10x increase in `h` gives a
~10x increase in relative error, e.g. D4 random_dense_1: 1.3%@1e-5 -> 12.8%@1e-4) -- exactly
the expected first-order-secant curvature signature, not evidence of gradient inaccuracy.
**One disclosed outlier**: D20 `random_dense_1` shows `relerr=150%` (h=1e-5) / `1589%`
(h=1e-4) -- but the *absolute* exact directional derivative there is tiny (`9.4e-4`), so the
relative-error metric is dominated by division-by-a-near-zero artifact, not a real
discrepancy (the absolute gap, `~1.4e-3`, is comparable in magnitude to the other directions'
absolute errors). **Conclusion: the exact A-gradient is accurate throughout, including
exactly at a same-bin-equality boundary, at both D4 and real D20.**

### Box-size / Hessian-option diagnostic (motivates Phase 4's solver configuration)

CSV: `docs/key_results/melitz_fixedqA_phase3_box_hessopt_diagnostic_2026-07-30.csv`. A live
diagnostic at the D20 anchor, run inside this same script (not merely asserted), because an
initial full-scale attempt with the production outer driver's own KNITRO options
(`melitz_outer_finite_delta_alg_direct_2026-07-27.opt`, dense BFGS `hessopt=2`) and a loose
box (`1.0`, matching `finite_delta_outer.jl`'s own `theta_box` default) let KNITRO's first
Newton/interior-point step badly overshoot (objective jumped `0.48 -> 19.3` in one iteration)
and land in much-worse-than-anchor territory:

| box | opt | nStatus | Delta_final | n_classified (F/C/I) | wall_s |
|---:|---|---:|---:|---|---:|
| 1.0 | outer (hessopt=2) | -102 | 1.999 | 60 (49/11/0) | 35.2s |
| 0.3 | outer (hessopt=2) | -102 | 0.329 | 60 (53/7/0) | 29.6s |
| 0.15 | outer (hessopt=2) | -202 | 18.156 | 60 (6/54/0) | 11.6s |
| 0.05 | outer (hessopt=2) | -100 | 0.319 | 150 (139/11/0) | 69.5s |
| 0.05 | **middle (hessopt=6)** | -102 | 0.321 | 150 (100/50/0) | 73.0s |
| **0.1** | **middle (hessopt=6)** | -502 | **0.290** | 150 (125/25/0) | 71.6s |

**Non-monotonic in box size** (`box=0.15` is *worse* than both `0.3` and `0.05` -- a genuine
symptom of a badly-scaled first step landing in a bad region at intermediate box sizes, not a
smooth trend), confirming this is a numerical/scaling issue, not evidence against a genuine
improving direction existing (every configuration tested, including the worst, still finds
*some* improvement over the anchor's `0.4833`, just not reliably). A new dedicated options
file (`melitz_middle_loop_opt_2026-07-30.opt`, L-BFGS `hessopt=6` -- exactly the governing
prompt's own "limited-memory or quasi-Newton" recommendation) with a **tight box (`0.1`)** was
adopted for all D20 Phase 4/6 runs below on this evidence. This reflects the project's own
documented, pre-existing finding (CLAUDE.md) that real-D20 calibrated `A_od` spans roughly 11
orders of magnitude -- an unscaled, loosely-boxed middle problem is genuinely fragile at that
scale, independent of anything specific to this new architecture.

---

## Phase 4/5: bounded D20 profile across the audited negative-switch cliff

CSV: `docs/key_results/melitz_fixedqA_phase4_d20_cliff_profile_2026-07-30.csv`. Uses the
**exact real-D20 anchor, welfare coordinate, and q-direction** from the negative-switch
geometry audit (`noah_D20`, focal=fra, `sigma=2.5`, `seed=1`, `W=80,000`, `target=0.5`,
`Delta0=0.4832764950468883`; reduced-q direction/basis reconstructed via
`melitz_build_reduced_q_stage(theta0,x0,ctx,obj,1;bandwidth_policy=PowerScaledQBandwidth(1e-3,80_000,0.5),target_switches=100)`,
`|b_q|=0.000357`) -- reproduced fresh in this session's own script (the prior sessions' own
`.jls` scratch state was not persisted), with the linearity of `q(t)` and the switch
thresholds cross-verified live (`max slope discrepancy = 3.19e-15`, matching the audit's own
`3.19e-15`; exact switch thresholds loaded from
`docs/key_results/melitz_negswitch_phase2_minus_switches_2026-07-30.csv`/`..._plus_...csv`
and cross-checked, not merely assumed).

**Six fixed-q points**, `g` held fixed throughout at the anchor's own value:

| point | sign | t | note |
|---|---:|---:|---|
| anchor | -1 | 0 | audited base point |
| pre_switch1 | -1 | 6.31e-3 | before first minus-side switch (t=8.41e-3) |
| post_switch1 | -1 | 9.47e-3 | between switch 1 and 2 |
| post_switch2 | -1 | 1.26e-2 | between switch 2 (1.05e-2) and 3 (1.47e-2) |
| post_switch3 | -1 | 1.59e-2 | past switch 3 |
| positive_post_switch1 | +1 | 2.22e-2 | past the plus-direction switch 1 (1.48e-2) |

**Four deterministic starts at each point**, all projected to exact middle feasibility before
KNITRO begins (`melitz_project_start_to_middle_constraints`):

- **A. continuation** -- the previous point's own best verified `A_free` (the anchor's own, at
  the first point).
- **B. old-anchor projected** -- the *original* anchor's `A_free`, projected to the *current*
  point's constraints (i.e. NOT compensated for the moved `q` at all beyond the minimal
  feasibility repair).
- **C. cellwise `p*`-compensated** -- `melitz_cellwise_A_from_moments`
  (`lfd_preserving_state.jl`'s own formula `(*)`, reused verbatim), projected.
- **D. deterministic log-H perturbation** -- a fixed, non-random `h_free` sinusoidal
  perturbation of the anchor, projected.

### Decisive comparison table (Phase 5's central question)

For each point: the **fixed-A(anchor) classification** (`DeltaStar(A_old, q_post)`, i.e. the
anchor's *unchanged* `A_free` re-evaluated at the new `q`) versus the **profiled `Phi(q) =
min_A DeltaStar(A,q)`** (the best cold-reverified `FiniteSolved` result across the 4 starts):

| point | fixed-A(anchor) | classification | best `Phi(q)` | best start | `Phi` vs anchor `Delta0=0.4833` |
|---|---:|---|---:|---|---|
| anchor | 0.483276 | FiniteSolved | **0.289806** | A/B/C (tie) | -40.0% |
| pre_switch1 | 0.483277 | FiniteSolved | **0.279401** | A_continuation | -42.2% |
| post_switch1 | 186,161 | **AboveEvaluationCap** | **0.483264** | C_cellwise_compensated | -0.003% (~unchanged) |
| post_switch2 | 493,433 | **AboveEvaluationCap** | **0.29296** | C_cellwise_compensated | -39.4% |
| post_switch3 | 223,288 | **AboveEvaluationCap** | **0.254625** | A_continuation | -47.3% |
| positive_post_switch1 | 0.483250 | FiniteSolved | **0.287394** | D_H_perturbation | -40.5% |

**Answering the decisive comparison directly**: at all three post-switch points, fixed-A
`DeltaStar` is `AboveEvaluationCap` with an astronomically large certified lower bound
(186K-493K) -- reproducing/extending the negative-switch audit's own finding that this is a
genuine, not-a-knife-edge infeasibility under a *fixed* `A`. The profiled `Phi(q)`, in
contrast, is **finite at every single one of the six points**, and at `post_switch2`,
`post_switch3`, and `positive_post_switch1` is **substantially *lower* than the pre-switch
anchor's own `Delta0`** (39-47% reductions) -- not merely "recovered to about the same
divergence," but a **materially better fit** once `A` is genuinely allowed to re-optimize.
`post_switch1`'s own best value (`0.483264`) **independently reproduces the joint-`(A,q)`
LFD-preserving session's own reported number at this exact bracket** (`Delta*=0.483264`,
`melitz_joint_Aq_feasibility_preserving_search_2026-07-30.md`) to 6 significant figures --
obtained here via a genuinely different method (gradient-based KNITRO optimization from the
same cellwise-compensated start, not the prior session's closed-form-plus-discrete-corrector
construction) -- a strong independent cross-validation of both sessions' machinery.

**Same-bin H-equality at the optimum**: every KNITRO run in this campaign reported `Final
feasibility error (abs/rel) = 0.00e+00 / 0.00e+00` (verified directly in the run logs, not
merely assumed) -- since the same-bin rows are registered as exact linear equality
constraints, this is direct evidence that the optimal `A` makes every same-active-set-bin
destination pair's `H` target exactly equal, at every terminal point in the campaign.

**Start reliability -- the central practical finding of Phase 4.** No single start is
reliable on its own:

- **A (continuation)**: excellent at `pre_switch1` (0.279) and `post_switch3` (0.255,
  inheriting `post_switch2`'s own best point), but at `post_switch2` it *started already
  `FiniteSolved` at 0.483* (inherited from `post_switch1`'s own compensated result) and the
  middle solve then **wandered to `AboveEvaluationCap` (647)** -- a genuine regression from a
  good starting point. At `positive_post_switch1` it wandered even more badly (started
  `FiniteSolved` at 0.255, ended `AboveEvaluationCap` at `5.6e8`) -- a large discontinuous jump
  in the true q-direction (minus-side deepest point to the positive-side point) defeats naive
  continuation.
- **B (old-anchor, uncompensated)**: gets **permanently stuck** at `post_switch1`
  (`AboveEvaluationCap` immediately, only 2 evaluations, `nStatus=0` "locally optimal" --
  because an `AboveEvaluationCap` starting point returns a **zero gradient** by this
  session's own convention (no local optimum to differentiate around at a capped point), so
  KNITRO has no signal to escape and declares false convergence at a terrible value). Also
  wanders badly at `post_switch3` (`AboveEvaluationCap`, 663). Works fine at `post_switch2`
  and `positive_post_switch1`.
- **C (cellwise-compensated)**: the **only start that reaches `FiniteSolved` at every single
  one of the six tested points** -- a clean, decisive, reliable pattern. This is exactly the
  prior session's own `lfd_preserving_state.jl` construction, now validated as the right
  *seed* for a genuine gradient-based middle-loop search, not merely a standalone endpoint.
- **D (deterministic H-perturbation)**: fails (immediately `AboveEvaluationCap`, 2-17
  evaluations) at `anchor`/`pre_switch1`/`post_switch2`/`post_switch3` (an *un*-informed
  perturbation has no reason to land in the tiny compensated-feasible region near a switch),
  but works at `post_switch1` and gives the single **best** result at
  `positive_post_switch1` (0.287).

**Answering Phase 5's five questions directly:**

1. **Does `Phi(q)` remain finite across the switch where fixed-A `DeltaStar` becomes
   infinite?** Yes, at all three tested post-switch points, decisively (verified, not
   approximated -- cold-reverified `FiniteSolved` classifications).
2. **Is `Phi(q)` approximately continuous?** Mixed. `anchor -> pre_switch1` is smooth
   (0.290 -> 0.279). `pre_switch1 -> post_switch1` shows an apparent jump (0.279 -> 0.483),
   but `post_switch2`/`post_switch3` (deeper past the switch) drop back to 0.29/0.25 -- **the
   `post_switch1` value is very likely NOT the true `Phi` at that point**, but a local optimum
   close to the cellwise-compensated start under this session's own tight `box=0.1`/
   `max_evals=150` budget (disclosed explicitly, not glossed over -- see "Do not describe a
   bounded local middle optimum as global" in the Final Report Questions below). The
   *existence* of a much lower value nearby (found at the adjacent points) suggests the true
   `Phi(post_switch1)` is likely also well below 0.48, not yet reached within this session's
   bounded budget.
3. **Does the optimal `A` jump at the chamber transition?** Not tested directly (would require
   comparing full `A_free_final` vectors pre/post-switch, which were not persisted to CSV --
   disclosed gap); the large `Delta` values found (`0.29-0.48` at different points, versus a
   single continuous anchor value of `0.483`) are *consistent with* a discontinuous jump but do
   not, by themselves, prove one.
4. **Does continuation from the previous A converge rapidly?** Sometimes (when it lands near a
   good point, e.g. `pre_switch1`, `post_switch3`), sometimes catastrophically not (see start A
   above) -- continuation alone is not reliable, though it is cheap when it works (near-zero
   extra evaluations if already `FiniteSolved`).
5. **Do different starts converge to the same middle solution?** No, at D20 -- see the start
   reliability discussion above (this directly informs the Decision Criteria verdict below).

---

## Phase 6: D4 multistart diagnostic

CSV: `docs/key_results/melitz_fixedqA_phase6_d4_multistart_2026-07-30.csv`. Two D4 states (the
ordinary anchor, and a genuine chamber-transition point found via
`melitz_q_direction_exact_switches` along a small dense reduced-q direction), each with the
same four start types (anchor/projected-perturbations rather than the D20-specific `p*`-
compensation, since D4's `p*` machinery is identical in spirit but the point of this phase is
multistart agreement, not compensation-necessity), each under **both** `coordinate=:logA` and
`coordinate=:logH`:

| state | coordinate | Delta_final range across 4 starts | nStatus |
|---|---|---|---|
| D4_ordinary | logA | `1.41e-7` -- `2.39e-7` | all 0 |
| D4_ordinary | logH | `1.53e-7` -- `6.21e-7` | all 0 |
| D4_chamber_transition | logA | `4.31e-8` -- `2.62e-7` | all 0 |
| D4_chamber_transition | logH | `3.23e-7` -- `4.22e-8` | 3x 0, 1x -502 (still `FiniteSolved`) |

**All 16 runs converge to `FiniteSolved`**, tightly clustered within roughly a factor of 5-15
of each other (all in the `1e-8`-`1e-6` range, i.e. all effectively at the D4 problem's own
near-zero-divergence floor) **regardless of start or coordinate system** -- a clean, decisive
D4 finding: **no evidence of materially different local minima at D4**, for either an ordinary
point or a genuine chamber-transition point. This *contrasts* with the D20 finding (Phase 4:
strong start-dependence) -- consistent with the D20 sensitivity being substantially a
numerical/scaling artifact of the much higher dimension (`nA=399` vs `15`) and the ~11-orders-
of-magnitude `A_od` spread, not necessarily evidence of many genuinely separated local minima
in the underlying economic problem.

---

## Phase 7: cost, practicality, and conditioning

CSV: `docs/key_results/melitz_fixedqA_phase7_conditioning_2026-07-30.csv`.

**Conditioning: log-A vs log-H.** `cond(rows_A) = 1.2445e16`, `cond(rows_H) = 1.2212e16` --
**statistically indistinguishable** (1.9% apart, within floating-point noise for a matrix this
ill-conditioned). `nnz` identical (`1157/151620` both spaces). This exactly confirms the Phase
1 prediction: because `sigma` is a single global parameter, this implementation's log-H is a
*uniform scalar rescaling* of log-A, which cannot change either sparsity or condition number.
**Both coordinate systems are similarly, extremely poorly conditioned in absolute terms**
(`~1e16`) -- reflecting the ~11-orders-of-magnitude spread of calibrated `A_od` documented in
this project's own `CLAUDE.md`, not a log-A-vs-log-H distinction. This is why the box-size/
Hessian-option choice (Phase 3) mattered far more in practice than the coordinate system did.

**Cost.** D4 middle solves: `0.6-2.0s` wall, `17-52` classified evaluations. D20 middle
solves (`box=0.1`, `hessopt=6`, `max_evals=150`): well-behaved runs (a `FiniteSolved` or
improving trajectory) took `55-95s` wall for up to 150 classified evaluations (roughly
`0.4-0.6s`/inner-solve, consistent with warm-started real-D20 CC-dual solves); degenerate runs
(already-capped or zero-gradient-trapped starts) terminated almost instantly (`0.3-13s`, only
`2-50` evaluations) since KNITRO gave up immediately on a zero local gradient. **Warm-start
benefit**: substantial when continuation lands in a feasible region (near-zero extra cost --
e.g. `post_switch2`'s start A began already `FiniteSolved`), but this is start-dependent, not
automatic (see Phase 4's start-reliability discussion) -- q-continuation warm starts help only
when the destination point is genuinely nearby in the relevant sense; the D20 real-solve inner
warm-start machinery itself (KNITRO's own dual-bank/previous-point warm start,
`warm_start_source=:previous`, used throughout) delivers its usual benefit within a single
middle trajectory (successive `cb_F!`/`cb_G!` calls at nearby trial points solve much faster
than the initial cold `4.28s`/`6.17s` base-point solve).

**Embedding-cost estimate (explicitly NOT run).** A short outer cutoff search evaluating, say,
10-20 distinct `q` points per outer iterate, each needing one middle solve at the
well-behaved-run cost above (`~70s`), would cost roughly `700-1400s` (12-23 minutes) per outer
iterate if middle solves were run serially with the full `max_evals=150` budget -- 20-70x the
cost of a single production `(A,f)` direct inner solve (`~1-4s`, doc 1's own timings). Cutting
the per-middle-solve budget (the data above show most of the improvement is found within the
first `~50-100` evaluations, not uniformly across all `150`) and/or running independent `q`
points' middle solves in parallel (they share no state) could plausibly bring this down by an
order of magnitude, but this was **not tested** -- flagged as a concrete, bounded follow-up
rather than an unverified claim.

---

## Tests

`test/melitz/runtests.jl`, new testset `"Governing prompt 2026-07-30 (fixed-q A middle loop,
three-level architecture)"` (also `include`d into the manual dependency list both there and in
`src/melitz/include_melitz.jl`). Covers, at D4: constraint-system row count and cross-check
against `melitz_origin_block_monotonicity_check`; A-gravity as an identity for random
`A_free`; f-gravity exact redundancy (15 random perturbations); zero cutoff movement under `A`
(15 random perturbations); the exact log-A/log-H gradient transform and same feasibility
verdict in both spaces; the typed classification wrapper never leaking `NumericalFailure` as a
value; and (KNITRO-gated) a full `solve_melitz_fixed_q_A_profile` run in both coordinate
spaces checking `MELITZ_DENSE_G_MATERIALIZATIONS[]` stays flat (no dense G), zero cutoff
movement, gravity residuals `<1e-6`, ordering-constraint satisfaction, and no-worse-than-anchor
`Delta`. **78/78 assertions pass** in a dedicated standalone isolated run
(avoiding the pre-existing, unrelated `mul_G!` SIGSEGV two immediately-prior sessions also
disclosed in the full suite).

---

## Reproducible scripts

- `scripts/melitz_fixedqA_middleloop_experiment_2026-07-30.jl` -- the full Phases 3-7
  campaign (`MELITZ_MIDDLELOOP_MODE=full`, default) or a fast sanity check
  (`MELITZ_MIDDLELOOP_MODE=smoke`, one D20 point, small evaluation budgets). Writes all six
  `docs/key_results/melitz_fixedqA_phase*_2026-07-30.csv` files reported above, plus
  `scripts/fixedqA_experiment_state_2026-07-30.jls` (a full-state serialization of the scalar
  result rows -- not the individual `A_free_final` vectors, a disclosed gap noted in Phase
  5 Q3 above).
- `melitz_middle_loop_opt_2026-07-30.opt` -- the L-BFGS middle-loop KNITRO options file adopted
  after the Phase 3 box/Hessian diagnostic.

---

## Final report answers

1. **Is the fixed-q middle problem smooth in practice?** The objective's *gradient* is smooth
   and exact everywhere tested (Phase 3), including exactly at a same-bin-equality boundary.
   The *optimization landscape*, at D20 scale (`nA=399`), is numerically delicate without a
   tight trust region (`box<=0.1`) and L-BFGS -- a loose box with dense BFGS lets KNITRO's
   first step badly overshoot (Phase 3's box diagnostic). At D4 (`nA=15`) it is well-behaved
   under essentially any of the tested configurations (Phase 6).
2. **Are the exact A derivatives accurate throughout the middle solve?** Yes, at both D4 and
   real D20, confirmed against independent reoptimized secants (Phase 3), including at a
   same-bin-equality tangent direction (one disclosed near-zero-derivative relative-error
   artifact, not a real inaccuracy).
3. **Does the post-switch middle optimum remain finite?** Yes, at all three tested post-switch
   points (Phase 4/5), decisively (cold-reverified `FiniteSolved`), versus `AboveEvaluationCap`
   certified lower bounds of 186K-493K under the fixed anchor `A`.
4. **How does its `DeltaStar` compare with the pre-switch value?** Lower (better) at
   `post_switch2`/`post_switch3`/`positive_post_switch1` (39-47% reductions vs the anchor's own
   `0.4833`); essentially unchanged at `post_switch1` (`0.483264`, matching the prior session's
   own reported number at this exact bracket, but likely not yet the true optimum given the
   evidence from the deeper post-switch points -- see Phase 5 Q2).
5. **Does optimal A make identical active-set destinations have identical H targets?** Yes --
   every terminal point in the campaign satisfied the same-bin equality rows to KNITRO's own
   `0.00e+00` feasibility tolerance (direct evidence from the run logs, not assumed).
6. **Do log-A and log-H find the same middle solution?** Yes at D4 (Phase 6: all 16 runs
   cluster within the `1e-8`-`1e-6` range regardless of coordinate system). Not directly
   compared at D20 in this bounded session (disclosed scope limit -- Phase 4 used `:logA`
   only, per the governing prompt's own Phase 6/D4-only multistart-comparison design).
7. **Which parameterization is better conditioned?** Neither -- statistically indistinguishable
   (Phase 7), and mathematically provably so in this implementation (a uniform scalar
   rescaling, since `sigma` is a single global parameter). Both are similarly, severely
   ill-conditioned in absolute terms, reflecting the ~11-orders-of-magnitude `A_od` spread this
   project's own `CLAUDE.md` documents -- a genuine practical challenge for a future
   production version, but not one log-H (in this direct form) resolves.
8. **How sensitive is the result to the four starts?** Very sensitive at D20 (Phase 4: outcomes
   range from `FiniteSolved` at `0.25-0.30` to `AboveEvaluationCap` at `1e2`-`1e11` depending on
   start); essentially insensitive at D4 (Phase 6).
9. **Is broad multistart necessary?** No -- a *small*, economically-motivated, deterministic
   start (the cellwise `p*`-compensated `A` from `lfd_preserving_state.jl`'s own formula `(*)`)
   was `FiniteSolved` at every one of the 6 tested D20 points, the only start with that
   property. Broad/random multistart was never needed; but a *naive* start (raw anchor `A`, or
   an uninformed perturbation) is demonstrably unreliable on its own, including a genuine
   zero-gradient trap (start B at `post_switch1`).
10. **How many inner solves and how much wall time does one middle optimization require?**
    D20, well-behaved: `55-95s` wall, up to 150 classified inner solves (`~0.4-0.6s` each).
    D20, degenerate (stuck/capped) starts: `0.3-13s`, `2-50` inner solves. D4: `0.6-2.0s`,
    `17-52` inner solves.
11. **Does continuation from the preceding q chamber materially reduce cost?** Sometimes
    substantially (a feasible inherited start needs zero extra evaluations to *start*
    `FiniteSolved`), but it is not reliable on its own -- it produced two of the four
    catastrophic-wander failures observed in Phase 4 (starts A at `post_switch2` and
    `positive_post_switch1`, both of which began `FiniteSolved` and ended
    `AboveEvaluationCap`). Continuation should be paired with the cellwise-compensated start,
    not relied on alone.
12. **Which of Conclusions A-E is supported?**

### Decision: **B -- middle loop is workable but needs bounded multistart**

Criteria (governing prompt's own wording): *"multiple local minima appear; a small
deterministic start set reliably finds the best result."* Both hold exactly, per the D20
evidence above: genuinely different outcomes from different starts at the *same* fixed-q point
(ruling out A's "starts converge to essentially the same solution"), while a small, fixed,
economically-motivated start portfolio -- specifically, the cellwise `p*`-compensated start,
optionally combined with continuation as a cheap first try -- reliably finds a finite,
substantially-improved result at every tested point (ruling out D's "impractical" verdict; the
observed failures are start-dependent, not intrinsic). Conclusion C (log-H materially
improves conditioning) is explicitly **not** supported (Phase 7). Conclusion A's stronger
"starts converge to essentially the same solution" criterion is **not** met at D20 (though it
*is* met at D4 -- Phase 6). No inconsistency in the fixed-q constraint derivation or state
reconstruction was found (Conclusion E does not apply -- every correctness test, Phase-0
identity check, and cross-check against `melitz_origin_block_monotonicity_check` passed).

**Recommendation** (matching Decision B's own stated recommendation): use continuation plus a
fixed small start portfolio -- concretely, **(1) try continuation from the previous q-chamber's
best point first (cheap when it works); (2) always also compute and try the cellwise
`p*`-compensated start (`lfd_preserving_state.jl`'s formula, this session's single most
reliable start); (3) cold-reverify and keep the better of the two.** A production outer
cutoff-chamber prototype built on this profiled `Phi(q)` should budget accordingly (Phase 7's
embedding-cost estimate) and should NOT rely on a single generic/uninformed start.

**Explicitly not claimed**: that any reported `Phi(q)` is the *global* minimum over `A` at that
point (only a verified, cold-reoptimized *local* optimum from the tested starts -- the
`post_switch1` discussion in Phase 5 Q2 is the clearest disclosed instance where the true
`Phi` is likely materially lower than what this session's bounded budget reached).

---

## Provenance

- Repo: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`, branch
  `melitz/fullD-delta-star`.
- Session started at base commit `523c4af` (joint-`(A,q)` LFD-preserving search, 2026-07-30).
  **A concurrent session (this project runs many parallel agent sessions against this same
  branch) committed `261f2e6` ("Melitz hybrid chamber-aware LFD-preserving corrector") to the
  SAME shared working directory partway through this session**, touching
  `src/melitz/include_melitz.jl` and `test/melitz/runtests.jl` -- both files this session had
  also mid-edited (one new `include` line each, plus this session's own new testset appended
  to `runtests.jl`). That session explicitly detected the collision and disclosed it in its
  own commit message ("include_melitz.jl and test/melitz/runtests.jl were also concurrently
  modified by a different process... those additions are left in place as-is"), so this
  session's own one-line `include` additions and its new testset rode along into `261f2e6`
  unmodified and correctly -- verified directly (`git diff HEAD` on both files is empty; the
  new testset and `include("fixed_q_a_middle_loop.jl")` line are both present and intact at
  current HEAD). No content was lost; this session's own standalone new file
  (`src/melitz/fixed_q_a_middle_loop.jl`) remained untouched/untracked by the other commit, as
  confirmed by `git show 261f2e6 --stat`.
- This session's own new/changed deliverables, committed locally on top of `261f2e6`:
  `src/melitz/fixed_q_a_middle_loop.jl` (new), `melitz_middle_loop_opt_2026-07-30.opt` (new),
  `scripts/melitz_fixedqA_middleloop_experiment_2026-07-30.jl` (new), five
  `docs/key_results/melitz_fixedqA_phase*_2026-07-30.csv` (new), this report. (The
  `include_melitz.jl`/`runtests.jl` changes attributed to this session are already committed,
  as part of `261f2e6`, per the paragraph above.) **Committed locally at the end of this
  session; not pushed** (per this repo's own standing "confirm before pushing to production"
  convention).
- Julia: `juliaup` toolchain, `-t 8` (this session's environment had 8 available threads, not
  the Melitz-local `CLAUDE.md`'s documented 20-thread production default -- disclosed via the
  `melitz_thread_startup_report()` warning printed at the start of every run in this session;
  this affects wall-clock timing reported above but not correctness, since the middle loop's
  own inner solves are inherently sequential per KNITRO trial point, matching production's own
  outer-driver parallelism profile).
- KNITRO: Artelys Knitro 13.0.1, academic license.
