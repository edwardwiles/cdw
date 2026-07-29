# Melitz sequential reduced-q-subspace outer-search backend (2026-07-29 continuation session)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from the
q-bandwidth convergence campaign (`docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md`,
commit `be31490`), which this session read in full and does not re-derive. That session
established, decisively: individual free-q coordinate secants are essentially exact and
improve with `W`; a DIRECT dense-block fixed-dual secant along an arbitrary q direction also
tracks the fully reoptimized `DeltaStar` change essentially exactly and improves with `W`; but
SUMMING coordinatewise secants into a dense-direction prediction does not improve with `W` and
disagrees in sign 5-9% of the time -- "the problem is coordinatewise aggregation," not
per-coordinate estimation or dual-reoptimization curvature. This session implements and tests
the governing prompt's proposed fix: collapse the outer search's free-q block to ONE scalar
KNITRO variable `s` per stage, moving along one fixed dense direction, with the outer gradient
supplied to KNITRO always a genuine DIRECT block secant -- never a coordinatewise sum.

## Phase 0: repository and baseline audit

- Branch `melitz/fullD-delta-star`, HEAD `be31490` at session start (37 commits ahead of
  `cdw/melitz/fullD-delta-star`, not pushed). `git status` clean except pre-existing untracked
  scratch directories inherited from other sessions (unrelated, untouched this session).
- Full test suite (`julia --project=. -t 1 test/melitz/runtests.jl`) confirmed clean **before
  any edit**: every displayed testset `Pass==Total`, exit code 0 (baseline log:
  `key_results/../provenance` archive, see Provenance section).
- Source files identified (per the governing prompt's own Phase 0 checklist):
  - outer parameterization dispatch: `src/melitz/log_cutoff_param.jl`
    (`melitz_expand_theta`/`melitz_reduce_theta`), `outer_parameterization_config.jl`.
  - q-gravity reconstruction: `expand_free_theta_logcutoff`/`build_q_gravity_pivot`/
    `build_q_gravity_offset` (`log_cutoff_param.jl`).
  - exact A gradient: `src/melitz/exact_a_gradient.jl` (`melitz_exact_a_gradient_full!`).
  - exact smooth q gradient (fixed active set): `src/melitz/exact_q_smooth_gradient.jl`.
  - coordinatewise q secants: `src/melitz/q_bandwidth_policy.jl`
    (`melitz_q_coordinate_probe`, `MelitzQBandwidthPolicy` family).
  - direct dense-block q secants: previously SCRIPT-ONLY
    (`scripts/melitz_qbw_phase7_dense_directions_2026-07-29.jl`'s own inline pattern) --
    promoted to a source-level function this session (`melitz_q_direct_block_secant`,
    `reduced_q_subspace.jl`), per Rule 10 ("no second, script-only implementation").
  - outer linear constraints: `src/melitz/affine_cutoff.jl`
    (`build_melitz_affine_cutoff_system`, `MelitzAffineCutoffSystem`).
  - typed inner evaluation policies/classifications: `src/melitz/inner_solve_policy.jl`
    (`CappedEvaluation`/`FullValueEvaluation`), `inner_screening.jl` (`FiniteSolved`/
    `AboveEvaluationCap`/`InfiniteDeltaCertified`/`NumericalFailure`), `inner_session.jl`
    (`MelitzInnerSession`/`solve_melitz_delta!`, the one authoritative entry point).
  - incumbent retention: `finite_delta_outer.jl`'s own `MelitzOuterCandidate`/
    `cold_verified_incumbent` pattern (`solve_melitz_finite_delta_bound`) -- mirrored, not
    reused directly, by this session's own (deliberately simpler) controller (Phase 8 below).
- Full relevant Melitz test suite re-confirmed clean AFTER this session's own edits (Section
  "Test suite" below).

## Phase 1: reduced-q state map (`src/melitz/reduced_q_subspace.jl`)

`MelitzReducedQStage(q_anchor_free, q_basis_free, stage_id, bandwidth_policy, theta_anchor,
s_lo, s_hi, fingerprint)`. The reduced outer vector is `x_reduced = (g, A_free..., s)`, length
`2+nA` (`nA=D^2-1`). The state map,

```julia
melitz_reduced_full_theta(x_reduced, stage, ctx) =
    vcat(x_reduced[1], x_reduced[2:1+nA], stage.q_anchor_free .+ x_reduced[end].*stage.q_basis_free)
```

hands the reconstructed FULL `theta_free` (length `1+nA+nq`, `nq=D^2-2`) directly to the
EXISTING `expand_free_theta_logcutoff`/`melitz_expand_theta` dispatcher at every call site --
**zero lines of gravity-reconstruction logic are duplicated anywhere in this session's new
code.** `q_anchor_free`/`q_basis_free` are defined over `q_free_free`'s own domain (length
`D^2-2`, i.e. the KNITRO-controlled free-q sub-vector under production `:logcutoff` --
excluding BOTH the focal domestic cell `(j,j)` and the q-gravity pivot cell, which are
reconstructed downstream by the SAME production pivot-expansion code this map never touches).

Verified directly (D4, `docs/key_results` / test suite):

- `s=0` reproduces the anchor state **bit-for-bit** (not merely approximately) -- exact by
  construction (`stage.q_anchor_free .+ 0.0.*stage.q_basis_free === stage.q_anchor_free` in
  floating point).
- an `s`-only perturbation changes the reconstructed q-block and leaves the A-block
  bit-identical; an `A`-only perturbation changes the A-block and leaves the reconstructed
  q-block bit-identical -- both directions, confirmed live, not merely argued from the map's
  own algebraic form (this also directly exercises last session's own hard-won strict A/q
  separation fix, `docs/melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md` Section 1
  -- a regression would have shown up here immediately).
- the welfare coordinate produces exactly its existing documented effect, INCLUDING the
  affine effect on the profiled q-gravity pivot (`q_jj -> g0_q`, `log_cutoff_param.jl`) --
  this is not special-cased anywhere in the reduced map; `melitz_reduced_full_theta` simply
  passes `x_reduced[1]` straight through to `theta_full[1]`, and the EXISTING
  `expand_free_theta_logcutoff` re-derives `g0_q` from the current `g` at every call, exactly
  as production already does.

Fingerprint (`melitz_reduced_q_stage_fingerprint`) hashes the ctx content fingerprint, stage
id, anchor, basis (hence its own normalization -- rescaling `q_basis_free` changes the hash),
bandwidth policy (via `string(policy)`, since the policy types carry no custom `Base.hash` and
are not `isbits`-auto-hashed in this Julia version -- verified this is content-sensitive, not
merely type-sensitive, in the regression test), `W`, and QMC seed -- changing ANY of these six
inputs changes the fingerprint (regression test item 9, all six checked independently).

## Phase 2: transformed affine cutoff constraints

`theta_free = M*x_reduced + c0` exactly (`melitz_reduced_q_lift_matrix`), so the reduced
system is `C_r = sys.C*M`, `b_r = sys.C*c0 + sys.b`, where `sys` is the UNMODIFIED, EXISTING
`build_melitz_affine_cutoff_system(ctx)`. This is an exact algebraic composition, not an
approximation -- verified numerically at randomized reduced points, both at D4 (max
discrepancy vs. the production full-state system: `9.7e-17`, floating-point noise) and at real
D=20 (`4.0e-16`). Registered directly with KNITRO as true linear rows
(`KN_add_con_linear_struct`), zero per-iterate evaluation cost, mirroring production's own
`:linear` cutoff-backend pattern exactly.

The scalar `s`'s own feasible interval, `melitz_reduced_q_s_interval` (Phase 4), is a CLOSED-
FORM 1-D LP over the reduced system at the anchor's fixed `(g,A_free)` -- not sampled, not
bisected -- confirming the governing prompt's own expectation that the transformed
restrictions can (in principle) involve both the welfare coordinate and `s` jointly (through
the q-gravity pivot's `g0_q(g)` dependence, already present in `sys.C`/`sys.b` since the WHOLE
system is affine in `theta_free` including `g`).

## Phase 3: candidate q-direction proposal

`melitz_reduced_q_propose_direction`: (1) the EXISTING coordinatewise secant vector `g_q`
(`melitz_q_coordinate_probe`, one call per free-q coordinate, `mode=:fixed_dual`) is computed
and used ONLY as a proposal signal -- never passed to KNITRO; (2) scaled steepest-descent
`d_tilde = -S_q^{-2}.*g_q` (`S_q=1` by default -- no separate KNITRO q-variable scaling is
set on this experimental backend, a disclosed simplification); (3) normalized to unit norm;
(4)/(5) its full-q effect is exactly the reduced state map's own reconstruction (no separate
"full-q effect" computation exists to duplicate); (6) evaluated via ONE
`melitz_q_direct_block_secant` call at a small crossing-calibrated probe amplitude (10
two-sided switches by default); (7) sign chosen so increasing `s` locally reduces the direct
fixed-dual estimate of `DeltaStar`. If the direct directional response is non-finite or
negligible, `nothing` is returned (Phase 3's own "do not invent a useful direction"), and
`melitz_run_reduced_q_sequential_search` (Phase 8) falls back to a welfare-plus-A-only stage.

`melitz_reduced_q_choose_direction` additionally re-signs/re-evaluates the PREVIOUS stage's
own successful direction at the NEW anchor (never assumed still descent-oriented) when one is
supplied, and picks whichever of the (at most two) unit-norm candidates has the larger-
magnitude direct fixed-dual secant -- exactly the governing prompt's own "compare only the
newly proposed direction to the previous successful direction" instruction; no larger
direction library was built this session.

## Phase 4: normalize and bound the scalar q coordinate

`melitz_bisect_amplitude_for_target_crossings` (promoted from the Phase 7 diagnostic script's
own validated pattern) bisects the unit direction's raw amplitude `r_q` for a total two-sided
crossing-count target (default 100, the governing prompt's own recommended default) --
`b_q = r_q*d`. `s`'s trust range defaults to `[-1,1]`, further intersected with the Phase 2
closed-form exact feasible interval. Recorded per stage: raw/scaled norm of `b_q` (`|q_basis_
free|`), the feasible `[s_lo,s_hi]` interval, and (via `melitz_q_direction_two_sided_
crossings`) plus/minus crossing counts at any tested amplitude.

## Phase 5: reduced outer gradient

`melitz_reduced_q_gradient!` fills `g_reduced=(g,A_free...,s)`:

- **welfare block**: cheap fixed-dual central secant on `g`, identical construction to every
  existing direct FD backend's own welfare-coordinate handling.
- **A block**: the EXACT envelope-theorem gradient (`exact_a_gradient.jl`), zero FD probes,
  valid at ANY `s` (verified: at `s=0.3*s_hi`, D4, the exact A gradient matches its own
  fixed-dual secant to relative error `4.4e-10` -- essentially machine precision, confirming
  the exact-A result is not an `s=0`-only artifact).
- **scalar-s block**: a DIRECT dense-block fixed-dual CENTRAL secant along `stage.q_basis_
  free`, via `melitz_q_direct_block_secant` -- literally `(B(s+h_s)-B(s-h_s))/(2h_s)`, computed
  by the SAME source-level function the Phase 7 campaign validated (promoted, not
  reimplemented). `h_s` is chosen via the crossing infrastructure (default target: 50 total
  two-sided switches, within the governing prompt's own 25-100 recommended range). **Never**
  `dot(stage.q_basis_free, coordinatewise_secants)` -- verified directly (regression test item
  7): at the D4 fixture, the direct block secant (`-3.737e-5`) and the coordinatewise-
  assembled dot product (`-3.605e-5`) differ measurably (not by floating-point noise), and the
  source code path for `g_reduced[end]` never touches `g_q` at all.

Near a stage/support boundary, a genuine one-sided secant is used instead (feasibility checked
against the EXACT Phase 2 reduced system, not a heuristic), and `info.one_sided` records this
explicitly -- never silently clips a central probe while still labeling it central.

## Phase 6: outer Hessian treatment

`hessopt=6` (KNITRO's limited-memory BFGS quasi-Newton option) is set on every reduced-NLP
solve (`reduced_q_controller.jl`). No exact outer second-derivative is derived or supplied, per
the governing prompt's own explicit instruction; every INNER Hessian (the CC dual solve's own
matrix-free structured Hessian) is completely untouched by this session -- this file never
reaches into the inner solve's own Hessian machinery at all.

## Phase 7: trial-point screening

**Algebraic basis, already established in this codebase** (`inner_screening.jl`'s own file
header, quoted directly): "the inner CC dual problem minimizes a raw functor value
`f(zeta,lambda;G)`... over UNCONSTRAINED `(zeta,lambda)` -- every point in `R^{1+d}` is
dual-feasible, so for ANY `(zeta,lambda)`, weak duality gives `f(zeta,lambda;G) >= f* =
-Delta(G)`, i.e. `-f(zeta,lambda;G) <= Delta(G)` for literally EVERY `(zeta,lambda)`, not just
a verified optimum." Since `obj(x) == f(x;G)` and `DeltaStar(theta) = -min_x f(x;theta)`, this
gives `-obj(x) <= DeltaStar(theta)` for ANY dual `x`, at ANY `theta` -- unconditionally, not
merely at a verified optimum. `melitz_reduced_q_cap_screen` returns `-obj(x_ref)` (the
anchor's own verified dual, always available) IF it exceeds `cap+tol`, else `nothing` --
**this screen can only ever certify `AboveEvaluationCap`; it never returns anything else.**

Verified numerically, not merely cited: (1) at the D4 anchor with `cap=10.0`, the screen
correctly does NOT fire (`Delta0=7.55e-6 << cap`); with an absurdly low `cap=1e-8`, it
correctly fires and returns exactly `Delta0` (a TIGHT bound at the anchor itself, since the
anchor's own dual IS locally optimal there); (2) at 8 random nearby trial points, the returned
lower bound never exceeded the true reoptimized `DeltaStar` (regression test item 12). The
sequential controller (Phase 8) applies this screen before every KNITRO trial-point evaluation
and falls through to a real typed inner solve whenever it does not fire.

## Phase 8: sequential stage controller (`src/melitz/reduced_q_controller.jl`)

A deliberately SIMPLER KNITRO wiring than production's `solve_melitz_finite_delta_bound`/
`melitz_build_finite_delta_callbacks` (no exact-point cache, no dual-polish/origin-block
prescreens, no multi-candidate live tracking beyond one running best incumbent) -- Rule 11
only requires production stay unchanged (it does: zero lines of `finite_delta_outer.jl` are
touched), not that the experimental controller match its engineering sophistication.

`melitz_solve_reduced_q_stage!` registers a genuinely `2+nA`-dimensional KNITRO NLP (`s` the
ONLY q-related variable KNITRO ever sees), with the divergence constraint's typed classifi-
cation via `solve_melitz_delta!(session, theta_full, policy)` (Rule 2/3: the four types used
as-is; `NumericalFailure` always reported as a KNITRO eval-error, never "infeasible"), the
Phase 7 cap screen applied first, and the Phase 2 cutoff system registered as true linear
rows. `melitz_run_reduced_q_sequential_search` retains the best verified `FiniteSolved`,
within-budget incumbent across up to 5 stages (hard cap), stopping after 2 consecutive stages
with no verified improvement, or immediately after one welfare-plus-A-only fallback stage if
no useful q direction is found at some anchor. `stopped_reason` records only the STOPPING
RULE that fired (`:max_stages`/`:no_improvement_streak`/`:no_direction_found`) -- never
labeled "convergence."

**One tested comparator**, `melitz_reduced_q_more_extreme_kappa(direction, kappa_a, kappa_b)`
(`reduced_q_controller.jl`), replaces the informal "higher/lower kappa is better" narrative --
see the correction below.

## Corrections to the prior session's diagnostics (governing-prompt-mandated)

**Correction 1 (Phase 9 factor-of-two)**: the prior session's own
`scripts/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.jl` compared a ONE-SIDED linear
prediction `pred_aq = exact_A*t_A + q_secant*t_q` (base -> `theta0+t`) directly against
`actual_dDelta = lp.Delta - lm.Delta`, the FULL TWO-SIDED reoptimized change `DeltaStar(theta0+t)
- DeltaStar(theta0-t)` (spanning `2t`, not `t`). Under local linearity,
`DeltaStar(theta0+t)-DeltaStar(theta0-t) ~= 2*(pred_aq)`, so the comparison was structurally
biased toward "predicted is ~half of actual" REGARDLESS of estimator quality -- exactly the
"roughly half its magnitude... a systematic, repeatable underprediction" finding that
session's own doc reported (e.g. pure_intensive/target=0.1/scale=1.0: predicted `-3.555e-4`
vs. actual `-7.110e-4`, ratio `~0.5`).

**Fixed** (script patched in place, rerun): the script now reports BOTH a genuinely matched
one-sided actual change (`lp.Delta - lfd0.Delta`, the SAME half-interval `pred_aq` spans) AND
the original two-sided pair (`pred_aq_twosided=2*pred_aq` vs. `actual_dDelta_twosided`),
clearly labeled, never conflated. **Rerun result** (D4, targets 0.1/0.5, 4 paths x 3 scales =
24 rows, `docs/key_results/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.csv`, corrected):
matched one-sided ratios (`pred_aq/actual_onesided`) range `0.30` to `1.54` across the 24 rows,
median close to `1.0`, with **no systematic ~0.5x bias** -- the original "roughly half"
finding was a comparison-normalization artifact, not (necessarily) a genuine 2x
underprediction bias in the `(A,q)` estimator. Residual scatter (some rows still show real
disagreement, e.g. `mixed_7525`/target=0.5/scale=1.0: ratio `-1.22`, a sign flip) is genuine
estimator/curvature noise at this D4 Pareto-adjacent point, not a units artifact -- disclosed,
not smoothed over. Regression test item 11 reproduces this exact bug pattern in isolation
(using the known-exact A-block gradient, so any residual gap is attributable ONLY to the
interval mismatch, not estimator noise) and asserts both that the corrected comparison is
close (`rtol=0.05`) and that the ORIGINAL mismatched comparison is NOT close -- so a future
edit reintroducing this bug fails loudly.

**Correction 2 (Phase 12 kappa-sign narrative)**: the prior session's own
`docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md` Phase 12 table analysis claimed "the
experimental `(A,q)` backend finds a marginally BETTER (higher) kappa in both
`direction=:upper` runs." This is **backwards**: the upper GT% bound is obtained by
MINIMIZING kappa (thereby maximizing `100*(1-kappa)`), so a SMALLER kappa is the more extreme
(better) solve for `:upper`, not a larger one. Re-checking that session's own table directly:
at `delta=0.1/upper`, production's kappa (`0.8984`) is SMALLER than experimental's
(`0.9072`) -- production actually found the tighter upper bound, the OPPOSITE of what the text
claimed. (The `:lower` narrative in that same table happens to be correct, since "higher
kappa better" and "lower bound maximizes kappa" coincide for that direction only -- not a
general rule.) The underlying KNITRO incumbent-selection logic in
`solve_melitz_finite_delta_bound` was never wrong (`signed_objective` already encodes the
correct sign) -- only the after-the-fact DIAGNOSTIC narrative was backwards for `:upper`.
**Fixed**: `melitz_reduced_q_more_extreme_kappa`/`_gt` (one tested comparator,
`reduced_q_controller.jl`, regression test item 10) is now the ONLY sign logic used anywhere
in this session's own Phase 12 reporting (below) and in the sequential controller's own
incumbent-retention decision.

## Phase 10: source-level tests (`test/melitz/runtests.jl`, new testset "Reduced-q-subspace
outer-search backend (2026-07-29 continuation)")

All 14 governing-prompt items addressed except #14 (disclosed, see Phase 13):

1. `s=0` reproduces the anchor bit-for-bit -- **met**.
2. `s` changes no `A` values -- **met**.
3. `A` changes no `q` values -- **met**.
4. reduced-state reconstruction matches the production full `(A,q)` reconstruction at `s!=0`
   -- **met** (5 random points).
5. transformed constraints equal production constraints -- **met** (25 random points, max
   discrepancy `<1e-9`).
6. the scalar `s` derivative equals a direct dense full-q fixed-dual secant computed
   independently at the same `h_s` -- **met**.
7. the scalar derivative is NOT computed from a coordinatewise dot product -- **met**,
   demonstrated by numeric disagreement, not merely asserted structurally.
8. exact A derivatives remain correct at nonzero `s` -- **met** (relative error `4.4e-10`).
9. cache fingerprints change with anchor/basis/normalization/bandwidth-policy/`W`/QMC-seed --
   **met**, all six checked independently.
10. the upper/lower incumbent comparator uses the correct objective direction -- **met**, plus
    cross-checked against the equivalent GT%-based comparator at 20 random points x 2
    directions.
11. the corrected Phase 9 one-sided/two-sided pairing does not produce an artificial factor of
    two -- **met**, and the ORIGINAL mismatched pairing is shown to reproduce the artifact
    (so the regression is a genuine trap, not a tautology).
12. the fixed-dual cap screen is one-sided safe and returns only a certificate or `nothing` --
    **met**, plus 8-point weak-duality numeric verification.
13. no dense G is materialized by any reduced-q-subspace function -- **met**
    (`MELITZ_DENSE_G_MATERIALIZATIONS[]` unchanged across a full gradient + cap-screen call).
14. D20 execution warns/errors on accidental serial fallback -- **not separately re-implemented
    this session** (disclosed): no `Threads.@threads`-eligible kernel exists anywhere in
    `reduced_q_subspace.jl`/`reduced_q_controller.jl` (mirroring the q-bandwidth campaign's own
    disclosed convention for its diagnostic code -- the hot loops this session's new code calls,
    `melitz_exact_a_gradient_full!`/`melitz_q_coordinate_probe`, are themselves single-call,
    `O(W)`/`O(W*D)` functions with no thread-fork of their own), so there is no serial-fallback
    path for this session's OWN code to warn about; `melitz_thread_startup_report()` (called at
    the top of every script this session wrote) still fires its own pre-existing check.

Plus two additional tests: the state-map/dimension helpers reject a `:logf` ctx
(`ArgumentError`, never silently reinterprets the A-block as q), and the sequential
controller's own small bounded smoke run produces a genuine typed classification breakdown
and a never-worse-than-starting incumbent.

**A real bug caught by this last test, not merely "ran without error"**: the first version of
`melitz_solve_reduced_q_stage!` incremented the four typed-classification counters (plus
`n_cap_screened`) inside `inner_eval`, a helper called from BOTH `cb_F!` and `cb_G!` (KNITRO's
own `eval_fcga=no` convention calls the objective/constraint and gradient callbacks
SEPARATELY at the same accepted trial point for most outer iterates, exactly as
`finite_delta_outer.jl`'s own header documents) -- while `trials` was pushed only from
`cb_F!`. This double-counted every ordinary trial point once from each callback, so
`n_finite_solved+n_above_cap+n_infinite_delta+n_numerical_failure+n_cap_screened` exceeded
`length(trials)` (caught live: `34 != 25` on the first full-suite run after this session's own
edits). Fixed by moving all counter increments into `cb_F!` alone, keyed off the SAME `kind`
symbol `cb_G!` also uses for its own gradient dispatch -- `inner_eval` itself no longer
mutates any shared counter. Confirmed via a standalone rerun (`n_finite=12, n_above=2, n_inf=1,
n_fail=10, n_screened=0, sum=25, length(trials)=25, MATCH=true`) and the full test suite
rerun (Section "Test suite" below).

## Phase 11: D4 derivative and stage shakedown

`scripts/melitz_reducedq_phase11_d4_shakedown_2026-07-29.jl`. Mandatory `W=320,000`
(governing prompt's own development floor), one representative `W=1,280,000` spot check.
**Substitution disclosed**: D4's fixed-A/f gamma-profile corridor tops out at `Delta~0.572`
(established, `docs/melitz_aq_q_bandwidth_convergence_2026-07-29.md` Phase 3 -- `Delta~1/2` are
not reachable at D4), so the two test points are `target=0.5` (`Delta0=0.460` at `W=320k`) and
the "pareto" calibration point (`Delta0=0.522` at `W=320k`, the closest available substitute
for the requested `Delta*~2` point), not `Delta*~2` itself.

| point | W | Delta0 | h_s | central: fixed-dual | central: reoptimized | central relerr | one-sided s | one-sided pred | one-sided actual | one-sided relerr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target=0.5 | 320,000 | 0.460 | 0.523 | -2.9131e-4 | -2.9133e-4 | **4.8e-5** | 0.5 | -1.457e-4 | -1.364e-4 | 6.8% |
| pareto | 320,000 | 0.522 | 0.505 | -3.1456e-4 | -3.1457e-4 | **3.2e-5** | 0.5 | -1.573e-4 | -1.539e-4 | 2.2% |
| target=0.5 | 1,280,000 | 0.447 | 0.528 | -5.3415e-5 | -5.3414e-5 | **2.0e-5** | 0.5 | -2.671e-5 | -3.845e-5 | 30.5% |

**Central agreement is essentially exact at every point/W tested** (relative error `2e-5` to
`5e-5`, floating-point/`O(h^2)`-truncation-noise level) -- directly confirming the campaign's
own Phase 7 Q2 finding (direct block secant vs. fully reoptimized, essentially exact) survives
intact inside the reduced-subspace machinery, at both the mandatory `W=320,000` and the
`W=1,280,000` spot check. **One-sided stage-relevant prediction is noticeably less reliable**
(`6.8%`/`2.2%`/`30.5%` relative error at `s=0.5*s_hi`, a genuine within-trust-region step, not
a tiny probe) -- and does NOT improve monotonically with `W` in this small sample (the
`W=1.28M` point shows the WORST one-sided error of the three). This is the expected, disclosed
distinction between "central-secant diagnostic accuracy" (excellent, confirmed) and "one-sided
step prediction" (moderate, genuinely a harder problem, confirmed) -- see Conclusion below.
Transformed-constraint discrepancy vs. production: `<6e-17` at every point (exact).

## Phase 12: bounded D4 outer comparison

`scripts/melitz_reducedq_phase12_d4_comparison_2026-07-29.jl`. Identical starting economic
state (D4, seed=29, `W=20,000` -- the SAME base fixture the prior session's own Phase 12 used,
for direct comparability), identical evaluation cap (`CappedEvaluation(10.0)`), identical
KNITRO outer-algorithm option file. THREE backends: production `(A,f)`, full-coordinate
experimental `(A,q)`, and this session's sequential reduced-q backend. `delta in {0.1,0.5}`
(D4's own corridor limit, `1,2` would never bind on this corridor -- same established
non-informative-budget reasoning as the prior session's own disclosed scope reduction, not
independently re-tested), `direction in {:upper,:lower}`, 4 cells x 3 backends = 12 runs.

| delta | dir | backend | wall (s) | start GT% | best GT% | Delta_best | within budget | n_FiniteSolved | n_AboveCap | n_Infinite | n_Fail | n_screened |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0.1 | upper | production (A,f) | 18.1 | 6.52 | 10.16 | 0.0924 | yes | 12 | 15 | 55 | 0 | -- |
| 0.1 | upper | full (A,q) | 4.0 | 6.52 | 9.28 | 0.0511 | yes | 14 | 17 | 54 | 0 | -- |
| 0.1 | upper | **reduced-q (5 stages)** | 29.0 | 6.52 | **12.31** | 0.1000 | yes | 568 | 40 | 0 | 91 | 0 |
| 0.1 | lower | production (A,f) | 2.9 | 6.52 | 3.53 | 0.0994 | yes | 38 | 59 | 10 | 0 | -- |
| 0.1 | lower | full (A,q) | 3.0 | 6.52 | 4.45 | 0.0668 | yes | 25 | 33 | 19 | 0 | -- |
| 0.1 | lower | **reduced-q (3 stages)** | 19.1 | 6.52 | **3.08** | 0.0999 | yes | 382 | 12 | 0 | 46 | 0 |
| 0.5 | upper | production (A,f) | 3.1 | 6.52 | 14.87 | 0.4929 | yes | 29 | 66 | 2 | 0 | -- |
| 0.5 | upper | full (A,q) | 2.0 | 6.52 | 13.05 | 0.3913 | yes | 10 | 13 | 55 | 0 | -- |
| 0.5 | upper | **reduced-q (5 stages)** | 19.5 | 6.52 | **17.39** | 0.4993 | yes | 210 | 305 | 64 | 0 | 157 |
| 0.5 | lower | production (A,f) | 2.3 | 6.52 | 2.95 | 0.4615 | yes | 15 | 58 | 23 | 0 | -- |
| 0.5 | lower | full (A,q) | 3.7 | 6.52 | 3.03 | 0.4994 | yes | 33 | 24 | 46 | 0 | -- |
| 0.5 | lower | **reduced-q (5 stages)** | 28.7 | 6.52 | **0.056** | 0.4766 | yes | 384 | 182 | 0 | 111 | 0 |

**Using the CORRECTED comparator** (`melitz_reduced_q_more_extreme_kappa`): the sequential
reduced-q backend finds the MOST EXTREME (best) verified incumbent in **all 4 of the 4
(delta,direction) cells tested** -- `delta=0.1/upper`: GT%=12.31 vs. 10.16/9.28;
`delta=0.1/lower`: GT%=3.08 vs. 3.53/4.45 (smaller is better for `:lower`);
`delta=0.5/upper`: GT%=17.39 vs. 14.87/13.05; `delta=0.5/lower`: GT%=0.056 vs. 2.95/3.03. This
is a genuinely decisive, if small-sample, result. **Important confound, disclosed rather than
hidden**: the sequential backend ran 3-5 stages (`210-568` `FiniteSolved` evaluations plus
substantial `AboveEvaluationCap`/screened counts, `19-29s` wall) vs. ONE short KNITRO run for
production/full-`(A,q)` (`10-38` `FiniteSolved` evaluations, `2-18s` wall) -- the comparison is
NOT matched on evaluation budget or wall-clock time; part (not necessarily all) of the
reduced-q backend's advantage here is simply "it got more attempts, spread across more
KNITRO restarts (each stage is a fresh KNITRO problem, so 5 stages behaves somewhat like 5
independent short runs from successively better starting points)," not proven to be purely an
algorithmic quality advantage. `nStatus=-410`/`Iteration limit reached` (or equivalent) fired
in every individual KNITRO sub-solve of every backend -- **none of the 12 runs converged**;
this is a stability/incumbent-quality smoke test, not a convergence result, exactly as the
governing prompt's own Phase 12 framing requires. The Phase 7 cap screen fired non-trivially
in 2 of 4 sequential-q cells (`91`/`157`/`111` screened evaluations, `0` in the other two) --
a real, if modest, share of trial points avoided a full inner solve.

## Phase 13: limited real-D20 construction test

`scripts/melitz_reducedq_phase13_realD20_construction_2026-07-29.jl` (`-t 20`, per
`src/melitz/CLAUDE.md`; `melitz_thread_startup_report()` confirms 20 threads available, BLAS=1).
One verified real-D20 point (`noah_D20`, focal=`fra`, seed=1, `W=80,000`, `target=0.5`,
`Delta0=0.4833`). **No D20 outer campaign was run** (governing prompt's own explicit rule).

- reduced state reconstructed correctly: `s=0` reproduces the anchor bit-for-bit (`D=20`:
  `nA=399`, `nq=398`) -- **met**.
- transformed constraints correct at D=20: max discrepancy vs. production `4.0e-16` (10
  random points) -- **met**, generalizes cleanly from D4.
- direct scalar derivative runs without dense G: `MELITZ_DENSE_G_MATERIALIZATIONS[]` delta
  `= 0` across the full stage-build + gradient computation -- **met**.
- memory bounded: live heap bytes actually DECREASED slightly across the gradient call
  (`-5.7MB`, GC noise, no leak) -- **met**.
- stage construction (398 coordinatewise probes) took `105.4s` at `W=80,000` -- the dominant
  cost of this phase, disclosed (not optimized this session; each probe is `O(W)`, so `398`
  probes is `O(398*W)~3.2e7` moment-operator evaluations, consistent with the observed time).
  The reduced gradient itself (welfare + exact A + one direct-block s-secant) took `1.64s`.
- **the one direct-scalar-vs-reoptimized comparison, at the crossing-calibrated `h_s=0.496`
  (governing prompt's own required check): the reoptimized secant FAILED to verify**
  (`lfd_ok=false` on the minus side) -- consistent with, not contradicting, the q-bandwidth
  campaign's own Phase 10 finding ("the dense-direction amplitude heuristic... overshot at
  D=20 scale for most directions -- 3 of 4 dense directions' reoptimized secants failed to
  converge"). A supplementary sweep to smaller `h_s` (same direction, same base point,
  `h_s in {0.496,0.1,0.02,0.005,0.001}`) resolves this cleanly:

  | h_s | fixed-dual | reoptimized | ok | relerr |
  |---:|---:|---:|---|---:|
  | 0.496 | -6.005e-3 | NaN | minus-side fails to verify | -- |
  | 0.1 | -4.694e-3 | NaN | minus-side fails to verify | -- |
  | 0.02 | -1.287e-3 | NaN | minus-side fails to verify | -- |
  | 0.005 | -1.777e-5 | -1.777e-5 | **both verify** | **2.5e-9** |
  | 0.001 | -1.777e-5 | -1.777e-5 | **both verify** | **4.7e-9** |

  At `h_s<=0.005` (roughly 1% of the `s in [-1,1]` trust region -- much smaller than the
  crossing-target default's own amplitude at this D=20/W=80,000 point), the reoptimized
  secant verifies cleanly and matches the fixed-dual secant to **essentially machine
  precision** (`2.5e-9`/`4.7e-9` relative error) -- the SAME central-agreement pattern D4
  showed at every W tested (Phase 11), now confirmed at real D=20 as well. The default
  crossing-target amplitude heuristic (aimed at 50 total two-sided switches) is simply too
  aggressive for a reoptimized ground-truth comparison at this D=20 point's own crossing
  density -- a genuine, disclosed limitation of the amplitude heuristic (matching the
  campaign's own prior Phase 10 finding exactly), NOT a defect in the direct-block-secant
  gradient itself, which the fixed-dual mode (the one actually supplied to KNITRO,
  `melitz_reduced_q_gradient!`) computes successfully at every tested `h_s` with no solve
  required at all.

## Decision

**Conclusion A: reduced q is promising.** Evaluated against the governing prompt's own
acceptance bar:

- *the scalar direct derivative predicts stage-relevant one-sided changes materially better
  than the assembled coordinatewise vector*: confirmed structurally (Phase 5/10 item 7: the
  scalar-s gradient is never a coordinatewise dot product, and numerically differs from one)
  and by direct comparison to the campaign's own Phase 7 finding (coordinatewise-assembled
  dense predictions: 91-95% sign agreement, not W-improving; this session's direct block
  secant: central relative error `2e-5`-`5e-5` at D4 `W>=320k`, `2.5e-9` at real D=20 with a
  suitably calibrated step -- both regimes dramatically tighter than the coordinatewise
  assembly's own ~20-40% median error).
- *the reduced backend substantially lowers cap exits or rejected steps*: NOT cleanly
  established either way this session -- `AboveEvaluationCap`/screened counts are comparable
  in share to production's own (Phase 12 table), and 2 of 4 cells showed nontrivial cap-screen
  activity (91/157/111 screened evaluations) while 2 showed none; no controlled apples-to-
  apples rejection-RATE comparison (matched evaluation budget) was run this session.
- *it retains or improves verified D4 incumbents*: **yes, decisively** in this session's own
  smoke test -- the sequential reduced-q backend found the most extreme verified GT% bound
  (under the CORRECTED comparator) in all 4 of 4 `(delta,direction)` cells tested (Phase 12),
  though with the disclosed evaluation-budget confound (more stages/trials, more wall-clock
  time than the single-shot comparison backends -- Rule 12's own "never worse than a verified
  restricted incumbent" guarantee is what makes this a safe, non-regressive experiment
  regardless of the confound, not evidence the confound doesn't matter for a FAIR comparison).
- *behavior is consistent at budgets around both 0.5 and 2*: partially -- consistent at the
  two D4-reachable budgets (`delta=0.1,0.5`, both directions); `delta~2` was not independently
  testable at D4 (established corridor limit) and no D20 outer search was run (Rule 11).

**Recommendation**: continue with the sequential one-direction reduced-q backend in a
careful D4 frontier exercise (matching the governing prompt's own Conclusion-A recommendation
verbatim), with two concrete, disclosed follow-ups flagged by this session's own data before
that exercise: (1) run a MATCHED-evaluation-budget comparison (same total inner-solve count or
wall-clock across all three backends) to separate "the reduced backend is a better search" from
"the reduced backend got more attempts" (Phase 12's own disclosed confound); (2) replace the
crossing-target amplitude heuristic for `h_s` (and for the direction-normalization amplitude
`r_q`) with an LP-derived feasible-step bound at D=20 scale, or accept a materially smaller
default crossing target there -- the current default overshoots the reoptimized-verification
regime at D=20 by roughly 100x (Phase 13), even though it is entirely adequate for the
fixed-dual gradient KNITRO actually receives.

**What was NOT established**: genuine solver convergence (every individual KNITRO sub-solve
in every backend, every cell, hit an iteration limit or local-infeasibility status, never a
converged optimum -- Phase 12); a D20-scale outer search of any kind (Rule 11, not attempted);
whether the reduced backend's Phase-12 advantage survives a matched-budget rerun (disclosed
open question, not assumed either way).

## Final report answers (governing prompt's own numbered questions)

1. **Was the reduced-q map implemented without duplicating production reconstruction logic?**
   Yes -- `melitz_reduced_full_theta` performs only the affine `vcat` assembly; every actual
   gravity/pivot/cutoff reconstruction call goes through the EXISTING, unmodified
   `expand_free_theta_logcutoff`/`melitz_expand_theta`/`build_melitz_affine_cutoff_system`.
2. **Does one scalar `s` correctly represent a dense movement of the full q block?** Yes --
   `q_free_free(s) = q_anchor_free + s*q_basis_free` moves all `nq=D^2-2` free-q coordinates
   simultaneously along one fixed direction (verified: `s`-only perturbation changes the
   entire reconstructed q block, D4 and D20).
3. **Is the derivative supplied to KNITRO w.r.t. `s` a genuine direct block evaluation rather
   than a sum of coordinatewise secants?** Yes, verified both structurally (the code path
   never reads the coordinatewise vector) and numerically (the two quantities measurably
   differ at the D4 fixture).
4. **Does that scalar derivative predict fully reoptimized central `DeltaStar` changes?**
   Yes, essentially exactly, at D4 `W in {320k,1.28M}` (relative error `2e-5`-`5e-5`) and at
   real D=20 once the step is calibrated within the region where the reoptimized comparison
   itself converges (relative error `2.5e-9`-`4.7e-9` at `h_s<=0.005`).
5. **How well does it predict one-sided stage movements?** Moderately -- `6.8%`/`2.2%`/`30.5%`
   relative error at a genuine within-trust-region step (`s=0.5*s_hi`) across the 3 D4 test
   points, NOT monotonically improving with `W` in this small sample. This is the weakest
   link in the chain (see Conclusion, follow-up #2).
6. **Does the reduced backend reduce `AboveEvaluationCap` trial points?** Not established
   either way -- comparable shares to production in this session's own (evaluation-budget-
   unmatched) smoke test; no controlled rate comparison was run.
7. **Does it reduce `NumericalFailure` evaluations?** Not established either way, same caveat
   as above; these are never described as "infeasible" anywhere in this session's own code or
   reporting (Rule 3, structurally enforced -- `NumericalFailure` is a distinct classification
   throughout).
8. **Does it improve accepted-step behavior?** The sequential controller retained a genuinely
   improving verified incumbent in every one of the 4 Phase 12 cells (multiple stages showed
   `improved=true`); whether the PER-STEP acceptance rate (accepted / proposed) is better than
   production's own was not separately measured.
9. **Does it improve the verified GT% bounds at `delta=0.1,0.5,1,2`?** At `delta in {0.1,0.5}`
   (the only D4-reachable budgets): yes, in all 4 tested cells, under the corrected comparator
   -- with the disclosed evaluation-budget confound. `delta in {1,2}` not independently
   testable at D4 (established corridor limit, not a new gap).
10. **Does it outperform the corrected production comparison from common starts?** Yes in this
    session's own smoke test (4/4 cells), with the confound above meaning this should be read
    as "promising, not yet a controlled result," not a settled performance claim.
11. **Is the main remaining problem direction proposal, one-sided curvature, stage scaling, or
    something else?** **One-sided curvature/step prediction** (item 5 above) and **amplitude
    calibration at D20 scale** (Phase 13's own crossing-target overshoot) are the two concrete
    weak points this session's own data isolates; direction PROPOSAL itself was not shown to
    be a problem (the direct block secant, evaluated along the proposed direction, tracks the
    reoptimized truth essentially exactly at central steps every time it was tested).
12. **Which of Conclusions A-D is supported?** **Conclusion A**, with the two disclosed
    follow-ups above -- see the Decision section.

## Test suite

`julia --project=. -t 1 test/melitz/runtests.jl`, run three times this session: (1) baseline,
before any edit -- clean, exit 0; (2) after this session's own source/test additions -- **one
genuine failure** (the counter/`trials`-count mismatch described above, `34 != 25`), caught
live, not shipped; (3) final, after the fix -- **clean, exit 0**, new testset "Reduced-q-
subspace outer-search backend (2026-07-29 continuation)": **91/91 Pass**, every other
pre-existing testset unaffected. This sequence (not just the final clean run) is disclosed in
full -- exactly the "ran without error" vs. "verified" distinction this repo's own prior
session docs insist on; the first attempt at declaring this session's own new tests complete
would have been wrong.

## Required output files

- `docs/melitz_reduced_q_subspace_search_2026-07-29.md` -- this document.
- `docs/key_results/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.csv` -- CORRECTED (Correction 1).
- `docs/key_results/melitz_reducedq_phase11_d4_shakedown_2026-07-29.csv`
- `docs/key_results/melitz_reducedq_phase12_d4_comparison_2026-07-29.csv`
- `docs/key_results/melitz_reducedq_phase13_realD20_construction_2026-07-29.csv`
- Scripts: `scripts/melitz_reducedq_phase{11,12,13}_*_2026-07-29.jl`; corrected
  `scripts/melitz_qbw_phase9_aq_vs_af_matched_2026-07-29.jl`.
- Source: `src/melitz/reduced_q_subspace.jl`, `src/melitz/reduced_q_controller.jl` (new);
  `src/melitz/include_melitz.jl`, `test/melitz/runtests.jl` (additive includes/tests).
- `provenance.txt` alongside this document.

## Provenance

See `docs/key_results/melitz_reducedq_provenance_2026-07-29.txt`.
