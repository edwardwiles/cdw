# Melitz production-fast backend: matrix-free port to production (2026-07-26 continuation)

Continuation of `docs/melitz_production_port_handoff_2026-07-26.md`. That document is the
handoff written at the end of the prior session: what existed (validation-only matrix-free
operator + freestanding KNITRO driver), what remained (making it the actual production
default), and the concrete plan. This document reports what this session did against that
plan: the permanent `MelitzCCBundle`, the defaults flip, the correctness regressions, and
the benchmarks -- all against the ACTUAL production entry points, not a validation-only
harness.

## 1. Final production call graph

- `build_melitz_psi_bundle` (`delta_star.jl`) and `build_melitz_psi_bundle_from_calibration`
  (`pareto_calibration.jl`) now default to `backend=:matrix_free`: they construct a
  `MelitzSortedTailContext` + `MelitzMomentOperator` + `MelitzCCBundle` (`mode=:delta`)
  instead of `PsiObjectiveBundleDelta`. `backend=:dense_reference` remains fully supported
  and unchanged.
- `build_melitz_implicit_bundle` (`finite_delta_outer.jl`) defaults to `backend=
  :auto_from_gradient_backend`, which resolves to `:matrix_free` unless the caller's own
  `gradient_backend` names one of the legacy dense-only families (`:B`, `:B_localized`,
  `:B_localized_parallel`, `:B_argument_localized_serial`, `:B_argument_localized_parallel`,
  `:D`), in which case it resolves to `:dense_reference` -- a caller explicitly asking for a
  dense-only gradient mechanism gets the bundle that mechanism actually needs, rather than
  silently getting a matrix-free bundle whose functor throws on the first jac_h call.
  `gradient_backend` itself now defaults to `:auto`, which resolves to
  `:B_direct_argument_sorted_parallel`/`_serial` (matrix-free / sorted-context present) or
  `:B_direct_argument_parallel`/`_serial` (plain direct), by `D`/thread-count, rather than
  the old unconditional `:B`.
- `melitz_classified_inner_solve` (`inner_screening.jl`) -- the function the ENTIRE finite-
  delta outer NLP's `cb_F!`/`cb_G!` route every inner evaluation through -- now calls two
  new Melitz-owned, bundle-agnostic dispatch functions instead of hardcoding
  `CS.select_G_from_H`/`obj.moments!`/`CS.inner_loop_internal`:
  `melitz_bundle_prepare_at_theta!(obj, theta)` and `melitz_bundle_inner_solve!(obj, theta)`
  (`cc_bundle.jl`). Each has a generic (untyped `obj`) method preserving the EXACT legacy
  dense behavior, and a specific `obj::MelitzCCBundle` method that updates the matrix-free
  operator / runs the Melitz-owned KNITRO driver. The `:B` range screen is skipped (not
  silently downgraded) when no dense `G` exists; the two functor-based screens (stored-dual,
  dual-polish) work on both bundle types unchanged, since they only ever call the functor.
- The FC-to-GA exact-point cache (`MelitzExactPointCache`, `finite_delta_outer.jl`) now
  stores either a dense `H` copy (legacy) or a `MelitzOperatorSnapshot` (new, `cc_bundle.jl`
  -- copies of the operator's small `coef`/`lambda`/`order`/`rank`/`bin`/`ell` fields, `O(W*D)`,
  strictly SMALLER than the dense `O(W*D^2)` entry it replaces) via
  `melitz_heavy_snapshot`/`melitz_heavy_restore!`/`melitz_heavy_bytes`/`melitz_heavy_recompute`
  dispatch. A `cb_G!` call at a theta `cb_F!` just solved restores the operator with ZERO
  re-equilibration (no `melitz_expand_theta`/`melitz_baseline_cutoff`/
  `melitz_update_moment_operator!` call) -- confirmed live (Section 5 below,
  `FC_to_GA_cache_hits == n_ga_calls` in the D=20 live campaign).
- `direct_gradient.jl`'s `_base_arg0!`/`_direct_coordinate_grad` and
  `sorted_crossing_gradient.jl`'s `_direct_coordinate_grad_sorted` gained new, MORE SPECIFIC
  `obj::MelitzCCBundle` methods (`cc_bundle.jl`) that reconstruct the base column value from
  operator fields (`_melitz_op_gbase`, `op.ell`) instead of reading `obj.H`. The generic
  (dense) methods are untouched.
- `melitz_recover_lfd`/`melitz_recover_lfd_from_solution`/`evaluate_melitz_delta`'s
  `store_G` path (`delta_star.jl`) gained `MelitzCCBundle` methods using `mul_G!`/`mul_Gt!`
  (never a dense `G`) for the primary path; `melitz_dense_G_from_operator` (diagnostic-only,
  never on a production-fast hot path) exists for a caller that explicitly wants a real
  dense `G` back from a matrix-free bundle.
- `fstar_direct.jl`'s `fstar_equal_weight_moments` gained a `MelitzCCBundle` method that
  computes `mean(G,dims=1) = G'*(ones(W)/W)` via a single `mul_Gt!` call -- genuinely
  matrix-free, not a dense-G-then-average shortcut.

## 2. Backend configuration and counters

`MelitzBackendConfig` (`backend_config.jl`, new) is an explicit, immutable record of every
backend choice (`inner_backend`, `moment_backend`, `outer_gradient_backend`,
`hessian_backend`, `screening_backend`, `cache_policy`, `evaluation_cap`,
`forbid_dense_fallback`). `:auto` fields resolve via `MELITZ_AUTO_PARALLEL_D_THRESHOLD=10`
(parallel iff `D>=10 && Threads.nthreads()>1`) -- derived from this session's own D=4 vs
D=20 benchmarks (Section 4-5): parallel variants win decisively at D=20, and D=4 is too
small a fixture to have been benchmarked as a parallel win, so the threshold sits strictly
between the two, closer to the D=20 side. `melitz_print_backend_summary(cfg, D)` prints the
resolved (not raw `:auto`) choice.

Eighteen global `Ref(0)` counters (`MELITZ_SORTED_MOMENT_CALLS`, `MELITZ_MATRIX_FREE_*`,
`MELITZ_DENSE_*`, `MELITZ_EXACT_POINT_CACHE_HITS`, `MELITZ_FC_TO_GA_CACHE_HITS`,
`MELITZ_OPERATOR_REBUILDS`, ...) mirror this codebase's own established global-counter
convention (`cc_algo`'s `INNER_SOLVE_COUNT`). `melitz_backend_counters_reset!()`/
`_snapshot()` give a caller an explicit before/after delta.
`melitz_check_no_dense_fallback!(cfg, counter_ref, call_site)` throws immediately when
`cfg.forbid_dense_fallback` and a dense counter would otherwise silently increment.

## 3. Melitz-owned production CC bundle (`cc_bundle.jl`, new)

`MelitzCCBundle` (`mode=:delta` or `:implicit`) replaces `PsiObjectiveBundleDelta`/
`PsiObjectiveBundleImplicit` for the production-fast path. No `cc_algo` type is subtyped, no
`cc_algo` generic function is extended -- own functor, own KNITRO driver
(`melitz_cc_inner_loop_knitro!`/`melitz_cc_inner_loop_internal!`/`melitz_cc_inner_loop`,
mirroring `inner_loop_KNITRO`/`inner_loop_internal`/`inner_loop`'s exact API sequence and
per-mode return convention -- ported, not called), own divergence conjugate
(`melitz_cc_Psi!`/`_dPsi!`/`_ddPsi!`, copied verbatim from `cc_algo/Psi.jl`), own
single-flight concurrency guard (`melitz_cc_guard_enter_inner_solve!`/`_exit_inner_solve!`,
independent state from `cc_algo/parallelism_guards.jl`'s own). Own KKT diagnostics capture
(`MELITZ_CC_LAST_OPT_ERR`/`_FEAS_ERR`, mirroring `cc_algo`'s `INNER_LAST_OPT_ERR`/
`_FEAS_ERR` but populated from Melitz's own `KN_get_abs_opt_error`/`_feas_error` calls, never
reading `cc_algo`'s globals).

Written duck-typed-compatible with every existing Melitz consumer: same field NAMES
(`γ`, `U`, `d`, `outer_constr_index`, `find_smallest`, `lower_limit`, `use_cached_x`, `x`,
`Psi!`, `threshold_crossed`/`threshold_crossing_*`) as the dense bundles. Two fields
(`needs_outer_moment_jacobian`, `jac_h`) are fixed at `false`/`zeros(0,0,0)` -- not a real
toggle, a truthful statement that this bundle has no dense jac_h theta-branch at all,
kept only so pre-existing consumers checking these fields see the honest answer instead of
a `FieldError`.

The functor's `(x, g, θ; jac, constr)` theta-gradient branch is NOT implemented (throws a
clear error on nonempty `θ`) -- confirmed unreachable on the production-fast path: the
default `gradient_backend` is always in the `:B_direct_argument_*` family, which bypasses
this branch entirely (`cb_G!` calls the direct-gradient function, never
`obj(x,g,theta;jac=...)`). The `constr` branch (used by `cb_F!` to read `Delta_theta`) IS
implemented (`constr[1] = -f*1e10`) -- the ONE genuinely-live sub-part of that branch for
Melitz's `outer_constr_index==d+1` convention; the "extra outer-constraint moments" GEMV
sub-part is correctly omitted as structurally vacuous (2026-07-26 handoff Section 5's own
finding, re-confirmed).

## 4. No-Ricardian-touch audit

`git diff --name-only` + `git status --porcelain` (new files): every changed/new file is
under `src/melitz/`, `test/melitz/`, or `docs/`. Zero diff in `cc_algo/`,
`production/fullA-exact/`, `full_aod_diag/`, or any other Ricardian path (`git diff --stat
cc_algo/` returns empty). `grep -rn "CounterfactualSensitivity\|PsiObjectiveBundle" cc_bundle.jl`
shows every code-level (non-docstring) hit confined to the GENERIC (untyped `obj`, dense-
fallback) methods -- legitimate CALLS into `cc_algo`'s pre-existing exported API
(`CS.select_G_from_H`, `CS.inner_loop_internal`), exactly matching what this codebase's
pre-existing dense consumers already did before this session. Zero new methods were added
to any `CounterfactualSensitivity`-namespaced function; zero cc_algo types subtyped.

Changed files (tracked): `src/melitz/delta_star.jl`, `src/melitz/finite_delta_outer.jl`,
`src/melitz/include_melitz.jl`, `src/melitz/inner_screening.jl`,
`src/melitz/pareto_calibration.jl`, `src/melitz/predictor_corrector.jl`,
`test/melitz/runtests.jl`. New files: `src/melitz/backend_config.jl`, `src/melitz/cc_bundle.jl`
(both new this session), plus the untracked `src/melitz/moment_operator.jl`/
`src/melitz/matrix_free_dual_solve.jl`/two docs from the PRIOR session's own working tree
(inherited, not created this session -- see the handoff doc's own Section 0).

## 5. Correctness regressions

### 5.1 Full existing test suite

48 top-level testsets, ~12,445+8,481+... assertions (same counts as the prior session's own
baseline run, reproduced at the START of this session before any edit), ALL GREEN against
the new matrix-free-by-default entry points -- required six iterations to reach (each prior
iteration surfaced one class of pre-existing test that assumed dense-bundle-specific
mechanics -- `obj.H` manipulation, `needs_outer_moment_jacobian`/`jac_h` field checks, a
`ctx.sorted_tail_ctx===nothing` assumption that no longer holds by default, a bare
`CounterfactualSensitivity.inner_loop_internal`/`inner_loop` call -- each fixed either by an
explicit `backend=:dense_reference` pin at the specific test (when the test's actual PURPOSE
is validating dense-only mechanics) or by a genuine additive fix (KKT diagnostics capture,
fixed `jac_h`/`needs_outer_moment_jacobian` fields, `moment_backend`/`gradient_backend`
auto-detection so an explicit legacy choice is honored rather than silently overridden).

### 5.2 Real D=20/W=80,000 inner-solve correctness (matrix-free vs dense, actual production
constructors)

| quantity | value |
|---|---:|
| objective (Delta) abs diff | `3.24e-17` |
| dual `x` max abs diff | `3.36e-10` |
| LFD weights max abs diff | `1.14e-17` |
| moment residual (matrix-free) | `2.34e-14` |
| moment residual (dense) | `1.92e-14` |
| dense wall (`melitz_recover_lfd`) | `9.70s` |
| matrix-free wall (`melitz_recover_lfd`) | `2.88s` |
| **speedup** | **`3.37x`** |

Both `nStatus==0`. Reached through the ACTUAL `build_melitz_psi_bundle_from_calibration`
constructor (`backend=:matrix_free` default vs `backend=:dense_reference`), not a
validation-only bundle.

Note this `3.37x` (not the prior session's own validation-harness `19.2x`) reflects a
different KNITRO options file (`melitz_inner_loop_options_capped_2026-07-24.opt`, this
session's own real-D20 fixture convention, vs whatever the validation harness used) and a
genuinely different number of KNITRO iterations on each side under that file -- not a
regression in the matrix-free arithmetic itself (Section 5.4's ablation below isolates the
per-iteration cost directly and shows a much larger ratio). Reproducing the exact prior
multiplier was not the goal here; demonstrating correctness and a genuine speedup through
the ACTUAL production entry point was.

### 5.3 Outer-gradient correctness (real D=20)

`melitz_classified_inner_solve` on the matrix-free Implicit bundle vs the dense one, at the
same theta: `Delta` diff `<1e-6` (both `FiniteSolved`). Sorted-parallel direct gradient
(matrix-free) vs plain dense-serial direct gradient (dense), at the two bundles' own
converged duals: gradients agree (max abs diff well inside the `1e10`-scaled tolerance
corresponding to `~1e-4` in real Delta-gradient units, consistent with two independently-
converged dual solves rather than one shared one).

### 5.4 Cumulative ablation (real D=20/W=80,000, complete inner-solve wall)

| Config | moment-build | inner-solve | **total** |
|---|---:|---:|---:|
| A: dense moments + dense inner solve | `2.04s` | `9.34s` | `11.39s` |
| B: sorted-tail-parallel moments + dense inner solve | `0.84s` | `4.48s` | `5.31s` |
| D: matrix-free operator + matrix-free inner solve | `0.28s` | `0.95s` | `1.23s` |

Moment-construction speedup (A→B): `2.45x`. Complete-FC speedup (A→D): **`9.28x`**.

Outer-gradient ablation, same fixture, at each config's own converged dual:

| Config | outer-gradient wall |
|---|---:|
| A: legacy `:B` (dense finite-difference of full moments, jac_h theta-branch) | **`2524.0s`** (42 min) |
| D: `:B_direct_argument_sorted_parallel` (production default) | `2.41s` |

**Speedup: `1049x`.** This is not a typo -- `:B` rebuilds the FULL `W x (D^2+1)` dense
moment matrix via finite differences for EVERY one of the `2D^2-2=798` free coordinates
(`~2s` per rebuild per Config A's own moment-build number, `x2` for `+h`/`-h`, `x798`
coordinates `~3192s` -- consistent with the measured `2524s`). This number is the single
clearest demonstration in this session of why flipping `gradient_backend`'s default away
from `:B` matters: at D=20 scale, the legacy default was not merely slower, it was
operationally unusable for a live outer optimization (a single GA call would eat 42
minutes).

### 5.5 Real D=20 live outer campaign (production-fast, actual `solve_melitz_finite_delta_bound`)

**CORRECTED 2026-07-26 (same-day, post-delivery) -- the first version of this section,
initially pushed to Dropbox, reported a run with a real configuration bug in the BENCHMARK
SCRIPT (not the production port) that made the campaign ~13.5x slower than it should be and
produced 31 spurious `NumericalFailure` results. Caught by direct user questioning ("this
should never happen -- what causes KNITRO to return a non-accepted status") rather than by
this session's own review. Root cause and corrected numbers below; the original (buggy)
numbers are kept, struck through, for the record.**

**Root cause**: `solve_melitz_finite_delta_bound`'s `delta_evaluation_cap` kwarg ALONE does
NOT activate the KNITRO-native `f <= obj.lower_limit` early-abort threshold inside the
functor -- that requires the SEPARATE `lower_limit_guard` (or `inner_solve_config`) kwarg,
which the first benchmark run never passed. Without it, `obj.lower_limit` stays at
`-KNITRO.KN_INFINITY` (`build_melitz_implicit_bundle`'s own `elseif lower_limit_guard===
nothing; lower_limit=-KNITRO.KN_INFINITY` branch) -- so a bad trial theta that SHOULD be
caught instantly by the crossing certificate instead runs the ENTIRE real KNITRO attempt to
`melitz_inner_loop_options_capped_2026-07-24.opt`'s own `maxtime_real=90` (a 90-SECOND
wall-clock cap per attempt) before giving up with no certificate at all -- classified
`NumericalFailure` by `melitz_classified_inner_solve`'s own (correct, pre-existing, and
deliberate) design, exactly as its docstring describes ("timeout is NumericalFailure unless
a valid lower-bound or infeasibility certificate already exists"). This is NOT a bug in the
matrix-free port -- the crossing mechanism (`MelitzCCBundle`'s own `threshold_crossed`/
`threshold_crossing_bound` fields) was already correctly implemented and works; confirmed by
a direct controlled test (20 identical perturbed trial thetas, same bundle, same everything
except `lower_limit_guard`): ALL 20 resolved as `NumericalFailure`-or-slow before the fix,
and ALL 20 resolved as fast `AboveEvaluationCap` certificates (`0.1-0.2s` each, mostly via
the free stored-dual-bank screen, no KNITRO call at all) after simply adding
`lower_limit_guard=1e-6` to the SAME call.

Real KNITRO outer optimization, `melitz_outer_finite_delta.opt` (`maxit=25`), `delta=1.0`,
`direction=:upper`, `delta_evaluation_cap=10.0`, `lower_limit_guard=1e-6` (the fix), real
D=20/W=80,000 calibration:

| | Original (buggy: no `lower_limit_guard`) | Corrected (`lower_limit_guard=1e-6`) |
|---|---:|---:|
| Wall clock | ~~`2062s` (~34 min)~~ | **`153s` (2.5 min)** |
| `n_fc_calls` / `n_ga_calls` | ~~`169` / `17`~~ | `173` / `17` |
| `n_inner_solved` | ~~`23`~~ | `17` |
| `n_above_cap_reject` | ~~`115`~~ | `156` |
| `n_numerical_failure_reject` | ~~`31`~~ | **`0`** |
| Points/minute | ~~`5.4`~~ | **`74.5`** |

**A `13.5x` wall-clock reduction and complete elimination of `NumericalFailure`, from fixing
a benchmark-script configuration bug, not the port itself.** Converged after 16 outer
iterations both times ("solution estimate < xtol for 3 consecutive iterations", a clean
stopping condition).

- **CORRECTED LABEL**: the value below is `g = theta_free[1] = log(gamma_prime_j)` (the raw
  outer-loop coordinate KNITRO actually optimizes, `MelitzOuterCandidate.objective`'s own
  definition, `finite_delta_outer.jl`) -- it is NOT the real gains-from-trade `kappa`/`GT_j`
  (user question, live session: "kappa is meant to live between 0 and 0.113" -- correctly
  flagged as impossible for the value below). The actual `GT_j = 1 - (w'/w)*exp(g)^(1/(sigma-1))`
  (`melitz_gains_from_trade`, `equilibrium.jl`) at this same point is `0.0203` (cross-checked
  directly against `acr_gains_from_trade`'s independent ACR sufficient-statistic formula to
  `~1e-11`, this codebase's own required agreement) -- comfortably in range. This was a
  mislabeling in this session's OWN reporting scripts (a same-named-sounding helper,
  `kappa_of_g` in `predictor_corrector.jl`, actually computes `(w'/w)*gamma_prime^(1/(sigma-1))`,
  the pre-"`1-`" term, not `GT` itself), not an error in the production economics.
- **`cold_verified` g EXACTLY EQUALS `best_live` g (`-0.4188822101422145`, both,
  UNCHANGED by the fix) -- this run found ZERO improvement over the cold-start incumbent,
  before or after.** Stated plainly (user question, live session): this is NOT a
  demonstration that the outer search itself works well -- it reproduces the same "mostly
  rejected trial points, no forward progress" pattern this repo's own prior sessions have
  already observed. This session deliberately did NOT investigate or redesign the outer
  coordinate-proposal/trust-region logic that produces this rejection rate (out of scope,
  per the governing prompt's own "do not begin... a new outer-algorithm redesign in this
  session"). The correct reading of this result is narrowly infrastructural: the
  matrix-free path runs a real, multi-iteration KNITRO outer optimization end-to-end without
  touching a dense fallback anywhere, at the expected speed, when correctly configured --
  not that it searches better.
- **Backend counters, corrected run**: `sorted_moment_calls=173`, `dense_moment_calls=0`;
  `matrix_free_objective_calls=1,950`, `matrix_free_gradient_calls=320`,
  `matrix_free_hessian_calls=256` (all far lower than the buggy run's `180,241`/`122,720`/
  `56,164` -- the buggy run's huge counts were themselves an artifact of many single inner
  solves each internally iterating for up to 90 seconds before giving up, not a sign of
  extra necessary work); `dense_inner_{objective,gradient,hessian}_calls=0`;
  `dense_G_materializations=0`; `production_dense_screen_calls=0`; `evaluation_cap_exits=34`
  (the KNITRO-native crossing certificate now genuinely fires, confirming the fix is real,
  not merely faster for an unrelated reason); `FC_to_GA_cache_hits=17` out of `17` GA calls
  (**100% hit rate**, unchanged by the fix -- every single GA call landed at a theta its own
  preceding FC call had just solved, reusing that exact operator state with zero
  re-equilibration); `operator_rebuilds=176`.

This is the acceptance-criteria-relevant result: an ACTUAL production outer optimization,
through the real driver, CORRECTLY CONFIGURED, with EVERY dense-path counter at exactly zero
and zero spurious numerical failures.

**Practical implication for any future caller of `solve_melitz_finite_delta_bound`/
`build_melitz_implicit_bundle`**: always pass `lower_limit_guard` (or `inner_solve_config`)
alongside `delta_evaluation_cap` -- passing `delta_evaluation_cap` alone silently leaves the
KNITRO-native early-abort disabled, exactly as `inner_solve_config.jl`'s own header already
warned (a distinct but related prior incident this session's own benchmark script managed to
reproduce anyway, because `solve_melitz_finite_delta_bound`/`build_melitz_implicit_bundle`
still allow the cap to be silently omitted -- only `solve_melitz_nuisance_min_delta` makes it
a mandatory, unomittable argument). Making this cap non-optional on `build_melitz_implicit_bundle`/`solve_melitz_finite_delta_bound` too would be a reasonable, low-risk
follow-up (not attempted this session -- flagged in Section 9).

## 5.6 D=4 outer-loop regression (dense vs production-fast, matched settings)

D=4 synthetic fixture (`seed=29`, `W=3,000`), `solve_melitz_finite_delta_bound`, `delta=1e-3`,
`direction=:upper`, identical `theta_init`/option files/evaluation cap on both sides
(`backend=:matrix_free` default vs `backend=:dense_reference` + `gradient_backend=
:B_direct_argument_serial` explicitly):

| | matrix-free (default) | dense-reference |
|---|---:|---:|
| `nStatus` | `-410` (iteration limit) | `-410` (iteration limit) |
| `n_fc_calls` | `85` | `63` |
| `n_ga_calls` | `26` | `26` |
| `cold_verified` g (= `log(gamma_prime_j)`, NOT kappa/GT -- see Section 5.5's correction) | `-0.045301157334125944` | `-0.045301157334127796` |
| wall clock | `11.9s` | `6.8s` (D=4/`W=3,000` is too small a fixture for the matrix-free
  operator's per-call overhead to pay for itself against a tiny dense `G` -- the real win is
  at D=20 scale, Section 5.4-5.5) |

Both sides hit the SAME KNITRO iteration limit (`maxit` in the default outer options file,
not tuned for this ad hoc smoke fixture) rather than converging -- expected for a quick
regression check, not a claim about this particular fixture's own economics. The
ECONOMICS agree to 7 significant figures (the `g` coordinate) despite the two backends
following different KNITRO iteration counts (`85` vs `63` FC calls) -- exactly the
"solver trajectories may differ slightly, economics must agree" acceptance criterion
(governing prompt Phase 15). **This run also did not pass `lower_limit_guard`** (the same
omission diagnosed in Section 5.5) -- its wall-clock numbers likely also understate the
achievable speed and are not being re-run/re-claimed here; only the (label-corrected)
economics-agreement conclusion is being asserted.

## 6. Allocation audit (real D=20/W=80,000, matrix-free bundle, post-JIT)

| callback | `@allocated` |
|---|---:|
| objective only | `176` bytes |
| objective + gradient | `144` bytes |
| objective + Hessian (`structured_parallel`, 16 threads) | `13,376` bytes |

Matches the prior session's own documented `~13KB` fixed, `W`/`D`-independent parallel-
Hessian task-spawn overhead almost exactly. Objective/gradient are effectively zero
(sub-200-byte residuals, not data-scaling).

## 7. Seed/W robustness

Attempted D=4 (`W=20,000`) at three ADDITIONAL seeds (`11`, `42`, `55`, all distinct from
`29`, the seed used throughout this repo's own D=4 test fixtures): all three hit a
PRE-EXISTING, unrelated `generate_fake_melitz_data` constraint (`export-selection
zhat[o,d]>=zhat[o,o] violated` -- the synthetic-fixture generator's own gravity/cutoff
feasibility check rejecting that random draw, nothing to do with this session's port).
`D=10, W=20,000, seed=29` succeeded and gives one genuine ADDITIONAL cross-check beyond the
D=4/seed=29 and real-D20 fixtures already used throughout Sections 5.2-5.5: matrix-free vs
dense agree to `1.38e-17` in `Delta` and `1.40e-14` in the dual.

This session's full correctness evidence spans THREE distinct `(D, data)` combinations
(D=4/synthetic/seed=29, D=10/synthetic/seed=29, D=20/real Noah data) -- not the "at least two
real-D20 QMC seeds" a genuinely exhaustive multi-seed sweep would need (Phase 18's full ask,
governing prompt). Finding additional D=4 seeds that pass the fixture generator's own
feasibility check would need either a seed search or a fixture-generator change, neither
attempted this session; flagged as a remaining gap (Section 9).

## 8. What's explicitly NOT ported this session (documented scope decisions, not oversights)

- `predictor_corrector.jl`'s `melitz_predictor_corrector_continuation` and
  `nuisance_profile.jl`'s `solve_melitz_nuisance_min_delta`/
  `melitz_build_nuisance_profile_callbacks` remain dense-bundle-only. Audited (Phase 8):
  neither is called by `solve_melitz_finite_delta_bound`, the main production outer driver
  -- both are separate, alternative outer-search strategies (Stage 2 profiling / predictor-
  corrector continuation). `predictor_corrector.jl`'s own internal bundle construction is
  now explicitly pinned to `backend=:dense_reference` (was implicitly dense before this
  session; now explicit and documented, not a silent default inheritance).
- `melitz_origin_block_screen` (a HiGHS-LP-based screen, default `false`, opt-in) remains
  dense-only -- not audited/ported this session.
- `melitz_range_screen` is skipped (not ported) for the matrix-free bundle -- a documented,
  deliberate choice (Section 1 above): it is a speed heuristic, not a correctness
  requirement, and a matrix-free equivalent would need either an `O(W*D)` histogram-based
  reimplementation (not attempted) or a dense `G` (defeats the point). Skipping it can only
  cost extra KNITRO attempts on already-infeasible points, never a wrong answer.
- The generic `cc_algo`-shared `melitz_calibration_outer_ctx`'s own `moment_backend` kwarg
  default was left at `:dense_reference` (unchanged) -- only the PRODUCTION entry point
  (`build_melitz_psi_bundle_from_calibration`) that wraps it was flipped, per this session's
  own `resolved_moment_backend` computation passed down explicitly.

## 9. Remaining gaps for a future session

- **`lower_limit_guard`/`inner_solve_config` is silently omittable on
  `build_melitz_implicit_bundle`/`solve_melitz_finite_delta_bound`, and omitting it produces
  a REAL, expensive, easy-to-hit failure mode** (Section 5.5's correction: `13.5x` slower,
  spurious `NumericalFailure`s, each burning the full `maxtime_real=90s` per bad trial
  point) -- confirmed live this session (caught by direct user questioning, not this
  session's own review). `solve_melitz_nuisance_min_delta` already makes this cap a
  mandatory, unomittable argument (`inner_solve_config.jl`'s own "impossible to omit"
  design, built after an earlier incident of the identical class); extending that same
  mandatory-cap requirement to `build_melitz_implicit_bundle`/
  `solve_melitz_finite_delta_bound` would close this gap for good rather than relying on
  every caller remembering to pass it. Not attempted this session (would require an
  API-breaking signature change touching ~90 existing test call sites, per
  `inner_solve_config.jl`'s own stated reason for not doing this originally) -- flagged as
  the single highest-value follow-up from this session's own experience.
- A genuine multi-seed REAL-D20 QMC sweep (Phase 18's full ask) -- only one real-D20 seed
  used throughout this session, matching every prior session's own precedent.
- A W=160,000 fixed-point check at real D=20 (Phase 18) -- not attempted (time budget).
- `predictor_corrector.jl`/`nuisance_profile.jl`'s own dense-only scope (Section 8) could be
  ported to the matrix-free bundle in a future session using the exact same dispatch pattern
  established here (`melitz_bundle_prepare_at_theta!`/`melitz_bundle_inner_solve!`).
- A genuine, from-scratch matrix-free reimplementation of `melitz_range_screen` (an `O(W*D)`
  histogram over `op.bin`, sketched but not implemented in Section 1's own commentary) would
  let production-fast runs recover this screen's speed benefit too.
- A stale note in `docs/melitz_matrix_free_inner_operator_2026-07-26.md` Section J ("Still
  not recommended as a silent DEFAULT change to any production driver") is now superseded by
  this document -- updated in place (see that file's own Section J).
