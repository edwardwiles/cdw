# Melitz production-fast backend: closure session (2026-07-26 continuation)

Continues `docs/melitz_production_fast_backend_2026-07-26.md` (the production-port session)
under the governing closure prompt's 13 phases + addendum. This document is written
progressively as each phase completes, not only at the end -- each section below is final as
of the phase it describes, later phases do not retroactively invalidate earlier ones unless
explicitly noted.

Working tree: `trade_robustness_modular`, branch `melitz/fullD-delta-star`, HEAD `1078398f`
at session start (uncommitted production-fast port on top, preserved throughout). No commit
made yet (Phase 13 will do the single local commit once all phases pass).

## Phase 0: inventory (complete)

- Confirmed branch/HEAD/uncommitted diff matched the handoff exactly.
- `00_READ_FIRST_CORRECTION.md` (named in the governing prompt's "read first" list) does
  **not** exist in this repo -- flagged, not fabricated. Its presumed content (the
  `lower_limit_guard` omission incident) is already written up in
  `melitz_production_fast_backend_2026-07-26.md` Section 5.5.
- Baseline full test suite (pre-closure-session state): 48 testsets, every `Pass==Total`,
  exit code 0, zero errors (one `ERROR:` line in the log is KNITRO's own console output for
  a deliberate infeasibility test, not a Test.jl failure).
- Environment: Julia 1.12.6 (juliaup), KNITRO 13.0.1 (this repo's own pinned version --
  `.knitro_env.sh`'s own documented reason: 14.x lacks a valid site license), 208 cores/3.0TiB
  host, `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`/`JULIA_NUM_THREADS=1` per this project's
  standing test-run convention.

## Phase 1: evaluation cap impossible to omit (complete, then simplified)

**The confirmed bug**: `build_melitz_implicit_bundle`'s cap-activation logic checked
`lower_limit_guard === nothing` *before* checking whether `delta_evaluation_cap` was even
supplied. A caller passing `delta_evaluation_cap=10.0` alone (looking active) fell through to
the disabled branch (`lower_limit=-KN_INFINITY`) purely because `lower_limit_guard` was
omitted. Since `solve_melitz_finite_delta_bound` always forwards a real `delta_evaluation_cap`
(default `10.0`, never `nothing`), every call to that high-level driver that didn't separately
pass `lower_limit_guard` silently ran fully uncapped -- exactly the incident already
documented in this repo's own `docs/melitz_production_fast_backend_2026-07-26.md` Section 5.5
(13.5x slower campaign, 31 spurious `NumericalFailure`s).

**The fix**: reordered the branch logic in `build_melitz_implicit_bundle` (`finite_delta_outer.jl`)
so `delta_evaluation_cap !== nothing` alone always activates the cap. Added a defensive
`@assert isfinite(obj.lower_limit)` directly in `solve_melitz_finite_delta_bound` as a live
runtime guarantee. Traced every existing test call site first (9 for the high-level driver,
~28 for the low-level constructor) -- none relied on the old disabled-by-omission behavior.

**Second pass (same day, direct user feedback)**: the initial fix additionally introduced a
small additive `guard`/`lower_limit_guard` margin on top of the cap
(`lower_limit = -(delta_evaluation_cap + guard)`), inherited unreflectively from an EARLIER,
different, and already-superseded design where the abort threshold was tied to the OUTER
BUDGET `delta` (which genuinely needed a margin, because aborting exactly AT `delta` was the
bug being fixed there). User correction: once the threshold is the evaluation cap itself (a
value chosen deliberately far from any routine `Delta`, e.g. `10.0` vs. routine values order
`1e-3`-`1`), no margin is needed -- the cap IS the threshold. Removed the guard/margin
concept **entirely**, including from `inner_solve_config.jl` (`MelitzInnerSolveConfig`/
`melitz_configure_lower_limit` no longer take a `guard` argument) and from an independently-
implemented copy of the identical `cap+guard` pattern found in `inner_screening.jl`'s
`melitz_dual_polish_screen`/`melitz_classified_inner_solve` (unrelated call path, same
vestigial idea). `lower_limit = -delta_evaluation_cap`, exactly, everywhere.

Fixed one flaky regression test of this session's own making along the way: a fixed random
seed occasionally perturbed a small D=4 fixture past its own convergence boundary
(`nStatus != 0`) -- replaced with a small deterministic candidate-seed search at a smaller,
safer perturbation magnitude.

**Result**: 48/48 testsets green, +9 new regression tests recreating the exact incident
pattern (cap-alone-no-guard must activate; the driver's own no-kwargs default must activate;
a bad point under a tight cap-alone must resolve as fast `AboveEvaluationCap`, not a full
re-derivation).

## Phase 2: strict production-fast forbids every dense fallback (complete)

`MELITZ_PRODUCTION_FAST` previously had `forbid_dense_fallback=false` -- an authoritative
"production-fast" preset silently PERMITTING a dense fallback was backwards. Corrected:

- `MELITZ_PRODUCTION_FAST`: `forbid_dense_fallback=true` (strict).
- `MELITZ_PRODUCTION_COMPAT` (new): otherwise identical, `forbid_dense_fallback=false`
  (explicit, documented dense fallback permitted -- for genuine ablation/cross-check needs).
- `MELITZ_DENSE_REFERENCE`: unchanged (`forbid_dense_fallback=false` by construction -- this
  preset IS the dense path).

Added a real `forbid_dense_fallback::Bool=false` kwarg to all four production entry points
(`build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`,
`build_melitz_implicit_bundle`, `solve_melitz_finite_delta_bound`) that throws an
`ArgumentError` **at construction time** -- before any moment/KNITRO work begins -- for every
known way a caller can end up on a dense path: explicit `backend=:dense_reference`, an
explicit `moment_backend` (which silently selects `:dense_reference` per this codebase's own
documented rule), `needs_outer_moment_jacobian=true` (same rule), or a legacy dense-only
`gradient_backend` (e.g. `:B`). Verified the failure is genuinely fast (all new tests
complete in ~1s total, not a real KNITRO trajectory).

**Result**: 19/19 new tests, all passing in ~1.0s; full suite remains green (48 testsets, no
regressions).

## Phase 3: centralize and type the welfare metrics (complete)

Confirmed the exact mislabeling incident the governing prompt describes:
`predictor_corrector.jl`'s `kappa_of_g` never computed the gains-from-trade `kappa`/`GT_j` --
only the pre-`1-` ratio term (`(w'/w)*gamma_prime^(1/(sigma-1))`) -- and its result was stored
under bare `.kappa`/`.final_kappa` fields on `MelitzPredictorCorrectorStep`/
`MelitzPredictorCorrectorResult`. (Note: `predictor_corrector.jl` was not previously included
in `test/melitz/runtests.jl` at all -- zero test coverage before this session; now included.)

Added, in `equilibrium.jl` (alongside the pre-existing, always-correct
`melitz_gains_from_trade`/`acr_gains_from_trade`):

- `MelitzWelfareMetrics` -- explicit fields `g`, `gamma_prime`, `wage_ratio`, `kappa_ratio`,
  `gains_from_trade`. Never store/print `g` under a `kappa`/`GT`-named field again.
- `melitz_welfare_metrics_from_g(g, wage_ratio, sigma)` / `(g, calib_or_ctx)`.
- `melitz_welfare_metrics(p::MelitzPrimitives, cf::MelitzCounterfactual)` -- population-level,
  wraps `melitz_gains_from_trade`'s own arithmetic exactly.
- `kappa_ratio_of_g` -- the renamed `kappa_of_g` (no external call sites existed outside
  `predictor_corrector.jl`, confirmed by grep, so renamed outright rather than kept as a
  deprecated alias). `predictor_corrector.jl` updated throughout
  (`.kappa`->`.kappa_ratio`, `final_kappa`->`final_kappa_ratio`, `kappa_cur`->
  `kappa_ratio_cur`).

**Result**: 27/27 new tests (gamma_prime==exp(g); kappa_ratio/GT arithmetic; GT agrees with
`melitz_gains_from_trade`; Pareto-reference GT agrees with the ACR sufficient statistic;
plausible real-data GT lies in `[0,1)`; a negative `g` is never aliased to
`kappa_ratio`/`gains_from_trade`). Full suite green, no regressions.

## Phase 4: reconcile production vs. validation inner-solver benchmarks (complete)

Four benchmark numbers now exist for "matrix-free vs dense, real D=20/W=80,000, complete
inner solve":

| # | Source | Construction | Options file | Wall (dense) | Wall (matrix-free) | Speedup |
|---|---|---|---|---:|---:|---:|
| 1 | Validation harness (`docs/melitz_matrix_free_inner_operator_2026-07-26.md` I.3) | freestanding `MelitzMatrixFreeDualBundle`/`melitz_matrix_free_inner_solve` (hand-rolled validation-only driver) vs. `build_melitz_psi_bundle`+`melitz_recover_lfd` | not preserved in-repo (ad hoc `/tmp` script, archived to Dropbox only) | `13.35s` | `0.69s` | `19.2x` |
| 2 | Production constructor comparison (`melitz_production_fast_backend_2026-07-26.md` 5.2) | `build_melitz_psi_bundle_from_calibration` both sides, `melitz_recover_lfd` both sides | `melitz_inner_loop_options_capped_2026-07-24.opt` | `9.70s` | `2.88s` | `3.37x` |
| 3 | Cumulative production ablation (`melitz_production_fast_backend_2026-07-26.md` 5.4) | component-decomposed (moment-build wall + inner-solve wall measured separately, Configs A/B/D) | not stated per-component | `11.39s` (total) | `1.23s` (total) | `9.28x` |
| 4 | **This session's matched benchmark** (below) | `build_melitz_psi_bundle_from_calibration` both sides, `melitz_recover_lfd` both sides, single-thread, post-JIT | `melitz_inner_loop_options_capped_2026-07-24.opt` | `4.516s` | `0.393s` | `11.49x` |

### 4.1 This session's matched run

Identical real-D20 calibration (`real_data/noah_D20`, `sigma=2.5`, `theta_star` estimated,
focal=`fra`), identical `W=80,000` seed, identical canonical options file
(`melitz_inner_loop_options_capped_2026-07-24.opt`), identical construction path
(`build_melitz_psi_bundle_from_calibration` -> `melitz_recover_lfd`, for BOTH
`backend=:matrix_free` and `backend=:dense_reference`), `OPENBLAS_NUM_THREADS=1`/
`JULIA_NUM_THREADS=1` (single-thread, isolating the pure per-call arithmetic difference from
any parallel-Hessian effect), post-JIT (a `W=2,000` warmup solve discarded before timing).

```
matrix_free     : wall=0.393s nStatus=0 Delta=4.072070e-04
dense_reference : wall=4.516s nStatus=0 Delta=4.072070e-04
speedup = 11.49x
Delta abs diff = 3.236e-17
```

Both sides agree to `3.236e-17` in `Delta` (both `nStatus==0`) -- correctness is not in
question at any of the four benchmarks; only the *magnitude* of the speedup varies.

**Side finding (worth flagging, Phase 10-adjacent)**: reproducing this benchmark from a
genuinely standalone script (no `cc_algo` loaded) throws
`MethodError: no method matching KN_add_vars(::KNITRO.Model, ::Int64)` inside
`MelitzCCBundle`'s own KNITRO driver (`cc_bundle.jl:349`). Root cause: `cc_bundle.jl` calls
the 2-argument convenience form `KNITRO.KN_add_vars(kc, n)`, which does **not** exist in the
installed `KNITRO.jl` package itself (only the 3-arg core method does) -- the 2-arg method is
a monkey-patch defined in `cc_algo/knitro_compat.jl` (`function KNITRO.KN_add_vars(kc,
nV::Integer) ... end`, extending the KNITRO module from outside the package). `test/melitz/runtests.jl`
happens to load this shim (via its own guarded `include(cc_algo/include_cc_algo.jl)` for the
`KNITRO_AVAILABLE` check, itself needed for other, genuinely cc_algo-calling tests), so this
dependency is invisible in the test suite but real: `MelitzCCBundle`'s own claimed "own
functor... independent of cc_algo" (production-port doc, Section 3) is accurate for the
economics/algorithm, but is NOT accurate for this one KNITRO-version-compatibility shim --
any genuinely standalone Melitz-only caller (no `cc_algo` in the process at all) would hit
this same crash. Not fixed this session (fixing it means either duplicating the 2-line compat
shim under `src/melitz/` -- trivial, zero economic risk -- or documenting the dependency as
permanent); flagged for Phase 10/13 follow-up.

### 4.2 Reconciliation

All four numbers are genuine (no fabricated/estimated figures) and all agree on the
qualitative conclusion (matrix-free is faster, by a wide margin, with identical economics).
The magnitude varies because the four benchmarks measure genuinely different things, not
because any of them is wrong:

1. **Benchmark 1** used a hand-rolled, validation-only KNITRO driver
   (`melitz_matrix_free_inner_solve`) on the matrix-free side, built specifically to mirror
   `inner_loop_KNITRO`'s call sequence for a first correctness check -- not the actual
   production constructor path, and its own options file is not preserved for exact
   comparison.
2. **Benchmark 2** uses the real production constructors for both sides, but wall-clock
   variance across the two backends is not purely a per-call-arithmetic effect: `MelitzCCBundle`'s
   own KNITRO driver (`melitz_cc_inner_loop_knitro!`) and `PsiObjectiveBundleDelta`'s
   (`cc_algo`'s `inner_loop_KNITRO`) are two INDEPENDENTLY-CODED KNITRO driver
   implementations (by explicit design -- "own KNITRO driver... ported, not called") that can
   take a genuinely different number of KNITRO iterations to reach the same converged point
   under identical options, simply because Hessian-callback registration, warm-start
   handling, and internal bookkeeping differ between the two implementations. This is a
   structural reason the ratio is not a universal constant, not a bug in either driver.
3. **Benchmark 3** decomposes wall-clock by component (moment-build vs. inner-solve) rather
   than measuring one end-to-end `melitz_recover_lfd` call -- a different, more granular
   measurement basis that isolates per-layer cost rather than whatever KNITRO trajectory each
   full solve happens to take.
4. **Benchmark 4** (this session) matches construction path, options file, calibration, seed,
   and thread policy as closely as the codebase currently allows, single-threaded to remove
   any parallel-Hessian confound -- `11.49x` sits within the `3.37x`-`19.2x` range the prior
   three benchmarks already established, closer to the high end, consistent with a clean
   single-thread comparison.

**Conclusion**: do not chase a single "the" multiplier. The defensible, reproducible claim is
"a large (>3x, often >10x), correctness-preserving speedup through the actual production
entry points" -- confirmed again here -- not a specific number that would hold across every
options file/thread policy/KNITRO version. Future benchmark scripts should report their own
exact options file/thread policy/construction path alongside any speedup figure (this
session's own table above does so for all four), rather than a bare multiplier.

### 4.3 Canonical option files (governing prompt 4.2)

- **Evaluation-cap / production solves**: `melitz_inner_loop_options_capped_2026-07-24.opt`
  (Melitz-owned, already the convention `melitz_production_fast_backend_2026-07-26.md`'s own
  Section 5.2-5.5 benchmarks used) paired with `melitz_outer_finite_delta.opt` for the outer
  NLP. Both Melitz-owned (`melitz_` prefix), never the `ek_*.opt` Ricardian-named files.
- **Full-value verification**: `melitz_inner_loop_options.opt` (Melitz-owned, uncapped-by-name
  default already used throughout `test/melitz/runtests.jl`'s own non-cap testsets).
- **Note (pre-existing, not changed this session)**: `build_melitz_psi_bundle`/
  `build_melitz_psi_bundle_from_calibration`'s own DEFAULT `inner_loop_opt`/`outer_loop_opt`
  kwargs still point at `ek_inner_loop_options.opt`/`ek_outer_loop_options.opt` (the
  Ricardian-named files) -- inherited from before this codebase's own Melitz-specific option
  files existed, and changing a default relied on by dozens of existing call sites is a
  larger, separate risk than this phase's own scope (flagged for Phase 10, not attempted
  here). Every call site THIS session added or modified passes an explicit Melitz-owned
  `inner_loop_opt` (`melitz_inner_loop_options.opt`/`melitz_inner_loop_options_capped_2026-07-24.opt`),
  never relying on that default.

## Phase 5: audit the 1e10 divergence-constraint scaling (complete)

**Key finding: the governing prompt's premise is stale.** It describes the current
implicit bundle as using `constr[1] = 1e10 * DeltaStar` (up to sign) as the REGISTERED
KNITRO constraint. Traced the complete chain in `finite_delta_outer.jl` (own file header +
`melitz_build_finite_delta_callbacks`/`melitz_register_finite_delta_knitro_problem!`) and
found this was already fixed in a **2026-07-23** correctness-repair session (well before this
closure session's own 2026-07-26 starting point), predating even the production-port work.

**The exact current trace** (governing prompt 5.1's own ask):

1. Raw inner objective: the shared cc_algo-convention functor computes
   `constr[1] = -f(x)*1e10` where `f` is the raw KNITRO-minimized dual objective; at the true
   optimum `-f(x*) = DeltaStar(theta)` exactly (Section 18's sign fix).
2. `Delta_theta = local_c[1] / 1e10` recovers the unscaled `DeltaStar(theta)`.
3. **Registered constraint value** (`cb_F!`, `finite_delta_outer.jl:1187`, pre-session):
   `evalResult.c[1] = Delta_theta / delta` -- dimensionless.
4. **Registered constraint bound** (`melitz_register_finite_delta_knitro_problem!`, pre-session):
   `KNITRO.KN_set_con_upbnd(kc, cIndices[1], 1.0)` -- literally `1.0`.
5. **Registered Jacobian** (`cb_G!`): `local_jac ./ (1e10 * delta)`, where
   `local_jac = d(1e10*DeltaStar)/dtheta` -- consistently rescaled with the value.
6. **Objective scale**: `evalResult.obj[1] = ±theta[1]` -- no scaling at all.
7. **KNITRO user scaling** (`var_scale`/`var_center`): a separate, orthogonal, opt-in
   mechanism (2026-07-25 session); `nothing` by default, not entangled with this constraint.

**The exact final constraint KNITRO solves**: `minimize/maximize ±theta[1] subject to
DeltaStar(theta)/delta <= 1` (plus separately-registered cutoff/gravity rows) -- already the
dimensionless representation the governing prompt asks to introduce as an option. `1e10`
survives only as an internal raw-functor convention that cancels exactly before reaching
KNITRO; it was never a live outer-conditioning problem in the CURRENT code (it may have been
in the pre-2026-07-23 state the governing prompt's authors may have had in mind).

**What this session added** (governing prompt 5.2, despite dimensionless already being
default): a real `divergence_constraint_scaling::Symbol=:dimensionless` kwarg on
`melitz_build_finite_delta_callbacks`/`solve_melitz_finite_delta_bound`/
`melitz_fixed_point_probe`, with `:legacy_1e10` as an explicit, diagnostic-only alternative
reproducing the OLD pre-2026-07-23 raw-magnitude registration
(`1e10*DeltaStar(theta) <= 1e10*delta`) for direct comparison. Both modes derive from ONE
`divergence_divisor` constant (`1e10*delta` for `:dimensionless`, `1` for `:legacy_1e10`) so
the registered value/bound/sentinel are always `raw/divisor <= (1e10*delta)/divisor` --
provably the identical feasible set for any divisor, confirmed by tests showing: (a) the
default (`:dimensionless`) reproduces the pre-existing value/bound exactly; (b)
`:legacy_1e10` reproduces the pre-2026-07-23 registration exactly; (c) both modes agree on
feasibility classification and gradient sign pattern at both a feasible and an infeasible
point, with value/Jacobian related by the exact constant scale factor
(`rtol=1e-8`). **Not made the production default** (already isn't -- `:dimensionless` already
was), per the governing prompt's own instruction not to change the default without matched
D=4/D=20 evidence.

**Result**: 19/19 new tests green (one flaky perturbation-magnitude issue in this session's
own test, same class as Phase 1's -- fixed the same way: a small deterministic candidate
search instead of one fixed seed). Full suite green, no regressions.

## Phase 6: port nuisance profiling to production-fast infrastructure (complete)

`nuisance_profile.jl`'s `solve_melitz_nuisance_min_delta`/`melitz_build_nuisance_profile_callbacks`
were entirely dense-only (`inner_loop(obj_inner, theta)` -- cc_algo's function, dispatched
only for `PsiObjectiveBundleDelta`; direct `obj_inner.H` reads/writes in the exact-point
cache). Audited the existing bundle-agnostic dispatch infrastructure already built during
the 2026-07-26 production-port session (`cc_bundle.jl`) and found MOST of what Phase 6 needs
already existed:

- `melitz_heavy_snapshot`/`melitz_heavy_restore!`/`melitz_heavy_recompute` -- already
  dispatched per bundle type (dense `Matrix` copy vs. `MelitzOperatorSnapshot`).
- `direct_gradient.jl`'s gradient backends -- already bundle-agnostic via multiple dispatch
  on the untyped `obj` parameter they already had (`_base_arg0!`/`_direct_coordinate_grad`
  gained `MelitzCCBundle`-specific methods in `cc_bundle.jl` during the production-port
  session; the closures calling them needed zero changes).

The ONE missing piece: an `inner_loop`-equivalent entry point for `MelitzCCBundle` (the
existing `melitz_bundle_prepare_at_theta!`/`melitz_bundle_inner_solve!` pair stops one level
short, at the unflipped `inner_loop_internal` convention). Added `melitz_bundle_inner_loop(obj,
theta)` (`cc_bundle.jl`) -- composed ENTIRELY from the existing pair plus the same one-line
`find_smallest` sign correction both `inner_loop` (cc_algo) and `melitz_cc_inner_loop` (this
file) already apply independently. No new algorithm; no trust-region/predictor-corrector
logic added, per the governing prompt's own explicit instruction.

Added `forbid_dense_fallback::Bool=false` to `melitz_build_nuisance_profile_callbacks`/
`solve_melitz_nuisance_min_delta` (throws if `obj_inner` is not a `MelitzCCBundle`), and
extended `gradient_backend` to also accept the two SORTED direct backends (previously only
plain serial/parallel).

**Two real bugs found and fixed along the way (both this session's own, not pre-existing)**:

1. A real-D20 test using `W=20,000` hit a genuinely ill-conditioned, UNCAPPED single inner
   solve -- confirmed live: dense and matrix-free objectives disagreed by 40% (`1.3e9` vs
   `2.2e9`) and one gradient entry was `~1e23`-`~1e24` (clear numerical garbage from a
   divergent trajectory). This exactly matches this repo's own documented finding (memory:
   "Melitz D=20 rank deficiency RESOLVES at W=80k") -- fixed by using `W=80,000`, after which
   both backends agree to `rtol=1e-4`/`1e-6`.
2. A test-ordering bug: the backend-usage counter reset was placed BEFORE the (legitimately
   dense) reference bundle's own callback calls, so the shared global counters picked up the
   dense bundle's own expected dense-path increments and misattributed them to the
   matrix-free run as a false "dense fallback leak." Fixed by moving the reset to immediately
   before the matrix-free calls only.
3. A full nested `solve_melitz_nuisance_min_delta` outer-search test at real D=20 scale was
   attempted first and appeared to hang (28+ minutes, no output) -- consistent with this
   repo's own documented real-D20 nested-KNITRO-solve fragility. Replaced with a single
   fixed-point `cb_F!`/`cb_G!` comparison (matching the governing prompt's own wording: "one
   real-D20 FIXED g point"), which tests the identical bundle-agnostic port without that risk.

**Result**: 17/17 tests green (D=4 matched dense-vs-matrix-free: 8/8 identical `Delta_min`/
`theta_final`/cutoff-gravity-residuals; real D=20/W=80,000 single-point `cb_F!`/`cb_G!`:
objective/gradient agree, zero dense-fallback counters). Full suite green, no regressions.

## Phase 7: matrix-free range screen (complete, enabled by default)

Implemented `melitz_range_screen(op::MelitzMomentOperator)` (`inner_screening.jl`) --
a matrix-free equivalent of the dense `melitz_range_screen(G)` that never materializes `G`.

**Algebra**: for trade cell `(o,d)`, `G[w,idx] = coef[o,d]*z_power[w,o] - lambda[o,d]` when
draw `w` is ACTIVE (`bin[w,o] >= rank[d,o]`), else the CONSTANT `-lambda[o,d]`. Since
`sorted_ctx.sorted_z_power[:,o]` is sorted ascending and `bin` is a non-decreasing function of
`z`, "active for threshold `t`" is exactly the sorted SUFFIX from `first_active_pos[t+1,o]` to
`W` -- so the active set's `z_power` extrema are just its two endpoints (no scan needed).
Added `first_active_pos::Matrix{Int}` ((D+1) x D) to `MelitzMomentOperator`, fused directly
into the existing `melitz_update_moment_operator!` merge sweep (no extra `O(W*D)` pass):
whenever the sweep's `ptr` advances (by one step, or several when a single draw crosses
multiple cutoffs at once), the current sorted position is recorded as the first position
reaching every newly-crossed level. The screen itself is then `O(D^2)` (D origins x D
destinations, O(1) lookup each) plus one `O(W)` pass for the always-dense focal-link column.

**Validation** (standalone script, no KNITRO needed -- direct `op`/`G` construction from
`melitz_outer_state`): D=4 (FIXTURE, 11 points), D=10 (synthetic, seed=100, 10 points), real
D=20 (`real_data/noah_D20` via calibration, W=20,000, 16 points, including several
over-budget/near-infeasible perturbations) -- **zero mismatches** across all 37 checked
points (same nothing/certificate, same column index, same `lo`/`hi` to `1e-9`).

**Performance** (real D=20/W=20,000, screen call only, given a pre-built `op`/`G` -- isolates
the screen from moment-construction cost): dense `~32.9ms`, matrix-free `~110.7us` --
**~300x**. This does not include `G`'s own `O(W*D^2)` construction cost, which a real
production run avoids entirely on the matrix-free path (not double-counted here, matching
the governing prompt's own request to report the screen's own wall separately).

**Decision**: the governing prompt's own bar ("enable it by default only if the measured net
return is positive") is decisively met -- `matrix_free_range_screen` defaults to `true` in
`melitz_classified_inner_solve` (still toggleable via the same kwarg; `false` remains
available for direct comparison/diagnostics). The origin-block LP screen remains dense-only
and opt-in (default `false`), unaudited this session, per the governing prompt's own scope
note.

**Result**: 23/23 new tests green (D=4/D=10/real-D20 equivalence; confirms the screen
actually engages by default for `MelitzCCBundle` and that `matrix_free_range_screen=false`
still disables it on request). Full suite green, no regressions.

## Phase 8: corrected D=4 regression (complete)

Reran the D=4 outer comparison with the fix the governing prompt specifically asked for:
`D=4`, `W=20,000`, `seed=29`, native `:linear` cutoff constraints, the mandatory
evaluation-cap configuration (`delta_evaluation_cap=10.0`, always active per Phase 1), and
all four `delta in {1e-3, 1e-2} x direction in {upper, lower}` combinations, strict
production-fast (`forbid_dense_fallback=true`) vs. dense reference, identical `theta_init`/
`theta_box`/options files on both sides.

**A real bug found in this session's OWN first attempt at this benchmark**: the first script
pinned only the INNER bundle's (`obj_inner`) backend, not `solve_melitz_finite_delta_bound`'s
own `backend` kwarg -- which defaults to `:auto_from_gradient_backend` and, since
`:B_direct_argument_serial` is not in the legacy dense-only `gradient_backend` list, silently
built a MATRIX-FREE outer-NLP callback bundle for BOTH configurations regardless of the
inner bundle's own backend. Caught by the dense run's own backend counters showing
`dense_moment_calls=0`/`matrix_free_objective_calls>0` -- structurally impossible for a
genuinely dense run. Fixed by passing `backend=` explicitly and using
`:B_argument_localized_serial` (a genuinely dense, but not artificially catastrophic, dense
gradient backend -- raw `:B` was measured elsewhere in this codebase at `1049x` slower than
the direct backends and would have made this routine regression benchmark impractically
slow) for the dense side.

**Results** (all four hit the outer options file's own `maxit` iteration cap,
`nStatus in (-410, -200)`, matching this repo's own documented "expected for a quick
regression check" convention -- not a claim about full convergence):

| delta | direction | backend | g (NOT kappa) | kappa_ratio | GT | DeltaStar | n_fc | n_ga | wall |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | upper | dense | -0.051084 | 0.929359 | 0.070641 | 9.601e-4 | 137 | 26 | 61.48s |
| 1e-3 | upper | matrix-free | -0.051625 | 0.929023 | 0.070977 | 9.527e-4 | 135 | 26 | 12.14s |
| 1e-3 | lower | dense | -0.035440 | 0.939102 | 0.060898 | 9.682e-4 | 121 | 26 | 39.13s |
| 1e-3 | lower | matrix-free | -0.036582 | 0.938387 | 0.061613 | 9.698e-4 | 112 | 26 | 27.36s |
| 1e-2 | upper | dense | -0.069470 | 0.918037 | 0.081963 | 9.413e-3 | 99 | 26 | 29.39s |
| 1e-2 | upper | matrix-free | -0.071432 | 0.916837 | 0.083163 | 9.619e-3 | 97 | 26 | 14.32s |
| 1e-2 | lower | dense | -0.025363 | 0.945432 | 0.054568 | 9.838e-3 | 107 | 26 | 35.09s |
| 1e-2 | lower | matrix-free | -0.027097 | 0.944340 | 0.055660 | 9.980e-3 | 86 | 21 | 6.04s |

Gravity residuals: `<7e-17` (both `A` and `f`) at every one of the 8 runs -- gravity-exact by
pivot construction, as expected, on both backends. Cutoff `min_slack` positive throughout
(feasible). `n_numerical_failure_reject=0` at every configuration (the evaluation cap is
doing its job -- zero uncertified failures). Dense-run counters confirm genuine dense
execution (`dense_moment_calls`/`dense_G_materializations`/`production_dense_screen_calls`
all `>0`, `matrix_free_*` all `0`); matrix-free-run counters confirm the reverse, with
`dense_*` counters all exactly `0` throughout (the `forbid_dense_fallback=true` structural
guarantee held).

**Economics agree closely, not bit-identically** (two independently-converged local KNITRO
trajectories on a nonconvex problem, following different iteration paths -- `n_fc_calls`
differ by up to ~20 between backends): `g` differs by `1e-3`-`2e-3` across configurations,
`GT`/`kappa_ratio` by similar relative magnitude, `DeltaStar` agrees to 2-3 significant
figures. This matches the qualitative "solver trajectories may differ slightly, economics
must agree (not bit-for-bit)" pattern already documented elsewhere in this repo for the D=20
case, now confirmed at D=4 with the CORRECTED (capped, matched-backend) configuration.

**Wall-clock -- corrected finding, updating the prior (uncapped, misconfigured) record**:
matrix-free is faster at ALL FOUR D=4 configurations this session actually measured
correctly (`1.4x` to `5.8x`), NOT slower as `docs/melitz_production_fast_backend_2026-07-26.md`
Section 5.6's own (uncapped, `gradient_backend=:B_direct_argument_serial`-mislabeled-as-dense)
D=4 comparison reported. Reported honestly either way per the governing prompt's own
instruction; this session's own numbers are more trustworthy given the backend-pinning bug
above was caught and fixed before reporting them, and the evaluation cap was genuinely
active on both sides (Section 5.6's own comparison flagged that its wall-clock numbers
"likely understate achievable speed" precisely because it lacked the cap -- consistent with
the direction of this correction).

## Phase 9: real-D20 robustness (complete -- honest negative result included)

Fixed-point-only checks (no outer campaign, per the governing prompt's own instruction) at
the real `noah_D20` calibration, comparing dense vs. matrix-free `evaluate_melitz_delta` at
`theta0`:

| config | matrix-free | dense | Delta abs diff | `G` rank | `G` cond |
|---|---|---|---:|---:|---:|
| W=80,000, canonical seed=1 | `nStatus=0 Delta=4.0721e-4` (9.50s) | `nStatus=0 Delta=4.0721e-4` (17.44s) | `3.236e-17` | 401/401 | `1.06e5` |
| W=80,000, **additional seed=2** | `nStatus=-103 Delta=3.70e8` (9.54s) | `nStatus=-401 Delta=1.00e10` (90.43s, ~full `maxtime_real`) | -- (both unverified) | **400/401** | **`6.80e17`** |
| W=160,000, canonical seed=1 | `nStatus=0 Delta=1.3346e-4` (10.23s) | `nStatus=0 Delta=1.3346e-4` (17.42s) | `4.887e-17` | not checked (skipped `store_G` at this `W`) | -- |

**Canonical seed=1 is well-posed and both backends agree to machine precision at BOTH
W=80,000 and W=160,000** (`~3-5e-17` absolute `Delta` difference, matching Phase 4's own
W=80,000 finding exactly). Matrix-free is consistently faster in this single-fixed-point
comparison too (`~1.7x`-`1.8x`), a smaller margin than the full-campaign ablations (expected --
a single cold `evaluate_melitz_delta` call is dominated by JIT/one-off setup relative to a
multi-iteration campaign) but a real, consistent, honestly-measured one.

**The additional seed (seed=2) genuinely FAILS to produce a well-posed fixed point at
W=80,000** -- the moment matrix `G` is measurably RANK DEFICIENT (`400/401`, condition number
`6.8e17`, i.e. numerically singular in double precision) at this theta0/seed combination,
and BOTH backends correctly fail to verify (dense hits its own `-1e10` uncapped-failure
sentinel after burning the full `90s` `maxtime_real`; matrix-free reaches `nStatus=-103`,
`Delta~3.7e8`, also unverified). This is reported explicitly, per the governing prompt's own
instruction ("do not search only until finding favorable seeds and suppress the failures") --
no further seed search was performed to find a second passing seed. The finite-support
geometry of this ONE real dataset (`noah_D20`) evidently does not guarantee a well-posed
fixed point at every QMC seed even at `W=80,000` (previously documented as resolving the
rank deficiency FOR THE CANONICAL seed only, memory: "Melitz D20 rank deficiency RESOLVES at
W=80k" -- this session's finding refines that: resolution is seed-dependent, not universal at
this `W`).

## Phase 10: hidden backend defaults (audited, no changes needed)

Audited every `moment_backend`/`backend`/`inner_backend` default across `src/melitz/*.jl`
(`grep` for `::Symbol=:dense_reference`-style defaults). Found exactly one:
`melitz_calibration_outer_ctx`'s own `moment_backend::Symbol=:dense_reference`
(`pareto_calibration.jl`) -- already flagged in
`docs/melitz_production_fast_backend_2026-07-26.md` Section 8 as a known, deliberately-scoped
gap. Traced its only two callers (grep-confirmed, no others exist in this codebase): (1)
`build_melitz_psi_bundle_from_calibration`, the production entry point, which ALREADY passes
`moment_backend=resolved_moment_backend` explicitly, never relying on this default; (2)
`melitz_calibration_roundtrip_check`, a pure algebraic reduce/expand consistency check (no
Monte Carlo, no KNITRO, no moment matrix ever touched regardless of this field) -- clearly
diagnostic by name and docstring, matching the governing prompt's own stated exception
("explicit diagnostic functions may default to `MELITZ_DENSE_REFERENCE` only if their names/
documentation clearly identify them as reference tools"). Documented this finding directly in
the function's own docstring; left the default unchanged (no behavior change needed).

The four real production entry points (`build_melitz_psi_bundle`,
`build_melitz_psi_bundle_from_calibration`, `build_melitz_implicit_bundle`,
`solve_melitz_finite_delta_bound`) already default through `backend=:matrix_free` (or
`:auto_from_...`, which resolves to matrix-free absent a legacy-dense-only override) --
confirmed BY CONSTRUCTION in Phase 2's own tests (`obj_ok isa MelitzCCBundle` with no backend
keyword passed, for all four) and Phase 6's nuisance-profile test (same check) -- no
additional tests needed for this phase's own "construct every production entry point with no
backend keyword" requirement; already covered.

## Phase 11: counter/cache thread-safety audit (one real race found and fixed)

**Counters**: grepped every `MELITZ_*[] += 1` site (18 total) against every
`Threads.@threads` region in `src/melitz/` (6 files: `localized_gradient.jl`,
`direct_gradient.jl`, `sorted_tail.jl`, `moment_operator.jl`, `sorted_crossing_gradient.jl`,
`argument_localized_gradient.jl`). **None of the 18 counter increments occur inside a
parallel region** -- every one lives in `cc_bundle.jl`/`finite_delta_outer.jl`/
`inner_screening.jl` at the single-call (one objective/gradient/Hessian dispatch, one cache
lookup, one screen) level; the `Threads.@threads` coordinate sweeps write only to disjoint
per-coordinate output array slots (`g[r]`/`H[...]`, already documented "no synchronization
needed" in prior sessions) and touch no shared global counter. The backend-usage counters
are therefore reliable evidence of dense-vs-matrix-free execution, exactly as this session's
own Phase 2/6/7/8 tests rely on them being.

**One genuine race found and fixed**: the Melitz-owned single-flight concurrency guard
(`melitz_cc_guard_enter_inner_solve!`, `cc_bundle.jl`) used a plain `Ref{Bool}`
(`MELITZ_CC_INNER_SOLVE_ACTIVE`) with a check-then-set pattern -- a classic TOCTOU race: two
threads calling it concurrently could BOTH observe `false` before either sets `true`, letting
a genuine concurrent-`KN_solve` violation slip past undetected. This exactly mirrors
`cc_algo/parallelism_guards.jl`'s own identical plain-`Ref` pattern (confirmed by reading,
not modifying, that file) -- inherited at the time this Melitz-owned copy was written, not a
new mistake this session. Fixed (Melitz-owned copy only; `cc_algo` untouched) by converting
to `Threads.Atomic{Bool}` with `Threads.atomic_cas!` for the acquire step -- now provably
race-free regardless of caller behavior. Confirmed this guard's actual exposure in the
CURRENT codebase is latent, not actively triggered: none of the six `Threads.@threads`
regions above ever call into this guard (they only do arithmetic on an already-converged
dual, never launch a new `KN_solve`) -- but a defensive guard whose own check can race is a
real gap worth closing now that it was found, independent of whether it has fired incorrectly
yet.

**Cache byte-boundedness**: `MelitzExactPointCache`'s heavy tier (storing either a dense
`Matrix` copy or a `MelitzOperatorSnapshot`) is already bounded by `heavy_max_bytes`/
`heavy_max_size` with LRU eviction, already tested ("Closure Phase D: content fingerprint +
compact/heavy cache split", 22/22, including a dedicated "tiny byte budget evicts even below
heavy_max_size" test). `melitz_heavy_bytes(snap::MelitzOperatorSnapshot)` correctly sums
EVERY held array (`coef`/`lambda`/`order`/`rank`/`bin`/`ell`) -- confirmed by reading, not
merely assumed -- so a real D=20/W=80,000 snapshot's true `O(W*D)` footprint (`bin` alone is
`W*D` bytes as `UInt8`, `~1.6MB` at this scale) is correctly accounted against the budget, not
undercounted. No unbounded accumulation is possible: the cache's own eviction policy already
enforces the byte budget regardless of how many distinct outer points are visited.

## Phase 12: final no-change production regression (complete)

**Fixed-point comparisons** (D=4, real D=20) and **callback replay** (classification, finite
`DeltaStar`, dual, moment residuals, welfare metrics, equilibrium diagnostics,
production-fast vs. dense agreement) were already established with fresh, direct evidence
across this session's own Phases 4, 6, 7, 8, and 9 -- not re-run a third/fourth time here
for its own sake: Phase 4 (real D=20 matched benchmark, `Delta` diff `3.236e-17`), Phase 6
(D=4 nuisance-profile matched comparison, 8/8 identical `Delta_min`/`theta_final`/gravity
residuals; real D=20 single-point `cb_F!`/`cb_G!` objective+gradient agreement), Phase 7
(range-screen equivalence, D=4/D=10/real-D20, 37 points, zero mismatches), Phase 8 (D=4 full
outer-NLP regression, 4 delta/direction configs, welfare metrics/cutoff/gravity residuals
reported with correct `g`/`kappa_ratio`/`GT` naming throughout), Phase 9 (real D=20 fixed-point
at W=80,000 AND W=160,000, `Delta` diff `~3-5e-17` at the canonical seed). D=10 was covered
in Phase 7 (range-screen equivalence only, not a full outer/inner-solve comparison -- no
D=10 fixture proved stable enough across available seeds for a heavier campaign in the time
available this session; flagged, not silently dropped).

**Short real-D20 production-fast campaign** (this phase's own new work): ran
`solve_melitz_finite_delta_bound` at the real `noah_D20` calibration, `W=80,000`,
`delta=1.0`, `direction=:upper`, `theta_box=0.02`, `delta_evaluation_cap=10.0`,
`backend=:matrix_free`, `forbid_dense_fallback=true` -- the SAME configuration as
`docs/melitz_production_fast_backend_2026-07-26.md` Section 5.5's own corrected campaign,
run again now after all Phase 1-11 closure changes to confirm nothing regressed. Result:
`249` FC calls, `26` GA calls, wall `423.3s`, terminal `nStatus=-400`. **Full 7-point
acceptance checklist, all confirmed true**:

1. mandatory evaluation cap active (`obj.lower_limit` finite): **true**
2. zero dense fallback (`dense_inner_objective/gradient/hessian`, `dense_G_materializations`,
   `production_dense_screen_calls` all `0`): **true**
3. matrix-free FC/GA active (`matrix_free_objective/gradient_calls > 0`): **true**
4. structured Hessian active (`matrix_free_hessian_calls > 0`): **true**
5. FC-to-GA cache reuse: **true** (`26/26` GA calls hit the cache -- `100%`, matching the
   ORIGINAL corrected campaign's own `17/17` `100%` finding)
6. zero spurious `NumericalFailure` from missing cap configuration: **true**
   (`n_numerical_failure_reject == 0`)
7. matrix-free range screen engaged (new this session, Phase 7): **true**
   (`matrix_free_range_screen_calls > 0`)

Welfare metrics correctly separated throughout (`MelitzWelfareMetrics`, Phase 3): cold-verified
`g=-0.424610`, `gamma_prime=0.654025`, `kappa_ratio=0.975891`, `GT=0.024109`, `DeltaStar=4.486e-3`
-- `GT` comfortably in `[0,1)`, in the same ballpark as the original campaign's own
independently-verified `GT=0.0203` (different terminal theta/iteration count, same fixture,
not expected to match exactly -- this session did not redesign the outer search, per the
governing prompt's own explicit instruction). **This is an infrastructure confirmation, not a
claim about outer-search quality** -- matching this repo's own established framing for this
exact campaign.

## Phase 13: commit and document the production closure (complete)

`git diff --name-only` (pre-commit): exactly 10 modified files, all under `src/melitz/` or
`test/melitz/` (`delta_star.jl`, `equilibrium.jl`, `finite_delta_outer.jl`,
`include_melitz.jl`, `inner_screening.jl`, `inner_solve_config.jl`, `nuisance_profile.jl`,
`pareto_calibration.jl`, `predictor_corrector.jl`, `test/melitz/runtests.jl`) plus 4 new
`src/melitz/*.jl` files inherited from the prior (uncommitted) production-port session
(`backend_config.jl`, `cc_bundle.jl`, `matrix_free_dual_solve.jl`, `moment_operator.jl`) and
4 `docs/melitz_*.md` files (3 inherited, 1 -- this document -- new). **Zero diff in
`cc_algo/`, `production/fullA-exact/`, `full_aod_diag/`, or any other Ricardian path** --
confirmed directly, not merely asserted.

Staged and committed exactly those 18 files (explicit filenames, never `git add -A`) --
excluded the large amount of PRE-EXISTING, unrelated untracked cruft in the working tree
(`full_aod_diag/batch_out_v2/`, `sequential_gravity/batch_out_*`, `results/fullA_d4/`, etc. --
inherited from other sessions' work, not part of this closure). Local commit only, per
explicit instruction -- **not pushed**.

**Commit**: `42460f8` on branch `melitz/fullD-delta-star` (18 files changed, 6090
insertions, 226 deletions).

**Final status against the governing prompt's acceptance criteria**:

1. Passing an evaluation cap necessarily activates KNITRO `lower_limit` -- **yes** (Phase 1).
2. No production caller can silently omit the cap configuration -- **yes** for
   `solve_melitz_finite_delta_bound`/`build_melitz_implicit_bundle` (Phase 1); `solve_melitz_nuisance_min_delta`
   already required it before this session (pre-existing).
3. Strict production-fast forbids every dense fallback -- **yes** (Phase 2), including the
   nuisance-profile driver (Phase 6) and the range screen path (Phase 7).
4. `g`, `kappa_ratio`, `GT` cannot be confused -- **yes** (Phase 3, `MelitzWelfareMetrics`).
5. Production/validation timing differences explained -- **yes** (Phase 4, four benchmarks
   reconciled, one new matched measurement added).
6. Nuisance-profile path uses the production-fast matrix-free backend -- **yes** (Phase 6).
7. Matrix-free range-screen value measured and adopted on evidence -- **yes** (Phase 7,
   ~300x, zero mismatches, enabled by default).
8. D=4 regression rerun with correct cap settings -- **yes** (Phase 8, a real backend-pinning
   bug in this session's OWN first attempt found and fixed before reporting).
9. Real-D20 additional seed or documented failure, plus W=160,000 attempt -- **yes** (Phase
   9: canonical seed passes at both W=80k/160k; additional seed genuinely fails, reported
   honestly).
10. Every production entry point defaults through one backend configuration -- **yes**
    (Phase 10 audit; one remaining dense default confirmed diagnostic-only).
11. All hot callbacks remain allocation-stable -- unchanged from the production-port
    session's own measurements (Phase 6/7 additions are new dispatch methods on the SAME
    already-allocation-audited hot path, not new hot-path allocations; not independently
    re-profiled this session).
12. Full tests pass -- **yes**, repeatedly, throughout every phase.
13. No Ricardian/shared source file changes -- **yes**, confirmed directly.
14. The complete port is captured in a local commit -- **yes**, `42460f8`.
15. The next session can focus entirely on outer-search behavior -- **yes**, modulo the
    addendum below (parameterization selection), which this same session also attempts.

**Remaining known gaps, carried forward explicitly** (not silently dropped):
- `predictor_corrector.jl` remains dense-only and explicitly diagnostic (governing prompt's
  own allowance -- "may remain dense-only... unless the user later chooses to revive it").
- The origin-block LP screen remains dense-only and opt-in (Phase 7's own scope note).
- A pre-existing `THREE`-dirname `real_data` path bug in several OTHER (not this session's)
  real-D20 tests in `test/melitz/runtests.jl` (e.g. "Section 17") silently skips via their own
  `isdir` guard -- found while building this session's own (correctly two-dirname) real-D20
  tests; flagged for a future session, not fixed here (those tests are not part of this
  session's own scope).
- D=10 was validated for the range screen only (Phase 7), not a full outer/inner-solve
  comparison (no D=10 fixture proved stable enough across available seeds for a heavier
  campaign in the time available).
- Hot-callback allocation stability (#11 above) was not independently re-profiled this
  session -- inherited from the production-port session's own measurements.
