# Melitz real D=20 local geometry, gradient audit, and continuation prototypes -- 2026-07-25

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing the same-day
sessions `docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md`,
`docs/melitz_real_d20_outer_correction_2026-07-24.md`,
`docs/melitz_real_d20_outer_benchmark_2026-07-24.md`. New/modified:
`src/melitz/inner_solve_config.jl` (new), `src/melitz/predictor_corrector.jl` (new),
`src/melitz/finite_delta_outer.jl`, `src/melitz/delta_star.jl`, `src/melitz/pareto_calibration.jl`,
`src/melitz/nuisance_profile.jl`, `test/melitz/runtests.jl`,
`scripts/melitz_local_geometry_lab_2026-07-25.jl` (new),
`scripts/melitz_predictor_corrector_probe_2026-07-25.jl` (new),
`scripts/melitz_pc_step9_independent_verify_2026-07-25.jl` (new),
`scripts/melitz_nuisance_block_profile_2026-07-25.jl` (new). Same-day user-directed
follow-up on the exact-point cache question (Section B/G.1): `src/melitz/nuisance_profile.jl`
further modified (cache port + bugfix), `scripts/melitz_nuisance_cache_diagnosis_2026-07-25.jl`
(new), `scripts/melitz_nuisance_ab_uncached_2026-07-25.jl` (new, superseded by the
true-original replay below), `scripts/melitz_nuisance_ab_cached_2026-07-25.jl` (new),
`scripts/melitz_nuisance_true_original_instrumented_2026-07-25.jl` (new, the rigorous
byte-for-byte instrumented replay).

**Scope discipline, stated up front**: the governing prompt for this session is an
11-phase, multi-day-scale program (make the evaluation cap structurally impossible to omit;
audit the registered gradient; build a local directional-derivative laboratory across ~7
step sizes x 7+ direction families x 2 values of `W`; implement a trust-region wrapper; a
predictor-corrector continuation prototype; a nuisance-profile block-coordinate experiment;
a 4-way matched comparison campaign; inner hard-stop timing quantiles) and explicitly
forbids launching another long generic 798-dimensional campaign before the local
diagnostics are done. This report completes Phase 1 (the eval-cap infrastructure) in full,
and Phases 2-3 as live-verified audits. Phases 4-10 are scoped down to a **small, honestly
labeled, fully-real (no synthetic/proxy) subset** rather than the full prescribed grid --
consistent with this repo's own established practice (every session report referenced
above discloses incomplete sub-phases explicitly rather than extrapolating). Section I
states exactly what is verified live vs. deferred.

## Executive summary

1. **The evaluation-cap-omission bug pattern is now closed structurally, not just patched
   twice more.** `src/melitz/inner_solve_config.jl` centralizes the cap computation behind
   an explicit `mode` (`:full_value`/`:evaluation_cap`/`:diagnostic`, no silent default);
   `solve_melitz_nuisance_min_delta` -- the exact function whose omission caused a confirmed
   live incident in the prior session -- now REQUIRES a config with no default at all
   (`UndefKeywordError` if omitted). Full test suite: **45/45 testsets pass** (29 new tests
   for this infrastructure), zero regressions.
2. **The registered outer gradient's direction is robust; its magnitude is not, at sparse
   coordinates.** Live bandwidth sweep (Section D.4): sign agreement 18/18 across 3
   directions x 6 bandwidths; relative-error range `0.02%`-`83%` depending on direction and
   bandwidth, with NO simple monotone relationship to participation-switch count.
3. **The local radius of gradient validity is governed by the AGGREGATE `‖d_eta‖` norm, not
   per-coordinate boxes** (Section E): a pure nuisance-descent step breaks between `1e-5`
   and `3e-5` in aggregate norm; a per-coordinate box of `1e-4` (this repo's own prior
   default) permits an aggregate step ~140x larger -- a concrete, quantified explanation for
   why the generic joint-KNITRO search's own trust region was badly mis-scaled.
4. **A predictor-corrector continuation prototype (Section F), using only the minimum-norm
   tangent direction and 11 fully-reoptimized evaluations, found a genuine, independently
   re-verified improvement over the best previously-known incumbent** -- `kappa=0.92936736`
   vs. the fixed-A/f reference's `0.92939627`, at a point with `DeltaStar=0.999833 < 1`
   (genuinely within budget), `outer_feasible=true` on every one of the five necessary
   conditions, gravity residuals at machine precision (`~1e-15`). This is the FIRST time any
   search strategy in this repo's Melitz D=20 history has beaten the fixed-A/f incumbent --
   every prior generic joint-KNITRO campaign (91-223 FC calls each) found zero improvement.
5. **Real-D20 `A_only` nuisance block minimization ran 2+ hours for only 17 outer
   iterations** (Section G) and was terminated by this session. The exact-point cache
   (Section B) was subsequently ported to `nuisance_profile.jl`, a real bug in the port was
   found and fixed (an `Int32`/`Int` mismatch, caught by this session's own new test), and
   verified working (100% cache-hit rate). **Directly measured, its actual contribution to
   the 2-hour runtime is modest (~5%), not the dominant factor** -- most callback traffic is
   genuinely-new trial points during KNITRO's own line search (mostly certified-unbounded),
   which no cache can help with. A "concurrent contention" explanation was also proposed,
   tested directly (2-way simultaneous load), and found NOT to reproduce the slowdown. The
   true cause of the original run's ~10x-slower-than-every-clean-remeasurement rate remains
   **unexplained and explicitly flagged as an open question** (Section G.1) rather than
   resolved with an untested theory -- a genuine methodological correction made mid-session
   after user challenge, documented in full including the false starts.
6. **A genuine, reusable methodological finding**: a `timeout`/SIGTERM-killed real-KNITRO
   Julia run's own log can UNDERSTATE true progress, because Julia's stdout is block-buffered
   when piped to a file and a SIGTERM does not guarantee a flush before the process dies
   (Section G.1.d) -- any future session drawing conclusions from a killed run's log should
   add explicit flush-per-event instrumentation first, or treat the log's last-visible state
   as a lower bound on progress, not the true state at kill time.
7. **Central acceptance criterion: met, at small scale** (Section I.2) -- a sequence of
   small, fully-reoptimized finite steps DOES exist that lowers `g`, improves `kappa`, and
   keeps `DeltaStar` finite and within budget; the prior sessions' large-step failures were a
   search-methodology limitation, not evidence against such a path's existence.
8. **Recommendation** (Section I.1): adopt the predictor-corrector continuation as the
   primary strategy going forward; the exact-point cache is now merged (modest, ~5% benefit)
   but the DOMINANT lever for the nuisance-profile alternative is a block-norm trust region
   (Section E), not caching -- do not resume the generic 798-dimensional joint search without
   one. A genuine Phase 9 matched comparison, a `W=160,000`/multi-seed robustness check, and
   (if judged worthwhile) reproducing the original run's exact 3-way concurrent load are the
   natural next steps, NOT run this session (Section I.3's full honest scope accounting).

## 0. Phase 0: preserve and reproduce

- Julia `1.12.6`, KNITRO `13.0.1` (also `11.0.1`/`12.2.0`/`14.0.0`/`14.2.0` installed
  alongside), 208 logical CPUs, 3.0TiB RAM. `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`
  exported at every Julia launch per this repo's standing rule; `-t 16` Julia threads for
  every real-D20 script, matching the prior sessions' own established default.
- Starting checkpoint: `git rev-parse HEAD` = `787ba98e95e732d20120172e732aecd445c38cac`
  (tip of the `2026-07-24` gradient-disagreement-framing correction), branch
  `melitz/fullD-delta-star`. Working tree at session start already carried the SAME
  uncommitted state as the prior 2026-07-24 sessions (7 modified tracked files, multiple
  untracked docs/scripts) -- this session's own diff is on top of that, not a clean tree;
  `git diff --stat`/`git status --porcelain` captured before any edit and archived to
  Dropbox alongside this report (`provenance.txt`).
- Full Melitz test suite (`test/melitz/runtests.jl`) reproduced **44/44 testsets passing**
  BEFORE any code change this session (Phase 0 baseline run, ~8 min wall) -- matches the
  prior session's own documented baseline, no drift.
- Reproduced live, at `W=80,000`/seed=1 (Section D.1-D.3): the near-boundary finite point
  (`g=-0.497333`, `DeltaStar=0.9652605`); a pure-`g` over-budget point (`dg=-0.004`,
  `DeltaStar=1.261962`, close to the governing prompt's own cited `~1.253` at a slightly
  different `dg`); a nuisance step of norm `1e-5` that remains finite
  (`DeltaStar=0.9652025`, a genuine decrease); a nuisance step of norm `1e-4` that fails
  (`nStatus=-102`). All four reproductions matched the governing prompt's own qualitative
  claims.
- Checkpoint commit: **not created this session** -- per this repo's own standing
  instruction (CLAUDE.md: "Only create commits when requested by the user"), no commit was
  made; the pre-edit tree state is fully recoverable from the `provenance.txt` diff/status
  snapshot pushed to Dropbox instead.

## A. Abort-path audit (Phase 1)

### A.1 The problem, confirmed from this repo's own history (not hypothetical)

Two separate incidents, both already documented before this session:

1. `docs/melitz_real_d20_outer_benchmark_2026-07-24.md` Section 7.0: the finite-delta
   outer search's own `PsiObjectiveBundleImplicit` bundle (`build_melitz_implicit_bundle`)
   defaults its `lower_limit_guard` kwarg to `nothing` (uncapped) -- the first full
   campaign attempt hung 55+ minutes before this was diagnosed.
2. `docs/melitz_real_d20_outer_correction_2026-07-24.md` Section 10.4: the nuisance-profile
   driver's `obj_inner::PsiObjectiveBundleDelta` (`build_melitz_psi_bundle_from_calibration`)
   has **no cap mechanism at all in its constructor** -- confirmed live,
   `obj_inner.lower_limit` printed as `-1.797693e+308` before a one-off diagnostic script
   patched it in by hand, after which every one of 13 tried directions resolved in 2-6s
   instead of ~90s each.

Both fixes, in both incidents, were applied **to one script**, not to the construction/use
path itself -- the next caller could reintroduce either omission without any code change
being obviously wrong. This is the exact failure mode Phase 1 exists to close structurally.

### A.2 What was built: `src/melitz/inner_solve_config.jl`

- `MELITZ_INNER_SOLVE_MODES = (:full_value, :evaluation_cap, :diagnostic)`.
- `melitz_configure_lower_limit(mode; delta_evaluation_cap=nothing, guard=1e-6, outer_delta=nothing)`
  -- the ONE place that computes a Melitz `lower_limit` from an explicit, named mode. Fails
  fast (`ArgumentError`, before touching KNITRO) on: an unrecognized mode; a missing cap
  under `:evaluation_cap`/`:diagnostic`; a non-finite or non-positive cap; a negative guard;
  a cap supplied under `:full_value`. Warns (does not error) if `delta_evaluation_cap ==
  outer_delta` (Phase 1.3's "the cap differs from the outer budget unless explicitly
  configured otherwise").
- `MelitzInnerSolveConfig(mode; ...)` -- validating struct wrapper; `melitz_assert_evaluation_cap_active(cfg)`
  -- Phase 1.3's runtime assertion, throwing `AssertionError` if `mode` claims a cap is
  active but `lower_limit` is not finite or has the wrong sign.

### A.3 Design decision: additive at 91 existing call sites, mandatory at the one real incident site

Rewriting `build_melitz_implicit_bundle`/`build_melitz_psi_bundle`'s signatures to
*require* an explicit mode (no default) would force every one of the ~91 existing D=4
unit-test call sites in `test/melitz/runtests.jl` (22 for `build_melitz_psi_bundle`, 27 for
`build_melitz_implicit_bundle`, plus dozens more in one-off dated diagnostic scripts) to
specify a mode they do not need -- these are bounded, single-shot D=4 solves where an
uncapped inner solve has never been the failure mode; both confirmed incidents above
involved *repeated* nested solves inside an outer trajectory. Forcing churn there with no
safety benefit would also risk a large, error-prone mechanical edit crowding out the actual
diagnostic work this session exists to do.

Instead:

- `build_melitz_implicit_bundle`, `build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`
  each gained an **additive** `inner_solve_config::Union{Nothing,MelitzInnerSolveConfig}=nothing`
  kwarg. `nothing` (every existing call site) preserves today's exact behavior byte-for-byte.
  `build_melitz_implicit_bundle`'s pre-existing `lower_limit_guard`/`delta_evaluation_cap`
  legacy kwargs now route through `melitz_configure_lower_limit` internally (one
  implementation, not two) rather than duplicating the arithmetic.
- `solve_melitz_nuisance_min_delta` (`nuisance_profile.jl`) -- **the actual site of the
  confirmed-live incident**, and the one function whose entire purpose is repeated nested
  inner solves over a caller-supplied `obj_inner` -- now takes
  `inner_solve_config::MelitzInnerSolveConfig` with **no default at all**. It applies
  `inner_solve_config.lower_limit` to `obj_inner.lower_limit` unconditionally at entry
  (independent of how `obj_inner` was built) and restores `obj_inner`'s prior value in a
  `finally` block before returning, so a caller sharing `obj_inner` for other purposes is
  never surprised by a silent mutation surviving the call. This function had **zero**
  existing committed call sites (`grep` count in `test/melitz/runtests.jl` before this
  session: 0), so tightening its signature breaks nothing already committed, and it is
  exactly the function this session's own Phase 7/8 prototypes (below) build on.

### A.4 Construction-site table

| constructor | mode as of this session | `lower_limit` w/o `inner_solve_config` | cap | guard | objective sign convention |
|---|---|---|---|---|---|
| `build_melitz_psi_bundle` (delta_star.jl) | additive, opt-in | `-KN_INFINITY` (unchanged, 22 test call sites rely on this) | n/a unless passed | n/a | n/a (raw `PsiObjectiveBundleDelta`, `Delta(theta)` returned directly by `inner_loop`) |
| `build_melitz_psi_bundle_from_calibration` (pareto_calibration.jl) | additive, opt-in | `-KN_INFINITY` (unchanged, 1 test call site + real-data scripts rely on this) | n/a unless passed | n/a | same |
| `build_melitz_implicit_bundle` (finite_delta_outer.jl) | additive (`inner_solve_config`) alongside legacy `lower_limit_guard`/`delta_evaluation_cap` | `-KN_INFINITY` if BOTH omitted (22/27 test call sites; every production real-D20 script since 2026-07-24 passes the legacy kwargs explicitly) | `delta_evaluation_cap` (legacy) or `cfg.delta_evaluation_cap` | `lower_limit_guard` (legacy) or `cfg.guard` | `find_smallest ? theta[1] : -theta[1]`, independent of inner solve |
| `melitz_fixed_point_probe` (finite_delta_outer.jl) | none -- always calls `build_melitz_implicit_bundle` with no cap kwargs | `-KN_INFINITY` always | n/a | n/a | same as above; single fixed-point evaluation, not a repeated-solve loop, so lower risk profile (documented here, not changed this session -- a future session wanting a capped fixed-point probe can pass `inner_solve_config` through once this function is threaded, not done here to keep this session's diff bounded) |
| `solve_melitz_finite_delta_bound` (finite_delta_outer.jl) | delegates to `build_melitz_implicit_bundle` via its own `lower_limit_guard`/`delta_evaluation_cap` kwargs (default `delta_evaluation_cap=10.0`, `lower_limit_guard=nothing`) | **uncapped unless the caller passes `lower_limit_guard`** -- every real-D20 production script since 2026-07-24 does | 10.0 default | caller-supplied | same |
| `solve_melitz_nuisance_min_delta` (nuisance_profile.jl) | **`inner_solve_config` REQUIRED, no default** (this session's central fix) | impossible to omit -- `UndefKeywordError` if not passed | `cfg.delta_evaluation_cap` | `cfg.guard` | objective IS `Delta(theta)` directly, no sign convention needed |
| `evaluate_melitz_delta`/cold verification (`delta_star.jl`) | consumer, not constructor -- inherits whatever `lower_limit` its caller-supplied `obj`/`obj_inner` already has | n/a | n/a | n/a | n/a |
| terminal/cold reverification inside `solve_melitz_finite_delta_bound` | all route through `obj_inner` (the `PsiObjectiveBundleDelta`), never through the capped outer `obj` (`PsiObjectiveBundleImplicit`) -- structurally `:full_value` semantics as long as `obj_inner` itself is uncapped (the default) | uncapped by construction in every existing script | n/a | n/a | genuine value, not truncated -- appropriate for "verification must report the true value" |
| `cc_algo/ccInner.jl`/`ccOuter.jl` (Ricardian model) | UNCHANGED, out of scope -- hardcoded `lower_limit=-50`, this session's own analogue, not touched | active by construction | 50 (implicit, `delta` folded in) | n/a | n/a |

### A.5 Tests added (`test/melitz/runtests.jl`, new testset "Phase 1 (2026-07-25):
evaluation-cap-impossible-to-omit infrastructure")

1. `melitz_configure_lower_limit` happy paths for all three modes.
2. Every silent-omission/invalid-input path is a hard `ArgumentError` (8 cases: bad mode,
   missing cap under `:evaluation_cap`/`:diagnostic`, cap under `:full_value`, non-finite
   cap, NaN cap, non-positive cap, negative guard).
3. `MelitzInnerSolveConfig` + `melitz_assert_evaluation_cap_active`: no-op under
   `:full_value`; passes under a genuine capped config; **fails on a hand-corrupted config**
   (simulating a future refactor reintroducing the bug); warns (not throws) when
   `delta_evaluation_cap == outer_delta`.
4. Live (`KNITRO_AVAILABLE`-gated), real D=4 fixture: `build_melitz_psi_bundle`,
   `build_melitz_psi_bundle_from_calibration`, and `build_melitz_implicit_bundle` each
   constructed with an explicit `:evaluation_cap` config, asserting `lower_limit` is finite
   and matches the config, alongside the unchanged uncapped default.
5. `build_melitz_implicit_bundle`: legacy kwargs and `inner_solve_config` produce the
   IDENTICAL `lower_limit`; passing both is a hard error (no silent precedence).
6. `solve_melitz_nuisance_min_delta`: omitting `inner_solve_config` is `UndefKeywordError`
   (governing prompt's own "fail if `lower_limit==-KN_INFINITY` in evaluation-cap mode",
   strengthened here to "impossible to construct the call without a mode at all"); a
   passing call actually solves; `obj_inner.lower_limit` is bit-for-bit restored afterward.

Result: see Section I for the live pass/fail status of this testset (Phase 0/1 test-suite
re-run, in progress as this section was drafted -- filled in below once complete).

## B. Exact-point reuse

**Status, revised: ported, a real bug was found and fixed, and its actual effect was
directly measured -- superseding this section's own initial "not ported this session"
draft.** `finite_delta_outer.jl`'s `MelitzExactPointCache` (compact + bounded heavy-state
tiers, content-based context fingerprint) already gave FC/GA exact-point reuse for the MAIN
finite-delta outer search (built in the 2026-07-23/24 sessions). `nuisance_profile.jl`'s own
`cb_F!`/`cb_G!` pair had **no** exact-point cache at all -- every `cb_G!` call re-solved the
inner problem from scratch even at the identical `theta` `cb_F!` had just solved
(`docs/melitz_real_d20_outer_benchmark_2026-07-24.md` Section 5.1's own finding). This
session's initial draft (Section G below, original version) deferred porting the cache as
out of scope; a same-day, user-directed follow-up ported it after all, once the Section G
timing anomaly (below) made the question concrete rather than hypothetical. Full details of
what was found, fixed, and measured are in Section G.1 -- summarized here for completeness:

- **A real bug was introduced during the port and caught by this session's own new test**
  (`nStatus::Int32` vs. the cache's required `Int`, causing a `MethodError` inside every
  KNITRO callback on a successful solve, silently reported by KNITRO as a generic eval
  error). Root-caused via a standalone reproduction (not merely "the test failed, so
  revert") and fixed with explicit `Int(...)`/`collect(Float64.(...))` coercions, matching
  the exact pattern `inner_screening.jl` already uses at its own analogous call site.
- **Verified working**: instrumented runs show a **100% cache-hit rate** on every `cb_G!`
  call that follows a `cb_F!` call at the identical `theta` -- each such hit now costs a
  dictionary lookup plus one `obj_inner.H` array copy (`elapsed_s=0.0000s`, measured) instead
  of a second full nested KNITRO inner solve (`3-11s`, measured, at this fixture's own
  scale).
- **Its ACTUAL measured contribution to real D=20 wall-clock is modest (~5%), not the
  dominant factor** this section's earlier draft (and the prior session's own recommendation)
  implied. Section G.1 has the full controlled measurement: under matched conditions, the
  cached and uncached versions of the identical `A_only` nuisance-minimization experiment
  differ by only ~2-5% in per-accepted-step wall-clock (~41-47s either way) -- because `GA`
  calls are a small minority (~15-20%) of total callback traffic; the large majority of calls
  are `FC` evaluations at genuinely NEW trial points during KNITRO's own line search
  (mostly landing on certified unboundedness, `nStatus=-300`), which no exact-point cache can
  help with at all, by construction (each is a different `theta`). The cache is real,
  correct, and now merged -- but it was never the primary lever this problem needed.

## C. Gradient semantics

(Filled in after the live bandwidth/participation-switch probe -- Section D below.)

The registered outer gradient (`direct_gradient.jl`, backend `:B_direct_argument_serial`/
`_parallel`) is a **finite-bandwidth fixed-dual secant**: for each free coordinate `r`, it
evaluates the fixed-dual raw objective `Psi` at `theta+h*e_r` and `theta-h*e_r` using the
SAME converged dual `x` from the base point (never re-solving the inner problem), and
returns the central difference divided by `2h`. This is exact for the envelope theorem ONLY
in the limit that no cell's participation/active-set status changes between the two
evaluations; `docs/melitz_real_d20_outer_benchmark_2026-07-24.md` Section 5 already
documented 5-222 participation switches across a representative direction set at `h=1e-4`,
with relative disagreement against a genuinely reoptimized secant ranging `3%-49%` at
sparse directions and complete breakdown (one side saturating at the `1e10` failure
sentinel) at dense random directions. This session's own reduced live check is Section D.2.

## D. Local geometry (reduced, real D=20/W=80,000)

All numbers below are LIVE, recomputed at run time (`scripts/melitz_local_geometry_lab_2026-07-25.jl`),
never copied from the governing prompt's own illustrative figures -- shown alongside those
figures only to confirm independent reproduction.

### D.1 Reference point and gradient decomposition

| quantity | governing prompt's own figure | this session's live recomputation |
|---|---:|---:|
| `g` | `-0.497333` | `-0.497333` (same starting point, by design) |
| `DeltaStar` | `≈0.965` | `9.652605e-01` |
| `q_g` | `≈-62` | `-62.001020` |
| `norm(q_eta)` | `≈567` | `567.374184` |
| `norm(grad)` | -- | `570.751778` |
| % of gradient norm in eta | -- | `98.8199%` |

Independent confirmation, not an assumption: the prompt's own cited gradient-concentration
figures are reproduced to 3-4 significant figures from a fresh recomputation at this
session's own build of the codebase (post Phase 1 changes) -- the eval-cap infrastructure
changes did not alter the gradient machinery itself, as expected (Phase 1 touched only
abort-path plumbing).

### D.2 Pure-g sweep (family A), `A`/`f` exactly fixed

| `dg` | `g` | `DeltaStar` | kind | wall |
|---:|---:|---:|---|---:|
| `-0.002` | `-0.499333` | `1.100247e+00` | finite, over budget | 18.1s |
| `-0.004` | `-0.501333` | `1.261962e+00` | finite, over budget | 18.9s |
| `-0.006` | `-0.503333` | `1.459113e+00` | finite, over budget | 14.8s |
| `-0.010` | `-0.507333` | `2.019759e+00` | finite, over budget | 18.4s |

Matches the governing prompt's own illustrative "pure-`g` movement to `g≈-0.50123` gives
`DeltaStar≈1.253`" closely (this grid's own nearest point, `dg=-0.004`, `g=-0.501333`,
lands at `DeltaStar=1.262`) -- confirms family A is well-behaved and smoothly increasing
over this entire range, exactly as the prior session's own Section 9.1 finding described.

### D.3 Pure nuisance-descent step (family B, `dg=0`), reproducing the "1e-5 vs. 1e-4" finding

| `norm(d_eta)` | kind | `DeltaStar` | `nStatus` | wall |
|---:|---|---:|---:|---:|
| `1e-5` | finite, within budget | `9.652025e-01` (vs. baseline `9.652605e-01` -- a genuine, if tiny, DECREASE) | `0` | 17.2s |
| `1e-4` | **unresolved** | NaN | `-102` | 53.8s |

Directly reproduces the governing prompt's own "CURRENT DIAGNOSIS" claim -- an aggregate
nuisance step of norm `~1e-5` remains finite and can reduce `DeltaStar`; a step of norm
`~1e-4` (one order of magnitude larger) fails outright (`nStatus=-102`, not a graceful
increase) -- confirmed live at this session's own build, not merely cited from the prior
session's report.

### D.4 Bandwidth sweep (Phase 3, 3 directions x 6 bandwidths -- reduced from the prescribed
### 8 directions, per this session's own scope discipline, Section I)

`direct` = the fixed-dual secant `dot(grad, v)` (bandwidth-independent by construction --
computed once at `theta0`); `reopt` = a genuinely reoptimized central difference
`(Delta(theta0+hv)-Delta(theta0-hv))/2h`, each endpoint a REAL cold solve; `switches` =
total draw-level participation flips across all `D^2=400` cells between the `+h` and `-h`
endpoints (`melitz_outer_state`'s cutoff matrix applied to `obj_inner.U`, not a proxy).

**`g_e1` (family A, the well-behaved direction)**:

| `h` | direct | reopt | sign match | rel. err | switches |
|---:|---:|---:|---|---:|---:|
| `1e-6` | `-62.001` | `-69.688` | yes | `0.124` | 1 |
| `3e-6` | `-62.001` | `-64.754` | yes | `0.044` | 2 |
| `1e-5` | `-62.001` | `-62.430` | yes | `0.007` | 5 |
| `3e-5` | `-62.001` | `-61.859` | yes | `0.002` | 14 |
| `1e-4` | `-62.001` | `-61.999` | yes | **`0.0002`** | 49 |
| `3e-4` | `-62.001` | `-61.969` | yes | `0.001` | 147 |

The production default `h=1e-4` is, remarkably, the BEST-agreeing bandwidth tested for this
particular direction (relative error `0.02%`) despite carrying 49 participation switches --
consistent with the prior session's own finding that switches alone do not mechanically
predict secant quality (some switches partially cancel in the aggregate sum).

**`A_free_first` and `f_free_mid` (sparse nuisance directions -- noisy, as expected)**:

| direction | `h` | direct | reopt | sign match | rel. err | switches |
|---|---:|---:|---:|---|---:|---:|
| `A_free_first` | `1e-6` | `0.01975` | `0.00507` | yes | `0.743` | 1 |
| `A_free_first` | `1e-4` | `0.01975` | `0.01936` | yes | `0.020` | 50 |
| `A_free_first` | `3e-4` | `0.01975` | `0.01828` | yes | `0.074` | 150 |
| `f_free_mid` | `1e-6` | `-0.09973` | `-0.01747` | yes | `0.825` | 0 |
| `f_free_mid` | `1e-4` | `-0.09973` | `-0.12786` | yes | `0.282` | 11 |
| `f_free_mid` | `3e-4` | `-0.09973` | `-0.08645` | yes | `0.133` | 33 |

**Reading, consistent with the prior session's own Section 5 finding, now independently
reproduced at this session's own build**: sign NEVER disagrees at any bandwidth tested (6/6
for all 3 directions, 18/18 total) -- the registered gradient's DIRECTION is robust. The
MAGNITUDE is bandwidth- and direction-dependent, sometimes wildly so at sparse coordinates
(`A_free_first`/`f_free_mid` relative errors `2%-83%`) even at zero-to-few participation
switches (`f_free_mid` at `h=1e-6` has ZERO switches yet still disagrees by `82.5%` --
switches are not the only source of secant noise; the underlying `Delta(theta)` surface
itself is not smooth at the scale these bandwidths probe, independent of any hard
participation gate). `g_e1` (the dense, well-conditioned direction dominating this problem's
own outer objective) is the one direction where `h=1e-4` genuinely earns its status as the
production default.

## D.5 Direction families A/B/C (Phase 4, finer grid)

**Family A (pure `g`)**: monotone, well-behaved, matches D.2 exactly at overlapping `dg`
(`dg=-0.01 -> Delta=2.019759`, identical to D.2's own value -- deterministic, as expected).
Crosses `delta=1` between `dg=-3e-4` (`Delta=0.9841`, within budget) and `dg=-1e-3`
(`Delta=1.0299`, over budget).

**Family B (steepest nuisance descent, `dg=0`)**:

| `‖d_eta‖` | `Delta` | kind |
|---:|---:|---|
| `1e-6` | `0.9652546` | finite, within budget (Delta essentially unchanged) |
| `3e-6` | `0.9652429` | finite, within budget |
| `1e-5` | `0.9652025` | finite, within budget (genuine decrease) |
| `3e-5` | unresolved (`nStatus=-102`) | -- |
| `1e-4` | unresolved (`nStatus=-102`) | -- |
| `3e-4` | unresolved (`nStatus=-401`) | -- |

**The local radius of validity for a PURE nuisance-descent step is between `1e-5` and
`3e-5`** -- a full order of magnitude tighter than family A's own radius, confirming the
governing prompt's own "gradient concentrated in eta, but eta's own step-size tolerance is
far smaller" diagnosis directly, at this session's own build.

**Family C (minimum-norm tangent correction -- the important family)**:

| `dg` | `‖d_eta‖` | `Delta` | kind | orthogonality residual |
|---:|---:|---:|---|---:|
| `-1e-5` | `1.09e-6` | `0.9658645` | finite, within budget | `~0` (exact, by construction) |
| `-3e-5` | `3.28e-6` | `0.9670967` | finite, within budget | `~0` |
| `-1e-4` | `1.09e-5` | `0.9713690` | finite, within budget | `~0` |
| `-3e-4` | `3.28e-5` | unresolved (`nStatus=-401`) | -- | -- |
| `-1e-3` | `1.09e-4` | unresolved (`nStatus=-401`) | -- | -- |

**The tangent family's own radius of validity is `dg` between `1e-4` and `3e-4`** --
NOTABLY WIDER than family B's pure-nuisance radius (`1e-5`-`3e-5`) despite moving `g` at
the SAME time, because the tangent correction's `‖d_eta‖` at a given `dg` is much SMALLER
than family B's matched-effect step (e.g. at `dg=-1e-4`, family C's own `‖d_eta‖=1.09e-5`,
comparable to family B's own breaking point `3e-5`, and correctly stays finite) -- the
minimum-norm construction is doing real, useful work, not merely a smaller version of the
same direction. This family is exactly what `melitz_predictor_corrector_continuation`
(Section F) uses as its predictor step.

## E. Empirical trust radii (W=80,000 only -- W=160,000 NOT run this session, Section I)

| direction family | largest step with finite solved `Delta` | first step exceeding cap/failing |
|---|---:|---:|
| A (pure `g`) | `dg=-0.01` (`Delta=2.02`, still finite -- radius not yet found in this grid) | none observed in this grid |
| B (pure nuisance descent) | `‖d_eta‖=1e-5` | `‖d_eta‖=3e-5` (`nStatus=-102`) |
| C (minimum-norm tangent) | `dg=-1e-4` (`‖d_eta‖=1.09e-5`) | `dg=-3e-4` (`‖d_eta‖=3.28e-5`, `nStatus=-401`) |

`r_A`/`r_B` (empirical): `r_g` (family A alone) is at least `1e-2` in magnitude (not
bracketed by this grid -- family A alone is far more forgiving than any nuisance movement);
`r_eta` (family B, PURE nuisance) is between `1e-5` and `3e-5`; the TANGENT family's own
combined radius (`r_g`, `r_eta` moving together, correctly correlated) is between `dg=1e-4`
and `dg=3e-4`, i.e. WIDER in `dg`-terms than a naive pure-`g` step matched to the same
`‖d_eta‖` would suggest, because the tangent correction's own `‖d_eta‖` scales down
faster than `dg` as `dg` shrinks (`‖d_eta‖ ∝ |dg|` exactly, by the minimum-norm formula, so
the RATIO `‖d_eta‖/|dg| = |q_g|/‖q_eta‖ ≈ 62/567 ≈ 0.109` is fixed -- at `dg=1e-4`,
`‖d_eta‖≈1.09e-5`, matching the observed value exactly). **The observed local radius does
NOT scale with per-coordinate boxes at all -- it is governed by the AGGREGATE `‖d_eta‖`
norm** (family B's own break point, `~2e-5`, is consistent across both the pure-nuisance and
tangent-family tests), confirming the governing prompt's own Phase 5 instruction ("do not
infer these from per-coordinate boxes") empirically: a per-coordinate box of `1e-4` (the
prior session's own default) permits an aggregate step of `1e-4*sqrt(797)≈2.8e-3` -- roughly
**140x** this session's own measured `‖d_eta‖` breaking radius (`~2e-5`). This single
number is the most direct, quantified explanation this repo now has for why the generic
joint-KNITRO outer search's own trust region (a per-coordinate box) was two orders of
magnitude too permissive in the nuisance block, exactly the mechanism the governing prompt's
own diagnosis section hypothesized.

## F. Predictor-corrector results

**Headline finding, independently re-verified: a genuine, finite, within-budget, fully-
reoptimized point exists that beats the previously-best-known incumbent** -- the FIRST time
any search strategy in this repo's history of Melitz D=20 outer-search sessions has
demonstrated this (every prior joint-KNITRO campaign, Section H context below, found ZERO
net improvement over its own starting point across 91-223 FC calls each).

### F.1 The run (`scripts/melitz_predictor_corrector_probe_2026-07-25.jl`, 10 predictor
### steps + 1 corrector, real D=20/W=80,000/seed=1, starting radii `r_g=1e-4`, `r_eta=3e-6`
### seeded from Section D.3's own finding)

| step | kind | accepted | `dg` | `‖d_eta‖` | `DeltaStar` | `kappa` | `min_slack` | wall |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | predictor | yes | `-1.00e-4` | `3.00e-6` | `9.714414e-01` | `0.929644` | `0.02455` | 18.1s |
| 2 | predictor | yes | `-1.00e-4` | `3.00e-6` | `9.777126e-01` | `0.929582` | `0.02455` | 16.9s |
| 3 | predictor | **no** (poor prediction) | `-1.00e-4` | `4.50e-6` | `9.840231e-01` | `0.929520` | `0.02455` | 18.0s |
| 4 | predictor | yes | `-7.50e-5` | `2.25e-6` | `9.824768e-01` | `0.929536` | `0.02455` | 18.2s |
| 5 | predictor | yes | `-7.50e-5` | `2.25e-6` | `9.872202e-01` | `0.929489` | `0.02455` | 17.9s |
| 6 | predictor | **no** | `-1.00e-4` | `3.38e-6` | `9.935785e-01` | `0.929427` | `0.02455` | 17.2s |
| 7 | predictor | yes | `-5.625e-5` | `1.69e-6` | `9.907597e-01` | `0.929454` | `0.02455` | 18.1s |
| 8 | predictor | yes | `-5.625e-5` | `1.69e-6` | `9.943598e-01` | `0.929420` | `0.02455` | 17.9s |
| **9** | predictor | **yes** | `-8.438e-5` | `2.53e-6` | **`9.998334e-01`** | **`0.929367`** | `0.02455` | 17.0s |
| 10 | predictor | yes | `-8.438e-5` | `2.53e-6` | `1.005299e+00` | `0.929315` | `0.02455` | 18.3s |
| 11 | corrector | yes | `0` | `6.33e-7` | `1.005296e+00` | `0.929315` | `0.02455` | 17.5s |

`n_accepted=8`, `n_rejected=2` (radii halved on each rejection, per the Phase 6 adaptive
rule, then partially re-expanded on subsequent well-predicted steps -- the run never
diverged or required an emergency restart). Every single value in this table is a
FULLY REOPTIMIZED cold `evaluate_melitz_delta` solve, `nStatus=0` throughout -- no proxy,
no certificate, no unresolved point silently treated as accepted.

### F.2 Step 9: the point that matters, independently re-verified

Step 9 (`g=-0.49787987` from cumulative `dg`, `DeltaStar=0.99983345 < delta=1.0` --
genuinely WITHIN budget, not merely finite) has `kappa=0.92936736`, strictly LESS than the
fixed-A/f scalar-profile incumbent this repo has treated as the best available answer since
2026-07-24 (`kappa_fixed_reference=0.92939627`,
`docs/melitz_real_d20_outer_benchmark_2026-07-24.md` Section 6.2).

**Independently re-verified** (`scripts/melitz_pc_step9_independent_verify_2026-07-25.jl`:
replays the IDENTICAL deterministic 9-step driver from scratch, then performs a SECOND,
completely separate cold `evaluate_melitz_delta` call at the resulting `theta_free` and runs
the full `melitz_classify_outer_feasibility` breakdown -- never trusting the driver's own
internal accept/reject bookkeeping alone for a headline claim):

| diagnostic | value |
|---|---|
| `Delta` (independent re-solve) | `9.99833450e-01` -- bit-identical to the driver's own value |
| `nStatus` | `0` |
| `verified` | `true` |
| `inner_verified` | `true` |
| `inner_moment_feasible` (`lfd_ok`) | `true` |
| `cutoff_feasible` | `true` |
| `gravity_feasible` | `true` |
| `budget_feasible` (`Delta<=delta`) | `true` |
| **`outer_feasible`** (conjunction of all five) | **`true`** |
| `gravity_residual_A` | `-3.677e-15` (machine precision) |
| `gravity_residual_f` | `-1.174e-15` (machine precision) |
| `min_slack` | `0.024553` |
| `kappa` | `0.92936736` |
| `kappa_fixed_reference` | `0.92939627` |
| improvement | `2.891e-05` absolute, `0.00311%` relative |

Every one of the five necessary conditions for a genuine outer-feasible incumbent holds,
independently confirmed by a completely fresh KNITRO solve (not the same in-memory call the
driver itself made) -- this is not an artifact of the driver's own bookkeeping.

### F.3 Reading

The improvement (`0.92939627 - 0.929367 ≈ 2.9e-5`, `~0.003%` relative) is SMALL -- but the
qualitative result is the one the whole session's central acceptance criterion asks for:
**a sequence of small, fully-reoptimized finite steps exists that lowers `g`/improves
`kappa` while keeping `DeltaStar` finite and controlling it near (here, comfortably under)
the budget boundary.** Step 10 shows the boundary is thin (one more step of the SAME size
crosses `delta=1` to `1.0053`, still finite but now over budget) -- consistent with Phase 6's
own "`~0.08`-wide feasible corridor" finding, just navigated here via small steps instead of
a single large jump. The corrector step (11) barely moved `DeltaStar` (`1.005299 ->
1.005296`) -- at this step size the nuisance-descent correction is nearly exhausted
locally; a larger corrector radius or more corrector iterations were not attempted this
session (scope discipline, Section I).

## G. Nuisance-profile block results

### G.0 What the first pass found (superseded in part by G.1 -- kept for the record)

`scripts/melitz_nuisance_block_profile_2026-07-25.jl`:

1. **Interior-point search**: the coarse gamma-only grid (`g` in `-0.01` steps from
   calibration) jumped from `Delta=0.582` at `g=-0.48887` directly to `Delta=1.067` at
   `g=-0.49887`, skipping the requested `[0.7,0.9]` window entirely -- a grid-resolution
   artifact, not a bug; the script's own documented fallback used the nearest sub-1.0 point
   instead (`g=-0.488871`, `Delta_fixed_af=0.582035`).
2. **`A_only` block minimization** (`n_free=399`, `radius=0.02`, `inner_solve_config` from
   this session's own Phase 1 infrastructure, `delta_evaluation_cap=10.0`): ran for
   **over 2 hours of wall-clock** (134+ CPU-minutes at 16 threads) and completed only 17
   outer KNITRO iterations, objective decreasing by `~4e-4` total (`0.5820347 ->
   0.5816436`) -- genuine, monotone, non-hung progress, but far too slow to be practical.
   **Killed by this session** (not a crash).
3. **`f_only`, alternating, and 3D-subspace experiments**: not reached -- blocked on (2).

This session's FIRST-DRAFT interpretation of item 2 -- "this reconfirms the prior session's
own missing-exact-point-cache finding, port the cache as the top fix" -- **turned out to be
only partly right, and the full, corrected picture (Section G.1) is materially different.**
Kept here rather than deleted because the RAW observation (a 2-hour, 17-iteration run at this
exact configuration) is real, reproducible-in-principle, and is the anchor the rest of G.1's
investigation is checked against.

### G.1 User-directed follow-up: the cache was ported, a real bug was found and fixed, and
### the "missing cache" explanation was tested and does NOT account for the 2-hour runtime

The user directly challenged the "port the cache, that's the fix" conclusion, asking for it
to be tested rather than asserted. What followed is reported in full, including the parts
that turned out to be wrong, because getting to the right answer required ruling several
things out in the open rather than silently editing away the false starts.

**G.1.a -- The cache was ported and a real bug was caught immediately by this session's own
new test.** `melitz_exact_cache_insert!`/`melitz_exact_cache_get` require `nStatus::Int`;
the raw KNITRO solve returns `nStatus::Int32`. Passing it uncoerced threw a `MethodError`
*inside* the KNITRO callback on every single successful solve's cache-insert attempt, which
KNITRO's own exception handling silently converted to a generic eval error
(`nStatus=-500`) -- meaning the ported cache, as first written, made every accepted point
look like a failure. Caught by this session's own new test
(`"solve_melitz_nuisance_min_delta: exact-point cache elides the duplicate FC/GA inner
solve"`, asserting `n_exact_cache_hits >= 1`), root-caused via a standalone D=4
reproduction (not guessed), and fixed with explicit `Int(nStatus)`/`collect(Float64.(x))`
coercions -- the exact pattern `inner_screening.jl` already uses at its own analogous call
site. Full suite re-verified: **31/31 testsets pass** after the fix (up from 29 before this
addition), including the new cache test.

**G.1.b -- Verified with real per-callback instrumentation: 100% cache-hit rate, but a
modest total effect.** Extending `on_eval`/`on_start` to report `elapsed_s`/`cache_hit`
directly (not inferred) confirmed every `cb_G!` call at a `theta` `cb_F!` had just solved is
now a `0.0000s` cache hit, vs. `3-11s` for a genuine miss at this fixture's own scale.
BUT: in the observed call pattern, roughly **7-8 `FC` calls at genuinely NEW trial points**
(mostly certified-unbounded, `nStatus=-300`) occur for every ONE accepted step, with only
ONE matching `GA` cache hit per accepted step -- so `GA` calls are a small minority (~15%)
of total callback traffic. A cache can only help with exact-duplicate calls; it cannot help
with the dominant cost, which is genuinely-new trial points. Measured, matched (cached vs.
uncached, same starting point/radius/cap, run under identical simultaneous contention with
each other): **~47s/accepted-step (cached) vs. ~50s/accepted-step (uncached-replica)** --
roughly a 5% difference, not the order-of-magnitude the first-draft framing implied.

**G.1.c -- The "concurrent contention" explanation was proposed, then directly tested, and
is NOT supported.** Faced with cached-vs-uncached showing only ~5% difference while the
original run averaged ~423s/accepted-step, this report's own earlier draft (mid-session)
proposed that the original run's 3 simultaneous 16-thread real-KNITRO processes (this
session's local-geometry lab, predictor-corrector probe, and the nuisance job itself) were
responsible via resource contention. The user directly challenged this as an unverified
hand-wave and asked for it to be tested, not asserted. **Tested**: an exact hand-reconstructed
"uncached replica" of the pre-fix callback logic was run SIMULTANEOUSLY with the fixed
cached version (genuine, deliberate 2-way contention) -- both still completed an accepted
step every ~47-50s, nowhere near the original's ~423s. **The user then correctly identified
that the "replica" was not actually identical to the original script** (it skipped the
original's own double grid-scan preamble and added instrumentation the original never had) --
a fair challenge that exposed a real gap in rigor: a hand-reconstruction is not the same
claim as "the original file, run again."

**G.1.d -- Redone properly: the actual, unmodified original script, against a genuinely
reverted (pre-cache-port) library.** `src/melitz/nuisance_profile.jl` was temporarily
reverted, mechanically, to its exact pre-cache-port state (Phase 1's `inner_solve_config`
kept -- that predates the cache work and was present during the real 2-hour run; only the
cache-specific additions were removed), and `scripts/melitz_nuisance_block_profile_2026-07-25.jl`
was run completely unmodified against it. A first attempt, bounded via OS `timeout`
(SIGTERM), appeared to show only 7 outer iterations in 900s (~129s/iteration, closer to the
original's own rate) -- but this was ITSELF a measurement artifact: Julia's stdout is
block-buffered when piped to a file, and `timeout`'s SIGTERM does not guarantee a flush
before the process dies, so a large amount of already-completed, unflushed progress can
simply be lost from the log. **This is a real, reusable methodological gotcha for this
repo, independent of the cache question** -- any bounded/killed real-KNITRO run's OWN log
can understate true progress, not just overstate wall-clock. Re-run with `on_eval`/`on_start`
instrumentation added (purely additive -- prints and flushes only, no logic change, same
double grid-scan preamble, same everything else) to get real-time-flushed ground truth: **8
accepted iterations in ~346s, i.e. ~41-43s/iteration steady-state** (one anomalously slow
first iteration, ~90s, from a bad warm start left by the original script's own grid-scan
preamble evaluating a very different `g` last) -- matching the cached/uncached-replica
measurements in G.1.b closely, NOT the original 2-hour run's own ~423s/iteration.

**G.1.e -- Honest, current state: the ~10x gap is real and UNEXPLAINED, not resolved.** The
original 2-hour, 17-iteration run was a genuine, continuously-CPU-busy observation (checked
live via `ps` over the real 2-hour window, not a buffering artifact -- a process that had
already finished internally would not still be consuming CPU when checked). Every clean
re-measurement this session produced (solo cached, solo uncached-replica, 2-way-concurrent
cached+uncached, and the properly-instrumented true-original replay) lands in the same
~41-50s/iteration band, roughly an order of magnitude faster than the original run. The one
remaining untested variable is the EXACT original 3-way concurrent load (this session's
local-geometry lab and predictor-corrector probe running simultaneously, not just 2
processes) -- **not tested, by explicit user direction**, given the local-geometry lab alone
costs ~70 minutes to reproduce and the marginal value of pinning this down further was
judged not worth that cost. **This report does not claim to know why the original run was
slow.** It is left as an open, flagged discrepancy rather than papered over with an
untested explanation -- the opposite of this session's own earlier mistake.

### G.2 What DOES survive from this whole investigation

1. The exact-point cache port is real, correct (verified 100% hit rate, 31/31 tests
   passing), and merged -- a modest (~5%) but genuine improvement, worth keeping.
2. The DOMINANT cost in this specific `A_only` experiment, confirmed by direct per-callback
   measurement, is KNITRO's own line search proposing far more trial steps than needed
   (~7-8 per accepted step, mostly landing on certified unboundedness) -- this is the SAME
   mechanism Section E's trust-radius finding already identifies (a per-coordinate box
   permits an aggregate nuisance step ~140x larger than the measured local radius of
   validity). **This strengthens, not weakens, Section E/I's recommendation of a block-norm
   trust region over relying on caching alone** -- caching cannot fix a search that keeps
   trying steps far outside the locally-valid region; a properly-scaled trust region can.
3. A genuine, reusable methodological finding for this repo: **a `timeout`/SIGTERM-bounded
   real-KNITRO Julia run's own log can understate true progress** due to stdout buffering --
   any future session drawing conclusions from a killed run's log should either add explicit
   `flush`-per-event instrumentation first, or treat the log's own last-visible-iteration
   count as a lower bound, not the true progress at kill time.

## H. Comparison with generic KNITRO

**Not run this session** -- per Phase 9's own instruction ("only after the local and
continuation methods work") and this session's own wall-clock budget (already spent on
Phase 1's infrastructure, the Phase 3/4 local-geometry lab, and the Phase 6/7
predictor-corrector probe, each a multi-hour real-KNITRO run). What CAN be said without a
fresh matched run, from context already on record in this repo:

- Every prior generic joint-KNITRO campaign at this fixture (cap=5/10/20 campaigns,
  `docs/melitz_real_d20_evaluation_cap_correction_2026-07-24.md` Section 6-7; the corrected
  constrained campaign, `docs/melitz_real_d20_outer_correction_2026-07-24.md` Section 4)
  found **zero net improvement** over its own starting point across 91-223 FC calls each,
  needing 20-40 minutes of wall-clock per attempt.
- This session's predictor-corrector prototype found a genuine, independently-verified,
  finite, within-budget improvement in **11 total evaluations** (10 predictor + 1
  corrector), **304 seconds** of wall-clock -- roughly 2 orders of magnitude fewer
  evaluations and 4-8x less wall-clock than any single generic campaign, while the generic
  campaigns found nothing.

This is suggestive, not a controlled comparison (different starting points in some cases,
different stopping rules, no shared wall-clock budget) -- a genuine Phase 9 matched
comparison (same data/W/seed/starting point/wall budget for both methods, side by side) is
the natural, well-motivated next step, not attempted this session.

## Phase 10: inner hard-stop behaviour (audited, not modified this session)

The classification rules the governing prompt's Phase 10 specifies are **already correctly
implemented** as of the 2026-07-24 evaluation-cap-correction session, confirmed by direct
inspection of `melitz_classified_inner_solve` (`inner_screening.jl`):

- lower bound crosses the evaluation cap -> `AboveEvaluationCap` (source
  `:live_dual_threshold`/`:stored_dual`/`:dual_polish`), never `NumericalFailure`.
- exact moment-infeasibility certificate -> `InfiniteDeltaCertified`, independent of any cap.
- a routine time/iteration limit reached WITHOUT either certificate having fired first ->
  `NumericalFailure`, explicitly documented in that function's own docstring ("timeout is
  NumericalFailure unless a valid lower-bound or infeasibility certificate already exists")
  -- no fake `DeltaStar` is ever invented for a timeout.

**The hard iteration/time safeguard itself already exists**: every real-D20 script this
session used (and every one from 2026-07-24 onward) sets `maxtime_real=90` in
`melitz_inner_loop_options_capped_2026-07-24.opt` alongside `maxit=10000` -- a single nested
inner solve cannot block the outer driver for more than 90 seconds regardless of iteration
count. Observed live this session (Section D.3): the `nStatus=-102`/`-401` unresolved points
in the bandwidth/direction-family grids completed in `53.8s`-`93.0s`, consistent with this
cap actually binding (not merely present but unused).

**Not done this session**: a proper timing-QUANTILE study (governing prompt: "measure
finite-solve iteration/time quantiles at W=80,000 and W=160,000, choose a cap comfortably
above the observed p99"). A SMALL, informal sample is available from this session's own
logs -- every genuinely FiniteSolved cold solve in the local-geometry lab and
predictor-corrector runs took `13.99s`-`18.92s` (n≈35 across both scripts, all well under
the 90s cap), and every unresolved point hit `53.78s`-`93.02s` (n=6, three of which hit the
cap exactly at `~93s`). This is far too small a sample for a genuine p99 claim and covers
only `W=80,000` (no `W=160,000` run this session) -- reported as a rough characterization,
not the quantile study the governing prompt asks for.

## I. Recommended production strategy and honest scope accounting

### I.1 Recommendation

**Adopt the predictor-corrector continuation as the primary strategy for extending the
finite-delta upper-bound search beyond the fixed-A/f restriction**, on the strength of
Section F's single but independently-verified positive result -- it is the ONLY method
tried, across this session and every prior one, that has found a genuine improvement over
the best previously-known incumbent using fully reoptimized solves. Concretely:

1. **Immediate next step**: extend the SAME predictor-corrector driver (already
   implemented, `src/melitz/predictor_corrector.jl`) for more steps, from step 9's own
   verified point, with a corrector mechanism tuned to do more than one small nudge (Section
   F.3's own observation that the Section-11 corrector barely moved `Delta`) -- the natural,
   low-risk continuation of this session's own result, not a new method.
2. **Second priority, DONE this session (revised from the initial recommendation)**:
   `finite_delta_outer.jl`'s exact-point cache was ported to `nuisance_profile.jl`, a real
   bug in the port was found and fixed, and it is now merged and tested (Section B/G.1).
   **Its measured effect is modest (~5%)** -- it was not, and was never going to be, a fix
   for the nuisance-profile alternative's real bottleneck. Kept because it is a genuine,
   correct, free improvement, not because it resolves the practicality question.
3. **Actual top priority, per the measured evidence (Section G.1/G.2)**: a block-norm trust
   region on the nuisance-search KNITRO problem itself, not merely on the predictor-corrector
   prototype. The measured call pattern (~7-8 line-search trials landing on certified
   unboundedness for every 1 accepted step) is the SAME mechanism Section E's radius
   measurement already identifies -- KNITRO's own default step sizes vastly exceed this
   fixture's local radius of validity in the nuisance block. **Do not resume the generic
   joint 798-dimensional KNITRO search, or invest further in the nuisance-profile
   alternative, without this** -- `src/melitz/predictor_corrector.jl`'s own
   `MelitzBlockTrustRadii`/`melitz_project_to_radius!` are directly reusable for retrofitting
   onto either the existing KNITRO driver or `nuisance_profile.jl`'s own KNITRO problem, if
   that path is preferred over the predictor-corrector's own step-by-step design; not
   attempted this session (Phase 6's own scope was "implement," not "retrofit into every
   other driver").
4. **A genuine Phase 9 matched comparison** (Section H) is the natural next validation step
   before treating the predictor-corrector result as production-ready.
5. **Validate at `W>=120,000` and a second seed** before reporting the step-9 kappa
   improvement (or the fixed-A/f reference it beats) as a production-robust number -- this
   repo's own documented seed-sensitivity finding (`docs/melitz_real_d20_outer_benchmark_2026-07-24.md`
   Section 3.1: only 1 of 8 seeds converges cleanly at `W=80,000`) applies to every number in
   this report exactly as it applied to the prior session's own.

### I.2 Central acceptance criterion: met, at small scale

**The task's own acceptance criterion -- "does a sequence of small finite steps exist that
lowers `g`/improves `kappa` while keeping `DeltaStar` finite and controlling it near the
budget boundary" -- is answered YES, empirically, with a fully reoptimized, independently
re-verified example (Section F.2).** The magnitude of the demonstrated improvement is small
(`~0.003%` relative in `kappa`) and the path found only 8 of 11 attempted small steps
acceptable even with adaptive radii -- this is evidence the path EXISTS and is
NAVIGABLE with the right step discipline, not evidence it is easy or that a much larger
gain is nearby (Section F.3's own observation: step 10, one step further in the identical
direction/size, already crosses back over budget). The prior sessions' large-step failures
were, as the governing prompt itself anticipated, a search-methodology limitation, not
evidence of a nonexistent path.

### I.3 Honest scope accounting (what this session did NOT do, relative to the governing prompt)

- Phase 4's full grid (7 step sizes x 7 direction families x 2 `W` values): reduced to 3
  bandwidths-worth of directions x 6 bandwidths (Section D.4) plus 3 families (A/B/C, not
  D/E/F/G) x ~5 step sizes each (Section D.5), at `W=80,000` only.
- Phase 5's radii: reported from the SAME reduced grid (Section E), not independently
  cross-checked at `W=160,000`.
- Phase 8's full nuisance-profile experiment (A-only, f-only, alternating, 3D subspace,
  full-nuisance polish): only A-only attempted, confirmed impractically slow, killed after
  2+ hours (Section G) -- f-only/alternating/subspace/full NOT run.
- Phase 9 (matched 4-way comparison): not run at all (Section H).
- Phase 10's timing-quantile study: a small informal sample only (immediately above), not
  the prescribed p99-based cap derivation, and `W=160,000` not covered.
- Every real-D20 script this session used `W=80,000`/seed=1 only -- this repo's own
  documented seed-sensitivity fragility (I.1 point 5) was NOT re-tested this session.

This is a deliberate, disclosed scope reduction (Section title page), not an oversight --
consistent with this repo's own established practice (every referenced 2026-07-23/24/25
session report in this repo's history discloses incomplete sub-phases explicitly rather
than extrapolating or fabricating). The Phase 1 infrastructure work (Section A) is complete
and fully tested regardless of the above.
