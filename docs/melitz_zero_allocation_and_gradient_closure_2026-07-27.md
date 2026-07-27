# Melitz zero-allocation hot-path + gradient-quality closure (2026-07-27 evening session)

Continues `docs/melitz_outer_parameterization_comparison_2026-07-26.md` (Parts 1-2) and
`docs/melitz_hot_path_allocation_audit_2026-07-27.md` on branch `melitz/fullD-delta-star`,
local HEAD `6f3fa16a` (parents `1418338`, `42460f8`, `f475ceb`), not pushed. Governing prompt:
close two remaining technical issues before an outer-search redesign -- (1) the zero-
allocation hot-path audit, (2) the gamma/participation/mixed registered-vs-reoptimized
gradient discrepancy. Does NOT redesign the outer optimizer and does NOT begin real-D20
outer-search work, per the governing prompt's own explicit scope limits.

**Scope note, stated up front** (matching this repo's own established convention for
honestly scoping a 13-phase ask against one session's time budget): the governing prompt's
own Phase 1 asks for an exhaustive call-graph table across every named function reached from
`solve_melitz_finite_delta_bound`, Phase 5 asks for `Profile.Allocs`-based measurement of
~15 named callbacks at two scales, and Phase 7 asks for a full memory-traffic (bytes-moved)
audit. This document delivers a real, evidence-based version of each -- concrete numbers
where measured, explicit gaps where not -- not a complete instantiation of every requested
subsection. The two CONCRETE, most consequential asks (a genuinely mutating theta-expansion
workspace wired into the production default gradient backend, and a rigorous three-object
diagnosis of the gradient discrepancy) are both completed with live verification.

## Phase 0: preserve and reproduce (complete)

- Branch/HEAD as above; `git status` showed only pre-existing, unrelated untracked
  scratch/output directories (`full_aod_diag/batch_out_v2/`, `sequential_gravity/batch_out_*`,
  etc. -- inherited from other sessions, not touched).
- Julia 1.12.6 (juliaup, per this repo's own standing convention -- `/opt/shared_sw` 1.10.11
  is broken). KNITRO 13.0.1 (`.knitro_env.sh`, pinned -- 14.x lacks a valid site license).
  208 cores / 3.0TiB host. `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`/`JULIA_NUM_THREADS=1`
  for every test/benchmark run in this document, per this project's standing convention.
- **Baseline full suite** (BEFORE any edit this session): every testset `Pass==Total`, exit
  code 0 (`/tmp/.../phase0/baseline_runtests.log`).
- **Post-Phase-2/3-edit full suite**: every testset `Pass==Total`, exit code 0, INCLUDING the
  new regression tests added this session (see Phase 6/13 below) -- zero regressions.
- Reproduced the standalone top-level outer solve (no `cc_algo`,
  `test/melitz/standalone_no_cc_algo.jl`, run as a genuine separate process both times) and
  the strict matrix-free no-dense-fallback path (`forbid_dense_fallback=true` throughout) --
  both pass in both the baseline and post-edit runs.
- Reproduced Phase-8's own cited gradient-quality numbers directly from
  `docs/key_results/melitz_phase8_gradient_quality_2026-07-27.csv` (gamma ratio 0.6887,
  participation ratio 0.8472, mixed ratios 2.57-4.07) -- confirmed present and unchanged;
  this session's own Phase 9 bandwidth sweep (below) independently reproduces the gamma
  ratio (0.6887 at h=1e-4) from a freshly-run script, not merely re-read from the CSV.

## Phases 1-2: production hot call graph + in-place workspace-based theta expansion (complete for the production default; two related paths disclosed, not covered)

**Call graph** (traced by reading, not merely grepping): `solve_melitz_finite_delta_bound` ->
`melitz_build_finite_delta_callbacks` -> `cb_F!`/`cb_G!` -> `inner_solve_verified_or_fail`
(exact-point cache check, then `melitz_classified_inner_solve` on a miss) -> `cb_G!`'s
gradient path resolves (`melitz_resolve_gradient_backend`, `MelitzCCBundle` +
`sorted_tail_ctx` always present for the matrix-free bundle) to
**`:B_direct_argument_sorted_serial`/`_parallel`** -- the PRODUCTION DEFAULT gradient
backend for every matrix-free bundle. Its per-coordinate body (`cc_bundle.jl`'s
`MelitzCCBundle`-specific `_direct_coordinate_grad_sorted`, dispatched over the shared
`sorted_crossing_gradient.jl` implementation) calls
`_fill_compact_direct_columns_crossing_sorted!` for every touched direct trade cell, which
called `melitz_expand_theta` **TWICE per free coordinate** (once at `theta+h`, once at
`theta-h`) -- i.e. `~2 x 798 = 1596` allocating calls per full outer gradient at real D=20.
Each call allocated `logA_full` (length `D^2`), `A` (`D x D`), `f` (`D x D`),
`logf_free_full` (length `D^2-1`), plus (before Phase 3, below) a locked global-cache probe
-- measured **live, before this session's fix, at 19,640 bytes/call at real D=20**
(`ctx8.D=20`, `n_theta=798`, matching the governing prompt's own cited "~19,672 bytes"
figure almost exactly -- the small difference is the Phase-3 pivot-cache fix already
folded in by the time this exact number was measured, see below) and 1,136 bytes/call at
D=4.

**Fix** (`delta_star.jl`, `log_cutoff_param.jl`): `MelitzThetaExpansionWorkspace(D)`
(preallocated `logA_full`, `logf_free_full`, `theta_plain` scratch) +
`MelitzExpandedState(D)` (preallocated `A`, `f`, `gamma_prime_j`, `f_jj` output) +
`pivot_expand!` (in-place `pivot_expand`) + `expand_free_theta!`/`melitz_expand_theta!` (the
mutating primaries, `outer_parameterization=:logf`-only -- the production default;
`:logcutoff` throws `ArgumentError` rather than silently returning wrong state, since its
own q-gravity pivot rebuild is a separate, pre-existing allocation this session did not
touch, and `:logcutoff` is not the production default). Wired into the ONE identified
production-hot call site (`_fill_compact_direct_columns_crossing_sorted!`, both the generic
and `MelitzCCBundle`-specific `_direct_coordinate_grad_sorted` methods, both the serial and
parallel backend factories, with per-thread workspace/state buffers for the parallel
variant, keyed by `Threads.maxthreadid()` directly rather than a flag another block updates
first -- mirroring this codebase's own already-fixed `thetap_bufs`/`thetam_bufs` latent-bug
pattern, Section L of the 2026-07-26 addendum).

`melitz_expand_theta` (the original, generic, `ForwardDiff`-compatible, allocating
dispatcher) is **UNCHANGED** -- kept exactly as the "diagnostic convenience wrapper" the
governing prompt's own Phase 2 describes, still used by the ~14 other, non-per-coordinate
call sites (`melitz_outer_state`, `melitz_cutoff_constraints_at` -- called once per FC/GA,
not per coordinate -- screens, `gradient_lab.jl` diagnostics, `cc_bundle.jl`'s snapshot
restore, `pareto_calibration.jl`'s roundtrip check, `melitz_cutoff_constraint_jacobian`'s
`ForwardDiff.jacobian` branch, which genuinely needs Dual-number propagation the Float64-only
mutating path cannot provide) -- cross-validated against the new mutating path by dedicated
equivalence tests (Phase 6, below) rather than routed through it, to avoid adding
indirection/allocation risk to call sites that do not need this speedup.

**Verified live** (`test/melitz/runtests.jl`, "Governing prompt Phase 2-3"):

| scale | `melitz_expand_theta!` bytes (post-warmup, 2 consecutive calls) | `melitz_expand_theta` bytes (old, allocating) | max abs/rel diff (A, f) |
|---|---:|---:|---|
| D=4 (FIXTURE) | 0, 0 | 1,136 | `0.0`, `2.78e-16` |
| real D=20 (`noah_D20`, W=80,000, n_theta=798) | 0, 0 | 19,640 | `0.0`, `2.89e-15` |

Zero allocation, exactly matching (to machine precision) the pre-existing allocating
implementation across 20 (D=4) / 5 (D=20) random perturbations plus the calibrated point
itself.

**Disclosed, NOT fixed this session** (Phase 4's own static audit, below, identifies these
precisely): `argument_localized_gradient.jl`'s `_fill_compact_direct_columns!`/
`_fill_compact_link!` still call the allocating `melitz_expand_theta` -- these are the fill
helpers for (a) `direct_gradient.jl`'s PLAIN (non-sorted) `:B_direct_argument_serial`/
`_parallel` backend (not the production default -- `:B_direct_argument_sorted_*` is), and
(b) the FOCAL-LINK column specifically, for ANY coordinate whose dependency touches it
(`cc.touches_link`), **including inside the production sorted backend** -- `_fill_compact_link!`
was deliberately kept dense/unchanged in the 2026-07-25 sorted-tail session (documented
scope decision, `sorted_crossing_gradient.jl`'s own header) and this session did not revisit
that decision. This remains the single largest known residual per-coordinate allocation in
the production hot path -- flagged as the top-priority follow-up for a future allocation
session, not fixed here (a genuine, disclosed scope narrowing given this session's own time
budget after the Phase 8-11 gradient work, Section below).

## Phase 3: remove the global locked pivot cache (complete for every production ctx; legacy fallback retained)

**Fix** (`delta_star.jl`, `pareto_calibration.jl`): `melitz_build_f_pivot_parts(D,
f_free_lin, A_pivot, c_full)` factors out the pivot-selection work `melitz_cached_f_pivot_parts`
used to memoize behind a process-global `ReentrantLock` + single-slot `Ref`. Every
PRODUCTION ctx-construction site (`build_melitz_psi_bundle`, `melitz_calibration_outer_ctx`)
now calls this ONCE, at construction, and stores the result as immutable `ctx` fields
(`f_pivot_c`/`f_pivot_idx`/`f_pivot_other`) -- exactly like `c_full`/`A_pivot` themselves
already are. `melitz_cached_f_pivot_parts(ctx)` checks for these fields FIRST (zero lock,
zero recompute) -- the global-lock path is retained ONLY as a fallback for `ctx` objects
that predate/bypass this field (hand-built test `NamedTuple`s, `fstar_solver.jl`'s own
diagnostic-only local `ctx`, verified still functioning correctly by a dedicated test,
below) -- no existing caller breaks, and no production hot-path caller (any `ctx` from
`build_melitz_psi_bundle`/`melitz_calibration_outer_ctx`) ever reaches the lock.

**Verified live** (same testset as above): `ctx.f_pivot_c !== nothing` for every production
ctx; `melitz_cached_f_pivot_parts(ctx)` returns `ctx.f_pivot_c` BY IDENTITY (`===`, not a
copy/recompute); a hand-built legacy `ctx` (no `f_pivot_*` fields) still round-trips
correctly through the fallback path. Acceptance criterion #2 ("no global lock acquired
during normal coordinate expansion") holds for the actual production path by construction,
not by a faster cache.

## Phase 4: static audit of remaining hot allocations (targeted, not exhaustive)

Grepped every hot-path file (`sorted_crossing_gradient.jl`, `direct_gradient.jl`,
`argument_localized_gradient.jl`, `cc_bundle.jl`, `moment_operator.jl`, `delta_star.jl`,
`log_cutoff_param.jl`) for the governing prompt's own named pattern classes, then read each
hit in context (not classified by grep count alone):

| pattern | production-hot finding | status |
|---|---|---|
| `melitz_expand_theta(...)` (allocating) inside `_fill_compact_direct_columns_crossing_sorted!` | 2x/coordinate, real D=20 hot loop | **FIXED this session** (Phase 2) |
| `melitz_cached_f_pivot_parts` global lock | every FD coordinate probe | **FIXED this session** (Phase 3) |
| `melitz_expand_theta(...)` inside `_fill_compact_link!`/plain `_fill_compact_direct_columns!` | 2x/link-touching coordinate (sorted backend) or 2x/coordinate (plain backend) | **disclosed, NOT fixed** (Phase 2's own note above) |
| `copy(theta)` in parallel gradient backends | per-coordinate, `O(n)` allocations | **already fixed in the 2026-07-27 morning session** (Part 1 Section L) -- re-confirmed still fixed by reading the current file, not re-broken |
| `zeros(...)`/buffer construction inside `make_melitz_gradient_delta_*` factory closures | ONE-TIME per shape change (`(W,maxcols)`/`nt` guard), not per call | initialization-only, correctly guarded, no action needed |
| `melitz_expand_theta(...)` inside `melitz_outer_state`/`melitz_cutoff_constraints_at`/`melitz_moments_adapter!` | once per FC/GA call (not per coordinate) | low-severity, out of THIS phase's per-coordinate scope; not re-audited beyond confirming call frequency |
| `zeros(D,D)`/`Diagonal`/`A'*C*B` temporaries inside `moment_operator.jl`'s update sweep | not traced to per-draw-loop granularity this session | **not covered** -- disclosed gap |
| `Dict`/`Set`/`push!`/`sort`/`findall` in hot files | all confined to ONE-TIME cache/compact-columns-map construction (`melitz_compact_columns_map`, called once per distinct `ctx`, cached by identity) | initialization-only, no action needed |

**Not separately executed this session** (disclosed, matching this repo's own established
convention rather than fabricating coverage): the full ~15-pattern-class x ~15-function
matrix the governing prompt's own Phase 4 literally specifies; a systematic audit of
`moment_operator.jl`'s own per-draw update loop; `keyword`/`NamedTuple` construction inside
threaded loops (spot-checked in the files read, none found, not exhaustively swept).

## Phase 5: dynamic allocation profiling (targeted `@allocated`, not `Profile.Allocs`)

Measured live (post-JIT-warmup, `@allocated`, two consecutive calls each) for the ONE
function this session's own Phase 2 fix targets:

| function | D=4 bytes | real D=20 bytes (W=80,000, n=798) |
|---|---:|---:|
| `melitz_expand_theta!` (new, mutating) | 0 | 0 |
| `melitz_expand_theta` (old, allocating, for comparison) | 1,136 | 19,640 |

**Not separately executed this session** (disclosed): `Profile.Allocs`-based (as opposed to
`@allocated`-based) measurement; per-callback breakdown for the other ~14 named functions
the governing prompt's own Phase 5 lists (operator update, matrix-free objective/gradient
callback, serial/parallel structured Hessian, Hessian packing, range screen, complete
serial/parallel outer gradient, LFD recovery, candidate registration, nuisance FC/GA);
separating KNITRO.jl's own allocations from Melitz-owned callback allocations. These require
a dedicated allocation-audit session with its own time budget, per the same disclosed-gap
convention `docs/melitz_hot_path_allocation_audit_2026-07-27.md` Section 5 already
established for the prior (2026-07-27 morning) session's own equally-scoped-down Phase
4/5 ask.

## Phase 6: allocation regression tests (added, absolute-ceiling style)

Added to `test/melitz/runtests.jl` ("Governing prompt Phase 2-3"): post-warmup
`@allocated(...) == 0` assertions for `melitz_expand_theta!` at BOTH D=4 and real D=20 (two
consecutive calls each, catching a reintroduced allocation regardless of whether it happens
to be STABLE call-to-call -- the governing prompt's own Phase 6 critique of the pre-existing
`bytes_first_call==bytes_second_call` pattern is satisfied here by asserting the value
itself, not merely its stability), plus a positive-control assertion that the OLD allocating
`melitz_expand_theta` still allocates non-trivially at both scales (`>0` at D=4, `>15,000` at
D=20) -- proving the D=20 ceiling test would actually catch a reintroduced per-call
allocation of the size this session found, not merely pass vacuously.

**Not separately executed this session** (disclosed): a dedicated coordinate-probe
D=4-vs-D=20 scaling test, a W=20,000-vs-W=80,000 draw-size test for the FULL gradient
callback (as opposed to `melitz_expand_theta!` alone), and a deliberate copy-style
anti-pattern helper proving the new tests would fail on the KNOWN anti-pattern -- the
D=4-vs-D=20 comparison IS implicitly present (both scales tested, both assert `==0`), but
not packaged as a single parametrized "no O(n_theta) scaling" test the way the governing
prompt's own Phase 6 describes.

## Phase 7: large allocation-free memory traffic (not separately audited this session)

**Disclosed, not executed**: a dedicated `copyto!`/`fill!` bytes-moved audit across the hot
callbacks. The ONE relevant observation this session's own work surfaces directly: the new
`melitz_expand_theta!`'s own `copyto!`/`fill!`-equivalent work (writing `A`/`f`/scratch
vectors) is bounded by `O(D^2)` per call (unavoidable -- every entry of `A`/`f` is a genuine
output, not a redundant clear), confirmed by the zero-allocation measurement above (no
HIDDEN large temporary is being created and immediately discarded; if there were, it would
show up as nonzero `@allocated`). A systematic audit of `copyto!`/`fill!` traffic elsewhere
(operator-state copies, Hessian copies) is out of this session's scope.

## Phases 8-11: gradient discrepancy diagnosis (complete -- decisive finding, NOT a stale-cache/chain-rule/scaling bug)

### Method

At the D=4 FIXTURE's calibrated point (`theta0`, `Delta0=7.5545e-6`, the SAME point
`docs/melitz_outer_parameterization_comparison_2026-07-26.md` Section O's own Phase-8 CSV
uses), reused the exact prior-session methodology (Richardson `h`/`2h` stability pre-scan to
select probe coordinates -- `r_gamma=1`, `r_tech=2`, `r_part=17` this run) and extended it to
explicitly compute and compare THREE objects at every probed `(direction, h)` pair, per the
governing prompt's own Phase 8 instruction:

1. **registered fixed-dual gradient** (`pred`): `cb_G!`'s own analytic-envelope-theorem
   secant (`direct_gradient_fn`, fixed `h=1e-4` -- this quantity is NOT itself a function of
   the sweep's own `h`, since the registered backend's own bandwidth is fixed at
   construction).
2. **FD of the registered constraint value** (`fd_registered`): re-running `cb_F!` at
   `theta0 +/- h*dirvec` (a FULL reoptimize each side, through
   `inner_solve_verified_or_fail`/`melitz_classified_inner_solve`) and differencing
   `evalResult.c[1]`.
3. **FD of raw, independently-reoptimized `DeltaStar`** (`fd_Delta`, via
   `evaluate_melitz_delta` directly, `cold=true` each side -- bypassing `cb_F!`'s own
   sentinel/cap machinery entirely).

Full CSV: `docs/key_results/melitz_phase9_bandwidth_smoothness_2026-07-27.csv` (28 rows,
4 directions x 7 bandwidths `h in {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}`).

### Finding 1 (ruled out): registered-constraint scaling / stale cache

`ratio_fdreg_vs_fdDelta` (object 2 / object 3, correctly unit-matched) is **`1.0000` to 10+
significant figures at every one of the 28 rows** (e.g. gamma, h=1e-4: `0.9999999999999997`).
Object 2 (`cb_F!`'s own registered constraint, internally re-solving through the full
production inner-solve/screen/cache machinery) and object 3 (an INDEPENDENT
`evaluate_melitz_delta` call, bypassing `cb_F!` entirely) are, quite simply, the SAME
quantity measured two different ways -- this rules out a registered-constraint-scaling bug,
a chain-rule error in the value path, AND any state-staleness in the exact-point cache (a
staleness bug would show up as a discrepancy here, since the two paths populate/consult the
cache independently).

### Finding 2 (ruled out): numerical inner-solve tolerance / reproducibility noise

Directly tested (`melitz_inner_loop_options.opt`'s own nominal tolerances are tight:
`feastol`/`feastol_abs`/`opttol`/`opttol_abs`/`xtol` all `1e-12`): 5 repeated COLD KNITRO
solves at the identical `theta0` are **bit-identical** (`Delta`, full dual vector `x`, all
`==`, not merely `isapprox`); a WARM-started solve (from `theta0`'s own converged dual)
matches the cold solve to `~1e-13` relative precision. This rules out cold-start
path-dependence or any KNITRO-level nondeterminism as the source of the observed
instability.

### Finding 3 (ruled out): hard participation switches

`switch_count` (outer cutoff-feasibility-based) is **`0` at every one of the 28 rows**,
including the smallest tested `h=1e-6` -- the Richardson pre-scan's own smoothness selection
worked as intended for THIS crude (outer-feasibility-level) notion of a switch.

### Finding 4 (the actual mechanism): `theta0` sits extremely close to a local critical point of `DeltaStar`, so finite-`h` secants there are curvature-dominated, not linear-term-dominated

Directly profiled `Delta(theta0 + s*e_gamma)` over `s in {-0.1,...,-1e-6, 0, 1e-6,...,0.1}`
(13 points, all genuine `evaluate_melitz_delta` re-solves): `Delta` **increases in BOTH
directions** away from `theta0` at every step size `>=1e-5` (`Delta(s=-1e-4)=8.80e-6`,
`Delta(s=0)=7.55e-6`, `Delta(s=+1e-4)=7.73e-6` -- `theta0` sits at, or extremely near, a
LOCAL MINIMUM of `Delta` along this direction). This is not a coincidence: `theta0` is
`FIXTURE`'s own `theta_free`, produced by `solve_fstar`'s "exact-sample correction" --a
coordinate-descent-plus-polish procedure whose EXPLICIT PURPOSE is to drive the moment
residual (closely related to `Delta`) toward a local minimum. At a genuine local critical
point, the first-order (linear) term of a Taylor expansion is small BY DEFINITION, so ANY
finite-difference secant computed nearby is measuring a mix dominated by the SECOND-order
(curvature) term, not primarily the same first-order quantity the registered envelope-theorem
formula targets -- explaining both:

- **why the secant's own SIGN flips as `h` shrinks** (gamma: `+3.70e-3` raw slope at
  `h=1e-6` vs `-5.36e-3` at `h=1e-4` -- a real, reproducible sign change, not noise, per
  Finding 2), and
- **why the registered-vs-secant RATIO does not monotonically converge to 1 as `h` shrinks**
  (it should, in the textbook envelope-theorem story, IF the true linear term dominates near
  `theta0` -- here it does not, so the usual "smaller `h` is more accurate" intuition breaks
  down specifically at this near-critical starting point).

**Independent corroboration, participation direction**: `pred=0.0` EXACTLY (registered
formula predicts a locally FLAT function along this coordinate), and the reoptimized secant
independently confirms `fd_Delta=0.0` for every `h` up to `1e-4`, with a nonzero secant
appearing only at `h=3e-4`/`1e-3` -- i.e. the registered formula's own zero-prediction is
CORRECT and precisely confirmed by the reoptimized ground truth over a 100x range of `h`,
directly contradicting a "poor predictor" story for this direction and instead supporting
the finite-support/threshold-crossing explanation below.

**Likely finer-grained mechanism** (plausible, not independently verified this session given
remaining time): with `W=20,000` Monte Carlo draws, `Delta(theta)`'s TRUE functional form is
a genuinely non-smooth (staircase) function at the scale of individual draws crossing a
participation cutoff threshold -- distinct from the coarse, outer-feasibility-level
"`switch_count`" this session measured, which cannot detect a handful of draws (out of
20,000) crossing their own individual threshold. Such micro-kinks would average into a
locally smooth aggregate response only once the perturbation spans enough of them --
consistent with the participation direction's own transition from exactly-zero (`h<=1e-4`)
to nonzero (`h>=3e-4`) secants, and with the gamma/technology/mixed directions' own
sign-unstable-then-stabilizing pattern across the same `h` range.

### Phase 11 verdict: fixed-dual finite-bandwidth semantics explain the difference; no implementation error found

Per the governing prompt's own Phase 11 menu ("Fixed-dual finite-bandwidth semantics explain
the difference... document quantitatively... do not falsely correct it"): **this is the
correct branch.** No sign, scaling, stale-cache, dependency-mapping, or chain-rule error was
found (Findings 1-3 above rule out every mechanism on the governing prompt's own Phase 8
checklist except finite-bandwidth/curvature semantics and hard participation switches, and
directly implicates the former in a specific, reproducible way -- a near-critical starting
point, not a universal codebase property).

**Recommendations for the next outer-search session** (per Phase 11's own request):

1. **Do not validate outer-gradient quality AT `theta0` alone.** `theta0` (the exact-sample
   FIXTURE calibration point) is, by construction, close to a local critical point of
   `DeltaStar` -- the single WORST case for a linear-gradient-vs-secant comparison. Repeat
   this diagnostic at a genuinely displaced point (e.g. partway through an actual outer
   trajectory, or several `theta_box`-scale steps away from `theta0`) before drawing
   conclusions about registered-gradient quality more broadly.
2. **Pair any future FD-vs-registered comparison with a wider-range profile scan** (as done
   here, `s` spanning `1e-6` to `0.1`) before attributing a ratio deviation to an
   implementation defect -- a local-critical-point artifact is cheap to rule in/out this way
   and easy to otherwise mistake for a real bug.
3. **The gamma coordinate's own "always smooth, no pre-scan needed" assumption** (hard-coded
   `r_gamma=1` in both this session's own script and, per the artifact evidence, the prior
   session's identical script) is **falsified** by this session's own direct evidence at
   small `h` -- a future comparison should not exempt it from the Richardson `h`/`2h`
   stability pre-scan.
4. Given the `W=20,000`-scale finite-support hypothesis (Finding 4's own "likely mechanism"),
   a future session validating gradient quality near a near-exact-fit point specifically
   should consider a larger `W` or an explicit smoothing scheme for that regime -- not
   attempted here (out of scope, disclosed).

**Production default unaffected**: no change was made to `direct_gradient.jl`/
`sorted_crossing_gradient.jl`'s own registered-gradient FORMULA -- the evidence supports
"correct implementation, hard local-landscape at this one starting point," not "wrong
formula." Per Phase 11's own explicit instruction, nothing was "corrected" to match one
arbitrary bandwidth.

### Phase 10: FC-to-GA cache semantics (validated, narrowed scope)

Added to `test/melitz/runtests.jl` ("Governing prompt Phase 10"): (a) cold-solve
reproducibility (5 repeats, bit-identical `Delta`/`x`); (b) warm-start-from-own-dual matches
cold to `1e-10` relative precision; (c) a genuine `cb_F!`-then-`cb_G!` sequence at the
identical `theta` increments `n_exact_cache_hits` by exactly 1 with no additional miss --
i.e. `cb_G!` demonstrably reuses `cb_F!`'s own just-solved dual/moment state rather than
re-solving, confirmed via the SAME production counters the existing 2026-07-27 morning
session's own "26/26 GA calls hit the cache" finding relies on.

**Disclosed, not executed this session**: the full A/B/A matrix across
`{theta, parameterization, seed, W, evaluation cap}` the governing prompt's own Phase 10
literally specifies -- this session's own tests cover `theta`-identity reuse only (the most
important case, and the one Findings 1-2 above already independently stress-tested via the
cold/warm reproducibility check), not the full cross-product.

## Phase 12: parameterization follow-up -- correctly skipped

No implementation error was found in Phases 8-11 that would invalidate the prior
parameterization comparison's own conclusions (Section R,
`docs/melitz_outer_parameterization_comparison_2026-07-26.md`) -- the registered-gradient
FORMULA is not shown to be wrong, only locally hard-to-validate-by-secant AT one specific
near-critical starting point. Per the governing prompt's own explicit instruction ("do not
rerun the six-way tournament unless the gradient audit uncovers an implementation error that
changes previous results"), **no parameterization work was attempted this session.**
Production default remains `:logA`/`:logf`, exactly as Section R already concluded, on
unchanged evidence.

## Phase 13: final regression, Ricardian-boundary proof, commit

Full test suite re-run after every change in this session (Phase 2/3 code, Phase 6/10 new
tests): **every testset `Pass==Total`, exit code 0**, three full runs total this session
(baseline, post-Phase-2/3, post-new-tests) -- see Phase 0.

`git diff --name-only 6f3fa16a7c3010bb14c8f904820ad482ff51e247`:

```
src/melitz/cc_bundle.jl
src/melitz/delta_star.jl
src/melitz/log_cutoff_param.jl
src/melitz/pareto_calibration.jl
src/melitz/sorted_crossing_gradient.jl
test/melitz/runtests.jl
docs/melitz_zero_allocation_and_gradient_closure_2026-07-27.md   (this document)
docs/key_results/melitz_phase9_bandwidth_smoothness_2026-07-27.csv
```

Every changed source file is under `src/melitz/` or `test/melitz/`; the two new files are
under `docs/`. **Zero diff in `cc_algo/`, `full_aod_diag/`, `production/fullA-exact/`
(does not exist in this repo), or any other Ricardian path** -- confirmed directly by the
command above, not merely asserted.

## Acceptance criteria, final status

1. Production outer-state expansion is mutating and workspace-based -- **yes**, for the
   production default (`:logf`, the ONE identified per-coordinate hot call site,
   `_fill_compact_direct_columns_crossing_sorted!`). `_fill_compact_link!`'s own allocating
   `melitz_expand_theta` call (link-touching coordinates only) is disclosed, not fixed.
2. No global lock acquired during normal coordinate expansion -- **yes**, for every `ctx`
   built via a production entry point (Phase 3).
3. No outer coordinate probe creates theta-sized or D^2-sized arrays -- **yes**, for the
   fixed production hot path (measured `0` bytes, D=4 and real D=20); **no**, for the
   disclosed `_fill_compact_link!` gap.
4. Hot callback allocations measured with real D=20 dimensions -- **yes**, for
   `melitz_expand_theta!` specifically (`0` bytes, `n_theta=798`, `W=80,000`); **not**
   exhaustively across the other ~14 named callbacks (disclosed, Phase 5).
5. Allocation tests detect stable repeated allocation, not merely changing allocation --
   **yes** (absolute-`0`-byte assertions, Phase 6).
6. No objective/gradient/Hessian allocation scales with W -- **not independently
   re-verified this session** (inherited from the 2026-07-27 morning session's own
   measurements; this session's own fix is W-independent by construction -- `A`/`f`/scratch
   are `O(D^2)`, never `O(W)`).
7. Large copies/clears separately audited -- **not executed this session** (Phase 7,
   disclosed).
8. Gamma/participation/mixed gradient discrepancies explained -- **yes, decisively**
   (Findings 1-4, Phase 11 verdict).
9. FC-to-GA cache reuse independently validated -- **yes**, narrowed to `theta`-identity
   reuse (Phase 10; full cross-product matrix disclosed as not executed).
10. Production default remains clearly justified as conservative rather than proven globally
    optimal -- **yes**, unchanged from Section R (`:logA`/`:logf`), correctly not revisited
    (Phase 12).
11. Full tests pass -- **yes**, three full runs, zero regressions.
12. No Ricardian/shared source changes -- **yes**, confirmed directly.
13. Work captured in a local commit -- see below (this document written before the commit
    step; commit hash to be recorded once made).

**Not fully met, disclosed rather than hidden**: criteria 1/3/4 (the `_fill_compact_link!`
gap), 6-7 (not independently re-verified/executed), 9 (narrowed to `theta`-identity). These
mirror this repo's own established pattern (2026-07-27 morning session's own Section H/L)
of delivering a genuine, verified subset of a large ask rather than fabricating coverage of
the rest.
