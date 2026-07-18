# jac_h runtime audit: the dense outer-moment Jacobian in the full-A cached/Method-B path

Branch `diag/fullA-d4-exact-jach-audit` (based on `diag/fullA-d4-exact` @ `b5c109d`). Investigates and,
having confirmed it safe, eliminates the unused dense `jac_h::Array{Float64,3}` allocation from the
full-A cached (`outer_loop_cached`) and Method-B production drivers. Every runtime claim below is
backed by a counter/timer from an actual run, not a static-analysis guess — this investigation has
been burned before by assuming behavior instead of measuring it (see `docs/fullA_d4_code_audit.md`'s
own winner-boundary-derivative finding for the canonical example of why).

## 1. Summary (measured, not assumed)

`jac_h` is a `PsiObjectiveBundleExplicit`/`Implicit`/`Delta` field
(`cc_algo/PsiObjectiveBundle.jl`), a dense `N × (d+2) × l` tensor (`N`=draws used for the outer
gradient, `d`=number of moments, `l`=length of the full θ vector) that `calculate_jac_θ!` /
`calculate_jac_θ_autodiff!` (`cc_algo/outer_loop_functions.jl`) fill via a full ForwardDiff Jacobian
pass through `moments!`, for the **legacy** outer-gradient/outer-Jacobian branch
(`length(g)>0 && length(θ)>0` in the bundle's own callable method, plus `ift!`'s implicit-function-
theorem contraction of it).

For the full-A cached/Method-B path as it actually runs today:

1. **Allocation**: happens exactly once per `PsiObjectiveBundleImplicit` construction (the
   `@with_kw` field default). At this investigation's D=4/W=8000 sizing (`N=8000, d=18, l=23`) this
   is **29,440,000 bytes (28.08 MB)**, confirmed via `Base.summarysize` and via the theoretical
   formula `8 * N * (d+2) * l` (they agree exactly, up to a fixed 56-byte `Array` header).
2. **Population**: **never happens** in the live full-A cached/Method-B drivers. Measured directly
   with a call counter (`JAC_H_POPULATE_COUNT`) across `evaluate_fullA` (warm+cold), an 8-probe
   central-FD-style sweep, the full `L_fix`/`lfix_incremental` machinery (base solve + 4 incremental
   perturbations), and a genuine 3-iteration `outer_loop_cached` KNITRO outer solve: **0 calls**.
3. **Read/contraction**: **never happens** either — the `length(θ)>0` branch that reads `jac_h`
   (`JAC_H_THETA_BRANCH_COUNT`) and `ift!`'s contraction of it (`JAC_H_IFT_COUNT`) were also **0**
   across the same exercise.
4. **The counters are not dead code**: forcing one direct call to the bundle's callable with a
   nonempty `θ` (bypassing the normal driver flow) makes all three counters increment to exactly 1,
   confirming the zero counts above are real evidence of the live path's behavior, not a broken
   instrumentation harness.

**Conclusion**: `jac_h` in this path is allocated, then sits as inert dead memory for the object's
entire lifetime. Eliminating the allocation removes **only** a one-time memory/allocation cost — it
does **not** remove any active AD/ForwardDiff computation, because none was ever running for this
object in this path (see §6).

A new opt-in `needs_outer_moment_jacobian::Bool=true` field on `PsiObjectiveBundleImplicit` makes the
allocation skippable (`false` → a `0x0x0` placeholder instead of the full tensor), with a clear
diagnostic error (not a bounds crash or silent wrong answer) if any legacy jac_h-touching code path
is subsequently invoked on such an object. Validated bit-identical (to a stated, justified tolerance)
against the default-`true` behavior across D=4 (calibration/upper/lower/5 random feasible points/a
short real outer-loop trajectory), and separately at D=6 and D=8. The flag is now wired into the two
named full-A production drivers (`run_fullA_D4_production.jl`, `run_fullA_D10_production.jl`);
everything else defaults to the old unconditional-allocation behavior, unchanged.

## 2. Call-graph audit (traced from code, cross-checked by counters in §4)

### 2.1 Bundle type constructed by each driver

| Driver | Bundle type | Has `jac_h`? |
|---|---|---|
| `full_aod_diag/d4_exact/context.jl::d4_exact_setup` (used by `oracle.jl`, `run_d4_optimized_fd.jl`, `candidate_registry.jl`, `test_oracle*.jl`, `test_lfix_incremental.jl`, most of `d4_exact/`) | `CS.PsiObjectiveBundleImplicit` | **Yes** (the field this audit targets) |
| `full_aod_diag/d4_exact/context_scaled.jl::d_exact_setup_scaled` (D/W-parametrized variant of the above) | `CS.PsiObjectiveBundleImplicit` | **Yes** |
| `full_aod_diag/run_fullA_D4_production.jl` — "new" (cached) path | `PsiObjectiveBundleImplicit` | **Yes** |
| `full_aod_diag/run_fullA_D4_production.jl` — "reference" path | `PsiObjectiveBundleImplicitMethodBFullA` | **No field at all** — this struct (`full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl`) simply does not declare `jac_h`; its own callable method overrides the `θ`-nonempty branch entirely with `_methodB_fullA_envelope_scalar` + `Q.gravity_grad(θ)`, so the question is moot for this type. |
| `full_aod_diag/run_fullA_D10_production.jl` (`make_fullA_obj`) | `PsiObjectiveBundleImplicit` | **Yes** |

So `PsiObjectiveBundleImplicit` is the only type in the live full-A cached/Method-B ecosystem that
actually carries a `jac_h` field.

### 2.2 Driver dispatch: `outer_loop` vs `outer_loop_cached`

- `d4_exact/oracle.jl::evaluate_fullA` calls `CS.inner_loop_internal(obj, θ_full)` directly, then
  `obj(inner_x, constr = @view(cbuf[1:ncon]))` — **`g` and `θ` are both left at their default empty
  vectors** (positional args 2/3 of the callable), so this is a constraint-**value**-only call. It
  never reaches the `length(g)>0` branches at all.
- `d4_exact/run_d4_optimized_fd.jl` drives its own raw KNITRO NLP whose gradient callback
  (`cb_G!`) calls `eval_grad_central_fd`, which itself only calls `Delta_of_w` →
  `evaluate_fullA` repeatedly (finite differences in the **reduced pivot-eliminated coordinate
  space**) — again never touching the bundle's own `θ`-nonempty branch.
- `cc_algo/outer_loop_cached.jl::outer_loop_cached` (used by both `run_fullA_D4_production.jl`'s
  "new" path and `run_fullA_D10_production.jl`) calls `obj(inner_x, constr = @view(cbuf[1:ncon]))`
  for constraint **values** (again both `g`,`θ` empty) and gets **gradients** entirely from
  caller-supplied free-only closures (`obj_grad_fn!`, `div_grad_fn!`, `gravity_grad_fn!`) — never
  from the bundle's own `θ`-nonempty branch. `div_grad_fn!` is
  `envelope_scalar_div_ctx` (`full_aod_diag/ad_benchmark/derivative_core.jl`), a **standalone**
  ForwardDiff-over-a-free-only-closure function that builds its own local `H` buffer — it does not
  read or write `obj.jac_h` at all.
- `full_aod_diag/d4_exact/lfix_incremental.jl` (`build_lfix_base_cache`, `lfix_incremental_at`) calls
  `obj.moments!` directly (once, for self-validation) and `CS.inner_loop_internal` — never the
  bundle's callable with nonempty `θ`, never `jac_h`.
- `cc_algo/local_sensitivity.jl::local_sensitivity` **does** read `obj.jac_h`, but it also **calls
  `calculate_jac_θ!(obj, θ)` itself first** (line 5) — so the new `needs_outer_moment_jacobian`
  guard in `calculate_jac_θ!` (added this audit, §5) already protects this caller automatically; no
  separate change was needed there. It is not called from anywhere in the current full-A
  cached/Method-B drivers.
- Three **obsolete/superseded** diagnostic scripts (`full_aod_diag/cond_diag.jl`,
  `full_aod_diag/solve_scaled.jl`, `full_aod_diag/validate_free_only_gradient_fullA.jl`) do read
  `obj.jac_h` directly — but each constructs its **own** `PsiObjectiveBundleImplicit` independently
  (not via `d4_exact_setup`/`d_exact_setup_scaled`), so they are entirely unaffected by anything in
  this audit (default `needs_outer_moment_jacobian=true` unchanged; these files were not touched).
  `validate_free_only_gradient_fullA.jl` in particular is the one place in this repo that
  legitimately *needs* the dense Jacobian (it validates the free-only gradient against it as ground
  truth) — exactly why the new flag defaults to `true` rather than removing the field outright.

### 2.3 Confirms the task's exact framing

> Do not assume all three occur [allocate / populate / read].

Confirmed: (1) occurs (once per construction); (2) and (3) do **not** occur anywhere in the live
full-A cached/Method-B path, only in the legacy `outer_loop`-driven callable branch, which nothing in
this investigation's current drivers reaches (`PsiObjectiveBundleImplicitMethodBFullA`, the one type
still driven through `outer_loop`, has no `jac_h` field to read in the first place).

## 3. Fixed-θ inner KNITRO callbacks: θ=[] confirmed

`cc_algo/inner_loop_functions.jl::callbackEvalFG_inner!`/`callbackEvalH_inner!` call
`obj(x, evalResult.objGrad)` / `obj(x, h=evalResult.hess)` — both 1- or 2-positional-argument calls,
leaving `θ` at its default `Float64[]`. This is architecturally guaranteed (not just observed): the
inner KNITRO problem optimizes over `(ζ,λ)` (or `(η,ζ,λ)` for the Explicit variant) at a FIXED outer
θ, so the inner callback signature has no reason to ever pass a nonempty θ. Cross-checked by the
`JAC_H_THETA_BRANCH_COUNT`/`JAC_H_POPULATE_COUNT` counters staying at 0 through every inner solve
performed during this audit's real-usage exercise (§4.2), including the deliberately-cold ones.

## 4. Instrumentation and runtime audit (measured)

### 4.1 Instrumentation added (`cc_algo/jac_h_instrumentation.jl`, purely additive)

Ref-based counters/timers, mirroring the existing `INNER_SOLVE_COUNT`/`INNER_INFEAS_COUNT` pattern
(`cc_algo/inner_loop_functions.jl`):

| Counter | Meaning |
|---|---|
| `JAC_H_ALLOC_COUNT` / `JAC_H_ALLOC_BYTES` / `JAC_H_ALLOC_TIME` | # times a real (nonempty) `jac_h` was constructed; cumulative bytes/wall-time |
| `JAC_H_SKIPPED_COUNT` | # times construction was skipped (`needs_outer_moment_jacobian=false`) |
| `JAC_H_POPULATE_COUNT` / `JAC_H_POPULATE_TIME` | # calls to `calculate_jac_θ!` (the function that fills `jac_h`); cumulative wall-time |
| `JAC_H_THETA_BRANCH_COUNT` | # times a bundle callable's `length(g)>0 && length(θ)>0` branch (the one that reads/contracts `jac_h`) is entered |
| `JAC_H_IFT_COUNT` | # calls to `ift!` (the other jac_h reader/contractor, for the Implicit/Delta variant) |

`_instrumented_jac_h_default(N,d,l)` wraps the previously-bare `zeros(N, d+2, l)` default expression
with these counters — **identical array contents/type/size**, only additionally counted. `reset_jac_h_counters!()` / `jac_h_counters_snapshot()` support scoped before/after measurement.
Included in `cc_algo/include_cc_algo.jl` right after `ObjectiveBundle.jl` (before
`PsiObjectiveBundle.jl`, so the wrapper functions are in scope for the `@with_kw` field defaults).

### 4.2 Real-usage counter run (D=4, `full_aod_diag/d4_exact/audit_jach.jl` Part 2)

Counters reset, then: `evaluate_fullA` (warm + cold) at the calibration point, 4×2=8 perturbed
`evaluate_fullA` calls (central-FD-style probe), `solve_base_state` + `build_lfix_base_cache` +
4 `lfix_incremental_at` calls at the `upper_maxit40` candidate:

```
alloc_count=0  populate_count=0  theta_branch_count=0  ift_count=0
```

Then, without resetting, a genuine 3-iteration `outer_loop_cached` KNITRO trajectory (Part 5.6,
`csw_outer_default_maxit3.opt`) from the calibration start point: still **0/0/0** for
populate/theta_branch/ift (asserted in-script, not just eyeballed).

### 4.3 Counters are not dead code (Part 3)

One direct forced call, `ctx.obj(inner_x, g_buf, θ_full; jac=jac_buf)` (nonempty `θ`), on the
default (`needs_outer_moment_jacobian=true`) object:

```
populate_count=1  theta_branch_count=1  ift_count=1   (outer_constr_index<=d holds here, so ift! IS reached)
```

Confirms the zero counts in §4.2 are real, not an artifact of a broken counter.

### 4.4 The guard, forced (Part 4)

The identical forced call on a `needs_outer_moment_jacobian=false` object throws immediately, with:

```
ErrorException: calculate_jac_θ!: this ObjectiveBundle was constructed with
needs_outer_moment_jacobian=false -- the dense outer-moment Jacobian (jac_h) was never
allocated, so the legacy ForwardDiff-through-moments!/analytic-Jacobian outer-gradient path
(calculate_jac_θ!, calculate_jac_θ_autodiff!, ift!) cannot be used on this object. ...
```

No out-of-bounds crash, no silent wrong result. The object remains usable afterward for the normal
(`θ`-empty) call pattern — verified by a subsequent successful `evaluate_fullA` call on the same
object.

## 5. The no-jac_h construction mode (implementation, purely additive)

`cc_algo/PsiObjectiveBundle.jl::PsiObjectiveBundleImplicit`:

```julia
needs_outer_moment_jacobian ::Bool      = true
jac_h  ::Array{Float64,3} = needs_outer_moment_jacobian ? _instrumented_jac_h_default(N, d, l) : _skipped_jac_h_default()
```

Default `true` preserves the exact prior unconditional-allocation behavior for every existing caller
that does not pass this kwarg (no breaking change). `PsiObjectiveBundleExplicit`/`Delta` were left
structurally unchanged (still always allocate `jac_h`, just now via the counted
`_instrumented_jac_h_default` wrapper) — they are not part of the full-A cached/Method-B path this
task scoped (`PsiObjectiveBundleExplicit` is unused by any full-A driver; `PsiObjectiveBundleDelta`
likewise), so adding the opt-out flag there was left out of scope rather than speculatively extended.

Guards added at every point a legacy jac_h-touching call could originate:
- `cc_algo/outer_loop_functions.jl::calculate_jac_θ!` — the single generic entry point both the
  autodiff and analytic-Jacobian branches share; errors immediately if
  `needs_outer_moment_jacobian=false`, before either branch would write into (or bounds-check
  against) the empty array.
- `cc_algo/PsiObjectiveBundle.jl`'s `ift!(λ, obj::Union{PsiObjectiveBundleImplicit,
  PsiObjectiveBundleDelta})` — a second, defense-in-depth guard, for any future/direct call to
  `ift!` that bypasses `calculate_jac_θ!`'s normal ordering (unreachable via the current callable
  flow, since `calculate_jac_θ!` is always called first and already errors).

`cc_algo/local_sensitivity.jl` needed no separate change (§2.2): it calls `calculate_jac_θ!` itself
before touching `obj.jac_h`, so it inherits the guard automatically.

`full_aod_diag/d4_exact/context.jl::d4_exact_setup` and
`full_aod_diag/d4_exact/context_scaled.jl::d_exact_setup_scaled` both gained a passthrough
`needs_outer_moment_jacobian::Bool=true` keyword (default unchanged), so validation/microbenchmark
scripts can request either mode without duplicating the setup logic.

## 6. What this change actually saves — memory/allocation only, not AD computation

Because `jac_h` was **never populated** in this path (§2, §4.2), the legacy
`calculate_jac_θ_autodiff!`'s full ForwardDiff Jacobian pass through `moments!` was **never running**
here to begin with — it is not "redundant active computation" being removed (contrast with
`docs/fullA_performance_profile_v2.md`'s Phase 1 finding, where a genuinely-executing redundant
`moments!` call was eliminated). The only thing this change removes is:

- the one-time `zeros(N, d+2, l)` allocation + zero-fill at bundle construction, and
- the corresponding heap memory sitting live (but untouched) for the object's entire lifetime.

No AD graph, no ForwardDiff `Dual` computation, no draws-loop work is avoided, because none of that
was ever happening for this object along this path.

## 7. Validation

### 7.1 Method and tolerance

Two contexts built from the **identical economy/draws** (asserted `ctx.θ0_up == ctx2.θ0_up`,
`ctx.U == ctx2.U`), differing only in `needs_outer_moment_jacobian`:

- **`TIGHT = 0.0`** (bit-identical) for quantities that never pass through a KNITRO solve —
  `moments!`'s `K`/`G` output, and `winner_hash` (pure `argmin`, no solver). Confirmed bit-identical
  in every check.
- **`SOLVER_TOL = 1e-8`** for quantities downstream of `ctx.obj`'s / `ctx2.obj`'s own **independent**
  inner KNITRO dual solve (two separate `KN_new()` problem instances from identical data are not
  contractually bit-reproducible against each other). Justified empirically: an initial 0.0-tolerance
  run found every such mismatch was **≤ 9.33 × 10⁻¹⁴** in absolute value (`lambda`, `m_star`, `zeta`,
  `Delta_dual`, etc.) — floating-point noise from independent BLAS/KNITRO internal solve paths, not a
  jac_h-caused discrepancy (`moments!` itself, zero solver involvement, was exactly bit-identical in
  that same run). `1e-8` is ~10⁶× looser than the observed noise floor, and matches this
  investigation's own established convention (`test_oracle.jl` TEST 4: warm-vs-cold KNITRO solves
  compared at "diff < 1e-8 : PASS", not bit-identical).
- **`TRAJ_TOL = 1e-4`** for the multi-iteration outer-loop trajectory comparison (chains several
  independent inner solves, so the single-solve noise floor can compound across outer iterations).

### 7.2 D=4 battery (`full_aod_diag/d4_exact/audit_jach.jl`, all PASS)

Points: calibration (`CS.pack_free(ctx.θ0_up, ctx.m)` — the direct calibration point, not a
pivot-elimination round-trip of it; see the in-script note on why those two differ and why the pivot
round-trip of the zero log-matrix is itself cold/warm-infeasible at this economy, a pre-existing,
unrelated property of `gravity_elimination.jl`'s pivot machinery, not a jac_h issue), `upper_maxit40`
(current upper incumbent), `lower_stalled` (current lower incumbent) — both loaded from
`candidate_registry.jl`, not hand-transcribed — plus 5 random feasible perturbations of `upper_maxit40`
(radius 0.01, 0 skipped as infeasible this run).

Compared per point: `moments!`'s `K`/`G`; `inner_status`; `K_hard`, `Delta_dual`, `Delta_primal`,
`zeta`, `lambda`, `moment_resid`, `max_abs_moment_kkt_resid`, `gravity_value`, `gravity_raw`,
`m_mean`/`m_min`/`m_max`, `winner_hash`, `primal_dual_gap`; the fixed-dual `BaseDualState`
(`ζstar`,`λstar`,`m_star`) and `fixed_dual_L` value (also cross-checked against `Delta_dual` itself).
Additionally: `L_fix` **incremental gradient** (tier `:incremental_o1`, 4 coordinates) at
`upper_maxit40` and `lower_stalled`; a 4-coordinate **optimized-value central-FD gradient**
(h=0.01) at `upper_maxit40`; and a genuine **3-iteration `outer_loop_cached` trajectory**
(`θ_min_full`, `objective`, `nStatus`) using the same free-only gradient closures
`run_fullA_D4_production.jl` uses (`envelope_scalar_div_ctx`, `gravity_grad_free!`).

**Result: 12/12 checks PASS.** jac_h counters stayed at exactly 0 through the entire 3-iteration
trajectory on both contexts (asserted in-script).

### 7.3 D=6/D=8 microbenchmark (`full_aod_diag/d4_exact/audit_jach_d6d8.jl`, own process)

Run in a **separate Julia process** from the D=4 script — `context.jl`/`context_scaled.jl` are not
designed to be `include()`'d twice within one process (re-including redefines the
`CounterfactualSensitivity` module and breaks type identity for already-constructed objects;
confirmed by an `UndefVarError`/export-ambiguity crash on a first attempt to do this inline).

Contrary to `docs/fullA_block_local_performance.md` §7's finding (a prior session's D=6 attempt found
the calibration-anchored point cold-infeasible), the calibration point (reached via
`CS.pack_free(ctxS.θ0_up, ctxS.m)`, matching §7.2's D=4 fix) was **feasible on the first attempt** at
both D=6 and D=8 this session — plausibly the same `pack_free`-vs-pivot-round-trip distinction found
at D=4 (§7.2), not re-diagnosed further here (out of scope for this audit).

`Delta_dual`/`zeta`/`lambda` agreement (`SOLVER_TOL=1e-8`): **PASS at both D=6 and D=8.**

## 8. Performance numbers

All at `W=Jac_W=8000` (this investigation's standard synthetic-economy scale). D=4 numbers from
`audit_jach.jl`; D=6/8 from `audit_jach_d6d8.jl` (separate process, so absolute wall-times are not
directly comparable to D=4's — each process has its own JIT-compilation history — but the
WITH-vs-WITHOUT deltas *within* a process are).

### 8.1 jac_h size: `8 * N * (d+2) * l` bytes, read from the actual bundle (not assumed)

| D | N (=W) | d (nTotalMoments) | l (l_full) | theoretical bytes | measured `Base.summarysize` |
|---|---|---|---|---|---|
| 4 | 8000 | 18 | 23 | 29,440,000 (28.08 MB) | 29,440,056 (+56B `Array` header) |
| 6 | 8000 | 38 | 45 | 115,200,000 (109.86 MB) | (formula confirmed exactly; header not re-measured) |
| 8 | 8000 | 66 | 75 | 326,400,000 (311.28 MB) | (formula confirmed exactly; header not re-measured) |

### 8.2 Bundle-construction wall time, WITH vs WITHOUT jac_h (median, isolating jac_h's own cost)

| D | WITH jac_h | WITHOUT jac_h | attributed jac_h alloc+zero cost |
|---|---|---|---|
| 4 (Part 1, n=20 reps) | 15.825 ms | 0.919 ms | 14.907 ms |
| 4 (Part 6.1, n=20 reps, re-measured) | 14.35 ms | 0.845 ms | 13.5 ms |
| 6 (n=5 reps) | 236.31 ms | 118.84 ms | 117.47 ms |
| 8 (n=5 reps) | 458.53 ms | 240.32 ms | 218.2 ms |

This is the **entire** measured effect of the change — construction happens once per outer-loop run,
not per iteration.

### 8.3 Warm, per-evaluation cost — JIT/ordering-controlled, alternating order, `@timed`

An initial (non-alternating, ctx-then-ctx2) comparison of a short outer-loop trajectory showed a
large apparent gap (ctx 5.80s vs ctx2 1.36s, 4.2×) — **not trusted**, and directly falsified by a
controlled re-measurement: ctx ran first in that comparison, so most of the gap is plausibly
first-in-process JIT compilation (`outer_loop_cached`, `ForwardDiff.GradientConfig`, KNITRO callback
closures all compile on their first call), not jac_h. Per this investigation's own standing
discipline (memory `feedback-verify-before-causal-claims`: never explain a timing difference from
wall-clock alone without eval-count/compile-time evidence), Part 6 re-measured with pre-warming and
alternating order:

| D | metric | WITH jac_h | WITHOUT jac_h | ratio (WITH/WITHOUT) |
|---|---|---|---|---|
| 4 | warm `evaluate_fullA`, n=30, alternating | 22.17 ms (median), 16,617,424 bytes, 0.0 ms GC | 21.86 ms (median), 16,617,424 bytes, 0.0 ms GC | 1.014 |
| 4 | short outer-loop (maxit=3), n=5, alternating | 0.405 s (median) | 0.371 s (median) | 1.092 |
| 6 | warm `evaluate_fullA`, n=10, alternating | 50.939 ms (median) | 44.453 ms (median) | 1.146 |
| 8 | warm `evaluate_fullA`, n=10, alternating | 117.781 ms (median) | 113.962 ms (median) | 1.034 |

**Bytes allocated per warm `evaluate_fullA` call are identical (16,617,424) between WITH/WITHOUT at
D=4**, and GC time is 0.0 ms for both — direct confirmation that jac_h contributes **zero** ongoing
per-call allocation or GC pressure once constructed (it is genuinely inert). The remaining ~1–15%
ratios above are within normal run-to-run noise for these wall-clock magnitudes (tens of
milliseconds) on a shared, busy multi-tenant host; not large enough, or monotonic enough across D, to
support a "large dead array causes measurable per-call slowdown via memory locality/GC-scan effects"
claim — flagged as a plausible but **unconfirmed** secondary hypothesis, not asserted.

### 8.4 Real production run (not just the synthetic audit harness)

`full_aod_diag/run_fullA_D4_production.jl`, with `needs_outer_moment_jacobian=false` now wired into
its "new" (cached) bundle, run end-to-end (full 25-outer-iteration KNITRO solve, both the new cached
path and the reference `PsiObjectiveBundleImplicitMethodBFullA` path):

```
NEW: status=-400  opt_err=... gamma'_focal=0.8886523793006442  kappa=0.17860287198055402  wall=29.3s
REF: status=-400  gamma'_focal=0.8983813737221186  kappa=0.16356043962883482  wall=36.11s
```

These numbers **match** `docs/fullA_d4_code_audit.md` §8's previously-logged baseline (recorded when
`jac_h` was still unconditionally allocated: "new=0.1786, ref=0.1636") to the digits quoted there —
direct, independent confirmation in a real (not synthetic-harness) production run that eliminating the
allocation changes nothing about the optimization outcome.

## 9. What was and was not done

- **Done**: instrumentation, call-graph audit, runtime counter confirmation, the opt-in no-jac_h
  mode + guards, full D=4 validation battery, D=6/D=8 microbenchmark validation, performance
  measurement (construction cost, warm per-call cost with JIT/ordering controlled, real production
  run), wiring `needs_outer_moment_jacobian=false` into both named production drivers
  (`run_fullA_D4_production.jl` — run end-to-end and cross-checked; `run_fullA_D10_production.jl` —
  edited identically but **not** run end-to-end this session, D=10 solves take substantially longer
  and the identical code path was already exercised at D=4/6/8).
- **Left unchanged (deliberately, scope discipline)**: `PsiObjectiveBundleExplicit`/`Delta` (not used
  by any full-A driver); the shared `d4_exact_setup`/`d_exact_setup_scaled` **default** (kept `true`
  — dozens of other scripts in this directory depend on it, a much larger blast radius than the two
  explicit production-driver call sites this task named); the three obsolete diagnostic scripts that
  legitimately read `obj.jac_h` from their own independently-constructed bundles
  (`cond_diag.jl`, `solve_scaled.jl`, `validate_free_only_gradient_fullA.jl`).
- **Not chased further**: why the D=6 calibration point was feasible this session when a prior
  session found it cold-infeasible (§7.3) — plausibly the same `pack_free`-vs-pivot-round-trip
  distinction found at D=4, out of scope for a jac_h audit.

## 10. Files touched

- `cc_algo/jac_h_instrumentation.jl` (new) — counters/timers, no-jac_h default helpers.
- `cc_algo/include_cc_algo.jl` — include ordering + exports.
- `cc_algo/PsiObjectiveBundle.jl` — `needs_outer_moment_jacobian` field + guards + counters.
- `cc_algo/outer_loop_functions.jl` — `calculate_jac_θ!` guard + populate counters.
- `full_aod_diag/d4_exact/context.jl`, `full_aod_diag/d4_exact/context_scaled.jl` — passthrough kwarg.
- `full_aod_diag/run_fullA_D4_production.jl`, `full_aod_diag/run_fullA_D10_production.jl` —
  `needs_outer_moment_jacobian=false` wired into the cached-path bundle construction.
- `full_aod_diag/d4_exact/audit_jach.jl` (new) — D=4 audit + validation + performance script.
- `full_aod_diag/d4_exact/audit_jach_d6d8.jl` (new) — D=6/8 microbenchmark, own process.
- `docs/fullA_jach_audit.md` (this file), `docs/fullA_performance_profile_v2.md` (pointer section).
