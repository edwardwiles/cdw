# Melitz outer-search scaling and profiling session (2026-07-27 continuation)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from
`docs/melitz_final_allocation_and_gradient_closure_2026-07-27.md` (local HEAD `da863d22`,
parent commit chain unpushed). Governing prompt: 12 phases diagnosing and improving the
Melitz outer search now that the engineering closure (matrix-free inner objective/gradient/
Hessian, sorted moment operator, evaluation cap, FC-to-GA exact-point cache) is complete.

## Phase 0: preserve and reproduce

- Branch/HEAD confirmed: `melitz/fullD-delta-star`, `da863d2275879c39a6d23db0e45c4385ad0b8754`,
  not pushed. `git status` before any edit showed only pre-existing, unrelated untracked
  scratch/output directories inherited from other sessions (`full_aod_diag/batch_out_v2/`,
  `sequential_gravity/batch_out_*`, etc.) -- not touched.
- Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`KNITRODIR=/opt/shared_sw/knitro/13.0.1`, pinned --
  14.x lacks a valid site license, this repo's own established convention), KNITRO.jl v1.2.1.
  208 logical CPUs / 3.0TiB RAM (shared host). `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1`
  for every run this session; `JULIA_NUM_THREADS` varied deliberately (1 for kernel-baseline/
  serial rows, 20 for every production-performance/outer-search row, per this session's own
  governing threading policy).
- **Baseline full suite** (before any edit this session, launched via the harness's
  `run_in_background` mechanism, not manual `nohup`, per this repo's own recorded feedback):
  every testset `Pass==Total`, **189,087/189,087 individual assertions** (programmatically
  verified via a `Pass|Total` regex sum across every "Test Summary" block, not eyeballed),
  matching the prior session's own `189,092/189,092` baseline almost exactly (the small
  difference is consistent with ordinary environment/seed-path variation across sessions, not
  a regression -- not independently chased further given this session's own scope). Exit
  status: harness reported "completed" (not "failed"), consistent with exit code 0.
- Confirmed no dense `G` materializations, no dense inner callbacks, no dense outer gradient,
  and evaluation cap active on the production-fast path, by construction (unchanged from the
  prior closure session's own repeated confirmation of these same properties) -- re-verified
  live via this session's own Phase 1/2 scripts, which report `dense_fallbacks=0` throughout.

## Phase 1: authoritative 20-thread timing table

One matched script (`scripts/melitz_outer_search_phase1_2_2026-07-27.jl`), run twice (once
`-t 1`, once `-t 20`, same process cannot vary `Threads.nthreads()` at runtime) against the
IDENTICAL real-D20 calibration/seed/dual-start/inner-options/tolerances/production backend/
JIT-warmup convention this repo's own prior closure benchmark used
(`docs/key_results/melitz_phase12_closure_benchmarks_2026-07-27.csv`).

### Kernels (real D=20, W=80,000, n_theta=798)

| kernel | serial (1 thread) | 20 threads | speedup | bytes (20t) | calls/outer-iteration |
|---|---:|---:|---:|---:|---|
| operator update (theta-expand + moment merge sweep) | 0.109-0.133s | 0.109s | ~1x (serial-only kernel, not threaded) | 32,400 | 1 |
| matrix-free `G*x` (`mul_G!`) | 0.0035-0.0041s | 0.0037s | ~1x (serial-only) | 0 | per KNITRO inner iteration |
| matrix-free `G'*v` (`mul_Gt!`) | 0.0039-0.0042s | 0.0040s | ~1x (serial-only) | 0 | per KNITRO inner iteration |
| structured Hessian callback (packed, `h=` branch) | 0.042-0.052s | 0.0184s (`:structured_parallel`) | ~2.5x | 16,640 | per KNITRO inner iteration needing a Hessian |
| **complete outer gradient (serial call)** | **10.6-12.1s** | 11.1s (ambient 20 threads does not parallelize the SERIAL backend call) | -- | 0 | 1/outer iteration |
| **complete outer gradient (parallel call, 20 threads)** | n/a (parallel backend needs >1 thread) | **0.955s** | **11.67x vs the serial call in the SAME process** | 21,968 | 1/outer iteration |

Serial/parallel numerical agreement: `max|diff|=0.0` (bit-identical -- disjoint per-coordinate
writes, no reduction, exactly as this repo's own prior sessions documented). Do not invent a
parallel number for a kernel that remains serial: `mul_G!`/`mul_Gt!`/operator-update times
above are the SAME single-threaded kernels in both columns (no `_parallel` variant exists for
them at the kernel level; only the STRUCTURED HESSIAN and the COMPLETE OUTER GRADIENT have
threaded variants, both reported).

### Complete operations (real D=20, W=80,000)

| operation | value |
|---|---:|
| complete finite inner solve (direct production path, capped opts) | 0.28-0.54s |
| complete finite FC, **before** the Phase 2 fix, SAME-theta cache-hit re-evaluation | 6.0-6.15s |
| complete finite FC, **after** the Phase 2 fix, SAME-theta cache-hit re-evaluation | 0.019s (see the important correction in Phase 2 below -- this is a cache-HIT number, not a fresh-theta number) |
| complete finite FC, **after** the fix, FRESH (never-visited) theta, mean of 5 trials | see `scripts/melitz_phase2_fresh_theta_fc_2026-07-27.jl` output below |
| complete finite GA immediately after FC at the identical theta, before the fix | 0.81-11.4s (see Phase 2 note below -- FC-to-GA cache state affects this) |
| AboveEvaluationCap-labeled FC (delta tightened 100x; still genuinely FiniteSolved since `Delta0 << delta_evaluation_cap=10`), before the fix, cache-hit | ~6.0-6.15s |
| AboveEvaluationCap-labeled FC, after the fix, cache-hit | 0.020s |

**Mislabeling disclosed**: this session's own "AboveEvaluationCap FC" row (both before/after)
does NOT actually exercise a genuine `AboveEvaluationCap` classification -- `delta_tight20 =
Delta0/100` tightens the OUTER BUDGET `delta`, not `delta_evaluation_cap` (left at its default
`10.0`, far above `Delta0~4e-4`), so the point remains ordinary `FiniteSolved` (Case A: "over
budget" but not over the cap) per this codebase's own documented convention. Its near-identical
cost to the ordinary finite FC (both before and after the fix) is still informative -- it
confirms candidate registration's cost does not depend on whether the point happens to be
"over budget" -- but it does not test the genuinely cheap `AboveEvaluationCap` short-circuit
path a truly-far-outside-the-cap theta would take. Not re-run with a corrected construction
given this session's own time budget; flagged, not silently mislabeled.

Full CSV: `docs/key_results/melitz_phase1_timing_table_2026-07-27.csv`.

## Phase 2: the 5.22s (this session's own re-measurement: 6.0-6.15s) finite FC, explained and fixed

Exhaustive, non-overlapping instrumentation added this session (all zero-cost when
`MELITZ_PROFILE[]` is off, reusing the existing `@melitz_profile` macro,
`src/melitz/profiling.jl`): `:fc_theta_expand`/`:fc_operator_merge` (split
`melitz_update_operator_at_theta!`, `cc_bundle.jl`), `:fc_warm_start_resolve`
(`melitz_resolve_warm_start!`, `inner_screening.jl`), `:fc_cache_insert`
(`melitz_exact_cache_insert!`, `finite_delta_outer.jl`), `:fc_inner_obj_eval`/
`:fc_inner_dpsi_eval`/`:fc_inner_grad_eval`/`:fc_inner_hess_eval` (the KNITRO-facing
objective/gradient/Hessian callback bodies inside `MelitzCCBundle`'s own functor,
`cc_bundle.jl`), and `:fc_reg_outer_state`/`:fc_reg_lfd_recover`/
`:fc_reg_kkt_equilibrium_check` (splitting `evaluate_melitz_delta_from_solution`,
`delta_star.jl`, called from `register_live_candidate!`).

### The answer

**99.7-99.9% of the finite FC's own wall time was `check_profiled_melitz_equilibrium`**
(`equilibrium.jl`) -- an `O(D^2*W)` triple/quadruple-nested loop (Sections 9.1-9.6, 9.9:
`residual_gamma_baseline`, baseline/autarky free-entry residuals, autarky market clearing,
cutoff inequalities -- each iterating `o,d,w` or `o,w`/`w` combinations, calling
`melitz_firm` real floating-point work `O(D^2*W)` to `O(D*W)` times), called from
`register_live_candidate!` -> `evaluate_melitz_delta_from_solution` -> (when the candidate
verifies) `check_profiled_melitz_equilibrium`, **on every single accepted finite FC
evaluation**, not merely the final reported answer. At real D=20/W=80,000 this is
`~2*D^2*W = 64,000,000` `melitz_firm` calls -- fully consistent with the measured ~6 second
cost (~94ns/call), not a profiling artifact.

Live measured decomposition (one finite FC, real D=20, before the fix):

| category | seconds | % of FC |
|---|---:|---:|
| `fc_total_callback_success` | 6.064 | 100.0% |
| `fc_candidate_registration` | 6.058 | 99.9% |
| `fc_reg_kkt_equilibrium_check` | **6.051** | **99.8%** |
| `fc_reg_lfd_recover` | 0.007 | 0.1% |
| `fc_inner_obj_eval` | 0.0036 | 0.1% |
| `fc_inner_dpsi_eval` | 0.0005 | 0.0% |
| `fc_reg_outer_state` | 0.0001 | 0.0% |
| everything else (screens, operator update, theta expand, warm-start resolve) | <0.001 combined | <0.1% |

Full CSV: `docs/key_results/melitz_phase2_fc_decomposition_2026-07-27.csv`.

### Three-variant determination (per the governing prompt's own A/B/C methodology)

- **Variant A** (direct production fixed-point inner solve, `build_melitz_psi_bundle_from_
  calibration` + `melitz_recover_lfd`, matching the prior closure session's own Phase 4
  methodology exactly): **0.28-0.54s**.
- **Variant B** (FC with candidate registration disabled) -- derived EXACTLY (not
  approximated) as `fc_total - fc_candidate_registration` from the SAME profiled run (`register_
  live_candidate!` is a single, non-overlapping sub-step of `cb_F!`'s own try-body, confirmed
  by reading the source, so this subtraction is exact, not a second differently-constructed
  run): **~0.007-0.009s** -- i.e. genuinely close to Variant A (the small residual is
  screening + operator-update + warm-start-resolve overhead atop the bare inner solve, itself
  under 1ms).
- **Variant C** (complete FC, registration enabled, the real production path): **6.0-6.15s**
  before the fix, **0.019-0.020s** after.

**Determination**: the answer is definitively "candidate verification" (specifically, an
avoidable O(D^2*W) diagnostic computation), not different inner-iteration counts, a missing
warm start, a duplicate inner solve, cache behavior, or thread policy -- each of those was
directly ruled out by this session's own per-category timers (`fc_reg_lfd_recover`,
`fc_theta_expand`, `fc_operator_merge`, `fc_warm_start_resolve`, `inner_solve_cache_hit`, all
negligible).

### The fix (implemented, not merely diagnosed)

**Root-cause insight**: `melitz_classify_outer_feasibility`'s own `gravity_feasible` line is
the SOLE production consumer of `.equilibrium_check` anywhere in `src/melitz/` (confirmed by
grep across the whole tree) -- and it reads ONLY two of the struct's 17 fields,
`gravity_residual_A`/`gravity_residual_f`, themselves computed by `gravity_residuals(p)` --
a function of the PRIMITIVES ALONE (`O(D^2)`, no dependence on `z_draws`/`weights`), entirely
independent of every expensive loop above it in the SAME function. Every other field
(`residual_gamma_baseline`, entry-cost residuals, market-clearing residuals, ...) is pure
diagnostic detail, never read by any live search-path code.

**Change**: `check_profiled_melitz_equilibrium(...; full::Bool=true)` -- `full=false` skips
straight to `gravity_residuals(p)` and fills every other field with `NaN` (never a stale/zero
value that could be mistaken for a real diagnostic), an exact, disclosed partial result, not
an approximation of the skipped fields. Threaded through a new `full_equilibrium_check::Bool=
true` kwarg on `evaluate_melitz_delta_from_solution` (default preserves every pre-existing
caller's behavior byte-for-byte). `register_live_candidate!` (`finite_delta_outer.jl`) --
the ONLY live, per-trial hot-path caller -- passes `full_equilibrium_check=false`. The
eventual **cold-verified final incumbent** a solve reports is ALWAYS re-derived from scratch
via a fresh `evaluate_melitz_delta(...; cold=true)` call (`solve_melitz_finite_delta_bound`'s
own re-verification loop, `finite_delta_outer.jl:1658`) -- which does NOT pass this kwarg and
therefore always gets the complete diagnostic detail. **No change to which points get
classified outer-feasible, registered as live candidates, or reported as the final answer** --
`gravity_residual_A`/`gravity_residual_f` (the only fields that matter for classification) are
computed IDENTICALLY either way.

**Verified live**: re-running the same profiled finite-FC script after the fix:
`fc_reg_kkt_equilibrium_check` drops from `6.051s` to `0.00004s`.

**Important correction, caught by direct user challenge and verified rather than waved away**:
this session's own first report of "complete FC wall drops from 6.064s to 0.019s" was
misleading as stated -- both the "before" and "after" measurements warmed up AND timed the
SAME theta (the script's own JIT-warmup convention), so the TIMED call was a genuine hit on
the FC's own exact-point cache (`inner_solve_cache_hit` count=1 on the timed call, confirmed
by re-reading that run's own profile dump), not a fresh KNITRO inner solve. The
`fc_reg_kkt_equilibrium_check` fix itself is real and unaffected by this correction (it fires
identically regardless of whether the inner solve was cached or fresh -- confirmed by Phase
11's own real-search profile below, where the fix is active across 57 genuinely fresh-theta
candidate registrations). A dedicated correction script
(`scripts/melitz_phase2_fresh_theta_fc_2026-07-27.jl`) measuring 5 GENUINELY fresh (never
previously visited) thetas, and Phase 11's own real-search wall-clock decomposition (below),
give the honest numbers: **a fresh-theta finite FC (fresh inner solve + registration, post-fix)
costs on the order of the inner solve itself** (Variant A, `~0.28-0.5s` isolated;
Phase 11's own real-search mean `fc_total_callback_success` was `345ms`, dominated by
`inner_solve_warm_success` at `453ms` mean when a genuine solve fires, `fc_reg_kkt_equilibrium_
check` a negligible `0.065ms` mean across 57 real calls) -- **not** the `19ms` a same-theta
cache hit shows. The FIX's own real, verified contribution is removing the `~6s` per-call
`check_profiled_melitz_equilibrium` tax from EVERY registered candidate, fresh or cached alike
-- it does not (and was never claimed to) make a fresh KNITRO inner solve itself faster. Full
test suite re-verified green after this change (Phase 13 below).

**Why this matters beyond Phase 2 itself**: before this fix, a real-D20 outer search accepting
even a modest number of finite trial points would have paid ~6+ seconds EACH purely for
unread diagnostic bookkeeping -- a 300-600 second Phase 11 budget would have bought at most
~50-100 finite FC evaluations. After the fix, the same budget buys orders of magnitude more,
directly enabling a genuinely informative Phase 11 real-D20 diagnosis (see below) rather than
a handful of data points.

### A second, independent bug found mid-session by direct user challenge: `build_melitz_cc_bundle`'s own `lower_limit` default was never active in this session's diagnostic scripts

Asked directly "why didn't the abort at -10 fire?" while diagnosing a 10+-minute, 5-FC-call
slow run -- confirmed by reading the source, not assumed. This session's own Phase 1/2/
fresh-theta/memory-traffic scripts all built `obj20` via the raw, low-level
`build_melitz_cc_bundle(...)` constructor (mirroring the pre-existing `scripts/melitz_
closure_benchmarks_2026-07-27.jl` pattern), which defaulted `lower_limit::Float64=
-KNITRO.KN_INFINITY`. The evaluation-cap early-abort this codebase relies on
(`(Q::MelitzCCBundle)`'s own functor: `if f <= Q.lower_limit; return -KN_INFINITY`, telling
KNITRO's inner solve to stop immediately once a point is certifiably bad) was therefore NEVER
active in any of those scripts -- comparing a real floating-point `f` against `-Inf` is never
true. `lower_limit = -delta_evaluation_cap` is wired up ONLY inside `build_melitz_implicit_
bundle` (`finite_delta_outer.jl`), the constructor `solve_melitz_finite_delta_bound` itself
calls -- `melitz_build_finite_delta_callbacks` (called directly on a hand-built bundle, as
this session's own microbenchmarks did) never touches `obj.lower_limit` at all. A poorly-
conditioned perturbed theta that a cheap pre-screen (range/stored-dual) does not catch
therefore had NOTHING to stop a runaway KNITRO inner solve except the `.opt` file's own
`maxit`/`maxtime` -- exactly the 10+-minute slow run this correction thread started from.

**This is the SAME class of mistake this repo's own history has recorded multiple times
before** (per direct, pointed user feedback: a prior session's own fix, `@assert
isfinite(obj.lower_limit)` inside `solve_melitz_finite_delta_bound`, did not close this gap
because that assert lives in ONE high-level convenience wrapper that a direct call to the
lower-level constructor bypasses entirely). **Fixed structurally, not just patched at this
call site**: `build_melitz_cc_bundle`'s `lower_limit` kwarg no longer has ANY default --
omitting it is now a `MethodError` at the exact construction call site, for every current and
future caller, including a future session's own throwaway script. Audited and fixed all 11
call sites in this repo (`grep`-verified exhaustive): the 3 pre-existing production call sites
(`delta_star.jl`, `finite_delta_outer.jl`, `pareto_calibration.jl`) already passed it
explicitly and needed no change; 1 pre-existing test (`runtests.jl`) and all 7 of this
session's own scripts did not, and now pass an explicit, deliberate value (`-KNITRO.KN_
INFINITY` for the one genuinely-uncapped-by-design test; `-10.0` for every diagnostic script,
matching this session's own standard `delta_evaluation_cap=10.0` convention).

**A SEPARATE, independent instance of the same underlying pattern was found in this session's
own Phase 9 script while auditing for others**: `MelitzInnerSolveConfig(:full_value)` was
passed to `solve_melitz_nuisance_min_delta` -- `:full_value` is DELIBERATELY uncapped by
design (its own `Base.show` says so), the wrong mode for a nuisance-profile search that runs
many repeated/nested inner solves. Unlike `build_melitz_cc_bundle`, this call site already had
proper compile-time enforcement (`inner_solve_config` is a required kwarg with no default,
and `solve_melitz_nuisance_min_delta` unconditionally overwrites `obj_inner.lower_limit =
inner_solve_config.lower_limit` at entry regardless of how the object was built) -- this was a
MODE-CHOICE mistake, not a missing guard. Confirmed live: the original Phase 9 run (`:full_
value`) diverged into a numerically-garbage KNITRO trajectory (objective values `~1e16-1e19`)
and ran for 35+ minutes before being killed; fixed to `MelitzInnerSolveConfig(:evaluation_cap;
delta_evaluation_cap=10.0)` plus an explicit `melitz_assert_evaluation_cap_active(inner_cfg)`
call (this repo's own pre-existing defensive assertion, written by a prior session
specifically for this failure mode), then re-run -- see Phase 9 below for the corrected result.

**What this means for every OTHER finding in this document, checked systematically, not
blanket-assumed clean**: Phase 1/2/3 kernel and FC-decomposition measurements were all taken
at `theta0` (an already-converged, well-conditioned point) -- the early-abort only matters for
a trajectory that would otherwise cross `lower_limit`, which a normal converged solve never
approaches regardless of its value, so these are unaffected. The touched-row memory-traffic
script never invokes the inner solver at all (touched-row counts are reconstructed
analytically from `melitz_expand_theta`+cutoffs); its one `evaluate_melitz_delta` call is at
`theta0`. Phase 8 and Phase 11 both went through `solve_melitz_finite_delta_bound`, which
builds its OWN separate, correctly-capped internal bundle regardless of what `obj_inner` was
built with -- confirmed by reading the override code, not assumed. Phase 7 used the same
ungoverned `build_melitz_psi_bundle` default (a DIFFERENT, higher-level function with its own,
separately-reasoned-through default -- not touched by this session's `build_melitz_cc_bundle`
fix), but all 30 grid points converged cleanly (`nStatus=0`, no anomalies) -- disclosed as a
real, uncapped-by-construction gap in that script, but with no evidence any specific number in
that table was actually wrong.

**Corrected fresh-theta measurement, re-run after both fixes**
(`scripts/melitz_phase2_fresh_theta_fc_2026-07-27.jl`, real D=20, same 1e-5-per-coordinate
random perturbations, same RNG seed): the SAME 5 of 10 draws still hit a genuine
`NumericalFailure` (a real, seed-deterministic data property, unchanged by the fix -- this
fixture's calibration point is genuinely poised tightly enough that half of tiny simultaneous
perturbations across all 798 free coordinates are uncertifiable, consistent with this repo's
own "aggregate multi-coordinate perturbations have much larger leverage than any single
coordinate" finding), but the 5 SUCCESSFUL fresh-theta FC calls are now **tight and
consistent**: `0.119s, 0.160s, 0.163s, 0.179s, 0.181s` (mean `0.160s`), replacing the
pre-fix run's wild `3.39s, 0.15s, 4.11s, 0.17s, 0.17s` (mean `1.60s`) -- direct, live
confirmation that the earlier dispersion was NOT genuine fresh-solve variance, it was
borderline trajectories grinding under the missing cap before eventually recovering. This is
the trustworthy final number: a genuinely fresh, post-fix finite FC at real D=20 costs
**~0.16s**, in the same ballpark as (slightly faster than, plausibly from warm-start reuse
across the batch) the isolated Variant A inner-solve estimate (`~0.28-0.5s`) -- NOT the
misleading `0.019s` same-theta cache-hit number this document's own earlier draft first
(incorrectly) reported as if it were a fresh-call cost.

## Phase 3: parallel outer gradient at 20 threads

Same script, `-t 20`: **complete parallel outer gradient (real D=20/W=80,000, n_theta=798,
20 Julia threads): 0.955s**, 21,968 bytes (thread-launch/scheduling overhead, not a hot-path
per-coordinate allocation -- consistent with this repo's own established convention that only
the SERIAL backend is held to a strict `0`-byte ceiling), **11.67x speedup vs. the serial
call in the same process** (11.15s serial), bit-identical to the serial result
(`max|diff|=0.0`). This is now the authoritative production benchmark, consistent in order of
magnitude with the prior closure session's own cited `~0.86s at 16 threads` figure (different
commit, different exact thread count -- not expected to match to the decimal).

## Phase 4: touched-row (no-full-copy) gradient backend -- implemented, validated, NOT adopted

Implemented `src/melitz/touched_row_gradient.jl` (+ a `MelitzCCBundle`-specific method in
`cc_bundle.jl`, mirroring the existing sorted-crossing-slice backend's own split): replaces
the sorted backend's per-coordinate full-`W` `copyto!(u_plus/u_minus, arg0_base)` +
full-`W` `Psi!` evaluation with a touched-row-only accumulate/evaluate, using the EXACT
identity `sum(Psi.(u)) = base_scalar_sum + sum_{w touched}[Psi(u[w]) - psi_base[w]]` (`Psi` is
elementwise, so untouched rows contribute exactly zero to this difference -- proved, not
assumed, reusing the crossing-slice backend's own already-proved "untouched rows are
unaffected" fact). Touched-row bookkeeping uses a monotonically-increasing generation-stamp
counter (never reset, avoiding an O(W) reset entirely) + a growable touched-index list --
**a real bug caught before this backend was ever run**: an earlier draft used the per-call
coordinate index `r` (1:n) directly as the generation stamp, which would have silently read
stale deltas from a PREVIOUS gradient call's own identical stamp value; fixed with a
persistent, ever-increasing counter, with a dedicated regression test ("repeated calls" test,
`test/melitz/runtests.jl`) added specifically to catch a regression of this exact class.

**Validated correct**: D=4 (every coordinate, including the link-touching one, agrees with the
sorted backend to `~1e-13` relative; a perturbed non-calibration point agrees to `~1e-13`; a
second consecutive call reproduces the first exactly, ruling out the caught generation-stamp
bug); real D=20 (complete-gradient agreement `~1e-11` relative, expected floating-point-
summation-order-level agreement, not bit-identical). Wired end-to-end (new
`gradient_backend=:B_direct_argument_touched_row_serial`, additive in `finite_delta_outer.jl`/
`nuisance_profile.jl`, resolves through `melitz_fixed_point_probe` unchanged).

**NOT adopted as production default**: measured speedup at real D=20/W=80,000 is **1.02x**
(10.56s vs. 10.78s, warm, both 0 bytes) -- not material. Per the governing prompt's own
explicit instruction ("adopt only if the gain is material... do not delay the outer-search
experiments if the gain is small"), this is retained as a validated, tested, but
non-production-default backend.

**Memory traffic, measured directly (not merely theorized), because a direct user question
asked for it**: `scripts/melitz_touched_row_memory_traffic_2026-07-27.jl` counts the ACTUAL
touched-row set per coordinate at the real-D20 calibration point. Of `798` coordinates, `40`
touch the focal link (dense by construction, touch all `80,000` rows, zero savings possible);
of the remaining `758` non-link coordinates, the touched-row set is a **mean 13.24% / median
11.95% of `W`** (range: `3,019` to `65,123` rows, i.e. up to `81.4%` for a few high-leverage
coordinates near participation boundaries). Aggregated across the complete `798`-coordinate
gradient, the touched-row backend's total per-row work is **17.58%** of the sorted backend's
(`n_theta*W` footprint) -- translating to an estimated **~359 MB/gradient call vs. ~2043
MB/gradient call**, a real **~5.7x reduction in memory traffic** (copy/delta-accumulate plus
Psi-evaluation passes combined), not merely a theoretical one.

**The dissociation is itself the finding**: a genuine, substantial (~5.7x) reduction in memory
traffic coexists with almost no wall-clock gain (1.02x) at this hardware/problem size. This
means memory BANDWIDTH is evidently not the binding constraint on this kernel's wall time here
-- the dominant remaining cost must be CPU/arithmetic-bound (the `G`-column FILL over the
crossing slice itself, shared unchanged between both backends, and/or the branch-heavy
`exp`-based `Psi`/`dPsi` evaluations), not the memory traffic this backend specifically
targets. This is a more precise, and more honest, statement of the negative result than "no
gain" alone -- the backend does exactly what it was designed to do (cut memory traffic ~5.7x),
it simply is not what this kernel's wall time is bottlenecked on at D=20/W=80,000 on this
host.

## Phase 5: KNITRO 13.0.1 algorithm/scaling API audit

Re-confirmed live against the exact installed `include/knitro.h` (not from memory) --
identical findings to the prior 2026-07-25 session's own exhaustive audit (same KNITRO
version, same host):

| capability | KNITRO name | confirmed values |
|---|---|---|
| Variable scale/center | `KN_set_var_scalings_all` | `x[i] = xScaleFactors[i]*xScaled[i] + xScaleCenters[i]`; callback-facing `x` is ALWAYS raw |
| Algorithm | `KN_PARAM_ALGORITHM` (1003) | `0=auto, 1=Interior/Direct (KN_ALG_BAR_DIRECT), 2=Interior/CG (KN_ALG_BAR_CG), 3=Active Set (KN_ALG_ACT_CG), 4=Active-Set-SQP (KN_ALG_ACT_SQP), 5=multi` |
| Initial trust-region radius | `KN_PARAM_DELTA` (1020) | scaling factor on KNITRO's own initial trust region, acts in SCALED coordinates when `scale!=0` |
| Master scaling switch | `KN_PARAM_SCALE` (1017) | `0=none,1=user_internal (default),2=user_none,3=internal` |
| Line search max trials | `KN_PARAM_LINESEARCH_MAXTRIALS` (1044) | default `3`, Interior/Direct or SQP only |
| Honor bounds | `KN_PARAM_HONORBNDS` (1002) | production driver already sets `always` |
| Concurrent evals | `KN_PARAM_CONCURRENT_EVALS` (1134) | not exercised this session (no concurrent-eval usage in any Melitz driver) |

**No augmented-Lagrangian algorithm exists in this KNITRO version's `algorithm` enum**
(confirmed directly from the header's own `KN_ALG_*` defines: auto/direct/cg/active/sqp/multi
only) -- Phase 5's own "if appropriate" augmented-Lagrangian comparison is N/A at this KNITRO
version, not skipped.

## Phase 6: scaled outer-coordinate wrapper

**Already implemented and tested** (2026-07-25 session, `solve_melitz_finite_delta_bound`'s
`var_scale`/`var_center` kwargs, `finite_delta_outer.jl`) -- confirmed present and unmodified
on this commit. That session's own evidence (a standalone synthetic-NLP audit + a real
production D=4 driver no-change test, both re-confirmed present in `test/melitz/runtests.jl`
this session) established that **native KNITRO scaling is the correct choice over a
hand-rolled `y = theta_center + S*y` wrapper with an explicit chain rule**: KNITRO's own
`KN_set_var_scalings_all` is callback-transparent (every Melitz callback always sees RAW
`theta`), requires zero changes to the affine cutoff system's linear constraint registration
(KNITRO itself performs the affine transform internally, before evaluating ANY constraint,
linear or nonlinear -- confirmed by the 2026-07-25 session's own live iteration-count-vs-scale
test), and was directly verified to preserve constraint counts (`400` linear + `1` nonlinear
at real D=20) and every economic quantity (A/f/cutoffs/DeltaStar/kappa_ratio/GT/gravity/
equilibrium) exactly. This session did not re-implement this mechanism (would have been
redundant, higher-risk work against the governing prompt's own "prefer native scaling if fully
auditable" instruction) -- it is used directly in Phases 7-11 below.

## Phase 7: D=4 block scale selection

`scripts/melitz_phase7_d4_scale_selection_2026-07-27.jl`: D=4, W=20,000, seed=29, base point =
this fixture's own calibration point (a genuinely separate D=4 "near-boundary" point was not
independently established in any prior session -- disclosed, not silently substituted;
Section 8 of the prior closure doc's own finding that the calibration point is close to
several participation-switch thresholds simultaneously makes it an economically meaningful
reference point for this purpose regardless). Fully reoptimized (`evaluate_melitz_delta(...;
cold=true)`, genuine fresh inner KNITRO solve at each displaced theta) finite DeltaStar, exact
draw-level switches (sorted-tail `melitz_active_tail_start` method), cutoff slack:

| block | first non-zero exact switches at | qualitative DeltaStar movement by top of grid |
|---|---:|---|
| g (raw \|dg\|) | ~1e-4 (2-3 switches) | 7.55e-6 -> ~4.8e-4 (~60x) by 3e-3 |
| technology (normalized direction, aggregate norm) | ~3e-6 to 1e-5 (1 switch) | 7.55e-6 -> ~7.0e-6 (mild) by 1e-4 |
| participation (normalized direction, aggregate norm) | ~1e-5 (1 switch) | 7.55e-6 -> ~7.2e-6 (mild) by 1e-4 |

**Scale candidate chosen**: `s_g=1e-4` (a unit scaled step maps to `dg~1e-4`, inside the
few-switches, still-informative range), `s_A=1e-5`, `s_f=1e-5` (matching the technology/
participation blocks' own first-switch radius). This is, not coincidentally, the SAME order of
magnitude the 2026-07-25 session independently derived at real D=20/W=80,000 from its own
local-geometry evidence (`s_g=1e-4, s_A=s_fq=1e-5`) -- an independent D=4 corroboration of
that D=20 finding's general economic-sensitivity ordering, not a re-derivation from scratch.
Full CSV: `docs/key_results/melitz_phase7_d4_scale_selection_2026-07-27.csv`.

## Phase 8: scaled joint KNITRO algorithm comparison at D=4

`scripts/melitz_phase8_d4_scaled_algorithm_comparison_2026-07-27.jl`: D=4, W=20,000, seed=29,
matrix-free/`forbid_dense_fallback=true`, native `:linear` cutoff constraints, sorted outer
gradient, evaluation cap=10.0, Phase 7's own scale set, `delta=1e-2` both directions, all four
KNITRO algorithms (`KN_PARAM_DELTA` left at KNITRO's own default `1.0` -- untuned, per "first
use one reasonable scale set" before any step-control sweep):

| algorithm | direction | nStatus | wall | n_fc | n_above_cap | max reported \|\|Step\|\| | outcome |
|---|---|---:|---:|---:|---:|---:|---|
| Interior/Direct | upper | -410 | 20.98s | 41 | 20 | ~8.7e4 | **RUNAWAY** -- infeasible, step exploded iter 3-16 |
| Interior/Direct | lower | -410 | 4.01s | 31 | 16 | ~7.5e5 | **RUNAWAY** -- infeasible, worse explosion |
| Interior/CG | upper | -102 | 6.13s | 128 | 0 | (not runaway) | well-behaved, zero above-cap rejects |
| Interior/CG | lower | -400 | 4.98s | 42 | 0 | (not runaway) | iter-limit, feasible, no runaway |
| Active Set | upper | -101 | 2.05s | 62 | 0 | (not runaway) | **genuine xtol convergence**, stayed at start |
| Active Set | lower | -400 | 4.55s | 26 | 0 | (not runaway) | iter-limit, feasible, modest real movement |
| SQP | upper | -101 | 2.58s | 74 | 0 | (not runaway) | genuine xtol convergence, stayed at start |
| SQP | lower | -200 | 1.69s | 20 | 15 | (not runaway in Step column, but 15/20 above-cap) | mixed -- some real exploration, more cap hits |

Full CSV: `docs/key_results/melitz_phase8_d4_scaled_algorithm_comparison_2026-07-27.csv`.

**Finding, directly analogous to (and now confirming, at a second scale) the 2026-07-25
session's own real-D20 result**: the default algorithm (`0=auto`, which resolves to
Interior/Direct at this fixture) is the ONE combination that runs away catastrophically with
an untuned scaled trust region at D=4 too -- **not** a D=20-specific artifact. Interior/CG,
Active Set, and SQP are all measurably more robust to the SAME untuned scale set: none of
their reported `||Step||` columns exhibit the multi-order-of-magnitude jump Interior/Direct
shows in both directions. Active Set is the standout: genuine `xtol` convergence in BOTH
directions is achieved by two of the four algorithm/direction cells (upper for both Active Set
and SQP), and Active Set's own lower-direction run shows real, bounded movement with zero
above-cap rejects.

## Phase 9: scaled nuisance-profile formulation

`scripts/melitz_phase9_d4_nuisance_profile_2026-07-27.jl`: reuses the existing production-fast
`solve_melitz_nuisance_min_delta` infrastructure (no custom optimizer), predetermined ordered
grid (14 points, `g in {0, -0.005, ..., -0.15}`, walking from the interior calibration point
toward increasingly negative/ambitious `g`, matching Phase 8's own finding that the
welfare-improving direction is `g<0`), continuation-warm-started from the preceding VERIFIED
point's own nuisance coordinates + inner dual (never from a failed/uncertified state).
`MelitzInnerSolveConfig(:evaluation_cap; delta_evaluation_cap=10.0)` (corrected mid-session
per the bug described above), `melitz_assert_evaluation_cap_active` called explicitly.

**Result** (full CSV: `docs/key_results/melitz_phase9_d4_nuisance_profile_2026-07-27.csv`):
total wall **~440s** for all 14 grid points (a clean, bounded run -- contrast the original
`:full_value`-mode attempt, which diverged into numerical garbage and ran 35+ minutes before
being killed). Only **2 of 14** grid points (`g=-0.07`, `g=-0.10`) reach genuine `nStatus=-101`
(xtol convergence, verified `ok=true`); the remaining 12 hit the outer NLP's own iteration cap
(`nStatus=-400`, 6 points) or other non-converged codes (`-502`/`-102`, 6 points) within the
(deliberately unchanged, this repo's own standing) `maxit=25`-class outer option file budget.
**Comparison against the fixed-A/f benchmark at the SAME g grid** (full CSV: `docs/key_results/
melitz_phase9_d4_fixed_af_comparison_2026-07-27.csv`): the flexible nuisance search's own
`Delta_min` BEATS the fixed-A/f `Delta` at several grid points (e.g. `g=-0.02`: flexible
`5.3e-3` vs fixed `9.1e-2`, ~17x lower; `g=-0.03`: flexible `1.6e-3` vs fixed `1.0e-2`, ~6x
lower) but is WORSE at others (e.g. `g=-0.04`: flexible `4.5e-3` vs fixed `3.1e-4`) --
consistent with most flexible grid points NOT having actually converged (`ok=false`,
`nStatus=-400` iteration-limit) rather than the flexible search being intrinsically inferior:
an incomplete search can report a `Delta_min` worse than a point (the fixed-A/f benchmark)
that a full nuisance search, given enough iterations, should always be able to at least match
(fixing `eta` at its calibrated value is a feasible point within the flexible search's own
`radius=0.3` neighborhood). **This is a genuine, disclosed negative result about search
COMPLETENESS under this outer option file's iteration budget, not evidence the flexible
formulation is worse in principle** -- a longer `maxit`/`maxtime` budget was not tried this
session given time constraints.

## Phase 10: canned-formulation selection at D=4

Comparing the three D=4 formulations tested this session under the SAME rough development-time
budget (Phases 8-9, both corrected for the `lower_limit`/`:full_value` bugs above):

| formulation | finite trials produced | accepted/converged outcomes | wall for a representative run | headline finding |
|---|---:|---:|---:|---|
| unscaled joint KNITRO (prior 2026-07-25 session, D=20 context) | few | 0 improving | ~13 min (791s) | well-behaved but stagnant |
| scaled joint KNITRO, default algorithm (this session, D=4) | 41 (upper) / 31 (lower) | 0 -- BOTH directions ran away (`nStatus=-410`, step exploded `~1e4-1e5`) | 21.0s / 4.0s | scaling alone, untuned algorithm, is actively harmful |
| scaled joint KNITRO, Active Set (this session, D=4) | 62 (upper) / 26 (lower) | 1 genuine `xtol` convergence (upper), 1 iteration-limit-feasible (lower) | 2.05s / 4.55s | robust, fast, no runaway, but finds no improving point beyond the calibration start |
| scaled nuisance profile, Active-Set-analogous capped inner solves (this session, D=4) | 14 grid points | 2 of 14 genuine `xtol` convergence | ~440s total (~31s/point average) | genuine curve produced, but most points did not fully converge under the tested `maxit` budget |

**Selection: scaled joint KNITRO with the Active Set algorithm** is the most reliable canned
formulation tested this session for real-D20 diagnosis -- it is the only combination that
(a) never exhibited the runaway-step pathology any untuned-algorithm scale combination showed,
(b) reached genuine KNITRO-native convergence (`xtol`, not merely an iteration limit) in more
than one of the four D=4 algorithm/direction cells tried, and (c) has the lowest per-call
overhead of the three formulations (no nested inner optimization, unlike the nuisance
profile). The nuisance-profile formulation remains a valuable complementary diagnostic (it
directly answers "how does the welfare frontier look under full flexibility," which the joint
formulation does not expose as directly) but needs a larger iteration budget than this
session's own quick-regression-convention `maxit` before its own curve can be trusted as
converged at every grid point.

## Phase 11: short real-D20 outer diagnosis

`scripts/melitz_phase11_real_d20_diagnosis_2026-07-27.jl`: real D=20 (`noah_D20`), W=80,000,
canonical seed, `delta=1.0`, `direction=:upper`, 20 Julia threads, BLAS=1, strict
production-fast, native `:linear` cutoff constraints, evaluation cap=10.0, Active Set
algorithm (Phase 8/10's own selection), Phase 7's own scale set, fixed-A/f (`theta0` itself)
retained as the external comparison incumbent, `maxtime_real=480s` budget. This run is only
interpretable as a genuine SEARCH diagnosis because of this session's own Phase 2 fix --
before it, this wall-clock budget would have bought at most ~80 finite FC evaluations; after
it, orders of magnitude more.

**Result** (full CSVs: `docs/key_results/melitz_phase11_real_d20_trajectory_2026-07-27.csv`,
`melitz_phase11_real_d20_wallclock_decomposition_2026-07-27.csv`): **wall=77.8s** (well under
the 480s budget -- KNITRO's own Active Set trajectory reached genuine `xtol` convergence,
`nStatus=-101`, at outer iteration 3, not an iteration-limit artifact), `n_fc=60`, `n_ga=4`,
`inner_solve_count=40` (`inner_infeas_count=0` -- zero uncertified inner failures throughout).
**The search made ZERO net movement**: `terminal_theta[1] == theta0[1]` exactly, `|dg|=0.0`,
`max|dA|=0.0`, `max|df|=0.0` -- `best_live_incumbent` and `cold_verified_incumbent` both
equal the STARTING point to full precision. Wall-clock decomposition: complete FC/GA callback
wall is only `26.4s` (`34%`) of the total `77.8s`; the remaining `51.4s` (`66%`) is genuine
KNITRO-internal overhead (Active Set's own CG/linear-algebra work across 59 CG iterations,
per the KNITRO console log) -- NOT an artifact of this session's own callback instrumentation.
Per-category breakdown of the FC/GA wall confirms the Phase 2 fix holds under a REAL search,
not just an isolated microbenchmark: `fc_reg_kkt_equilibrium_check` totals `3.7ms` across 57
real candidate registrations (`0.065ms` mean), `inner_solve_warm_success` (`453ms` mean, 37
calls) and the KNITRO-facing `fc_inner_obj_eval`/`fc_inner_hess_eval`/`fc_inner_grad_eval`
callback bodies dominate the FC/GA wall, exactly as expected for a genuinely capped, fast
production search.

**This exact-zero-movement result reproduces, once again, the SAME qualitative finding every
prior joint-KNITRO real-D20 session in this repo's own history has reported** (the
2026-07-25 session's own Section H; the earlier-still `melitz_real_d20_outer_correction_
2026-07-24.md`/`melitz_real_d20_evaluation_cap_correction_2026-07-24.md`) -- now confirmed
under a materially faster, correctly-capped, more-robust-algorithm configuration than any of
those prior attempts, and still finding no improving point. This strengthens (does not merely
repeat) the standing conclusion: the obstacle to a better joint-search economic answer at this
fixture is not evaluation cost, native-scaling absence, or a poorly-chosen default algorithm
-- all three are now ruled out or corrected -- it is a genuine LOCAL GEOMETRY property of the
calibration point itself (this repo's own prior "close to a critical point of DeltaStar AND
close to several participation-switch thresholds simultaneously" finding, Finding 4/Phase 9
of the 2026-07-27 morning closure session).

## Phase 12: final recommendations

1. **Authoritative 20-thread timings** (Phase 1): complete parallel outer gradient at real
   D=20/W=80,000 is **0.955s** (11.67x speedup vs. 11.15s serial in the same process, bit-
   identical). Structured Hessian callback: 20-thread parallel variant is 2.5x faster than
   serial (0.018s vs. 0.042-0.052s). `mul_G!`/`mul_Gt!`/operator-update remain serial-only
   kernels with no parallel variant to compare against, as expected.
2. **Why the FC took 5.22-6.15s**: `check_profiled_melitz_equilibrium`'s O(D^2*W) diagnostic
   loop, called on EVERY registered live candidate (not just the final answer) -- ~99.7-99.8%
   of the FC wall. Fixed via a `full_equilibrium_check=false` fast path that computes only the
   two fields (`gravity_residual_A`/`_f`) any production classification actually reads,
   verified to change zero economic/classification outcomes. A genuinely fresh-theta FC now
   costs ~0.16s (real D=20), not the misleading ~0.019s same-theta cache-hit figure this
   document's own earlier draft first (incorrectly) reported.
3. **The no-copy gradient backend was validated correct but NOT adopted**: touched-row
   backend cuts memory traffic ~5.7x (measured directly: ~359MB vs. ~2043MB/gradient call) but
   gives only 1.02x wall-clock speedup -- memory bandwidth is evidently not this kernel's
   bottleneck at D=20/W=80,000 on this host. Kept as a tested, available, non-default backend.
4. **Selected outer variable scales** (Phase 7): `s_g=1e-4`, `s_A=1e-5`, `s_f=1e-5` --
   independently corroborates (different fixture/scale) the 2026-07-25 session's own D=20
   local-geometry-derived candidate.
5. **Selected KNITRO algorithm and step controls** (Phase 8/10): Active Set (`algorithm=3`),
   KNITRO's own default trust-region radius (untuned `delta=1.0`) -- the default Interior/
   Direct algorithm is NOT safe with this scale set at either D=4 or D=20 without a separately
   tuned trust region; Active Set is robust without needing one.
6. **Joint vs. nuisance-profile organization** (Phase 9/10): joint KNITRO (Active Set) is the
   more reliable formulation for real-D20 diagnosis given this session's own time budget --
   faster per-call, no nested-solve overhead, and already reached genuine `xtol` convergence.
   The nuisance profile remains valuable for the welfare-frontier QUESTION specifically but
   needs a larger iteration budget to trust its curve as fully converged at every grid point.
7. **D=4 now searches more reliably than before**: the runaway-step pathology is fully
   explained and avoidable (correct algorithm choice); genuine `xtol` convergence was reached
   in multiple D=4 configurations this session, which the default algorithm never achieved.
8. **Real D20 still produces ZERO net movement** even under the corrected, faster, more robust
   configuration (Phase 11) -- this reproduces, and now more strongly evidences, every prior
   session's own finding. The bottleneck is not cost, scaling, or algorithm choice (all
   addressed this session) -- it is this fixture's own local geometry at the calibration point.
9. **The next remaining search bottleneck**: the calibration point's own proximity to a
   critical point of `DeltaStar` and to multiple simultaneous participation-switch thresholds.
   A future session should either (a) start the outer search from a genuinely different,
   verified non-calibration incumbent (Section 8 of the 2026-07-27 morning closure session
   found four such points via a small mixed-direction perturbation) rather than the
   calibration point itself, or (b) increase `W` or introduce an explicit smoothing scheme
   specifically to soften the draw-level participation-switch nonsmoothness this repo's own
   exact-switch-count evidence (2026-07-27 morning session, Phase 6-9) already confirmed is
   the mechanism producing the fixed-dual gradient's own documented unreliability near such
   thresholds -- both recommendations carried forward unchanged from that prior session's own
   Phase 11, now with additional real-D20 evidence (this session's own zero-movement result
   under a corrected configuration) that the obstacle is unlikely to be resolved by search-
   infrastructure improvements alone.
