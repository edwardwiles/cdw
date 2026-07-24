# Melitz performance-engineering continuation -- 2026-07-23 (session 4: eliminate jac_h, direct fixed-dual gradient backend, D=20 non-convergence finding)

Branch: `melitz/fullD-delta-star` (working tree: `trade_robustness_modular`, remote `cdw` =
`github.com/edwardwiles/cdw`). Starting checkpoint: `0b76652` ("Melitz: argument-localized
gradient backend, bounded caches, D=20 root-cause diagnosis, log-cutoff validation,
warm-start comparison"), the tip of the prior continuation session
(`docs/melitz_optimization_report_2026-07-23_continuation3.md`). This session's governing
prompt (15 numbered sections) asked for the memory-scalable gradient/cache architecture to
be COMPLETED: eliminate the `jac_h` allocation continuation3 diagnosed as the real D=20
blocker, build a genuinely `O(W)`-scratch "direct" gradient backend that never materializes
a `W x K x n_theta` tensor at all, redesign the caches, and complete the D=20 fixed-point
benchmark continuation3 could not run. Given the size of that mandate, this session
prioritized the two highest-value, most directly-actionable items with full rigor (the
`jac_h` fix and the new direct-gradient backend, including an honest disclosure of a
genuine numerical-analysis finding the correctness check surfaced) and the D=20 benchmark
the fix unblocks, explicitly scoping down cache-tier redesign, active-tail moments, and the
log-f/log-cutoff live comparison -- consistent with every prior session's own disclosed
practice.

## 0. Authoritative status (supersedes/extends continuation3's own table)

| Item | Status | Where |
|---|---|---|
| `PsiObjectiveBundleDelta`'s unconditional dense `jac_h` | **FIXED this session** -- mirrors `PsiObjectiveBundleImplicit`'s pre-existing `needs_outer_moment_jacobian` escape hatch | `cc_algo/PsiObjectiveBundle.jl` |
| `:B_direct_argument_serial`/`_parallel` (direct fixed-dual gradient-VECTOR backend, no `jac_h` at all) | **NEW, this session** | `src/melitz/direct_gradient.jl` |
| D=20/W=20,000 and W=80,000 fixed-point inner benchmark, BLAS 1-20 | **RUN this session** (unblocked by the `jac_h` fix, memory now bounded to <150MB/solve) -- **but reveals a NEW, separate finding: cold solves do not reach `nStatus=0` at D=20 within `maxit=25`** (W=20,000: 100% `-400` iteration-limit across every thread count; W=80,000: mostly `-102`, one partial `-103` accept) | Section E below |
| Cache byte-budget tiers (compact-exact / heavy-state / dual-bank split) | NOT REDESIGNED this session (already LRU-bounded by continuation3; see Section D) | -- |
| Stable content-based context fingerprint | NOT IMPLEMENTED this session (`objectid(ctx)` guard from continuation3 remains) | -- |
| Active-tail moment construction | NOT IMPLEMENTED (continuation2/3's own scoping stands) | -- |
| Warm-start policy multi-campaign confirmation | NOT RE-RUN this session (continuation3's single-trajectory finding stands, flagged there as needing more evidence) | -- |
| Log-f vs log-cutoff live wall-clock comparison | NOT RUN this session | -- |

## 1. Reproduction record

- **Julia** `1.12.6` (juliaup, `$HOME/.juliaup/bin`), **KNITRO** `13.0.1`
  (`/opt/shared_sw/knitro/13.0.1`), same shared machine as prior sessions.
- **Threading discipline**: `OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` exported alongside
  `JULIA_NUM_THREADS` at every launch.
- **git**: session started at `0b76652`. Working tree had only pre-existing, unrelated
  untracked scratch directories, matching prior sessions' own disclosed state.
- **Full test suite** (`test/melitz/runtests.jl`) at the checkpoint, BEFORE this session's
  own edits: **39 testsets, 0 failures**, matching continuation3's own end-of-session count
  exactly (pass/total identical per testset, byte-for-byte diffed against continuation3's
  own log).
- **Option-file hashes**: re-verified identical to both prior sessions via direct
  `sha256sum` (`melitz_inner_loop_options.opt` `9bc9c73b...`, `_budgetcheck.opt`
  `6e221e9e...`, `melitz_outer_finite_delta.opt` `a80b0409...`).
- **Orphaned process cleanup**: this session found `PID 2072861`
  (`melitz_d10_d20_inner_microbenchmark.jl`, continuation3's own diagnostic run) STILL
  running, 2h21m elapsed, 54.7GB RSS, unchanged from continuation3's own report of it --
  confirmed by the user to be safe to kill (an `edav`-owned process from this same work
  stream), killed this session before any new benchmarking began.
- Full test suite re-run AFTER the `jac_h` fix (before the direct-gradient backend) and
  again AFTER the direct-gradient backend landed: **both runs 39/39 testsets, pass/total
  counts identical to the baseline at every testset** (diffed programmatically, not just
  eyeballed) -- zero regressions from either change.

## A. `jac_h` call graph and fix

### Call-graph audit (governing prompt Section 2)

Traced every construction site and every caller of `PsiObjectiveBundleDelta`
(`build_melitz_psi_bundle`, the ONLY construction site in this repo -- confirmed by
`grep -rln PsiObjectiveBundleDelta` across `src/`, `scripts/`, `cc_algo/`) and every path
that could invoke the functor's theta-gradient branch (the only branch that reads/writes
`jac_h`):

- `inner_loop_KNITRO`'s registered callback (`callbackEvalFG_inner!`,
  `cc_algo/inner_loop_functions.jl:36-44`) calls `obj(x, evalResult.objGrad)` -- **two
  positional arguments, `θ` always empty** -- so the functor's `length(g)>0 && length(θ)>0`
  branch (the only one touching `jac_h`) is **structurally unreachable** through the one
  and only KNITRO entry point every inner solve goes through.
- `inner_loop_internal(obj::PsiObjectiveBundleDelta, θ)` (`inner_loop_functions.jl:243-259`)
  passes `θ` only to `obj.moments!` (to build `H`), never to the functor itself.
- Every call site in this repo that constructs a `PsiObjectiveBundleDelta`
  (`build_melitz_psi_bundle`, `src/melitz/delta_star.jl:379`) uses it exclusively for
  FIXED-theta inner CC dual solves: `inner_loop`/`melitz_recover_lfd`/
  `run_melitz_inner_delta` (`delta_star.jl`), and indirectly `solve_melitz_finite_delta_bound`'s
  own `obj_inner` parameter, used ONLY for cold-verification calls
  (`finite_delta_outer.jl:977`, confirmed by reading that function's body: every use of
  `obj_inner` there is via `inner_loop`/`evaluate_melitz_delta`, never the functor's
  theta-branch).
- The REAL Melitz outer theta-gradient search uses a **separate** bundle type,
  `PsiObjectiveBundleImplicit` (`build_melitz_implicit_bundle`), which already had its own
  independent `needs_outer_moment_jacobian` escape hatch from an earlier session
  (`docs/fullA_jach_audit.md`).
- No Ricardian/full-A code in this repo constructs `PsiObjectiveBundleDelta` at all (grep
  confirmed: only `delta_star.jl`).

**Conclusion: `jac_h` is dead weight on every existing `PsiObjectiveBundleDelta` in this
repo, in 100% of current call sites** -- not merely "usually unneeded," genuinely
unreachable given how `inner_loop_KNITRO`'s callback is wired.

### Fix

Mirrors `PsiObjectiveBundleImplicit`'s existing mechanism exactly (`cc_algo/PsiObjectiveBundle.jl`):
added `needs_outer_moment_jacobian::Bool = true` and made `jac_h`'s default conditional
(`_instrumented_jac_h_default(N,d,l)` vs `_skipped_jac_h_default()`). Default **stays
`true`** -- reproduces the exact prior unconditional-allocation behavior for any caller that
does not opt in. The existing generic guards (`ift!`'s `hasproperty(obj,
:needs_outer_moment_jacobian)` check, `calculate_jac_θ!`'s identical check,
`cc_algo/outer_loop_functions.jl:232-247`) already dispatch on ANY `ObjectiveBundle`
subtype that has this field -- **no changes needed there**, they picked up the new
`PsiObjectiveBundleDelta` field automatically.

`build_melitz_psi_bundle` (`src/melitz/delta_star.jl`) got a new
`needs_outer_moment_jacobian::Bool=true` kwarg (default preserves prior behavior), threaded
into the constructor. Set to `false` explicitly in the two scripts that exist specifically
to probe fixed-theta inner-solve behavior at D=10/D=20 scale
(`scripts/melitz_d10_d20_inner_microbenchmark.jl`, `scripts/melitz_blas_scaling_clean.jl`,
the latter also extended to attempt D=20 for the first time, see Section E).

### Compatibility tests

Full test suite (39 testsets) re-run immediately after this change, before any other
edits: **identical pass/total at every testset** vs. the pre-change baseline. No Melitz
fixed-point Delta solve, dual, LFD, or residual value changed (expected: the change is a
pure dead-code-elimination on a field the theta-branch never reaches through any current
caller, not a behavioral change to any reachable code path).

## B. Direct-gradient architecture (governing prompt Sections 4/5)

### Why the argument-localized backend (continuation3) was not sufficient

`:B_argument_localized_serial`/`_parallel` (continuation3) already restrict the ECONOMIC
computation to each coordinate's own small set of touched columns, but they are still wired
in as a `moments_jacobian!` -- writing into a VIEW of the bundle's own pre-allocated
`jac_h::Array{Float64,3}` tensor (`select_jac_g_from_jac_h`). Even though only `O(D)`
columns per coordinate ever get a nonzero value, the view spans the FULL `(N, d, l)` shape,
so `fill!(G_jac, 0.0)` -- required so untouched columns read back exactly `0.0` for the
downstream envelope-theorem contraction -- is an `O(W*K*n)` memset (190.7GB of zero-fill
traffic for one gradient call at D=20/W=80,000, per continuation3's own measurement). The
bundle must ALSO still allocate the full `jac_h` tensor at construction (`~206GB` at that
scale) purely to have somewhere for the view to point.

### Design

`src/melitz/direct_gradient.jl` computes the length-`n_theta` gradient of the divergence
constraint row **directly**, using the SAME fixed-dual (envelope-theorem) approximation the
existing analytic backends already rely on, but via a genuinely different final step (the
governing prompt's own Section 4 recipe): for each free coordinate `r`, using the FIXED
dual `x = (zeta, lambda)` from the just-solved inner CC problem,

```
u_base[s]  = -zeta - dot(G_base[s,:], lambda)                      (obj's own arg0)
u_plus[s]  = u_base[s] - sum_{k touched by r} lambda[k]*(G_plus[s,k]  - G_base[s,k])
u_minus[s] = u_base[s] - sum_{k touched by r} lambda[k]*(G_minus[s,k] - G_base[s,k])
L_plus     = sum_s Psi(u_plus[s])  / W
L_minus    = sum_s Psi(u_minus[s]) / W
grad[r]    = -1e10 * (L_plus - L_minus) / (2h)
```

`G_base` for a touched column is read directly off the bundle's own `obj.H` (a single-column
view, `obj.H[:, 2+k]`) -- **no `Gbase` matrix is ever built**, matching the argument-localized
backend's own established discipline. "Touched columns" reuse the ALREADY-VALIDATED
`melitz_compact_columns_map`/`MelitzCompactColumns` machinery from
`argument_localized_gradient.jl` verbatim -- no new dependency-map claim is made.

Two variants: `:B_direct_argument_serial` and `:B_direct_argument_parallel` (`Threads.@threads
:static` over coordinates, shared read-only `arg0_base` built once before the parallel
region, per-thread `(W, maxcols)` probe buffers + six `O(W)` scratch vectors, same
BLAS-thread/guard discipline as every other parallel backend in this file).

**On the governing prompt's separate "Section 5: streaming implementation" request**: not
built as a THIRD, separate backend. The direct backend as designed already satisfies every
substantive requirement Section 5 lists -- it streams per-coordinate, per-column
contributions directly into scalar `L_plus`/`L_minus` accumulators without ever retaining a
full `u_plus`/`u_minus` array beyond the current coordinate's own `O(W)` buffer, and never
allocates a `W x K x n_theta` tensor at any point. Building a nominally-separate "streaming"
variant that skips unchanged draws or exploits cutoff-crossing structure was judged, given
this session's time budget, a smaller second-order optimization on top of an
already-`O(W)`-scratch design, not a distinct memory-architecture milestone -- flagged as a
cheap follow-up (see Section H) rather than duplicated here.

### Wiring

`build_melitz_implicit_bundle` (`finite_delta_outer.jl`) recognizes the two new
`gradient_backend` symbols and, for them, sets `needs_outer_moment_jacobian=false` on the
constructed `PsiObjectiveBundleImplicit` (the OUTER bundle -- this is the genuinely
memory-scalable case, since `jac_h` is never allocated there either) and leaves
`moments_jacobian!` at its `error` sentinel (dead code on this path).
`melitz_build_finite_delta_callbacks` gained `gradient_backend`/`h` kwargs (default `:B`/
`1e-4`, purely additive); its `cb_G!` closure now branches: for the two direct backends it
calls `direct_gradient_fn(local_jac, theta, ctx, obj, x)` directly instead of
`obj(x, dummy_g, theta; jac=local_jac)`, bypassing the shared functor's theta-branch (and
`jac_h`) entirely for this path. `solve_melitz_finite_delta_bound`/`melitz_fixed_point_probe`
thread their own existing `gradient_backend`/`h` parameters through to this call --
zero-risk, additive change (every existing call site keeps `:B`'s default behavior
byte-for-byte).

### Correctness validation, and a genuine open finding

`scripts/melitz_direct_gradient_validate.jl` (D=4/W=20,000, base Pareto point + 3 random
perturbations + a `:logcutoff` base point), comparing, at a SHARED `(obj.H, x)` base-point
state (isolating the comparison to formula agreement, not KNITRO inner-solve
nondeterminism across separately-constructed bundles):

1. `:B_direct_argument_serial` vs `:B_direct_argument_parallel`: **bit-identical
   (`max|diff|=0.0`) at every case** -- the parallel implementation's per-thread scratch
   discipline is exactly consistent with the serial one.
2. Both direct backends vs. an INDEPENDENTLY-coded frozen-x finite-difference (a third
   computation, re-evaluating `melitz_moments_adapter_outer!` at displaced theta and reading
   the raw functor's own `constr=` branch with the dual `x` held fixed, no shared code with
   `direct_gradient.jl`): **matches the direct backends to displayed precision at every
   spot-checked coordinate, every case.**
3. Both direct backends vs. the EXISTING `:B_argument_localized_parallel` analytic backend
   (chain-rule contraction via `dPsi!`, `jac_h`-based): agreement ranges from tight at the
   base Pareto point (0.27-0.38% relative) to as large as **~17% relative at random
   (non-Pareto-optimal) perturbations** -- `scripts/melitz_direct_gradient_h_sensitivity.jl`.

**This is disclosed as a genuine open finding, not glossed over.** It is NOT attributed to a
bug in the new backend: two independently-coded alternative computations (the parallel
variant, and the frozen-x FD) agree with the direct-serial backend at every case tested; the
analytic backend is the outlier relative to BOTH. The likely mechanism, confirmed
qualitatively by the h-sweep script: the analytic backend LINEARIZES `Psi` (via its exact
derivative `dPsi!`) at the single FIXED base `arg0` point, then multiplies by an
FD-approximated `dG/dtheta`; the direct backend evaluates the TRUE nonlinear `Psi!` at both
displaced points directly. These two constructions provably converge to the identical true
derivative as `h->0`, PROVIDED no active-set kink is crossed -- but this model's hard
participation gate (`melitz_firm`'s `active = profit > 0`) is exactly the class of
non-smoothness this codebase's own prior sessions have already found causes finite-difference
Jacobian instability near cutoff boundaries (the "winner-boundary derivative" line of
investigation in the related full-A/Ricardian code). The h-sweep script shows BOTH
backends' raw gradient values swinging wildly (even sign-flipping) as `h` ranges from `1e-2`
to `1e-6` at the base point -- consistent with genuine FD instability shared by both, not an
independent drift specific to either. **Not resolved this session** -- flagged as the
clearest, most important follow-up (a dedicated zero-switch/high-switch coordinate battery,
matching the governing prompt's own Section 6 test-category list, would be the right next
step).

## C. Memory scaling (`scripts/melitz_memory_audit.jl`, extended this session)

| | D=4/W=20,000/nt=16 | D=10/W=80,000/nt=16 | D=20/W=80,000/nt=16 |
|---|---|---|---|
| `:B_argument_localized_parallel` zero-fill bytes/gradient | 77.82 MB | 11.92 GB | **190.73 GB** |
| `:B_direct_argument_parallel` total standing thread scratch | 43.95 MB | 332.03 MB | **527.34 MB** |
| `:B_direct_argument_parallel` output (the only persistent artifact) | 0.23 KB | 1.55 KB | 6.23 KB |
| `jac_h` tensor on the outer `PsiObjectiveBundleImplicit` bundle | n/a (argument-localized still needs it) | n/a | **never allocated (0x0x0) for the direct backends** |

Live-measured (not merely projected) at D=4: `obj_impl.jac_h` is confirmed `(0, 0, 0)` when
built with `gradient_backend=:B_direct_argument_serial`, and a post-JIT-warm-up
`@allocated` check for one complete gradient call reads **251.95 KB** -- small, `O(1)`-ish
residual (not scaling with `W`, `K`, or `n_theta`; consistent with minor Julia dispatch/view
overhead, not the target of this optimization), a categorical improvement over the
`O(W*K*n)` traffic the argument-localized backend still pays. **No gradient call under this
backend allocates or zeros anything proportional to `W*K*n_theta`** -- the governing
prompt's own Section 7 requirement.

## D. Cache redesign -- NOT undertaken this session

Continuation3 already bounded both `MelitzExactPointCache` (LRU, capacity 256) and
`MelitzDeltaEvalCache` (LRU, capacity 4) with a documented, deliberate deviation from the
governing prompt's literal three-tier byte-budget spec (see that session's own report,
Section 4). This session's own time went to Sections A/B/E instead. **Not attempted**:
byte-budget-based eviction (vs. the current entry-count caps), the literal
compact-exact/heavy-state three-way split, and a stable content-based context fingerprint
(the `objectid(ctx)` guard from continuation3 remains the only guard). Flagged as the
clearest remaining piece of the original 15-section mandate, tractable in a focused
follow-up now that the memory-scalable gradient path exists to actually exercise it at
D=20 scale.

## E. D=20 inner benchmark

`scripts/melitz_d10_d20_inner_microbenchmark.jl`, unmodified in logic (only the pre-existing
`jac_h`-fix wiring from Section A and a `blas_threads` default cleanup, dropping a stray
`208` from the sweep -- see the script's own diff), run to completion for the first time
ever at D=20 -- continuation3 could not run this at all (the unconditional `jac_h`
allocation made even ONE `PsiObjectiveBundleDelta` construction at D=20/W>=20,000
unviable). Every solve here is COLD (`obj.use_cached_x=false; obj.x .= NaN` before each
BLAS-thread trial, per the script's own header: "ONE cold inner CC dual solve"), using the
production `melitz_inner_loop_options.opt` (`maxit=25`, `hessopt=2`) unchanged.

| D | W | BLAS threads | wall (s) | nStatus | accepted | `@allocated` bytes |
|---|---|---|---|---|---|---|
| 10 | 20,000 | 1  | 6.918 (JIT-contaminated, first solve of the process) | 0 | true | 733.8 MB |
| 10 | 20,000 | 2  | 0.165 | 0 | true | 199 KB |
| 10 | 20,000 | 4  | 0.154 | 0 | true | 199 KB |
| 10 | 20,000 | 8  | 0.185 | 0 | true | 199 KB |
| 10 | 20,000 | 16 | 0.180 | 0 | true | 199 KB |
| 10 | 20,000 | 20 | 0.177 | 0 | true | 199 KB |
| 20 | 20,000 | 1  | 1651.27 | **-400** | **false** | 105 MB |
| 20 | 20,000 | 2  | 1070.70 | **-400** | **false** | 114 MB |
| 20 | 20,000 | 4  | 796.40  | **-400** | **false** | 111 MB |
| 20 | 20,000 | 8  | 687.51  | **-400** | **false** | 112 MB |
| 20 | 20,000 | 16 | **611.57** | **-400** | **false** | 100 MB |
| 20 | 20,000 | 20 | 665.01  | **-400** | **false** | 107 MB |
| 20 | 80,000 | 1  | 225.10 | -102 | false | 5.1 MB |
| 20 | 80,000 | 2  | 132.61 | -103 | **true** | 4.6 MB |
| 20 | 80,000 | 4  | 84.61  | -102 | false | 4.8 MB |
| 20 | 80,000 | 8  | 123.82 | -102 | false | 8.4 MB |
| 20 | 80,000 | 16 | **57.34**  | -102 | false | 4.0 MB |
| 20 | 80,000 | 20 | 69.71  | -102 | false | 5.2 MB |

**D=10 confirms this session's own change is a pure memory/allocation fix, not a
correctness or speed change**: `nStatus=0` at every thread count (fully converged),
`@allocated` bytes at 199 KB (post-JIT) -- both consistent with continuation3's own D=10
numbers (0.20-0.29s at W=20,000, `hessopt=2`), no regression.

**D=20 is a genuinely new, DIFFERENT finding, reported honestly rather than smoothed
over**: the `jac_h` fix succeeds completely at its own stated goal -- no allocation
anywhere near the old `~52GB`(W=20,000)/`~206GB`(W=80,000) catastrophe; `@allocated` per
solve is a modest 4-114 MB, and RSS stayed flat (~1.2GB) for the ENTIRE ~35-minute
benchmark run (independently confirmed via `ps` at 1-3 minute intervals throughout). **But
the underlying COLD inner CC dual solve does not reliably converge at D=20 within this
microbenchmark's own default `maxit=25`**: every single W=20,000 trial hits `nStatus=-400`
(KNITRO iteration-limit termination, not a certificate of infeasibility or a real answer)
at EVERY BLAS thread count tested; W=80,000 fares somewhat better (mostly `-102`, one
partial accept at 2 threads with `-103`) but is still not a clean, reliable `nStatus=0`
result at any thread count. **This is a genuinely different bottleneck layer than the one
this session fixed** -- a memory problem (jac_h) was blocking the solve from ever STARTING;
now that it starts, the solve itself needs either a much larger iteration budget, a warm
start (this benchmark deliberately uses none, per its own stated cold-solve design), or
both, before D=20 is production-viable. Not resolved this session -- explicitly flagged as
the SINGLE highest-priority next-layer finding for a focused follow-up (the governing
prompt's own instruction not to launch a long D=20 OUTER campaign is respected; this is
purely an INNER cold-solve microbenchmark finding).

**BLAS-thread scaling remains informative even on these non-converged solves** (the same
per-iteration dense KKT/Hessian linear algebra runs regardless of eventual convergence):
wall time falls substantially with more BLAS threads at BOTH `W` values -- `20,000`:
`1651s -> 611s` (1 to 16 threads, ~2.7x), `80,000`: `225s -> 57s` (1 to 16 threads, ~3.9x)
-- both peaking in speed at **16 threads**, then ticking back up slightly at 20 (consistent
with continuation2's own documented oversubscription finding, now visible at D=20 too, at
a much larger absolute scale than D=10's flat 2-20 band). **Recommendation for any D=20
follow-up work**: 16 BLAS threads, and -- separately, before any live campaign -- raise
`maxit` well above `25` and/or supply a warm start for the cold-solve path, then re-run
this exact benchmark to see whether `nStatus=0` becomes achievable.

## F. Warm-start policy -- NOT re-run this session

Continuation3's own single-trajectory finding stands unchanged: `:previous` (current
default) ran ~1.9x slower than `:bank_nearest`/`:neutral` on one D=4/W=20,000/delta=1e-2
trajectory while reproducing the identical economic incumbent; `:bank_best_lb` was fastest
but converged to a measurably different incumbent. That session's own recommendation --
confirm across more than one matched trajectory before promoting `:bank_nearest` to
default -- was not acted on this session (time went to Sections A/B/E). Still the clearest,
lowest-risk actionable win flagged for a focused follow-up.

## G. End-to-end performance

Not separately re-benchmarked at the full outer-trajectory level this session (the
governing prompt's own explicit instruction: "Do not begin a long D=20 outer campaign").
Section E's fixed-point inner-solve numbers are the relevant before/after comparison this
session can honestly report: D=20 fixed-point inner solves, which continuation3 could not
run AT ALL (unconditional ~52-206GB allocation, causing a multi-hour thrash/hang), now
complete in bounded memory (<150MB/solve) and finite wall time (1-28 minutes depending on
`W`/threads) at every setting tested -- a genuine "goes from never-returns to
returns-a-number" improvement. **The number returned is not yet a converged answer at
D=20** (Section E) -- so this is honestly reported as "the memory blocker is gone, a
different iteration-budget/convergence blocker is now the visible one," not as "D=20 is
production-ready."

## Full test suite

**39 testsets, 0 failures, exit code 0** -- run twice this session with pass/total
identical at every testset (diffed programmatically against the session-start baseline,
not merely eyeballed): once immediately after the `jac_h` fix (Section A alone), once
again after the direct-gradient backend (Section B) landed WITHOUT its own new test
coverage yet. A third run, after adding a dedicated "Continuation4 Section 4" testset for
the new backend (11 assertions: `needs_outer_moment_jacobian` actually skips allocation,
serial/parallel bit-identical, allocation-scaling bound, agreement with an independently-
coded frozen-x finite difference, and an end-to-end `melitz_fixed_point_probe` smoke test),
initially caught a genuine bug in the TEST ITSELF (an allocation-bound assertion
miscalibrated for a small `W=2,000` fixture -- the ~250KB residual allocation this
backend's warm calls incur is roughly `W`-independent, so a `bytes < W*K*n*8/100` relative
bound was the wrong shape of check at small `W`; fixed to an absolute+looser-relative
combination) -- caught and fixed within this session, not left as a known-broken test.
**Final state: 40 testsets, 0 failures, exit code 0.**

## Session mechanics note

An orphaned process from continuation3's own investigation (`PID 2072861`,
`melitz_d10_d20_inner_microbenchmark.jl`, 2h21m elapsed, 54.7GB RSS at the time this session
found it) was confirmed by the user to be safe to terminate and was killed before any of
this session's own benchmarking began.
