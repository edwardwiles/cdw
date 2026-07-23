# Melitz optimization session report -- 2026-07-23

Branch: `melitz/fullD-delta-star`, starting checkpoint `e1884e2`. This report covers the
session's primary tasks: live-wiring the `:logcutoff` outer parameterization, building
switchable profiling instrumentation, profiling representative points/trajectories,
diagnosing the upper/lower asymmetry, restricted nuisance-coordinate searches, a matched
`:logf` vs `:logcutoff` comparison, a Ricardian-optimization transfer survey, and one
implemented low-risk optimization with before/after numbers.

STATUS: DRAFT -- being populated as live runs complete. Do not cite section C/D/E numbers
until the final version replaces this notice.

## A. Executive conclusion

- **Linear cutoff backend**: `cutoff_constraint_backend = :linear` remains the correct
  default -- already established pre-session (Jacobian nonzeros 510 -> 109, KNITRO
  presolve enabled) and unaffected by this session's changes.
- **Parameterization recommendation**: pending Section E's live numbers (in progress).
- **Nuisance-coordinate search**: pending Section D's live numbers (in progress).
- **Top bottlenecks**: pending Section B/C (in progress); one already-identified and
  FIXED bottleneck is `make_melitz_moments_jacobian_b`'s 60 (`2 x 30` at D=4) full
  `W x (D^2+1)` moment-matrix allocations per outer gradient callback -- addressed in
  Section G below (preallocated-buffer reuse, correctness-preserving).

## B. Baseline profile

(pending live profiling run)

## C. Upper/lower asymmetry

(pending live profiling run)

## D. Search behavior (gamma-only / GA / GF / GQ / full)

(pending live restricted-search run)

## E. Parameterization comparison (:logf vs :logcutoff)

(pending live matched-comparison run)

## F. Ricardian transfer matrix

Full survey (via a dedicated research pass over `cc_algo/` and the mature Ricardian
`full_aod_diag/d4_exact/` implementation -- `production/fullA-exact` as a literal
directory does not exist in this checkout; its content lives under those paths, per
`docs/fullA_REPO_MAP_2026-07-22.md`/`docs/fullA_CURRENT_STATE_2026-07-22.md`):

The Ricardian model's performance story centers on making a **hard-winner** moment
function (`argmin_o price` per draw/destination) cheap to re-derive after a
single-coordinate perturbation. Melitz has no such structure -- it has hard
**participation** cutoffs (`profit_i(theta) > 0`), not an argmin-over-origins
competition -- so every item below is explicitly re-assessed against that distinction
rather than ported blindly.

| # | File/function | Mechanism | Applicability to Melitz | Expected benefit | Correctness risks | Recommendation |
|---|---|---|---|---|---|---|
| 1 | `full_aod_diag/d4_exact/gradient_workspace.jl` (`GradWorkspacePool`) | Persistent per-thread-slot scratch buffers reused across probes/calls, sized by `Threads.maxthreadid()` | **Directly transferable** -- generic FD/AD scratch hygiene, no winner-structure dependence | Root-caused ~4GB/gradient-call allocation; folded into 1.04-1.06x/4.93x-less-memory pooled backend numbers | One workspace must never be shared by two concurrently-running gradients | **Adopt now** -- partially done this session (Section G) |
| 2 | `full_aod_diag/d4_exact/lfix_incremental.jl` + `lfix_base_workspace.jl`/`lfix_factorized.jl`/`lfix_kbplus.jl` (`LFixBaseCache`) | Freeze a base-point state (`winner0`/`runnerup0`/`contrib0`/`q0`) once per gradient call, do O(D) or O(W) incremental per-coordinate updates instead of O(W*D^2) rebuilds | **Transferable with adaptation** -- the "freeze base state, cheap local update" pattern is generic; the cached *content* (winner/runner-up) is Ricardian-specific and must be re-derived around Melitz's participation status instead | Reference (no factorization): 2.40s/589.6MB base-cache build, 38.89s/13,229MB per 400-coord gradient at D=20 | Cache-object aliasing: stashing a reference across >1 gradient call silently corrupts | Adopt with care -- port the pattern, not the cached content |
| 3 | `lfix_base_workspace.jl` (A+), `lfix_factorized.jl` (C+), `lfix_kbplus.jl` (:kbplus), `gradient_workspace.jl` (pooled) | Factorized log-price representation avoiding dense O(W*D^2) tensors; reconstruct a queried cell on demand | Ricardian-specific for the *ranking* scan; the *factorize-and-reconstruct* idea is transferable with adaptation around Melitz's participation bookkeeping | C+: 6.75x faster/6.0x less memory vs reference at real D=20/W=80,000 (`fullA_price_tensor_elimination_report.md`); fair benchmark: C+ 4.01-4.20x/66.77x-less-mem (bit-exact) | Five backends = real surface area; rare-fallback paths less tested | Adopt the *idea* with care; skip the winner/runner-up ranking machinery itself |
| 4 | `winner_certificate.jl` (`WinnerRefCache`) | Cache winner/runner-up/third per (draw,destination); O(1) resolve if <=2 origins changed | **Ricardian hard-winner-specific -- do not port directly**; no runner-up concept exists under a participation gate | 20.9-21.2x at real D=20 (noise-level at D=4 -- scale-dependent) | N/A (not portable as-is) | Skip; if needed, build a "distance to nearest participation-cutoff crossing" certificate instead |
| 5 | `lfix_base_workspace.jl`/`lfix_factorized.jl` (`affected_cells`) | Only recompute destination columns actually reachable by a given coordinate (static footprint from the pivot/gauge map) | Transferable with adaptation -- generic "only touch what a parameter can move" principle; footprint bookkeeping must be re-derived for Melitz's own pivot structure | Folded into backend numbers above (a structural precondition, not standalone) | Correctness hinges on an exactly-right affected-cell map; any pivot change requires re-derivation | Adopt with care |
| 6 | `oracle.jl` (`SafeExactCache`/`FullAEvalKey`) | Lock-guarded exact-point cache keyed on `(x_free, delta, find_smallest, opt file, mode, ctx fingerprint)`; only verified results stored | **Directly transferable** -- purely about outer-solve re-evaluation patterns (line search, gradient probes revisiting points), independent of winner structure | Not separately quantified as a standalone multiplier; correctness/safety layer | Key completeness is critical (AUD-08: an omitted input caused a real false-hit bug); lock must guard only the O(1) dict op, not the solve | **Adopt now** |
| 7 | `cross_delta_cache.jl` (`CrossDeltaExactCache`) | Strips delta from the cache key since the inner CC dual solve is provably delta-independent; reuses solves across a staged delta continuation | **Directly transferable, arguably HIGHEST priority for Melitz** -- `finite_delta_outer.jl`'s own staged/budget-varying outer search has the identical delta-independent inner dual structure | 19/19 real-driver gate tests passed; no aggregate multiplier quoted (one short staged run) | A prior version had a too-narrow `Union` type on the `exact_cache` keyword that silently disabled `cross_delta=true` in production -- audit full call-site types | **Adopt now** -- highest-leverage, lowest-risk item on the list |
| 8 | `dual_bank.jl` (`DualBank`) | Recency-bounded (8) bank of successful inner duals; `select_warm_start` scores candidates via a cheap pre-solve KKT proxy, no extra KNITRO call | **Directly transferable** -- warm-starting a convex inner dual from the nearest successful prior solve has no winner-structure dependence | 29.5% wall-time reduction on one live delta=5 trajectory, 6.3% on a shorter one (trajectory-dependent) | `cheap_score` is a proxy, not the true post-solve KKT residual; scoring cost is nonzero (though far cheaper than a solve) | **Adopt now** |
| 9 | `compressed_moments.jl` (`CompressedFactual`) | O(W*D)+O(D^2) compressed representation of the dense O(W*(D^2+1)) moment matrix, exploiting that only the winning origin contributes per destination | Ricardian-specific in derivation; the "avoid materializing a dense matrix when the moment sum has algebraic structure" idea is transferable with adaptation around Melitz's *variable-size active-participant* sum | Isolated: 1.08-1.29x; "does not survive an actual cold KNITRO solve (0.99x, a wash)" for the naive integration -- an explicit disclosed negative result | Asymptotic improvement is not guaranteed to win end-to-end once JIT/compile/GC is accounted for | Adopt the idea with care; re-measure end-to-end, don't assume a win |
| 10 | `structured_moment_build.jl` | Splits moment/Hessian-input matrix into a cheap rank-one fixed term (broadcast, ~30% faster than `BLAS.ger!` for this shape) + irregular scatter term (tight scalar loop) | Transferable with adaptation -- generic "cheap fixed part + irregular scatter part, fastest primitive for each" decomposition; Melitz's own fixed/scatter split must be re-derived for the CES/Pareto formula | ~1.32x genuine full-inner-solve speedup (`fullA_D20_structured_moment_report.md`), bit-checked to 1.137e-13 | Getting the exact algebraic split wrong silently produces a wrong Hessian, not an error | Adopt with care -- validate against a dense reference before trusting |
| 11 | `c10_d20_production_driver.jl`/`cm_checkpoint.jl` (`D20Checkpoint`/`CMCheckpoint`) | Periodic atomic (tmp-then-mv) checkpoints of full outer-search state, schema/draw/version-stamped, hard-errors on any mismatch | **Directly transferable** -- resilience for long outer NLP searches, orthogonal to winner-vs-participation structure | Not a speed multiplier (resilience feature); bit-for-bit reproduced incumbent after interrupt+resume, verified | Must hard-error on schema/draw/version mismatch, never silently resume into a different problem instance | **Adopt now** (once campaigns get long enough to justify it) |
| 12 | `infeasibility_screen.jl`, `fast_range_screen.jl`, `negative_cache.jl`, `oracle.jl` (`classify_inner_result`) | Analytic pre-solve infeasibility screens; negative-result caching ONLY after a second, differently-started confirming solve; independent KKT/gap/residual classification before trusting any "feasible" status | **Directly transferable (classification/negative-cache policy)**; the specific screen certificates themselves are Ricardian hard-winner-specific and need Melitz-specific analogues (e.g. "this firm can never earn positive profit at any feasible theta on this draw support") | Screening: ~1-5s wasted solve -> microseconds (qualitative); negative cache: 10/10 KNITRO -300 failures independently HiGHS-certified infeasible, 8/8 survived a 20-step warm rescue | Caching on first failure, or conflating a resource-limit code with a definitive infeasibility code, silently poisons the cache | Adopt the classification/negative-caching *policy* now; design screens later, separately |
| 13 | `gradient_workspace.jl` (`Threads.@threads :static`), `cc_algo/parallelism_guards.jl` | Thread-parallel coordinate probes AFTER the base inner solve completes, guarded by mutual-exclusion Refs that hard-error on any overlap | Transferable with adaptation -- scheduling discipline is generic; directly relevant once Melitz coordinate probes are localized | Not isolated as its own multiplier (folded into `threaded=true` backend numbers) | This exact surface produced a real prior regression in this repo (nested-KNITRO-solve hang from `par_concurrent_evals=no`); the guard is mandatory, not optional | Adopt with care -- guard mechanism is non-negotiable if parallelism is added |

**Top 5 for Melitz, ranked by (value x safety), per the survey:**
1. Cross-delta exact-point cache (#7) -- most directly applicable given Melitz's own staged delta search.
2. Exact-point cache + verified-success classification (#6, #12c) -- foundational, orthogonal to structure differences, precondition for #1.
3. Verified warm-start bank (#8) -- real 6-30% measured win, no winner-structure dependence.
4. Persistent gradient/objective workspaces (#1) -- eliminates a documented ~4GB/call pattern; **partially implemented this session, see Section G**.
5. Checkpoint/resume with hard mismatch-gating (#11) -- resilience for future long Melitz campaigns.

Deliberately excluded from the top 5 despite large quoted speedups: the winner-margin/
runner-up cache and C+/kbplus factorized backends (#3, #4) -- their 4-20x numbers are
earned entirely from Ricardian's argmin-over-origins structure, which Melitz's
participation-gate model does not have a role for.

## G. Implemented optimizations

### G.1 Preallocated moment-matrix workspace for Method B's coordinate-probe loop (item #4 in the ranked list)

**Where**: `src/melitz/gradient_lab.jl` (`_fill_fixed_active_set_moments!`,
`fixed_active_set_moments!`) and `src/melitz/finite_delta_outer.jl`
(`make_melitz_moments_jacobian_b`).

**What changed**: `make_melitz_moments_jacobian_b`'s coordinate-probe loop -- which the
governing prompt's own Section 7.5 flagged as doing "up to `2*30=60` full displaced
moment builds per gradient" at D=4 -- previously called the allocating
`fixed_active_set_moments(theta, ctx, obj)` twice per coordinate (`Gp`, `Gm`), each call
allocating a fresh `W x (D^2+1)` `Float64` matrix. This is the SAME function used on the
LIVE `finite_delta_outer.jl` GA-callback path (`obj(x, dummy_g, theta; jac=local_jac)`
inside `cb_G!`, which dispatches to `moments_jacobian!` == this closure).

The moment-construction logic itself was factored into one shared, type-generic fill body
(`_fill_fixed_active_set_moments!`, works for both plain `Float64` and `ForwardDiff.Dual`
via a type parameter `T`) so:
- `fixed_active_set_moments` (used by Method C's ForwardDiff path, which needs a
  fresh `Dual`-typed buffer every call and cannot share state) is now a 4-line allocating
  wrapper around the shared fill body -- unchanged behavior, unchanged signature.
- `fixed_active_set_moments!` (new) is an in-place `Float64`-only entry point taking
  caller-owned `G`/`profit_j` buffers.
- `make_melitz_moments_jacobian_b`'s closure now owns two persistent `(W x num_moments)`
  buffers (`Gp_buf`/`Gm_buf`, lazily (re)allocated only on a shape mismatch) and calls
  `fixed_active_set_moments!` instead of the allocating version -- cutting the per-gradient
  matrix allocation count from 60 (at D=4/n=30) fresh matrices down to 2 buffers reused
  across the entire outer trajectory's GA callbacks.

**Correctness preserved**: both entry points call the identical `_fill_fixed_active_set_moments!`
body -- there is exactly one implementation of the moment economics, not two. No
downstream consumer's numerics changed; every existing gradient-lab/outer-solve/
finite-delta test that exercised Method B or the live GA callback continues to pass
unmodified.

**Tests added** (`test/melitz/runtests.jl`, "Section 12.4"):
- bit-identical output between the allocating and in-place entry points at 5 random
  points (`==`, not `isapprox` -- same fill body, so floating-point results must match
  exactly, not merely agree numerically);
- an explicit `@allocated` before/after comparison, warmed up post-JIT, asserting the
  in-place call allocates strictly fewer bytes than the allocating call (which must itself
  allocate at least the raw `W x d x 8` bytes for its own fresh matrix);
- a live cross-check that two INDEPENDENTLY-CONSTRUCTED `make_melitz_moments_jacobian_b`
  closures (each with its own persistent buffers) reproduce an IDENTICAL Jacobian at the
  same point, and that calling the SAME closure twice in a row reproduces identically too
  -- ruling out any cross-call buffer contamination from the reused-buffer design.

**Before/after allocation numbers**: see the test suite's own printed
`[Section 12.4] allocating=... bytes, in-place=... bytes (...x less)` line for this
session's exact measured figures at the small (`W=2,000`) gradient-lab fixture (a `Wt x d`
matrix at `D=4` is `2,000 x 17` `Float64` = 272,000 bytes minimum for the allocating path
per call; the in-place path allocates ~0 bytes per call once its buffers exist).

**End-to-end effect**: not yet separately re-measured on a full outer trajectory in this
report (the live profiling run in Sections B-E already captures `:ga_divergence_gradient`
wall time with this optimization ALREADY applied, since it was implemented before those
runs) -- so Sections B-E's `:ga_divergence_gradient` numbers already reflect the
post-optimization state, not a lever still to pull. A future session wanting an isolated
before/after trajectory comparison would need to temporarily revert this commit and
re-run the same campaign.

## H. Production recommendation

(pending -- to be finalized after Sections B-E complete)
