# Continuation 12: common-marginals restriction for the full-A_od exact solver

Status: D=4 gates PASSED (architecture built, validated, and a real constrained outer solve
produced at 3 grid sizes). D=20 scaling NOT started -- see Section 9 "what's not done" and the
resume commands at the end for exactly how to continue.

## 1. Repo audit (section 1 of the standing brief)

- Production full-A branch: `diag/fullA-d4-exact`, worktree `gravity-fullA-d4`, HEAD `02583bc`
  at the start of this continuation. That worktree had substantial UNCOMMITTED work from
  continuation 11 (an active overnight delta=1 push, PID still running at session start) — not
  touched, not built on top of.
- This continuation's work lives on a NEW additive branch, `diag/fullA-d4-exact-common-marginals`,
  in a NEW worktree `gravity-fullA-d4-c12-common-marginals`, branched from `diag/fullA-d4-exact`
  commit `02583bc`. Three further sibling worktrees/branches (`-cm-interval-hessian`,
  `-cm-hessian-arch`, `-cm-conditioning`) were created off THIS branch for parallel subagent
  workstreams (see Section 9).
- D=4 synthetic driver: `full_aod_diag/d4_exact/context.jl::d4_exact_setup(...)` is the canonical
  context builder every script in that directory uses. Trusted oracle: `oracle.jl::evaluate_fullA`
  (dense/reference); `oracle_fast.jl::evaluate_fullA_fast` (dense/compressed dispatch, not used by
  this continuation's new code yet — see Section 8, "what's not done").
- D=20 real-data driver: `context_real_d20.jl::d20_real_setup`, production batch driver
  `c10_d20_production_driver.jl`.
- Dense CC inner solver: `cc_algo/PsiObjectiveBundle.jl` (`PsiObjectiveBundleImplicit`, dense BLAS
  `hessian!`), `cc_algo/inner_loop_functions.jl` (KNITRO dual-solve bundle).
- Compressed CC inner solver: `full_aod_diag/d4_exact/compressed_*.jl`, dispatched via
  `oracle_fast.jl`'s `moment_representation=:dense|:compressed`. NOT used by this continuation's
  CM implementation (the CM block is dense-appended; compressed's winner-form representation does
  not naturally accommodate the CM columns, which are draw-level bin-membership functions, not
  winner-index functions — flagged as future work, not attempted this continuation).
- Structured moment construction: `structured_moment_build.jl` (continuation 10). Not touched.
- Exact Hessian: `cc_algo/PsiObjectiveBundle.jl::hessian!` (dense BLAS gemm, no block structure);
  chunked-over-draws variant `chunked_hessian.jl` (D=20/large-W memory bounding, not relevant here).
- Fast outer gradient: `composite_gradient_fast.jl`/`lfix_incremental.jl` (the "Lfix" machinery).
  **NOT yet wired to be CM-aware** — see Section 7/9 "what's not done."
- Checkpoint/resume + infeasibility screening: `c10_d20_production_driver.jl`'s `D20Checkpoint`;
  `infeasibility_screen.jl`'s exact pairwise certificate (`S_sod = B_so + a_od`).
- Candidate registry: `candidate_registry.jl`, literal `w` (pivot-reduced-log-A) vectors, current
  D=4 headline upper `kappa=0.17245688540655113`, headline lower `kappa=0.004387827651021192`.
- **No existing full-A common-marginals implementation existed anywhere in this repo** before this
  continuation. A sequential-method (not full-A) analog exists and was validated/reused as the
  math reference: `sequential_gravity/common_marginals_moments.jl` on branch
  `fix/cm-fixed-dual-gradient` (same git-common-dir as this repo — a same-repo branch, not an
  external file). A disconnected LEGACY full-A-adjacent implementation exists
  (`moments/pairewiseIndependenceMoment!.jl`, `sameMarginalsMoment`/`independenceMoment` flags in
  `moments/moments!.jl`) but is explicitly asserted OFF (`error(...)`) in the current exact
  `gammanorm`/`directgp` moments path this branch's whole machinery is built around — not reused.

**Correction mid-session (user-flagged, verified in code)**: the LATEST full-A production driver
(`c10_d20_production_driver.jl`, `composite_gradient_fast.jl`) does NOT enforce the gravity
moment as a live KNITRO outer constraint. It analytically ELIMINATES gravity via
`gravity_elimination.jl::build_pivot_elimination`/`pivot_expand` — one A_od entry per D×D block
is solved as an affine function of the other D²−1 (in log-A coordinates, where the gravity moment
is exactly linear), so gravity is satisfied by CONSTRUCTION and KNITRO's outer decision variables
are the reduced `zfree` (D²−1 dims), never the raw A_od block. A second, older production path
(`run_fullA_D4_production.jl`, `CS.outer_loop_cached`) DOES carry gravity as a live outer
constraint with an analytic tariff-based gradient, over the FULL (unreduced) A_od parameterization
— both conventions currently coexist in the codebase. This does not change the CM implementation's
column-layout requirement (gravity must stay the sole outer-only suffix column in `G` regardless
of how the caller constructs θ — see Section 2), but it does mean any FAST/production-grade
CM-aware outer-loop wiring (Section 7/9) should reuse `zfree`/pivot-elimination, not the raw
17-dim layout used for this continuation's fixed-point validation and first correctness-focused
outer solve (Section 6).

## 2. Mathematical formulation implemented

CDW eq. 35 (cumulative-CDF, anchored-to-reference-country form), exactly as already validated in
the sequential-method codebase: for non-reference origin `o` and quantile cutoff `z_l` (an
evenly-spaced-probability empirical quantile of the reference origin `1`'s raw baseline Exp(1)
draws), impose

```
E_F[ 1{U_o <= z_l} - 1{U_1 <= z_l} ] = 0,   o = 2..D,  l = 1..L
```

under the least-favorable reweighted F the CC inner dual solve finds. Ported verbatim (math
unchanged, attributed) into `full_aod_diag/d4_exact/common_marginals_moments.jl`, including the
already-validated orthonormal-contrast alternative (`contrasts=:orthonormal`, a closed-form
`R=(BB')^{-1/2}` Helmert-like reparameterization removing the shared-reference-country
correlation the anchored form induces — same restriction count, exact invertible transform).

**New (full-A-specific) glue**, also in that file:
- `wrap_moments_with_cm(core_moments!, ncore_full, CM)`: builds a `moments!`-compatible closure.
  CRITICAL layout fact (found by reading `cc_algo/PsiObjectiveBundle.jl`'s callable, not assumed):
  moment columns `outer_constr_index:d` are OUTER-only (evaluated at θ directly, never
  inner-CC-reweighted); in this codebase that suffix is currently exactly ONE column, the
  gravity/orthogonality moment, which `newGravityMoment!.jl` unconditionally writes to `G[:,end]`
  of whatever view it receives. The common-marginals restriction is an INNER moment (imposed on
  the least-favorable F via its own dual multiplier), so it must be spliced BEFORE the gravity
  column, not after — `wrap_moments_with_cm` calls the core moments function into a same-eltype
  temporary sized to the ORIGINAL column count (so gravity lands at ITS local last position),
  then relocates: pre-gravity core -> `G`'s prefix, CM block -> the next `ncm` columns, gravity ->
  `G`'s new last column.
- `build_cm_augmented_obj(ctx, CS; L, contrasts=:anchored)`: builds a NEW
  `PsiObjectiveBundleImplicit` (via `CS.PsiObjectiveBundleImplicit`) with `d` and
  `outer_constr_index` both grown by `ncm=(D-1)*L`, leaving the original context's `obj`
  untouched (fully additive, no destructive edits to trusted production code).

## 3. Dense reference validation (section 2/12 of the brief)

`c12_validate_dense_cm.jl`, `c12_d4_fixed_param_battery.jl` (both committed). At D=4, W=8000,
calibration point (A_od=1), L in {10,20,50}, both `:anchored` and `:orthonormal` contrasts:

- Inner KNITRO dual solve converges cleanly (`nStatus=0`) at every L/contrast combination.
- KKT residuals (`sum(m*G_j)/W` at the solved LFD weights `m`) are at MACHINE PRECISION
  (~1e-16 to 1e-17) for both the pre-existing core moments and the new CM block — the strongest
  form of "this restriction is correctly imposed on the least-favorable F" evidence available.
- Anchored and orthonormal contrasts agree exactly (same max residual, same `Delta_dual`,
  same `gamma'_focal`) — confirms the rotation is a true invertible reparameterization, not an
  accidental change to the feasible set.
- `Delta_dual` (the CC divergence needed to satisfy ALL restrictions from calibration) increases
  monotonically with `L` (0.0010 baseline -> 0.0033 (L=10) -> 0.0051 (L=20) -> 0.0115 (L=50)) —
  expected: more restrictions shrink the feasible-F set, so more divergence is needed to still
  rationalize the same θ.
- **Economically meaningful finding**: the EXISTING unrestricted D=4 upper headline candidate
  (`candidate_registry.jl`'s `w_up40`, kappa=0.17246) is badly INFEASIBLE under the CM
  restriction — `Delta_dual` jumps from ~1.0 (its own unrestricted value) to 1.87 (L=10) /
  1.99 (L=20) / 2.18 (L=50), all far past `delta=1`. The restriction has real economic bite; the
  restricted upper bound at delta=1 will be materially smaller than 0.1725.
- A deliberately structurally-infeasible point (one A_od entry driven to ~1e-6, so that origin can
  essentially never win a destination cell under exact hard-max) is correctly rejected
  (`nStatus=-300`, KNITRO's unbounded-dual code) at every L tested — the CM-augmented bundle
  inherits the base solver's infeasibility detection correctly, no special-casing needed.

## 4. Sign-convention finding (not in the original brief, discovered mid-session)

The plain generic outer-loop path (`CS.outer_loop`, `cc_algo/outer_loop_functions.jl`) and the
specialized cached/envelope-gradient path (`CS.outer_loop_cached`,
`run_fullA_D4_production.jl`) use DIFFERENT internal sign conventions for `find_smallest` under
the `directgp` objective convention (`K = gamma'_focal` directly, not `kappa`). The cached path's
own `obj_grad_fn!` applies a compensating `(-1)^find_smallest` flip so `find_smallest` keeps its
"finds smallest kappa" meaning; the plain generic path does NOT apply this compensation, so under
it `find_smallest=true` empirically finds the SMALLEST `gamma'_focal`, i.e. the LARGEST kappa
(upper bound) — verified empirically (not just derived from reading code, which is easy to get
backwards here), `c12_sign_convention_smoke_test.jl`: `find_smallest=true` moved kappa from 0.064
(calibration) toward the known upper anchor (reaching 0.164 in a 25-iteration smoke run, target
~0.1725); `find_smallest=false` moved it toward the known lower anchor (reaching 0.0039, target
~0.0044). **Any future CM-aware outer-loop script using the plain `CS.outer_loop` path for an
UPPER bound must use `find_smallest=true`** — the opposite of a naive reading of
`moments_gammanorm.jl`'s own docstring, which describes the CACHED path's convention, not this one.

## 5. D=4 delta=1 CM-constrained upper-bound solve (section 13)

Driver: `c12_d4_delta1_upper_cm.jl` (usage: `julia --project=. full_aod_diag/d4_exact/c12_d4_delta1_upper_cm.jl <L> <delta>`).
Deliberately uses the GENERIC ForwardDiff/dense-jac_h outer-loop path (`CS.outer_loop`,
`needs_outer_moment_jacobian=true` — safe at D=4's scale, unlike D=20 where this tensor caused a
prior server-wide memory incident, see `docs/fullA_jach_audit.md`), NOT the specialized
Method-B/envelope-gradient cached path (`run_fullA_D4_production.jl`'s `div_grad_fn!`): that path
hardcodes a direct call to `EK_moments_gammanorm_directgp!` inside
`ad_benchmark/derivative_core.jl::moment_map!`, bypassing `obj.moments!` entirely — confirmed by
reading the code, not assumed — so it would silently ignore the CM columns if reused naively. The
plain `CS.outer_loop` path dispatches `calculate_grad_k_autodiff!`/the dense-Jacobian gradient
generically THROUGH `obj.moments!`, so it correctly picks up the CM wrapper. This is the
correctness-first choice; a CM-aware fast envelope-gradient path is future work (Section 9).

**L=10 result** (`csw_outer_100.opt`, maxit=100, generic ForwardDiff outer path):

| L | status | outer_iters | opt_err | feas_err | gamma'_focal | kappa | Delta_dual (recheck) | gravity | CM max KKT | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | -400 (maxit hit) | 100/100 | 0.0195 | 1.3e-7 | 0.9056808059 | **0.1522028746** | 0.95766 (<=1) | -8.3e-9 | 1.8e-14 | 217s |

`status=-400` fired because `outer_iters` exactly hit the 100-iteration cap (`csw_outer_100.opt`),
not a real failure -- `opt_err=0.0195` shows the KKT optimality gap hasn't fully closed yet, while
`feas_err=1.3e-7` shows the point is tightly feasible. This is the SAME "best-feasible-found-in-a-
fixed-budget, not yet a certified optimum" situation continuation 11 documented repeatedly for the
unrestricted problem (see `[[d20-delta1-round4-independent-verification]]` in memory) -- more
outer budget would likely push kappa higher still (the direction of travel, 0.0642 calibration ->
0.1522, is toward the unrestricted anchor 0.1725, consistent with the restriction's real but
not-total bite).

**Independent re-verification is genuinely independent of the search's own bookkeeping**: a
FRESH cold `evaluate_fullA` call at the reported solution (not reusing KNITRO's own internal
state) gives `nStatus=0` (clean, unrelated to the OUTER status), confirms `Delta_dual=0.9577<=1`
(the delta=1 constraint holds), `gravity_value=-8.3e-9` (~0, correctly eliminated), and separately
recomputes the CM-block's own KKT residual at that exact point: `1.8e-14`, machine precision --
the common-marginals restriction is genuinely, exactly satisfied by the least-favorable F at this
solution, not merely "close enough."

**Headline comparison**: unrestricted upper kappa=0.17246 vs CM-restricted (L=10) kappa=0.15220
-- an **11.8% reduction** in the upper bound purely from imposing this restriction, at the same
delta=1 divergence budget. Confirms the Section 3 finding (the unrestricted candidate is badly
CM-infeasible) translates into a real, materially different answer once the restricted problem is
actually re-optimized rather than just checked at the old optimum.

**L=20 and L=50 results** (same setup, `csw_outer_100.opt`, maxit=100):

| L | status | outer_iters | opt_err | feas_err | gamma'_focal | kappa | Delta_dual (recheck) | gravity | CM max KKT | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | -400 (maxit hit) | 100/100 | 0.0195 | 1.3e-7 | 0.9056808059 | 0.1522028746 | 0.95766 | -8.3e-9 | 1.8e-14 | 217s |
| 20 | -101 (genuine KKT convergence) | 76/100 | 0.00206 | 1.3e-18 | 0.9029744779 | **0.1564209371** | 0.99969 | 8.1e-20 | 6.3e-15 | 248s |
| 50 | -102 (genuine KKT convergence, looser) | 42/100 | 0.0485 | 1.1e-15 | 0.9037514955 | 0.1552107445 | 0.99988 | -6.8e-17 | 1.4e-16 | 371s |

All three are independently re-verified feasible (Delta_dual right at the delta=1 boundary, as
expected for an upper-bound problem at the optimum -- the search should fully use the divergence
budget) with CM-block KKT residuals at or near machine precision in every case.

**Important caveat, found while writing this up -- do not over-read the L=10/20/50 ordering as a
clean comparison**: kappa is NOT monotonically decreasing in L (0.1522 < 0.1552 < 0.1564), which
would be surprising if these were a strictly NESTED sequence of restrictions (more restrictions
can only shrink the feasible-F set further, so kappa_max should be weakly decreasing in a nested
sequence, ASSUMING both searches find their own true global optimum). Checked directly: the
L-quantile grids as constructed (`k/L` for `k=1..L-1`) give **L=10's 9 cutpoints as an exact
subset of L=20's 19** (both are multiples of 0.05) -- so L=10 vs L=20 IS a clean nested comparison
-- but **L=50's 49 cutpoints are NOT a superset of L=20's 19** (0.05 is not a multiple of 0.02,
confirmed directly: `set(L20_grid).issubset(set(L50_grid))` is `False`) -- so L=20 vs L=50 is not
nested at all, compounded by L=50 having the loosest optimizer convergence of the three
(`opt_err=0.0485`).

For the genuinely nested L=10-vs-L=20 pair, ruled out the "L=10 just needs more budget" hypothesis
directly: re-ran L=10 with a 3x larger iteration cap (`csw_outer_300.opt`, 300 vs 100) and it
converged to a genuine KKT point (`status=-102`) after only 116 iterations at
**kappa=0.1517529468** -- barely different from the original 100-iteration value (0.1522, a
-0.3% move), not the "still climbing toward 0.156" pattern budget-starvation would predict. Both
L=10 runs are tightly converged and land at ~0.152, while L=20 lands at 0.156, a genuine ~2.6%
higher value on a STRICTLY LARGER feasible set (L=10's 9 restrictions are a subset of L=20's 19,
so L=20's problem is MORE constrained, and its optimum should be weakly BELOW L=10's, not above
it). **Conclusion, more precise than the earlier draft of this section**: this is not an
optimizer-budget artifact -- it is most likely a NON-CONVEX LANDSCAPE / single-start local-optimum
artifact. The outer KNITRO search here is single-start (one initial point, the calibration
equilibrium) over a genuinely non-convex problem (this matches continuation 11's own documented finding for the UNRESTRICTED problem: a 3-point
multistart there found a 6-9% spread in kappa at matched budgets, see
`docs/fullA_continuation11_handoff.md` Section 4 in the parent `diag/fullA-d4-exact` branch) --
so a single L=10
run converging to a WORSE local optimum than a single L=20 run is entirely plausible without
either run being "wrong." **This L-dependence finding therefore needs multistart at each L (not
just more single-start iterations) before it can be read as "kappa vs L," and is flagged
explicitly as unresolved** -- the economically solid, well-supported finding from this pass is
narrower but still real regardless of this ambiguity: **at every L tested, from every start tried,
the CM-restricted upper bound (0.152-0.156) is meaningfully below the unrestricted headline
(0.1725)** -- a 9-12% reduction -- not the specific value/ordering of the L-dependence itself.

## 6. Parallel subagent workstreams (sections 3-6, 8-11 of the brief)

Three subagents ran in sibling worktrees off this branch, each with the validated dense reference
as ground truth to equivalence-test against. All three branches are now MERGED into this one
(clean merges, disjoint new files, no conflicts). Full reports:
`docs/fullA_cm_interval_lookup_report.md`, `docs/fullA_cm_hessian_architecture_report.md`,
`docs/fullA_cm_conditioning_and_adaptive_grid_report.md`.

### 6a. Interval/bin-index reformulation + lookup kernels (sections 3-6)

Cumulative<->interval transform built two independent ways (block-cumsum and an explicit
`kron(upper-tri-ones, I_nO)` matrix), agreeing with each other and reconstructing the dense
reference to 0-1.11e-16 at L in {10,20,50}, both contrast modes. The interval-augmented CC inner
solve matches the dense one (Delta_dual/Delta_primal ~1e-15-1e-19, LFD weights ~1e-14-1e-16) at
calibration/perturbed/infeasible points, correctly rejecting the infeasible one identically.

O(D)-per-draw lookup FG kernels (interval basis, plus a cumulative-suffix-sum diagnostic variant)
reproduce the dense callable to machine precision and were wired into a REAL live KNITRO inner
solve (not just an isolated microkernel), matching the untouched dense baseline to the same
tolerance across 12 configs. Two real bugs caught and fixed en route: a reshape-orientation bug
that silently transposed origin<->bin, and an off-by-one BLAS slice bound.

**Honest end-to-end benchmark**: the isolated FG-kernel cost shows the expected O(D) vs O(D*L)
scaling clearly (dense 324/511/998 mus/call vs lookup flat ~325-365 mus/call at L=10/20/50 ->
1.0x/1.4x/2.9x). But the FULL cold KNITRO inner solve only speeds up 1.18-1.29x, because (a) at
D=4 the problem converges in just 4-5 KNITRO iterations, so KNITRO's own per-iteration overhead
dominates the callback saving, and (b) the Hessian callback was deliberately left dense/unoptimized
for this workstream (that's Section 6b's job), so its cost grows with L and partially offsets the
FG win. **Conclusion: the lookup formulation is a real, verified, correct win, but a MODEST one
end-to-end at D=4 -- the bigger payoff is expected at D=20/L=50 where KNITRO iteration overhead is
a much smaller share of total cost and the O(D) vs O(D*L) gap is larger in absolute terms**, not
yet measured. Threaded histogram building: ~25-30% win from 1->8 threads, REGRESSED at 16 threads
(workload too small for W=8000) -- a real but limited finding, not a production recommendation.

### 6b. Exact Hessian architecture comparison + block elimination (sections 8, 10)

Four architectures compared (A: trusted dense BLAS baseline; B: structured/cached moment-block
materialization; C: exact structured Hessian via weighted bin-contingency tables `T_op`/`T_o1`/
`T_1p`/`T_11`; D: matrix-free HVP via KNITRO's CG mode, diagnostic-only per the brief).

**Winner: Architecture C.** Hessian-callback time 1.4x (L=10) -> **5.2x (L=50, anchored)** faster
than dense BLAS; full cold KNITRO inner solve 1.1x -> **2.7x (L=50)** faster. All four architectures
agree with the trusted baseline to ~1e-15 -- but only after a real bug was found and fixed: C's
first implementation was off by a uniform 2x on the `H_EC` (economic-CM cross) block because it was
written only into the upper-triangle position and never mirrored before a symmetrization step
averaged it against an unset zero. Architecture B gave a modest, real 5-8% cold-solve speedup from
avoiding `wrap_moments_with_cm`'s fresh-allocated `G_tmp` per call. Architecture D validated
correctly as an independent oracle but was 4-8x SLOWER overall (138-394 HV calls per solve vs 4
dense calls) -- confirmed diagnostic-only, as the brief anticipated.

**Block elimination (Schur complement on `H_CC`) does NOT help**: 1.46x SLOWER than direct
factorization of the full Hessian, because `H_CC` (150x150 at L=50) already costs 81% of factoring
the entire 168x168 matrix -- eliminating the LARGE block gives no asymptotic benefit here (this
would likely flip at larger D where the CM block is a smaller fraction of the total). Correctness
of the reorganization itself was verified (~1e-27 at a KKT point, ~2e-9 on a random RHS) even
though it isn't the right lever to pull.

**D=20/L=50/W=80000 projection** (not run, explicitly out of scope for this workstream): C's
nominal FLOP advantage shrinks somewhat with D (322x at D=4 -> ~216x at D=20 projected, since the
core-moment count grows ~D^3 and partially catches up to the ~D^2-scaling CM block at fixed L), but
memory is likely the more decisive factor given this repo's prior 109GB jac_h incident (see
Section 1) -- C is the only architecture avoiding an ~828MB dense Hessian-step scratch buffer
(needs only ~297MB of its own bin-table scratch instead). A fully memory-lean D=20 pipeline would
also need the FG callback itself restructured to match (Section 6a's lookup kernels, not yet
combined with Architecture C's Hessian -- see Section 9 "what's not done").

### 6c. Conditioning experiments + adaptive quantile activation (sections 9, 11)

96 runs (4 bases x 3 L's x 2 points x 4 reference countries), all converged. Conditioning ranking,
CONSISTENT at every L and both test points: **interval < std_interval < orthonormal < anchored**.
At L=50, interval beats anchored by ~11-13x and orthonormal by ~3-4x in `cond(Hessian)`. Critically,
anchored's conditioning DEGRADES ~4x from L=10 to L=50 while interval's stays flat -- i.e. the
anchored default (what Sections 3-5 above and the D=4 outer solves used) is the WORST-scaling
choice as L grows, exactly the regime a future D=20/L=50 run would stress hardest.

Sparsity tradeoff (structural, not just numerical): anchored and interval columns are each a
function of exactly 2 origins' raw draws (provably sparse); orthonormal's rotation matrix is fully
dense (every column touches all D origins), forfeiting the per-origin sparsity this codebase's
compressed-moment machinery elsewhere already exploits. **Net recommendation from this workstream:
interval basis dominates orthonormal on BOTH conditioning AND sparsity -- there is no scenario
found where orthonormal is the better choice.** (Standardizing interval columns further made
conditioning ~2x WORSE, not better -- not adopted.)

Reference-country sensitivity is real but an order of magnitude smaller than basis choice
(single-digit % spread for anchored/interval/orthonormal across all 4 possible reference
countries, up to ~19% for standardized-interval) and never changed feasibility or iteration count
in any of the 96 runs -- basis choice is the first-order lever, not reference-country choice.

Rank deficiency: exactly one redundant column at the exact calibration point (A_od=1) only, traced
to a pre-existing degeneracy in the CORE (non-CM) moment set at that one symmetric point -- not
introduced by CM in any basis, absent at any perturbed point, nothing removed (out of scope, not a
bug in this continuation's work).

Adaptive quantile activation: **found no exploitable redundancy across quantile levels at D=4** --
starting from a 7-threshold seed and adding the largest violations, the loop always escalated to
the FULL 50/50 (or 20/20) candidate grid; a follow-up check showed 41-43 of 43 un-imposed
thresholds still violate even a loose 1e-3 tolerance after only 7 are imposed. This is a genuine,
if not the hoped-for, negative result: each quantile threshold carries distinct binding
information at this D and W, so a coarse active subset is a poor approximation of the dense grid.
The final fully-covered active set DOES agree with the independent dense L=50 reference to
5.2e-18/3.2e-16 (Delta_dual) -- so the MACHINERY is correct, it just doesn't buy fewer constraints
here. Warm-restart (reusing the previous dual solution) was verified correct against a cold
re-solve and saves ~35% wall time even though iteration count doesn't change at D=4's easy scale.

## 7. D=20 scaling (sections 15-16)

NOT STARTED this continuation — gated on the D=4 outer-solve result and subagent findings above
per the brief's own decision criteria ("proceed only after the D=4 equivalence and outer-solve
gates pass"), which have now passed. Next command to resume: adapt
`context_real_d20.jl::d20_real_setup` + `common_marginals_moments.jl::build_cm_augmented_obj`
following the exact same pattern as Section 5's D=4 driver, starting with the production
microbenchmark table (brief Section 15's "First") before any long outer solve. Given Section 6's
findings, the D=20 attempt should NOT simply copy the D=4 driver's choices verbatim: use the
INTERVAL basis (not anchored -- Section 6c's clear conditioning+sparsity win), Hessian
Architecture C (Section 6b), and budget for the fact that the lookup-kernel FG win (Section 6a)
is expected to matter much more at D=20 scale than the modest 1.2-1.3x seen at D=4.

## 7b. Production decision criteria (brief Section 16) -- assessed against this continuation's evidence

1. Cumulative and interval formulations produce the same finite-grid result: **YES**, verified to
   ~1e-14-1e-19 across multiple points and L (Section 6a).
2. Optimized implementation avoids repeated construction of fixed moments: **PARTIAL** -- bin
   indices/CM block are precomputed once per context as designed, but `wrap_moments_with_cm`
   still allocates a fresh `G_tmp` temporary per `moments!` call (flagged, not yet fixed -- a
   known, modest cost per Architecture B's 5-8% measurement).
3. Exact inner solutions remain numerically stable: **YES** -- every converged run (dozens, across
   all three subagent workstreams plus the D=4 outer solves) hit KKT/complementarity residuals at
   or near machine precision; the one deliberately-infeasible test point was correctly rejected
   every time, never silently "solved."
4. Memory remains safe at D=20/L=50/W=80000: **PROJECTED safe for Architecture C specifically**
   (Section 6b: ~297MB own scratch vs dense's ~828MB), memory behavior of the FG path (Section 6a)
   at that scale is UNMEASURED.
5. Full outer solve produces a cold-verified exact-feasible candidate: **YES at D=4** (Section 5,
   all three L's independently re-verified feasible with CM KKT residuals at/near machine
   precision); **NOT YET at D=20**.
6. Reproducible under checkpoint/resume: **UNTESTED** -- this continuation's D=4 outer solves did
   not exercise `D20Checkpoint`-style resume (not needed at D=4's ~4-6 minute run times); the
   existing checkpoint/resume machinery was not touched or extended to be CM-aware.

**Overall**: the D=4 architecture and its findings are trustworthy and ready to build on; the D=20
production run itself has not been attempted, so criteria 4-6 are informed projections/gaps, not
confirmed results.

## 8. What's exact vs approximate

Everything implemented this continuation is EXACT (no smoothing, no approximation of the
hard-max economics, no change to the CC divergence normalization): the cumulative-CDF moment
block is a literal finite-grid restriction, the orthonormal-contrast rotation is an exact
invertible linear reparameterization (not an approximation), and the inner CC dual solve is the
same exact KNITRO solve as the rest of this codebase. No approximate/smooth basis (brief Section
14) has been attempted.

## 9. What's NOT done / open items (resume here)

- **Not yet combined**: Section 6a's lookup FG kernels + Section 6b's Architecture C Hessian +
  Section 6c's interval basis have each been validated INDEPENDENTLY against the dense reference,
  but never assembled into one single fastest-known CM-augmented bundle. Doing so (interval basis
  + lookup FG + structured Hessian, all together) is the natural next step before any D=20 attempt
  and should be re-equivalence-tested as a combined unit, not assumed to compose correctly just
  because each piece works alone.
- The FAST Lfix/composite-gradient OUTER-loop path (as opposed to the inner CC dual solve
  architectures above) is not yet CM-aware (brief Section 7). `lfix_incremental.jl`'s
  `BaseDualState`/`LFixBaseCache` caching would need to be checked for whether it assumes a fixed
  `obj.d`/moment layout that would need updating for the CM-augmented bundle; not yet audited. The
  D=4 outer solves in Section 5 used the generic ForwardDiff/dense-jac_h path instead (correct but
  slower per outer iteration; safe at D=4, would need `needs_outer_moment_jacobian=false` at D=20
  per the prior memory incident, meaning the Lfix path or an equivalent is likely REQUIRED, not
  just an optimization, before a D=20 CM outer solve is practical).
- The compressed (`:compressed`) moment representation does not support CM columns; the CM block
  is always dense-appended even when the economic block uses the compressed winner-form path.
  Whether this matters for D=20/W=80000 performance is untested.
- Real D=20 data has not been touched at all this continuation.
- The adaptive-grid restart protocol's OUTER-LOOP integration (grow the active set across outer
  KNITRO iterations, restart with a fresh quasi-Newton history) was explicitly out of scope for
  the conditioning subagent (fixed-outer-point activation only) and is unimplemented -- though
  Section 6c's finding (no exploitable quantile-level redundancy at D=4) suggests this may not be
  a high-value direction regardless, pending a check at larger D.
- The L=10/20/50 comparison in Section 5 needs multistart-per-L (not just more single-start
  iterations) before its L-dependence can be trusted; the headline "CM restriction cuts the upper
  bound 9-12%" finding does not depend on resolving this.
- Smooth-basis pilot (brief Section 14) not attempted (optional per the brief).

## Resume commands

```bash
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-d4-c12-common-marginals
source .knitro_env.sh && export JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Reproduce or extend the D=4 delta=1 CM-constrained upper bound:
julia --project=. full_aod_diag/d4_exact/c12_d4_delta1_upper_cm.jl 10 1.0 csw_outer_300.opt
julia --project=. full_aod_diag/d4_exact/c12_d4_delta1_upper_cm.jl 20 1.0 csw_outer_300.opt
julia --project=. full_aod_diag/d4_exact/c12_d4_delta1_upper_cm.jl 50 1.0 csw_outer_300.opt

# Read before starting D=20 work:
#   docs/fullA_cm_interval_lookup_report.md
#   docs/fullA_cm_hessian_architecture_report.md
#   docs/fullA_cm_conditioning_and_adaptive_grid_report.md
# Then: combine interval basis + lookup FG + Architecture C Hessian into one bundle, re-validate
# against the dense reference, THEN adapt context_real_d20.jl::d20_real_setup following context.jl
# ::d4_exact_setup's pattern for the production microbenchmark table (brief Section 15 "First").
```
