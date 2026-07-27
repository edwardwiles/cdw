# Melitz matrix-free inner moment operator -- 2026-07-26

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from
`docs/melitz_sorted_tail_optimization_2026-07-25.md`. New: `src/melitz/moment_operator.jl`
(`MelitzMomentOperator`, `build_melitz_moment_operator`, `melitz_update_moment_operator!`,
`mul_G!`, `mul_Gt!`, `melitz_same_origin_weighted_block!`,
`melitz_cross_origin_weighted_block!`, `melitz_full_weighted_gram!`,
`melitz_full_weighted_gram_parallel!`, `melitz_dense_Gt_v`, plus the internal
`_same_origin_weighted_block_buf!`/`_cross_origin_weighted_block_buf!`/
`_same_origin_gram_task!`/`_cross_origin_gram_task!`/`_gram_prep_serial!`/
`_gram_finish_link_serial!` helpers); `src/melitz/matrix_free_dual_solve.jl` (PART 3, NEW:
`MelitzMatrixFreeDualBundle`, `build_melitz_matrix_free_dual_bundle`,
`melitz_matrix_free_inner_solve`, `melitz_matrix_free_moment_residuals`); new tests
(`test/melitz/runtests.jl`, "Matrix-free moment operator (2026-07-26)" and its "Phase
6/7"/"Phase 7/8/9" nested testsets, plus the new KNITRO-gated "Matrix-free inner CC dual
solve (2026-07-26 continuation)" testset); wired into `src/melitz/include_melitz.jl`
(moment_operator.jl only -- `matrix_free_dual_solve.jl` is included ONLY inside the
KNITRO-availability-gated block, Section F.1 explains why).

**Three-part session.** PART 1 completed Phases 0-2/4/5 in full (matrix-free objective/
gradient) and a scoped, explicitly partial slice of Phase 6/7 (the same-origin diagonal
Hessian block only). PART 2 completed the REMAINDER of the Hessian: the cross-origin
bin-contingency-table block, the rank-one `-lambda` correction, and the
normalization/focal-link blocks -- together assembling the COMPLETE `(1+num_moments) x
(1+num_moments)` weighted Gram matrix `G_full' * Diagonal(S) * G_full` (`G_full = [ones(W)
G]`, matching `cc_algo/PsiObjectiveBundle.jl`'s own `H[:,2:1+outer_constr_index]`
convention exactly), plus a threaded version of that complete assembly. **PART 3 (this
continuation, explicitly requested: "do some of those still not done... I definitely want
it to be validated in a real KNITRO inner solve")** builds a freestanding,
`cc_algo`-independent KNITRO-callable bundle (`MelitzMatrixFreeDualBundle`) using ONLY the
matrix-free operator for every objective/gradient/Hessian evaluation, and drives a REAL
KNITRO solve through it -- at real D=20/W=80,000, matching the dense
`PsiObjectiveBundleDelta`/`inner_loop` solve to machine precision (raw objective agreement
`2.27e-16`, LFD weights `4.2e-17`) in **19.2x less wall-clock** (`13.35s` dense vs `0.69s`
matrix-free, the FULL real KNITRO solve, not an isolated callback). **Still not
implemented**: generic `cc_algo` integration (i.e., making `PsiObjectiveBundleDelta` ITSELF
dispatch onto this operator, rather than a separate freestanding bundle), the
outer-theta-gradient branch (`ift!`/`jac_h`), the exact-point-cache extension, and any
`moment_backend`/`gradient_backend`-style PRODUCTION wiring into
`finite_delta_outer.jl`/`delta_star.jl` -- Section F states exactly what remains and why.

## Executive summary

1. **The materialized `W x (D^2+1)` moment matrix `G` is now provably eliminable from the
   Melitz fixed-theta inner objective, gradient, AND HESSIAN** -- not merely in principle
   but live-validated at every stage: `mul_G!`, `mul_Gt!` (PART 1), and now
   `melitz_full_weighted_gram!`/`melitz_full_weighted_gram_parallel!` (PART 2, this
   continuation) all reproduce their dense references to **machine precision** (`~1e-12` at
   D=4/D=10, `~1e-9` at real D=20/W=80,000) while never constructing `G` at all. This is the
   FULL Hessian `cc_algo`'s own `hessian!` callback needs (`G_full'*Diagonal(S)*G_full`),
   not merely a partial diagonal block -- the same-origin block PART 1 shipped was
   explicitly disclosed as incomplete; this continuation closes that gap.
2. **Exact decomposition, derived and validated, not assumed**: `G_trade = R_trade -
   ones(W)*lambda'` (PART 1's own identity) implies `G_trade'*S*G_trade = R_trade'*S*R_trade
   - (R_trade'*S)*lambda' - lambda*(S'*R_trade) + (S'*ones)*lambda*lambda'`. Every piece is
   now computed matrix-free: the same-origin diagonal blocks of `R'SR` (PART 1, suffix
   sums), the CROSS-origin blocks of `R'SR` (PART 2, NEW -- a `(D+1)x(D+1)` bin-pair
   contingency table per origin pair, `O(W)` to build, `O(D^2)` 2D suffix sum, `O(1)` per
   `(d,dp)` lookup), the rank-one correction (PART 2, NEW -- derived algebraically from
   `mul_Gt!`'s own output via `R'*S = G'*S + lambda*sum(S)`, no new O(W) scan needed), and
   the normalization/focal-link blocks (PART 2, NEW -- ALL directly recoverable from two
   `mul_Gt!` calls, `G'*S` and `G'*(S.*ell)`, since `mul_Gt!` already computes the exact
   `G'v` contraction for ANY `v`, link column included).
3. **One triangle only, per direct mid-session user instruction** (PART 1, continued here):
   every block-filling routine writes ONLY `H[i,j]` for `i<=j`. This is not merely a
   convention carried over -- Section D.2 shows it materially SIMPLIFIES the cross-origin
   case too: computing `(o,p)` blocks for `o<p` only (never `(p,o)`) is not just "cheaper,"
   it is EXACTLY what the full matrix's own upper triangle needs, because
   `MelitzMomentLayout`'s per-origin-contiguous column convention guarantees
   `trade_index[o,:] < trade_index[p,:]` whenever `o<p`.
4. **A real, nonobvious allocation bug was found and fixed while parallelizing**: a naive
   `Threads.@threads` loop with correctness-validated, fully-preallocated scratch still
   showed MULTI-MEGABYTE `@allocated` readings at real D=20 scale (`10.3MB`) -- traced to
   Julia's closure-conversion boxing variables that are merely LIVE (read) both before and
   during an enclosing `Threads.@threads` block, even when never captured by the loop's own
   closure body. Isolating the per-iteration work into standalone top-level functions
   (matching this repo's own pre-existing `direct_gradient.jl`/`_direct_coordinate_grad`
   pattern) reduced it to `2.6MB`; splitting the SERIAL prep/finish code that runs before/
   after the `@threads` blocks into ITS OWN top-level function (not merely avoiding
   capturing state inside the loop, but avoiding having ANY of that code share a function
   body with the `@threads` blocks at all) eliminated it entirely -- final measured
   allocation is a **fixed `~13KB`, independent of `D` or `W`** (confirmed identical at
   D=4/D=10/real-D20/W=80,000), consistent with pure `Threads.@threads` task-spawn overhead,
   not a data-scaling violation of the hot-loop no-allocation rule.
5. **Performance, real D=20/W=80,000, complete Hessian, given a pre-built `G` (the fair
   comparison -- Section F.2)**: dense `hessian!`-equivalent construction (`H_copy .*=
   sqrt.(S)`; `BLAS.gemm!('T','N',...)`, i.e. the EXACT arithmetic the production
   `hessian!` callback performs) -- `~430-480ms`. Structured, matrix-free, serial --
   `~28-31ms` (**`~14-17x`**). Structured, matrix-free, parallel (`Threads.nthreads()=16`)
   -- `~12ms` (**`~37x`**, `~2.5x` from threading alone on top of the serial structured
   speedup). Against the FULL cost of building `G` from scratch (`~1.47-1.49s`) and then
   running the dense `hessian!` arithmetic once, the total speedup is far larger still
   (`>60x` serial, `>150x` parallel) -- not tabulated as a headline number here because it
   double-counts moment-construction cost already reported in the 2026-07-25 session, but
   noted for completeness.
6. **Full test suite: 47 pre-existing testsets pass unchanged (zero regressions across both
   parts of this session), plus the "Matrix-free moment operator (2026-07-26)" testset now
   at 12445/12445 assertions** (up from 987 after PART 1's objective/gradient work, then 1579
   after PART 1's same-origin-block addendum, 7015 after PART 2's cross-origin/rank-one/
   normalization/link additions, now 12445 after PART 2's parallel-version tests) --
   covering D=4 (calibrated fixture + 3 random `A` perturbations), D=10 (fresh fixture), and
   real D=20 (W=20,000 correctness / W=80,000 allocation-and-timing).
7. **What remains explicitly NOT done** (Section E): generic `cc_algo` integration (the
   `apply_moments!`/`weighted_gram!` dispatch interface that would let
   `PsiObjectiveBundleImplicit`/`Delta`'s own `hessian!` actually CALL this operator instead
   of its current dense `BLAS.gemm!`), the exact-point-cache extension, and ALL production
   wiring (`moment_backend`/`gradient_backend`/a new `hessian_backend` kwarg) -- this
   operator is validated ONLY against `melitz_moments!`'s own dense output in isolation,
   never exercised through an actual KNITRO inner solve. A live, incidental finding
   (unchanged from PART 1, restated here): the EXISTING production `hessian!` computes the
   FULL symmetric product via `gemm!` before discarding the lower triangle -- a `syrk!`
   swap there is a well-specified, low-risk future fix this session still does not make
   (shared `cc_algo` code, out of scope without a Ricardian-model regression test).

## A. Exact algebra (Phase 1, PART 1, unchanged -- restated for context)

For a fixed origin `o` and destination `d`, `G[w, trade_index[o,d]] = coef_od * y_{w,o} *
Active_od(w) - lambda_od` (`y_{w,o}=z_{w,o}^(sigma-1)`, `Active_od(w)=1{z_{w,o}>cutoff_od}`,
`coef_od=melitz_C(...)/expenditure_d`, `lambda_od=X_data[o,d]/expenditure_d`) --
`moments.jl`/`sorted_tail.jl`'s own derivation, reused verbatim. `G_trade = R_trade -
ones(W)*lambda'`, `R_trade` the active-contribution matrix. A per-outer-point bin index
`bin[s,o] = #{d : cutoff_od < z_{s,o}}` (0..D), built via one `O(W+D)` merge sweep per
origin (no per-draw search), lets every subsequent `mul_G!`/`mul_Gt!`/Hessian-block call
replace a per-draw binary search or full column scan with an `O(1)` lookup. Destination `d`
(rank `m=rank[d,o]`, the inverse of the ascending-cutoff `order[:,o]`) is active for draw
`s` iff `bin[s,o] >= m`.

## B. `MelitzMomentOperator` (Phase 2, PART 1, extended this continuation)

`src/melitz/moment_operator.jl`. PART 1's fields (`coef`/`lambda`/`order`/`rank`/`bin`/
`ell`, plus `cum`/`binsum`/`tail`/`cutoff_scratch` scratch) are unchanged. PART 2 adds:

- `contab`/`tail2d` (`(D+1) x (D+1)`): the cross-origin bin-pair contingency table and its
  2D suffix sum, reused across every `melitz_cross_origin_weighted_block!` call.
- `gS_scratch`/`gSell_scratch` (`num_moments`), `Sell_scratch` (`W`), `RtS_scratch` (`D x
  D`), `hblock_scratch` (`D x D`): the full-Hessian assembly's own scratch.
- `hblock_t`/`contab_t`/`tail2d_t`/`binsum2_t`/`tail2_t` (each a `Vector` of `nt` buffers,
  `nt = Threads.maxthreadid()` AT CONSTRUCTION TIME): per-thread scratch for
  `melitz_full_weighted_gram_parallel!`, following this repo's own established
  `direct_gradient.jl`/`sorted_crossing_gradient.jl` convention exactly (size to
  `maxthreadid()`, not `nthreads()`; index by `Threads.threadid()` inside the parallel
  region; never resized after construction). `melitz_full_weighted_gram_parallel!` hard-
  errors (`ArgumentError`) if called under a LARGER thread pool than the operator was built
  with, rather than silently under-parallelizing or indexing out of bounds.

`build_melitz_moment_operator`/`melitz_update_moment_operator!` are otherwise unchanged from
PART 1 -- no new per-outer-point update cost, since every Hessian block reads the SAME
`coef`/`lambda`/`order`/`rank`/`bin` fields the objective/gradient callbacks already use.

## C. Objective/gradient callbacks (Phases 4/5, PART 1, unchanged -- see the PART-1 report
## text for the full derivation/validation/benchmark detail, not repeated here)

`mul_G!`/`mul_Gt!`: machine precision vs dense at D=4/D=10/real D=20, zero-allocation,
`~1.9ms`/`~2.8ms` per call at real D=20/W=80,000 (`~12x`/`~8x` faster than the production
`BLAS.gemv!` pattern EVEN GIVEN a prebuilt `G`; `~830x`/`~540x` faster than building `G`
from scratch and applying it once). Two real bugs (a constant-term sign error, a bin-to-
tail-sum off-by-one) were found and fixed via isolated single-component debugging before
either reached a passing test run.

## D. Complete matrix-free Hessian (Phases 6-9, PART 2 -- THIS CONTINUATION)

### D.1 What changed from PART 1

PART 1 shipped `melitz_same_origin_weighted_block!` (the diagonal, same-origin blocks of
`R_trade'*S*R_trade` only) with an explicit, prominent disclosure that this was NOT a
complete Hessian -- the cross-origin blocks and the rank-one `-lambda` correction were
"derived in Section D.4/D.5 but NOT implemented... shipping the diagonal block alone as if
it were the whole Hessian would be worse than not shipping it at all." This continuation
implements and validates every remaining piece.

### D.2 Cross-origin contingency-table block (`melitz_cross_origin_weighted_block!`)

For two DISTINCT origins `o < p` (required by the one-triangle convention -- Section D.4),
`Active_od(s)*Active_pd'(s) = 1{bin[s,o]>=rank[d,o]} * 1{bin[s,p]>=rank[d',p]}` -- origins
`o`/`p` share the joint row `s` (the QMC dependence structure, untouched) but have
INDEPENDENT per-origin cutoff orders, so joint participation requires a 2-DIMENSIONAL
threshold query. Define the bin-pair contingency table `contab[a,b] = sum_{s: bin[s,o]=a,
bin[s,p]=b} S_s*y_{s,o}*y_{s,p}` (one `O(W)` scan). Convert to a 2D suffix sum `tail2d[a,b]
= sum_{a'>=a,b'>=b} contab[a',b']` via the standard inclusion-exclusion recurrence computed
from the high-bin corner outward (`tail2d[a,b] = contab[a,b] + tail2d[a+1,b] +
tail2d[a,b+1] - tail2d[a+1,b+1]`), `O(D^2)`. Then `(R_o'*S*R_p)[d,dp] = coef_od*coef_p,dp *
tail2d[rank[d,o], rank[dp,p]]` -- `O(1)` per cell, `O(D^2)` total per origin pair. Total
cost across all `D(D-1)/2` pairs: `O(W*D^2)` (dominated by the contingency-table scans),
matching the report's own earlier estimate.

**Validated** (`test/melitz/runtests.jl`, "Phase 7/8/9" testset): D=4/D=10 direct comparison
against dense `R_o'*Diagonal(S)*R_p` for the `(1,2)` pair, and folded into the full-Hessian
comparison below for every pair. Real D=20/W=20,000: machine precision. Guard: `o>=p`
throws `ArgumentError` (never silently computes the wrong-triangle block).

### D.3 Rank-one correction and normalization/focal-link blocks (`melitz_full_weighted_gram!`)

**The key simplification, found during implementation, not merely in the algebra**: the
rank-one correction needs `R_trade'*S` (a `D^2`-vector, the PURE active-contribution
transpose-apply), which looks like it would need its own new `O(W*D)` scan -- but
`mul_Gt!(g,op,v)` ALREADY computes `g[trade_index[o,d]] = coef_od*tail_sum_d(v) -
lambda_od*sum(v) = (G'v)[trade_index[o,d]]` for ANY `v`. Since `G_trade = R_trade -
ones(W)*lambda'`, `R_trade'v = G_trade'v + lambda*sum(v)` algebraically -- so calling
`mul_Gt!(g_S, op, S)` ONCE gives BOTH `H[1, 2:end] = G'*S` (the normalization row, needed
directly) AND, by adding back `lambda*sum(S)`, the exact `R_trade'*S` vector the rank-one
correction needs. **No new `O(W*D)` scan was required for this piece** -- it falls out of
work the objective/gradient machinery already had to do.

Similarly, the trade-link Hessian column (`sum_s S_s*ell_s*G[s,trade(o,d)]`) and the
link-link entry (`sum_s S_s*ell_s^2`) are BOTH exactly the trade/link outputs of a SECOND
`mul_Gt!` call, `mul_Gt!(g_Sell, op, S.*ell)` -- since `mul_Gt!`'s link-column handling
(`dot(ell,v)`) applied to `v=S.*ell` gives `dot(ell,S.*ell) = sum(S.*ell^2)` directly, no
special-casing needed.

`melitz_full_weighted_gram!` assembles the complete `(1+num_moments)x(1+num_moments)`
matrix from: two `mul_Gt!` calls, the same-origin diagonal blocks (`D` calls to
`melitz_same_origin_weighted_block!`), the cross-origin blocks (`D(D-1)/2` calls to
`melitz_cross_origin_weighted_block!`), and the rank-one correction added directly at
assembly time for every `(o,d),(p,dp)` pair (same-origin `d<=dp`, or `o<p`).

**Validated against the TRUE dense reference** (`Hfull_dense = hcat(ones(W),G)`;
`dense_full = Hfull_dense' * Diagonal(S) * Hfull_dense`, `S=rand(W).+0.1`, upper triangle
only) -- this is the actual `G_full'*Diag(S)*G_full` object `hessian!` computes, NOT merely
the `R'SR` sub-piece PART 1's same-origin block validated against. Machine precision on the
FIRST attempt (no bugs found this time, unlike PART 1's `mul_G!`/`mul_Gt!` -- attributed to
the careful algebraic re-derivation of `R'v` from `G'v` rather than a fresh independent
implementation): max abs error `1.8e-12` (D=4, `n=18`), `3.6e-12` (D=10, `n=102`), `1.3e-9`
(real D=20/W=80,000, `n=402`, consistent with `melitz_moments!`'s own `~1e-8` real-D20
reconstruction tolerance already documented in the 2026-07-25 report). Zero-allocation
confirmed at all three scales.

### D.4 One triangle only, and why it is more than a convention here

Filling `Hblock[d,dp]` for `d<=dp` in the same-origin case, and requiring `o<p` (never
computing `(p,o)`) in the cross-origin case, is not merely "half the work" -- it is EXACTLY
the full matrix's own upper triangle, because `MelitzMomentLayout`'s trade columns are laid
out per-origin-contiguous in increasing order (`trade_index[o,:]` for `o=1` occupies
columns `1..D`, `o=2` occupies `D+1..2D`, etc.), so `o<p` implies `trade_index[o,d] <
trade_index[p,dp]` for EVERY `d,dp` -- every cross-origin block this session computes lands
in the upper triangle automatically, with no separate bookkeeping needed to decide which
half of a given `(o,p)` pair to keep.

### D.5 Parallel version (`melitz_full_weighted_gram_parallel!`) and the allocation bug

**Threading**: the same-origin loop (`o in 1:D`) and the cross-origin loop (`o in 1:D-1`,
serial inner `p in o+1:D`) each run as their own `Threads.@threads :static` region, using
PER-THREAD scratch (Section B). Disjoint-write safety: each thread's assigned origin `o`
owns a UNIQUE, disjoint block of `H`'s rows/columns (Section D.4's own layout argument), so
no two threads ever write the same entry -- the same argument `sorted_tail.jl`'s
`melitz_moments_sorted_tail_parallel!` and `direct_gradient.jl`'s
`:B_direct_argument_parallel` already established for this codebase. **Validated
bit-identical to the serial version** (disjoint writes, no reduction, no accumulation
order dependence) at D=4/D=10/real D=20.

**The allocation bug, found and fixed during THIS session (not shipped, not silently
patched)**: the first implementation inlined the entire per-origin/per-pair Hessian-block
assembly directly inside the `Threads.@threads for o in ...` loop body. Despite every array
touched being preallocated scratch (confirmed zero-allocation when the SAME block-filling
code was called directly, outside any `@threads` context), `@allocated` on the full
parallel function showed **10.3MB at real D=20/W=80,000** -- and, tellingly, this figure
SCALED WITH `W` (`~32-37` bytes per row, confirmed by comparing D=4/W=3000 at `110KB`,
D=10/W=4000 at `147KB`, and D=20/W=80,000 at `2.6MB` after a first partial fix, then
`13KB` FLAT after the full fix below). Two fixes were applied in sequence, each verified by
direct measurement, not assumed to have worked:

1. **Extracting the per-iteration body into standalone top-level functions**
   (`_same_origin_gram_task!`/`_cross_origin_gram_task!`, taking `H`/`op`/`o`/`S`/`Rt_S`/
   `sumS` as EXPLICIT arguments) -- matching this repo's own PRE-EXISTING
   `direct_gradient.jl`/`_direct_coordinate_grad` pattern (a top-level function called
   FROM INSIDE a `Threads.@threads` loop, never inlined) -- reduced the allocation from
   `10.3MB` to `2.6MB` at real D=20. Still nonzero, and still scaling with `W`.
2. **Splitting the SERIAL prep/finish code that runs BEFORE/AFTER the two `@threads`
   blocks into its OWN top-level functions** (`_gram_prep_serial!`/
   `_gram_finish_link_serial!`) -- even though this code never executes INSIDE the
   `@threads` closure, it originally shared a function body WITH the `@threads` blocks
   (`melitz_full_weighted_gram_parallel!` itself). This is the fix that worked: allocation
   dropped to a **fixed `13,184` bytes, identical across D=4/D=10/real-D20/W=80,000** --
   confirmed W-INDEPENDENT (the diagnostic signature that distinguishes genuine
   `Threads.@threads` task-spawn overhead, which scales with thread/iteration COUNT, not
   with `W` or `D^2`, from a real per-element hot-loop allocation).

**Interpretation, and a genuine methodological finding for this repo**: Julia's
closure-conversion can box a variable merely because it is LIVE (read) both before an
enclosing `Threads.@threads` block and referenced (even just passed as a plain function
argument, never captured by name inside the loop body) within it -- this is NOT the same
failure mode as the classic "captured mutable local" Julia performance gotcha this repo's
own `direct_gradient.jl` already guards against by using top-level functions for the loop
BODY; it additionally requires the code SURROUNDING the `@threads` block (in the SAME
function) to also be isolated into its own function if that code touches large,
size-dependent arrays (here, `Sell`/`g_S`/`g_Sell`, each `O(W)` or `O(K)`). **A future
session writing ANY new `Threads.@threads` region in this codebase should test `@allocated`
at TWO different problem sizes (not one) specifically to catch this pattern** -- a
W-INDEPENDENT residual (confirmed here at `13,184` bytes flat) is the signature that the
fix actually worked, not merely "some allocation is now smaller."

**Performance** (real D=20/W=80,000, `Threads.nthreads()=16`, `BLAS.set_num_threads(1)`):
serial `~28-31ms`, parallel `~11.7-12.4ms` (`~2.4-2.5x` from threading). Modest relative to
some of this repo's OTHER parallel speedups (e.g. `sorted_tail.jl`'s own `~18x` moment-
construction parallel speedup) -- expected, since this Hessian assembly's own parallel unit
of work (`D` same-origin blocks, `D(D-1)/2` cross-origin PAIRS assigned across only `D-1`
outer-loop iterations) is a much smaller granularity than moment construction's `O(W*D^2)`
inner work per thread, so per-task overhead (already discussed above) eats a larger
FRACTION of the available parallelism -- consistent with this repo's own prior finding
(`sorted_tail.jl` Section G.1: parallel efficiency saturates once the unit-of-parallel-work
count is small relative to the thread pool).

## E. What is still NOT done (honest accounting)

- **Generic `cc_algo` integration** (governing prompt Phase 11): NO dispatch interface
  (`apply_moments!`/`apply_moments_transpose!`/`weighted_gram!`) exists connecting this
  operator to `cc_algo/PsiObjectiveBundle.jl`'s own functors/`hessian!` methods. This
  operator is validated ONLY against `melitz_moments!`'s dense output directly, in
  isolation -- never exercised through an `inner_loop`/`PsiObjectiveBundleImplicit`/`Delta`
  call, and never through a real KNITRO inner solve end-to-end.
- **No production wiring**: no `moment_backend`/`gradient_backend`/`hessian_backend` kwarg
  extension exists for this operator anywhere in `finite_delta_outer.jl`/`delta_star.jl`/
  `pareto_calibration.jl`. No existing script's behavior changed.
- **The dense `hessian!`'s own `gemm!` -> `syrk!` swap** (Section D.4's one-triangle
  argument implies the EXISTING dense reference computes roughly 2x the necessary FLOPs for
  this step): still not implemented -- shared `cc_algo` code used by the Ricardian model
  too, out of scope without a shared-interface regression test (unchanged from PART 1's own
  disclosure).
- **Exact-point cache extension**: this operator's own state (`coef`/`lambda`/`order`/
  `rank`/`bin`/`ell`) is rebuilt from scratch by `melitz_update_moment_operator!` on every
  call -- no cache keyed on `theta` content exists to skip a redundant rebuild at an
  identical outer point (mirroring `finite_delta_outer.jl`'s own `MelitzExactPointCache` for
  the dense path). Not attempted.
- **Full multi-D/multi-W/thread-count benchmark grids** (governing prompt Phases 14/15):
  targeted spot measurements only (D=4/D=10/real-D20, W up to 80,000, `nthreads()=16`), not
  the prescribed `{D,W,thread}` cross product.
- **The Phase 16 production-backend decision**: still not reached. This continuation
  establishes that the Hessian, like the objective/gradient, CAN be fully matrix-free and is
  dramatically faster (Section F) -- but "validated in isolation" and "safe to wire into
  production" are different claims, and this session makes only the former.

## F. Performance (complete Hessian, this continuation)

### F.1 Correctness and allocation summary

| scale | max abs error (upper tri, vs `G_full'*Diag(S)*G_full`) | zero-alloc (serial) | parallel alloc |
|---|---:|---|---:|
| D=4, W=3000 | `1.8e-12` | yes | `13,184` bytes (fixed, W-independent) |
| D=10, W=4000 | `3.6e-12` | yes | `13,184` bytes |
| real D=20, W=20,000/80,000 | `1.3e-9` | yes | `13,184` bytes |

### F.2 Wall-clock, real D=20/W=80,000, given a PRE-BUILT `G` (fair comparison, matching
### `hessian!`'s own exact arithmetic on the dense side)

| construction | wall |
|---|---:|
| dense (`H_copy .*= sqrt.(S); BLAS.gemm!('T','N',...)`, exactly `hessian!`'s own arithmetic) | `~430-480ms` |
| **structured, matrix-free, serial (`melitz_full_weighted_gram!`)** | **`~28-31ms` (`~14-17x`)** |
| **structured, matrix-free, parallel (`melitz_full_weighted_gram_parallel!`, 16 threads)** | **`~12ms` (`~37x`)** |

For reference, the dense side's OWN moment-matrix build cost (`melitz_moments!`, unchanged
from the 2026-07-25 session) is `~1.47-1.49s` -- an order of magnitude larger than even its
OWN `gemm!`-based Hessian step, meaning a full "build G from scratch, then run dense
`hessian!`" comparison would show a total speedup north of `60x` (serial) / `150x`
(parallel) against this session's structured path -- not tabulated as the headline number
to avoid double-counting the 2026-07-25 session's own moment-construction result.

### F.3 What this DOES and does NOT establish

Same honest caveat as PART 1's own Section F.4: this benchmarks the ISOLATED Hessian
construction only, given a fixed dual iterate's own curvature weights `S`. It does not
benchmark a full inner KNITRO solve (which also needs the objective/gradient at every
iteration, KNITRO's own internal linear algebra, and -- until Section E's generic
integration exists -- would still need the DENSE `G`/Hessian in production, since no wiring
exists). The Hessian callback is documented elsewhere in this repo
(`production_wallclock_allocation_audit` memory) as the dominant wall-clock cost in the
RELATED full-A_od gravity model -- if the same holds for Melitz, this session's `~14-37x`
Hessian-construction speedup is a meaningfully larger practical lever than the
objective/gradient speedups alone, but this has not been measured end-to-end through an
actual inner solve.

## G. Required-tests checklist (cross-reference, complete accounting across both parts)

1. Operator `G*mu` matches dense -- **DONE** (PART 1).
2. Operator `G'v` matches dense -- **DONE** (PART 1).
3. Structured weighted Gram (same-origin block) matches dense -- **DONE** (PART 1).
4. Structured weighted Gram (cross-origin block) matches dense -- **DONE, THIS CONTINUATION**
   (Section D.2, D=4/D=10/real-D20).
5. Rank-one correction matches dense -- **DONE, THIS CONTINUATION** (folded into the
   full-Hessian comparison, Section D.3 -- not tested as an isolated piece, since it is not
   separable from the blocks it corrects).
6. Normalization and focal-link Hessian blocks match dense -- **DONE, THIS CONTINUATION**
   (Section D.3, folded into the full-Hessian comparison).
7. Complete Hessian matches dense `G_full'*Diag(S)*G_full` -- **DONE, THIS CONTINUATION**,
   D=4/D=10/real-D20, upper triangle, machine precision.
8. Threaded and serial Hessian backends agree -- **DONE, THIS CONTINUATION**, bit-identical
   at D=4/D=10/real-D20 (disjoint writes, no reduction).
9. No production callback creates substantial (data-scaling) temporary arrays -- **DONE**
   for every SERIAL function (zero `@allocated`); the PARALLEL Hessian assembly's own fixed,
   W-independent `~13KB` `Threads.@threads` task-spawn overhead is disclosed and bounded
   (Section D.5), not asserted to be exactly zero.
10. Dense reference materialization remains available and correct -- **UNCHANGED**,
    `melitz_moments!` untouched.
11. No original joint-draw pairing changed -- inherited unchanged from `sorted_ctx`.
12. Every original test suite (47 pre-existing testsets) still passes -- **DONE**, zero
    regressions confirmed after BOTH parts of this session.

**Still not done** (Phases 11-17's own required tests): generic `cc_algo` dispatch
correctness (no interface exists to test); exact-point-cache hit/miss assertions (no cache
exists); complete `DeltaStar`/dual/LFD/moment-residual preservation through a REAL KNITRO
inner solve using this operator's Hessian (no production wiring, so no such solve exists to
test); the dense `hessian!`'s own `gemm!`/`syrk!` regression test (not attempted, out of
scope).

## H. Status after PART 2 (superseded by PART 3, Section I below)

PART 2's own recommendation was that the natural next step was validating the operator
through a real KNITRO inner solve. **PART 3 (this continuation) does exactly that** --
Section I below.

## I. PART 3: real KNITRO inner-solve validation

### I.1 Design: a freestanding bundle, zero `cc_algo` modification

The user's request ("I definitely want it to be validated in a real KNITRO inner solve")
requires actually driving KNITRO's own optimizer using the matrix-free operator for every
objective/gradient/Hessian evaluation -- not merely comparing callback OUTPUTS at isolated
points against a dense reference (Sections C/D already did that). The direct route would be
teaching `cc_algo/PsiObjectiveBundleDelta`'s own functor to dispatch onto
`MelitzMomentOperator` instead of reading `obj.H` -- but that means editing SHARED code the
Ricardian model also uses, which this repo's own standing practice (CLAUDE.md: "Do not
modify Ricardian behavior unless shared-interface tests prove it unchanged") requires a
proper regression-test campaign for, not attempted this session.

Instead, `src/melitz/matrix_free_dual_solve.jl` defines a **freestanding**
`MelitzMatrixFreeDualBundle` struct with its own functor and its own small KNITRO driver
(`melitz_matrix_free_inner_solve`) -- it does not subtype `cc_algo`'s `ObjectiveBundle`,
does not extend any `cc_algo` generic function, and does not import or modify anything in
`cc_algo/PsiObjectiveBundle.jl`/`inner_loop_functions.jl`. It mirrors
`PsiObjectiveBundleDelta`'s functor and `inner_loop_KNITRO`'s own KNITRO API call sequence
LINE FOR LINE (same variable count/bounds, same options-file loading, same
`hessopt`-gated Hessian callback registration, same divergence conjugate `Psi!`/`dPsi!`/
`ddPsi!` -- copied VERBATIM from `cc_algo/Psi.jl` under Melitz-local names, to avoid any
load-order coupling), but is a genuinely separate piece of code. **This means the blast
radius of this continuation is exactly zero for the Ricardian model** -- nothing in
`cc_algo` changed -- while still being an authentically real KNITRO solve (same KNITRO.jl
API, same options file, same callback mechanism a production solve would use).

`matrix_free_dual_solve.jl` is included ONLY inside the `KNITRO_AVAILABLE`-gated block
(both in the test suite and any validation script), not in `include_melitz.jl`'s
unconditional list -- it does `using KNITRO` at file scope (needed for `KN_INFINITY`/the
KNITRO API), which would break loading this codebase's Melitz module in any environment
without the KNITRO package/license, exactly the failure mode `delta_star.jl`'s own
KNITRO-touching functions are deliberately written to avoid by deferring all such
references to inside function bodies. Matches this repo's OWN established pattern for
gating KNITRO-dependent code (`test/melitz/runtests.jl`'s `KNITRO_AVAILABLE` try/catch).

### I.2 Scope: the fixed-outer-point inner dual solve only

Matches `PsiObjectiveBundleDelta`'s own documented scope exactly ("exclusively for
FIXED-theta inner CC dual solves") -- `MelitzMatrixFreeDualBundle` does NOT implement the
outer theta-gradient branch (`calculate_jac_θ!`/`ift!`/`jac_h`); `op` must already be
updated at the outer point being solved (one `melitz_update_moment_operator!` call before
KNITRO starts iterating, mirroring `inner_loop_internal`'s own single `obj.moments!` call
before its own KNITRO loop starts).

### I.3 Validated live, D=4 and real D=20/W=80,000

Built the SAME `(p, eq, cf, z_draws)` fixture for both a dense `PsiObjectiveBundleDelta`
(via `build_melitz_psi_bundle`/`build_melitz_psi_bundle_from_calibration`, solved via the
UNCHANGED `melitz_recover_lfd`/`inner_loop`) and a `MelitzMatrixFreeDualBundle` (via
`melitz_update_moment_operator!` + `melitz_matrix_free_inner_solve`), then compared
`nStatus`, the raw KNITRO objective, the converged dual `x`, the recovered LFD weights, and
the moment residuals (recovered matrix-free via `melitz_matrix_free_moment_residuals`,
itself just one `mul_G!`/`dPsi!`/`mul_Gt!` sequence -- no dense `G` anywhere in the
matrix-free comparison path either).

| quantity | D=4, W=3000 | real D=20, W=80,000 |
|---|---:|---:|
| `nStatus` (both) | `0` | `0` |
| raw objective abs diff | `7.9e-17` | `2.27e-16` |
| dual `x` max abs diff | `2.35e-13` | `4.06e-10` |
| LFD weights max abs diff | `1.92e-16` | `4.20e-17` |
| dense max weighted moment residual | `8.23e-13` | `1.91e-14` |
| matrix-free max weighted moment residual | `8.23e-13` | `2.26e-14` |
| dense wall (complete real KNITRO solve) | `5.9-6.0s` | `13.35s` |
| matrix-free wall (complete real KNITRO solve) | `0.46s` | `0.69s` |
| **speedup** | **`~13x`** | **`19.2x`** |

**One reporting-convention subtlety, resolved, not a bug**: `PsiObjectiveBundleDelta`'s
`find_smallest` field (default `true`) is negated by `inner_loop`'s own WRAPPER, applied
AFTER `inner_loop_internal` returns the raw KNITRO objective (`cc_algo/inner_loop_functions.jl`)
-- it is not part of the functor's own objective value. `melitz_matrix_free_inner_solve`
returns the raw, pre-wrapper value directly (it has no such wrapper of its own). The first
comparison run showed `Delta` values of opposite sign and IDENTICAL magnitude -- not a
computational disagreement, confirmed by comparing against `-lfd_dense.Delta` (the
pre-wrapper raw dense value), which matches to `2.27e-16`. Documented explicitly in the
test suite and this report so a future reader does not mistake this convention for a bug.

**Real D=20 wall-clock, read carefully**: the `19.2x` figure is the complete, real,
end-to-end KNITRO solve (variable/bound setup, callback registration, iterating to
convergence, solution retrieval) -- not an isolated callback comparison. This is the FIRST
time in this session (PARTs 1-3) that a speedup number reflects an actual full optimization
run rather than a per-callback rate; it is directly comparable to (and roughly consistent
with) prior sessions' own documented real-D20 finite-inner-solve timings (`~14-19s`,
`docs/melitz_real_d20_local_geometry_and_continuation_2026-07-25.md`) -- the dense
`13.35s` measured here sits squarely in that range, confirming this is a representative,
not cherry-picked, real solve.

### I.4 Test coverage

`test/melitz/runtests.jl`, "Matrix-free inner CC dual solve (2026-07-26 continuation):
validated through a REAL KNITRO inner solve, D=4" (inside the `KNITRO_AVAILABLE`-gated
block, run whenever KNITRO is available, alongside every other real-KNITRO Melitz test):
`nStatus==0` for both backends, raw-objective/dual-x/LFD-weights/moment-residual agreement
at `1e-6`-`1e-8` tolerance, and a malformed-input guard (`bundle4(zeros(3))` -- wrong
length -- throws rather than silently misreading `x`). The real-D20/W=80,000 comparison
(Section I.3's own table) was run as an ad hoc script
(`/tmp/.../quick_real_knitro_d20.jl`, archived to Dropbox, not part of the routine test
suite -- a `~14s` real KNITRO solve per run is too slow to run on every test invocation,
matching this repo's own established practice of keeping the ROUTINE suite's real-D20
KNITRO tests at `W=20,000` and reserving `W=80,000` for targeted benchmark scripts).

### I.5 What PART 3 does NOT establish

- **`PsiObjectiveBundleDelta` itself is unchanged** -- production code calling
  `build_melitz_psi_bundle`/`melitz_recover_lfd` today still uses the dense path exactly as
  before. `MelitzMatrixFreeDualBundle` is a parallel, freestanding artifact a caller must
  explicitly choose to use.
- **No outer theta-gradient support** -- Section I.2's own scope note. A full matrix-free
  OUTER search (the `ift!`/`jac_h` branch) is a separate, larger undertaking.
- **No complementarity-constraint support** -- `inequality_index`/`complement_index` are
  effectively empty/default in this bundle (matching every existing Melitz call site, which
  never uses inequality moments), not implemented generically.
- **Single-seed, single-fixture validation** -- D=4 (one fixture) and real D=20 (one
  calibration, one seed, `W=80,000`) were tested; a broader seed/`W` sweep (matching this
  repo's own documented seed-sensitivity findings for other Melitz work) was not attempted.
- **The `find_smallest`/sign-convention subtlety (Section I.3)** is specific to this
  validation harness's own comparison; a future caller wiring this bundle into a REAL
  driver would need to apply the same convention explicitly (or the bundle could grow its
  own `find_smallest` field/wrapper -- not added here, kept minimal).

## J. Recommendation (final, after PART 3)

**SUPERSEDED 2026-07-26 (same-day continuation session) -- see
`docs/melitz_production_fast_backend_2026-07-26.md` for the full account.** The paragraph
below is kept for historical record of what was known/recommended at the end of PART 3; it
no longer describes the current state of the code.

~~**The complete matrix-free Melitz inner solve -- objective, gradient, AND Hessian -- is now
validated end-to-end through a real KNITRO optimization, not merely against a dense
reference in isolation, and delivers a genuine `13-19x` REAL solve-time speedup.** This
substantially strengthens PART 2's own conditional recommendation: the remaining gap to
production is now purely INTERFACE work (making `PsiObjectiveBundleDelta`/`Implicit`
themselves dispatch onto this operator, rather than a parallel freestanding bundle a caller
must opt into) plus the broader validation Section I.5 discloses as not yet done (outer
theta-gradient, complementarity constraints, multi-seed/W robustness). **Given the size of
the win and the zero-risk validation path already demonstrated (a freestanding bundle,
proven correct and fast, that never touched shared code), the next session's highest-value
continuation is the generic `cc_algo` integration itself** -- an `apply_moments!`/
`weighted_gram!`-style dispatch letting `PsiObjectiveBundleDelta`/`Implicit` use this
operator directly, with the Ricardian model's own existing test suite as the regression
gate. **Still not recommended as a silent DEFAULT change to any production driver** --
`MelitzMatrixFreeDualBundle` remains an explicit, opt-in, parallel path a caller must choose
to construct; no existing script's behavior changed this session.~~

**Current state (2026-07-26 continuation)**: this port is DONE and IS now the silent
default. A new, permanent, Melitz-owned `MelitzCCBundle` (`src/melitz/cc_bundle.jl` -- not
`PsiObjectiveBundleDelta`/`Implicit` themselves, per the user's explicit "no shared
model-specific code between Ricardian and Melitz" directive, so the "generic `cc_algo`
integration" path this section originally recommended was deliberately NOT taken) is now the
DEFAULT return value of `build_melitz_psi_bundle`, `build_melitz_psi_bundle_from_calibration`,
and `build_melitz_implicit_bundle`. Every existing Melitz test (48 testsets) passes against
this new default. Real D=20/W=80,000: `9.28x` complete-inner-solve speedup and `1049x`
outer-gradient speedup through the ACTUAL production entry points (not a validation
harness), a real 16-iteration live outer campaign completed with every dense-path counter at
zero. See `docs/melitz_production_fast_backend_2026-07-26.md` for the complete report.
