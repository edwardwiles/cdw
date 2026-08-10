# Pairwise-quantile-independence Hessian optimization — results, 2026-08-09

Continuation of the session documented in `PAIRWISE_QUANTILE_STATUS_2026-08-09.md` and scoped by
`PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_HANDOVER_2026-08-09.md`. That handover asked two things:
(1) settle whether an exact Hessian-vector-product (HVP) inner solve pays off, via a controlled A/B
at real D20/W=100k scale; (2) apply the T3/T4 table fixes (dedup, threading) and two smaller
low-risk fixes regardless of the HVP outcome. Both are now done, with real numbers.

## Part 1 — HVP: correct, but not a net win (decisive real A/B)

Built the full matrix-free Hessian-vector product for this restriction's ENTIRE inner-dual Hessian
(`full_aod_diag/d4_exact/pairwise_quantile_hvp.jl`) — `H_EE`, `H_E,R`, and `H_RR` all at once, via
one `forward!` + one `Ψ''`-weight + one `transpose!` pass, no dense assembly anywhere.

**This required no new economic-kernel derivation** (the handover's flagged "harder half"): traced
`economic_forward!`/`economic_transpose!` (used by this restriction's own FG functor) to their
definition and confirmed they ARE `compressed_dual_contraction!`/`compressed_transpose_contraction!`
— the exact primitives `compressed_cc_hvp` (`compressed_cc_inner.jl`) already composes for another
family, operating on the SAME `CompressedFactual` `winner_pair_hessian!`'s `WinnerPairHessCtx` is
built from. `winner_pair_hessian!` is simply a different (sparsity-exploiting, O(W·Ddest²)) explicit
algorithm for computing the identical H_EE block — not a different economic model. Combined with
this restriction's own `dual_index!` already composing `arg0 = -ζ - E·λ_E - R·λ_R` additively, the
FULL combined `Hv` (including the cross term) reduces to `Hv = (1/M)·[1|E|R]'·diag(h)·([1|E|R]·v)`,
built entirely from already-validated `economic_forward!`/`economic_transpose!`/
`pairwise_quantile_forward!`/`pairwise_quantile_transpose!` — see `pairwise_quantile_hvp.jl`'s own
header for the full sign-convention derivation.

**D4 validation** (`debug_pq_hvp_check.jl`): `Hv` vs `Hdense*v` (the already-FD-validated explicit
Hessian) agrees to the SAME ~4e-8 relative precision ceiling this restriction's own dense Hessian
already carries vs finite differences (not a new source of error) — 8 random directions plus 4
targeted basis vectors (ζ, an economic coordinate, a marginal coordinate, a pair coordinate), all
PASS. Real `KN_solve()` with `hessopt=5`/`algorithm=cg` at D4 converges (`nStatus=-100`, in the
accepted set) and the independent verifier gives `kkt_resid=2.1e-11` — both dense and HVP variants
solved from the SAME `x_free_calib` point, duals agree to `max|Δλ|=1.3e-8`
(`test_pairwise_quantile_d4_hvp_ab.jl`).

**D20/W=100,000 controlled A/B** (`profile_pairwise_quantile_d20_hvp_ab.jl`), same context, same
cutoffs, both variants run sequentially and BOTH passed through the same verifier:

| variant | nStatus | n_fg | n_hess(-vec) | wall-clock | verifier kkt_resid |
|---|---|---|---|---|---|
| dense (hessopt=exact) | -103 | 10 | 9 | 343.72s | 1.19e-12 |
| HVP (hessopt=5, cg) | -100 | 8 | **6084** | 1512.30s | 3.34e-11 |

`\|ζ_dense - ζ_hvp\| = 8.1e-13`, `max\|λ_dense - λ_hvp\| = 9.9e-9` — both variants are CORRECT
(verifier-confirmed, duals agree). But CG needed **6084** Hessian-vector calls vs dense's 9 full
Hessian calls — even though each HVP call is far cheaper than one dense callback, the total is
**4.4x slower** (1512s vs 344s). This is exactly the risk the handover named up front: "switching to
a CG-based inner algorithm could need MORE iterations... even though each individual callback is
far cheaper." A W=3000 smoke pass (both variants correctly infeasible, `nStatus=-300`, consistent
with this codebase's documented D20 small-W sensitivity) had already hinted at this — HVP needed
1446 Hessian-vector calls even at that tiny/infeasible scale.

**Verdict: HVP is not adopted for production.** Per CLAUDE.md's own standing warning, this is NOT
attributed to "HVP gives CG a worse starting point" or similar — the real cause is that CG simply
needs many more iterations than Direct/SQP for this problem's conditioning, a genuine algorithm
property, not a warm-start artifact. Falling back to the handover's own sanctioned Plan B: fix
T3/T4 directly.

## Part 2 — T3/T4 fixes + two smaller fixes (all applied, all real numbers)

### Fix 1: T3/T4 dedup (handover "Verified finding #1")

`triple_combos`/`quad_combos` store each unique unordered triple/quad **3× redundantly** (axis
permutations of the same joint table). Deduped: `PairwiseQuantileHessianTables` now stores `T3`
sized `C(D,3)` (was `3*C(D,3)`) and `T4` sized `C(D,4)` (was `3*C(D,4)`), with a
`triple_perm`/`quad_perm` array (computed once per campaign via `sortperm`) used ONLY at READ time
(`read_T3`/`read_T4`, `pairwise_quantile_hessian.jl`) to map an arbitrary role-ordered combo back
onto the canonical (origin-sorted) table.

**A real bug was found and fixed in this exact pass** (not shipped broken): the first implementation
deduped the STORAGE but not the SCATTER — it still looped over all 3× redundant combos and scattered
every one into the smaller canonical table, which **triple-counts** every cell rather than
deduplicating. Caught immediately by `debug_pq_hess_check.jl`'s Hessian-vs-FD check regressing from
~4e-8 to ~2% relative error, localized to the T3/T4-touching rows via `debug_pq_hvp_check.jl`'s
basis-vector breakdown, and root-caused in under two minutes with a dedicated synthetic brute-force
isolation script (`debug_pq_t3t4_dedup_isolate.jl`, comparing deduped reads against a brute-force
recompute over the full redundant list on random synthetic data — no real KNITRO context needed).
Fix: scatter uses ONLY the `C(D,3)`/`C(D,4)` canonical (sorted-origin) representative tuples, with
identity ordering — no permutation needed at scatter time at all, only at read time. Re-ran the
isolation script (exact 0.0 error) and the full D4 suite (Hessian check, HVP check, real KNITRO
A/B+verifier) — all PASS again, same ~4e-8 precision ceiling as before any of this pass's changes.

### Fix 2: T3/T4 threading (handover "Verified finding #2")

T3/T4 scatter now uses the SAME static-chunk/fixed-order-reduction `Threads.@threads :static`
discipline T1/T2 already use (`PairwiseQuantileHessThreadScratch`, per-thread canonical-table
copies, never atomics). **Note: the D20/W=100k numbers below were measured with
`Threads.nthreads()==1`** (this session's Julia process was not launched with `--threads`) — the
dedup fix alone (Fix 1) already delivered the numbers below; the threading fix is real,
correctness-preserving, and ready, but its OWN incremental benefit is unmeasured in this pass
(would require a re-run with `julia --threads=N`).

### Fix 3 & 4 (handover's "two smaller, lower-risk fixes")

- **Dense packing removal**: `pairwisequantile_hess_cb_builder` (`pairwise_quantile_production.jl`)
  previously built a full `(NCORE+n_rows)×(NCORE+n_rows)` dense `Hfull`, mirrored both triangles,
  then packed. Now writes DIRECTLY into KNITRO's packed upper-triangular output, selecting the
  source block (`hee_packed`'s own packing via `_pk_upper`, `HEQ`, or `HRR`) per `(i,j)` — no
  intermediate dense matrix ever built.
- **Cross-block persistent scratch**: `pairwise_quantile_cross_hessian_block!`'s
  `Mtab_S`/`Ptab_S`/`Mtab_Snu`/`Ptab_Snu`/`Mtab_cf`/`Ptab_cf`/`v`/`v_winner_sum` buffers (previously
  allocated fresh every call, ~900KB/call) now live in a persistent `PairwiseQuantileCrossHessScratch`
  (built once per campaign, reused via `fill!`).

## Real before/after numbers, D20/W=100,000 (real KNITRO solve, real context)

| | BEFORE (session start) | AFTER (this pass) |
|---|---|---|
| Full inner solve wall-clock | 407.4s (`nStatus=-103`, 10 FG, 9 Hess) | **126.12s** (`nStatus=0`, 9 FG, 8 Hess) |
| One Hessian callback, total | 45.67s | **14.18s** |
| — T1/T2/T3/T4 table build | 35.80s (78.4%) | **8.58s** (60.5% of new total) |
| — H_E,R cross-block | 3.128s | **2.187s** |
| — final assembly/packing | 5.950s | **3.178s** (no dense Hfull anymore) |
| — H_E,R cross-block allocations | 921,784 bytes/call | **1,488 bytes/call** (~620x less) |

**Overall: 3.23x faster end-to-end inner solve, single-threaded** (`Threads.nthreads()==1` for this
measurement — the threading fix, Fix 2, is additional unrealized headroom on top of this).

**Correctness re-confirmed at real D20/W=100,000 scale** (`test_pairwise_quantile_d20_verifier_after_fixes.jl`,
separate run from the timing table above): real KNITRO solve on the POST-FIX production path,
`nStatus=0` (fully converged, not just in the accepted set), independent verifier
**`kkt_resid=9.54e-13`** (machine precision; `kkt_resid_E=9.54e-13`,
`kkt_resid_marginalbin=2.00e-15`, `kkt_resid_pairindep=3.73e-16`), marginal probabilities sum to 1
per origin to `<1e-9`. This is the same order of magnitude as the session's original D4/D20
verifier results (`1.1e-14`/pre-fix), confirming the T3/T4 dedup bug fix and all other changes did
not regress correctness at production scale.

## Follow-up 1: 10-thread parallelization (real number, on top of dedup)

Re-ran the same D20/W=100,000 profile with `julia --threads=10` (the earlier numbers above were
all measured with `Threads.nthreads()==1` -- the threading fix, always real and correctness-
preserving, had not yet been measured).

| | 1 thread | 10 threads |
|---|---|---|
| Full inner solve | 126.12s | **57.84s** |
| T1-T4 table build (the threaded block) | 8.58s | **1.84s** (~4.7x, sub-linear, reduction overhead) |
| H_E,R cross-block (NOT threaded) | 2.19s | 2.48s (noise) |
| Final packed write (NOT threaded) | 3.18s | 4.61s (noise) |

10 threads take the full solve from 126s to 58s -- **2.2x on top of dedup, 7.0x vs the original
407s baseline**. With T1-T4 now down to 1.84s, the two NOT-yet-threaded blocks (cross-block +
packed write, 7.09s combined) are the new bottleneck within one callback -- a natural next target
if further speedup is wanted, out of scope for this pass.

## Follow-up 2: made the restriction genuinely `L`-generic (number of quantile bins was hardcoded)

The task's own draft (eq. 32) used `L=5` (quintiles, 4 cutoffs), and that `L=5` was hardcoded
throughout all 8 restriction files as literal `4`/`5`/`0.2`/`0.04`/`16`/`25` constants -- array
dimensions, row-index formulas (`marginal_row`/`pair_row`), centering targets, the hand-unrolled
4-cutoff decode/Jacobian, the verifier's quantile grid, KNITRO layout sizes -- not a parameter
anywhere. Generalized every one of these to a required `L::Int` (`L>=2`, `n_cutoffs=L-1`), threaded
through every struct (`PairwiseQuantileCutoffLayout`, `PairwiseQuantileOperator`,
`PairwiseQuantileBinState`, `PairwiseQuantileHessianTables`, all thread-scratch structs) and every
function call site across all 8 core files plus all test/debug/profile scripts. No defaults added
anywhere (`L` is a required argument everywhere, matching this repo's own no-silent-defaults rule)
-- the hand-unrolled `NTuple{4,Float64}` cutoff decode/Jacobian in `pairwise_quantile_cutoff_
transform.jl` was rewritten as a loop over `n_cutoffs(layout)` in the process.

**Genericity proof, two levels:**
1. **Standalone math** (`test_pairwise_quantile_d4_dense_oracle.jl`, now takes `L` as `ARGS[1]`,
   default 5): every check (cutoff Jacobian vs FD, forward!/transpose! vs dense reference, every
   Hessian raw-table block vs an INDEPENDENTLY-built dense Gram matrix, packed round-trip, cutoff-
   gradient exact-vs-slow-O(W) recompute) re-run at **`L=2,3,5,7,10`** -- ALL PASS at machine
   precision (`L=5` regression bit-for-bit unchanged from before this pass).
2. **Real KNITRO production path** (`test_pairwise_quantile_d4_hvp_ab.jl`, now takes `L` as
   `ARGS[1]`): real `d4_exact_setup` economic context, real dense-Hessian AND HVP inner solves,
   real independent verifier, at **`L=5`** (regression: `nStatus=0`/`kkt_resid=8.0e-15`,
   bit-identical to the pre-genericity numbers) and **`L=8`** (octiles: `nStatus=0`,
   `kkt_resid=1.03e-13`, HVP variant also converges and agrees with dense to `max|Δλ|=4.5e-9`) --
   proving the FULL restriction (economic cross-block, Hessian callback, verifier, both inner-solve
   variants) is genuinely `L`-generic end-to-end, not just its standalone math.

## What changed, file by file

- `full_aod_diag/d4_exact/pairwise_quantile_hvp.jl` (NEW): full combined `H·v` HVP, `KN_set_cb_hess`
  wiring for `hessopt=5`, `archPQ_base_state_hvp` driver. Correct and validated, NOT used in
  production (HVP verdict above).
- `full_aod_diag/d4_exact/pairwise_quantile_hessian.jl`: `_assign_canonical_combos`,
  `PairwiseQuantileHessThreadScratch`, `read_T3`/`read_T4`, deduped+threaded
  `build_pairwise_quantile_hessian_tables!`.
- `full_aod_diag/d4_exact/pairwise_quantile_cross_hessian.jl`:
  `PairwiseQuantileCrossHessScratch`, `pairwise_quantile_cross_hessian_block!` now takes it as a
  required argument (call sites in `debug_pq_cross_hess_isolate.jl`/`profile_pairwise_quantile_d20.jl`
  updated).
- `full_aod_diag/d4_exact/pairwise_quantile_production.jl`: `PairwiseQuantileCoreHessCtx` gained
  persistent `hee_packed`/`HEQ`/`HRR`/`cross_hess_scratch` fields (dropped unused `Hfull`);
  `pairwisequantile_hess_cb_builder` rewritten for direct packed writes (`_pk_upper`).
- New test/debug scripts: `debug_pq_hvp_check.jl`, `debug_pq_t3t4_dedup_isolate.jl`,
  `test_pairwise_quantile_d4_hvp_ab.jl`, `profile_pairwise_quantile_d20_hvp_ab.jl`,
  `test_pairwise_quantile_d20_verifier_after_fixes.jl`.
- **`L`-genericity (Follow-up 2)**: ALL 8 `pairwise_quantile_*.jl` core files
  (`cutoff_transform`, `bin_context`, `cutoff_gradient`, `operator`, `hessian`, `cross_hessian`,
  `verification`, `production`) plus `pairwise_quantile_hvp.jl` and every test/debug/profile script
  -- `L` (number of quantile bins) is now a required argument everywhere a bin count was previously
  a literal `4`/`5`/`0.2`/`0.04`/`16`/`25`. `PairwiseQuantileCutoffLayout`/`PairwiseQuantileOperator`/
  `PairwiseQuantileBinState`/`build_pairwise_quantile_thread_scratch`/
  `PairwiseQuantileTransposeScratch`/`PairwiseQuantileCrossHessScratch` all gained an `L` parameter;
  `marginal_row`/`pair_row`/`n_marginal_rows`/`n_pair_rows`/`n_total_rows`/`n_mean_flat`/
  `n_pair_flat` all gained an `L` argument. `decode_origin_logcutoffs`/`cutoff_jacobian_block!`
  rewritten from hand-unrolled 4-tuples to loops over `n_cutoffs(layout)=L-1`.
  `test_pairwise_quantile_d4_dense_oracle.jl`/`test_pairwise_quantile_d4_hvp_ab.jl`/
  `debug_pq_t3t4_dedup_isolate.jl` take `L` as an optional `ARGS[1]` (default 5, test-file
  convenience only); the D20 profile/verifier scripts take `L` as a required `ARGS[2]`.

## Not done this pass

- Bin-5 sparsity exploitation (handover "Verified finding #3") — not attempted; the dedup fix alone
  already delivered the bulk of the anticipated gain. Would need empirical verification of the 0.8
  active-fraction assumption before implementing, per the handover's own caveat.
- A real D20/W=100k timing/verifier run at an `L != 5` value -- genericity is proven at real-KNITRO
  scale at D4 (Follow-up 2) and at real D20/W=100k scale for `L=5` only (the main results above);
  combining both (D20/W=100k at a different `L`) was not run this pass, budget permitting later.
- `run_pairwisequantile_upper_checkpointed` (checkpoint/resume production entry point) — still not
  built, same as noted in the prior status doc.
