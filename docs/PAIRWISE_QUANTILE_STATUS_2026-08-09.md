# Pairwise-quantile-independence restriction (draft eq. 32) — session status, 2026-08-09

Branch: `prototype/pairwise-quantile-independence-2026-08-09`, forked from
`fix/zc-cmzc-exclude-row-k2-k3-2026-08-07` @ `c22d831` (confirmed to contain the Variant D
k=(σ-1) row-omission fix, NOT yet in the stale `production/fullA-exact` ref).

Full design rationale, reuse-point citations, and math proofs live in
`docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md` and the approved implementation
plan (this session's plan-mode output).

**Update (later in this session): the restriction now runs end-to-end through a REAL KNITRO inner
dual solve against the real D4 economic context** (`d4_exact_setup`), not just standalone
synthetic unit tests. This required building the actual KNITRO wiring
(`pairwise_quantile_production.jl`, new this pass) — `OperatorPsiBundle`/`prime_operator!` (true
no-dense-H bundle), a callable FG state mirroring `OriginZCOperatorState`, and a real
`KN_set_cb_hess` Hessian callback combining H_EE (unchanged shared backend) + H_E,R (this
restriction's economic cross-block) + H_MM/MP/PP (this restriction's own block). Getting this
running surfaced and fixed **three real integration bugs** that the earlier standalone/synthetic
testing could not have caught (see "Bugs found via real KNITRO testing" below) — this is exactly
why the user was right to push back on stopping at "no live KNITRO context."

## Fully validated end-to-end (real D4 KNITRO solve, real economic context)

Run in sequence, each building on the last (all in `full_aod_diag/d4_exact/`):

1. **`test_pairwise_quantile_d4_dense_oracle.jl`** — 15/15 PASS, standalone synthetic-draw
   correctness of the restriction's own math (cutoff transform + Jacobian, forward, transpose,
   every Hessian block family, packing, cutoff-gradient shortcut) against dense/brute-force
   references. Unchanged from the first pass of this session.
2. **`debug_pq_fg_check.jl`** — real `d4_exact_setup` context, real `cf_build`/`prime_operator!`,
   real `economic_forward!`/`economic_transpose!`. FG functor's analytic gradient matched finite
   differences to **2.9e-12 across all 130 real coordinates** (18 economic + 112 restriction).
3. **`debug_pq_cross_hess_isolate.jl`** — isolated the H_E,R cross-block (economic × restriction)
   against finite differences of the real gradient, column by column. Caught and fixed two real
   bugs (below); final state: max error **~4e-8** across every tested column.
4. **`debug_pq_hess_check.jl`** — full 130×130 packed Hessian (H_EE + H_E,R + H_MM/MP/PP together)
   vs. finite differences of the real gradient: relative error **~3.9e-8**.
5. **`test_pairwise_quantile_real_d4_knitro.jl`** — real `KN_solve()` through the actual KNITRO
   C library (this host, `demand.mit.edu`, is the licensed one — confirmed live). **`nStatus=0`**
   (converged), 5 FG calls, 4 Hessian calls.
6. **`test_pairwise_quantile_real_d4_verifier.jl`** — real solve + independent
   `verify_inner_solution_operator_pairwisequantile!` recompute from fresh scratch. **KKT residual
   1.1e-14** (machine precision). Marginal bin probabilities land at 0.1999–0.2001 (target 0.2)
   across all 4 origins; `sum(marginal_prob[o,:]) == 1` to 1e-9 for every origin.

## Bugs found via real KNITRO testing (none were, or could have been, caught by synthetic-draw
unit tests alone — this is the concrete payoff of insisting on the live context)

1. **`cf_build`/`prime_operator!` corrupted by passing the wrong `ctx`.** `archPQ_base_state`
   originally called `prime_operator!(obj, θ_econ0, ctx_cm, ...)` using the MERGED `ctx_cm` (whose
   `.obj` field had been overwritten with the restriction-augmented `OperatorPsiBundle`).
   `cf_build` reads dimensionality off `ctx.obj` internally, so this silently corrupted `cf.oci`
   (17 → 129). Fixed by threading the ORIGINAL, unaugmented `ctx` through separately (mirrors
   origin-ZC's own `octx.econ_ctx` field, which exists for exactly this reason — confirmed by
   reading `cm_originzc_production.jl` after hitting this).
2. **Cross-block row 1 / NuZ correction used raw, uncentered feature sums.**
   `winner_pair_cross_hessian_zc_block!`'s own `Z` argument is documented as "already centered"
   (`x - t`); this restriction's raw bin-indicator lookups are never centered by construction, so
   the `- t*(sum of weights)` step (already used in `pairwise_quantile_transpose!`) had to be
   added explicitly for row 1 and the `NuZ` correction. Caught because row 1 was off by ~2 orders
   of magnitude in the isolated FD check.
3. **Bilateral (winner-slot) block needed a PER-SLOT centering correction, not a global one.**
   Each bilateral row only sums over the draws whose actual winner matches that row's slot — a
   different subset per row — so its centering term is `t * v_winner_sum[j]` (accumulated in the
   same winner-loop pass), not `t * sum(v)`. Fixed by adding a `v_winner_sum` accumulator.
4. **Marginal-cell row-index convention mismatch between the Hessian code and the FG functor.**
   `marginal_row(o,a)=(o-1)*4+a` (an O-MAJOR flat layout, self-consistently used and validated
   throughout `pairwise_quantile_hessian.jl`'s own D4 dense-oracle test) does NOT match Julia's
   `reshape(x_slice, D, 4)` column-major (A-MAJOR) convention used when `dual_index!`/the FG
   functor slice λ_M out of the real KNITRO solution vector. This silently swapped two marginal
   dual coordinates' Hessian entries. Fixed by reshaping as `reshape(x_slice, 4, D)'` instead
   (matches `marginal_row` exactly). The pair-cell layout (`pair_row`) needed no fix — its
   `(b-1)*4+a` local ordering already matches `reshape(v,4,4,npair)`'s natural column-major layout.

None of these bugs were in the restriction's own core math (forward/transpose/H_MM/MP/PP/cutoff-
gradient) — all four were integration-layer bugs at the boundary between this restriction and the
existing production KNITRO/economic-context wiring, which is exactly the category of bug a
synthetic standalone test cannot see.

## D20/W=100,000 profiling

Ran at D=20/W=3,000 first (fast sanity check: context build 88s, Hessian sub-block timing
collected for one real callback — `H_EE` 0.012s, `T1/T2/T3/T4` table build 0.75s, `H_MM/MP/PP`
fill 0.47s, centering 0.07s, `H_E,R` cross-block 0.09s; `nStatus=-300` at this W, consistent with
this codebase's own well-documented D20 small-W sensitivity, not a bug — see CLAUDE.md/memory on
checking W-sensitivity before treating a D20 KNITRO failure as a real bug). A W=100,000 run was
then launched; see the session's final message for its outcome (may have completed after this
document was written — check the D20 profiling report file if present, or the raw log for the
authoritative numbers rather than trusting this paragraph if it looks stale).

## Not done this session

- **`run_pairwisequantile_upper_checkpointed`** (the full `cm_originzc_checkpoint.jl`-style
  production entry point with checkpoint schema/resume, ~200-kwarg surface). The functional
  equivalent (`archPQ_base_state` + `PairwiseQuantileCoreHessCtx` + `build_pairwise_quantile_
  augmented_obj`, all in `pairwise_quantile_production.jl`) is built, real-KNITRO-tested, and
  correct — what's missing is checkpoint persistence/resume and the wide kwarg-compatibility
  surface real production campaigns expect, not correctness.
- `:draft_cumulative_diagonal` mode is still an explicit not-implemented placeholder (see
  `pairwise_quantile_config.jl`) — same reasoning as the first pass of this session.

## Recommended next steps

1. If the W=100k profiling run didn't complete in this session, rerun
   `profile_pairwise_quantile_d20.jl 100000` (takes a raw `W` positional arg, no silent default)
   and confirm the `H_PP`-disjoint 4-way table build is or isn't the dominant cost at real scale.
2. Build `pairwise_quantile_checkpoint.jl` (checkpoint schema + `run_pairwisequantile_upper_
   checkpointed`) using `archPQ_base_state` as the validated inner-solve core.
3. Implement `:draft_cumulative_diagonal` for real if replication against the literal draft
   equation is ever needed (currently out of scope, explicitly not the scientific default).
