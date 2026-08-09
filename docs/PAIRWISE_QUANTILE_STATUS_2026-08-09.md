# Pairwise-quantile-independence restriction (draft eq. 32) — session status, 2026-08-09

Branch: `prototype/pairwise-quantile-independence-2026-08-09`, forked from
`fix/zc-cmzc-exclude-row-k2-k3-2026-08-07` @ `c22d831` (the fix branch confirmed to contain the
Variant D k=(σ-1) row-omission fix, NOT yet in the stale `production/fullA-exact` ref — see the
branch-choice exchange earlier this session).

Full design rationale, reuse-point citations, and math proofs live in:
- `docs/PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md` (equivalence proofs for the
  marginal and joint conditions, `:all_cross` vs the draft's diagonal-only condition)
- the approved implementation plan (this session's plan-mode output)

## What's done and independently validated (standalone, synthetic draws, no live KNITRO needed)

All of the following passed against dense/brute-force references — see
`full_aod_diag/d4_exact/test_pairwise_quantile_d4_dense_oracle.jl` (run it: `julia -t N
test_pairwise_quantile_d4_dense_oracle.jl` from that directory), currently **15/15 PASS**:

1. **`pairwise_quantile_cutoff_transform.jl`** — softplus-based ordered-cutoff transform (80 raw
   coords at D=20), analytic Jacobian confirmed against finite differences (err ~1e-9).
2. **`pairwise_quantile_bin_context.jl`** — per-origin bin decode (`bin[w,o]::UInt8`), presorted
   draws, T3/T4 combo-lookup registries (`triple_lookup`/`quad_lookup`, plain Int matrices, no
   Dicts). D=20 counts asserted: 80/190/3040/3120.
3. **`pairwise_quantile_operator.jl`** — forward contraction and transpose, both matched a naive
   dense `G*λ` / `-((1/W)G'w)` reference to ~1e-15. Threaded reduction confirmed bit-identical
   across repeated calls (deterministic).
4. **`pairwise_quantile_hessian.jl`** — the hardest, most bug-prone piece: raw table families
   T1/T2/T3/T4, the H_MM/H_MP/H_PP block-fill exploiting exactly the structure the task specifies
   (same-origin diagonal, cross-origin/same-pair/same-origin-in-pair all reuse T2, disjoint-origin/
   shared-origin both reuse T3, fully-disjoint uses T4 only), the centering identity, and packing —
   matched an INDEPENDENTLY-CONSTRUCTED dense reference to ~1e-16 at both D=4 and D=6 (D=6 exercises
   every block sub-case: same-origin, cross-origin, same-pair, MP same-origin both slots, MP
   disjoint, PP shared-origin, PP disjoint).
5. **`pairwise_quantile_cutoff_gradient.jl`** — the fixed-dual boundary-crossing secant method. The
   core `ΔR_w`/crossed-draw-range shortcut matched a slow full O(W) recompute to ~1e-15/1e-16 in
   both directions (an EXACT identity check, not an approximate one — see the file's own note on
   why this statistic is a genuine step function of any single cutoff, making "compare to an
   infinitesimal finite difference" a meaningless test here). Full orchestration (bandwidth
   selection + central/one-sided secant + Jacobian chain-rule mapping to raw coordinates) verified
   against a from-scratch manual replica to exact match.
6. **`pairwise_quantile_verification.jl`**'s probability/residual reporting logic — spot-checked
   against synthetic independent draws (probabilities land at ~0.04/0.2 as expected, max cumulative
   residual ~0.007) and a deliberately-dependent pair (residual correctly jumps to 0.236, proving
   the diagnostic actually detects a real violation, not just always-small noise).

## What's written but NOT independently exercised this session (needs live production context)

7. **`pairwise_quantile_cross_hessian.jl`** (H_E,restriction economic cross-block) — written
   directly against the researched `WinnerPairHessCtx`/`WinnerZCCrossScratch`/
   `winner_pair_cross_hessian_zc_prep!` field contracts (`core_exact_hessian.jl`,
   `winner_pair_cross_hessian.jl`), following `winner_pair_cross_hessian_cm_block!`'s scatter-
   accumulate pattern with the feature accessor swapped for a `state.bin` lookup instead of a
   materialized `Z` column (avoiding the forbidden W×3120 matrix). Syntax-checked only — needs a
   real `WinnerPairHessCtx`/`CompressedFactual` from a live D20 (or D4) economic context to actually
   run and cross-check against a dense reference.
8. **`pairwise_quantile_verification.jl`**'s `verify_inner_solution_operator_pairwisequantile!`
   itself (the KKT-residual half, not the probability-report half already validated above) — needs
   real `economic_forward!`/`economic_transpose!`/`cf`/`econ_ws` and `record_operator_verification!`
   from the production chain to run; its structure mirrors `verify_inner_solution_operator_originzc!`
   exactly (same call shape, same per-block residual convention).
9. **`pairwise_quantile_config.jl`** — self-contained, syntax- and behavior-checked (mode
   resolution, error paths). Note the explicit, disclosed limitation: `:draft_cumulative_diagonal`
   is accepted as a config value but `resolve_pairwise_quantile_mode` errors if selected, rather
   than silently running `:all_cross`'s math under the draft's name — see that file's own docstring
   for why (the draft's condition is a cumulative/block-sum functional, genuinely different from
   the interval-cell moments this prototype builds everywhere else; implementing it correctly is
   flagged as follow-up, not attempted here).

## Not done this session (explicitly out of reach without a live campaign)

- **`run_pairwisequantile_upper_checkpointed`** (the actual `run_*_checkpointed`-style production
  entry point + new checkpoint schema, mirroring `cm_originzc_checkpoint.jl`'s "new schema number,
  own struct, own save/load, zero edits to existing types" pattern). This needs to be threaded
  through `cm_checkpoint.jl`-style ~200-line kwarg surface and real KNITRO callback registration
  (`KN_set_cb_hess` etc.) — writing this blind, without a way to run it, risked producing
  plausible-looking but unverified glue code, which seemed worse than being explicit that it's the
  clear next step. The pattern is fully researched (exact function names/line numbers are in the
  plan's Section 9); wiring it up is mechanical once a live context is available to test against.
- **D4 KNITRO smoke run** through that entry point + the verifier (plan Section "Verification plan"
  item 2) — blocked on the item above.
- **D20/W=100k profiling** (plan Section 11) — blocked on the checkpoint entry point existing and
  a live campaign run; this was always meant to be the LAST step, informing whether/how to optimize
  the `H_PP`-disjoint 4-way table build (flagged throughout the code as the most likely wall-clock
  bottleneck, exactly as the task anticipated).
- Final D20 counts are asserted mechanically (`assert_pairwise_quantile_d20_counts`,
  `pairwise_quantile_verification.jl`) but have not been exercised against a real D20 dataset —
  they're pure arithmetic (`4*D`, `C(D,2)`, etc.) so this is low-risk, but flagging it as unexercised
  rather than claiming it as "run and confirmed."

## Recommended next steps, in order

1. Get a live D4 (or small-D) `CompressedFactual`/economic context (e.g. via `d4_exact_setup` or
   `context_real_d20.jl`'s D4 analog) and exercise `pairwise_quantile_cross_hessian_block!` against
   a dense reference the same way the restriction-only Hessian was validated here.
2. Write `run_pairwisequantile_upper_checkpointed` (`pairwise_quantile_checkpoint.jl`, not yet
   created) following `cm_originzc_checkpoint.jl`'s exact pattern, wiring in `PairwiseQuantileConfig`,
   the cutoff-coordinate layout as new outer-coordinate registrations, and
   `pairwise_quantile_cross_hessian_block!`/`fill_pairwise_quantile_hessian_raw!` into a real KNITRO
   `hess_cb_builder` closure.
3. Run the D4 KNITRO smoke test, then the D20/W=100k profiling campaign per plan Section 11.
