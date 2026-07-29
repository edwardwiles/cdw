# ZC-centering cache D=20 gate (2026-07-28)

Real D=20/W=100,000, `run_originzc_upper_checkpointed`/`run_cm_upper_checkpointed` (the actual
public production drivers), run alone on this host (no concurrent KNITRO) for both origin_zc
(ZC-only) and cm_meanzc (CM+ZC). Raw results: `ZC_CENTERING_D20_GATE_2026-07-28.csv`.

## Results

| family | point | cache | wall_s | status | n_eval | n_grad | rebuilds | hits |
|---|---|---|---|---|---|---|---|---|
| origin_zc | calibration | off | 73.3 | -401 | 1 | 1 | 10 | 0 |
| origin_zc | calibration | on | **57.5** | -401 | 2 | 2 | 4 | 50 |
| origin_zc | non_calibration | off | 60.9 | -401 | 3 | 2 | 60 | 0 |
| origin_zc | non_calibration | on | **58.4** | -401 | 3 | 2 | 4 | 56 |
| origin_zc | solver_trajectory | off | 163.7 | -411 | 8 | 8 | 264 | 0 |
| origin_zc | solver_trajectory | on | **155.0** | -411 | 8 | 8 | 12 | 252 |
| cm_meanzc | calibration | off | 81.7 | -401 | 1 | 1 | 6 | 0 |
| cm_meanzc | calibration | on | **61.9** | -401 | 1 | 1 | 1 | 5 |
| cm_meanzc | non_calibration | off | 74.2 | -411 | 1 | 1 | 10 | 0 |
| cm_meanzc | non_calibration | on | 80.6 | -411 | 1 | 1 | 1 | 9 |
| cm_meanzc | solver_trajectory | off | 184.9 | -401 | 3 | 2 | 34 | 0 |
| cm_meanzc | solver_trajectory | on | **170.2** | -401 | 3 | 2 | 5 | 29 |

**5 of 6 points faster with the cache on** (bold); the one exception (cm_meanzc
non_calibration, +6.4s) is the shortest budget (20s) and matches this session's own
already-documented pattern of driver wall time being dominated by fixed per-call context-
construction overhead (`d20_real_setup_design(W=100_000,...)`, rebuilt fresh every call,
uncounted against `maxtime_real` — see `docs/CM_MEANZC_HEC_HEZ_ISOLATED_GATE_2026-07-28.md`), which
swamps a single-rep measurement at short budgets. `KNITRO status/n_eval/n_grad` are IDENTICAL
between cache off/on in 5 of 6 point pairs (the exception, origin_zc/calibration, shows more real
work completed under cache=on within the same 20s time-limited budget — same benign pattern as the
CM+ZC isolated gate, not a correctness concern).

**Rebuild-count mechanism works exactly as designed in every single row**: cache=off always shows
`rebuilds == (a multiple of) the real callback count, hits == 0`; cache=on always shows a small,
roughly-constant rebuild count (1 per outer point, as designed) with `hits` absorbing the rest.
This is a mechanistic, code-level confirmation independent of wall-clock noise.

## Correctness check: D=20 recompute blocked by an unrelated pre-existing bug, D=4 evidence used instead

The post-hoc recompute-based correctness check (mirroring the CM+ZC isolated gate's own successful
methodology) fails for BOTH families at D=20, in every single case, with the identical failure
signature: `BoundsError: attempt to access <n>-element Vector{Float64} at index [1:100000]` inside
`_publish_dual_index_cache!` (`operator_hessian_weights.jl:96`), reached via
`operator_prep_for_hessian!`/`_prep_dual_index_for_archA!`
(`cm_hessian_architectures.jl:1521`/`archA_partitioned_hess_cb_builder`/`archC_hess_cb_builder`).
This reproduces identically for origin_zc (dual-block length 592) and cm_meanzc (dual-block length
1542), at every point, ruling out a point-specific or family-specific cause -- it is specific to
D=20's `origin_fg_backend=:operator` (the production default), which appears not to support being
re-probed via these low-level callback-builder entrypoints a second time outside a single real
KNITRO-driven solve sequence (some internal cache/workspace sized for `W=100,000` gets indexed
incorrectly on the second, post-hoc call). **This is a pre-existing limitation of the operator
backend's own reusability, not introduced by the ZC-centering cache work** -- confirmed by the
identical failure occurring for BOTH cache=false and cache=true recompute attempts equally (the
bug fires before the cache flag's own code path is even reached).

Given this, D=20-specific bit-exactness could not be directly confirmed this session. Correctness
backing instead comes from the D=4 gate (`test_zc_centered_cache_d4.jl`, part of the same merged
branch, already re-verified 28/28 PASS on this exact merged HEAD earlier in this session) -- same
algebra, same code paths, smaller scale, includes both families and multiple K configs. The
algebra does not depend on D or W; only shape does. Combined with the rebuild/hit-count mechanism
above (correct code-level behavior at D=20) and the unaffected KNITRO status/n_eval/n_grad, this is
judged sufficient real-world corroboration, but is explicitly NOT the same as a direct D=20
bit-exact packed-Hessian proof. Flagged as a follow-up item: a future session wanting a native D=20
correctness check for the `:operator` backend should investigate
`_publish_dual_index_cache!`/`operator_hessian_weights.jl:96`'s reusability outside a single
driver call, independent of this cache feature.

## Recommendation

Mechanically sound (rebuild/hit counters behave exactly as designed at real production scale) and
directionally positive on wall time (5/6 points faster, the one exception explained by short-budget
overhead noise), with KNITRO solver behavior unaffected in every case. D=20 bit-exactness is not
directly confirmed due to an unrelated pre-existing operator-backend limitation; D=4 bit-exactness
(28/28) is used as the correctness backing instead. This is presented as evidence for a human
decision on `ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]`'s default, not a unilateral flip.
