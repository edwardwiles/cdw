# Restricted-Family Dual-Bank Warm-Start Release — Phase D — 2026-07-26

**State: MATCHED_AB_PASSED** (not merged to production/fullA-exact; on
`port/remediate-production-5x7-audit-2026-07-26`).

## What was found

`DualBank`/`record_success!` (`dual_bank.jl`) are generic (a plain history of `(eval_id, zfree,
x_solved)` tuples) and reusable as-is. But `select_warm_start`/`cheap_score` — the actual
CANDIDATE-SCORING logic — are unrestricted-family-specific: `cheap_score` calls
`compressed_cc_value_grad(...; cf::CompressedFactual, ...)`, which scores only the ECONOMIC-CORE
columns of a candidate dual and has no concept of a CM threshold grid or a mean/pair restriction
block. Calling it unmodified against a restricted family's WIDER inner-solve vector
(`x = [zeta; lambda_core; lambda_restriction]`) is not a safe drop-in — either a dimension
mismatch or a silently-incomplete score that ignores the restriction columns' own contribution to
solve quality.

## Scope decision: distance-only selection, not KKT-proxy scoring

Building a restriction-aware analogue of `cheap_score` would require new per-family
gradient-evaluation kernels — comparable in scope to Phase B1's lookup-FG work — and was judged
out of scope for this pass (consistent with the user's own earlier direction on Phase B1: ship the
tractable, well-scoped item now, flag genuinely new-kernel-development work as a separate
follow-on rather than silently absorbing it).

Instead, `cm_dual_bank_production.jl` implements a simpler, still scientifically sound policy:
among the bank's history plus the current `obj.x` "actual" slot, select whichever solved dual's
own ECONOMIC coordinate (`x_free`, extended with the restriction-parameter vector `nu` for
CM+ZC/origin-ZC) is nearest, by scaled Euclidean distance, to the target point. This is a real,
functioning warm-start bank — physically motivated (nearby points in parameter space tend to have
nearby optimal duals) — just without the unrestricted family's own richer KKT-based re-ranking.
Falls back to the neutral (zero) start only when the bank is empty and `obj.x` is not itself a
valid finite start (`norm(obj.x) < 1e6`, the same guard `inner_loop_initial_values` already uses).

## What was built and wired

- `RestrictedDualBank` (wraps a plain `DualBank` plus parallel `x_free` history for distance
  comparison), `select_warm_start_restricted`, `record_success_restricted!`.
- Public counters (task-required): `RestrictedDualBankCounters` (`queries`, `hits`, `misses`,
  `warm_inner_solves`, `cold_inner_solves`, `selected_distance_sum` -> mean, `warm_start_failures`
  — incremented if a warm-started solve comes back non-feasible), module-level
  `RESTRICTED_DUAL_BANK_COUNTERS` Ref, `print_restricted_dual_bank_counters()`.
- Wired via new optional `dual_bank`/`eval_id` kwargs threaded through the full call chain for
  **all four families**: `archC_base_state`/`archC_verified_state` (plain flexible CM),
  `archC_meanzc_verified_state` (CM+ZC, distance computed on `vcat(x_free, nu)`),
  `archC_frechet_verified_state` (common-Fréchet), `archOZ_verified_state` (origin-ZC, distance on
  `vcat(x_free, nu)`) — plus every intervening `*_screened` wrapper (`archC_verified_state_screened`,
  `archC_frechet_verified_state_screened`, and the four `cm_*_production_value_verified_screened`
  functions) updated to pass the kwargs through unchanged rather than dropping them.
- New `use_dual_bank::Bool=true, dual_bank_size::Int=8` kwargs on both `run_cm_upper_checkpointed`
  and `run_originzc_upper_checkpointed`; a fresh `RestrictedDualBank` built once per driver call;
  `cb_F!` in both drivers now passes `dual_bank=dual_bank, eval_id=n_eval[]` through to the value
  computation (only reached on an exact-cache miss — a cache hit correctly skips both the inner
  solve and any warm-start selection, since nothing needs solving).
- Existing per-family Hessian math (`archC_hess_cb_builder`, `archC_frechet_hess_cb_builder`,
  `_originzc_hess_cb_builder`) is completely untouched — the bank only ever overwrites `obj.x`
  before the inner solve and reads the converged `inner_x` after.

## Correctness gate (real, D=4, ALL PASS)

`test_phaseD_dual_bank_correctness.jl`, through the real `archC_verified_state` call path:

- First call (bank empty, fresh `obj.x`): solves feasibly, correctly recorded as **cold**
  (neutral start), bank grows to one entry. **PASS.**
- Second call at a nearby perturbed point: solves feasibly, correctly recorded as **warm** (the
  bank's one entry selected as the start), bank grows to two entries. **PASS.**
- Bank-on vs. a completely independent bank-off run at the SAME two points: `Delta_dual` agrees
  exactly (`diff=0.0` at both points) — confirms warm-starting changes only *how* KNITRO reaches
  the solution, never *what* it converges to. **PASS.**
- Final counters: `queries=2 hits=1 misses=1 warm_inner_solves=1 cold_inner_solves=1
  warm_start_failures=0 mean_selected_distance=0.114448` — matches the test's own call sequence
  exactly.

## Not yet done (tracked separately)

- A live D=20/full-KNITRO-driver confirmation (same status as Phase C — planned alongside Phase
  I's 300s profiles, which will report real bank hit rates and warm/cold solve time differences
  from an actual outer trajectory rather than a 2-point synthetic gate).
- The KKT-proxy-scored selection policy the unrestricted family's own `DualBank` uses remains a
  legitimate, larger follow-on if the distance-only policy's real-world hit rate/benefit (to be
  measured in Phase I) turns out to leave meaningful value on the table.
