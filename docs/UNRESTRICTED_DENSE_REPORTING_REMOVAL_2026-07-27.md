# Unrestricted Dense-Reporting Removal — 2026-07-27

## What changed

`evaluate_fullA_fast_compressed` (`compressed_live.jl`, the unrestricted family's real production
tail under `moment_representation=:compressed`/`:operator`) unconditionally ran a dense-G block on
every call, regardless of `verification_backend`, solely to serve three reporting-only outputs not
consumed by `classify_inner_result`/`is_cacheable_result`/`is_verified_success`:

- `gravity_raw` — via `obj(inner_x, constr=cbuf)`, a full dense constraint evaluation, then read
  `cbuf[2]`;
- `benchmark_unweighted_moment_mean` / `max_abs_moment_resid` — via
  `CS.select_G_from_H(obj, obj.H)`, requiring `obj.H`'s dense columns to be materialized first
  (`materialize_dense_factual_structured!`/`fill_gravity_column!`).

This was already flagged, disclosed, and left unfixed by the prior verification-defaults phase (see
that block's own pre-existing comment, unchanged in provenance) — this task's Goal 9 closes it.

New kwarg `dense_reference_diagnostics::Bool = false` on `evaluate_fullA_fast_compressed` (and
threaded through `evaluate_fullA_fast(...; moment_representation=:compressed, ...)`). At the new
default:

- the dense materialization / `select_G_from_H` / `obj(inner_x, constr=...)` block is skipped
  entirely (`need_dense_block = (verification_backend === :dense_reference) ||
  dense_reference_diagnostics` — still runs unconditionally when `verification_backend ===
  :dense_reference`, since that backend genuinely needs the same dense state for its own
  Delta_dual/Delta_primal/KKT residual, not just for reporting);
- `gravity_raw = NaN`, `benchmark_unweighted_moment_mean = Float64[]`,
  `max_abs_moment_resid = NaN` — same convention the pre-existing `!solved` failure branch in this
  function already used for these fields, not a new pattern.

Pass `dense_reference_diagnostics = true` to restore the full dense reporting values (e.g. for an
explicit debug/diagnostic run, or a test that wants exact dense-vs-compressed field parity).

`K_hard` was investigated and found to be ALREADY dense-G-free (`obj.H_save`, set directly from
`obj.H[1,1]` which `inner_loop_internal_compressed` fills from `θ_full[3+ctx.D] .* SW` — a
compressed-native O(W) computation, never routed through `select_G_from_H`) — no change needed
there; the task brief's own mention of `K_hard` as a target was a naming imprecision, not a real
gap (verified by direct code read, not assumed).

## Correctness: verification-critical fields provably untouched

`Delta_dual`, `Delta_primal`, `mean_m_resid`, `max_abs_moment_kkt_resid`, `weight_norm_resid`,
`m_mean`/`m_min`/`m_max`, `inner_status` are computed entirely inside the
`verification_backend===:operator`/`:dense_reference` dispatch block, which is untouched by this
change — confirmed identical between `dense_reference_diagnostics=false` and `=true` calls at the
same point (test below), not just argued from the code structure.

## Gates (D=4, real KNITRO, both PASS)

**`test_unrestricted_dense_reporting_removal_2026-07-27.jl`** (new, dedicated):
```
=== default call (dense_reference_diagnostics not passed, i.e. false) ===
  gravity_raw = NaN  (expect NaN)
  benchmark_unweighted_moment_mean = Float64[]  (expect empty)
  max_abs_moment_resid = NaN  (expect NaN)
  PASS: reporting-only dense fields are NaN/empty at the new default

=== dense_reference_diagnostics=true ===
  gravity_raw = 5.115329513541167e-17  (expect a real number)
  benchmark_unweighted_moment_mean length = 18  (expect > 0)
  PASS: reporting-only dense fields are real when explicitly requested

PASS: verification-critical fields identical regardless of dense_reference_diagnostics
ALL PASS
```

**`test_compressed_live_integration.jl`** (pre-existing, dense-vs-compressed field-by-field parity
suite — updated to pass `dense_reference_diagnostics=true` in its `compare()` helper, since that
test's whole purpose is exact parity including the reporting fields): re-run in full after this
change, **ALL 25 comparisons PASS** (gamma-profile sweep, 15 random perturbations, cold-start
upper/lower/calibration points, 5-step warm trajectory, tie-injection fallback) — worst-case field
diff `gravity_raw: 2.033e-20`, `max_abs_moment_resid: 1.110e-16`, both at floating-point noise
scale, confirming `dense_reference_diagnostics=true` reproduces the exact pre-change numeric
behavior.

## Not done / left for a future session

- `gravity_val`/`logA`/`gravity_R_sum`/`gravity_R_mean`/`gravity_R_beta` were investigated and
  found to already be fully dense-G-free (computed directly from `θ_full`/`ctx.γ`/`ctx.τ` via
  `aod_pow_matrix`-equivalent algebra, never touching `obj.H`) — no change needed, confirmed by
  direct code read.
