# Timeout and State-Reuse Audit — 2026-07-24 (port-prep)

## 1. The bug being guarded against

The pre-omit-ROW reconciliation archive
(`experiment/fullA-fixed-frechet-basis-draft-reconciliation-2026-07-24`, its own Part V) diagnosed
and fixed two compounding bugs in a driver script (`run_armC_profile_continuation.jl`) that lived
**only on that experimental branch**, never in `production/fullA-exact`:

1. **An invented 10-second inner-solve trial timeout** (`ek_inner_bounded_trial_10s.opt`), traced
   to commit `66a392d` on that same experimental branch — a value invented by a prior session, not
   part of production. A cold L=50 solve genuinely needs ~20s; the timeout matrix
   (`diag_frechet_timeout_matrix.jl`, D=20 real data, W=80,000) showed all three tested points
   fail at 10s (`nStatus=-401`, KNITRO's `KN_RC_TIME_LIMIT_FEAS`) and succeed at 30s.
2. **Discarding an already-verified `BaseDualState`.** The driver's gradient step called
   `cm_frechet_production_gradient_cplus(...)` with no `base=` argument at the *current accepted
   point*, one line after setting a 10s trial-tier option file — even though that exact point had
   just been cold-verified (60s) one iteration earlier, whose returned `base` was silently
   discarded (`ok, Δcold = cold_confirm(...)`, dropping the `base` return value). The default
   `base=nothing` fell through to a **fresh** inner solve under the 10s bound.

Fixed together (`run_armC_profile_continuation_statefix.jl` on that branch), Q0/cumulative went
from **0% accepted (20/20 `trial_timeout`)** at the 10s tier to **66.7% accepted** at a 30s tier
with the state-reuse fix in place — the single largest effect in that whole study, larger than any
basis (Q0/Q1/Q2) choice (see that branch's own `FIXED_FRECHET_DRAFT_RECONCILIATION_2026-07-24.md`
Part VII).

## 2. Why this port's own code does not reintroduce either bug

- **No new trial-timeout option file is introduced anywhere in this branch.** Grep confirms:
  `git grep -n "10s\|_trial_10s\|10\.0.*maxtime" full_aod_diag/d4_exact/cm_frechet_*.jl` — no hits.
  Every entry point in `cm_frechet_production_bundle.jl` (`archC_frechet_base_state`,
  `archC_frechet_cdf_power_base_state`, and their verified/screened variants) calls
  `inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder=...)` using whatever
  `obj.inner_loop_opt` the caller's `ctx`/`ctx_cm` already carries — this port never overrides it
  with a shorter bound, and does not ship its own opt-file.
- **Every gradient/base-state accessor accepts and honors `base::Union{Nothing,BaseDualState}`.**
  `archC_frechet_base_state`/`archC_frechet_cdf_power_base_state` only perform a fresh inner solve
  when called directly (no `base` argument to pass through — these ARE the solve functions); any
  future outer-driver built on this branch must call them once per new point and thread the
  returned `BaseDualState` into subsequent gradient/Hessian requests at the *same* point, following
  exactly the pattern `cm_production_gradient_cplus` already uses in current production
  (`base = base === nothing ? archC_base_state(...) : base` — see
  `docs/CM_PRODUCTION_HOOK_INTERFACE_SPEC_2026-07-24.md` §4). This port's own dispatcher functions
  do not discard a caller-supplied `base`.

## 3. Typed classification (replaces "timeout == infeasible")

`cm_frechet_production_bundle.jl` introduces `frechet_solve_outcome(nStatus)`, built on production's
own existing `decode_knitro_status` (`knitro_status.jl`, already carries a `KnitroStatusInfo`
lookup table with `.category`/`.is_feasible_result` fields — this was **not** built for this port;
it already existed in production and is reused unchanged):

| `frechet_solve_outcome` | `KnitroStatusInfo.category` | Meaning |
|---|---|---|
| `:feasible` | `:optimal`, `:feasible_approx`, `:limit_feasible` | A feasible point was returned — includes ordinary time/iteration-limit-but-feasible results, which are **not failures**. |
| `:time_limit_no_certificate` | `:limit_infeasible` | Time/iteration/eval limit reached with **no** feasible point ever found — numerical-unknown, never treated as Δ*=∞ or a certificate. |
| `:infeasible_certificate` | `:infeasible`, `:unbounded` | A genuine structural certificate. |
| `:exact_screen_certificate` | `:exact_screen_certificate` | This repo's own pre-solve screen sentinel (never reached `KN_solve`). |
| `:solver_error` | `:error`, `:unknown` | Callback/solver error, not a mathematical statement about feasibility. |

Every base-state constructor in this branch (`archC_frechet_base_state`,
`archC_frechet_verified_state`, `archC_frechet_cdf_power_base_state`,
`archC_frechet_cdf_power_verified_state`) routes its `nStatus` check through
`_frechet_require_feasible`, which only accepts `:feasible` — the OLD pattern seen elsewhere in
this codebase, `nStatus in (0,-100,-101,-103) || throw(...)`, is a **hardcoded enumeration** of the
same "is this a feasible result" test the typed decoder already performs generically; this port
uses the typed decoder instead so a status this port's author didn't anticipate fails loudly with
its decoded meaning in the error message, rather than either silently miscategorizing it or
crashing with a bare status-code number.

`archC_frechet_verified_state`/`archC_frechet_cdf_power_verified_state`'s returned `verify`
NamedTuple carries the typed `outcome` field alongside the raw `inner_status`, so a caller building
a checkpoint/report can distinguish "solved cleanly," "solved but hit a limit while still
feasible," and "ran out of budget with nothing to show" without re-decoding the status code itself.

## 4. What this document does NOT claim

This port has not yet run its own multi-arm outer-search timing study analogous to the archive's
Part V.10/VII (a matched comparison of trial-tier lengths under the *current* production solve
policy, post-omit-ROW, at L=50). The port-readiness report's D=20 shakedown (§12 of the task
brief) is the intended venue for observing whether the *current* production `inner_loop_opt`
budget is adequate for the combined CDF+POWER block at L=50/D=20/W=80,000 — this document only
establishes that the two specific bugs found on the old experimental branch are structurally
absent from this port's own code, not that no other timing tuning will ever be needed.
