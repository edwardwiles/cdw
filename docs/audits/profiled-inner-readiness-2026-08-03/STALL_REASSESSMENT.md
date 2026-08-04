# REDUCED inner-stall reassessment — post-fix — 2026-08-03

## unrestricted

Sentinel panel at real production dims (D=20, Ddest=19, W=100,000), run after this branch's
`lower_limit`/method-collision fix (`54b0a94`):
`full_aod_diag/d4_exact/test_unrestricted_stall_sentinel_w100k_2026-08-03.jl`.

| Point | wall | nStatus | KKT resid |
|---|---|---|---|
| calibration (feasible) | 7.14s | 0 | 2.010e-13 |
| small perturbation (feasible) | 1.56s | 0 | 3.408e-13 |
| adversarial A (large uniform A_free shift) | 1.33s | -300 | 1.881e+04 |
| adversarial B (large alternating-sign A_free shift) | 1.34s | -300 | 2.331e+04 |

Both adversarial points reject in ~1.3s via the correct native `nStatus=-300` (KN_RC_UNBOUNDED),
consistent with the salvage package's own measured post-fix timing (~7.0-7.2s at W=100,000 for a
*different*, historically-captured point) and a dramatic improvement over the pre-fix behavior
(genuinely-unbounded points burning the full iteration budget for a slow `nStatus=-400` exit,
measured at 77-90s at this same W by the salvage package before this fix existed anywhere).
These are freshly-constructed adversarial points, not a byte-for-byte replay of a specific
historically-captured one (that capture lives only in an uncommitted Dropbox archive from a prior
session, not pulled this pass) — the substantive claim under test is that the clamp mechanism
itself now fires correctly at real production scale, which it does.

`REDUCED_INNER_STALL (unrestricted) = resolved`.

## flexible_CM ("eval18")

Already fully diagnosed as **not a defect** in a prior session (memory
`eval18-forensic-verdict-genuinely-unbounded-2026-08-02`, not re-derived this pass): a genuinely
unbounded/infeasible D20/W=100,000 outer trial point, where the Hessian's condition number
explodes from ~3e5 at cold start to ~1e10-3.8e11 along the trajectory, driving the objective past
`lower_limit=-50` only very slowly (crossing at iteration 299 when `maxit` was raised from 100 to
1000, correctly firing `nStatus=-300`). This is a **separate** issue from the method-collision bug
fixed on this branch — that bug meant the clamp could never fire at *any* iteration count
regardless of conditioning; eval18's own symptom is that a working clamp is just slow to reach
for this specific, badly-conditioned point. Confirmed distinct: the fix landed in this session
targets the missing-clamp defect (now closed for all 5 families' bundle constructors); eval18's
own conditioning-driven slowness is a real, known, already-documented property of this specific
family+point combination, not something this session's fix could or should have changed.

`REDUCED_INNER_STALL (flexible_CM / eval18) = resolved_as_diagnosis (not a defect; pre-existing
finding re-confirmed, not re-derived this pass)`.

## Escalation not required

Per task §9's own branching instruction ("if no residual stall remains, record the evidence; if a
residual stall remains, perform a bounded escalation on unrestricted only"): no residual stall was
found for unrestricted after this session's fix, so the bounded-escalation program (per-iterate
capture, callback freshness test, FD gradient/Hessian-vector-product cross-checks, exact/FD ×
quasi-Newton solver arm matrix) was not needed and was not run.
