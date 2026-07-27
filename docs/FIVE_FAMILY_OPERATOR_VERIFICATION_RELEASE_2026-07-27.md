# Five-Family Operator Verification Release — 2026-07-27

Task §3 asks for one composable `verify_inner_solution_operator!(...)` covering all 5 families,
independently recomputing `r = -zeta*1 - G*lambda`, objective, gradient, KKT residual, and
feasibility classification via the shared economic operator + family-specific restriction
operator + independent scratch, no dense G, with `verification_backend = operator` as the
production default and `skip_cm_fill_ref` removed once operator verification exists.

## Honest status: 2 of 5, unchanged this session

| Family | Operator verification function | Status |
|---|---|---|
| Origin-ZC (ZC-only) | `verify_inner_solution_operator_originzc!` | Inherited proof-of-concept, D=4 gated. NOT wired as production default (`verification_backend` stays `dense_reference`). |
| CM+ZC | `verify_inner_solution_operator_cmmeanzc!` | Inherited (added by `port/finish-operator-stack-...` before this reconciliation), D=4 gated, 3 configs ALL PASS, operator-recomputed KKT residual agrees with the dense verifier's own to ~1e-15. NOT wired as production default. |
| Flexible-CM | none | NOT built, any session. |
| Common-Frechet | none | NOT built, any session. Would additionally require `skip_cm_fill_ref`-toggled dense CM-column fill removed first per the inherited branches' own scope note — moot until the function itself exists. |
| Unrestricted | none | NOT built, any session. Its own FG is already fully operator-based since Addendum Part A, but the VERIFICATION step specifically was never built. |

This session did not extend operator verification to any additional family, and did not wire
`verification_backend = operator` as a production default for the 2 families that already have
one. This is the largest gap between this release and the original task's own ambitions for
§3 — flagged explicitly rather than left implicit.

## Why this wasn't reached this session

Priority order followed the task's own explicit guidance to close well-scoped items with real
evidence over spreading thin, and this session's actual time went to: reconciling three divergent
branches (mechanical but real work, ~59 commits classified), a real crash fix discovered mid-task
(rectangular D-vs-Ddest layout bug across 7 files), generalizing the persistent L-fix cache from
square-only to rectangular (previously undiscovered as a usable building block), wiring 2 more
families onto the shared A-gradient, and — the largest single time investment — root-causing and
fixing a genuine correctness bug in the common-Frechet lookup FG backend that three prior sessions
had only diagnosed, not fixed. Operator verification extension was never reached given that
sequence, not silently skipped in favor of something less important.

## What building the remaining 3 would look like (for the next session)

- **Flexible-CM** is the natural next target (no ZC block, `CMLookupState`'s own forward/backward
  already exists from the FG retrofit, so the component functions a verifier would call already
  exist) — this was already the inherited branches' own stated next priority and remains correct.
- **Common-Frechet** needs the level-anchor block's own independent verification math derived
  (genuinely new, not wiring) plus the `skip_cm_fill_ref` dependency resolved for that family
  specifically (distinct from the Hessian bug this session fixed, which was about a DIFFERENT
  `skip_cm_fill_ref` call site's safety, not about removing the toggle).
- **Unrestricted** needs an operator verification function built from scratch (none exists in any
  form yet, proof-of-concept or otherwise) — likely the smallest lift given the family's FG is
  already fully operator-based, but genuinely new code, not adaptation of an existing pattern.
- Only after all 3 exist: flip `verification_backend = operator` as the production default per
  family, remove `skip_cm_fill_ref` project-wide, and run the full D=4 + real D=20 gate matrix this
  task's §8 asks for across every family.

## Verdict

```text
VERIFICATION_DEFAULT =
    unrestricted:dense_reference (no operator verification exists)
    flexible_cm:dense_reference (no operator verification exists)
    common_frechet:dense_reference (no operator verification exists)
    cm_plus_zc:dense_reference (operator verification exists, inherited proof-of-concept, not wired default)
    zc_only:dense_reference (operator verification exists, inherited proof-of-concept, not wired default)
SKIP_CM_FILL_REF_REMOVED = false (one specific unsafe usage fixed for correctness reasons --
    see COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md -- the general architectural removal
    task §3 asks for, gated on operator verification existing for all families, was not reached)
```
