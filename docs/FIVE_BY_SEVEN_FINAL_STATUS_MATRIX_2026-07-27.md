# Five-by-Seven Final Status Matrix — 2026-07-27 (continuation session)

Columns: economic moment builder, economic FG default, restriction FG default, verification
default, A-gradient default, H_EE coverage, H_ER status. (7 columns x 5 families.) This is this
SESSION's own honest status, not a claim that Phase C's full gate matrix was run -- see
`FINAL_OPERATOR_STACK_RELEASE_MASTER_REPORT_2026-07-27.md` for why Phase C was not reached.

| Family | Economic moment builder | Economic FG default | Restriction FG default | Verification default | A-gradient default | H_EE | H_ER |
|---|---|---|---|---|---|---|---|
| Unrestricted | compressed_operator | compressed_operator | n/a | dense_reference (operator verifier NOW EXISTS, not default) | composite_gradient_at_fast_buffered (`:shared` opt-in, real-driver-gated this session) | shared winner-pair | n/a (no restriction block) |
| Flexible CM | shared_economic_operator | shared_economic_operator (cm_lookup) | cm_lookup | dense_reference (operator verifier NOW EXISTS, not default) | shared_inplace_pooled (economic_A_gradient!) | shared winner-pair | primitive_built_and_gated, NOT wired (D=4 8/8 + D=20/L=50 2/2 PASS, machine precision) |
| Common Frechet | shared_economic_operator available, dense_reference default | dense_reference (cm_frechet_lookup correct+gated, not default) | dense_reference (cm_frechet_lookup available) | dense_reference (operator verifier NOW EXISTS, not default) | shared_inplace_pooled (economic_A_gradient!, WIRED this session) | shared winner-pair + level extension | not_attempted |
| CM+ZC | shared_economic_operator | operator backend | operator backend | dense_reference (operator verifier exists, inherited, not default) | shared_inplace_pooled (economic_A_gradient!) | shared winner-pair | not_attempted |
| ZC-only | shared_economic_operator | operator backend | operator backend | dense_reference (operator verifier exists, inherited, not default) | shared_inplace_pooled (economic_A_gradient!, inherited) | shared winner-pair | not_attempted (H_EZ) |

## Reading notes

- "NOW EXISTS" = built this session (Phase A continuation); "inherited" = built by a prior
  session on this same release branch; "not default" applies to every operator verifier across all
  5 families -- none has been flipped from `dense_reference` to `operator` as the production
  default, for any family, in any session to date.
- Flexible-CM's `H_ER` is the only cross-Hessian cell with real progress this session: a validated,
  gated primitive exists, but it is not called from any production code path yet.
- This matrix intentionally does NOT claim gate results for conditions this session did not run
  (upper/lower public-driver smokes, exact-cache reuse, checkpoint save/kill/resume, cold
  verification) -- those remain exactly as the inherited branch's own `4fb9839` state left them,
  which this session did not re-verify or extend.
