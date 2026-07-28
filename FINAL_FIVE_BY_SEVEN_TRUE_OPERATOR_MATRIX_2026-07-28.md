# Final five-by-seven true-operator matrix (2026-07-28) — FINAL: ALL FIVE PASS

| family | bundle_type | has_H_field | has_moments_field | FG backend dense-free | Hessian backend dense-free | D=4 gate | D=20/W=100k gate |
|---|---|---|---|---|---|---|---|
| unrestricted | `OperatorPsiBundle` | false | false | true | true | pass, exact | pass, exact (nStatus 0/0) |
| flexible_cm | `OperatorPsiBundle` | false | false | true | true | pass, exact (serial+threaded) | pass, exact (nStatus -103/-103) |
| common_frechet | `OperatorPsiBundle` | false | false | true | true (after fixing serial-branch bug) | pass, exact | pass, exact (nStatus 0/0) |
| cm_plus_zc | `OperatorPsiBundle` | false | false | true | true (after fixing build_bin_tables_threaded! bug) | pass, exact | pass, exact (nStatus 0/0) |
| zc_only | `OperatorPsiBundle` | false | false | true | true (after fixing scratch-field bug) | pass, exact | pass, exact (nStatus 0/0) |

All five families reached `OPERATOR_BUNDLE=OperatorPsiBundle`/`has_H_field=false`, validated by
real KNITRO gates at both D=4 (synthetic, exact 0.0 agreement on objective/gradient/packed
Hessian/full inner solve) and real D=20/W=100,000 production data (exact agreement on the full
inner solve's accepted status, delta*, and complete dual vector -- the direct machine-precision
solution-equivalence proof). See `FIVE_FAMILY_NO_H_BUNDLE_GATE_2026-07-28.md` for the full detail,
including the three real (pre-existing, not bundle-introduced) bugs found and fixed while
generalizing beyond flexible CM, and the architecture note on common-Frechet's own duplicated
(not shared-and-parameterized) Hessian-callback implementation.

CSV companion: `FINAL_FIVE_BY_SEVEN_TRUE_OPERATOR_MATRIX_2026-07-28.csv`.
