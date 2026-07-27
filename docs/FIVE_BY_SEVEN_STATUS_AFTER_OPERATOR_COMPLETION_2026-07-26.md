# Five-by-Seven Status Matrix After Operator Completion — 2026-07-26/27

Machine-readable version: `FIVE_BY_SEVEN_STATUS_AFTER_OPERATOR_COMPLETION_2026-07-26.csv`.
Rows = 5 families, columns = 7 architecture dimensions (economic FG, restriction FG, verification,
H_EE, H_ER, H_RR, full-G materialization). See
`FIVE_FAMILY_OPERATOR_STACK_COMPLETION_2026-07-26.md` for narrative and the full final verdict.

| Family | Economic FG | Restriction FG | Verification | H_EE | H_ER | H_RR | Full-G |
|---|---|---|---|---|---|---|---|
| unrestricted | compressed_operator/**default** | n/a | dense_reference (not attempted) | shared winner-pair | n/a | n/a | zero (confirmed) |
| flexible_cm | shared_economic_operator/**default (NEW)** | cm_lookup/default (unchanged) | dense_reference (not attempted) | shared winner-pair | dense economic-column dependent (confirmed) | family-specific (unchanged) | zero in FG (confirmed) |
| common_frechet | shared_economic_operator/available, **not default** | cm_frechet_lookup/available, not default | dense_reference (not attempted) | shared winner-pair | dense economic-column dependent (confirmed) | family-specific (unchanged) | operator available; default pending D=20 perf |
| cm_plus_zc | shared_economic_operator/**default (FLIPPED)** | operator/**default (FLIPPED)** | operator/available, **not default (NEW)** | shared winner-pair | dense economic-column dependent (confirmed) | family-specific (unchanged) | zero in FG (confirmed) |
| zc_only | shared_economic_operator/**default (FLIPPED)** | operator/**default (FLIPPED)** | operator/available, not default (inherited) | shared winner-pair | exact ZC cross-block (unchanged) | family-specific (unchanged) | zero in FG (confirmed) |

**Bold** = new or changed this session (2026-07-26/27 continuation). Everything else was already
true before this session (either from the immediately-prior `shared-inner-fg-operator-port`
session or earlier production merges).

**Read alongside**: `docs/NO_DENSE_G_GLOBAL_RUNTIME_PROOF_2026-07-26.md` (evidence for the
"Full-G" column), `docs/phaseB_no_dense_g_five_family_smoke_log.txt` (raw counter output this
table's "confirmed" claims are based on).
