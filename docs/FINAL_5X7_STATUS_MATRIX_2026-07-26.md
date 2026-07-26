# Final 5×7 Status Matrix — 2026-07-26

**This is a STATUS matrix (what backend is active, what is verified), not a performance matrix.**
Task §11's real-runtime-attributed 5×7 performance matrix was not produced this session — see
`FIVE_FAMILY_FINAL_PROFILE_STATUS_2026-07-26.md`. Every cell below reflects this session's own
verified state, not an inference from a sibling family.

| Family | inner_fg_backend | full_G_materialization | exact_cache | dual_bank | cross_hessian_backend | cm_basis | A_coordinate_mode |
|---|---|---|---|---|---|---|---|
| Unrestricted | shared exact winner-pair (pre-existing, unchanged by this task) | N/A (not a restricted family) | pre-existing unrestricted-family cache (unchanged) | pre-existing KKT-scored `DualBank` (unchanged, already default) | shared exact winner-pair core Hessian (pre-existing) | N/A | `powered_aspace` (verified: upper+lower+resume+legacy-refusal, Phase 1.1) |
| Flexible CM | `:dense_reference` default; `:cm_lookup` available, not default (allocation regressed +12.6%, inherited Phase B1) | present (chunked-dense fill, not operator — see `NO_FULL_G_MATERIALIZATION_AUDIT`) | wired, verified D=20 all-invariants (Phase 1.2) | opt-in only (`false` default — real trajectory evidence shows non-trivial warm-start failures, `RESTRICTED_DUAL_BANK_FINAL_DECISION`) | shared exact winner-pair H_EE (pre-existing); cross-block not audited this session | cumulative + orthonormal (pre-existing decision, reconfirmed) | `powered_aspace` (verified live smoke, Phase 8) |
| Common Fréchet | `:dense_reference` (no lookup-FG variant exists for this family — inherited session's own scope note) | present (chunked-dense fill, same as flexible CM plus level-anchor block) | wired, verified D=20 (Phase 1.2) | opt-in only (same finding: real warm-start failures, up to 100% in this session's benchmark) | shared exact winner-pair H_EE; level-block extension separate (pre-existing) | cumulative + orthonormal | `powered_aspace` (verified live smoke, Phase 8) |
| CM+ZC | `:dense_reference` (no lookup-FG variant — CM-grid-specific kernel doesn't cover the mean/pair block) | present, **explicitly retained dense by benchmark decision** (Phase E part 2) — furthest from compliant | wired, verified D=20 (Phase 1.2) — includes the constructor bugfix (`879c8bb`) that made CM+ZC constructible again | opt-in only | shared exact winner-pair H_EE; mean/pair cross block not audited this session | cumulative + orthonormal | `powered_aspace` (verified live smoke, Phase 8) |
| Origin-ZC | N/A (no CM grid; small fixed-size raw power features) | small, fixed-size block (K_mean/K_pair scale) — lower priority than the other three | wired, verified D=20 (Phase 1.2 — the gap the inherited session explicitly never closed) | opt-in only; real-trajectory evidence pending this session's corrected re-run (see master report) | shared exact winner-pair H_EE (per the corrected docstring, Phase F item 3); H_ER/H_RR dense | N/A (no CM grid) | `powered_aspace` promoted; live-smoke pending corrected re-run |

## Companion CSV

See `key_results/final_5x7_status_matrix_2026-07-26.csv` in the Dropbox push package for the
machine-readable version of this table.
