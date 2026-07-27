# Operator-Based Inner Verification — 2026-07-26

## What Job 2 asked for

Replace dense-`G` strict post-solve verification with operator-based verification for all five
families; remove the `skip_cm_fill_ref` mutable-switch pattern once operator verification exists;
production default `verification_backend = :operator`, dense reference retained only as an explicit
debug backend.

## What this branch actually did (proof of concept, one family)

`verify_inner_solution_operator_originzc!` (`operator_verification.jl`) — see
`ORIGIN_ZC_OPERATOR_FG_PORT_2026-07-26.md` for the validation results (15/15 D=4 checks, KKT
residual matching the dense verifier to ~1e-15 to ~1e-16 across three `K_mean/K_pair` configs).
This proves the approach is sound and reuses the SAME shared operators
(`economic_forward!`/`economic_transpose!`/`restriction_forward!`/`restriction_transpose!`) the FG
callback itself uses — exactly the addendum's intent — with genuinely independent scratch (fresh
`EconomicFGWorkspace`/`ZCRestrictionWorkspace` every call, not the live FG callback's own `st`).

## What was NOT done (explicit, honest gap list)

1. **CM+ZC's own operator verification.** Would need the CM-grid block's transpose contribution
   folded into the KKT residual (straightforward extension, reusing
   `cumulative_backward_gradient!` the same way `restriction_transpose!` is already reused for the
   Z block) — not attempted.
2. **Flexible CM's and common Fréchet's operator verification.** Their FG is still dense-E
   (`CMLookupState`/`CMFrechetLookupState`'s own economic block), so an operator-based verifier for
   them would need the flexible-CM E-block retrofit (`FLEXIBLE_CM_OPERATOR_FG_FINAL_GATE_2026-07-26.md`'s
   own recommended follow-on) done first, or would itself be a dense-E verifier wearing an
   "operator" label for the CM/level blocks only — neither attempted.
3. **`skip_cm_fill_ref` removal.** This switch belongs to flexible CM's own `moments!` closure
   (`wrap_moments_with_cm_archB`) and `archC_verified_state`'s post-solve recompute
   (`cm_production_bundle.jl`) — it is NOT touched by origin-ZC's own verification path at all
   (origin-ZC never had this switch; it has no CM-grid block). Removing it requires flexible CM's
   verification to go operator-based first (item 2 above), which did not happen this session. The
   switch remains exactly as the inherited port left it.
4. **`verification_backend = :operator` as a production default / dispatch point.** No family's
   production `archX_verified_state` function was changed to dispatch between `:operator` and
   `:dense_reference` verification — origin-ZC's own `archOZ_verified_state` still calls its
   pre-existing dense recompute (`obj(inner_x, constr=...)`, `CS.select_G_from_H`) unchanged;
   `verify_inner_solution_operator_originzc!` exists and is validated as a standalone function, not
   yet wired as an alternative inside the production verified-state call.

## Honest verdict

`VERIFICATION_BACKEND[origin_zc] = dense_reference (production, unchanged)`,
`operator (validated standalone, not wired as production default)`. All other families:
`dense_reference` only, unchanged. `verification_backend = :operator` production-default wiring is
a real, scoped, achievable follow-on — not attempted here given the time this task had available
after Job 1's two new operator families.
