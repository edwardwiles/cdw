# Remaining Dense G Consumers — Handoff for winner_aware_H_ER_cross_hessian — 2026-07-27

Task's explicit boundary: "Do not claim full G has been eliminated globally... this release should
prove `dense G absent from FG and strict verification`. The remaining dense economic-column
consumers should be listed precisely for the next standalone task: winner-aware H_ER cross-Hessian."

## What THIS release proves is dense-G-free

Real runtime-counter evidence (`smoke_no_dense_g_five_families.jl`, re-confirmed on this release's
own merged HEAD): at their PRODUCTION DEFAULT (not merely opt-in), flexible-CM, CM+ZC, and ZC-only
run their ordinary FG hot path with zero dense-G reads (`operator_FG_calls>0`,
`dense_economic_G=0`, `full_G=0`). Unrestricted has been fully operator-based (compressed, no
dense/operator branch to select) since an earlier session's Addendum Part A. Common-Frechet's
default (`dense_reference`) is, honestly, dense by design — its own operator alternative
(`cm_frechet_lookup`) is now correct and gated (this release's own fix) but not the default.

## What remains dense — precise, file-and-line list

Grep census, this session, of `obj.H[:, 2:1+NCORE]`-style dense economic-column reads in the
Hessian files touched by the five families (both are `@view`s into `obj.H`, not copies — but
`obj.H`'s economic columns must still be DENSELY FILLED by `moments!` on every outer point for
these views to be valid):

```
cm_hessian_architectures.jl:606:  E = @view H[:, 2:1+NCORE]   # H_EC cross-block (flexible CM / CM+ZC, Architecture C)
cm_frechet_hessian.jl:59:         E = @view H[:, 2:1+NCORE]   # H_EC cross-block (common Frechet)
```

Both are `PRODUCTION_HOT_PATH` — the Hessian callback runs on every KNITRO Hessian evaluation for
these families, not merely at setup. This is an explicit, documented, deliberate scope boundary of
the entire FG/verification work this and the inherited sessions did (`economic_operator.jl`'s own
header: "Eliminating THAT dependency would be Hessian cross-block rework, explicitly out of scope
for this task").

## A related, THIS-session-confirmed finding: the dense fill these views depend on is NOT always
## safely skippable

`COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md` documents a real bug this session found and fixed:
an inherited `skip_cm_fill_ref` optimization incorrectly assumed common-Frechet's shared Hessian
callback (`archC_frechet_hess_cb_builder`) never reads the dense CM/level columns the `H_EC`
cross-block view above depends on — it does, and skipping the fill produced a real, reproducible
infeasible KNITRO termination. This is directly relevant context for whoever picks up the winner-
aware `H_ER` cross-Hessian task: the dense economic/CM/level columns these cross-blocks read are
NOT a vestigial dependency that can be silently dropped without care — any future FG/Hessian
matrix-free rework touching these families must independently re-verify (not merely assert) which
Hessian paths actually read which dense columns before skipping any fill, exactly as this session
had to do the hard way.

## What the next task needs to build

Per the task's own formula: `H_ER = Q'SR - pi(nu'SR)`. This requires a genuinely new derivation of
the economic-x-restriction cross-Hessian block in operator form (analogous to how
`economic_forward!`/`economic_transpose!` already eliminated the dense economic-block FG reads,
but for the SECOND-derivative cross term, not the gradient) — real, correctness-sensitive numerical
work, not wiring or wrapping existing pieces. This is exactly why this task's own instructions
scoped it out explicitly: "Do NOT implement the winner-aware (H_ER) cross-Hessian in this task."

## Verdict

```text
FG_AND_VERIFICATION_DENSE_G =
    zero_at_production_default: unrestricted, flexible_cm, cm_plus_zc, zc_only
    present_at_production_default: common_frechet (dense_reference is the default; a correct,
        gated, non-default operator alternative now exists, see COMMON_FRECHET_FG_D20_FINAL_GATE)
    present_by_design_all_5_families: Hessian H_EC/H_ER cross-blocks (task Section 10 scope,
        NOT attempted this task, precisely listed above for the next standalone task)
NEXT_STANDALONE_TASK = winner_aware_H_ER_cross_hessian
```
