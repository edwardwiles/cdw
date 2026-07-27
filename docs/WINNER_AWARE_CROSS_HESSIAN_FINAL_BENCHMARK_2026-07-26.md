# Winner-Aware Cross-Hessian — Status 2026-07-26/27 (NOT attempted this session)

Task §10 asks for a winner-aware economic × restriction cross-Hessian backend (`E = Q - νπ'`,
exploiting the winner-sparse structure of `E` the same way the shared winner-pair `H_EE` backend
already does) for flexible-CM/common-Fréchet's `H_EC`, CM+ZC's `H_EC`+Z-cross, and origin-ZC's
Z-cross — eliminating the dense economic-column read this session's own
`docs/NO_DENSE_G_GLOBAL_RUNTIME_PROOF_2026-07-26.md` confirmed (by grep) still exists at
`cm_hessian_architectures.jl:606` and `cm_frechet_hessian.jl:59`.

**This was not attempted this session.** Rationale: this session's available time was allocated to
(a) finishing the FG-operator retrofits for the 2 remaining restricted families (flexible-CM,
common-Fréchet) plus fixing a real, severe allocation regression found in common-Fréchet along the
way, (b) extending operator verification to a 2nd family, (c) resolving an open question from the
prior session (`select_G_from_H`'s true cost) via a dispatched allocation audit, which itself
surfaced a larger, more consequential allocation problem in a DIFFERENT subsystem (the outer
A-gradient) that this session judged higher-priority to dispatch a follow-on investigation for than
this task's own Hessian cross-block item. Winner-aware cross-Hessian work is a substantial,
self-contained body of numerical-methods work (deriving `Q'SR - π(ν'SR)` for R ∈ {CM bins, Z
features} for THREE different R structures, plus gates) comparable in scope to any ONE of the three
background investigations this session already dispatched (allocation audit, outer A-gradient,
CM basis diagnosis) — attempting it without dedicated time would risk a rushed, unvalidated
implementation, which this task's own instructions explicitly discourage ("Do not merge to
production... Every fix requires... before/after allocation" for the comparable A-gradient task;
the same standard should apply here).

## Recommendation for the next session

This is the correct next Phase B item once the currently-dispatched three background
investigations (allocation audit — done; outer A-gradient; CM basis diagnosis) land. The economic
side is now well-understood (this session's `economic_forward!`/`economic_transpose!` retrofit
work + the shared winner-pair `H_EE` backend it's modeled on) — the missing piece is purely the
restriction-side winner-bin/winner-feature cross contraction, which should follow the same
"exploit winner sparsity, one shared primitive, family-specific composition" pattern this session's
own FG-operator work already established as reproducible practice in this codebase.
