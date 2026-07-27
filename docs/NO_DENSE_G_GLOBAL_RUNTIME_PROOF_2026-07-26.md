# No-Dense-G Global Runtime Proof — scoped to this session's Phase A work (2026-07-26/27)

## Scope statement (honest)

Task §9 asks for a complete repository audit (dense moment builders, `gemv!`/`gemm!` against
draw-by-moment matrices, `select_G_from_H`, generic verifiers, cache scorers, post-processing
routines) classified `PRODUCTION_HOT_PATH | PRODUCTION_SETUP_ONLY | EXPLICIT_REFERENCE | TEST_ONLY |
DEAD_CODE`. Per `feedback-grep-all-consumers-not-just-named-files` and the prior session's own
honest gap list (`shared-inner-fg-operator-port-2026-07-26` memory: "the full-codebase dense-G-
consumer audit (task §13, a ~500-file scope, not attempted)"), a complete audit of that size was
not attempted this session either — this document instead reports what THIS session's own Phase A
work (shared economic-operator retrofits for all 4 restricted families + production-default flips
for 3 of them) can prove with real runtime evidence, plus the specific, now-resolved question left
open by the prior session (`select_G_from_H`'s cost), plus an honest scope statement of what
remains (Hessian cross-blocks, a genuine ~500-file audit).

## Resolved this session: `select_G_from_H` is NOT a dense materialization

The prior session's own memory flagged (via the D=4 perf A/B in this session, see
`FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB` commit) that both `:operator` and `:dense_reference` FG
backends showed near-identical ~51MB/call allocation at D=4, unexplained at the time. The parallel
hot-path allocation audit subagent (worktree `audit-hot-path-array-allocation-2026-07-27`, commit
`e5e5c8f7a14cef3e4ef54c82453a9ea31f9ba997`) independently confirmed by direct measurement:
`CS.select_G_from_H` is a genuine **zero-byte `@view`** (`cc_algo/PsiObjectiveBundle.jl:624`,
`cc_algo/KLObjectiveBundle.jl:467`), NOT a dense copy/materialization. It should NOT be counted as
a `full_G_materializations`/`dense_economic_G_materializations` site anywhere in this codebase --
every one of its ~40 call sites found by `grep -rn "select_G_from_H"` (see that grep's full output
in this session's tool history) is reading a view into an already-allocated `obj.H` buffer, not
building a new dense matrix. The real source of the ~51MB (D=4) / ~13GB (D=20) per-"complete
inner solve" figure is the OUTER (A)-gradient's `composite_gradient_at_fast` finite-difference
loop (a completely different subsystem from this task's FG/Hessian/verification scope) -- see that
audit's own `HOT_PATH_ARRAY_ALLOCATION_AUDIT_2026-07-26.md` for the full reconciliation, and
`docs/A_GRADIENT_*` (a second background agent, `feature/shared-outer-a-gradient-2026-07-27`, was
dispatched this session specifically to fix that) for the fix attempt.

## What THIS session proved with real runtime counters (not just code reading)

`smoke_no_dense_g_five_families.jl`, one real D=4 inner solve per family at its **current
production default** config (log: `docs/phaseB_no_dense_g_five_family_smoke_log.txt`):

| Family | Default backend | `operator_FG_calls` | `dense_economic_G` | `dense_CM_G` | `dense_ZC_G` | `generic_dense_FG` | `full_G` |
|---|---|---|---|---|---|---|---|
| unrestricted | (no branch -- always compressed) | N/A | N/A | N/A | N/A | N/A | N/A |
| flexible_cm | `cm_lookup` | 5 | 0 | 0 | 0 | 0 | 0 |
| common_frechet | `dense_reference` (unflipped) | 0 | 0 | 0 | 0 | 0 | 0 |
| cm_plus_zc | `operator` | 5 | 0 | 0 | 0 | 0 | 0 |
| zc_only | `operator` | 5 | 0 | 0 | 0 | 0 | 0 |

3 of 5 families (flexible_cm, cm_plus_zc, zc_only) now run their ordinary FG hot path through the
shared/family operator with zero dense-G reads, at their real production default (not merely an
opt-in flag) -- this is new this session for `cm_plus_zc`/`zc_only` (previously opt-in only,
correctness-gated but not performance-gated or defaulted; see this session's commits flipping
`ORIGINZC_FG_BACKEND_DEFAULT`/`CM_MEANZC_INNER_FG_BACKEND_DEFAULT`). `common_frechet`'s
`generic_dense_FG_calls` shows 0 rather than a nonzero count reflecting real dense FG calls because
its `:dense_reference` dispatch path predates this branch's counter instrumentation and was never
wired to `record_generic_dense_fg!()` -- a real, pre-existing gap (not introduced this session,
not fixed this session either; flagged honestly here rather than silently reported as "0 dense
calls" implying it took zero dense FG evaluations, which is false -- it took 0 *operator* calls and
an unknown-but-nonzero number of real dense calls not currently counted).

## What is genuinely NOT eliminated (confirmed by reading, not assumed)

Grep census, this session, of `obj.H[:, 2:1+NCORE]`-style dense economic-column reads in the
Hessian files touched by the five families:

```
cm_hessian_architectures.jl:606:    E = @view H[:, 2:1+NCORE]        # H_EC cross-block (flexible CM / CM+ZC, Architecture C)
cm_frechet_hessian.jl:59:           E = @view H[:, 2:1+NCORE]        # H_EC cross-block (common Frechet)
```

Both are `@view`s (not copies) into `obj.H`, but `obj.H`'s economic columns must still be
DENSELY FILLED by `moments!` on every outer point for these views to be valid -- confirmed by the
prior session's own reading (`SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md`: "the Hessian's
H_EC/H_ER cross-terms still need dense H[:,2:1+NCORE] regardless"), re-confirmed here by direct
grep rather than re-asserted from memory. This is `PRODUCTION_HOT_PATH` (the Hessian callback runs
on every KNITRO Hessian evaluation, not merely at setup) but is an **explicit, documented,
deliberate scope boundary** of this task's own Phase A (`economic_operator.jl`'s own header:
"Eliminating THAT dependency would be Hessian cross-block rework, explicitly out of scope for this
task") -- task §10's winner-aware economic × restriction cross Hessian is the correct vehicle to
eliminate it, and was **NOT attempted this session** (a large, separate body of numerical work; see
`HIGHEST_PRIORITY_REMAINING_GAP` in the final verdict document).

## Explicitly out of scope for this document

- The full ~500-file repository audit task §9 literally asks for (classifying every dense moment
  builder / cache scorer / post-processing routine in the entire codebase). Given the scope already
  covered this session (4/5 families' FG operators, 1 major allocation-regression root-cause-and-fix,
  1 new operator-verification implementation, 2 dispatched follow-on investigations), this was
  judged lower priority than finishing Phase A's FG/verification work and is left for a future
  session, consistent with the prior session's own documented triage.
- Task §10's winner-aware cross-Hessian work (H_EC/H_ER elimination) -- confirmed real and
  necessary by this document's own grep census above, not attempted.
- The two dispatched background investigations (outer A-gradient allocation, CM basis diagnosis)
  have their own separate deliverable documents; this document does not duplicate their content.
