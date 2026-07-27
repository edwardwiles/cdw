```text
CM_FEATURE_IMMUTABILITY = pass
CM_BASIS_DEFAULT = cumulative
ORIGIN_CONTRAST_DEFAULT = anchored
INTERVAL_HESSIAN = correct_not_faster
```

# CM Basis and Contrast Default Reconciliation — 2026-07-27

This document closes out the one open question `cm-basis-diagnosis-2026-07-27`
(`diag/cm-basis-interval-orthonormal-2026-07-27`, commit `92b9d7b`, adopted into this release —
see `THREE_BRANCH_RECONCILIATION_PLAN_2026-07-27.md`) left unresolved: it found a real doc/code
mismatch (three 2026-07-26 documents claim `contrasts=:orthonormal` is already the production
default; the actual wired driver defaults to `:anchored`) and, per that session's own scope
(no authorization to flip a default without evidence *for* the change), left the verdict as
`ORIGIN_CONTRAST_DEFAULT = inconclusive` rather than picking a side.

**This task's own instructions resolve that ambiguity explicitly**: "Anchored and orthonormal
contrasts were effectively tied at D=20. Actual code defaults to anchored; update stale
documentation that claimed orthonormal was already default." That is exactly what this document
and the correction banners added to the three stale documents (below) do.

## What changed

**No code changed.** `contrasts::Symbol = :anchored` was already the actual default at every real
production call path (`run_cm_upper_checkpointed`, `cm_production_bundle.jl`, `cm_frechet_level.jl`,
`cm_config.jl`'s `CMConfig`) before this reconciliation — see
`ORTHONORMAL_CONTRAST_FINAL_DIAGNOSIS_2026-07-27.md` §2 for the exact call-path trace. There was
never a case for changing the CODE; the gap was three DOCUMENTS asserting the opposite of what the
code does.

**Three documents corrected** (a banner added at the top of each, pointing here; original content
preserved below the banner rather than deleted, since the underlying D4 benchmark numbers in each
are not disputed — only the "already decided, already default" framing was wrong):

1. `docs/ORTHONORMAL_ORIGIN_CONTRAST_BENCHMARK_2026-07-26.md` — originally titled "This decision
   was already made and is already the production default", asserting `:orthonormal`.
2. `docs/FIVE_FAMILY_OPTIMIZATION_COMPLETION_MASTER_REPORT_2026-07-26.md` — final-verdict block
   read `ORIGIN_CONTRAST_DEFAULT = orthonormal (pre-existing 2026-07-22 decision, reconfirmed not
   re-decided)`.
3. `docs/FINAL_5X7_STATUS_MATRIX_2026-07-26.md` — all three CM-grid-bearing families' `cm_basis`
   column read "cumulative + orthonormal (pre-existing decision, reconfirmed)".

None of these three documents ran any real evidence of their own for the orthonormal claim — each
cites the SAME 2026-07-22 review (`fullA_cm_conditioning_and_adaptive_grid_report.md`), which is a
D4-only conditioning benchmark that never claims to have set a production default, let alone
checked D=20. The "already decided, already default" framing appears to have originated from
over-reading that D4 benchmark as a decision record, then been repeated verbatim by two later
documents citing the first rather than re-checking the code — a citation chain, not independent
verification. This is worth naming explicitly since it is a reusable failure pattern (a claim
becomes "established" by repetition across documents that all trace back to one unverified
original), distinct from, but structurally similar to, this project's own recurring "A_od≡1 is not
calibration" trap that CLAUDE.md already flags.

## The evidence (real, this reconciliation's own contribution is only the resolution + doc fixes; the empirical work is `cm-basis-diagnosis-2026-07-27`'s)

Real D=20/W=80,000 (`c15_d20_four_arm_basis_contrast_comparison.jl`, seed 20260719,
`destination_sample=:exclude_row`, L=50) — the first real-D20 check of this exact question in this
codebase's history:

| point | basis | cond(anchored) | cond(orthonormal) | ratio (ortho/anchored) |
|---|---|---|---|---|
| calibration | cumulative | 1.1095e+06 | 1.1108e+06 | 1.0012 |
| perturbed_2pct | cumulative | 1.4619e+06 | 1.4626e+06 | 1.0005 |
| calibration | interval | 4.2165e+07 | 4.2314e+07 | 1.0035 |
| perturbed_2pct | interval | 4.8028e+07 | 4.8838e+07 | 1.0169 |

Conditioning-equivalent to within 0.05%-1.7% (an order of magnitude smaller than run-to-run KNITRO
iteration-count noise), with anchored fractionally *better* at every point checked — the opposite
direction from the D4-only "orthonormal wins by 2.6-3.4x" claim the three corrected documents were
built on. `Delta_dual` agrees between contrasts to 1.6e-16-2.2e-16 at every point (exact, by
construction — `CM[:, cols] .= block * R` with `R'R = I`, an orthogonal congruence transform, not
an independently-estimated equivalence). Full derivation and the D4 numbers (which DO show a real,
reconfirmed 2.2x-3.4x orthonormal advantage that simply does not survive to D20) in
`ORTHONORMAL_CONTRAST_FINAL_DIAGNOSIS_2026-07-27.md`.

## Verdict

```text
CM_BASIS_DEFAULT = cumulative
    (interval is exact, machine-precision-verified, but 32.9x-38.1x worse conditioned at real D=20
    under both contrasts and did not improve complete-solve time -- INTERVAL_VS_CUMULATIVE_FINAL_
    DIAGNOSIS_2026-07-27.md. Retained as an exact reference/opt-in mode, not default.)
ORIGIN_CONTRAST_DEFAULT = anchored
    (matches the actual wired code, unchanged. Orthonormal and anchored are conditioning-equivalent
    at D=20 -- neither the D4-only "orthonormal wins" claim the corrected documents made, nor any
    new D20 finding, clears this task's own bar for preferring one over the actual wired default.
    Retained as an exact reference/opt-in mode.)
CM_FEATURE_IMMUTABILITY = pass
    (empirically confirmed, real D=20, rebuilds_due_to_A_or_gp=0 -- IMMUTABLE_CM_FEATURE_OPERATOR_
    FINAL_2026-07-27.md)
INTERVAL_HESSIAN = correct_not_faster
```

## Scope not reopened

Per this task's own explicit instruction ("Do not reopen interval-versus-cumulative basis R&D.
Preserve the diagnosis unless new evidence reveals a correctness problem") and per
`ORTHONORMAL_CONTRAST_FINAL_DIAGNOSIS_2026-07-27.md`'s own honest gap list, common Fréchet and
CM+ZC were not independently re-run at D20 for the contrast axis (the `cctx.R !== nothing`
congruence mechanism is shared identically across all three CM-grid families per
`production_backend_manifest.jl`'s family-agnostic `restriction_hessian_backend` label, making a
differing verdict there unlikely but unmeasured) — this remains an honest, disclosed gap, not
re-investigated by this reconciliation since no correctness problem was found that would warrant
reopening it.
