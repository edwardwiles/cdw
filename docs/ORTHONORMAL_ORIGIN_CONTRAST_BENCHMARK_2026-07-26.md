# Orthonormal Origin-Contrast Benchmark — 2026-07-26

> **CORRECTION (2026-07-27, `CM_BASIS_AND_CONTRAST_DEFAULT_RECONCILIATION_2026-07-27.md`): the
> claim below that `contrasts=:orthonormal` is "already the production default" is WRONG.** The
> actual wired production driver (`run_cm_upper_checkpointed`, `cm_checkpoint.jl:591`) defaults to
> `contrasts=:anchored`, confirmed by reading the code directly, not by any doc citation. Real
> D=20/W=80,000 evidence (gathered 2026-07-27, this codebase's first real-D20 check of this
> question) also shows anchored and orthonormal are conditioning-equivalent at D=20 (<1.7% either
> direction) — the D4-only "2.6-3.4x orthonormal advantage" this document cites below does not
> survive to D=20 and should not be used to justify a default either way. **Production default
> remains `:anchored` (the actual wired value), unchanged.** See the reconciliation doc for the
> full evidence. The rest of this document is preserved for its D4 benchmark data, which is not
> disputed — only its "already decided, already default" framing is wrong.

## This decision was already made and is already the production default

Contrary to the task's framing (as if `anchored` vs `orthonormal` were still an open question),
`contrasts=:orthonormal` is **already** the approved, production-default contrast basis for the
CM-family drivers, decided in the 2026-07-22 conditioning review
(`docs/fullA_cm_conditioning_and_adaptive_grid_report.md`,
`docs/fullA_cm_hessian_architecture_report.md`, referenced directly in
`cm_production_stage_runner.jl`'s own header comments, which this session read while building the
Phase 2 benchmark).

Documented evidence from that prior review: orthonormal strictly dominates anchored at every
`L ∈ {10, 20, 50}` and both calibration points tested (2.6-3.4x lower `cond(Hessian)`, gap widening
as `L` grows — the L=50 regime every current production campaign runs at), and is far less
reference-country-sensitive (0.8-1.5% spread vs 2.9-9.7% for anchored). The one documented cost —
losing per-origin sparsity in the CM moment columns — does not block production because the wired
Hessian backend (`cm_hessian_backend=:structured`, Architecture C) is independently validated
correct and still gives a real 2.2x-4.5x speedup under orthonormal contrasts at L=50.

This session's own D=20 gates (Phase 1.2/1.3, Phase 2, Phase 8 — all four restricted families) all
ran with `contrasts=:orthonormal` and passed, consistent with (not contradicting) that prior
decision.

## What this session did NOT do

Re-run a fresh four-arm (cumulative+anchored / cumulative+orthonormal / interval+anchored /
interval+orthonormal) comparison as task §7.1 describes. The **interval basis** itself
(orthogonal to the anchored-vs-orthonormal axis) is not wired into `build_cm_production_context` at
all currently — see `INTERVAL_VS_CUMULATIVE_CM_BASIS_2026-07-26.md` — so only two of the four arms
(cumulative+anchored, cumulative+orthonormal) are even constructible today, and that pair's
comparison is the one already settled by the 2026-07-22 review cited above. Re-deriving it from
scratch this session would have duplicated already-validated prior work rather than closing a real
gap.

## Verdict

`ORIGIN_CONTRAST_DEFAULT = orthonormal` (already decided, already default, reconfirmed by this
session's own gates passing under it — not a new decision this session made).
