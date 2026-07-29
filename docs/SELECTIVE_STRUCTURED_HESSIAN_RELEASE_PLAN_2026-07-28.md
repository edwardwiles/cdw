# Selective structured Hessian production release — plan (2026-07-28)

Branch: `release/selective-structured-hessian-production-2026-07-28`.
Base: `origin/production/fullA-exact@d455e43` (current tip at plan time; the two commits ahead of
this task's original cut point `5b4f9da` are `550f613`/`d455e43`, an unrelated outer-driver
lower-direction wiring feature + a five-family multistart shakedown campaign — touch
`cm_checkpoint.jl`/`cm_originzc_checkpoint.jl` only in `find_smallest`/incumbent-comparison
plumbing, purely additive with `find_smallest=true` preserving every existing caller's behavior
byte-for-byte; confirmed by reading the full diff before merging. No Hessian-internals overlap).

Merged in whole: `optimize/production-structured-CM-ZC-hessian-2026-07-28@3fde0b0` (one trivial,
purely-additive merge conflict in `cm_checkpoint.jl` — two independent `include` lines added by
each side — resolved by keeping both).

## Commit classification (inherited from `optimize/production-structured-CM-ZC-hessian-2026-07-28`)

| commit | subject | class |
|---|---|---|
| `4e4838e` | provenance + algebra docs | MERGE_NOW (docs only) |
| `9377952` | threaded H_EC/H_EZ/H_CZ kernels, opt-in | MERGE_NOW (correctness-gated D=4, opt-in, zero default-behavior change at commit time) |
| `ce309c8` | H_ZZ BLAS/threaded candidates | MERGE_NOW (opt-in, correctness-gated) |
| `f4af26e` | tests, lifecycle audit, master report | MERGE_NOW (docs/tests) |
| `410c154` | provenance freeze | MERGE_NOW (docs) |
| `a4c55ed` | D=20 profiling: live-handle stash | MERGE_NOW (diagnostic instrumentation, opt-in) |
| `4d35b2c` | **WinnerBinCrossScratch constructor arg-order fix** | MERGE_NOW — real bug fix, independent of every optimization candidate, see §3 of the master report |
| `5380773` | common_frechet D=4 threaded H_EC gate | MERGE_NOW (test) |
| `2642d0f` | D=20 profiling: sub-block instrumentation (flexcm/frechet) | MERGE_NOW (diagnostic, opt-in) |
| `bbda590` | same constructor fix, found independently on 2nd sub-branch | MERGE_NOW (content-identical to `4d35b2c`, kept for history) |
| `7c13f3a` | Zc-caching implementation | MERGE_AFTER_CONCISE_GATE — implemented + D=4-validated, needed a real D=20 gate before any default flip (see §9 gate below) |
| `76a3836` | Zc-caching docs + algebra | MERGE_NOW (docs) |
| `df9b96f` | D=20 sub-block profile + gates (flexcm/frechet) | MERGE_NOW (evidence this plan's §4 defaults rely on) |
| `97b2a5d` | D=20 orchestration + H_ZZ benchmark scripts | MERGE_NOW (scripts) |
| `8998f6f` | D=20 raw per-run CSVs | MERGE_NOW (raw evidence) |
| `379faed` | D=20 profiling final deliverables | MERGE_NOW (docs) |
| `e52e21f`, `5e2de90` | sub-branch merges | MERGE_NOW (merge commits) |
| `6c8d8f4` | master report + corrected Section 3 | MERGE_NOW (docs — includes the live retraction of the original "cm_meanzc production regression" claim, replaced with the correct KNITRO-concurrency-contention finding) |
| `9a020bc`, `3fde0b0` | manifest + handover note | MERGE_NOW (docs) |

No commits classified `REWORK` or `DROP` — nothing in this branch is a failed low-level profiling
harness or unrelated exploratory code; everything shipped was either a real fix, an opt-in kernel
with a correctness gate, or evidence/docs. No cherry-picking was needed — the whole branch merged
cleanly.

## This continuation session's additions (on top of the merge)

- `8d2c613`: flipped `CROSS_HESSIAN_THREADED_DEFAULT[]` to `true` (flexible_cm/common_frechet/
  origin_zc H_EC/H_EZ) — MERGE_AFTER_CONCISE_GATE, gate already satisfied by the inherited D=20
  matched-driver evidence (`COMPLETE_INNER_SOLVE_BEFORE_AFTER_2026-07-28.csv`) plus a fresh full
  D=4 regression re-run (0 failures) on this exact merged HEAD. `cm_meanzc`'s own constructor
  pinned to `false` explicitly — not affected by this flip.
- CM+ZC (`H_EC`/`H_EZ` for `cm_meanzc`), the H_ZZ realistic resource gate, and the ZC-centering
  D=20 gate are still open at plan-freeze time — tracked as MERGE_AS_CANDIDATE, not yet enabled.
  See `docs/SELECTIVE_STRUCTURED_HESSIAN_RELEASE_MASTER_2026-07-28.md` for the final verdict once
  those land.

## Architecture invariants re-confirmed at merge time

All preserved unchanged from the inherited branch (re-verified by reading, not just trusting the
inherited report): canonical `OperatorPsiBundle`, zero legacy-H allocations, zero composite-G
materializations, zero production `moments!`/`select_G_from_H` calls, no CM-bin/Fréchet-level
rebuilds inside any Hessian callback, no lower-triangle reads/writes anywhere in the threaded
kernels (output-ownership design, matches `core_exact_hessian.jl`'s existing H_EE idiom).

Common Fréchet continues to share the exact same (E)/(C) implementation as flexible CM
(`build_cm_bin_ctx`/`hessian_cm_structured_v2!`) — the threaded H_EC default flip in `8d2c613`
applies identically to both since they route through the same constructor.
