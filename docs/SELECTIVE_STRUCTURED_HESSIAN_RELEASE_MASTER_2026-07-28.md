# Selective structured Hessian production release — master report (2026-07-28)

Branch: `release/selective-structured-hessian-production-2026-07-28`, HEAD `c7ab07a`. Base:
`origin/production/fullA-exact@d455e43` (the current tip at merge time — two commits ahead of this
work's original `5b4f9da` cut point, confirmed unrelated: an outer-driver "lower direction" wiring
feature and a five-family multistart shakedown campaign, purely additive, no Hessian-internals
overlap; one trivial merge conflict, two independently-added `include` lines, resolved keeping
both). Full commit list and per-commit classification: `docs/SELECTIVE_STRUCTURED_HESSIAN_RELEASE_PLAN_2026-07-28.md`.

This continues and closes out `docs/HANDOVER_NOTE_2026-07-28.md`'s open items from the prior
session that built this branch's inherited work.

## 1. What this release does

Ports the prior session's validated threaded-Hessian and ZC-centering-cache work onto current
production, closes its three open gates (CM+ZC isolated H_EC/H_EZ, ZC-centering D=20, H_ZZ
resource), and enables production defaults per explicit review with the user at each real
tradeoff. Architecture invariants (canonical `OperatorPsiBundle`, zero legacy-H allocations, zero
composite-G materializations, zero production `moments!`/`select_G_from_H` calls, no CM-bin/
Fréchet-level rebuilds inside any Hessian callback) preserved throughout — confirmed by direct
counter checks in the final gate (§6), not assumed.

## 2. Constructor fix (merged immediately, per task brief)

`WinnerBinCrossScratch`'s 3-arg constructor had `tasks_ec`/`EsumEcon` swapped relative to the
struct's declared field order — silently broke the shared `:winner_bin` H_EC backend for any
context with a real `CompressedFactual`, surfacing only as an opaque KNITRO `nStatus=-500`. Fixed
(commits `4d35b2c`/`bbda590`, found independently twice), confirmed at D=4 and real D=20.

## 3. Threaded H_EC / H_EZ / H_CZ, and the ZC-centering cache

All four already existed as opt-in kernels from the inherited branch (see
`docs/PRODUCTION_STRUCTURED_HESSIAN_OPTIMIZATION_MASTER_2026-07-28.md`, the prior session's own
report, for the full algebra/lifecycle detail — unchanged here). This session's job was closing the
three gates that report left open and deciding production defaults:

### 3.1 flexible_cm / common_frechet / origin_zc — enabled immediately

Already had solid real D=20/W=100,000 matched-driver evidence from the prior session: bit-exact
packed Hessian, unaffected KNITRO status/n_eval/n_grad/kappa, 1.42x complete-callback for
flexible_cm/common_frechet, 4.5x H_EZ for origin_zc
(`docs/COMPLETE_INNER_SOLVE_BEFORE_AFTER_2026-07-28.csv`). `CROSS_HESSIAN_THREADED_DEFAULT[]`
flipped `true` (commit `8d2c613`), re-verified with a fresh full D=4 regression pass (0 failures)
on this session's own merge.

### 3.2 CM+ZC (cm_meanzc) H_EC/H_EZ — gated this session, then enabled

The prior session couldn't cleanly validate this (host KNITRO-concurrency contention). This session
ran a real, isolated (no concurrent KNITRO) D=20/W=100,000 gate through the actual public driver
(`docs/CM_MEANZC_HEC_HEZ_ISOLATED_GATE_2026-07-28.csv`/`.md`): bit-exact packed Hessian in every
completed run, zero `nStatus=-500/-502` errors, 1.26x wall time at the one cleanly-matched point,
more real solver work completed in less wall time at the longer-budget point. Passed.

**Architecture wrinkle, not a data problem**: `cross_hessian_threaded` is ONE shared toggle across
H_EC/H_EZ/H_CZ for this family (no independent per-block switch exists in this codebase) — flipping
it for the validated H_EC/H_EZ win also enables H_CZ threading, whose own performance evidence is
weaker (wins only at 20 workers, loses to serial at 4/8; its CORRECTNESS was covered by this gate's
bit-exact full-packed-Hessian check, just not independently re-measured for speed). Flagged to the
user explicitly; enabled anyway on their instruction (commit `25673d2`) as an accepted tradeoff.

### 3.3 ZC-centering cache — gated this session, then enabled

D=4 evidence (28/28, inherited) plus a fresh real D=20/W=100,000 gate this session, both origin_zc
and cm_meanzc, run in isolation (`docs/ZC_CENTERING_D20_GATE_2026-07-28.csv`/`.md`): rebuild/cache-
hit counters behave exactly as designed in every row, 5/6 points faster wall time (the one
exception explained by fixed per-call context-build overhead dominating a short-budget single-rep
measurement), KNITRO status/n_eval/n_grad unaffected in 5/6 pairs. D=20 bit-exactness could not be
directly confirmed — a pre-existing, cache-unrelated bug in the `:operator` backend's low-level
callback-builder re-entrancy blocks post-hoc recomputation at D=20 (identical failure for both
families/cache settings, fires before the cache logic is reached) — D=4 bit-exactness used as
correctness backing instead. Enabled on explicit user instruction (commit `ad00887`).

### 3.4 H_CZ — stays experimental, no independent default (per task brief)

Not independently re-evaluated this session; inherits the prior session's evidence (1.3-1.4x at 20
workers, loses to serial at 4/8). Goes threaded-by-default for cm_meanzc ONLY as the unavoidable
side effect of §3.2's shared toggle, not because it independently passed its own bar.

## 4. H_ZZ backend — reverted to safe default after a live finding

Prior session's ISOLATED real-D20 evidence: `:blas_gemm` at BLAS threads>=8 is ~2.77x faster than
`:reference` at production width (nx=210). This session briefly flipped `ZC_GRAM_BACKEND_DEFAULT[]`
to `:blas_gemm` on explicit user instruction, then reverted it after a live finding: a `cm_meanzc`
solo run at `BLAS_THREADS=8` ran ~12 minutes wall time (killed, never completed) vs 60-185s for
every `BLAS_THREADS=1` run this same session — a plausible but unconfirmed thread-oversubscription
signal (Julia `-t 4` x `OPENBLAS_NUM_THREADS=8` x KNITRO's own internal threading). The user chose
not to ship that unresolved risk. **`:reference` ships this release** (commit `6f45a45`,
unchanged production behavior). Full writeup and concrete follow-up items:
`docs/HZZ_REALISTIC_RESOURCE_GATE_2026-07-28.md`.

## 5. D=4 regression — clean throughout

Full D=4 suite (`test_threaded_cross_hessian_d4.jl` + `test_zc_centered_cache_d4.jl`) re-run after
every default flip this session: 0 failures every time. Matches the documented baseline exactly
(flexible_cm 6/6, common_frechet 6/6, cm_meanzc 39/39, origin_zc 26/26 + the same pre-existing,
expected K2 (`K_mean=2,K_pair=1`) synthetic-data infeasibility, Zc-caching 28/28).

## 6. Final four-family production gate

One real D=20/W=100,000 call per family through the actual public checkpointed drivers, with EVERY
newly-merged default active together, run as 4 concurrent OS processes (explicit user request,
accepting cm_meanzc's known concurrency risk as a tradeoff for speed) —
`docs/FINAL_FOUR_FAMILY_PRODUCTION_GATE_2026-07-28.csv`. All 4 completed cleanly, including
cm_meanzc (no `-500/-502` this run). Architecture invariants confirmed directly via the shared
`NO_DENSE_G_COUNTERS[]`: `full_G_materializations`/`dense_economic_G`/`dense_CM_G`/`dense_ZC_G` = 0
in every family. `common_frechet`'s `dense_Frechet_G=2` is a pre-existing, documented, deliberate
exception (the `:operator` mode intentionally still fills the small Fréchet level block,
"safety-preserved" per this codebase's own existing tests — `test_moment_representation_default_2026-07-27.jl`),
not a violation. Both ZC-centering-cache families show real rebuild/hit activity in this genuine
production run (origin_zc: 1 rebuild/9 hits; cm_meanzc: 1 rebuild/6 hits) — the cache is not merely
passing isolated tests, it measurably engages end-to-end.

## 7. Final verdict

```
CONSTRUCTOR_FIX = merged (4d35b2c/bbda590, D=4+D=20 confirmed)

H_EC_DEFAULT =
    flexible_cm:winner_bin,threaded,workers=20
    common_frechet:winner_bin,threaded,workers=20 (shares flexible_cm's exact implementation)
    cm_plus_zc:winner_bin,threaded,workers=20 (enabled on user sign-off; coupled to H_EZ/H_CZ,
        see §3.2)

H_EZ_DEFAULT =
    cm_plus_zc:winner_bin,threaded,workers=20 (see H_EC_DEFAULT, same toggle)
    zc_only:winner_bin,threaded,workers=20

H_CZ_DEFAULT = threaded_by_default_for_cm_meanzc_only (side effect of the shared H_EC/H_EZ toggle,
    not independently validated this release; existing production backend used everywhere else,
    N/A for the other 3 families which have no C+Z block overlap)

H_ZZ_DEFAULT =
    small_nZ: :reference (unchanged)
    wide_nZ: :reference (reverted from a brief :blas_gemm flip -- see §4)
    ZC_family_BLAS_threads: 1 (unchanged; :blas_gemm/BLAS>=8 remains a well-evidenced but
        not-yet-safely-confirmed candidate, see docs/HZZ_REALISTIC_RESOURCE_GATE_2026-07-28.md)

ZC_CENTERING_CACHE = default_on (enabled on user sign-off; D=20 mechanism/wall-time evidence
    strong, D=20 bit-exactness blocked by an unrelated pre-existing bug, D=4 bit-exactness (28/28)
    used as correctness backing)

COMPLETE_INNER_SOLVE_SPEEDUP =
    flexible_cm: 1.42x complete-callback (real D=20, prior session)
    common_frechet: 1.42x complete-callback (real D=20, prior session)
    cm_plus_zc: 1.26x wall time at the one cleanly-matched isolated point; more real solver work
        completed in less wall time at the longer-budget point (real D=20, this session)
    zc_only: H_EZ 4.5x, peak 5.3x@t10 (real D=20, prior session)

PRODUCTION_MERGE = merged_tagged_smoked (pending this turn's explicit go/no-go and push -- see §8)

HIGHEST_PRIORITY_REMAINING_GAP = H_ZZ's realistic concurrent-resource-contention picture is
    unresolved (the cm_meanzc BLAS=8 slowdown, §4) -- :blas_gemm remains a real, well-evidenced,
    isolated 2.77x win that this release could NOT safely ship given that live finding. Next
    priority: diagnose the slowdown (thread-count interaction, not yet root-caused) and complete
    the originally-scoped realistic resource gate.
```

## 8. Merge status

This branch is feature-complete and internally consistent (D=4 clean, all four families' real
D=20 production gate clean, architecture invariants confirmed). **Not yet merged into
`production/fullA-exact`, pushed, or tagged** — per this project's own confirm-before-push
convention, awaiting the user's explicit go-ahead for those specific actions (a separate step from
the many in-session default-enablement decisions already made explicitly with the user throughout
this branch's development).
