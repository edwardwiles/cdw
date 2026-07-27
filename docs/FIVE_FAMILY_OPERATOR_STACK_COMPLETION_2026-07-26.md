# Five-Family Operator Stack Completion — Master Report — 2026-07-26/27

Continuation of `port/shared-inner-fg-operator-and-verification-2026-07-26`, on branch
`port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26`. See
`docs/PHASE0_PROVENANCE_AND_COMMIT_CLASSIFICATION_2026-07-26.md` for provenance/branch discipline.

## What this session did (real, gated, all with actual test runs — not read-only inspection)

### Phase A — finish all five FG operators (COMPLETE for FG; verification 3/5)

1. **Flexible CM** (`cm_lookup_kernels.jl`/`cm_lookup_production.jl`): retrofitted `CMLookupState`'s
   economic (core) block from a dense `obj.H` `BLAS.gemv!` to the shared `economic_forward!`/
   `economic_transpose!`, via the existing `core_cf_ref` box `wrap_moments_with_cm_archB` already
   publishes for the winner-pair Hessian. Falls back safely (byte-identical) when no compressed
   factual is available. Verified D=4 + real D=20/W=80,000, ALL PASS, diffs ≤2.2e-11.
2. **Common Fréchet** (`cm_frechet_lookup_kernels.jl`/`cm_frechet_lookup_production.jl`): fixed the
   inherited **14,066,064 bytes/callback allocation regression** by parameterizing
   `CMFrechetLookupState{O}` (`obj::O` instead of `obj::Any`) — an **8,140x reduction to 1,728
   bytes/callback**, confirmed at BOTH D=4 and real D=20/W=80,000 (not a scale-dependent artifact).
   Also retrofitted the same struct's economic block to the shared operator (same pattern as
   flexible CM). Verified D=4 + real D=20/W=80,000, ALL PASS, diffs ≤3.7e-11.
3. **CM+ZC and Origin-ZC** (inherited opt-in operators from the prior session): ran the
   complete-inner-solve performance A/B the prior session left as the single largest open gap
   (`FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`'s own "NOT MEASURED" list). Real
   D=4 + D=20/W=80,000 results: correctness ALL PASS, speedup 1.001x–1.084x (within the 5%
   criterion), allocation at parity (~1.0002x — not materially reduced, for the same
   "shared post-solve overhead dominates" reason this codebase's OWN `CM_INNER_FG_BACKEND_DEFAULT`
   flip precedent already documents). **Flipped both families' production default to `:operator`**
   (new `ORIGINZC_FG_BACKEND_DEFAULT`/`CM_MEANZC_INNER_FG_BACKEND_DEFAULT` Refs,
   `core_exact_hessian.jl`), eliminating the last dense `obj.H` read from their ordinary FG path.

### Phase A — operator verification (extended 1→2 of 5 families)

Extended the origin-ZC-only `verify_inner_solution_operator_originzc!` proof-of-concept
(inherited) to CM+ZC (`verify_inner_solution_operator_cmmeanzc!`, `operator_verification.jl`),
reusing CM+ZC's own free-function kernels against fresh, independent scratch. D=4 gate, 3 configs,
ALL PASS — operator-recomputed KKT residual agrees with the dense verifier's own
`max_abs_moment_kkt_resid` to ~1e-15, both effectively zero (stationarity independently confirmed).
Flexible-CM and common-Fréchet's own operator verification were NOT attempted this session (see
`HIGHEST_PRIORITY_REMAINING_GAP` below).

### Phase B — no-dense-G proof (scoped)

Resolved the prior session's open `select_G_from_H` question (confirmed a genuine zero-byte
`@view`, not a dense materialization — see `docs/NO_DENSE_G_GLOBAL_RUNTIME_PROOF_2026-07-26.md`
for the full writeup and the real culprit this misdirected investigation toward, the outer
A-gradient's `composite_gradient_at_fast`). Real runtime-counter evidence (not just code reading)
that flexible_cm/cm_plus_zc/zc_only run their ordinary FG path dense-G-free AT THEIR PRODUCTION
DEFAULT (not merely opt-in). Confirmed by grep that Hessian H_EC/H_ER cross-block dense-column
reads remain (a deliberate, documented Phase A scope boundary) — task §10's winner-aware
cross-Hessian rework was NOT attempted this session. The full ~500-file repository audit task §9
literally asks for was also NOT attempted (scope reasons, consistent with the prior session's own
triage).

### Dispatched background work — all three now COMPLETE, each self-pushed to Dropbox

Three large follow-on investigations this session judged too large to fold into the main thread
were dispatched as separate background agents, each in its own isolated git worktree/branch. All
three finished before this report's final revision and are folded in below; their own deliverable
docs are the authoritative source, not this summary.

**1. Hot-path array allocation audit** (worktree `audit-hot-path-array-allocation-2026-07-27`,
commit `e5e5c8f7a14cef3e4ef54c82453a9ea31f9ba997`, pushed to
`dropbox:.../hot_path_array_allocation_audit_2026-07-27`): resolved the `select_G_from_H` question
above (zero-byte view, not a dense materialization); identified the outer A-gradient
(`composite_gradient_at_fast`) as the true source of the large per-call allocation figures other
sessions had misattributed to the FG/verification path — directly motivating investigation #2.

**2. Shared outer A-gradient allocation rework** (worktree `shared-a-gradient-2026-07-27`, branch
`feature/shared-outer-a-gradient-2026-07-27`, HEAD `9af2af3`, pushed to
`dropbox:.../shared_outer_a_gradient_session_2026-07-27`): **partial_merge**. Built
`shared_a_gradient.jl`'s `economic_A_gradient!` shared entry point (bit-identical to reference at
D=4 and real D=20) and wired it as default for ZC-only ONLY (1 of 5 families) — 93.0%/86.3%
allocation reduction at real D=20/W=80,000 (9296MB cold / 4474MB warm unbuffered →
651MB cold / 615MB warm). Flexible-CM/common-Fréchet/CM+ZC remain on the legacy unbuffered path,
honestly reported as not yet wired (not a false "merged_all_families" claim). **Two real bugs
found**: (a) a genuine, previously-undetected correctness bug in `dest_contrib_incremental_o1!`
(`gradient_workspace.jl`) — used origin-count `D` instead of destination-count `Ddest` as an index
stride, silently correct only when `D==Ddest` (i.e. every square-case gate this codebase has ever
run), producing ~75x-magnitude-wrong gradients under the CURRENT real-D20 production default
(`destination_sample=:exclude_row`, D=20/Ddest=19) — **found and fixed** this session; (b) both
`composite_gradient_at_fast_buffered` and `_pooled` (the functions the prior allocation audit's own
"756.5MB pooled" figure was based on) **hard-crash** under today's D=20 production default due to a
hardcoded square reshape — found and disclosed, NOT fixed (out of this agent's time budget). This
means the prior audit's own headline pooled-allocation number cannot currently be reproduced at
all under the actual production default, a materially important correction.

**3. CM basis diagnosis** (worktree `cm-basis-diagnosis-2026-07-27`, branch
`diag/cm-basis-interval-orthonormal-2026-07-27`, commit `92b9d7b`, pushed to
`dropbox:.../cm_basis_diagnosis_session_2026-07-27`): found that two DIFFERENT 2026-07-26 sessions'
own docs claiming "interval basis not attempted" / "orthonormal already the production default"
were BOTH stale — an earlier, already-merged "Continuation 12/13" lineage had already built and
validated the interval basis + its from-scratch Hessian, and the actual wired driver default
(`run_cm_upper_checkpointed`) is `:anchored`, not orthonormal, contradicting those docs. New real
D=20/W=80,000 evidence this session: the D=4-only "orthonormal is 2.6-3.4x better conditioned"
claim does NOT survive to D=20 (anchored/orthonormal are conditioning-equivalent there, <1.7%
either direction); the interval-vs-cumulative D=20 conditioning gap (32.9x-38.1x worse for
interval) was reconfirmed under both contrast schemes. Verdict:
`CM_FEATURE_IMMUTABILITY=pass`, `CM_BASIS_DEFAULT=cumulative` (unchanged),
`ORIGIN_CONTRAST_DEFAULT=inconclusive` (the doc/code mismatch is reported, not resolved),
`INTERVAL_HESSIAN=correct_not_faster`. No production defaults flipped.

## Final verdict (this session's own scope — Phase A/B; see separate docs for Phase C / A-gradient)

```text
ECONOMIC_FG =
    unrestricted:compressed_operator/default (unchanged, Addendum Part A)
    flexible_cm:shared_economic_operator/default (NEW this session)
    common_frechet:shared_economic_operator/available_not_default (CM_FRECHET_INNER_FG_BACKEND_DEFAULT stays :dense_reference pending a clean D=20 perf confirmation -- D=4 shows 1.57x-1.87x speedup, D=20 correctness ALL PASS, D=20 timing run itself crashed/inconclusive this session, see below)
    cm_plus_zc:shared_economic_operator/default (flipped this session)
    zc_only:shared_economic_operator/default (flipped this session)

RESTRICTION_FG =
    unrestricted:not_applicable
    flexible_cm:cm_lookup/default (unchanged, inherited)
    common_frechet:cm_frechet_lookup/available_not_default (unchanged default; economic block now retrofitted)
    cm_plus_zc:operator/default (unchanged, inherited default, now perf-gated this session)
    zc_only:operator/default (unchanged, inherited default, now perf-gated this session)

VERIFICATION =
    unrestricted:dense_reference (not attempted this session)
    flexible_cm:dense_reference (not attempted this session)
    common_frechet:dense_reference (not attempted this session)
    cm_plus_zc:operator/available_not_default (NEW this session, D=4 gated ALL PASS)
    zc_only:operator/available_not_default (inherited from prior session)

HESSIAN_EE = shared winner-pair backend, all 5 families (unchanged, pre-existing)
HESSIAN_ER = family-specific, still dense economic-column-dependent for flexible_cm/common_frechet/cm_plus_zc (confirmed by grep this session, not eliminated); origin-ZC's own exact ZC cross-block unchanged (pre-existing, out of this session's scope)
HESSIAN_RR = family-specific, unchanged (pre-existing)

FULL_G_MATERIALIZATION =
    zero_in_ordinary_FG_callback_for_4_of_5_families (unrestricted, flexible_cm, cm_plus_zc, zc_only all confirmed by real runtime counters; common_frechet has the operator available but not yet default) |
    present_in_Hessian_cross_blocks_by_design (H_EC/H_ER dense-column reads remain, deliberate Phase A scope boundary, task Section 10 not attempted)

CM_FEATURE_IMMUTABILITY = pass (per cm-basis-diagnosis-2026-07-27 agent)
CM_BASIS_DEFAULT = cumulative, unchanged (per cm-basis-diagnosis-2026-07-27 agent; interval confirmed correct but 32.9x-38.1x worse conditioned at real D=20)
ORIGIN_CONTRAST_DEFAULT = inconclusive (per cm-basis-diagnosis-2026-07-27 agent; also found the WIRED driver default is :anchored, contradicting two prior sessions' own docs claiming :orthonormal was already default -- a real doc/code mismatch, not resolved this session)
INTERVAL_HESSIAN = correct_not_faster (per cm-basis-diagnosis-2026-07-27 agent)

A_GRADIENT_BACKEND (per shared-a-gradient-2026-07-27 agent) =
    unrestricted:composite_gradient_at_fast_buffered (unchanged)
    flexible_cm/common_frechet/cm_plus_zc:legacy_unbuffered (not wired this session)
    zc_only:shared_inplace_pooled (WIRED, DEFAULT -- 93.0%/86.3% allocation reduction at real D=20/W=80,000)
A_GRADIENT_CORRECTNESS_BUG_FOUND_AND_FIXED =
    dest_contrib_incremental_o1! used D instead of Ddest as index stride -- silently correct only
    when D==Ddest, ~75x-magnitude-wrong gradients under the CURRENT real-D20 production default
    (destination_sample=:exclude_row, D=20/Ddest=19); fixed this session
A_GRADIENT_CORRECTNESS_BUG_FOUND_NOT_FIXED =
    composite_gradient_at_fast_buffered/_pooled both hard-crash under the current D=20 default
    (hardcoded square reshape) -- the prior allocation audit's own "756.5MB pooled" figure is
    NOT currently reproducible under the real production default at all

PRODUCTION_MERGE = port_ready_not_merged
    (real, gated, committed work on THIS session's own branch plus two further branches from
    dispatched background agents -- see their own commits above; nothing pushed to origin or
    merged to production/fullA-exact this session -- per
    feedback-confirm-before-pushing-to-real-remote-2026-07-25, push/merge requires explicit user
    authorization, not granted this session. The three branches (this session's own
    port/finish-operator-stack-..., feature/shared-outer-a-gradient-2026-07-27,
    diag/cm-basis-interval-orthonormal-2026-07-27) are NOT merged into each other either --
    they share a common ancestor (a69b32d) but diverge from there; reconciling them is a
    next-session task, not attempted here.)

HIGHEST_PRIORITY_REMAINING_GAP =
    the shared-a-gradient agent's bug #2 (composite_gradient_at_fast_buffered/_pooled crash under
    the real D=20 production default) -- this blocks wiring the shared A-gradient backend for the
    4 remaining families and means NO restricted family currently has a working allocation-
    efficient outer gradient at real D=20 scale except ZC-only. This is more consequential than
    common-Frechet's own still-unresolved D=20 perf-bench crash (second priority: reproducible,
    2/2 attempts, same failure point, not diagnosed -- see
    docs/phaseA_item4_frechet_d20_perf_bench_crash_log.txt).
```
