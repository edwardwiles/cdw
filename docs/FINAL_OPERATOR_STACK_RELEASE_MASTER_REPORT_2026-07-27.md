# Final Operator Stack Release — Master Report — 2026-07-27 (continuation session)

## What this session was asked to do

Continue `release/shared-FG-verification-and-A-gradient-2026-07-27` (inherited HEAD `4fb9839`,
70 commits ahead of `production/fullA-exact@f1fa8e7`) through three ordered phases: (A) finish the
mature shared-FG/verification/A-gradient work that release left honestly incomplete; (B) implement
winner-aware `H_ER` cross-Hessian blocks and eliminate dense economic moment columns; (C) run final
five-family gates, merge passing commits, and produce a final 5x7 status matrix.

## What this session actually did

**Phase A: substantially completed.** Five new commits (`4367abe`..`a4e8ece`) on top of the
inherited branch:

1. Wired the already-existing, already-rectangular-generalized (but never-connected)
   `LFixBaseWorkspace`/`build_lfix_base_cache!` into `economic_A_gradient!`'s plain (`cache=nothing`)
   path. Real D=20/W=80,000 warm allocation: **614.83 MB -> 27.30 MB (95.6% reduction)**, exceeding
   the task's own >=80% target, bit-identical to the reference gradient.
2. Wired common-Frechet's `cm_frechet_production_gradient` onto the shared
   `economic_A_gradient!` backend (default `:shared_inplace_pooled`, `:legacy_unbuffered` kept as
   explicit reference) -- D=4 bit-identical, ALL PASS.
3. Wired the unrestricted family onto `economic_A_gradient!` via a new opt-in
   `price_cache_backend=:shared` in the real production driver
   (`c10_d20_production_driver.jl`) -- exercised through the actual `run_polish_checkpointed`
   KNITRO driver at real D=20/W=80,000, kappa matches `:buffered` exactly (diff=0.0), no crashes.
   Unrestricted's own default (`:cplus`) is unchanged; this is additive/opt-in only.
4. Built `verify_inner_solution_operator!` for the 3 families that had none (flexible-CM,
   unrestricted, common-Frechet) -- closing the largest gap the inherited release flagged in its
   own `FIVE_FAMILY_OPERATOR_VERIFICATION_RELEASE_2026-07-27.md`. All 5 families now have an
   operator verification function; the flexible-CM and unrestricted ones are gated at D=4 AND real
   D=20 (unrestricted to machine precision, ~1e-12 to ~1e-15); common-Frechet's is gated at both
   scales too (the genuinely new piece -- the level-anchor block had no prior verification-side
   precedent at all).
5. Ran the task's own worker sweep (1,4,8,10,20) for `economic_A_gradient!` at real D=20: winner is
   **20 threads** (3.83s vs 15.59s serial, ~4.1x), consistent with this project's existing H_EE
   worker-count convention.

**Deferred within Phase A** (not reached, documented rather than skipped silently): the broader
3-point (calib/near-delta=1/hard-point) common-Frechet dense-vs-operator gate the task explicitly
asks for (the inherited branch's own 2-point gate, `COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md`,
is the most recent evidence and was not extended this session); CM+ZC/ZC-only's missing
complete-inner-solve + short public-driver performance gates; `skip_cm_fill_ref` project-wide
removal (still gated on operator verification being wired as a *default*, not merely existing);
flipping `verification_backend=operator` as a default for any family.

**Phase B: one family's primitive built and gated, not wired into production.** Implemented
`winner_pair_cross_hessian.jl` (`H_ER = Q'SR - pi*(nu'SR)` for flexible-CM's CM-grid restriction),
reusing the already-validated `WinnerPairHessCtx` (H_EE's own winner-pair backend) rather than
re-deriving `E`'s decomposition from scratch. **Two real bugs were found and fixed during
derivation** (a row-index off-by-one from an all-ones `H` column, and a missing accumulator for
the "cf"/common-factor column) -- both caught by the dense-vs-new gate before any commit, both
would have been large, visible errors (0.31 and 0.18 against a matrix scale of ~1) if shipped
unvalidated. Final gates: D=4 (8 configs) and real D=20/W=80,000/L=50 (both contrasts) ALL PASS to
machine precision. See `WINNER_AWARE_H_ER_FINAL_PORT_2026-07-27.md` for full detail.

**NOT done in Phase B:** wiring the validated primitive into `hessian_cm_structured!` as an actual
backend (it exists standalone, gated against the dense reference, but nothing in production calls
it yet); extending to the other 3 families (common-Frechet, CM+ZC, ZC-only) -- each needs either a
direct reuse of the CM-grid half (common-Frechet, CM+ZC) or a wholly new derivation (the mean/pair
ZC cross block, the level-anchor cross column); global no-dense-G runtime counters.

**Phase C: NOT reached.** No final five-family/seven-condition gate matrix was run with real
KNITRO across all five families at calibration/near-delta=1/hard-point/upper-driver/lower-driver/
exact-cache/checkpoint-resume. No merge to `production/fullA-exact`, no fast-forward, no tags, no
push. This is a genuine, load-bearing gap, not an oversight -- see "Why Phase C was not attempted"
below.

## Why Phase C was not attempted this session

Phase C's own scope (5 families x 7+ conditions, each requiring a real D=4 AND real D=20/W=80,000
KNITRO run, plus process-group hard-kill/resume tests, plus a full merge-and-tag sequence) is, by
this project's own historical pace (every comparably-sized single piece of this release --
the L-fix cache wiring, the operator verification build-out, the H_EC cross-Hessian derivation --
consumed a meaningful fraction of one full session each), realistically multiple additional
sessions of work, not a tail end of this one. Given real, gated, honest partial progress is this
project's own consistently established standard (every single deliverable doc read this session
follows the same "report exactly what passed, exactly what didn't, no fabrication" pattern),
attempting to compress Phase C into the remaining time in this session would have meant either
fabricating gate results that were never actually run, or rushing real KNITRO gates without the
time to properly investigate any failure -- both worse outcomes than an honest, clearly-scoped
handoff. `production/fullA-exact` remains untouched; nothing was pushed to `origin`.

## Commits this session (all local to the release branch, not merged, not pushed)

```text
4367abe Wire persistent LFixBaseWorkspace into economic_A_gradient!, killing the 615MB warm alloc
823bf99 Wire common-Frechet onto the shared economic_A_gradient! backend (default)
d7d2d2f Wire unrestricted family onto economic_A_gradient! (opt-in price_cache_backend=:shared)
a4e8ece Build operator verification for the 3 remaining families: flexible-CM, unrestricted, common-Frechet
ab279a9 Phase B: winner-aware H_EC cross-Hessian for flexible-CM (H_ER = Q'SR - pi*(nu'SR))
ed5d197 Phase B: real D=20/L=50 validation of winner-bin H_EC + final port status doc
```

Branch: `release/shared-FG-verification-and-A-gradient-2026-07-27`, 6 commits ahead of the
inherited `4fb9839` (which was itself 70 commits ahead of `production/fullA-exact@f1fa8e7`) -- 76
commits total ahead of base. Working tree clean after each commit; nothing staged or uncommitted.

## Final verdict

```text
ECONOMIC_MOMENT_BUILDER =
    unrestricted:compressed_operator (Addendum Part A, unchanged this session)
    flexible_cm:shared_economic_operator (cm_lookup, unchanged this session)
    common_frechet:shared_economic_operator_available_not_default (dense_reference remains default;
        :cm_frechet_lookup correct+gated, see COMMON_FRECHET_FG_D20_FINAL_GATE_2026-07-27.md)
    cm_plus_zc:shared_economic_operator (operator backend, unchanged this session)
    zc_only:shared_economic_operator (operator backend, unchanged this session)

ECONOMIC_FG_DEFAULT = unchanged this session for all 5 families (see above)
RESTRICTION_FG_DEFAULT = unchanged this session for all 5 families

VERIFICATION_DEFAULT =
    unrestricted:dense_reference (operator verification NOW EXISTS this session, not wired default)
    flexible_cm:dense_reference (operator verification NOW EXISTS this session, not wired default)
    common_frechet:dense_reference (operator verification NOW EXISTS this session, not wired default)
    cm_plus_zc:dense_reference (operator verification exists, inherited, not wired default)
    zc_only:dense_reference (operator verification exists, inherited, not wired default)

A_GRADIENT_DEFAULT =
    unrestricted:composite_gradient_at_fast_buffered (legacy default UNCHANGED; :shared now exists
        as opt-in via price_cache_backend=:shared, real-driver-gated this session)
    flexible_cm:shared_inplace_pooled (economic_A_gradient!, WIRED prior session, unchanged)
    common_frechet:shared_inplace_pooled (economic_A_gradient!, WIRED this session)
    cm_plus_zc:shared_inplace_pooled (economic_A_gradient!, WIRED prior session, unchanged)
    zc_only:shared_inplace_pooled (economic_A_gradient!, inherited, unchanged)

A_GRADIENT_WARM_D20_ALLOCATION = 27.30 MB (was 614.83 MB; 95.6% reduction, WIRED this session)
A_GRADIENT_WORKERS = 20 (worker sweep 1/4/8/10/20 run this session; 20 is the real, gated winner)

HESSIAN_H_EE = shared winner-pair backend, all 4 families with a core block, UNCHANGED this session
HESSIAN_H_ER =
    flexible_cm: primitive_built_and_gated_not_wired (D=4 8/8 + D=20/L=50 2/2 PASS, machine precision)
    common_frechet: not_attempted
    cm_plus_zc: not_attempted
    zc_only: not_attempted
HESSIAN_H_RR = untouched this session (out of this task's H_EE/H_ER scope; dense, unchanged)

FULL_G_MATERIALIZATION =
    present_flexible_cm,common_frechet,cm_plus_zc (H_EC/H_ER cross-block dense-column reads
    remain in PRODUCTION for all but the newly-gated-but-unwired flexible-CM primitive -- see
    REMAINING_DENSE_G_CONSUMERS_HANDOFF_2026-07-27.md, confirmed still accurate this session)

DUPLICATE_ECONOMIC_STATE_BUILDS = 0 (no new duplicate-build sites found or introduced this session)
SILENT_FALLBACKS = 0 (no new silent fallback introduced; both Phase B bugs were CAUGHT by gates,
    never shipped)

PRODUCTION_MERGE = port_ready_not_merged
    (every commit on this branch is individually gated with real command output; nothing pushed to
    origin or merged into production/fullA-exact -- per this project's own standing rule, that
    requires explicit user authorization not granted/sought this session, AND Phase C's own final
    gate matrix was never run, which this project's own established practice treats as a hard
    prerequisite for a merge decision, not merely a formality)

HIGHEST_PRIORITY_REMAINING_GAP = wire winner_pair_cross_hessian_cm_block! into
    hessian_cm_structured! as a real (even if opt-in, non-default) backend -- the primitive exists
    and is machine-precision-validated at both D=4 and real D=20, but nothing in production calls
    it, so Phase B's own "eliminate dense economic moment columns" ask remains unrealized for
    flexible-CM (and unattempted for the other 3 families) until that wiring step happens
```
