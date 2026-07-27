# Winner-Aware H_ER Phase — Master Report — 2026-07-27

## Scope and provenance

Continuation of `release/shared-FG-verification-and-A-gradient-2026-07-27` (inherited HEAD `6435d2b`,
77 commits ahead of `production/fullA-exact@f1fa8e7`, confirmed in sync with `origin` at session
start). New release branch `release/winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27`, built
on that HEAD — see `docs/WINNER_AWARE_HER_PHASE_RECONCILIATION_2026-07-27.md` for the full
per-commit classification. Final HEAD after this phase: `76d2f27`, 106 commits ahead of
`production/fullA-exact`. Working tree clean.

Five ordered goals (task brief): (1) wire the validated flexible-CM winner-aware `H_ER` cross-
Hessian into production; (2) extend the same architecture to common-Fréchet, CM+ZC, and ZC-only;
(3) make operator verification the production default for all five families; (4) complete the
common-Fréchet operator-FG default gate; (5) prove ordinary inner solves no longer materialize or
read dense `G`. **All five were reached and closed** — this phase completes every numbered task
section (0, 1, 2, 3, 3.3, 4, 5, 6, 7), not a subset.

## Execution model

Section 2 (flexible-CM, the foundation every other family's H_ER reuses) was done directly, in
sequence, since it establishes the shared architecture (backend-selection field on `CMBinHessCtx`,
persistent `WinnerBinCrossScratch`, "no silent fallback" counter discipline) everything else builds
on. Once gated and committed, Sections 3, 4+5, 6, and 7 were dispatched to four parallel agents,
each in its own git worktree branched from that same commit, each with a self-contained brief
including the exact shared primitives to reuse, the exact dense call sites to replace, and the exact
gate discipline (D=4 then real D=20/W=80,000, machine-precision correctness before any default flip,
no fabricated passes). All four branches were then merged back into the release branch one at a
time, resolving one genuine content conflict (two agents both appended new functions to
`winner_pair_cross_hessian.jl` at the same point — resolved by reconstructing the file from each
branch's own authoritative content via `git show`, verifying valid Julia syntax, then re-running
five independent D=4 gates against the merged result, all of which passed). The session was
interrupted once by a host disconnect partway through the parallel dispatch; all four agents were
still in their initial-setup phase with zero commits at that point, so nothing was lost — the
interrupted worktrees were cleaned up and the four agents relaunched from the same briefs.

## Section-by-section results

### Section 2 — Flexible-CM winner-aware H_ER (done directly)

`hessian_cm_structured!`/`hessian_cm_structured_v2!` gained a `:winner_bin` cross-Hessian backend
(`cctx.cm_cross_hessian_backend`), reusing the already-validated `winner_pair_cross_hessian_fill!`/
`_cm_block!` primitive instead of the dense `CScum` bin-prefix-sum built from `E = H[:,2:1+NCORE]`.
D=4 (36 comparisons) and real D=20/W=80,000/L=50 (8 comparisons) both **ALL PASS** to machine
precision (`max|ΔH|` 6e-16 to 2e-13). `CM_CROSS_HESSIAN_BACKEND_DEFAULT` flipped to `:winner_bin`.
Real D=20: ~7x faster cold (serial), ~2x faster warm (threaded, production default). See
`docs/FLEXIBLE_CM_WINNER_BIN_HER_RELEASE_2026-07-27.md`.

### Section 3 — Common-Fréchet winner-aware H_ER + operator-FG default gate

Reused Section 2's CM-grid winner-bin block directly (no duplication); derived the genuinely new
level-anchor cross block (`H_E,level`) via two new functions (`winner_pair_cross_hessian_colsum!`/
`_esum!`) built on the SAME `WinnerBinCrossScratch`, extended with one new field (`EsumEcon`).
**Two real bugs found and fixed**, both caught by the gates before any commit: (1) a production bug
— the "cf" common-factor column's contribution was silently dropped from `Esum`, caught by the D=4
gate's first run (66/114 failures, cleanly isolated to the `H_E,level` slice); (2) a test-harness bug
— a spurious D=20 "hard point" failure traced to a stale `obj.H` on one side of the comparison, not
a production defect. D=4 (114/114) and real D=20/L=50 (44/44) **ALL PASS** to machine precision
(3.6e-15 to 1.1e-10 against Hessian scale 1590-5198). `CM_FRECHET_CROSS_HESSIAN_BACKEND_DEFAULT`
flipped to `:winner_bin`. The separate operator-FG default gate (Section 3.3) ran the full broader
comparison the task asked for (D=20, both contrasts, 3 points, 24 checks, all correct) but found a
consistent 3.94% allocation regression despite a 1.1-1.2x speedup — correctly did **not** flip that
default, holding this codebase's own established allocation-parity bar. See
`docs/COMMON_FRECHET_WINNER_AWARE_HER_RELEASE_2026-07-27.md` and
`docs/COMMON_FRECHET_OPERATOR_FG_DEFAULT_FINAL_GATE_2026-07-27.md`.

### Sections 4+5 — CM+ZC and origin-ZC winner-aware H_ER via one shared primitive

One new shared function, `winner_pair_cross_hessian_zc_block!`/`_prep!`, computes `H_EZ = E'SZ` for
an arbitrary already-centered restriction matrix `Z` — deliberately takes `Z` pre-centered
(`Φ-1t'`) rather than raw `Φ`+targets, since both wiring sites already have the centered columns in
`obj.H` for free, avoiding threading per-outer-point target state through the Hessian callback.
Wired into CM+ZC's `_fill_cm_HEE!` widening branch (`HEM` block only, `HMM`/H_RR untouched, out of
scope) and origin-ZC's `archA_partitioned_hess_cb_builder` (`HER` block only, `HRR` untouched).
Investigated (and correctly declined, with reasoning) extending CM+ZC's separate CM-grid block to
`:winner_bin` — its widened `E` has no representation in the core-only `wctx`, so the existing
`ncore_core==NCORE` guard correctly stays as-is; documented as genuine future work, not silently
skipped. D=4 and real D=20/W=80,000 gates for both families **ALL PASS** (max|ΔH| 1.3e-15 to 2.0e-12
across both). Both `zc_cross_hessian_backend` defaults flipped to `:winner_bin` on
correctness-plus-no-dense-G grounds (timing was a wash at this small restriction width, honestly
reported, not a regression). Found and honestly flagged (not fixed, out of scope, confirmed
pre-existing at the untouched base commit) an unrelated test-harness bug in
`test_cm_meanzc_d4_gates.jl`. See `docs/CM_MEANZC_WINNER_AWARE_HER_RELEASE_2026-07-27.md` and
`docs/ORIGIN_ZC_WINNER_AWARE_HEZ_RELEASE_2026-07-27.md`.

### Section 6 — Operator verification defaults, all five families

Added an explicit `verification_backend::Symbol` (`:operator`|`:dense_reference`) toggle at each
family's post-solve verification tail, backed by per-family global `Ref`s (kept out of the
Hessian-backend structs two other agents were concurrently editing). Built a shared
`verify_namedtuple_from_operator` helper so both backends produce the identically-shaped result,
letting existing cache/incumbent-admission logic consume either one unchanged. **176/176**
correctness checks passed (draw-level dual index, objective, complete dual gradient, KKT residual,
feasibility/moment residual, status classification, cache admission, incumbent admission, cold
verification) across all 5 families at D=4 and real D=20/W=80,000. **All five defaults flipped to
`:operator`.** Section 6.2 (removing `skip_cm_fill_ref` and similar toggles) was correctly deferred
as explicitly lowest-priority and unsafe to verify independently mid-parallel-dispatch — documented,
not silently dropped. See `docs/FIVE_FAMILY_OPERATOR_VERIFICATION_DEFAULT_RELEASE_2026-07-27.md`.

### Section 7 — Repository-wide dense-G audit + counter completeness

Full classified audit (`PRODUCTION_HOT_PATH`/`PRODUCTION_SETUP_ONLY`/`EXPLICIT_REFERENCE`/
`TEST_ONLY`/`DEAD_CODE`) of every dense-`G` read site across `full_aod_diag/d4_exact/`. Key finding:
the real per-Newton-iterate `:dense_reference` FG callback lives in `cc_algo/PsiObjectiveBundle.jl`
(outside `d4_exact/` entirely) and had zero counter instrumentation before this phase. Measured,
not assumed, the required invariant per family at both D=4 and real D=20 — see the table below.
Zero production files were edited beyond the counter file itself (pure audit, by design, to avoid
conflicting with the three concurrently-editing agents). See
`docs/REMAINING_DENSE_MOMENT_CONSUMERS_2026-07-27.md` and
`docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md`.

## Post-merge integration verification

After merging all four agent branches (one genuine conflict, in `winner_pair_cross_hessian.jl`,
resolved by reconstruction + syntax check, not by discarding either side), five independent D=4
gates were re-run against the fully merged state to confirm the reconciliation didn't silently break
anything: the flexible-CM regression gate, the standalone ZC cross-primitive gate, and the CM+ZC/
origin-ZC/common-Fréchet wiring gates. **All five ALL PASS** on the merged HEAD (`76d2f27`) — see
`docs/key_results_2026-07-27/postmerge_*.log`-equivalent output captured in this report's own
provenance below.

## The rectangular-layout gap, corrected

Every family's own release doc (Sections 2-5) flagged "rectangular (`D≠Ddest`) and non-last-
omitted-destination configurations were not separately exercised" as a known gap, because no
rectangular D=4 CM context builder exists in this repository. **This was re-examined during
reconciliation and found to be substantially overstated**: every single real-D=20 gate run in this
entire phase (by every one of the five agents, without exception) used
`destination_sample=:exclude_row`, which `context_real_d20.jl` documents produces `Ddest = D - 1`
— i.e. every real-D=20 correctness gate in this phase, across all five families, IS a genuinely
rectangular (`D=20, Ddest=19`) configuration, machine-precision-validated. What remains genuinely
untested is narrower than originally flagged: (a) rectangular coverage specifically at the small D=4
scale (redundant with, not more rigorous than, the D=20 coverage already proven), and (b)
*non-last*-omitted-destination specifically (`:exclude_row` always omits the last index by
construction, per that file's own comment). Both are small, explicitly scoped residuals, not an
open correctness question about rectangular layouts in general.

## Deliverables

- `docs/WINNER_AWARE_HER_PHASE_RECONCILIATION_2026-07-27.md`
- `docs/FLEXIBLE_CM_WINNER_BIN_HER_RELEASE_2026-07-27.md`
- `docs/COMMON_FRECHET_WINNER_AWARE_HER_RELEASE_2026-07-27.md`
- `docs/COMMON_FRECHET_OPERATOR_FG_DEFAULT_FINAL_GATE_2026-07-27.md`
- `docs/CM_MEANZC_WINNER_AWARE_HER_RELEASE_2026-07-27.md`
- `docs/ORIGIN_ZC_WINNER_AWARE_HEZ_RELEASE_2026-07-27.md`
- `docs/FIVE_FAMILY_OPERATOR_VERIFICATION_DEFAULT_RELEASE_2026-07-27.md`
- `docs/GLOBAL_NO_DENSE_G_INNER_SOLVE_PROOF_2026-07-27.md`
- `docs/REMAINING_DENSE_MOMENT_CONSUMERS_2026-07-27.md`
- `docs/INNER_STACK_PERFORMANCE_AB_2026-07-27.md` (this session's synthesis of all measured numbers)
- `docs/SHA256_MANIFEST_WINNER_AWARE_HER_2026-07-27.txt`
- `docs/key_results_2026-07-27/` (dense-G audit's raw logs), plus raw gate logs embedded/referenced
  in each family's own release doc

## Final verdict

```text
ECONOMIC_FG_DEFAULT =
    unrestricted:compressed_operator (unchanged this phase)
    flexible_cm:cm_lookup (unchanged this phase)
    common_frechet:dense_reference (broad Section 3.3 gate run this phase, correct+faster but
        allocation regression -- NOT flipped, held to established parity bar)
    cm_plus_zc:operator (unchanged this phase)
    zc_only:operator (unchanged this phase)

RESTRICTION_FG_DEFAULT =
    unrestricted:not_applicable
    flexible_cm:cm_lookup (unchanged this phase)
    common_frechet:cm_frechet_lookup (available, not default -- same Section 3.3 gate)
    cm_plus_zc:operator (unchanged this phase)
    zc_only:operator (unchanged this phase)

VERIFICATION_DEFAULT =
    unrestricted:operator (FLIPPED this phase, 11/11 D4+D20 PASS)
    flexible_cm:operator (FLIPPED this phase, 22/22 D4 + 11/11 D20 PASS)
    common_frechet:operator (FLIPPED this phase, 22/22 D4 + 11/11 D20 PASS)
    cm_plus_zc:operator (FLIPPED this phase, 33/33 D4 + 11/11 D20 PASS)
    zc_only:operator (FLIPPED this phase, 33/33 D4 + 11/11 D20 PASS)

HESSIAN_H_ER =
    unrestricted:not_applicable
    flexible_cm:winner_bin/default (FLIPPED this phase, D4 36/36 + D20 8/8 PASS, ~2-7x faster)
    common_frechet:winner_bin/default (FLIPPED this phase, D4 114/114 + D20 44/44 PASS, ~2-7x faster)
    cm_plus_zc:winner_bin/default (FLIPPED this phase, D4+D20 ALL PASS, timing a wash)
    zc_only:winner_bin/default (FLIPPED this phase, D4+D20 ALL PASS, timing mixed/small)

FULL_G_INNER_SOLVE =
    present_common_frechet_FG_path (the :dense_reference economic FG callback in
        cc_algo/PsiObjectiveBundle.jl, still the default for that one family/backend combination --
        every OTHER production hot path measured this phase shows zero dense-G reads, confirmed by
        instrumented runs, not assumed)

DENSE_CROSS_HESSIAN_CALLS = 0 in every family's default configuration except CM+meanZC's own
    CM-grid block (structurally always dense, ncore_core<NCORE, out of scope per task Section 4)
DENSE_REFERENCE_VERIFICATION_CALLS = 0 in every family's default configuration (all 5 flipped)
SILENT_FALLBACKS = 0 (every fallback path in every backend added this phase is counted, not silent
    -- confirmed by direct code read in every family's own gate, not assumed)

PRODUCTION_MERGE = port_ready_not_merged
    (every commit on release/winner-aware-HER-and-no-dense-G-inner-solve-2026-07-27 is individually
    gated with real command output; branch is clean and 106 commits ahead of
    production/fullA-exact@f1fa8e7; per this project's standing rule, merging to
    production/fullA-exact and pushing to origin requires explicit user authorization, sought but
    not yet granted as of this report)

NEXT_PHASE =
    final_five_family_release_gates_and_canonical_merge (pending user go-ahead to merge to
    production/fullA-exact) |
    remaining_narrow_items: common-Fréchet FG default (deliberately not flipped, correct as-is),
    Section 6.2 toggle removal (deliberately deferred, low-risk to leave), D=4-scale rectangular/
    non-last-omitted-destination test coverage (D=20 rectangular already proven, this is
    redundant-but-not-yet-built small-scale coverage)
```
