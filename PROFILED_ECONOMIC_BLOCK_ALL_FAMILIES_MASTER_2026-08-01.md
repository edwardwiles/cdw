# Profiled economic block, all-families port — master status (2026-08-01)

## Branch / commits

Branch: `architecture/profiled-economic-block-all-families-2026-08-01`
Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-economic-block-all-families-2026-08-01`
Base: `production/fullA-exact@cd17235` (confirmed direct ancestor, 0 divergence) + the diagnostic
branch's 32 commits of validated profiled-unrestricted work.

```
774a223 Fix pre-existing stale include lists / moment_representation defaults in 4 existing tests
6741ec1 Surgically amend H_EF (colsum!/esum!) for the profiled economic layout
b2bb418 Surgically amend H_EZ (winner_pair_cross_hessian_zc_block!) for the profiled economic layout
10b5940 Surgically amend H_EC (winner_pair_cross_hessian_cm_block!) for the profiled economic layout
c49f843 Design: exact profiled cross-block formulas mapped onto current H_EC/H_EF/H_EZ code
a06fcc4 Snapshot source branch + audit existing optimized economic/cross-block functions
```

## Evidence context this port sits inside

The archive that motivated this task (`profiled_scales_unrestricted_outer_ab_2026-08-01.zip`,
`00_READ_FIRST_CORRECTION.md`) states its own verdict as `PORT_TO_RESTRICTED_FAMILIES =
insufficient_evidence` for the unrestricted-only profiled result: a real but small 3.85-7.99%
outer-loop edge, from 3 points, one seed each, upper-bound direction only. This session's work is
evidence-gathering infrastructure toward resolving that verdict (prove the shared cross-block
formulas are correct, build toward five-family equivalence gates), not a claim that the case for
merging is already made. `PRODUCTION_DEFAULT_CHANGED = false`, `CAMPAIGN_LAUNCHED = false` hold
throughout — no production code changed, no campaign run.

## What is DONE and VALIDATED this session

**1. Source snapshot** (`PROFILED_ALL_FAMILY_SOURCE_SNAPSHOT_2026-08-01.md`) — confirmed clean
ancestry, new isolated worktree/branch, explicitly avoided the one other active job on this
machine (`campaign-sigma3-w500k-fullA-10x10-launch-2026-08-01`, a `profile_full_table_2026-08-01.jl`
run in a separate worktree).

**2. Audit** (`EXISTING_OPTIMIZED_ECONOMIC_CROSS_BLOCK_MAP_2026-08-01.md`) — confirmed H_EE and the
economic FG (`fill_core_hessian_upper!`, `economic_forward!`/`economic_transpose!`,
`core_exact_hessian.jl`/`economic_operator.jl`) are *already* one shared, cf-agnostic
implementation across all five families — **no change needed there**. Pinpointed the exact three
cross-block functions requiring surgical amendment, all in `winner_pair_cross_hessian.jl`:
`winner_pair_cross_hessian_cm_block!` (H_EC), `winner_pair_cross_hessian_colsum!`/`_esum!` (H_EF),
`winner_pair_cross_hessian_zc_block!`/`_threaded!` (H_EZ). Also flagged several 2026-07-28 docs
describing Fréchet-CM duplication as now stale (fixed via thin-wrapper delegation).

**3. Formula design** (`PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md`) — resolved the one genuine
ambiguity in the mission's abstract math: the destination-specific correction total `T^R_{d,k}`
must be built from `cf.wval` (raw, always-defined per-draw destination value), **not** by summing
kappa0-scaled winner buckets over origins — the latter hits an undefined-kappa0 problem at the
omitted anchor cell. Derived by cross-checking against `reduced_homogeneous_hessian_2026-08-01.jl`,
the already-validated H_EE reduction a prior session in this branch built, so the new cross-block
code is consistent with that convention rather than inventing a new one. Also resolved (by directly
reading `cm_frechet_hessian.jl`, not assuming) that H_EF's level restriction shares the identical
`(x,l)` CM-grid as H_EC — no separate level-native table needed.

**4-6. Surgical amendments — H_EC, H_EZ, H_EF**, each following the identical, minimal-diff pattern:
- New accumulator(s) added to scratch structs (`WinnerBinCrossScratch`: `MTab`/`MCScum`/`T0_slot`/
  `MSumX`; `WinnerZCCrossScratch`: `SnuWval`/`TZ_buf`), filled **unconditionally but cheaply**
  inside the *existing* draw loops (one extra multiply-add per iteration — confirmed no new O(W)
  pass added anywhere).
- Each amended function (`winner_pair_cross_hessian_cm_block!`, `_zc_block!`, `_colsum!`, `_esum!`)
  gained a `use_profiled_correction::Bool = false` keyword. `false` (the default, every existing
  production caller) is **byte-for-byte unchanged** — confirmed to `max|Δ| = 0.0` exactly in the
  H_EC gate, and to machine precision (~1e-10 to ~1e-15) in the H_EZ/H_EF gates.
- The France/cf row's profiled correction is now also implemented (commit `1afe1be`, follow-up to
  the original three amendments): `build_winner_pair_ctx` gained an optional `bi_slot::Int=0`
  keyword (`dest_slot(ctx, ctx.bi)`) — when supplied, every amended function applies the identical
  profiled correction to the France row that every other row already gets; when omitted (default,
  every pre-existing caller), the France row falls back to the old destination-independent
  correction exactly as before, so nothing existing changes. Validated to ~1e-12–1e-16 against
  independent brute-force references for H_EC/H_EZ/H_EF, and bit-identical threaded-vs-serial for
  H_EZ's threaded twin (`test_profiled_france_row_d4_2026-08-01.jl`,
  `test_profiled_hez_threaded_d4_2026-08-01.jl` re-run with `bi_slot` supplied). Along the way, a
  genuine (if previously harmless) out-of-bounds `@inbounds` read was found and fixed in H_EC's and
  H_EF's generic per-column loops (they iterate the cf column too, and were reading
  `target_slot[jcf]`'s sentinel `0` as an array index before this fix).
- The "keep" (winner) term in every block is **completely untouched** — only the target-correction
  term was ever edited, per the mission's own instruction.

**Gate results (all real KNITRO D4 solves, this session's own new tests)**:

| Block | Test file | Combos | Regression (false) | New path (true) |
|---|---|---|---|---|
| H_EC | `test_profiled_hec_correction_d4_2026-08-01.jl` | 2 contrasts × 2 L × 2 pts = 8 | max|Δ|=0.0 exact, 8/8 | max|Δ|~1e-15 vs independent brute force, 8/8 |
| H_EZ | `test_profiled_hez_correction_d4_2026-08-01.jl` | CM+ZC, 3 pts | max|Δ|~1e-15, 3/3 | max|Δ|~1e-15 vs independent brute force, 3/3 |
| H_EF | `test_profiled_hef_correction_d4_2026-08-01.jl` | 2 contrasts × 2 L × 2 pts, colsum!+esum! = 16 | max|Δ|~1e-10, 16/16 | max|Δ|~1e-10 vs independent brute force, 16/16 |

Every "independent brute force" reference was built directly from raw `cf.winner`/`cf.wval`/`Bidx`
data, re-deriving both the unchanged "keep" term and the new correction term from scratch — not
reusing `MCScum`/`MSumX`/`TZ_buf`/`T0_slot` at all, so a bug shared between the check and the code
under test cannot hide. Two real test bugs were found and fixed during this process (an off-by-one
row index in the H_EZ test, a missing `kappa0[j]` scale factor in the H_EF test) — both confirmed,
via debug scripts kept in the repo for provenance, to be bugs in the *test*, not in the
`winner_pair_cross_hessian.jl` implementation.

**7 (partial). Pre-existing test suite regression check** — two of the four directly-relevant
pre-existing (unmodified-logic) test files were unblocked from a *pre-existing*, unrelated stale-include
bug (`HCZ_PREP_BACKEND_DEFAULT`/`resolve_cross_hessian_workers_default` missing from their include
lists — confirmed via a control run of the untouched originals *before* any of this session's edits)
and a *pre-existing*, unrelated `moment_representation` production-default drift
(`:dense_reference`→`:operator`), then re-run:
- `test_winner_pair_cross_hessian_cm_d4.jl`: **ALL PASS, 20/20** (H_EC via the real production
  `hessian_cm_structured!` callback, dense-vs-winner-bin comparison).
- `test_winner_pair_cross_hessian_zc_d4.jl`: **ALL PASS, 28/28** (H_EZ, both CM+ZC and origin-ZC
  families, via the real production callback).
- `test_frechet_winner_bin_her_wiring_d4.jl` / `test_threaded_cross_hessian_d4.jl`: hit a
  *different, deeper* pre-existing drift (specific to `cm_frechet_cplus.jl`'s dense-mode
  construction and an `obj.moments!` dependency in the H_ZZ BLAS gate) — not fixed this session,
  out of scope; H_EF's correctness is independently covered by this session's own new test instead.

**8. Threaded H_EZ twin** (`winner_pair_cross_hessian_zc_block_threaded!`, `threaded_cross_hessian.jl`)
— amended with the identical `use_profiled_correction` keyword and `TZ` gemm as the serial version
(commit `81b6225`). D4-gated (`test_profiled_hez_threaded_d4_2026-08-01.jl`, 3 points ×
worker_counts {1,2,4} = 9 combos each) for **bit-identical** agreement (`max|Δ|=0.0`, not just
close) against the already-validated serial path, both under the new `use_profiled_correction=true`
and the regression `=false` path — matching this file's own stated design guarantee exactly. ALL
PASS, 9/9 + 9/9.

## What is NOT done (honest accounting against the mission's 23 sections / 12 commits)

- **Outer A/gp gradient sharing (mission §13, commit 6)**: not started. The profiled machinery
  (`ProfiledLFixCache`/`build_profiled_lfix_cache`/`profiled_composite_gradient_at_incremental`,
  `profiled_lfix_incremental_2026-08-01.jl`) exists for unrestricted only; making it generic over a
  family's fixed-restriction contribution (mirroring `build_lfix_base_cache_cm`/`_originzc`/
  `_cm_frechet`'s existing pattern for the OLD cache) is real remaining work, scoped precisely in
  `PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §6 but not implemented.
- **Wiring `use_profiled_correction=true` through the higher-level orchestrators** (commits 7-9):
  `hessian_cm_structured!`/`_v2!`, `archA_partitioned_hess_cb_builder`, and the per-family callback
  builders do not yet pass the new keyword through — this session validated the *primitives*
  directly (standalone-primitive style, matching how `winner_pair_cross_hessian_cm_block!` etc. were
  themselves originally gated per their own docstrings), not the full wired KNITRO Hessian callback.
- **A genuine reduced/anchor-omitting `CompressedFactual` for the four restricted families**: the
  profiled economic moment layout (`ProfiledEconomicMomentLayout`, anchor omission) exists only for
  unrestricted so far. Every D4 gate in this session used the OLD/full economic layout (all D·Ddest
  bilateral columns retained) to validate the `T^R_{d,k}` *mechanism* — this is a real, valid,
  necessary check, but it is not the same as a restricted family actually running on the reduced
  layout end-to-end. Building that plumbing is the largest remaining piece of work.
- **D20 operator-reference gates** (mission §16, commit 8): not run. Requires the wiring above.
- **Five-family inner-solve equivalence** (mission §17, commit 9): not run.
- **All-family outer-gradient gates** (mission §18, commit 10): not run.
- **Performance/allocation profiling old vs new** (mission §19, commit 11): not run. (The additive
  scratch fields' cost was reasoned about at design time — O(W·Ddest·D) added to an existing
  O(W·Ddest·D) loop, i.e. same complexity class, not a new pass — but not empirically measured.)
- **W500k public-entry smokes per family** (mission §21, commit 12): not run.

## Verdict block

```text
SHARED_ECONOMIC_LAYOUT =
    all_families   # H_EE/FG already shared and cf-agnostic, confirmed by audit; no new duplication introduced

SHARED_ECONOMIC_FG =
    all_families_efficient   # unchanged this session -- already true per audit

H_EE =
    existing_optimized_reused   # untouched, per mission instruction

H_EC =
    existing_optimized_function_surgically_amended   # winner_pair_cross_hessian_cm_block!, D4-validated

H_EF =
    existing_optimized_function_surgically_amended   # colsum!/esum!, D4-validated

H_EZ =
    existing_optimized_function_surgically_amended   # winner_pair_cross_hessian_zc_block!, D4-validated; threaded twin _threaded! ALSO amended, D4-validated bit-identical to serial (commit 81b6225)

RESTRICTION_ONLY_BLOCKS_CHANGED =
    none

CROSS_BLOCK_FORMULA_GATE =
    D4: pass (H_EC/H_EZ/H_EF primitives, standalone, real KNITRO, independent brute-force references)
    D20: not_run

ALL_FAMILY_INNER_EQUIVALENCE =
    unrestricted: not_run_this_session   # prior sessions' D4/D20 gates on the diagnostic branch stand, not re-verified here
    flexible_CM: not_run
    common_Frechet: not_run
    ZC_only: not_run
    CM_plus_ZC: not_run

SHARED_A_GP_GRADIENT =
    not_started   # scoped in design doc §6, no code written

PERFORMANCE =
    FG: not_measured
    H_EE: not_measured
    H_EC: not_measured (reasoned: same complexity class, no new O(W) pass)
    H_EF: not_measured (reasoned: same complexity class, no new O(W) pass)
    H_EZ: not_measured (reasoned: one additional BLAS gemm per call, cheap)
    outer_gradient: not_applicable (not started)

W500K_PUBLIC_ENTRY_SMOKE =
    not_run

PRODUCTION_DEFAULT_CHANGED = false

MERGE_STATUS =
    not_ready_full_family_reduced_layout_wiring_and_D20_gates_incomplete

CAMPAIGN_LAUNCHED = false
```

## Recommended next steps (in order)

1. **Build the reduced/anchor-omitting `CompressedFactual` construction for the restricted
   families** — the single largest remaining piece, and the prerequisite for everything else on the
   list. Follow `ProfiledEconomicMomentLayout`'s existing conventions (already family-agnostic in
   its own design) rather than writing family-specific variants.
2. **Thread `use_profiled_correction` through `hessian_cm_structured!`/`_v2!` and
   `archA_partitioned_hess_cb_builder`** — mechanical once (1) exists, since the primitives
   themselves are already validated.
3. **D4 dense-reference gates per mission §15**, now against a genuine reduced-layout `cf` for each
   restricted family (not the OLD/full layout this session's gates used).
4. Only after (1)-(3): D20 operator-reference gates, inner-solve equivalence, outer-gradient
   sharing, performance profiling, W500k smokes — in that order, per the mission's own commit
   sequence.

Given the archive's own `insufficient_evidence` verdict on the unrestricted-only result, it may be
worth pausing after step 3's D4 equivalence gates to reassess whether the remaining, expensive
D20/W500k campaign work is worth running before more outer-loop A/B evidence (beyond the 3 points
already gathered for unrestricted) justifies it — a judgment call for the user, not resolved here.
