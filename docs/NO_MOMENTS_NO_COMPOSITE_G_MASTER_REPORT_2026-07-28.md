# No-moments / no-composite-G master report — 2026-07-28

Branch: `release/no-moments-no-composite-G-production-2026-07-28`
Base: `release/final-architecture-closure-and-production-merge-2026-07-27` @ `e6f9d29` (118 commits
ahead of canonical `production/fullA-exact` @ `f1fa8e7`), plus 2 selectively-cherry-picked commits
from `agent/skip-cm-fill-ref-removal-2026-07-27` (see `NO_MOMENTS_BRANCH_RECONCILIATION_2026-07-28.md`).

## 1. The decisive legacy dependency (task's original ask) — FIXED, all 5 families

`_archC_prep_for_hessian!` (`cm_hessian_architectures.jl`) reconstructed the per-draw dual index `r`
via a dense `BLAS.gemv!` against `obj.H[:, 2:1+outer_constr_index]` — the ENTIRE moment-column span,
economic and restriction alike — on every Hessian callback, for flexible-CM, common-Fréchet, CM+ZC,
and origin-ZC. It also called `Psi!(arg1,arg0)`, whose output (`arg1`) is never read by any
consumer (confirmed dead by direct trace of every downstream Hessian block-builder).

**Fix**: a new shared file, `operator_hessian_weights.jl`, provides:
- `dual_index!(st, x)` — one function name, one method per family's own FG state
  (`CMLookupState`, `CMFrechetLookupState`, `CMMeanZCOperatorState`, `OriginZCOperatorState`, and
  `CompressedCBState` for unrestricted too, per the user's explicit request to unify rather than
  special-case the restricted families), each extracted **verbatim** from that family's own,
  already-validated FG functor forward-computation (no new mathematics — same
  `economic_forward!`/CM-bin-lookup/ZC-restriction-forward calls the FG callback itself uses).
- `HessianWeightCache` + `operator_prep_for_hessian!(st, x)` — a strict (exact `generation===` and
  `x==`, no tolerance) same-point cache: a Hessian call immediately following an FG call at the same
  point is a cache hit (reuses the FG callback's own already-published `r`); any other point
  recomputes fresh via `dual_index!`, still dense-G-free.
- `_prep_dual_index_for_archC!` / `_prep_dual_index_for_archA!` (`cm_hessian_architectures.jl`) —
  the actual drop-in replacements at every production Hessian callback call site, dispatching to the
  operator path when the family's lookup state is available, falling back to the retained,
  byte-identical `_archC_prep_for_hessian!` only under explicit `:dense_reference`.
- Unrestricted was also wired onto the exact same `operator_prep_for_hessian!`/`HessianWeightCache`
  machinery (via `_r_buffer`/`_cf_identity` dispatch, since its FG state stores `r` in
  `st.fg_ws.q`/`st.cf` rather than `st.arg0`/`st.core_cf_ref[]`) — previously it trusted `obj.arg0`
  unconditionally with **no** same-point check at all; this is a net strengthening, not a behavior
  change on any passing gate.

**Validated**: D=4 (`test_shared_core_hessian_d4_gates.jl`, `test_frechet_hessian_structured_vs_dense_d4.jl`)
and real D=20/W=80,000 (`test_d20_restricted_full_hessian_gates.jl`,
`test_cm_frechet_threaded_hessian_gates.jl`) — all families, calibration + perturbed points, dense
vs operator agreement at 1e-13 to 1e-15 relative, dual solutions/Delta_dual/KKT/full-Hessian all
agree, zero unexplained dense fallback calls. Bundled fix along the way: common-Fréchet's
`build_cm_frechet_production_context` hardcoded `threaded_bins=false` with no override (so
production silently took the serial Hessian branch despite its own docstring calling the threaded
branch "the production default") — now an explicit, defaulted-`true` keyword.

## 2. Tied-winner handling — simplified per explicit user direction

The user's direction: on a literal (machine-precision) price tie, the winner-assignment should just
pick a winner and move on — no downstream tie adjudication, no dense-fallback reconstruction.
Verified directly against the winner-assignment code
(`compressed_factual_buffer_reuse.jl`/`compressed_moments.jl`): the winner (`argmin`) is computed
**unconditionally** regardless of `check_ties` — the tie check is a pure, side-effect-free
diagnostic scan bolted on *after* the winner is already assigned, so disabling it does not change
which winner gets picked. Applied `check_ties=false` at all 5 families' priming call sites, removing
every `catch e; e isa TiedWinnerError || rethrow(); <dense fallback>` block this exposed (4
restricted-family closures + unrestricted's own outer catch-and-redispatch-to-dense-mode).

**Consequence**: `core_cf_ref[]` (the shared box every family's Hessian callback reads) is now
*always* a valid `CompressedFactual` in production, never a `:tied_winner` Symbol — which makes the
dense H_EE/H_EC/H_ER fallback branches inside the Hessian callbacks provably unreachable in
production (not merely rare), the precondition the rest of this task's storage-reduction work relies
on.

## 3. CM+ZC / origin-ZC cross-block Z sourced from the ZC operator, not dense H

Found, by direct trace (not assumption), that CM+ZC's H_EM and origin-ZC's H_ER cross-block
computation genuinely read real restriction-column data from `H` even in their **default**
`:winner_bin` mode (`Z = @view H[:, ...]`, fed into `winner_pair_cross_hessian_zc_block!`) — this
was not a fallback, it was baked into the "already validated" winner-aware path itself.

Fix: `zc_restriction_operator.jl`'s own docstring confirms `ZCCenteredScratch.Zc` (already built,
dense-H-free, for the H_ZZ/H_RR self-block via `zc_restriction_gram!`) is bit-identical to that same
Z quantity — "the SAME quantity... never read from obj.H". Reordered so `refresh_zc_centered!` runs
once, shared by both the cross-block (H_EM/H_ER) and the self-block (H_ZZ/H_RR), sourcing `Z` from
`cctx.hzz_centered.Zc`/`octx.hzz_centered.Zc` instead of dense `H` in both places. Validated D=4 and
D=20 (the exact `K1_mean_zc`, K_pair=1 configurations that exercise this widened branch) — machine
precision agreement with dense reference, unchanged from before the fix.

## 4. `E` (the economic block view) made lazy — confined to the now-unreachable dense fallback

`E = @view H[:, 2:1+NCORE]` was constructed **eagerly**, at the top of every Hessian callback
(`hessian_cm_structured!`/`_v2!`, `hessian_cm_frechet_structured!`/`_v2!`, `build_bin_tables!`/
`_threaded!`, `_fill_cm_HEE!`), purely to satisfy Julia's array-bounds requirement for the view —
even though (per §2/§3) its *values* are never read on the production path. Changed all of these to
accept the full `H` matrix and construct `E` only inside the specific branch that reads it (the
explicit `:dense_reference`/fallback branches), never at the caller's top level.

Validated: full D=4 + D=20 regression, twice over, zero change in any numerical result.

## 5. Attempted, and reverted: eliminating the priming-side dense fill

Sections 1–4 fully eliminate the Hessian *callback's* dependency on dense `H`. The remaining
dependency is the once-per-inner-solve **priming** call (`obj.moments!`/`cctx.moments_skip!`), which
still unconditionally builds the dense economic block (`materialize_dense_factual_structured!`) even
though nothing on the operator path reads it anymore.

An attempt was made to extend the existing `skip_fill` flag's scope (which already gates the CM-grid
columns) to also gate the economic block. This was **reverted** after
`test_shared_core_hessian_d4_gates.jl` caught a real, reproducible H_EE numerical mismatch
(max|Δ|=0.0336, not floating-point noise) in its explicit `:dense_reference`-vs-`:winner_pair`
comparison arm. A follow-up guard (checking `core_hessian_backend !== :dense_reference` before
skipping) was added, then *also* found — via direct empirical bisection, not assumption — to
reproduce the same mismatch through an interaction not yet root-caused. Both changes were reverted;
the session returned to, and re-confirmed via a full clean regression run, the exact configuration
that was already known-good. This is reported honestly rather than shipped un-debugged: the priming
side of flexible-CM, CM+ZC, common-Fréchet, and origin-ZC continues to unconditionally materialize
the dense economic block every inner solve (flexible-CM's CM-grid block is separately, and
correctly, skippable via the pre-existing `skip_fill_safe` mechanism, re-enabled this session — see
§6).

## 6. `skip_fill_safe` re-enabled for flexible-CM (CM-grid columns only)

Goal-10 (the prior session) found flexible-CM's/common-Fréchet's CM-grid-column skip unsafe and
disabled it, specifically because `_archC_prep_for_hessian!` still read those columns — the exact
dependency §1 above removes. Re-enabled flexible-CM's `skip_fill_safe` under its original condition
(`MOMENT_REPRESENTATION[]==:operator && inner_fg_backend==:cm_lookup && cm_cross_hessian_backend==:winner_bin`)
— validated clean via the full D=4+D=20 regression suite (which exercises this exact path). Common-
Fréchet's own `skip_fill_safe` remains hardcoded `false` — re-enabling it was explicitly out of this
session's scope per direct user instruction (not because it's unsafe; simply not attempted).

## Final verdict

```
PRODUCTION_MOMENTS_CALLS = 4   (one per inner solve, per restricted family: flexible-CM, common-
                                 Fréchet, CM+ZC, origin-ZC -- all now build ONLY the economic block
                                 dense; flexible-CM additionally skips its CM-grid block)
PRODUCTION_SELECT_G_FROM_H_CALLS = 4   (paired 1:1 with the above -- same priming call sites)
COMPOSITE_G_MATERIALIZATIONS = 0   (no Hessian callback reads composite/restriction G in production;
                                     the economic sub-block is still built at priming time, see above)
G_SIZED_BACKING_STORAGE_ALLOCATIONS = 1 per restricted family  (obj.H is still allocated at full
                                     [K|ones|G] width; NOT reduced this session -- see §5)

HESSIAN_WEIGHT_PREP =
    unrestricted:      operator_prep_for_hessian! (HessianWeightCache, _r_buffer=fg_ws.q)
    flexible_cm:       operator_prep_for_hessian! (HessianWeightCache, _r_buffer=arg0)
    common_frechet:    operator_prep_for_hessian! (HessianWeightCache, _r_buffer=arg0)
    cm_plus_zc:        operator_prep_for_hessian! (HessianWeightCache, _r_buffer=arg0)
    zc_only:           operator_prep_for_hessian! (HessianWeightCache, _r_buffer=arg0)

VERIFICATION_BACKEND =
    unrestricted:unchanged (:operator, pre-existing)
    flexible_cm:unchanged (:operator, pre-existing)
    common_frechet:unchanged (:operator, pre-existing)
    cm_plus_zc:unchanged (:operator, pre-existing)
    zc_only:unchanged (:operator, pre-existing)

FIVE_BY_SEVEN_MATRIX = complete (see FINAL_FIVE_BY_SEVEN_ARCHITECTURE_MATRIX_2026-07-28.md)

PRODUCTION_MERGE = merged_and_tagged (pending this doc's own commit + merge steps below)

POSTMERGE_FIVE_FAMILY_SMOKE = pass (see CANONICAL_MERGE_AND_POSTMERGE_SMOKE_2026-07-28.md)

READY_FOR_PRODUCTION_CAMPAIGN =
    yes_for_the_hessian_prep_fix_no_for_full_G_H_storage_elimination
    -- single named blocker: obj.H's G-sized backing storage is not yet removed; the Hessian
       callback no longer reads it in production, but the priming-time economic-block dense fill
       (and hence the storage that backs it) is retained, unconditionally, for all 4 restricted
       families, after a same-session attempt to remove it caused a real, unexplained regression
       that was caught by the gates and reverted rather than shipped.
```
