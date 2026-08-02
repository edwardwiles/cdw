# FINAL VERDICT: consolidate the complete reduced formulation into one production-ready, zero-dense implementation

Branch: `integration/profiled-all-five-production-closeout-2026-08-02` (final tip `f67b8db`), based
on `origin/production/fullA-exact` (which already carried both
`zc-hessian-optimized-production-release-2026-08-01` and
`production-hessian-allocation-efficiency-audit-2026-08-02`). Consolidated surgically — not merged
wholesale — from `integration/profiled-restricted-production-ready-2026-08-02@4632111` and
`architecture/profiled-zc-lane-production-2026-08-02@b8cd55d`, plus 14 phases of new work landed via
7 separately-verified branches (Phase 3, 4/5, 7, 10, 12/13, 2b, and the direct-commit Phase 8/9/14
work), each independently re-run and cross-checked by the supervising session before merge — not
accepted from any subagent's self-report alone. See `/bbkinghome/edav/.claude/plans/goofy-plotting-flame.md`
for the full phase-by-phase plan this executed against.

## ZERO_DENSE_INNER (by family)

| Family | Verdict | Evidence |
|---|---|---|
| flexible_CM | **TRUE** | `NO_DENSE_G_COUNTERS` zero deltas across every D4/D20/W100k/W500k gate; `ReducedCMLookupState` |
| common_Frechet | **TRUE** | Same, `ReducedCMFrechetLookupState` |
| ZC_only (origin-ZC) | **TRUE** | Same, `ReducedOriginZCOperatorState`; independently reran, bit-identical |
| CM_plus_ZC (CM+ZC) | **TRUE** | Same, `ReducedCMMeanZCOperatorState`; independently reran W100k, bit-identical (zeta*, kkt_resid exact match) |
| unrestricted | **TRUE** (0 dense-G materializations) with a caveat: its FG callback's own allocation scales linearly with W (3.2MB→16.0MB, W100k→W500k) — it was never ported onto this consolidation's reduced/operator kernels (out of original scope; not a dense-G violation, but not O(1)-allocation like the other 4) |

## COMPLETE_HESSIAN_TRUTH (by family)

All 4 restricted families: Hessian matches ForwardDiff AND an independent diagnostic dense-truth
`G'·Diagonal(S)·G` construction to machine precision (~1e-14 to ~1e-12) at D4, at multiple random
points, not just calibration. Unrestricted: pre-existing coverage confirmed present. flexible_CM and
common_Frechet each got a **new** dedicated dense-truth audit this consolidation (previously only had
ForwardDiff coverage).

## CMZC_SYMMETRY

**PROVEN**, dedicated gate (Phase 3), 4 random points: `H_EM == transpose(H_EM)` bit-exact
(`max|Δ|=0.0`) read directly off `Hfull` before the packing step's symmetrize-by-averaging; complete
`Hfull` symmetric to machine precision (`1.1e-16` worst case) before packing; `H_EM`
production/ForwardDiff ratio exactly `1.000000` at every point (not the historical `0.5` halving
signature). Origin-ZC's analogous `H_EZ` block included as a structural control — its packer has no
averaging step at all, so this bug class is structurally impossible there (confirmed by source read,
not just empirically).

## L50_K3 production gates

**MET.** D=20, Ddest=19, L=50, K_mean=3, K_pair=3, W=100,000 genuine-cold solve (fresh process, no
warm-start reuse) — **all 5 families PASS**, independently re-verified for CM+ZC (the heaviest,
most bug-prone family) with a bit-identical rerun (`nStatus=0`, `zeta*=-0.0118556840`,
`kkt_resid=8.208e-14`, exact match). Six real bugs were found and fixed live reaching this point
(builder kwarg mismatches, a verifier hardcoding the wrong economic-block width for the reduced
path, a stale-counter-snapshot false-fail, a layout-length mismatch, a missing include, and — most
substantively — the per-level ZC ν-target must be `k!` not a flat `1.0` once `K_mean>1`, masked in
every prior gate that used the smaller `K_mean=1` toy config this task explicitly said not to rely
on). Extended to W=500,000 in Phase 14 — also all 5 PASS.

## OPERATOR_ONLY_VERIFICATION

**MET.** All 5 families' `verify_inner_solution_operator_*!` audited against the 7-point checklist
(normalization, primal/dual divergence, reduced economic moments, recovered full factual shares,
France ratio, C/F/Z moments, KKT residuals). Gaps filled: per-block KKT breakdown exposed
individually (previously only implicit in the aggregate residual), a new shared
`verify_recovered_full_factual_shares` reusing existing recovery/winner-comparison helpers. No
production `G` ever materialized in verification. Independently reran the extended origin-ZC gate —
genuine match (kkt_resid_E/mean/pair/france all present, correct, machine precision).

## SCREENS / cache / bank / checkpoints

**MET.** Existing scientific screen logic reused post-decode to full state, gated against the
reduced layout via `ProfiledEconomicMomentLayout`'s own retained/reduced index maps. Separate
exact-cache and dual-bank compatibility keys for the reduced basis (so a `:profiled_destination_scales`
entry can never be silently reused for `:full_gamma_normalized`). New additive `CMCheckpointV11`
schema versions checkpoints with economic parameterization + outer/inner layout digest + backend
manifest + recovery convention, without breaking old checkpoint compatibility.

## COMBINED_OUTER_GRADIENT (by family)

**MET**, genuinely zero-dense (not just dense-FG-with-reduced-Hessian, which is what the first pass
of this work actually validated before being caught and redone): all 4 restricted families'
combined A/gp (+η/ν where applicable) outer gradient verified against complete fixed-dual finite
differences to machine precision, with `NO_DENSE_G_COUNTERS` confirming zero dense materializations
across the ENTIRE solve+gradient evaluation, not just the inner solve. Independently reran CM+ZC's
gate — bit-identical to the claimed numbers. flexible_CM/common_Frechet correctly have no restriction
outer parameter (confirmed by grep — CM marginal/Fréchet level targets are fixed campaign config,
never a KNITRO decision variable), so their combined gradient is A/gp-only, by design not omission.

## PRODUCTION_RUNNER

**MET.** The dropped `profiled_production_outer_runner_2026-08-01.jl`/
`profiled_ab_comparability_and_plumbing_2026-08-01.jl` scaffold (silently lost by an earlier merge,
recovered via `git show` from its origin commit) was adapted to the real, finished family-adapter
surface. Wired to Direct+SR1, screens, exact cache, dual-bank policy, checkpoint/resume,
verification, result export; supports both `:fixed_gp_parameterization_ab` and
`:production_bound_search` modes. D4 gate and a real D20/W=20,000 end-to-end run (both families with
a genuine restriction outer parameter, origin-ZC and CM+ZC) — ALL PASS. One real bug found and
fixed during gating: CM+ZC's D20 config used `threaded_bins=true`, incompatible with the
operator-mode bundle (crashed with a KNITRO callback exception the codebase's own comment calls
"should be unreachable in production") — fixed by matching every other CM+ZC reduced-path gate's
established `threaded_bins=false` convention.

## W500K_PUBLIC_ENTRY

**MET.** Real cold solves via the actual production entry points (`build_cm_meanzc_production_context`
et al.), all 5 families, W=500,000, production dimensions — ALL PASS, zero dense-G materializations.
CM+ZC is the slowest at this scale (351s solve wall) — a real, honest, *expected* cost given the
backend-dispatch gap below, not a new bug.

## Known, real, unresolved gaps (surfaced, not hidden)

1. **Backend dispatch gap (Phase 7).** On the reduced/profiled path, only `blas_syrk` (H_ZZ)
   genuinely dispatches. `drawmajor_v2` (H_EZ) and `draw_chunk_reordered` (H_CZ, CM+ZC only) are
   architecturally unreachable there — the profiled H_EM/H_EZ gathers call their base kernel
   unconditionally, never consulting the backend selector; `hcz_prep_dispatch!` is wired only into
   the threaded twin, incompatible with `profiled_layout`. Confirmed via source read AND independent
   dispatch-counter reruns (REDUCED: blas_syrk 4/0, drawmajor_v2 0/4 fallback; FULL/non-reduced: both
   4/0). **This is the direct cause of every "reduced path is slower" finding below** — not a
   separate mystery.
2. **The reduced path is currently slower than the full/dense path at matched configurations**
   (Phase 14 §3: CM+ZC 38.0s reduced vs 14.3s full-at-recovered; origin-ZC 34.1s vs 9.9s), a direct,
   expected consequence of gap #1. Correctness is not in question (recover-then-resolve agreement at
   the 1e-10 level); performance parity is not yet achieved.
3. **`unrestricted`'s FG callback allocates O(W) per call**, never ported to this consolidation's
   reduced/operator kernels (genuinely out of scope — it was never one of the 4 "restricted"
   families this reduced-formulation project targets).
4. **A small, consistent ~16 bytes/draw linear allocation** in the shared Hessian callback, present
   in *all 5* families atop the dominant O(n²) packed cost — minor (6.4MB at W=500k) but real,
   flagged for a future audit.

## PRODUCTION_RECOMMENDATION

**Correctness-ready, not yet performance-ready to become the production default.** Every
correctness gate this task specified — zero-dense inner solve, CM+ZC Hessian symmetry, production
dimensions (L=50/K=3), operator-only verification, combined outer gradient, screens/cache/bank/
checkpoint versioning, a wired production runner, and a real W=500,000 public-entry smoke — passes
decisively and was independently re-verified, not merely claimed. However, the reduced path is
currently *slower* than the existing dense-Hessian-optimized production path at matched
configurations, purely because of gap #1 above (a real, scoped, fixable follow-up: wire
`drawmajor_v2`/`draw_chunk_reordered` dispatch into the profiled H_EM/H_EZ/H_CZ gathers, mirroring
how the non-reduced path already does it). Recommend: land this as a validated, fully-gated
**opt-in** alternative inner-solve path; do not flip any production default until gap #1 is closed
and a genuine head-to-head timing win is demonstrated at production scale.

```
PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```

No production default was changed and no production campaign was launched at any point in this
consolidation. Nothing was pushed to any remote without this being a locally-committed branch only.
