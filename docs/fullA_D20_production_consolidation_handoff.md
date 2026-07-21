# Full-A_od D=20 production consolidation, timing audit, and δ=5 improvement

Status: IN PROGRESS (written incrementally as phases complete)

Integration branch: `integration/fullA-d20-runtime-delta5`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d20-runtime-delta5`, base `98983bd`
(`diag/fullA-d4-exact`, treated as production).

## 1. Source-stream / commit integration map

| Source | Branch (worktree) | Tip | What it contributed | Status here |
|---|---|---|---|---|
| Canonical rerun | `diag/fullA-d20-canonical-rerun` (`gravity-fullA-d20-canonical-rerun`) | `98983bd` (+ uncommitted) | δ∈{0.1,1,2,5} D=20/W=80,000 frontier, pairwise-screen crash root-cause + uncommitted fix, organic infeasibility traces/checkpoints, first algorithm=2 experiment at δ=5 | Pairwise fix ported+committed here (`afe09a9`). Traces/checkpoints to be reused in Phase D from that worktree directly (not copied). Marked superseded once this branch merges. |
| Warm-start replay | `diag/fullA-d20-warmstart-replay` (`gravity-fullA-d20-warmstart-replay`) | `baec13e` (7 commits past `98983bd`, clean) | `SafeExactCache` (lock-guarded exact-point cache), KKT-proxy-scored successful-dual bank (policy P3), live δ=5 A/B (-29.5% wall, trajectory-dependent) | To be ported in Phase B. Marked superseded once merged. |
| Fast range screen integration | `integration/fullA-fast-range-screen` (`gravity-fullA-fast-range-screen-integration`) | `98983bd` (clean, = base) | Envelope/winning-range/safety-net screens, already merged into base | Base only, nothing to port. |
| QMC investigation | `diag/fullA-d20-qmc-delta1` (`gravity-fullA-d20-qmc-delta1`) | `5882c16` | Validated pseudorandom/Sobol(Cranley-Patterson-shifted)/Halton(Owen-style-digit-scrambled) draw generators for D=20/W=80,000, generator validation report, Stage A-E replicate results | To be ported as an explicit `draw_design` option in Phase A2/B2 (mid-session addendum), default unchanged. Marked superseded once merged. |

## 2. KNITRO version (resolved, no further investigation needed)

`.knitro_env.sh` in every worktree points `KNITRODIR` at `/opt/shared_sw/knitro/14.2.0`, and the
canonical-rerun handoff doc's header says "KNITRO 14.2.0" — **this is wrong for what actually
runs**. `KNITRO.jl`'s `deps/deps.jl` (`~/.julia/packages/KNITRO/LHqTK/deps/deps.jl`) hardcodes:

```julia
const libknitro = "/opt/shared_sw/knitro/13.0.1/lib/libknitro.so"
```

Confirmed live in this worktree:

```
KNITRO.jl loaded library path: /opt/shared_sw/knitro/13.0.1/lib/libknitro.so
```

regardless of `.knitro_env.sh`/module state. **13.0.1 is the real, only, always-loaded version on
this machine.** All timing comparisons in this consolidation use 13.0.1 throughout — there is no
cross-version comparison risk since nothing here has ever actually run on 14.2.0. The
warmstart-replay branch's commit `c408858` already documented this correctly; the canonical-rerun
handoff's "14.2.0" header should be read as a documentation error, not a real solver-version
difference.

## 3. Pairwise-screen crash fix (Phase A, DONE)

Commit `afe09a9` on this branch. See commit message + `test_pairwise_screen_meta_ranged.jl` for
the regression test (6/6 checks pass): constructs a genuinely pairwise-certified-infeasible D=4
point via the same adversarial recipe as `test_infeasibility_screen.jl`'s test 9, confirms
`evaluate_fullA_screened_ranged`'s `:pairwise_certified_infeasible` branch now returns
`worst_o`/`worst_d`, and that the driver's exact field-access pattern
(`(stage=:pairwise, o=screen_meta.worst_o, d=screen_meta.worst_d, ...)`) no longer throws.

## 4. Exact-point cache — TBD (Phase B)

## 5. Successful-dual / KKT-scored bank — TBD (Phase B)

## 6. Draw-design port (pseudorandom / sobol_randomized / halton_scrambled) — TBD (addendum)

## 7. Timing-regression audit — TBD (Phase C)

## 8. Granular δ=1/2/5 profiling — TBD (Phase D)

## 9. Organic -300 certification — TBD (Phase D)

## 10. Staged δ=5 workflow — TBD (Phase E)

## 11. Canonical post-integration A/B — TBD (Phase F)

## 12. Final merge / rollback — TBD (Phase F)
