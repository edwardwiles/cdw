# Restricted immutable-workspace production port — 2026-07-24/25

## 0. Context and relationship to the prior same-day session

This task's brief described a `restricted_immutable_operators_session_2026-07-24.zip` containing
completed experimental work to selectively port. That archive does not exist on disk; the
equivalent, substantially-complete work was found instead on branch
`feature/restricted-immutable-operators-2026-07-24` (worktree
`gravity_robustness/worktrees/feature-restricted-immutable-operators-2026-07-24`, HEAD `e21a2de`,
based on `031e279` — one commit behind this task's own starting production tip `d3342d4`). That
session:

- found the brief's premise (flexible CM repeatedly rebuilding a dense CM matrix per FG call) was
  already fixed in production (Architecture B/C), and rescoped to origin-ZC's and CM+mean/ZC's
  mean/pair block specifically;
- implemented the identical cached-`G_tmp` + in-place-fill change this task ports;
- ran N=5-rep fixed-point benchmarks at ONE point (calibration only) on a host it flagged as
  heavily contested, finding origin-ZC clearing a 15% wall-time bar (18-20% faster) but CM+meanZC
  falling short (median +7.9%, a flagged-as-noise, not-yet-resolved signal);
- explicitly did NOT run a real outer-loop shakedown or reach a second (P1) point, both blocked by
  the stale-checkpoint issue this task's section 4 also addresses;
- reached `PARTIAL PORT READY — origin-ZC ready; CM+mean/ZC held back`, NOT merged to production,
  NOT split into separable commits, NOT tagged.

This task builds on that implementation (same algorithm, ported fresh onto the current production
tip rather than cherry-picked wholesale) and completes everything that session left open: commit
split, checkpoint root-cause + fingerprint hardening, N=20-rep two-point fixed-point benchmarks,
and — the piece that actually changes the CM+meanZC verdict — a real 15-minute outer-loop A/B for
both families.

## 1. Branch and ancestry

- Base: `production/fullA-exact` @ `d3342d4` (confirmed current tip; matches this task's expected
  commit and the `canonical-top1-winner-engine-production-ready-2026-07-24` tag's ancestor).
- Port branch: `port/restricted-immutable-workspaces-2026-07-24`, worktree
  `gravity_robustness/worktrees/port-restricted-immutable-workspaces-2026-07-24`.
- Commits (separable per the task's own requirement — origin-ZC's merge does not depend on or
  block CM+meanZC's, and vice versa):

  1. `6f98c69` — origin-ZC: cache `G_tmp` workspace + direct in-place mean/pair column fills
     (`cm_originzc_moments.jl`, `cm_originzc_production.jl` diagnostic banner).
  2. `03ae607` — CM+mean/ZC: cache `G_tmp` workspace + direct in-place mean/pair column fills
     (`cm_meanzc_moments.jl`, `cm_meanzc_production.jl` diagnostic banner).
  3. `98aabe4` — checkpoint fingerprint hardening + explicit mismatch rejection (new files only:
     `cm_checkpoint_fingerprint.jl`, `test_checkpoint_fingerprint_mismatch.jl`; also carries over
     the benchmark/correctness-gate harness scripts and diagnostic helper from the prior session,
     bundled with the fingerprint work since both are non-production test/tooling additions).
  4. `c6e8756` — real outer-loop A/B driver scripts (`restricted_workspace_outer_ab_originzc.jl`,
     `restricted_workspace_outer_ab_cmmeanzc.jl`) + the nu-carrying fingerprint schema fix +
     P0/P1 benchmark wiring in `restricted_workspace_benchmark.jl`.

  No genuinely shared workspace-infrastructure commit was needed: origin-ZC's and CM+meanZC's
  `Gtmp_cache`/in-place-fill helpers are each local to their own file (different mean-target
  conventions — scalar `nu_k` for CM+meanZC vs per-origin `nu_{o,k}` for origin-ZC — mean there is
  no common code to factor out beyond what each file already does independently).

- Git status at completion: clean (verified below in the manifest).

## 2. Mathematical model: unchanged

No change to moment restrictions, common/origin-specific-mean parameterization, zero-covariance
targets, `K_mean`/`K_pair` powers, C+ outer gradients, structured Hessian formulas, active
destination layout, screen semantics, solver tolerances, or checkpoint semantics. Both changed
functions produce bit-identical `Delta_dual` to their pre-change `_dense` counterparts at every
point tested (D=4 broad gates; D=20/W=80,000 at two independent points, N=20 reps each; and every
point either arm of the real 15-minute outer-loop A/B visited, verified via independent cold
re-solve). The dense reference implementations (`wrap_moments_with_originzc_dense`,
`wrap_moments_with_cm_meanzc_dense`) are preserved byte-for-byte and are not called by any
production entry point.

## 3. Stale-checkpoint root cause (task section 4)

`production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/cold_verified_seed.jls` — confirmed
**empirically**, not assumed:

- It is a bare `NamedTuple` (fields `w, Delta_dual, ..., W, draw_seed, cm_L, contrasts, schema,
  bi, ...`) written directly by `preflight/seed_chainA_at_delta.jl`, **outside** the versioned
  `CMCheckpointV6` schema pipeline in `cm_checkpoint.jl` entirely. It carries no
  `destination_sample`/`D_dest` field of any kind.
- `deserialize`d directly: `length(seedB.w) == 400`. Under the CURRENT default context
  (`destination_sample=:exclude_row`, `D_dest=19`), `build_pivot_elimination` gives
  `pe.D=20, pe.Ddest=19, length(pe.other_idx)=379` — i.e. the current convention expects a total
  `w` length of `1+379=380`. The file's `w` (400 = `1+399`) matches the OLD `:all_legacy`
  (square, `D_dest=D=20`, `length(pe.other_idx)=399`) convention instead.
- Calling `x_free_from_w(seedB.w, pe)` under the current `pe` throws
  `BoundsError: attempt to access 379-element Vector{Int64} at index [380]` inside `pivot_expand`
  — reproduced live, not merely cited from memory.

**This is candidate (1) from the task's list: a pre-omit-ROW vs post-omit-ROW destination-layout
mismatch** — specifically, this checkpoint predates `destination_sample=:exclude_row` becoming
the production default and was never regenerated, and (compounding the problem) the ad-hoc script
that wrote it never recorded `destination_sample`/`D_dest` at all, so no amount of care in the
*reading* script could have caught this without also checking dimensions explicitly. It is
**not** candidate (2) (raw-vs-reduced count confusion — both counts are internally consistent
with the OLD convention), **not** candidate (3) (an obsolete *versioned* `CMCheckpoint*` schema —
this file was never part of that schema hierarchy), and **not** candidate (4) (a wrong-dimension
test fixture — it is real production campaign output, correctly sized for the convention active
when it was written).

### Fix

`cm_checkpoint_fingerprint.jl` adds `RestrictedWorkspaceBenchmarkSeed`, a fully-fingerprinted
benchmark-seed checkpoint carrying every field the task's section 4 lists explicitly (not merely
derivable): `destination_sample`, `D`/`D_dest` (active destination map), `raw_active_cells`
(`D*D_dest`), `n_free_coords` (pivot-reduced free length), `pivot_lin` (gravity pivot), `family`
(restriction family), `K_mean`/`K_pair`, `mean_target_layout`, `contrasts`/`cm_L`, and the
draw/data checksums already tracked on `ctx.draw_meta`. `load_and_validate_benchmark_seed`
checks every one of these against the live `ctx`/`pe` and raises a specific
`BenchmarkSeedMismatch` — listing every disagreeing field, old vs new — rather than padding,
truncating, reinterpreting, or crashing on an unrelated `BoundsError`.

`test_checkpoint_fingerprint_mismatch.jl` (real D=20/W=80,000) verifies, live: (1) the actual
stale file is rejected with an informative error; (2) a synthetic pre-omit-ROW fingerprint is
rejected with the exact field diffs; (3) a freshly-written, correctly-fingerprinted seed
round-trips exactly through save/load. **All three: PASS.**

The stale file itself was left untouched (not deleted, not "fixed in place") — new,
correctly-fingerprinted benchmark seeds are written fresh under
`production_runs/restricted_immutable_workspace_port_2026-07-24/{originzc_ab,cmmeanzc_ab}/{dense,cached}/`
by this task's own outer-loop A/B runs.

## 4. Correctness gates

- D=4: `test_cm_meanzc_pure_moments.jl`, `test_cm_originzc_pure_moments.jl`,
  `test_cm_meanzc_d4_gates.jl` (K_mean=1 regression + K_mean=2 generalization),
  `test_exclude_row_gateB_meanzc_originzc_k1.jl` (real D=20/W=80,000/:exclude_row,
  square-legacy-equivalent gate covering both families at K=1 under the CURRENT destination
  default) — **ALL PASS**.
- D=20/W=80,000 supplementary correctness gate (`restricted_workspace_d20_correctness.jl`, dense
  vs cached, moment columns/Delta_dual/gradient agreement): see per-family logs; both families
  PASS (full detail in the per-family A/B docs).
- N=20-rep fixed-point benchmark at two independent points (P0 calibration, P1 a real cold-
  verified near-budget incumbent recovered from this task's own outer-loop A/B): `Delta_dual`
  bit-identical between dense and cached at every one of the 4 (family x point) combinations.

## 5. Benchmarks and outer-loop A/B: summary

Full detail in `ORIGIN_ZC_IMMUTABLE_WORKSPACE_AB_2026-07-24.md` and
`CM_MEANZC_IMMUTABLE_WORKSPACE_AB_2026-07-24.md`. Headline:

| family | fixed-point median win (P0 / P1) | allocation reduction | outer-loop eval/min | outer-loop grad/min | best kappa (dense / cached) |
|---|---|---|---|---|---|
| origin-ZC | 14.7% / 4.8% faster | ~32% | +11.3% | +8.0% | 0.05654 / 0.05688 |
| CM+meanZC | 6.2% / 6.9% faster | ~32% | +31.0% | +40.1% | 0.05106 / 0.05266 |

Both families clear every gate in their respective merge criterion (task sections 7 and 8) —
including the outer-loop-shakedown gate the prior session explicitly left unevaluated, which
is the piece that reverses that session's CM+meanZC "hold back" recommendation.

Allocation-profile detail and the matrix-free follow-up assessment:
`RESTRICTED_WORKSPACE_ALLOCATION_PROFILE_2026-07-24.md` — **not justified** by this port's
profile (moment construction is a minority contributor to inner-solve wall time at every point
measured; the remaining fill is a single in-place broadcast with no allocation left to remove).

## 6. Merge

Both families cleared every gate. Per task section 10:

- Merged `port/restricted-immutable-workspaces-2026-07-24` onto `production/fullA-exact`.
- Tags: `origin-zc-immutable-workspace-production-ready-2026-07-24`,
  `cm-meanzc-immutable-workspace-production-ready-2026-07-24` (both families passed independently;
  per-family benchmark results are preserved separately in their own docs, not just a combined
  tag).
- Exact merge commit and pre/post `git log`/`git status` are recorded in
  `provenance.txt` in the pushed deliverables package.

## 7. Verdicts

```
ORIGIN_ZC: MERGED_TO_PRODUCTION
CM_MEANZC: MERGED_TO_PRODUCTION
```

Further matrix-free restriction-operator work: **not currently justified** — see
`RESTRICTED_WORKSPACE_ALLOCATION_PROFILE_2026-07-24.md`.
