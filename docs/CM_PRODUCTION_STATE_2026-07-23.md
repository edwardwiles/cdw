# CM production state — 2026-07-23 (CM-C+ promotion)

Canonical record of the CM-C+ gradient-backend integration into `production/fullA-exact`,
superseding `docs/CM_PRODUCTION_STATE_2026-07-22.md` for anything CM-gradient-backend related.
Everything in that prior document about the CM driver, checkpoint mechanics, and cache absence is
otherwise still accurate.

## Release identity

- Integration branch: `integration/cm-cplus-production-2026-07-23`, fast-forwarded into
  `production/fullA-exact` on both `cdw` (canonical remote) and local.
- Release tag: `cm-cplus-production-ready-2026-07-23` (annotated), pointing at the same commit as
  `production/fullA-exact` after this integration. Use `git rev-parse production/fullA-exact` (or
  the tag) for the authoritative current hash — not hand-copied here, per this repo's existing
  convention of treating `git rev-parse`/`git ls-remote` as the source of truth over a hardcoded
  value that can go stale.
- Archival tag `archive/cm-cplus-tested-evidence-2026-07-23` preserves the full tested-evidence
  branch tip (`perf/fullA-cm-cplus-statecache-overnight-2026-07-22` @ `e347ca4`) after that branch
  itself was deleted.
- Julia: 1.12.6 (juliaup; never `/opt/shared_sw`). KNITRO: 13.0.1 (`ZIENA_LICENSE`/`KNITRODIR` via
  `.knitro_env.sh`), licensed on `demand.mit.edu`.

## Default backend / fallback

- `cm_gradient_backend = :cplus` is now the **production default** for
  `run_cm_upper_checkpointed` (`full_aod_diag/d4_exact/cm_checkpoint.jl`).
- `:reference` remains fully supported as an explicit, documented fallback/validation backend
  (byte-identical to the pre-C+ code path).
- Unknown backend symbols fail immediately (`cm_gradient_backend in (:reference, :cplus)` guard).
- The resolved backend is printed at process startup (`cm_gradient_backend=:cplus (production
  default)` / `:reference (fallback/validation backend)`) and written to a `<label>_gradient_backend.txt`
  sidecar and into the checkpoint's own persisted schema (`CMCheckpointV3.cm_gradient_backend`).
- `cm_production_stage_runner.jl` and `scripts/cm_production_supervisor.sh` read
  `CM_GRADIENT_BACKEND` (default `cplus`) and `CM_ALLOW_BACKEND_SWITCH` (default off) from the
  environment; both are logged into each stage's `run_meta.txt`.

## Checkpoint backend-provenance policy

- Checkpoint schema bumped 2 → 3 (`CMCheckpointV3`); legacy schema-1/2 files remain readable
  forever via `CMCheckpoint` (permanently frozen layout) + `upgrade_schema2` (fills
  `cm_gradient_backend = :reference`, provably correct for every schema-2 file that exists, since
  `:reference` was the only value the real 2026-07-22 campaign's stage runner ever passed).
- Resuming under a different `cm_gradient_backend` than the checkpoint was written with is a
  **hard error** unless `allow_backend_switch=true` is passed explicitly.
- An explicit, audited switch: logs the switch, writes a `<label>_backend_switch_audit.txt`,
  clears the persisted `bandwidth_cache` (tuned under the old backend's own selection formula),
  and **independently cold-re-verifies the resumed incumbent** under the new backend before it is
  trusted (added this integration; validated live in Gate D test 4/5, `|diff|=0.0`).

## Canonical divergence / gradient call path

Both backends share the identical inner solve and function value:

```
supervisor → stage runner → run_cm_upper_checkpointed
  cb_F! → cm_production_value_verified → archC_verified_state → Delta_dual (canonical, corrected F1 formula)
  cb_G! → cm_production_gradient (:reference) | cm_production_gradient_cplus (:cplus)
           both call archC_base_state(x_free0, pcx.ctx_cm, pcx.cctx) for the shared base dual state
```

`cm_production_gradient_cplus` (`lfix_cm_cplus.jl`) composes the existing, unmodified
`build_lfix_base_cache_C!` (factorized Backend C+ economic-block builder) with the same
`cm_fixed_contribution` the Reference CM path already uses — one shared CM-algebra
implementation, not a second derivation. Tie handling (`TiedWinnerError`) and nonfinite-probe/
retry discipline are reused from the existing Backend C+ kernel, not duplicated. Production
contrast basis: `:orthonormal` (unchanged, `cm_production_stage_runner.jl`'s own documented
2026-07-22 decision).

## Complete-state cache: NOT integrated

Classified **PROTOTYPE_ONLY_NEGLIGIBLE_VALUE**. Shadow-mode measurement on three real independent
D=20/W=80000/L=50 trajectories found **zero** exact-point revisits across all `cb_F!` calls — an
interior-point/barrier solver does not naturally revisit exact prior iterates on this path, so a
complete-state cache has no reachable hit opportunity in measured real usage. Not ported to
production; the prototype/design doc/shadow-measurement commits remain only on the archived
evidence branch/tag.

## Release gate results (2026-07-23)

| Gate | Scope | Result |
|---|---|---|
| A | Existing CM checkpoint/tie/resume/schema/verified-result regression suite (8 scripts, incl. the ported 54/54 and 60/60 C+ batteries and the directional sign audit) | **PASS**, 8/8 scripts, 0 failures |
| B | Expanded D=4 C+ equivalence battery (every free coord, multiple bandwidths, ties/near-ties, incumbent-swap, top-3-vs-fallback, nonfinite-probe) | **PASS**, 60/60 |
| C | Real D=20/W=80000/L=50 five-point gate vs. the completed 2026-07-22 campaign checkpoints | **PASS**, 5/5 points; every point: 399/399 finite, cosine=1.0000000000, 0/399 sign mismatches, max\|Δg\| 3.9e-15 – 2.2e-13 (well inside the 5e-12 envelope), C+ 6.13x–7.02x faster (median complete-gradient callback wall time) |
| D | Checkpoint/backend-provenance suite | **PASS**, 17/17 (incl. same-backend resume both directions, cross-backend resume with/without override, cold re-verification of a switched incumbent, interruption+resume, corrupt-file rejection) |
| E | One end-to-end supervisor smoke test, C+ as resolved default, real D=20/W=80000/L=50 | **PASS** — startup printed `cm_gradient_backend=cplus`, checkpoint written with C+ provenance, cold-verified by the supervisor itself, C+ resume accepted the checkpoint, Reference resume without override was correctly rejected |

**CM-C+ vs Reference D=20 timing** (median of 3 reps, complete gradient callback, per Gate C point):

| Point | Reference median wall | C+ median wall | Speedup | Reference alloc | C+ alloc |
|---|---|---|---|---|---|
| chain1_delta0.1 | 6.964s | 1.076s | 6.47x | 13881 MB | 162 MB |
| chain1_delta1.0 | 6.999s | 1.001s | 7.00x | 13881 MB | 162 MB |
| chain1_delta2.0 | 6.999s | 0.997s | 7.02x | 13881 MB | 162 MB |
| chain2_delta1.0 | (see gate log) | 1.114s | 6.13x | — | 162 MB |
| chain3_delta1.0 | (see gate log) | 1.008s | 6.90x | — | 162 MB |

These are **callback-level** speedups/allocation reductions (the CM gradient callback only), not
whole-solver wall-clock speedups — the outer KNITRO trajectory itself is not required or expected
to be identical between backends (see the retained 600s-trajectory-divergence note below).

## Known non-blocking observations (not fixed this integration, out of scope)

- **Supervisor kill-signal scope**: `scripts/cm_production_supervisor.sh`'s wall-budget/stall
  `kill -TERM "$pid"` targets the subshell wrapper PID, not the actual KNITRO/Julia grandchild
  process it launches; observed live during Gate E's first (90s-budget) attempt, where the
  grandchild survived as an orphan after its wrapper was killed. Pre-existing, backend-agnostic
  (affects `:reference` identically), not a CM-C+ regression. Gate E was re-run with a realistic
  300s budget and passed cleanly; the kill-scope issue is noted here for future hardening, not
  fixed in this integration.
- **600s trajectory-divergence observation (carried over from the tested-evidence branch)**:
  Reference and C+ can take a different number of outer KNITRO iterations on the same real
  problem despite gradients agreeing to near machine precision — expected under floating-point
  perturbation of a nonconvex solver's path, not a correctness defect. C+ did not produce a worse
  verified incumbent in that comparison. Not something this integration attempts to equalize.

## Launch commands

Production default (C+):
```
cd /bbkinghome/edav/gravity_robustness/gravity-production-fullA-exact   # or the post-merge worktree
source .knitro_env.sh
scripts/cm_production_supervisor.sh <chain_id 1|2|3> production_runs/<campaign_dir>/chain<chain_id>
```

Explicit Reference fallback for a whole campaign:
```
CM_GRADIENT_BACKEND=reference scripts/cm_production_supervisor.sh <chain_id 1|2|3> production_runs/<campaign_dir>/chain<chain_id>
```

Explicit, audited backend switch on a resumed stage (single stage runner invocation, not the
supervisor):
```
CM_GRADIENT_BACKEND=cplus CM_ALLOW_BACKEND_SWITCH=1 julia --project=. full_aod_diag/d4_exact/cm_production_stage_runner.jl <stage_dir> <delta> <budget_s> resume <ckpt_path> <chain_perturb_seed>
```
