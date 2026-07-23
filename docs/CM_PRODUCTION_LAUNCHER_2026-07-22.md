# CM production launcher — 2026-07-22

Built in branch `perf/fullA-cm-postclosure-2026-07-22` (based on tag
`cm-production-ready-2026-07-22`), then merged into `production/fullA-exact`. No production
launcher existed before that session (confirmed by `find` for `*launch*`/`*supervisor*`/
`*campaign*` under `full_aod_diag/d4_exact/` — nothing matched).

**Closed by a later same-day session** (return-path safety, 4-way stage state machine, seed
reproducibility/provenance, contrast-basis decision — see
`docs/SESSION_SUMMARY_2026-07-22_cm_production_readiness.md` for the full account and the final
canonical commit/tag). `perf/fullA-cm-postclosure-2026-07-22`'s worktree and branch (both local
and on `cdw`) were removed once confirmed fully incorporated into `production/fullA-exact` — do
not look for that worktree/branch; work from `production/fullA-exact` (or the
`cm-production-ready-2026-07-22-r3` tag) directly. Everything below describing the launcher's
architecture and configuration is still accurate; only the state-machine/return-path/seeding
mechanics inside `scripts/cm_production_supervisor.sh` and
`full_aod_diag/d4_exact/cm_production_stage_runner.jl` changed — see the session summary for
specifics.

## Files

- `full_aod_diag/d4_exact/cm_production_stage_runner.jl` — runs exactly one (chain, delta) stage.
  Three seeding modes: `calibration` (stage 1 only, optionally perturbed per chain),
  `seed_w0` (fresh run at a prior stage's cold-verified incumbent — the normal cross-delta
  transition), `resume` (continue the same interrupted stage from its own latest checkpoint).
- `full_aod_diag/d4_exact/cm_cold_verify.jl` — standalone process: loads a `CMCheckpoint`,
  rebuilds `ctx`/`pcx` strictly from the checkpoint's own recorded provenance, asserts draw
  checksums match, cold-re-evaluates `best_feasible.w` via `cm_production_value_verified`
  (no cache exists on the CM path at all, so "fresh process" already means "cache disabled" —
  see `docs/CM_PRODUCTION_STATE_2026-07-22.md`), hard-refuses to hand the vector forward unless
  `is_verified_success` and `Delta_dual <= delta` both hold.
- `scripts/cm_production_supervisor.sh` — one supervisor process per chain; walks the delta
  ladder, launches each stage, polls it, detects stalls, restarts, cold-verifies between stages.

Both `.jl` files parse cleanly (`Meta.parseall`, verified this session). Full exercise (real
KNITRO solve) was done via the deployment smoke test below.

## Production configuration (fixed inside the scripts, matching the brief)

D=20 real data, France focal (`baseIndex=2`, via `d20_real_setup`/`d20_real_setup_design`),
W=80,000, common-marginal cumulative basis (`cm_basis=:cumulative`, hardcoded in `CMCheckpoint`
construction), **contrasts=:orthonormal** (changed from the original `:anchored` default by the
launcher-closure session, per the conditioning evidence in
`docs/fullA_cm_conditioning_and_adaptive_grid_report.md` — orthonormal strictly dominates
anchored in conditioning at every L/point tested there, with the gap widening at L=50; the
production Hessian backend below was independently validated correct and still real-world faster
than dense under orthonormal at L=50, so the conditioning win is not traded against a broken fast
path — see `docs/fullA_cm_hessian_architecture_report.md`), L=50, `JULIA_NUM_THREADS` capped at 20
(env var, default 20, never overridden upward by the supervisor), canonical
`run_cm_upper_checkpointed` driver, canonical `Delta_dual` (unchanged production code — confirmed
in `docs/CM_PRODUCTION_STATE_2026-07-22.md`), schema-2 `CMCheckpoint` (schema-1 resume is a hard
error in `load_cm_checkpoint`, unchanged), typed `is_verified_success`/`CMExpectedSolveFailure`
gating (unchanged production code), direction-specific gamma-prime box (unchanged —
`find_smallest=true` hardcoded, matches the only wired direction), independent checkpoint
directory per chain **and per delta stage within a chain** (`$CKPT_ROOT/delta_<delta>/`), no
experimental backend (`cm_hessian_backend=:structured`, the production default, unchanged).

Chain-perturbation starting points are now derived from a fixed integer formula (not Julia's
generic `hash()`, which is not a stable cross-version API) and are recorded in full, per stage, in
`$CKPT_DIR/w0_used.jls` — see the session summary for the exact formula. Cross-delta `seed_w0`
transitions now assert the seed's recorded W/draw_design/draw_seed/cm_L/contrasts/bi/schema all
match the new stage's fixed context (only `delta` may differ) rather than only logging a note.

**Best-verified-incumbent-distinct-from-terminal-iterate**: enforced structurally — the next
stage's `w0` always comes from `cm_cold_verify.jl`'s output (`best_feasible.w`, cold-re-verified),
never from the checkpoint's own `g`/`zfree` terminal-iterate fields. The supervisor's own
`seed_w0` mode never reads `g`/`zfree` at all.

**Cold verification after each delta**: the supervisor calls `cm_cold_verify.jl` immediately after
every stage completes, before advancing to the next delta; a failed cold-verification aborts the
chain rather than silently seeding the next stage from an unverified point.

## Exact three-chain launch commands

`perf/fullA-cm-postclosure-2026-07-22` (worktree and branch, local and on `cdw`) has been removed
-- it was fully incorporated into `production/fullA-exact` and confirmed byte-identical for every
launcher file before removal (see the session summary). Launch from a fresh worktree/checkout of
the **`cm-production-ready-2026-07-22-r3`** tag (or later `-rN`, whichever the session summary
names as current) — never the pre-launcher `cm-production-ready-2026-07-22` tag (no supervisor
existed at that commit) or the intermediate `-r2` tag (predates the return-path/state-machine/
seed-provenance/contrast-basis fixes). Each chain's supervisor is a long-running foreground (or
`&`-backgrounded, tracked by its own PID — this is the script's own concern, not the harness's) process:

```bash
export JULIA_NUM_THREADS=20   # hard cap per the brief; the supervisor also sets this itself
ROOT=/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22

nohup scripts/cm_production_supervisor.sh 1 "$ROOT/chain1" > "$ROOT/chain1_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain1_supervisor.pid"

nohup scripts/cm_production_supervisor.sh 2 "$ROOT/chain2" > "$ROOT/chain2_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain2_supervisor.pid"

nohup scripts/cm_production_supervisor.sh 3 "$ROOT/chain3" > "$ROOT/chain3_supervisor.out" 2>&1 &
echo $! > "$ROOT/chain3_supervisor.pid"
```

(Add `RESUME_CAMPAIGN=1` before `nohup` only if intentionally resuming into an already-nonempty
`$ROOT/chainN` directory from a prior partial launch -- the supervisor now refuses a fresh launch
into a nonempty campaign directory otherwise.)

(`nohup`/`&` here is the intended, correct way to run a genuine multi-hour unattended production
supervisor after this session ends — distinct from the house rule against `nohup`/`disown` inside
*this conversation's own* Bash tool calls, which exists so THIS session's own background work
stays visible to the harness. This campaign is explicitly meant to keep running after the
conversation ends.)

Each chain writes to its own `$ROOT/chain{1,2,3}/` directory; within each chain,
`delta_{0.1,0.5,1.0,2.0}/` are independent subdirectories. `$ROOT/chain*/supervisor.log` and
`restarts.log` record every action; `$ROOT/chain*/delta_*/stage.log` is the raw Julia output for
that stage.

## Known operational caveat carried into this launcher

See `docs/CM_PRODUCTION_STATE_2026-07-22.md` — a real interrupt→resume hang was observed and only
partially root-caused during the prior closure session. This supervisor's stall-detection
(>=600s of no log growth AND no checkpoint-mtime advance, well above any ordinary 90-130s
callback) is the mitigation: it cannot prevent the hang, but it detects and recovers from it
without a human watching the terminal. See `docs/CM_STALL_INVESTIGATION_2026-07-22.md` for this
session's own reproduction attempt.
