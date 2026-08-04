# Checkpoint (.jls) contents verification — resolves handoff's flagged caveat

The handoff explicitly flagged: "The cell inventory and seed manifest ... reference the `.jls`
path/checksum for both 'outer vector' and 'inner dual' fields, but this has NOT been confirmed to
actually contain a separable, directly-reusable inner dual state." Resolved by actually
deserializing a real seed file with the real production include chain loaded (not guessed from
memory), per the handoff's own instruction for what the next session should do first.

## Method

`load_checkpoint_unified(path)` (`c10_d20_production_driver_unified.jl:100`), loaded via the exact
include chain `campaign_unrestricted_runner.jl` itself uses (`c10_d20_production_driver.jl`,
`flexible_theta.jl`, `flexible_theta_aspace_production.jl`, `outer_coordinate_layout.jl`,
`c10_d20_production_driver_unified.jl`), run against a real seed:
`unrestricted/upper/delta_0.5/start_2/unrestricted_upper_d0.5_s2_unified_latest.jls` (this exact
file is `CONTINUATION_SEED_MANIFEST_2026-08-03.csv`'s `unrestricted_upper` seed role
`A_envelope_incumbent_delta_0.5`).

## Finding: the schema has TWO different "outer point" concepts — use the right one

- `checkpoint.zfree`/`checkpoint.g` (379-length A-block + scalar gp): the **last outer iterate the
  solver tried** when the checkpoint was written — `g=0.9489165095118315`, not necessarily feasible
  or verified.
- `checkpoint.best_feasible` (a `NamedTuple`): the **actual best verified incumbent** —
  `gp=0.9524246329235422`, `w` (380-length, `[gp; A_nonpivot_native]`), `Delta=0.5000008965197175`
  (matches `CONTINUATION_SEED_MANIFEST_2026-08-03.csv`'s `Delta_star` for this seed exactly),
  `gravity=2.5e-18` (~0, feasible), `kkt=6.0e-13`, `inner_status=0`.
- **Seeds must be built from `checkpoint.best_feasible.w`, not `checkpoint.zfree`/`checkpoint.g`.**
  Using the raw last-iterate fields would seed a continuation run from an unverified probe point
  instead of the actual incumbent the campaign found.

## Finding: `dual_warm_start` is real and populated, but tied to the LAST iterate, not `best_feasible`

`checkpoint.dual_warm_start` is a genuinely separate, populated field (length 382, all 382 entries
nonzero, e.g. `[-0.9906, -0.0013, -0.6129, ...]`) — structurally confirms the handoff's caveat is
resolvable: the inner dual state is NOT bundled ambiguously inside some opaque blob, it's its own
field. But `best_feasible` stores no dual vector of its own, and `dual_warm_start` was captured
whenever the checkpoint was periodically written (`checkpoint_interval_s`), which is not
necessarily the same outer point as `best_feasible.w` (the checkpoint interval and the moment the
best point was found are independent events).

**Practical consequence for the orchestration (Section 10 of the task spec):** seed a continuation
cell from `best_feasible.w` directly; do not attempt to splice in `dual_warm_start` from the
checkpoint verbatim, since it may correspond to a different outer point. The first inner solve at
`best_feasible.w` in the new cell will produce its own genuine warm dual state there (cheap — one
inner solve relative to the overall cell budget) rather than risk seeding KNITRO with a dual vector
computed at a different point under the same key. This is conservative but correct; it does not
contradict CLAUDE.md's warm-start note (start type affects speed, not whether the true optimum is
reachable) — it just avoids asserting a specific point-correspondence that was not actually true
per this direct inspection.

## Same schema shape confirmed (by direct struct read, not re-deserialized) for CM/origin-ZC families

`CMCheckpointV9` (`cm_checkpoint.jl:362`) and `CMCheckpointV10` (`cm_originzc_checkpoint.jl:268`)
both have the identical `g`/`zfree`/`eta_nu`/`dual_warm_start`/`best_feasible` field pattern as
`D20CheckpointUnified` — same seeding rule applies uniformly across all 5 families. Loaded via
`load_cm_checkpoint(path)` (`cm_checkpoint.jl:506`), which auto-upgrades any older on-disk schema
(V3/V4/V6/V8) to the current V9 shape before returning.

## Verdict

```
CHECKPOINT_CONTENTS_CONFIRMED = true
SEED_FIELD = best_feasible.w   (NOT zfree/g)
DUAL_REUSE_POLICY = fresh_solve_at_seed_point   (dual_warm_start not spliced verbatim)
```
