# Fixed transformed-A production port — master summary, 2026-07-25

Branch: `port/transformed-A-and-flexible-theta-production-2026-07-25`, based on
`cdw/production/fullA-exact@39b89c5` (current canonical tip at time of this port; the allocation/
Hessian production release is already merged there, so this port's Part I §1 rebase blocker was
cleared before this port started).

## What this release is

`trade_elasticity_mode=:fixed`, `A_coordinate_mode=:powered_aspace` (`a := log(AodPow)`, the
theta-decoupled bilateral productivity coordinate) as a new opt-in alongside the existing
`A_coordinate_mode=:legacy_z` (current production's `z=log(Aod_theta)`), both reachable through
the one shared `run_polish_checkpointed_unified` driver and `OuterCoordinateLayout` abstraction.
Exact reparameterization at fixed theta — see `TRANSFORMED_A_COORDINATE_MATHEMATICS_2026-07-25.md`
for the math and `COMMON_OUTER_COORDINATE_ARCHITECTURE_2026-07-25.md` for the shared driver
architecture.

## Gate summary

| Gate | Result |
|---|---|
| D=4 unified-layout (21/21 assertions) | PASS, decisive chain-rule `rel_err=4.3e-12` |
| D=20 unified-layout (real production scale) | PASS, decisive chain-rule `rel_err=8.5e-13` |
| Checkpoint/resume (5/5) | PASS |
| Smoke (all 3 arms, post-reconciliation) | PASS, `fixed_aspace` bit-identical to pre-rebase reference |
| Matched comparison, 300s x {δ=1,δ=2} | transformed-A beats legacy-z: +8.8% / +1.4%, both cold-verified, zero hangs |

Full detail: `OUTER_COORDINATE_PUBLIC_DRIVER_ASSERTIONS_2026-07-25.md` (gates),
`FIXED_TRANSFORMED_A_MATCHED_COMPARISON_2026-07-25.md` (campaign).

## Restriction-family compatibility (task §15)

Not empirically tested against CM/ZC/origin-ZC in this session (see the public-driver-assertions
doc's "not run" section). The coordinate change is architecturally family-agnostic — it only
changes how the outer vector's A-block is decoded/gradient-rescaled, never the restriction basis,
moment construction, or Hessian machinery this task was explicitly scoped to avoid touching — but
that is an architectural argument, not a tested one, for this session.

## Default and replication mode (task §21)

`A_coordinate_mode=:legacy_z` remains a first-class, fully supported mode (not a compatibility
shim — it is the literal untouched production behavior when a caller doesn't opt into
`:powered_aspace`). This port does **not** flip any existing production entry point's default;
`run_polish_checkpointed`/`run_profile_checkpointed` are completely untouched. Recommending
`:powered_aspace` as the *new-campaign* default (task §19's "make transformed a-space the default
for new fixed-theta campaigns only after these gates") is a call for the user to make at merge
time, not something this port changes unilaterally.

## Final verdict

```
FIXED_TRANSFORMED_A = PORT_READY_NOT_MERGED
FIXED_DEFAULT_A_COORDINATE = legacy_z   (unchanged; :powered_aspace available opt-in,
                                          recommend switching new-campaign default at merge time
                                          given the clean matched-comparison win)
LEGACY_Z_REPLICATION_MODE = retained
POST_MERGE_SMOKE = not_applicable   (not merged this session)
```

Held at `PORT_READY_NOT_MERGED` because this task's standing instructions require explicit user
confirmation before any `git push`/merge to `cdw/production/fullA-exact` — not because of any
unmet technical gate. All correctness gates pass; the matched comparison is a clean, cold-verified
win at both tested budgets; zero regressions found.
