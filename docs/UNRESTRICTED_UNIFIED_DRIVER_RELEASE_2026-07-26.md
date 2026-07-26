# Unrestricted Unified-Driver Launch-Path Release — 2026-07-26

**State: MATCHED_AB_PASSED** (not yet merged to production/fullA-exact; on
`port/remediate-production-5x7-audit-2026-07-26`).

## What changed

`unrestricted_stage_runner.jl` (the only committed CLI/process-launchable entry point for the
UNRESTRICTED family) previously called `run_profile_checkpointed` (`c10_d20_production_driver.jl`)
— a legacy, z-space-only, fixed-theta-only driver that hardcodes KNITRO `algorithm=3` by default
and predates the transformed-A/flexible-theta coordinate architecture entirely. `git log` dates
this script to 2026-07-24 10:58, a full day before `run_polish_checkpointed_unified`
(`c10_d20_production_driver_unified.jl`, 2026-07-25 09:55) even existed — confirmed stale wiring,
not a deliberate design choice.

New default behavior:
- New campaigns (`MODE=calibration` or `MODE=resume`) go through
  `run_polish_checkpointed_unified`, with `A_coordinate_mode=:powered_aspace` ("transformed_a")
  and `trade_elasticity_mode=:fixed` as the new defaults (both overridable via env vars).
  `A_COORDINATE_MODE=legacy_z` remains a fully supported, explicit replication option.
- `MODE=legacy_profile_resume` retains the OLD `run_profile_checkpointed` driver, completely
  unmodified, solely to finish a genuinely in-flight pre-fix campaign.
- Startup diagnostics (`unrestricted_public_driver`, `outer_problem_type`, `A_coordinate_mode`,
  `trade_elasticity_mode`, `outer_algorithm`, `checkpoint_schema`) are printed at launch for every
  mode.
- Resuming a legacy (schema-4, `D20CheckpointV4`) checkpoint under the new default path is
  refused with an explicit migration message pointing at `MODE=legacy_profile_resume`, rather than
  silently reinterpreted or converted (per the task's A2 requirement — no padding/truncation
  between the two incompatible checkpoint schemas).

## Correctness gates (all real, all passed)

1. **Decode/eval/gradient equivalence** (`test_unrestricted_legacy_vs_unified_equivalence.jl`,
   real D=20/W=80,000/`:exclude_row`/seed 20260719, both `find_smallest=true` and `false`):
   compares the OLD driver's own decode path (`build_pivot_elimination`/`pivot_reduce`/
   `pivot_expand`) against the NEW unified driver's `A_coordinate_mode=:legacy_z` decode
   (`build_pivot_elimination_cheap`/`decode_outer_unified`) at the SAME real calibration point,
   built from one shared, identically-seeded context. Result: **ALL PASS** both directions —
   `x_free` agrees to ~6e-8 absolute / effectively machine-precision relative (dominated by
   entry magnitude), `screened_eval`'s `Delta_dual` agrees to ~4e-12 absolute
   (`0.002486773477940637` vs `0.0024867734779406418`), `inner_status` identical (both feasible,
   code 0).
2. **Live CLI smoke test** (`unrestricted_stage_runner.jl`, real D=20/W=80,000, 25-30s KNITRO
   budget, default `A_coordinate_mode=:powered_aspace`): real KNITRO outer solve, converged to a
   feasible incumbent (`kappa=0.0586-0.0649` across two runs depending on random timing cutoff),
   correct startup diagnostics, `D20CheckpointUnified` written successfully. **PASS.**
3. **Migration-refusal smoke test**: resuming a deliberately incompatible checkpoint file under
   the default path prints the intended migration message and exits nonzero (correct — refusal is
   the intended behavior, not a bug). **PASS.**
4. `legacy_profile_resume` mode was not independently live-exercised this session (would require
   generating a genuine in-flight legacy checkpoint first) — the code path itself is an unmodified
   passthrough to the pre-existing, already-battle-tested `run_profile_checkpointed`, so the risk
   here is limited to the new ARGS/MODE dispatch logic around it, not the scientific driver itself.

## Bugs found and fixed during this work (both in NEW code written this session, not production)

- A Julia top-level soft-scope gotcha (assigning to `res` inside a `try` block when a global `res`
  already exists is ambiguous and silently creates a shadowing local, not a global update) caused
  an `UndefVarError`/`FieldError` on the first smoke-test attempt. Fixed with an explicit
  `global res = ...` inside the `try`. This is the sibling trap to this project's own documented
  `julia-toplevel-catch-scoping-gotcha` memory entry (that one covers `catch`-only assignment;
  this one covers `try`-side assignment when a global of the same name is already in scope).

## Not yet done (tracked separately)

- Phase G's live-resolved `outer_algorithm` capture (this release reports the *configured* value,
  `auto` per the `.opt` file, `pin_outer_algorithm=false`) — actual KNITRO-resolved algorithm
  requires the cross-family instrumentation Phase G adds.
- A dedicated live CLI smoke test under `A_COORDINATE_MODE=legacy_z` (covered indirectly by the
  decode-equivalence gate, which is the stronger test, but not by a separate live run).
