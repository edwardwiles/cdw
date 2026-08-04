# Section 15 — regression suite and no-mess checks

## Real regression found and fixed

Adding a hard `isdefined` requirement to `profiled_production_outer_constrained_2026-08-02.jl`
(section 6, requiring `profiled_coordinate_mode_dispatch_2026-08-04.jl` to be included first) is a
breaking change to that file's own include contract — any PRE-EXISTING caller that includes it
without also including the 3 new dependency files now fails immediately. This broke 4 pre-existing
files this session had not otherwise touched:

- `test_all_family_checkpoint_resume_2026-08-03.jl`
- `test_profiled_upper_constrained_checkpoint_resume_2026-08-03.jl`
- `test_zc_free_nu_production_driver_d20_2026-08-04.jl`
- `test_zc_free_nu_production_driver_d4_2026-08-04.jl`

Found by actually running the regression suite (`test_all_family_checkpoint_resume_2026-08-03.jl`
failed on first run, immediately after this session's own section 6/6b commits). Fixed by adding
the same 3-file include block (`cm_aspace_coordinate.jl`, `profiled_powered_relative_a_2026-08-04.jl`,
`profiled_coordinate_mode_dispatch_2026-08-04.jl`) before `profiled_production_outer_constrained_2026-08-02.jl`
in all 4 files — the same fix already applied to `bin/run_profiled_model.jl` and the two other
production driver scripts during section 12.

**Re-run after the fix: `test_all_family_checkpoint_resume_2026-08-03.jl` — 52/52 PASS, 0 FAIL**
(real D4 KNITRO round trips, all 5 families, checkpoint write/round-trip/resume-cumulative-counts/
best-feasible-monotonicity/cross-family-refusal/cross-W-refusal, all pass).

## Pre-existing, unrelated failure found (NOT fixed — out of scope)

`test_profiled_mock_family_gate_2026-08-01.jl` fails with a DIFFERENT error
(`profiled_lfix_incremental_2026-08-01.jl requires lfix_factorized_workspace.jl to be included
first`) — confirmed via `git log` that this file was last touched by the PRIOR session's own
commit `92e45da` (before this continuation started) and has never included
`lfix_factorized_workspace.jl` at all. This is a genuine pre-existing bug, unrelated to anything
this session touched (this session never modified `test_profiled_mock_family_gate_2026-08-01.jl`
or `profiled_lfix_incremental_2026-08-01.jl`'s own include requirements). Documented here for
transparency, not fixed — outside this continuation's own scope, and the "real" (non-mock)
successor gate (`test_profiled_reduced_flexcm_frechet_adapters_2026-08-02.jl`) already covers the
same ground with genuine KNITRO solves rather than mocks.

## No-mess checks

- **No silent defaults**: confirmed by section 4's own exhaustive audit (every new required kwarg
  checked, all call sites pass explicitly).
- **No dense path added or called**: this session added no new dense-G/dense-H code anywhere;
  `NO_DENSE_G_COUNTERS`-style assertions were not touched. Confirmed by grep: none of this
  session's new files reference `dense_reference`/`build_dense`/similar dense-path symbols except
  as pre-existing kwarg VALUES passed through unchanged (e.g. `inner_fg_backend = :dense_reference`
  in test setup code copied verbatim from existing test patterns, not a new dense computation this
  session introduced).
- **FULL production source/behavior unchanged**: confirmed by grep — this session's diff never
  touches `cm_checkpoint.jl`, `cm_outer_driver.jl`, `cm_originzc_checkpoint.jl`, or any of the
  `_run_full*` functions in `bin/run_profiled_model.jl` (only `_run_reduced`/
  `_run_reduced_zc_free_nu` and their include lists were touched).

## Not done this session

- The full "run all prior functional-readiness tests" / "all manifest/registry/A-B gate tests"
  suite (dozens of files across `full_aod_diag/d4_exact/`) was not exhaustively re-run — only the
  checkpoint/resume regression (the one most directly exercising code this session's changes
  touch) was run to completion. A broader sweep is real remaining work.
