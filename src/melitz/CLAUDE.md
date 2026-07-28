# Melitz-local operational notes

This file applies only to `src/melitz/`, `test/melitz/`, `docs/melitz_*`, and Melitz-only
scripts under `scripts/`. It does not apply to `cc_algo/`, any other Ricardian source, or any
Ricardian tests/options/cache/campaign drivers.

## Use 20 Julia threads by default for Melitz performance/production work

For Melitz performance and production work, use 20 Julia threads by default for any safely
parallel outer-gradient, structured-Hessian, or moment-construction kernel, unless the user
explicitly requests otherwise. Do not silently benchmark or run these paths serially. Always
report the active Julia and BLAS thread counts.

Concretely:

- Launch Melitz production/benchmark scripts with `julia --project=. -t 20 ...`.
- Prefer `gradient_backend=:auto` / `moment_backend=:auto` / `hessian_backend=:auto`
  (`MelitzBackendConfig`'s own defaults, `src/melitz/backend_config.jl`) over hardcoding a
  `*_serial` variant — `:auto` already resolves to the parallel backend whenever `D>=10` and
  `Threads.nthreads()>1`.
- If you must pass an explicit `*_serial` gradient backend (e.g. a deliberate serial-vs-
  parallel comparison), that's fine, but do not do it silently in a script whose header claims
  `-t 20`/production timing — `melitz_note_explicit_gradient_backend_choice` will print a
  warning and increment `MELITZ_EXPLICIT_SERIAL_GRADIENT_DESPITE_PARALLEL_COUNT` when this
  happens; check that counter is 0 (or its increments are all deliberate/disclosed) before
  treating a campaign's wall-clock numbers as representative of the parallel default.
- Call `melitz_thread_startup_report()` (`src/melitz/backend_config.jl`) at the start of any
  production/benchmark script to print Julia threads available and BLAS threads selected, and
  to warn (or, with `strict=true`, throw) if fewer than 20 Julia threads are available.
- This repo has repeatedly regressed to serial execution in later sessions despite this
  guidance — see memory `feedback-melitz-thread-defaults-repeatedly-regress` (if present) and
  `docs/melitz_outer_search_gradient_redundancy_and_sensitivity_2026-07-XX.md` Phase 0/1 for a
  concrete, live example (a `-t 20` script that hardcoded a serial gradient backend and paid a
  ~9s-per-call cost instead of the ~0.95s parallel cost on the identical real-D20 point).

Do not run a thread-count optimization study — 20 threads (or all available, with a warning,
if fewer) is the fixed default; the question to check is only "did the parallel path actually
get entered," not "what is the optimal thread count."
