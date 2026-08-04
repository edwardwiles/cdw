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

## Never infer invalidity from materialized zero LFD weights

A recovered LFD weight materializing as Float64 `0.0` is NOT by itself evidence of an invalid
or divergent inner solve. `dPsi!`/`melitz_cc_dPsi!` (the divergence's conjugate derivative)
compute `exp(arg0)` directly; for an extreme-tail QMC draw `arg0` can be around -800 to -1100,
which underflows to the literal bit pattern `0.0` in Float64 while still being mathematically a
genuinely tiny POSITIVE density ratio (confirmed live via BigFloat, 2026-08-04: every such
weight is strictly positive at arbitrary precision, with true mass below `1e-300`, i.e. this is
unavoidable post-normalization underflow, not an avoidable common-offset/normalization bug —
see `test/melitz/test_primal_divergence_underflow_2026-08-04.jl`). Since
`lim_{m->0+} m*log(m) - m + 1 = 1`, the correct contribution of an underflowed-to-zero weight to
`melitz_primal_divergence` is the finite limit value `1.0`, not `Inf`.

`melitz_primal_divergence` (`src/melitz/delta_star.jl`) used to return `Inf` for the WHOLE
divergence sum the instant even one of `W` recovered weights underflowed to exactly `0.0`
(commit before `1d97e4c`), which silently failed `lfd_ok` on otherwise-perfectly-good points —
and this got MORE likely, not less, exactly at the extreme boundary-pushing points an outer
search cares about most. Fixed 2026-08-04 (commit `1d97e4c`); see
`docs/melitz_verification_gate_false_rejection_fix_2026-08-04.md` for the full
find/fix/validation writeup. This is the same false-rejection bug class independently found on
the Ricardian side the same day (`full_aod_diag/d4_exact/oracle.jl`'s `m_min_floor` check).

**Use the central `melitz_primal_divergence`/`melitz_recover_lfd`/`melitz_recover_lfd_from_solution`
gate (`src/melitz/delta_star.jl`, shared by both the dense `PsiObjectiveBundleDelta`/`Implicit`
path and the matrix-free `MelitzCCBundle` path via `cc_bundle.jl`) and its typed `lfd_ok`
result. Do not add a separate strict `weight > 0`/`minimum(weights) > 0` check anywhere else in
the Melitz tree** — a genuinely invalid recovery is still (correctly) caught by this same gate
via negative/non-finite weights, `s<=0` on the raw pre-normalization sum, normalization
residual, moment residual, or `isfinite(primal_dual_gap)` — never by a materialized-zero check.
