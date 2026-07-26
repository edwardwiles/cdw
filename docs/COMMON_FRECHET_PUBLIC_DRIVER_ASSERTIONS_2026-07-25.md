# Common-Fréchet public driver assertions — 2026-07-25/26 (Part V)

Confirms `marginal_restriction=:common_frechet` is genuinely live through the real public entry
point (`run_cm_upper_checkpointed`), not merely reachable through a test-script bypass
(`build_cm_production_context_v2` alone). Full wiring detail:
`COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md`.

## Startup manifest assertions (real run, `smoke_frechet_checkpointed_driver.jl`)

Captured verbatim from a real D=20/W=80,000/L=10 run:
```
[frechet_smoke] cm_gradient_backend=cplus (production default) destination_sample=exclude_row (production default)
[frechet_smoke] marginal_restriction=common_frechet (fixed Frechet as CM plus a common-level anchor)
marginal_restriction = common_frechet
frechet_feature_set = cdf_only
frechet_basis = cm_contrasts_plus_common_level
frechet_grid_size = 10
cm_contrast_count = 19
frechet_level_count = 1
total_marginal_moments = 200
cm_contrasts = anchored
core_hessian_backend = exact_winner_pair_parallel
[frechet_smoke] core_hessian_backend=exact_winner_pair_parallel (Architecture C)
```
Every field matches the task's required exact format and resolves to the correct live values
(`cm_contrast_count = D-1 = 19` at D=20; `total_marginal_moments = D*L = 200`).

## Real KNITRO execution confirmation

Same run: `10` real outer evaluations, `6` gradients, KNITRO's own solver banner confirms
`Number of variables: 380` (matches `D*D_dest = 20*19` under `:exclude_row`), genuine outer
progress (`gp: 0.9878→0.9702`, feasible, verified), checkpoint written and independently reloaded
(`schema=8, marginal_restriction=common_frechet`).

## Runtime backend-use counters

`core_hessian_backend=exact_winner_pair_parallel` confirmed both at startup-manifest print time
(static config) and via the real solve's Hessian-callback path (`archC_frechet_hess_cb_builder`,
Architecture C, winner-pair `H_EE`) — not the `:dense_reference` fallback. Screen-summary counters
(`pairwise_hits=13 inner_solves_avoided=13`, zero `hard_winner_hits`/`witness_hits` at calibration)
show no anomalous fallback behavior.

## Config-guard assertions (`test_frechet_cm_config_wiring_d4.jl`)

`(:common_frechet, cm_basis=:interval)` correctly raises an error (unsupported combination, not
silently mishandled). `(:common_frechet, cm_hessian_backend=:structured)` is correctly *accepted*
(Part III landed) — this assertion was updated mid-session when Part III completed (a pre-Part-III
version of the test asserted the opposite; caught and fixed as a stale-test issue, not a code
regression — see the `Rebase onto transformed-A...` commit for the full account).

## Verdict

`PUBLIC_DRIVER_ASSERTIONS = pass`, real execution, not simulated.
