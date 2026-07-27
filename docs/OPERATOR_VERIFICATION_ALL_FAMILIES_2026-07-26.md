# Operator-Based Verification — All Families — Status 2026-07-26/27

## Done this session

- `verify_inner_solution_operator_cmmeanzc!` (`operator_verification.jl`) — CM+ZC's `G=[E|Z|C]`
  independently reapplied via `economic_forward!`/`economic_transpose!` +
  `restriction_forward!`/`restriction_transpose!` (Z) + the CM-grid free-function kernels
  (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`/`build_weighted_histogram!`/
  `cumulative_backward_gradient!`), all against FRESH scratch, never touching the live FG
  callback's own buffers. D=4 gate (`test_operator_verification_cmmeanzc.jl`), 3 K_mean/K_pair/L
  configs, ALL PASS: operator-recomputed KKT residual agrees with the dense verifier's own
  `max_abs_moment_kkt_resid` to ~1e-15 (both effectively zero, confirming stationarity
  independently of the dense path).

## Inherited, unchanged this session

- `verify_inner_solution_operator_originzc!` — origin-ZC's `G=[E|Z]`, prior session's own
  proof-of-concept, D=4 gated.

## NOT done this session (honest gap)

- Flexible-CM operator verification (its own dense verification still reads `obj.H` via
  `obj(inner_x, constr=...)`/`select_G_from_H` — now confirmed cheap, so this is not urgent for
  allocation reasons, but is still architecturally required by task §8).
- Common-Fréchet operator verification (same gap, plus its own CM-block verification also needs
  the `skip_cm_fill_ref`-toggled dense CM-column fill removed first).
- Unrestricted family: verification backend not investigated this session at all (its own FG is
  already fully operator-based since Addendum Part A, but the VERIFICATION step specifically
  was out of this session's time budget).
- `verification_backend = :operator` production wiring (task §8's "Wire ... production default
  family by family after gates") — not done for ANY family this session; all operator
  verification remains available-but-not-default, consistent with `verify_inner_solution_operator!`
  functions being newly-built/newly-extended rather than long-validated.
- `skip_cm_fill_ref` removal (task §8's explicit ask) — correctly NOT attempted, since it requires
  flexible-CM's own verification to go operator-based first (not done, see above).

## Priority for next session

Flexible-CM's operator verification is the natural next target: it is the simplest remaining gap
(no ZC block, `CMLookupState`'s own forward/backward already retrofitted this session so the
component functions needed already exist), and closing it would unblock `skip_cm_fill_ref`'s
removal.
