# Flexible-theta cache / checkpoint audit — production port, 2026-07-25 (task §10/§11)

Test script: `full_aod_diag/d4_exact/test_flexible_theta_aspace_cache_checkpoint.jl` (rectangular
D=4/D_dest=3 sample — mechanics tested here are scale-independent; wall-clock cache/checkpoint
activity at D=20 scale is exercised separately by the matched-comparison campaign). **Result: 18/18
PASS.** Full log excerpt: `docs/key_results/flexible_theta_aspace_cache_checkpoint_gate_log_2026-07-25.txt`
(committed to this repo).

## 1. Exact-point cache fingerprint composition

Every cache/state fingerprint required by task §10 is covered by one of two layers:

- **`FullAEvalKey.x_free`** (`oracle.jl`, unmodified): built from `xf = [mu; gp; Aod_levels...]`
  (the decoded output of `decode_and_expand_flexible_A`) — theta/mu, gp, and the full powered-A
  vector are ALREADY part of this key for free, because flexible mode's `xf` genuinely contains
  `mu` at position 1 (via `make_flexible_theta` moving mu into `ctx.free_idx[1]`). No change to
  `FullAEvalKey` or `oracle.jl` was needed or made.
- **`context_fingerprint_flexible_theta(ctx)`** (`flexible_theta.jl`, new, additive): extends the
  unmodified `context_fingerprint(ctx)` (which already hashes draws — raw-draw checksum via
  `ctx.draw_meta.checksum_uniform`/`checksum_transformed` — and destination_sample/layout) with a
  block hashing `(trade_elasticity_mode, A_coordinate_mode, theta_lo, theta_hi,
  theta_param_version, sigma, destination_sample)`. Verified: (a) flexible vs fixed-tagged ctx ->
  different fingerprint; (b) different theta bounds -> different fingerprint; (c) same ctx -> same
  fingerprint (deterministic, re-checked twice).
- **Transformed-draw checksum**: inherited unchanged from `context_fingerprint`'s existing
  `ctx.draw_meta.checksum_transformed` — flexible mode does not redraw or re-transform `U` (raw
  draws are fixed for the whole campaign per task §5; only theta-dependent DERIVED objects like
  winners/moments are rebuilt on a theta change, never `U` itself).
- **Solver/backend version**: inherited unchanged from `context_fingerprint`'s existing
  `LOADED_KNITRO_RELEASE` + option-file-contents hashing.

**Known scope limitation, documented not fixed**: `screened_eval`'s OWN internal `FullAEvalKey`
construction (inside `c10_d20_production_driver.jl`, unmodified) calls the base
`context_fingerprint(ctx)`, NOT `context_fingerprint_flexible_theta(ctx)` — i.e. the exact-point
cache `screened_eval` builds internally does not, by itself, distinguish two flexible contexts that
differ ONLY in theta bounds (same γ/U/destination_sample). This is not a practical aliasing risk
under this port's own driver (`run_polish_checkpointed_flexible_theta_A` constructs a FRESH
`SafeExactCache()` per call by default — a new cache object each run, never shared across runs with
different theta bounds unless a caller explicitly passes `exact_cache_override=`, which this port's
matched-comparison script never does). `context_fingerprint_flexible_theta` exists so a FUTURE
cross-theta-bound cache-sharing feature (analogous to `CrossDeltaExactCache`) has the right
fingerprint ready to wire in — flagged here rather than silently left for a future session to
rediscover.

## 2. A/B/A exact-cache-hit invariant (task §10's required test)

Solve at `theta_A = theta_star`; solve at `theta_B = theta_star*1.03` (genuinely different point,
cache grows 1->2); return to the IDENTICAL `theta_A` + outer coordinates. Result: cache size stays
at 2 (no new entry — exact hit), `Delta_dual` **bit-identical** (`===`, not merely `≈`) between the
first and third evaluation, `xf` bit-identical, `cache_hit` flag correctly set. Re-verified
independently with a second, fresh `SafeExactCache()` instance (two calls at the identical point,
second call hits). **All required behavior confirmed exactly as specified.**

## 3. Checkpoint schema (task §11)

`D20CheckpointFlexA` (`c10_d20_production_driver_flexible_theta_A.jl`) — new struct name carrying
`D20CheckpointV4`'s COMPLETE field set (schema 4, current production, UNCHANGED) plus:
`trade_elasticity_mode`, `A_coordinate_mode`, `eta_theta`, `theta`, `theta_lo`, `theta_hi`,
`a_nonpivot`. `zfree`/`logA_full` retain their V4 semantics exactly (genuine z-space
`log(Aod_theta)`, theta-consistent at the checkpoint's own theta) — the SAME convention the
original experimental port's `FlexibleThetaCheckpoint` used, so a checkpoint's economic content is
recoverable without needing to know which parametrization wrote it.

- Save/resume at `theta_star`: `eta_theta`/`a_nonpivot`/`logA_full` all round-trip bit-identical.
- Save/resume OFF `theta_star` (theta_B): `theta` round-trips bit-identical and is confirmed
  genuinely different from `theta_star`.
- **Reject loudly, not silently**: `load_checkpoint_flexA` throws on (a) a file that does not
  contain a `D20CheckpointFlexA` at all (simulating a fixed-mode or foreign checkpoint file — the
  Julia `struct`-name-based dispatch this codebase's CM-family checkpoints
  (`CMCheckpointV3`..`V7`) already use for exactly this purpose), and (b) a wrong `schema` field on
  an otherwise-valid `D20CheckpointFlexA`. Both confirmed to throw (not silently proceed) in the
  gate script.
- The driver itself (`run_polish_checkpointed_flexible_theta_A`) additionally hard-errors on resume
  if `theta_lo`/`theta_hi`/`destination_sample`/`draw_design` supplied by the caller disagree with
  what the checkpoint recorded — mirroring `run_profile_checkpointed`/`run_polish_checkpointed`'s
  own existing mismatch-refusal discipline for draw_design/destination_sample.

**Not separately gated in this port** (acknowledged limitation, not silently skipped): an explicit
"deliberate z-space-vs-a-space mismatch" checkpoint-rejection test, since this port did not wire
`D20CheckpointFlexA`/`load_checkpoint_flexA` to accept a z-space checkpoint at all (there is no
production z-space flexible-theta checkpoint type in this codebase to construct a genuine
mismatch from — the z-space arm exists in this port only as D=4 test/comparison scaffolding, see
the mathematical-parameterization doc §10, not as a checkpointable production driver). The
type-name-based rejection in item (a) above already covers "wrong checkpoint kind entirely," which
is the operative protection.

## 4. DualBank — theta-blind by design (documented limitation, not a correctness issue)

`screened_eval_flexible_A` passes `zfree=d.z_nonpivot` into the EXISTING, unmodified
`screened_eval`/`DualBank`/`select_warm_start` machinery — the warm-start nearest-neighbor
distance is computed over z-space A-coordinates only, with no theta axis (unlike the original
z-space port's `select_warm_start_flexible`, which folded `eta_theta` into the distance metric —
NOT ported here, a deliberate scope decision to minimize new/unvalidated code in a first
production port). Smoke-tested (no crash on record/select through the flexible-A path). This is a
potential WARM-START QUALITY/performance consideration (a bank entry from a very different theta
might be chosen over a same-theta entry purely on A-proximity), never a CORRECTNESS issue — every
retrieved warm start is only ever used to seed `ctx.obj.x` for a subsequent inner KNITRO solve,
which is independently verified feasible/converged before being trusted (per `screened_eval`'s
own warm-then-cold-retry and `is_verified_success` discipline, unchanged). Flagged as a candidate
follow-up, not implemented in this port.
