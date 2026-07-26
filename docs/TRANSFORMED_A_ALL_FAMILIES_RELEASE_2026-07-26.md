# Transformed-A All-Families Release — 2026-07-26

## Decision: promoted `A_coordinate_mode=:powered_aspace` to the production default for all four restricted families

Task §8 (and the inherited remediation's own Phase F item 5, explicitly sized but deferred pending
direction) asked to promote transformed-A to the default for the restricted families, matching the
unrestricted family's own Phase A promotion. This session completed it.

## Evidence, in order

1. **Coordinate-transform correctness** (`test_cm_aspace_coordinate_gates.jl`, pre-existing,
   re-run this session at real D=20/W=80,000): 6/6 PASS — `cm_a_from_z`/`cm_z_from_a` round-trip to
   machine precision (`5.3e-15`), a-space decode reconstructs the *identical* `logA_full` as the
   direct z-space path at the real calibration point (`<1e-9`), gradient rescale matches
   `gradient_transform_unified` exactly. This is the shared, family-agnostic decode/encode/
   gradient-rescale boundary (`cm_aspace_coordinate.jl`) every restricted family goes through — not
   re-derived per family.
2. **Default flip**: `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl`'s `A_coordinate_mode` kwarg
   default changed `:legacy_z → :powered_aspace` for both `run_cm_upper_checkpointed` (flexible CM,
   common Fréchet, CM+ZC all share this entry point) and `run_originzc_upper_checkpointed`.
   `:legacy_z` remains fully supported as an explicit replication mode.
3. **Live smoke test** (`phase8_transformed_a_default_smoke.jl`, new this session): real D=20/
   W=80,000 campaigns through the real public drivers, **default kwargs** (i.e. exercising the new
   default, not overriding it), for all four families, from a correctly-constructed a-space `w0`.
   Confirmed real, non-crashing, progressing KNITRO solves:
   - Flexible CM: **PASS** (wall=78.2s, n_eval=2, kappa=0.0431278)
   - Common Fréchet: **PASS** (wall=107.1s, n_eval=1, kappa=0.0202939)
   - CM+ZC: **PASS** (wall=92.4s, n_eval=3, kappa=0.0317378)
   - Origin-ZC: **PASS** — but only after this session's smoke script needed **three** rounds of
     its own bugfixes: (1) wrong include order for `cm_originzc_config.jl`/
     `cm_originzc_target_layout.jl`, (2) an invalid `distribution_restriction` symbol value, and
     (3) a **missing include** (`cm_originzc_cplus.jl`) that let a real KNITRO gradient-callback
     error (`nStatus=-500`, `UndefVarError(:cm_originzc_production_gradient_cplus)`) slip through
     as a false "PASS" under this script's first, too-weak check (`n_eval > 0` alone — a callback
     that errors after one evaluation still has `n_eval=1`). Caught by re-reading the actual log
     rather than trusting the printed PASS line. Tightened the check to require a genuinely
     feasible-or-timelimit KNITRO status (excluding `-500`), fixed the missing include, and
     re-ran: real iterations now proceed (previously died at iteration 0), `ALL PHASE 8
     TRANSFORMED-A DEFAULT SMOKE TESTS PASSED`.

   **`ALL PHASE 8 TRANSFORMED-A DEFAULT SMOKE TESTS PASSED` — all four families, real D=20/
   W=80,000, default kwargs, genuinely feasible-or-timelimit KNITRO status, no callback errors.**

## Checkpoint safety

Old coordinate-mode checkpoints are not silently reinterpreted: `CMProductionEvalKey` (the exact-
cache key, Phase C) and the checkpoint schema both carry `A_coordinate_mode` as an explicit field,
so a resume attempt against a checkpoint written under the other coordinate convention will not
silently produce a wrong-basis result — consistent with the unrestricted family's own Phase A
migration-refusal discipline (see Phase 1.1's run 5 in the master report for the analogous refusal
test on that family; an equivalent explicit refusal test for a restricted-family legacy-z
checkpoint was not run this session, since no restricted-family campaign had been run under the
old default long enough to produce one worth testing against).

## Verdict

`TRANSFORMED_A_DEFAULT`: flexible_cm=promoted+verified, common_frechet=promoted+verified,
cm_plus_zc=promoted+verified, zc_only=promoted+verified (all four families' live smoke tests
PASSED under the corrected, tightened check).
