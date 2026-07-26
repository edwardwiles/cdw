# Production merge record — 2026-07-26

## What happened

`cdw/production/fullA-exact` fast-forwarded from `d3e561a` to `da62166` (28 commits), bringing
in both the transformed-A and flexible-theta production ports. Pushed by explicit user
confirmation after a final freshness check (production had moved twice more during this task's
execution — once mid-task for a shared winner-pair core-Hessian merge, once again immediately
before push for a C+ A-gradient micro-optimization — both were caught, rebased onto, and
re-verified before the push went out).

```
git push cdw port/transformed-A-and-flexible-theta-production-2026-07-25:production/fullA-exact
   d3e561a..da62166  -> production/fullA-exact   (fast-forward)
git push cdw fixed-transformed-A-production-ready-2026-07-26 flexible-theta-transformed-A-production-ready-2026-07-26
```

## Final verdicts

```
FIXED_TRANSFORMED_A = MERGED_TO_PRODUCTION
FLEXIBLE_THETA = MERGED_OPT_IN
FIXED_DEFAULT_A_COORDINATE = legacy_z
LEGACY_Z_REPLICATION_MODE = retained
POST_MERGE_SMOKE = pass
THETA_DERIVATIVE_BACKEND = cplus_fixed_dual_secant
THETA_GENERIC_MOMENTS_CALLS = 0
THETA_BLOCK_SPEEDUP = 6.43x
THETA_BLOCK_ALLOCATION_REDUCTION = 99.66%
FLEXIBLE_THETA_POST_OPTIMIZATION = still_not_faster
PRODUCTION_MERGE = complete
```

## Key decisions made by the user, recorded here for provenance

1. **Flexible theta merged despite failing its own practical-value gate** (loses to fixed
   transformed-A by 3.0%/11.4% at δ=1/δ=2, confirmed both before and after a 6.4x theta-derivative
   speedup specifically built to test whether wall-clock cost was the cause — it wasn't). Explicit
   user decision: merge anyway as an inert, non-default, not-recommended opt-in, available for
   future refinement rather than held back entirely.
2. **CM/CM+meanZC/origin-ZC regression matrix run before merging** (user's explicit choice over
   proceeding with unrestricted-only validation). Result: zero regression, confirmed via real
   D=20/W=80,000 calls through the actual `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`
   public drivers (`test_backend_manifest_cm_originzc.jl`, ALL PASS). Note: this confirms no
   regression *to* those families from this merge — transformed-A/flexible-theta working *in
   combination with* CM/ZC/origin-ZC restrictions remains unimplemented (no driver wires that
   combination), unchanged from the original task's scope.
3. **One combined merge with two separate tags**, rather than two literal git merge commits — the
   fixed and flexible coordinate systems were built as one shared architecture from the start (by
   design, to avoid duplicate near-identical drivers), so the code isn't cleanly separable into
   two independent diffs without re-authoring history. The two tags at the same commit distinguish
   the two releases' independent readiness certifications, which is what substantively matters.

## Post-merge state

- Existing production entry points (`run_profile_checkpointed`, `run_polish_checkpointed`,
  `run_cm_upper_checkpointed`, `run_originzc_upper_checkpointed`) are **completely unmodified** in
  behavior — `A_coordinate_mode=:legacy_z` remains the default and only reachable mode through
  them; `run_polish_checkpointed_unified` is the new opt-in entry point for both
  `A_coordinate_mode=:powered_aspace` (fixed theta) and `trade_elasticity_mode=:flexible`.
- No existing default was flipped. Adopting `:powered_aspace` as the default for *new* fixed-theta
  campaigns (the task brief's own suggested next step, given the clean matched-comparison win) is
  a decision for whoever runs the next campaign, not made unilaterally by this merge.
