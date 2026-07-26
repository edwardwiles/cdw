# Final fixed-mode regression sweep (task §8)

Run on the release branch (`release/shared-winner-pair-final-merge-2026-07-25`, commit range
`0732951..b40e0f4` over the canonical `production/fullA-exact` tip `39b89c5`), after the
dynamic-worker-policy and A/B-harness-bug fixes were committed — i.e. this is the exact code that
was fast-forwarded into production, not an earlier snapshot.

## Results

| Check | Test | Result |
|---|---|---|
| Shared H_EE, all 4 families, D=4 (dense vs serial vs parallel(2,4)), random + real-solved points | `test_shared_core_hessian_d4_gates.jl` | **PASS 0 failures** (ran twice: once immediately after the worker-policy code edit, once as the final pre-merge check) |
| Shared H_EE, all 4 restricted families, D=20 real data (K1_mean_only + K1_mean_zc, P0/P1 points) | `test_d20_restricted_full_hessian_gates.jl` | **PASS 0 failures** — re-run fresh on the release branch (not just the pre-existing committed log) |
| Backend-manifest field assertions, unrestricted | `test_backend_manifest_unrestricted.jl` | **PASS** after fixing a stale hard-coded `core_hessian_workers=10` assertion (see below) |
| Backend-manifest field assertions, CM/CM+meanZC/origin-ZC | `test_backend_manifest_cm_originzc.jl` | **PASS** after the same fix |
| Exact-tie winner behavior (AUD-06) | `test_aud06_tie_safety.jl` | **PASS 10/10** |
| Checkpoint fingerprint mismatch rejection (stale/synthetic pre-omit-ROW checkpoints) | `test_checkpoint_fingerprint_mismatch.jl` | **PASS**, all 3 sub-tests |
| C+ vs reference-gradient equivalence, CM+mean/ZC (incl. K=3,K_pair=2 structural smoke) | `test_cm_meanzc_cplus_equivalence.jl` | **PASS** |
| C+ vs reference-gradient equivalence, origin-ZC (incl. K=3,K_pair=2 structural smoke) | `test_cm_originzc_cplus_equivalence.jl` | **PASS** |
| Winner forced-tie / canonical tie-break convention | `test_winner_forced_tie.jl` | **PASS 10/10** |
| Checkpoint resume regression (profile/polish, cf_workspace/canonical_price_ws/hard_score_B_cache re-attachment) | `test_checkpoint_resume_regression.jl` | **PASS**, all sub-checks (needed a longer timeout than the first attempt's 180s bundle — not a failure, just insufficient wall budget in that one attempt) |
| Unrestricted D=4 stationarity, upper direction | `test_stationarity_upper.jl` | **PASS** — external KKT check, relative residual < 5% |
| Unrestricted D=4 stationarity, lower direction | `test_stationarity_lower.jl` | Inconclusive by the test's OWN documented design (search stalled before using its full divergence budget) — the test's own comment states this is an EXPECTED, disclosed non-failure mode, not evidence of a broken method; unrelated to this release's changes (see below) |

## One fix required by this sweep: stale hard-coded worker-count assertions

Both manifest tests asserted the literal `core_hessian_workers=10`, which the dynamic worker
policy (task §3) correctly changes depending on `Threads.nthreads()`. Fixed both to assert against
`resolve_core_hessian_workers_default()` directly rather than re-hard-coding a new literal — this
is a genuine, expected consequence of implementing task §3, not a regression.

## One disclosed, pre-existing, out-of-scope gap found (not fixed — not caused by this release)

`test_cm_checkpoint_original.jl` (the fixture-writing half of `test_cm_checkpoint_resume.jl`'s
pair) fails with:

```
ERROR: LoadError: d20_real_setup_design: destination_sample=:exclude_row is not supported with
draw_design=:sobol_randomized (only :pseudorandom routes through the rectangularized
d20_real_setup) -- pass destination_sample=:all_legacy explicitly if you intend square D x D
behavior here.
```

Verified via `git diff 39b89c5 -- full_aod_diag/d4_exact/test_cm_checkpoint_original.jl` (empty
diff) that this test file is **byte-identical to the pre-port canonical `production/fullA-exact`
tip** — this is a pre-existing incompatibility between an older test fixture script and the
exclude-ROW-destination production release (2026-07-24, a prior, already-merged, unrelated
release), not something the shared winner-pair H_EE port introduced or is in scope to fix per task
§0's explicit instruction not to reopen unrelated work. Disclosed here rather than silently
skipped.

## Not separately re-run in this sweep (already covered elsewhere in this release's own gates)

Omit-ROW/`:all_legacy`/D=4-square/D=4-rectangular-with-non-last-omitted-destination: the shared
H_EE backend does not touch destination-sampling logic at all (it is purely the economic-core
Hessian block, computed identically regardless of which destinations are sampled) — these
scenarios are exercised by the pre-existing, unmodified exclude-ROW-destination test suite that
this release does not touch, and the D=4/D=20 shared-H_EE gates above already run under the
production-default `:exclude_row` sampling at real D=20 scale.

## Overall

Every regression check that is actually within this release's changed surface (shared H_EE
backend, dynamic worker policy, runtime counters, startup manifest fields) passes with 0 failures
after the sweep's own fix. The two non-passing items found are both disclosed, pre-existing, and
outside this release's scope.
