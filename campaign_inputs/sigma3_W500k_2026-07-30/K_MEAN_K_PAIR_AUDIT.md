# K_mean/K_pair semantics audit for "K=2" (ZC-only, CM+ZC families)

**Question**: the campaign spec requests "K=2" for the ZC-only and CM+ZC families, describing it
as restrictions for "k=1, k=2, corresponding to first and second moments." The codebase's actual
restriction machinery is parameterized by a PAIR `(K_mean, K_pair)`, not a single `K`, with the
invariant `0 <= K_pair <= K_mean`. Before setting `K_mean=K_pair=2`, the campaign brief requires
proving this is the exact intended implementation, not an assumption.

## Finding: K_mean=2, K_pair=2 is confirmed correct, from three independent sources

1. **`full_aod_diag/d4_exact/cm_originzc_target_layout.jl`** (`OriginByPowerLayout`/
   `SharedByPowerLayout`): `K_mean` = number of "power levels" `k=1:K_mean` that get an
   equal-mean-type restriction (`mean_targets`); `K_pair` = how many of those levels (a prefix
   `1:K_pair`) ALSO get a pairwise/zero-covariance-type restriction (`pair_targets`). The file's
   own `layout_fingerprint` docstring gives `"origin_by_power:D=20:K_mean=2:K_pair=2"` as its
   worked example — i.e., `K_mean=2:K_pair=2` at real D=20 is already an anticipated,
   documented configuration in this codebase, not a novel combination.
2. **`full_aod_diag/d4_exact/originzc_production_stage_runner.jl`** (header comment, the real
   production stage-runner's own CLI contract): `ORIGIN_K_PAIR` env var — "default = ORIGIN_K_MEAN
   for the zero-covariance restriction, 0 for the mean-only restriction." Both requested families
   (ZC-only, CM+ZC) are zero-covariance-type restrictions (`origin_specific_moments_zero_covariance`
   / `:cm_plus_equal_means_zero_covariance` / `:cm_plus_moments`), so under this codebase's own
   established default rule, requesting "K=2" for a ZC family unambiguously means
   `K_mean=2, K_pair=2` — K_pair defaulting to anything less than K_mean is specifically the
   *mean-only* (non-ZC) arm, which is not what was requested.
3. **Plain reading of the task brief**: "restrictions for k=1, k=2" (both levels), "corresponding
   to first and second moments" — i.e. both power levels are meant to be restricted, not just one
   with the other partially/un-restricted. This matches `K_mean=2` (both levels get the mean
   restriction) with `K_pair=2` (both levels also get the zero-covariance restriction), not e.g.
   `K_pair=1` (only level 1 gets zero-covariance).

**Conclusion: `K_mean=2, K_pair=2` is confirmed as the exact intended implementation** for both
the ZC-only (family 4) and CM+ZC (family 5) restrictions in this campaign, via
`OriginByPowerLayout(D=20, K_mean=2, K_pair=2)` (origin_zc) and
`CMMeanZCConfig(cm_extension=:cm_plus_moments, meanzc_K_mean=2, meanzc_K_pair=2)` (cm_meanzc+zc).

## Not yet done (deferred to Phase D driver construction / Phase E preflights)

- The 2026-07-28 shakedown campaign's `campaign_cm_family_runner.jl:61-62` hardcodes
  `MEANZC_K = ORIGINZC_K = 1` — this is that PRIOR campaign's own script and is left untouched
  here (editing a past campaign's runner in place is not the right fix); the new sigma3/W500k
  campaign driver (Phase D) will set `K_mean=K_pair=2` explicitly when it is built, informed by
  this audit.
- Restriction counts by k/type (`n_eta = K_mean*D = 40` at D=20 for the mean-column count is
  immediate from `n_eta(::OriginByPowerLayout) = K_mean*D`; the pair-column count per level is
  `D*(D-1)/2 = 190`, so `K_pair=2` levels contribute `2*190=380` pairwise restrictions on top of
  the `2*20=40` mean restrictions — these are read off the layout formulas directly, not yet
  cross-checked against a live-built context's actual packed dimensions), inner dimension, packed
  Hessian dimension, and workspace-memory estimate will be reported from the REAL live context
  objects (not re-derived by hand) as part of Phase E's "Confirm K=2" preflight, per the campaign
  brief's own instruction to source these facts from live objects.
