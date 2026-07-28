# Common Fréchet / flexible-CM harmonization: canonical merge (2026-07-28)

## Merge method

Clean fast-forward. `refactor/harmonize-frechet-with-flexible-CM-2026-07-28` was branched from
`cdw/production/fullA-exact@7a185ec` (the Phase 1 no-H release head) and never diverged from it
locally, so merging back was a pure fast-forward, no rebase or merge commit.

## Steps taken (in order)

1. Confirmed `cdw/production/fullA-exact` had not advanced beyond `7a185ec` since Phase 1
   (`git fetch cdw --tags`; `git merge-base --is-ancestor cdw/production/fullA-exact HEAD` → true).
2. Asked the user explicitly whether to merge Phase 2 to production now; user confirmed
   "Yes, merge and push now."
3. `git branch -f production/fullA-exact db786c0` (local fast-forward).
4. `git tag -a common-frechet-shared-CM-core-release-2026-07-28 db786c0`.
5. `git push cdw production/fullA-exact` → `7a185ec..db786c0`.
6. `git push cdw common-frechet-shared-CM-core-release-2026-07-28` → new tag.
7. `git fetch cdw --tags` + `git merge-base --is-ancestor db786c0 cdw/production/fullA-exact` →
   **CONFIRMED**.
8. `git status --short` → clean.
9. Fresh, independent `git worktree add` at the new remote head (detached, verified identical to
   `cdw/production/fullA-exact`), ran post-merge smokes for flexible CM and common Fréchet.

## Exact production SHA

```
cdw/production/fullA-exact = db786c00d323f17d452e7c62ac3abf91396c4d22
```

(previous canonical tip, Phase 1's release: `7a185ec865228760da42560da813823d17230366`)

## Exact tag

```
common-frechet-shared-CM-core-release-2026-07-28
  tag object: 89069ef3c25d981cefa3d809529ab025a3bfacb5
  points at:  db786c00d323f17d452e7c62ac3abf91396c4d22
```

## Ancestry proof

```
git merge-base --is-ancestor db786c00d323f17d452e7c62ac3abf91396c4d22 cdw/production/fullA-exact
  -> exit 0 (true)
```

## Post-merge smokes (fresh, independent worktree at the new production head)

| Family | wall (s) | n_eval | n_grad | kappa | Matches Phase 1 smoke? |
|---|---|---|---|---|---|
| flexible CM | 185.9 | 4 | 3 | 0.05060338369820172 | yes, exact |
| common Fréchet | 169.0 | 2 | 2 | 0.020553415111748796 | yes, exact |

Both smokes reproduce Phase 1's own post-merge smoke results (same `kappa`, same eval counts,
same forbidden-counter values -- `dense_Frechet_G_materializations=14` for common Fréchet,
otherwise all 0) **exactly**, confirming the harmonization refactor changed zero observable
production behavior for either family beyond the intended internal consolidation.

## Clean status

```
git status --short  -> (empty)
```

## Verdict

```
PHASE_2_PRODUCTION = merged_tagged_smoked
CANONICAL_PRODUCTION_SHA = db786c00d323f17d452e7c62ac3abf91396c4d22
TAG = common-frechet-shared-CM-core-release-2026-07-28
REMOTE = cdw (git@github.com:edwardwiles/cdw.git)
```
