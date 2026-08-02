# Profiled restricted-family outer-gradient: source snapshot (2026-08-01)

## Purpose

This branch (`architecture/profiled-restricted-outer-gradient-2026-08-01`) is the parallel
outer-gradient-layer workstream described in the task prompt. It owns profiled outer-coordinate
decoding, the shared economic A/gp fixed-dual gradient, family adapters, the gravity-pivot chain
rule, outer-gradient caches, gradient verification, and the matched outer-search A/B harness. It
does **not** own reduced economic layouts, restriction-family inner dual layouts, inner FG/Hessian,
or inner KNITRO debugging — that is `architecture/profiled-restricted-inner-endtoend-2026-08-01`
(or its descendant), running in parallel in a separate worktree.

## Repo / worktree

- Real repo: `/bbkinghome/edav/cdw` (bare-ish; all work happens in `git worktree` checkouts under
  `/bbkinghome/edav/gravity_robustness/worktrees/`).
- New worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-restricted-outer-gradient-2026-08-01`
- New branch: `architecture/profiled-restricted-outer-gradient-2026-08-01`
- Created via: `git worktree add <path> -b architecture/profiled-restricted-outer-gradient-2026-08-01 f439109`

## Base commit selection

Candidate tip commits inspected on `diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`:

```
f439109 (2026-08-01T09:44:32-04:00) Corrected A/B results after gp-drift bugfix: profiled +4-8% at matched iterations
1aeec43 (2026-08-01T08:59:51-04:00) CRITICAL FIX: profiled A/B harness let gp drift instead of holding it fixed
f483cae Parametrize gp-fraction A/B scripts; launch sweep at gp={0.995,0.99,0.988,0.985}
60042eb Add production's adaptive per-coordinate FD bandwidth; launch A/B at non-trivial gp start
1b4ec1b Fixed-iteration A/B (maxit=60, fast gradient): full still ~7.5x better, decisively
7560ca0 (2026-08-01T07:16:24-04:00) Build O(1)-incremental profiled gradient: 10-15x faster, validated to machine precision
fa61ee9 Run matched full-vs-profiled unrestricted outer A/B (upper, delta=1): full wins decisively
5302d86 (2026-08-01T05:31:24-04:00) Wire profiled outer gradient (fixed-dual FD) + evaluator; D4/D20 gates all PASS
2abb119 (2026-08-01T04:51:40-04:00) Commit validated unrestricted profiled inner formulation and D20 omit-ROW gates
```

**Selected base: `f439109`** (current tip of `diagnostic/profiled-scales-unrestricted-outer-ab-2026-08-01`
at branch time). Verified this is the "latest corrected descendant" required by the task, not `7560ca0`:

1. **gp genuinely held fixed in fixed-gp tests** — `1aeec43` is a live-user-triggered fix for exactly
   the opposite bug (the profiled A/B harness let `gp` drift instead of holding it fixed); `f439109`
   re-ran the A/B with `gp` genuinely fixed and is its correction commit. `7560ca0` predates this fix.
2. **Corrected gp derivative, no spurious `LPrime_bi*S_m` term** — confirmed by reading
   `full_aod_diag/d4_exact/profiled_lfix_incremental_2026-08-01.jl` (unchanged since `7560ca0`,
   still present at `f439109`) directly:
   `profiled_gp_component_analytic` returns
   `-cache.κ_cf * cache.σ * gp^(cache.σ - 1) * Tslot_bi / cache.M` — no `LPrime_bi*S_m` term. The
   file's own header/docstring documents the two-stage correction history: an *earlier* fix (in
   `PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md` section "3c corrected", now itself
   superseded) added an `LPrime_bi*S_m` term; that addition was found to be **spurious** — `cf.cf_raw`
   (`compressed_moments.jl:264`) is rebuilt fresh at the current `gp` and already contains its own
   `-gp^σ·wPrime_bi·LPrime_bi` term that exactly cancels `reduced_homogeneous_dual_contraction`'s
   separate `const_cf` term, collapsing the true derivative to the single `Tslot_bi` term. This
   matches the task prompt's description of the corrected result exactly (see task §6).
3. **Corrected unrestricted A/B traces** — `f439109`'s own commit message: "Retracts sections 6 and
   6b of the master doc (both computed with the gp-drift bug ... Re-ran the matched fixed-iteration
   A/B at three points with gp now genuinely held fixed ... Corrected picture: a small, consistent
   4-8% edge for the profiled formulation."

So `f439109` post-dates and supersedes `7560ca0` on all three counts the task asked to verify before
assuming a base commit. `f439109`'s working tree is clean except one untracked, non-code results
directory (`results/profiled_ab_2026-08-01/`, 8.1MB of A/B trace CSVs) which was left behind and is
irrelevant to this branch's base.

## Relationship to the inner-pipeline branch

`git merge-base --is-ancestor f439109 <inner-tip>` confirmed **`f439109` is already an ancestor of**
`architecture/profiled-restricted-inner-endtoend-2026-08-01`'s tip (`f1f969b` at branch time) — the
inner workstream already rebased onto this exact corrected unrestricted state. This branch and the
inner branch therefore share a common, already-validated unrestricted foundation; this branch does
not need to defer any unrestricted-layer fix to the inner branch, and the inner branch's own
`profiled_restricted_family_base_2026-08-01.jl` (read-only reference, not modified here) already
reuses `ProfiledEconomicMomentLayout`/`build_profiled_economic_moment_layout` unchanged for every
restricted family's economic block — confirming the mission's core premise (one shared economic
layout across families) is already an established design decision on the inner side, not something
this branch needs to negotiate.

## Key files reused unchanged from the base commit (SHA256 at branch time)

```
c410279...  full_aod_diag/d4_exact/profiled_outer_gradient_fd_2026-08-01.jl        (full-rebuild fixed-dual reference)
e2838f6...  full_aod_diag/d4_exact/profiled_lfix_incremental_2026-08-01.jl          (O(1) incremental gradient, corrected gp)
390b6b6...  full_aod_diag/d4_exact/profiled_economic_moment_layout_2026-08-01.jl    (shared economic-moment layout)
c2c00e4...  full_aod_diag/d4_exact/gravity_pivot_on_retained_2026-07-31.jl          (gravity-pivot chain rule)
3f8fd23...  full_aod_diag/d4_exact/outer_coordinate_layout_profiled_2026-07-31.jl   (decode_outer_profiled / outer_dim_profiled)
91ae5b9...  full_aod_diag/d4_exact/profiled_outer_evaluator_2026-08-01.jl           (evaluate_profiled_point; smooth-regime analytic reference kept as cross-check only)
```

`lfix_incremental.jl` (`update_winner_o1` / top-3 winner-cache mechanics, pre-existing production
code, not a 2026-08-01 file) is required and reused unchanged as documented in
`profiled_lfix_incremental_2026-08-01.jl`'s own `isdefined` guard.

## Working tree state at snapshot time

`git status`: clean (`nothing to commit, working tree clean`) immediately after `git worktree add`.
All new files this branch adds are listed in the master doc's manifest.
