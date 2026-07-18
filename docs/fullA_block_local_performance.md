# Full-A D=4: block-local and incremental `L_fix` performance

Phase 2 deliverable ("the most important scaling optimization"). Builds
`full_aod_diag/d4_exact/lfix_incremental.jl` — three additive, equivalence-tested evaluators for the
fixed-dual scalar `L_fix`, benchmarked against the existing full-rebuild `fixed_dual_L`
(`three_way_derivatives.jl`, unmodified).

## 1. Dependency graph (verified from code, not assumed)

Traced directly from `moments/hFunction.jl` and `full_aod_diag/moments_gammanorm.jl` (§ full detail
in `lfix_incremental.jl`'s header comment):

- Perturbing `Aod_theta[o,d]` (one entry of the free `D×D` matrix) changes **only** the `D` moment
  columns of destination `d`'s block (`d1 = d + (o'-1)*D` for `o'` in `1:D`) — because `MinInd!`
  picks a winner across **all** `D` origins at that destination, so the winner can switch even
  though only one origin's underlying price moved.
- If `(o,d) == (baseIndex,baseIndex)`, it **additionally** changes the single counterfactual column
  (index `D²+1`) via `hFunctionCounter!`'s `constConsσ[baseIndex,baseIndex]`.
- In the pivot-reduced coordinates, one `z_free` coordinate maps to exactly **one direct** `Aod`
  entry plus the **pivot** `Aod` entry (`gravity_elimination.jl::pivot_expand`'s affine combination)
  — up to **2** affected destinations per coordinate, never more, independent of `D`.
- `gamma'_focal` changes **only** the counterfactual column; zero `Aod` entries change;
  `hFunctionCounter!`'s `counterType==1` branch never calls `MinInd!` — this coordinate is smooth,
  no winner-switching machinery needed at all.

## 2. Closed-form simplification

Within any destination-`d` block, a "loser" column's raw value is `-P[d1]*denom[d]` — a **data
constant**, independent of which origin lost. Only the **winning** origin's column carries
θ-dependence. This collapses the λ*-weighted contribution of an entire destination block to:

```
contrib[s,d] = (SW[s]/gammafac) * ( CONST_d[d] + lambda*[d1(winner(s,d))] * pTsigma(winner(s,d),s) )
```

where `CONST_d[d]` is a pure data+λ* constant (zero draws-loop cost, computed once). A perturbation
therefore only needs the **new winner** and their `pTsigma` value per draw — not a full moment-row
rebuild.

## 3. Three tiers implemented

| Tier | What it recomputes per affected destination | Winner determination |
|---|---|---|
| `:block_local` (Tier 1) | ALL `D` origins' price/pTσ | full O(D) rescan |
| `:incremental` (Tier 2) | ONLY the 1-2 changed origin(s)' price/pTσ, reuses cached prices for the rest | full O(D) rescan (can't avoid touching all D cached values to find the true min/second-min) |
| `:incremental_o1` (Tier 3) | ONLY the 1-2 changed origin(s)' price/pTσ | **O(1) per draw** using cached (winner, winner_price, runner-up, runner-up_price) — see `update_winner_o1`'s docstring for the exact case-analysis proof. Falls back to Tier 2's O(D) rescan only in the rare case of 2 changed origins landing in the same destination (chaining two O(1) updates through an inexact intermediate runner-up was traced through and found to risk a wrong winner in a narrow edge case — correctness prioritized over speed there) |

## 4. Equivalence (mandatory before any timing was trusted)

`full_aod_diag/d4_exact/test_lfix_incremental.jl`: all three tiers checked against the trusted
`fixed_dual_L` across **every one of the 16 reduced coordinates × both signs × up to 4 h values
(0.02/0.01/0.005/0.001)**, at `upper_maxit40`, `lower_stalled`, and 8 random feasible perturbations
(2 skipped for genuine base-solve infeasibility, a separately-documented phenomenon this
investigation has certified via an independent LP, not a bug here). **ALL PASS at machine precision**
(`max|tier - true| ~ 1e-16 to 1e-15` throughout, 704 total checks). `profile_lfix_tiers.jl`
additionally cross-checks the full **16-dim gradient** itself (not just the per-perturbation scalar)
across all four methods (full-rebuild, block-local, incremental, incremental_o1, incremental_o1
threaded) — also PASS, `max|g_tier - g_full| < 1e-8` (looser tolerance appropriate for a
central-difference gradient assembled from 32 already-near-machine-precision scalars).

Three real bugs were found and fixed during derivation (see the Phase 2 commit messages for the full
trace): a `lambda[o,d]` transpose-indexing error, a missing `U.^(-mu)`/`Uσ.^(-mu)` power transform,
an `Aod`-level-vs-`AodPow` conflation in the counterfactual formula, and a sign error in the
perturbation-delta assembly (`q_s = -ζ*-λ*'G_s`, so an increase in a destination's contribution
**decreases** `q`, not increases it). All four were caught by the self-validating cache constructor
(errors loudly on a >1e-8 mismatch rather than silently producing wrong values) or by the mandatory
equivalence test — direct evidence the "equivalence-test before trusting" discipline this
investigation follows is doing real work, not a formality.

## 5. Single-threaded timing, D=4/W=8000

`full_aod_diag/d4_exact/profile_lfix_tiers.jl`, N=20 full 16-dim central-FD gradients (32
perturbations each), `upper_maxit40`, commit `bbc0e47`:

| method | median (16-dim gradient) | speedup vs. full-rebuild |
|---|---|---|
| `full_rebuild` (`fixed_dual_L`, baseline — 32 full moment-matrix rebuilds) | 0.384037s | 1.00x |
| `block_local` (Tier 1) | 0.203115s | **1.89x** |
| `incremental` (Tier 2) | 0.071237s | **5.39x** |
| `incremental_o1` (Tier 3) | 0.052339s | **7.34x** |

**Realized, not theoretical.** The progression block-local → incremental → incremental_o1 tracks
exactly the mechanism each optimization targets: block-local already saves the destination-blocks
that aren't touched at all (~1.9x, matching roughly "recompute 1-2 of 4 destinations instead of 4" at
D=4 — a ratio that should IMPROVE at larger D, see §7); incremental additionally saves the redundant
D-1 unchanged origins' price recomputation within each touched block (another ~2.8x); incremental_o1
additionally removes even the O(D) rescan itself via the proven O(1) winner update (another ~1.4x).

## 6. Parallel FD scaling (thread-safe by construction)

Unlike `fixed_dual_L` (which calls `obj.moments!`, itself internally `Threads.@threads`-parallel over
draws — nesting that from multiple probe-threads would oversubscribe, not attempted), the incremental
tiers never touch `ctx.obj`'s mutable state or call any threaded production function — each probe
allocates its own local scratch, making `Threads.@threads` over the 32 FD probes safe with zero
coordination. Same N=20/16-dim-gradient benchmark, `incremental_o1` tier, `Threads.@threads` over the
16 coordinate probes (32 evaluations):

| `JULIA_NUM_THREADS` | median (16-dim gradient) | speedup vs. full-rebuild (1.00x baseline) | speedup vs. incremental_o1 single-threaded (0.052339s) |
|---|---|---|---|
| 1 | 0.050607s | 7.59x | 1.03x |
| 2 | 0.028370s | 13.53x | 1.78x |
| 4 | 0.023916s | 16.06x | 2.12x |
| 8 | 0.012497s | 30.73x | 4.05x |
| 16 | 0.008625s | 44.53x | 5.87x |

**Parallel efficiency drops off well below linear** (16 threads gives 5.87x, not 16x) — expected at
D=4: only 16 coordinates (32 signed probes) exist to distribute, so beyond ~4-8 threads each thread
gets very little work relative to Julia's own task-scheduling overhead for a workload this fine-grained
(each single probe is already sub-millisecond at incremental_o1's cost). **This ceiling should relax
at larger D** (more coordinates = more independent probe-work per gradient), not measured this
session — flagged as a specific, falsifiable prediction for the D=6/8 pilot (Phase 8), not asserted.

## 7. D-scaling — NOT measured this session, explicitly flagged as a gap

An attempt to benchmark D=6 (`context_scaled.jl::d_exact_setup_scaled(D=6,...)`) found the
calibration-anchored base point cold-infeasible (`nStatus=-300`) at that D/seed combination — the
same phenomenon already documented for the D=4 calibration point (`docs/fullA_continuation3_resume_audit.md`
§4) — and this session did not have a validated D=6 upper/lower candidate on hand to substitute (that
is Phase 8's job, gated on Phase 1-4 completing first, per the task's own ordering). **No D=6/8/10
timing claim is made here.** The code has no D=4-specific hardcoding (`D`, `D²`, `Aod_offset` are all
read from `ctx` throughout `lfix_incremental.jl`), so it is structurally expected to generalize, but
"structurally expected" is explicitly not the same standard as "measured" that this investigation
otherwise holds itself to — do not cite a D=6+ speedup number from this document.

**Reasoning for why the case should strengthen at larger D** (qualitative, not yet quantified):
block-local/incremental tiers' per-coordinate cost is `O(D·W)` (touch ≤2 destination blocks, each
`O(D)` origins), independent of the total `n_free = D²+1` coordinate count's own growth — vs.
full-rebuild's `O(D²·W)` per-coordinate cost (the ENTIRE moment matrix, every time). The RATIO
(full-rebuild cost) / (incremental cost) should grow roughly linearly in `D`, on top of `n_free`
itself growing as `D²` — i.e., the total-gradient speedup should grow, not just hold steady, as D
increases. This is exactly the mechanism `docs/fullA_scaling_projection.md`'s D^3.5-3.8 empirical
fit was probing for, now with a candidate structural explanation, not measured confirmation.

## 8. What Phase 2 changes about the outer-loop cost picture

Combined with Phase 1's finding (`docs/fullA_performance_profile_v2.md` §8: a full-rebuild `L_fix`
gradient is ~100% dominated by `lfix_perturbation_moments`, the 32 full moment rebuilds) — Phase 2's
`incremental_o1` tier directly attacks that exact cost, at D=4 replacing it with a per-coordinate cost
dominated by O(1) arithmetic and 1-2 cheap `O(W)` price recomputes. The realized 7.34x
single-threaded / 44.5x-at-16-threads speedup over the ALREADY-cheap-relative-to-`Delta_FD`
`fixed_dual_L` baseline (itself already ~3x cheaper than `Delta_FD` per Phase 1's `profile_gradient_methods.jl`
finding) compounds: incremental_o1(threaded) vs. `Delta_FD` (the original optimized-value ground
truth) is plausibly a **20-130x** wall-clock advantage at D=4 for the gradient alone (not yet directly
measured head-to-head in a single script — the ~3x `L_fix`-vs-`Delta_FD` figure and this section's
`incremental_o1`-vs-`L_fix` figures come from different profiling scripts at different points; a
direct `Delta_FD` vs. `incremental_o1` head-to-head is Phase 4's job).

## 9. Artifacts

- `results/fullA_d4/bbc0e47/profile_lfix_tiers/profile_lfix_tiers_nthreads{1,2,4,8,16}.csv`
- `results/fullA_d4/bbc0e47/profile_lfix_tiers/summary_nthreads{1,2,4,8,16}.txt`
- Equivalence test output: re-run `julia --project=. full_aod_diag/d4_exact/test_lfix_incremental.jl`
