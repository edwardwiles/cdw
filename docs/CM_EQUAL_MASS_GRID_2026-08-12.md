# `L` now means "L equal-mass buckets" for the production CM / CM+ZC / Common-Fréchet families

**2026-08-12.** Worktree `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`, branch
`feature/pq-free-mass-reparam-2026-08-10`. Task brief: `docs/TASK_CM_EQUAL_MASS_GRID_2026-08-12.md`.

**This is a scientific change, not a refactor.** It changes which restriction the three CM-flavoured
paper families impose. Every existing CM result at `L = 50` is on the old grid. Read §6 before
resuming anything.

---

## 1. The brief's claims, re-measured before anything was changed

The brief asked for its own measurements to be reproduced rather than trusted. All three reproduce.

**Claim 1 — CM's thresholds are already closed-form theoretical, and that part is correct.** Confirmed.
`precalc_common_marginals_cdf` (`common_marginals_moments.jl:215`) computes
`z = theoretical_u_threshold.(probs_used)` with `theoretical_u_threshold(p) = -log(1-p)`, the exact
Exp(1) quantile. Unchanged by this task.

**Claim 2 — the GRID of probability levels was the problem, and three of them existed.** Confirmed by
running the brief's snippet against the real `nested_quantile_grids.jl`:

| where | grid | at L=50 |
|---|---|---|
| CM's bare default (`probs === nothing`, `cm_equal_grid_probs`) | `range(1/L,(L-1)/L,length=L)` | 50 levels, spacing **0.0195918** |
| what the campaign injected (`resolve_cm_probs` → `nested_grid_sequence([10,20,50])[50]`) | dyadic largest-gap bisection | 50 levels → **51 buckets**, masses only **0.015625 or 0.03125** |
| CM+PQ family #7 (`cm_pq_probs_grid`) | `k/G`, `k=1..G-1` | 49 levels → **50 buckets of exactly 0.02** ✔ |

Also measured, and worth recording: the dyadic grid at L=10 gives masses {0.0625, 0.125} and at L=20
{0.03125, 0.0625}. It contains 0.25 and 0.5 but **not** 0.2 — every one of its points is a multiple
of 1/64, so `k/50` and the dyadic grid share exactly one point (`p = 0.5`). They are not nested in
either direction; see §5.

**Claim 3 — which families are affected.** Confirmed by reading both files. `paper_upper_v1.toml`
sets `L = 50` with no explicit `probs` for `COMMON_MARGINALS`, `COMMON_FRECHET` and `CM_PLUS_ZC`
(lines 97, 108, 140); `family_start_chain.jl::fam_kwargs()` then injected
`probs = resolve_cm_probs(kw.L)`. `ORIGIN_ZC` and `UNRESTRICTED` have no CM grid and are untouched.

---

## 2. The decision: `L` = 50 means 50 buckets, hence **49 levels**

The brief required this to be decided deliberately and stated. **`L` is a bucket count.** At `L = 50`
the cutpoints are `k/50` for `k = 1..49`, giving 50 buckets of mass exactly 0.02.

`p = 1` is excluded and the count is *not* padded back to 50. Its CDF contrast
`1{U_o ≤ ∞} − 1{U_ref ≤ ∞}` is identically zero — a structurally zero moment column and a singular
KKT, not merely an uninformative row. `p = 0` is degenerate the same way from the other end. Family
#7 reached the identical conclusion independently (`cm_pairwise_quantile_config.jl`, "WHY G-1 LEVELS
AND NOT G"), and this change makes `cm_pq_probs_grid(G)` and `cm_equal_mass_probs(G)` numerically
identical functions — deliberately still separate, see §4.

This forces a distinction the codebase previously did not need, because every old grid happened to
have `n_levels == nominal L`:

* every CM **config** surface — `protocols/*.toml`'s `L`, `FamilySeedSpec.L`, `CMConfig.cm_grid_size`
  — states a number of **BUCKETS**;
* every CM **core** function's `L` argument — `precalc_common_marginals_cdf` and everything it feeds
  (bin tables, Hessian architecture, `n_cm_moments`) — is a number of **LEVELS**.

The translation is `n_levels = length(probs)`, which is the single authority everywhere. It is
already enforced from underneath by `precalc_common_marginals_cdf`'s own `length(probs) == L`
assertion, so a call site that forgets fails loudly at context-build time rather than silently
solving a different problem.

---

## 3. What changed, file by file

| file | change |
|---|---|
| `common_marginals_moments.jl` | **NEW** `cm_equal_mass_probs(n_buckets)` = `(1:(n_buckets-1))./n_buckets`; hard-errors for `n_buckets < 2`. Full buckets-vs-levels docstring. Also re-checked and documented that `theoretical_u_threshold`'s grid-symmetry step survives: `1 − k/L = (L−k)/L`, and `k ↔ L−k` is a bijection of the new grid onto itself, so the `p → 1−p` relabelling its derivation relies on still holds. |
| `multistart_seed_generator.jl` | `resolve_cm_probs(L)` now returns `cm_equal_mass_probs(L)` for **all** `L` — no `{10,20,50}` special case, no fallback branch. **NEW** `cm_n_levels(spec) = length(spec.probs)`. All four CM builder call sites (`:cm_zc`, `:common_frechet`, `:cm_only`, and the CM-only companion inside `companion_implied_nu_cmzc`) now pass `L = cm_n_levels(spec)` instead of `spec.L`. |
| `paper_upper_v1_orchestrator/family_start_chain.jl` | `fam_kwargs()` does the one buckets→levels translation, `kw = merge(kw, (L = length(probs), probs = probs))`, and **logs it explicitly** each run. |
| `cm_checkpoint.jl` | **NEW** top-of-function check in `run_cm_upper_checkpointed`: `probs === nothing \|\| length(probs) == L`, with an error message that names the buckets-vs-levels distinction. Turns the deep `@assert` into a self-explaining error at the boundary a caller controls. |
| `cm_config.jl` | `cm_equal_grid_probs` **unchanged**; prominent docstring warning that it is neither equal-mass nor the production grid, and why it was left alone (§4). |
| `nested_quantile_grids.jl` | Header note: no longer the production grid, kept for the genuine nesting property, who still depends on it. |
| `cm_pairwise_quantile_config.jl` | Comment only: records that the two grids now coincide numerically and why they stay separate functions. |
| `smoke_objective_mode_min_delta_fixed_gp.jl` | Two call sites paired `L = 50` with `resolve_cm_probs(50)`; now derive `L = length(CM_PROBS_L50)`. |
| `test_multistart_seed_generator.jl` | Grid testset rewritten (see §7). |
| `diag_cm_equal_mass_grid_ab_2026-08-12.jl` | **NEW** — the before/after Δ\* harness of §6. |

### What was deliberately NOT changed

* **`cm_equal_grid_probs` / `CMConfig`'s `:equal` rule.** Changing its return length would break ~40
  historical diagnostic and benchmark scripts that pair it with a matching `L` (`probs =
  cm_equal_grid_probs(L)` then `build_cm_production_context(...; L = L, probs = probs)`), silently
  redefining their restriction while fixing nothing on the production path — `CMConfig`'s `:equal`
  rule drives benchmark harnesses (`c14_*`, `matched_outer_benchmark_cm_*`), and
  `run_cm_upper_checkpointed` takes `probs` explicitly and treats `cm_grid_rule` as checkpoint
  metadata only. It is left alone with a loud docstring instead. **This is a real remaining
  inconsistency, flagged rather than half-fixed:** `CMConfig(cm_grid_rule=:equal, cm_grid_size=50)`
  still means 50 levels / 51 unequal buckets. Say the word and it becomes a follow-up.
* **`nested_quantile_grids.jl` itself**, and every direct caller of `nested_grid_sequence` — see §5.
* **Family #7 (CM+PQ)**, which stays on `cm_pq_probs_grid` and on `family_start_chain.jl`'s
  `NO_PROBS_DRIVERS` exemption. Its `L` is PQ bins and its CM grid arrives separately as
  `cm_grid_size`, so routing it through `resolve_cm_probs(spec.L)` would be the wrong grid derived
  from the wrong quantity; and the `L | G` superset condition it rests on must not silently follow
  whatever CM's production grid does next.

---

## 4. The nesting trade-off (the brief's trap 2), resolved by measurement

`nested_quantile_grids.jl` exists for a real and different reason: the dyadic grid makes
`Q_10 ⊂ Q_20 ⊂ Q_50` hold by construction, which is what licenses reading a `kappa(L)` sweep as
"more restrictions ⇒ weakly lower kappa" (`docs/fullA_common_marginals_handoff.md` §5 records the
non-monotone `0.1522 < 0.1552 < 0.1564` that motivated it).

**Equal-mass grids nest only when the sizes divide.** `Q_10 ⊂ Q_20` (10 | 20) and `Q_10 ⊂ Q_50`
(10 | 50) still hold; `Q_20 ⊄ Q_50` does not, since 20 ∤ 50. So exactly one link of the ladder is
lost.

**Nothing on the live production path depends on it.** Checked, not assumed:

* `paper_upper_v1` — the live protocol — runs `L = 50` **only** (all three CM families). No L-sweep,
  no cross-L comparison. This is the whole of the production dependency, and it is empty.
* The multi-L users all call `nested_grid_sequence` **directly** and are untouched:
  `c13_d20_cm_upper_continuation.jl` (warm-start L=10→20→50 continuation, which already has an
  explicit "start point infeasible under this L, fall back to calibration" branch),
  `CMConfig`'s `:nested_family` rule, `cm_production_stage_runner.jl` (referenced only from docs, not
  from any live orchestrator — `paper_upper_v1_orchestrator/` is the live one),
  `campaign_cm_family_runner*.jl`, and ~50 `c13_*`/`c14_*`/`diag_*` scripts.

So this is a trade-off the change makes, not a dependency it breaks. **If a future L-ladder is
wanted with equal-mass grids, choose sizes that divide** — e.g. `{10, 50}`, `{5, 10, 50}`,
`{2, 5, 10, 50}` — and the nesting is exact and free.

---

## 5. Gates

Run with `OPENBLAS_NUM_THREADS=1`, Julia from `~/.juliaup/bin`.

| gate | result |
|---|---|
| `test_cm_pairwise_quantile_d4_dense_oracle.jl` (family #7 must keep working) | **136/136 PASS**, before and after. Check-by-check diff against the pre-change baseline: identical except six threaded-reduction residuals that move at the 1e-15 level (e.g. check 4 `rel L2` 1.163e-15 → 5.899e-16). |
| `test_multistart_seed_generator.jl` (the direct gate on what changed) | **892/892 PASS, 0 failed, 0 errored**, 15 testsets, at `-t 10`. Includes the rewritten 33-assertion grid testset, both spec-constructor testsets, and the real D=20 `build_family`/`evaluate_family`/`qualify_economic_point`/reproducibility paths that now run on the new grid. |
| `test_cm_lookup_fg_twofamily_2026-08-05.jl` (CM D=4 dense-truth FG) | `:suffix` — the real production basis — **12/12 PASS**. `:interval` **12 FAIL**. **Pre-existing, not caused by this change**: re-run at pristine `HEAD` in a throwaway git worktree, the failure list is byte-identical (13 PASS / 12 FAIL, same residuals to the last digit, e.g. `max\|Δg\|=0.020372097257968146`). Flagged for whoever owns the interval basis; out of scope here. |
| `test_phaseB1_cmlookup_production_correctness.jl d4` | Does not run, at `HEAD` or with this change: `UndefKeywordError: keyword argument include_truncated_moment not assigned` on its first call. **Pre-existing bit-rot** from the 2026-08-05 required-kwarg hardening, unrelated to grids. Not fixed here (out of scope, and fixing it would mean choosing a scientific value for that script). |

An honest note on process: my first `test_multistart_seed_generator.jl` run errored with
`hessian_core_winner_pair!: workers=6 not in workspace's precomputed worker_counts`. That was my
launch mistake — `-t 6` is not one of the precomputed counts `[1,2,4,8,10,19,20]`
(`core_exact_hessian.jl:602`) — not a regression. Re-run at `-t 10` (production's own thread count).

---

## 6. The scientific result: Δ\* before and after, real D=20

`full_aod_diag/d4_exact/diag_cm_equal_mass_grid_ab_2026-08-12.jl`. Fixed-state value-only evaluation
(`build_family` + `evaluate_family` — the same production evaluator the seed qualifier and the outer
driver's verification path call) at the **real calibration point**, with the economic point, draws,
σ, W, δ, family definitions and every solver option held exactly fixed and **only the grid varying**.
Arms interleaved by grid within each family. `paper_upper_v1`'s own `[scientific]` block: D=20,
W=100,000, δ=1.0, σ=3.0, `sobol_randomized`/20260719, `exclude_row`, Brazil–Korea gravity
exclusions, `inner_lower_limit=-10.0`.

| family | CM moments | Δ\*(θ_calib) **OLD** (dyadic, 50 levels) | Δ\*(θ_calib) **NEW** (equal-mass, 49 levels) | NEW/OLD |
|---|---|---|---|---|
| `COMMON_MARGINALS` | 950 → 931 | 5.128734484e-4 | 5.112589898e-4 | 0.99685 |
| `COMMON_FRECHET`   | 950 → 931 | 5.141318470e-4 | 5.123876018e-4 | 0.99661 |
| `CM_PLUS_ZC`       | 950 → 931 | 3.897339266e-3 | 3.921222767e-3 | 1.00613 |

**All six arms converged and verified** (`verified=true`, `class=VerifiedSolved`). Five returned
`inner_status=0`; `CM_PLUS_ZC` under the new grid returned `-100`, which is the documented
"opttol_abs below the achievable floor" convergence, not a failure, and it still passed the full
verification gate.

**Reading this.** Δ\* moves by 0.3–0.6% — small, finite, same order, nothing near the unbounded
regime. That is the expected shape: both grids impose genuine CDF-contrast restrictions at the same
point, at differently-placed cutoffs and with one fewer level. **No monotonicity is implied and none
should be read into the mixed signs** (CM-only and Common-Fréchet tick down, CM+ZC ticks up): the two
grids are *not* nested in either direction — they share exactly one cutpoint, `p = 0.5` — so
"fewer restrictions ⇒ weakly lower Δ\*" simply does not apply.

Wall-clock in the log is **not** a valid comparison: the first arm of each family absorbs JIT
(`COMMON_MARGINALS` OLD 56.8s vs NEW 6.7s is compilation, not the grid).

**A small-W control worth recording.** The same harness at W=8000 fails all six arms with
`nStatus=-300` (confirmed infeasible) — **including both old-grid arms**. That is the known D=20
small-W infeasibility (950 CM restrictions on 8,000 draws), not anything to do with the change, and
it is why the A/B is reported at the production W=100,000 only.

---

## 7. What a resume does when the grid changes (the brief's trap 3)

**Existing checkpoints are self-protecting, but silently so.** `run_cm_upper_checkpointed` records
`cm_probs` — the exact cutpoints, schema field 197, "NOT re-derived from L on resume, taken
verbatim" — and on resume executes `L = resumed.cm_L; probs = resumed.cm_probs`
(`cm_checkpoint.jl:1140`), **overriding the caller**. So:

* an old `L = 50` dyadic checkpoint resumed today continues on the **old** grid, correctly and
  consistently — there is no way to half-mix two grids inside one run;
* but the caller is **not told** its requested grid was ignored. Unlike `destination_sample`,
  `marginal_restriction` and `cm_feature_family_count`, which hard-refuse on mismatch, the grid is
  silently inherited.

I did not add a grid-mismatch refusal, because the inherit-and-continue behaviour is the *correct*
one for a mid-flight resume and hard-refusing would strand every in-progress checkpoint. **The
practical rule: an old checkpoint is an old-grid run. Do not treat a resumed old checkpoint's Δ\* or
kappa as comparable to a fresh run's.** If you want the grid change to apply, start fresh.

Fresh runs are also distinguishable after the fact: `family_spec_descriptor` embeds the full probs
list in the manifest digest, so old-grid and new-grid seed qualifications hash differently and
cannot be confused.

---

## 8. Follow-up (same session): closing the bare-`L` gap

The first pass made `L` mean buckets on the routes that go through `resolve_cm_probs` — the protocol
orchestrator and the seed-spec constructors. It left one reachable route that did **not**: calling
`run_cm_upper_checkpointed(...; L = 50)` with **no `probs`** fell through to
`precalc_common_marginals_cdf`'s own `probs === nothing` branch, `range(1/L,(L-1)/L,length=L)`.
Measured at L=50 that is 50 levels → **51 buckets, 49 of mass 0.0195918 and 2 of mass 0.02**. So
`L = 50` still meant two different restrictions depending on whether the caller also passed a grid.

**Fixed.** `run_cm_upper_checkpointed` now resolves the grid itself:

* `probs` omitted → `L` is read as a **bucket count**, `probs = cm_equal_mass_probs(L)`, then
  `L = length(probs)`. Logged on every run.
* `probs` supplied → `L` is the **level count** and must equal `length(probs)`, else a hard error
  naming the buckets-vs-levels distinction.

Either way `L == length(probs)` holds from that line onward, which is what every builder below
requires. Note this also changes bare `L = 10` calls in old scripts from 10 levels to 10 buckets
(9 levels) — the same scientific change, applied consistently, rather than a second silent grid.

**New gate**: `test_cm_grid_l_means_buckets_2026-08-12.jl` — **103/103 PASS**. It checks the *promise*
(bucket masses re-derived from cutpoints, not the cutpoint formula asserted) across all six routes
that can state an `L`: `cm_equal_mass_probs`, `resolve_cm_probs`, the three `*_family_spec`
constructors, `paper_upper_v1.toml`'s own `L` through `fam_kwargs()`'s translation, a bare-`L`
driver call, and family #7's `cm_pq_probs_grid`. Plus negative controls that the old default and the
dyadic grid really are *not* equal-mass, so the gate has content.

**End-to-end proof of the bare-`L` route**, real D=20/W=20,000 through the real driver, 60s budget:

```
[bare_L50] CM grid: L=50 equal-mass buckets (mass 0.02 each) -> 49 cutpoints k/50, k=1..49 (probs was not supplied)
checkpoint cm_L         = 49   (LEVEL count)
checkpoint length(probs)= 49
buckets induced         = 50
unique bucket masses    = [0.02]
VERDICT: 50 equal-mass buckets? true
```

(`knitro_status = -401` is the 60s wall-clock limit — this smoke gates grid resolution reaching the
solver, not the quality of the outer optimum. It did find a feasible incumbent, `Δ = 0.943 ≤ 1`.)

---

## 9. Is this in production? — **Yes, as of 2026-08-12: `origin/production/fullA-exact = e49c0ed`**

⚠️ **This section previously said "No".** That was true when written and is now superseded — the
branch was merged and pushed later the same session, on the user's instruction. Recorded as a
correction rather than silently rewritten, because the earlier version was pushed to Dropbox.

* Merge commit **`e49c0ed`**, a clean fast-forward of `production/fullA-exact` (`184750c` → `e49c0ed`;
  first parent is the old production tip, so no history was rewritten). Pushed to
  `git@github.com:edwardwiles/cdw.git`. **Rollback point: `184750c`.**
* The merge lands far more than this grid change: `paper_upper_v1` (protocol *and* orchestrator),
  the pairwise-quantile family (#6) and CM+PQ (#7) did not exist on production at all. Production's
  own 7 commits (CROSS families, `linsolver ma97`, the inner-opttol fix, `H_CZ :j_parallel`, the
  shared-`.opt` opt-in) are preserved and reconciled, not overwritten.
* Still true from the old text: **no campaign has been re-run**, and per §7 an old checkpoint
  resumed today stays on the old dyadic grid. Every completed `paper_upper_v1` result at `L = 50`
  predates this change.

**One file conflicted** — `multistart_seed_generator.jl`, three hunks, all additive divergence,
resolved as unions. **The conflict resolution was not the merge.** Three further defects existed
only in the merged tree, and each was found by running something rather than by the merge resolving
cleanly:

1. `build_family`'s `:cm_zc_cross` branch passed `L = spec.L`. That family (production, 2026-08-09)
   is CM-flavoured and takes its grid from `resolve_cm_probs`, which now returns `L-1` equal-mass
   cutpoints — so it handed a 50-**bucket** count to an argument counting **levels** and hard-errored
   against a 49-long grid. Now `cm_n_levels(spec)`, the same fix the four pre-existing CM builder
   call sites carry. `evaluate_family` and `companion_implied_nu_cmzc` already route the cross kinds
   through the diagonal path and were already correct.
2. The merged dependency guard is the **union** of both sides' required symbols, so all 28 consumers
   of `multistart_seed_generator.jl` need both sets. **26 branch-side consumers lacked the CROSS
   symbols — including `paper_upper_v1_orchestrator/family_start_chain.jl`, the live campaign cell
   runner, which died at include time.** 2 production-side cross tests lacked `fast_range_screen.jl`.
   All 28 fixed; CROSS ordering copied from `test_cross_seed_family_dispatch_2026-08-09.jl`.
3. The branch widened `FamilySeedSpec` from 9 fields to 12 for family #6, but production's
   `origin_zc_cross`/`cm_zc_cross` constructors still passed 9 positionally — `MethodError` at spec
   construction. Both widened with the same `:none`/`0`/`:none` off-sentinels every other non-PQ
   kind uses.

Gates on the merged tree, **both sides of the merge**:

| gate | result |
|---|---|
| `test_cmzc_cross_wiring_2026-08-09.jl` (production side) | **29/29 ALL PASS** |
| `test_cross_seed_family_dispatch_2026-08-09.jl` (production side) | **15/15 ALL PASS** |
| `test_cm_pairwise_quantile_d4_dense_oracle.jl` (family #7) | **136/136** |
| `test_multistart_seed_generator.jl` | **892/892**, 0 failed, 0 errored |
| `test_cm_grid_l_means_buckets_2026-08-12.jl` | **103/103** |
| `family_start_chain.jl` | loads through every include (dies only on a deliberately bogus protocol path) |

**Still open, and now more urgent because it is on production**: this tree does **not** carry the
2026-08-03 "require scientific params" hardening that `CLAUDE.md` describes.
`run_cm_upper_checkpointed` still defaults `W::Int = 80000`, `L::Int = 10`,
`draw_seed::Int = 20260719`, `delta::Float64 = 1.0` and `σHat = 3.0`. `L` is named explicitly in that
rule's own parameter list, so a caller who omits `L` entirely silently gets a 10-bucket CM. Worth a
deliberate pass.

**Note for the family #7 owner**: family #7 was merged to production as it stood at branch tip
`7fc668f`. Your notes have it as "ready EXCEPT seeds". Nothing about it was changed by this merge
beyond the include-list additions in item 2 above.

---

## 10. Open items for the user

0. **DONE — merged and pushed** (§9): `origin/production/fullA-exact = e49c0ed`. Rollback point
   `184750c`. No campaign has been re-run on the new grid yet; that is the next decision.
1. **`CMConfig`'s `:equal` rule still means levels, not buckets** (§3). Deliberate, to avoid silently
   redefining ~40 diagnostic/benchmark scripts. It is now the *only* remaining route on which "L"
   does not mean buckets, and it is benchmark-only. Flag if you want it unified.
2. **Existing `L=50` CM results are on the old grid** (§7). Nothing was re-run; the paper campaign's
   completed waves are dyadic-grid results.
3. **Two pre-existing broken CM gates** found while running the gate set (§5): the `:interval` arm of
   `test_cm_lookup_fg_twofamily_2026-08-05.jl` and all of
   `test_phaseB1_cmlookup_production_correctness.jl`. Both fail identically at pristine `HEAD`.
   Neither is caused by, nor fixed by, this change.
