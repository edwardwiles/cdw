# Handover: impose the ZC/mean restrictions on the EXPONENTIAL draw U, not the Fréchet z

Paste everything below the line as your task prompt in a fresh Claude Code session.

---

## Your goal

Add a **feature-basis option** to the ZC/mean restriction families so the power restrictions can be
imposed on the underlying exponential draw `U` instead of the Fréchet productivity `z = U^(-μ)`.

**Why.** `z ~ Fréchet(1, 1/μ)` has finite moments only up to order `1/μ`. The CC regularity condition
requires the `2K`-th moment to be finite, so `2Kμ < 1`, i.e. `K < 1/(2μ)`. At the real D20 calibration
`μ̂ = 0.13352` that caps `K ≤ 3`; at the D4 toy `μ = 1/6` exactly it caps `K ≤ 2` (see memory
`d4-toy-mu-is-one-sixth-so-k3-second-moment-infinite` — the existing D4 `K=3` gates are already
outside the regularity condition). `U ~ Exp(1)` has **all** moments finite (`E[U^k] = k!`), so the
ceiling disappears and `K` can in principle go as high as you like.

**This should be a small change**, and the user's framing is right: it is the same set of
restrictions applied to a different power of the same draw. The restriction *form* is unchanged —
mean `E[φ_o^k] = ν_k`, diagonal pair `E[φ_o^k φ_p^k] = ν_k²`, cross pair
`E[φ_o^{k1} φ_p^{k2}] = ν_{k1} ν_{k2}` — only the feature `φ` and the target *value* change. In
particular **no new `MeanZCTargetLayout` subtype is needed**: `mean_targets`/`pair_targets` are
already written in terms of `ν` alone and are basis-agnostic.

| | current (Fréchet basis) | new (exponential basis) |
|---|---|---|
| feature | `φ = z^k = U^{-μk}` | `φ = U^k` |
| population target `ν_k` | `Γ(1 − μk)` | `Γ(k+1) = k!` |
| finite iff | `k < 1/μ` | always |
| CC regularity `2K` | `K < 1/(2μ)` | always |

## Read these first, in this order

1. `/bbkinghome/edav/gravity_robustness/CLAUDE.md` — project rules. The ones that will bite here:
   **no defaults on any scientific parameter** (a basis choice is unambiguously scientific — it
   changes which restriction is imposed, so it must be an explicit kwarg, and the default must be the
   existing Fréchet basis so no existing caller silently changes meaning); **never add a
   `:dense_reference` fallback**; **check background jobs within 30–60 s** via `ps` + log tail; **push
   a Dropbox package at the end**.
2. Auto-memory, especially `ozc-cross-kpair2-grid-build-2026-08-09` (the most recent work on these
   exact files, including measured cost scaling you will need), plus the three memories named in the
   "framing hazard" section below — **read those three before writing any code**.
3. `cm_meanzc_moments.jl`'s `frechet_power_feature` (the single canonical feature builder) and
   `build_raw_mean_pair_matrix_levels`; `cm_originzc_cross_moments.jl`'s
   `build_raw_cross_pair_matrix_levels`.

## ⚠️ THE FRAMING HAZARD — read this before touching anything

This repo has a **documented history of `U^k` vs `z^k` being a genuine BUG**:

- `feedback-power-weighted-features-need-z-not-u-transform`
- `feedback-zc-families-do-achieve-finite-delta-star-k3` ("root cause was a `U^k` vs `z^k` bug")
- trap 1 of `HANDOVER_CMZC_CROSS_AND_CAMPAIGN_2026-08-09.md` (`nu0` seeded as `mean(U^k)` — "wrong
  twice over: the feature is `z = U^(-μ)` not `U`")

You are now deliberately introducing `U^k` as a **legitimate, opt-in alternative**. That is not a
contradiction of those memories — they are about code that used `U^k` while *intending* `z^k`. But a
future reader (or a future you) skimming the new code will see `U^k` in a restriction feature and
"fix" it back.

**So: every new code path that uses the exponential basis must carry a loud comment saying it is
deliberate, naming the basis kwarg that selected it, and explicitly distinguishing it from the known
bug.** Update those memories at the end to record that `U^k` is now a sanctioned basis *when
explicitly selected*, and still a bug when it appears in a Fréchet-basis path.

## ⚠️ THE IMPLEMENTATION TRAP — `frechet_power_feature` has TWO unrelated callers

Do **not** globally swap it. Verified call sites (2026-08-10):

**(A) ZC/mean restriction features — these are what you make basis-switchable:**
- `cm_meanzc_moments.jl:163` (`build_raw_mean_pair_matrices`, feeds `build_raw_mean_pair_matrix_levels`)
- `cm_meanzc_moments.jl:205` (`nu_feasible_interval`)
- `cm_originzc_config.jl:170` and `:195` (`originzc_default_nu_bounds`, both layout methods)
- indirectly: `cm_meanzc_config.jl`'s `meanzc_default_nu_bounds` (calls `nu_feasible_interval`)
- `cm_originzc_cross_moments.jl`'s `build_raw_cross_pair_matrix_levels` consumes `Zraw_all`, so it
  follows automatically — no edit needed there.

**(B) flexible-CM eq.36 truncated-power `Pow`, ALL at `k = σ−1` — MUST STAY FRÉCHET, do not touch:**
- `cm_hessian_architectures.jl:807`, `common_marginals_moments.jl:268`,
  `cm_meanzc_production.jl:150`, `cm_frechet_level.jl:313`

Group (B) is a different moment family entirely (the CM-grid's truncated-power block). Switching it
would silently corrupt the CM block. Consider renaming or adding an assertion so the two uses cannot
be confused again.

## What to build

1. A named feature-basis surface, e.g. `restriction_feature_basis::Symbol` with
   `:frechet_power` (existing behaviour, the default so nothing changes for existing callers) and
   `:exponential_power` (new). Thread it through the (A) call sites above, the context builders, the
   config structs (`CMMeanZCConfig`, `OriginZCConfig`), and the checkpointed drivers. **Persist it in
   the checkpoint and hard-refuse a resume mismatch** — exactly as `meanzc_target_layout` does
   (`CMCheckpointV11`, added 2026-08-09; you will likely need `V12`). **Give it a distinct
   `family_tag`** for the exact-eval cache: at matched `K` the two bases are different economic
   problems with different Δ*, so a cache-key collision would be a silent wrong answer.
2. The exponential feature builder itself — trivially `U .^ k`, ideally expressed so it shares the
   numerically-stable `exp(k*log(U))` form.
3. The target/seed change: `ν_k = Γ(k+1) = k!` instead of `Γ(1 − μk)`, wherever `ν0` is seeded or a
   `ν` box is derived (`nu_feasible_interval`, `*_default_nu_bounds`, every smoke/seed script).
   Population mean, not a sample average — the standing rule.

**Everything downstream should need no change**: `ZCRestrictionOperator`, `refresh_zc_targets!`,
`restriction_forward!`/`_transpose!`, `zc_restriction_gram!`, the Hessian blocks, the ν-gradient
envelope formulas, and the C+ backends are all written against "a raw feature matrix and a target
vector". Confirm that by reading, then by the gates — do not assume it.

## ⚠️ Variant D does NOT carry over — resolve this, don't port it

`meanzc_profiled_level`/`originzc_profiled_level` (Variant D, focal `k* = σ−1` mean-row omission)
exists because the autarky/counterfactual price-index moment enforces
`E[z_bi^{σ−1}] = cf_denom/cf_num` — an exact affine relation to the **Fréchet-basis** mean row at
level `k* = σ−1`, which makes that row exactly redundant (a genuine KKT rank deficiency).

In the exponential basis the features are `U^k`; there is no such relation, so the redundancy
argument does not apply and `ν_{k*}` has no derived value. **Make Variant D a hard error when the
exponential basis is selected**, unless you can derive and verify an analogous relation — do not
silently reuse `meanzc_profiled_nu_value`, which computes a Fréchet-basis quantity.

## ⚠️ The real new risk is CONDITIONING, not moment existence

`E[U^k] = k!` and the Gram involves `E[U^{2k}] = (2k)!`. These are finite but grow explosively:
`ν_10 = 3.6e6`, `E[U^20] = 2.4e18`. Raw `U^k` columns at large `K` will differ in scale by many
orders of magnitude and `H_ZZ` will be catastrophically ill-conditioned. **"K as high as I like" is
true mathematically; conditioning is what will actually stop you, and measuring it is a core
deliverable of this task, not an afterthought.**

Cheapest principled fix, and the first thing to try: **rescale each restriction row**, e.g. feature
`U^k / k!` with target `ν_k / k!` (≈ 1). A per-row diagonal rescaling leaves the imposed restriction
set mathematically identical (each moment condition is just multiplied by a constant; the dual λ
absorbs it), so it is a pure conditioning improvement with no scientific change. **Verify that
claim numerically**: at a low `K` where both are computable, the rescaled and unrescaled exponential
bases must give the same Δ* to solver tolerance. Decide explicitly whether `ν` denotes the raw or the
rescaled moment and document it — that choice propagates into the `η = log ν` box and the checkpoint.

A more aggressive option if rescaling is not enough: **Laguerre polynomials**, which are orthonormal
under the `Exp(1)` measure, so the Gram would be near-identity. Note `{L_1..L_K}` and `{U^1..U^K}`
span the same space, so for the **mean block** they impose an equivalent restriction set; for the
**diagonal pair block** they do **not** (a Laguerre pair restriction mixes `Cov(U_o^i, U_p^j)` terms
with `i ≠ j` that the diagonal family leaves unrestricted). Treat this as an untested idea to
evaluate only if rescaling proves insufficient, and check the equivalence claim before relying on it.

## How high can K actually go? — cost, from measured scaling

Measured 2026-08-10 (D20/W=100k/L=50, CM+ZC, cold solve): the Hessian callback is dominated by
`H_ZZ`, the restriction gram, which scales as `O(W · nx²)` in the restriction width `nx`. Confirmed
near-exactly: `nx` 630 → 1770 took `H_ZZ` 1.41 → 7.52 s/call = **8.02×**, versus the `nx²` prediction
7.89×. Restriction-independent blocks were flat (~0.95×), and the genuine economic Hessian is
~0.03 s/call — negligible. See `key_results/15_iters_vs_per_iteration_2026-08-10.txt` and
`00_READ_FIRST_CORRECTION.md` in `dropbox:.../cmzc_cross_and_campaign_2026-08-09/`.

**Extrapolating that measured quadratic law** (these are predictions, not measurements — verify):

| family, D=20 | `nx` at K | predicted `H_ZZ` s/call |
|---|---|---|
| diagonal, K=3 | 630 | 1.41 (measured) |
| diagonal, K=5 | 1050 | ~3.9 |
| diagonal, K=8 | 1680 | ~10 |
| diagonal, K=10 | 2100 | ~16 |
| **cross**, K=10 | **19200** | **~1300** |

So: **high `K` is for the DIAGONAL families** (`cm_meanzc`, `origin_zc`). The `K_pair²` CROSS families
become infeasible well before K=10 and should stay at low K. Do not spend effort making cross work at
high K without checking the cost first.

## Suggested order of work

1. **D4 first, always** — seconds per solve, catches essentially every wiring bug. Note D4's `μ = 1/6`
   caps the Fréchet basis at `K ≤ 2` for CC regularity, so D4 is also where the *motivation*
   demonstrates most cleanly: the exponential basis should run at `K = 4, 6, 8` where the Fréchet
   basis cannot.
2. Feature builder + `ν` targets/bounds + basis kwarg through the (A) sites. Smoke at D4: inner solve
   converges and **recovered residuals against the new `E[U^k] = k!` targets are ~0** — that is the
   check that the feature and the target actually correspond.
3. **Control against the existing Fréchet basis before bug-hunting anything** (repo rule, memory
   `feedback-control-against-base-family-before-bug-hunting`): at matched `K` the two bases should
   both converge and give *different* Δ* (they are different restrictions — do not expect equality).
4. FD-check the ν-gradient at D4. The envelope formula is unchanged in form, but the FD is cheap and
   this repo has been bitten before. **Use `h = 1e-6`, not `1e-4`** — the eta_nu objective is steep
   enough that `h = 1e-4` shows a spurious 3–5% relative gap that is pure O(h²) truncation error
   (established 2026-08-10 by an h-sweep against the unmodified base family).
5. Conditioning study: `cond` of the centred restriction data vs `K`, for raw `U^k` and for the
   rescaled variant. **This is the deliverable that answers "how high can K go".** Report Δ*,
   `inner_status`, iteration count, and wall time as a function of K.
6. Real D20 through the actual checkpointed driver on its **default** `cm_gradient_backend=:cplus`,
   plus a resume test, then campaign/seed-generator integration if the results justify it.

## Traps carried over from the immediately preceding work (all confirmed live)

1. **`ν0` = theoretical population mean, never a sample average** of the same draws the restriction is
   imposed on. Here that is `k!`.
2. **`draw_seed` is INERT under `draw_design = :pseudorandom`** (an inner `Random.seed!(888)` overrides
   it). Use `:sobol_randomized`. If a "different seeds" comparison returns identical numbers, that is
   why.
3. **`W = 8000` does not work at D20** — degenerate calibration point. Use `W ≥ 80,000`; iterate at D4.
4. **`cm_gradient_backend = :cplus` is the production driver's DEFAULT.** If you add a family/branch,
   implement its C+ path — do not make it error, or the family is unusable in its own default config.
5. **When A/B-ing two gradient backends, pass BOTH the same `h_mode` AND the same shared
   `bandwidth_cache` Dict** — a mismatch fakes a ~1e-3 gap that looks like a real bug.
6. **Julia world-age**: a runtime `include()` inside a function body followed by calling the
   just-defined method in the same frame throws `MethodError`. Fixed with `Base.invokelatest` in
   `build_originzc_core_hess_ctx` and `build_cm_meanzc_bin_ctx`; the pattern may exist elsewhere.
7. **Run `campaign_cm_family_runner.jl` end to end, not just the driver** — its family arms drift out
   of sync with driver required-kwarg hardening and only fail at launch (memory
   `feedback-run-campaign-runner-end-to-end-not-just-driver`).
8. **KNITRO `-100` is `KN_RC_NEAR_OPT`** — a stall ("no further progress possible"), *not* an
   iteration cap, and it often clears on restart. `-400/-401` are the limit codes.
9. **Do not benchmark with a warm re-solve at the same point.** The base family's warm re-solve does
   *zero* iterations and returns in ~2 s, which makes any "steady-state" ratio meaningless. Compare
   cold solves.
10. **Checkpoint loaders differ by driver**: `run_cm_upper_checkpointed` writes `CMCheckpointV11`
    (load with `load_cm_checkpoint`); `run_originzc_upper_checkpointed` writes `OriginZCCheckpointV10`
    (`load_cm_checkpoint_v10`). Field is `checkpoint_reason`, not `stop_reason`.

## Where to start — branch decision

The most recent work (CM+ZC-CROSS + campaign integration) is **uncommitted working-tree state** on
`feature/ozc-cross-2026-08-09` in `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`, based on
`origin/production/fullA-exact`@`4df5254`.

**Recommendation: a fresh worktree off `origin/production/fullA-exact`.** The basis change is small
and orthogonal to the cross work, high `K` is a *diagonal*-family story anyway (see the cost table),
and building on an unmerged uncommitted branch compounds risk. Take the cross branch instead only if
you specifically want exponential-basis × cross-grid — and if so, read
`docs/CMZC_CROSS_AND_CAMPAIGN_INTEGRATION_2026-08-09.md` there first.

Confirm the choice with the user before building on the uncommitted branch.

## Environment

```bash
export PATH="$HOME/.juliaup/bin:$PATH"     # NOT /opt/shared_sw — that Julia is broken here
export OPENBLAS_NUM_THREADS=1              # hard rule under Julia threading
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}
julia --project=. -t 8 full_aod_diag/d4_exact/<script>.jl
```

Standard D20 production config:
```julia
d20_real_setup_design(W = 100_000, δ = 1.0, find_smallest = true,
    draw_design = :sobol_randomized, draw_seed = 20260719,
    destination_sample = :exclude_row, exclude_diagonal_gravity = true,
    gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea(),
    σHat = 3.0, inner_lower_limit = -10.0)
# L = 50, contrasts = :orthonormal, include_truncated_moment = true,
# A_coordinate_mode = :powered_aspace, cm_gradient_backend = :cplus (driver default)
```

## Deliverable

A `docs/*.md` write-up stating: what the basis option is and how to select it; the gate results;
**the conditioning-vs-K study and a defensible answer to "how high can K actually go"**; the cost at
the K values you tested; and an explicit statement that the two bases are *different* restrictions
with different Δ*, not a reparameterisation. Then update the memories named in the framing-hazard
section and push a Dropbox package to a NEW subfolder under
`dropbox:Gravity robustness/Analysis/Server Output/`.
