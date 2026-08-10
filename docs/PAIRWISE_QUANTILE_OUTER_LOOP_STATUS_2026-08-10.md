# Pairwise-quantile-independence: OUTER loop built, validated, and wired as family #6 — 2026-08-10

Continuation of `PAIRWISE_QUANTILE_OUTER_LOOP_INTEGRATION_HANDOVER_2026-08-10.md`. That document
scoped the work; this one records what was built, what was validated, **two real bugs it found**,
and the one decision left open for the user.

Branch: `integration/pairwise-quantile-outer-loop-2026-08-10`, worktree
`/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`. Not pushed to any remote.

---

## Headline

The outer loop exists, is validated against reoptimized finite differences, and runs end-to-end at
real D=20 through a checkpointed production driver. **Two genuine bugs were found and fixed in the
process, both of which the pre-existing test suite passed cleanly** — which is the main reason this
task was worth doing carefully rather than quickly.

| # | Item | Status |
|---|------|--------|
| 0 | Branch reconciliation (restriction ⊕ orchestrator) | **done**, clean merge, zero file overlap |
| 1 | Objective value (`Delta_dual`) returned by the inner solve | **done** (`archPQ_verified_state`) |
| 2 | Cutoff gradient wired to a real converged inner solve | **done** — *found bug #1* |
| 3 | Reoptimized-FD validation gate | **done**, passes; includes a negative control |
| 4 | Combined outer gradient (economic ⊕ cutoffs) | **done** — *found bug #2* |
| 5 | Checkpointed outer driver (23-step mirror) | **done**, smoke-tested at real D=20 |
| 6 | Family-#6 registration | **code done**; protocol TOML deliberately **not** touched |
| 7 | End-to-end smoke test | **done, all checks pass** |

---

## The two bugs

### Bug 1 — the cutoff gradient's `dR` had the wrong sign

`pairwise_quantile_forward!` *subtracts* the restriction contribution into the dual index
(`arg0[w] -= Rw`, matching the economic block's own `arg0 .-= econ_buf`), so
`r = -ζ − E·λ_E − G_R·λ_R` and the change in `r` when a draw changes bin is **minus** the change in
`G_R·λ_R`. `fixed_dual_delta_f` used `dR = +dG`.

**Why the existing suite missed it.** `test_pairwise_quantile_d4_dense_oracle.jl` CHECK 8-9 built
its own reference as `r_current = -pairwise_quantile_forward!(zeros, …)` and evaluated
`psi_scalar.(-a0)` — negating *both* the reference `r` and the brute-force recompute. The two sign
errors cancelled, and the check passed to 1e-16 under a convention no real solve ever uses. This is
exactly the risk the handover doc flagged when it noted the module "has never been called with a
real converged `r_current`."

**How it was found.** Not by inspection — by an isolation experiment
(`debug_pq_cutoff_sign_isolate.jl`, EXPERIMENT 0) that compared the `O(k_crossed)` shortcut against
a *full independent fixed-dual recompute at the real converged dual*, in the production convention.
5/5 probed cutoffs disagreed in sign.

**Fix + gate.** `dR = -dG`, with the derivation and the history recorded at the site. The oracle's
own convention was corrected, so CHECK 8-9 now genuinely gates the sign — verified by negative
control: reverting the fix makes it fail (`err=4.1e-2` vs `tol=1e-10`).

### Bug 2 — the economic gradient's base cache omitted the restriction's `q0` contribution

Every other restricted family folds its own `G_R·λ_R` into the `LFixBaseCache`'s `q0` via `with_q0`
(`build_lfix_base_cache_originzc`). The first version of this family's gradient did not, on the
reasoning that *"the restriction's bin memberships depend only on `U` and the cutoffs, never on
theta, so the restriction contributes nothing to the theta-gradient."*

**That reasoning is wrong**, and the way it is wrong is worth recording because it is seductive: it
is a true statement about the *derivative* and irrelevant to `q0`. `q0` is the per-draw **level**
`−ζ* − Σ_j λ*_j G[s,j]` that the economic block linearizes *around*. Omitting a term from it
linearizes the economic gradient about the wrong base point.

**How it was found.** The FD gate's section 3 flagged it (two of four probed economic coordinates
~70× off). It is now fixed by `build_lfix_base_cache_pairwise_quantile`, which folds via the same
`pairwise_quantile_forward!` operator the inner solve uses and then **cross-checks the result
against the verifier's independently recomputed `r`** — the same quantity by two routes, so the
check is exact rather than statistical:

```
max|q0 (with fold)    − independently recomputed r|  =  6.1e-16     PASS
max|q0 (WITHOUT fold) − independently recomputed r|  =  4.7e-01     (control: must be large)
```

That cross-check then immediately caught a *third* issue as a by-product: `pq_bin_state` is mutable
per-outer-point state, and the outer-gradient path was reading whatever a later solve had left
there. `ensure_pq_bins!` now makes that path self-consistent, with the `q0` check retained behind it
as the backstop.

---

## Validation

### Cutoff gradient vs reoptimized FD (`test_pairwise_quantile_outer_gradient_fd.jl`)

Every FD probe **re-solves the inner dual from scratch** — the same standard this codebase holds
origin-ZC's `nu` gradient to. Probes are taken at **matched bandwidth**, using the same
`cutoff_probe_points` helper the analytic secant uses: `Delta_dual` is a genuine step function of a
cutoff, so an FD at a step crossing zero draws returns exactly 0.0 and would "disagree" with any
correct gradient.

Real D=4 KNITRO, L=5, W=8000, min_crossed=200, at a deliberately **off-optimum** cutoff point:

| metric | value | gate |
|---|---|---|
| sign agreement on signal-carrying coords | **16/16** | pass |
| correlation(analytic, FD) | **0.9834** | > 0.95 |
| relative L2 error | **0.2204** | < 0.35 |
| median ratio analytic/FD | **0.8006** | ∈ [0.75, 1.35] |
| negative control (sign flip removed) | **−0.9834** | must anti-correlate |

**Why an off-optimum point is the one that counts.** At the natural starting point (cutoffs at each
origin's empirical quantiles), `Delta*` turns out to be at a local **minimum in every cutoff
coordinate** — 16/16, confirmed by direct up/down reoptimized probes. That is structural, not
coincidental: the restriction's targets are the fixed constants `1/L` and `1/L²`, and cutoffs at the
empirical quantiles are exactly where the unweighted draws already come closest to meeting them. So
the true gradient is near zero there and any finite probe is curvature-dominated — a reassuring-
looking but weak test. The gate reports that point and *enforces* at the off-optimum one.

**The residual ~20% gap is a bandwidth artifact, not a bias.** Measured, not asserted:

| min_crossed | median analytic/FD | correlation | rel L2 |
|---|---|---|---|
| 25 | 1.028 | 0.886 | 0.260 |
| 50 | 0.939 | 0.938 | 0.195 |
| 100 | 0.852 | 0.974 | 0.153 |
| 200 | 0.801 | 0.983 | 0.220 |

The gap shrinks monotonically as the probe narrows (fixed-dual vs reoptimized secants differ at
second order in the probe width). A systematic scale error would not do that. At real campaign `W`
the same `min_crossed` is a far smaller fraction of the draws, so the effective bandwidth is much
tighter than in this D=4/W=8000 gate.

### Economic block

Not FD-gated, deliberately, and this is a correction to the handover's own suggested plan: the
economic A-block gradient is itself an **adaptive bandwidth-selected secant**, and
`select_bandwidth`'s own docstring states that a probe below its `h_floor=1e-4` "would just
reproduce Method-A's known-wrong winner-boundary-dropping gradient". An earlier draft of this gate
FD'd it at `h=1e-5` and produced a confident-looking failure on exactly the coordinates where the
winner-boundary term dominates. The gate now uses the checks this codebase actually uses for that
block:

- `q0` fold exact against the independent verifier recompute (above) — **6.1e-16**
- `economic_A_gradient!` **bit-identical** to `composite_gradient_at_fast` on the same cache — `0.0`
- matched-bandwidth FD reported as a diagnostic only; on the two coordinates carrying real signal
  the ratios are 0.957 and 1.065.

### End-to-end driver smoke (`smoke_pairwise_quantile_outer_driver.jl`)

Real D=20 data, W=20,000, L=3, real KNITRO outer solve. **All checks passed.**

- `:min_gp`: 26 evaluations, 10 gradients, verified incumbent **gp = 0.97219, Δ = 0.09920**
- the outer solve genuinely **moves the cutoff coordinates** (max move 2.0e-3). This check is
  guarded on `n_eval ≥ 1`: without that guard it *false-passed* at W=8000 where every point was
  rejected and the checkpoint simply held a different terminal iterate.
- checkpoint round-trip, resume (carried `n_eval` 26 → 33), and a hard error on an `L` mismatch
- `:min_delta_fixed_gp` drove **Δ 0.00523 → 0.00189** with `gp` pinned
- omitting `σHat` / `draw_seed` / `L` / `min_crossed` each raises `UndefKeywordError`

### Attainability: do not run this family at small W

Measured at the real D=20 calibration point with `δ=50` (so the early-abort threshold is `Inf` and
cannot be the cause):

| W | L | rows | Δ* | status |
|---|---|---|---|---|
| 8,000 | 3 | 800 | — | **inner solve fails** (unbounded/infeasible) |
| 20,000 | 2 | 210 | 0.004822 | VerifiedSolved (11.1 s) |
| 20,000 | 3 | 800 | 0.005235 | VerifiedSolved (4.4 s) |
| 80,000 | 2 | 210 | 0.000714 | VerifiedSolved (10.9 s) |
| 80,000 | 3 | 800 | 0.000750 | VerifiedSolved (11.1 s) |

Same finite-sample pattern the earlier `pairwise_grid_common_marginal` restriction showed
(memory `pairwise-grid-independence-restriction`), where the W-sweep was also non-monotonic — so
**sweep, don't extrapolate**, before choosing a campaign `(W, L)`.

---

## Family-#6 registration: what is wired, and the one open decision

**Done (code):**
- `FamilySeedSpec` gains `min_crossed` (`0` = not applicable for the other five kinds — the same
  off-sentinel convention already used for their `K_mean`/`K_pair`/`L`, not a silent default)
- `pairwise_quantile_family_spec`, plus `build_family` / `evaluate_family` arms
- `paper_six_family_seed_specs` as a **separate** function rather than an edit of
  `paper_five_family_seed_specs`, so the already-frozen protocol's meaning cannot change silently
- `family_start_chain.jl`'s `call_driver` arm. This family fits the uniform convention with no
  wrapper (unlike `UNRESTRICTED`); `L`/`min_crossed` arrive generically through `fam_kwargs()`.

**Deliberately NOT done — needs the user's call.** `protocols/paper_upper_v1.toml` is frozen by its
own header: *"Once any cell of this protocol has started, THIS FILE MUST NOT BE MUTATED. Any
substantive change requires a new protocol version (`paper_upper_v2.toml`) with its own root output
tree."* Adding a 6th family is a larger change than the in-place `protocol_sha` re-freezes this
branch has done after bug fixes, and the two options have different costs:

- **`paper_upper_v2.toml`** — clean provenance, but a fresh output root: the existing five-family
  results are not reused.
- **In-place re-freeze** — reuses existing results, but a protocol that was declared immutable
  changes meaning after cells have run.

Also still pending, and cheap once that is decided: `launch_wave.sh`'s hardcoded `FAMILIES=(...)`
array, the `[concurrency]` recompute (5→6 families per start), and the `[projection]` nesting edges.

---

## Files

New: `pairwise_quantile_outer_production.jl`, `pairwise_quantile_checkpoint.jl`,
`test_pairwise_quantile_outer_gradient_fd.jl`, `smoke_pairwise_quantile_outer_driver.jl`,
`debug_pq_cutoff_sign_isolate.jl`.
Modified: `pairwise_quantile_cutoff_gradient.jl` (sign fix + `cutoff_probe_points`),
`test_pairwise_quantile_d4_dense_oracle.jl` (convention fix), `multistart_seed_generator.jl`,
`paper_upper_v1_orchestrator/family_start_chain.jl`.

Commits: `407f365` (carried-forward Hessian/L-genericity work), `ce67c25` (outer layer + both bug
fixes), `302efcb` (driver + registration).

## What a next session should not redo

- Do **not** "simplify" `dR = -dG` back to `dR = dG`, and do not re-negate the oracle's
  `r_current`/`psi_scalar` — that pair of changes reintroduces bug #1 and hides it again.
- Do **not** drop the `q0` restriction fold on the argument that the restriction rows are
  theta-independent. That argument is true and irrelevant; see bug #2.
- Do **not** FD the economic A-block at small `h` and treat disagreement as a bug.
- Do **not** judge the cutoff gradient at the empirical-quantile starting point: it is a local
  minimum, so a good-looking agreement there means little.
