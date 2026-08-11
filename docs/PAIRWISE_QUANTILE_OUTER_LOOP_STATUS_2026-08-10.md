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

**Deliberately NOT done, and now confirmed as the user's decision (2026-08-10): leave the protocol
alone — code only, decide later.** `protocols/paper_upper_v1.toml` is frozen by its own header
(*"Once any cell of this protocol has started, THIS FILE MUST NOT BE MUTATED"*), and a live
`paper_upper_v1` campaign was in fact running on this host while this work was done
(`screen -S paperupper_resume`, `repo_scratch/paper_upper_v1/resume_remaining_screen.sh`), which
makes leaving it untouched clearly right rather than merely cautious.

Consequence: the family is fully runnable today via its own driver and qualifies through the seed
generator (`paper_six_family_seed_specs`), but `launch_wave.sh` / `paper_upper_v1.toml` still run
five families. Still pending whenever the protocol question is revisited: the
`[families.PAIRWISE_QUANTILE]` block, `launch_wave.sh`'s hardcoded `FAMILIES=(...)` array, the
`[concurrency]` recompute (5→6 families per start), and the `[projection]` nesting edges.

## Campaign target (user, 2026-08-10): W=100,000, L=10 if feasible, else L=5

All other settings as previous production (σ=3.0, `draw_design=:sobol_randomized`,
`draw_seed=20260719`, `destination_sample=:exclude_row`, `inner_lower_limit=-10.0`, Brazil–Korea
gravity exclusions). `L` is genuinely new to this family and has no production precedent.

**Scale at D=20** (memory is not the constraint — this host has ~3 TB, ~2.3 TB free):

| L | moment rows | outer cutoff coords | dense `HRR` | T4 tables | KNITRO packed H |
|---|---|---|---|---|---|
| 5 | 3,120 | 80 | 0.08 GB | 0.03 GB | 0.04 GB |
| 10 | 15,570 | 180 | 1.94 GB | 0.76 GB | 0.97 GB |

**Wall-clock is the real constraint.** Prior profiling measured the L=5/W=100k inner solve at
407 s, of which ~78% is the T1–T4 Hessian table build. Rows grow 5× from L=5 to L=10 and the dense
Hessian blocks scale ≈ rows², with each T4 combo growing `(L-1)^4` = 256 → 6561 (≈25×). So a naive
projection is **hours per inner solve at L=10**, and an outer bound search needs hundreds of them.
That is consistent with the earlier `pairwise_grid_common_marginal` restriction, which at D=20/L=10
took ~27 min per inner solve at W=80,000.

⚠️ **`pairwise_quantile_hvp.jl` is NOT the escape hatch — it was measured and rejected.** An
earlier revision of this document said it "should be tried before falling back to L=5"; that was
wrong. The 2026-08-09 session already ran the decisive controlled A/B at real D=20/W=100,000
(`profile_pairwise_quantile_d20_hvp_ab.jl`), both variants verifier-confirmed correct with duals
agreeing to ~1e-8:

| variant | n_hess(-vec) calls | wall-clock |
|---|---|---|
| dense (`hessopt=exact`) | 9 | **343.72 s** |
| HVP (`hessopt=5`, CG) | **6084** | **1512.30 s** |

Each HVP callback is far cheaper, but CG needs 676× more of them — **4.4× slower overall**, verdict
"not adopted for production", and explicitly a conditioning property rather than a warm-start
artifact. See `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md` Part 1. Whether the
ratio narrows at L=10 is an unmeasured hypothesis in both directions and must not be acted on
without an L=10 A/B.

### Probe result: **L=10 IS feasible at W=100,000** (measured, not projected)

One value-only inner solve at the real D=20 calibration point, `δ=50` (early-abort threshold `Inf`,
so it cannot be mistaken for a failure), `JULIA_NUM_THREADS=16`. Log:
`logs/pq_W100k_L5_L10_probe.log`; script `full_aod_diag/d4_exact/probe_pairwise_quantile_W100k_L5_L10.jl`.

| W | L | rows | Δ* | class | FG / Hess | inner solve |
|---|---|---|---|---|---|---|
| 100,000 | 5 | 3,120 | 0.000706 | VerifiedSolved | 6 / 5 | **39.9 s** |
| 100,000 | 10 | 15,570 | 0.003011 | VerifiedSolved | 6 / 5 | **1084.6 s (18.1 min)** |

So L=10 converges cleanly and is verified — the earlier worry that it might be unattainable at this
scale is **not** borne out. The cost growth is also milder than the naive `(L-1)^4` projection
suggested: ~217 s per Hessian call at L=10 vs ~45.7 s measured at L=5, i.e. ≈4.7×, not ≈25×.

**But feasible ≠ practical for a full bound search.** At ~18 min per inner solve, the smoke test's
own 26-evaluation outer run would take ~8 hours, and a real bound search needs substantially more
than that. Concretely: L=10 is usable today for single-point evaluation and seed qualification, and
for an outer search the remaining lever is the un-threaded cross-block and packed-write path (see
the HVP note above for why the HVP route is not it). L=5 at 39.9 s/solve is comfortably practical
for an outer search right now.

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
