# Pairwise-quantile independence: reparameterized to FIXED cutoffs + FREE bin masses — 2026-08-10

Implements `PAIRWISE_QUANTILE_FREE_MASS_REPARAMETERIZATION_HANDOVER_2026-08-10.md` in full.
Branch `feature/pq-free-mass-reparam-2026-08-10`, off
`integration/pairwise-quantile-outer-loop-2026-08-10`, worktree
`/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`. **Not pushed to any remote.**

---

## Headline

The restriction's outer coordinates are now the bin **masses** `mu_{o,a}` on the simplex, with the
quantile cutoffs fixed once per campaign. `Delta*` is therefore smooth in every outer coordinate,
and its gradient is an **exact closed-form envelope derivative** — a mirror of origin-ZC's
production-validated `d_delta_dual_d_eta_origin_vec`, not a new gradient engine.

**The measured agreement against reoptimized finite differences is 4.1e-10 relative L2**, at a
deliberately non-uniform `mu` where the pair term's partner index is observable. Version A's
bandwidth-secant cutoff gradient could only reach ~0.22 at the same D=4 scale. That is the whole
point of the change, and it is the number to quote.

| # | Item | Status |
|---|------|--------|
| 1 | Simplex (stick-breaking) mass transform + analytic Jacobian | **done**, FD-gated |
| 2 | Fixed cutoffs, two explicit sources, recorded in the checkpoint | **done** |
| 3 | Inner solve: forward / transpose / Hessian centering / cross-Hessian | **done**, dense-oracle gated |
| 4 | HVP path | **needed no edit** (composes forward!/transpose!); A/B re-gated |
| 5 | Exact closed-form outer mass gradient | **done**, reoptimized-FD gated at 4.1e-10 |
| 6 | Version-A cutoff-gradient machinery | **deleted**, not kept as a fallback |
| 7 | Checkpointed outer driver, new schema, resume guards | **done**, real-D20 smoke |
| 8 | Family-#6 registration (seed generator, orchestrator) | **done** (protocol TOML still untouched) |
| 9 | Version-A ↔ version-B equivalence anchor | **done** |

---

## 1. The reparameterization

**Version A** — cutoffs `q_{o,r}` free (softplus-ordered), targets pinned at `1/L`, `1/L^2`:

```
g^M_{o,a}(z) = 1{b_o(z)=a} − 1/L                    a = 1..L-1
g^P_{op,ab}(z) = 1{b_o(z)=a, b_p(z)=b} − 1/L²       a,b = 1..L-1
```

**Version B** — cutoffs FIXED for the campaign, masses `mu_{o,a}` free:

```
g^M_{o,a}(z) = 1{b_o(z)=a} − mu_{o,a}
g^P_{op,ab}(z) = 1{b_o(z)=a, b_p(z)=b} − mu_{o,a}·mu_{p,b}
```

Outer restriction coordinate count is unchanged, `(L-1)·D`. Both are pure **dependence**
restrictions: in A the marginal rows merely define `q` as the reweighted distribution's own
quantiles, in B they merely define `mu` as its own bin masses. `mu` is **per origin** and stays that
way — a common `mu` would silently add a marginal restriction on top of the independence one, which
is a different (strictly stronger) object.

### The two modelling choices, both required with no default

**1. Where the cutoffs are fixed** — `cutoff_source`, `:frechet_theoretical` | `:empirical_quantile`.

`ctx.U` holds **Exp(1)** draws in this codebase (`draw_design.jl`: `U = -log(1-U01)`, one shared
transform for every design), and the Fréchet productivity is `z_o = U_o^{-mu_hat}`, a strictly
DECREASING bijection of `U_o`. So:

- `:frechet_theoretical` puts the cutoffs at the **population** `r/L` quantiles, which in the `U`
  coordinate are `-log(1 - r/L)` and are therefore identical across origins. Binning `U` there is
  the same partition of draws as binning `z` at its own Fréchet quantiles `(-log(r/L))^{-mu_hat}`;
  only the bin LABELS reverse, and this restriction is invariant to relabelling bins. Cutoffs common
  across origins is **not** the same thing as masses common across origins.
- `:empirical_quantile` puts them at each origin's OWN empirical `r/L` quantile, using version A's
  own `sorted[clamp(round(Int,(r/L)*W),1,W)]` convention. This is the setting under which version B
  reproduces version A exactly at `mu = 1/L`.

Both are legitimate; neither is a default. The chosen source AND the resulting `(L-1)×D` cutoff
matrix are both recorded in the checkpoint — the rule alone does not pin the numbers
(`:empirical_quantile` depends on the draws) and the numbers alone do not record the intent.

Every marginal bin and every joint cell is asserted non-degenerate at context build
(`assert_pairwise_quantile_bins_nondegenerate`, floor `min_bin_count`, also required with no
default — a joint cell holds only ~W/L² draws in expectation).

**2. How `mu` is parameterized** — **stick-breaking** on unconstrained reals, per origin:

```
v_{o,a} = logistic(s_{o,a})                          (conditional prob. of bin a | bin >= a)
R_{o,0} = 1,  R_{o,a} = R_{o,a-1}·(1 - v_{o,a})
mu_{o,a} = v_{o,a}·R_{o,a-1}       a = 1..L-1        mu_{o,L} = R_{o,L-1}   (the remainder)
```

so `mu > 0` and `sum_{a<L} mu_{o,a} < 1` hold for every real vector, by construction, never by a
KNITRO constraint — the same discipline the softplus-ordered cutoff transform used for ordering.
Chosen over softmax because (a) it is the cumulative/one-increment-at-a-time construction the math
note is *already written in* (`P(z_o<q_r)=p_r`, `F(r,s)=p_r p_s`), so those equivalence proofs carry
over verbatim with `p_r` free; (b) it is the direct structural analog of the ordered-cutoff
transform it replaces; (c) it has **no redundant dimension** — `(L-1)` reals map bijectively onto
the `(L-1)`-dimensional simplex interior, unlike softmax over `L` reals whose Jacobian is rank
deficient by construction.

Analytic Jacobian (lower-triangular, available from `(mu, v)` alone):

```
d mu_a / d s_k  =  0                for k > a
                =  mu_a·(1 - v_a)   for k = a
                =  −mu_a·v_k        for k < a
```

**Bounds** (`default_raw_mass_bounds`) are data-derived on the same discipline
`default_raw_cutoff_bounds` inherited from `originzc_default_nu_bounds`: the box is centred on the
draws' own conditional bin frequency `vhat_{o,a} = n_{o,a}/(W - sum_{j<a} n_{o,j})` and widened by an
explicit **16× log-odds** margin, then clamped so no coordinate can request a conditional
probability the `W` draws cannot resolve at all (`v ∈ [1/W, 1-1/W]`). No hand-tuned level.

---

## 2. Inner solve: the edit really is small

The moment rows differ from version A **only in the constant subtracted per row**, so the indicator
machinery, the bin lookups and the threaded histogram builder are untouched. Concretely:

- `pairwise_quantile_forward!` — one expression: `C_lambda = Σ λ^M_{o,a}·mu_{o,a} +
  Σ λ^P_{op,ab}·mu_{o,a}·mu_{p,b}`, still a per-draw constant, still hoisted out of the `w` loop.
- `pairwise_quantile_transpose!` — the same constant, on the aggregated sums.
- `build_pairwise_quantile_hessian_tables!` / `fill_pairwise_quantile_hessian_raw!` — **unchanged**.
  Expanding `H_RR[I,J] = (1/W) Σ_w h_w (ind_I − c_I)(ind_J − c_J)` shows the T1–T4 tables (~78% of
  inner-solve wall-clock) are target-independent.
- `center_and_scale_pairwise_quantile_hessian!` and `pairwise_quantile_cross_hessian_block!` — take
  `mu` instead of `1/L`, `1/L^2`. Because the target is now row-dependent and appeared inline at
  five separate sites, there is now ONE definition of it (`pairwise_quantile_target_vector!`) that
  both call.
- `pairwise_quantile_hvp.jl` — **no edit at all**: it composes `forward!`/`transpose!`, so it
  inherited the substitution. Re-gated anyway (below).
- `winner_pair_hessian!` (H_EE) — untouched, shared backend.

**Free simplifications that came with fixed cutoffs.** `bin[w,o]` is now campaign-constant, so it
is computed ONCE in the operator's constructor and lives on the immutable `PairwiseQuantileOperator`
rather than on mutable per-outer-point state. `sorted_z`/`sorted_idx` (which existed *only* to
support the cutoff-crossing gradient) are gone with it, as is the whole class of "some later solve
overwrote the bin state" hazard that `ensure_pq_bins!` guarded against. The equivalent hazard now
attaches to the masses and is guarded the same way (`ensure_pq_masses!`).

**This buys exactness, not speed.** Same row count, same T1–T4 cost. The measured inner-solve times
from 2026-08-10 stand unchanged: real D=20, W=100,000, L=5 → 39.9 s/solve, L=10 → 1084.6 s/solve.

> The L=10 cost lever is **not** `pairwise_quantile_hvp.jl` — that was measured at exactly this
> scale on 2026-08-09 and rejected (dense `hessopt=exact`: 9 Hessian calls, 343.72 s; HVP
> `hessopt=5`+CG: 6084 Hessian-vector calls, 1512.30 s — **4.4× slower**, both verifier-confirmed
> correct). See commit `b0c7c7a` and
> `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md` Part 1. The measured pointer is
> instead the *unthreaded* downstream assembly (centering, cross-block, packed write), per
> `PAIRWISE_QUANTILE_HESSIAN_ASSEMBLY_OPPORTUNITY_2026-08-10.md`.

**Version B does make that assembly work easier, and this implementation already meets its two
preconditions**, without having attempted the optimization itself: `bin[w,o]` is now a
campaign-lifetime constant owned by the immutable operator (so the T1–T4 scatter pattern is fixed
forever and only the weights `h_w` move), and the centering vector `t` — the only thing the free
masses touch — now comes from ONE shared builder, `pairwise_quantile_target_vector!`, rather than
being written inline at five sites. Threading `center_and_scale_pairwise_quantile_hessian!`'s
`nrow²` read-modify-write loop, or making it BLAS-2, is a change to that single function.

---

## 3. The outer gradient

`mu` enters `r` only through `C_lambda`, so with `A_{o,a} ≡ dC_lambda/dmu_{o,a}` (draw-independent):

```
∂C_λ/∂mu_{o,a} = λ^M_{o,a} + Σ_{p≠o} Σ_b λ^P_{op,ab}·mu_{p,b}   =  A_{o,a}
∂r_w/∂mu_{o,a} = +A_{o,a}                       (r = … − (G_R·λ_R)_w, and forward! SUBTRACTS)
∂f/∂mu_{o,a}   = (1/W) Σ_w Ψ'(r_w)·A_{o,a} = A_{o,a}·mean_m
```

and by the envelope theorem at the converged `(ζ*, λ*)`, with `Delta* = −f*`:

```
   d(Delta_dual)/d(mu_{o,a})  =  −mean_m·( λ^M_{o,a} + Σ_{p≠o} Σ_b λ^P_{op,ab}·mu_{p,b} )
```

which is origin-ZC's own formula with the pair index changed:

```
   d(Delta_dual)/d(nu_{o,k})  =  −mean_m·( λ_mean,o,k + Σ_{p≠o} nu_{p,k}·λ_pair,op,k )
```

`d_delta_dual_d_mu` mirrors that function line for line. **The one real difference is the pair
index**: origin-ZC's pair dual is indexed by the power `k`, so its product term pairs `nu_{p,k}`
with `lambda_pair,op,k` at the SAME `k`; here the pair dual is indexed by the bin pair `(a,b)`, so
the mass multiplying `λ^P_{op,ab}` in `dC/dmu_{o,a}` is `mu_{p,b}` — the **partner's** mass at the
**partner's** bin index, summed over `b`. Reading it as `mu_{p,a}` is the natural transcription
error and is **invisible at `mu = 1/L`**, which is why both gates below are enforced at a
non-uniform `mu` and carry an explicit negative control for exactly that mistake.

There is **no sign flip at the production layer** (version A's `pairwise_quantile_cutoff_gradient_vec`
had one): `d_delta_dual_d_mu` differentiates `Delta_dual` directly — the `−mean_m` prefactor IS
the `Delta = −f` sign.

The `q0` restriction fold (`build_lfix_base_cache_pairwise_quantile`) and its exact cross-check are
**kept**, unchanged in reasoning: `q0` is the per-draw LEVEL the economic block linearizes around,
not a derivative, so "the restriction rows don't depend on theta" remains true and remains
irrelevant.

---

## 4. Validation — measured, not asserted

### Gate 1+2 — standalone D=4 dense oracle (`test_pairwise_quantile_d4_dense_oracle.jl`)

Synthetic draws, no solver. Run at **L=5 and L=7** (L-genericity), all checks pass.

| check | measured | tol |
|---|---|---|
| stick-breaking Jacobian vs FD of the decode | 1.7e-11 | 1e-8 |
| mass conservation `Σ_a dmu_a/ds_k = −dmu_L/ds_k` | 1.0e-11 | 1e-8 |
| `uniform_mass_raw` decodes to `mu == 1/L` | 2.8e-17 | 1e-14 |
| `empirical_mass_raw` decodes to the draws' bin frequencies | 0.0 | 1e-13 |
| shared target vector vs the dense reference's own column constants | 0.0 | 1e-15 |
| `forward!` vs dense `G·λ` (mu-centered) | 3.6e-15 | 1e-9 |
| `transpose!` vs dense `−(1/W)Gᵀw` (mu-centered) | 8.3e-17 | 1e-9 |
| centered Hessian block vs INDEPENDENT centered-dense reference | 6.9e-17 | 1e-8 |
| closed-form `d(Delta)/dmu` vs fixed-dual FD (relative) | **2.1e-9** | 1e-6 |
| chain rule to raw coords vs fixed-dual FD (relative L2) | **3.9e-10** | 1e-6 |
| **negative control**: `mu[p,a]`-instead-of-`mu[p,b]` pair index | **0.55** (must be > 1e-3) | fires |

The dense reference builds the centered feature matrix directly and forms `G'WG`, while the
production path centers the RAW indicator via the `X'WX − rt' − tr' + Stt'` identity — two
independent routes, so agreement is a genuine cross-check of the centering algebra.

### Gate 3 — version-A ↔ version-B equivalence anchor (`test_pairwise_quantile_version_ab_anchor.jl`)

Real D=20, `:empirical_quantile` cutoffs, `mu` pinned at `1/L`: the moment matrix is then identical
to version A's at its own start point. Checked structurally (every marginal-row constant exactly
`1/L`, every pair-row constant exactly `1/L²`) *and* numerically against version A itself, re-run
live from a detached worktree at the pre-reparameterization commit `a387a07`
(`/bbkinghome/edav/cdw_worktrees/pq-versionA-ref-2026-08-10`,
`full_aod_diag/d4_exact/ref_versionA_anchor_point.jl`).

Real D=20, W=20,000, L=3, `sigma=3.0`, `sobol_randomized`, `draw_seed=20260719`, `:exclude_row`,
`inner_lower_limit=-10`, Brazil–Korea gravity exclusions, `delta=50` (so the early-abort threshold
is `Inf` and cannot be mistaken for a failure). Both versions `VerifiedSolved`, `nStatus=0`.

| quantity | value |
|---|---|
| version A (live re-run at `a387a07`) | `Delta_dual = 0.0052348411957184159` (11.6 s) |
| version B, `:empirical_quantile` cutoffs, `mu = 1/L` | `Delta_dual = 0.0052336470814719595` (11.8 s) |
| relative difference | **2.281e-04** |
| recorded 4-sig-fig reference in the previous status doc | 0.005235 — reproduced |

Structural checks, all exact: the fixed cutoffs equal version A's own empirical-quantile
construction (`max|diff| = 0.0`); `uniform_mass_raw` decodes to `mu = 1/L` exactly
(`max|mu − 1/L| = 0.0`); every marginal-row centering constant is exactly `1/L` and every pair-row
constant exactly `1/L²` (`max|t − target| = 0.0`).

**The 2.281e-04 is fully attributed, not hand-waved.** Version A never used the empirical quantiles
directly: it *encoded* them into raw coordinates (`log q_1`, `log(expm1(gap))`) and *decoded* them
back (`exp`, `softplus`) on every outer point, and that round trip is 1-ulp-lossy
(`max|Q_roundtrip − Q| = 2.2e-16`). The empirical quantile **is** one of the draws, so a draw
sitting exactly on a cutoff can change bin under a cutoff one ulp below it. Measured directly in the
anchor: the round trip moves **2 of 400,000** bin assignments (0.0005%), **both** of them draws
sitting exactly on a cutoff, at most one per (origin, cutoff).

**And the loop closes.** Re-solving version B on version A's own round-tripped cutoffs — so the two
partition the draws identically and the moment matrices are element-for-element equal — gives

```
   version B at version A's cutoffs :  0.005234841195718461
   version A                        :  0.0052348411957184159
   relative difference              :  8.616e-15      (gate: < 1e-12, solver reproducibility)
```

i.e. the two versions are the same object to round-off once the partition is the same, and the
2.281e-04 above is exactly the 2-draw partition difference. That is a stronger statement than the
handover asked for (it asked for the 4-significant-digit reproduction), and it rules out any
centering or indexing error outright.

### Gate 4 — reoptimized-FD gate on the combined gradient (`test_pairwise_quantile_outer_gradient_fd.jl`)

Real D=4 KNITRO, W=8000, L=5, `:empirical_quantile`. Every FD probe re-solves the inner dual from
scratch. **Plain small `h` — no bandwidth, no matched-secant machinery**, because `Delta_dual` is
now smooth in these coordinates.

At the enforced (non-uniform `mu`) point, `Delta_dual = 0.10167786557963647`, `VerifiedSolved`:

| h | relative L2 | max abs diff | cosine |
|---|---|---|---|
| 1e-3 | 1.153e-06 | 3.068e-07 | 1.0000000000 |
| 1e-4 | 1.154e-08 | 3.068e-09 | 1.0000000000 |
| 1e-5 | **4.132e-10** | 7.167e-11 | 0.9999999999999999 |

Clean `h²` convergence, exactly as a smooth objective should give. At the uniform-`mu` start point
(reported, not enforced — see the pair-index note above) the relative L2 is 1.02e-07 at h=1e-4.

Negative controls, both fire:

| control | result |
|---|---|
| sign-flipped gradient must anti-correlate with FD | cosine = **−0.9999999999999999** |
| pair product term dropped must fail the same gate | relative L2 = **0.376** (gate is 1e-6) |

Economic block, gated the way this codebase actually gates it (NOT by a small-`h` FD — the A-block
gradient is itself an adaptive bandwidth-selected secant whose own docstring says a sub-`h_floor`
probe reproduces a known-wrong gradient):

| check | measured |
|---|---|
| `q0` restriction fold vs the independent verifier recompute | **7.8e-16** |
| CONTROL: the same, WITHOUT the fold (must be large) | **0.477** |
| `economic_A_gradient!` vs `composite_gradient_at_fast` | **0.0** (bit-identical) |
| combined `vcat(g_econ, g_mass)` layout + tail identity | pass |

**The gate threshold is 1e-6 relative L2.** The handover set the expectation at ~1e-5; the code
achieves 4.1e-10. The threshold is therefore stricter than the stated expectation and still ~2400×
above what is actually achieved — it is not a number tuned until green. Version A could only reach
~0.22 here, and "a few percent" would mean something is wrong.

### Gate 5 — real D=20 end-to-end driver smoke (`smoke_pairwise_quantile_outer_driver.jl`)

Real D=20 data, W=20,000, L=3, `cutoff_source=:empirical_quantile`, `min_bin_count=1111`
(= W/2L², data-derived), real KNITRO outer solve, 240 s budget per stage. **All checks passed, no
failures.**

Occupancy at that setting, reported by the context builder: min marginal bin 6,666, min joint cell
2,203 — comfortably above the declared floor.

- `:min_gp` — 35 evaluations, 12 gradients, verified incumbent **gp = 0.97188**, and the outer solve
  genuinely **moves the mass coordinates** (max move 4.79e-3). That check is guarded on
  `n_eval ≥ 1 && n_grad ≥ 1`: without the guard it false-passed in the past at W=8000 where every
  point was rejected and the checkpoint merely held a different terminal iterate.
- checkpoint round-trip through its own loader; records `L`, `cutoff_source`, `min_bin_count`,
  `n_raw`, `D`, sigma and draw provenance, the `raw_masses`, and the fixed cutoff matrix itself —
  the last verified **bit-identical to regenerating it from `cutoff_source`**.
- resume carried `n_eval` 35 → 56; resuming under a different `L` is a hard error; resuming under a
  different `cutoff_source` is a hard error.
- `:min_delta_fixed_gp` drove **Δ 0.005234 → 0.001582** with `gp` pinned exactly at `gp_fixed`.
- the no-defaults rule is enforced: omitting `σHat`, `draw_seed`, `L`, `cutoff_source` or
  `min_bin_count` each raises `UndefKeywordError`.

Sanity cross-check worth noting: the driver's very first evaluation (calibration point, `mu = 1/L`)
reports `Delta = 0.005233647081471936` — the same value the equivalence anchor solves for
independently above, to every printed digit.

### Re-gated, unchanged code

- `test_pairwise_quantile_real_d4_knitro.jl` — real KNITRO inner solve, `nStatus=0`. PASS.
- `test_pairwise_quantile_real_d4_verifier.jl` — independent verifier, `kkt_resid` small; LFD
  marginal probabilities exactly `1/L` per bin and cumulative residuals ~1e-16 at the optimum. PASS.
- `test_pairwise_quantile_d4_hvp_ab.jl` — dense-Hessian vs HVP inner solve: `|ζ_A − ζ_B| = 1.4e-15`,
  `max|λ_A − λ_B| = 6.4e-9`. PASS. (The HVP file needed no edit; this confirms it.)

---

## 5. What was deleted, and why not kept as a fallback

`pairwise_quantile_cutoff_transform.jl` and `pairwise_quantile_cutoff_gradient.jl` in full
(`softplus`/`dsoftplus`, `PairwiseQuantileCutoffLayout`, `decode_all_cutoffs!`,
`cutoff_jacobian_block!`, `default_raw_cutoff_bounds`, `bandwidth_target`, `crossed_draw_range`,
`fixed_dual_delta_f`, `cutoff_probe_points`, `cutoff_secant_gradient!`), plus `matched_raw_steps`
and `d_delta_dual_d_cutoff_fd`, the matched-bandwidth logic in the FD gate,
`pairwise_quantile_start_cutoffs`, `refresh_pairwise_quantile_bins!`, `ensure_pq_bins!`, and
`debug_pq_cutoff_sign_isolate.jl`.

All of it existed solely to cope with a step-function objective that version B does not have.
Keeping it "just in case" would keep exactly the tuning burden the reparameterization removes.

**Two things carried forward deliberately, and they must not be "restored":**

1. The `dR = −dG` sign convention's *reason* — `forward!` SUBTRACTS into its accumulator, so
   `r = −ζ − E·λ_E − G_R·λ_R`. Every version-B derivation (the Hessian centering, the HVP, the
   `+A_{o,a}` in the mass gradient) is stated against that convention.
2. The D4 oracle's corrected `r_current`/`psi_scalar` convention (`r_current = r0`,
   `psi_scalar.(a0)`, **no negation**). Version A briefly negated both sides; the two errors
   cancelled, the check passed to 1e-16 under a convention no real solve uses, and a genuine sign
   bug survived it. See memory `feedback-self-cancelling-test-convention-cannot-gate-a-sign`.

---

## 6. Production layer

- **Checkpoint schema**: `PairwiseQuantileMassCheckpointV1` (fresh name, verified unique by grep
  across the tree — `Serialization.deserialize` resolves types by NAME and this repo has been bitten
  once already). Records `L`, `cutoff_source`, the `(L-1)×D` **cutoff matrix itself**,
  `min_bin_count`, `n_raw`, `raw_masses`. A version-A `PairwiseQuantileCheckpointV1` is rejected
  outright rather than reinterpreted — its restriction coordinates were cutoffs, not masses.
- **Resume guards**: hard error on `L`, `cutoff_source`, and a **bit-identity check of the
  regenerated cutoff matrix** against the checkpoint's, on the same footing as the `L` guard
  (different cutoffs = a different restriction). `min_bin_count` differences are logged, not
  refused — it is a validation floor, not part of the problem's identity.
- **Bounds**: `default_raw_mass_bounds`, built after the context decodes the bins (the box is
  derived from their occupancy), replacing `default_raw_cutoff_bounds`.
- **`cb_G!`**: closed form; no bandwidth cache for the restriction block (the economic block still
  has its own).
- **Family #6 registration**: `FamilySeedSpec` swaps `min_crossed` for
  `(cutoff_source, min_bin_count, mass_start)` — `:none`/`0` off-sentinels on the other five kinds,
  the same convention already used for their `K_mean`/`K_pair`/`L`/`contrasts`.
  `pairwise_quantile_family_spec` and `paper_six_family_seed_specs` take them as required kwargs.
  `nu_policy` now records `:pq_mass_start_uniform` / `:pq_mass_start_empirical` rather than a
  cutoff-flavoured label.
- **Orchestrator bug found and fixed while wiring this**: `family_start_chain.jl`'s `fam_kwargs()`
  auto-injects `probs = resolve_cm_probs(L)` for any family whose manifest specifies `L`. This
  family's `L` is its number of quantile BINS and `run_pairwise_quantile_upper_checkpointed` has no
  `probs` kwarg at all, so every call through the orchestrator would have failed with an
  unsupported-keyword `MethodError`. The guard is now on the DRIVER, not on the presence of `L` —
  `L` is precisely what the two families have in common. (Latent in version A too; it could not
  fire because the protocol never registered the family.)
- **`protocols/paper_upper_v1.toml` NOT touched.** It is frozen by its own header and a live
  `paper_upper_v1` campaign was running on this host throughout this work
  (`screen -S paperupper_resume`). That decision stands until the user revisits it.

---

## 7. Verifier change worth knowing about

Version A built its probability report from the **unweighted** draws and compared it to the
constants `r/L`. Under version B the targets are the free masses, so an unweighted report would show
a large "residual" at any point where `mu` differs from the draws' own bin frequencies — i.e. it
would flag the search doing exactly what it is supposed to do.

The report is now computed **under the LFD** (`m_w = Ψ'(r_w)`), which is both the quantity the
restriction actually constrains and the human-readable form of the KKT residuals: `g_M[o,a] = 0` is
exactly `Mraw[o,a]/S_m = mu[o,a]`. Cumulative factorization residuals are taken against
`Pcum[o,r] = Σ_{a≤r} mu_{o,a}` (the math note's own cumulative form, with `p_r` now free).

This costs nothing: `pairwise_quantile_transpose!` already builds the m-weighted full-`L` tables on
its way to `g_M`/`g_P`, so the report reads them instead of making a second `O(W·(D+npair))` pass —
one pass fewer than version A.

---

## 8. What a next session should NOT redo

- Do **not** reintroduce a bandwidth/secant for the mass gradient. It is exact; if an FD disagrees,
  move `h`, don't widen the tolerance.
- Do **not** negate `pairwise_quantile_mass_gradient_vec`'s output. `d_delta_dual_d_mu` already
  differentiates `Delta_dual`. The FD gate's sign negative control exists to catch this.
- Do **not** read the pair term as `mu[p,a]`. It is `mu[p,b]` — the partner's mass at the partner's
  bin index. Both gates carry a negative control for it, and both are enforced at non-uniform `mu`
  because the error is invisible at `mu = 1/L`.
- Do **not** drop the `q0` restriction fold on the argument that the restriction rows are
  theta-independent. That argument is true and irrelevant; `q0` is a level.
- Do **not** make `mu` common across origins — that adds a marginal restriction.
- Do **not** re-negate the D4 oracle's `r_current`/`psi_scalar`.
- Do **not** FD the economic A-block at small `h` and treat disagreement as a bug.
- Do **not** expect a speedup, and do **not** reach for `pairwise_quantile_hvp.jl` as the L=10
  performance fix — it was measured at real D=20/W=100k and rejected (4.4× slower; CG needs 6084
  calls vs dense's 9). Read `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md` Part 1
  first. (The HVP path is still *correct* and is still gated here — it is not the speed lever.)

## 9. Follow-up round (same session, after the first commit)

Three further user directives, all implemented and gated.

### 9.1 The restriction is now stated on the FRÉCHET productivity

Previously the code binned `ctx.U` (the Exp(1) draws) at the complementary Exp(1) quantiles and
argued that this is the *same partition* as binning the Fréchet `z` — true, but with the bin labels
reversed, which made the cumulative report and the bin ordering read backwards relative to the math
note and the economics. It now bins `z` directly:

```
z_o(w) = U_o(w)^(-muHat)         pairwise_quantile_frechet_features  (wraps the shared
                                 frechet_power_feature -- no second copy of the formula)
:frechet_theoretical  q_r = (-log(r/L))^(-muHat)        exact population quantile, closed form
:empirical_quantile   q_r = sorted(z_o)[round(rW/L)]    each origin's own
```

Bin 1 is now the LOWEST-productivity bin. `muHat` is `ctx.μHat`, required with no default, and is
recorded on the context. The D=4 oracle checks the closed form directly: `P(z <= q_r) == r/L` to
**1.1e-16**.

The equivalence anchor got *stronger* as a result. Version B's Fréchet-z partition is version A's
U-partition relabelled with **0 of 400,000 assignments differing** (it was 2 under U-binning). The
remaining `Delta*` residual — 0.005233647081 vs version A's 0.005234841196, rel **2.281e-04** — is
therefore not a partition difference in the construction, and it is not a centering error either.
It is version A's own 1-ulp-lossy `log`/`softplus` cutoff round trip, which moves **2 of 400,000**
assignments away from the exact empirical quantiles, both of them draws sitting exactly ON a cutoff.
Two draws are worth `2 * Delta*/W ≈ 5e-7` of `Delta*`, against the 1.2e-6 observed — the right order.
Version A is the one that drifted, not version B, and the anchor now measures and prints this rather
than asserting it.

### 9.2 The L=10 cost — and a CORRECTION to what this document first said

**⚠️ The first version of this section, and the commit message and status update that went with it,
claimed the final packed write was 86.16 s and 46% of the L=10 solve. That was wrong by ~45×.** It
is recorded here rather than quietly edited out, because the way it was wrong is the useful part.

`PAIRWISE_QUANTILE_HESSIAN_ASSEMBLY_OPPORTUNITY_2026-08-10.md` §2 nominated the centering pass as
"the single most promising item" and §4 said to measure first. The profile
(`logs/pq_L10_blockprofile.log`) appeared to say: centering 0.37 s, packed write **86.16 s**, sum of
blocks 96.72 s, i.e. assembly = 51% of the 1323 s solve and the packed write alone = 46%.

The centering number was right (and threading + the one-triangle change took it to **0.034 s**). The
packed-write number was an **artifact of the profiler**. `profile_pairwise_quantile_d20.jl` did not
call the production callback; it hand-inlined a copy of the loop at TOP-LEVEL scope, where
`HRR`/`HEQ`/`hee_packed`/`n` are non-const globals, so every one of ~127M element accesses was a
dynamic dispatch. Measured side by side at the identical L=10 dimensions:

| the same loop | wall |
|---|---|
| inside a function, serial, row-walk — **what production actually ran** | **1.90 s** |
| inside a function, threaded, column-walk — production today | **0.40 s** |
| at top-level scope reading non-const globals — **what the profiler measured** | **84.70 s** |

The stride fix and the threading in §9.2's earlier draft are real but small (1.90 → 0.40 s). The
`evalResult.hess` hoist committed alongside them is worth essentially nothing: Julia specializes the
callback on the concrete `EvalResult`, so that field access was already concrete — the untyped-`Any`
diagnosis had the right *mechanism* but the wrong *location*. It is kept only because the type
assertion it added is a cheap tripwire.

**Corrected accounting at D=20/L=10/W=100k**, per Hessian callback:

| block | wall |
|---|---|
| T1–T4 raw table build | 3.48 s |
| H_E,R cross-block | 3.79 s |
| H_MM/MP/PP raw block-fill | 1.73 s |
| final packed write | ~0.40 s |
| centering correction | 0.03 s |
| H_EE | 0.08 s |
| **total assembly** | **≈ 9.5 s** |

With 7 callbacks in a ~1200 s solve that is **≈ 6% of wall-clock, not 51%**. So §4's own decision
rule — "if the block sum is a small fraction, KNITRO's own work dominates and the honest answer is
that L=10 is expensive for a structural reason" — resolves the other way: **~94% of the L=10 solve
is KNITRO**, factorizing a dense ~15,952² KKT system per interior-point iteration.

`profile_pairwise_quantile_d20.jl` now times the **real** callback via
`pairwisequantile_hess_cb_builder` and prints the sum-of-blocks against it, so this class of error
cannot recur silently.

### 9.3 Only one triangle is computed now

KNITRO receives a **packed upper triangle** (`KN_DENSE_ROWMAJOR`), so each unordered pair is read
exactly once — but `fill_pairwise_quantile_hessian_raw!` was mirroring every value into both halves
and the centering pass was correcting all `nrow²` entries, for a half nothing ever read. Both now
touch the lower triangle only (`_lo_write!` stores at `[max(i,j), min(i,j)]`), which is precisely
the half the packed write walks by column; the centering pass is threaded over columns too. The
defensive `0.5*(H[i,j]+H[j,i])` in `pack_upper_pairwise_quantile_hessian!` is gone — it would now
average a real value against a structural zero — and the symmetry it defended is enforced upstream
by storing each pair exactly once.

Gated in the D=4 oracle: centered lower triangle vs the independent dense reference **5.6e-17**; the
upper triangle asserted untouched; the dense reference asserted symmetric; and the **packed vector**
— what KNITRO actually receives — checked against the full dense reference at **5.6e-17**. At real
D=20 the change moves `Delta*` by ~1e-16 relative, i.e. floating-point reassociation only.

### 9.4 Measured effect, and what it does and does not buy

Real D=20, W=100,000, `:frechet_theoretical`, `mu = 1/L`, one verified inner solve at the
calibration point (`n_fg=6`, `n_hess=5` in both cases):

| L | `Delta*` | before | after |
|---|---|---|---|
| 5 | 0.00070555 | — | **38.1 s** |
| 10 | 0.00300686 | 1084.6 s | **922.5 s** |

So L=10 is ~15% faster and L=5 is unchanged in the ~40 s band. **This does not make L=10 viable for
an outer search**: at ~15 min per inner solve a 75-minute stage buys about 5 evaluations. The
profile's own arithmetic says why — after the assembly fix the residual ~646 s is KNITRO factorizing
a dense ~15,970² KKT system per interior-point iteration, which no amount of faster filling touches.
That is the structural answer §4 of the note asked for.

## 10. Production run — LAUNCHED

`run_pairwise_quantile_production.jl` + `launch_pairwise_quantile_production.sh`.

Running now: **L=5, `cutoff_source=:frechet_theoretical`, W=100,000**, under
`screen -S pq_prod_L5`, campaign root
`/bbkinghome/edav/repo_scratch/pq_freemass_production_2026-08-10/L5_frechet_theoretical`.

It walks the frozen protocol's own delta grid `{0.1, 0.5, 1.0, 2.0}` with the protocol's own
per-delta stage structure and budgets — **read from `paper_upper_v1.toml` and restated explicitly in
the runner, with that file left untouched**:

| stage | objective | algorithm | budget |
|---|---|---|---|
| A | `:min_gp` | primary, `pin_outer_algorithm` (CG + L-BFGS) | 75 min |
| B | `:min_delta_fixed_gp` at A's incumbent gp | default | 30 min |
| C | `:min_gp`, resuming from B (or A) | alternate, `outer_direct_hessopt=:sr1` | 75 min |

All other settings are the protocol's: σ=3.0, `:sobol_randomized`, seed 20260719, `:exclude_row`,
`exclude_diagonal_gravity`, Brazil–Korea exclusions, `inner_lower_limit=-10`,
`A_coordinate_mode=:powered_aspace`, `z_halfwidth=30`, `find_smallest=true`. This family's own three
choices: `L=5`, `cutoff_source=:frechet_theoretical`, `min_bin_count = W/(2L²) = 2000` (half the
expected joint-cell occupancy, derived from this run's own W and L), `mass_start=:uniform`.

Operationally: the launcher **freezes a `git archive` snapshot** of the exact commit into
`_source` and **refuses to start from a dirty tree** (a dirty tree would silently run committed-only
code); every stage skips itself if its checkpoint exists, so the bounded retry loop resumes rather
than restarts; KNITRO pinned at 13.0.1, which is what every gate here was validated under (the live
`paper_upper_v1` campaign runs 14.2.0 — do not merge results across the two without re-gating).

Measured occupancy at launch: min marginal bin 19,996, min joint cell 3,966, against the declared
floor of 2,000.

**L=10 is deliberately not launched as a grid.** At 922 s/solve it cannot make meaningful outer
progress in a protocol-sized stage; the same runner will do it (`launch_... 10 frechet_theoretical`)
if a much longer budget is ever allocated, and single-point evaluation at L=10 works today.

## 11. Still open

- `protocols/paper_upper_v1.toml`: the `[families.PAIRWISE_QUANTILE]` block, `launch_wave.sh`'s
  hardcoded `FAMILIES=(...)`, the `[concurrency]` recompute (5→6 families per start), and the
  `[projection]` nesting edges — all deliberately untouched, pending the user's decision.
- L=10 outer-search cost. The route indicated by the measured evidence is the unthreaded downstream
  Hessian assembly (`PAIRWISE_QUANTILE_HESSIAN_ASSEMBLY_OPPORTUNITY_2026-08-10.md`), not HVP.
  Nothing in that direction was attempted here; this session's contribution to it is that version B
  makes the scatter pattern and the centering site fixed and single (§2).
- The campaign's `cutoff_source` is a scientific choice the user has not yet made. The user's
  suggestion in the handover was Fréchet values; `:frechet_theoretical` is implemented and is the
  cleaner object (population quantiles, no Monte Carlo error in where the bins sit), while
  `:empirical_quantile` is what makes version B and version A the same restriction. Both are
  available, neither is a default, and the choice is recorded in every checkpoint.
