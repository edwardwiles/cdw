# Corrected full-equilibrium audit: endogenous autarky entrant mass, shared entry cost, and non-ACR GT (2026-07-31)

**This document corrects and supersedes the interpretation** (not the raw numbers, which are
independently re-verified below) of `docs/melitz_final_incumbent_full_equilibrium_audit_2026-07-31.md`.
That prior audit incorrectly (a) treated `N'_j = N_j` as a required restriction that this
closure does not impose, and (b) treated disagreement with the ACR/Chaney sufficient statistic
as a material equilibrium failure. Both are wrong. This session corrects both, re-derives the
proper closure algebraically, and re-verifies every equation under the corrected reading.

This is an **audit only**: nothing was optimized, repaired, or searched from the incumbents; the
Ricardian implementation was not touched.

## States audited

| | GT | Δ* | checkpoint |
|---|---:|---:|---|
| **Upper** | 8.84934137951131% | 0.49812250980571005 | `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/phase6_upper_m1.jls` |
| **Lower** | 0.2722655222019532% | 0.11376383122706066 | `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/phase6_lower_m-1.jls` |

Repo `trade_robustness_modular`, branch `melitz/fullD-delta-star`, D=20, real data
(`real_data/noah_D20`), focal country France, σ=2.5, θ*=8.75177334, W=80,000, seed=1. Both
checkpoints loaded fresh, fingerprints re-verified, fresh cold-solve `Δ` drift = 0.0 exactly at
both. Provenance/fingerprints: `docs/key_results/melitz_corrected_endogenous_entry_full_equilibrium_audit_2026-07-31/`
(reuses the SHA-256 checksums already recorded in the prior audit's `provenance_2026-07-31.txt`,
same checkpoints, same data files).

**All calculations independent** (`scripts/melitz_corrected_endogenous_entry_audit_2026-07-31.jl`):
own Kahan-compensated Float64 loops over the real reference draws + recovered LFD weights, plus a
BigFloat cross-check for the autarky zero-profit residual — never calling `melitz_moments!`,
`mul_G!`, `mul_Gt!`, or a cached production moment residual.

## Headline result

**Conclusion A — the full maintained equilibrium holds**, under the corrected closure. Every
equation the maintained model actually requires — including the endogenous autarky entrant mass
`N'_j` (correctly NOT forced equal to `N_j`) and the shared entry-cost parameter `f_Ej` — is
satisfied independently at both incumbents, to machine precision or to the ~1e-8-level tolerance
inherited from the underlying calibration data. The ACR/Chaney sufficient statistic is reported
descriptively only, as instructed, and its disagreement with the reported GT away from the exact
Pareto calibration point is **not** an equilibrium failure — it is expected once `γ'_target` is
searched away from its calibration value, since ACR is a Pareto-benchmark implication, not a
general equilibrium restriction of this closure.

---

## Corrected closure (what changed vs. the prior audit)

The prior audit's Phase 7 "N′ cross-check" computed two formulas for the autarky entrant mass
and found them to agree (`N′_market_clearing == N′_price_index`, relative difference ~1e-14), then
dismissed this as "near-tautological" and looked instead for agreement with `N_j` (the factual
mass) and with the ACR formula — neither of which the maintained model requires. The correct
reading, confirmed here:

- **`N_o` (factual, all origins) is normalized/calibrated as currently implemented** — the active
  closure sets `N_o ≡ 1` for every origin (a units normalization, not a restriction with economic
  content); this was already correctly identified in the prior audit.
- **`N'_j` (focal autarky) is endogenous.** It is *determined by* the autarky equilibrium system
  (specifically, the price-index equation), not fixed to any value, and in particular is **not**
  required to equal `N_j`. Both incumbents confirm `N'_j ≠ N_j` (`N'_j = 0.7864` upper, `1.1656`
  lower, vs. the universal `N_j ≡ 1`) — this is the correct, expected behavior of the closure, not
  a defect.
- **The same `f_Ej` must rationalize both the factual and autarky free-entry conditions.** This
  is exactly what the current implementation's single focal moment enforces (proof below) — it is
  not merely "close to zero," it is the *same equation*, written once, as profiling out a shared
  `f_Ej`.
- **GT is computed from the factual and autarky price indices**, `GT_j = 1 - (w'_j/w_j)(γ_j/γ'_j)^{1/(1-σ)}`,
  which the implementation's own `melitz_gains_from_trade`/checkpoint already does (under the
  universal `γ_j≡1` normalization, this reduces to the simplified form used in production) — not
  from the ACR formula, which is a **Pareto-benchmark sufficient statistic**, not a general
  restriction of this closure away from that benchmark.

---

## Part I: factual equilibrium (all origins)

| check | formula | upper | lower |
|---|---|---:|---:|
| F1 trade shares, all D²=400 cells | `N_o·E_LFD[x_od·1{active}] = X̄_od` (N_o≡1) | max\|resid\| = 8.44e-15 | max\|resid\| = 2.91e-14 |
| F2 price index, all D destinations | `γ_d = Σ_o N_o·E_LFD[p_od^{1-σ}·1{active}]` | max\|γ_d−1\| = 7.06e-8 | max\|γ_d−1\| = 7.06e-8 |
| F4 resource/labor equation (this repo's actual **iceberg** convention — see note below) | `w_o·L_o = Σ_d X_od` | max\|resid\| = 1.43e-5 (data); 1.43e-5 (model-predicted) | same |
| F5 zero-profit cutoff, all D²=400 cells | `π_od(ẑ_od)=0` (forward `melitz_C`/`melitz_cutoff`, independent formula path) | max\|resid\| = 1.25e-15 | max\|resid\| = 1.39e-15 |

**F4 note**: this repo's maintained implementation uses a **pure iceberg** trade-cost convention
— confirmed by direct code reading (`equilibrium.jl`'s `melitz_solve_wages_ge`/`population_X`) and
by the repo's own documentation (`docs/melitz_delta_star.md` §13.6: *"Already satisfied by
construction, no change needed: `population_X`'s market-clearing condition is the plain
`w_o*L_o = sum_d X_od` (income=sales), with NO `1/tau_od` tariff-revenue term anywhere... `tau_od`
enters only through marginal cost... consistent with an iceberg-cost interpretation throughout."*).
The audit prompt's offered alternative formula (with a `(σ-1)/(σ(1+t_od)) + 1/σ` weighting) applies
to an ad-valorem-tariff-with-revenue-rebate convention this repo does not implement; per the
governing instruction to "use the exact production definitions where they differ," the plain
income=sales identity is the correct target, and its residual (1.43e-5, inherited entirely from
the underlying calibration data — identical whether evaluated on raw data or the LFD-predicted
trade flow, since F1 holds to 1e-14) is reported above.

### F3 / factual free-entry table (`f_Eo`, all 20 origins)

Full tables: `factual_entry_table_{upper,lower}.csv`. `f_Eo := Π_o^F/w_o` where
`Π_o^F = E_LFD[Σ_d max(0,π_od(z_o))]`; the free-entry residual `Π_o^F − w_o·f_Eo` is identically
zero by this construction (reported explicitly per the instruction not to just assert it) and
**nonnegativity holds for every origin at both incumbents**:

| | upper | lower |
|---|---|---|
| all 20 `f_Eo ≥ 0`? | **true** | **true** |
| `f_Eo` range across origins | [0.3315, 85.09] | [0.3303, 85.21] |
| focal `f_Ej` (France) | 2.6456 | 1.5384 |

The ~257× spread in implied entry costs across origins is a genuine feature of this reduced
closure (no data target pins `f_Eo` for any single origin individually — see Part III) and is
reported as a plausibility observation, not a pass/fail criterion.

---

## Part II: focal autarky equilibrium

### C1: autarky zero-profit cutoff (`ẑ'_jj = 1`)

BigFloat-precision independent recomputation of `π'_jj(1)`:

| | upper | lower |
|---|---:|---:|
| `π'_jj(1)` (BigFloat) | 1.850e-16 | −2.841e-16 |

Holds by construction (`derive_fjj_from_autarky_cutoff` solves `f_jj` exactly for this), confirmed
independently at both incumbents.

### C2: the same `f_Ej` satisfies factual and autarky free entry — proven algebraically, not just numerically

**Claim**: the current `D²+1` system's single focal moment,

```
g_link(θ) = Π_j^F(θ)/w_j − Π_j^A(θ)/w'_j
```

is **exactly, algebraically** equivalent to profiling out a shared `f_Ej`, not merely a separate
restriction that happens to be small.

**Proof**: define, unconditionally and by construction (for any candidate state), the factual and
autarky implied entry costs

```
f_Ej^F := Π_j^F / w_j        f_Ej^A := Π_j^A / w'_j
```

Substituting these definitions directly into `g_link`:

```
g_link(θ) = f_Ej^F(θ) − f_Ej^A(θ)
```

`g_link(θ) = 0` is therefore *literally the same equation*, written with two different variable
names, as `f_Ej^F = f_Ej^A` — "a single entry-cost parameter rationalizes both the factual and
autarky free-entry conditions." No additional assumption, approximation, or separate derivation is
needed: since `f_Ej` is never introduced as an independent field anywhere in the code (confirmed:
`MelitzPrimitives` has no `f_entry` field at all — it is recovered post-solve, never stored as a
primitive), the *only* way this closure can enforce "one `f_Ej` for both regimes" is via exactly
this algebraic substitution. This numerically holds trivially given the link residual is small,
and is verified directly here:

| | upper | lower |
|---|---:|---:|
| `f_Ej^F` (factual) | 2.645567540971803 | 1.538411... |
| `f_Ej^A` (autarky) | 2.645567540971797 | 1.538411... |
| difference | −6.22e-15 | −1.10e-13 |
| FE-link residual (independent) | 6.22e-15 | 1.10e-13 |

**Answer to Final Question 4: yes** — the implemented FE-link moment is exactly equivalent to
profiling out a shared `f_Ej`, both algebraically (above) and numerically (residuals at machine
precision, both directions).

### C3/C4: autarky price-index equation, endogenous `N'_j`, and the resource equation — proven algebraically equivalent given `ẑ'_jj≡1`

**C3 (price index)**: `γ'_j = N'_j · Q'_j` where `Q'_j := E_F[p'_jj(z_j)^{1-σ}]` (the participation
indicator is a.s. 1 given `ẑ'_jj≡1` and the reference Pareto(1,θ*) support `[1,∞)`). Solving:
`N'_{j,γ} := γ'_j / Q'_j`.

**C4 (resource)**: `w'_j·L_j = N'_j · E_F[x'_jj(z_j)·1{z_j≥1}]`. Computed here via a **direct,
independent formula** (not through the entry-cost detour): `N'_{j,resource} := (w'_j·L_j) / E_LFD[x'_jj(z)·1{z≥1}]`.

**Proof these are the same equation**: given `ẑ'_jj≡1` exactly and Pareto(1,·) support (so
`1{z_j≥1}=1` a.s.), substitute `x'_jj(z_j) = E'_j·p'_jj(z_j)^{1-σ}/γ'_j` into C4:

```
w'_j·L_j = N'_j · (E'_j/γ'_j) · E_F[p'_jj(z_j)^{1-σ}]  =  N'_j · (E'_j/γ'_j) · Q'_j
```

Using the maintained autarky income identity `E'_j = w'_j·L_j` (confirmed directly in the code:
`expenditure_prime = ctx.w_prime * ctx.L[j]`):

```
w'_j·L_j = N'_j · (w'_j·L_j/γ'_j) · Q'_j   ⟹   1 = N'_j·Q'_j/γ'_j   ⟹   N'_j = γ'_j/Q'_j
```

— **exactly the C3 solution.** The price-index and resource equations are therefore the *same
equation*, algebraically, given the maintained autarky cutoff normalization; their independent
numerical agreement (below) confirms internal consistency of the LFD-based reconstruction (a
legitimate, valid equilibrium check under the maintained closure, per the corrected instructions),
not two separate restrictions that happen to coincide.

| | upper | lower |
|---|---:|---:|
| `γ'_j` | 0.590381 | 0.675642 |
| `Q'_j` | 0.750695 | 0.579659 |
| `N'_{j,γ}` (price index) | 0.786445 | 1.165585 |
| `N'_{j,resource}` (direct, independent formula) | 0.786445 | 1.165585 |
| relative difference | **0.0** | 1.9e-16 |
| price-index residual (`γ'_j − N'_{j,γ}·Q'_j`) | 0.0 | 0.0 |
| resource-equation residual (shared `N'_j`) | 0.0 | −3.55e-15 |

Both `N'_j` values are **positive and finite** (0.786 upper, 1.166 lower) and, correctly, **not**
equal to `N_j ≡ 1` — exactly as the corrected closure requires.

### C5: entry-resource accounting — nothing is left unaccounted

Derived directly (not previously in either audit): with markup `μ=σ/(σ-1)`, a firm's revenue
decomposes as `revenue = σ·(π + w'·f_jj)` (immediate from `π = revenue/σ − w'·f_jj`), so its total
labor payment is `revenue − π = (σ−1)·π + σ·w'·f_jj`. Aggregating over the `N'_j` entrants (all
active, `Pr(active)=1` given `ẑ'_jj≡1`), and using autarky free entry `w'·f_Ej = Π_j^A` (C2):

```
Total labor demanded = N'_j·[(σ−1)·Π_j^A + σ·w'·f_jj] + N'_j·w'·f_Ej          (production+fixed) + (entry)
                      = N'_j·[(σ−1)·Π_j^A + σ·w'·f_jj + Π_j^A]
                      = N'_j·σ·(Π_j^A + w'·f_jj)
                      = N'_j · E_F[revenue·1{active}]                          (since revenue=σ(π+w'f_jj))
                      = N'_j · E_F[x'_jj(z)·1{z≥1}]
```

— **exactly the C4 resource equation's right-hand side.** This proves the single revenue-based
resource equation already fully embeds variable production labor, destination (here, domestic)
fixed-cost labor, *and* sunk entry-cost labor, once free entry is imposed — nothing is left
unaccounted for, and this holds for **any** `N'_j` (the derivation never assumed `N'_j = N_j`).

### C6: gains from trade, from the factual/autarky price indices

`GT_j = 1 − (w'_j/w_j)·(γ_j/γ'_j)^{1/(1−σ)}`, with `γ_j` the factual (baseline) price-index power
at the focal country, independently recomputed via F2's own loop (not assumed to be exactly 1):

| | upper | lower |
|---|---:|---:|
| `γ_j` (factual, independently recomputed) | 1.0000000669 | 1.0000000669 |
| `GT_general` (full formula, using the actual `γ_j`) | 8.849345% | 0.272270% |
| `GT_simplified` (assumes `γ_j≡1` exactly, matches the production formula) | 8.849341% | 0.272266% |
| **checkpointed GT** | **8.849341%** | **0.272266%** |
| `\|GT_general − checkpoint\|` | 4.1e-6 pp | 4.5e-6 pp |

`GT_general` reproduces the checkpoint to within 4-5e-6 percentage points — fully explained by
`γ_j` sitting at `1.0000000669` rather than exactly 1 (the same ~7e-8-level calibration artifact
already found in F2, not a new discrepancy), confirming the production formula's `γ_j≡1`
simplification is valid to that same tolerance. **The price-index-based GT independently
reproduces the checkpointed GT.**

### ACR (descriptive only, non-binding)

| | upper | lower |
|---|---:|---:|
| `GT_ACR = 1 − λ_jj^{1/θ*}` (data-anchored domestic share) | 2.030281% | 2.030281% |
| `\|GT_general − GT_ACR\|` | 6.819 pp | 1.758 pp |

Reported descriptively only, per the corrected instructions: ACR is a sufficient-statistic
implication of the exact Pareto calibration benchmark (confirmed to hold there to ~1e-14 in the
repo's own Gate A record, `docs/melitz_delta_star.md` §14.2), not a general restriction of this
closure at an arbitrary LFD/candidate-model point. Its disagreement here — expected once `γ'_target`
is searched away from its calibration value — **is not treated as an equilibrium failure.**

---

## Part III: reconciliation of the current `D²+1` implementation

| step | maintained-model reduction | does the code do exactly this? |
|---|---|---|
| 1. `f_Eo = Π_o^F/w_o` for all origins | factual free entry profiled out via this definition | **Yes** — `recover_entry_costs_from_lfd` (equilibrium.jl) computes exactly this; verified independently for all 20 origins, all nonnegative |
| 2. `Π_j^F/w_j = Π_j^A/w'_j` | focal factual+autarky free entry reduced to one link moment via a shared `f_Ej` | **Yes** — `moments.jl`'s `g_free_entry_link` column is literally `Π_j^F/w_j − Π_j^A/w'_j`; proven algebraically equivalent to `f_Ej^F=f_Ej^A` above (C2) |
| 3. `N'_j = γ'_j / E_F[p'_jj^{1-σ}]` | autarky price-index equation profiled out via the endogenous `N'_j` | **Reconstructable and consistent, but not consumed anywhere in the active moment/outer-search system** — `recover_N_prime_price_index`/`recover_N_prime_market_clearing` (equilibrium.jl) compute exactly this, but `moments.jl`'s `melitz_moments!` never references `N'_j` (it doesn't need to: the focal link moment only needs *profit*, which the free-entry substitution shows is independent of `N'_j`) |
| 4. same profiled `N'_j` used consistently elsewhere | resource equation and any reported equilibrium objects use the SAME `N'_j` | **Consistent by proof (C3/C4 above) and by direct numerical check** (relative difference 0.0 / 1.9e-16) — but since no other active equation in the code actually *uses* `N'_j` (see row 3), this consistency is a property of the two available reconstruction formulas agreeing, not of a stored value being threaded through multiple consumers |

**Answer**: this is exactly the correct dimensional reduction, and the code implements it
correctly for steps 1-2 (both directly, in the active moment system) and correctly *makes
available* (but does not itself consume) steps 3-4 via `equilibrium.jl`'s existing
reconstruction functions. `N'_j`'s existence, positivity, finiteness, and cross-formula
consistency are **mathematically guaranteed** by the algebraic proof above (C3/C4), independent
of whether any calling code happens to read `MelitzCounterfactual.entrant_mass_prime` — and this
audit confirms the guarantee holds numerically at both incumbents.

---

## Part IV: nonfocal counterfactual closure

The maintained exercise evaluates **unilateral focal autarky**: the focal country `j` is placed
in isolation (`τ'_jj=1`, no imports/exports), while every nonfocal object — `w_o` (o≠j), `A_od`,
`f_od`, `τ_od` for cells not involving `j`'s autarky construction, and every nonfocal destination's
baseline price index `γ_d` (d≠j) — is held **fixed at its factual/baseline value**. Confirmed
directly from the code: `MelitzCounterfactual` carries only focal-country fields
(`target_country`, `w_prime`, `expenditure_prime`, `cutoff_prime`, `trade_flow_prime`,
`entrant_mass_prime`); no nonfocal autarky wage, price index, or entrant mass exists anywhere in
the type system or the moment construction. This is the standard "sufficient-statistic"
partial-equilibrium autarky counterfactual (in the spirit of ACR/Melitz-Redding, though here
computed via direct price-index reconstruction rather than the ACR formula) — **not** a claim to
re-solve a full multi-country general equilibrium under focal autarky. No additional nonfocal
free-entry or price-index equations are required for the focal `GT_j` calculation, because
`GT_j` depends only on the ratio of `j`'s own factual and autarky real wages, which this
unilateral construction fully determines.

---

## Decision

**A — full maintained equilibrium holds.**

- All factual trade-share (F1), price-index (F2), resource (F4), and zero-profit (F5) equations
  hold independently, at both incumbents, at machine precision or at the ~1e-8/1e-5 level
  inherited from the underlying calibration data.
- All factual free-entry equations hold after correctly reconstructing `f_Eo` (nonnegative at
  every one of the 20 origins, both incumbents).
- The same focal `f_Ej` satisfies both factual and autarky free entry — proven algebraically
  equivalent to the implemented link moment, and verified numerically to ~1e-13 or tighter.
- The autarky price index identifies a positive, finite `N'_j` (0.786 upper, 1.166 lower),
  correctly **not** equal to `N_j≡1`.
- The same `N'_j` satisfies the autarky resource equation — proven algebraically equivalent to
  the price-index equation given `ẑ'_jj≡1`, and verified numerically (relative difference ≤2e-16).
- Zero-profit, gravity (re-affirmed from the prior audit, unaffected by this correction), and the
  price-index-based GT formula all hold; GT independently reproduces the checkpoint to ~5e-6
  percentage points.
- ACR disagreement is descriptive only and does not bear on this decision.

---

## Final questions

1. **Factual `f_Eo` for each incumbent?** `factual_entry_table_{upper,lower}.csv` — 20 origins
   each; ranges [0.3315, 85.09] (upper) and [0.3303, 85.21] (lower); focal France: 2.6456 (upper),
   1.5384 (lower).
2. **Do all factual free-entry equations hold?** Yes — by construction (residual ≡ 0 given the
   `f_Eo := Π_o^F/w_o` definition) and all 20 are nonnegative at both incumbents.
3. **Does the same focal `f_Ej` satisfy autarky free entry?** Yes — difference ~6e-15 (upper) /
   1e-13 (lower).
4. **Is the FE-link moment exactly equivalent to profiling out `f_Ej`?** Yes — proven
   algebraically (C2) and confirmed numerically.
5. **What endogenous `N'_j` is implied by the price-index equation?** 0.786445 (upper), 1.165585
   (lower).
6. **Is `N'_j` positive and finite?** Yes, both incumbents.
7. **Does the same `N'_j` satisfy autarky resource/market clearing?** Yes — proven algebraically
   equivalent given `ẑ'_jj≡1` (C3/C4), confirmed numerically to ≤2e-16 relative difference.
8. **Is the price-index equation explicitly imposed, or is `N'_j` correctly profiled out?**
   `N'_j` is correctly profiled out — reconstructable and internally consistent via two
   independent formulas, but not itself consumed by the active moment/outer-search system (which
   does not need it).
9. **Are entry costs and entry labor consistently accounted for?** Yes — proven directly (C5):
   the resource equation's revenue-based right-hand side already embeds variable production,
   fixed-cost, and entry-cost labor once free entry is imposed, for any `N'_j`.
10. **Does the price-index-based GT formula reproduce the checkpoint?** Yes, to ~5e-6 percentage
    points (fully explained by `γ_j` sitting at 1.0000000669 rather than exactly 1).
11. **Does ACR hold at the Pareto reference point?** Yes (previously verified in the repo's own
    Gate A record, `docs/melitz_delta_star.md` §14.2, to ~1e-14); it does not, and is not required
    to, hold at these searched incumbents.
12. **Are the upper and lower incumbents feasible for the actual maintained model?** Yes.
13. **Which conclusion is supported?** **A.**

## Required outputs

1. This document.
2. `docs/key_results/melitz_corrected_endogenous_entry_full_equilibrium_audit_2026-07-31/factual_entry_table_{upper,lower}.csv`
3. `docs/key_results/melitz_corrected_endogenous_entry_full_equilibrium_audit_2026-07-31/focal_autarky_table_{upper,lower}.csv`
4. Reconciliation table: Part III above.
5. Independent GT reconstruction: C6 above.
6. Reproducible script: `scripts/melitz_corrected_endogenous_entry_audit_2026-07-31.jl`; provenance
   (checkpoint/data fingerprints, git HEAD) reuses `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/provenance_2026-07-31.txt` (same checkpoints, same data, unchanged).
