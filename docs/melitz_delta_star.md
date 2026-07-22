# Full-D Melitz Δ\* benchmark: design document

Branch: `melitz/fullD-delta-star` (created from `origin/production/fullA-exact` @ `670eac4`).
This document is written *before* the bulk of the implementation, per the project brief,
and is updated with actual results once the D=4 benchmark runs (§13-14).

## 0. Scope of this milestone

Build the first clean, full-country (no rest-of-world aggregation) Melitz/Chaney
implementation, validate it end-to-end against the Christensen–Connault (CC)
minimum-divergence machinery already in `production/fullA-exact`, and demonstrate
**Δ\* ≈ 0** for a D=4 synthetic economy generated from the Pareto reference distribution
F\*. Finite-δ upper/lower bound programs are explicitly **out of scope** for this
milestone (per the brief).

---

## 1. The mathematical model

### 1.1 Firm problem (pure Melitz, constant markup)

A continuum of potential entrants in origin `o` draw productivity `z_o ~ Pareto(1, θ*)`,
i.i.d. across draws, independently across origins. A firm with productivity `z` selling
into destination `d` faces

```
marginal_cost_od(z) = w_o · τ_od / (A_od · z)
price_od(z)         = markup · marginal_cost_od(z),   markup = σ/(σ-1)
```

`A_od` is the paper's efficiency shifter (higher `A` ⟹ lower marginal cost — see §2 for
why this needs no adapter against the existing repo). `τ_od = 1+t_od` is the full variable
trade-cost wedge, diagonal = 1. There is **no** `ρ`, no perfect-competition fallback, no
hybrid branch: `markup` is always `σ/(σ-1)` and the model requires `σ > 1`.

Demand for a firm's variety in `d` is CES, so unconstrained revenue is

```
price_power_d           = Σ_o N_o · E_F[ price_od(z)^(1-σ) · active_od(z) ]     (≡ "γ_d")
unconstrained_revenue_od(z) = expenditure_d · price_od(z)^(1-σ) / price_power_d
operating_profit_od(z)  = unconstrained_revenue_od(z)/σ − w_o · f_od
active_od(z)            = operating_profit_od(z) > 0
```

with `realized_revenue`/`realized_operating_profit` zero when `!active` (the `>0`
convention is used consistently everywhere; the boundary has probability zero under the
continuous Pareto). `N_o` (paper: `N_o`; code: `entrant_mass`, or bare `N` when
unambiguous) is the **economic mass of potential entrants** — never a numerical firm
count, and never divides revenue or fixed costs. `W`/`num_draws` is a pure Monte-Carlo
integration-accuracy setting.

### 1.2 The `price_power_d ≡ γ_d = 1` normalization

Multiplying every `A_od` (fixed `d`) by a constant `c_d` divides every price in
destination `d` by `c_d` and leaves `p_od(z)^(1-σ)/price_power_d` — hence every economic
object (revenue, profit, cutoffs, entry) — unchanged, since `price_power_d` scales by
`c_d^(σ-1)`. This is pure destination-specific scale freedom in `A_od`'s level, with no
economic content. We use it to set **`price_power_d = 1` for every baseline destination
`d`** (§4 shows this holds automatically, at machine precision, once `expenditure_d` is
defined as `Σ_o X_od` — no separate rescaling step is needed for the synthetic fixture).
With this normalization, `realized_revenue_od(z) = expenditure_d · price_od(z)^(1-σ)`
directly (no division needed since `price_power_d ≡ 1`), matching the addendum's §3.4.

The counterfactual `price_power'_d` (only needed for the focal/target country in the
autarky counterfactual) is **not** independently normalized — it is an endogenous object
computed from the normalized baseline `A`.

### 1.3 Zero-profit cutoff and the analytical Pareto aggregate

Define `C_od = expenditure_d · (markup·w_o·τ_od/A_od)^(1-σ)`, so `realized_revenue_od(z) =
C_od · z^(σ-1) · active`. The zero-profit cutoff solves `C_od·ẑ_od^(σ-1) = σ·w_o·f_od`, i.e.

```
ẑ_od = (σ·w_o·f_od / C_od)^(1/(σ-1))
```

For `z ~ Pareto(1, θ*)` (density `θ*·z^(-θ*-1)`, `θ* > σ-1` required for finiteness):

```
Pr(z > ẑ)              = ẑ^(-θ*)                                     (ẑ ≥ 1)
E_F[z^(σ-1)·1{z>ẑ}]    = θ*/(θ*-σ+1) · ẑ^(σ-1-θ*)
```

so aggregate bilateral trade under F\* is

```
X_od = N_o · C_od · θ*/(θ*-σ+1) · ẑ_od^(σ-1-θ*)                        (*)
```

and expected per-entrant operating profit (derivation in §1.5) collapses to

```
E_F[operating_profit_od(z)] = C_od·(σ-1)/(σ·(θ*-σ+1)) · ẑ_od^(σ-1-θ*)
```

### 1.4 `N_o` is derived, closed form, from `L_o` and `f_entry_o` — matching Melitz-Redding eq. (22)

This derivation went through three passes in one session, worth recording because the
final answer is a genuine reconciliation of two seemingly conflicting facts, not a
correction of one by the other.

**Pass 1** tried to use free entry to *solve for* `N_o` directly and failed (a
D-dimensional NLsolve system that would not converge even after homotopy).

**Pass 2** found that `N_o` cancels out of the free-entry condition entirely once `C_od`
is expressed the "structural" way (`C_od = expenditure_d·(markup·w_o·τ_od/A_od)^(1-σ)`,
no `N_o`), concluding `N_o` is a free scale like the Ricardian repo's `L` — this is true,
but only as a statement about the *data-inversion* parameterization of §1.6 (`C_od` built
from `X_od/N_o`): for **that specific parameterization**, `N_o` and `A_od` trade off
against each other holding `X_od` fixed, so `N_o` is not separately identified from
*aggregate trade data alone*. This is a real, defensible econometric point (§1.7) — but it
does not mean `N_o` is unconstrained by the *model's own primitives*.

**Pass 3**, prompted by checking Melitz & Redding's own closed-form mass-of-entrants
result (their eq. 22, `M_Ei=(σ-1)/(kσ)·L_i/f_Ei`, derived from a genuine labor
income/resource identity, `w_i·L_i` = revenue = mass-of-entrants × average revenue per
potential entrant), found the same identity holds here: combining
`f_entry_o = S_o·(σ-1)/(σ·θ*·w_o)` (`S_o = Σ_d C_od·M_od`, from §1.6's `entry_cost_from_free_entry`,
still valid regardless of parameterization) with `N_o = w_o·L_o/S_o` (the income/sales
identity, already enforced by `melitz_solve_wages`, §1.5) and eliminating `S_o` gives

```
N_o = (σ-1)/(σ·θ*) · L_o/f_entry_o
```

— exactly Melitz-Redding's eq. (22), extended unchanged to the bilateral-`A_od` case (the
cutoff/`A`/`τ` dependence cancels algebraically, exactly as in their result). **Verified
numerically** (`verify_closed_form_N.jl`, this session) against the already-working
Pass-2 construction: computing `N` this way from the *already-derived* `f_entry`
reproduces the *already-chosen* `N` to 1e-9 — confirming Pass 2 and Pass 3 are the same
system of equations, just solved in opposite directions (Pass 2: choose `N`, derive
`f_entry`; Pass 3: choose `f_entry`, derive `N`).

**Resolution:** `f_entry_o` (with `L_o`) is the primitive; `N_o` is derived via the
closed form above — matching Melitz-Redding, matching the brief's original parameter
table, and matching standard Melitz practice. Pass 2's non-identification point is not
wrong, it is simply about a *different* question (§1.7: what can be recovered from
*aggregate trade data alone*, without knowing `L_o`/`f_entry_o` — relevant to the
eventual F\* *solver*, §9) than what this section answers (what pins `N_o` in the
*forward, deep-primitives* construction of the synthetic fixture).

### 1.5 Data, not deep primitives: mirroring the Ricardian repo's actual architecture

A second, related correction (also caught mid-session): the Ricardian repo's data object
is trade **shares** `λ_od`, not trade-flow **levels** `X_od`. Shares alone under-determine
levels — levels require `expenditure_d = w_d·L_d`, and wages must be *solved* so that
income (`w_o·L_o`) equals sales (`Σ_d X_od`) for every country *simultaneously* (no
deficits). An earlier draft of this file chose `X_od` levels directly and only impose
`expenditure_d = Σ_o X_od` (destination-side balance) — that silently drops the
origin-side balance (`w_o·L_o = Σ_d X_od`) and can run an undetected trade deficit.

The fix reuses `prestep/iterWagesPreStep!.jl`'s exact damped-Jacobi fixed point verbatim
(`melitz_solve_wages` in `equilibrium.jl`): `w1 = λ·(w0.*L)./L`, iterated to convergence,
normalized `w[1]=1`. Given `λ` (data, columns sum to 1), `L` (chosen labor endowment),
this is the **only** numerical fixed point in the entire construction — well-behaved
(effectively finding the stationary vector of a column-stochastic-weighted system),
nothing like the abandoned steep power-law system of §1.4's first attempt. Given the
solved `w`, `expenditure_d := w_d·L_d` and `X_od := λ_od·expenditure_d` follow in closed
form and satisfy **both** row and column balance by construction.

### 1.6 Profiling the D² trade-flow equations analytically (the addendum's parameterization)

Given `N_o` (§1.4, derived from `L_o`/`f_entry_o`) and *any* candidate cutoff `ẑ_od ≥ 1` (with `ẑ_od ≥
ẑ_oo` for `d≠o`, per the export-selection restriction), `(*)` can be solved for `C_od`
directly:

```
C_od = (X_od/N_o) · (θ*-σ+1)/θ* · ẑ_od^(θ*-σ+1)
```

and then, inverting the definitions of `C_od` and the cutoff condition,

```
A_od = markup·w_o·τ_od · (C_od/expenditure_d)^(1/(σ-1))
f_od = C_od·ẑ_od^(σ-1) / (σ·w_o)
```

This is exactly the derivation sketched in the addendum §1.5 (independently re-derived
here; the two agree term for term). **The D² trade-flow equations are therefore satisfied
by construction for any choice of `{ẑ_od}`** — they never separately identify `A_od` and
`f_od`; the free parameterization is the `D²`-dimensional cutoff matrix `{ẑ_od}`
(equivalently, any `A/f` pair related to it by the formulas above), subject only to the
two gravity restrictions and the support/export-selection inequalities. We use this
cutoff parameterization as the practical implementation device in `fake_data.jl`, but
**do not claim it is the economically identified object** — see §9 and the addendum §1.3.

### 1.7 Non-identification of `A_od`/`f_od` individually (why this must not be oversold)

Because `price_power_d ≡ 1` removes `N_o` from firm-level revenue, and because the
data-inversion parameterization of §1.6 shows `N_o` and `A_od`'s level trade off against
each other holding *aggregate trade data* fixed (§1.4's Pass 2), the object that
aggregate F\*-Melitz trade **data alone** identifies, once double-differenced, is the
composite the addendum derives:

```
r = θ*·a + β·h,     a = ΔΔ log A,  h = ΔΔ log f,  β = 1 - θ*/(σ-1)
```

not `a` and `h` separately. §1.6's cutoff parameterization is one convenient *computational*
way to pick a point in the affine set of decompositions consistent with `r`; it is not a
claim that `ẑ_od` (or the implied `A_od`/`f_od` split) is identified by the data. The
acceptance tests (§13) check the identified composite and the two orthogonality
restrictions, never elementwise recovery of `A`/`f`.

### 1.8 Autarky: the target country's own baseline cutoff is derived, not chosen

The brief requires `ẑ'_{target,target} = 1` (the autarky cutoff sits exactly at the Pareto
lower support) while *also* holding the target country's resources fixed across baseline
and autarky (same labor endowment, same wage numeraire `w'=w=1`) — necessary for the
ACR/Chaney cross-check to be a meaningful comparison at all (ACR compares two equilibria
with the *same* endowment; if `expenditure_prime` were allowed to differ from the
baseline's own `expenditure[target]`, the price-index ratio would reflect a scale change,
not a pure market-access change, and would have no reason to match `1-λ_dd^(1/θ*)`).

An earlier pass at this derivation got the *direction* backwards — it tried to derive
`f[target,target]` from `f_entry[target]` and treat `N[target]`/`expenditure_prime` as
free. Redone correctly (holding `expenditure_prime[target] = expenditure[target]`,
`w'=1`, `τ'_tt=1`, and combining the baseline and autarky zero-profit conditions with the
price-power definitions for cell `(target,target)`), the algebra shows the *baseline*
cutoff for that one cell, not `f_entry` or `N`, is what must be pinned:

```
ẑ[target,target] = (expenditure[target]·w[target] / X[target,target])^(1/θ*)
                  = (w[target] / λ_tt)^(1/θ*)
```

(`λ_tt = X[target,target]/expenditure[target]`, the baseline domestic trade share — the
resemblance to the ACR formula `1-λ_tt^(1/θ*)` is exactly why the cross-check below comes
out exact). This was verified two ways: symbolically, and numerically (re-deriving it by
substitution gave an apparently different, `N`-dependent formula on a first pass; a direct
numeric probe in `test_equilibrium.jl` showed the `N`-dependent formula is an identity that
holds automatically once this cutoff value is used, for *any* `N[target]` — confirming
`N[target]` is free (§1.4) and this cutoff is the one genuine normalization). `f_entry` and
`f[target,target]` are then computed by the *same* closed-form formulas as every other
cell (§1.6) — no special-casing beyond fixing this one cutoff rather than letting it be
gravity-projected like the other `D²-1` cells (§10's projection step treats it as a fixed
constant, exactly as it already treated the old, now-superseded, `f[target,target]`
formula).

**Validated end-to-end** (`test_equilibrium.jl`, this session): with this cutoff fixed,
`price_power_d≡1` for all `d`, the autarky cutoff comes out at exactly `1`, and
`GT_model` (via the price-power ratio) matches `GT_ACR = 1-λ_tt^(1/θ*)` to **2.2e-16**
— including after changing `N[target]` from `1.5` to `7.3`, confirming free entry data
matches to machine precision independent of the free `N` choice.

---

## 2. Mapping to the repository's notation (`A`, `c`, `γ`)

**`A` (productivity):** confirmed independently (not merely inferred from the legacy
file) that `production/fullA-exact`'s underlying convention is `MC_od = w_o·τ_od/(A_od·z_o)`
— i.e. *higher* `A` ⟹ *lower* marginal cost, **exactly the paper's convention**. The repo
internally stores the reciprocal `AodPow = 1/A_od` for its own numerics/gradients, but the
economic object already matches the paper. **No adapter is needed**; the Melitz module
defines its own `A[o,d]` directly in this convention and does not reuse `cHat`/`AodPow`.

**`γ` (price-related object) — deliberately renamed, not reused:** the repo's own `γ`
satisfies `γ_d^σ · gdp_d = P_d^(1-σ)` (a GDP-and-σ-rescaled Fréchet price-power aggregate
built from `SpecialFunctions.gamma`, the Euler Gamma special function — an unrelated
naming collision the repo's own code comments flag). The Melitz module's analogous object
(§1.1, `price_power_d = Σ_o N_o E_F[p_od^(1-σ)·active]`) is a **different formula** (no
Euler-Gamma special function; the truncation comes from the Pareto tail, not a Fréchet
extreme-value moment) and is named `price_power` throughout, never `gamma`, to avoid the
collision. `computeGamma.jl` is not reused.

**`doubleDiff` vs. `withinTransform`:** the repo has two double-differencing-style
operators (`misc/doubleDiff.jl`): the cell-referenced `doubleDiff(z)` (reference row 1 /
column 2) and `withinTransform(z)` (two-way fixed-effects "within" transform). Production's
*own* live gravity moment uses `withinTransform`, because `doubleDiff` does not reproduce
an OLS two-way-FE elasticity-regression coefficient — but that concern is about
*estimating θ* by regression, which the Melitz module never does (θ\* is a fixed,
calibrated benchmark parameter here, §5). The two Melitz gravity restrictions are direct
scalar covariance restrictions `⟨ΔΔlogτ, ΔΔlogA⟩=0` / `⟨ΔΔlogτ, ΔΔlogf⟩=0` on the
model's own `A`/`f`, which is exactly what `doubleDiff` computes (and both live in the
same `(D-1)²`-dimensional double-differenced space the addendum's math uses). The Melitz
module therefore uses **`doubleDiff`**, matching the addendum's `ΔΔ` notation and the
brief's explicit instruction to reuse it, and documents (here) why this differs from
production's own `withinTransform` usage rather than silently picking one.

---

## 3. Integration draws vs. `N_o`

- `z_o ~ Pareto(1, θ*)`, one draw per **origin**, `W × D` reference-draw matrix (matches
  the repo's own `UoModel=1` convention, `prepare_cc/drawU.jl`). Generated once via a
  single `Random.seed!(seed)` + `rand!` + inverse-CDF transform (`z = (1-u)^(-1/θ*)`),
  mirroring `prepare_cc/genRands.jl`'s "seed once, draw once, reuse everywhere" discipline
  — not `cc_algo/rhalton.jl` (validated but unused on the production hot path; available
  as an opt-in alternative, §11).
- `W`/`num_draws` is a pure accuracy dial. `N_o`/`entrant_mass[o]` is the economic mass of
  potential entrants (§1.1, §1.4) and is never divided into revenue or fixed costs, and
  never confused with `W`.
- `N'_o = N_o` is imposed by construction: the counterfactual moment code receives the
  identical `entrant_mass` vector as the baseline (never an independently searched
  `NPrime`).

---

## 4. Moment functions

All moments are built from the same shared per-draw firm routine (`firm_quantities.jl`),
called identically in baseline and counterfactual — never a separate manual counterfactual
formula and never evaluated at a single draw index (the two anti-patterns the legacy audit
flags).

### A. D² bilateral trade-flow moments — matched as SHARES, not levels

```
lambda_od = X_data[o,d] / expenditure_d
g_trade[o,d](z) = entrant_mass[o]·realized_revenue_od(z_o)/expenditure_d − lambda_od
```

for **every** `(o,d)` pair — all `D²` cells, never gated on `baseIndex`/"rest of world".
**Corrected mid-session from an earlier level-based draft** (`g_trade =
entrant_mass[o]·realized_revenue − X_data[o,d]`, optionally rescaled by the cell's own
`X_data[o,d]` for conditioning): live user feedback pointed out the Ricardian repo's own
`moments/hFunction.jl` matches **shares**, not levels — confirmed by reading it directly:
`G[ω,d1] = pricesTemp[o] − P[d1]·denom[d]`, where `P[d1]` is literally the vectorized
trade-*share* data (`prepare_cc/buildObjectsForMoments.jl`) and `denom[d]=γ[d]^σ·gdp[d]`
is a destination-level rescaling constant applied to make it comparable to the simulated
term — the fundamental data object matched is the share. `setup/createFakeData.jl`
confirms this is a **closed-form** object (`lambda = phi./sum(phi,dims=1)` under the
Fréchet gravity equation), never a Monte Carlo sample average — the Melitz module's own
`X_data`/`entry_target` default (`build_melitz_psi_bundle`) mirrors this: the closed-form
population values `(eq.trade_flow, w.*f_entry)`, not a sample mean over the same draws
used to evaluate the moments (see §14 for why that distinction matters numerically).
Matching all `D²` flows with `expenditure_d = Σ_o X_od` implies `price_power_d=1`
automatically (§1.6's derivation: `price_power_d = (1/expenditure_d)·Σ_o N_o C_od M_od =
(Σ_o X_od)/expenditure_d ≡ 1`) — **no separate price-index moment is added.**

### B. D free-entry moments (per-entrant, not multiplied by `N_o`)

```
g_entry[o](z) = Σ_d realized_operating_profit_od(z_o) − w_o·f_entry_o
```

`N_o` never appears here (§1.1: it multiplies aggregate trade and price-power
contributions, not the per-potential-entrant free-entry condition — this is a modeling
requirement, not a simplification we chose for convenience; §1.4's closed form for `N_o`
would be circular/wrong if `N_o` also entered this equation).

### C. Two gravity restrictions (outer, F-independent equality constraints)

`⟨doubleDiff(τ), doubleDiff(A)⟩ = 0` and `⟨doubleDiff(τ), doubleDiff(f)⟩ = 0`. Following
production's own treatment of its analogous F-independent gravity moment (it is carved out
as an **outer**-loop equality constraint on θ via `outer_constr_index`, not an inner-loop
column matched over draws — `prepare_cc/master_prepare_cc.jl`'s `nOuterLoopMoments`
accounting), the Melitz module does the same: these two restrictions are evaluated once
from `(A,f,τ)` (no `U` dependence) and enter as equality constraints in the outer θ-search,
**not** as additional `G` columns — avoiding the brief's "do not duplicate the same
restriction in both places."

### D. Factor-market / expenditure closure

Following the Ricardian repo's own minimal convention (confirmed: no `deficit`, no
`tariff` revenue, and — critically — **no `gdp_adjustment` object anywhere** in
`production/fullA-exact`; GDP is simply `w·L`), the Melitz module adopts the analogous
closure `expenditure_d := Σ_o X_data[o,d]` (an accounting identity, not a free parameter),
with `L_o := expenditure_o/w_o` reported as a derived diagnostic (labor income), not as an
independent primitive requiring its own equilibrium condition. This is a deliberately
minimal closure matching the Ricardian repo's own level of ambition (§11) — it is *not*
a microfounded labor-resource constraint on production/fixed/entry costs, which the
brief's "any factor-market... restrictions required by the current paper" leaves to the
existing architecture's own precedent.

---

## 5. Parameter inventory

| object | dims | searched / fixed / derived | transform | normalization | role | code location |
|---|---|---|---|---|---|---|
| `σ` | scalar | fixed/calibrated | `log σ` internally if ever searched | `σ>1` required | CES elasticity | `types.jl` |
| `θ*` | scalar | fixed/calibrated | `log θ*` internally if ever searched | `θ*>σ-1` required | Pareto shape | `types.jl` |
| `λ[o,d]` | D×D | **data** (chosen for the synthetic fixture; columns sum to 1) | — | none | trade share (mirrors Ricardian's own data object) | `fake_data.jl` |
| `L[o]` | D | fixed/chosen (labor endowment) | `log L` | none | endowment | `fake_data.jl` |
| `w[o]` | D | **solved** via `melitz_solve_wages` (reuses `iterWagesPreStep!`'s exact fixed point, given `λ`, `L`) | — | `w[1]=1` (numeraire) | wage | `equilibrium.jl` |
| `w'[target]` | scalar | fixed | — | `w'[target]=1` (counterfactual numeraire) | autarky wage | `equilibrium.jl` |
| `τ[o,d]` | D×D | fixed (data) | — | diag = 1 | trade cost | `fake_data.jl` |
| `expenditure[d]` | D | derived, closed form: `w_d·L_d` | — | `expenditure_d = Σ_o X_od` (holds automatically given the wage solve) | destination spend | `equilibrium.jl` |
| `X[o,d]` | D×D | derived, closed form: `λ_od·expenditure_d` | — | `Σ_d X_od = w_o·L_o` (income=sales, holds automatically) | trade flow (data level) | `equilibrium.jl` |
| `f_entry[o]` | D | **primitive (chosen/calibrated)** | `log f_entry` | none | entry cost | `fake_data.jl` |
| `entrant_mass[o]` (`N_o`) | D | **derived, closed form**: `(σ-1)/(σθ*)·L_o/f_entry_o` (§1.4 — matches Melitz-Redding eq. 22) | — | `N'_o=N_o` (shared object) | mass of potential entrants | `equilibrium.jl` |
| `ẑ[o,d]` | D×D | free computational parameterization (§1.6), not economically identified — **except** `ẑ[target,target]` | — | `ẑ_od≥1`; `ẑ_od≥ẑ_oo` (d≠o); `ẑ[target,target]` **derived** (§1.8) | baseline cutoff | `fake_data.jl` |
| `A[o,d]` | D×D | derived, closed form from `(X,N,w,τ,expenditure,ẑ)` (§1.6), non-identified split (§1.7) | `log A` | `price_power_d≡1` holds automatically given the closure above | efficiency shifter | `equilibrium.jl` |
| `f[o,d]` | D×D | derived, closed form (§1.6) | `log f` | none beyond `ẑ[target,target]`'s own normalization | fixed market-access cost | `equilibrium.jl` |
| `price_power[d]` | D | derived diagnostic | — | `≡1` by construction (§1.2/1.5) | CES price-power aggregate | `equilibrium.jl` |
| `price_power'[target]` | scalar | derived (autarky, closed form) | — | none (endogenous) | counterfactual price-power | `equilibrium.jl` |
| `ẑ'[target,target]` | scalar | normalized | — | `=1` exactly (Pareto lower support) | autarky cutoff | `equilibrium.jl` |
| Pareto lower support | scalar | fixed | — | `=1` | F\* primitive | `pareto.jl` |

---

## 6. Firm-level calculations

Single shared routine `melitz_firm!`/`melitz_firm` in `firm_quantities.jl`, used for both
baseline and counterfactual, evaluated at every draw (never draw index 1 only):
`marginal_cost → price → unconstrained_revenue → operating_profit → active →
realized_revenue/realized_operating_profit`, exactly the formulas in §1.1.

---

## 7. Why free-entry is not multiplied by `N_o`

`f_entry_o` is the entry cost **per potential entrant** — the condition `E_F[Σ_d
operating_profit_od(z)] = w_o f_entry_o` says a representative potential entrant expects
zero net profit from paying the entry cost and then discovering `z` and choosing where to
sell. Multiplying by `N_o` would conflate "zero expected profit per entrant" with "zero
*aggregate* profit across the whole mass of entrants" — a different condition, and it is
precisely because this equation is `N_o`-free that combining it with the labor/income
identity yields `N_o`'s own closed form (§1.4) — multiplying by `N_o` here would corrupt
that derivation, not just be economically wrong on its own terms.

---

## 8. Baseline cutoffs and autarky cutoff normalization

Baseline cutoffs `ẑ[o,d]` are the free computational parameterization of §1.6 (chosen
subject to `ẑ≥1`, `ẑ_od≥ẑ_oo`) — **except** `ẑ[target,target]`, which is **derived**
(§1.8's closed form) rather than freely projected, so that the autarky cutoff
`ẑ'[target,target]=1` holds exactly while holding the target country's resources fixed.
The cell is simply excluded from the gravity-projection step (§10) that produces the
other `D²-1` cells of `ẑ` (equivalently `A`/`f`), with its known, fixed contribution to
the restriction's inner product netted out of that projection.

---

## 9. `N'_o = N_o`

The equilibrium/moment code takes a single `entrant_mass::Vector` and passes the same
vector to both the baseline and counterfactual firm/moment evaluators — there is no
`NPrime` object anywhere in the Melitz module (the legacy code's `NumberActiveFirmsPrime`
pattern, which computes a separate value and is later force-equated to `NumberActiveFirms`
via an `impose_M_Mprime_equality` flag, is explicitly rejected).

---

## 10. Full-D gravity restrictions (construction)

`fake_data.jl` generates raw `log ẑ` (the cutoff parameterization, §1.6) with
heterogeneous cell-specific noise plus origin/destination fixed effects (which vanish
under `doubleDiff`). Because `log A_od` and `log f_od` are each *affine* in `log ẑ_od`
given the other data fixed (coefficients `(θ*-σ+1)/(σ-1)` and `θ*` respectively — derived
this session, §1.6), the two gravity restrictions reduce to a single linear requirement
on `⟨doubleDiff(τ), doubleDiff(log ẑ)⟩`, itself pinned by requiring the trade-flow data's
own composite restriction `⟨T, ΔΔlogX + θ*T⟩ = 0` to hold (§1.7's identified-composite
condition). `fake_data.jl` projects the *free* cells' double-differenced `log ẑ`
component onto that required value (Gram-Schmidt-style), with `ẑ[target,target]` excluded
and held at its §1.8 value throughout (its known, fixed contribution to the inner product
is netted out so the free cells absorb the correction). Both restrictions are verified
numerically to machine precision as part of fixture construction, and `std(log A) > 0`,
`std(log f) > 0`, `std(doubleDiff(log A)) > 0`, `std(doubleDiff(log f)) > 0` are asserted
(never trivially-all-ones matrices).

---

## 11. Ricardian routines reused / adapted

**Reused as-is:** `misc/doubleDiff.jl`'s `doubleDiff` (§2, §10); the general "seed once,
draw once" discipline of `prepare_cc/drawU.jl`/`genRands.jl` (adapted to Pareto inverse-CDF
instead of Exp(1)+Fréchet transform, §3); `prestep/iterWagesPreStep!.jl`'s damped-Jacobi
fixed point `w1 = λ·(w0.*L)./L` reused **verbatim** as `melitz_solve_wages` (§1.5 — not
just "the same pattern": this is the one and only numerical fixed point anywhere in the
Melitz construction, taking trade *shares* as data exactly as the Ricardian repo does);
the CC minimum-divergence inner loop
(`cc_algo/ccInner.jl`, `cc_algo/inner_loop_functions.jl`, `PsiObjectiveBundleDelta` from
`cc_algo/PsiObjectiveBundle.jl`) — the Melitz module supplies its own `moments!(K,G,θ,U,obj)`
function and plugs it into an unmodified `PsiObjectiveBundleDelta`; the outer Δ\*
minimization via the method-agnostic `cc_algo/outer_loop_cached.jl` +
`cc_algo/free_param_map.jl` (recommended integration point per the infrastructure survey,
since it accepts caller-supplied gradients rather than assuming the Fréchet-specific
`jac_h`/autodiff-through-`moments!` path); `cc_algo/Psi.jl`'s divergence functions
(model-agnostic).

**Not reused, with reason:** `prestep/computeGamma.jl` (Fréchet-specific, uses
`SpecialFunctions.gamma`, wrong formula for Melitz — §2); `moments/hFunction.jl`/
`moments/moments!.jl` (Fréchet-specific per-draw firm formulas — Melitz has its own,
§6); `withinTransform` for the gravity restriction itself (used for elasticity
*estimation* in production, not applicable here — §2, though the function remains
available if a future extension needs it).

---

## 12. Legacy routines adapted or rejected

See `docs/melitz_legacy_audit.md` for the full file/line inventory. In summary: the
constant-markup price/revenue/profit algebra and the Pareto tail-expectation *shape* were
useful cross-checks (re-derived independently in §1.3-§1.7, not copied); the
`f[target,target] = f_entry·(θ*-σ+1)/(σ-1)` formula in particular was found, isolated, and
confirmed correct — but only after re-deriving it from scratch (§1.7) and then cross-
checking against the buried legacy expression, not by trusting the legacy code directly.
Everything else — foreign-country aggregation, the ρ/σ hybrid branch, `N`-as-firm-count,
single-draw counterfactuals, `gdp_adjustment=ones(D)`, repurposed parameter slots — is
rejected per that document.

---

## 13. Reproducing the D=4 Δ\* test

```
source .knitro_env.sh   # KNITRO license only resolves on demand.mit.edu
julia --project=. scripts/run_melitz_delta_star_fake.jl --D 4 --seed 1234 --draws 80000
```

Prints normalizations, the true parameter summary, moment residuals (with the
`min_active_draw_count` diagnostic — see §14), the GT/ACR cross-check, and `Delta(theta*)`
via the real, unmodified CC/KNITRO inner loop. The formal automated equivalent is

```
julia --project=. test/melitz/runtests.jl
```

(104 tests; the last two testsets require KNITRO and are skipped with a warning, not a
hard failure, if `cc_algo`/KNITRO cannot be loaded in the current environment).

## 14. Numerical result and tolerance

**Verified live, `seed=1234`, `--draws 80000`:**

| quantity | value |
|---|---|
| max ⎜bilateral-flow residual⎟ (economic units, Monte Carlo vs. closed-form target) | 0.0424 |
| max ⎜free-entry residual⎟ (economic units) | 0.0249 |
| `A` gravity residual `⟨ΔΔlogτ, ΔΔlogA⟩` | −1.04e-17 |
| `f` gravity residual `⟨ΔΔlogτ, ΔΔlogf⟩` | −4.86e-16 |
| `GT` (model, price-power ratio) | 0.0362015 |
| `GT` (ACR/Chaney, `1−λ_dd^(1/θ*)`) | 0.0362015 |
| ⎜GT_model − GT_ACR⎟ | 1.11e-16 |
| min active-draw count (any cell) at `W=80,000` | 69 |
| **Δ(θ\*)** (real KNITRO CC inner loop) | **1.07e-4** |
| KNITRO status | 0 (optimal) |
| max ⎜optimal dual variable⎟ | 0.212 |

Both gravity restrictions and the ACR cross-check hold to machine precision (1e-16 to
1e-17) by construction, independent of `W` — they never touch the Monte Carlo draws.
`Δ(θ*)` itself is genuinely small but **not** machine-precision zero, and this is the
correct, expected result once the target moments are the closed-form population values
(§4A) rather than a sample average over the same draws used to evaluate the moments (the
brief's own "Mode 1" construction — useful only as a degenerate code-correctness check,
since it makes `Δ(θ*)=0` a tautology, not a validation; confirmed this reduces to exactly
`0.0` with an all-zero dual solution when tested, correctly flagged as suspicious and
investigated live rather than reported as the headline result).

**A real numerical finding, diagnosed live:** at smaller `W` (500 to 32,000 tested), the
KNITRO inner solve does not converge — its dual variables diverge to `~1e14`–`1e16` rather
than settling on a large-but-finite `Δ`. Root cause, confirmed directly: bilateral
participation probability under Pareto is `Pr(active) = ẑ_od^(-θ*)`, which is small enough
for high-cutoff (especially export) cells that some cells have **zero** active draws in
the *entire* Monte Carlo sample at small `W` for this fixture (verified: cell `(1,2)` has
0 active draws at `W=500`; cell `(2,3)` has 0 at `W=2000`, 5 at `W=8000`, 69 by `W=80,000`).
A cell with zero active draws has a perfectly constant, never-zero moment residual across
every single draw — no reweighting of the Ψ-divergence dual problem can bring that to
zero, so the optimal dual variable is genuinely unbounded (not a solver bug). This is a
substantive, previously-undocumented numerical property of applying the CC minimum-
divergence machinery to models with rare extensive-margin participation (the Ricardian/EK
model has no analog: its "argmin" winner-take-all selection is essentially always likely
for at least one competitor). `min_active_draw_count` (`moments.jl`) makes this checkable
before running KNITRO; `melitz_inner_loop_options.opt` raises the shared default's
`maxit=100` to `10000` (did not by itself fix small-`W` divergence — confirming the issue
is genuine dual unboundedness, not merely an iteration-limit shortfall).

**`Δ(θ*)` shrinks with `W`, as expected** (all KNITRO status 0, optimal): `2.10e-4`
(`W=64,000`) → `1.37e-4` (`W=100,000`) → `3.71e-5` (`W=150,000`) → `2.63e-5`
(`W=200,000`).

**Remaining discrepancy:** the moment-residual magnitudes (0.02–0.04, economic units) at
`W=80,000` reflect ordinary Monte Carlo noise on a heavy-tailed Pareto-revenue statistic,
not a construction error — both gravity restrictions and the ACR identity, which don't
depend on the draws at all, are exact to machine precision at the *same* parameter values,
confirming the underlying economics is correct; only the finite-sample moment matching (an
intrinsically stochastic quantity) carries residual noise, and it visibly shrinks with `W`.
