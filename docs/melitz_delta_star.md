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

### 1.4 `N_o` is a genuinely free scale; free entry instead pins `f_entry_o`

An earlier pass at this derivation tried to use free entry to *solve for* `N_o` — this
was wrong, caught two ways in the same session: (a) empirically, a first `equilibrium.jl`
built a D-dimensional NLsolve system on that premise and it would not reliably converge
even after homotopy/continuation; (b) conceptually, cross-checking against Melitz &
Redding (2014, *Handbook of International Economics* ch. 1) shows their closed-economy
free-entry condition `f·J(φ*) = f_E` (their eq. 5) pins the cutoff **independent of the
mass of entrants/market size** — entrant mass is a separate object, determined only via
the aggregate-trade/market-demand equations (their eq. 11), which are linear in `M_E`
given the data and cutoffs.

Redoing the algebra with that lesson: summing the profit expression over `d` and
imposing `Σ_d E_F[operating_profit_od] = w_o·f_entry_o`, and substituting the identity
`C_od·M_od = X_od/N_o` (from solving `(*)` for that product) gives

```
f_entry_o = [ Σ_d C_od·(σ-1)/(σ·(θ*-σ+1))·ẑ_od^(σ-1-θ*) ] / w_o
```

This is a genuine, closed-form, **exact** result for `f_entry_o` **given `C_od` (hence
given `N_o`, already chosen)** — it is not an equation that determines `N_o`; `N_o`
cancels out of the free-entry condition entirely once `C_od` is expressed the "structural"
way (`C_od = expenditure_d·(markup·w_o·τ_od/A_od)^(1-σ)`, no `N_o`), and only reappears
once we *choose* to express `C_od` via the data-based inversion `(X_od/N_o)·(...)` (§1.5)
— at which point `f_entry_o`'s formula literally has an `N_o` in the denominator of every
term, that is **exactly cancelled** by the same `N_o` implicit in `X_od/N_o`. **`N_o` is
therefore a genuinely free scale choice, exactly analogous to the Ricardian repo's `L`**
— confirmed numerically (`test_equilibrium.jl`, this session): re-running the full
construction with `N[target]` changed from `1.5` to `7.3` leaves every downstream
object (`A`, `f`, cutoffs, `GT`, the ACR cross-check) unchanged to machine precision.

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

Given `N_o` (§1.4, freely chosen) and *any* candidate cutoff `ẑ_od ≥ 1` (with `ẑ_od ≥
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

Because `price_power_d ≡ 1` removes `N_o` from firm-level revenue, and because §1.4 shows
`N_o` is a free scale untouched by free entry, the object that aggregate F\*-Melitz trade
actually identifies, once double-differenced, is the composite the addendum derives:

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

### A. D² bilateral trade-flow moments

```
g_trade[o,d](z) = entrant_mass[o]·realized_revenue_od(z_o) − X_data[o,d]
```

for **every** `(o,d)` pair — all `D²` cells, never gated on `baseIndex`/"rest of world".
Scaled by `X_data[o,d]` for conditioning (economic zero set unchanged; unscaled residuals
always also reported, per the brief). Matching all `D²` flows with `expenditure_d = Σ_o
X_od` implies `price_power_d=1` automatically (§1.5's derivation: `price_power_d =
(1/expenditure_d)·Σ_o N_o C_od M_od = (Σ_o X_od)/expenditure_d ≡ 1`) — **no separate
price-index moment is added.**

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
| `entrant_mass[o]` (`N_o`) | D | **genuinely free choice** (§1.4 — not pinned by free entry; verified numerically) | `log N` | `N'_o=N_o` (shared object) | mass of potential entrants | `fake_data.jl` |
| `ẑ[o,d]` | D×D | free computational parameterization (§1.6), not economically identified — **except** `ẑ[target,target]` | — | `ẑ_od≥1`; `ẑ_od≥ẑ_oo` (d≠o); `ẑ[target,target]` **derived** (§1.8) | baseline cutoff | `fake_data.jl` |
| `A[o,d]` | D×D | derived, closed form from `(X,N,w,τ,expenditure,ẑ)` (§1.6), non-identified split (§1.7) | `log A` | `price_power_d≡1` holds automatically given the closure above | efficiency shifter | `equilibrium.jl` |
| `f[o,d]` | D×D | derived, closed form (§1.6) | `log f` | none beyond `ẑ[target,target]`'s own normalization | fixed market-access cost | `equilibrium.jl` |
| `f_entry[o]` | D | derived, closed form from free entry (§1.4) | `log f_entry` | none | entry cost | `equilibrium.jl` |
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
*aggregate* profit across the whole mass of entrants" — a different condition. See §1.4:
`N_o` in fact cancels out of the free-entry condition algebraically (confirmed both
symbolically and numerically) once `C_od` is written the structural way — multiplying by
`N_o` here would reintroduce a dependence that isn't economically there.

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
julia --project=. scripts/run_melitz_delta_star_fake.jl --D 4 --seed 1234 --draws 20000
```

(exact CLI finalized alongside the script; see §14 for the run's actual output once
executed.)

## 14. Numerical result and tolerance

*(filled in after the implementation runs — see the final report for this session.)*
