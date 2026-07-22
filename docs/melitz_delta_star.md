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

### 1.4 Free entry pins `N_o` in closed form — independent of the `A_od`/`f_od` split

Summing the profit expression over `d` and imposing `Σ_d E_F[operating_profit_od] =
w_o·f_entry_o` (the free-entry condition, §1.6), and using `(*)` to substitute
`C_od·ẑ_od^(σ-1-θ*) = X_od·(θ*-σ+1)/(N_o·θ*)` (an *identity*, obtained by solving `(*)` for
that product — it does **not** depend on `ẑ_od` separately), gives, after cancellation:

```
N_o = (σ-1) / (θ*·σ·w_o·f_entry_o) · Σ_d X_od
```

This is a genuine, closed-form, **exact** result: `N_o` depends only on total exports of
origin `o`, `f_entry_o`, `w_o`, `σ`, `θ*` — not on how origin `o`'s exports split across
destinations, not on `A_od`/`f_od` individually. It is the reason §1.6's decomposition
result (that aggregate Melitz trade under Pareto identifies only a composite of `A`/`f`)
does not "leak" into `N_o`: **the entrant mass is fully identified even though the
bilateral productivity/fixed-cost split is not.**

### 1.5 Profiling the D² trade-flow equations analytically (the addendum's parameterization)

Given `N_o` (§1.4) and *any* candidate cutoff `ẑ_od ≥ 1` (with `ẑ_od ≥ ẑ_oo` for `d≠o`,
per the export-selection restriction), `(*)` can be solved for `C_od` directly:

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

### 1.6 Non-identification of `A_od`/`f_od` individually (why this must not be oversold)

Because `price_power_d ≡ 1` removes `N_o` from firm-level revenue, and because §1.4 shows
`N_o` is *separately* pinned by free entry, the object that aggregate F\*-Melitz trade
actually identifies, once double-differenced, is the composite the addendum derives:

```
r = θ*·a + β·h,     a = ΔΔ log A,  h = ΔΔ log f,  β = 1 - θ*/(σ-1)
```

not `a` and `h` separately. §1.5's cutoff parameterization is one convenient *computational*
way to pick a point in the affine set of decompositions consistent with `r`; it is not a
claim that `ẑ_od` (or the implied `A_od`/`f_od` split) is identified by the data. The
acceptance tests (§13) check the identified composite and the two orthogonality
restrictions, never elementwise recovery of `A`/`f`.

### 1.7 Autarky: `f[target,target]` is derived, not chosen

The brief requires `ẑ'_{target,target} = 1` (the autarky cutoff sits exactly at the Pareto
lower support). At `ẑ=1`, `Pr(active)=1` for every firm, so both the autarky operating-
profit expectation and the zero-profit-at-cutoff condition (`C'_{tt}·1^(σ-1) = σ·w'_t·f_tt`,
`w'_t=1` by the counterfactual numeraire) can be combined directly (full derivation
verified symbolically in this session, cross-checked against the legacy file's — otherwise
rejected — `Melitz_transform_θ` formula, which reduces to the same expression once
isolated from its surrounding hybrid logic):

```
f[target,target] = f_entry[target] · (θ* - σ + 1) / (σ - 1)
```

This is a genuine **normalization-by-construction**: `f[target,target]` is *derived* from
`f_entry[target]` (not independently drawn/projected like the other `D²-1` cells of `f`),
documented here exactly because the brief requires every normalization to state why it is
necessary. It does not conflict with the gravity restriction on `f` because the projection
step (§10) treats `f[target,target]` as a fixed constant and solves the free cells' raw
values so the *overall* `⟨T, ΔΔ log f⟩ = 0` restriction holds including that fixed cell's
contribution.

### 1.8 Baseline general equilibrium: one non-degenerate NLsolve system

Unlike `N_o`, the baseline `expenditure_d` cannot be solved for in closed form ahead of
`A_od`/`f_od`, because `ẑ_od` (hence `C_od` and the price-power aggregate) depends on
`expenditure_d`, which we also want to satisfy `expenditure_d = Σ_o X_od` (so that
`price_power_d ≡ 1`, §4) and `Σ_d E_F[operating_profit_od(z)] = w_o f_entry_o` (free
entry) simultaneously. Given `A`, `f`, `τ`, `w`, `f_entry`, `σ`, `θ*` fixed, the baseline
equilibrium is the joint fixed point in `(N_o, expenditure_d)_{o,d=1}^D` (2D unknowns, 2D
equations: D price-power-normalization residuals + D free-entry residuals), solved with
**`NLsolve.nlsolve`** (canned solver, matching the user's stated preference — no hand-rolled
Newton code) using the analytical Pareto tail expectations of §1.3 (no Monte Carlo needed
for the equilibrium solve itself — matching the brief's preference for analytical
expectations in the F\* pre-step). This mirrors the Ricardian repo's own
`iterWagesPreStep!`/`iterWagesTheory!` fixed-point pattern (§11) — a small, well-posed
system solved with a canned iterative method — rather than a bespoke closed-form chain.

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
| `w[o]` | D | fixed (baseline), solved (autarky wage of non-target n/a — only target reweighted) | `log w` | `w[target]=1` (baseline numeraire) | wage | `fake_data.jl` |
| `w'[target]` | scalar | fixed | — | `w'[target]=1` (counterfactual numeraire) | autarky wage | `equilibrium.jl` |
| `τ[o,d]` | D×D | fixed (data) | — | diag = 1 | trade cost | `fake_data.jl` |
| `A[o,d]` | D×D | constructed (gravity-projected), non-identified split (§1.6) | `log A` | `price_power_d≡1` via `A_od` level (§1.2); autarky uses the *same* `A[target,target]` | efficiency shifter | `fake_data.jl` |
| `f[o,d]` | D×D | constructed (gravity-projected) except `f[target,target]` | `log f` | `f[target,target]` **derived** (§1.7), not projected | fixed market-access cost | `fake_data.jl` |
| `f_entry[o]` | D | fixed/calibrated | `log f_entry` | none | entry cost | `fake_data.jl` |
| `entrant_mass[o]` (`N_o`) | D | derived, closed form (§1.4) | — | `N'_o=N_o` (shared object) | mass of potential entrants | `equilibrium.jl` |
| `expenditure[d]` | D | derived, NLsolve joint w/ `N` (§1.8) | — | `expenditure_d = Σ_o X_od` | destination spend | `equilibrium.jl` |
| `price_power[d]` | D | derived diagnostic | — | `≡1` by construction (§1.2) | CES price-power aggregate | `equilibrium.jl` |
| `price_power'[target]` | scalar | derived (autarky GE) | — | none (endogenous) | counterfactual price-power | `equilibrium.jl` |
| `ẑ[o,d]` | D×D | free computational parameterization (§1.5), not economically identified | — | `ẑ_od≥1`; `ẑ_od≥ẑ_oo` (d≠o) | baseline cutoff | `fake_data.jl` |
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
*aggregate* profit across the whole mass of entrants" — a different (and, given `N_o` is
otherwise determined by §1.4's closed form, circular) condition. See §1.4 for why this
matters mechanically: multiplying by `N_o` here would make `N_o` cancel out of its own
determining equation.

---

## 8. Baseline cutoffs and autarky cutoff normalization

Baseline cutoffs `ẑ[o,d]` are the free computational parameterization of §1.5 (chosen
subject to `ẑ≥1`, `ẑ_od≥ẑ_oo`). The autarky cutoff `ẑ'[target,target]=1` is imposed by
**deriving** `f[target,target]` from `f_entry[target]` (§1.7's closed form) rather than by
an overwrite of a previously-computed value — the cell is simply never included in the
gravity-projection step that produces the other `D²-1` cells of `f` (§10).

---

## 9. `N'_o = N_o`

The equilibrium/moment code takes a single `entrant_mass::Vector` and passes the same
vector to both the baseline and counterfactual firm/moment evaluators — there is no
`NPrime` object anywhere in the Melitz module (the legacy code's `NumberActiveFirmsPrime`
pattern, which computes a separate value and is later force-equated to `NumberActiveFirms`
via an `impose_M_Mprime_equality` flag, is explicitly rejected).

---

## 10. Full-D gravity restrictions (construction)

`fake_data.jl` generates raw `log A`, `log f` with heterogeneous cell-specific noise plus
origin/destination fixed effects (which vanish under `doubleDiff`, so don't affect the
restriction and are free to use for e.g. enforcing sensible cutoff rankings). It then
projects the *free* cells' double-differenced component off `doubleDiff(log τ)`
(Gram-Schmidt: subtract the component along `doubleDiff(logτ)` from `doubleDiff(logA_raw)`,
similarly for `f`), with `f[target,target]` held fixed at its §1.7 value throughout the
projection (the projection coefficient is solved treating that one cell's contribution to
`⟨doubleDiff(τ), doubloDiff(f)⟩` as a constant offset, so the *other* free cells absorb the
correction). Both restrictions are verified numerically to machine precision as part of
fixture construction, and `std(log A) > 0`, `std(log f) > 0`, `std(doubleDiff(log A)) > 0`,
`std(doubleDiff(log f)) > 0` are asserted (never trivially-all-ones matrices).

---

## 11. Ricardian routines reused / adapted

**Reused as-is:** `misc/doubleDiff.jl`'s `doubleDiff` (§2, §10); the general "seed once,
draw once" discipline of `prepare_cc/drawU.jl`/`genRands.jl` (adapted to Pareto inverse-CDF
instead of Exp(1)+Fréchet transform, §3); the `NLsolve`-based fixed-point pattern of
`prestep/iterWagesPreStep!.jl`/`setup/iterWagesTheory!.jl` (§1.8 — same "small system,
canned solver" shape, new equations); the CC minimum-divergence inner loop
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
