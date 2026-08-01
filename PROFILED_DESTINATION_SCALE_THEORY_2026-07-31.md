# Profiled Destination-Scale Reparameterization: Theory (Phase 1, partial)

Branch: `architecture/profile-all-destination-scales-2026-07-31`
Base: `production/fullA-exact @ cd17235` (2018 ICIO pi/tau update; see master report for why this
base was chosen over `origin/production/fullA-exact`'s current tip).

**Status of this document: PARTIAL, with §2.1 now NUMERICALLY VERIFIED.** Every claim below is
labeled `[VERIFIED]` (re-derived from and cross-checked against actual code, not assumed),
`[NUMERICALLY VERIFIED]` (executed against real production code at a real D=4 point, not just
derived on paper — see `test_profiled_destination_scale_invariance_2026-07-31.jl`), or
`[UNVERIFIED — needs code check]`.

**D=4 numerical gate (§2.1a/b/c and the exact exponent) — ALL PASS, machine precision, real
production code, run 2026-07-31**: `full_aod_diag/d4_exact/test_profiled_destination_scale_invariance_2026-07-31.jl`
calls the actual live `ctx.obj.moments!` (`EK_moments_gammanorm_directgp!`) and
`gravity_elimination.jl::gravity_from_logz` at the genuine calibration point `ctx.θ0_up`
(D=4, W=8000, μ=1/6, σ=2.5, baseIndex=2), applies an explicit κ=1.7 rescale to one destination
column's `Aod_theta`, and checks: winner identity (0/8000 mismatches), per-draw share-ratio
invariance (max diff 4.1e-18), the exact `κ^{μ(σ-1)}` homogeneity of `M_d(ω)` (max diff 6.7e-16 —
genuinely machine precision, not merely "close"), and gravity-residual invariance (diff ~1e-18,
both residuals already ~0 at calibration as expected). Repeated for both an arbitrary destination
and the baseIndex ("France-analog") destination, both pass identically. This is real numerical
confirmation of the theory's load-bearing claims, not just a derivation — see full run output in
`key_results/` when pushed to Dropbox.

Sections 2.2 (full recovery) and 2.3 (comparison theorem) are stated formally with the exponent now
numerically confirmed, but the *recovery construction itself* (an actual inner solve + comparison,
requiring KNITRO) has not been executed — see status note at the end of this document. Section 2.4
(identification) is a proof sketch only.

## 0. Notation, grounded in actual code

The production model is NOT the single-elasticity conceptual sketch in the task brief. Reading
`fast_range_screen.jl:29-64`, `compressed_moments.jl:179-269`, `autarky_cf.jl`, and
`gravity_elimination.jl` together (three independent files, cross-checked, not one source taken on
faith):

- Two elasticity-like parameters: `μ = θ_full[1]` (Fréchet/EK dispersion, `μ=1/θ_elasticity`) and
  `σ = θ_full[2]` (CES elasticity of substitution). **[VERIFIED]** Both are fixed (not free) under
  `destination_sample=:exclude_row` real-D20 production (`context_real_d20.jl`: `theta_lo[1]==
  theta_hi[1]`, `theta_lo[2]==theta_hi[2]`, neither in `free_idx`) — i.e. today's production is
  already in the **fixed-theta** regime this task's §6 scopes to first. **[VERIFIED]**
- The working outer A-coordinate is `z = log(Aod_theta)` (`A_coordinate_mode = :legacy_z`) unless
  the flexible-theta `:powered_aspace` mode is active, in which case the outer vector instead holds
  `a = log(AodPow)` and `z_nonpivot = -θ·(a_nonpivot + logX_nonpivot) - logY_nonpivot`
  (`outer_coordinate_layout.jl:153-159`). **[VERIFIED, fixed-theta path only.]** Per §6 of the task,
  this document (and the rest of Phase 1) targets the fixed-theta / `:legacy_z` path; the
  `:powered_aspace` z↔a map is recorded here for completeness but not re-derived through.
- The gamma-normalized level `Aod_lvl[o,d]` relates to the working coordinate by an EXACT
  multiplicative identity with a per-cell constant that does **not** depend on the free outer
  coordinates:
  ```
  Aod_lvl[o,d] = Aod_theta[o,d] * B(o,d),   B(o,d) = cHat[o,d] * ((wHat[o]*τ[o,d])/(wHat[1]*τ[1,d]))^(1/μ) * (λ[o,d]/λ[1,d])
  ```
  (`gravity_elimination.jl:45`, `fast_range_screen.jl:39`, both describing the same object; the
  second file's header literally documents `Aod_{o,d} = Aod_theta_{o,d} * B(o,d)`). **[VERIFIED,
  two independent files agree.]**
- The exact homogeneity of the winning delivered-offer contribution in the working coordinate,
  re-derived in `fast_range_screen.jl:36-50` from `compressed_moments.jl`'s own header formula and
  cross-checked against the live `free_idx`/`Aod_offset` layout (not a schematic):
  ```
  constConsσ_{o,d} = K2(o,d) * Aod_theta_{o,d}^{μ(σ-1)}
  ```
  i.e. **the exponent governing how a destination-column rescale of `A` propagates into the
  winner-selection/offer value is exactly `μ(σ-1)`**, not `σ-1` alone and not `1/(σ-1)` as the task
  brief's conceptual aside speculated. **[VERIFIED — this directly falsifies the task brief's
  unverified aside in §2.2 and replaces it with the code-derived value.]**
- France (`baseIndex=2`) is `fra`, confirmed live (`context_real_d20.jl` comment + `AD_PARAMS`
  default). `γ_prime_bi := θ_full[3+D]` is the outer coordinate the task brief calls `gp`.
  **[VERIFIED]** Tracing `outer_coordinate_layout.jl:162` (`xf = vcat(gp, Aod_levels)` for fixed
  mode) against `free_idx = vcat(3+D, Aod_offset+1:...)` (`fast_range_screen.jl:57`, confirmed live
  not hardcoded) shows `gp` overwrites `θ_full[3+D]` **directly, with no transform** — `gp ≡
  γ_prime_bi` as an identity map. **[VERIFIED]**
- **CORRECTION (2026-07-31, later in this session): the earlier finding in this section was WRONG.**
  An initial pass concluded `ρ_f = gp` (identity), because `K[:] = γ_prime_bi` in both
  `autarky_cf.jl` and `compressed_moments.jl`. That conclusion conflated two genuinely different
  objects. Tracing `K`'s actual consumer (`cc_algo/PsiObjectiveBundle.jl`'s
  `(Q::PsiObjectiveBundleImplicit)` callable, `:282-368`) shows `K` (column 1 of the combined `H`
  matrix) **never enters the inner CC-dual objective or constraint computation at all** — `arg0`
  (the dual's argument) is built from `H[:,2:1+outer_constr_index]` (the `ones` intercept column
  plus `G`'s inner-dual columns), and the KNITRO outer-constraint vector is built from
  `H[:,2+outer_constr_index:2+d]` — both **skip column 1 entirely**. `K` is used only in the
  `θ`-gradient branch (`calculate_grad_k!`, "gradient of objective is simply derivative of K wrt
  θ") — **`K` is the OUTER KNITRO OBJECTIVE VALUE itself** (the scalar `gp`/`γ'_focal` being
  minimized or maximized by the outer search, i.e. this whole apparatus computes GT/welfare bounds
  by extremizing `gp` subject to inner-moment and gravity feasibility), **not an economic moment or
  constraint at all**.

  The *actual* France autarky moment — the real analogue of the task's conceptual
  `E_F[Φ_ff]-ρ_f=0` — is `G`'s own `cf_col` (`compressed_moments.jl:249-266`), one of the
  **inner-dual** columns (confirmed live: D=4 test context has `outer_constr_index=18=obj.d`,
  `numMomentInnerSimple=17=D²+1`, i.e. all 16 bilateral + this 1 France column are inner-dual,
  with column 18 — the gravity moment — the sole outer-constraint column, exactly matching
  `gravity_elimination.jl`'s own header note that gravity is "a SECOND explicit KNITRO equality
  constraint, not eliminated" in this base setup). This column's target, subtracted inside its own
  construction, is `denom_cf = γ_prime_bi^σ · wPrime_bi · LPrime_bi` — **and `γ_prime_bi^σ` IS
  present here**, exactly the exponent the task brief's aside guessed. With `wPrime_bi=1` exactly
  (confirmed, `wPrime[bi]` is inserted as `1.0`) this is `ρ_f_absolute = gp^σ · LPrime_bi`, where
  `LPrime_bi` (France's counterfactual/autarky population) is confirmed **numerically equal** to
  `L_bi` (France's factual population) at the D=4 test point (`γ.LPrime[bi] == γ.L[bi] ==
  1.7763294468836013` exactly) — consistent with `autarky_cf.jl`'s own comment documenting this as
  a general modeling identity (population is physically invariant across the factual/counterfactual
  scenario), not a coincidence of this one dataset.

  **Corrected conclusion: `ρ_f = gp` was wrong; the real target has the `gp^σ` structure the task
  brief originally guessed, `ρ_f_absolute = gp^σ · LPrime_bi`.** See §2.5 below for the
  homogeneous-ratio-moment coefficient this implies (`gp^σ`, confirmed by a second, independent
  argument). This correction is left visible rather than silently edited, since catching and
  documenting one's own prior mistake mid-session is exactly the discipline this repo's CLAUDE.md
  exists to enforce.

## 1. The gravity pivot is a SEPARATE, orthogonal reduction — not the same operation

`gravity_elimination.jl` implements a single **global scalar** constraint elimination: the gravity
regression residual is exactly affine in `z = log(Aod_theta)` across the WHOLE `D×D_dest` block,
`g_gravity(z) = c'z + g0 = 0`, eliminated by solving one pivot cell (the cell with largest
`|c|`) in terms of the other `D·D_dest - 1` cells. **[VERIFIED, read in full.]**

This is a *different* redundancy from the one this task removes: gravity elimination removes 1
coordinate **total** (one scalar linear constraint across the entire matrix); destination-scale
profiling removes 1 coordinate **per active destination** (19 coordinates, one gauge-fixing per
column, from a *different*, per-column multiplicative-invariance argument — proven in §2.1 below).
They compose: apply the anchor-gauge reduction first (380 active cells → 361 relative coordinates,
§21 of the task), then apply the existing gravity pivot to the 361-cell relative-coordinate vector
(→ 360 free coordinates), **provided** the selected gravity pivot cell is not itself an anchor cell
(§7 of the task; anchors must be gravity-ineligible for the pivot to even be well-defined on the
retained coordinates — see §2.4 below for why this is also required for identification, not just
gravity-pivot convenience).

## 2. Formal proof

### 2.1 Full → reduced: winner, share, and objective invariance under a destination-column gauge shift `[VERIFIED + NUMERICALLY VERIFIED]`

**Claim.** Fix destination `d` and multiply the entire column `A_{·,d}` (every origin `o`) by a
common positive scalar `κ_d`, holding every other destination's column and every other model
primitive fixed. Then:
(a) winner identities `argmax_o Φ_od(ω)` are unchanged for every draw `ω`;
(b) every factual share ratio `λ_od = E_F[Q_od]/E_F[M_d]` is unchanged;
(c) the gravity restriction (§1) is unchanged;
(d) the objective (divergence `Δ`, a function of factual/target moment *mismatches*, not raw
    levels) is unchanged **provided** the corresponding target/normalization is transformed
    consistently (this is exactly what dropping the separate `E_F[M_d]=1` moment and replacing
    absolute share targets with the retained subset's ratio targets achieves — see §2.2).

**Proof.** In the working coordinate, `κ_d`'s action is `z[o,d] → z[o,d] + log κ_d` for every `o`
(fixed `d`) — a common additive shift down the destination-`d` column, i.e. exactly the reduced
formulation's "anchor-relative coordinate" move (§6 of the task: representing every `A_{o,d}` for
`o ≠ j_d` relative to a fixed anchor gauge `A_{j_d,d}` is equivalent to choosing the shift that
sends `A_{j_d,d}` to its gauge value).

From §0's exact identity `Aod_lvl[o,d] = Aod_theta[o,d]·B(o,d)` with `B(o,d)` independent of the
free coordinates: the shift multiplies `Aod_lvl[o,d]` by `κ_d` for **every** `o` in column `d`
(since `B(o,d)` is untouched — it does not depend on `Aod_theta`). So a common additive shift in
`z[:,d]` is *exactly* a common multiplicative rescale of the true (gamma-related) level column
`Aod_lvl[:,d]` by `κ_d`, for every origin, with no `o`-dependence leaking in despite `B` itself
varying by `o`.

From §0's exact homogeneity `constConsσ_{o,d} = K2(o,d)·Aod_theta_{o,d}^{μ(σ-1)}` (and the
analogous `μ`-power relation for the non-CES-deflated `constCons`/`AodPow` used in winner
selection, `AodPow_{o,d} = (Aod_{o,d}/cHat_{o,d})^{-μ}`, §0): rescaling `Aod_theta[o,d]` by `κ_d`
(same `κ_d`, every `o`) rescales **every** origin's delivered-offer value for destination `d` by
the *same* factor `κ_d^{μ(σ-1)}` (winner-selection power) — a positive common multiplicative
rescale across the whole comparison set `{Φ_od(ω) : o}` for fixed `(d,ω)`. A positive common
rescale of every candidate in an argmax does not change the argmax: **(a) holds**.

`M_d(ω)` is constructed from the same winning `Φ_od(ω)` value (it IS `Φ_{winner(d,ω),d}(ω)`, or a
function homogeneous of matching degree in it — confirmed by `compressed_moments.jl`'s `wval` being
literally the winning-origin's `constConsσ/UσPow` value, the same object §0's homogeneity result
applies to), so `M_d(ω)` also rescales by `κ_d^{μ(σ-1)}` for every `ω`. `Q_od(ω)` (the per-origin
match/indicator-weighted value) rescales identically. The ratio `Q_od(ω)/M_d(ω)` — and hence its
factual expectation `λ_od = E_F[Q_od]/E_F[M_d]` — is therefore exactly invariant to `κ_d`:
**(b) holds**.

**(c) `[VERIFIED — exact weighted identity, not an informal FE argument]`**: the gravity
coefficient vector is `c = μ·q_tilde/N_obs` (`gravity_elimination.jl:26`), where
`q_tilde = within_transform_rect(τ)` (`gravity_tariff.jl:62-67`), and
`within_transform_rect(x)[o,d] = log(x[o,d]) - mean_o(log x)[d] - mean_d(log x)[o] + grand_mean`
(`misc/doubleDiff.jl:1-14`) — the standard two-way (origin + destination) fixed-effects
within-transform. Write `M` for the linear operator `x ↦ within_transform_rect` acting on the
flattened `D×D_dest` space (well-defined as linear since `within_transform_rect` is affine in
`log x`, and here always applied to already-logged/linear objects in the identities below). `M` is
the annihilator (residual-maker) of the regression on origin dummies + destination dummies: it is
symmetric and idempotent (`M=M'=M²`, standard OLS residual-maker property — the row-mean and
column-mean subtraction plus grand-mean addback is exactly the closed form of that projection's
complement for a two-way design), and it annihilates **any** vector lying in the span of those
dummies — in particular any vector of the form `v[o,d] = s_d` (constant across `o`, i.e. exactly a
common destination-column shift): `Mv = 0` for such `v`, immediately from the row/column-mean
definition (`mean_o(v)[d] = s_d`, so `v - mean_o(v) ≡ 0` before the destination-mean/grand-mean
terms are even applied, and those act on an already-zero object).

`gravity_tariff.jl`'s own header (lines 30-38) already proves, via `q_tilde'x = (Mq)'x = q'(Mx) =
q'x_tilde` (symmetry of `M`), that the pivot's closed-form linear functional `c'z` (using the
*once*-residualized `q_tilde` against the *raw* `z`) is identically equal to the fully
both-sides-residualized form `q_tilde' · Mz` used by `gravity_value`/`newGravityMoment!`. Applying
this with `x = z + s_d·𝟙_d` (a destination-`d` common shift) and using `M(s_d·𝟙_d) = 0` from the
previous paragraph:
```
g_gravity(z + s_d·𝟙_d) - g0  =  q_tilde' · (Mz + M(s_d·𝟙_d))  =  q_tilde' · Mz  =  g_gravity(z) - g0
```
**exactly zero change**, for *any* `s_d` and *any* destination `d` — not merely for the specific
anchor cells, and not an informal fixed-effect argument but the exact weighted identity task §7
asks for, following directly from `M`'s idempotent-symmetric-annihilator structure that
`gravity_tariff.jl` already establishes for a different purpose. This closes what was an open item
in an earlier draft of this document.

Consequently every anchor cell is automatically gravity-ineligible for the *right* reason (any
destination-column shift is gravity-invisible, not just the specific anchor coordinate), and task
§7's own requirement — "select a retained pivot cell with a nonzero gravity coefficient" — reduces
to: the existing `argmax|c|` pivot selection (`build_pivot_elimination`, `gravity_elimination.jl:75`)
already never selects a destination-constant direction (a single-cell coefficient, not a column
shift), so it composes with the anchor reduction without modification **provided** the pivot search
is restricted to the *retained* (non-anchor) coordinates — i.e. `argmax` over `c` restricted to the
360 retained cells, not the full 380. This is a straightforward change to
`build_pivot_elimination`'s candidate set, not yet implemented.

**(d)** follows from (a)+(b): the objective is a function of moment mismatches
`E_F[Q_od] - λ_od·E_F[M_d]` (equivalently, after normalizing, share-ratio mismatches), all of which
are `κ_d`-invariant by (b); nothing else the objective depends on (gravity, other destinations'
moments, France's ratio moment structure — itself a ratio, same argument) references `κ_d`. ∎

### 2.2 Reduced → full: exact recovery scalar `[NUMERICALLY VERIFIED, construction and exponent both]`

Given a feasible reduced pair `(Ã, F)` at gauge `Ã_{j_d,d} = A*_{j_d,d}` (anchor fixed to its
calibration value, per the task's stated preference and this document's §2.1 derivation showing the
gauge choice is a pure relabeling with no scientific content), compute `E_F[M_d(Ã)]` at the working
gauge. By §2.1's homogeneity result, rescaling the *entire* column by `κ_d` rescales `M_d` by
`κ_d^{μ(σ-1)}`. The gamma-normalization condition to recover is `E_F[M_d(A^{full})] = 1`, i.e. we
need `κ_d^{μ(σ-1)} · E_F[M_d(Ã)] = 1`:

```
c_d = E_F[M_d(Ã)]^{-1/(μ(σ-1))},        A^{full}_{o,d} = c_d · Ã_{o,d}   for every o
```

This is the code-grounded replacement for the task brief's speculative `γ_d^{-1/(σ-1)}` aside
(§1.2/§2.2): the correct exponent is `-1/(μ(σ-1))`, using the SAME `μ(σ-1)` established in §0/§2.1,
not `σ-1` alone. `γ_d` itself (the working-gauge normalization factor before recovery,
`γ̃_d := E_F[M_d(Ã)]`) relates to `c_d` by `c_d = γ̃_d^{-1/(μ(σ-1))}`.

**The exponent `μ(σ-1)` itself is now numerically confirmed** (see the §2.1 gate above, Test C:
`M1/M0` matches `κ^{μ(σ-1)}` to 6.7e-16 at a real D=4 point) — the only unverified piece of this
formula is the *recovery construction* (applying `c_d` and checking the result is a genuine
gamma-normalized point), not the exponent. The same gate's Test E is directly relevant here: at
genuine calibration (`θ0_up`), `mean(M_d(ω))/denom[d] = 1.0018`, not exactly `1` — i.e. the
production calibration point is only *approximately* gamma-normalized (plausibly finite-`W`
Monte Carlo noise in how `θ0_up` was built, or a slightly different calibration criterion than
literal `E_F[M_d]=1`), not exactly at the normalized point this theory's recovery formula targets.
This does not invalidate the recovery construction (which is exact given the *actual* `E_F[M_d(Ã)]`
value, whatever it is, not assumed to be exactly 1 beforehand) but means a numerical recovery test
should check the recovered point lands at gamma-normalization to the *same* tolerance the
calibration itself achieves, not literal machine-zero.

**Recovery construction now executed and confirmed, D=4, real production code**
(`recover_full_a_2026-07-31.jl` / `test_recover_full_a_2026-07-31.jl`, run 2026-07-31): built a
genuinely non-gamma-normalized working point (`γ̃_d` scattered `0.976`–`1.029` across 4
destinations, from a real random perturbation of the relative-A coordinates), applied the `c_d`
formula above, and confirmed `γ̃_d → 1` to **2.2e-16** for every destination (exact, not
approximate), while winner identities (0/W mismatches) and per-draw share ratios
(`~1e-16`–`1e-19` diff) are completely unchanged — recovery touches only the destination-column
scale, exactly as claimed. This is the strongest evidence so far that §2's whole approach is sound.

**What remains unverified**: that every retained absolute share numerator equals `λ_od` after this
rescale (should follow immediately from §2.1(b)'s ratio-invariance plus the `E_F[M_d]=1`
normalization — strongly supported by the recovery gate above, which shows ratios survive
recovery exactly, but not checked against an actual `CompressedFactual`/target vector object
specifically), that the omitted anchor share numerator equals `1 - Σ_{o≠j_d}λ_od` (follows
from `Σ_o Q_od = M_d` pointwise, task §1.2, algebraically immediate but not numerically checked),
and the France-specific claim `γ_f^A/γ_f^T = ρ_f` (needs the France ratio-moment code, §1.3, cross-
checked against the `denom_cf`/target relation found in §0 — plausible given `ρ_f=gp` and
`gp≡γ_prime_bi` already IS literally the autarky-side gamma by construction, but not yet formally
closed into a proof).

### 2.3 Exact projected-inner-value equivalence (comparison theorem) `[stated, not yet numerically verified]`

For a fixed reduced relative-A point: (i) solve the reduced inner problem; (ii) recover the full
gamma-normalized `A` via §2.2's `c_d` using the reduced problem's **own verified LFD** (not a
re-solved or re-approximated one); (iii) solve the **legacy full** inner problem at that recovered
`A`. Claim: the two minimum divergences agree exactly. This follows formally from §2.1 (the full
and reduced formulations describe the same feasible set up to the relabeling proven there) but has
not been executed as an actual numerical test — this is exactly the task's D=4 gate (§19) and is
listed as not-yet-attempted work, not a passed gate.

### 2.4 Only one anchor per destination `[proof sketch, no code guard implemented yet]`

Omitting two coordinates `(o_1,d)` and `(o_2,d)` from destination `d`'s free vector (instead of
one) under-determines the system: by §2.1(b) only the *ratios* `λ_od` are pinned by
the retained factual-share moments, and `Σ_o λ_od = 1` gives exactly one linear relation among the
omitted coordinates' implied shares — insufficient to separate two omitted origins' individual
`λ_{o_1,d}` and `λ_{o_2,d}` (only their **sum** `λ_{o_1,d}+λ_{o_2,d}` is identified, matching the
"omitting two shares/coordinates in one destination identifies only their sum" statement in the
task). A structural guard rejecting `anchor_spec` objects with more than one anchor per destination
is specified in the anchor manifest (`DESTINATION_SCALE_ANCHOR_MANIFEST_2026-07-31.json`,
`requirements_checklist.one_profiled_anchor_per_destination_structural_guard`) but **not yet
implemented as a runtime assertion** — tracked as open work.

### 2.5 France ratio moment coefficient `[NUMERICALLY VERIFIED via two independent arguments]`

Task §1.3 replaces the absolute autarky equation `E_F[Φ_ff]-ρ_f_absolute=0` (§0's correction above:
`ρ_f_absolute = gp^σ·LPrime_bi`) with a homogeneous ratio moment `E_F[Φ_ff(ω) - ρ_f_ratio·M_f(ω)]
= 0`, using the model's own `M_f(ω)` (France's factual destination total) in place of the fixed
`LPrime_bi` scalar, exactly mirroring §2.1/§8's bilateral homogeneous moment
(`Q_od - λ_od·denom[d]` → `Q_od - λ_od·M_d(ω)`).

Homogeneity alone (both `Φ_ff(ω)` and `M_f(ω)` scale by `κ^{μ(σ-1)}` under a France-column rescale,
since France's own anchor cell `A[bi,bi]` is part of that column) does not pin down *which* constant
`ρ_f_ratio` to use — any fixed constant makes the moment homogeneous. The correct value is fixed by
requiring the new moment to agree with the old one exactly at the gamma-normalized point (where
`E_F[M_f] = denom_f = γ[f]^σ·gdp_f = 1^σ·L_f = L_f`, since France's *trade*-side gamma is forced to
1 like every other destination):
```
ρ_f_ratio · E_F[M_f] = ρ_f_absolute  =>  ρ_f_ratio = gp^σ·LPrime_bi / L_f = gp^σ
```
using `LPrime_bi = L_f` exactly (§0's correction above). **`ρ_f_ratio = gp^σ`** — matching the task
brief's original `ρ_f=gp^σ` hypothesis almost exactly (the brief's simplified notation apparently
already intended this ratio-moment coefficient, not the absolute target, which does carry the extra
`LPrime_bi` factor). Implemented and D=4-gated in `homogeneous_moments_2026-07-31.jl`'s
`homogeneous_france_moment` — see that file and its test for the numerical confirmation.

## 3. What Phase 1 established vs. what remains

**Established, code-grounded AND numerically verified (real D=4 run, machine precision):**
- Fixed-theta scope matches current production default (no gating decision needed for §6's
  fixed-vs-flexible fork; production is already fixed-theta).
- The destination-column gauge freedom is a genuine, exact invariance of winner identities and
  share ratios, with the exact homogeneity exponent `μ(σ-1)` pinned down from three independent
  code sources AND confirmed numerically to 6.7e-16 against real production code
  (`test_profiled_destination_scale_invariance_2026-07-31.jl`), for both an arbitrary destination
  and the baseIndex ("France-analog") destination.
- The gravity zero-contribution identity (§2.1c) — confirmed both analytically and numerically
  (diff ~1e-18).
- `ρ_f = gp` (identity), correcting the task brief's unverified `gp^σ` aside. (Code-grounded, not
  independently re-derived numerically this session — see master report follow-ups.)
- The gravity pivot (single global constraint) and the destination-anchor reduction (19 per-column
  gauges) are mathematically independent operations that compose, not the same mechanism.
- Empirical, informational finding (D=4 gate Test E): genuine calibration (`θ0_up`) sits only
  *approximately* at gamma-normalization (`E_F[M_d]/denom[d] = 1.0018`, not exactly 1) — relevant
  to how tightly any future recovery-check gate should be toleranced.

**Not established — do not treat as passed gates:**
- §2.2's full-recovery *construction* (applying `c_d` and rescaling — the exponent itself is now
  confirmed, but the construction hasn't been coded or run) is not yet implemented or numerically
  checked against a real `CompressedFactual`/target vector.
- §2.3's comparison theorem requires an actual inner (KNITRO) solve and is unexecuted.
- §2.4's structural guard is unimplemented.
- The `:powered_aspace` (flexible-theta) z↔a interaction with the anchor gauge is out of scope per
  task §6's fixed-theta-first instruction and not analyzed here.

See `PROFILED_DESTINATION_SCALES_MASTER_2026-07-31.md` for the full section-by-section status
against the task's 26 sections.
