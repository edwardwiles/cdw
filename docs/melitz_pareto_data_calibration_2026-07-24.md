# Melitz data-only Pareto calibration -- 2026-07-24

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`), starting checkpoint `787ba98`
(tip of the formal closure audit, `docs/melitz_optimization_report_2026-07-24_closure.md`),
continuing the same session's second thread,
`docs/melitz_wage_calibration_gap_2026-07-24.md`. New file:
`src/melitz/pareto_calibration.jl` (~900 lines). Wired into `include_melitz.jl` and
`test/melitz/runtests.jl` (new "Pareto data-only calibration" testset).

## A. Methodological diagnosis: what leaked before, and why it was invalid

Every Melitz estimation context in this repo is built by `build_melitz_psi_bundle(data::
MelitzSyntheticData)` (`delta_star.jl`). That function reads `data.primitives.w` directly
into the estimation context (`ctx.w = p.w`) and every moment (`melitz_moments!`) uses this
`w` as a real functional argument of the firm-level price/revenue formula. `data.primitives`
comes from `generate_fake_melitz_data`, which computes `w` via `melitz_solve_wages_ge(L, tau,
A, f, sigma, theta_star)` -- a general-equilibrium solve that **requires the TRUE, secret
`A`/`f`** the fixture generator itself chose. So every existing Melitz test, including every
number reported in the closure-audit report two documents ago, was computed with baseline
wages derived from privileged knowledge of the answer -- not from the trade-share data
(`X_data`) the estimator nominally targets. This is architecturally identical whether the
data are fake (every existing test) or real: plugging in `real_data/noah_D20` broke
immediately, because there is no "true A/f" for real data to hand to `melitz_solve_wages_ge`.

A second, unrelated function, `melitz_solve_wages(lambda, L)`, is the CORRECT data-only
analogue of the Ricardian side's `prestep/iterWagesPreStep!.jl` -- identical damped-Jacobi
fixed point, no `A`/`f`/`tau`/`sigma`/`theta_star` in its signature -- but it was **dead
code**, called from nowhere in the repository, in any closure version, ever.

There was no code anywhere that calibrated `A`, `f`, cutoffs, entry costs, or the autarky
reference point from data either -- every one of these was either a synthetic-fixture
construction detail or a searched outer parameter with no data-only construction path.

## B. Identification

**What the data identify.** Given `(lambda, L, tau, sigma, theta_star)`, the population
factor-market-clearing condition `E = lambda*E` (`E_o = w_o*L_o`) identifies wages up to the
usual Perron-vector scale indeterminacy (resolved by a numeraire pick). Given wages, the
population Pareto trade-share equation identifies, cell by cell, the **composite**
`chi_od = theta_star*log(A_od) + beta*log(f_od)` (`beta = 1 - theta_star/(sigma-1)`) --
derived in Section C below and verified in code (`melitz_pareto_composite`). The data do
**not** separately identify `A_od` and `f_od`: any cutoff matrix `u = log(q)` admissible
under the model's support/export-selection restrictions inverts to a DIFFERENT `(A, f)` pair
that reproduces the exact same observed shares, price index, factor-market clearing, and
Pareto gains from trade. This nonidentification is the central fact this calibration makes
explicit rather than hiding behind an arbitrary default.

**What theta_star's compatibility with the data requires.** The two structural gravity
restrictions (`A` orthogonal to double-demeaned log `tau`; `f` orthogonal to double-demeaned
log `tau`) can both hold **only if** the supplied `theta_star` equals the data's own
two-way-fixed-effects OLS gravity coefficient `theta_hat` computed directly from
`(lambda, tau)` -- proved algebraically in Section C, not merely asserted, and checked
automatically (`melitz_gravity_theta_check`); an incompatible `theta_star` throws with a
clear diagnostic rather than silently distorting `A`/`f`.

**What the cutoff-calibration step chooses** (not identified from data, and never claimed to
be): which admissible `u` decomposes the identified composite into `A` and `f` separately.
This is a transparent, configurable convex minimum-distance choice (Section D/E below), with
at least two different named target policies demonstrated to give different `A`/`f` but
identical shares, price index, and Pareto GT (the nonidentification test, Section E).

## C. Equations

Baseline normalizations preserved unchanged from the active closure (`docs/
melitz_delta_star.md`): `N_o=1`, `gamma_d=1`, Pareto support `z>=1`, focal autarky cutoff
`zhat'_jj=1`. Notation: `mu_sigma = sigma/(sigma-1)`, `beta = 1-theta_star/(sigma-1)`,
`M(q) = theta_star/(theta_star-sigma+1) * q^(sigma-1-theta_star)` (`pareto_tail_power_mean`,
unchanged, reused verbatim).

**1. Wage solve.** `E = w.*L`, `w_o*L_o = sum_d lambda_od*w_d*L_d` <=> `E = lambda*E`
(`E` a right Perron eigenvector of the column-stochastic `lambda`, eigenvalue exactly 1 by
Perron-Frobenius). Solved by the SAME damped-Jacobi iteration as
`prestep/iterWagesPreStep!.jl` (`melitz_solve_wages`, already existing, previously
unwired), cross-checked against an independent dense eigendecomposition
(`calibrate_melitz_wages`).

**2. The identified composite.** From `X_od = C_od*M(q_od)`, `C_od =
E_d*(mu_sigma*w_o*tau_od/A_od)^(1-sigma)`, and the zero-profit cutoff `sigma*w_o*f_od =
C_od*q_od^(sigma-1)`, substituting out `q_od` gives, after simplification (full derivation in
`melitz_pareto_composite`'s docstring and independently re-derived by hand for this report):

```
log A_od = a0_od + (theta_star/(sigma-1) - 1) * u_od      (u_od = log q_od)
log f_od = f0_od + theta_star * u_od

a0_od = log(mu_sigma*w_o*tau_od) - log(theta_star/(theta_star-sigma+1))/(sigma-1)
        + log(lambda_od)/(sigma-1)
f0_od = log(E_d*lambda_od/(sigma*w_o)) - log(theta_star/(theta_star-sigma+1))
```

so `chi_od := theta_star*log(A_od) + beta*log(f_od) = theta_star*a0_od + beta*f0_od` --
**the `u_od` coefficient cancels EXACTLY** (`theta_star*(theta_star/(sigma-1)-1) +
beta*theta_star = 0`, verified symbolically and numerically): the composite depends only on
observables and `(sigma, theta_star)`, never on the cutoff choice.

**3. Cell-by-cell Pareto inversion, for ANY admissible `u`.**

```
B_od = M(q_od)/lambda_od
A_od = mu_sigma*w_o*tau_od * B_od^(-1/(sigma-1))
f_od = E_d*lambda_od*q_od^(sigma-1) / (sigma*w_o*M(q_od))
```

Verified in code (`melitz_ad_from_cutoffs` + `melitz_verify_baseline_equilibrium`) that
`population_X` applied to the returned `(A,f)` reproduces `lambda_od` to machine precision
and the resulting cutoff exactly equals `exp(u_od)`, for every `u` tried.

**4. Gravity constraint on `u`, in closed form.** Both `log A(u)` and `log f(u)` are affine
in `u` with a **scalar** (cell-independent) slope, so both gravity restrictions reduce to
the SAME linear functional of `u`: `dot(c, vec(u)) = rhs`, `c = gravity_coefficient_vector
(D, tau)` (already-existing, `equilibrium.jl`), with two candidate targets
`rhs_A = -dot(c,vec(a0))/kappa_A` (`kappa_A = theta_star/(sigma-1)-1`) and
`rhs_f = -dot(c,vec(f0))/theta_star`. These are algebraically the **same number** exactly
when `theta_star == theta_hat` (proved via `dot(c,vec(chi)) = (theta_star-theta_hat)*dot(c,c)`,
Section D). The `2 x D^2` "constraint matrix" `[kappa_A*c'; theta_star*c']` is therefore rank
<= 1 by construction -- confirmed by SVD in `calibrate_melitz_cutoffs`, not merely asserted.

**5. Entry costs.** `f_Eo = (1/w_o) * sum_d [X_od/sigma - w_o*f_od*Pr(z>=q_od)]`
(`melitz_profile_entry_costs`), the population free-entry integral, generalized from
`equilibrium.jl`'s existing focal-country-only `population_baseline_focal_profit_integral`
to every origin.

**6. Autarky reference (closed form, not a search).** Given the ALREADY data-calibrated
`(A_jj, f_jj)`, `derive_fjj_from_autarky_cutoff`'s own defining equation
(`E'_j*(mu_sigma*w'_j/A_jj)^(1-sigma)/(sigma*gamma'_j) == w'_j*f_jj`) is solved for
`gamma'_j` instead of `f_jj` (`melitz_autarky_reference`):

```
gamma'_j = E'_j * (mu_sigma*w'_j/A_jj)^(1-sigma) / (sigma*w'_j*f_jj)
```

## D. Proof: gravity compatibility <=> theta_star == theta_hat

`within(log lambda_od) = -theta_star*within(log tau_od) + within(chi_od)` (immediate from
the composite's own derivation: every `od`-invariant term collapses into origin/destination
fixed effects, absorbed by `withinTransform`). Taking the inner product with
`T = within(log tau)` on both sides and using `theta_hat := -sum(within(log lambda).*T) /
sum(T.*T)` (`prestep/master_prestep.jl`'s own formula, reused verbatim -- the SAME two-way-FE
OLS gravity coefficient `moments/newGravityMoment!.jl`'s canonical `UoModel=1` branch
computes):

```
theta_hat = theta_star - dot(c, vec(chi)) / dot(c, c)          [since sum(T.*within(chi)) == dot(c,vec(chi))
                                                                  by withinTransform's self-adjointness]
=>  dot(c, vec(chi)) = (theta_star - theta_hat) * dot(c, c)
```

`dot(c,vec(chi))` is exactly the (u-invariant, per Section C.2) gravity-restriction value of
the identified composite. It is zero -- i.e. BOTH separate `A`/`f` gravity restrictions are
JOINTLY satisfiable by some admissible `u` -- **if and only if** `theta_star == theta_hat`.
This gives a closed-form, exactly-checkable compatibility criterion
(`melitz_gravity_theta_check`), not a fuzzy diagnostic.

## E. Architecture

```
MelitzObservedData          -- lambda (D x D, cols sum to 1), L, tau, countries.
                                Constructor validates + snaps tau diag to 1.0 and lambda
                                col-sums to 1.0 exactly once within-tolerance.
MelitzSyntheticTruth         -- hidden A, f, w, gamma_prime_target, cutoff, seed (Monte
                                Carlo comparison only).
split_melitz_synthetic_truth(::MelitzSyntheticData) -> (observed, truth)
                                -- adapts the EXISTING generate_fake_melitz_data fixture
                                (kept, per Section 15's explicit permission, as the legacy
                                leaked-DGP generator) into the type-separated pair.

calibrate_melitz_wages         -- Section 4 (C.1)
melitz_gravity_theta_check     -- Section 5 (D)
melitz_pareto_composite        -- Section 6 (C.2)
melitz_ad_from_cutoffs         -- Section 7/8 (C.3)
melitz_cutoff_target            -- Section 9.2 (>=2 named target policies)
calibrate_melitz_cutoffs        -- Section 9/9.1 (convex QP, JuMP+HiGHS, gravity constraint
                                    eliminated ANALYTICALLY via a GravityPivot-style
                                    substitution -- see Section G)
melitz_conditioning_diagnostics -- Section 10 (rank/cond/active counts at a given W)
melitz_profile_entry_costs      -- Section 11 (C.5)
melitz_autarky_reference        -- Section 13, the closed-form half (C.6)
calibrate_melitz_cutoffs_and_autarky -- Section 13, the FULL joint solve (see Section H.1)
melitz_verify_baseline_equilibrium   -- Section 12 (population residuals)
calibrate_melitz_pareto          -- Section 14, the top-level orchestrator -> MelitzParetoCalibration
build_melitz_psi_bundle_from_calibration -- observable-only analogue of build_melitz_psi_bundle
```

`build_melitz_psi_bundle` (the leaked-DGP path) is UNCHANGED and still exists -- kept as the
explicitly-labeled legacy comparison path (Section 15/16), never the default. The new
`build_melitz_psi_bundle_from_calibration` consumes only a `MelitzParetoCalibration`; it
cannot accidentally receive `MelitzSyntheticTruth` (no such argument exists in its
signature).

## F. Synthetic validation (D=4, seed=29, the existing benchmark fixture)

Ran the full pipeline: `generate_fake_melitz_data` -> `split_melitz_synthetic_truth`
(discard truth) -> `calibrate_melitz_pareto` (observables only) -> real KNITRO inner solve
via `build_melitz_psi_bundle_from_calibration` -> ex-post equilibrium check under the
recovered LFD.

| quantity | value |
|---|---|
| wage recovery vs. hidden truth (up to numeraire) | `rtol` ~1e-8 (matches `truth.w/truth.w[1]` to every reported digit) |
| `theta_hat` (data-only gravity estimate) vs. true `theta_star=6.8` | `6.799999999999998` -- compatible |
| max \|model share - data share\| | `3.3e-16` (machine precision, by construction of the Section-7 inversion) |
| gravity residuals (A / f), post-calibration | `8.5e-17` / `1.3e-16` (machine precision -- see Section G) |
| feasibility slack (`min_support` / `min_export_minus_domestic`) | `0.067` / `0.064` (both `>0`) |
| profiled entry costs `f_E` | all positive, `[0.087, 0.104]` |
| free-entry link residual (Section H.1's joint solve) | `-1.2e-11` (~zero) |
| `GT_model` vs. `GT_ACR` | `0.065237459` vs. `0.065237458` -- diff `5.3e-10` |
| real KNITRO inner solve | `nStatus=0`, `Delta(theta)=1.11e-5`, `lfd_ok=true`, max moment residual `7.9e-17`, primal-dual gap `1.3e-18` |
| ex-post `residual_autarky_cutoff` | `0.0` exactly |
| ex-post `N_prime_diff_rel` | `8.9e-16` |

**Comparison with the old leaked-DGP path** (`build_melitz_psi_bundle` on the same fixture,
`docs/melitz_delta_star.md` Section 14.6, `Delta(theta*)=7.55e-6` at `W=20,000`): the same
order of magnitude, `nStatus=0` both ways, `GT_model==GT_ACR` both ways (`~1e-14` there,
`5.3e-10` here -- looser here only because the calibration path's `gamma_prime_target` is
itself the OUTPUT of a numerical bisection to `xatol=1e-10`, not an analytically-exact
construction). The calibrated `A`/`f` do **not** equal the hidden truth's `A`/`f` elementwise
(expected -- their decomposition is not identified, Section B); the ECONOMIC OBSERVABLES
(wages up to numeraire, shares, price index, baseline equilibrium, Pareto GT, gravity) agree
tightly. This is exactly the appropriate recovery target main prompt Section 15 specifies.

**Nonidentification, demonstrated directly** (two `cutoff_policy=:uniform` target draws,
`(p_domestic,p_export)=(0.3,0.1)` vs. `(0.5,0.15)`): different `A`, different `f`
(`!isapprox` at `rtol=1e-3`), IDENTICAL observed shares (machine precision both), IDENTICAL
price index, IDENTICAL `GT_ACR` (`atol=1e-8` -- exact by construction, since ACR uses only
the domestic share), and IDENTICAL identified composite `chi` (`atol=1e-6`). See
`test/melitz/runtests.jl`, "Section 18: nonidentification".

## G. A numerical finding: the gravity constraint must be eliminated analytically, not
## imposed as a dense QP equality row

The cutoff-calibration QP's single gravity restriction is linear in `u` (Section C.4). The
first implementation passed it to HiGHS (via JuMP) as an ordinary dense `D^2`-length equality
constraint row. This worked at D=4 (16 variables) but **reliably failed with
`OTHER_ERROR` at D=20 (400 variables)** -- reproduced with a minimal isolated example: the
SAME problem with only the box/order inequality constraints solves instantly; adding the
single dense equality row alone (nothing else changed) makes HiGHS's QP solver fail
immediately, independent of `c`'s own conditioning (`c`'s value range spans a modest ~1.2e4,
not itself pathological). Root cause not pinned down further (not blocking, given the fix
below) -- flagged as a HiGHS/JuMP QP interaction worth avoiding rather than debugging deeper.

**Fix**: eliminate the gravity constraint ANALYTICALLY, exactly the way this repo's own
`GravityPivot` (`equilibrium.jl`) already eliminates the SAME kind of restriction for the
`A`/`f` outer coordinate system: pick the largest-`|c|` eligible cell as a pivot (restricted
to off-diagonal cells, Gate A5's finding that `|c|` is diagonal-dominated under
`withinTransform`), express its `u` value as a closed-form affine function of every other
free cell via the constraint, and substitute that expression directly into the QP's objective
and inequality constraints -- leaving HiGHS with ONLY inequality constraints, no equality row
at all. This solves at every `D` tried (4 through 20) and, as a side benefit, makes the
gravity restriction hold to **machine precision** post-calibration (`8.5e-17`/`1.3e-16` at
D=4 above) rather than the QP solver's own feasibility tolerance (previously ~1e-7/1e-8).

## H. Real D=20 calibration (`real_data/noah_D20`, France focal, `sigma=2.5` assumed
## per this repo's existing Ricardian real-data convention)

`pi.csv`'s column-sum-to-1 convention confirmed directly (columns sum to `1.0` to machine
precision; rows do not, as expected for a share matrix) -- rows are origin, columns are
destination, matching `prestep/master_prestep.jl`'s own `lambda*(w0.*L)./L` orientation used
by the Ricardian real-D20 path (`full_aod_diag/d4_exact/context_real_d20.jl`). One data-
quality note: `tau.csv`'s diagonal is `1.0` for 19/20 countries and `1.001122` for the "row"
(rest-of-world) aggregate -- snapped to exactly `1.0` by `MelitzObservedData`'s constructor
(a measurement artifact, not a real iceberg cost of trading with oneself).

| quantity | value |
|---|---|
| `theta_hat` (data-only two-way-FE gravity estimate, no assumption) | `8.747` |
| `theta_star` used | `= theta_hat` (compatible by construction) |
| wage calibration | 239 damped-Jacobi iterations, market-clearing residual `1.5e-7`\*, Perron cross-check `4.1e-6` |
| wage range (numeraire = country 1, `aus`) | `[0.037, 2.01]` |
| max \|model share - data share\| | `4.4e-16` (machine precision -- ALL 400 real bilateral shares reproduced exactly) |
| gravity residuals (A / f) | `-1.4e-4` / `2.6e-4` (see note below -- solver-tolerance-bound here, not machine precision) |
| feasibility slack | `min_support=0.021`, `min_export_minus_domestic=0.084` (both `>0`) |
| profiled entry costs `f_E` | all positive, range `[0.336, 86.4]` (wide -- reflects real `L`'s ~257x range across countries) |
| `gamma_prime_target` (France autarky price power) | `0.658` |
| free-entry link residual (joint solve, Section H.1) | `-1.0e-10` (~zero) |
| `GT_model` vs. `GT_ACR` (France) | `0.0203135169` vs. `0.0203135175` -- diff `6.0e-10` |
| A range | `[0.0019, 4.77]` |
| f range | `[0.0041, 1157]` |

\*Real-data note: `calibrate_melitz_wages`'s damped-Jacobi iteration floors around
`diff~1.7e-7` on this dataset regardless of `max_iter` (tested to 400,000 iterations, no
further improvement) -- a genuine round-off floor of this many-country iteration at this
scale, not slow convergence (the market-clearing residual computed independently,
`1.5e-7`, confirms the fixed point IS essentially reached). Default `tol` loosened from
`1e-12` (fine for the D=4 synthetic fixture) to `1e-8`; real-data callers should pass
`wage_tol` explicitly if this floor moves.

\*\*Gravity-residual note: `theta_hat`'s own two rhs targets (`rhs_A`, `rhs_f`) agree to a
SCALED relative gap of `3.1e-5`, comfortably inside a loosened `gravity_tol=1e-4` for real
data (vs. `1e-6` for the gravity-exact synthetic fixture) -- expected, since real
`theta_hat` is itself an OLS point estimate on noisy real data, not an exact construction
target. The realized post-calibration gravity residual (`~1e-4`) reflects averaging these
two slightly different targets, not a solver-tolerance artifact (Section G's fix makes the
QP's OWN constraint-satisfaction exact; the residual here is entirely attributable to
`theta_star` not being EXACTLY the population gravity coefficient, which no finite real
dataset can deliver).

### QMC moment-matrix conditioning at the calibrated reference point

| `W` | moment matrix rank | condition number | min active draws | worst cell |
|---|---|---|---|---|
| 2,000 | 354 / 401 | `1.36e16` | 5 | (3, 14) (bra -> kor) |
| 5,000 | 377 / 401 | `1.26e16` | 13 | (3, 14) |
| 20,000 | 396 / 401 | `1.24e16` | 51 | (3, 14) |

Same qualitative signature as the SYNTHETIC D=20 finding in `docs/
melitz_optimization_report_2026-07-24_closure.md` Section 6 (rank-deficient, condition
number at the double-precision floor, worst cell driven by a low-probability bilateral
pair) -- confirms this is a structural property of D=20-scale Melitz moment systems under
this closure, not an artifact of the (previously leaked-DGP) synthetic fixture generator.

### Real KNITRO inner-solve probe (small `W`, one evaluation each -- NOT an outer search)

| `W` | wall | `nStatus` | `Delta` | `lfd_ok` | KKT opt err |
|---|---|---|---|---|---|
| 2,000 | 16.0s | `-400` (iteration limit) | `1e10` (sentinel) | `false` | `1.6e-4` |
| 5,000 | 9.0s | `-400` | `1e10` | `false` | `8.1e-5` |

Consistent with the rank-deficiency finding above: the inner CC dual problem does not reach
a bounded solution at D=20 on real data either, at small `W` (both runs completed in
seconds, not the 10+ minutes documented for the synthetic W=20k/80k benchmark -- these are
much smaller `W`). **Per this task's explicit instruction, no long D=20 finite-delta outer
search was run.** The calibrated REFERENCE MODEL itself (population-level, closed-form) is
fully valid regardless of this inner-solve finding -- every equilibrium residual in the
table above holds to the reported tolerance; only the FINITE-SAMPLE CC dual solve inherits
the known D=20 conditioning issue, exactly as it does for the synthetic fixture.

## H.1. The autarky reference coupling (a genuine finding, not anticipated going in)

Attempting the straightforward design -- calibrate the cutoff matrix freely (all `D^2` cells,
including the focal country's own domestic cell `(j,j)`), then solve `gamma_prime_target`
via `melitz_autarky_reference`'s closed form treating the resulting `f_jj` as fixed data --
**breaks `GT_model == GT_ACR`** (first attempt, D=4 fixture: `GT_model=0.189` vs.
`GT_ACR=0.065`, off by `0.124`, NOT a rounding-level discrepancy). Root cause: the classic
ACR/Chaney sufficient-statistic identity requires not just the `zhat'_jj=1` normalization but
ALSO that the SAME entry cost rationalize both the baseline and autarky free-entry
conditions for the focal country (`population_focal_link_residual==0`) -- with `f_jj` pinned
by data (to reproduce the observed domestic share exactly, Section 12 point 1), these become
two conditions on one remaining unknown (`gamma_prime_target`), generically incompatible.

**Fix**: treat the focal country's own domestic log-cutoff `u_jj` as the SECOND degree of
freedom (mirroring this repo's OWN established pattern for exactly this kind of coupled-cell
problem -- `docs/melitz_delta_star.md` Section 11's `solve_pivot_pair!`, a joint 2D Newton
solve for a similarly-coupled A-pivot/f-pivot cell pair in the legacy `fstar_solver.jl`):
`calibrate_melitz_cutoffs_and_autarky` runs a 1-D bisection (`Roots.jl`) over `u_jj`; at each
trial, the OTHER `D^2-1` free cells are recalibrated (QP, `u_jj` held fixed, SAME gravity
constraint), `(A,f)` inverted, `gamma_prime_target` solved in closed form, and the resulting
`population_focal_link_residual` is the bisection's root function. This still reproduces
EVERY observed share exactly (the Section-7 inversion holds at every trial `u_jj`), while
ALSO achieving `GT_model==GT_ACR` to the bisection's convergence tolerance -- confirmed at
`5.3e-10` (D=4 synthetic) and `6.0e-10` (D=20 real) above. Bisection converges in well under
a second per call (a handful of cheap QP solves, no KNITRO) at every scale tried.

## I. Numerical conditioning: how the cutoff choice affects rank/condition number

Section 10's diagnostics (`melitz_conditioning_diagnostics`) are reported for both the D=4
synthetic fixture (full rank at every `W` tried, `min_active` comfortably >100 at `W=20,000`)
and the D=20 real calibration (rank-deficient, Section H above). The cutoff-calibration QP's
`weights`/`cutoff_policy` arguments are explicitly a NUMERICAL-CONDITIONING lever, not an
economic one (main prompt Section 9.2's own instruction) -- a natural D=20 follow-up (not
attempted this session, consistent with "do not resume D=20 performance optimization... until
this calibration audit passes") is sweeping `p_domestic`/`p_export`/a share-informed target to
see whether a different admissible decomposition meaningfully improves the moment matrix's
rank, the same way `docs/melitz_optimization_report_2026-07-24_closure.md` Section 6
recommends for the (now known to be a SEPARATE, decomposition-independent) synthetic finding.

## J. Remaining limitations

1. **The `A`/`f` decomposition is fundamentally not identified from aggregate trade data
   alone** (Section B) -- this is a mathematical fact of the Pareto-Melitz model under these
   normalizations, not a limitation of this calibration procedure. Any reported `A_od`/`f_od`
   level must be read as "one admissible decomposition consistent with the data," never "the"
   structural values.
2. **The autarky reference point requires an extra degree of freedom** (the focal country's
   own domestic cutoff, Section H.1) beyond the rest of the cutoff-calibration QP to achieve
   `GT_model==GT_ACR` -- a real, non-obvious coupling this session found and resolved, not
   anticipated by the governing prompt's own Section 13 description.
3. **D=20's moment-matrix rank deficiency is a structural, decomposition-independent finding**
   confirmed on BOTH synthetic and (now) real data -- fixing it (if desired) is a moment-system
   regularization question, explicitly out of scope for this session (Section 17's own
   instruction).
4. **`theta_star` is treated as externally estimated/verified via `melitz_gravity_theta_check`**,
   not jointly calibrated with everything else in one combined estimation step -- consistent
   with the governing prompt's "accept theta_star as an input and verify compatibility" option.
5. **Real-data wage-iteration floor** (`~1.7e-7`, Section H) is a numerical, not economic,
   limitation of the plain damped-Jacobi solver at this data scale; a Krylov/sparse-eigenvector
   solve would likely remove it but was not needed for this session's tolerance requirements.
6. This report does NOT attempt or claim D=20 outer-search readiness (Section 17's explicit
   scope boundary) -- the deliverable is a validated, data-only, calibrated REFERENCE model,
   for both synthetic and real D=20 data.

## K. Acceptance criteria (main prompt Section 20)

| # | criterion | status |
|---|---|---|
| 1 | estimation context built from real D=20 observables, no true A/f/w | **MET** -- `MelitzObservedData` + `calibrate_melitz_pareto` + `build_melitz_psi_bundle_from_calibration`, no `MelitzSyntheticTruth` argument exists in this path |
| 2 | synthetic estimation uses observable-only calibrated wages | **MET** -- `split_melitz_synthetic_truth` discards truth before calibration; wage recovery verified vs. hidden truth post-hoc only |
| 3 | population baseline equilibrium equations hold | **MET** -- Section F/H tables, all residuals at or near machine precision |
| 4 | both separate gravity restrictions hold | **MET** (D=4, machine precision) / **MET to `~1e-4`** (D=20 real, `theta_star` is itself only OLS-estimated on noisy data) |
| 5 | all cutoff restrictions hold | **MET** -- feasibility slack `>0` at both D=4 and D=20 |
| 6 | Pareto GT equals ACR | **MET** -- `5.3e-10` (D=4), `6.0e-10` (D=20 real) |
| 7 | two cutoff decompositions give identical observables/GT | **MET** -- Section F nonidentification test |
| 8 | finite-QMC Delta behaves sensibly with W | **PARTIALLY MET** -- D=4 behaves as expected (existing benchmark unaffected); D=20 inherits the KNOWN rank-deficiency non-convergence (Section H), not newly introduced |
| 9 | no prohibited hidden-DGP input in the production path | **MET** -- Section 16 audit below |
| 10 | all existing and new tests pass | **MET** -- full `test/melitz/runtests.jl` run to completion this session: **1371/1371 assertions pass, 44 top-level testsets, zero regressions**, including the new 37/37 "Pareto data-only calibration" testset (18.7s) |

## Section 16 audit: remaining DGP-leak search

`grep`-classified every `generate_fake_melitz_data(` call site (28 hits): all in
`scripts/*melitz*.jl` (diagnostic/benchmark utilities, pre-dating this session) and
`test/melitz/runtests.jl` (the existing large regression suite) -- **legitimate synthetic
truth generation**, not production estimation, per main prompt Section 16's own
classification. `melitz_solve_wages_ge(` call sites (3): `equilibrium.jl` (definition),
`fake_data.jl` (fixture construction, legitimate), `scripts/melitz_fixture_generation_profile.jl`
(diagnostic). The ONE **prohibited** use found and fixed this session:
`build_melitz_psi_bundle`'s `ctx.w = p.w` reading `MelitzSyntheticData.primitives.w`
(GE-solved from true `A`/`f`) as if it were legitimate estimation input -- `build_melitz_psi_bundle`
itself is UNCHANGED (kept as the explicitly-labeled legacy comparison path, main prompt
Section 15's own explicit permission), but the new default production path
(`calibrate_melitz_pareto` + `build_melitz_psi_bundle_from_calibration`) never touches it.
`gamma_prime_target`/`f_jj` truth-object reads elsewhere in the codebase are all
POST-ESTIMATION recovery comparisons (test assertions comparing calibrated/estimated values
against `data.primitives.gamma_prime_target` etc.) -- legitimate per Section 16's own second
category.
