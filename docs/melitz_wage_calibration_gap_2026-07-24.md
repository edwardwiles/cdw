# Melitz wage-calibration gap and real-data investigation -- 2026-07-24 handoff

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`). Continuation of the same
session as `docs/melitz_optimization_report_2026-07-24_closure.md` (the formal closure
audit). This document covers a SEPARATE thread that opened up when the user asked to try
the D=20 diagnosis against real data: it surfaces a real, unrelated-to-D=20 methodological
gap in how Melitz estimation contexts are built, and records the (partial, inconclusive)
real-data investigation itself. **No code changes were made for this thread** -- per the
user's explicit instruction, this is a verify-and-write-up handoff, the fix is deferred to a
new session.

## 1. The finding: no active pre-step calibrates wages (or anything else) from data alone

### What should happen (per the user, and per how the Ricardian side of this repo actually works)

A structural estimator -- whether run on real data or on a fake/simulated dataset used to
test the estimator itself -- should only ever be handed **observables**: bilateral trade
flows/shares, labor endowments, trade costs. Any quantity the model treats as "known" going
into the moment conditions (here: baseline wages `w`) must be independently CALIBRATED from
those observables, never leaked from knowledge of the secret data-generating parameters
(here: the true `A_od`, `f_od` chosen to build a synthetic fixture). This holds even when
testing on fake data -- the whole point of a fake-data recovery test is to verify the
estimator can find the truth FROM OBSERVABLES ALONE; handing it a DGP-derived object defeats
that test silently.

The Ricardian side of this repo does this correctly: `prestep/iterWagesPreStep!.jl` takes
`(w0, L, lambda)` -- `lambda` is the OBSERVED bilateral trade-share matrix (`pi.csv` for
real data) -- and solves a linear damped-Jacobi fixed point `w1 = lambda*(w0.*L)./L` for
wages, using nothing about the model's underlying technology/cost primitives at all (there
are none to know, for Ricardian's own EK closure, that aren't already embedded in `lambda`
itself).

### What Melitz actually has

**Two wage-solving functions exist in `src/melitz/equilibrium.jl`:**

1. **`melitz_solve_wages(lambda::Matrix{Float64}, L::Vector{Float64}; damping=0.6, ...)`**
   (line 32) -- the CORRECT, data-only analogue of `iterWagesPreStep!.jl`. Identical
   `w1 = lambda*(w0.*L)./L` update, identical `damping=0.6`. Takes ONLY a trade-SHARE
   matrix (columns sum to 1) and labor endowments -- no `A`, `f`, `tau`, `sigma`, or
   `theta_star` anywhere in its signature. This is EXACTLY the function the user is asking
   about.

   **This function is called from nowhere in the entire repository.** Confirmed via
   `grep -rn "melitz_solve_wages(" --include="*.jl" .` across `src/`, `scripts/`, `test/`:
   the only two matches are the function's own docstring and definition. It is dead code.
   Its own docstring says it "replac[es] the superseded exact-sample-correction pathway's
   data-driven `melitz_solve_wages`" -- i.e. it was built for an EARLIER closure
   (`docs/melitz_delta_star_v1_superseded_closure.md`-era), and appears to have never been
   reconnected to anything when that closure was superseded by the current
   population-Pareto one -- and, more importantly, **it was never called from the active
   ESTIMATION context builder (`build_melitz_psi_bundle`) in ANY closure version**, not just
   the current one. This is not a recent regression from this session's own work; it
   predates everything touched in the closure audit.

2. **`melitz_solve_wages_ge(L, tau, A0, f, sigma, theta_star; damping=0.1, ...)`** (line
   128) -- the function actually wired into `fake_data.jl`'s
   `generate_fake_melitz_data`/`build_at` (the ONLY fixture-construction path in this repo).
   Requires the TRUE `A`, `f` as direct arguments -- it recomputes trade flows from the
   Melitz-Pareto closed form (`population_X`) at every iterate using the SECRET primitives
   the fixture generator itself chose, and returns the resulting general-equilibrium wage.

### Where the leak enters the estimator

`generate_fake_melitz_data` (`fake_data.jl`) calls `melitz_solve_wages_ge` and stores the
result as `primitives.w` (`r.w` in the code). `build_melitz_psi_bundle`
(`src/melitz/delta_star.jl:391`) then builds the estimation context directly from this:
`ctx = (..., w=p.w, ...)`. This `ctx.w` is NOT decorative -- confirmed directly in
`melitz_moments!` (`src/melitz/moments.jl:68,106`): `firm = melitz_firm(p.w[o], ...)` --
`p.w[o]` (origin `o`'s baseline wage) is a REAL functional argument to the firm-level
price/revenue formula that produces every trade-share moment. Every real KNITRO solve, every
`Delta(theta)` value, every test in the ENTIRE existing test suite (including everything
re-verified in this session's own closure audit) is computed with wages that came from
knowing the true `A`/`f`, not from calibrating against the observed trade-flow data
(`X_data`) the estimator is nominally targeting.

**Consequence**: as currently wired, this codebase cannot honestly claim to test "does the
estimator recover the true parameters from observables alone" -- it has been silently
testing "does the estimator recover the true parameters when also handed one true
GE-equilibrium object computed from those same true parameters." This is architecturally
identical whether the underlying data is fake (as in every existing test) or real (which is
why plugging in `real_data/noah_D20` breaks immediately -- there IS no "true A/f" for real
data to hand to `melitz_solve_wages_ge`, so the whole fixture-construction pipeline has no
path forward at all for real data, independent of anything about D=20 specifically).

### What a fix looks like (NOT attempted this session -- for the next one)

The natural fix: replace (or supplement) `ctx.w=p.w` in `build_melitz_psi_bundle` with a
call to `melitz_solve_wages(lambda, L)`, where `lambda = X_data ./ sum(X_data, dims=1)` (the
OBSERVED trade-share matrix -- already directly computable from `X_data`, which the
estimator already treats as observed data). This should be done for BOTH the fake-data
testing path (to make existing recovery tests methodologically honest) AND as the mechanism
that makes real-data D=20 runs possible at all. This is very likely a bigger change than a
one-line swap:

- `fake_data.jl`'s current construction is intertwined -- `melitz_solve_wages_ge` ALSO
  rescales `A` per-destination to enforce the baseline price-index normalization
  `gamma_d==1` (the `s = (E/colsum(X))^(1/theta_star)` step), which the data-only
  `melitz_solve_wages` does not do (it has no `A` to rescale). Untangling "solve wages from
  data" from "enforce gamma_d==1 in the SYNTHETIC fixture's own true primitives" needs
  care -- these are two genuinely different jobs that happen to be fused in the current GE
  solve.
- `melitz_solve_wages` has a genuine wage-SCALE indeterminacy (normalizes `w[1]=1` --
  documented explicitly in `melitz_solve_wages_ge`'s own docstring: "unlike
  `melitz_solve_wages`... this GE solve ties absolute wage levels to the REAL primitives...
  there is NO free wage-scale normalization here"). Whether/how a numeraire choice interacts
  with the rest of the active closure (gravity pivots, the focal link moment, `gamma_prime_target`)
  needs to be worked through, not assumed.
- Whether `melitz_reduce_theta`'s use of the TRUE `(A,f,gamma_prime_target)` to seed
  `theta0` (the outer search's starting/reference point in every existing test) is ALSO a
  problem worth flagging in the same pass: using the true parameter as an optimizer start
  point is a much more standard and defensible Monte-Carlo-study convention than leaking a
  computed equilibrium OBJECT into the moment conditions themselves, but it does mean no
  existing test in this repo has ever exercised a genuine "search FROM an uninformed start"
  recovery test either. Worth a decision, not necessarily a fix, in the next session.

**This is a real, load-bearing methodological gap, independent of the D=20 rank-deficiency
finding in the closure-audit report.** It should likely be prioritized ABOVE any further
D=20 conditioning work, since D=20's own diagnosis (Section 6 of the closure report) was
itself produced entirely on fixtures built through this same leaking pipeline -- not wrong
(the rank-deficiency is a real, mechanical property of the realized `A`/`f`/`tau` regardless
of how `w` was obtained), but any FOLLOW-UP work that tries to fix D=20 by retuning the
fixture generator should be aware the generator itself has this separate issue.

## 2. The real-data investigation (partial, inconclusive -- for context)

Triggered by: "try D=20 using real data... the Ricardian repo should point you to where to
find it... wondering if the fake data generation process... is junk."

### Real D=20 data located

`real_data/noah_D20/{countries,L,pi,tau}.csv` -- 20 real countries (aus, fra, bra, can, che,
chn, deu, esp, gbr, idn, ind, ita, jpn, kor, mex, nld, rus, tur, usa, row), already used by
the RICARDIAN full-A_od real-data path (`full_aod_diag/d4_exact/context_real_d20.jl`,
France/`fra` focal, `fakeData=3` in `setup/importData.jl`). `pi.csv` is the real bilateral
trade-SHARE matrix; `L.csv` real labor endowments (4.9M to 1.26B, ~257x range); `tau.csv`
real iceberg trade costs (1.0 to 1.83, vs. the synthetic generator's tuned 1.05-1.14 band).

### Attempt 1: substitute real tau/L into `generate_fake_melitz_data`, keep A/f synthetic

Blocked by the wage-calibration gap above: with real `L`'s extreme heterogeneity,
`melitz_solve_wages_ge`'s damped-Jacobi map (tuned/proven only against the synthetic
generator's mild 1.3x `L` range) converges extremely slowly -- oscillates in a 1e-9 to
1e-13 band rather than cleanly crossing even a loosened tolerance, across every `max_iter`/
`tol` setting tried (20,000/1e-13, 60,000/1e-9, 100,000/soft-non-throwing-fallback). **This
was chasing the wrong problem** -- per Section 1 above, `melitz_solve_wages_ge` should not
be part of this pipeline at all when real (or honestly-treated fake) trade-share data is
available; `melitz_solve_wages` (data-only, `damping=0.6`, matching Ricardian's own proven-
robust linear iteration) is the function that should have been used, and would very likely
not have hit this convergence wall (Ricardian's own real-D20 run, `context_real_d20.jl`,
uses exactly this style of linear share-based wage solve against the SAME real data
successfully in production). Not independently re-verified this session (ran out of time
once the deeper gap was found) -- flagged as the first thing to try once the pre-step is
wired in.

### Direct mechanism test: does real tau avoid the within-origin cutoff-clustering hypothesis?

Bypassing the wage-solve entirely, a no-solving-required direct comparison (both from raw
`tau` data, D=20, no GE, no KNITRO): does real `tau_od` avoid the "narrow synthetic range
packs 19 destinations too close together, causing near-duplicate cutoffs and collinear
moment columns" mechanism hypothesized as the likely driver of the closure report's
rank-362/401 finding?

| | Synthetic (fake_data.jl, seed=29) | Real (`tau.csv`) |
|---|---|---|
| within-origin spread (max-min)/mean | 0.071 | 0.154 (**2.16x more**) |
| within-origin coefficient of variation | 0.023 | 0.041 (**1.77x more**) |
| destination-pairs within 1% of each other | 22.2% | **31.4% (MORE, not fewer)** |

**Mixed/inconclusive result, reported honestly rather than smoothed over.** Real tau IS
more dispersed on average (driven by a few extreme outliers -- max within-origin spread
0.76 for real vs. 0.08 for synthetic), but the BULK of the real distribution is more tightly
clustered than the synthetic generator's uniform draws (a classic long-tailed real-economic-
geography pattern: many similar/nearby countries share near-identical trade costs to a given
destination, then a few outliers). By the SPECIFIC near-duplicate-pair proxy for the
collinearity mechanism, real data is not obviously better and may be mildly worse. This does
NOT settle whether "fake_data.jl is junk" -- it shows the one isolated, cheaply-testable
piece of the hypothesis (raw tau dispersion) is genuinely mixed, and a real answer requires
an actual real-data fixture (moment matrix rank, computed the same way as the closure
report's Section 6), which needs the wage-calibration gap fixed first.

### Bottom line for the next session

1. **Fix the wage-calibration gap first** (Section 1) -- wire `melitz_solve_wages` (or a
   clearly-designed replacement) into `build_melitz_psi_bundle`, sourced from `X_data`, for
   both the fake-data testing path and as the enabling step for real data.
2. **Then** revisit real-data D=20: with a working data-only wage calibration, building a
   real-tau/L (or fully real-data, if `A_od`/`f_od` calibration from `pi.csv` is also
   tackled) D=20 fixture should not hit the convergence wall this session did, and the
   moment-matrix rank/conditioning test (identical to the closure report's Section 6) can
   actually be run against it.
3. **A_od/f_od themselves have no real-data analogue at all** in this codebase currently --
   they are Melitz-specific structural unobservables. Using real data for D=20 in any full
   sense (not just tau/L) will eventually need either a genuine calibration procedure
   (invert `A`,`f` from real trade flows given `sigma`/`theta_star`) or an explicit decision
   to keep them synthetic/calibrated-to-match-real-shares as a deliberate, documented choice
   -- not resolved here, flagged for scoping in the next session.
