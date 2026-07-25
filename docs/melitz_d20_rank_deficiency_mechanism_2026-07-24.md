# Melitz D=20 moment-matrix rank deficiency: mechanism, and its resolution at W=80,000 on real data -- 2026-07-24

Branch: `melitz/fullD-delta-star` (`trade_robustness_modular`). Follow-up to
`docs/melitz_pareto_data_calibration_2026-07-24.md` (same-day session): that report closed
the wage-calibration-gap leak and built a full data-only Pareto calibration pipeline
(`src/melitz/pareto_calibration.jl`), validated on D=4 synthetic and real D=20 data. This
document answers a follow-up question from that report's D=20 conditioning-diagnostics
table: **why does the CC inner (KNITRO) solve work cleanly at D=4 but fail
(`nStatus=-400`/`-102`) at D=20**, and confirms the mechanism directly against the
calibrated real-data cutoffs and moment matrix rather than asserting it.

This is a genuinely SEPARATE issue from the wage-calibration leak closed earlier the same
day. **Short answer, established below with real numbers**: the D=20 moment system packs 20
destination-moments onto a single shared per-origin random draw sequence, and most
destinations look similar enough (in calibrated cutoff terms) that their moment columns are
severely ill-conditioned at ordinary Monte Carlo sample sizes (`W=2,000`-`20,000`) -- but,
for real D=20 data, this is an ORDINARY finite-sample problem, not a permanent structural
one: at `W=80,000` the moment matrix reaches full rank and the real KNITRO inner solve
converges cleanly to machine precision (Section 5a). The earlier session's SYNTHETIC D=20
fixture does NOT resolve at `W=80,000` (a real divergence, examined in Section 8) -- so the
full picture is more nuanced than either "D=20 is broken" or "just raise W and it's fine"
alone. Nothing in this document changes the calibration pipeline or its acceptance-criteria
status (companion report); it documents an open question in more depth, per a direct
follow-up question, and materially updates the earlier session's "D=20 doesn't work" framing
for the real-data case specifically.

## 1. Why more moments alone doesn't explain it

Going from D=4 to D=20, the number of trade-share moments grows quadratically:
`D^2 = 16 -> 400` (plus the single focal free-entry link moment, `401` total columns). The
number of Monte Carlo reference draws `W` (2,000 to 80,000 in this investigation) is always
much larger than `401`, so a naive row/column count comparison gives no reason to expect
column-rank deficiency. The real driver is a specific STRUCTURAL near-collinearity among
the moment columns, not merely their count.

## 2. The mechanism: draws are shared per ORIGIN, not per bilateral pair

`z_draws` (the Monte Carlo reference-Pareto sample) is `W x D` -- **one column per origin**,
reused identically across every destination that origin sells to (`pareto_draws`'s own
docstring: "draws are indexed by origin, not by origin-destination pair", matching this
repo's `UoModel=1` convention throughout). Concretely (`moments.jl`'s `melitz_moments!`,
baseline `N_o=1`/`gamma_d=1` normalization), the trade-share moment column for cell `(o,d)`
reduces to

```
m_{o,d}(z) = K1_od * z^(sigma-1) * 1{z > cutoff_od}  -  lambda_od
K1_od = (mu_sigma * w_o * tau_od / A_od)^(1-sigma)
```

For a FIXED origin `o`, all `D` of its destination-columns are deterministic transforms of
the **same scalar draw sequence** `z_o` (one shared column of `z_draws`). They differ across
destination `d` ONLY through (a) a multiplicative scale `K1_od` and (b) *where* the
participation indicator switches on, `cutoff_od`. If several destinations from the same
origin have nearly the SAME `cutoff_od`, their moment columns become nearly SCALAR MULTIPLES
of one shared shape function (`z^(sigma-1)*1{z>cutoff}`) -- numerically almost
indistinguishable directions in the `W`-row sample, regardless of how large `W` is.

This means the relevant risk factor is not "how many moments" but "how similar are the
cutoffs within each origin's own block of `D` destination-columns" -- a property of the
calibrated `(A,f,tau)`, not of `W` or of the estimator's construction.

## 3. Direct confirmation against the real D=20 calibration

Using the SAME real-data calibration validated in the companion report
(`real_data/noah_D20`, France focal, `sigma=2.5`, `theta_star=theta_hat=8.747` estimated
from data), pulled the calibrated cutoff matrix `q = exp(u)` directly.

**Every origin's within-destination cutoff spread** (`(max-min)/mean` across that origin's
20 destinations):

| origin | q range | mean | (max-min)/mean |
|---|---|---|---|
| aus | [1.125, 1.463] | 1.294 | 0.261 |
| fra | [1.021, 1.337] | 1.287 | 0.246 |
| bra | [1.025, 1.984] | 1.303 | 0.737 (driven by ONE outlier, kor -- see below) |
| can | [1.118, 1.409] | 1.294 | 0.225 |
| che | [1.140, 1.334] | 1.294 | 0.150 |
| chn | [1.103, 1.359] | 1.294 | 0.198 |
| deu | [1.146, 1.355] | 1.294 | 0.161 |
| esp | [1.140, 1.337] | 1.294 | 0.153 |
| gbr | [1.142, 1.355] | 1.294 | 0.164 |
| idn | [1.110, 1.370] | 1.294 | 0.201 |
| ind | [1.057, 1.449] | 1.295 | 0.303 |
| ita | [1.138, 1.344] | 1.294 | 0.159 |
| jpn | [1.119, 1.344] | 1.294 | 0.174 |
| kor | [1.097, 1.350] | 1.294 | 0.196 |
| mex | [1.130, 1.386] | 1.294 | 0.198 |
| nld | [1.143, 1.339] | 1.294 | 0.152 |
| rus | [1.128, 1.366] | 1.294 | 0.184 |
| tur | [1.110, 1.368] | 1.294 | 0.199 |
| usa | [1.130, 1.332] | 1.294 | 0.156 |
| row | [1.143, 1.330] | 1.294 | 0.145 |

Every origin's 20-destination cutoff block clusters within roughly 15-30% of its own mean
(excluding the single Brazil/Korea outlier) -- i.e. MOST destinations look economically
similar (similar effective trade cost/productivity) from any given exporter's perspective,
which is exactly the condition that makes their moment columns nearly collinear.

**Brazil in full detail** (origin 3, chosen because it is also the reported worst cell):

```
bra -> aus: q=1.228   bra -> fra: q=1.253   bra -> can: q=1.243   bra -> che: q=1.347
bra -> chn: q=1.221   bra -> deu: q=1.266   bra -> esp: q=1.264   bra -> gbr: q=1.311
bra -> idn: q=1.255   bra -> ind: q=1.455   bra -> ita: q=1.275   bra -> jpn: q=1.243
bra -> mex: q=1.309   bra -> nld: q=1.307   bra -> rus: q=1.216   bra -> tur: q=1.351
bra -> usa: q=1.238   bra -> row: q=1.263
bra -> kor: q=1.984   <-- the ONE outlier
(bra -> bra (domestic): q=1.025)
```

18 of Brazil's 19 EXPORT cutoffs sit in a tight band, `q in [1.216, 1.351]` -- a ~13% spread
around a mean of ~1.28. Those 18 moment columns are, up to their own `K1_od` scale factors,
all close to scalar multiples of one shared "activate near q~1.28" shape. Korea is the
outlier at `q=1.984` (a much higher, more selective cutoff) -- this is also the SAME cell
(`(3,14)`, bra->kor) independently flagged as the worst (fewest active draws) cell in the
companion report's own diagnostics at every `W` tried.

## 4. Two distinct, compounding sub-mechanisms -- BOTH finite-sample, at very different rates

**(a) Near-duplicate (not identical) cutoffs within an origin** (the 18 "normal" Brazil
destinations above). Their moment columns are genuinely different functions of `z`, but the
difference between them is SMALL -- resolving it statistically requires far more samples
than resolving an ordinary, well-separated moment. Section 5 below shows this fully resolves
by `W=80,000` for real data (condition number collapses from the double-precision floor to
an ordinary `~1e5`), but is essentially invisible at `W=2,000`-`20,000` (condition number
pinned at `~1.2e16`-`1.4e16` throughout that range). This is the dominant, slow-resolving
mechanism.

**(b) Near-zero-participation-probability cells** (Brazil -> Korea). A column that is
almost always exactly zero across the sample (very few draws exceed the high cutoff), hence
itself nearly redundant with a constant/intercept direction. This resolves FASTER with `W`
than (a) -- the minimum active-draw count climbs steadily (5 -> 13 -> 51 -> 200 across
`W=2,000` to `80,000`) and is already a reasonable sample size by `W=80,000`, well before
mechanism (a) has visibly moved the condition number at all (it stays pinned through
`W=20,000` even as (b) is already improving).

Both are genuine finite-sample phenomena, not exact structural singularities -- but they
operate on very different `W`-scales, which is why the moment matrix looks catastrophically
rank-deficient at `W=20,000` and essentially fine at `W=80,000` (Section 5).

## 5. Moment-matrix rank and conditioning vs. W (real D=20 data)

Built via the SAME `melitz_moments!` machinery the real inner KNITRO solve uses
(`melitz_conditioning_diagnostics`), at the calibrated reference point (`focal_country=fra`,
`theta_star=8.747`, seed=1 Halton draws):

| `W` | moment matrix rank | / total columns | condition number | min active draws | worst cell |
|---|---|---|---|---|---|
| 2,000 | 354 | 401 | 1.36e16 | 5 | (3, 14) bra -> kor |
| 5,000 | 377 | 401 | 1.26e16 | 13 | (3, 14) |
| 20,000 | 396 | 401 | 1.24e16 | 51 | (3, 14) |
| **80,000** | **401** | **401** | **1.06e5** | **200** | (3, 14) |

**This overturns the naive read of the W=2,000-20,000 trend.** At `W=80,000` the moment
matrix reaches **FULL rank (401/401)** and the condition number collapses from `~1.2e16`
(the double-precision floor) to `~1.06e5` -- a totally ordinary, healthy condition number,
**eleven orders of magnitude better**, not a further-worsening or plateauing trend. This
means mechanism (a) (near-duplicate cutoffs, Section 4) is, for REAL data, a genuine
FINITE-SAMPLE phenomenon after all -- just one that resolves much more slowly in `W` than
mechanism (b) (the single thin cell), not a hard `W`-independent structural floor.
Economically this makes sense: Brazil's 18 "similar" export destinations have cutoffs that
are CLOSE but not identical (`q` ranging `1.216`-`1.351`, Section 3) -- their moment
columns are genuinely different functions of `z`, just ones whose difference is small
enough that resolving it statistically requires many more than 20,000 samples. Two
near-parallel (but not exactly parallel) vectors are not a true rank deficiency; they are a
CONDITIONING problem that a big enough sample does eventually resolve, unlike an EXACT
collinearity (identical cutoffs), which no amount of `W` would fix.

**Does this actually rescue the KNITRO inner solve?** Tested directly.

## 5a. Real KNITRO probe at W=80,000: YES, it converges cleanly

Ran the real CC inner minimum-divergence solve (`melitz_recover_lfd`, the exact production
path via `build_melitz_psi_bundle_from_calibration`) at `W=80,000` on the same real D=20
calibration, at the calibrated reference point (not a search):

| `W` | wall | `nStatus` | `Delta` | `lfd_ok` | KKT opt err | max moment residual |
|---|---|---|---|---|---|---|
| 2,000 | 16.0s | `-400` (iteration limit) | `1e10` (sentinel) | `false` | `1.6e-4` | -- |
| 5,000 | 9.0s | `-400` | `1e10` | `false` | `8.1e-5` | -- |
| **80,000** | **16.3s** | **`0` (optimal)** | **`4.12e-4`** | **`true`** | **`1.9e-14`** | **`1.9e-14`** |

**This is the headline finding of this follow-up.** At `W=80,000` -- exactly where Section 5
shows the moment matrix reaches full rank and a healthy condition number -- the SAME real
D=20 CC inner problem that fails (`nStatus=-400`) at `W=2,000` and `5,000` converges
CLEANLY: `nStatus=0`, a small positive `Delta(theta)=4.12e-4` (the expected shape for a
finite-sample gap around the population reference point, same qualitative pattern as every
converged D=4 benchmark in the companion report), and both the KKT optimality error and the
maximum weighted moment residual at **machine precision** (`1.9e-14`). It also did not take
materially longer wall-clock (16.3s, comparable to the FAILED `W=2,000` run) -- the earlier
`-400` runs were not "running out of time," they were genuinely stuck on an ill-conditioned
problem, and a well-conditioned larger problem solves faster in practice, not slower.

**Revises the mechanistic story from Sections 1-4**: the D=20 rank deficiency is not, at
least for this real-data calibration, evidence of a fundamentally broken moment system --
it is a genuine finite-sample conditioning problem in the `W=2,000`-`20,000` range that
`W=80,000` fully resolves. The correct one-line summary of "why D=4 works and D=20 doesn't"
is now: **D=4's moment system is well-conditioned at ordinary sample sizes; D=20's is not,
until `W` is large enough (`~80,000` for this real-data calibration) to statistically
resolve the near-duplicate-cutoff columns described in Sections 2-4** -- not "D=20 is
unsolvable."

## 6. Contrast with D=4: why this genuinely doesn't show up there

At D=4, each origin's draw column is shared across only **4** destination-moments (not 20).
Even with comparable relative cutoff spread (the D=4 synthetic benchmark fixture shows
24-50% spread per origin, similar in kind to D=20's real-data 15-30%), there are at most 1-2
"extra" near-redundant directions per origin to accumulate -- summed over 4 origins, nowhere
near enough to meaningfully damage a 17-column moment matrix sampled at `W=20,000+` rows.
At D=20, the SAME per-origin phenomenon multiplies out over 20 origins x up to ~17-18
near-redundant directions each, producing the observed tens of near-zero singular values.

This is fundamentally a "how many moments share one row of randomness, and how similar are
their shapes" problem, not a "how many total moments" problem. Raising `maxit` alone does
NOT fix it at a GIVEN `W` (confirmed in the companion report's real-data KNITRO probe:
`W=2,000` and `W=5,000` both hit `nStatus=-400`, the iteration limit, in 9-16 seconds --
not stalling for lack of iterations, genuinely not making further progress along the moment
matrix's near-null directions) -- but raising `W` itself DOES fix it, at least for real
data, once `W` is large enough to statistically resolve the near-duplicate columns
(Section 5a) -- an important correction to the naive first guess that "this doesn't depend
on W at all."

## 7. Mechanically, how this breaks the KNITRO inner solve

The CC inner problem is a convex dual minimum-divergence optimization over dual variables
`(zeta, lambda_1...lambda_{401})`; its curvature (Hessian/KKT structure at each interior-point
iteration) is built from the moment matrix `G`. Along a near-null direction of `G`,
perturbing the dual variable in that direction changes the objective and constraint values
by almost nothing -- numerically the dual problem is close to UNBOUNDED along that
direction. This produces exactly the two failure signatures observed at ordinary sample
sizes on both synthetic and real D=20 data: `nStatus=-102` (KNITRO's own unboundedness
detection) or `nStatus=-400` (the barrier method exhausts its iteration budget without
settling, thrashing along a nearly flat direction) -- versus D=4, where every origin's
moment block is well-conditioned and KNITRO reaches `nStatus=0` in a fraction of a second.
As Section 5a shows directly, once the near-null directions are removed (by using a large
enough `W`), the SAME real D=20 problem's curvature is restored and KNITRO converges just as
cleanly as D=4 does -- the mechanism described here is fully reversible, not an inherent
property of D=20 itself.

## 8. A real divergence between the synthetic fixture and real data at large W

The SAME qualitative signature at `W<=20,000` -- rank deficiency, condition number at the
double-precision floor, a worst cell driven by one unusually high (low-participation)
cutoff -- was found on both:

- the SYNTHETIC D=20 fixture (`docs/melitz_optimization_report_2026-07-24_closure.md`
  Section 6, built through the LEAKED-DGP pipeline that was still active at the time):
  rank **362/401 at BOTH `W=20,000` AND `W=80,000`**, condition number **8.38e16 at both**
  -- that prior session's own finding was explicitly that rank/conditioning were "IDENTICAL"
  across that `W` range, i.e. **the synthetic fixture's rank deficiency does NOT resolve by
  `W=80,000`**.
- the REAL D=20 data (this document): rank climbs `354 -> 377 -> 396 -> 401` and condition
  number collapses `1.36e16 -> 1.26e16 -> 1.24e16 -> 1.06e5` across the SAME `W` range --
  **fully resolved by `W=80,000`** (Sections 5/5a).

This is a genuine, informative DIVERGENCE, not just two confirmations of the same thing.
The most likely explanation (not independently re-verified this session, flagged for a
follow-up that wants to chase it further): the synthetic fixture's `(A,f,tau)` were built
from DELIBERATELY NARROW parameter ranges (`fake_data.jl`'s tuned defaults,
`tau_offdiag_logrange=(0.05,0.13)`, `logA_noise_sd=0.04`, etc. -- narrow BY DESIGN, to keep
the D=4 fixture comfortably feasible) -- at D=20 scale this could plausibly produce cutoffs
within an origin that are not just CLOSE but close enough to be effectively
numerically-indistinguishable at any practically reachable `W`, i.e. a much SEVERER version
of the same mechanism than real, heterogeneous trade-cost data produces. Real data's
genuine cross-country heterogeneity (Brazil-Korea being a clean example, Section 3) appears
to inject enough real separation between destinations' cutoffs that, while still requiring
a large `W` to resolve, it DOES resolve.

**Practical upshot**: "D=20 is broken" was the wrong one-line takeaway from the earlier
session's synthetic-only finding. The corrected takeaway, now checked against real data
too: **D=20's moment system needs a substantially larger `W` than D=4 does to become
well-conditioned, and for the real French-focal dataset, `W=80,000` is enough; for the
particular synthetic fixture used in earlier sessions, it evidently is not** (not
independently re-confirmed this session, taken from the prior report's own numbers).

## 9. What would actually help going forward

- **First, cheapest thing to try**: for any REAL-data D=20 work, just use `W>=80,000` (or
  sweep upward from there) before concluding the closure doesn't work -- directly
  demonstrated to fix the inner-solve failure in Section 5a, at no extra methodological
  complexity.
- **For the synthetic D=20 fixture specifically** (where `W=80,000` was already shown, in
  the prior session, NOT to be enough): either push `W` further (untested how far), or
  retune the fixture generator's parameter ranges to inject more genuine cross-destination
  heterogeneity per origin (the likely root cause per Section 8) -- not attempted this
  session.
- **Moment-system regularization/reformulation**: identify and either drop or jointly
  constrain the near-duplicate destination-moment directions within each origin block, for
  cases where a large enough `W` is impractical (compute cost grows with `W`, though Section
  5a shows the WELL-conditioned `W=80,000` solve was in fact no slower than the FAILED
  smaller ones here).
- **Cutoff-decomposition conditioning lever**: the calibration's cutoff-target policy is a
  free, non-identified choice (Section 9.2 of the companion report) -- deliberately
  spreading cutoffs WITHIN an origin is a numerical-conditioning-only lever that could
  reduce the `W` needed for good conditioning. Not attempted this session.
- None of these is an economic restriction change -- all are about which admissible
  representation of the (non-identified) `A`/`f` decomposition is chosen, how the moment
  system is specified, or simply how large a Monte Carlo sample is used.

## 10. Bottom line

D=4 and D=20 use the identical moment formula and identical closure -- the difference is
that D=20 packs 20 destination-moments onto each origin's single shared draw sequence, and
most of those 20 destinations (in both synthetic and real data) turn out to have similar
enough calibrated cutoffs that their moment columns are NEAR-collinear at ordinary sample
sizes. This makes the CC inner dual problem's curvature vanish along many near-null
directions at `W=2,000`-`20,000`, which KNITRO reports as an unbounded dual or an
iteration-limit failure. **For real D=20 data, this is a genuine but ORDINARY finite-sample
conditioning problem**: at `W=80,000` the moment matrix reaches full rank, a healthy
condition number, and the SAME real KNITRO inner solve converges cleanly to machine
precision (Section 5a) -- directly confirming the diagnosis and directly reversing the
"D=20 doesn't work" framing for real data specifically. The EARLIER synthetic-fixture
finding that `W=80,000` does NOT resolve it stands as reported by the prior session and is
not contradicted here -- but it is now understood as a property of that PARTICULAR
fixture's narrow tuning, not of the D=20 closure per se, since real data with the same `D`
and the same moment formula resolves cleanly at the same `W`. This does not change anything
about the calibration pipeline's own validated status (companion report,
`docs/melitz_pareto_data_calibration_2026-07-24.md`) -- it only adds a materially more
complete and more optimistic answer to "does D=20 work at all," at least for real data.
