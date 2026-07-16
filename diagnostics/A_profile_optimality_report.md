# Is A_od* profile-optimal in the unrestricted full-A CC program?

**Scope note.** This is the "decisive core" subset of the full 9-diagnostic
spec (Diagnostics 1, 8, 4 only), chosen with the user because the codebase
architecture already pointed to a specific, testable mechanism before any
numerics were run. The remaining diagnostics (2, 3, 5, 6, 7, 9 — feasible
tangent-space construction beyond the single gravity constraint, reduced
Hessian/Lanczos, multistart, full sequential-linearization iteration audit)
were not built; see "What this doesn't establish" below for exactly what
would be needed to extend this to the full spec.

## Bottom line

**Evidence that moving A_od away from A\* lowers the profiled divergence
δ\*(A,γ'), and evidence that the outer derivative is wrong because it omits
the winner-boundary (Dirac) term.** These two findings are causally linked,
not independent: the outer derivative's blindness to winner-switching is the
*mechanism* by which the solver fails to find the profitable moves that a
fully re-solved search does find. This is **not** "A\* is genuinely optimal" —
across 3 independent diagnostics (an exact-Dual AD/analytical comparison, an
independent finite-difference/winner-recount check, and a fully re-solved,
no-linearization directional search), the same conclusion recurs.

## Architecture recap (full detail was reported to the user separately; summarized here for the record)

Repo: `trade_robustness_modular_perf` (branch `feature/sequential-inversion-perf`).
The "unrestricted Ricardian version" = `full_aod_diag/run_fullA_D10_production.jl`
+ `PsiObjectiveBundleImplicitMethodBFullA`, with `EK_moments_gammanorm_directgp!`
(`full_aod_diag/moments_gammanorm.jl`) — the γ_d≡1-for-all-d gauge, ALL D²
`A_od` entries free (no `A[1,d]=1` pin), direct-γ'_focal objective. This is a
genuinely different, more restrictive method than `sequential_gravity/`'s
focal-only/profiled approach (which is not touched here).

The production outer gradient for the divergence-budget constraint
(`full_aod_diag/ad_benchmark/derivative_core.jl::envelope_scalar_div_ctx`, and
`PsiObjectiveBundleImplicitMethodB_fullA.jl::_methodB_fullA_envelope_scalar`)
is an envelope-theorem `ForwardDiff.gradient` over `Aod_theta`. It re-solves
the trade-flow moments fresh at every perturbed θ (winners are NOT frozen
across iterations), **but** the winner indicator itself
(`misc/smoothMinIndNew!.jl::MinInd!`, `xInd[i] = (x[i] > xMin) ? 0 : 1`) is a
hard `Bool` branch on the *value* of a `ForwardDiff.Dual`, which structurally
strips all partial-derivative information through that branch. The codebase's
own analytic-Jacobian file (`moments/hFunction_jacobian.jl`) adds a
`SmoothDirac` boundary term for exactly this reason, with its own comment
noting the ForwardDiff/a.e. derivative "omits the indicator's boundary term"
— but that file is architecturally unreachable from the full-A production
driver (`use_Jacobian=0` is hardcoded everywhere in `full_aod_diag/`).

## Economic setup and the coordinate mapping (worked out before coding)

Per the user's setup, `b_od = (A_od/(w_o τ_od))^(σ-1)`, `α_od = log(b_od)`,
`Φ_od = b_od X_o`, `g_od = Φ_od·1{o=argmax}`. The code's actual free outer
parameter is `Aod_theta[o,d]` (θ block, `full_aod_diag/moments_gammanorm.jl`),
related to `A_od` (structural, `= 1/AodPow`) via `Aod_theta → Aod → AodPow`,
an exact elementwise chain (fixed μ) giving

```
d log(A_od[o,d]) / d log(Aod_theta[o,d]) = μ         (exact identity, any point)
d α_od / d log(Aod_theta[o,d])           = μ(σ-1) = 1/β     (β = θ*/(σ-1) = (1/μ)/(σ-1))
```

so the analytical Jacobian, in the code's actual coordinates, is the user's
formula rescaled by `1/β`:

```
∂E[g_od]/∂log(Aod_theta[j,d]) = λ_od + c·λ_od²        (j=o),   c=(1-β)/β
                               = c·λ_od·λ_jd           (j≠o)
```

**Two implementation-specific unit corrections were required** (both found by
running the diagnostic against the actual production code, not derivable from
the abstract setup alone — recorded here so a future session doesn't
re-discover them the hard way):

1. **The Fréchet benchmark's `Aod_theta` is NOT all-ones** under the γ_d≡1
   gauge. `build_theta_gammanorm` rescales each destination column by
   `s_d = γ0_d^(-σ/(μ(σ-1)))`, which is 1 only if the *old*-gauge `γ0_d`
   happened to be 1. At D=5 here, `s_d ∈ {1.85, 0.46, 0.38, 0.71, 0.76}` —
   materially different from 1. Using `ones(D²)` as "A\*" gives nonsense
   (E_F[G] off by 0.6+ instead of ~1% MC noise). Fixed by reading `Aod_theta`
   directly off the prestep-derived `θ0_up`.
2. **Production's own `G` is gamma-normalized**: `EK_moments_gammanorm_directgp!`
   divides all raw moments by `gamma(μ(1-σ)+1)` before returning. This factor
   is invisible in the user's abstract economic-setup formula (stated in raw
   `E[U^(σ-1)]` units) but is baked into every object the AD/FD Jacobians
   actually differentiate. Missing it produced a suspiciously *uniform* ~18%
   gap across all 25 diagonal entries (ratio 0.81–0.83, tightly clustered);
   `1/gamma(μ(1-σ)+1) = 0.8160` for this D=5 economy's (μ,σ) — matches to
   <2%, i.e. within Monte Carlo noise. Fixed by dividing the analytical
   formula through by this same constant.
3. (Caught but not a "finding": an early `ad_jacobian_full` differentiated
   w.r.t. the raw *level* of `Aod_theta` instead of its `log`, silently
   introducing an extra `Aod_theta[col]` rescaling per column since the
   benchmark level isn't 1 — see point 1. Fixed by reparametrizing the AD
   call as `Aod0 .* exp.(logdelta)`, evaluated at `logdelta=0`, matching the
   FD convention exactly.)

## Diagnostic 1 (+ decomposition): analytical vs AD vs FD moment Jacobian at A\*

D=5, W=8000, θ*=6, σ=2.5, μ=1/6, **β=4** (`diagnostics/check_A_profile_optimality.jl`,
`diagnostics/jacobian_checks.jl`). Sanity checks at A\*: raw-G zero-moment
check max|E_F[G_od]|=1.9e-2, Fréchet identity max|E_F[share_od]−λ_od|=1.0e-2
— both consistent with W=8000 Monte Carlo noise (~1/√W ≈ 1.1%).

**The decisive comparison** decomposes the full analytical Jacobian into a
"smooth" part (λ_od/(β·gamma_norm), the exact tautological quantity AD *can*
see — a mathematical near-identity: `d(price^σ·1{o wins})/dα_od =
1{o wins}·d(price^σ)/dα_od` since the indicator's own partial is exactly
zero) and a "boundary" remainder (everything else):

| Comparison | diag max rel err | off-diag max abs err |
|---|---|---|
| AD vs smooth-only prediction | **8.8%** (≈ MC noise) | **0.0 exactly** |
| AD vs FULL analytical (incl. boundary) | 75% | 9.0e-2 (=100% missing) |

Every one of the 25 off-diagonal AD entries checked (D=5, all `(o,j,d)` with
`j≠o`) is **exactly 0.0**, while the analytical prediction ranges
−0.003 to −0.08. AD's diagonal entries match the smooth-only prediction to
within Monte Carlo noise and disagree with the full (boundary-inclusive)
analytical value by up to 75%. This is not "AD is imprecise" — it is AD
exactly reproducing the piece of the derivative that survives differentiating
through a hard-thresholded `Bool` branch, and exactly missing the piece that
doesn't.

The raw finite-difference-vs-analytical comparison (perturbing `Aod_theta`
directly, no Duals, `h∈{1e-2,...,1e-6}`) is **not** clean at this W: errors
are non-monotonic in h and largest at the smallest h — a real finite-sample
effect, not a bug, confirmed and explained by Diagnostic 8 below.

## Diagnostic 8-lite: winner-switch fraction vs h

Focused on the `(j,d)` pair with the single largest analytical off-diagonal
magnitude (D=5: `Aod_theta[j=4,d=4]`, off-diagonal row `o=2`,
analytical value **−0.0902**; analytical diagonal **0.2673**):

| h | frac. draws switching winner | FD diag | FD off-diag |
|---|---|---|---|
| 0.1 | 2.4% | 0.2647 | −0.0881 |
| 0.03 | 0.71% | 0.2601 | −0.0880 |
| 0.01 | 0.26% | 0.2470 | −0.0790 |
| 0.003 | 0.09% | 0.2254 | −0.0602 |
| 0.001 | 0.025% | 0.1964 | −0.0406 |
| 0.0003 | 0.0125% (1 draw) | 0.2436 | 0.0000 |
| 0.0001 | 0.0125% (1 draw) | 0.4946 | 0.0000 |

At `h=0.1` (the largest step, i.e. the *most* draws switch), FD matches the
analytical value to **1–2%**. As h shrinks, the number of switching draws in
the finite W=8000 sample drops toward 0/1, and the FD estimate visibly
degrades/becomes erratic — at h≤3e-4 exactly zero draws switch in this
particular sample, so FD reports exactly 0 for the off-diagonal entry, missing
the boundary contribution entirely at that step size. This is precisely the
finite-W numerical-precision floor the user's prompt anticipated ("the
fraction tending to zero does not imply the aggregate contribution divided by
h vanishes") — confirmed directly: the aggregate contribution does **not**
vanish, it is well-approximated at moderate h and is under-sampled (not
mis-specified) at very small h.

## Diagnostic 4-lite: fully re-solved directional δ\* test at A\*

For each of 3 `γ'_focal` targets (Fréchet point + one toward each bound),
built a separate `PsiObjectiveBundleDelta`-style D²+1-moment "exact δ\*" bundle
(no gravity column, to avoid a column-collision bug — see
`diagnostics/directional_resolve.jl` header), fully re-solved (no
linearization, `feastol/opttol=1e-12`) at `A(h)=A\*⊙exp(h·v)` for the
production envelope-gradient direction and 3 random directions, all projected
to be **exactly** gravity-feasible (gravity is exactly linear in
`log(Aod_theta)`, so projecting out `vec(q_tilde)` keeps `R(A(h))=0` exactly
for every h, not just to first order — no linearization approximation here
either). One cold start per direction cross-checked against the warm-started
chain; results were bit-identical.

**Result: at all 3 γ' targets, multiple independent directions lower the
re-solved δ\* below the A\* baseline**, by up to **1.2–2.2%** (not
solver-tolerance noise — the inner solve tolerance is 1e-12, four orders of
magnitude tighter than the observed effect):

| γ'_focal target | baseline δ\* | best found | Δ | direction |
|---|---|---|---|---|
| 0.8931 | 0.168268 | 0.166148 | −1.26% | competitor_swap, h=−0.03 |
| 0.9227 (Fréchet) | 0.001959 | 0.001917 | −2.17% | random1, h=−0.003 |
| 0.9343 | 0.023317 | 0.023035 | −1.21% | competitor_swap, h=+0.03 |

Descent was found in **multiple, independently-drawn random directions** at
every target (not only the direction hand-picked to target the largest
off-diagonal Jacobian entry) — this is not a single lucky direction. A\*
behaves like a saddle in the profiled objective: some signs/directions raise
δ\*, others reliably lower it; the production envelope gradient (which by
construction cannot see the boundary term) reports a direction/magnitude
inconsistent with this realized behavior at several points (e.g. at the
Fréchet target, `‖g_prod_proj‖=0.149` yet the envelope-predicted slope along
the actually-descending directions has the wrong sign in some cases —
consistent with the derivative being missing a term, not merely imprecise).

## Classification

**Evidence that moving A lowers divergence, but the current method misses
it — AND evidence that the outer derivative is wrong because winner-boundary
effects are omitted.** Both hold simultaneously and are mechanistically
linked: Diagnostic 1 shows exactly which term is missing and why (a
structural property of differentiating through `MinInd!`'s `Bool` branch);
Diagnostic 8 independently confirms (via full re-derivation of winners, no
AD at all) that this missing term is real and non-negligible; Diagnostic 4
confirms it matters in the object that's actually optimized (the profiled
δ\*), via a fully re-solved search with no linearization anywhere.

This directly explains the user's original observation: an outer solver whose
gradient is provably blind to the one class of directions where the actual
profitable moves live will report "no reason to move away from A\*" — which
looks identical to "A\* is optimal" unless you fully re-solve off-axis, as
Diagnostic 4 does.

## What this doesn't establish (honest bounds on the "decisive core" scope)

- **Not shown**: how *large* the achievable gain is at realistic scale
  (D=10/20, W≥80000) or over the *full* feasible tangent space (only 4
  directions × 3 γ' targets were tried at D=5; no systematic search, no
  reduced Hessian, no multistart). The ~1-2% δ\* reductions found here are a
  lower bound on the true gain, not an estimate of it.
- **Not built**: Diagnostic 2's general tangent-space machinery (only needed
  here in its simplest form — a single linear gravity constraint — since no
  other A_od pins exist under the current gauge; a richer normalization would
  need the fuller construction), Diagnostic 7's reduced Hessian/Lanczos
  (would show whether A\* is a saddle or a max along the descending
  directions — plausible given the mixed-sign pattern above, not verified),
  Diagnostic 6's systematic multistart, Diagnostic 9's sequential-linearization
  iteration audit (this diagnostic suite only tested the full-A method, not
  `sequential_gravity/`'s separate profiled/focal approach).
- **Not resolved**: a corrected outer-derivative implementation (e.g. wiring
  `hFunction_jacobian.jl`'s existing `SmoothDirac` boundary term into the
  full-A production path, or a likelihood-ratio/boundary-integral estimator)
  was explicitly out of scope per "do not immediately redesign the
  algorithm" — this report identifies the mechanism and quantifies that it
  matters, but does not implement a fix.

## Files

- `diagnostics/context.jl` — shared setup (D-configurable full-A gammanorm
  context, Fréchet A\* baseline, share-recovery helper, winner detector).
- `diagnostics/jacobian_checks.jl` — Diagnostic 1 (analytical/AD/FD/smooth-only
  Jacobians, error reports) + Diagnostic 8 (winner-switch report).
- `diagnostics/directional_resolve.jl` — Diagnostic 4 (exact δ\* bundle,
  envelope gradient, gravity-feasible projection, directional re-solve).
- `diagnostics/check_A_profile_optimality.jl` — driver; run via
  `DVAL=5 WVAL=8000 julia --project=. diagnostics/check_A_profile_optimality.jl`
  (KNITRO env required, see `.knitro_env.sh`).
- `diagnostics/diag1_jacobian_5.csv`, `diagnostics/diag8_winnerswitch_5.csv`,
  `diagnostics/diag4_directional_5.csv` — raw numerical output.
