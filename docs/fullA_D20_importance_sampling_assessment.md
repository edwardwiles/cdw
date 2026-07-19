# Continuation 10, Section 8: importance sampling — paper-only assessment + tiny diagnostic (NOT integrated)

Branch `c10-qmc-is` (worktree `gravity-fullA-d4-c10-qmc-is`, forked from
`diag/fullA-d4-exact` @ `690b8f5`), machine `demand.mit.edu`. This document is
explicitly a **separate, higher-risk investigation from Section 7's QMC
work** (`fullA_D20_qmc_investigation_report.md`) — the two are not blended
here or in the final summary. Per the task brief: assessment only. **No
importance-sampling code is integrated into the real D=20 pipeline, and none
was run against it.** The only code artifact is a small, standalone toy
diagnostic (`full_aod_diag/d4_exact/c10_phase8_is_toy_diagnostic.jl`,
D=6-origin synthetic economy, no KNITRO, ~3 seconds to run) built solely to
probe importance-sampling *mechanics* (effective sample size, weight-tail
behavior, a toy Hessian-conditioning proxy) — not to produce any number that
could be mistaken for a real kappa/Delta result.

## 0. Headline verdict

**Importance sampling (IS) is not a drop-in extension of this codebase's
Monte Carlo/QMC draw generation — it requires real, non-trivial rework of the
CC dual/primal machinery, and the existing code's only precedent for a
per-draw weight vector is (a) narrower than what IS needs and (b) contains an
actual unfixed bug in one of its two call sites (§2.3 below).** QMC
(Section 7) is the lower-risk path forward: it changes only the *source* of
the uniform draws feeding the model's own inverse-CDF transform, preserving
F* as the target distribution exactly, with zero changes to the dual/primal
formulas. IS changes the *target* draws come from (a proposal Q ≠ F*), which
touches the objective, gradient, Hessian, and primal-weight-recovery formulas
simultaneously. **IS is deferred pending independent mathematical
derivation/validation of the modified formulas in §3 below** — this document
identifies exactly what would need to be derived and re-validated, but does
not do that derivation to production-ready rigor, and does not implement it.

## 1. What Christensen–Connault (2023) say about this

Section 3.1 ("Computation") of the published paper (`papers/Christensen
Connault 2023 ECMA.pdf`) states their own numerical approach directly:

> "The expectations in the objective functions (13), (14), and (16) are
> available in closed form for certain settings... Otherwise, the
> expectations will need to be computed numerically... In the empirical
> applications, we used a randomized quasi-Monte Carlo approach based on
> scrambled Halton sequences as in Owen (2017)."

This is the exact citation `cc_algo/rhalton.jl`'s header references ("Adapted
from R code of Art B. Owen... for the r program rhalton") — i.e., this
repository's existing (if previously unused-for-this-purpose) Halton
generator is literally an implementation of the same method CC's own paper
used. This is strong independent support for Section 7's QMC direction, and
by contrast, **the paper's own "Practical Considerations" section says
nothing about importance sampling, weighted quadrature, or nonuniform
draws** — every discussion of "computing the expectations... numerically" in
CC (2023) assumes draws are generated so as to directly target F* (Monte
Carlo or QMC via F*'s own inverse-CDF), never a proposal distribution Q ≠ F*
requiring a likelihood-ratio correction. The dual criterion functions
themselves, equations (11)–(14) (p.273), are written as expectations under a
GENERIC F* (`K^δ(θ;γ,P) = sup_{η,ζ,λ} {-η E_{F*}[φ*((k+ζ+λ'g)/(-η))] - ηδ -
ζ - λ12'P}`, and the dual analogue for the sup/inf pair) — nothing in the
paper's formal apparatus rules out F* being approximated by a weighted
empirical measure, but nothing in it discusses how either, since their own
implementation never needed to.

## 2. Are nonuniform base quadrature weights already supported in the CC code?

**This is the single most important fact-finding item, and the answer is:
partially and narrowly — not in the way IS would need, and with one
confirmed dead/buggy code path.**

### 2.1 The dual objective's own empirical average is HARDCODED uniform (1/W)

Read directly (not inferred) from `cc_algo/PsiObjectiveBundle.jl`, both the
`Explicit` and `Implicit` bundle types (the latter is what the real D=20
economy uses, confirmed via `context_real_d20.jl`'s
`PsiObjectiveBundleImplicit` construction):

```julia
# PsiObjectiveBundleExplicit callable, line ~65:
f = η * (sum(arg1) / M + δ) + ζ

# PsiObjectiveBundleImplicit callable, line ~190:
f = sum(arg1) / M + ζ
```

`M = size(U)[1]` (the draw count, `= W` in this investigation) is a plain
scalar divisor of an UNWEIGHTED `sum(arg1)`. Every gradient/Jacobian/Hessian
computation downstream in the same file uses the identical pattern —
`BLAS.gemv!('T', 1/N, ...)`, `BLAS.gemv!('T', -1/N, ...)`, `BLAS.gemm!('T',
'N', 1/N, ...)` — literally dozens of call sites, all with a plain `1/N`
(or `1/M`) scalar, never a per-draw weight vector. **There is no `w::Vector`
parameter anywhere in `PsiObjectiveBundle`, `PsiObjectiveBundleImplicit`,
`PsiObjectiveBundleDelta`, `KLObjectiveBundle*`, `ccOuter.jl`, or
`ccInner.jl`** — confirmed by grep across every file in `cc_algo/`, not
assumed.

### 2.2 A DIFFERENT, narrower weight mechanism exists one layer up: `SamplingWeights`

`prepare_cc/buildObjectsForMoments.jl` does carry a `SamplingWeights` field
(default `ones(1)`, i.e. effectively "off"), threaded into `obj.γ` and
applied inside `moments/moments!.jl`:

```julia
# moments/moments!.jl, EK_moments_simple!, near the end:
for im ∈ 1:numMomentsSimple
    @. G[:, im] *= SamplingWeights[1:W]
end
@. K[:] *= SamplingWeights[1:W]
```

This multiplies BOTH the divergence column `K` (which becomes `H[:,1]`, the
`k(U,θ,γ)` term inside `arg0`) and every moment column `G` (which becomes
`H[:,2+...]`, the `g(U,θ,γ)` terms) by a per-draw scalar **before** `arg0` is
assembled and passed through the nonlinear `Psi!`/`dPsi!` map. This is
**fundamentally different from correct IS reweighting**, which needs the
weight `w_i = dF*/dQ(u_i)` to multiply the *already-Psi-transformed*
quantity — i.e. `(1/W) Σ w_i · φ*(arg0_i)`, not `(1/W) Σ φ*(w_i · raw_arg0_i)`
(the latter is what `SamplingWeights` actually computes, since `w_i` is baked
into `arg0_i` before the nonlinear transform, not applied to the transform's
output). Because `Psi!`/`φ*` is nonlinear (piecewise exp/quadratic — see
`cc_algo/Psi.jl`), these are NOT the same quantity. `SamplingWeights` is a
purpose-built device for this codebase's two existing schemes
(`importanceSampling∈{1,2}` in `prepare_cc/genRands.jl`, stratified sampling
and a scalar-rate-shifted-exponential "importance sampling" of the SAME
exponential family used for variance reduction of specific moments/CDF
estimates elsewhere in the code, e.g. `precalcCDFs.jl`/`precalcIndependence.jl`)
— it is not a general nonuniform-quadrature-weight facility for the CC
dual/primal formulas themselves, and does not touch `PsiObjectiveBundle.jl`'s
own `1/M` averages at all (confirmed: `SamplingWeights` never appears in
`cc_algo/*.jl`, only in `moments/*.jl` and `prepare_cc/*.jl`).

### 2.3 A concrete, confirmed bug: `importanceSampling==1`'s weight computation is a silent no-op

Reading `prepare_cc/genRands.jl::genExpRandsImportanceSampling!` line by
line:

```julia
function genExpRandsImportanceSampling!(U, λ, ImportanceSampleingWeight)
    rand!(U)
    for i in 1:length(U)
        U[i] = -log(1 - U[i])/λ
    end
    W = size(U, 1); D2 = size(U, 2)
    ImportanceSampleingWeight = ones(W)              # <-- REBINDS the local name to a NEW array
    for od = 1:D2
       @. ImportanceSampleingWeight[:] = ImportanceSampleingWeight[:] .* exp.((λ-1) .* U[:,od]) ./ λ
    end
end
```

The line `ImportanceSampleingWeight = ones(W)` is a plain `=` reassignment of
the local binding to a brand-new array object — it does **not** mutate the
array the caller passed in (`prepare_cc/drawU.jl`'s `SamplingWeight`, itself
`master_prepare_cc.jl`'s `SamplingWeight = ones(W)`). Every subsequent
`@.` update mutates this new, orphaned local array; the caller's actual
`SamplingWeight` is untouched and remains `ones(W)` regardless. **This means
the `importanceSampling==1` code path's weight computation has never had any
effect on any downstream computation** — a real, confirmed, unfixed bug,
found by reading the code for this task, not by running it (this
investigation does not use `importanceSampling==1` in any of its own
production configurations, so the bug has apparently gone unnoticed). This
is not something this task fixes (out of scope, and this task's instructions
say not to touch production code) — it is reported here as direct evidence
that **the codebase's only existing precedent for a "weight w_i" mechanism
is itself unreliable**, reinforcing that IS support would need to be built
and validated from scratch, not adapted from an existing working feature.
(`genExpRandsStratified!`'s analogous mechanism, by contrast, mutates its
weight array correctly via indexed assignment — `@.
StratifiedSamplingWeight[1:strata_size] = ...` — not a rebind — so stratified
sampling's weight propagation is real; this is not a blanket claim that every
weight mechanism in the file is broken, only the `importanceSampling==1`
one.)

## 3. Exactly which formulas would change, written out

Notation: `η,ζ,λ` are the inner dual variables; `k_i,g_i` are the
`k(U_i,θ,γ)`/`g(U_i,θ,γ)` moment values at draw `i`; `φ*=Psi!` is the convex
conjugate (piecewise exp/quadratic, `cc_algo/Psi.jl`); `dφ*=dPsi!`,
`ddφ*=ddPsi!`. Currently (`M`=`W` draws from F* directly, via the model's own
inverse-CDF transform):

**Dual objective** (`PsiObjectiveBundleImplicit`'s callable, `find_smallest`
sign/`δ` term omitted for brevity):

```
f(η,ζ,λ)      = (1/W) Σ_i φ*( (k_i+ζ+λ'g_i)/(-η) ) + ζ            [current, uniform]
f_IS(η,ζ,λ)   = (1/W) Σ_i w_i · φ*( (k_i+ζ+λ'g_i)/(-η) ) + ζ       [IS, w_i = dF*/dQ(u_i)]
```

**Gradient w.r.t. (ζ,λ)** (currently `g[1] = 1 - (1/M)Σφ*'(arg0_i)`,
`g[2:end] = -(1/M) H[:,3:...]' · φ*'(arg0)`):

```
∂f/∂ζ         = 1 - (1/W) Σ_i    dφ*(arg0_i)
∂f_IS/∂ζ      = 1 - (1/W) Σ_i w_i·dφ*(arg0_i)
∂f/∂λ_j       = -(1/W) Σ_i    g_{i,j}·dφ*(arg0_i)
∂f_IS/∂λ_j    = -(1/W) Σ_i w_i·g_{i,j}·dφ*(arg0_i)
```

i.e. every `BLAS.gemv!('T', ±1/N, H[...], arg1, ...)` call becomes
`BLAS.gemv!('T', ±1/N, H[...], w .* arg1, ...)` (element-wise pre-multiply
`arg1 = dφ*(arg0)` by the weight vector before the matrix-vector product —
mechanically a one-line change PER call site, but there are ~10 such sites
across `PsiObjectiveBundle.jl`'s three bundle types and both value/gradient
branches, all of which would need the same edit, consistently, and the
outer-θ envelope-theorem gradient branch (`calculate_jac_θ!`/`jac_h`-based)
needs the identical `w_i` weighting applied at the point where `jac_h[:,1,:]`
values get contracted against `arg1[1:N]` — currently unweighted there too).

**Hessian w.r.t. (ζ,λ)** (`hessian!`, uses `ddPsi!`): the same `w_i` must
multiply every `ddφ*(arg0_i)` term feeding the Hessian's outer-product-sum
assembly — i.e. a weighted second-moment matrix `(1/W) Σ_i w_i·ddφ*(arg0_i)·
h_i h_i'` where `h_i` is the relevant per-draw covariate row. This is exactly
the quantity the toy diagnostic (§4) probes for conditioning sensitivity.

**Primal weight recovery** (`oracle.jl::primal_divergence`,
`m_weights = obj.arg1` i.e. `dφ*(arg0)` at the solved dual point):

```
p_i           = m_i / Σ_j m_j                                       [current: normalizes over F*-draws]
Delta_primal  = (1/W) Σ_i φ(W·p_i)                                   [current]

# Under IS, m_i is still recovered the same way (dφ* at the solved dual point), but it now
# represents dF/dQ at draw i, NOT dF/dF* -- the RN derivative of the recovered worst/best-case
# F relative to F* requires an EXTRA factor of w_i = dF*/dQ:
p_i_IS        = (w_i · m_i) / Σ_j (w_j · m_j)                        [reweight to renormalize vs F*]
Delta_primal_IS = (1/ESS_or_W) Σ_i w_i · φ(W_eff·p_i_IS)             [effective sample size, not W]
```

The exact normalizing constant in the last line (`W` vs. an effective sample
size `ESS = (Σw_i)²/Σw_i²`, §4) is precisely the kind of detail that needs
independent, careful re-derivation from the KL/φ-divergence definition
relative to F* (not Q) before trusting any number it produces — sketched
here to show the shape of the needed change, not asserted as validated.

**Divergence itself**: `D_φ(F,F*)` in the paper's own formulation (`Nδ = {F :
D_φ(F,F*) ≤ δ}`, p.271) is always defined relative to **F***, never Q — this
does not change under IS, but every empirical estimator of any
`E_F*[·]`-shaped quantity in the code (the objective, its gradient/Hessian,
and the primal divergence) needs the `w_i` correction consistently applied,
or the estimated `Delta`/`kappa` will be biased toward Q instead of F*.

## 4. Toy diagnostic: rare-winner oversampling, ESS, weight tails, conditioning

`full_aod_diag/d4_exact/c10_phase8_is_toy_diagnostic.jl` — a self-contained,
D=6-origin, W=20,000-draw toy (reuses the REAL `cc_algo/Psi.jl` `ddPsi!`, not
a reimplementation; everything else is a deliberately simplified stand-in,
not the real D=20 model). One origin (`o*`, cost `c_{o*}=3.0` vs. `~1.0-1.2`
for the other five) is a genuine rare winner: **zero winning draws out of
20,000 under plain Monte Carlo from F*=Exp(1)^D** — a direct, small-scale
illustration of exactly the zero-winner-incidence phenomenon Section 7's
comparison also measures at the real D=20 scale. Proposal `Q`: identical to
F* on every origin except `o*`, whose shock is drawn `Exp(rate=0.15)` instead
of `Exp(rate=1)` (mean shock ~6.7× larger, which lowers `o*`'s price and
makes it win far more often) — a "deliberately oversample a rare-winner
origin" mixture, exactly the kind of proposal the task brief asks about.

Findings (one run, `S_RATE=0.15`, seed fixed for reproducibility — a single
toy configuration, not a swept sensitivity study):

- **Unbiasedness confirmed mechanically**: reweighting `o*`'s under-`Q` win
  count (0.70%, i.e. 140/20,000 draws) by the correct likelihood ratio
  `w_i = f*(u_i)/q(u_i)` recovers an estimate of `o*`'s true F*-share
  (3.4e-6, consistent with "genuinely ≈0, just not exactly measured at
  W=20,000") — and the reweighted shares for ALL SIX origins match the
  direct F*-Monte-Carlo baseline to within 0.53 percentage points (small,
  consistent with ordinary Monte Carlo noise at this ESS, not a formula bug).
  This is the one unambiguously reassuring result: **the IS weight formula
  itself, correctly applied, is unbiased**, as basic theory guarantees.
- **Effective sample size drops to 27.8% of nominal W** (ESS≈5,564 of
  20,000) from this single, fairly moderate rate-shift on ONE of six
  columns — i.e. even a modest, single-dimension proposal shift meaningfully
  degrades the effective precision of every downstream estimate.
- **Weight tails are meaningfully heavy but not catastrophic here**: max
  weight is 6.6× the mean weight; the top 1% of draws carry 6.4% of total
  weight (vs. 1% under uniform weighting) — a real distortion, not extreme,
  at this toy's scale.
- **Toy Hessian conditioning barely moved** (condition number ratio
  IS/plain = 1.04×, on a deliberately small 2×2 toy Hessian). **This
  understates the real risk**: the real D=20 A-block Hessian/gradient is
  ~400-dimensional, not 2-dimensional, and concentrating effective sample
  mass onto a handful of draws (as ESS-drop and weight-tail results above
  already show happens) plausibly has a much larger effect on a
  400-dimensional curvature matrix's conditioning than on this toy's 2×2
  one — the toy's near-1.0 ratio should NOT be read as "IS is safe for
  conditioning," only as "this specific low-dimensional slice didn't show a
  large effect; the real object was not tested."

## 5. Overall risk assessment

**QMC (Section 7) is the lower-risk path and should be pursued first, if
either is pursued further.** It preserves F* as the exact target distribution
via the model's own inverse-CDF transform — the only change is which
low-discrepancy or pseudorandom sequence feeds that transform — so it
requires ZERO changes to the dual/primal formulas in `cc_algo/`, and CC
(2023) themselves used exactly this approach.

**IS is a genuinely higher-risk, deferred workstream**, for concrete reasons
found in this assessment, not a generic caution:
1. Nonuniform quadrature weights are **not currently supported** in the part
   of the code that would need them (`cc_algo/PsiObjectiveBundle.jl`'s
   hardcoded `1/M` averages) — real implementation work, not a config flag.
2. The formulas that need to change (objective, gradient, Hessian, primal
   weight recovery — §3) are mechanically identifiable but their exact
   normalization (especially the primal-divergence/ESS question) needs
   careful independent derivation and validation against the paper's own
   KL/φ-divergence definition before any number from it could be trusted.
3. The codebase's only existing precedent for a per-draw weight
   (`SamplingWeights`) is architecturally the WRONG shape for this purpose
   (pre-nonlinear-transform scaling, not post-transform reweighting) and one
   of its two call sites is a confirmed, silent no-op bug (§2.3) — i.e.
   there is no working template to adapt, only a cautionary example.
4. The toy diagnostic (§4) shows real, non-trivial effective-sample-size loss
   and weight-tail heaviness even from a single, moderate, one-dimensional
   proposal shift — and explicitly could not rule out (in either direction)
   what this does to the real 400-dimensional A-block Hessian's conditioning.

**Recommendation**: do not pursue IS integration until (a) the modified
dual/gradient/Hessian/primal formulas in §3 are independently re-derived and
validated (ideally against a small closed-form or synthetic example where the
true answer is known), and (b) a dedicated conditioning study is run at
real D=20 scale (not this toy's 2×2 slice) to establish whether
weight-concentration meaningfully degrades the actual A-block Hessian used
by this investigation's outer-loop solves.

## 6. Files

New, all under `full_aod_diag/d4_exact/`: `c10_phase8_is_toy_diagnostic.jl`
(the toy diagnostic described in §4; standalone, ~3s runtime, no KNITRO, only
dependency is the real `cc_algo/Psi.jl`). No production file modified. This
document: `docs/fullA_D20_importance_sampling_assessment.md`.
