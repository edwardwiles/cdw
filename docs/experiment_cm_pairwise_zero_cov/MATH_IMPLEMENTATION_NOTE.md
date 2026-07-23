# Math implementation note: nested CM / CM+mean / CM+mean+ZC arms

Branch `experiment/fullA-cm-pairwise-zero-cov`, forked from `remediation/fullA-exact-2026-07-22`
@ `82dd485` (strict descendant of `production/fullA-exact` @ `670eac4`). Trial branch — not a
promotion candidate.

## 1. Why one outer scalar ν suffices

Common marginals force every origin's productivity draw to share one CDF (up to the finite-grid
approximation). If all origins share a marginal, they share every marginal moment, in particular
the mean: `E_F[z_o] = ν` for a single scalar `ν`, identical across `o`. So imposing "all D means
are equal" is not new information beyond CM's grid — but the current finite-grid CM restriction
(`precalc_common_marginals_cdf`, `common_marginals_moments.jl:50`) only matches `L` CDF cutpoints,
which does **not** force exact mean equality at finite `L` (a finite set of CDF-contrast equalities
does not pin down `E[z]`, an integral over the whole support). That is why Section 1's arms treat
"exact common means" as a distinct, additional restriction rather than something CM already
implies — confirmed empirically in Section 6.1/6.3 below (CM-only leaves a nonzero mean residual
at finite L that CM+mean drives to exactly zero by construction).

Given exact common means, the natural pairwise zero-covariance target is `ν²` (not a separately
fitted `E[z_o]E[z_p]`), because under the mean restriction `E[z_o]E[z_p] = ν·ν = ν²` for every
pair. So there is still only one free scalar (`ν`) even after adding 190 pair restrictions at
D=20 — the pair moments are not new *parameters*, they are new *equality constraints on the same
ν*. This is exactly why the outer parameter count is identical (1 scalar) between arm 2 and arm 3
while the inner moment count jumps from 20 to 210 (Section 1's table).

The mean-only arm is not a redundant subset of the ZC arm — it isolates how much of any observed
tightening comes from merely forcing equal means (a first-moment restriction with no dependence
content) versus the pairwise-covariance restriction itself (a genuine, new independence-flavored
restriction). Reporting only CM-vs-(CM+mean+ZC) would conflate these two effects; Section 6.3
requires both be measured and both nesting inequalities hold.

## 2. Parameterization

`η_ν = log ν`, `ν = exp(η_ν)`, `nu_parameterization = :log` (only option currently implemented;
kept as a Symbol field per this codebase's existing dispatcher convention — see `CMConfig`'s
`cm_grid_rule`/`cm_basis`/`cm_hessian_backend` — for a future alternative, e.g. a bounded logistic
map, without a call-site signature change).

Positivity is required because `z` is Exp(1) (via `genExpRands!`, `prepare_cc/genRands.jl:2-8`,
strictly positive support), so any feasible common mean must be strictly positive; `log` maps a
bounded-away-from-zero interval (Section 3) onto all of `ℝ`, which is what KNITRO's box-constrained
outer variable wants.

## 3. Direct vs. anchored basis

Direct reference basis (Section 2 of the task brief):

```
g_mean_o(s; ν)   = z_{s,o} - ν                      for every origin o = 1..D
g_pair_{op}(s; ν) = z_{s,o} z_{s,p} - ν²             for every unordered pair o<p
```

This is the trusted reference implementation (`build_cm_meanzc_augmented_obj` with
`meanzc_basis = :direct`, `mean_zero_cov_moments.jl`). It is also the **production candidate**:
because `d g_mean_o/dν = -1` and `d g_pair_op/dν = -2ν` are the same scalar for every `o`/pair, the
direct basis needs no anchoring to avoid a redundant equation, and it keeps every moment residual
individually interpretable ("origin o's mean deviates from ν by this much"), which is exactly what
Section 6.1's per-origin/per-pair residual checks want. The anchored contrast basis
(`g_anchor = z_{s,r} - ν`, `g_contrast_o = z_{s,o} - z_{s,r}` for `o ≠ r`, `r` = the existing CM
reference origin `ctx.γ.refIndex1`) is algebraically equivalent — same feasible set, since
`{z_o = ν ∀o} ⟺ {z_r = ν, z_o = z_r ∀o≠r}` — and is implemented as `meanzc_basis = :anchored` for
the conditioning/speed comparison Section 2 asks for, reusing the same pair-moment block (pair
covariance targets don't change under either mean basis, only the mean block's D columns are
reparameterized as one anchor + (D−1) contrasts). **Conditioning verdict (Section 8, D=20):**
[to be filled in after the D=20 conditioning comparison — see PERFORMANCE_REPORT.md].

Both bases produce IDENTICAL moment column counts (D mean-block columns either way) so no other
code needs to branch on `meanzc_basis` beyond `mean_zero_cov_moments.jl`'s own construction.

## 4. Column layout and why it makes the Hessian free

`common_marginals_moments.jl`'s existing convention places CM's `ncm` finite-grid columns
immediately before the trailing gravity column, and `cm_hessian_architectures.jl`'s Architecture C
treats the leading `NCORE` columns as a dense "economic" block (`H_EE` via one BLAS `gemm!` on
`E .* sqrt.(w)`) and the CM `ncm` columns as a bin-indexed block (`H_CC`/`H_EC` via prefix-summed
tables, since CM columns are locally-constant step functions of the reference-origin's CDF bin).

The new mean/pair columns are **not** step functions — they are smooth (linear/quadratic) in the
raw draws — so they do not fit CM's bin/prefix-sum trick, but they fit the *other* block trivially:
they are just more continuous columns, exactly like the economic block. So the implementation
inserts them as `[economic (NCORE_econ−1) | mean (D) | pair (0 or D(D−1)/2) | CM-grid (ncm) |
gravity]` and passes `NCORE_ext = NCORE_econ + n_mean + n_pair` into the existing, UNMODIFIED
`CMBinHessCtx`/`hessian_cm_structured!` machinery. `H_EE` then becomes one BLAS `gemm!` over the
widened `E_ext = [E | mean cols | pair cols]` — this is why Section 4's brief ("prefer weighted
BLAS Gram products ... before considering a different estimand or approximate Hessian") is
satisfiable with **zero new Hessian code**, only a wider slice. The dense D=4 reference
(`archA_hess_cb_builder`) is generic over any `moments!` function and requires no new code either.

## 5. Why ν never touches θ_full / FreeParamMap

`ν` is threaded through a captured `Ref{Float64}` (`nu_ref`) inside the new `moments!` closure
(`wrap_moments_with_cm_meanzc`), **not** appended as a new slot of `θ_full`/`l_full`/`free_idx`
(`context_real_d20.jl:83-88`, `cc_algo/free_param_map.jl`). Reasons:

1. `l_full`, `Aod_offset`, `free_idx = [gp; vec(A_od)]`, and the `@assert n_free(m) == 1+D²`
   invariant are load-bearing for gravity elimination (`pivot_expand`), the checkpoint schema, and
   every D-dependent test in the existing suite. Adding a slot would force a coordinated change to
   all of them for a parameter that is mathematically orthogonal to the gravity/A_od reparameterization.
2. The task brief requires the ν-derivative be a **hand-derived envelope formula**, verified against
   finite differences, not an automatic-differentiation path. Keeping ν outside `θ_full` guarantees
   no ForwardDiff/Jacobian machinery (`moments_jacobian!`, `three_way_derivatives.jl`) ever
   differentiates through it by accident — the only way `∂Δ/∂ν` is ever computed is the explicit
   formula in Section 6 below.
3. It matches this file family's own "additive only" convention (`lfix_cm_aware.jl`'s docstring):
   `nu_ref` is set once per outer KNITRO evaluation, exactly analogous to `θ` itself being
   reconstructed fresh once per evaluation — during a single inner CC dual solve, ν is exactly as
   fixed as `g`/`A_od` are, so this is not a hidden mutable-global hazard, it is a second read-only
   input to the same call, sequenced correctly (the outer driver must set `nu_ref[]` before calling
   the inner solve — enforced by a `@assert nu_ref[] == ν_expected` guard at the top of the new
   `cb_F!`/`cb_G!` callbacks, see `c40_meanzc_outer_driver.jl`).

The KNITRO outer decision vector is extended for the two non-baseline arms only:
`w = [g; zfree; η_ν]` (append, do not insert), so `w[1:end-1]` is byte-identical in meaning to the
existing `w = [g; zfree]` convention and `cm_extension = :cm_only` is byte-identical to today's CM
path (no η_ν slot exists at all when the config's `cm_extension` is `:cm_only` — the driver
function for that arm is literally the pre-existing `cm_outer_driver.jl`/`cm_checkpoint.jl`,
unmodified, per Section 6.5's regression requirement).

## 6. Envelope derivative — derivation from the ACTIVE code convention

From `oracle.jl:394` / `cm_production_bundle.jl:102-108`: `Delta_dual = cbuf[1]/1e10 =
-(mean_s[Psi(q_s)] + ζ*)`, where (from `_archC_prep_for_hessian!`, `cm_hessian_architectures.jl:372-377`
and `cc_algo/PsiObjectiveBundle.jl`'s callable) `q_s(θ) = -ζ* - Σ_j λ_j* G_j,s(θ)`, sum over every
inner (dual-reweighted) moment column `j`, and `m_s = Psi'(q_s)` is the recovered weight
(`obj.arg1`, confirmed by `archC_verified_state`'s `mean_m_resid = |sum(m)/W - 1|` check, i.e.
`mean_s[m_s] ≈ 1` at a verified solution).

By the envelope theorem, at a converged inner solve `(ζ*, λ*)` is a stationary point of the inner
problem, so `∂Delta_dual/∂ν` at fixed `(ζ*,λ*)` equals the TOTAL derivative (first-order changes in
`ζ*,λ*` induced by `dν` contribute zero to first order). Only the mean/pair moment columns depend
on `ν`:

```
∂G_mean,o,s/∂ν = -1            (every origin o, every draw s)
∂G_pair,op,s/∂ν = -2ν          (every pair o<p, every draw s)
```

so

```
∂q_s/∂ν = - Σ_o λ_mean,o* · (-1)  - Σ_{o<p} λ_pair,op* · (-2ν)
        = Σ_o λ_mean,o*  + 2ν Σ_{o<p} λ_pair,op*
```

and since `∂Psi(q_s)/∂ν = Psi'(q_s)·∂q_s/∂ν = m_s·∂q_s/∂ν` (same for every `s`, as `∂q_s/∂ν` has no
`s`-index — a scalar shared across all draws):

```
∂Delta_dual/∂ν = - mean_s[m_s] · ( Σ_o λ_mean,o* + 2ν Σ_{o<p} λ_pair,op* )
               ≈ -( Σ_o λ_mean,o* + 2ν Σ_{o<p} λ_pair,op* )     [since mean_s[m_s] ≈ 1]
```

which matches the task brief's candidate formulas EXACTLY (mean-only arm: drop the pair sum). This
independent re-derivation from the live sign/scaling convention is the check the brief asks for
("do not trust this sign mechanically") — it is re-verified numerically in Section 6.4 against
central finite differences of both the fixed-dual scalar and an independently reoptimized inner
value, at multiple D=4 points and ≥2 real D=20/L=50 points, before being trusted for any outer
gradient. `∂Delta_dual/∂η_ν = ν · ∂Delta_dual/∂ν` (chain rule through `ν=exp(η_ν)`).

Cost: this requires only `Σ_o λ_mean,o*` and `Σ_{o<p} λ_pair,op*` (two scalar sums of an
already-available `λ*` vector) plus `mean_s[m_s]` (already computed by the verified-success path) —
no `W`-scale reconstruction beyond what the base solve already produced, satisfying the brief's
"should not require a W-scale winner reconstruction or coordinate sweep."

## 7. What "pairwise zero covariance" actually asserts, precisely

The implemented restriction is exact-equality-in-expectation-under-F of `E_F[z_o z_p]` to `ν²`
for every unordered pair, imposed on top of exact common means. This is **zero covariance**
(`Cov_F(z_o,z_p) = E_F[z_o z_p] - E_F[z_o]E_F[z_p] = E_F[z_o z_p] - ν² = 0`), which equals zero
*correlation* only where `Var_F(z_o), Var_F(z_p) < ∞` and strictly positive — verified as a
diagnostic in Section 6.1 (a positive-variance check on the recovered least-favorable `F`) before
ever describing the restriction as "zero correlation" in a report. It is emphatically **not**
independence and this experiment does not implement or claim general independence.
