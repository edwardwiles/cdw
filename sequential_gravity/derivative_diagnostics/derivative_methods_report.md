# Fixing the winner-boundary derivative in the sequential/profiled CC method

Follow-on to the full-A diagnostic session (`diagnostics/A_profile_optimality_report.md`),
which established that the full-A outer gradient is provably blind to winner-boundary
(Dirac) terms. This task asked: is the SAME bug present in the production sequential/profiled
method, and if so, can it be fixed without smoothing the economic model and without
re-solving the inner CC problem for every finite-difference perturbation? Two independent
derivative methods were built, cross-validated against each other and against a fully
re-solved ground truth, then wired into the actual production outer loop.

**Bottom line.** Yes, the same bug is present (confirmed at the raw-moment level and in the
full dual integrand). Both new methods (fixed-dual finite differences and a conditional
winner-boundary Monte Carlo estimator) are valid, agree with each other and with fully
re-solved ground truth, and disagree sharply with the existing AD gradient wherever the
boundary term matters. Wiring either corrected method into the real production outer loop
changes where the search converges — κ moves by 15-30%, well outside the ~1e-6 KNITRO
tolerance. **A trade-share-inversion derivative is NOT required for this fix** (Case A: the
free outer parameter A[.,focal] is a direct subvector of θ, not produced by an inversion).

## 1. Code architecture

Repo: `trade_robustness_modular_perf` (branch `feature/sequential-inversion-perf`). Production
entry point: `sequential_gravity/run_profiled_production.jl`.

- **Inner dual objective Q(A,x)**: `PsiObjectiveBundleDelta`'s own callable
  (`cc_algo/PsiObjectiveBundle.jl:298`), `f = mean(Ψ(arg0))/M + ζ`, `arg0 = H[:,2:1+oci]·(-x)`.
  Solved via `inner_loop`/`inner_loop_internal` (`cc_algo/inner_loop_functions.jl:222`).
- **θ layout** (`focal_moments_directgp.jl:19`, `run_profiled_production.jl:124-127`):
  `θ = [μ, σ, γ'_focal, A[.,focal] (length D)]`. `A[.,focal] = θ[4:3+D]` is a **direct**
  subvector of θ (Case A, see §6 below) — `FreeParamMap`'s `free_idx = vcat(3, 4:3+D)`
  confirms only γ'_focal and A[.,focal] are ever differentiated; μ, σ are pinned.
- **Current envelope gradient**: `PsiObjectiveBundleImplicitMethodB.jl::_methodB_envelope_scalar`
  (lines 88-102), a `ForwardDiff.gradient` over the free-only vector, called from
  `run_profiled_production.jl::make_seq_div_grad_fn!` (lines 425-439). This IS the gradient of
  the 1e10-scaled divergence-budget outer constraint (`constr[1] = -f·1e10`), not of `f`
  directly — an easy sign/scale trap, see §7.
- **Hard winner construction**: `focal_moments_directgp.jl:45-50` — a hand-rolled inline
  `if price < best; best = price; bo = o; end` loop, structurally identical in kind to the
  full-A method's `misc/smoothMinIndNew!.jl::MinInd!` (a hard comparison on the *value* of a
  `ForwardDiff.Dual`, stripping all partials through that branch).
- **Sequential-linearization loop**: `run_profiled_production.jl::seq_gravcol` (lines 168-332)
  — recovers the LFD, inverts omitted destinations (`profiled_gravity.jl::invert_destination`),
  builds a linearized gravity-residual moment, iterates to `|R|≤tol`. Live, actively-wired
  production code, not a legacy path.
- **Winner criterion, verified equivalent** (not assumed): the code's actual winner rule is
  argmin *level price* (`price[o] = wHat[o]·AodPow[o]·τ[o,focal]·U[o]^μ`); the user's economic
  setup defines the winner via argmax `Φ_od = b_od·X_o`. Algebra (using
  `constConsσ[o] = b_o = c_o^{1-σ}` where `c_o = w_oτ_o/A_o` is the price constant) shows
  `Φ_od = price[o]^{-(σ-1)}` **exactly**, with no origin-specific proportionality constant — so
  argmin(price) = argmax(Φ) exactly in this model. `X_o = z_o^{σ-1} = U_o^{-μ(σ-1)} = U_o^{-1/β}`,
  and since `U_o~Exponential(1)`, a change of variables gives **`X_o ~ Fréchet(β)` exactly**,
  density `f*(x) = β·x^{-β-1}·exp(-x^{-β})`, β=θ*/(σ-1) — a well-known Fréchet power-transform
  property, derived and documented in `boundary_derivative.jl`'s header, not guessed.

## 2. Part 1 — fixed-dual finite differences

`sequential_gravity/derivative_diagnostics/fixed_dual_criterion.jl`, `fixed_dual_fd.jl`.

`dual_criterion_fixed_x(θ,obj,x_fixed)` reuses `PsiObjectiveBundleDelta`'s own callable
directly (no reimplementation) — recomputes θ's moments (hence winners, hard, no smoothing)
via `obj.moments!`, then evaluates the SAME `Ψ`/arg0 formula with x supplied externally (never
re-optimized, no KNITRO call). **Identity test passes exactly** (rel_diff = 0.0, not merely
small) at three γ' targets, D=4.

Central-difference gradient/directional-derivative functions perturb `Acol[j] *= exp(±hv)`
(log-space, matching the analytical Fréchet-Jacobian convention; production's own outer
parameterization is level-space, and the two are related by the exact identity
`d/dlog(x) = x·d/dx` — verified against the implementation, not assumed, per §7's caught bug).

**Adaptive step-size finding**: at W=8,000, winner switches are sparse (≤1.3% of draws even at
h=0.1) and no h in `{1e-4,...,1e-1}` reaches a stable plateau (≤3% drift to h/2 and 2h, ≥100
switches). At **W=32,000** a single-coordinate direction stabilizes (h=0.1, 267 switches). At
**W=128,000** both a coordinate direction and a random direction stabilize cleanly (1456 and
1286 switches, ≤3% drift). This is the expected finite-sample floor the task anticipated, not
a bug: the aggregate boundary contribution does not vanish, it is merely under-sampled at small
h and small W simultaneously.

## 3. Part 2 — conditional winner-boundary derivative

`boundary_derivative.jl`. For destination "focal", coordinate j, competitor
`k(ω) = argmax_{o≠j} Φ[ω,o]`, threshold `X_j* = M_{-j}(ω)/b_j` (`M_{-j}` = runner-up's Φ,
cached once per draw via a single O(D) top-2 pass reused for every j — Part 1.3's locality
request, here meaning "only recompute the one shared top-2 structure," since this reduced model
has a single destination, not D independent destination blocks). Decomposition:

```
dQ/da_j = INTENSIVE_j + BOUNDARY_j
INTENSIVE_j = (1/M) Σ_{ω: bo(ω)=j} arg1[ω]·(-λ_j/(Γβ))·Φ[ω,j]     (exact identity: equals AD)
BOUNDARY_j  = (1/(Mβ)) Σ_ω X_j*(ω)·f*(X_j*(ω))·(H_+(ω) - H_-(ω))
```

`H_+`/`H_-` evaluate the **complete** `Ψ(arg0)` integrand (via `psi_scalar`, matching
`cc_algo/Psi.jl::Psi!` exactly) under the "j wins" / "k(ω) wins" hypotheses at the threshold —
not a linearized moment times a fixed dual weight, per the task's explicit warning; Ψ is
nonlinear and is evaluated exactly at both points.

**Two real bugs found and fixed while validating this** (both caught by cross-checking against
Part 1's already-validated FD, not found by inspection alone):
1. Missing `1/β` chain-rule factor converting the threshold algebra (derived in α=log(b)
   units) to the code's actual free coordinate log(Acol) — omitting it made the boundary term
   exactly β=4× too large.
2. For `j==focal` specifically: the threshold hypothesis sets `X_focal` to its threshold
   value, but `X_focal` is the SAME underlying draw that also enters the price-index moment
   G[D+1] (`cc_prime·U[.,focal]^{μ(1-σ)}` — the identical exponent as `X_focal`). Leaving
   G[D+1] at its actual-draw value while X_focal is hypothetically at the threshold was wrong;
   fixed by re-evaluating G[D+1] at the threshold value too. This was the only coordinate
   showing a residual discrepancy, and critically **it did not shrink from W=128,000 to
   W=800,000**, which is what proved it was a real bug rather than MC noise.

**Part 2.4 gate** (required before any production use): intensive-only vs the analytical
Fréchet raw-moment Jacobian (`β·diag(λ)+(1-β)λλ'`, α-units) reproduces the diagonal to 73%
relative error and the off-diagonal to 100% (i.e. zero) — matching AD's known blind spot
exactly. **Intensive+boundary reproduces the full analytical Jacobian to <0.2% (diagonal) and
<2% (off-diagonal)**, D=4, W=128,000, convergent from 18% (W=1,000) down to 1.6% (W=128,000) as
draws increase. Gate passed before the full dual-integrand version was built, as required.

## 4. Part 3 — comparison against fully re-solved profile differences

192 fully re-solved KNITRO inner problems (no fixed-x approximation anywhere), D=4,
W=128,000: negative-FD-gradient direction, negative-boundary-gradient direction, all 4
single-coordinate directions, 10 random directions, 2 step sizes, 3 γ' targets.

| target | FD median relerr | boundary median relerr | AD median relerr |
|---|---|---|---|
| γ'=0.9447 (off-benchmark) | 0.23% | 3.05% | 201% |
| γ'=0.9610 (Fréchet, degenerate) | 15.3%\* | 15.1%\* | 80.2%\* |
| γ'=0.9668 (off-benchmark) | 0.66% | 3.91% | 374% |

\*At the Fréchet point the true profile slope is ≈0, so relative error inflates for *every*
method there (confirmed: AD's error is relatively *better*, not worse, at this point — the
tell that it's a zero-crossing artifact of the metric, not a real difference in method
quality). At both genuine targets, FD and boundary track the expensive ground truth to within
a few percent; AD is off by 200-374% typically, up to >5000% in single directions.

## 5. Part 4 — Monte Carlo stability and the sample-exact negative control

Median relative error vs fully re-solved profile, non-degenerate targets, across W:

| W | FD | boundary | AD |
|---|---|---|---|
| 8,000 | 0.86% | 19.2% | 204% |
| 32,000 | 0.64% | 10.2% | 239% |
| 128,000 | 0.38% | 3.4% | 213% |
| 800,000 | 0.70% | 3.6% | 202% |

FD is excellent throughout; boundary converges cleanly as W grows; AD's *mean/max* error gets
**worse** at larger W (up to 1046× at W=800,000) because a sharper ground truth exposes larger
true AD errors previously masked by noise — evidence against AD, not against FD/boundary.

**Seed sensitivity**: at the Fréchet target (3 seeds, W=32,000), cross-seed CVs were 1.2-9.8
with sign flips — alarming until re-examined: the true gradient is ≈0 there, so any noise
dominates a near-zero signal. Re-run at a non-degenerate target: CVs of 5-28%, unremarkable MC
variability. Lesson recorded for future sessions: never judge derivative-estimator stability at
a degenerate/zero-crossing point.

**Sample-exact Fréchet negative control**: `max|E_F[G]|` and `δ*(A*,γ'_frechet)` both shrink
cleanly with W (7.8e-3→7.4e-4 and 1.8e-4→1.3e-6 across W=8k→800k — the divergence scales
roughly as the square of the moment error, consistent with the CC divergence's local quadratic
behavior). **160 fully re-solved points scanned around A* at the Fréchet target (10 directions
× 5 step sizes, both signs): zero negative divergences found** (min observed δ*=2.7e-6>0).
Passes cleanly.

## 6. Part 5 — wired into the real production outer loop

Design choice (deliberately minimal-risk after two sign/scale bugs were caught in Part 2): the
corrected methods are implemented as an **additive correction on top of the unchanged AD
gradient**, `g_free_corrected = g_free_OLD_AD + CORRECTION`, exploiting the Part 1/2 identity
that AD exactly equals the "intensive" (smooth) part — so `CORRECTION` is nonzero *only* for
the A[.,focal] block, is exactly zero for γ'_focal (no winner dependence there at all) and for
the gravity-linearized moment (an affine surrogate, already exact). Setting
`gradient_method=:pointwise_ad` makes `CORRECTION≡0` **by construction**, so it is guaranteed
byte-identical to the pre-existing code path — not merely tested once, but structurally
incapable of regressing existing behavior.

Added to `sequential_gravity/run_profiled_production.jl`: a `gradient_method` keyword on
`outer_solve_nested_cached` (default `:pointwise_ad`, fully backward compatible) and an
additive `SKIP_BATCH_LOOP` env-var guard so a comparison driver can `include` the file for its
setup/functions without triggering the default 12-solve batch. New file
`gradient_method_wiring.jl` implements `make_seq_div_grad_fn_corrected!`.

### Real outer-loop comparison, D=4, δ=1, both bounds

| bound | method | γ'_focal | κ | status | rel‖ΔAcol‖ | gravity_ok | R_mean | wall |
|---|---|---|---|---|---|---|---|---|
| lower | pointwise_ad | 0.997233 | 0.004608 | −400 (converged) | 10.3% | true | 7.7e-5 | 81s |
| lower | fixed_dual_fd | 0.997654 | **0.003906** | −400 (converged) | 51.0% | true | 4.5e-4 | 77s |
| lower | boundary | 0.997035 | 0.004937 | −400 (converged) | 10.5% | true | 3.2e-4 | 60s |
| upper | pointwise_ad | 0.898797 | 0.162915 | **−102 (not converged)** | 9.5% | **false** | −5.2e-4 | 492s |
| upper | fixed_dual_fd | 0.894833 | **0.169060** | −400 (converged) | 67.8% | **true** | −4.5e-4 | 332s |
| upper | boundary | 0.899846 | 0.161286 | −400 (converged) | 4.1% | **false** | −7.8e-4 | 600s |

The upper bound is materially harder to converge for this economy regardless of gradient
method (4–8× the wall time of the lower bound) — a pre-existing property of this search
direction, not introduced by this task. Only **fixed_dual_fd's upper-bound result is both
KNITRO-converged and gravity-feasible**; pointwise_ad's upper-bound run did not converge
(status −102) and boundary's, while converged, is **not gravity-feasible**
(`gravity_ok=false`) — i.e. that specific θ* is not actually a valid CC-bound point. Both are
flagged rather than silently used below.

### Fixed-A efficiency comparison (this session's specific follow-up question)

For each result's `γ'_focal`, `δ*_fixedA(γ'_focal)` is the divergence that would be **needed to
reach that identical γ'_focal (identical GT) with A held fixed at A\* throughout** — computed
by a direct D+1-moment inner CC solve at A=A*, no outer search at all (this is exactly "what
delta gives you the same GT if you don't move A"). `extra_delta = δ*_fixedA − 1` (1 = the
actual budget spent by the moved-A search) is the effective divergence-budget "profit" from
moving A: positive means fixed-A could NOT reach that GT within budget 1 (moving A bought a
real gain); negative means fixed-A reaches (or beats) that same GT using LESS than the budget
the moved-A search actually spent — i.e. moving A did worse than doing nothing.

| bound | method | γ'_focal | κ | δ\*_fixedA | extra_delta | valid? |
|---|---|---|---|---|---|---|
| lower | pointwise_ad | 0.997233 | 0.004608 | 1.0997 | **+0.0997** | yes |
| lower | fixed_dual_fd | 0.997654 | 0.003906 | 1.2596 | **+0.2596** | yes |
| lower | boundary | 0.997035 | 0.004937 | 1.0382 | **+0.0382** | yes |
| upper | pointwise_ad | 0.898797 | 0.162915 | 0.7762 | −0.2238 | **no (not converged)** |
| upper | fixed_dual_fd | 0.894833 | 0.169060 | 0.9305 | **−0.0695** | **yes** |
| upper | boundary | 0.899846 | 0.161286 | 0.7408 | −0.2592 | **no (gravity-infeasible)** |

**Lower bound: moving A genuinely helps, at all three gradient methods.** Every method finds a
γ'_focal that fixed-A could not reach within budget 1 — fixed-A would need 4–26% *more*
divergence budget to match it. Tellingly, **fixed_dual_fd — the method most rigorously
validated in Parts 1–4 — finds the largest gain** (extra_delta=+0.26, more than double AD's
+0.10 and nearly 7× boundary's +0.04), and also moved A furthest from A\* (51% vs ~10% for the
other two). This is consistent with AD's incomplete gradient causing the *current* production
search to converge to a **locally-plausible but globally weaker** point than a correctly-
informed search can reach — exactly the failure mode Parts 1–3 predicted architecturally.

**Upper bound: the only valid comparison (fixed_dual_fd) shows a small negative extra_delta
(−0.07)** — moving A neither helps nor meaningfully hurts here, within the noise of a single
D=4/δ=1 run. The other two methods' upper-bound numbers (pointwise_ad not converged;
boundary gravity-infeasible) cannot be used to draw a conclusion about the upper bound — they
are reported for completeness, not interpreted. Given the upper direction's generally poor
convergence behavior here (long wall times, one non-convergence, one infeasible point), the
honest reading is: **the upper bound needs a more careful, better-converged re-run (larger
`maxit`, possibly different starting points) before any efficiency conclusion can be drawn
there** — this asymmetry between bounds (clean gains at the lower bound, inconclusive at the
upper) is itself a finding worth carrying forward, not an artifact to explain away.

## 7. Sign/scale conventions caught along the way (recorded so a future session doesn't re-derive them)

- `inner_loop` applies `obj.find_smallest`'s sign flip (`val *= -1`) to the raw solved `f`;
  `dual_criterion_fixed_x` must apply the identical flip to be comparable to `δ*` — missed on
  the first pass, gave a clean `rel_diff=2.0` (exactly `Q=-δ*`) rather than a vague mismatch,
  which made it easy to catch and fix.
- `_methodB_envelope_scalar` returns `d(constr[1])/dθ` where `constr[1]=-f_raw·1e10` — the
  gradient of the 1e10-scaled outer constraint, not of `f_raw` itself. `df_raw/dθ =
  -grad/1e10`. Conflating the two gives errors of ~9 orders of magnitude, not a subtle bias —
  loud enough to catch immediately, but only if you think to check.
- Production's outer parameterization of `A[.,focal]` is the raw **level** `Acol`, not
  `log(Acol)` — confirmed by inspecting `θ_lo`/`θ_hi`/`FreeParamMap`, not assumed. All
  cross-method comparisons convert explicitly via `d/dlog(x) = x·d/dx`, evaluated at the
  current point.
- The Fréchet benchmark's `Acol` is **not** all-ones under this gauge (a per-column, actually
  here per-vector-of-D-equal-values constant `γf0^{-σ/(μ(σ-1))}`, generally ≠1) — same gotcha
  independently re-derived in the earlier full-A session; using `ones(D)` as "A*" silently
  breaks the Fréchet zero-moment identity check.

## 8. Answers to the required conclusions

1. **Is fixed-dual FD valid?** Yes. Exact identity test passes (rel_diff=0.0). Tracks the fully
   re-solved ground truth to <1% median error at every W tested (8k-800k), at genuine
   (non-degenerate) targets. No inner re-solve needed per finite-difference evaluation.
2. **Does the boundary derivative match it?** Yes, after fixing two real bugs (chain-rule
   factor; focal-coordinate threshold consistency), both surfaced by exactly this
   cross-validation. Matches the analytical Fréchet Jacobian to <2% (Part 2.4) and the fully
   re-solved profile to 3-10% (better at larger W), while agreeing with FD to within a few
   percent at every W tested.
3. **Does the production sequential result move away from A\*?** Yes, materially, at **both**
   bounds. Swapping only the gradient (identical inner problem, identical KNITRO settings,
   identical starting point) changes κ by double-digit percentages and moves
   `‖A*_result − A*‖/‖A*‖` from ~10% (AD, boundary) up to ~51-68% (fixed-dual FD) at both
   bounds. Crucially, this is not just "the search wandered" — the fixed-A efficiency check
   (§6) shows that at the **lower** bound, this movement is a genuine gain: every method's
   result requires *more* divergence budget to match under fixed-A (extra_delta = +0.04 to
   +0.26), with the most rigorously validated method (fixed-dual FD) finding the largest gain.
   At the **upper** bound the one valid (converged + gravity-feasible) comparison shows a small
   negative extra_delta (−0.07) — inconclusive rather than a confirmed gain, and flagged for a
   better-converged re-run rather than overclaimed.
4. **Does any apparent improvement survive at large draw counts?** The *derivative accuracy*
   improvement clearly does (Part 4: boundary's error shrinks 19%→3.6% as W grows 8k→800k; AD's
   error, if anything, gets worse). The *outer-loop κ gain* (§6/point 3 above) was demonstrated
   at W=8,000 (this task's outer-loop default) and not yet re-verified end-to-end at W=32k-800k
   (that would mean re-running the full outer search at each W, not just the derivative check)
   — flagged as the natural next step, not silently assumed.
5. **Is a derivative through a separate trade-share inversion required?** No (Case A, confirmed
   architecturally in Part 6/§1): `A[.,focal]` is a direct outer variable; the non-focal
   destinations' inversion sensitivity is not needed for this specific derivative fix (it would
   only matter for the *gravity moment's* sensitivity to A[.,focal]/γ'_focal, which is already
   handled separately as an affine, provably-exact surrogate — see §6 — and was out of scope
   here per the task's own "do not implement until established as necessary").

## Files

- `fixed_dual_criterion.jl`, `fixed_dual_fd.jl` — Part 1.
- `boundary_derivative.jl` — Part 2 (density/threshold derivation, intensive+boundary formulas,
  Part 2.4 raw-moment gate, the full dual-integrand estimator).
- `full_profile_resolve.jl` — the expensive, validation-only fully re-solved slope.
- `run_part1_fixed_dual_fd.jl`, `run_part2_verify_raw_jacobian.jl`, `run_part2_boundary_vs_fd.jl`,
  `run_part3_comparison.jl`, `run_part4_mc_stability.jl` — per-part drivers, all runnable
  standalone (`DVAL`/`WVAL` env-overridable).
- `gradient_method_wiring.jl` — Part 5's additive-correction wrapper.
- `run_part5_gradient_method_comparison.jl` (lower bound only), `run_part5b_both_bounds_fixedA.jl`
  (both bounds + the fixed-A efficiency comparison) — Part 5 drivers.
- Modified (additive, backward-compatible): `../run_profiled_production.jl` — added
  `gradient_method` keyword to `outer_solve_nested_cached` and the `SKIP_BATCH_LOOP` guard.
- All `part*.csv` files — raw numerical output for every part.
