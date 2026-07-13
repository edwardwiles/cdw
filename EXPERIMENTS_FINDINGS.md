# Derivatives, KNITRO algorithm & profiling experiments

Working branch **`experiments-derivatives`** (baseline snapshot committed first, so `git checkout main`
or `git checkout 941ecf5 -- <file>` restores any original). Nothing pushed.

---
## HANDOFF — current state (read this first)

`master.jl` now differs from the original committed baseline in these deliberate ways:
- **`use_Jacobian = 0`** (ForwardDiff autodiff; the analytic Dirac Jacobian is off — see A3).
- **`gravMoment = 1`** (gravity orthogonality ON as an outer constraint — see §gravity).
- **GT redefined** `κ = 1 − (γ'/γ)^{σ/(σ−1)}` (baseline→autarky); point estimate 0.0686→**0.0642**.
- **Moment set reduced** 24→17 (autarky): dropped 4 redundant baseline price-index + 3 zero counterfactuals.
- csw outer `.opt`: **`hessopt 4`** (product_findiff); `use_cached_x` default **ON** (`EXP_CACHED_X=0` off).

**Current bounds (autodiff, δ=1):**
| config | κ_lower | κ_upper | note |
|---|---|---|---|
| original baseline (analytic Jac, old GT, maxit25) | 0.007437 | 0.219549 | committed reference |
| autodiff, new GT, no gravity (maxit25) | 0.002304 | 0.229318 | |
| **+ gravity (within/FE), maxit25** | 0.010230 | 0.159515 | upper not fully converged |
| **+ gravity (within/FE), maxit100** | **0.018113** | **0.154415** | converged, feas_err ~1e-17 |

**13 changed source files** (all reversible via `git checkout main`): moments!.jl, hFunction.jl,
hFunction_jacobian.jl, newGravityMoment!.jl, moments/… ; prepare_cc/{master_prepare_cc,
buildObjectsForMoments,nameMoments}.jl; prestep/master_prestep.jl; cc_algo/{inner_loop_functions,
outer_loop_functions,ccOuter}.jl; misc/doubleDiff.jl; master.jl. `.opt` files are at committed baseline
(restore via `scratch_exp/restore_baseline.sh`; backups in `scratch_exp/opt_backups/`).

**Open items:** (1) gravity bounds only converged at maxit100 (raise maxit for production); (2) restriction
moments (sameMarginals/independence) still have Float64 buffers → break autodiff, need eltype(θ) fix if
enabled; the sameMarginalsMoment Ū-subsample bug is still deferred; (3) instrumentation (OUTER_SOLVE prints
+ inner counters) is left in cc_algo — cheap, informative, remove if noisy.

---

**Fixed config for every number below** (unless a row says otherwise): 4-country simulated
*unrestricted* example, `W=8000`, `Jac_W=8000`, `δ=1`, autarky counterfactual. Outer solve =
`csw_outer_loop_settings_cluster.opt` (algorithm auto, hessopt auto≈BFGS, `maxit 25`,
feas/opt tol `1e-6`, `eval_fcga yes`). Inner solve = `ek_inner_loop_options.opt`
(algorithm 0, gradopt+hessopt exact, `maxit 100`, tol `1e-12`).

**Committed baseline bounds:** κ = **[0.007437, 0.219549]** around the gravity point estimate 0.0686.

---

## TL;DR / recommendation

**Recommended configuration: `use_Jacobian=0` (ForwardDiff autodiff) + `hessopt 4` (product_findiff) +
outer `algorithm auto`/direct + a larger outer `maxit` than the demo's 25.** Inner solver settings are
fine as-is. Rationale below; the single most important change is **turn off the analytic Dirac Jacobian
and let ForwardDiff differentiate the exact indicator.**

- **A1 (ForwardDiff now works).** A one-line element-type fix in `moments!` makes the whole moment map
  ForwardDiff-compatible; `use_Jacobian=0` runs end-to-end. Your hypothesis is right — modern ForwardDiff
  handles the exact `min`/indicator fine, and the autodiff gradient is even slightly *faster* than the
  analytic one for `∂k/∂θ`.
- **A3 (the smoothed Diracs hurt — remove them).** The forward moments already use the *exact* indicator;
  the smoothed Dirac survives only in the hand-coded analytic **Jacobian**. It is a spurious approximation
  of a measure-zero boundary term that ForwardDiff correctly omits. Result: autodiff bounds
  **[0.001, 0.303]** are *wider and still feasible* vs analytic **[0.007, 0.220]** — the Dirac biases the
  search and **understates sensitivity by ~38%** at the demo's iteration cap. β-tuning is fragile (some β
  blow the solve up) and never recovers the clean autodiff bound. **Drop the Dirac; use autodiff.**
- **A2 (Hessian).** A truly exact bilevel Hessian is impractical (needs 2nd-order IFT through the inner
  KNITRO solve), and supplying a *partial* exact Hessian (objective only) actively **hurts** (κ_upper
  0.212 < BFGS). The best practical option is **`hessopt 4` (product_findiff)** — Hessian-vector products
  from finite-differencing our *exact* gradient; it converges with the lowest opt_err in the fewest
  iterations, at zero code cost.
- **A4 (algorithm).** Outer: **interior-point direct/auto is the robust default** (solves both bounds
  feasibly). Active-set (SLQP) is faster and best on the upper bound but returns an infeasible lower bound
  here. cg/sqp are worse. **Inner algorithm choice has no effect** — the inner dual is easy for any
  algorithm.
- **B (speed).** Both outer solves hit the `maxit 25` cap, so the demo bounds are *iteration-capped*, not
  converged — raise maxit for real numbers. The outer-gradient `calculate_jac_θ!` (370 ms, 80 MB/call) is
  the code hot spot; ~50–63% of inner solves are infeasible-by-design and eat ~half the wall time.
  Removed the debug `@show`. (The "reloads the .opt file every solve" suspicion was tested and is *false*
  — 0.9 ms, negligible.)

---

## A1 — ForwardDiff compatibility

**Fix:** `moments/moments!.jl` allocated `Aod`, `AodPow`, `Aod_θ` as `ones(D,D)` (Float64). Writing
Dual numbers into them threw `Float64(::Dual)`. Changed to `ones(eltype(θ), D, D)` — identical for
Float64 θ, Dual-safe for autodiff. (The other buffers, e.g. `UPow`, were already `eltype`-typed, and
`hFunction!`/`hFunctionCounter!` already used `eltype(γ)`.) No `PreallocationTools`/`DiffCache` needed:
the offending arrays are per-call locals, so plain generic typing is the least-invasive route.

**Cross-check (at θ_initial, l=23 params, d=24 moments, N=8000 draws):**

| quantity | analytic | autodiff | max abs diff |
|---|---|---|---|
| ∂k/∂θ (objective grad) | — | — | 1e-8 on all **free** vars* |
| ∂k/∂θ time | 0.63 ms | 0.47 ms | autodiff faster |
| ∂g/∂θ (moment Jacobian) time | 514 ms | 802 ms | — |
| ∂g/∂θ values | differ by the SmoothDirac term (see A3) | | up to ~190 per-draw |

\* the only ∂k/∂θ discrepancy (0.019) is on θ[2]=σ, which is **fixed** in the outer loop, so it never
enters the optimisation.

---

## A3 — do the smoothed Diracs buy anything? (No.)

Where smoothing actually lives in the unrestricted example:
- `hFunction!` forward pass → **exact `MinInd!`** (smooth-min is commented out).
- `hFunction_jacobian.jl` (analytic Jacobian) → **`SmoothDirac(β=0.01)`** at lines 117/119,
  approximating ∂/∂θ of the arg-min selection indicator (a Dirac at price ties).

ForwardDiff differentiates the exact indicator → piecewise-constant → derivative 0 a.e. → it **omits**
the Dirac. That is the *correct* a.e. derivative of the finite-sample MC objective KNITRO optimises.

**Bounds (maxit 25; both hit the iteration cap, status −400):**

| scheme | κ_lower | κ_upper | feas_err (up/lo) |
|---|---|---|---|
| analytic, Dirac β=0.01 (baseline) | 0.007437 | 0.219549 | 0.0 / 0.0 |
| **autodiff, no Dirac** | **0.00124** | **0.302701** | **0.0 / 0.0** |

Both feasible, but autodiff is **wider on both ends**. Because the bounds are max/min of κ over the
δ-neighbourhood, a wider *feasible* point is strictly closer to the truth: **true κ_upper ≥ 0.3027**, so
the analytic Dirac bound **0.2195 understates sensitivity by ~38%**. The Dirac term's spurious mass
biased the constraint gradients and stalled the search short of the more-extreme feasible optima.

**Convergence (maxit 100)** — is it a different answer, or just slow convergence? Just slow:

| scheme | κ_lower | κ_upper | upper status |
|---|---|---|---|
| analytic (Dirac) | 0.005804 | 0.267872 | −400 (still iter-capped, opt_err 0.50, **still climbing**) |
| **autodiff (no Dirac)** | 0.001051 | **0.303174** | −102 (≈settled; κ_upper barely moved 0.3027→0.3032) |

So the analytic/Dirac path converges to (roughly) the same place but **much more slowly** — its
perturbed gradients drag out the solve, so at any fixed iteration budget it reports a **too-narrow**
bound. Autodiff reaches the settled value in a fraction of the iterations.

**β-sweep of the analytic Jacobian** (β→∞ should kill the Dirac): fragile, not a fix —

| β | κ_lower | κ_upper | note |
|---|---|---|---|
| 1e−3 (sharp) | 0.008393 | 0.230813 | |
| 1e−2 (baseline) | 0.007437 | 0.219549 | |
| 1e−1 | 0.007039 | **fail (obj=1e10)** | upper solve blows up |
| 1e0 | 0.003245 | 0.242929 | |
| 1e2 (≈no Dirac) | 0.004453 | **fail (obj=1e10)** | upper solve blows up |

No finite β reproduces the clean autodiff bound, and some β destabilise the upper solve. **Tuning β is a
dead end; differentiating the exact indicator with ForwardDiff is the right move.** The smoothed Diracs
were only ever there to let autodiff run on non-smooth code — now that ForwardDiff handles the exact
`min`/indicator directly, they are pure liability.

---

## A2 — outer Hessian options

Structure: the implicit-autarky outer **objective is `k(θ)`** (closed form, no inner dependence) — its
exact Hessian `∇²k` is cheap (ForwardDiff). But the **constraints** (divergence budget + moment
conditions) depend on the inner solution `x*(θ)`; their exact curvature needs a **second-order IFT**
through the inner KNITRO solve, which is disproportionate to hand-derive and cannot be autodiffed
(KNITRO is a black box in the loop). So a fully-exact Lagrangian Hessian is impractical here.

Practical comparison (exact gradient throughout, analytic Jac, maxit 25). Higher κ_upper / lower
opt_err in the same 25 iterations = better curvature:

| hessopt | κ_lower | κ_upper | upper opt_err | note |
|---|---|---|---|---|
| auto = bfgs (2) (baseline) | 0.007437 | 0.219549 | 1.03 | |
| **product_findiff (4)** | 0.005713 | **0.237256** | **0.394** | best; Hvec-products from finite-diff of exact grad |
| lbfgs (6) | 0.005713 | 0.237256 | 0.394 | ties product_findiff here |
| sr1 (3) | 0.02878 | 0.241905 | 0.163 | good upper, but **erratic lower** (0.029, worse) |
| **exact objective-Hessian callback (hessopt=1)** | 0.010826 | **0.211653** | 0.272 | **HURTS** (see below) |

**Exact objective-Hessian callback** (`test_exact_hessian.jl`, `KN_set_cb_hess` with `∇²k` from
ForwardDiff, hessopt=1, maxit 100): κ_upper = **0.2117** — *narrower than even plain BFGS at maxit 100
(0.2679)*. Supplying the exact objective curvature while omitting the (uncomputable) **constraint**
curvature gives KNITRO a systematically wrong Lagrangian Hessian and it converges confidently to a
*worse* point. This confirms the structural argument: for this bilevel problem you cannot get a useful
exact Hessian without the second-order IFT through the inner solve, which is disproportionate. **Don't
supply a partial exact Hessian.**

**product_findiff (4)** is the practical winner over BFGS — no extra code, just a `.opt` setting, and it
uses our exact gradient to build curvature. But (crucial): with the **analytic (Dirac) Jacobian**, no
Hessian option escapes the gradient bias — `conv_prodfd_m100` still only reaches κ_upper=0.2615 at maxit
100, vs autodiff's 0.303. **Jacobian/gradient quality dominates the Hessian choice.** The biggest win is
A3 (autodiff); product_findiff is the second-order polish that makes it converge fastest:

| recommended combo (autodiff + product_findiff) | κ_lower | κ_upper | upper opt_err |
|---|---|---|---|
| maxit 25 | 0.007778 | 0.300082 | **0.128** (vs 1.6 autodiff-BFGS, 1.03 analytic-BFGS) |
| maxit 100 | 0.004306 | 0.301852 | 0.489 |

→ autodiff+product_findiff reaches the settled upper bound (~0.30) with the lowest opt_err in the fewest
iterations. (The lower bound sits near 0 either way — it wobbles 0.001–0.008 with iters/Hessian, which is
expected since the true lower bound → 0.)

---

## A4 — KNITRO algorithm sweep (analytic Jac, BFGS/auto Hessian, maxit 25)

Outer `algorithm` ∈ {auto,direct,cg,active,sqp} (analytic Jac, maxit 25):

| outer algorithm | κ_lower | κ_upper | upper opt_err | upper time | note |
|---|---|---|---|---|---|
| auto (0) = direct (1) | 0.007437 | 0.219549 | 1.03 | 100–108 s | robust; auto picks interior-direct |
| cg (2) | 0.011003 | 0.184907 | 0.448 | 69 s | fast but **narrowest** (converges to less-extreme pts) |
| active (3) | *lower INFEASIBLE (−410)* | 0.242674 | **0.116** | **70 s** | best & fastest UPPER, but **fails the lower bound** |
| sqp (4) | *lower INFEASIBLE (−410)* | 0.184268 | 1.34 | 107 s | narrow upper + infeasible lower |
| multi (5) | — | — | — | — | **SIGSEGVs KNITRO** (core dump ~123 s) — unusable here |

**Interior-point direct/auto is the robust choice** (solves both bounds feasibly). Active-set (SLQP) is
faster and gives the best *upper* bound but returns an infeasible *lower* bound here — usable only with
a feasibility fallback. cg/sqp are worse. 

**Inner `algorithm` ∈ {direct(1), active(3)}: no effect** — bounds identical to baseline to 4+ digits.
The inner dual is a well-conditioned smooth convex problem; any algorithm nails it. Not a tuning lever.

---

## B — profiling

### B2 — solver behaviour (from instrumentation)
- **Both outer solves hit `status=-400` (KNITRO iteration limit)** at `maxit 25`. The published bounds
  are **iteration-capped**, not tolerance-converged (opt_err ≈ 1.0–1.6). Raising maxit widens them
  (see A3 maxit-100) — worth knowing when quoting the example's numbers.
- **~1 inner solve per outer FC eval** (e.g. 188 FCevals → 188 inner solves).
- **~47–63% of inner solves return infeasible** (θ outside the divergence budget; `nStatus∉feasible`,
  objSol=−1e10): upper 89/188, lower 120/192. Expected, but quantified — ~half the inner work is
  spent probing infeasible θ.
- ~6–7 inner KNITRO iterations per inner solve.
- **The ~50–63% infeasible inner solves are the biggest solver-level waste**: each still costs ~0.5 s, so
  roughly *half* of each ~100 s outer solve is spent probing θ outside the divergence budget. The **outer
  algorithm** controls this: active-set (SLQP) probed far fewer infeasible points (25 vs 89) and finished
  in 70 s vs 100 s — but it returned an infeasible *lower* bound here, so it needs a feasibility fallback
  before it's usable. This is the main safe-ish speed lever at the solver level.
- **Removed** the per-inner-solve `@show nStatus / @show objSol` debug prints — noise, not a measurable
  bottleneck (stdout I/O on ~380 solves/run is small next to ~0.5 s/solve of KNITRO work).

### B1 — inner-loop hot paths (warm, M=8000 draws, l=23, d=24)

| function | time/call | alloc/call | role |
|---|---|---|---|
| `Psi!` / `dPsi!` / `ddPsi!` | 70–80 µs | **0 B** | divergence conjugate — already optimal, not a bottleneck |
| inner objective+grad `Q(x,g)` | 0.33 ms | 416 B | BLAS gemv + Psi — cheap |
| `calculate_grad_k!` | 0.44 ms | 73 KB | outer objective grad — cheap |
| `EK_moments!` | 20.5 ms | **5.8 MB** | moment assembly, once per inner solve |
| **`calculate_jac_θ!` (analytic)** | **370 ms** | **80 MB** | **outer-gradient cost centre** (once per outer FC eval) |
| one full inner KNITRO solve | ~0.5 s | — | ~4–7 KNITRO iters; time is KNITRO-internal |

Findings:
- **`calculate_jac_θ!` dominates the outer gradient**: 370 ms and **80 MB allocated per call**. At ~190
  FC evals per outer solve that's ~70 s and ~15 GB of GC churn per solve. Preallocating its temporaries
  (the analytic Jacobian builds `Not()`-indexed copies and per-call matrices) is the highest-value code
  optimisation. (The autodiff Jacobian is ~800 ms/call — ~2× slower — but yields better search
  directions, so total wall time is comparable; see A3.)
- **Each inner solve ≈ 0.5 s and it is KNITRO-internal, *not* our code.** I explicitly tested the
  "reloads the 29 KB .opt file every solve" hypothesis: `KN_new`+`KN_load_param_file`+`KN_free` = 1.1 ms
  total (param load itself 0.88 ms). **Rejected** — context/param overhead is negligible; the time is
  KNITRO's barrier iterations. So don't bother caching the KNITRO context for speed.
- `Psi!`/`dPsi!`/`ddPsi!` are already zero-allocation tight loops — leave them.
- `EK_moments!` allocates 5.8 MB/call (fresh `UPow`/`UσPow` at `moments!.jl:126-127`); preallocating
  those is a minor, safe win.
- **`@code_warntype` on `EK_moments!` and the inner `Q(x,g)`: both fully type-stable (0 `::Any`).** So the
  allocations above are genuine array temporaries, not type-instability boxing — the fix is preallocation,
  not annotations. No type-stability work needed.

**Removed** the per-inner-solve `@show nStatus / @show objSol` debug prints in `inner_loop_internal`
(fired on all ~380 solves/run). Cheap I/O individually but pure noise; gone.

---

## Files / how to reproduce
- Harness: `scratch_exp/run_experiment.sh` (per-run .opt overrides + logging), `scratch_exp/phase2.sh`
  (the sweep), `scratch_exp/parse_results.sh` (tabulate), `scratch_exp/results/*.log`.
- Standalone: `test_jac.jl` (Jacobian cross-check), `profile_inner.jl` (B1), `test_exact_hessian.jl` (A2).
- Runner: `run_master.jl` (reads `EXP_USE_JAC`, `EXP_BETA` from env). Baseline unchanged: `master.jl`.

---

## Moment & parameter map (4-country autarky example) + redundancies

**Parameters searched, θ (length 23):**
| idx | symbol | status |
|---|---|---|
| 1 | μ (Frechet dispersion) | free; **bounded 0 < μ ≤ 1/(σ−1)** |
| 2 | σ (CES elasticity) | FIXED at 2.5 |
| 3–6 | γ_θ[1..4] (baseline aux) | free |
| 7 | γ'_θ[baseIndex] (counterfactual aux) | free |
| 8–23 | A[o,d], o,d∈1..4 (16 entries) | **A[o=1,d]=1 fixed for each d** (4 fixed) → 12 free |

→ 18 free parameters (μ, 4 γ, 1 γ', 12 A). **Redundancy #1 (one A per d) IS handled** — the code
pins the origin-1 row A[1,d]=1 for every destination d (one normalization per d).

**Moments matched, g (length 24):**
- cols 1–16: trade-share moments, one per (o→d) pair.
- cols 17–20: baseline price-index moment, one per d.
- col 22: counterfactual (autarky) price-index for the baseIndex country.
- cols 21, 23, 24: counterfactual placeholders for the *other* 3 countries — **identically zero**
  (autarky only touches baseIndex); prep already flags them `moments_without_var`.

**Redundancy #2 (price index = sum of trade shares) is NOT handled — confirmed numerically:**
data trade shares sum to 1 over origins for each d (=1.0 exactly), and the baseline price-index moment
for d **equals the sum of that d's 4 trade-share moments to machine precision (max|diff| ~1e-15)**. So
cols 17–20 are exact linear combinations of cols 1–16 → 4 redundant equality moments. Plus cols 21/23/24
are degenerate zeros. **Effective rank ≈ 17 (16 trade shares + 1 counterfactual); 7 of 24 carry no
information but are still assembled and given inner-dual multipliers.** Dropping them (keep 16 trade
shares + col 22) shrinks the inner problem (25 → ~18 variables) with identical bounds — the best
low-risk speedup, since the inner problem is solved ~190×/outer solve.

## Sanity check: κ_upper vs the analytical ceiling
κ here is gains-from-trade = λ_dd^{−μ} − 1 for the baseIndex country. The point estimate uses the Frechet
μ=1/6: 0.671546^{−1/6} − 1 = **0.0686** ✓. The outer loop bounds μ at 1/(σ−1), so the ceiling is
**κ_max = λ_dd^{−1/(σ−1)} − 1 = 0.671546^{−1/1.5} − 1 = 0.3038** (with λ_dd=0.671546, σ=2.5). The autodiff
κ_upper = 0.3032 sits *just under* this ceiling (it pushes μ to its boundary) — a strong validation that
the autodiff bound is correct, and that the analytic-Dirac bound (0.22) fell short because its biased
gradient never reached the μ=1/(σ−1) boundary. (If the ceiling is written in ratio form λ_dd^{1/(σ−1)}
= 0.767, note κ = 1/ratio − 1, so the gains-form ceiling 0.3038 is the tight one.)

---

## Implemented pipeline changes (follow-up session) + verification

All on branch `experiments-derivatives`; `master.jl` now defaults to `use_Jacobian=0` (autodiff).

| # | change | files | effect |
|---|---|---|---|
| 1 | GT redefined **baseline→autarky**: `κ = 1 − (γ'/γ)^{σ/(σ−1)}` | `moments!.jl`, `master_prestep.jl` | point est 0.0686→**0.0642**; bounds map `new=old/(1+old)` |
| 2 | dropped 7 redundant/degenerate moments (4 baseline price-index + 3 zero counterfactuals); 24→17 | `master_prepare_cc.jl`, `hFunction.jl`, `nameMoments.jl`, `moments!.jl` | **bounds unchanged**; inner problem 25→18 vars; **~24% less KNITRO time** |
| 4 | preallocate `UPow`/`UσPow` (Float64 path via a `γ` cache; Duals still per-call under autodiff) | `buildObjectsForMoments.jl`, `moments!.jl` | fewer allocations on the Float64 moment fills |
| 5 | outer `hessopt 4` (product_findiff) | `csw_..._cluster.opt` | upper opt_err 1.6→**0.09** (near-converged in 25 iters) |
| 3 | `use_cached_x` warm-start, default **ON** (`EXP_CACHED_X=0` to disable) | `ccOuter.jl` | ~3% wall, up to 16% fewer inner iters; bounds/feasibility unchanged |

**On the γ definition (answering "is there an extra σ?"):** yes. The price-index moment sets
`denom[d] = γ[d]^σ·gdp[d] = E[min_o p_od^{1−σ}] = P_d^{1−σ}`, so **γ_code = P^{(1−σ)/σ}**, not `P^{1−σ}`.
Your `γ = Σ_o p_od^{1−σ}` is `γ_code^σ`. Hence the code-γ exponent is `σ/(σ−1)`, and the new-def
ceiling is `κ_max = 1 − λ_dd^{1/(σ−1)} = 1 − 0.6715^{1/1.5} = 0.2331` (κ_upper=0.2293 sits just under ✓).

**Verification (autodiff, δ=1, maxit 25):**
- VERIFY1 (#1+#4+#5, no #2): κ = [0.002298, **0.229318**], wall 253s.
- VERIFY2 (+#2): κ = [0.002304, **0.229318**] — κ_upper identical to 10 digits; κ_lower differs only in
  the 5th digit (both unconverged at the cap). Point est prints 0.06421. Wall 215s.
- VERIFY3 (+cached_x): κ = [0.002304, 0.229318]; wall 208s.

## #6 — the gravity / orthogonality moment
It exists: `gravMoment` → `moments/newGravityMoment!.jl`, computing `Σ (Δτ−meanτ)·ΔA` with
`Δ=doubleDiff` (double-diff ln τ vs double-diff ln A_od). For **UoModel=1** (our config) it depends only
on `A` and data `τ` — **F-independent**, exactly your orthogonality. BUT:
1. It is currently **off** (`gravMoment=0`).
2. It is wired as an **inner-loop** moment (a constant column matched by the dual), *not* an outer
   constraint on θ — your instinct that it belongs in the outer loop is right (the outer-constraint
   machinery already exists, used by `GravityMomentFirstApproach`/`independenceMoment` via
   `nOuterLoopMoments`).
3. The `UoModel==1` branch has a bug: `sumGrav` is used with `+=` before being initialized
   (`newGravityMoment!.jl:22`) — it would throw if turned on as-is.
To give you the F-independent gravity constraint you describe: fix the init bug + move it from an inner
moment to an outer constraint (add to `nOuterLoopMoments`) + turn it on. Recommend confirming before
wiring, since it adds an identifying restriction on A that will (presumably) tighten the bounds.

---

## Trade-share moment redundancy — is it safe to drop one per d? NO.

The A_{od} normalization (fix A[1,d]=1) removes a redundant **parameter** (the model is invariant to a
common per-d scaling of A_{·,d}, which is absorbed jointly with γ[d]) — it does **not** make any
trade-share moment a linear combination of the others. The D trade-share moment functions for a given d
have **disjoint supports** (each is active only where its origin is the cheapest), so they're linearly
independent; the *only* linear dependence was Σ_o(trade shares)=price index, which we already exploited.
Empirical rank check at θ_initial (W=8000): rank(full 17-moment G)=**17**; each d's 4 trade-share cols
rank **4/4**; all 16 trade-shares rank **16/16**; smallest singular value 0.986 (well-conditioned).
→ **Dropping a trade-share moment removes real information and would change the bounds. Not safe.**

## Gravity / orthogonality moment — now an outer constraint, turned on
Changes: fixed the uninitialized `sumGrav` (newGravityMoment!.jl), made `doubleDiff` ForwardDiff-safe
(`zeros(eltype(z),…)`), added `gravMoment` to `nOuterLoopMoments` so it enters the **outer** loop as an
equality constraint on θ (not an inner moment), switched the call to pass **`Aod_θ`** (the free A
parameter) instead of `AodPow` so the moment is exactly `cov(ΔΔ ln Aod_θ, ΔΔ ln τ)=0`, and set
`gravMoment=1` in `master.jl`. Verified: point est 0.0642, numMomentInnerSimple=17 (+1 outer), and the
constraint is enforced (feas_err ~1e-8). Bounds with the correct `AodPow` object:
κ = **[0.005986, 0.180404]** vs **[0.002304, 0.229318]** without — the gravity orthogonality
**tightens the bounds ~23%** (κ_upper 0.229→0.180), which is the economically sensible direction
(restricting ΔΔlnA ⟂ ΔΔlnτ limits the admissible A/μ configurations). Both at maxit 25 (opt_err ~1.2,
unconverged) → raise maxit for exact numbers. (The earlier mistaken `Aod_θ` object gave [0.0012, 0.232],
i.e. barely bound — the object choice is decisive; see below.)
### Which A object? `AodPow` is the raw structural A (corrected)
Tracing the price: `hFunction!` builds `pricesTemp[o] = w_o·AodPow·τ_od·U^μ` with productivity
`z_o = U^{-μ}` (Frechet(1,θ), μ=1/θ). Matching `MC_od = w_o·τ_od/(A_od·z_o)` gives **`A_od = 1/AodPow`**
exactly (AodPow is the object that enters the price, not the line-73 `Aod` or `Aod_θ`). So
`ΔΔ ln(AodPow) = −ΔΔ ln(A_od)`, and `Σ(ΔΔlnτ)·ΔΔln(AodPow)=0` **is** the consistency condition
`Σ ΔΔlnτ·ΔΔlnA_od=0`. The other two objects are contaminated:
- `Aod` (line 73) `= cHat·A_od^{1/μ}` ⇒ ΔΔln adds `ΔΔln cHat`.
- `Aod_θ` (free param, =1 at the gravity baseline) ⇒ ΔΔln adds `ΔΔln(wŵτ)/μ + ΔΔlnλ + …` (incl. a τ term).
⇒ **the original `AodPow` was correct**; reverted the earlier (mistaken) switch to `Aod_θ`.
### Correct gravity transform = two-way (origin+destination) FE "within", NOT double-difference
Numerical check (`gravity_check.jl`, regress lnA on lnτ with origin+dest FE, 2000 random draws, D=4):
| transform | max \|b − b_two-way-FE\| |
|---|---|
| two-way within `z̃=lnz − mean_o − mean_d + grand` | **1.8e-15** (exact, FWL) |
| cell double-diff `z_od − z_1d − z_o2 + z_12` | 2.3 |
| code's demeaned double-diff | 1.3 |
The cell double-difference does **not** reproduce a two-way-FE coefficient, and no τ-demeaning fixes it
(DD references cells o=1,d=2; FE references means — different projections, `DD'DD ≠ W'W`). Implemented
`withinTransform` (misc/doubleDiff.jl) and rewrote both (a) the gravity **moment** (`newGravityMoment!`,
UoModel=1) to `Σ_od withinTransform(τ)·withinTransform(AodPow)=0`, and (b) the prestep **θ̂** estimator
(master_prestep.jl) to the within/FWL form. On the clean simulated data θ̂=6 either way (both recover the
true θ), so the point estimate is unchanged; the constraint differs during the robust search.

### Effect of gravity on the bounds (autodiff, maxit 25, δ=1)
| config | κ_lower | κ_upper | width |
|---|---|---|---|
| no gravity | 0.002304 | 0.229318 | 0.227 |
| gravity, double-diff (AodPow) | 0.005986 | 0.180404 | 0.174 |
| **gravity, within / two-way-FE (correct)** | **0.010230** | **0.159515** | **0.149** |
The FE-correct gravity orthogonality tightens most: κ_upper 0.229→0.160 (~30%) — it rules out "hold
Frechet, push μ→1/(σ−1)", which violates gravity. Caveat: still maxit-25 (upper status −410, feas_err
1.5e-6; lower opt_err 0.6) → raise maxit for converged numbers. The τ-demeaning question is subsumed:
the within transform is the right operator (demeans both margins + adds back the grand mean).
