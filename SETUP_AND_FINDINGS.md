# Trade-Model-Robustness — modular working copy

Working copy of `habibiscoding/Trade-Model-Robustness`, reorganized so the **modular
pipeline (`master.jl` + subfolders) is the main version**, made to run end-to-end, and
trimmed of the experimental starting-point machinery. Set up + verified on
**`demand.mit.edu`** on 2026-07-10.

## Status: WORKING ✅

The 4-country simulated **unrestricted** example runs end-to-end and produces the
distribution-agnostic gains-from-trade bounds from the CDW paper. At δ=1:

| quantity | value |
|---|---|
| κ_lower | **0.0074** (→ 0, the paper's lower bound) |
| point estimate at F\* (gravity/EK) | 0.0686 |
| κ_upper | **0.2195** (~3.2× the gravity value) |

This is the shape of **CDW Figure 2**: the gravity point estimate sits inside wide bounds.
The legacy monolith (`legacy/GravityRobustness_run.jl`) gives the same point estimate (≈0.0708).

## The papers (context)

`papers/` holds the two references (see also the Claude memory note):
- **CDW** = Chenguiti Ansari, Donaldson & Wiles, *"Quantifying the Sensitivity of Quantitative
  Trade Models"* (`CDW_Draft_June_2026.pdf`) — the paper; defines the goal.
- **CC 2023** = Christensen & Connault, *"Counterfactual Sensitivity and Robustness"*, ECMA
  (`Christensen Connault 2023 ECMA.pdf`) — the computational method.

Method ↔ code map: counterfactual κ = `GT_d = 1 − (γ_d/γ'_d)^(1/(1−σ))` (autarky); F\* = iid
Frechet(1,θ\*) with θ\* from the gravity double-difference regression; bounds = max/min of GT_d
over F within **hybrid-divergence** δ of F\* (`Psi.jl` = the hybrid divergence), via CC's convex
inner dual + outer search over ψ={γ,γ′,A}; the LFD is recovered from `dPsi!`.

## How to run

KNITRO only licenses on **`demand.mit.edu`** (see the Claude memory note for why):

```bash
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.juliaup/bin:$PATH"   # julia 1.12.6
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular
julia --project=. master.jl
```

Output CSVs land in the repo root: the main one `NoJacob_..._DR_1_<date>.csv` holds
`δ, κ_lower, κ_upper`; `Theta_*` and `LFD_*` hold the bound-achieving parameters and
least-favorable distributions. (KNITRO prints harmless `mip_*_gap` deprecation warnings.)

### Example vs. paper-scale config (`master.jl` params)

The committed config is a **fast 4-country example** (~4 min), not paper precision:
- `W = 8000` (draws) and `Jac_W = 8000` — the paper uses ~80000; **`Jac_W` must be ≤ `W`**.
- `csw_outer_loop_settings_cluster.opt` has `maxit 25` (the outer solve is a slow nested
  bilevel program; unlimited `maxit 0` did not finish in >7 min even at D=4).

For paper precision: set `W=80000, Jac_W=25000` in `master.jl` and `maxit 0` in the csw opt
file (expect a long run — this is why the "cluster" opt file exists).

## Environment

`Project.toml` / `Manifest.toml` are committed (Julia 1.12.6 via juliaup). Deps: KNITRO
**v1.2.1**, Distributions, Plots, JLD2, Parameters, NLsolve, ForwardDiff, Calculus,
SpecialFunctions, InvertedIndices + stdlibs.

## KNITRO — two important compatibility facts

1. **License is machine-locked to `demand.mit.edu`** (not `supply`). Set the env vars above.
2. **`cc_algo/knitro_compat.jl`** — the loop code was written for the KNITRO.jl 0.13/0.14
   high-level API; v1.2.1 dropped a few convenience wrappers. This shim restores them
   (`KN_add_vars`/`KN_add_cons` returning indices, string-name `KN_get_int_param`, 2-arg
   `KN_set_var_lo/upbnds`). Everything else (callbacks, `KN_set_con_eqbnds`, `KN_get_solution`,
   …) matched v1.2.1 already.

## What was fixed to make the modular pipeline run

`master.jl` had never been run as `julia master.jl` (only in a pre-loaded REPL). Because
only `cc_algo/` is a module and the other subfolders are `include`d into `Main`:
- **`master.jl` imports** — added the full Main-scope `using` block (Parameters, Base.Threads,
  Random, Dates, NLsolve, DelimitedFiles, ForwardDiff, Calculus, LinearAlgebra,
  SpecialFunctions, InvertedIndices) before the includes.
- **`setup/setwd.jl`** — `user==2` now `cd`s into this repo (so the `.opt` files and outputs
  resolve here); `master.jl` sets `user = 2`.
- **`misc/checkParams.jl`** — rewritten (dropped the nonexistent-`calculateLFD` branch).
- **params block** — completed and set to the unrestricted core config (autarky, closed-form
  Frechet prestep, Psi/hybrid divergence, OuterLoop=1).
- **`use_Jacobian = 1` is required** — `moments!` uses preallocated Float64 caches (e.g.
  `AodPow`) that break ForwardDiff (`Float64(::Dual)`), so the autodiff outer-gradient path
  cannot work. The analytic Jacobian (`moments/moments_Jacobian!.jl`) uses `Not` (InvertedIndices).

## Derivatives / Jacobians (how each loop feeds KNITRO)

- **Inner loop** (convex dual over multipliers η, ζ, λ at fixed θ): **exact analytic gradient AND
  exact analytic Hessian**. `ek_inner_loop_options.opt` sets `gradopt exact`, `hessopt exact`; the
  loop registers both an FG callback (`callbackEvalFG_inner!`) and a Hessian callback
  (`callbackEvalH_inner!`). Both come from closed forms — BLAS ops on the moment matrix `H` with
  the divergence-conjugate derivatives `dPsi!`/`ddPsi!`.
- **Outer loop** (search over structural θ): **exact analytic gradient/constraint-Jacobian, with a
  quasi-Newton (BFGS) Hessian**. `gradopt exact` in both outer opt files; the θ-derivatives
  (`calculate_grad_k!`, `calculate_jac_θ!`, with `use_Jacobian=1`) use the hand-derived analytic
  `EK_moments_Jacobian!` **plus the implicit function theorem** (`ift!`) to differentiate through
  the inner optimizer (how x\*(θ) shifts with θ). Smooth-min operators (`SmoothDirac`) keep the
  cheapest-supplier arg-min differentiable in closed form. `hessopt auto` + no Hessian callback
  registered ⇒ KNITRO uses its own quasi-Newton Hessian for the outer problem.
- **Not used:** finite differences (nowhere; `Calculus` is imported but never called). A ForwardDiff
  autodiff **fallback** exists for the outer gradient (`calculate_grad_k_autodiff!`, used only when
  `use_Jacobian=0`) but is broken by the Float64 caches above — hence `use_Jacobian=1`.

## Cruft removed (this pass)

Per the agreed scope — **only the experimental starting-point machinery**, keeping all the
restriction moments (common-marginals / independence / Frechet-marginals; core to CDW §5) and
the KL / Conditional objective bundles:
- Deleted `prestep/preStepGeneralDistribution.jl` (general-distribution / RN / BB presteps).
- Deleted the Frechet-copula draw functions from `prepare_cc/drawU.jl` (kept the Exp(1) `drawU`).
- Removed the starting-point routing + flags (`useFiniteSamplePrestep`,
  `useFrechetCopulaStartingPoint`, `useRNStartingPoint`, `UseUnput_Θ`) from `master_prepare_cc.jl`,
  `buildObjectsForMoments.jl`, `PMM.jl` (γHat), `checkParams.jl`, and the `master.jl` params.
- Verified: bounds unchanged (`[0.0074, 0.2195]`), zero errors.

## Known issues / deferred

- **`sameMarginalsMoment!` bug (common-marginals restriction).** `calculate_grad_k` runs
  `moments!` on a 2-row `K`/`G` subsample, but `sameMarginalsMoment!` indexes the full `Ū`
  (row mismatch). So restricted runs (`sameMarginalsMoment=1`) currently break under the
  analytic Jacobian. The unrestricted example is unaffected. **Deferred** — fix by passing a
  row-consistent `Ū` subsample. The other restriction moments should be checked similarly.
- **`legacy/`** — the old monolith, kept only as a validated reference; `GravityRobustness_run.jl`
  is a runnable copy (with leftover debug `@info` in its outer callbacks). Safe to delete.
- **`fakeData=2`** path (`createFakeDataGeneric.jl` + `genRands.jl`, uses Bigsimr) is an
  alternative data source, left in place but not needed for the Frechet example.
- Outer solve performance (unlimited `maxit`) is the main scaling concern for larger D.
