# Experiment findings (live notes) — derivatives & KNITRO algo + profiling

Branch: `experiments-derivatives` (baseline snapshot committed at `941ecf5`).
Config unless noted: 4-country unrestricted, W=8000, Jac_W=8000, δ=1, autarky, csw outer opt
(algorithm auto, hessopt auto=BFGS, maxit 25, feas/opt tol 1e-6, eval_fcga yes), inner ek opt
(algorithm 0, gradopt/hessopt exact, maxit 100, tol 1e-12).
BASELINE bounds (committed): κ=[0.007437, 0.219549]; total wall ~257s (with debug @show).

## Code changes made (all reversible, on experiments branch)
1. moments/moments!.jl: `Aod`/`AodPow`/`Aod_θ` allocated with `ones(eltype(θ),D,D)` instead of
   `ones(D,D)` → moments! now accepts ForwardDiff Duals (A1 fix). Float64 path unchanged.
2. moments/hFunction_jacobian.jl: SmoothDirac width β now ENV-overridable (`EXP_BETA`, default 0.01).
3. cc_algo/inner_loop_functions.jl: removed per-inner-solve `@show nStatus/@show objSol` debug prints;
   added instrumentation counters (INNER_SOLVE_COUNT/INFEAS/ITERS) + Ref-based KNITRO stat getters.
4. cc_algo/outer_loop_functions.jl: OUTER_SOLVE summary line (status/iters/FCevals/feas_err/opt_err/
   time/inner-solve counts/obj) per outer solve.
5. run_master.jl (new), scratch_exp/*.sh (harness), test_jac.jl / profile_inner.jl (standalone tests).

## A1 — ForwardDiff compatibility: SUCCESS
- moments! runs under Duals after the one-line eltype fix. The doc's "broken autodiff path" is fixed.
- grad_k (∂k/∂θ): analytic vs autodiff match to ~1e-8 on ALL FREE variables. Only diff (0.0189) is
  on θ[2]=σ, which is FIXED in the outer loop (bounds equal) → irrelevant.
- Speed (per call, θ has l=23, d=24 moments, N=8000 draws):
    grad_k:  analytic 0.63 ms   autodiff 0.47 ms  (autodiff slightly FASTER)
    jac_θ:   analytic 514 ms     autodiff 802 ms   (autodiff ~1.5x slower)

## A2/A3 — the analytic Jacobian's SmoothDirac term vs ForwardDiff (no Dirac): THE KEY FINDING
- The forward moment map (hFunction!) already uses the EXACT indicator MinInd! (smooth min is
  commented out). SmoothDirac only appears in the analytic *Jacobian* (hFunction_jacobian.jl:117/119),
  where it approximates ∂/∂θ of the argmin-selection indicator (a Dirac at ties).
- ForwardDiff differentiates the exact indicator → piecewise-constant → derivative 0 a.e. → it OMITS
  the Dirac boundary term. So autodiff jac and analytic jac DIFFER by exactly this term:
    per-draw |Δjac_h| up to 192 (huge, spikes at near-tie draws); mean |jac_h| 0.133 (ana) vs 0.112 (ad).
    Discrepancy concentrated in col 1 (μ) and cols 8-23 (free Aod params).
- FULL-RUN BOUNDS (maxit 25, both hit iteration limit status=-400):
    analytic (with Dirac, β=0.01):  κ = [0.007437, 0.219549]   feas_err(upper)=0.0
    autodiff (no Dirac):            κ = [0.00124,  0.302701]    feas_err=??? (pending)
  => The smoothed Diracs DO change the bounds materially (κ_upper 0.2195 -> 0.3027, +38%).
  Interpretation pending the feasibility check (below) + β sweep + convergence runs.

  DISCRIMINATOR RESOLVED (maxit 25, both status=-400 unconverged but feasibility is decisive):
    x_ana_m25      (with Dirac): κ=[0.007437, 0.219549]  feas_err=0.0/0.0  opt_err=1.03/1.41
    x_autodiff_m25 (no Dirac):   κ=[0.00124,  0.302701]  feas_err=0.0/0.0  opt_err=1.6 /0.117
  BOTH bounds are FEASIBLE (feas_err=0). The autodiff (exact-derivative, no-Dirac) bounds are WIDER
  and still feasible => the true κ_upper >= 0.3027, so the analytic Dirac bound 0.2195 is TOO NARROW.
  CONCLUSION: the smoothed Diracs are NOT buying accuracy; the spurious Dirac term in the analytic
  Jacobian biases the constraint gradients and keeps KNITRO from reaching the more-extreme feasible
  optima, UNDER-STATING model sensitivity (bad for a paper whose point is to quantify sensitivity).
  Recommendation forming: drop the Dirac / use ForwardDiff on the exact indicator (use_Jacobian=0).
  (Confirm with maxit-100 convergence runs: do both settle, and does the ranking hold.)

  MAXIT-100 UPDATE: x_ana_m100 upper STILL status=-400 at 100 iters (opt_err 0.503, feas 0.0),
  κ_upper climbed 0.2195 (m25) -> 0.2679 (m100), STILL widening. So the analytic/Dirac path is not
  converging to a different answer — it converges SLOWLY (Dirac-perturbed gradients), so at any fixed
  iteration budget it under-reports the bound. Autodiff hit 0.3027 in just 25 iters. => exact-derivative
  (autodiff) both converges faster AND is unbiased. Awaiting x_autodiff_m100 to see if it settles >=0.3027.
  (Analytic m100 upper cost: 383s knitro, 870 FCevals, 290 infeasible inner.)

## B2 — solver behaviour (from instrumented baseline, maxit 25)
- Both outer solves hit status=-400 (KNITRO ITERATION LIMIT). The "baseline bounds" are maxit-25-
  CAPPED, not tolerance-converged (opt_err ~1.0). Convergence runs (maxit 100) queued.
- ~1 inner solve per outer FC eval (188 FCevals -> 188 inner solves).
- 47-63% of inner solves return INFEASIBLE (θ exceeds divergence budget): upper 89/188, lower 120/192.
  Expected, but quantified — half the inner solves are "wasted" exploring infeasible θ.
- ~6-7 inner KNITRO iters per inner solve (1280 iters / 188 solves).
- Removing debug @show: outer-solve elapsed 178.6s (vs 257s total with @show incl prep). Need clean
  A/B on @show cost (TBD in B1).

## Pending (phase2 queue, ~2h): 
- x_ana_m25 / x_autodiff_m25 (feas_err), x_ana_m100 / x_autodiff_m100 (convergence)
- a3 β-sweep {1e-3,1e-1,1e0,1e2}; a2 hessopt {prodfd(4),sr1,lbfgs,bfgs}; a4 outer algos {1,2,3,4};
  a4 inner algos {1,3}; conv_prodfd_m100
## Pending (after phase2): B1 profiling (profile_inner.jl); A2 exact objective-Hessian callback attempt.

## PHASE2 HARVEST (all maxit 25 unless noted; analytic Jac unless "autodiff")
CONVERGENCE (maxit 100): autodiff κ=[0.001051, 0.303174] (upper ~converged, status -102);
  analytic κ=[0.005804, 0.267872] (upper still climbing, status -400). Autodiff wider & more converged.
A3 β-sweep (analytic Jac): β=1e-3 [0.008393,0.230813]; β=1e-2 [0.007437,0.219549]; β=1e-1 UP FAILED(1e10);
  β=1 [0.003245,0.242929]; β=1e2 UP FAILED(1e10). Fragile/noisy; no β reproduces autodiff 0.303.
A2 hessopt (analytic Jac, m25): auto/bfgs κ_up=0.2195 opt_err 1.03; product_findiff(4) κ_up=0.2373 opt_err 0.394;
  lbfgs(6) = prodfd (0.2373); sr1(3) up=0.2419 opt_err 0.163 but LO erratic 0.0288. => prodfd/lbfgs > bfgs.
  conv_prodfd_m100 κ=[0.00538,0.261537]: with analytic Jac, Hessian option can't overcome the Dirac-biased
  gradient (still ~0.26 < autodiff 0.303). => GRADIENT quality dominates Hessian option.
A4 outer algo (m25): auto=direct 0.2195; cg narrower 0.1849 (fast, few infeas); active up=0.2427 opt_err 0.116
  (fast 70s) but LOWER INFEASIBLE status -410; sqp up=0.1843, lower infeasible. => interior/direct robust;
  active-set best-but-fragile.
A4 inner algo: direct/active IDENTICAL to baseline => inner algorithm choice is a no-op.
RECOMMENDATION forming: use_Jacobian=0 (autodiff) + hessopt=4 (product_findiff) + algorithm auto/direct;
  raise maxit for real bounds. Inner solver settings fine as-is.

## PHASE3
rec_ad_prodfd_m25 (autodiff + product_findiff, m25): κ=[0.007778, 0.300082], UP opt_err 0.128 (vs
  autodiff-bfgs 1.6, analytic-bfgs 1.03) => combo converges upper to ~0.300 in just 25 iters. Best combo.

rec_ad_prodfd_m100 (autodiff+prodfd, m100): κ=[0.004306, 0.301852]. Upper ~0.302 (=autodiff-bfgs m100).
  prodfd helps EARLY (m25 opt_err 0.128) but by m100 converges same as bfgs. Lower bound wobbles 0.001-0.008.
A2 EXACT objective-Hessian callback (hessopt=1, obj-Hessian only, zero constraint curvature, analytic Jac, m100):
  κ=[0.010826, 0.211653]. Upper 0.2117 is NARROWER than bfgs m100 (0.2679) and even prodfd (0.2615)!
  => providing exact OBJECTIVE Hessian but omitting (uncomputable) CONSTRAINT curvature MISLEADS KNITRO.
  Definitive: exact-obj-Hessian HURTS; product_findiff (hessopt=4) is the right practical choice.
B1 PROFILE (warm, M=8000): EK_moments! 20.5ms/5.8MB; Psi/dPsi/ddPsi ~75us/0-alloc (fine);
  inner Q(x,g) 0.334ms; inner_loop_internal ~561ms (avg in-loop 99s/188=0.53s, matches) — ~0.5s is
  KNITRO-INTERNAL (NOT param load: KN_new+load 29KB opt+free = 1.1ms, param load only 0.88ms => NOT a
  bottleneck, hypothesis REJECTED); calculate_jac_θ! (analytic) 370ms + 80MB alloc/call = the outer-grad
  cost center (190 calls/solve => 70s + 15GB GC churn); calculate_grad_k! 0.44ms (fine).
B1/B2 LEVERS: (1) ~50-63% of inner solves infeasible, each ~0.5s => ~half the outer solve wasted probing
  outside the divergence budget; active-set outer had 25 vs 89 infeasible & was faster (70 vs 100s).
  (2) calculate_jac_θ! 80MB alloc/call — preallocate temporaries. (3) EK_moments! 5.8MB/call — preallocate
  UPow/UσPow. (4) removed per-solve @show. NOT a lever: param-file reload, inner algorithm, Psi funcs.

A4 multi=5: SIGSEGV/core dump (exit 139) — unusable. code_warntype: EK_moments! & inner Q both
type-stable (0 ::Any) => allocs are real temporaries, not instability.
