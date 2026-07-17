# Full-A D=4 exact-formulation code audit

Diagnostic branch `diag/fullA-d4-exact`, worktree `../gravity-fullA-d4`, branched from
production commit `53ffb58e8d9b18498279fab25da4d1b7cc47556a` (branch `sequential-profiled-gravity`
in `trade_robustness_modular`) on 2026-07-17. Full environment snapshot: `environment.txt`
(Julia 1.12.6, KNITRO 13.0.1 on demand.mit.edu, 208 cores, single-threaded Julia by default).

This audit covers **only** the true full-A method: every `A_od` entry (D² of them) free in the
outer KNITRO loop, as distinct from the *sequential/profiled* method that `docs/reference/
sequential_methodology.pdf` documents (which keeps only the focal column, D+1 free params, and
eliminates gravity via a linearized influence-function moment re-solved every sequential-loop
iterate). The two methods share almost no code below the outer-loop level except the moment
kernel (`hFunction!`) and the CC inner-dual machinery (`cc_algo/`).

## 1. Where the newest full-A code lives

All in `full_aod_diag/` (dates below are file mtimes at branch-off, not necessarily commit dates —
git blame was not separately run per-file given the volume; the directory as a whole was last
touched in commits up to `53ffb58`):

| Role | File | Key symbol(s) |
|---|---|---|
| Moment kernel (γ_d≡1 gauge, direct-γ' objective) | `full_aod_diag/moments_gammanorm.jl` | `EK_moments_gammanorm_directgp!`, `EK_moments_gammanorm!`, `theoretical_gammaprime_bounds`, `build_theta_gammanorm` |
| Production D=4 driver (cached, free-only) | `full_aod_diag/run_fullA_D4_production.jl` | wires `FreeParamMap` + `OuterEvalCache` + closed-form gravity gradient |
| D=10-named but D-general driver | `full_aod_diag/run_fullA_D10_production.jl` | same pattern, hardcodes `fakeData=1`/`W=8000` (not env-wired — a known gap per prior-session handoff notes) |
| Method-B diagnostic bundle (no cache) | `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl` | `PsiObjectiveBundleImplicitMethodBFullA`, `_methodB_fullA_envelope_scalar`, `make_gravity_grad` |
| Closed-form gravity value/gradient | `full_aod_diag/gravity_tariff.jl` | `gravity_value`, `gravity_grad_free!`, `precompute_q_tilde` |
| Free/fixed parameter split | `cc_algo/free_param_map.jl` | `FreeParamMap`, `pack_free`, `reconstruct_full`, `round_trip_check` |
| Exact-point outer cache | `cc_algo/outer_eval_cache.jl` | `OuterEvalCache`, `ensure_inner!`, `ensure_grad!` |
| Cached outer-loop driver | `cc_algo/outer_loop_cached.jl` | `outer_loop_cached` |
| Envelope-scalar core (Method B) | `full_aod_diag/ad_benchmark/derivative_core.jl` | `envelope_scalar_div_ctx`, `moment_map!` |
| Shared setup / data context | `full_aod_diag/ad_benchmark/setup_context.jl` | `AD_PARAMS`, `build_ad_context`, `make_ad_obj` |
| Prior audit notes (this session builds on, does not duplicate) | `full_aod_diag/ad_benchmark/call_graph_audit.md`, `derivative_formulas.md`, `README.md` | — |
| Prior validated tests | `full_aod_diag/test_free_param_and_gravity.jl`, `validate_free_only_gradient_fullA.jl` (+ `.log`s) | round-trip, gravity-gradient FD, free-only-vs-dense-Jacobian gradient equivalence — all previously PASS |
| Governing session summary | `full_aod_diag/SESSION_SUMMARY_2026-07-12.md` | single source of truth for the γ_d≡1 normalization change |
| Conditioning study (OLD normalization) | `full_aod_diag/report.md`, `cond_diag.jl` | superseded numbers, kept for history |
| Winner-boundary hard branch | `misc/smoothMinIndNew!.jl` | `MinInd!` (line 28-36) — the non-differentiable Bool branch |
| Moment kernel that calls it | `moments/hFunction.jl` | `hFunction!` line 69, `hFunctionCounter!` line 157 |
| Known-correct-but-unwired analytic Jacobian | `moments/hFunction_jacobian.jl` | SmoothDirac-corrected; **not reachable from the full-A driver** (`use_Jacobian=0` hardcoded in `AD_PARAMS`) |

## 2. Data / draws / configuration used by this code path

`full_aod_diag/ad_benchmark/setup_context.jl::AD_PARAMS`: **synthetic** economy
(`fakeData=1, DFake=4, seedFakeData=889, seedU=888`), `σHat=2.5`, `baseIndex=2`, `W=Jac_W=8000`,
`counterType=1` (autarky), `gravMoment=1`, `OuterScaling=1`, `use_Jacobian=0`, `δ_ref=1`. This is
**not** the real BACI/Teti tariff data — it is the repo's standard synthetic D=4 test economy used
throughout the prior full-A diagnostic sessions (`full_aod_diag/report.md`,
`SESSION_SUMMARY_2026-07-12.md`). Per memory `d20-realdata-w-sensitivity`, W=8000 is known to
understate kappa relative to W≥80,000 at δ≥1 on the *real* data; whether the same holds on this
synthetic D=4 economy is untested and out of scope for this methodology audit unless flagged
otherwise. All numerics in this report should be read as "does the exact full-A optimization
method work reliably," not "what is the calibrated gains-from-trade number."

## 3. Free-parameter count — verified from code, not assumed

`run_fullA_D4_production.jl:53-61`:
```julia
free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))   # γ'_focal + all D² A_od entries
fixed_idx = vcat(1, 2, collect(3:2+D))                          # μ, σ, and the D now-inert γ_θ slots
m = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
@assert CS.n_free(m) == 1 + D^2
```
At D=4: **n_free = 17** (1 γ'_focal + 16 A_od entries), confirmed by a passing assertion and by
`test_free_param_and_gravity.log` ("n_free = 17 (expect 17)", round-trip PASS). This is **not**
D²−D (the PDF's own quoted figure for a *destination-column-pinned* gauge) — it is D², because
`moments_gammanorm.jl` (§4 below) removes the `A[1,d]=1` pin from **every** column, not just the
focal one. μ and σ are held fixed (equal KNITRO bounds) and structurally excluded from
`free_idx`, so they never appear in ForwardDiff's Dual-partial tuples either (this is the
"free-only ForwardDiff" optimization the driver's header comment describes). Gravity is **not**
eliminated at this stage — see §5.

## 4. Normalization actually implemented — resolves the PDF's flagged ambiguity

The task brief flagged two inconsistent dimensionality statements in the methodology PDF (D²−D
free after a per-column `A[1,d]=1` pin, vs. `γ_focal≡1` leaving all D focal-column entries free)
and asked which one the code actually uses for full-A.

**Verified from `moments_gammanorm.jl`**: neither, exactly — the code generalizes the second
convention to *every* destination, not just focal. `EK_moments_gammanorm_directgp!` (called by
this driver) hardcodes `γ = ones(T, D)` for **all** d (line 247: old `γ_θ` slots read but ignored,
comment at line 79/222 says so explicitly) and leaves the entire `A_od` matrix
(`θ[Aod_offset+1:Aod_offset+D²]`) free — no `A[1,d]=1` pin anywhere. This is a genuine
reparameterization, not cosmetic: `γ[d]` enters `denom[d] = γ[d]^σ * gdp[d]`
(`hFunction.jl:37`), the normalizer for **every** destination-d trade-share moment, for every d —
so fixing `γ_d≡1` for all d changes the whole moment map, and the compensating degree of freedom
shows up as `A[1,d]` becoming free instead of pinned. `κ` remains a strictly monotone transform of
`γ'_focal` alone (only baseIndex's γ/γ' pair is economically free; `EK_moments_gammanorm_directgp!`
makes `K ≡ θ[3+D]` directly, skipping the `σ/(σ-1)`-power transform of `EK_moments_gammanorm!`).
This matches memory `gamma-d-normalization-and-direct-gp` and is the best-conditioned full-A
variant found in prior sessions (D=4 opt_err improved 0.598→0.0039 lower / 1.51→0.0087 upper in
that session's own run — **not yet reproduced or independently re-verified this session**, see §7
open items).

## 5. Gravity: currently a second explicit KNITRO constraint, NOT eliminated

`PsiObjectiveBundleImplicitMethodBFullA`/`outer_loop_cached` treat gravity as `constr[2]`, a
second outer equality constraint solved directly by KNITRO's SQP alongside the divergence-budget
inequality — `obj.outer_constr_index == obj.d` asserted in the driver (exactly one extra moment).
Its gradient is closed-form and exact (`gravity_tariff.jl`, elementwise, no ForwardDiff, no draws
loop — verified against finite differences, relerr 2.5e-10, `test_free_param_and_gravity.log`).
This is architecturally different from the *sequential* method's approach (linearize gravity into
an extra CC moment, re-solved every sequential-loop iterate — PDF §6-9) and from what the task
brief's §10 asks for (reduce dimensionality via sparse pivot elimination or an orthonormal
nullspace parameterization, dropping gravity as an explicit KNITRO constraint entirely). **Exact
gravity elimination is new work for this investigation, not something already implemented for
full-A.**

## 6. THE CENTRAL OPEN ISSUE: winner-boundary derivative bug is confirmed present and unfixed here

Per prior-session diagnosis (memory `full-a-winner-boundary-derivative-bug`, done at D=5 in a
different worktree, diagnosis only, no fix), the full-A divergence-budget gradient differentiates
`ForwardDiff.gradient` through `hFunction!`'s winner-selection call, which is the **hard** branch:

```julia
# misc/smoothMinIndNew!.jl:28-36
function MinInd!(xInd, x, D)
    xMin = minimum(x)
    for i = 1:D
        xInd[i] = (x[i] > xMin) ? 0 : 1     # hard Bool, zero a.e. derivative, undefined at ties
    end
end
```
called from `hFunction!` line 69 and `hFunctionCounter!` line 157 — i.e. **inside** the exact
function (`EK_moments_gammanorm_directgp!`) that both `envelope_scalar_div_ctx` (in
`run_fullA_D4_production.jl`'s `make_div_grad_fn!`) and `_methodB_fullA_envelope_scalar` (in
`PsiObjectiveBundleImplicitMethodB_fullA.jl`) call `ForwardDiff.gradient` over. **This is
independently confirmed by direct code inspection this session** (not re-trusting the D=5 memory
finding): the smoothed variant (`smoothMinIndNew!`) exists in the same file and is never called by
either of these two paths; only the hard `MinInd!` is used. The Method B envelope-scalar approach
being algebraically identical to a raw ForwardDiff pass over the dual objective at fixed
`(ζ*, λ*)` (validated relerr ~1e-15 against the dense-Jacobian production path, per
`ad_benchmark/correctness_results.csv` and `validate_free_only_gradient_fullA.log`) does **not**
fix this — that validation only confirms Method B **reproduces production's existing (buggy)
gradient exactly**, not that production's gradient is the true derivative. The three finite-step
checks in `validate_free_only_gradient_fullA.log` (relerr ~1e-11 to 2e-13) are directional central
differences at `h` small enough that, in a finite-W=8000 sample, few or no draws switch winner —
exactly the finite-sample floor the D=5 memory already documented as consistent with, not
contradicting, the boundary-term bug (small-h FD and a derivative that's missing an
O(switch-probability) term agree until h is large enough for switches to register).

**Important scoping distinction from the codebase's existing "fixed" work**: the memory entries
`sequential-winner-boundary-derivative-fix` and `full-d2-winner-boundary-fix` describe a
validated, wired-in correction — but for the **sequential/profiled** method's reduced `D+2`-moment
problem (only the focal column's `D` entries + γ'_focal free, `n_free=D+1`). That fix
(`full_fixed_dual_criterion.jl`, `full_gradient_method_wiring.jl`) is **not applicable as-is** to
this full-A path: the moment count, moment definitions, and free-parameter vector are entirely
different (`d=18` full moments incl. gravity vs. the sequential method's `D+2=6`; `n_free=17` vs.
`D+1=5`). Building the analogous `fixed_dual`/`optimized`/`frozen-adjoint` machinery for the true
full-A problem is core new work for this investigation (task §4, §12).

## 7. KNITRO options — confirmed silent Hessian-mode fallback

`full_aod_diag/csw_outer_25.opt` (outer solve): `algorithm auto`, `gradopt exact`, **`hessopt 4`**
(requested: BFGS) but also `eval_fcga yes`. Per memory `full-aod-conditioning` (independently
re-confirmed by reading the opt file directly this session, not just citing the memory):
`eval_fcga yes` blocks KNITRO's product-finite-difference machinery that `hessopt 4` needs, so
KNITRO silently falls back to `hessopt 6` (L-BFGS) — this is exactly the failure mode task §16
warns about ("previous runs may have silently used L-BFGS rather than the intended Hessian
option"). **Not yet re-verified from a fresh KNITRO log this session** (next step: run with
`outlev` raised and grep for KNITRO's own "Changing hessopt" message). `maxit 25` (outer),
`opttol 1e-06`. `full_aod_diag/ek_inner.opt` (inner CC dual): `algorithm 0` (auto), `gradopt exact`,
`hessopt exact` (the small inner dual problem *can* afford an exact Hessian), `opttol 1e-12`,
`maxit 100` — these look appropriate and are not flagged as a problem.

## 8. Previous D=4 full-A output — inconclusive, not yet a clean baseline

`full_aod_diag/run_fullA_D4_production.log` (existing run, this driver): both the cached/free-only
path ("new") and the reference full-theta-AD path ("ref") terminate at **status=-400** (KNITRO
non-convergent termination — hit the `maxit=25` outer-iteration cap without reaching KKT
tolerance; `opt_err≈0.017` for the ref path, not near zero) and **disagree with each other** by
Δκ≈0.015 (new=0.1786 vs ref=0.1636) — i.e. this existing log is **not** a validated, converged D=4
result and should not be treated as a baseline number; it demonstrates the driver runs end-to-end,
nothing more. This is a different (worse) opt_err than the `gamma-d-normalization-and-direct-gp`
memory's own headline D=4 numbers (0.0039 lower / 0.0087 upper) — the discrepancy is unexplained
(possibly a different δ, bound direction, or maxit) and is an open item for §18 (short D=4 runs)
to reconcile with fresh, fully-logged, reproducible runs under this branch.

## 9. Discrepancies with `sequential_methodology.pdf`

1. The PDF documents the **sequential/profiled** method only; it explicitly frames full-A as the
   non-scaling comparison method it exists to justify avoiding (§1: "This is exactly the
   'full-A-in-outer-loop' comparison method run alongside the sequential method as a cross-check").
   It contains no description of full-A's own gravity handling, gauge, or gradient — everything in
   this audit's §3-7 had to be independently derived from `full_aod_diag/` code, not read off the
   PDF.
2. The PDF's γ-gauge discussion (§2.3, §4) is scoped to the focal column only (`γ_focal≡1`,
   `A_{1,focal}=1` pin replaced). Full-A's actual code (§4 above) generalizes this to *every*
   column (`γ_d≡1 ∀d`) — consistent in spirit, not literally described.
3. The PDF's outer gradient correction (§10.1, `gradient_method=fixed_dual_fd_full`) is for the
   **sequential** method's `D+2`-moment problem. No equivalent correction exists yet for full-A's
   `d`-moment problem (§6 above) — this is the primary gap this investigation must close or clearly
   characterize as unfixable at reasonable cost.

## 10. Immediate next steps (feeds §6-7 of the task brief)

1. Build the machine-generated parameter table + pack/unpack/perturbation tests (extends the
   already-passing `FreeParamMap` round-trip test to the full task-mandated coverage: duplicate-
   mapping detection, unused-coordinate detection, per-coordinate perturbation effect on the full
   moment system).
2. Wrap `run_fullA_D4_production.jl`'s machinery into the `evaluate_fullA` oracle (task §7) —
   reuse `FreeParamMap`, `OuterEvalCache`'s inner-solve/gradient caching pattern (add exact-value
   provenance fields: inner status, KKT residual, cache hit/miss, warm/cold, elapsed by component),
   rather than rebuilding equivalent machinery from scratch.
3. Implement the three-way derivative distinction (frozen-adjoint / fixed-dual / optimized-value)
   for the true full-A `d`-moment problem, analogous to (but not reusing, per §6 above)
   `full_fixed_dual_criterion.jl`.
