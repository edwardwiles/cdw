# Sequentially-linearized profiled full-gravity — progress note

Branch: **`sequential-profiled-gravity`** (off `experiments-derivatives`, in `trade_robustness_modular`).
Fully reversible: `git checkout experiments-derivatives` restores the prior state. Nothing here
touches the live `master.jl` pipeline; all new code is under `sequential_gravity/` and gated in
standalone scripts. Companion docs: **`SEQUENTIAL_GRAVITY_DESIGN.md`** (the spec→code derivation and
the §18 integration limits); the Claude memory note `sequential-profiled-gravity`.

Everything below is on the **D=4 simulated example** (focal country = `baseIndex=2`), not real data.

---

## 1. The problem this solves

The distribution-agnostic bounds put the structural parameters θ in an outer optimizer. All bilateral
shifters `A[o,d]` currently live in θ: 16 entries (12 free) at D=4, but **361 (~342 free) at D=19** —
the outer bilevel program does not scale. The coauthor's "France-only gravity moment" compromise
(keep only destination = France rows, drop the other `A` columns) is too noisy.

**This approach** keeps only `A[·,focal]` in θ and *profiles out* every other column: for each omitted
destination d, recover `A[·,d]` by **inverting** its observed trade-share column under the current
least-favorable distribution F. The full origin+destination-FE gravity restriction `R = 0` is then
imposed via a **sequentially-linearized** moment (an influence function), so gravity can shape which F
CC selects — without carrying the omitted A's as parameters.

---

## 2. Exact model ↔ ChatGPT-spec mapping (derived and verified)

For draw s and origin o (`UoModel=1`, `θConstant=0`):
- `log_x[s,o] = μ·log(Uσ[s,o]) = μ(1−σ)·log(U[s,o])`  (destination-independent; U~Exp(1), Uσ=U^{1−σ})
- `u[o,d] = (σ−1)(log A_od − log w_o − log τ_od)`  — the spec's competitiveness index exactly, using
  the structural `A_od = 1/AodPow`
- winner(s,d) = `argmax_o (u[o,d] + log_x[s,o])` = argmin level price
- `share_d[o] = Σ_s p[s]·(winner value) / Σ_s p[s]·(value)` — matched to the observed column `λ̂[·,d]`
- LFD weights recovered exactly as in `lfd/LFD.jl:50-53` (`dPsi!` of the dual solution).

This mapping is the foundation for everything else.

---

## 3. What was built, phase by phase (all committed)

| phase | deliverable | status |
|---|---|---|
| 1 | `sequential_gravity/profiled_gravity.jl` — inversion `I_d`, gravity residual `R`, share-Jacobian `H_d`, adjoint, influence function ψ̄ | **validated** |
| 1 | `test_profiled_gravity.jl` (synthetic) + `validate_on_pipeline.jl` (real draws) | **8/8 tests pass** |
| 2a | `focal_moments.jl` — reduced θ (D+4) & inner moments (D+1) | **==full focal columns to 2e-14** |
| 2a | `run_focal_bounds.jl` — reduced focal-only CC solve | **runs, bounds** |
| 2b | `run_sequential.jl` — sequential loop at fixed θ | **R→0 in 2 iters** |
| 2c | `run_profiled_bounds.jl` — nested §18 outer integration | **works, gravity-consistent** |
| 2c | `time_pieces.jl` — head-to-head timing | **measured** |

### Phase 1 — the linearization core (the mathematically risky part)
Carries a softmax temperature ρ throughout (ρ→0 = hard max) so the finite-sample winner-switch
graininess is smoothed; the influence function ψ_R is then the **exact** derivative of the smoothed
residual. Tests C–J all pass, including the **mandatory directional-derivative gate** (ψ_R vs exact
re-inversion): rel err ~1e-4 synthetic, **3.8e-5 on real pipeline draws**. On the real 4-country data,
recovered `u` = the calibrated `(σ−1)(logA−logw−logτ)` within the finite-draw MC error (1.8e-2 ≈
1/√8000). This is the key scientific validation — the whole method rests on this influence function
being right, and it is.

### Phase 2a — reduced focal-only CC mode (the spec's hidden prerequisite)
`EK_moments_focal!` builds the reduced problem: outer θ = `[μ, σ, γ_focal, γ'_focal, A[·,focal]]`
(D+4 params vs D²+3), inner moments = D focal trade shares + 1 counterfactual (vs D²+1). Unit-tested
to reproduce the full `EK_moments!` focal columns to ~2e-14 at θ_initial and 5 perturbed θ. Plugs into
the **existing** CC solver unchanged (ForwardDiff, no hand-written reduced Jacobian).

### Phase 2b — the sequential loop at a fixed θ (spec §14)
Recover LFD → invert omitted → build ψ̄ → add one moment `E_F[ψ̄]=−R_k` → re-solve → re-invert → damp.
At θ_initial it drives the **exact** residual `R_beta` from 8.4e-3 → 1.7e-5 in **2 iterations**
(full α=1 steps), omitted share error 3.7e-12.

### Phase 2c — nested outer integration (spec §18)
A stateful moment closure recomputes the sequential loop (Δ_sequential) at **every θ** the outer
optimizer visits. Correctness rests on three robustness rules (see §5 and DESIGN §5b):
gravity-infeasible θ → unsatisfiable `INFCOL` moment → outer rejects; inversion divergence guard;
trust the linearized column only when the loop converged.

---

## 4. Results (D=4, δ=1)

**maxit-25 (iteration-capped, status −400 — not converged):**

| config | κ_lower | κ_upper | exact R at θ* |
|---|---|---|---|
| focal-only, no gravity | 0.00055 | 0.2326 | — |
| **profiled full-gravity** | **0.0107** | **0.0990** | ~4e-4 (both) ✓ |
| legacy all-A gravity (reference) | 0.0102 | 0.1595 | — |

The profiled lower bound (0.0107) **matches the legacy all-A method (0.0102)** — the cross-validation
we wanted: profiling the omitted A's out gives the same feasible set as carrying all of them. The
gravity residual at both bound-achieving θ* is ~4e-4 (gravity genuinely satisfied). The upper-bound gap
(0.099 vs 0.160) is most likely iteration-capping; **maxit-100 runs are in progress** to check whether
the two methods reconcile.

**Timing primitives (real pipeline, W=8000, D=4):** one destination inversion — cold 22 ms, warm
~8 ms — is **comparable to one min-divergence CC solve (~7 ms)**. (The inversion was ~350 ms before the
speedups: `dest_share` avoids the S×D allocation, and in the smooth ρ>0 regime a Newton +
monotone-decrease backtrack replaces the ~80-bisection exact line search — cold 350 ms → 22 ms.)

**Speed caveat, honestly:** at D=4 the profiled method is *slower* than all-A (per-θ sequential-loop
overhead when the all-A outer is only 23 params). The payoff is at D=19, where all-A has 361 outer
params and profiled has ~19. That crossover is untested here.

---

## 5. Conceptual points worth recording (came up in review)

- **What F is linearized around:** the min-divergence LFD that matches the *focal* moments at θ
  (≈ F* at θ_initial), per spec §14 — not F* and not the counterfactual-extremizing LFD.
- **Why the loop inverts ~twice per iteration:** it is not redundant — spec §14/§15. Invert at the
  current LFD to *build* ψ̄, then re-invert at the candidate LFD to *verify* the exact residual (and,
  under damping, to check `|R_α|<|R_k|`). α=1 is usually accepted, so damping adds no extra inversions.
- **Why "reuse last column on failure" was wrong (the R≈0.99 bug):** the linearized moment
  `E_F[ψ̄]=−R` is only a valid *local* surrogate for gravity when R is already small. When gravity is
  unachievable at a θ (its focal A column is gravity-incompatible; residual stalls at ≈−1.4), reusing a
  stale column let the outer "match gravity" trivially and accept a θ where the true R was 0.99. Fix:
  such θ are gravity-**infeasible** and must be rejected (`INFCOL`).
- **"Diverge before rejected" = the inversion, not the min-div solver.** At extreme θ (μ→0, origins
  near-tied ⇒ shares insensitive to u) the inversion Newton iterates run off to huge u → huge R. The
  min-div solve is fast (~7 ms) and cleanly succeeds/fails. Guarded now (`‖u‖>1e8` → bail).
- **Why the outer gradient is only approximate for gravity, though it's exact for every other moment:**
  the ordinary moments are explicit closed forms `g(θ,U)` — ForwardDiff gives ∂g/∂θ, and the *one*
  inner CC solve's θ-dependence is handled by the implicit function theorem (`ift!`), never by autodiff
  through KNITRO. The profiled gravity "moment" is instead the **output of a nested optimization**
  (inversions + an interior min-div KNITRO solve), so extra solvers sit *inside* the moment where the
  single outer IFT doesn't reach. In the all-A method gravity *is* an explicit `g(θ,U)` (A's are
  parameters), which is why its gradient works there. The exact profiled gradient is the same *kind* of
  machinery (chain IFT through the inversion via the validated ∂u/∂F, and through the min-div solve via
  `ift!`) — real work, deferred until the maxit-100 timing shows how much the frozen gradient hurts.

---

## 6. Known limitations / open items

1. **Not converged:** the reported bounds are maxit-25 iteration-capped. maxit-100 runs pending.
2. **Approximate outer gradient** (gravity column frozen on the autodiff pass) — see §5. Likely the main
   reason the outer hits its iteration cap.
3. **Speed at scale:** D=4 is slower than all-A (overhead); D=19 is the target and is untested. Needs
   the inversions parallelized across the D−1 omitted destinations (embarrassingly parallel) and the
   exact gradient.
4. **Not wired into `master.jl`** behind an `orthogonality_mode` flag yet — it lives in standalone
   `sequential_gravity/` scripts.
5. **ρ (softmax temperature)** is used for the inversion + influence (ρ=2e-3 in the loop); the exact
   residual for the convergence check is evaluated at that ρ. ρ→0 is the strict hard-max; the O(ρ) bias
   is small but present.

---

## 7. File map & how to run

Under `sequential_gravity/`:
- `profiled_gravity.jl` — the Phase-1 core module (dependency-light: LinearAlgebra, ForwardDiff).
- `test_profiled_gravity.jl` — synthetic tests C–J. **No KNITRO:** `julia --project=. sequential_gravity/test_profiled_gravity.jl`.
- `validate_on_pipeline.jl` — Phase-1 core on the real draws (no outer solve).
- `focal_moments.jl` / `test_focal_moments.jl` — reduced focal-only moments + unit test.
- `run_focal_bounds.jl` — reduced focal-only CC bounds.
- `run_sequential.jl` — the sequential loop at a fixed θ (R→0 demo).
- `run_profiled_bounds.jl` — the full nested §18 profiled-gravity bounds.
- `time_pieces.jl` — inversion vs min-div-solve timing.

KNITRO-using scripts need the env (demand.mit.edu; see `SETUP_AND_FINDINGS.md`):
```
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
```
