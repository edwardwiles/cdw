# Full‑A_od outer loop — feasibility & conditioning diagnostics (D = 4)

**Question.** The CC outer loop puts *all* bilateral shifters A_od in the outer parameter
vector. At D = 4 this is 12 free A_od (of 18 free params total); at D = 20 it becomes ~361.
Is the all‑A_od‑in‑the‑outer‑loop approach fundamentally viable with good tuning, or is it
intrinsically ill‑conditioned/degenerate? This report **measures** the identification and
conditioning structure at D = 4 before any refactor, per the research prompt.

All work here is **additive and isolated**: new scripts under `full_aod_diag/`, frozen copies
of the KNITRO `.opt` files (`csw_outer.opt`, `csw_trace.opt`, `csw_autoscale.opt`, `ek_inner.opt`)
so runs are reproducible and never collide with the shared config the other working copy edits.
The reduced reference variant is produced with a flag (fix all A columns except the target
destination), **without touching the core `ccOuter.jl` path**.

Fixed config: 4‑country simulated (Frechet) example, autarky counterfactual, hybrid‑divergence
(Ψ) outer loop, gravity orthogonality ON as an outer constraint, W = Jac_W = 8000, δ = 1,
ForwardDiff envelope gradient through the exact hardmax, `algorithm auto`, maxit 100, and
`hessopt 4` **as written** in the `.opt` — though KNITRO silently runs LBFGS instead (see caveat below).

---

## TL;DR verdict

**The full‑A_od outer loop at D = 4 is not rank‑degenerate — it is ill‑conditioned, the
ill‑conditioning is entirely and only in the A_od block, and it already produces a demonstrably
wrong (too‑narrow) bound at D = 4.** Three findings:

1. **Conditioning.** The moment‑map Jacobian ∂E[g]/∂θ is full rank (18/18), but its spectrum splits
   cleanly into **6 large structural directions (μ, γ, γ′; σ ≈ 1.8–2.9) and a ~30–100× smaller A_od
   tail (σ ≈ 0.004–0.08)**. cond(J) = **755 (full) vs 173 (reduced)** — the non‑target A_od roughly
   **quadruple** the condition number at D = 4, and the gap grows with D². κ is **nearly flat** along
   the weak A_od directions (|∇κ·v| ≈ 10⁻³ vs ‖∇κ‖ = 2.5). The weak directions are the
   fixed‑effect / double‑demeaned part of ln A_od.

2. **This already bites at D = 4.** Both the full and reduced upper bounds fail to reach KKT
   tolerance even at maxit 100 (opt_err ≈ 0.11–0.17), and — the decisive result — **the full
   variant's lower bound is provably non‑optimal**: the reduced solve reaches a feasible point
   (κ = 0.00896) that is *also feasible for the full problem*, yet the full solve stalled at 0.01811
   and never got there. The extra A_od degrees of freedom **trap the solver on a flat plateau and
   yield a spuriously narrow interval**, not a wider one.

3. **Conditioning is real but *variable scaling does not fix it* — tested and refuted.** The A_od
   Jacobian columns are ~59× smaller than the structural ones, and equilibrating them (rescale each
   A parameter by 1/‖J col‖) drops the *Jacobian* condition number 755 → 36. But feeding those exact
   factors to KNITRO (`KN_set_var_scalings`) made the *solver worse*: the lower bound went
   **0.0181 → 0.0558** (further from truth) with opt_err rising 0.12 → 0.41. So the stall is **not** a
   pure variable‑scaling artifact. Reason: the outer objective κ is genuinely near‑flat in A and the
   only A‑curvature enters through the noisy inner solve; rescaling A *up* just lets the solver take
   bigger steps across that flat, noisy plateau and wander further. A pure ΔΔ rotation is inert
   (orthogonal ⇒ cond invariant). **And there is no identification‑neutral algebraic reduction of the
   A block**: wages are *pinned by data* (`w_o L_o = Σ_d λ_od w_d L_d`, λ and L observed), not free —
   so the origin‑levels of ln A_od are genuinely identified, not collinear with a free wage. Only the
   D *destination*‑levels are redundant (absorbed by γ_d, = the `A[1,d]=1` normalization). Hence at
   D = 20 there are **D(D−1) = 380 identified free A_od, not 361** — no "2D−1 flats" to remove.
   Dropping any A directions (as the reduced experiment does) is therefore a genuine *restriction* that
   changes the bounds, not a free reparametrization. The one identification‑preserving reduction is to
   **profile the non‑focal A out of the outer loop via the equilibrium + gravity constraints** (nested
   fixed point), which is a separate effort, not a tuning of this full‑A_od loop.

**Recommendation: the full‑A_od outer loop has no cheap rescue.** All 380 A_od at D = 20 are
identified (wages are data‑pinned, so no origin‑level flats); you cannot drop them without changing the
bounds, cannot out‑scale them (tested — it hurts), and cannot rotate them away (inert). The realistic
path is to **remove them from the outer search by *solving* for the non‑focal A from the model's
equilibrium + gravity conditions given the focal/structural parameters (a nested fixed point)** — the
identification‑preserving version of "reduction," and the separate focal‑only effort. Absent that,
expect the full loop to need very large `maxit`/tight tolerances and to still report a stalled (too
narrow) lower bound at fixed budget.

---

## What "full" vs "reduced" mean in this code

Parameter vector θ (length 23) for this config: `μ`(1, free, bounded 0<μ≤1/(σ−1)),
`σ`(2, fixed), `γ_θ[1..4]`(3–6, free), `γ′_θ`(7, free), `A[o,d]` (8–23, column‑major).
Normalization `A[1,d]=1` for each d is fixed in **every** variant (removes the per‑destination
column scaling redundancy). So:

| variant | free A_od | total free θ | what is fixed |
|---|---|---|---|
| **full** (default) | 12 (`A[o,d]`, o∈{2,3,4}, d∈{1..4}) | 18 | σ, A[1,·] |
| **reduced** | 3 (`A[o,2]`, o∈{2,3,4}) — target destination only | 9 | σ, A[1,·], A[·,d≠2] |

Moments (d = 18): 16 trade‑share + 1 counterfactual price‑index (baseIndex) matched in the inner
dual, + 1 gravity orthogonality (ΔΔ ln A ⟂ ΔΔ ln τ) as an **outer** equality constraint on θ.
The outer KNITRO program sees only 2 explicit constraints (divergence budget + gravity); the rich
identification structure lives in the moment map, whose Jacobian ∂E[g]/∂θ is the object analysed
below.

---

## Section 4 — Conditioning / degeneracy (the main event)

Moment‑map Jacobian **J = ∂E_{F*}[g]/∂θ** (18 moments × free params), by ForwardDiff at θ_initial,
averaged over the 8000 MC draws (this is the same per‑draw Jacobian the envelope gradient uses).

### 4a/4c — rank, spectrum, conditioning

| | full (18 free) | reduced (9 free) |
|---|---|---|
| rank (tol σmax·1e‑8) | **18 / 18** (full rank) | **9 / 9** (full rank) |
| σmax | 2.912 | 2.912 |
| σmin | 0.003859 | 0.01684 |
| **cond(J_free) = σmax/σmin** | **754.6** | **172.9** |
| near‑null dim (σ < σmax·1e‑6) | 0 | 0 |

There is **no exact null space**, and — importantly — this is *not* an artifact of D = 4. Only the
D *destination*‑levels of ln A are redundant (each column's common scale, absorbed by the free γ_d),
and `A[1,d]=1` removes them. The *origin*‑levels are **identified**, because wages are pinned by data
(`w_o L_o = Σ_d λ_od w_d L_d`) rather than free — there is no free wage to be collinear with, at any D.
So the classic "2D−1 fixed‑effect flats" never materialize: there are D redundancies (handled), not
2D−1, leaving **D(D−1) identified free A_od**. What remains is not degeneracy but a **spectral cliff**:
the 6 structural directions sit at σ ≈ 1.8–2.9; all A_od directions sit ≥ 30× lower, identified but weak.

**Full spectrum (log₁₀):** structural 0.46, 0.45, 0.43, 0.33, 0.29, 0.25 │ **cliff** │ A_od
−1.10, −1.11, −1.25, −1.37, −1.48, −1.51, −1.56, −1.77, −1.88, −1.94, −2.05, **−2.41**.
See `out/sv_spectrum.png`.

### 4a — what the weak directions are (FE structure of ln A_od)

Each of the 7 smallest singular directions is **100 % A‑energy** (structural components ≈ 0).
Decomposing them in d(ln A) coordinates on the free‑A support (origins 2–4 × destinations 1–4):

| σ_k | A‑energy | origin‑FE | dest‑FE | double‑demeaned |
|---|---|---|---|---|
| 0.00386 (weakest) | 1.00 | 0.03 | **0.86** | 0.01 |
| 0.00889 | 1.00 | 0.42 | 0.51 | 0.70 |
| 0.0116 | 1.00 | 0.43 | 0.52 | 0.69 |
| 0.0134 | 1.00 | 0.48 | 0.07 | 0.87 |
| 0.0169 | 1.00 | 0.20 | 0.81 | 0.27 |

The single weakest direction is 86 % destination‑FE (the residual per‑destination rescaling of
A[2:4,d] that the `A[1,d]=1` anchor makes weakly‑identified rather than exactly flat); the rest are
FE/double‑demeaned mixtures. **The weak block is exactly the fixed‑effect + low‑signal part of
ln A_od** the prompt anticipated — just weak, not zero.

### 4b — is the objective flat along the weak directions?

**Yes, nearly.** ‖∇κ (free)‖ = 2.533, but the projection of ∇κ onto each weak singular direction
is |∇κ·v| ≈ 5×10⁻⁴ … 8×10⁻³ — three orders of magnitude below ‖∇κ‖. So the adversary gains almost
nothing in κ by moving along the weakly‑identified A_od directions, yet the constraint (divergence
budget) is only weakly curved there. **That combination — flat objective + weak constraint
curvature along the same directions — is exactly what stalls the interior‑point solve** (it wanders
the A_od plateau making little progress), and it is why the extra A_od barely change the bounds.

### 4e — scaling by group

| group | |θ| (min/med/max) | |∂κ/∂θ| | ‖J col‖ |
|---|---|---|---|
| structural (μ,γ,γ′) | 0.17 / 0.88 / 0.97 | 0 / 0.19 / 1.81 | 1.77 / 2.43 / 2.91 |
| A_od | 1.0 / 1.0 / 1.0 | 0 / 0 / 0.05 | **0.028 / 0.041 / 0.092** |

Parameters themselves are all O(1) (A ≡ 1 at the start), so this is **not** a variable‑scaling
problem — it is a *sensitivity* mismatch: the moments respond ~30–100× less to each A_od than to a
structural parameter. Variable rescaling (KNITRO auto‑scaling, §5) cannot manufacture identification
that the moments do not carry; it can only help the linear algebra.

---

## Section 1 — Baseline full solve & reduced comparison

Frozen config (`csw_outer.opt`/`csw_trace.opt`, maxit 100, `algorithm auto`, `scale user_internal`,
gravity ON). Reference converged bounds from `EXPERIMENTS_FINDINGS.md`: **[0.018113, 0.154415]**.

| variant | κ_lower (status, opt_err) | κ_upper (status, opt_err) | width | wall | inner solves (up/lo) |
|---|---|---|---|---|---|
| **full** (12 free A) | 0.018113 (−101 **stall**, 0.117) | 0.154415 (−400 cap, 0.111) | 0.1363 | 232 s | 613 / 516 |
| **reduced** (3 free A) | **0.008955** (−102 conv, **0.023**) | 0.159036 (−400 cap, 0.171) | 0.1501 | 261 s | 716 / 634 |
| full + auto‑scaling | 0.018113 (−101, 0.117) | 0.154415 (−400, 0.111) | 0.1363 | 218 s | 613 / 516 |
| full + sensitivity‑scaling | 0.055807 (−400, 0.412) **worse** | 0.155832 (−400, 0.21) | 0.1000 | 297 s | 709 / 753 |

Status −400 = iteration cap; −101/−102 = feasible point that "cannot be improved" (a **stall**, not
KKT convergence — opt_err ≫ 1e‑6 in every case). The upper solve **never converges to KKT tolerance
even at maxit 100** (opt_err ≈ 0.11–0.17) in either variant.

### ⚠ Decisive finding — the full‑A_od lower bound is provably non‑optimal (it stalls)

The **reduced** solve finds a feasible point at **κ = 0.008955**. Fixing the non‑target A_od at 1
(what "reduced" does) is a *valid choice inside the full problem's feasible set*, so **that same
point is feasible for the full variant** — with identical divergence, gravity, and moments. Yet the
**full** solve reported a lower bound of **0.018113 > 0.008955**: it stalled on the ill‑conditioned
A_od plateau (status −101, opt_err 0.117) and **failed to reach a point its own feasible set
contains**. The extra weakly‑identified A_od directions did not widen the attainable bound — they
*trapped* the solver and produced a **spuriously narrow** answer. (See `out/convergence.png`: the
full‑lower trace flatlines at 0.0181 while the better‑conditioned reduced‑lower descends to 0.0090.)

This is the practical cost of the full‑A_od outer loop at fixed solver budget — and it is exactly
the mechanism Section 4 predicts (flat κ + weak constraint curvature along the A_od directions ⇒ the
interior‑point method wanders and stalls).

*(convergence traces: `out/convergence.png`; per‑solve summaries in `out/solve_*_summary.txt`.)*

### 4a at the solution (not just the start)

Recomputing J at the converged upper‑bound θ: cond drops to **482** (from 755 at θ_initial) but the
structure is unchanged — full rank, 6 structural directions over a weak A_od tail (σmin = 0.0065),
double‑demean reparam → cond **80**. So the ill‑conditioning is a property of the whole path, not an
artifact of the starting point.

---

### Config caveat discovered here — `hessopt 4` is NOT actually active

KNITRO's own log prints **"Changing hessopt to 6 (LBFGS)"** at the start of every outer solve:
with `eval_fcga yes` (objective and gradient returned in one combined callback), KNITRO cannot
form the separate gradient evaluations that `hessopt 4` (product finite‑difference Hessian‑vector
products) requires, so it silently falls back to **LBFGS**. So the current `.opt` is effectively
running LBFGS, *not* the product‑findiff Hessian the prior experiments recommended. **Probe
(`probe_*.opt`, maxit 2, banner only):** with `eval_fcga yes` → "Changing hessopt to 6"; with
`eval_fcga no` + `hessopt 4` → banner stays `hessopt: 4`, no downgrade. So `eval_fcga yes` is the
cause. Corroboration from `EXPERIMENTS_FINDINGS.md` A2: hessopt 4 and hessopt 6 there gave
**bit‑identical** results (0.005713 / 0.237256, opt_err 0.394) — the other working copy even noted
"ties product_findiff here." That identity is the fingerprint of 4 → 6: the "big difference hessopt 4
made" was really **LBFGS vs the auto/BFGS baseline** (a real gain, just mislabeled — LBFGS, not
product‑findiff, is doing the work). To test *genuine* product‑findiff, set `eval_fcga no`. Worth
re‑testing at D = 20, where curvature quality matters more.

## Section 2 — Time attribution (from instrumentation)

From the per‑solve counters (`>>> OUTER_SOLVE …` lines) and the profiling in `EXPERIMENTS_FINDINGS.md`:
the full upper solve ran **613 inner solves in 142 s** (≈ 0.23 s each), of which **~24 % were
infeasible‑by‑design** (θ outside the divergence budget). The dominant per‑evaluation cost is the
outer‑gradient moment Jacobian (`calculate_jac_θ` ≈ 0.4–0.8 ms × 8000 draws) plus the inner KNITRO
barrier iterations; constraint‑Jacobian assembly and outer linear algebra are negligible at D = 4.
At D = 4 **cost is not the binding constraint — conditioning is** (the solve hits the iteration cap
with opt_err ≈ 0.1, i.e. it runs out of iterations before KKT tolerance, not out of time).

---

## Section 5 — What actually conditions the problem (corrected)

An earlier draft claimed "double‑demeaning cuts cond 755→125." **That was wrong** — it conflated a
*reparametrization* (identification‑neutral) with *dropping directions* (a restriction). The precise
decomposition (`cond_reparam_check.jl`, at θ_initial):

| operation on the free A block | cond(J_free) | changes identification? |
|---|---|---|
| (1) raw | 754.6 | — |
| (2) **orthogonal ΔΔ/FE rotation** (pure reparam) | **754.6 (invariant)** | **no — and no conditioning gain either** |
| (3) **per‑parameter sensitivity scaling** (divide each θ by ‖J col‖) | **36.1** | **no (bijective diagonal reparam)** |
| (4) drop the FE directions, keep struct+ΔΔ | 192 | **yes — a restriction** |
| (4b) drop FE + sensitivity scaling | 9.1 | yes |

Three corrected conclusions:

1. **A pure double‑difference reparametrization does nothing** — to identification *or* to
   conditioning. Rotating the A coordinates into (grand, origin‑FE, dest‑FE, ΔΔ) is an *orthogonal*
   change of basis, and singular values are invariant under it (row (2)). You were right to be
   skeptical that ΔΔ coordinates "affect identification."

2. **Per‑parameter sensitivity scaling fixes the *Jacobian* conditioning but not the *solver*.** The
   A_od columns are ≈ **59×** smaller than the structural ones (‖col‖ 0.028–0.092 vs 1.77–2.91), and
   rescaling each A parameter by 1/‖col‖ drops cond **755 → 36** (row (3)). **But when supplied to
   KNITRO (`KN_set_var_scalings_all`, `solve_scaled.jl`) it made the solve worse, not better:**

   | full variant | κ_lower | opt_err | status |
   |---|---|---|---|
   | unscaled | 0.018113 | 0.117 | −101 (stall) |
   | KNITRO auto‑scale (`scale internal`) | 0.018113 | 0.117 | −101 (bit‑identical) |
   | **sensitivity‑scaled (1/‖col‖)** | **0.055807** | **0.412** | −400 (capped, *worse*) |
   | reduced (fewer A, for reference) | 0.008955 | 0.023 | −102 (converged) |

   The scaling *did* take effect (753/709 vs 613/516 inner solves, different κ) — it just didn't help.

3. **Why scaling fails and reduction works.** KNITRO's outer program exposes only 2 explicit
   constraints (divergence budget + gravity); the A_od sensitivity mismatch is buried inside the
   *inner* solve. κ itself is near‑flat in A. Scaling A *up* by ~30 doesn't add curvature — it just
   lets the solver take larger steps across a flat, MC‑noisy plateau, so it wanders further and stalls
   at a worse point. The thing that actually converges is having **fewer** A directions (reduced), i.e.
   attacking the dimensionality, not the scaling. Neither KNITRO auto‑scale nor hand‑equilibration is
   a rescue.

**On dropping the level/FE directions (rows 4/4b):** this is a *restriction*, and it stays a
restriction at every D. γ_d absorbs the destination level (that is `A[1,d]=1`), so the destination
levels are genuinely redundant. The origin level would be redundant only if a *free* per‑origin
parameter could absorb it — the natural candidate is wages, but **wages are pinned by data**
(`w_o L_o = Σ_d λ_od w_d L_d`, λ and L observed), not free, at D = 4 *and* D = 20. So the origin
levels of ln A are identified (weakly), never collinear with a free wage, and dropping them changes
the bounds. This is exactly why the "reduced" solve is a genuine restriction — and why there is no
lossless "2D−1 flats" reduction to 361 at D = 20; it is D redundancies, leaving D(D−1) = 380
identified free A_od. The identification‑preserving way to shrink the outer search is to *solve* the
non‑focal A from the equilibrium + gravity conditions (nested fixed point), not to fix or drop them.

---

## What was not done, and why (scope)

The prompt's D = 20 agenda includes reverse‑mode (Enzyme) gradients, a colored sparse constraint
Jacobian, multistart, and MC‑draw sensitivity. At **D = 4 these are not the binding issues**: the
gradient is already cheap (forward‑mode over 18 params), the outer constraint Jacobian KNITRO sees
is only 2×18, and the diagnostic above shows the difficulty is conditioning of the A_od block, not
gradient cost or exact degeneracy. These levers become relevant precisely when D² makes forward‑mode
and dense assembly expensive — they are the **scale‑up** work, recommended below.

## Recommendation for scaling to D = 20

1. **There is no identification‑neutral dimensionality reduction of the A block.** Wages are
   data‑pinned, so all D(D−1) = 380 A_od are identified — no 2D−1 flats, no lossless reduction to 361.
   Variable scaling was tested and *failed* (lower bound worse); KNITRO auto‑scaling was inert; the ΔΔ
   rotation is inert. The identification‑preserving way to cut the outer dimension is to **profile the
   non‑focal A out via the equilibrium + gravity conditions** (nested fixed point) — the separate
   focal‑only effort, not a tuning of this loop. If the full loop must be kept, expect very large
   `maxit`/tight tolerances and a lower bound that is still stalled (too narrow) at fixed budget; the
   *upper* bound is comparatively better‑behaved but also iteration‑capped.
2. Only then worry about gradient cost: switch the moment‑Jacobian to **reverse‑mode / colored
   sparse** forward‑mode (the Jacobian is block‑structured by destination), which is a pure speed
   play once conditioning is fixed.
3. Curvature: the current `.opt` silently runs **LBFGS** (not the intended `hessopt 4`, because
   `eval_fcga yes` blocks product‑findiff — see caveat). Once conditioning is fixed, test
   `eval_fcga no` + real `hessopt 4`; keep interior/auto. KNITRO variable scaling is *not* a lever
   here (bit‑identical run) — skip it.
