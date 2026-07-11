# Sequentially-linearized profiled full-gravity — design note

Branch: **`sequential-profiled-gravity`** (off `experiments-derivatives`). Reversible:
`git checkout experiments-derivatives` restores the prior state; nothing here overwrites
the existing pipeline until Phase 2 wiring (and that is gated behind a new
`orthogonality_mode` flag).

This note translates the ChatGPT specification ("sequentially linearized profiled-gravity
procedure") into the **concrete objects of this codebase**, records the exact model↔spec
mapping, and states the implementation plan. It is the Section-23 deliverable.

---

## 0. Why this, and what the spec assumes that we don't yet have

**The dimensionality problem.** All bilateral shifters `A[o,d]` currently sit in the outer
parameter vector θ (`get_θ_initial` in `prepare_cc/buildObjectsForMoments.jl`; the block
`Aod_initial`). At D=4 that is 16 entries (12 free after the `A[1,d]=1` normalization). At D=19
it is 361 entries (~342 free) — the outer bilevel program does not scale.

**The old France-only compromise** (in the coauthor's code, not here): keep only the gravity
moment rows with destination = France (baseIndex) and drop every `A[o,d]` for `d≠France` plus
their trade-share moments. Valid because those `A` columns enter nothing else, but the
France-only gravity moment is too noisy.

**The new idea (this spec).** Keep only `A[o,France]` in the outer vector. Recover every omitted
column `A[·,d]` (d≠France) by **inverting** its observed trade-share column under the *current
least-favorable distribution* F. Impose the **full** origin+destination-FE gravity restriction
`R(F) = 0`, where `R` is a function of F through those inversions. Because the inversion map
`I_d(F)` is a nonlinear functional of F (not expressible in CC's linear-moment language),
**linearize it** around the current F: add one extra CC expectation moment
`E_F[ψ̄] = −R_k` (an influence function ψ̄), re-solve, re-invert, recompute the exact nonlinear
`R`, and iterate until `|R|` is below tolerance. This lets gravity influence *which* distribution
CC selects — unlike an ex-post check.

**Prerequisite the spec assumes but this repo lacks.** The spec is written as if a "reduced CC
problem" (focal-only trade-share moments; only `A[·,France]` in outer) already exists. **It does
not** — the current code (`gravMoment=1` on branch `experiments-derivatives`) keeps *all* `A[o,d]`
in outer and *all* trade-share moments in the inner loop, with gravity as an F-independent outer
constraint on θ. So the work splits:

- **Phase 1 (this note's core, KNITRO-free, validated first):** the inversion `I_d`, the exact
  gravity residual `R`, the AD share-Jacobian `H_d`, the adjoint, and the draw-level influence
  function ψ — all as standalone tested functions operating on `(draws, LFD weights, observed
  shares)`. This is where the mathematical risk is; it is validated by the mandatory
  directional-derivative test **before** any solver wiring.
- **Phase 2 (after Phase 1 passes):** implement the reduced focal-only CC mode (outer θ =
  `A[·,France]` only; inner moments = France trade shares + counterfactual + normalizations),
  add ψ̄ as one extra inner moment, and drive the sequential loop from the outer optimizer.

---

## 1. Exact model ↔ spec mapping (derived from the code)

Source of truth: `moments/hFunction.jl`, `moments/moments!.jl`, `prepare_cc/createUDerivatives!.jl`.

Base draws `U[s,o] ~ Exp(1)` (`genExpRands!`), i.i.d. across s and o. `UoModel=1` ⇒ one draw per
**origin** o (not per od pair), shared across destinations. With `θConstant=0`, `U` is kept raw
and `Uσ = U.^(1-σ)`.

Price of origin o in destination d, draw s (from `hFunction!`, with the argument named `Aod`
actually being `AodPow`):

    p_od(s) = w_o · AodPow_od · τ_od · U[s,o]^{μ}          (level; μ = θ[1])
    z_o(s)  = U[s,o]^{-μ}                                   (productivity)

with the **raw structural shifter** `A_od = 1/AodPow_od` (see `moments!.jl:161-166` and the
EXPERIMENTS note). The winning origin is `argmin_o p_od(s)`.

The **σ-transformed value** carried by the winner (the trade-share magnitude) is
`p_od^{1-σ} = wPow_o·(AodPow_od τ_od)^{1-σ}·Uσ[s,o]^{μ}` (via `pricesTempσ` and
`UσPow=Uσ^{-μ}`). Since σ>1, `p^{1-σ}` is monotone-decreasing in `p`, so

    argmin_o p_od(s)  =  argmax_o p_od(s)^{1-σ}  =  argmax_o ( u[o,d] + log_x[s,o] ),

where — and this is the clean collapse to the spec's notation —

    log_x[s,o] = μ · log(Uσ[s,o]) = μ(1-σ) · log(U[s,o])     (destination-independent) ★
    u[o,d]     = (σ-1)·( log A_od − log w_o − log τ_od )       ★★

★★ matches the spec's `u[o,d] = (σ-1)(log A[o,d] − log wage[o] − log τ[o,d])` **exactly**
(using `A_od = 1/AodPow`, `(1-σ)log AodPow = (σ-1)log A`). ★ matches the spec's `log_x[s,o]`
(destination-independent, precomputable). `tau[o,d] = 1 + tariff` here is `τData`.

**Empirical share** (fixed draws, LFD weights p[s]):

    value[s,o] = exp( u[o,d] + log_x[s,o] ),   winner[s] = argmax_o value[s,o]
    share_d[o] = Σ_{s: winner[s]=o} p[s]·value[s,winner] / Σ_s p[s]·value[s,winner]

i.e. only the winner's value counts on each draw. This is the object the inversion matches to
`λ̂[·,d]` = observed trade-share column (`γ.P` reshaped; `λData`).

**Normalization / gauge.** Shares are invariant to adding a constant to a whole `u[·,d]` column
(it cancels in the ratio). The code pins `A[1,d]=1` (origin-1). The spec pins `u[d,d]=0`. Both
are valid gauges; the gravity residual is gauge-invariant after two-way demeaning (test F). We
use **reference origin o=1** (`u[1,d]=0`) to match the existing `A[1,d]=1` convention. Configurable.

---

## 2. LFD weights (spec §4) — already computed here

`lfd/LFD.jl:50-53` is exactly the spec's `m_k`/`p_k` recovery: given the inner-dual solution
`x=(ζ,λ)` at a θ, the density ratio on draw s is

    m_k[s] = dPsi!( arg0 )[s],   arg0[s] = −ζ − dot(G[s, 1:oci-1], λ)

and F* has uniform base weights `1/W`, so `p_k[s] = m_k[s] / Σ_r m_k[r]` (spec §4, with
`π*[s]=1/W`). `Σ_s p_k[s]=1` because `E_{F*}[dF/dF*]=1`. Phase 2 reuses this verbatim; Phase 1
tests accept an arbitrary valid `p` (uniform, tilted, or a saved LFD).

---

## 3. The pieces (spec §§5–13), in this codebase's terms

Let free coordinates drop reference origin o=1 (`u[1,d]=0`).

**Inversion `I_d(F)` (§6).** Minimize convex potential over `u_free`:

    φ_d(u) = logsumexp_s( log p[s] + max_o( u[o] + log_x[s,o] ) ),   u[1]=0
    ∇_free φ_d = share_d[free]           (so ∇(φ_d − λ̂·u) = share − λ̂ )

Damped Newton with the closed-form Hessian below; converge on `‖share − λ̂‖_∞`.

**Share Jacobian `H_d = ∂share/∂u_free` (§9).** Away from ties, winners are locally fixed, so φ_d
is a logsumexp of affine functions and its Hessian is the softmax covariance of the winner
indicator:

    H_d = ( diag(share) − share·share' )[free, free]         (PSD, generically PD on free coords)

Default per spec: obtain `H_d` from **ForwardDiff.hessian(φ_d_free, u_free)** (hard-max AD:
differentiates the selected branch; second derivative of the piecewise-linear max is 0 a.e., so
AD returns exactly this covariance away from ties). We compute both and cross-check (test E). The
spec's caveat holds: hard-max AD is the exact Hessian of the *finite-sample* potential away from
ties; it omits the population extensive-margin (boundary-switching) term — which is why the
directional-derivative test against exact re-inversion (§16) is mandatory.

**Gravity residual `R` (§7).** Build the D×D matrix `u[o,d]` (focal column from outer A; omitted
columns from inversion), then

    logA[o,d] = log w_o + log τ_od + u[o,d]/(σ-1)
    Q = log τ ;  Q̃ = withinTransform-style two-way demean of Q ;  logÃ likewise
    R_sum = Σ_od Q̃·logÃ ,  R_beta = R_sum / Σ Q̃² ,  S_Q = Σ Q̃²

Identity (test G): `logÃ = Q̃ + ũ/(σ-1)` (log w_o is a pure origin FE ⇒ demeaned away), so
`R_sum = Σ Q̃·(Q̃ + ũ/(σ-1))`. This reuses `misc/doubleDiff.jl::withinTransform` and is
sign-consistent with the existing `newGravityMoment!` (which uses `within(logAodPow) = −logÃ`).
We use **R_beta** as the sequential constraint (scale-stable).

**Draw-level share objects (§8).** With `Vexp[s,d] = exp(max_o(u[o,d]+log_x[s,o]))`,
`winner[s,d]`, and `M_d = Σ_s p[s]·Vexp[s,d]`:

    r[s,o,d]  = Vexp[s,d]·1{winner[s,d]=o}
    ξ[s,·,d]  = r[s,·,d] − λ̂[·,d]·Vexp[s,d]        (E_F[ξ_d]=0 at a converged inversion)

**Implicit derivative (§11).** For a score perturbation h with E_F[h]=0,
`D_F u_d[h] = −H_d^{-1} · (1/M_d) · E_F[h·ξ_free_d]` (never form the inverse).

**Adjoint + influence function (§§12-13).** With `c_d[o] = Q̃[o,d]/(σ-1)` (free; divide by S_Q for
R_beta), solve `H_d a_d = c_d` (H_d symmetric), then

    ψ_R[s] = − Σ_{omitted d} dot(a_d, ξ_free[s,·,d]) / M_d ,   ψ̄ = ψ_R − E_F[ψ_R]

Self-consistency (used as an internal check): E_F[ψ_R]=0 automatically, and
`influence of h on R = E_F[h·ψ_R]` (chains `dR/du_d=c_d` with the implicit derivative). The next
CC problem adds the single equality moment `E_F[ψ̄] = −R_k`; on the F* support this is
`E_{F*}[m(z)·ψ̄(z)] = −R_k`.

---

## 4. Sequential inner algorithm (§14) and outer return (§18) — Phase 2

At a fixed outer point x (= μ, γ's, γ'_baseIndex, A[·,France]):

1. Solve reduced CC (focal moments only) → LFD `p_0`.
2. Iterate k: invert all omitted d under `p_k`; build `R_k`; if `|R_k|≤tol` and inversions/CC
   moments meet tolerance → converged. Else form `H_d`, solve adjoints, build ψ̄_k, add the one
   moment `E_F[ψ̄_k]=−R_k`, re-solve CC → `p_candidate`, re-invert, recompute exact `R`, and set
   `p_{k+1}` by damping (`p_α = (1-α)p_k + α p_candidate`, backtrack α∈{1,½,¼,…} on `|R_α|<|R_k|`).
   Replace (do not accumulate) the linearization each iteration.
3. Return `Δ_sequential(x)` = converged divergence from F*; feasibility `Δ_sequential(x) ≤ δ`.
   Gains-from-trade objective unchanged.

**Modes** (`orthogonality_mode`): `:focal_only` (reduced CC, no full gravity — must reproduce a
reduced baseline), `:sequential_profiled_full` (this procedure),
`:profiled_full_ex_post_diagnostic` (solve reduced once → invert → check gravity ex post;
**diagnostic only, not the full calculation**). The current all-A-in-outer `gravMoment=1` path is
retained unchanged as the existing exact-but-non-scaling reference.

---

## 5. Non-goals (§22)

No change to: the divergence (`Psi.jl`), the productivity draws, the gains-from-trade objective,
the focal-destination economic model, or the Melitz path. Omitted `A[o,d]` (d≠France) are **not**
added to outer θ; omitted trade-share moments are **not** added explicitly to each CC problem; the
default share Jacobian is AD, not finite differences; no silent regularization; convergence is
declared only against the **exact nonlinear** `R` after re-inverting.

---

## 6. File plan

- `sequential_gravity/profiled_gravity.jl` — Phase-1 core (this note §3), dependency-light
  (LinearAlgebra, ForwardDiff). No KNITRO.
- `sequential_gravity/test_profiled_gravity.jl` — tests C,D,E,F,G,H,I/J (spec §20), synthetic +
  self-contained; runnable with `julia --project=. sequential_gravity/test_profiled_gravity.jl`.
- `sequential_gravity/validate_on_pipeline.jl` — Phase-1 validation on the *real* draws/data
  (builds U, Uσ, wages, τ, σ, μ, observed shares via setup+prestep; still no outer solve).
- Phase 2 (later): new `cc_algo`/`moments` wiring behind `orthogonality_mode`.
