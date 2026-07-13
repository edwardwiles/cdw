# Call-graph audit (§2)

Traced every use of `calculate_jac_θ!`, `ForwardDiff.jacobian`/`.gradient`, the moment-map
Jacobian, the solved CC multipliers, the outer objective gradient, the divergence/gravity
constraint gradients, and the KNITRO gradient/Jacobian callbacks. Source: `cc_algo/outer_loop_functions.jl`,
`cc_algo/PsiObjectiveBundle.jl`, `cc_algo/inner_loop_functions.jl`.

## 1. The vector-valued function whose Jacobian is currently constructed

`calculate_jac_θ_autodiff!` (`cc_algo/outer_loop_functions.jl:243-256`) differentiates

```
H(θ) = [K(θ) | 1 | G(θ)]      (an N × (d+2) matrix, N = obj.N = Jac_W = 8000, d = nTotalMoments = 18)
```

via `ForwardDiff.jacobian!` on the FLATTENED output `reshape(obj.jac_h, N*(d+2), l)`. This is
**per-draw**, not a sample mean — every one of the N draws' K and G values is differentiated
separately w.r.t. every θ component.

## 2. Dimensions at D=4 (this session's γ_d≡1 + direct-γ' variant)

- `m = N*(d+2) = 8000*20 = 160,000` (the flattened Jacobian's row count)
- `p = l = 23` (θ length: μ, σ(fixed), 4×γ_θ(now inert), γ'_focal, 16×A_od)
- Dense Jacobian size: 160,000 × 23 ≈ 3.68M entries ≈ 29.4MB (Float64) — matches the repo's
  own prior profiling note (`derivative-algo-experiments` memory: "calculate_jac_θ! =
  370ms+80MB/call = hot spot"; the extra allocation is ForwardDiff's Dual-number buffers).

## 3. Where is the FULL matrix genuinely consumed?

**Nowhere.** It is always immediately contracted:

- **Objective gradient**: never touches `jac_h` at all. `calculate_grad_k!` /
  `calculate_grad_k_autodiff!` (`outer_loop_functions.jl:259-284`) computes `∂K/∂θ` directly via
  a SEPARATE `ForwardDiff.gradient!` call, evaluated on just **2 draws** (K is draw-invariant
  under `counterType==1`/autarky — a constant repeated across draws). This is already exactly
  "Method B" (direct scalar gradient, no dense Jacobian), just for the objective only, and
  predates this audit.
- **Divergence-budget constraint** (`PsiObjectiveBundle.jl:216-222`, the implicit-dependence
  case used by this config): `jac_h[:,3:1+outer_constr_index,i]` (the G-columns matched by the
  inner CC dual) is contracted with the fixed multiplier vector `λ` (a `BLAS.gemv!`, one call per
  θ-parameter `i`), then that per-draw N-vector is dot-producted with `arg1` (the per-draw
  Ψ-derivative weight) to give one scalar per parameter — i.e. `∂c_∂θ[1,:]`, an l-vector. This is
  a **bilinear-form gradient**, exactly the "envelope scalar" the audit spec (§4) describes:
  `s(θ) = (1e10/N)·Σ_draws arg1[draw]·(G[draw,1:17](θ)·λ)`, with λ, arg1 held fixed (envelope
  theorem — this constraint has **no** `ift!` correction in the source, uniquely among the outer
  constraints; see §4 below).
- **Gravity constraint** (`∂c_∂θ[2,:]`, `PsiObjectiveBundle.jl:225-236`): the same kind of
  contraction, PLUS an implicit-function-theorem correction (`ift!`) for how the inner solution
  `x*(θ)` itself moves — a genuine total derivative, not a pure envelope scalar. `ift!` also
  consumes `jac_h`, but again only through further BLAS contractions (`gemm!`/`gemv!`), never the
  raw dense array.

So **every consumer is a contraction** (`J'·λ`-type product or a small number of weighted
row/column combinations), never the whole matrix, exactly per the audit's guiding question.

## 4. Is the full Jacobian needed for...

| use | needs full J? |
|---|---|
| CC inner solve | No — inner solve doesn't differentiate at all (KNITRO solves the dual LP/QP-like inner program directly on the *evaluated* H, not its θ-derivative) |
| outer objective gradient | **No** — already a separate 2-draw scalar autodiff call, bypasses `jac_h` entirely |
| divergence-budget gradient | **No** — pure envelope scalar (verified: Method B reproduces it to ~1e-15 relative error, see `correctness_results.csv`) |
| gravity gradient | **Partially** — the direct (envelope) piece doesn't need it either, but the `ift!` total-derivative correction is a separate, more involved computation not re-derived in this audit (see Non-goals below) |
| diagnostics (SVD/conditioning) | **Yes** — this is the legitimate use case for a full/dense Jacobian, e.g. `full_aod_diag/cond_diag.jl`'s conditioning study earlier this session |

## 5. How many explicit constraints does KNITRO see in the outer problem?

**2.** `outer_constr_index = numMoments + 1 - nOuterLoopMoments = 18 + 1 - 1 = 18` (only the
gravity moment is an outer constraint in this config), and `outer_loop_constraints!` for
`PsiObjectiveBundleImplicit` adds `obj.d - obj.outer_constr_index + 2 = 18-18+2 = 2` constraints:
the divergence budget (`constr[1]`) and gravity (`constr[2]`). Confirms the existing memory note
("outer program has only 2 explicit constraints").

## Conclusion for §10 decision rules

Production does **not** need the full moment Jacobian for the objective gradient (already
avoided) or the divergence-budget constraint (provably avoidable — Method B validated exact).
The gravity constraint's `ift!` correction is the one piece genuinely requiring more than a bare
envelope scalar, and was **not** re-derived here (scope: this audit targets the divergence-budget
constraint and the objective, per the spec's explicit "the gravity moment should use the existing
analytic gradient — do not replace it with AD merely for uniformity"; a full IFT re-derivation for
gravity is future work, not a backend swap). The dense Jacobian's only legitimate remaining use
in this codebase is off-critical-path diagnostics (SVD/conditioning), which is exactly how
`full_aod_diag/cond_diag.jl` already uses it — a use case Method D (colored sparse) is well suited
to speed up, separately from the production gradient/constraint path.
