# Theta C+-style fixed-dual derivative: mathematical derivation — 2026-07-26

Every formula below is derived directly from the live source (`gravity_elimination.jl`,
`compressed_moments.jl`), not reconstructed from memory or from prior docs' prose. Where the
task's own framing turned out to need correcting once checked against code, that correction is
stated explicitly rather than silently reconciled.

## 1. The live hard-winner score, exactly, in transformed-A coordinates

`compressed_moments.jl`'s own verified header gives the factual winner rule:

```
winner_{s,d} = argmin_o price_{s,o,d},   price_{s,o,d} = constCons_{o,d} / U_{s,o}^{-μ}
constCons_{o,d} = wHat_o * AodPow_{o,d} * τ_{o,d}
```

so `price_{s,o,d} = constCons_{o,d} * U_{s,o}^{μ}`, and in logs:

```
log(price_{s,o,d}) = [log(wHat_o) + log(AodPow_{o,d}) + log(τ_{o,d})]  +  μ * log(U_{s,o})
```

For every **non-pivot** cell, `a_{o,d} := log(AodPow_{o,d})` is *literally the free outer
coordinate this port searches* (by construction, `A_coordinate_mode=:powered_aspace`), held fixed
while θ perturbs. So, defining

```
α_{o,d} = log(wHat_o) + a_{o,d} + log(τ_{o,d})     (data + fixed coordinate; θ-independent)
b_{s,o} = log(U_{s,o})                              (raw draw; θ-independent, shared across all d)
```

the score is **exactly**:

```
s_{o,d}(μ) = α_{o,d} + μ·b_{s,o}       for every (o,d) except the single pivot cell
winner_{s,d} = argmin_o s_{o,d}(μ)     (score MINIMIZED — price, not utility)
```

## 2. The pivot cell — corrected from the task's own framing

The task's brief frames the pivot's contribution as "an intercept affine in μ." **Checked against
`gravity_elimination.jl::pivot_expand_cheap` directly, this is not quite right — the pivot's
z-value is affine in θ, not μ — and the μ-affine form only emerges one algebraic step later, at
the `AodPow`/score level.** Both steps below are shown because the distinction matters for
implementation (which quantity you cache and interpolate).

`pivot_expand_cheap` (unchanged by this port, reused as-is):
```
g0(μ) = pgc.a + pgc.b·μ                          (verified affine in μ — a different, more primitive quantity)
intercept(μ) = -g0(μ) / (μ·c0[pivot])  =  -pgc.a/c0[pivot]·θ  -  pgc.b/c0[pivot]     (μ·θ=1 substitution)
z_pivot(θ) = intercept(θ) + Σ_k slope[k]·z_nonpivot_k(θ)
```
Since every non-pivot `z_nonpivot_k(θ) = -θ·(a_k + logX_k) - logY_k` (exactly linear in θ, `a_k`
fixed — the a↔z map, unchanged), the whole right-hand side collapses to a single degree-1
polynomial in **θ**:
```
z_pivot(θ) = B1·θ + B0
  B1 = -pgc.a/c0[pivot] - Σ_k slope[k]·(a_k + logX_k)
  B0 = -pgc.b/c0[pivot] - Σ_k slope[k]·logY_k
```
(`B1`, `B0` are constants for the lifetime of the outer base point — they depend on `pgc`, the
fixed non-pivot `a`'s, and data only, never on the current θ probe.)

Now apply the a-space map's inverse to get `log(AodPow_pivot)`:
```
log(AodPow_pivot) = -μ·(z_pivot(θ) + logY_pivot) - logX_pivot
                   = -μ·B1·θ - μ·B0 - μ·logY_pivot - logX_pivot
                   = -B1  -  μ·(B0 + logY_pivot)  -  logX_pivot        (μ·θ = 1 exactly)
```
The θ-linear term becomes a **constant** (`μθ=1`), and only a μ-linear term survives — so
`log(AodPow_pivot)`, and hence the pivot cell's price score, **is** exactly affine in μ, just via
this two-step route rather than directly:
```
α_pivot = log(wHat_{o*}) - B1 - logX_pivot + log(τ_{o*,d*})
b_pivot(s) = log(U_{s,o*}) - (B0 + logY_pivot)
s_pivot(μ) = α_pivot + μ·b_pivot(s)
```
**Net result: every one of the D·Ddest cells — pivot included — fits the identical template
`s_{o,d}(μ) = α_{o,d} + μ·b_{o,d}(s)`**, with the pivot cell's `α`/`b` computed via the two-step
route above instead of directly from `a_{o,d}`. This is what makes a single uniform scan loop
(§below) correct without a special-cased branch for the pivot beyond precomputing its `α`/`b`
differently.

## 3. Winning CES value — the task's shortcut formula confirmed, after checking `Uσ`'s real definition

The task suggests `v = exp[(1-σ)·s]`. This is **not automatically true for an arbitrary definition
of `Uσ`** — checked directly (`Uσ = U .^ (1 - σHat)`, found in the live context-construction path,
not assumed):
```
pTσ_{s,o,d} = constConsσ_{o,d} / Uσ_{s,o}^{-μ} = constConsσ_{o,d} · Uσ_{s,o}^{μ}
constConsσ_{o,d} = wHat_o^{1-σ} · (AodPow_{o,d}·τ_{o,d})^{1-σ}
log(pTσ_{s,o,d}) = (1-σ)·α_{o,d} + μ·log(Uσ_{s,o}) = (1-σ)·α_{o,d} + μ·(1-σ)·b_{s,o} = (1-σ)·s_{o,d}(μ)
```
Because `Uσ = U^{1-σ}` (not, e.g., `U^σ` — a wrong guess here would have introduced a spurious
extra `σ` factor on the slope), the `(1-σ)` factors out cleanly and the task's shortcut is
**exactly correct**:
```
v_{s,d}(μ) = exp[(1-σ)·s_{winner_{s,d}(μ), d}(μ)]
```

## 4. What already implements exactly this, in production, today

`compressed_moments.jl::build_compressed_factual(θ_full, ctx)` already computes — for a *given*
`θ_full` — precisely `winner[s,d]` (via `canonical_winner_argmin`, the shared fast log-additive
argmin) and `wval[s,d] = v_{s,d}` (the winning CES value), plus every normalization constant
(`gdiv`/`nrm`/`PMM`/`denom`/`Pmat`/`gammafac`) needed to evaluate the fixed-dual objective, in
**O(W·D²)** compute / **O(W·Ddest)** memory (never materializing the O(W·(D·Ddest+2)) dense moment
matrix). `compressed_cc_value_grad(ζ, λ, cf; Psi!, dPsi!)` then evaluates the exact fixed-dual
objective value (and gradient, unused here) from `cf` plus a **held-fixed** `(ζ,λ)` — this is
*already* the "fixed-dual index" formulation the task's §2 asks for
(`r_ω = ζ + λ'g_ω(η)`, evaluated via the compressed contraction, not the dense matrix).

**This is the existing, already-validated machinery this task's fast evaluator should call**,
rather than re-deriving `Δr_ω = λ'[g_ω(η±h)-g_ω(η)]` from scratch by hand — `build_compressed_factual`
already does a full, exact re-scan of all D origins per (draw, destination) at whatever θ it's
given (satisfying the addendum's "fused full-origin plus/minus scan is the correctness reference,
recompute the hard winner at both perturbed theta values" requirement directly), and
`compressed_cc_value_grad` already implements the exact same `f = sum(Ψ(-ζ-λ'g))/M + ζ` formula
the dense `obj(x)` functor computes, just from the compressed representation. The performance gap
identified in `FLEXIBLE_THETA_DERIVATIVE_PERFORMANCE_ANALYSIS_2026-07-26.md` (theta secant costing
~1.4s/743MB per probe) exists because `theta_fixed_dual_delta_pivot_A` calls the **general dense**
`obj.moments!`/`CS.reconstruct_full`/`obj(x)` path instead of this **already-existing compressed**
path — not because computing this quantity is inherently expensive.

## 5. Moment columns enumerated (task §1's "every unrestricted core moment that changes with θ")

At `D=20, Ddest=19`: `oci-1 = 382` inner-dual columns = `D·Ddest = 380` bilateral hard-winner
columns (§1-3 above) + 1 counterfactual price-index column (`cf_col = D·Ddest+1`, per
`build_compressed_factual`'s `cf_raw` block — same `AodPow` object, same μ-exponent structure,
fits the identical `s(μ)` template via the *same* pivot-or-nonpivot `α/b` for whichever cell `bi`
lands on) + 1 further column (the observed `numMoments=382` in the D=20 logs already includes both
extra columns beyond the D·Ddest bilateral block — confirmed by direct count against the live
`build_compressed_factual` field structure and the D=20 gate log's own `numMoments=382` print, not
assumed). **All θ-dependence in every column flows through exactly the same two objects**:
`winner[s,d]` (which origin, via the uniform `argmin_o s_{o,d}(μ)`) and `v_{s,d}=exp[(1-σ)s]` (the
winning value) — plus one *global* scalar, `gammafac = Γ(μ(1-σ)+1)`, which rescales the bilateral
+ counterfactual columns' `gdiv` uniformly and must be recomputed at each θ probe (cheap, O(1),
already how `build_compressed_factual` computes it).

## 6. Winner-stability crossing radius (task §4 — kept as an available diagnostic, not the default
correctness path; see the addendum below)

For base winner `w` at a given `(s,d)`, challenger `o`: `g_o = s_o(μ0) - s_w(μ0) ≥ 0` (price gap,
θ0), `d_o = b_o - b_w` (slope gap). A positive `Δμ` perturbation can only flip the winner to `o` if
`d_o < 0` (o's price falls faster), crossing at `Δμ_o^+ = g_o / (-d_o)`; a negative `Δμ` flips only
if `d_o > 0`, crossing at magnitude `g_o/d_o`. The nearest such crossing across all `o≠w` bounds
how far `μ` can move before the base winner is no longer certifiably correct. **Per the user's
mid-task addendum, this is not used as a shortcut/optional-backend in this port** — every θ probe
gets an exact, full re-scan via `build_compressed_factual` regardless. The formula is recorded here
in case a future session wants a diagnostic (e.g. to report how close to a winner flip a given
probe was), not as a correctness-critical code path.

## 7. Addendum compliance

Per the user's mid-task instruction: the implementation below performs an **exact fixed-dual
central secant recomputing the hard winner at both perturbed θ values in full** (no
runner-up/top-3/largest-U shortcut, no optional analytic stable-winner backend). The speedup comes
entirely from (a) reusing the already-existing O(W·D²)/O(W·Ddest) compressed pipeline instead of
the O(W·382)-with-dense-allocation generic path, and (b) eliminating redundant work around that
core computation (outer-vector copies, repeated pivot-cache rebuilds, a third, unused
reconstruction) — never from weakening what gets recomputed.
