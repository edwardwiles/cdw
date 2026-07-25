# Flexible-theta rectangular gravity restriction audit — production port, 2026-07-25

Task §6 deliverable. Re-derives the a-space gravity restriction on the ACTIVE, POST-OMIT-ROW
sample (20 origins, 19 active destinations under the real D=20 default, `destination_sample=
:exclude_row`) — does NOT reuse any square-sample residual or theta estimate from the pre-omit-ROW
experimental branch (that branch predates `context_real_d20.jl`'s omit-ROW rectangularization
entirely; its own `GRAVITY_SAMPLE_VERSION=2`/`THETA_CALIBRATION_VERSION=2` comments already
establish that theta itself is *re-estimated on the resolved rectangular sample*, not carried over
from a square fit — this audit's own pivot/gravity re-derivation follows the same discipline).

## 1. The restriction

Current production (`gravity_elimination.jl`) represents the single exact gravity restriction as
affine in `z := log(Aod_theta)` (`D × Ddest`, `Ddest = ctx.D_dest` under `:exclude_row`):

```
g_gravity(z) = sum(c .* z) + g0,   c = gravity_linear_coeffs(ctx; μ) = μ .* q_tilde ./ N_obs
```

`q_tilde`/`N_obs` (`precompute_q_tilde(τ)`, `context_real_d20.jl`) are built from the ACTIVE `τ`
sample only (already rectangular — `context_real_d20.jl`'s own `τ = γ.τ` reflects the resolved
`destination_sample`), so `c`/`g0` are automatically computed on the active `D × Ddest` cells, not
a square superset with a column dropped after the fact.

`g0 = gravity_offset(ctx; μ)` (the affine intercept, evaluated at `z=0` — a coordinate device, see
the file header's standing CLAUDE.md caveat, NOT a claim about the calibration point) is computed
by `gravity_from_logz`, which reconstructs the FULL rectangular price/level object
(`Aod_lvl`, `AodPow`) using the SAME `lambda = reshape(γ.P, (Ddest, D))'` reshape convention
`fast_range_screen.jl`'s `Pmat` construction and this port's own `precompute_aspace_XY` use — this
is the convention verified correct for the active-destination-sliced `γ.P` vector (column-major,
destination-slot-fast), NOT the square `reshape(P,(D,D))` the pre-omit-ROW experimental prototype
used (which would silently misalign columns whenever `Ddest != D`).

## 2. Pivot elimination — theta-invariance, re-derived (not assumed) on the rectangular sample

`c(mu) = mu .* c0` where `c0 := q_tilde./N_obs` is **pure data on the active sample**, independent
of `mu`/`gp`/`A`. Since `mu = 1/theta > 0` always (theory-safe domain), for ANY two positive
scalars `mu_1, mu_2`: `argmax|c(mu_1)| = argmax|c0| = argmax|c(mu_2)|` — a positive scalar multiple
never changes an argmax. **Pivot index is theta-invariant** on the rectangular sample, by the same
argument as the (pre-omit-ROW) square case, now re-verified with `c0`/`argmax` computed on the
active `D*Ddest` cells rather than `D^2`.

Pivot reconstruction (current production, `pivot_expand`):
`z[pivot] = (-g0 - sum_{j != pivot} c[j]*z[j])) / c[pivot]`. Substituting `c(mu) = mu.*c0`:

```
z[pivot] = (-g0(mu)/mu - sum_j c0[j]*z[j]) / c0[pivot]
         = intercept(mu) + sum_j slope[j]*z[j],   slope[j] = -c0[j]/c0[pivot]  (THETA-INVARIANT)
         intercept(mu) = -g0(mu) / (mu * c0[pivot])
```

**Pivot reconstruction slopes are exactly theta-invariant** (the `mu` that appears in both
numerator's `c[j]=mu*c0[j]` and denominator's `c[pivot]=mu*c0[pivot]` cancels exactly, leaving
`slope[j] = -c0[j]/c0[pivot]`, no `mu` dependence at all). **Only the intercept term
`intercept(mu) = -g0(mu)/(mu*c0[pivot])` depends on theta**, and it depends on theta ONLY through
the single scalar `g0(mu)`.

## 3. `g0(mu)` is exactly affine in `mu` — verified, not assumed

Claim: at a fixed `(gp, γ)` base state (i.e. holding every OTHER free coordinate fixed),
`g0(mu) = a_fit + b_fit * mu` exactly (not merely approximately) for `mu` in the theory-safe
domain.

Re-derivation (rectangularized): `gravity_from_logz(zeros(D,Ddest), ctx; μ=mu)` evaluates
`Aod_θ = exp(0) = 1` (a fixed all-ones matrix, independent of `mu`), then:

```
Aod_lvl(mu) = 1 .* cHat .* X.^(1/mu) .* Y     -- depends on mu only through the X.^(1/mu) factor
AodPow(mu)  = (Aod_lvl(mu)./cHat).^(-mu) = (X.^(1/mu).*Y).^(-mu) = X.^(-1) .* Y.^(-mu)
```

`gravity_value` (unchanged, `gravity_tariff.jl`) is a WEIGHTED LINEAR functional of
`log(AodPow)`-type residuals against the observed trade shares (the same `q_tilde/N_obs` structure
that makes `c(mu)` linear in `mu` in the first place) — `g0(mu) = gravity_value(τ, AodPow(mu),
q_tilde, N_obs)` reduces, after collecting terms, to a LINEAR functional of `log(AodPow(mu)) =
-log(X) - mu*log(Y)`, which is itself affine in `mu` (constant term `-log(X)`, linear term
`-mu*log(Y)`). A linear functional of an affine-in-mu object is affine in `mu`. This is the same
structural fact `q_tilde/N_obs`'s linearity in `c(mu)` already exploits — not a new assumption for
this audit, but re-verified explicitly here on the ACTIVE rectangular `X`/`Y` (§1) rather than
inherited from the square pre-omit-ROW claim.

**Empirical re-verification** (not just algebra): `build_pivot_elimination_cheap(ctx; mu_probe1,
mu_probe2)` fits `g0(mu) = a_fit + b_fit*mu` from exactly two `gravity_offset` evaluations at
arbitrary distinct `mu_probe1 != mu_probe2`. `test_flexible_theta_aspace_d4.jl` gate 7d
(rectangular D=4/D_dest=3 sample) round-trips a `logA_full` reconstruction through
`decode_and_expand_flexible_A` → `pivot_expand_cheap` → `reduce_to_w_ext_A` back to the starting
`w_ext_a`, to `rtol=1e-10` — this would fail immediately if the affine fit were only approximate
(any curvature in `g0(mu)` would show up as a probe-choice-dependent residual once `mu` moves away
from the two fit points, which the round-trip test exercises at `theta_star`, generally distinct
from both probes `1/theta_lo`/`1/theta_hi`).

## 4. Consequence for the outer-loop cost model

No outer iterate that only moves `theta` (the theta secant's two probes, §6 of the math
parameterization doc) needs to rebuild the pivot choice, `other_idx`, or re-derive the affine
offset from scratch — `pivot_expand_cheap` evaluates `intercept(mu)` in `O(1)` given the cached
`(a_fit, b_fit, c0, pivot_lin, other_idx, slope)`, itself built once per outer base point
(`O(D*Ddest)`, whenever `gp` changes) via `build_pivot_elimination_cheap`. This avoids paying a
full `gravity_from_logz` `O(D*Ddest)` reconstruction at every one of the theta secant's two probes
per `cb_G!` call — a real, if modest at D=20 scale, saving (`D*Ddest=380` vs `O(1)`).

## 5. D=4 and D=20 reconstruction test coverage

- D=4 square (`context_scaled.jl`, `row_idx=nothing`) and rectangular (`row_idx=4`, `D_dest=3`,
  last-position omission — production's own only supported `row_idx` position): both at
  `theta_star` and at off-theta values (`theta_min`, `theta_max`, `theta_star*1.05/0.95`) — see
  `test_flexible_theta_aspace_d4.jl` gates 1, 2, 3, 6, 7a, 7b, 7d.
- D=20 real post-omit-ROW sample: see
  `docs/FLEXIBLE_THETA_D20_DERIVATIVE_VALIDATION_2026-07-25.md` for the gravity-reconstruction and
  cold-verification results at calibration and off-theta probes on the real `D=20, D_dest=19` data.

## 6. What was NOT re-derived from scratch

The underlying `gravity_value`/`gravity_tariff.jl` weighted-residual formula, the `q_tilde`/`N_obs`
precomputation (`precompute_q_tilde`), and `pivot_expand`/`pivot_reduce`'s generic scatter/gather
logic (they operate on `PivotGravityElim`'s `(pivot_lin, other_idx, c, g0)` fields generically,
regardless of how those fields were computed) are ALL current production code, unmodified. This
audit re-derives and re-verifies the THETA-DEPENDENCE STRUCTURE of `c`/`g0` on the active
rectangular sample — it does not re-implement or second-guess the underlying gravity-residual
economics itself.
