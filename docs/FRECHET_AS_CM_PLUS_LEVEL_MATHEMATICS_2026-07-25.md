# Fixed Fréchet as flexible CM plus a common-level anchor: mathematical formulation — 2026-07-25/26

Part I of the fixed-Fréchet-as-CM-plus-anchor production port. Establishes the exact live
flexible-CM contrast basis (recovered from source, not assumed), the common-level restriction that
turns it into fixed Fréchet, and a full-rank proof of exact equivalence to the direct
country-by-country formulation. Pure mathematics/verification — no production code changed in this
document.

## 0. Provenance

Base: `production/fullA-exact@61a3bd6` (post shared-winner-pair-core-Hessian merge — see
`SHARED_WINNER_PAIR_RELEASE_COMPLETION_2026-07-25.md` on that branch). Branch:
`port/frechet-as-cm-plus-anchor-production-2026-07-25`, worktree
`gravity_robustness/worktrees/port-frechet-as-cm-plus-anchor-production-2026-07-25`.

## 1. The live flexible-CM contrast, recovered from source

The production flexible-CM restriction is built by `precalc_common_marginals_cdf`
(`full_aod_diag/d4_exact/common_marginals_moments.jl:50-118`), called (via `build_cm_augmented_obj`
→ `build_cm_production_context`, `cm_production_bundle.jl:54-97`) from the real production driver
`run_cm_upper_checkpointed` (`cm_checkpoint.jl:597`).

For threshold index `l` with level `z_l` (an empirical quantile of the reference origin's own
baseline draws, `z_l = quantile(U[:,refIndex1], p_l)`, `refIndex1 = ctx.γ.refIndex1`), and for each
non-reference origin `o`, production computes (`common_marginals_moments.jl:87-90`):

```
block[:, oi] = 1{U_o <= z_l} - 1{U_ref <= z_l}
```

Writing `f_l(ω) = (1{z_1(ω)<H_l}, …, 1{z_D(ω)<H_l})'` (the task brief's own notation, `H_l ≡ z_l`),
this is exactly `(e_o − e_ref)' f_l(ω)` for each `o ≠ ref`. So the **raw** (pre-rotation) contrast
matrix is

```
C ∈ ℝ^{D×(D-1)},   column for origin o (o≠ref):  C[:,oi] = e_o − e_ref
```

Production then optionally right-multiplies by a second matrix depending on the `contrasts::Symbol`
keyword (`common_marginals_moments.jl:59,78,92-96`, threaded through `CMConfig.contrasts`,
`cm_config.jl:47`, and `build_cm_production_context`'s own `contrasts::Symbol = :anchored` default,
`cm_production_bundle.jl:54`):

- **`contrasts = :anchored` (the production default at every call site in the real driver chain —
  `CMConfig`, `build_cm_production_context`, `build_cm_augmented_obj`)**: `R = nothing`, i.e. the
  live restriction is **`C` alone**, unrotated. This is *not* an orthonormal contrast matrix.
- **`contrasts = :orthonormal`** (available, used in some benchmark/diagnostic scripts, e.g.
  `bench_cm_bintable_decomposition.jl:78`, but not production's own default): `R =
  orthonormal_contrast_matrix(D)` (`common_marginals_moments.jl:33-37`,
  `R = I_{D-1} + \frac{1/\sqrt D - 1}{D-1}\mathbf 1\mathbf 1'`), and the live restriction is `C·R`.

**Correction to the task brief's assumption**: the brief describes `R` as "expected to be an
orthonormal contrast system" and asks to "verify from source." Verified: production's actual
default is the **unrotated anchored contrast `C`**, not orthonormal. `R` (`orthonormal_contrast_matrix`)
alone is not even orthonormal in isolation (numerically `max|R'R−I|` is O(1), not small) — what
*is* exactly orthonormal is the **composed** map `C·R` (verified below, §2). Both `C` and `C·R`
have the same column space (right-multiplying by the invertible `R` is a reparametrization, not a
change of subspace — proof in §3), so everything below is derived once, generically, for whichever
of `{C, C·R}` production is actually using, and holds for both.

Thresholds/bins: `precalc_common_marginals_cdf` (`common_marginals_moments.jl:70-75`, quantile
grid `range(1/L,(L-1)/L,length=L)` by default) and `compute_bin_indices`
(`common_marginals_interval.jl:72`). Forward/adjoint moment operators: `wrap_moments_with_cm_archB`
/ `fill_cm_columns_from_bins!` (forward) and `cm_lookup_kernels.jl`'s `interval_backward_gradient`
/ `cumulative_backward_gradient` (adjoint), both keyed off the same bin indices and the same `R`
(or `nothing`) — i.e. the CM-plus-level extension below only needs to add columns/targets to this
existing machinery, not replace it (Part II).

## 2. The common-level vector

Define `u = \mathbf 1_D / \sqrt D`. Claim: `u` satisfies the task's three requirements —
`C'u = 0` (equivalently `R'u=0` after rotation, see below), `u'u = 1`, `u'\mathbf 1 > 0` —
**regardless of which contrast mode (`:anchored` or `:orthonormal`) production is using.**

**Proof `C'u=0`**: each column of `C` is `e_o − e_ref`, and `\mathbf 1'(e_o-e_ref) = 1-1 = 0` for
every `o`. So every column of `C` is already orthogonal to `\mathbf 1`, hence orthogonal to
`u = \mathbf1/\sqrt D`, exactly, by construction — no dependence on `D` or on which origin is the
reference. Verified numerically to machine precision at `D=4` and `D=20`
(`max|C'u| = 0.0` exactly, both `D`; see `key_results/basis_conditioning_2026-07-25.txt`).

For the rotated case, `(CR)'u = R'C'u = R'·0 = 0` — follows immediately, no separate argument
needed.

**`u'u`**: `u'u = D·(1/\sqrt D)^2 = 1` exactly.

**`u'\mathbf1`**: `u'\mathbf1 = \sqrt D > 0`.

So the level restriction (task §2), instantiated for production's actual `u`, is: for each
threshold `l`,

```
u'f_l(ω) − (u'\mathbf1)·F*(H_l) = 0
  ⟺  (1/√D)·Σ_o 1{z_o(ω)<H_l} − √D·F*(H_l) = 0
  ⟺  (1/D)·Σ_o 1{z_o(ω)<H_l} = F*(H_l)
```

i.e. **the cross-origin average realized CDF at threshold `H_l` equals the target common Fréchet
CDF at `H_l`** — a direct, symmetric, economically transparent statement of "fixed Fréchet," not an
artifact of the basis choice. The complete fixed-Fréchet block for threshold `l` is
`[C'f_l(ω);\ u'f_l(ω) − (u'\mathbf1)F*(H_l)]`, i.e. `(D-1)` CM difference moments plus 1 level
moment, `D` total, matching task §2's `[R\ u]` construction with `R↦C` (or `C·R` under
`:orthonormal`).

## 3. Exact equivalence to direct country-by-country fixed Fréchet

Let `q_l(ω) = f_l(ω) − F*(H_l)\mathbf1 ∈ ℝ^D` (task's own notation). Direct fixed Fréchet imposes
`q_l=0` (`D` restrictions per threshold). The new formulation imposes `C'q_l=0` and `u'q_l=0`
(`(D-1)+1=D` restrictions per threshold) — note `C'f_l(ω) = C'q_l(ω)` exactly, since
`C'\mathbf1=0` (shown above) kills the `F*(H_l)\mathbf1` term; likewise the level restriction as
written in §2 literally **is** `u'q_l(ω)=0`. So the new formulation's restrictions are exactly
`M'q_l=0` where `M=[C\ u]∈ℝ^{D×D}` (or `[CR\ u]` under `:orthonormal`).

**Claim: `M` is invertible** (either mode) **⟹ `M'q_l=0 ⟺ q_l=0`.**

*Proof `M` is full rank.* `C`'s `D-1` columns `{e_o-e_ref}_{o≠ref}` are linearly independent (each
has a nonzero entry, `+1` at coordinate `o`, appearing in no other column — a standard argument:
if `Σ_o α_o(e_o-e_ref)=0` then reading off coordinate `o≠ref` gives `α_o=0` for every `o`). All
`D-1` columns lie in `\mathbf1^⊥` (shown in §2), which is itself exactly `(D-1)`-dimensional — so
`C`'s column space, being a `(D-1)`-dimensional subspace of a `(D-1)`-dimensional space, **equals**
`\mathbf1^⊥` exactly. `u=\mathbf1/\sqrt D∉\mathbf1^⊥` (since `u'\mathbf1=\sqrt D≠0`), so
`\mathrm{col}(C)⊕\mathrm{span}(u)=ℝ^D`, i.e. `M=[C\ u]` has full column rank `D`, hence is square
and invertible. Under `:orthonormal`, `R` itself is invertible (`R=I+c\mathbf1\mathbf1'` with
`c=(1/\sqrt D - 1)/(D-1)`; eigenvalues `1` (mult. `D-2`, on `\mathbf1^⊥∩ℝ^{D-1}`) and `1+c(D-1) =
1/\sqrt D ≠0` (on `\mathbf1_{D-1}`) — both nonzero, so `R` invertible), so right-multiplication by
`R` is a change of basis of the *same* subspace: `\mathrm{col}(CR)=\mathrm{col}(C)=\mathbf1^⊥`
unchanged, and the identical full-rank argument applies to `[CR\ u]`. Numerically confirmed at
`D=4,20`: `rank(M)=D` exactly, `cond(M_{:anchored})=\sqrt D` (`2.0` at `D=4`, `4.472…` at `D=20` —
a clean closed form, mildly growing but never remotely ill-conditioned at any realistic `D`),
`cond(M_{:orthonormal})=1` exactly (`M` is genuinely orthogonal, `M'M=I_D` to `~1e-16`) — see
`key_results/basis_conditioning_2026-07-25.txt`.

*Proof of the equivalence, given `M` invertible.* `M'` is invertible (transpose of invertible is
invertible), so `M'q_l=0 ⟺ q_l = (M')^{-1}·0 = 0`. ∎

**Consequences** (task §3's four bullets, now proven rather than asserted):

1. **Exactly the same feasible set** as direct fixed Fréchet — `M'q_l=0` and `q_l=0` are the
   identical event, for every draw `ω`, every threshold `l`.
2. **Exactly `DL` restrictions**: `(D-1)L` (CM) `+ L` (level) `= DL`, matching direct fixed
   Fréchet's own `D` restrictions × `L` thresholds exactly — not `DL±ε`.
3. **Literally flexible CM's `(D-1)L` restrictions plus `L` common-level restrictions** — the level
   block is additive, not a re-derivation of the CM block.
4. **Not a relaxation or approximation** — a basis change of an already-square, already-invertible
   linear map. It is **stricter than flexible CM by exactly `L` moments** (flexible CM only imposes
   `C'q_l=0`, i.e. `q_l∈\mathrm{span}(u)`, i.e. `z_1(ω),…,z_D(ω)` share a common CDF value at
   `H_l`, without pinning what that shared value *is*; fixed Fréchet additionally pins it to
   `F*(H_l)` via the level restriction).

This is a **basis change and nested implementation of the same feasible set**, not a new economic
restriction — exactly the framing task §3 requires.

## 4. Why the common-level vector, not country 1

Task §4 asks: if the live CM basis is not orthonormal, construct the corresponding null-space
level vector and report conditioning, without silently switching production's own contrast
convention. Done above: `u=\mathbf1/\sqrt D` is *already* the correct null-space vector for
production's actual default (`:anchored`) — no construction beyond identifying it was needed,
because `C`'s column space is `\mathbf1^⊥` regardless of the reference origin `refIndex1`'s
identity. This is the key reason `u` does **not** privilege country 1 (or whichever origin
`refIndex1` happens to resolve to): although `C` itself is built by differencing against a
*chosen* reference origin — making the *stored* `(D-1)`-column matrix visually asymmetric across
origins — its column space (`\mathbf1^⊥`) is fully origin-symmetric (permutation-invariant), and
`u=\mathbf1/\sqrt D` lives entirely outside that subspace, symmetric across all `D` origins by
construction. Switching `refIndex1` would change `C` but not `\mathrm{col}(C)`, hence not `u`.

Conditioning summary (`key_results/basis_conditioning_2026-07-25.txt`, `D∈{4,20}`):

| | `:anchored` (production default) | `:orthonormal` |
|---|---|---|
| `M'M=I_D`? | no | yes, to `~1e-16` |
| `cond(M)` | `\sqrt D` (`2.0`@D=4, `4.47`@D=20) | `1.0` exactly |
| `rank(M)` | `D` (full) | `D` (full) |

Both are well-conditioned at any `D` this project uses; `:orthonormal` is exactly orthogonal and
therefore the numerically cleaner choice for the Hessian congruence transform in Part III, but
`:anchored` (production's real default) is not a numerical concern either — `cond=\sqrt{20}≈4.47`
is far from a conditioning problem. Part II implements the restriction generically over whichever
mode the live `CMConfig.contrasts` is set to, since the math above holds for both without
modification.

## Summary for Part II

- `U = [C\ u]` (or `[CR\ u]`), `D×D`, always invertible, `\mathrm{col}(U)=ℝ^D`.
- New columns needed beyond flexible CM: exactly `L` (one `u'f_l(ω)` level moment per threshold),
  reusing production's existing `z_l`/bin-index/`f_l(ω)` machinery unchanged.
- New targets: `(u'\mathbf1)F*(H_l) = \sqrt D·F*(H_l)` per threshold, a length-`L` vector computed
  once from the fixed theoretical Fréchet CDF (theta-independent, like the CM block itself).
- `u` itself needs no per-call computation — it is the constant vector `\mathbf1_D/\sqrt D`,
  independent of `D`'s reference origin, independent of `contrasts` mode.
