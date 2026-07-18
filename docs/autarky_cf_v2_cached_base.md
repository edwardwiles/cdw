# Focal-autarky CF moment — V2 cached-base-vector specialization

Branch `diag/fullA-d4-autarky-cf-v2` (worktree
`.../.claude/worktrees/manual-autarky-cf-v2`, off `5cd37f7`). All work is
**additive** in new files under `full_aod_diag/d4_exact/`; the trusted generic
constructor (`EK_moments_gammanorm_directgp!`), the generic `hFunctionCounter!`,
the already-adopted `autarky_cf.jl` (v1) path, the tie rule (`MinInd!`), and all
other production code are **not modified**.

New files:
- `full_aod_diag/d4_exact/autarky_cf_v2.jl` — the cached-base path + `enable_autarky_cf_v2!`.
- `full_aod_diag/d4_exact/test_autarky_cf_v2.jl` — equivalence (vs generic AND vs v1) + exponent + independence + reuse.
- `full_aod_diag/d4_exact/benchmark_autarky_cf_v2.jl` — isolated / CF-only-sweep / full-build / value-call benchmarks.

## TL;DR

The task's premise holds and is implemented: μ, σ, the domestic wage/trade-cost,
the focal labour, and the draw matrix are **provably fixed** for a whole outer
run (`context.jl` pins `θ_lo[1]==θ_hi[1]` for μ, `θ_lo[2]==θ_hi[2]` for σ, and
holds indices 1,2,3:2+D out of the free-param map). So V2 precomputes **once**
the draw-level reciprocal `inv_uσ[s] = 1/UσPow[s,o1]`, and each outer point's CF
column is a single per-draw multiply-subtract `cf_num·inv_uσ[s] − cf_denom` with
**no per-draw divide, power, log, exp, min, or branch**.

**The task's `A_dd^(σ-1)` hint is not exactly right.** The verified scalar on the
raw free variable `A_dd = Aod_θ[bi,bi]` is `A_dd^(μ(σ-1))` — the μ factor is
real and must be kept. (Under the code's `AodPow` convention the factor is
`AodPow[bi,bi]^(1-σ)`, exponent `1-σ = -(σ-1)`.) `cf_denom` is **A_dd-independent**.

**Equivalence: tight, not bit-identical** (deliberately). V2 turns the per-draw
divide into a multiply against a once-formed reciprocal, so it agrees with both
the generic routine and the v1 `autarky_cf.jl` path to **max abs 3.55e-15** (one
reciprocal ULP of the O(1) numerator term) — vs v1's exact bit-identity. Through
the live inner solve (`evaluate_fullA`) the downstream `Delta_dual` is
**bit-identical (|diff|=0.00e+00)** — the 3.5e-15 moment perturbation is far
below KNITRO's tolerance.

**Speed.** In a full production build V2 is **flat** vs v1 vs prod (CF is ~0.6%
of the build; the value call is inner-solve-bound) — no wall-clock gain, and it
gives up v1's bit-identity. **Recommendation: keep v1 (`autarky_cf.jl`) as the
production oracle default; adopt V2 only for the CF-only-sweep niche** (γ'/A_dd
schedules at fixed draws that need only the CF column, e.g.
`delta_star_schedule`/`gamma_profile`), where V2 builds the CF column **35.7×
faster** than recomputing it from raw Uσ because it skips the W-length σ-power.

## 1. Verified exact formula, scalar, exponent, sign (from code)

Production CF column, raw, from `hFunctionCounter!`'s autarky branch
(`moments/hFunction.jl:201`), which the production constructor
`EK_moments_gammanorm_directgp!` calls with `w=wPrime, τ=τPrime, γ=γ_prime,
Aod=AodPow, L=LPrime` and the `Uσ` argument bound to the materialized `UσPow`:

```
G[s, D^2+1] (raw) = constConsσ[bi,bi] / UσPow[s,o1] − denom[bi]
  constConsσ[bi,bi] = wPrime[bi]^(1-σ) · (AodPow[bi,bi]·τPrime[bi,bi])^(1-σ)   (= cf_num)
  denom[bi]         = γ'[bi]^σ · (wPrime[bi]·LPrime[bi])                        (= cf_denom)
  UσPow[s,o1]       = Uσ[s,o1]^(-μ),   o1 = bi (UoModel==1)
then /Γ(μ(1-σ)+1) and ·SamplingWeights[s], exactly as every column.
```

`AodPow[bi,bi]` is built (`moments_gammanorm.jl:246-251`) as
`AodPow[bi,bi] = (Aod[bi,bi]/cHat[bi,bi])^(-μ)` with
`Aod[bi,bi]/cHat[bi,bi] = Aod_θ[bi,bi] · B_bi` (the `cHat` cancels), where
`B_bi = ((wHat[bi]·τ[bi,bi])/(wHat[1,1]·τ[1,bi]))^(1/μ) · (λ[bi,bi]/λ[1,bi])` is
built entirely from DATA (`wHat, τ, P`) and fixed μ. Hence, with
`A_dd ≡ Aod_θ[bi,bi]` (the raw θ entry, index `3+D+(bi-1)D+bi`):

```
cf_num = [ wPrime[bi]^(1-σ)·τPrime[bi,bi]^(1-σ)·B_bi^(-μ(1-σ)) ] · A_dd^(-μ(1-σ))
       = C_num_fixed · A_dd^(μ(σ-1))
```

- **Exponent on raw A_dd: `μ(σ-1)`** (verified numerically to rel-err 3.7e-16;
  at the calibration μ=1/6, σ=5/2 ⇒ exponent = 0.25). The hint `A_dd^(σ-1)`
  omits the μ factor.
- **Exponent under the AodPow convention: `AodPow[bi,bi]^(1-σ)`**, i.e. `1-σ`.
- **`cf_denom = γ'[bi]^σ·wPrime[bi]·LPrime[bi]` does not depend on A_dd** — it is
  a pure additive constant that shifts only with the objective variable γ'[bi].

The production/v1 code computes these two scalars in `autarky_cf.jl::autarky_cf_scalars`
from the primed quantities; **V2 reuses that function verbatim**, so the scalar,
its exponent and its sign are byte-identical to the already-adopted path — V2 does
not re-derive them. (The exponent above is proved separately in the test as a
cross-check, not relied on in the hot path.)

## 2. What is fixed vs A_dd-varying (independence verified)

| quantity | status across the whole outer run |
|---|---|
| μ = θ[1] | **fixed** — `context.jl:40` pins `θ_lo[1]==θ_hi[1]`, index 1 in `fixed_idx` |
| σ = θ[2] | **fixed** — `context.jl:39` pins `θ_lo[2]==θ_hi[2]`, index 2 in `fixed_idx` |
| draw matrix Uσ, wHat, τ, cHat, P (λ) | **fixed** — data, set once per ctx |
| wPrime[bi]≡1, τPrime[bi,bi], LPrime[bi] | **fixed** — autarky sets wPrime[bi]=1; τPrime/LPrime are data |
| `inv_uσ[s]=1/Uσ[s,o1]^(-μ)` | **fixed** ⇒ cached once |
| **A_dd = Aod_θ[bi,bi]** (θ index 3+D+(bi-1)D+bi) | **free** — enters cf_num only, as `A_dd^(μ(σ-1))` |
| γ'[bi] = θ[3+D] (objective var) | **free** — enters cf_denom only |
| all 15 OTHER A_od entries | **do not enter the CF column at all** |

**Independence check (test C3):** perturbing each of the 15 non-`(bi,bi)` A_od
entries by ×1.7 changes `cf_num` and `cf_denom` by exactly **0.0**. The CF column
reads only `AodPow[bi,bi]`, which depends on `Aod_θ[bi,bi]` alone. The single-cached-
vector premise is therefore sound: the base vector never needs invalidation as the
optimizer moves any A_od entry (it depends on none of them), and A_dd/γ' enter only
through the two scalars.

## 3. Implementation and composition choice

`autarky_cf_v2.jl` layers **on top of** `autarky_cf.jl` (it `include`s it and
reuses `autarky_cf_scalars`). `AutarkyCFBase` holds the once-built `inv_uσ` plus
the `(μ, o1)` it was built at, validated (never assumed) on every use — if ever
asked for a different μ/o1 it rebuilds, so it is safe outside the fixed-μ
convention too (degrades to "rebuild on change", never returns a stale vector).
`fill_autarky_cf_column_v2!` is the single per-draw `cf_num·inv_uσ[s] − cf_denom`
loop. `EK_moments_gammanorm_directgp_autarkyCF_v2!` mirrors the v1 constructor
byte-for-byte except the CF-column fill; the factual `hFunction!`, the UPow/UσPow
materialization it needs, and all post-processing are verbatim. `enable_autarky_cf_v2!(ctx;
pow_cache)` opt-in-wires it by the same single-field rebind as `enable_pow_cache!`
/`enable_autarky_cf!` and composes with the pow cache.

**Why layered on v1, not a replacement:** V2 needs v1's scalar computation
(guarantees identical A_dd/γ' dependence) and v1 remains the bit-identical
reference/fallback for the production oracle. V2 is the *further* specialization
for CF-only sweeps.

## 4. Equivalence (all PASS — `test_autarky_cf_v2.jl`)

- **Target A (vs generic `EK_moments_gammanorm_directgp!`)** and **Target B (vs
  v1 `autarky_cf.jl`)**, full G and K, 11 points (calibration + upper + lower + 8
  random-feasible): worst **abs = 3.55e-15** (≈ one reciprocal ULP of the O(1)
  numerator), floored-rel ≤ 1.8e-10 only where the centered CF column crosses
  zero. Base built once, reused 10×.
- **Target C1 (exponent):** `cf_num(a1)/cf_num(a2)` vs `(a1/a2)^{μ(σ-1)}` over 5
  A_dd values: worst rel-err **3.7e-16**; `cf_denom ⊥ A_dd` exactly (0.0).
- **Target C2 (reuse across A_dd):** SAME cached base reused across 5 distinct
  A_dd (`n_build=1, n_reuse=4`); CF column matches generic to abs **3.55e-15** at
  every A_dd — proving the cache is genuinely reusable without invalidation.
- **Target C3 (independence):** 0.0 over all 15 other-A_od perturbations.
- **Live wiring (benchmark C):** `evaluate_fullA` with `enable_autarky_cf_v2!` vs
  generic ⇒ **`Delta_dual` bit-identical, |diff|=0.00e+00** through the full inner
  KNITRO solve.
- **Baseline suite re-run clean on this worktree** before any new numeric claims:
  `test_oracle`, `test_lfix_incremental`, `test_composite_gradient_fast`,
  `test_moments_fast`, `test_compressed_moments`, `test_compressed_cc_inner`,
  `test_winner_certificate`, `test_autarky_cf` — all PASS.

### Bit-identical vs tight — the deliberate trade

`cf_num·(1/UσPow)` ≠ `cf_num/UσPow` bit-for-bit (one extra rounding on the cached
reciprocal). So V2 is **tight (≤1 ULP)** where v1 is **exactly bit-identical**.
The `-cf_denom` centering means the CF column crosses zero, so absolute error
(≤3.55e-15), not relative error, is the meaningful metric. A bit-identical
divide-preserving variant is possible (cache `UσPow[:,o1]` and keep the divide)
but that defeats the task's "single multiply, no per-draw divide" goal, so V2
takes the multiply and the ≤1-ULP cost.

## 5. Benchmark (D=4, W=8000, JULIA_NUM_THREADS=20, N=200 median, function-barrier'd)

**(A) Isolated CF component, UσPow already materialized** (the full-build context,
where the factual block materializes UσPow anyway):

| construction | median | alloc |
|---|---|---|
| generic `hFunctionCounter!` (autarky) | 0.0290 ms | 65 288 B |
| v1 `autarky_cf.jl` (scalars + divide) | 0.0068 ms | 0 B (4.24× vs gen) |
| **v2 cached-base (scalars + multiply)** | 0.0064 ms | 0 B (**4.50× vs gen, 1.06× vs v1** ≈ tie) |

**(A′) CF column FROM RAW Uσ** (the CF-only-sweep scenario — no pre-materialized
UσPow; v1/generic pay a W-length σ-power per call, v2 hits its once-built cache):

| construction | median |
|---|---|
| v1 from-scratch (power + divide) | 0.2110 ms |
| **v2 cached-base (multiply only)** | 0.0059 ms (**35.7× vs v1**) |

**(B) Full moment build** (`obj.moments!`) — flat, six configs:

| config | median | alloc |
|---|---|---|
| prod (generic CF, recompute pow) | 5.060 ms | 4 224 696 B |
| prod + pow_cache | 4.836 ms | 3 690 056 B |
| v1 autarkyCF (recompute pow) | 5.142 ms | 4 115 384 B |
| v1 autarkyCF + pow_cache | 4.661 ms | 3 581 848 B |
| v2 cached-base (recompute pow) | 5.137 ms | 4 115 224 B |
| v2 cached-base + pow_cache | 4.624 ms | 3 581 784 B |

**(C) Exact value-call** (`evaluate_fullA`, cold, inner KNITRO dominates):
`Delta_dual` **bit-identical (|diff|=0.00e+00)**; generic 29.87 ms vs v2 28.82 ms
= 1.04× (noise). Base `n_build=1, n_reuse=81` over the call.

## 6. Recommendation

- **Keep v1 (`autarky_cf.jl`) as the production oracle default.** For the actual
  D=4 upper/lower runs the CF column is ~0.6% of the moment build and the value
  call is inner-solve-bound, so V2 delivers **no wall-clock gain** over v1 there,
  and it gives up v1's exact bit-identity for ≤1-ULP tight equivalence. v1 stays
  the right default; the generic `hFunctionCounter!` stays the reference/fallback
  and for every non-autarky counterfactual.
- **Adopt V2 (`enable_autarky_cf_v2!`) for CF-only sweeps** — γ'/A_dd schedules at
  fixed draws that evaluate only the focal-autarky CF moment (e.g.
  `delta_star_schedule`, `gamma_profile`-style traces). There V2's cached base
  eliminates the W-length σ-power entirely, building the CF column **35.7× faster**
  than recomputing from raw Uσ, at ≤1-ULP agreement (and bit-identical downstream
  `Delta_dual`). This is the genuine, well-scoped payoff of the further
  specialization; it does not exist inside a full build because UσPow is already
  materialized there for the factual block.

Net: the further specialization is **correct, verified, and worth having as an
opt-in tool for the sweep use-case**, but is **not** a reason to change the
production oracle, which v1 already serves optimally.
