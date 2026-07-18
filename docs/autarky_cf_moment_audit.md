# Focal-autarky counterfactual price-index moment — audit + specialized path

Branch `diag/fullA-d4-autarky-cf` (worktree
`.../.claude/worktrees/manual-autarky-cf`, off `86caa75`). All work is **additive**
in new files under `full_aod_diag/d4_exact/`; the trusted dense constructor
(`full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!`), the generic
`hFunctionCounter!` (`moments/hFunction.jl`), the tie-breaking rule (`MinInd!`),
`lfix_incremental.jl`, and `composite_gradient_fast.jl` were **not modified**.

New files:
- `full_aod_diag/d4_exact/autarky_cf_audit.jl` — data-convention + formula audit.
- `full_aod_diag/d4_exact/autarky_cf.jl` — specialized path + `enable_autarky_cf!` wiring.
- `full_aod_diag/d4_exact/test_autarky_cf.jl` — bit-identical moment-build equivalence.
- `full_aod_diag/d4_exact/verify_autarky_cf_wiring.jl` — live `evaluate_fullA` field-for-field + gradient.
- `full_aod_diag/d4_exact/benchmark_autarky_cf.jl`, `benchmark_autarky_cf_valuecall.jl` — benchmarks.

## TL;DR

The current autarky counterfactual is **already** a direct domestic-only O(W)
broadcast — the generic full counterfactual loop is dead code under `counterType==1`.
Most of the inefficiencies the brief asks about are therefore **absent**. The real,
measurable waste is confined to the `hFunctionCounter!` **call wrapper** (per-chunk
allocations + a full unused `D×D` `constCons`/`constConsσ`/`denom` rebuild + a
`W`-length column copy inside its broadcast). The specialized path removes exactly
these: it is **3.55× faster and allocates 64.8 KB less on the isolated CF component**,
**bit-identical** (max|Δ|=0.0) everywhere, but — because the CF column is only ~0.6%
of the full moment build and the moment build is a minority of the inner-solve-bound
value call — the **full-build and value-call wall times are unchanged** at D=4/W=8000.
Its value is (a) a modest allocation/GC reduction, (b) removal of dead work that grows
as O(Th·D²) with dimension (a D-scaling hygiene win), and (c) a provably-minimal,
audit-clear CF construction. **Recommended for adoption behind `enable_autarky_cf!`**,
with the generic path retained as reference/fallback and for all non-autarky counterfactuals.

## 1. Verified current formula, normalization, and code path (read from code + data)

Active constructor for these runs: `EK_moments_gammanorm_directgp!`
(confirmed from `context.jl`: `(moments!) = EK_moments_gammanorm_directgp!`), autarky
`counterType==1`, `UoModel==1`, `OuterScaling==1`, `θConstant==0`. The counterfactual
price-index column is produced by `hFunctionCounter!`'s **autarky branch**
(`moments/hFunction.jl:192–216`), whose active line is a single broadcast (line 201):

```julia
@. G[:, D^2+1] = constConsσ[baseIndex,baseIndex] ./ Uσ[:, o1] .- denom[baseIndex]
```

(the `Uσ` argument here is the pre-transformed `UσPow = Uσ.^(-μ)`; the generic
per-origin loop and counterfactual-wage code at lines 140–191 are the `counterType!=1`
branch and never execute under autarky). Exact raw formula, verified line-by-line and
numerically (`autarky_cf_audit.jl`):

```
G[s, D^2+1] (raw) = constConsσ_cf[bi,bi] / UσPow[s,bi]  -  denom_cf
  constConsσ_cf[bi,bi] = wPrime[bi]^(1-σ) · (AodPow[bi,bi]·τPrime[bi,bi])^(1-σ)
  denom_cf             = γ'[bi]^σ · (wPrime[bi] · LPrime[bi])
  UσPow[s,bi]          = Uσ[s,bi]^(-μ)     (focal origin bi's σ-draw term)
then ÷gammafac=Γ(μ(1-σ)+1) and ×SamplingWeights[s], exactly as every column.
```

This confirms the brief's premise exactly: the counterfactual supplier is
**mechanically bi on every draw** (`p^aut_bi(s) = p_{bi,bi}(s)`); the column reads only
column `bi` of `UσPow`; there is **no minimization, no winner search, no price vector
for o≠bi**.

### Factual-vs-counterfactual domestic scalar (the brief's specific question)

Measured at the upper candidate (`autarky_cf_audit.jl`, and calibration-invariant since
these are data constants):

| quantity | factual (o=d=bi) | counterfactual (autarky) |
|---|---|---|
| wage | `wHat[bi] = 1` | `wPrime[bi] = 1` |
| domestic trade cost | `τ[bi,bi] = 1` | `τPrime[bi,bi] = 1` |
| focal labor | `L[bi] = 1.7763294…` | `LPrime[bi] = 1.7763294…` (equal) |
| CES **numerator** scalar | `AodPow[bi,bi]^(1-σ) = 1.31222602…` | `AodPow[bi,bi]^(1-σ) = 1.31222602…` (**identical**) |
| **denom** (constant) | `1·wHat[bi]·L[bi] = 1.77633` | `γ'[bi]^σ·LPrime[bi] = 1.60802` (**differs**) |

**Answer to "are the factual and counterfactual domestic price terms identical?"**
The per-draw **CES numerator is bit-identical** (`constConsσ_cf[bi,bi] ==
constConsσ_fac[bi,bi]`, ratio = 1.0 exactly) because `wPrime[bi]=wHat[bi]=1` and
`τPrime[bi,bi]=τ[bi,bi]=1` in this calibration — the shared draw term is
`1/UσPow[s,bi]`. The **only** thing that differs is the additive constant `denom`:
`denom_cf = γ'[bi]^σ · LPrime[bi]` versus `denom_fac = L[bi]`. Since `LPrime[bi]=L[bi]`
here, the sole operative difference is the free **`γ'[bi]^σ` factor** (γ'[bi]=θ[3+D],
the optimizer's variable). The specialized path incorporates this scalar exactly by
computing `denom_cf` from the primed quantities (`γ'`, `wPrime`, `LPrime`, `τPrime`),
so it stays correct even if a future calibration breaks the `w'=w`, `τ'=τ`, `L'=L`
coincidences.

### Audit of the specific inefficiency questions

| brief's question | verdict for the CURRENT autarky path |
|---|---|
| calls a **generic** counterfactual price routine? | **No.** The specialized `counterType==1` branch runs; the generic per-origin loop (lines 140–191) is dead code under autarky. |
| fills prices for all origins with +∞ except d? | **No.** No price vector is filled at all in the autarky branch. |
| repeats powers / logs / exps? | **Partially — yes, dead work.** `hFunctionCounter!` rebuilds the full `D×D` `constCons` (**never used** under autarky), the full `D×D` `constConsσ` (only `[bi,bi]` used), `wPow` (D), and the full `denom` vector (only `[bi]` used) — once **per thread-chunk** (×`Th`), i.e. O(Th·D²) redundant work per build. |
| allocates a counterfactual price matrix? | **Yes (dead).** `pricesCounterVec = zeros(D²)` plus `pricesTemp/pricesTempσ/pricesInd/denom/constCons/constConsσ/wPow` are all allocated per call and unused/underused in autarky. Additionally, the broadcast on line 201 materializes a **W-length column copy** (`Uσ[:,o1]` is not `@view`-ed): measured **+64.8 KB/call**. |
| performs a redundant min or argmin? | **No.** No `MinInd!`/`findmin` in the autarky branch. |
| recomputes the domestic term despite the factual pass? | **Yes, but cheap.** `hFunction!` computes the factual domestic σ-value `constConsσ_fac[bi,bi]/UσPow[ω,bi]` for destination bi on every draw and discards it (loop scalar); the CF recomputes the same per-draw `1/UσPow[s,bi]` (1 divide/draw). Reusing it would require storing a W-vector during the factual pass and a multiply in the CF — the same per-draw cost plus a W-allocation, i.e. **dominated by the direct broadcast**; not worth it. |

## 2. Specialized path (`autarky_cf.jl`)

`EK_moments_gammanorm_directgp_autarkyCF!` is a byte-for-byte mirror of the production
constructor **except** it replaces the per-chunk `hFunctionCounter!` call with
`autarky_cf_scalars` (two scalars, no D×D rebuild, no allocation) +
`fill_autarky_cf_column!` (one `@views` O(W) broadcast, no column copy). The factual
`hFunction!` and every post-processing step are called verbatim, so output is
bit-identical. `enable_autarky_cf!(ctx; pow_cache=…)` opt-in-wires it by rebinding the
mutable `obj.moments!` field — the same single-field-mutation mechanism as
`enable_pow_cache!` — and composes with the pow cache.

## 3. Equivalence (all PASS)

- **`test_autarky_cf.jl`** — specialized moment build vs production, 11 points
  (calibration + 2 candidates + 8 random feasible), with and without a shared
  `MuSigmaPowCache`: **max|ΔG| = max|ΔK| = max|ΔCF-column| = 0.0 (bit-identical)**.
- **`verify_autarky_cf_wiring.jl`** — live `evaluate_fullA`, generic-CF ctx vs
  autarky-CF-wired ctx, field-for-field at `tol = 0.0`, 8 points: **worst|Δ| = 0.0** on
  every field — optimized dual variables (`λ`, `ζ`), primal & dual divergence
  (`Delta_primal`, `Delta_dual`), recovered weights, moment residual, KKT residual,
  gravity value/R-family, winner hash, inner status. Divergence-gradient (central FD)
  **bit-identical** at the feasible candidate (at calibration all 17 components are NaN
  in **both** ctx — FD steps infeasible identically — i.e. still identical).
- **Baseline suite** re-run clean on this worktree before any new numeric claims:
  `test_oracle`, `test_lfix_incremental`, `test_composite_gradient_fast`,
  `test_moments_fast`, `test_compressed_moments`, `test_compressed_cc_inner` — all PASS.

## 4. Benchmark (D=4, W=8000, JULIA_NUM_THREADS=20, median of N reps)

**(A) Isolated CF component** — the direct measurement of the specialization (writes
`G[:,D²+1]` only):

| construction | median | alloc/call |
|---|---|---|
| generic `hFunctionCounter!` (autarky branch) | 0.0318 ms | 66 088 B |
| specialized scalars + `@views` broadcast | 0.0089 ms | 1 248 B |
| **speedup** | **3.55×** | **−64 840 B** |

**(B) Full moment build** (`obj.moments!`), four configs:

| config | median | alloc/call | vs prod |
|---|---|---|---|
| prod (generic CF, recompute pow) | 4.952 ms | 4 224 376 B | 1.00× |
| prod + pow_cache | 4.841 ms | 3 690 056 B | 1.02× |
| autarkyCF (recompute pow) | 5.057 ms | 4 115 000 B | 0.98× |
| autarkyCF + pow_cache | 4.836 ms | 3 581 880 B | 1.02× |

The full build is **flat** — the CF column is ~0.6% of it (the factual O(W·D²)
bilateral block dominates). The specialization removes **~109 KB/call** (CF column copy
+ `hFunctionCounter!` setup allocs); the pow cache removes the larger ~534 KB (UPow/UσPow);
together ~642 KB/call (15%).

**(C) Exact value-call** (`evaluate_fullA`, cold, inner KNITRO solve dominates):
bit-identical `Delta_dual` (|Δ|=0.0). Interleaved on a **single** ctx (swapping the
`moments!` binding between reps, N=60) generic vs autarkyCF = **1.004×** (identical
within noise; the ~28 ms value call is inner-solve-bound). The naive cross-ctx 0.90×
seen when timing two separate ctx objects was KNITRO run-to-run variance, not a
regression — the moment build is bit-identical with fewer allocations.

## 5. Recommendation

**Adopt the specialized autarky path for these runs, behind `enable_autarky_cf!`**, and
**retain the generic `hFunctionCounter!` as reference/fallback** and for every
non-autarky counterfactual. Justification, honestly scoped:

- It is **provably equivalent** (bit-identical G/K and every downstream live-oracle
  field, including duals, divergences, weights, gradients).
- It **removes real dead work**: the unused `D×D` `constCons`/`constConsσ`/`denom`
  rebuild and per-chunk allocations, plus a W-length column copy — a **3.55× / 64.8 KB**
  win on the CF component and **~109 KB/call** on the full build.
- The **wall-time win at D=4/W=8000 is negligible** (CF is ~0.6% of the build; value
  call is inner-solve-bound) — do not expect a live speedup here. The payoff grows with
  **D** (the removed setup is O(Th·D²)) and is the natural companion to the pow-cache
  and compressed-moment D-scaling levers.
- Compose it with `enable_pow_cache!` (both are single-field-mutation opt-ins) for the
  full ~15%/642 KB per-call allocation reduction.

For the immediate D=4 upper/lower runs the correctness/clarity and allocation/GC
reduction are the benefit; treat the specialization primarily as **D-scaling hygiene**,
not a D=4 wall-clock win.
