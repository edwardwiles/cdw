# `price0`/`pTσ0` call-site audit (addendum Step 1)

Written 2026-07-21/22, for the persistent-preallocation + tensor-elimination experiment.
Scope confirmed by `grep -rl LFixBaseCache` / `grep -rl build_lfix_base_cache`: **every**
consumer of these two tensors lives in `full_aod_diag/d4_exact/` (~30 files reference the
cache/builder, but only construct-and-pass-through; the dense tensors themselves are read
directly in exactly 5 files, listed below). Nothing outside this directory touches them.

## 1. What the two tensors are

Defined in `lfix_incremental.jl:262-263` (`LFixBaseCache` struct), built in
`build_lfix_base_cache` (`lfix_incremental.jl:381-493`):

```julia
price0 = Array{Float64}(undef, W, D, D)   # levels: price0[ω,o,d] = delivered price
pTσ0   = Array{Float64}(undef, W, D, D)   # sigma-transformed: constConsσ/Uσ^(-μ)
```

Filled in one loop (`lfix_incremental.jl:397-399`) via `price_and_pTsigma_cell!`, which computes
**both from independent formula chains** (not `pTσ = price0.^(1-σ)` computed literally):

```julia
price = constCons_od ./ (U[:,o] .^ (-μ))          # price0
pTσ   = constConsσ_od ./ (Uσ[:,o] .^ (-μ))         # pTσ0, constConsσ_od = constCons_od^(1-σ)
```

`constConsσ_od == constCons_od^(1-σ)` exactly (verified algebraically:
`wHat^(1-σ)*(AodPow*τ)^(1-σ) == (wHat*AodPow*τ)^(1-σ)`). **`Uσ == U.^(1-σ)` element-wise,
confirmed** — computed exactly once, this way, at context-build time
(`prepare_cc/createUDerivatives!.jl:18`: `Uσ = U .^ (1 - σHat)`), stored as a separate array in
`ctx.γ.Uσ`, never re-derived from `U` again afterward. So `pTσ0 == price0.^(1-σ)` holds
**exactly** (up to ordinary floating-point rounding from the two formulas' different evaluation
paths, not an approximation) — confirmed the hard way: Backend B's first tie-fixture test
(`test_lfix_pTsigma_only.jl`) patched only `ctx.U` (mirroring `test_winner_certificate.jl`'s own
established tie-injection recipe) and did NOT reproduce a tied `pTσ0`, because `ctx.γ.Uσ` is
stored SEPARATELY from `ctx.U` and doesn't auto-update — patching both consistently (`Uσtie =
Utie.^(1-σ)`) fixed it. This is a real, useful finding about this codebase's data layout (`U`
and `Uσ` are two independently-mutable arrays that happen to satisfy a fixed relationship at
construction, not a single source of truth), not a flaw in the monotonicity argument itself.

**Mathematical fact this experiment leans on**: since `σ > 1`, `x ↦ x^{1-σ}` is strictly
*decreasing* on `x > 0`. So `argmin_o price0[ω,o,d] == argmax_o pTσ0[ω,o,d]` for every
`(ω,d)` — ranking by ascending price is exactly ranking by descending pTσ. This is the
license for Backend B (drop `price0`, rank via `pTσ0` with flipped comparisons) and is
already partially exploited by a *different* file (`winner_certificate.jl`, see §4).

## 2. Exact reader/writer map

| File | Line(s) | Reads/writes | Classification | Notes |
|---|---|---|---|---|
| `lfix_incremental.jl` | 391-399 | **writer**, both tensors | construction | `price_and_pTsigma_cell!` fills both densely, D² cells × O(W) each |
| `lfix_incremental.jl` | 336-350 (`detect_price_ties`) | reads `price0` (dense scan) | tie handling | exact bit-equality scan for 2+ origins tied at the row min; O(W·D) |
| `lfix_incremental.jl` | 418-423 (`min_secondthirdmin_with_idx` over `price0[ω,:,d]`) | reads `price0` (dense scan) | winner/runner-up/third ranking | THE initial O(W·D²) ranking pass that populates `winner0/runnerup0/third0` + their `*_price0` levels |
| `lfix_incremental.jl` | 424-432 | reads `pTσ0` (indexed at `third0`) | CES/moment value at 3rd place | builds `third_pTσ0`, free given `pTσ0` already dense |
| `lfix_incremental.jl` | 444-449 (`contrib0` build) | reads `pTσ0` at winner | economic moment (contribution formula) | `contrib0[ω,d]` — the actual q0-building CES aggregate, uses ONLY `pTσ0[ω,winner,d]`, never raw price |
| `lfix_incremental.jl` | 561 (`dest_contrib_incremental`) | reads `price0[ω,o,d]` (col fallback) | winner runner-up ranking (O(D) rescan fallback) | 2-changed-origin generic tier |
| `lfix_incremental.jl` | 564 | reads `pTσ0[ω,wo,d]` | economic moment | same function, transformed value at resolved winner |
| `lfix_incremental.jl` | 604, 606 (`dest_contrib_incremental_top3`) | reads `pTσ0` at r1/r2 | economic moment | top-3-cache fast path |
| `lfix_incremental.jl` | 623 | reads `price0[ω,o,d]` (defensive fallback) | winner ranking | unreachable-in-practice full rescan (D≤3 & \|Cd\|≥3 edge case) |
| `lfix_incremental.jl` | 626 | reads `pTσ0[ω,bo,d]` | economic moment | same defensive fallback |
| `lfix_incremental.jl` | 670 (`dest_contrib_incremental_o1`) | reads `pTσ0[ω,wo,d]` | economic moment | the TRUE O(1) single-changed-origin tier |
| `composite_gradient.jl` | 90-98 (`count_winner_flips`) | none (uses `winner_price0`/`runnerup_price0`, not the dense tensor) | winner-switch check | pure winner-flip counting for bandwidth selection, never touches `pTσ0` at all |
| `composite_gradient.jl` | 127, 184 | reads `price0[ω,o,d]` (col fallback) | winner-switch check / ranking | same 2-changed-origin fallback pattern, duplicated for the *counting*-only variant |
| `bandwidth_quantile.jl` | 87-115 (`exact_flip_thresholds!`) | reads `price0[ω,o,d]`, `winner_price0`, `runnerup_price0` | near-tie / bandwidth-crossing selection | `log(threshold_price/price0[ω])/c` — closed-form flip-time; **provably rewritable in pTσ-space** (see §3) |
| `gradient_workspace.jl` | 102-116 (`dest_contrib_incremental_o1!`) | reads `pTσ0[ω,wo,d]` (in-place variant) | economic moment | pooled-workspace fast path, same formula as the allocating version |
| `c14_verify_lfix_buffer_fix.jl` | 52-53 | reads both (diagnostic print) | diagnostics only | one-time bit-for-bit equivalence check, not a hot path |

**Every single reader of `pTσ0` is an economic-moment/CES-contribution use** (`contrib0`,
`dest_contrib_*`, `cf_contrib_at`-adjacent). **Every reader of raw `price0` is ranking, tie
detection, or near-tie bandwidth selection** — never a CES/moment computation. This is a clean
split: *nothing* in this codebase uses raw price for anything except comparisons.

## 3. Is raw `price0` mathematically required anywhere? — No.

Checked every ranking/tie/bandwidth call site above against the σ>1 monotonicity fact:

- **Winner/runner-up/third ranking** (`min_secondthirdmin_with_idx`, all O(D) rescans): argmin
  price ⟺ argmax pTσ. Direct substitution, just flip `isless`/comparison direction.
- **Exact ties** (`detect_price_ties`): a tie in price is a tie in pTσ (strict monotone bijection
  on positive reals) — tie detection transfers exactly, no tolerance-widening.
- **O(1) winner update** (`update_winner_o1`, all its `<=`/`<` comparisons): every comparison is
  between two price *levels*; rewriting in pTσ-space flips every inequality direction (since the
  map is decreasing) but is otherwise a mechanical substitution.
- **Bandwidth quantile's closed form** (`bandwidth_quantile.jl` §26-37): `h_flip =
  log(thr/p0)/c`. Since `pTσ = price^{1-σ}`, `log(thr_pTσ/p0_pTσ) = (1-σ)·log(thr/p0)`, so
  `log(thr/p0) = log(thr_pTσ/p0_pTσ)/(1-σ)` — the SAME `h_flip` value, computed from pTσ ratios
  with one extra scalar division by the constant `(1-σ)`. Not an approximation; exact algebraic
  identity.

**Conclusion**: raw `price0` is never load-bearing for anything except monotone comparisons.
Backend B (§4 of the companion implementation, not yet built) can eliminate the whole
`price0` tensor, keeping only `pTσ0`, provided every comparison site above is rewritten with
flipped direction. Diagnostics that want to *print* a raw price for a specific (ω,o,d) can
reconstruct it on demand: `price = pTσ0^{1/(1-σ)}` (O(1) per queried cell, not a stored tensor).

## 4. Prior art already in this repo: `winner_certificate.jl`'s `WinnerRefCache`

**Important existing finding, not something this session built**: `winner_certificate.jl`
(continuation 7/8, "winner-margin certificate", see memory `fullA-continuation7-parallel-workstreams.md`)
already implements almost exactly what the addendum's Backend C (§5) asks for — a *factorized*
representation, **not** a dense `W×D×D` tensor:

```julia
S_{sod} = logCC_{o,d} + mulU_{s,o}     # log price = log(constCons_od) + mu*log(U_{s,o})
```

- `logCC0`: `D×D` bilateral matrix (one-time, O(D²))
- `mulU`: `W×D` origin-draw kernel (one-time, O(W·D), NOT O(W·D²))
- Winner/runner-up/third ranking via `top3_scan` over the *summed* score — exactly the
  factorization `K_{·o}·B'_{od}` the addendum's §5 describes, just in log-space (a sum, not a
  product, which is equivalent and numerically nicer — avoids the overflow/underflow concern
  in addendum §6, see below).
- `certified_winner_update`: an O(D²) (not O(W·D²)) *margin certificate* that certifies most
  draws' winners are provably unchanged after a bilateral shift, only rescanning the
  uncertified minority exactly — this is MORE aggressive than anything the addendum's Backend
  C sketch asks for, and already measured 8-18x on winner-recompute cost (per that file's own
  §"Measured justification": ~1.85ms cold vs ~6.5ms for `price0`/`pTσ0` construction, 3.5x
  before persistence; certified fraction 89-100% at realistic step sizes, per
  `docs/winner_certificate_report.md`).

**Why this hasn't already eliminated `price0`/`pTσ0` from `LFixBaseCache`**: scope, by explicit
design (own docstring, §518-539). `WinnerRefCache`'s persistence layer
(`PersistentWinnerCache`/`certified_winner_update`) is proven exact **only for the winner
identity** (and hence `wval` = pTσ at the winner) — the proof does NOT extend to runner-up/
third-place staleness under the *certified* (skip-most-draws) update path. `lfix_incremental.jl`'s
O(1)/top-3 incremental tiers (`dest_contrib_incremental_o1`, `_top3`) need EXACT runner-up and
third-place at every step (to handle the rare 2-changed-origin-same-destination case across
**400 sequential single-coordinate FD probes per gradient**, not just one nearby evaluation) —
a fundamentally different consumption pattern from `WinnerRefCache`'s "one certified nearby
point" use case. `coord_winner_update!` (`winner_certificate.jl` §442-516) DOES do exact
top-3-aware coordinate updates and would very plausibly serve `lfix_incremental.jl`'s need if
wired together — **this unification is exactly what Backend C should attempt**, reusing
`constCons_matrix`/`top3_scan`/`coord_winner_update!` rather than re-deriving a third
factorized representation from scratch.

## 5. Answers to the audit's own required questions

1. **Is raw `price0` mathematically required anywhere?** No (§3). Every use is a monotone
   comparison; σ>1 makes pTσ a strict order-reversing bijection of price.
2. **Can all winner/runner-up logic use transformed competitiveness?** Yes, mechanically —
   flip every comparison direction. Bandwidth-quantile's closed form transfers with one
   constant rescaling (÷(1-σ)).
3. **Does `LFixBaseCache` need a NEW factorized representation, or can it reuse
   `WinnerRefCache`'s?** The log-score factorization (`logCC + mulU`) already exists,
   already validated, already faster than `price0`/`pTσ0` construction — Backend C should
   adapt/reuse it rather than re-derive, but needs the exact-top-3-under-repeated-single-
   coordinate-perturbation guarantee `coord_winner_update!` already provides, NOT the
   certified/uncertified persistence layer (that layer's exactness scope is narrower than
   what `lfix_incremental.jl` needs).

## Next: Step 2 (benchmark current allocation/lifetime) and Backend A implementation.
