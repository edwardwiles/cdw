# Profiled cross-block formulas — exact mapping onto current code (2026-08-01)

This resolves the mission's abstract math (its §4-9) onto the concrete sufficient statistics
already in `full_aod_diag/d4_exact/`, using the H_EE reduction that a prior session in this branch
already built (`reduced_homogeneous_hessian_2026-08-01.jl`) as the load-bearing precedent for how
anchor omission must be represented — this is not a fresh derivation, it is forced to be consistent
with that already-validated code.

## 1. What "M_d(w)" actually is in this codebase

The mission's `M_d(w)` ("the winning offer/value for destination d") is **not** a winner-conditioned
quantity in the code's own representation — it is `cf.wval[w, slot]` (also exposed as `wctx.wval` on
`ReducedHomogeneousWinnerPairHessCtx`), documented there as "raw per-draw destination value, ALWAYS
DEFINED (winner-independent)". Confirmed identically in the OLD/full context builder
(`core_exact_hessian.jl:391`, `build_winner_pair_ctx`): `y[w,slot] = kappa0[j]*cf.wval[w,slot]` where
`j` is whichever column won — i.e. `wval` is the single per-(draw,destination) scalar that gets
kappa0-rescaled by whichever origin happens to win; the origin identity only ever selects *which*
kappa0 to apply, never changes the underlying value. This resolves the one real ambiguity in the
mission brief's abstract notation: `T^R_{d,k}` must be built from `wval` directly (raw, unconditional
on winner), **not** by summing kappa0-scaled per-origin winner buckets over `o` — those buckets each
carry a *different* origin's kappa0[j], so summing them would produce a kappa0-inconsistent quantity
that could not be pulled out as `pi_vec[j] * T^R_{d,k}` (a single scalar factored against the row's
own `pi_vec[j]`) the way the mission's formula requires. Using raw `wval` sidesteps the anchor's
"undefined kappa0" problem entirely: `wval[w,slot]` is defined for every draw regardless of who wins,
including draws where the omitted anchor wins.

This also means the "one destination-total accumulator in the same existing draw loop" the mission's
§6 asks for is **simpler** than a from-winner-buckets reconstruction: it does not need to be
winner-conditioned at all, and does not need an "anchor-sum" side table keyed on `winner_reduced_col
== 0`. It is a plain `Snu[w]*wval[w,slot]`-weighted bin/feature accumulation, exactly parallel to the
existing `NuTab`/`SOnlyTab` accumulators in `winner_pair_cross_hessian_fill!` (which are themselves
already un-winner-conditioned, run over `x in 1:D` for every draw) — just also indexed by destination
slot, since `T^R_{d,k}` (unlike `NuTab`) genuinely varies with `d`.

## 2. H_EC: `winner_pair_cross_hessian_fill!` / `winner_pair_cross_hessian_cm_block!`

**Old correction** (`winner_pair_cross_hessian.jl:206-210`, current code):
```julia
Hraw_EC[1, oi] = (SOnlyCScum[o, l] - SOnlyCScum[refIndex1, l]) * invM
nu_diff = NuCScum[o, l] - NuCScum[refIndex1, l]              # destination-INDEPENDENT
for j in 1:wctx.ncolI
    q_diff = QCScum[j, o, l] - QCScum[j, refIndex1, l]         # KEEP — the winner term, unchanged
    Hraw_EC[j + 1, oi] = (q_diff - pi_vec[j] * nu_diff) * invM  # nu_diff same for every j
end
```
`nu_diff` matches the mission's OLD formula exactly: `NuCScum[x,l] = Σ_{w:bin<=l} S[w]*nu[w]`
(mission's `S_w`), so `pi_vec[j]*nu_diff = λ_od(j) * Σ_w S_w R_k(w)` — a single global total, the
same for every economic row `j` regardless of its destination. This is the term the mission's §5
calls out as the thing that must change.

**New correction, additive, one new table**: add a THIRD accumulator alongside `NuTab`/`SOnlyTab` in
`winner_pair_cross_hessian_fill!`'s existing `for x in 1:D` per-draw loop (`winner_pair_cross_hessian.jl:136-145`):
```julia
# NEW, added to the SAME existing loop (one extra multiply-add per (w,x), no new pass):
MTab[slot, x, b] += Sw * nu[w] * wval[w, slot]     # requires promoting this loop from "for x" (D-shaped) to
                                                     # "for slot, for x" (Ddest*D-shaped) -- see note below
```
Because `T^R_{d,k}` genuinely varies with destination `d` (unlike `NuTab`/`SOnlyTab`, which do not),
this accumulator must be `Ddest x D x (L+1)`, not `D x (L+1)` — it cannot be folded into the existing
w-only loop (`for x in 1:D`, no slot dependence) without adding a `slot` dimension. The cheapest way
to do this without adding a second O(W) pass is to fold it into the file's OTHER existing per-draw
loop that already IS slotted — the `for slot in 1:Ddest; for w in 1:W` loop at
`winner_pair_cross_hessian.jl:147-160` (which builds `QTab`/`EsumEcon`) — adding one more line inside
its own `for x in 1:D` inner loop:
```julia
@inbounds for slot in 1:Ddest
    for w in 1:W
        o = winner[w, slot]; j = slot + (o - 1) * Ddest
        snuy = (S[w] * nu[w]) * y[w, slot]
        EsumEcon[j] += snuy
        mv = (S[w] * nu[w]) * wval[w, slot]        # NEW: winner-UNCONDITIONAL, uses wval not y
        for x in 1:D
            QTab[j, x, Bidx[w, x]] += snuy
            MTab[slot, x, Bidx[w, x]] += mv          # NEW
        end
    end
end
```
This requires threading `wval` into `WinnerPairHessCtx`/`WinnerBinCrossScratch`'s inputs (currently
absent from the OLD `WinnerPairHessCtx` — needs one new field, or simply reading `cf.wval` directly
since the fill function is always called with the live `cf` in scope at its call sites). Cost:
`O(W*Ddest*D)`, identical complexity class to the existing `QTab` fill it rides alongside (literally
the same loop nest, one more array write) — **zero new W-scale passes**, matching the mission's
explicit requirement.

Cumulative sum (added next to the existing `NuCScum`/`SOnlyCScum`/`QCfCScum` prefix-sum loop,
`winner_pair_cross_hessian.jl:163-172`):
```julia
MCScum = ws.MCScum   # Ddest x D x L, new field on WinnerBinCrossScratch
@inbounds for slot in 1:Ddest, x in 1:D
    acc = 0.0
    for l in 1:L
        acc += MTab[slot, x, l]
        MCScum[slot, x, l] = acc
    end
end
```

**Amended block fill** (`winner_pair_cross_hessian_cm_block!`, `winner_pair_cross_hessian.jl:198-222`)
— the row-1/keep-term lines are untouched; only the per-row correction changes, and it now needs to
know row `j`'s destination (`target_slot[j]`, already computed identically in
`reduced_homogeneous_hessian_2026-08-01.jl` — for the OLD full-index case this is simply
`((j - 1) % Ddest) + 1` from the `j = slot + (o-1)*Ddest` convention, or an explicit
`wctx.target_slot` field mirroring `ReducedHomogeneousWinnerPairHessCtx`'s own field of that name):
```julia
@inbounds for (oi, o) in enumerate(origins)
    Hraw_EC[1, oi] = (SOnlyCScum[o, l] - SOnlyCScum[refIndex1, l]) * invM
    for j in 1:wctx.ncolI
        q_diff = QCScum[j, o, l] - QCScum[j, refIndex1, l]        # UNCHANGED — the keep term
        d = wctx.target_slot[j]
        T_diff = MCScum[d, o, l] - MCScum[d, refIndex1, l]         # NEW — replaces nu_diff, now destination-specific
        Hraw_EC[j + 1, oi] = (q_diff - pi_vec[j] * T_diff) * invM
    end
    # cf row: same pattern, using MCScum[bi_slot, ...] in place of nu_diff
end
```
Anchor rows are never emitted (per `ProfiledEconomicMomentLayout`, `j` only ranges over retained
columns) but `MCScum` correctly includes the anchor's own contribution to `T_diff` because it was
built from `wval` unconditionally, not from `QTab`'s winner-conditioned buckets — no separate
anchor-sum table is needed, resolving the mission §6 concern by construction rather than by a second
accumulator.

**Dispatch, not duplication**: the OLD (full-index) callers (flexible CM, common Fréchet, CM+ZC —
none of which use the profiled layout yet) must keep getting the OLD destination-independent
`nu_diff` behavior unchanged. The cleanest surgical route, preserving one function body: give
`WinnerPairHessCtx` (old) a `target_slot` field too (trivial, `((j-1) % Ddest)+1`, or store it
directly at construction like `ReducedHomogeneousWinnerPairHessCtx` already does) and populate
`MCScum` identically for both — for the OLD/full ctx, `MCScum[d,x,l]` summed over `d` for a FIXED
`x,l` collapses back to `Ddest * (a destination-independent quantity)` **only if** every destination
contributes equally, which is not generally true, so **the OLD ctx must keep its own `nu_diff`
formula** (destination-independent, matching its own un-reduced normalization convention — the OLD
family's economic block is not the same statistical object as the profiled one, it does not have a
"T^R_{d,k}" concept at all in its own math, only `Σ_w S_w R_k(w)`). Do not force old callers through
the new `MCScum` path. Instead: add the new `T_diff`/`MCScum` machinery as an **additive** capability
(new fields on `WinnerBinCrossScratch`, filled unconditionally since it's cheap, but only *read* by a
new code path), and give `winner_pair_cross_hessian_cm_block!` a `use_profiled_correction::Bool`
keyword (default `false`, preserving current behavor bit-for-bit for the four families/contexts not
yet using the profiled layout) that switches which correction term is applied. This is the literal
reading of the mission's "surgical, in-place amendment, preserve everything else" instruction applied
to a function that currently has exactly one caller family for the profiled path (none, yet) and four
for the old path.

## 3. H_EZ: `winner_pair_cross_hessian_zc_block!`

Old correction (`winner_pair_cross_hessian.jl:395-406`):
```julia
NuZ = Z' * Snu                      # BLAS.gemv!, destination-INDEPENDENT, single (nx,)-length vector
HEZ[j+1, x] = invM * (HEZ[j+1, x] - pi_vec[j] * NuZ[x])    # SAME NuZ for every j
```
New correction: replace the single global `NuZ` (`nx`-vector) with a `(Ddest x nx)` matrix
`TZ[d, x] = Σ_w Snu[w] * wval[w, d] * Z[w, x]`, computed as **one BLAS gemm**, not a loop — build
`SnuWval[w, d] = Snu[w] * wval[w, d]` once (`W x Ddest`, O(W*Ddest), trivial), then:
```julia
TZ = SnuWval' * Z    # (Ddest x nx) = (Ddest x W) * (W x nx), one BLAS.gemm! call
```
against the same `Z` the row-1/cf-row `gemv!` calls already read — no extra dense-matrix
materialization beyond the already-small `SnuWval` (`W x Ddest`, the same size class as `wctx.y`
itself). Amended per-row correction:
```julia
@inbounds for j in 1:nbilateral
    d = wctx.target_slot[j]
    pij = pi_vec[j]
    for x in 1:nx
        HEZ[j+1, x] = invM * (HEZ[j+1, x] - pij * TZ[d, x])   # TZ[d,:] replaces NuZ
    end
end
```
Same `use_profiled_correction` dispatch discipline as H_EC — this is genuinely simpler to implement
than H_EC (one gemm vs. a new binned/cumulative table) because `Z` has no threshold-bin structure to
thread through.

## 4. H_EF: `winner_pair_cross_hessian_colsum!` / `winner_pair_cross_hessian_esum!`

Old correction uses `sumNu = Σ_{o=1}^D NuCScum[o,l]` (`colsum!`) and `t0 = Σ_w S[w]*nu[w]` (`esum!`)
— both single global scalars, reused for every `colsum[j+1]`/`Esum[j+1]` regardless of `j`'s own
destination. Once §2's `MCScum`/target_slot machinery exists, the amendment is mechanical and
strictly smaller than H_EC's (no new table, just swap which existing quantity is read):
```julia
# colsum!, amended:
@inbounds for j in 1:wctx.ncolI
    d = wctx.target_slot[j]
    sumT = 0.0
    for x_dummy in 1  # MCScum has no "x" dependence needed here -- see note below
    end
    colsum[j + 1] = sumQ - pi_vec[j] * MCScum_summed_over_x_or_direct[d, l]   # see open question below
end
```
**Open question, to resolve during implementation, not assumed here**: H_EF's level restriction may
or may not route through the SAME `Bidx`/CM-grid `x` dimension H_EC uses (its own file header says
the level feature is a threshold/bin-indexed step function of a *level* target, which reads as
plausibly the same grid machinery, but this needs confirming against `CMFrechetExtension`'s actual
`Wtab`/`T1` construction in `cm_frechet_hessian.jl` before writing the real patch — flagged here
rather than guessed, per this task's own "do not write new algorithms from a blank page" instruction
extending equally to "do not guess an index correspondence and hope it's right"). If the level
feature has its own bin index distinct from `x`, `MCScum` needs a level-native twin (`MLevelCScum`,
built the identical way, in the identical existing Fréchet-only accumulation pass) rather than reuse
of `MCScum` directly.

## 5. What does NOT change

- The "keep" (winner) term in every block (`q_diff`, `QCScum`-based, and H_EZ's winner-scatter
  `HEZ[j+1,x] += v[w]*Zx[w]` accumulation) is untouched — confirmed by re-reading, these already
  implement exactly `W^R_{od,k}` per the mission's own notation, no edit needed.
- Row 1 (the ones/ζ column) and its `SOnlyCScum`-based correction is untouched — it has no `pi_vec`
  entry (mission's formulas are about the `λ_od`-weighted rows only).
- `economic_forward!`/`economic_transpose!`/`fill_core_hessian_upper!` (H_EE core) — confirmed in
  `EXISTING_OPTIMIZED_ECONOMIC_CROSS_BLOCK_MAP_2026-08-01.md` to already be cf-agnostic and require
  no changes; the profiled cf's reduced `ncolI`/`winner`/`y` flow through them unmodified (this is
  exactly what `reduced_homogeneous_hessian_2026-08-01.jl` already relies on structurally — its own
  `u`/`r`/`QQ`/`R2` accumulators use the identical "skip when winner_reduced_col==0" pattern as the
  argument above, just at the H_EE diagonal rather than a cross-block).
- `pack_upper_cm_hessian!`, threading, scratch ownership, sparse/BLAS conventions — none of this
  changes; every amendment above is a formula swap inside an existing accumulation, not a
  restructuring.

## 6. Outer A/gp gradient sharing (mission §13)

`profiled_lfix_incremental_2026-08-01.jl`'s `ProfiledLFixCache`/`build_profiled_lfix_cache`/
`profiled_composite_gradient_at_incremental` are already generic over the economic block (they do not
hardcode "unrestricted") — what they lack, per the audit's §9 finding, is (a) a persistent-workspace
variant analogous to `LFixBaseWorkspace` (currently allocates fresh every call — an honest,
documented gap, not a correctness issue) and (b) a hook for a family's *own* fixed-restriction
contribution to fold into `q0`, mirroring how `build_lfix_base_cache_cm`/`_originzc`/`_cm_frechet`
already do this for the OLD cache (`EXISTING_OPTIMIZED_ECONOMIC_CROSS_BLOCK_MAP_2026-08-01.md` §8).
Porting this is smaller in scope than H_EC/H_EF/H_EZ: add a `restriction_contrib0` field/closure to
`ProfiledLFixCache`, computed once per family by a thin wrapper (`build_profiled_lfix_cache_cm`
etc.), analogous to the existing OLD-cache wrappers — no new winner-selection or bandwidth-selection
logic required, since `update_winner_o1`/`profiled_select_bandwidth` are already family-agnostic.
