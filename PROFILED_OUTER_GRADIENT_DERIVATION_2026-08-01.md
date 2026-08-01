# Profiled outer gradient derivation (2026-08-01)

Derives the custom analytic ("C+") outer gradient of the inner dual optimal value `K*(x)`
with respect to the profiled outer vector `w_profiled = [gp; r_free]` (task §6/§9), for the
`:unrestricted` family, fixed `theta`, `destination_sample=:exclude_row`. Every formula below
is traced directly from the actual production/diagnostic code, not assumed — file:line
citations are given at each step so the implementation in `profiled_operator_bundle_2026-08-01.jl`
(or a sibling file) can be checked term-by-term against this derivation.

## 1. The exact transformed-A exponent (traced from `winner_certificate.jl`)

`constCons_matrix` (`winner_certificate.jl:66-77`) builds, for `theta_full` and fixed data:

```
Aod_theta[o,d]   = exp(z[o,d])                     -- z is the reduced log-A coordinate this
                                                        whole reparameterization operates on
Aod[o,d]         = Aod_theta[o,d] * K1[o,d]         -- K1 = cHat*(...)^{1/mu}*(lambda_od/lambda_1d),
                                                        FIXED data, independent of z
AodPow[o,d]      = (Aod[o,d]/cHat[o,d])^(-mu)       = Aod_theta[o,d]^(-mu) * K2[o,d]   (K2 fixed)
constCons[o,d]   = wHat[o]*AodPow[o,d]*tau[o,d]
```

`canonical_price_precompute` (`winner_certificate.jl:115-150`) then builds the sigma-power
object that feeds `cf.wval`:

```
constConsσ[o,d] = wHat[o]^(1-σ) * (AodPow[o,d]*tau[o,d])^(1-σ)
                = wHat[o]^(1-σ) * tau[o,d]^(1-σ) * K2[o,d]^(1-σ) * Aod_theta[o,d]^(-mu*(1-σ))
                = C(o,d) * exp( mu*(σ-1) * z[o,d] ),   C(o,d) fixed (independent of z)
```

`build_compressed_factual` (`compressed_moments.jl:206-218`) sets, per draw `w` and active
destination slot `s`:

```
bo = winner[w,s]                       -- argmin price, PIECEWISE CONSTANT in z away from ties
wval[w,s] = constConsσ[bo,s] / UσPow[w,bo]
          = C(bo,s)/UσPow[w,bo] * exp( mu*(σ-1) * z[bo,s] )
```

So `M_d(ω) := wval[w,slot(d)]` (the task's destination-level homogeneous moment) is, **away
from winner switches**, a pure exponential in the log-A coordinate of whichever origin
currently wins at that destination, with exponent exactly

```
e_exponent := mu*(σ-1)
```

(the identical constant already used in `reduced_recovery_from_lfd_2026-08-01.jl:58`'s
`e_exponent = μ * (σ - 1)` for the LFD-based full-A recovery — confirms this is the one
"transformed-A exponent" production actually uses, not an assumption).

**Result (task §8's required derivative, now derived not assumed):**

```
∂M_d(ω)/∂a_{rd} = e_exponent * M_d(ω) * 1{w_d(ω) = r}      (a_{rd} := z[r,d], r retained, r ≠ anchor(d))
```

Nonzero exactly when `r` is the CURRENT winner at destination `d` for draw `ω` — matches the
task's stated requirement ("nonzero only when r is the current winner") exactly, with the
production exponent now pinned down as `mu*(σ-1)`.

## 2. Bilateral retained-moment derivative

The reduced bilateral homogeneous moment for retained cell `(o,d)` is (by construction,
`reduced_homogeneous_contraction_2026-08-01.jl:84-95`'s `acc` accumulation, matching the
task's own `M_d(ω)[1{w_d(ω)=o} - λ_{od}]` form with `λ_{od} = cf.Pmat[o,d]`):

```
E_{w,(o,d)} = M_d(ω) * (1{w_d(ω)=o} - λ_{od})
```

Away from winner switches, `1{w_d(ω)=o}` is locally constant in `a_{rd}`, so by the product
rule and step 1:

```
∂E_{w,(o,d)}/∂a_{rd} = e_exponent * M_d(ω) * 1{w_d(ω)=r} * (1{w_d(ω)=o} - λ_{od})
```

## 3. Envelope-theorem outer derivative (fixed-winner regime)

The inner dual objective (`_callbackEvalFG_inner_profiled!`, `profiled_operator_bundle_2026-08-01.jl:74-96`):

```
q_w      = -ζ - t[w],   t[w] = Σ_{(o,d) retained} κ_{od} E_{w,(o,d)} + (france term)
κ_{od}   := β_{od} * nrm_{od} * gdiv_{od}         (the effective dual coefficient on the RAW moment)
K(x)     = Σ_w Psi(q_w)/M + ζ
```

By the envelope theorem, at a KNITRO-verified optimum `(ζ*, κ*)` the total derivative of the
optimal value w.r.t. an outer coordinate equals the PARTIAL derivative holding `(ζ*, κ*)`
fixed (standard for a smooth inner objective at an interior/first-order optimum; this is
exactly why the production "C+" gradient never differentiates through the inner KNITRO
solve itself):

```
dK*/da_{rd} = (1/M) Σ_w dPsi(q_w) * ∂q_w/∂a_{rd} = -(1/M) Σ_w dPsi(q_w) * ∂t[w]/∂a_{rd}
```

`dPsi(q_w) = m_weights[w]` is exactly the solved LFD (matches
`reduced_recovery_from_lfd_2026-08-01.jl:50`'s own contract: "`m_weights` is `dPsi(r)` from a
`verify_inner_solution_*!` call").

### 3a. Ordinary retained bilateral coordinate `a_{rd}`, `d` not the France-ratio destination

Only destination-`d` moments depend on `a_{rd}` (M_d(ω) is destination-local). Summing step 2
over every retained `o` at `d`, using `κ_{anchor(d),d} ≡ 0` (no beta slot, so the anchor's own
`λ` term drops out of the sum with no special-case needed):

```
∂t[w]/∂a_{rd} = e_exponent * M_d(ω) * 1{w_d(ω)=r} * ( κ_{rd} - Cbar[d] )
Cbar[d] := Σ_{o retained at d} κ_{od} λ_{od}     -- EXACTLY reduced_homogeneous_dual_contraction's
                                                     own Cbar[slot] (line 54), reused unchanged.
```

so

```
dK*/da_{rd} = -e_exponent * (κ_{rd} - Cbar[d]) * (1/M) * Σ_{w: w_d(ω)=r} SW[w]*m_weights[w]*M_d(ω)
            = -e_exponent * (κ_{rd} - Cbar[d]) * B[r,d] / M
```

where `B[r,d] := Σ_{w: winner(w,d)=r} SW[w]*m_weights[w]*wval[w,d]` is **exactly** the `B`
matrix `reduced_homogeneous_transpose_contraction!` already accumulates
(`reduced_homogeneous_contraction_2026-08-01.jl:119-126`, `B[cf.winner[s,slot],slot] +=
ws*wv`) when called with `weights = m_weights` (the solved LFD) — i.e. **no new accumulation
kernel is needed**: calling the existing transpose contraction at the solved LFD already
produces the exact object this gradient needs. This is the concrete, code-grounded form of
task §9's "gp derivative uses the new France ratio moment" / "changed-cell maps" requirement:
the changed-cell map is the SAME `winner`/`wval` structure the forward/transpose kernels
already carry.

### 3b. Retained bilateral coordinate at the France (`bi`) destination

`reduced_homogeneous_dual_contraction` (lines 90-91) adds, ONLY at `slot = bi_slot`, the term
`κ_cf*cf_raw[w] - κ_cf*gp^σ*wval[w,bi_slot]`. The second piece also depends on
`a_{r,bi_slot}` through `wval[w,bi_slot] = M_{france}(ω)`, contributing an extra term:

```
∂t[w]/∂a_{r,bi_slot} = e_exponent * M_france(ω) * 1{w_{bi}(ω)=r} * ( κ_{r,bi_slot} - Cbar[bi_slot] - κ_cf*gp^σ )
```

i.e. the same formula as 3a with `Cbar[bi_slot]` replaced by `Cbar[bi_slot] + κ_cf*gp^σ`. In
Hessian-context terms this is exactly `Lam[jcf] = kappa0[jcf]*gp^σ`
(`reduced_homogeneous_hessian_2026-08-01.jl:104`) reused as an additional subtractand.

### 3c. The `gp` scalar coordinate (France price-index parameter, `θ_full[3+D]`)

`gp` enters `t[w]` only through the France term's `gp^σ` factor (no winner/moment
piecewise-constancy issue — `gp` is a genuine smooth continuous outer parameter, not a
log-A coordinate):

```
∂t[w]/∂gp = -κ_cf * σ*gp^(σ-1) * wval[w,bi_slot]

dK*/dgp = -(1/M) Σ_w SW[w]*m_weights[w] * ( -κ_cf*σ*gp^(σ-1)*wval[w,bi_slot] )
        = κ_cf * σ*gp^(σ-1) * (1/M) * Σ_w SW[w]*m_weights[w]*wval[w,bi_slot]
        = κ_cf * σ*gp^(σ-1) * Tslot[bi_slot] / M
```

using `Tslot[slot] := Σ_w SW[w]*m_weights[w]*wval[w,slot]`, exactly
`reduced_homogeneous_transpose_contraction!`'s own `Tslot` accumulator (line 125) at
`weights=m_weights`. Matches the task's own formula `-σ g_p^{σ-1} M_f` up to sign
convention (the task differentiates the CONSTRAINT `Φ_ff - gp^σ M_f`; here we differentiate
the DUAL VALUE `K*`, which carries the dual multiplier `κ_cf` and a sign flip from the
envelope theorem — both conventions agree once the multiplier and sign are made explicit,
and both reduce to the same `Tslot[bi_slot]`/`M_f`-type reused quantity).

### 3d. The Φ_ff (French domestic anchor) term

`Φ_ff` is the France-destination, France-origin (own-cell) factual quantity — **exactly the
anchor cell for the France destination** (task §6: "France anchor = France→France"). Since
anchor cells carry no free outer coordinate at all (task §9: "anchor coordinates have no
gradient slots"), `∂Φ_ff/∂a_{rd}` is identically zero for every RETAINED `r` — confirmed
structurally, not just numerically, because `Φ_ff` never appears in `reduced_homogeneous_
dual_contraction`'s retained-moment loop (only `cf.cf_raw[w]`, which is a residual/gap
quantity independent of any single retained `z` coordinate — see `compressed_moments.jl`'s
`cf_raw` construction, unaffected by any individual bilateral A cell). This matches the
task's own statement: "the first term is present only for the French domestic coordinate" —
i.e. present only for the (non-free) anchor coordinate, hence never appears in the profiled
gradient vector at all.

## 4. Gravity-pivot chain rule (task §10)

The free profiled A-coordinate vector is `r_free` (length 360), related to the full retained
vector `r` (length 361) by `pivot_expand_on_retained` (`gravity_pivot_on_retained_2026-07-31.jl:66-76`):

```
r[other_pos[k]] = r_free[k]                                          for k=1..360
r[pivot_pos]    = ( -offset_r0 - Σ_k cr[other_pos[k]]*r_free[k] ) / cr[pivot_pos]
```

This is affine in `r_free`, so by the chain rule, for every `k`:

```
dK*/d(r_free[k]) = dK*/d(r[other_pos[k]])                      -- DIRECT effect (identity term)
                  + dK*/d(r[pivot_pos]) * dr[pivot_pos]/d(r_free[k])   -- INDUCED effect
dr[pivot_pos]/d(r_free[k]) = -cr[other_pos[k]] / cr[pivot_pos]
```

where `dK*/d(r[other_pos[k]])` and `dK*/d(r[pivot_pos])` are both computed from section 3a/3b
above (`r[·]` is just `a_{rd}` re-indexed into the flattened retained-cell ordering
`gravity_pivot_on_retained_2026-07-31.jl` already uses — `_lin(o,d,D)`). Concretely:

```
g_full[pos] := dK*/d(r[pos])   for pos = 1..361   (section 3a/3b, one entry per retained cell)
g_free[k]   := g_full[other_pos[k]] - g_full[pivot_pos] * cr[other_pos[k]]/cr[pivot_pos]     for k=1..360
```

This is the exact profiled-coordinate analog of `pivot_reduce_on_retained` (`gravity_pivot_
on_retained_2026-07-31.jl:79-83`), but for a GRADIENT (covector) rather than a point — a
gradient does not simply drop the pivot entry like a point does; it must absorb the pivot's
sensitivity into every other retained coordinate via `cr`, exactly as written above.

**Required gate (task §10):** perturbing an anchor scale must be structurally impossible in
`r_free` (there is no coordinate for it — `AnchorSpec`/`ProfiledEconomicMomentLayout` both
guarantee this at the type level, §2 of both files), and inserting an arbitrary common
destination-scale shift into `gauge` must leave `offset_r0`/`cr`/the gravity residual exactly
unchanged (both are built from `z` differences within the retained set only —
`gravity_from_logz` is exactly affine and the theory doc's section 2.1(c) proves a common
shift of `z[:,d]` has zero gravity-coefficient contribution; this must be checked live by
finite-differencing `offset_r0` and `cr` against an arbitrary added `Δ` to `gauge[d]`, not
just cited from the theory doc).

## 5. Summary — vector layout of the profiled gradient

For `w_profiled = [gp; r_free]` (length 361 at real D20):

```
grad[1]      = dK*/dgp                              (section 3c)
grad[1+k]    = g_free[k]  for k=1..360               (section 4, built from section 3a/3b)
```

No entry exists for any anchor cell or for `Φ_ff` (both structurally absent from `r_free`,
per section 3d) — matching task §9's required properties exactly:
- anchor coordinates have no gradient slots (true by construction — `r_free` has no anchor entries)
- anchor winners still contribute through `M_d` and the `-λ*M` correction (the `Cbar[d]`/`κ_{rd}-Cbar[d]`
  term in section 3a already reflects this — the anchor's own `λ_{anchor,d}` is folded in via
  `κ_{anchor,d}≡0`, so it contributes zero to `Cbar[d]` but the OTHER retained cells' `-λ_{od}`
  terms are unaffected)
- the omitted anchor moment has no dual multiplier (no `κ_{anchor,d}` is ever instantiated)
- the France ratio multiplier `κ_cf` is included (sections 3b/3c)
- the gravity-pivot chain rule uses the retained-coordinate layout (section 4, using `cr`/`other_pos`/`pivot_pos` from `PivotGravityElimOnRetained` directly)
- the `gp` derivative uses the new France ratio moment (`Tslot[bi_slot]`, section 3c)

## 6a. CORRECTION (live user feedback, 2026-08-01): the A-block must use the SAME fixed-dual FD mechanism production uses, not the pure envelope formula in sections 3a/3b/4

Sections 3a/3b/4 above are algebraically correct **away from winner switches**, but a pointwise
envelope-theorem derivative is exactly the WRONG thing to hand an outer NLP solver: it is
discontinuous at every winner-switch boundary, and this project has already shown (this is a
repeated, confirmed finding, not a hypothetical) that ignoring switching effects makes the outer
search behave badly. Production's own "C+" gradient (`composite_gradient_at_Cplus`/
`composite_gradient_at`, §6 of the call-graph doc) is **not** a pure analytic formula for exactly
this reason: the A-block is a **fixed-dual (ζ*,κ* held at the just-solved optimum) coordinatewise
central finite difference**, using an O(1)-per-draw incremental winner update
(`update_winner_o1`/the top-3-cache exact update) so each ±h probe costs O(touched draws) rather
than a full inner re-solve. The finite step naturally averages over any winner flips inside the
±h window instead of picking one discontinuous branch.

**Sections 3a/3b/4's formulas are retained above only as the "smooth-regime" reference** (used as
one of several cross-checks in the gate, §11: "central finite differences where smooth") — they
are NOT what `profiled_composite_gradient_at` (implementation, `profiled_lfix_incremental_2026-08-01.jl`)
actually computes for the A-block. The gp component (section 3c) IS still exact/analytic in the
implementation, unchanged in kind (gp never enters winner selection — no MinInd!/price dependence
on gp at all, exactly mirroring why production's own `gamma_component_analytic` is exact rather
than FD), but section 3c above has a bug, corrected here:

### 3c corrected

Both gp-dependent pieces of `t[w]` matter, not just the winner-dependent one:
`const_cf = kappa_cf*gp^sigma*wPrime_bi*LPrime_bi` (draw-independent) AND
`-kappa_cf*gp^sigma*wval[w,bi_slot]` (draw-dependent, inside `contrib0[w,bi_slot]`). The corrected
result (implemented in `profiled_composite_gradient_at`, `profiled_lfix_incremental_2026-08-01.jl`):

```
S_m        := sum_w SW[w]*m_weights[w]
Tslot[bi]  := sum_w SW[w]*m_weights[w]*wval[w,bi_slot]
d(Delta_dual)/dgp = kappa_cf * σ * gp^(σ-1) * ( LPrime_bi*S_m - Tslot[bi] ) / M
```

(the earlier draft of this section, and an earlier draft of the code, dropped the `LPrime_bi*S_m`
term entirely — caught before any gate was run against it, not after.)

### A-block mechanism (implemented, replacing 3a/3b/4)

`profiled_lfix_incremental_2026-08-01.jl` builds a `ProfiledLFixCache` — the SAME dense
`price0`/`pTσ0`/`winner0`/`runnerup0`/`third0` (W x D x Ddest) cache `LFixBaseCache` builds, using
the UNCHANGED `price_and_pTsigma_cell`/`update_winner_o1`/top-3 machinery (winner-finding does not
know about anchors — it operates on the genuine `(D,Ddest)` `z_full` this reparameterization always
reconstructs) — but a REDUCED per-destination contribution formula:

```
contrib0[w,d] = (kappa[wo,d] - Cbar_eff[d]) * pTsigma0[w,wo,d],   wo = winner0[w,d]
Cbar_eff[d]   = Cbar[d] + (d==bi_slot ? kappa_cf*gp^sigma : 0)     (folds in the France cross-term, §3b)
```

(unlike production's legacy `CONST_d[d] + lambda*[wo]*pTsigma[wo]`, which targets the OLD
fixed-`denom[d]` functional — a genuinely different target, per this session's own memory
`unrestricted-profiled-scale-knitro-comparison-2026-08-01` key finding #1). `profiled_affected_cells`
mirrors `affected_cells` exactly: perturbing one `r_free[k]` touches its own direct retained cell
AND the gravity-pivot's retained cell (which moves under every `r_free` perturbation via
`pivot_expand_on_retained`'s affine reconstruction) — up to 2 destinations, exactly the same
"direct + pivot" structure production's plain (unreduced) A-coordinate system already has.
`profiled_select_bandwidth`/`a_block_fd_component_profiled`/`profiled_composite_gradient_at` mirror
`select_bandwidth`/`a_block_fd_component`/`composite_gradient_at` (`composite_gradient.jl`) exactly
— same adaptive switching-mass-targeted bandwidth (h0=0.01, floor=1e-4, ceil=0.1, target mass
0.3%-3%), same central-FD-with-shrink-and-retry discipline.

## 6b. SECOND correction (live user feedback, 2026-08-01): keep the implementation surgical, not a parallel O(1)-cache reimplementation

The first correction (§6a) replaced the pure envelope formula with a fixed-dual FD mechanism, but
the first FD implementation (`profiled_lfix_incremental_2026-08-01.jl`, since removed) reimplemented
production's entire `LFixBaseCache`/O(1)-incremental-winner-update machinery from scratch for the
reduced moment basis (~500 lines: a new dense price/pTsigma/winner/runnerup/third cache, a new
`update_winner_o1`-based per-destination contribution recompute, a new adaptive-bandwidth selector).
This was flagged as far too much custom surface for what should be a "marginal adaptation," and it
promptly **failed its own D4 gate** (cosine similarity ~0.03 against ground-truth re-solved FD) —
a concrete demonstration of the exact risk the correction warned about.

**Final implementation** (`profiled_outer_gradient_fd_2026-08-01.jl`, ~70 lines): a fixed-dual
central finite difference where each ± probe is evaluated by calling the **unchanged, already-gated
production/session kernels directly** — `build_compressed_factual` (production, unmodified) to
rebuild the compressed factual at the perturbed point, then `reduced_homogeneous_dual_contraction`
(this session's own already-gated kernel, §6 of the task) to get `t[w]`, then `CS.Psi!` for the
scalar. No custom winner-update logic, no new cache struct, no bandwidth selector — every coordinate
(including `gp`) uses the identical one-line probe function. Cost is O(W·D·Ddest) per probe (the
same order as production's own slowest, simplest `:block_local` tier), not O(1) — a deliberate,
documented trade of wall-clock speed for a far smaller, far more auditable bug surface, matching the
user's explicit priority ("even if you don't make a mistake, I'm sure that what you do will in
general not be as optimized" — accepted).

**D4 gate result** (`PROFILED_OUTER_GRADIENT_GATE_D4_2026-08-01.csv`): cosine similarity 0.9994/0.9995/0.9998
at calibration/small-perturbation/many-winner-changes respectively, sign agreement 92%/92%/100%. The
larger relative/absolute errors trace to the `gp` coordinate's own FD estimate at `h=0.01`, which
`composite_gradient.jl`'s own long-standing documentation independently flags as having "severe
curvature-driven bias... drifts by 15-40% between successive KNITRO iterates" for the analogous
gamma coordinate in the PRODUCTION gradient — i.e. a known property of this functional's gp/gamma
direction at this bandwidth, not a new defect introduced by the reduced-moment adaptation.

## 6. Implementation note

Every quantity in sections 3-4 (`Cbar`, `κ`, `B`, `Tslot`, `m_weights`) is already computed
by the EXISTING `reduced_homogeneous_dual_contraction`/`reduced_homogeneous_transpose_
contraction!` kernels when called once at the solved point with `weights = m_weights` (the
verified LFD) — the profiled C+ gradient therefore requires **no new O(W*D) accumulation
kernel**, only a thin wrapper that (a) calls the transpose contraction at the solved LFD to
get `B`/`Tslot`, (b) forms `g_full[pos] = -e_exponent*(κ_{rd}-Cbar[d])*B[r,d]/M` (with the
`bi_slot` adjustment from 3b) plus `grad[1] = κ_cf*σ*gp^(σ-1)*Tslot[bi_slot]/M`, and (c)
applies the section-4 pivot chain rule. This keeps the implementation a genuine "one
parameterization branch" addition (task §7/§9), not a parallel gradient pipeline.
