# Fixed-Fréchet Full Spec: Math and Targets — 2026-07-24 (port-prep)

Branch `feature/fixed-frechet-post-omit-row-port-prep-2026-07-24`, branched off
`production/fullA-exact @ c55e81e`. This document states the exact mathematical
specification being ported, and is intentionally scoped to be checkable
line-by-line against both the paper draft's §5.4 and this branch's code.

## 1. What is being imposed

For every origin `o ∈ {1,...,D}` (D = 20, **origin count**, invariant across
`destination_sample`) and every grid point `H_ℓ`, `ℓ = 1..L` (L = 50):

**CDF family (eq. 37)**

  E_F[ 1{z_o < H_ℓ} ] = F*(H_ℓ)

**Truncated-power family (eq. 38)**

  E_F[ z_o^(1-σ) · 1{z_o < H_ℓ} ] = E_F*[ z^(1-σ) · 1{z < H_ℓ} ]

Both families together contribute `2·D·L = 2000` restrictions at D=20, L=50 —
this is the complete paper specification. `:cdf_only` (CDF alone, `D·L=1000`
restrictions) is retained as an explicit legacy/diagnostic option; the
scientifically-named `marginal_mode=:frechet_reference` mode defaults to
`frechet_feature_set=:cdf_power`.

This is a restriction on **all D=20 origin productivity coordinates**
(`ctx.U`, a `W×D` matrix — one column per origin, including ROW). It is
**not** destination-indexed and is **not** resized under
`destination_sample=:exclude_row`: ROW is dropped as a destination (core
trade-share block shrinks from `20×20` to `20×19`) but is retained as an
origin, and its marginal restriction is imposed identically to every other
origin's. See §3 below for the dimension-safety argument in the actual code.

## 2. Why the U-space representation makes the targets closed-form

The codebase's productivity draws `ctx.U` are i.i.d. `Exp(1)` by construction
(`genExpRands!`), and every CM/Fréchet moment architecture operates directly
on `U`, not on a transformed `z`-space object. That means the benchmark
distribution `F*` — literally "the distribution these draws already have" —
is the standard exponential CDF, so:

- **CDF-family threshold/target**: `u_ℓ* = F*^{-1}(p_ℓ) = -log(1-p_ℓ)` (Exp(1)
  quantile), `t_ℓ* = p_ℓ` exactly. No estimation, no numerical quadrature —
  `p_ℓ` is read straight back out as the target.
- **Power-family target**: `E_Exp(1)[ U^pw · 1{U ≤ u} ] = ∫_0^u t^pw e^{-t} dt
  = γ(pw+1, u)` (lower incomplete gamma, unnormalized), with
  `pw = (1-σ)/θ*`. Requires `pw+1 > 0` (else the integral diverges at the
  origin); production `σ>1, θ*>0` gives `pw<0` in general but the incomplete
  gamma remains well-defined provided `pw>-1`, which is asserted rather than
  silently producing `NaN`/`Inf`.

Grid: `L` evenly-spaced probabilities excluding the endpoints,
`p_ℓ = ℓ/(L+1)` for `ℓ=1..L` (deliberate — including `p=0` or `p=1` produces a
degenerate all-zero or all-one contrast column).

## 3. Live calibration — θ* is NEVER hard-coded

  θ* = 1 / μ̂      (μ̂ = `ctx.μHat`, the live gravity-elimination estimate
                    associated with the ACTIVE destination_sample)
  σ  = ctx.σ

Both are read from the canonical context at construction time, not from any
historical or draft-reported constant. **The draft's own reported θ*=6.8 is
not reproduced or targeted anywhere in this port.** A pre-omit-ROW
reconciliation pass (`experiment/fullA-fixed-frechet-basis-draft-reconciliation-2026-07-24`)
already measured live θ*≈8.747 at the pre-omit-ROW calibration — a real,
disclosed, unresolved vintage gap versus the draft's 6.8 (see that branch's
`docs/FIXED_FRECHET_DRAFT_VS_CURRENT_SPEC_2026-07-24.md`, Finding F1). Under
`destination_sample=:exclude_row`, gravity is re-estimated on the rectangular
19-destination sample, so θ*/μ̂ generally differ again from both the draft and
the pre-omit-ROW reconciliation value. This port reports the live value at
startup and makes no claim that it reproduces any historical figure — it
implements the same restriction *family* at whatever θ*/μ̂ the active
production calibration currently produces.

At startup, this port's code must report: active `destination_sample`; live
θ*, μ̂, σ; `D` (origin count) and `D_dest` (destination count); `L`; feature
set (`:cdf_only`/`:cdf_power`); basis (`:cumulative`/`:interval`); and the
SHA-256 fingerprint of the analytic CDF target vector (provenance/checkpoint
use — see the timeout/state-reuse and canonical-winner-integration docs for
the full fingerprint field list).

## 4. Dimension discipline — the D vs. D_dest trap

Confirmed by direct inspection of `context_real_d20.jl` (`d20_real_setup`,
`production/fullA-exact @ c55e81e`, lines ~93-187):

- `ctx.D` (`Dact` in that function) = **origin count**, always 20, invariant
  across `destination_sample`.
- `ctx.D_dest` (`Ddest`) = **destination count**, `20` under `:all_legacy`,
  `19` under `:exclude_row` (`Ddest = row_idx === nothing ? Dact : Dact - 1`).
- `ctx.U` is `W × ctx.D` (one column per **origin**) — confirmed by
  `d20_real_setup`'s own screen-construction comment ("D = origin count,
  D_dest = destination count... precompute_pairwise_M needs only D (it's an
  origin × origin object)") and by `ctx.U` flowing unchanged from
  `build_ad_context_real_d20`'s `pp.U`, never resliced or reshaped by
  destination count anywhere in that function.
- The core `A_od` parameter block and core trade-share moments use the
  **`D_dest`-strided** convention (`Aod_free_pos = [1 + (d-1)*Dact + o for o
  in 1:Dact, d in 1:Ddest]`); the fixed-Fréchet feature block must **never**
  touch this convention or substitute `D_dest` for `D` anywhere in its own
  construction — see the project memory
  `moments-vs-aod-linear-index-convention` for the general form of this trap,
  confirmed as a real, previously-hit bug elsewhere in this codebase.

**Consequence for the port**: the pre-omit-ROW reconciliation archive's
`cm_frechet_bases.jl` already constructs every Fréchet moment block by
iterating `origins = [o for o in 1:D if o != refIndex1]` against `ctx.D` and
`ctx.U` — **never `ctx.D_dest`**. This is dimension-correct as written and
requires no `D`→`D_dest` fix. What it does *not* yet handle is composing
correctly with a **rectangular** (`20×19`, not `20×20`) core block through
the current production `moments!`/Jacobian-count wiring
(`ncore`/`outer_constr_index`/`d`) — that composition is what §4 of the port
readiness plan (and the code in this branch) must get right, and is verified
explicitly with a D=4 rectangular-layout gate (§11.11 of the task brief) in
addition to the D=4 square gate.

Explicit runtime assertions ported into this branch (see
`full_aod_diag/d4_exact/frechet_reference_targets.jl` and
`cm_frechet_bases.jl` in this branch): `size(ctx.U, 2) == ctx.D`,
`length(origins) == ctx.D - 1`, `ncm == D*L` (`:cdf_only`) or `2*D*L`
(`:cdf_power`) — independent of `ctx.D_dest`.

## 5. Restriction-block architecture

  [ g_ω^core ; h_ω^FF ]

`g^core`: ordinary trade/equilibrium block, shared unmodified with
unrestricted and other restricted models via the production core-moment
interface (see `FIXED_FRECHET_CANONICAL_WINNER_INTEGRATION_MANIFEST_2026-07-24.md`
for the exact current entry point and how this port consumes it).

`h^FF`: CDF + (if active) truncated-power block, built from `ctx.U` and the
analytic `FrechetReferenceTargets` bundle. Depends only on the benchmark
draws and closed-form targets — **not** on the outer `A_od` coordinates:

  ∂h_ω^FF / ∂A ≡ 0

so the production C+ (analytic outer-gradient) architecture for the `A`
coordinates is unmodified; the Fréchet block only affects the inner dual
multipliers/Hessian, never requiring a direct A-derivative kernel of its own.

## 6. Basis families ported

- `:cumulative` (Q0): existing-style cumulative CDF/truncated-moment
  coordinates — direct extension of the archived `precalc_frechet_reference_cdf`
  pattern with the eq. (38) power block added.
- `:interval` (Q1): disjoint-bin first-difference coordinates, built directly
  from bin membership (not by differencing Q0 columns) — an independently
  cross-checked code path.
- (Q2/whitened is ported as an available diagnostic transform on top of Q1,
  since the pre-omit-ROW study found it improves Gram conditioning
  substantially at large L but did not materially change outer-search
  acceptance rate at L=50 — see §5 of the port-readiness report for the
  basis-default decision and why `:cumulative` remains the temporary
  production default in this port.)

`:cumulative` and `:interval` are exact invertible reparameterizations of the
identical finite-grid restriction (proved via the triangular transform `S` in
the pre-omit-ROW reconciliation's
`FIXED_FRECHET_CUMULATIVE_INTERVAL_EQUIVALENCE_2026-07-24.md`, re-verified at
D=4 in this branch's own gates) — for any fixed grid they yield the same
feasible set, Δ*, primal weights, and counterfactual bound.

## 7. What is genuinely new in this port (not present in the reconciliation archive)

1. Dimension-safety assertions and a rectangular-layout D=4 gate (`D_dest ≠
   D`) — the archive never ran under `destination_sample=:exclude_row`.
2. A fast structured (Architecture-C-style) Hessian for the **truncated-power
   block and its cross-terms with the CDF block and the core block** — the
   archive explicitly left this unbuilt ("no fast structured Hessian exists
   yet for the truncated-power feature block" — its own Part VIII, item 3).
   See `FIXED_FRECHET_STRUCTURED_POWER_HESSIAN_2026-07-24.md`.
3. Consumption of the current production core-moment/winner-state interface
   (the archive predates the omit-ROW/unrestricted-core/threshold-10 merges
   entirely and used its own ad hoc Arm C driver, not any current production
   driver).
4. Production-shaped config (`marginal_mode=:frechet_reference`,
   `frechet_feature_set`, `frechet_basis`, `cm_grid_size`) and checkpoint
   fingerprint fields consistent with current production's established
   checkpoint schema conventions.
