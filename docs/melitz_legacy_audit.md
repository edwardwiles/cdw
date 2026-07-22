# Legacy Melitz code audit

Source files reviewed (read in full, treated as reference material only — not imported):

- `Nested_hFunction.jl` (40,158 bytes / 1,107 lines) — pulled via rclone from
  `dropbox:Gravity robustness/Analysis/Melitz_XVersion_20260721/Nested_hFunction.jl`
- `Nested_moments!.jl` (66,404 bytes / 1,709 lines) — pulled via rclone from the same
  Dropbox folder

Both files implement a **hybrid nested Bertrand/Melitz/EK/perfect-competition model**
(`Nested_equilibrium_quantities!` branches on `ρ`, `σ`, and a firm-count ceiling to pick
one of four equilibrium regimes), always restricted to a single `baseIndex` focal country
plus, in most Melitz code paths, a single aggregated "rest of world" destination (index
`1`). Nothing here is a finished, general full-`D` Melitz implementation, and the file
contains four progressively-abandoned attempts at the parameter-transform step
(`Melitz_transform_θ`, `_θ0`, `_θ1`, `_θ2`, `_θ_1`), each full of dead code, `@show`
debugging left in, and hand-derived fixed points that are not independently verified.

This audit exists to make explicit what is reused, what is rejected, and what is expected
to come from `production/fullA-exact` instead, per the project brief.

## Safe to reuse or adapt

- **Constant-markup Melitz price/revenue/profit algebra.** In
  `Nested_equilibrium_quantities_Melitz!` (`Nested_hFunction.jl:729-739`):
  `price = marginal_cost * σ/(σ-1)`, `sales = price^(1-σ) * constGammaPow_d * gdp_d`,
  `profit = sales/σ - fixedcost`, entry mask `profit > 0`. This is exactly the paper's
  constant-markup formula and matches the brief's firm-quantity spec once rewritten in
  terms of `A[o,d]` (economic productivity) rather than the file's implicit unit-cost
  object. Safe to reimplement cleanly in `src/melitz/firm_quantities.jl`.
- **Zero-profit cutoff / marginal-cost-at-cutoff algebra.**
  `Nested_zero_profits_marginal_cost_from_ϵ` (`Nested_hFunction.jl:906-917`, the `ρ==σ`
  branch only) gives the Melitz zero-profit marginal cost in closed form:
  `mc_hat = (fixedcost*σ / (gdp*constGammaPow))^(1/(1-σ)) * (σ-1)/σ`. Algebraically
  consistent with the brief's `C_od * zhat^(σ-1) = σ*w_o*f_od` cutoff condition and worth
  cross-checking against, but re-derive and re-test independently rather than porting the
  function (it is entangled with the rejected `ρ`/`ϵ` Bertrand generalization).
  Genuinely Melitz-only cutoff/tail algebra is safe to reuse; the `ρ!=σ` generalization is not.
- **Pareto tail-expectation shape.** The recurring closed form
  `((1/μ)/(1/μ - σ + 1)) * max(k,1)^(1 - 1/(μ*(σ-1)))` (e.g. `Nested_moments!.jl:574`,
  `1012`, `1308`) is the Pareto truncated-tail expectation `E[z^(σ-1) 1{z>k}]` written in
  terms of `μ = 1/θ*`. The functional form is a useful cross-check for
  `src/melitz/pareto.jl`'s analytical tail moments, but re-derive from first principles and
  verify against numerical integration per the brief — don't copy the expression verbatim,
  since every instance in the file is embedded in one of the rejected ad-hoc
  parameterizations below.
- **Autarky zero-profit normalization idea.** The general idea in
  `Melitz_transform_θ` (`Nested_moments!.jl:536-545`) — that fixing the autarky cutoff
  pins down `f[d,d]` (or equivalently the autarky price aggregate) via the zero-profit
  condition — is the right shape of argument for the brief's autarky cutoff normalization.
  The specific formula there is baseIndex-only, uses `N` as a firm count, and is not
  independently tested; treat only the *logic* (cutoff-one ⟹ one pinned-down primitive) as
  reusable, and re-derive the exact algebra under this repo's own conventions.
- **Vectorized firm-level equilibrium evaluation.** The general pattern in
  `Nested_equilibrium_quantities_Melitz!` — compute marginal costs for a whole draw vector
  at once, then apply the entry mask via elementwise multiplication rather than branching
  per firm — is a reasonable, allocation-light vectorization pattern worth mirroring
  (though the new code should vectorize over origins/destinations too, which this file
  never does for Melitz).

## Reject

The following patterns are explicitly rejected per the project brief and must not appear
in the new implementation:

- **Foreign-country aggregation via `baseIndex` + `Not(baseIndex)` / index `1` as "rest of
  world".** Throughout `Nested_hFunction_Melitz!`, the destination loop is gated by
  `if d != baseIndex && d != 1; continue; end` and the origin loop by
  `if o != baseIndex && Melitz == 1; continue; end` (`Nested_hFunction.jl:117-125,
  141-145`). Only cell `[baseIndex,baseIndex]` and the aggregated `[baseIndex, RoW]` cell
  are ever computed for Melitz; the other `D^2 - 2` bilateral cells are never touched. This
  is precisely the "destination 1 as rest-of-world" / "only loops over focal + one foreign
  destination" anti-pattern the brief bans. The new moment builder must compute all `D^2`
  cells explicitly (verified by a dedicated indexing test).
- **Hybrid `ρ`/`σ` branch and perfect-competition fallback.**
  `Nested_equilibrium_quantities!` (`Nested_hFunction.jl:675-712`) branches on
  `maximum(NumberActiveFirms_d) > max_firms || ρ > ρ_max` (perfect competition),
  `ρ == σ` (Melitz), else a general Bertrand nested-logit solve via `nlsolve`. `ρ` is
  fixed at `100.0` or `σ` throughout the transform functions specifically to force one of
  these branches. None of this belongs in a pure Melitz module — there is no `ρ`, no
  firm-count ceiling, and no Bertrand market-share solve in the new code.
- **Numerical-firm-count interpretation of `N` and `1/N` scaling.** `constGammaPow[d] =
  N^(-1) * γ[d]^(σ-1)` and `fixedcosts_d = repeat(w .* fixedcosts[:,d] ./ N, inner = N)`
  (`Nested_hFunction.jl:22-23, 127-129`) treat `N` as a number of simulated firms per
  origin and divide both the price-power aggregate and fixed costs by it, then replicate
  values `N` times. The brief bans exactly this: `N_o`/`entrant_mass` is an economic mass
  that multiplies aggregate quantities, and `num_draws`/`W` is a pure integration-accuracy
  setting that must never divide revenue or fixed costs.
- **Manual counterfactual using only the first productivity draw.**
  `Nested_hFunctionCounter!` (`Nested_hFunction.jl:650-672`) computes
  `this_price = marginal_costs[1]*σ/(σ-1)` — i.e. it evaluates the autarky counterfactual
  at a single arbitrary draw index and broadcasts (`sales .= this_sales*entry_decision`)
  rather than averaging over the reference draws. The new counterfactual code must use the
  same per-draw firm-quantity routine as the baseline, evaluated at every draw.
- **Reuse of unused parameter slots for unrelated cutoffs.** Comments like
  `"this parameter is not used in the outerloop, so I use it here..."`
  (`Nested_moments!.jl:572, 1010, 1156, 1298-1299`) repurpose `NumberActiveFirms_θ[1]`,
  `NumberActiveFirms_θ[3]`, `NumberActiveFirms_θ[4]` etc. as ad-hoc cutoff parameters
  `k_ox`, `k_oo`, `k_oo_prime` with hardcoded multipliers (`3 *`, `1.01 *`, `2 *`,
  `0.1 *`) that change between the four transform-function versions with no
  documented rationale. Every parameter in the new implementation must have one
  documented name and role (see the parameter inventory table in
  `docs/melitz_delta_star.md`).
- **Parameters read and then silently overwritten.**
  `gdp_adjustment = ones(D)` is computed meaningfully on lines 9-10 of
  `Nested_hFunction_prep` (`gdp_adjustment = (1 .+ ξ) ./ (sum(lambda ./ τ, dims=1))'`,
  then immediately `gdp_adjustment = (1 .+ ξ)`) and then unconditionally clobbered again
  on line 14 (`gdp_adjustment = ones(D)`), regardless of the `Counter` branch that
  produced a different value above it. This is the exact "hard-coded
  `gdp_adjustment = ones(D)` after calculating another adjustment" anti-pattern the brief
  names. `NumberActiveFirms[baseIndex]` is likewise computed several different ways in
  sequence within a single `Melitz_transform_θ*` variant and then overwritten by
  `impose_M_Mprime_equality` blocks. The new code must compute each object once, from one
  documented formula.
- **Hard-coded `M = 1`.** `Nested_equilibrium_quantities!`: `M = 1 # for now....`
  (`Nested_hFunction.jl:679`). Dropped entirely — the new implementation has no `M`
  (varieties-per-good) concept.
- **Silently setting all foreign/other fixed costs equal.**
  `fixedcosts_dd = fixedcosts_d[N*d] # for the hybrid model with M = 1, we impose that
  fixed costs are the same for all exporters` (`Nested_hFunction.jl:129`), and
  `fixedCosts[baseIndex, Not(baseIndex)] .= fixedCosts[baseIndex, 1]`
  (`Nested_moments!.jl:637, 1221, 1417`) — every foreign destination's fixed cost is
  overwritten with the single `[baseIndex, 1]` value after being computed. The new `f[o,d]`
  matrix must retain independent, heterogeneous values for every cell.
- **`gdp_adjustment = ones(D)` after calculating another adjustment.** See above —
  flagged separately because it recurs verbatim (`Nested_hFunction.jl:6-14`).
- **Disabled cutoff checks / dead `if false` blocks.** Numerous `if false` guarded blocks
  survive in both files (e.g. `Nested_hFunction.jl:374-381, 423-433, 594-600`;
  `Nested_moments!.jl` has entire multi-hundred-line commented-out alternative derivations
  inside `Melitz_transform_θ1`/`_θ2`, lines ~726-909 and ~1420-1591). None of this dead
  code is carried forward.
- **Unused "proportional profits" parameter.**
  `proportional_profits_in_BaseIndex` is threaded through every function signature,
  clipped (`min(1.0 - profit_share_offset, max(...))`, see next bullet), but its only
  effect anywhere in `Nested_hFunction.jl` is commented out
  (`Nested_hFunction.jl:174, 182`: `#@. @view(G[:, Profits_Of_BaseIndex_index2]) = ...`).
  It is dead weight carried through the whole parameter vector. Not reused.
- **Broad clipping with `max`/`min` that changes the economic model.**
  `proportional_profits_in_BaseIndex = min(1.0 - profit_share_offset, max(θ[...],
  minimum_profit_share))` (repeated in every `Melitz_transform_θ*` variant) and
  `k_ox = max(1, k_ox)` / `k_od = max(k_od, 1)` silently move parameters to a boundary
  without surfacing that the constraint bound. The brief specifically asks for
  positivity via log-parameterization rather than `max(0, ·)`-style clipping; the new
  code follows that instead.
- **Ambiguous inverse-cost convention mixed with the paper's `A`.**
  `Aod[:, :] = cHat[:, :] .^ (-1)` (`Nested_moments!.jl:525, 709, 972, 1109, 1285,
  1492`) shows the legacy code already had to invert the repository's `cHat` object to
  get something it then calls `Aod` and uses as productivity. This confirms the addendum's
  concern directly: the Ricardian repo's own `cHat`/similar objects are a unit-cost
  shifter, not the paper's efficiency shifter. The new Melitz module defines `A[o,d]` in
  the paper's convention from scratch and does not silently reuse `cHat`.

## Replaced by Ricardian infrastructure

The following generic functionality comes from `production/fullA-exact` rather than being
reimplemented, per the independent infrastructure-mapping pass (see
`docs/melitz_delta_star.md` §"Reused Ricardian infrastructure" for the full file/line
inventory):

- Reference-draw generation: production's actual pattern is `prepare_cc/drawU.jl` +
  `prepare_cc/genRands.jl` (seeded `rand!`, one `Random.seed!(seedU)` call up front, shape
  `W×D` under `UoModel=1`) — not `cc_algo/rhalton.jl`, which is validated but unused on the
  production hot path. The new Melitz module follows the same "seed once, draw once, reuse
  everywhere" discipline, adapted to Pareto via inverse-CDF.
- The origin-destination double-differencing operator `doubleDiff` (`misc/doubleDiff.jl`)
  for both gravity restrictions — legacy code re-implements one gravity moment ad hoc
  inline (`Nested_hFunction_Gravity_EK!`, `Nested_hFunction.jl:440-475`) using its own
  `doubleDiff` import; the new code reuses the repo's tested version directly. Note
  production's own *live* gravity moment actually uses a different operator,
  `withinTransform` (two-way fixed-effects "within" transform), because `doubleDiff`'s
  cell-referenced construction does not reproduce an OLS two-way-FE coefficient — but that
  distinction matters for *elasticity estimation*, not for evaluating the scalar covariance
  restrictions `⟨ΔΔlogτ, ΔΔlogA⟩=0`/`⟨ΔΔlogτ, ΔΔlogf⟩=0` the addendum defines, which match
  `doubleDiff`'s cell-referenced double difference exactly (both live in the same
  `(D-1)²`-dimensional space). The new code uses `doubleDiff`, as the brief names, and
  documents this distinction rather than silently picking one.
- The CC minimum-divergence inner loop (`cc_algo/ccInner.jl`,
  `cc_algo/inner_loop_functions.jl`, `PsiObjectiveBundle.jl`) — legacy code has no
  min-divergence machinery at all; `Nested_moments_simple!` only ever fills `G`/`K` for a
  *fixed* `θ`, it never computes `Delta(θ)`. Likewise the outer Delta-star minimization
  (`cc_algo/ccOuter.jl`, `outer_loop_functions.jl`, and in particular the newer
  method-agnostic `outer_loop_cached.jl`/`free_param_map.jl`), KNITRO setup/warm-starts,
  and result serialization — none of this exists in the legacy files, which only
  construct moment matrices for an externally driven `θ`.
- Wage/expenditure/factor-market-clearing conventions: **correction to an earlier draft of
  this document** — production's real code has no `gdp_adjustment` object or concept at
  all (confirmed by exhaustive grep of `production/fullA-exact`); GDP is simply `w .* L`
  with no deficits and no tariff revenue anywhere in this branch, and wages are solved by
  two plain damped fixed-point iterators (`prestep/iterWagesPreStep!.jl`,
  `setup/iterWagesTheory!.jl`). There is therefore no `gdp_adjustment` treatment to reuse;
  the correct lesson from the legacy code's `gdp_adjustment = ones(D)` bug (Reject section)
  is simply that GDP/expenditure must be computed once, from one documented formula, with
  no silent adjustment factor — which is what the Ricardian code already does by never
  introducing one.
- The `A`/productivity sign convention itself: confirmed independently (not just inferred
  from the legacy file's `cHat.^(-1)` pattern) that `production/fullA-exact`'s underlying
  structural convention is `MC_od = w_o·τ_od/(A_od·z_o)` — i.e. *higher* `A` means *lower*
  marginal cost, exactly the paper's convention. The repo internally stores the reciprocal
  `AodPow = 1/A_od` for its own numerics, but the economic object matches the paper
  directly. The new Melitz module defines its own `A[o,d]` in this convention from
  scratch (no adapter needed, since the underlying convention already agrees with the
  paper) and does not reuse `cHat`/`AodPow` themselves.
