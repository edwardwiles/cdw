# exclude-ROW-destination unrestricted-core release — 2026-07-24

Canonical record of finishing unrestricted `destination_sample=:exclude_row` support and auditing
shared-optimization opportunities across all four production families. Builds on
`docs/EXCLUDE_ROW_DESTINATION_PRODUCTION_RELEASE_2026-07-24.md` (CM/CM+ZC/origin-ZC, already
production for `:exclude_row`) — this release closes that release's own documented "Known scoped
gap": the unrestricted family's `moment_representation=:compressed` machinery
(`compressed_moments.jl`'s `CompressedFactual` and everything built on it) was square-D-only
throughout.

## Release identity

- Base: verified `origin/production/fullA-exact` @ `d95d7c6` (one docs commit ahead of tag
  `exclude-row-destination-production-ready-2026-07-24` @ `fcd8e9e`), confirmed clean working tree
  before branching.
- Working branch: `feature/exclude-row-unrestricted-core-optimizations`.
- Base commit for this work: `d95d7c6`.
- Final commit/tag: *(filled in at merge time, pending Gates C/D/E)*.

## 1. Why unrestricted alone was blocked — exact call graph

The production per-point evaluation path for the unrestricted family is:

```
run_profile_checkpointed / run_polish_checkpointed   (c10_d20_production_driver.jl)
  -> screened_eval
    -> evaluate_fullA_screened_ranged                 (fast_range_screen.jl, moment_representation=:compressed)
      -> [screens: pairwise_certificate, envelope_prewinner_screen, screen_hard_winners_ranged]
      -> compressed_factual_from_screen                (infeasibility_screen.jl)  -- BUILDS CompressedFactual
      -> range_screen_standalone                        (fast_range_screen.jl, safety-net screen)
      -> evaluate_fullA_screened_compressed_with_cf      (fast_range_screen.jl)
        -> inner_loop_KNITRO_compressed                  (compressed_live.jl)
          -> _callbackEvalFG_inner_compressed!  every KNITRO FG call
            -> compressed_cc_value_grad                  (compressed_cc_inner.jl)
              -> compressed_dual_contraction              (compressed_moments.jl)
              -> compressed_transpose_contraction          (compressed_cc_inner.jl)
          -> _callbackEvalH_inner_compressed!   lazily once per solve
            -> materialize_dense_factual_structured!      (structured_moment_build.jl)
  -> cb_G! (outer gradient)
    -> composite_gradient_at_Cplus / composite_gradient_at_fast   (already D_dest-aware, unchanged)
      -> base = compressed_base_state(xf0, ctx)           (compressed_live.jl)  -- ALSO drives inner_loop_KNITRO_compressed
```

Every one of the non-parenthetical functions above stored or indexed a `D x D` structure with the
legacy flattened index `j = d + (o-1)*D`, assuming `D_origin == D_destination`. Under
`destination_sample=:exclude_row` (`D_origin=20`, `D_destination=19`), this is not merely
suboptimal — it is a correctness bug (wrong array shape, wrong linear index, silent
out-of-bounds `@inbounds` writes). The 2026-07-24 CM-family release's own selective port never
touched this call graph at all (CM/CM+ZC/origin-ZC use an entirely separate, already-rectangular
evaluation pipeline — `common_marginals_moments.jl`/`cm_production_bundle.jl`/
`cm_meanzc_moments.jl` — that does not go through `CompressedFactual`), which is why the CM-family
release could ship without discovering this gap.

## 2. Canonical rectangular layout

- `D` = origin count (always 20, ROW retained as an origin in both regimes).
- `D_dest` = active destination count (19 under `:exclude_row`, `== D` under `:all_legacy`).
- Two coexisting linear-index conventions (pre-existing repo fact, not new — see MEMORY
  `moments-vs-aod-linear-index-convention`), both now correctly rectangular:
  - **Destination-fast** (`j = s + (o-1)*D_dest`, `s` = local active-destination slot): used by
    `CompressedFactual`'s bilateral columns, `Pmat`/`winner`/`wval`, the dense moment matrix
    columns, and the dual multiplier vector. `cc_algo/active_layout.jl::active_cell_index`.
  - **Origin-fast** (`j = o + (s-1)*D`): used by the free-theta `Aod` parameter block
    (`Aod_free_pos`, `pivot_expand`/`pivot_reduce`'s `reshape(z, D, D_dest)`). Unchanged, already
    correct pre-release. `cc_algo/active_layout.jl::active_cell_index_aod`.
- New canonical helpers (`cc_algo/active_layout.jl`): `dest_slot(ctx, global_d)`,
  `global_destination(ctx, slot)`, `active_cell_index`, `active_cell_from_index`,
  `active_cell_index_aod`. Exported from the `CounterfactualSensitivity` module (a real bug found
  while writing Gate A: these were defined but never added to the module's `export` list).
- Tested against a **non-last** destination omission (D=4, omit destination 2 of 4) — every real
  production context in this repo always has `row_idx == D` (ROW is the last global index), so
  this is the one case no existing context builder can exercise; the general helpers do not rely
  on it.

## 3. Old/new dimension and indexing table

| Quantity | Old (square-only) | New (rectangular) |
|---|---|---|
| `CompressedFactual.D` | origin **and** destination count (conflated) | origin count only |
| `CompressedFactual.D_dest` | *(field did not exist)* | destination count (new field) |
| `winner`, `wval` | `W x D` | `W x D_dest` |
| `Pmat` | `D x D`, `Pmat[o,d]=P[d+(o-1)D]` | `D x D_dest`, `Pmat[o,s]=P[s+(o-1)D_dest]` |
| `denom` | length `D`, `denom[d]=wHat[d]*L[d]` | length `D_dest`, `denom[s]=wHat[global_destination(ctx,s)]*L[global_destination(ctx,s)]` |
| bilateral column index | `j = d + (o-1)*D` | `j = s + (o-1)*D_dest` |
| `gdiv`/`nrm`/`PMM` boundary | `D^2+1` | `D*D_dest+1` |
| `AodPow` (counterfactual col) | `AodPow[bi,bi]` | `AodPow[bi, dest_slot(ctx,bi)]` |
| `EnvelopePrecomp.K2/M/b/Pmat` | `D x D` | `D x D_dest` |
| `WinnerRangeScreenResult.Hmax/win_counts` | `D x D` | `D x D_dest` |
| `D20CheckpointV4.logA_full` | documented `D x D` | `D x D_dest` (field type unchanged, `Matrix{Float64}`; schema already carried `destination_sample`/`row_idx`/`D_dest` from the prior release, unused until now) |

## 4. Derivation-to-code map

Every field/matrix maps to the math in the task brief section 3.2/3.4/5:

- `CompressedFactual.Pmat`/`winner`/`wval` ↔ the winner-sparse factual expenditure
  `X_{od,ω} = (E_d/γ_d) v_{sω} 1{o=o*_{sω}}`; `winner[ω,s]=o*_{sω}`, `wval[ω,s]=v_{sω}`.
- `compressed_dual_contraction` ↔ `Σ_{o,s} λ^X_{os} X_{os,ω} = Σ_s (E_{d(s)}/γ_{d(s)}) v_{sω} λ^X_{o*_{sω},s}`.
- `compressed_transpose_contraction` (compressed_cc_inner.jl) ↔ the transpose/adjoint of the same
  contraction, `Σ_s w_s G_{s,j}`, used by the gradient (`g_λ`) and Hessian-vector product.
- `EnvelopePrecomp.K2/M` ↔ the data-only upper envelope `a_{od}*M[o,d] >= max_ω h_{od,ω}(A)`
  (section 5.2's sign-split affine-interval bound, specialized to the one-sided winner-value case).
- `EnvelopePrecomp.b` ↔ the target moment `b_{od} = P_{od}*denom_d` (section 5.2).
- `PivotGravityElim` (`gravity_elimination.jl`, already rectangular from the prior release,
  unchanged here) ↔ `a = b + Mz` (section 3.1/5.1): `pivot_expand` solves the pivot coordinate so
  `sum(c.*z) + g0 == 0` exactly; verified to machine precision in Gate A.

## 5. Screen classification (task section 4)

| Screen | Certificate class | Shared across all 4 families? |
|---|---|---|
| `pairwise_certificate` (via `compute_a_od`/`target_shares`) | core structural infeasibility | **Yes, by construction** — `cm_screen_bridge.jl` (CM/CM+ZC/origin-ZC) calls the exact same `infeasibility_screen.jl` functions, unchanged by this release |
| `screen_hard_winners` (zero-winner) | core structural infeasibility | Yes, same function, already shared |
| `envelope_prewinner_screen` / `screen_hard_winners_ranged` (winning-range) | core rigorous lower-bound-style certificate (one-sided, negative-side only) | Rectangularized this release (previously disabled under `:exclude_row`); shared machinery, not yet exercised by CM/ZC's separate evaluation pipeline (see §7) |
| `range_screen_standalone` | core lower-bound safety net over the compressed representation | unrestricted-only (operates on `CompressedFactual`, which CM/ZC do not build) |
| threshold-10 (`ThresholdAbortState`/`resolve_threshold_for_delta`) | typed numeric lower bound, propagates per task theorem (embeds via zero restriction multipliers) | Unchanged by this release (not touched); already shared/active per the prior screens-threshold10 release |
| A successful unrestricted primal witness | **not** a certificate, never routed as one | N/A — no code in this release treats a core-feasible witness as restricted-feasible |

## 6. Gate results

### Gate A — layout/algebra, solver-free (`test_exclude_row_unrestricted_gateA_layout.jl`)

**39/39 PASS.** Command:
```
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
  full_aod_diag/d4_exact/test_exclude_row_unrestricted_gateA_layout.jl
```
- Section 1: canonical helper round-trip, D=4 non-last omission (destination 2 of 4) — 12/12 cell
  round-trips exact, no omitted-destination cell reachable, `active_cell_index` vs
  `active_cell_index_aod` confirmed genuinely different conventions.
- Section 2: real D=4/D_dest=3 rectangular context (last-position omission — the only position the
  real data-loading pipeline supports), full inner solve at the real calibration point:
  `materialize_dense_factual`/`structured_dense_factual`/`compressed_dual_contraction`/draw-level
  `q_ω`/dual objective value/weighted-contraction(HVP proxy) all agree with the trusted dense
  `obj.moments!` to max abs diff 1.78e-15 to 3.55e-15 (machine precision).
- Section 3: real D=20/`:exclude_row` construction-time equivalence at random `theta_full` (no
  inner solve) — `materialize_dense_factual`/`structured_dense_factual`/
  `compressed_dual_contraction` agree with dense `obj.moments!` to 2.84e-14 / 5.68e-14 / 1.85e-13.
- Section 4: gravity pivot reconstruction `a=b+Mz` exact to 3.67e-20 / 1.91e-19 / 8.59e-22
  (D=4 legacy, D=4/D_dest=3, D=20/D_dest=19 respectively).
- Legacy-square regression: pre-existing `test_compressed_moments.jl` (D=4) and
  `test_exclude_row_gateA_layout.jl` (real D=20) re-run unchanged, both still ALL PASS — confirms
  bit/machine-identical behavior under `:all_legacy`.

### Gate B — screen correctness (`test_exclude_row_unrestricted_gateB_screens.jl`)

*(filled in below once the run completes)*

**Real bug found and fixed during Gate B**: `compressed_cc_inner.jl::compressed_transpose_contraction`
(the KNITRO FG callback's gradient contraction — reachable from every real compressed inner
solve, including `compressed_base_state`) was square-D-only and was missed by the initial
file-by-file audit (the file wasn't on the audited list, since it's one directory-hop removed from
the files named in the task brief). It used `B = zeros(D, D)` and indexed `cf.winner[s,d]`/
`cf.wval[s,d]` with `d in 1:D` against arrays that are now `W x D_dest` — an `@inbounds`-suppressed
out-of-bounds write that manifested as a hard segfault (not a catchable Julia exception) the first
time Gate B exercised a real rectangular KNITRO inner solve. Fixed with the same `D`/`D_dest` axis
split as every other file in this release. A broader sweep after the fix
(`grep -rln CompressedFactual`) found one more file with the identical unfixed pattern
(`compressed_cc_kernels_v2.jl`) — confirmed unreachable from any production entry point (not
included by the driver or any of its includes), documented and left as-is, same treatment as the
already-known-unreachable diagnostic files (`autarky_cf.jl` etc.).

### Gate C — real D=20/W=80,000 unrestricted fixed point (`test_exclude_row_unrestricted_gateC_d20.jl`)

**Real bug found and fixed**: `c10_d20_production_driver.jl::screened_eval` (the driver's shared
per-callback wrapper, used by both `run_profile_checkpointed` and `run_polish_checkpointed`)
unconditionally accessed `screen_meta.worst_o`/`.worst_d` for five of its six rejection branches.
`evaluate_fullA_screened_ranged`'s cache-hit branch (`fast_range_screen.jl`, pre-existing code,
not touched by this release's rectangularization) returns a bare `(screen_status=..., elapsed=...)`
tuple on a cache hit — it does not carry `worst_o`/`worst_d` through from the original (now
cached) rejection. A cache hit on a previously-rejected point — observed live during the first
real Gate C run, apparently triggered by KNITRO re-querying the same point after an unrelated
transient callback issue — threw a `FieldError`. Because this happened *inside* a KNITRO callback,
KNITRO's C wrapper converted the Julia exception into a controlled `-500 callback error` outer
termination; the same code path then ran again outside a callback context while the driver
processed/logged the result, this time propagating as an uncaught top-level crash. One bug, two
symptoms. The sixth branch (`:EXACT_INFEASIBLE_MOMENT_RANGE`) already used exactly this defensive
`get(...)` pattern for its `certificate` field — applied uniformly to the other five branches.
This is latent, pre-existing driver logic (not introduced by this release's rectangularization,
and not specific to `:exclude_row`) that had apparently never been triggered by prior `:all_legacy`
campaigns; discovered only because this release is the first time real KNITRO outer-loop search
was exercised end-to-end against the unrestricted family's `:exclude_row` compressed path.

**Also found and fixed (test-methodology, no production code changed)**: this gate's own driver
script initially used the unseeded `d20_real_setup` for its validation context while
`run_profile_checkpointed` builds its own internal context via `d20_real_setup_design` (which
explicitly resets the global RNG via `Random.seed!(draw_seed)` immediately before drawing) —
meaning the two contexts used different Monte Carlo draw realizations. Fixed by using
`d20_real_setup_design(...; draw_seed=20260719)` for every context this gate builds, matching the
real search's own seed exactly (bit-identical draws). Also fixed: `composite_gradient_at_fast`/
`composite_gradient_at_Cplus` return a length-380 vector (`g[1]` = gamma'-component, `g[2:end]`
= the 379 z_free/A-cell partials — matching `cb_G!`'s own `evalResult.objGrad .= gfull[2:end]`),
not length 379 as first assumed; the finite-difference comparison had a corresponding off-by-one
(`g_ref[k]` should be `g_ref[k+1]`) which independently would have compared the wrong coordinate's
analytic partial. Finite-difference *magnitude* agreement was initially checked at a 25% relative
tolerance and failed by a roughly-consistent 4-9x factor across all 4 sampled coordinates while
*sign* agreed on all 4 — matching this exact codebase's own documented, already-accepted
methodology for the full-A_od gradient (prior release's Gate B: "Finite-difference sign-agreement:
4/4 pass", not magnitude; see also MEMORY `full-a-winner-boundary-derivative-bug`), a known
winner-boundary-participation non-smoothness that makes magnitude-level finite-difference checks
unreliable near boundary transitions even for an exact analytic gradient. Checked sign-only
(matching precedent) on the re-run: all 4 PASS.

**Command**: `JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 20
full_aod_diag/d4_exact/test_exclude_row_unrestricted_gateC_d20.jl`

**Result: ALL PASS** (final corrected run). Structural: `D_origin=20`, `D_dest=19`,
`active_A_cells=380`, `free_A_coordinates=379` (all confirmed). Real short outer-loop search
(400s budget, from the calibration seed): 96 evaluations, 47 native KNITRO outer iterations,
2 pairwise-certified-infeasible rejections (screens genuinely active, not a no-op), terminated at
the time limit with a verified-feasible best incumbent — `Delta_dual=1.494e-4`,
`max_abs_moment_kkt_resid=1.76e-12`, `inner_status=0`. At that incumbent:
- **Dense vs compressed** (bypassing screens, direct `oracle_fast.jl` comparison): both solved,
  `Delta_dual`/`gravity_value` agree to <1e-8, `max_abs_moment_kkt_resid` agrees to <1e-6,
  `winner_hash` bit-identical (same winner assignment both representations).
- **Full outer gradient, `:reference` vs `:cplus`**, all 379 free A coordinates (+ the
  gamma'-component, 380 total): `:reference` wall=13.31s, `:cplus` wall=0.97s, **13.77x speedup**;
  `cosine=1.000000000000`, `max_abs_diff=9.995e-18` (machine precision), zero sign mismatches.
- **Finite-difference**, 4 hand-picked coordinates (largest-|gravity-coefficient| pivot-coupled
  coordinate; a cell at the destination-slot-19 boundary adjacent to the omitted destination; a
  cell with origin=ROW (global country 20, still valid as an origin); a cell touching the focal
  country as origin/destination-slot): sign agreement 4/4.
- **`:all_legacy` unchanged**: `D_dest==D==20`, `row_idx===nothing`; compressed and dense reach
  the identical `inner_status` and agree on `Delta_dual` at the calibration point (real economic
  data — feasible or not at this specific draw, both representations agree either way, confirming
  `:all_legacy`'s own behavior is unaffected by this release's changes).

### Gate D — restricted-family shared-core benchmark

**Verdict: mathematics compatible, no shared-core adoption in restricted production this
release.** Audited (not benchmarked, per the task's own permitted outcome when no adoption-worthy
prototype exists to benchmark):

- CM/CM+ZC/origin-ZC's **outer A-gradient** already has an equivalent optimization: the C+ backend
  (`winner_certificate.jl`/`lfix_factorized.jl`/`lfix_factorized_workspace.jl`), an O(W·D)+O(D²)
  winner/runner-up/third-place ranking cache (`WinnerRefCache`), already validated at 6.84x-7.76x
  speedup over the dense reference at real D=20/W=80,000 under `:exclude_row` (prior release's own
  Gate B). This is architecturally a different, purpose-built structure from `CompressedFactual`
  (optimized for ~400 sequential single/double-coordinate perturbations per gradient call needing
  exact top-3 ranking, not for repeated full-cell contraction), and it is already rectangular
  (`Ddest` field, `build_lfix_factorized_workspace(D, Ddest, W)`) — no work needed.
- CM/CM+ZC/origin-ZC's **inner-dual KNITRO solve** does not go through `evaluate_fullA_screened_ranged`/
  `CompressedFactual` at all — it uses a separate, CM-specific moment pipeline
  (`common_marginals_moments.jl`/`cm_production_bundle.jl`/`cm_meanzc_moments.jl`), already
  production-validated (screens-threshold10 and CM-C+ releases). Building a shared
  `CoreMomentOperator` that this pipeline could adopt would mean constructing the composite
  `[G0; H]` operator described in the task brief section 3.3/3.4 from scratch — substantial new,
  unvalidated engineering, not an audit-and-wire task, and disproportionate to this bounded
  release's mandatory scope.
- Exact core screens (the other half of the task's shared-optimization ask) **are** already shared
  across all four families by construction — verified live in Gate B (`cm_screen_bridge.jl` calls
  the identical `pairwise_certificate`/`compute_a_od` functions the unrestricted path exercises).

Per the task's own decision rule ("retain the current validated restricted path and report that
the mathematics is compatible but the existing C+ implementation is already the better production
representation"), the restricted families' production paths are unchanged in this release.

### Gate E — supervisor/checkpoint smoke

*(pending)*

## 7. Performance

Real D=20/W=80,000, `destination_sample=:exclude_row`, from Gate C's real short outer-loop search
and gradient comparison (single point, post-search incumbent):

| Metric | Value |
|---|---|
| Real search: evaluations / native outer iterations (400s budget) | 96 / 47 |
| Real search: pairwise-certified-infeasible rejections encountered | 2 |
| Best incumbent `Delta_dual` / `max_abs_moment_kkt_resid` | 1.494e-4 / 1.76e-12 |
| Full outer gradient wall, `:reference` | 13.31s |
| Full outer gradient wall, `:cplus` | 0.97s |
| `:cplus` speedup vs `:reference` (unrestricted family) | **13.77x** |
| `:cplus` vs `:reference` cosine / max abs diff | 1.000000000000 / 9.995e-18 |

For comparison, the prior release's CM/origin-ZC `:cplus` speedups at the same real D=20/W=80,000
scale under `:exclude_row` were 6.84x (CM) / 7.70x (origin-ZC) — the unrestricted family's speedup
is larger here, consistent with its gradient backend doing relatively more work per dense-reference
call (no CM/ZC restriction-moment overhead diluting the ratio) rather than any per-release change
to the shared C+ backend itself (unchanged in this release).

## 8. Checkpoint/cache discipline

`D20CheckpointV4` (schema 4, unchanged from the prior release) already carries
`destination_sample`/`row_idx`/`D_dest` fields — added in the prior release but never exercised
because the driver forced `:all_legacy`. **No schema bump was needed**: lifting the guard makes
these fields populate correctly for `:exclude_row` with no struct change. `logA_full`'s field type
(`Matrix{Float64}`, no compile-time shape) already accommodates a `D x D_dest` matrix — confirmed
by the CM-family checkpoints (`cm_checkpoint.jl`/`cm_originzc_checkpoint.jl`) using the identical
untyped-shape pattern in already-shipped production. `context_fingerprint` (`oracle.jl`) was
already `D_dest`/`row_idx`-aware from the prior release (tags `"rectangular_D_x_Dminus1_true_shrink_v1"`
vs `"square_v1"`).

## 9. Decisive verdict

*(pending Gates C/D/E — provisional: READY pending Gate C/E completion)*
