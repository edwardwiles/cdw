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

*(pending)*

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

*(real D=20/W=80,000 timing/allocation table — pending Gate C)*

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
