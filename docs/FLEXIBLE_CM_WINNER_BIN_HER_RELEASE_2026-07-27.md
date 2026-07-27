# Flexible-CM Winner-Bin H_ER Release — 2026-07-27 (Section 2)

## What changed

`hessian_cm_structured!` (serial, `cm_hessian_architectures.jl`) and `hessian_cm_structured_v2!`
(threaded, `cm_hessian_threaded.jl`, the real production default via `archC_hess_cb_builder`) now
support `cctx.cm_cross_hessian_backend = :winner_bin`, reusing the already-validated
`winner_pair_cross_hessian_fill!`/`winner_pair_cross_hessian_cm_block!` primitive
(`winner_pair_cross_hessian.jl`, built and gated by the inherited session at commit `ab279a9`) to
fill the economic x CM-grid cross block `H_EC` instead of the dense `CScum` bin-prefix-sum built
from `E = H[:, 2:1+NCORE]`.

`build_bin_tables!`/`prefix_sum_tables!` (both serial and threaded variants) gained a `fill_S`
kwarg; when `:winner_bin` is active, `fill_S=false` skips the `S[x,j,bx] += ws*E[s,j]` accumulation
entirely -- the only remaining reader of `E` in the Hessian callback becomes the H_CC block's own
`T`/`CT` tables, which were built from `Bidx`/`w` alone and never touched `E` to begin with. For a
plain flexible-CM context (`cctx.ncore_core == cctx.NCORE`, no CM+mean/pair-ZC widening) with the
winner-pair core-Hessian backend already active, this Hessian callback now performs **zero** reads
of `obj.H`'s dense economic columns.

`CMBinHessCtx` gained two fields: `cm_cross_hessian_backend::Symbol` and
`cross_scratch::Union{Nothing,WinnerBinCrossScratch}` (persistent, rebuilt only on a genuine
`(ncolI,D,L)` size change -- confirmed via `objectid` stability across repeated warm calls, never
per Hessian callback).

`CM_CROSS_HESSIAN_BACKEND_DEFAULT` (`core_exact_hessian.jl`) flipped `:dense_reference` ->
`:winner_bin` after both gates below passed. `build_cm_meanzc_bin_ctx` (CM+meanZC) keeps a
hardcoded `:dense_reference` default, independent of this flip -- structurally out of scope (see
below). `hessian_cm_frechet_structured!` (common-Fréchet) does not call into this dispatch at all
(a separate function; Section 3's scope).

## Scope guard

`:winner_bin` only activates when:
1. `cctx.cm_cross_hessian_backend === :winner_bin` (requested),
2. `cctx.ncore_core == cctx.NCORE` (no CM+mean/pair-ZC widening -- that case folds mean/pair
   columns into the same dense NCORE block and is explicitly out of this section's scope, see task
   Section 4), and
3. `cctx.core_ws !== nothing && cctx.core_ws_for === cf` (the winner-pair H_EE precompute was
   *actually* refreshed for the current outer point by `_fill_cm_HEE!` just now, not stale or a
   dense fallback).

Any failure falls back to `:dense_reference`, recorded via `record_dense_cross_hessian_call!` (not
silent). CM+meanZC's `ncore_core < NCORE` whenever `K_mean>0` or `K_pair>0` means condition 2 always
fails there, so it is automatically and correctly excluded without needing its own guard logic.

## Gates (both ALL PASS)

**D=4** (`test_flexible_cm_winner_bin_her_wiring_d4.jl`): 2 contrasts (anchored, orthonormal) x 3
`L` values (10, 20, 50) x 3 points (calibration, 2 independent perturbed points) x 2 architectures
(serial `hessian_cm_structured!`, threaded-production `hessian_cm_structured_v2!`) = 36 complete
packed-Hessian comparisons (`:dense_reference` vs `:winner_bin`), plus per-`(contrasts,L)`
inner-solve status/dual-point match and persistent-workspace identity checks. **ALL PASS**,
`max|ΔH|` in `[6.1e-16, 1.04e-15]` against a Hessian scale of ~1.0-1.02.

**Real D=20/W=80,000/L=50** (`test_flexible_cm_winner_bin_her_wiring_d20.jl`,
`destination_sample=:exclude_row`): 2 contrasts x 2 points (calibration, a near-delta=1 perturbed
point) x 2 architectures = 8 complete packed-Hessian comparisons, plus inner-solve/workspace
checks. **ALL PASS**, `max|ΔH|` in `[9.24e-14, 2.06e-13]` against a Hessian scale of ~3980-4303.
Complete inner solve status matches (`nStatus=-103` both backends, both contrasts) and dual point
agrees to `<1e-12`.

Timing (informational, not the basis for the default decision alone -- correctness gated first):
real D=20, serial cold ~7x faster (`3.6s -> 0.49s`, anchored/calib); threaded_v2 (production
default) warm ~2x faster (`0.66-0.84s -> 0.27-0.40s` across contrasts/points). Warm
`hessian_cm_structured_v2!` (`:winner_bin`) allocates a stable 9.4 MB/call at real D=20, with zero
persistent-workspace resizes across repeated calls.

## Runtime counters (task Section 7, partial)

`dense_cross_hessian_calls`/`operator_cross_hessian_calls`/`winner_cross_hessian_calls` added to
the shared `NoDenseGCounters` (`no_dense_g_counters.jl`, consolidated there rather than a separate
Ref -- see commit `1aeac63`). D=4 gate run: `dense_cross_hessian_calls=116` (the explicit
`:dense_reference` half of every comparison), `winner_cross_hessian_calls=128`
(`=operator_cross_hessian_calls`, the `:winner_bin` half).

## Not done in this section (deferred to Sections 3-5)

- Common-Fréchet, CM+ZC, ZC-only H_ER (separate subagent tracks, see master report).
- Rectangular (`D != Ddest`) and non-last-omitted-destination D=4 configurations were **not**
  separately exercised -- no rectangular-layout D=4 CM context builder exists in this repository
  (confirmed by search; `d4_exact_setup`/`d20_real_setup` are both always-square). The underlying
  `WinnerPairHessCtx`/`winner_pair_cross_hessian.jl` primitives carry `Ddest` as a first-class
  dimension throughout (not hardcoded equal to `D`), so this is an untested-not-unsupported gap
  worth closing in a future session, not a known failure.
