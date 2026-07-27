# Origin-ZC Winner-Aware H_ER Release — 2026-07-27 (Section 5)

## What changed

Origin-ZC's `archA_partitioned_hess_cb_builder` (`cm_hessian_architectures.jl`) now supports
`octx.zc_cross_hessian_backend = :winner_bin`, reusing the SAME shared cross-Hessian primitive
Section 4 built for CM+ZC -- `winner_pair_cross_hessian_zc_block!`/`_prep!`
(`winner_pair_cross_hessian.jl`) -- to fill `H_ER` (`∂∂f_∂∂x[1:NCORE, NCORE+1:n]`), this family's
**only** restriction block (origin-ZC has no CM-grid at all, `G = [E | Z]`). `H_RR`
(`∂∂f_∂∂x[NCORE+1:n, NCORE+1:n]`) stays dense BLAS **unconditionally, unchanged** -- explicitly out
of this section's scope, matching CM+ZC's own `HMM` treatment.

`OriginZCCoreHessCtx` gained `zc_cross_hessian_backend::Symbol` and `zc_cross_scratch::
Union{Nothing,WinnerZCCrossScratch}`. `build_originzc_core_hess_ctx` gained a
`zc_cross_hessian_backend` kwarg, defaulting to `ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT[]`
(`core_exact_hessian.jl`). `build_originzc_production_context` gained a passthrough kwarg of the
same name.

## Structural difference from CM+ZC (Section 4), and why the wiring differs

Origin-ZC has NO widened "economic" block -- `octx.NCORE` is the core's own true width (no
mean/pair folding), and the restriction block `H[:, 2+NCORE:1+n]` is a genuinely separate set of
columns after the core, not spliced into the middle of it. The pre-existing dense code therefore
computed `HC_core` (sqrt(`arg2`)-weighted economic columns) and `HC_eta` (sqrt(`arg2`)-weighted
restriction columns) as two SEPARATE slices of `H_copy`, then one `gemm!` for `HER` and a second
for `HRR`. Since `HRR` needs `HC_eta` regardless of which backend fills `HER`, the wiring computes
`HC_eta` (and its sqrt-weighting) unconditionally up front, and only computes `HC_core` (which
requires densely reading `H[:, 2:1+NCORE]`, the winner-conditioned economic columns) inside the
dense fallback branch. In the `:winner_bin` branch, `H[:, 2:1+NCORE]` is **never read** -- only
`H[:, 2+NCORE:1+n]` (the mean/pair restriction columns, already read for `HC_eta`/`HRR` regardless)
is passed to the shared primitive, unweighted (the primitive applies its own `S=arg2` weighting
internally, so it needs the RAW column, distinct from `HC_eta`'s pre-weighted copy used by `HRR`'s
own gemm).

## Gates

**Standalone primitive**: shared with Section 4 -- see
`test_winner_pair_cross_hessian_zc_d4.jl`'s origin-ZC half. D=4, `K_mean=1/K_pair=1` and
`K_mean=2/K_pair=2`, calibration + 2 perturbed points, against an independently-built dense
`(1/M)*E'*diag(S)*Z` reference. **ALL PASS**, `max|Δ|` in `[1.8e-15, 2.0e-14]`.

**D=4 wiring** (`test_originzc_winner_bin_her_wiring_d4.jl`, invoking
`archA_partitioned_hess_cb_builder`'s closure directly with mock KNITRO `evalRequest`/`evalResult`
NamedTuples -- only `.x`/`.hess` are ever read/written by that closure, so this exercises the exact
production code path without needing a full KNITRO callback registration): `K_mean=1/K_pair=1` and
`K_mean=2/K_pair=2`, calibration + 2 perturbed points, plus a complete real KNITRO inner solve per
config (independent `aug`/`obj_cm` per backend). **ALL PASS**, `max|ΔH|` in `[1.5e-15, 3.1e-14]`
against a Hessian scale of `3.13` to `2.6e3`. Complete inner solve status matches (`nStatus=0`) and
dual point agrees to `<1e-8`.

**Real D=20/W=80,000** (`test_originzc_winner_bin_her_wiring_d20.jl`,
`destination_sample=:exclude_row`, `K_mean=1/K_pair=1`, matching
`d20_meanzc_release_gates.jl`'s own Point A config): calibration + a near-delta=1 perturbed point.
<!-- RESULTS_PLACEHOLDER -->

## Runtime counters

Reuses `dense_cross_hessian_calls`/`winner_cross_hessian_calls`/`operator_cross_hessian_calls`
(`no_dense_g_counters.jl`), no new counter names. D=4 wiring gate run:
`dense_cross_hessian_calls=15`, `winner_cross_hessian_calls=19` (`=operator_cross_hessian_calls`).

## Default backend

<!-- DEFAULT_PLACEHOLDER -->

## Not done / left for a future session

- `H_RR` stays dense BLAS unconditionally, unchanged -- explicitly out of scope.
- Rectangular (`D != Ddest`) D=4 configurations were not separately exercised (same gap disclosed
  in Sections 2 and 4's own docs).
- `fg_backend=:operator` (the ZC-restriction-operator FG path, `port/shared-inner-fg-operator...`)
  was not exercised together with `zc_cross_hessian_backend=:winner_bin` in the same gate run --
  these are independent dispatch points (FG vs Hessian) with no code-level coupling found during
  this session, but the combination itself was not explicitly tested.
- `K_pair=0` (mean-only) was not separately gated for `:winner_bin`, same caveat as Section 4.
