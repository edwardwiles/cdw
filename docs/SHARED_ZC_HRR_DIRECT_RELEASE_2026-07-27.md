# Shared Direct H_ZZ (mean/pairwise-ZC self-Gram) Release — 2026-07-27

## What changed

CM+ZC's `HMM` block (mean/pair x mean/pair, task's `H_ZZ`) and origin-ZC's `HRR` block (this
family's ONLY restriction self-block, same math) both computed `(1/M) * EM'*EM` /
`(1/M) * HC_eta'*HC_eta` -- a small dense BLAS gemm, but sourced from `sqrt(S)`-weighted columns
read directly out of `obj.H` (`E[:, ncore+1:NCORE]` for CM+ZC, `H[:, 2+NCORE:1+n]` for origin-ZC).
This is architecturally fine by the prior winner-aware-H_ER phase's own reasoning (that column is
cheap, already-centered, and always populated regardless of backend) but this session's task
brief explicitly asks for `H_ZZ` to be computed directly from each family's own raw/reusable ZC
restriction feature state instead -- decoupling it from `obj.H`'s column layout/fill discipline
entirely, and (per the brief) doing so with ONE shared routine rather than two independent
per-family derivations.

**New shared primitive** (`zc_restriction_operator.jl`, the pre-existing shared home for
`ZCRestrictionOperator`/`ZCRestrictionWorkspace`, used by both families' `:operator` FG backend
already):

- `ZCCenteredScratch` -- persistent `(W, max_nx)` scratch: `Zc[w,j]` = the centered restriction
  feature value (`Φ[w,j] - t[j]`), recomputed directly from `op.Zraw_all`/`op.Zpairraw_all` (the
  campaign-lifetime raw feature matrices, theta-independent) and the CURRENT outer point's targets
  (`ws.targets_mean`/`targets_pair`, refreshed via the pre-existing `refresh_zc_targets!`) --
  NEVER read from `obj.H`. `ZcS[w,j] = S[w]*Zc[w,j]`, the current Hessian callback's weighted copy
  -- shared by this routine AND CM+ZC's new `H_CZ` primitive (see the companion doc,
  `CM_MEANZC_BLOCK_PARTITION_AND_HCZ_RELEASE_2026-07-27.md`), built once per callback, not twice.
- `refresh_zc_centered!(cs, op, ws, S)` -- fills `Zc`/`ZcS` for the current callback.
- `zc_restriction_gram!(HZZ, cs, op, M)` -- `HZZ = (1/M) Zc' diag(S) Zc`, a single small dense
  BLAS gemm (`n_restriction(op) x n_restriction(op)`, at most a few hundred wide for production
  `K_mean<=2/K_pair<=2` configs -- no attempt made to avoid this small dense op, per the task
  brief's own "a small dense BLAS calculation is fine" allowance).

**ONE routine, TWO call sites**: `_fill_cm_HEE!`'s widened `HMM` computation
(`cm_hessian_architectures.jl`) and `archA_partitioned_hess_cb_builder`'s `HRR` computation both
call `zc_restriction_gram!` directly -- no per-family reimplementation.

**Dedicated raw-feature state, kept separate from the FG operator's own state**: each family gained
NEW, dedicated fields (`hzz_zc_op`/`hzz_zc_layout`/`hzz_zc_ws` on `CMBinHessCtx`/
`OriginZCCoreHessCtx`), built ALWAYS (independent of `inner_fg_backend`/`fg_backend`) rather than
reusing the EXISTING `meanzc_zc_op`/`fg_zc_op` fields (which are gated on `inner_fg_backend===
:operator`/`fg_backend===:operator` and used elsewhere as an explicit "was this built with
:operator" prerequisite check -- e.g. `archOZ_verified_state`'s `:operator` verification branch
hard-errors if `octx.fg_zc_op === nothing`; repurposing that field for the Hessian's own
always-needed use would have silently defeated that check). The current outer point's ν is
published into a new `nu_ref` field by `archC_meanzc_base_state`/`_verified_state`
(`cm_meanzc_production.jl`) and `archOZ_base_state`/`_verified_state`
(`cm_originzc_production.jl`), mirroring the pre-existing `core_cf_ref` "wrapper publishes, Hessian
callback reads" pattern -- both functions already receive ν directly as an argument, so this is a
one-line addition at each of the four call sites.

## Backend flag reused, not duplicated

Both dispatch points reuse the EXISTING `zc_cross_hessian_backend` field (`CMBinHessCtx`/
`OriginZCCoreHessCtx`) -- when `=== :winner_bin`, the Hessian callback now ALSO routes `H_ZZ`
through the new direct routine, in addition to the pre-existing `H_EZ`/`H_ER` winner-bin routing
that flag already controlled. No new backend-selection flag was introduced (the task brief's own
"don't invent new counter/flag names unless the existing ones genuinely don't fit" guidance) --
`CM_MEANZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT`/`ORIGINZC_ZC_CROSS_HESSIAN_BACKEND_DEFAULT` were
ALREADY `:winner_bin` in production (flipped by the prior winner-aware-H_ER phase), so this
session's `H_ZZ` work is already live at its production default the moment it lands, with no
separate flip step.

## Gates (D=4 and real D=20, both PASS)

D=4/D=20 correctness for `H_ZZ` was exercised via the SAME gate scripts used for `H_CZ` (CM+ZC:
`test_cm_meanzc_hcz_hzz_direct_d4.jl`/`_d20.jl`, comparing the COMPLETE packed Hessian, which
includes `HMM`) PLUS the pre-existing `zc_cross_hessian_backend=:winner_bin` wiring gates, RE-RUN
this session without modification (they now ALSO exercise the new direct `H_ZZ` routine, since it
shares the same backend flag):

- **CM+ZC D=4** (`test_cm_meanzc_winner_bin_hez_wiring_d4.jl`, re-run unmodified): K1/K2 x
  anchored/orthonormal, ALL PASS, max\|ΔH\| in `[1.4e-15, 4.5e-13]` (unchanged from the prior
  phase's own numbers -- confirms no regression).
- **CM+ZC D=4, dedicated H_ZZ+H_CZ gate** (`test_cm_meanzc_hcz_hzz_direct_d4.jl`): see the
  companion doc for the full table -- the "Z-rows sub-block" check in that gate isolates
  `H_EZ`/`H_CZ`/`H_ZZ` specifically and passed at the same machine-precision scale.
- **Origin-ZC D=4** (`test_originzc_winner_bin_her_wiring_d4.jl`, re-run unmodified): K1/K2, ALL
  PASS, max\|ΔH\| in `[1.6e-15, 9.1e-13]`.
- **Origin-ZC real D=20/W=80,000** (`test_originzc_winner_bin_her_wiring_d20.jl`, re-run
  unmodified, `destination_sample=:exclude_row`, `K_mean=1/K_pair=1`): ALL PASS.
  `calib`: max\|ΔH\|=5.329e-13 (scale 3.970e+03). `near_delta1_perturbed`: max\|ΔH\|=2.039e-12
  (scale 4.618e+03). Complete inner solve status matches (`nStatus=0` both backends) and dual point
  agrees to `<1e-16`. Warm `archA_partitioned_hess_cb_builder` (direct `H_ZZ`) allocates ~11.4 MB/
  call, `zc_cross_scratch` object identity stable across calls.
- **CM+ZC real D=20/W=80,000** (`test_cm_meanzc_hcz_hzz_direct_d20.jl`): see companion doc
  (`CM_MEANZC_BLOCK_PARTITION_AND_HCZ_RELEASE_2026-07-27.md`) for the full 8-row table -- max|ΔH|
  in [6.395e-13, 9.024e-13] across both contrasts, both points, both architectures, ALL PASS.

## K_pair=0 smoke (both families, real D=20)

- **Origin-ZC** (`test_originzc_hzz_kpair0_smoke_d20.jl`): `hzz_zc_op` built, inner solve feasible
  (`status=0`), Hessian call completed with no NaN/Inf. PASS.
- **CM+ZC**: folded into `test_cm_meanzc_hcz_hzz_direct_d20.jl`'s own final section (contrasts=
  :orthonormal): `hzz_zc_op` built, inner solve feasible (`status=0`), Hessian call completed with
  no NaN/Inf. PASS.

## Counters

Both families' production defaults route `H_ZZ` through the same `zc_restriction_gram!` — CM+ZC's
isolated production-only invariant check (no dense-reference call anywhere in the process) confirms
`dense_cross_hessian_calls=0` for the combined `H_CZ`+`H_ZZ` production path (full snapshot in the
companion doc's "Production-only invariant check" section). Origin-ZC's own `H_ZZ` reuses the
IDENTICAL routine and backend flag (already flipped to `:winner_bin` by the prior winner-aware-H_ER
phase), so the same zero-dense-cross-hessian-calls invariant applies by construction, not by a
separate re-derivation.

## Not done / left for a future session

- `H_CC`/origin-ZC's `H_EE` self-block computations are untouched, out of scope.
- The small `zc_restriction_gram!` gemm was not benchmarked for speed (task brief: "don't try to
  make it allocation-free or multi-threaded" -- correctness/decoupling from `obj.H`, not raw FLOPs,
  was the goal here, exactly mirroring the prior phase's own `H_EZ`/`H_ER` framing).
