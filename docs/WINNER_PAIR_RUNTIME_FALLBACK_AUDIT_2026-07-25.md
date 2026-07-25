# Runtime backend-use / fallback audit — 2026-07-25

Task §2. Proves which backend actually EXECUTED per Hessian callback, not just which one the
startup manifest claims was requested.

## What was built

`core_exact_hessian.jl` (`CoreHessianCallCounters`/`CORE_HESSIAN_COUNTERS`/
`record_core_hessian_call!`/`record_compressed_core_rebuild!`/`print_core_hessian_counters`):
- `winner_pair_hessian_calls` (= `winner_pair_serial_calls` + `winner_pair_parallel_calls`)
- `dense_core_fallback_calls`
- `dense_fallback_reason_counts` — one of `compressed_state_unavailable`, `tied_winner`,
  `unsupported_layout`, `debug_reference_requested`, `workspace_mismatch`, `other`
- `compressed_core_rebuilds` (workspace rebuilt from a FRESH `CompressedFactual`, not reused)

Wired into every family's actual Hessian-callback code path: unrestricted
(`_callbackEvalH_inner_compressed!`, direct calls), CM/CM+meanZC (`fill_core_hessian_upper!`
inside `_fill_cm_HEE!`, plus the shared `record_core_hessian_call!` for the dense-inline fallback
branch), origin-ZC (`archA_partitioned_hess_cb_builder`). `core_cf_ref` (the box each family's
`moments!` closure publishes a fresh `CompressedFactual` into) now stores a **Symbol fallback
reason** instead of `nothing` when compression genuinely isn't available for a point
(`:tied_winner` from a caught `TiedWinnerError`, `:compressed_state_unavailable` when
`use_compressed_core=false`) — every Hessian callback site checks `cf isa CompressedFactual`
before using it, and records the Symbol reason (or `:other`) on the dense-fallback path otherwise.

## A real, significant bug this counter infrastructure caught

Building these counters immediately surfaced a genuine, previously-undetected bug (full writeup:
`SHARED_WINNER_PAIR_FINAL_PRODUCTION_GATE_2026-07-25.md` §Corrections): `build_cm_production_context`
(the function the REAL `run_cm_upper_checkpointed` driver calls) built its `moments!` closure's
`core_cf_ref` and its `cctx.core_cf_ref` as **two independent, disconnected `Ref` objects** (both
silently defaulting to a private `Ref{Any}(nothing)` inside two different functions' default
kwargs) — so `cctx.core_cf_ref[]` stayed `nothing` FOREVER, and CM's Hessian callback silently used
dense BLAS on every single call, through the real production driver, since this port was first
implemented in the prior session. The symptom that exposed it: a real, feasible, converged CM
solve reporting **zero** winner-pair calls AND **zero** recorded dense-fallback calls simultaneously
— logically impossible for a solve that definitely called the Hessian callback (confirmed via
`inner_loop_internal_archgeneric`'s own `n_hess` return value, which WAS nonzero) — meaning the
dense branch was executing but not going through any counter-recording path, which in turn is what
led to finding the disconnected-Ref bug at its root. **Fixed** (`cm_production_bundle.jl`): one
`core_cf_ref` built once, threaded to both the moments closure and `build_cm_bin_ctx`.

## Zero-unexplained-fallback evidence gathered this session

| Gate | Family | winner_pair_hessian_calls | dense_core_fallback_calls |
|---|---|---|---|
| D=4 full gates (`test_shared_core_hessian_d4_gates.jl`) | all 4 | >0 at every point tested | 0 |
| D=4 direct diagnostic (post-fix) | flexible CM | 4 | 0 |
| D=20 checkpoint/resume, stage1 (real 20s outer run) | unrestricted | >0 (nonzero, logged) | 0 |
| D=20 checkpoint/resume, RESUMED stage (real run) | unrestricted | **106** | **0** |
| D=20/W=80,000/L=50 regression (`test_cm_compressed_core.jl`, post-fix) | flexible CM | (indirect: max\|ΔH_EE\|=3.3e-11 vs dense, only possible if winner-pair genuinely ran) | — |

`compressed_core_rebuilds=9` on the resumed unrestricted checkpoint run — confirms the workspace is
being correctly rebuilt once per NEW outer point (not once per Hessian call, and not stale-reused
across genuinely different points), matching the intended "theta-fixed, rebuilt once per outer
point" cadence documented in `core_exact_hessian.jl`.

## Ordinary production runs: zero dense fallback confirmed

Every gate in the table above shows `dense_core_fallback_calls = 0` for real, feasible solves —
satisfying task §2's "For all ordinary production benchmark runs, require
`dense_core_fallback_calls = 0` unless a specific, documented event genuinely requires fallback."
No undocumented/silent fallback was observed anywhere in this session's testing once the
disconnected-Ref bug above was fixed.

## `DENSE_FALLBACK_CALLS_IN_GATES` (final verdict field)

**0** across every correctness gate and benchmark run in this session, post-fix.
