# Shared Economic FG — Final Port Status — 2026-07-26/27

`economic_forward!`/`economic_transpose!` (`economic_operator.jl`, unchanged this session — a thin
wrapper around the already-validated unrestricted-family `compressed_dual_contraction!`/
`compressed_transpose_contraction!`) is now consumed by **all four restricted families**' own FG
callables, not just origin-ZC/CM+ZC (the prior session's scope):

| Family | Struct | Consumed since |
|---|---|---|
| flexible_cm | `CMLookupState` | THIS session (`cm_lookup_kernels.jl`) |
| common_frechet | `CMFrechetLookupState{O}` | THIS session (`cm_frechet_lookup_kernels.jl`) |
| cm_plus_zc | `CMMeanZCOperatorState` | prior session |
| zc_only | `OriginZCOperatorState` | prior session |

Every consumer follows the IDENTICAL pattern (deliberately, per the addendum's "one shared
implementation, not family copies" requirement): `cf = core_cf_ref[]`; if `cf isa
CompressedFactual`, lazily build/cache an `EconomicFGWorkspace` keyed on object identity, call
`economic_forward!`/`economic_transpose!`; else fall back to the pre-existing dense `obj.H`
`BLAS.gemv!` and increment `record_dense_economic_G!()`. The fallback exists for the tied-winner /
compressed-state-unavailable case and for the 50+ pre-existing ad hoc test/benchmark scripts that
construct these structs without wiring a real `core_cf_ref` — it is NEVER taken in the real
production entry points once `core_cf_ref` is threaded through (confirmed by
`n_dense_econ_fallback == 0` in every gate this session ran with a real production context).

No new economic-block math was written this session — `economic_forward!`/`economic_transpose!`
themselves are byte-for-byte unchanged from the prior session's Addendum Part A implementation.
This document's only claim is about CONSUMPTION: 4/5 families now share one implementation instead
of flexible-CM/common-Frechet each re-deriving their own dense BLAS call.

See `FIVE_FAMILY_OPERATOR_STACK_COMPLETION_2026-07-26.md` for the correctness/performance evidence
per family, and the individual commits (`5cdd4bb`, `ae0bcd5`) for the exact diffs.
