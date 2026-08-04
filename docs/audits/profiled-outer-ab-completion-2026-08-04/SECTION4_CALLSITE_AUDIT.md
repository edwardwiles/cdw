# Section 4 — required-argument call-site audit

Task: prove no silent defaults remain for `threaded`/`threaded_gradient`/`validity_radius`, every
public caller passes them explicitly, the argument has one consistent meaning at every layer, and
FULL production behavior is unchanged.

## Method

`git diff --stat 395dec3..4964966` (prior session's start commit through this continuation's doc
commit) gives the exact 30-file set touched since these kwargs were introduced. Grepped every
function definition taking `threaded::Bool`/`validity_radius::Float64` in that set, then grepped
every call site of each such function across the whole repo (not just the touched-file set, to
catch any older caller not in the diff).

## Definitions found (all required, no default)

| Function | File | Signature |
|---|---|---|
| `shared_family_outer_gradient` | `profiled_shared_economic_gradient_engine_2026-08-01.jl:150` | `(...; threaded::Bool)` |
| `instrumented_shared_family_outer_gradient` | `profiled_matched_gradient_instrumentation_2026-08-04.jl:124` | `(...; threaded::Bool)` |
| `instrumented_composite_gradient_at_fast` | `profiled_matched_gradient_instrumentation_2026-08-04.jl:211` | `(...; threaded::Bool)` |
| `reduced_originzc_outer_gradient_with_eta` | `profiled_zc_free_eta_2026-08-04.jl:146` | `(...; threaded::Bool)` (passes through to `shared_family_outer_gradient`) |
| `reduced_cmzc_outer_gradient_with_eta` | `profiled_zc_free_eta_2026-08-04.jl:202` | `(...; threaded::Bool)` (same pass-through) |
| `ReducedBandwidthCache` constructor | `profiled_reduced_bandwidth_cache_2026-08-04.jl:63` | `(validity_radius::Float64)` |

## Call sites (exhaustive, production + test)

Every call site found (34 total across production drivers and tests — `bin/run_profiled_model.jl`,
`profiled_production_outer_constrained_2026-08-02.jl`, `profiled_production_outer_runner_2026-08-01.jl`,
`profiled_zc_free_nu_production_driver_2026-08-04.jl`, and all `test_*.jl` files in
`full_aod_diag/d4_exact/`) passes `threaded = <value>` or `validity_radius = <value>` explicitly.
No call site relies on a default — because none of these functions define one, an omitted argument
would be a hard `UndefKeywordError`/`MethodError`, not a silent substitution.

`bin/run_profiled_model.jl`'s `--threaded-gradient` CLI flag is in the required-arguments list
(`parse_cli`'s own `for req in (..., "threaded-gradient")` loop, line 77) and is value-validated
(`"true"`/`"false"` only, line 84-85) before being threaded through as `threaded_gradient::Bool` —
a plain required positional/kwarg at every downstream layer, never re-defaulted.

## Consistent meaning

`threaded` means exactly one thing everywhere it appears in this call graph: whether
`shared_family_outer_gradient`'s own coordinate loop runs its per-coordinate finite-difference/
analytic sub-evaluations across `Threads` or serially. It is never overloaded to mean anything else
(e.g. FULL's separate, pre-existing `cross_hessian_threaded`/Hessian-parity `threaded` kwargs in
`cm_hessian_architectures.jl`/`c8_gammabranch_core.jl`/`composite_gradient_fast.jl` etc. are a
different, older, unrelated flag on a different code path — Hessian computation, not this session's
outer-gradient engine — and were correctly left untouched, confirmed by direct read; conflating the
two would be the actual bug this audit was checking for, and it is not present).

## FULL production unchanged

FULL's own production gradient (`cm_production_gradient_cplus`/`cm_meanzc_production_gradient_cplus`/
`cm_frechet_production_gradient_cplus`, called from `cm_checkpoint.jl:1344-1365`) is a structurally
different function, not touched by this session's `threaded`/`threaded_gradient` additions — it
remains unconditionally `threaded=true, h_mode=:cached` as before. Confirmed by grep: none of the
30 files in the session diff include `cm_checkpoint.jl`.

## Result

No remaining silent defaults. PASS.
