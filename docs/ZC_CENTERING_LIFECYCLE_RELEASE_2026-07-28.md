# ZC centering (Zc) lifecycle release — 2026-07-28

Addresses the `CROSS_HESSIAN_PRECOMPUTATION_LIFECYCLE_AUDIT_2026-07-28.md`'s own
`HIGHEST_PRIORITY_REMAINING_GAP`: `refresh_zc_centered!` rebuilt `Zc` (the centered mean/pair ZC
restriction matrix, `Zc[w,j] = Φ[w,j] - t[j]`) on every Hessian callback even though `Zc` depends
only on the current outer point's fixed targets `t` (via `refresh_zc_targets!`), never on the
dual-dynamic `S = Ψ''(r)` — the audit flagged this as recomputed at dual-callback frequency when it
is actually only outer-point-static, and should be rebuilt once per inner solve, not once per
Hessian callback within that solve. This document describes the opt-in fix, the D=4 validation
evidence, and one correction this release makes to the audit doc's own wording.

## Correction to the lifecycle audit doc: call frequency vs value stability

The audit doc states: *"`refresh_zc_targets!` ... called once per inner solve by
`_fill_cm_HEE!`/`archA_partitioned_hess_cb_builder`"*. Direct code reading while building this
release shows this is **not accurate as a statement about call frequency**:
`refresh_zc_targets!(cctx.hzz_zc_ws, op, cctx.hzz_zc_layout, cctx.nu_ref[])` sits inside
`_fill_cm_HEE!`'s `winner_bin_ok && zc_direct_ready` branch
(`cm_hessian_architectures.jl:809`), and `_fill_cm_HEE!` is called unconditionally from
`hessian_cm_structured_v2!` (`cm_hessian_threaded.jl:202`) — the actual per-Hessian-callback
function. So `refresh_zc_targets!` fires on **every** Hessian callback, exactly like
`refresh_zc_centered!` does. Confirmed empirically for origin-ZC too:
`archA_partitioned_hess_cb_builder`'s single callback body calls `refresh_zc_targets!` (and
`refresh_zc_centered!`) **twice** per callback — once for the HER block, once for the HRR block
(`cm_hessian_architectures.jl:1581` and `:1629`) — a second, independent redundancy this release's
own D=4 gate quantifies directly (see "Rebuild-count evidence" below).

What *is* true, and is the actual basis for this release's cache: the **values**
`refresh_zc_targets!` computes are outer-point-static (idempotent — calling it twice in a row with
the same `νfull` produces the same `targets_mean`/`targets_pair`), because `ν` is fixed for the
whole inner solve. The cache below is keyed on that value-stability, not on the (incorrect) claim
about call frequency.

## Design

`ZCRestrictionWorkspace` (`zc_restriction_operator.jl`) gains two new fields:

- `gen::Int` — a generation counter. Bumped by `refresh_zc_targets!` **only when the `νfull`
  argument's object identity changes** (`ws.last_nu !== νfull`), not on every call. This is safe
  and correct because the only two production callers (`archC_meanzc_base_state`,
  `archOZ_base_state`) do `cctx.nu_ref[] = collect(νvec)` — allocating a **fresh** vector object —
  exactly once per inner solve (before the KNITRO solve starts), and never touch `nu_ref[]` again
  until the next inner solve. This mirrors `core_ws_for !== cf`'s exact "rebuild only when the
  outer-point identity changed" idiom (`cm_hessian_architectures.jl:774`) with an integer
  generation counter instead of direct object-identity comparison on the cache itself (needed here
  because `refresh_zc_targets!` mutates `targets_mean`/`targets_pair` in place rather than
  replacing them wholesale, so there is no fresh *target* object to compare against — but there is
  a fresh *source* (`νfull`) object to key on).
- `last_nu::Union{Nothing,AbstractVector{Float64}}` — the `νfull` object `gen` was last bumped for.

`ZCCenteredScratch` gains one new field:

- `built_gen::Int` (default `-1`, meaning never built) — the `ws.gen` value `cs.Zc` was last built
  against.

`refresh_zc_centered!` gains one new keyword:

- `cache_across_callbacks::Bool = ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]` — **opt-in**, the global
  `Ref{Bool}` defaults to `false` (today's unchanged always-rebuild-every-callback behavior), per
  this codebase's universal convention. When `true` and `cs.built_gen == ws.gen`, the `Zc` rebuild
  pass is skipped entirely (existing `cs.Zc` contents are reused verbatim — correct, since the
  inputs that produced them have not changed). `cs.ZcS` (the `S`-weighted copy, genuinely
  dual-dynamic) is **completely unaffected** by this flag: it is always refreshed from the current
  `cs.Zc` whenever `fill_S=true`, exactly as before this release.

Two new runtime counters (`no_dense_g_counters.jl`): `zc_centered_rebuilds` /
`zc_centered_cache_hits`, incremented by `refresh_zc_centered!` on the rebuild/skip branches
respectively, surfaced through the existing `no_dense_g_report()`.

**No existing struct's positional-constructor call sites were put at risk.** All three affected
constructors (`ZCRestrictionWorkspace(op)`, `ZCCenteredScratch(W, max_nx)`) are only ever
constructed through their own explicit convenience constructors across the whole repo (confirmed by
`grep -rn "ZCRestrictionWorkspace(\|ZCCenteredScratch("` — every call site uses the 1- or 2-arg
wrapper, never the raw memberwise constructor) — this was checked deliberately, having just found
and fixed a real field-order-vs-constructor-arg bug in a sibling struct
(`WinnerBinCrossScratch`, see below) added by an earlier phase of this same task.

No change was made to any default. `ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]` defaults to `false`;
flipping it to `true` is left to a future session after real-D=20 validation (out of this task's
D=4 scope).

## D=4 validation

New gate: `full_aod_diag/d4_exact/test_zc_centered_cache_d4.jl`. Run per family to avoid the
documented cross-family KNITRO flakiness:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PATH="$HOME/.juliaup/bin:$PATH" \
  ONLY_FAMILY=cm_meanzc julia --project=. -t 4 full_aod_diag/d4_exact/test_zc_centered_cache_d4.jl
ONLY_FAMILY=origin_zc julia --project=. -t 4 full_aod_diag/d4_exact/test_zc_centered_cache_d4.jl
```

### Bit-exact equivalence

For `cm_meanzc` (`K_mean1_pair0`/`K1`/`K2`) and `origin_zc` (`K_mean1_pair0`/`K1` — `K2` is the
pre-existing infeasible config, see below), at 3 distinct simulated dual points each: the complete
packed Hessian computed with `cache_across_callbacks=false` vs `=true` is **bit-exact identical**
(`maxdiff=0.0` in every case, 15/15 checks) — confirming this is a pure caching/timing change, not
an algebra change.

### Rebuild-count evidence ("centered-Z rebuilds per Hessian callback = 0")

Five simulated Hessian callbacks (distinct dual points) fired within **one** inner solve (same
`nu_ref` object throughout, established immediately before each measurement via a fresh
`archC_meanzc_base_state`/`archOZ_base_state` call so no cache state leaks in from a prior
sub-test):

| family | config | `cache_across_callbacks` | `zc_centered_rebuilds` | `zc_centered_cache_hits` |
|---|---|---|---|---|
| cm_meanzc | K_mean1_pair0 | false | 5 | 0 |
| cm_meanzc | K_mean1_pair0 | true | **1** | 4 |
| cm_meanzc | K1 | false | 5 | 0 |
| cm_meanzc | K1 | true | **1** | 4 |
| cm_meanzc | K2 | false | 5 | 0 |
| cm_meanzc | K2 | true | **1** | 4 |
| origin_zc | K_mean1_pair0 | false | 10 (2×5, see below) | — |
| origin_zc | K_mean1_pair0 | true | **1** | — |
| origin_zc | K1 | false | 10 (2×5) | — |
| origin_zc | K1 | true | **1** | — |

`cache=false` reproduces exactly the documented always-rebuild baseline (`rebuilds ==
n_callbacks` for cm_meanzc; `== 2*n_callbacks` for origin_zc, since that family's own callback
calls `refresh_zc_centered!` twice per callback — once for HER, once for HRR — an independent,
pre-existing redundancy this release did not introduce or change, just quantified). `cache=true`
rebuilds **exactly once** across all 5 callbacks regardless of family — i.e. after the first
callback, `centered-Z rebuilds per Hessian callback = 0`, confirmed by direct counter assertion
(20/20 checks pass across both families), not merely inferred from Hessian-output equality.

A companion check confirms a genuinely new outer point (fresh `nu_ref` object, identical values)
correctly forces a fresh rebuild even with `cache_across_callbacks=true` (1/1 pass) — the cache
does not "stick" past the outer point it was built for.

## Relationship to the WinnerBinCrossScratch bugfix (this same session)

While root-causing the pre-existing `cm_meanzc` D=4 gate failure (`nStatus=-500`, unrelated to this
Zc-caching work — see `PRODUCTION_ZC_CM_CROSS_HESSIAN_ALGEBRA_2026-07-28.md`'s own notes and the
git log for the dedicated fix commit), a genuine bug was found in `WinnerBinCrossScratch`'s 3-arg
constructor (`winner_pair_cross_hessian.jl`): the struct's last two fields
(`tasks_ec::Vector{Task}` then `EsumEcon::Vector{Float64}`) were populated in the OPPOSITE order by
the constructor. That fix is unrelated to the Zc-caching change described in this document but was
made in the same session/branch — see the dedicated commit
("Fix WinnerBinCrossScratch constructor arg-order bug...") for the full writeup. It is mentioned
here only because it directly informed this release's own extra caution around NOT repeating that
exact class of bug when adding fields to `ZCRestrictionWorkspace`/`ZCCenteredScratch` (see "Design"
above).

## Status

- Implementation: done, opt-in, default unchanged (`ZC_CENTERED_CACHE_ACROSS_CALLBACKS[] = false`).
- D=4 bit-exact validation: done, both ZC families, PASS.
- D=4 rebuild-count validation: done, both ZC families, PASS (`centered-Z rebuilds per Hessian
  callback = 0` confirmed under the new opt-in mode).
- Real D=20 validation: **not done** — out of this task's D=4 scope, left for a future session
  before flipping `ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]`'s default or wiring it into production
  call sites' own kwarg defaults.
- Production wiring (flipping the default, or threading an explicit kwarg through
  `build_cm_meanzc_bin_ctx`/`build_originzc_production_context`'s own signatures): **not done**,
  deliberately out of scope per the task's "do NOT change any default backend/behavior" instruction.
