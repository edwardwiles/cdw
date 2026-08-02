# Production Hessian Type-Stability / Dynamic-Dispatch Audit (2026-08-02)

Scope: `full_aod_diag/` and `campaign_inputs/` at production HEAD `21fa6ec`, across all 5 families.
Methodology: repo-wide `rg` sweep for every concurrency/dispatch primitive named in the task brief,
followed by manual inspection of every hit's consuming code. `@code_warntype`/`code_typed` spot
checks on the 5 hess_cb closures are recorded in the harness build log (section below); a full
per-block `@code_warntype` pass on every sub-block function is deferred (see "not yet done").

## 1. `fetch(::Task)` audit (the task brief's specifically-named historical bug class)

```
rg -n "fetch\(" full_aod_diag campaign_inputs
```

**12 hits total, all in `full_aod_diag/d4_exact/`** (zero in `campaign_inputs/`):

| file:line | binds return value? | consumer |
|---|---|---|
| `core_exact_hessian.jl:758` | **YES** | `Ss, t0v, s0v = fetch(tasks[wk])::Tuple{Float64,Float64,Float64}` |
| `core_exact_hessian.jl:776` | no (barrier) | discarded |
| `zc_gram_blas_candidates.jl:218` | no (barrier) | discarded |
| `threaded_cross_hessian.jl:147,169,260,327,388` | no (barrier) | discarded |
| `hez_drawmajor_candidate_2026-08-01.jl:130` | no (barrier) | discarded |
| `hez_drawmajor_v2_candidate_2026-08-01.jl:117` | no (barrier) | discarded |
| `hcz_reordered_candidate_2026-08-01.jl:62` | no (barrier) | discarded |
| `hcz_drawchunk_candidate_2026-07-29.jl:78` | no (barrier) | discarded |

`core_exact_hessian.jl:758` is the **historical site** the task brief describes ("prior production
code had a real issue where `fetch(::Task)` returned `Any`, causing boxed scalar accumulation") —
confirmed via `git log`: fixed by commit `377f48e "Fix fetch(::Task) type-instability in
hessian_core_winner_pair!"`, an ancestor of the current HEAD. It is now explicitly
`::Tuple{Float64,Float64,Float64}`-annotated at the call site, eliminating the `Any`-typed boxed
accumulation the brief describes.

`rg -n "= fetch\("` (assignment-form only, repo-wide) confirms this is the **only** site anywhere
in `full_aod_diag`/`campaign_inputs` that binds a `fetch()` return value to a variable consumed
downstream. Every other `fetch()` call is a pure synchronization barrier: the spawned
`Threads.@spawn` task writes into a pre-allocated, closed-over shared array as its side effect, and
the `Any`-typed return of the bare `fetch()` statement is never read. A barrier `fetch()` returning
`Any` costs nothing at runtime beyond the (already-required) task-completion wait — there is no
boxed value to propagate.

**Verdict: this bug class is already fully remediated at production HEAD.** No further fetch/Task
type-stability fix is needed or proposed by this audit.

## 2. `Task{...}`/`@async` audit

```
rg -n "Task\{|@spawn|Threads\.@spawn|@async" full_aod_diag campaign_inputs
```

- **Zero** bare `Task{...}` type annotations anywhere (nothing to widen/narrow).
- **Zero** `@async` usage anywhere — all concurrency in this codebase is `Threads.@spawn` (CPU-bound
  parallel work, correctly not `@async`, which is for I/O-bound cooperative concurrency).
- Every `Threads.@spawn` call site (`core_exact_hessian.jl:743,764`; `zc_gram_blas_candidates.jl:203`;
  `threaded_cross_hessian.jl:132,154,242,315,373`; `hez_drawmajor_v2_candidate_2026-08-01.jl:86`;
  `hez_drawmajor_candidate_2026-08-01.jl:112`; `hcz_reordered_candidate_2026-08-01.jl:45`;
  `hcz_drawchunk_candidate_2026-07-29.jl:66`) assigns into a **persistent, pre-sized `tasks::Vector`**
  field on the enclosing workspace struct (built once at context-construction time, reused across
  every subsequent callback invocation) — never a fresh `Vector{Task}` allocated per call. This
  matches `winner_pair_cross_hessian.jl`'s own documented "persistent `Threads.@spawn` task buffer"
  convention. No per-callback `Vector{Any}`/`Vector{Task}` allocation from this pattern.

## 3. Backend-dispatch mechanism (Symbol-based, inside hot loops)

Every family's backend selection (`core_hessian_backend`, `zc_gram_backend`, `zc_ez_backend`,
`hcz_prep_backend`, `cm_cross_hessian_backend`, ...) is a `Symbol` field read via `if`/`===`
comparisons **once per callback invocation**, at the top of the callback closure, to choose which
concrete kernel function to call — not inside any W-scale inner loop. This is standard "dispatch
once, then run a type-stable kernel" structure, not a hot-loop dynamic-dispatch cost. Confirmed by
inspection of `archC_hess_cb_builder` (`cm_hessian_architectures.jl:1810-1850`),
`archA_partitioned_hess_cb_builder` (`cm_hessian_architectures.jl:1618` on), `zc_gram_dispatch!`
(`zc_gram_blas_candidates.jl:272`), `hcz_prep_dispatch!` (`hcz_drawchunk_candidate_2026-07-29.jl`).

## 4. `@code_warntype` spot checks on the 5 hess_cb closures (frozen-state, W=20,000)

Run against the closures the canonical harness (`production_all_hessian_audit_harness_2026-08-02.jl`)
already builds and validates for all 5 families. **Status: smoke-run only (5 closures, single
invocation each) — a full per-sub-block pass (§8's "targeted inference checks on every hot
production entry point," meaning each `_fill_cm_HEE!`/`hessian_core_winner_pair!`/
`winner_pair_cross_hessian_zc_block_drawmajor_v2!`/etc. individually) is NOT YET DONE; queued as
the next audit step, to run once the W=100,000 canonical harness pass (in progress) completes.**

## 5. Other §8-named patterns -- preliminary read, not yet exhaustively verified

- **Abstractly typed context fields**: `CMBinHessCtx`/`OriginZCCoreHessCtx` fields are concretely
  typed per prior sessions' own remediation (see memory `feedback-abstract-function-args-not-a-perf-issue`
  -- this codebase's prior audits already distinguish "abstract function *argument* type" (fine,
  idiomatic, not a perf issue) from "abstract *struct field* type" (a real perf issue) and fixed the
  latter class in earlier tasks). Not independently re-verified line-by-line in this pass.
- **Union-typed optional workspaces**: `cctx.tls` (thread-local scratch), `cctx.frechet_ext_cache`
  are `Union{Nothing,T}`-typed by design (lazy first-build). This is a deliberate, standard Julia
  pattern (small `Union` — 2 concrete types — compiles to an efficient dispatch, not boxing) rather
  than a defect; not itself a finding.
- **Generator expressions / views / SubArrays / heterogeneous tuples / function fields / Any-valued
  Dicts inside the Hessian callback hot path**: not yet swept individually. Deferred.

## Interim verdict (subject to the full per-block pass above)

```
TYPE_STABILITY = pass (fetch/Task class) | pending_full_sweep (remaining §8 categories)
```

The ONE historically-documented type-instability bug class named in the task brief
(`fetch(::Task)` → `Any` → boxed scalar accumulation) is confirmed already fixed at production HEAD,
verified by direct inspection of every fetch() call site in the audited directories, not merely by
the presence of a fix commit in git log. No NEW instance of that bug class was found. The broader
§8 checklist (per-block `@code_warntype`, generator expressions, SubArrays, heterogeneous tuples)
has not yet been exhaustively swept and is queued as follow-up work within this same audit branch.
