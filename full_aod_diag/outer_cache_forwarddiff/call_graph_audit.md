# Outer-loop callback / caching audit

Isolated, additive diagnostics (`full_aod_diag/outer_cache_forwarddiff/`). No file under
`cc_algo/` was modified. This audit is a different question from
`full_aod_diag/ad_benchmark/` (which asks "which AD backend computes the divergence-constraint
gradient fastest, once we're inside one inner+gradient evaluation"): this one asks "does the
outer KNITRO solve invoke that expensive inner CC solve MORE TIMES than there are distinct θ
points, and can that be fixed without touching the model?"

## Normalization/parameter-count correction (carried over from the conversation)

The initiating prompt for this audit assumed the old `A[1,d]=1` normalization and a
`D*(D-1)` free-bilateral-A-parameter count. The actual, current configuration (this session,
`γ_d≡1`-for-all-d + direct-γ' objective, see `../SESSION_SUMMARY_2026-07-12.md`) has:
- θ layout: `[μ, σ, γ_θ(D, now PINNED/inert), γ'_focal, A_od(D×D, fully free)]`.
- Free bilateral A parameters = **D²** (16 at D=4, not 12; 400 at D=20, not 380).
- Free outer parameters = **D² (A_od) + 1 (γ'_focal) + 1 (μ, when free)** = 18 at D=4 — confirmed
  numerically in this audit's own driver (`audit_outer_callbacks.jl` prints
  `free_outer_params=18`), not "12 bilateral A + ~6 other" as the initiating prompt assumed. The
  *count* ≈18 happens to be close to the prompt's guess; the *composition* is not.

## 1. Which KNITRO callbacks are registered?

`cc_algo/outer_loop_functions.jl::outer_loop` branches on `KN_get_int_param(kc, "eval_fcga")`:
- `eval_fcga == 1`: ONE combined callback, `callbackEval_and_ConsFG_outer!` (objective + gradient
  + constraints + Jacobian all in one KNITRO callback).
- `eval_fcga != 1`: TWO callbacks — `callbackEval_and_ConsF_outer!` (objective + constraint
  values only) registered via `KN_add_eval_callback`, and `callbackEval_and_ConsG_outer!`
  (gradient + Jacobian only) registered separately via `KN_set_cb_grad`.

Both callback shapes call `inner_loop_internal(obj, θ)` (`cc_algo/inner_loop_functions.jl`) —
the actual KNITRO-based inner CC solve — **unconditionally, every single call**, with **no
memoization of any kind** at the production-code level. `inner_loop_internal` has no notion of
"have I already solved this θ."

## 2. Are function values and gradients requested together or separately? — depends on the `.opt` file, and production uses the SEPARATE path

This is the crux of the audit, and it was not previously tested in this session.

- Every earlier full_aod_diag diagnostic this session (`compare_directgp.jl`,
  `ad_benchmark/*`, `run_fullA_D10*.jl`) uses `full_aod_diag/csw_outer*.opt`, which sets
  **`eval_fcga yes`** — the COMBINED callback path. Under that path, `inner_loop_internal` is
  called exactly once per KNITRO function/constraint evaluation, confirmed by the pre-existing
  instrumentation in `outer_loop_functions.jl`/`inner_loop_functions.jl` (`INNER_SOLVE_COUNT`):
  every "OUTER_SOLVE" log line this session has `inner_solves == outer_FCevals` exactly.
- **Every actual production driver** — `cc_algo/ccOuter.jl` (6 call sites, the main entry
  point), `sequential_gravity/run_sequential.jl`, `lfd/LFD.jl`, `prepare_cc/PMM.jl` — hardcodes
  `outer_loop_opt = "ek_outer_loop_options.opt"`, which sets **`eval_fcga no`** — the SEPARATE
  callback path. **None of this session's prior timing/comparison work exercised this
  configuration.**

## 3-5. Does the objective/constraint/gradient callback solve the inner CC problem?

Yes to all three, unconditionally, via `inner_loop_internal`. Under `eval_fcga=no` (production's
actual default), a single accepted outer iterate at θ* triggers:
1. `callbackEval_and_ConsF_outer!(θ*)` → `inner_loop_internal(obj, θ*)` (solve #1)
2. `callbackEval_and_ConsG_outer!(θ*)` → `inner_loop_internal(obj, θ*)` (solve #2, IDENTICAL θ)

## 6-7. Repeated calls at bit-identical theta / at same theta with different request codes?

**Confirmed empirically**, not just structurally. `audit_outer_callbacks.jl` ran a real D=4
outer solve (γ_d≡1+direct-γ' config, `ek_outer_loop_options_audit.opt` — a diagnostics-only copy
of the PRODUCTION `ek_outer_loop_options.opt` with only `maxit`/`maxtime_real` bounded so the
audit finishes quickly; `eval_fcga` left at production's `no`). Chronological trace
(`callback_trace.csv`, reproduced here):

```
idx  kind  theta_hash            same_as_prev_call
1    F     10319837004897719324  false
2    G     10319837004897719324  true   <- SAME theta as call 1
3    F     7768691897361044819   false
4    G     7768691897361044819   true   <- SAME theta as call 3
5    F     9049241939847989669   false
6    G     9049241939847989669   true   <- SAME theta as call 5
7    F     17363810434499655786  false  <- a rejected trial step (no G follows)
8    F     17709404322636709886  false
9    G     17709404322636709886  true   <- SAME theta as call 8
```

9 total callback invocations, only **5 unique θ points** — every accepted point pays for the
inner CC solve exactly **twice** (once from F, once from G, at bit-identical θ). This is a
textbook duplicate-solve pattern, present in production's actual default configuration today.

## 8-9. Is the inner solution or the divergence gradient cached and reused? — No (before this audit)

No. `obj.x` (`cc_algo/PsiObjectiveBundle.jl`'s `x` field) stores only the MOST RECENT successful
inner solution, used purely as a **warm-start seed** for the NEXT inner solve
(`inner_loop_initial_values`, gated by `obj.use_cached_x`) — not as a cache. It gets
unconditionally overwritten by every call, including calls at the identical θ that just produced
it. There is no θ-keyed lookup anywhere in `cc_algo/`.

## 10. Are destination inversions / moment arrays reused? Partially, via `obj.H`

`obj.H` (the moments matrix) IS overwritten fresh by `moments!` inside `inner_loop_internal`
every call — no reuse across calls even at identical θ. Not a bug in the sense of wrong output
(H must be recomputed if θ changed), but it's further work redone identically when θ hasn't
changed.

## 11. Is a warm start being confused with a cache? — No confusion in the code, but worth stating precisely

`use_cached_x=true` (set in every full_aod_diag diagnostic obj constructor, and NOT the default —
struct default is `false`) makes the inner solve start from the previous solution rather than
zero. This measurably helps (see `warmstart_audit.jl` below) but is **not** a cache: it still
pays for a full KNITRO inner solve, including at least one KNITRO iteration even when the seed
is already exactly optimal (§ Warm-start audit).

## Fix implemented (additive, not touching `cc_algo/`)

`cached_outer_loop.jl` defines `OuterEvalCache` (exact Float64-equality keyed, cache-owned copy
of θ — never aliases KNITRO's buffer) plus parallel callbacks `cb_F!`/`cb_G!`/`cb_FG!` and
`outer_loop_instrumented(obj, ...; use_cache)`, which use the SAME `obj.outer_loop_opt` file (so
it can be run against production's real `eval_fcga=no` config) and the SAME unmodified
`inner_loop_internal`/`(Q::PsiObjectiveBundleImplicit)` callable, but route every call through
`ensure_inner!`/`ensure_grad!`, which skip the KNITRO inner solve (or the gradient/Jacobian
computation) entirely when θ exactly matches the cached point. See `cache_design.md` for the
struct and invalidation rules.

## Result: re-running the identical D=4 audit with the cache on

```
                        run1 (no cache)   run2 (exact-point cache)
total callbacks         9                 9
unique theta points     5                 5
inner solves             9                 5   <- exactly 1/unique theta, as intended
grad computations        4                 4
cache hits               0                 4
kappa (bound found)     0.912028119649117  0.912028119649117   <- bit-identical
```

Duplicate inner solves eliminated: 4 of 9 (44%). `κ` output is bit-identical between the two
runs — caching changes *what work is redone*, not the answer, exactly as required.

See `cache_benchmark.csv` for the machine-readable version and `benchmark_forwarddiff.jl`'s
results for how this compounds with the (separately, currently dominant at D=4) dense-Jacobian
gradient cost.
