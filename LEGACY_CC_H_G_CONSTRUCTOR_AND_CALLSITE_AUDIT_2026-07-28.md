# Legacy CC `H = [K | ones | G]` constructor and call-site audit — 2026-07-28

Base: `campaign/five-family-bounds-2026-07-28@93f26df7`. Method: independently re-derived by a
read-only research pass over the checked-out commit (not copied from the prior session's own
`docs/NO_MOMENTS_NO_COMPOSITE_G_MASTER_REPORT_2026-07-28.md`, though that document was read first
for background and every claim in it was independently re-verified against current source). Full
raw file:line detail is preserved in this session's working notes; this document is the curated
summary. See `PARALLEL_LEGACY_CC_H_REMOVAL_PROVENANCE_2026-07-28.md` for isolation/provenance.

## 0. The struct all 5 families still construct

`cc_algo/PsiObjectiveBundle.jl:222-276`, `@with_kw mutable struct PsiObjectiveBundleImplicit{T}`.
Relevant fields:

```julia
moments!  ::Function                                                # K/G-filling closure
H         ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d)) # K | ones | G, M×(2+d)
H_copy    ::Array{Float64,2} = hcat(zeros(M), ones(M), zeros(M, d))
arg0,arg1,arg2 ::Array{Float64,1} = zeros(M)                        # each
jac_h     ::Array{Float64,3} = ... (N × (2+d) × l, when needs_outer_moment_jacobian)
```

`select_G_from_H(obj, H) = @view(H[:, 3:end])` (`cc_algo/PsiObjectiveBundle.jl:624`).

**No production call site anywhere overrides `H` explicitly** — every family's constructor call
passes `d = <family's own moment count>` and lets `H` fall through to the `@with_kw` default,
i.e. every family, including unrestricted, allocates the full `[K|ones|G]`-width dense matrix
(`W × (2+d)`, `W` = draw count) at every `PsiObjectiveBundleImplicit(...)` call. `build_moments`
as a literal symbol does not exist anywhere in the tree (grep confirmed zero matches) — the
codebase's own name for this is `moments!`.

## 1. Per-family classification

| Family | Live `obj.H` allocation | Priming (`moments!`) call | `select_G_from_H` call | Throwaway extra `obj.H` allocation |
|---|---|---|---|---|
| Unrestricted | 1×, at context build (`context.jl:58` / `context_real_d20.jl:132`) — struct-mandated, only 2-3 of its columns are ever written under the default backend | **none** — `inner_loop_internal_compressed` never calls `obj.moments!` | **none** | none |
| Flexible-CM | 1× live (`cm_production_bundle.jl:129`) | **once/inner solve** (`cm_lookup_production.jl:124`) | **once/inner solve**, paired 1:1 | **1×** per context build — `build_cm_augmented_obj` (`common_marginals_moments.jl:229`) constructs a full `W×(2+ncore+ncm)` object solely to harvest scalar config fields, then discards it entirely (its `moments!`/`H` are never touched again) |
| Common-Fréchet | 1× live (`cm_frechet_level.jl:362`) | **once/inner solve** (`cm_frechet_lookup_production.jl:96`) | **once/inner solve**, paired 1:1 | **1×** per context build — `build_cm_frechet_level_augmented_obj` (`cm_frechet_level.jl:163`), identical throwaway pattern |
| CM+ZC | 1× live (`cm_meanzc_moments.jl:439`) | **once/inner solve** (`cm_meanzc_lookup_production.jl:80`), no skip mechanism | **once/inner solve**, paired 1:1 | none (single-constructor family, no archB/archA duplicate) |
| Origin-ZC | 1× live (`cm_originzc_moments.jl:233`) | **once/inner solve** (`cm_originzc_lookup_production.jl:101`), no skip mechanism | **once/inner solve**, paired 1:1 | none |

`PRODUCTION_MOMENTS_CALLS = 4`, `PRODUCTION_SELECT_G_FROM_H_CALLS = 4` — matches the prior
session's own verdict; independently re-confirmed against current source, not merely re-cited.

## 2. What the priming closures actually fill, right now

`wrap_moments_with_cm_archB` (flexible-CM, `cm_hessian_architectures.jl:213-308`) and
`wrap_moments_with_cm_frechet_archB` (common-Fréchet, `cm_frechet_level.jl:208-282`) are
structurally identical: inside the returned `(K, G, θ, U, obj) -> ...` closure,

1. `cf = cf_build(θ, ctx; check_ties=false)` — always, cheap, publishes into the shared
   `core_cf_ref[]` box the Hessian callback reads (this publish is what actually matters for the
   operator Hessian path).
2. `materialize_dense_factual_structured!(@view(Gtmp[:,1:pregrav]), cf)` — the **economic block**
   dense fill — **unconditional**, regardless of `skip_fill`.
3. `fill_K_directgp!(K, θ, ctx)` — a separate O(W) scalar broadcast into `obj.H[:,1]`, always.
4. `@views G[:,1:pregrav] .= Gtmp[...]` — copies the economic block into `G` (i.e. `obj.H`).
5. Only when `skip_fill=false`: `fill_cm_columns_from_bins!` (CM-grid block) and, for
   common-Fréchet only, `fill_frechet_level_columns_from_bins!` (level-anchor block).

`wrap_moments_with_cm_meanzc` (CM+ZC, `cm_meanzc_moments.jl:280`) and
`wrap_moments_with_originzc` (origin-ZC, `cm_originzc_moments.jl:93`) have **no `skip_fill` kwarg
at all** — every block (economic + CM-grid/mean/pair) is unconditionally dense-filled every
inner solve.

**The economic block (step 2/4) is the one piece never skippable in any of the 4 restricted
families today.** A same-branch attempt to extend `skip_fill`'s scope to also cover it was made
and reverted after `test_shared_core_hessian_d4_gates.jl` caught a real (not floating-point-noise)
H_EE mismatch of `max|Δ|=0.0336`. See
`OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md` §3 for this session's root-cause analysis of
that regression (not fully closed — see that document's own honest gap statement).

## 3. K / ones classification (task §3)

- **K (`obj.H[:,1]`)**: filled via `fill_K_directgp!` — a trivial `O(W)` scalar broadcast
  (`K[s] = θ_full[3+D] * SamplingWeights[s]`), **not** a winner-search or dense-BLAS operation, and
  **not** read as part of any cross-column BLAS operation in the operator path. Classification:
  **COUNTERFACTUAL_EQUILIBRIUM_MOMENT** — this is the genuine `ζ`-side scalar objective quantity
  the inner dual needs, not a removable "CC payoff artifact." It must stay computed, but there is
  no scientific reason it needs to live inside a `W×(2+d)` matrix column rather than a standalone
  `Vector{Float64}(undef, W)` once the matrix itself is gone.
- **ones (`obj.H[:,2]`)**: written via `obj.H[:,2] .= 1.0`, a broadcast into an existing column, not
  a separate allocation at the priming call site. However, the struct's own `@with_kw` **default**
  expression, `H = hcat(zeros(M), ones(M), zeros(M,d))`, does call `ones(M)` as a genuine, separate
  temporary array at **every** `PsiObjectiveBundleImplicit(...)` construction (i.e., once per
  context build, not once per inner solve) — immediately `hcat`'d away, but a real, countable
  allocation. This is the literal site `ONES_VECTOR_ALLOCATIONS` should count today: **not zero**,
  once per family per context build (plus once more for flexible-CM/common-Fréchet's throwaway
  `obj_cm`, see §4). Classification: the constant-`ζ` column has a real algebraic reason to exist
  (the dual index's `-ζ·1 - ...` term) but, per the task's own inner-dual formula
  (`f(ζ,λ) = mean(Ψ(-ζ - g'λ)) + ζ`), it is representable as a **scalar** `ζ` broadcast at the point
  of use (`arg0 .= -ζ .- ...`), not a materialized length-`W` vector at all — matching the task's
  explicit framing ("The constant column is represented by the scalar ζ, so no W-vector of ones is
  needed"). The unrestricted family and the operator FG paths for all 4 restricted families already
  do exactly this (their `dual_index!` methods take a scalar `ζ`, never a materialized ones column)
  — the *only* place a real `ones(W)` still gets allocated is the legacy struct's default `H` field,
  which the operator paths never read for this purpose but which is still built regardless because
  nothing yet prevents it structurally.
- **No standalone `K = zeros(W)` / `ones(W)` buffer feeding any family's *live* `obj.H`** was found
  outside the struct-default pattern above. All other `K = zeros(W); G = zeros(W,d)`-style patterns
  found by grep (`oracle.jl`, `oracle_fast.jl`, `winners.jl`, `parameter_table.jl`, `c1x_*.jl`
  diagnostic/benchmark scripts, the `lfix_*.jl` legacy-outer-gradient-backend family) are local,
  throwaway buffers for a **non-default** explicit backend (`:legacy_unbuffered` gradient, or a
  standalone test/benchmark script) — classified **TEST_ONLY / EXPLICIT_DENSE_REFERENCE**, not live
  production storage.

```
CC_PAYOFF_K_VECTOR_ALLOCATIONS  = 0   (K is a COUNTERFACTUAL_EQUILIBRIUM_MOMENT, not a removable
                                        CC-payoff artifact; retained as the O(W) fill it already is)
ONES_VECTOR_ALLOCATIONS         = 1 per PsiObjectiveBundleImplicit construction (struct-default
                                    `ones(M)`, still present; target is 0 once the operator bundle
                                    type no longer carries an `H` field with this default at all)
```

## 4. Newly-confirmed finding this session: throwaway `obj_cm` allocations

**Not named in the prior session's report.** `build_cm_augmented_obj`
(`common_marginals_moments.jl:197-224`) and `build_cm_frechet_level_augmented_obj`
(`cm_frechet_level.jl:140-193`) each construct a full `PsiObjectiveBundleImplicit` (`W×(2+ncore+ncm)`
`H`/`H_copy`, plus `jac_h`, `arg0/1/2`, etc.) purely to compute `CM`/`z`/`origins`/`ncore`/`ncm` and
carry a handful of **scalar/pass-through** config fields forward. Traced every field the production
call site (`cm_production_bundle.jl:129`, `cm_frechet_level.jl:362`) actually reads from this
throwaway object: `δ, find_smallest, γ, d, outer_constr_index, inequality_index, complement_index,
l, U, N, lower_limit, use_cached_x, threshold_state, outer_loop_opt, inner_loop_opt,
needs_outer_moment_jacobian` — **every one of these except `d`/`outer_constr_index` is an unchanged
pass-through of `ctx.obj`'s own existing fields**, and `d`/`outer_constr_index` are simple sums
(`ncore+ncm`, `obj0.outer_constr_index+ncm`). **None of the throwaway object's `H`, `H_copy`,
`jac_h`, `arg0/1/2`, `x`, `∂x_∂θ`, `∂c_∂θ`, `∂∂f_∂∂x`, `∂∂f_∂x∂θ` fields are ever read** on this path.

This is genuinely dead, one-time-per-context-build (not per-inner-solve) waste, confirmed by
direct field-by-field trace, not assumption. **Not fixed this session**: `build_cm_augmented_obj` /
`build_cm_frechet_level_augmented_obj` are shared utilities with **~30 other callers**
(`c12_*`/`c13_*`/`c14_*`/`c15_*`/`test_*` diagnostic and correctness-comparison scripts), several of
which legitimately use the returned `obj_cm` as a real, solvable dense-reference object — so this
cannot be fixed by shrinking the shared function's allocation unconditionally. The safe fix is to
split the function into a cheap "compute CM/z/origins/ncore/ncm" half and an opt-in "build a full
solvable `obj_cm`" half, with `build_cm_production_context`/`build_cm_frechet_production_context`
calling only the cheap half — a real, bounded, low-risk refactor, left as the top actionable item
for the next session on this task (see `HIGHEST_PRIORITY_REMAINING_GAP` in the final verdict doc).

## 5. `select_G_from_H` / `moments!` reachability outside the 5 families

`cc_algo/inner_loop_functions.jl`, `cc_algo/outer_loop_functions.jl`, `cc_algo/local_sensitivity.jl`,
`cc_algo/KLObjectiveBundle*.jl` are **base-module generic dense implementations** that every
family's `PsiObjectiveBundleImplicit` construction still inherits its *type* from, but none of
these functions is called by any of the 5 families' actual production entry-point chains (each
family has its own `inner_loop_internal_*` in `full_aod_diag/d4_exact/` that shadows the generic
one). Classification: **DEAD_CODE relative to the production chain**, confirmed by trace, not
merely absence-of-evidence. The `legacy/` directory (a separate, differently-named module) is
**DEAD_CODE**, unreachable from `full_aod_diag/d4_exact/` by any `include`.

The base struct's own callable functor (`(Q::PsiObjectiveBundleImplicit)(x,g,θ;...)`,
`cc_algo/PsiObjectiveBundle.jl:282-368`) reads `H` via a dense `BLAS.gemv!` unconditionally when
invoked — but is **not** the registered KNITRO FG/Hessian callback for any of the 5 families under
default (`:operator`/`:cm_lookup`/`:operator`/`:operator`) settings; each family registers its own
closure-based callback instead. The functor's only confirmed live callers are
`archC_verified_state`'s explicit `:dense_reference` verification arm (non-default,
`CM_VERIFICATION_BACKEND_DEFAULT[]==:operator`) and `delta_dual_from_base` (no production caller
found, only `c13_*`/`c14_*` diagnostics). Classification: **EXPLICIT_DENSE_REFERENCE**.

## 6. Requirement check (this session's starting point, before any fix)

```
PRODUCTION_MOMENTS_CALLS               = 4   (unchanged from prior session's handoff)
PRODUCTION_SELECT_G_FROM_H_CALLS       = 4   (unchanged)
COMPOSITE_G_MATERIALIZATIONS           = 0   (Hessian-callback side; priming side still builds it)
G_SIZED_BACKING_STORAGE_ALLOCATIONS    = 1 per restricted family LIVE
                                          + 1 per flexible-CM/common-Fréchet THROWAWAY (new finding, §4)
CC_PAYOFF_K_VECTOR_ALLOCATIONS         = 0   (K is a required scalar-fill moment, not a removable artifact)
ONES_VECTOR_ALLOCATIONS                = 1 per PsiObjectiveBundleImplicit construction (struct default)
```

Both counts the task requires at zero (`PRODUCTION_MOMENTS_CALLS`,
`PRODUCTION_SELECT_G_FROM_H_CALLS`, `G_SIZED_BACKING_STORAGE_ALLOCATIONS`) remain **greater than
zero** at the start of this session — this is the same honest gap the prior session left, now with
a newly-identified second contributor (§4) and a sharper (though not fully resolved) root-cause
lead on the blocking regression (see `OPERATOR_DENSE_BUNDLE_TYPE_SEPARATION_2026-07-28.md`).
