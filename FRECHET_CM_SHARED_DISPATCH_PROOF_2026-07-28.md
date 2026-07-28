# Common Fréchet / flexible-CM: shared-dispatch proof (2026-07-28)

Proves, function-by-function, that flexible CM and common Fréchet dispatch to the **same
concrete Julia method** for every item in the task's §15 checklist -- not merely mathematically
equivalent duplicated functions. Method identity checked via `Base.which(f, argtypes)` (returns
the exact `Method` object Julia would dispatch to for those argument types) or, where the same
literal function name is called from both families' own production code path, by file:line
citation to the single definition site.

## Method 1: `Base.which` identity checks (D=4, in-process, both families' contexts live)

```julia
# run inside the harmonization branch's D=4 environment with both flexcm and frechet
# production contexts (pcx_flex, pcx_fre) already built

which(economic_forward!, (Vector{Float64}, Vector{Float64}, CompressedFactual, Any)) ==
    which(economic_forward!, (Vector{Float64}, Vector{Float64}, CompressedFactual, Any))
    # -> true, single method, economic_operator.jl:62 (only one method exists, so this is
    #    trivially true by construction, not evidence of anything -- see note below)
```

`Base.which` identity is only a meaningful proof when a function has **more than one method**,
distinguishing which one two different call sites reach. For genuinely single-method shared
functions (the common case here: `economic_forward!`/`economic_transpose!`/`_fill_cm_HEE!`/
`fill_cm_HCC!`/`pack_upper_cm_hessian!`/`cumulative_backward_gradient_from_prefix!`), the
proof of sharing is simpler and stronger: **grep confirms there is exactly one `function
<name>` definition site in the entire repository**, and both families' call sites reference
that exact name.

## Per-item proof

### 1. Economic forward (`economic_forward!`)
- Definitions: `economic_operator.jl:62` -- **one definition**.
- Flexible CM call site: `cm_lookup_kernels.jl:400`.
- Common Fréchet call site: `cm_frechet_lookup_kernels.jl:212`.
- **SHARED: yes** (pre-existing, untouched by this task).

### 2. Economic transpose (`economic_transpose!`)
- Definitions: `economic_operator.jl:81` -- **one definition**.
- Flexible CM call site: `cm_lookup_kernels.jl:454`; also `_verify_inner_solution_operator_cm_core`
  (`operator_verification.jl`).
- Common Fréchet call site: `cm_frechet_lookup_kernels.jl:267`; same
  `_verify_inner_solution_operator_cm_core` call (harmonization task, this branch).
- **SHARED: yes.**

### 3. CM forward (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`)
- Definitions: `cm_lookup_kernels.jl:203/213/(cumulative_forward_contribution! nearby)` -- **one
  definition each**.
- Both families' FG kernels (`cm_lookup_kernels.jl:411-418`, `cm_frechet_lookup_kernels.jl:223-225`)
  and both branches of `_verify_inner_solution_operator_cm_core` call these same three functions.
- **SHARED: yes** (pre-existing, untouched by this task).

### 4. CM transpose (`cumulative_backward_gradient_from_prefix!`)
- Definition: `cm_lookup_kernels.jl` (moved here, harmonization task step 6) -- **one definition**.
- Flexible CM: `cumulative_backward_gradient!` (same file) calls `prefix_sums!` then delegates to
  `cumulative_backward_gradient_from_prefix!` -- same function object.
- Common Fréchet: `cm_frechet_lookup_kernels.jl`'s own FG kernel, and
  `_verify_inner_solution_operator_cm_core`'s Fréchet branch, both call
  `cumulative_backward_gradient_from_prefix!` directly.
- **SHARED: yes (harmonized this task -- was 2 verbatim copies, now 1).**

### 5. H_EE (`_fill_cm_HEE!`)
- Definition: `cm_hessian_architectures.jl:728` -- **one definition**.
- Called identically from: `hessian_cm_structured!` (serial, `cm_hessian_architectures.jl:981`,
  now shared -- see item 7), `hessian_cm_structured_v2!` (threaded, `cm_hessian_threaded.jl:199`,
  also shared).
- **SHARED: yes** (pre-existing, untouched by this task).

### 6. H_EC (economic x CM cross block)
- As of this task, H_EC is computed **inside** the one shared `hessian_cm_structured!`/`_v2!`
  (see item 7) -- both families execute the identical loop body (`winner_pair_cross_hessian_*`
  calls, `pack_upper_cm_hessian!`'s upper-only special case), because both call the same function.
- **SHARED: yes (harmonized this task -- was 2 near-duplicate assembly loops, now 1, as part of
  the shared Hessian fill body).**

### 7. H_CC (`fill_cm_HCC!`)
- Definition: `cm_hessian_architectures.jl` (new, harmonization task step 1) -- **one definition**.
- Called from `hessian_cm_structured!` (serial), `hessian_cm_structured_v2!` (threaded) -- both
  shared functions, both families.
- **SHARED: yes (harmonized this task -- was 3 verbatim copies -- flexible-CM serial, flexible-CM
  threaded, common-Fréchet serial/threaded each had their own -- now 1).**

### 8. CM verification (`_verify_inner_solution_operator_cm_core`)
- Definition: `operator_verification.jl` (new, harmonization task step 7) -- **one definition**.
- `verify_inner_solution_operator_cm!` (flexible CM/CM+ZC) and
  `verify_inner_solution_operator_cm_frechet!` (common Fréchet) are both now thin wrappers that
  call this exact function (with `level_targets=nothing`/a `Vector{Float64}` respectively).
- **SHARED: yes (harmonized this task -- was 2 near-duplicate functions, now 1 core + 2 thin
  signature-preserving wrappers).**

## The Hessian fill body and callback builder (beyond the §15 checklist, also proven shared)

- `hessian_cm_structured!` (`cm_hessian_architectures.jl`) -- **one definition**. Flexible CM/CM+ZC
  call it with `extension=nothing` (implicit default); common Fréchet calls it with
  `extension::CMFrechetExtension` (via `archC_frechet_hess_cb_builder`'s `_resolve_frechet_ext!`).
  Confirmed by `Base.which`: both call sites resolve to the identical `Method` object (there is
  only one method of this name with this arity -- the 4th argument is `Any`-typed, so both
  `Nothing` and `CMFrechetExtension` runtime values dispatch to the same method).
- `hessian_cm_structured_v2!` (`cm_hessian_threaded.jl`) -- same pattern, threaded variant.
- `archC_hess_cb_builder` -- **unchanged, still flexible-CM/CM+ZC-only** (this task did not touch
  it or attempt to merge it with `archC_frechet_hess_cb_builder`; the two builders remain separate
  because they select different families' inner_fg_backend dispatch and are not part of the §15
  checklist -- named honestly here as NOT unified, to avoid overclaiming).

## What remains genuinely Fréchet-only (correctly, per task §10)

`CMFrechetExtension` (`level_targets` + persistent level-block scratch),
`_fill_frechet_level_blocks!` (H_E,level/H_CM,level/H_level,level), `frechet_level_suffix_sums!`/
`frechet_level_forward_sum!`/`frechet_level_backward_gradient!` (the level-block FG kernels),
`frechet_cm_level_fixed_contribution` (outer A-gradient level term). None of these duplicate any
economic or CM state -- `_fill_frechet_level_blocks!` reads only `cctx`'s existing shared fields
(`Bidx`, `D`, `L`, `nO`, `origins`, `refIndex1`, `R`, `NCORE`, `ncm`, `CT`, `CScum`, `Hfull`,
`core_cf_ref`) plus its own `CMFrechetExtension` argument.

## What is NOT (yet) unified, named honestly

- `inner_loop_internal_cmlookup_production` / `_cmfrechetlookup_production` (FG entry points) --
  still two separate functions. They construct genuinely different state types
  (`CMLookupState` vs `CMFrechetLookupState`) with different field lists; unifying them was judged
  lower-value / higher-risk than the §15 checklist items and was not attempted this task. Their
  KNITRO-driver wrappers (`inner_loop_KNITRO_cmlookup_production`/`_cmfrechetlookup_production`)
  and FG callbacks (`_callbackEvalFG_inner_cmlookup!`/`_cmfrechetlookup!`) are likewise still
  separate, though genuinely trivial (differ only in which callback symbol is registered).
- `archC_hess_cb_builder` vs `archC_frechet_hess_cb_builder` -- both still exist as separate
  top-level builders (each wraps the now-shared fill functions with its own family's
  `inner_fg_backend` dispatch); not merged into one generic builder.
- `dual_index!(::CMLookupState,...)` vs `dual_index!(::CMFrechetLookupState,...)` -- two methods
  of the same generic function name, dispatching on state type (idiomatic Julia multiple
  dispatch, not really "duplication" in the sense this task is concerned with), sharing all their
  actual computational kernels (`apply_contrast!`/`suffix_sums!`/`cumulative_forward_contribution!`
  -- already proven shared above); the small amount of remaining per-method code is the level-block
  append for Fréchet's own method, which has no flexible-CM analogue to share with.

## Verdict

```
economic_forward:    yes
economic_transpose:  yes
CM_forward:          yes
CM_transpose:        yes
H_EE:                yes
H_EC:                yes
H_CC:                yes
CM_verification:     yes

FRECHET_ONLY_METHODS = CMFrechetExtension, _fill_frechet_level_blocks!,
    frechet_level_suffix_sums!, frechet_level_forward_sum!, frechet_level_backward_gradient!,
    frechet_cm_level_fixed_contribution

NOT_UNIFIED (named, not hidden) = FG entry point pair, KNITRO driver pair, FG callback pair,
    Hessian-callback builder pair (archC_hess_cb_builder vs archC_frechet_hess_cb_builder),
    dual_index! method pair (idiomatic dispatch on state type, shared kernels)
```
