# No-Dense-G Runtime Proof — 2026-07-26

## Scope statement (read this before trusting the numbers below)

`no_dense_g_counters.jl` implements the task's requested counters (§12) and an opt-in
`FAIL_FAST_ON_DENSE_G` guard, wired at **this branch's own call sites**:
`economic_forward!`/`economic_transpose!` (`economic_operator.jl`), the dense-economic-fallback
branches in `OriginZCOperatorState`/`CMMeanZCOperatorState`, and the `:dense_reference`/`:operator`
dispatch points for origin-ZC and CM+ZC. This is **not** the full-codebase audit task §13 asks for
("search the repository for every dense moment builder; `gemv!` against draw-by-moment matrices;
`select_G_from_H`; `obj.H[:, ...]`; generic verifier; diagnostic or cache scorer; post-processing
routine... classify each use") — that is a separate, much larger item this session did not attempt
across the ~500-file `full_aod_diag/d4_exact/` tree. What follows is an honest report of what these
counters DO prove, not a claim that every dense-G consumer in the codebase has been found.

## Live runtime proof (real, this session)

Real D=4 sequences through the actual production entry points (`cm_originzc_production_value_verified`,
`cm_originzc_production_gradient`, `cm_meanzc_production_value_verified`,
`cm_meanzc_production_gradient`, `verify_inner_solution_operator_originzc!`):

**Origin-ZC, `fg_backend=:operator`, `FAIL_FAST_ON_DENSE_G[]=true`** (value + gradient + a repeated
value-then-verify sequence + one operator verification call):

```
full_G_materializations = 0
dense_economic_G_materializations = 0
dense_CM_G_materializations = 0
dense_ZC_G_materializations = 0
generic_dense_FG_calls = 0
operator_FG_calls = 7
operator_forward_calls = 7
operator_transpose_calls = 7
operator_verification_calls = 1
dense_reference_verification_calls = 0
```

Ran to completion with the fail-fast guard **armed** — proves zero dense-G materializations were
attempted through the covered call sites during this sequence, not just that the counter would be
zero if checked.

**CM+ZC, `inner_fg_backend=:operator`** (value + gradient):

```
full_G_materializations = 0, dense_economic_G_materializations = 0,
dense_CM_G_materializations = 0, dense_ZC_G_materializations = 0, generic_dense_FG_calls = 0,
operator_FG_calls = 5, operator_forward_calls = 5, operator_transpose_calls = 5
```

**Both families, `:dense_reference`** (one value call each, counters reset first): the guard
correctly attributes each dense-path inner solve to `generic_dense_FG_calls`:

```
generic_dense_FG_calls = 2, operator_FG_calls = 0
```

This confirms the counters correctly discriminate the two paths — a necessary sanity check on the
counters themselves, not just the code they're counting.

## What is NOT covered by these counters (explicit gaps)

- Flexible CM's and common Fréchet's own economic-core block, still dense this branch (see
  `SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md` §5) — their dense `E*λ` `BLAS.gemv!` calls are
  **not instrumented** (no `record_dense_economic_G!()` call site added there).
  `dense_economic_G_materializations` therefore reads 0 even when flexible CM or common Fréchet ran
  in `:dense_reference` or `:cm_lookup`/`:cm_frechet_lookup` mode — it is **not** a global claim
  that those two families are dense-E-free (they are not, per the design doc's own table).
- Verification for CM+ZC/flexible-CM/common-Fréchet still reads dense `obj.H`/`select_G_from_H`
  (unchanged this branch) — `dense_reference_verification_calls` is not incremented there (no call
  site added).
- The 500-file-wide dense-G-consumer classification task §13 asks for
  (`PRODUCTION_HOT_PATH`/`PRODUCTION_SETUP_ONLY`/`EXPLICIT_REFERENCE`/`TEST_ONLY`/`DEAD_CODE`) was
  not performed.

## Honest verdict for this counter

`FULL_G_MATERIALIZATION`: **not** `zero_all_production_families`. Correct verdict:
`present_flexible_cm,common_frechet` for the economic block (dense E unchanged), plus every
family's Hessian-cross-term dependency on dense `H[:,2:1+NCORE]` (out of scope, see the design
doc's §2 for why this is a deliberate boundary, not an oversight).
