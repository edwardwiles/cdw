# Literal method-reuse proof — flexCM/Fréchet FG harmonization (2026-07-29)

## What was extracted

Four new shared functions, all added to `cm_lookup_kernels.jl` (the file both families already
depended on for their low-level numeric kernels; load order across every entry point in the repo
already guarantees it loads before `cm_frechet_lookup_kernels.jl` — confirmed by grepping every
`include(...)` site, see below):

```julia
economic_forward_into_arg0!(st, x) -> cf
cm_forward_contribution!(st, λ_cm, method::Symbol) -> st.arg0
economic_transpose_into_g1_and_gE!(g, st, cf) -> sum_dPsi
cm_transpose_into_g!(g, st, method, D, ncore1, ncm, M) -> st.g_block
```

Each is duck-typed on the field subset `CMLookupState` and `CMFrechetLookupState` already share
verbatim (`obj`, `ncore`, `core_cf_ref`, `econ_ws`, `econ_ws_for`, `econ_buf`, `xsub`, `arg0`, `nO`,
`L`, `λmat_block`, `R`, `λmat_ext`, `bins`, `refIndex1`, `origins`, `cm_contrib`, `arg1`, `hist_h`,
`hist_partials`, `Hpre`, `g_block`, `g_stored`) — no abstract supertype was introduced; this mirrors
the exact pattern `hessian_cm_structured!`/`_verify_inner_solution_operator_cm_core` already use to
share the Hessian/verification cores across both families.

Both families' `dual_index!` and FG functor now call these SAME four functions for their shared
`[E|C]` prefix; common Fréchet's own code, after the refactor, is reduced to:

```julia
function dual_index!(st::CMFrechetLookupState, x)
    ...
    economic_forward_into_arg0!(st, x)              # [E] -- identical call flexible_cm makes
    cm_forward_contribution!(st, λ_cm, :suffix)     # [C] -- identical call flexible_cm makes
    frechet_level_suffix_sums!(st.P_level, λ_level) # [F] -- extension only, genuinely new
    frechet_level_forward_sum!(...)                 # [F]
    ...                                              # [F] const-term/arg0 update
end
```

i.e. exactly the task's target composition `frechet_cm_fg = economic_fg + cm_fg +
frechet_extension_fg`, adapted to this codebase's no-(H)-operator, in-place-mutating-state FG
architecture (there is no separate top-level `frechet_cm_forward!`/`frechet_cm_transpose!` wrapper
function in this codebase's idiom — the FG functor callable `(st::CMFrechetLookupState)(x,g)` IS
that composition point, and it now literally is `economic_transpose_into_g1_and_gE!` +
`cm_transpose_into_g!` + inline `[F]` code, in that order).

The `frechet_cm_level_fixed_contribution`/`cm_fixed_contribution` duplicate (item 4 in the
duplication audit) was fixed the same way: `cm_fixed_value_contribution(λ_cm, nO, L, refIndex1,
origins, bins, R)` (new, `lfix_cm_aware.jl`) is now called by BOTH `cm_fixed_contribution` and
`frechet_cm_level_fixed_contribution`, replacing Fréchet's previously-inlined verbatim copy.

## Proof method

Numerical equality alone is not proof of literal reuse (two independently-maintained
implementations of the same formula could agree numerically while remaining organizationally
duplicated). This task establishes literal reuse three ways:

1. **By construction**: the shared functions were extracted by literal code motion (cut the
   duplicated lines out of each `dual_index!`/FG functor, paste into ONE new function, replace both
   call sites with a call to it) — no arithmetic, variable, or operation ordering was changed.
   `git diff` for `cm_lookup_kernels.jl`/`cm_frechet_lookup_kernels.jl` shows this directly: the
   function BODIES that appear in the new shared functions are byte-identical to what was previously
   inlined in `CMLookupState`'s own methods, and `CMFrechetLookupState`'s corresponding methods
   shrank to calls into those same functions (see the diff for the exact before/after).
2. **By call-site inspection**: `grep -n "economic_forward_into_arg0!\|cm_forward_contribution!\|economic_transpose_into_g1_and_gE!\|cm_transpose_into_g!" full_aod_diag/d4_exact/cm_lookup_kernels.jl full_aod_diag/d4_exact/cm_frechet_lookup_kernels.jl` shows both families' `dual_index!`/FG-functor call the identically-named functions defined once in `cm_lookup_kernels.jl` — there is no `_flexcm`/`_frechet`-suffixed sibling of any of the four.
3. **By numerical equivalence at machine precision** against the untouched dense reference
   (`FLEXCM_FRECHET_D4_EQUIVALENCE_GATE_2026-07-29.csv`, `FLEXCM_FRECHET_D20_EQUIVALENCE_GATE_2026-07-29.csv`)
   — this is necessary-but-not-sufficient on its own (see above), included as the complementary
   check that the refactor did not silently change the math while appearing to share code.

## Load-order verification (no new include-guard risk introduced)

```
$ grep -rn "cm_frechet_lookup_kernels.jl" full_aod_diag/d4_exact/*.jl | grep include
operator_verification.jl:166:isdefined(Main, :CMFrechetLookupState) || include(joinpath(@__DIR__, "cm_frechet_lookup_kernels.jl"))
```

`operator_verification.jl:76` guards/includes `cm_lookup_kernels.jl` (and therefore the new shared
functions) BEFORE its own line 166 guard/include of `cm_frechet_lookup_kernels.jl`; every production
entry point (`campaign_cm_family_runner.jl` et al.) includes `cm_production_bundle.jl` (which
includes `operator_verification.jl`) before `cm_frechet_level.jl`/`cm_frechet_lookup_production.jl`/
`cm_frechet_cplus.jl`. `cm_frechet_lookup_kernels.jl` already called several `cm_lookup_kernels.jl`
functions unguarded before this task (`apply_contrast!`, `suffix_sums!`,
`cumulative_forward_contribution!`, etc.) — this task's new shared functions follow the exact same
existing, already-safe convention, adding no new load-order requirement.

## Results

- `c12i_validate_lookup_fg.jl` (pre-existing, unmodified) re-run against the refactored
  `CMLookupState`: ALL 12 (L×contrasts×method) cells PASS, worst `ferr` 4.4e-16, worst `grel`
  1.1e-15 — unchanged in character from the pre-refactor implementation (this script only exercises
  the refactored code path, so a pass here after the refactor is itself part of the proof).
- `bench_frechet_operator_fg_default_gate_2026-07-27.jl` (pre-existing, unmodified) re-run against
  the refactored `CMFrechetLookupState` at real D=20 (`W=80,000`, `L=50`, both contrasts, 3 points):
  ALL PASS, `zeta*` agreement 3.5e-18 to 3.3e-13 across cells, speedup 1.21x-1.36x vs dense,
  isolated per-FG-callback allocation 3,376 bytes (no W-scale allocation).
- New `harmonization_d4_equivalence_gate_2026-07-29.jl`: 96/96 cells PASS (both families, L in
  {10,20,50}, both contrasts, 8 points each including the real solver-derived dual) at machine
  precision (worst `ferr_abs` ~1e-17, worst `grel` ~1e-15).
