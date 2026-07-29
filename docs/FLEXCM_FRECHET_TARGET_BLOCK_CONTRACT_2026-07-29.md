# Authoritative FG block layout contract — flexible_cm / common_frechet (2026-07-29)

## Column ordering (unchanged by this task — already consistent pre-refactor)

Both families' inner-solve dual vector `x` uses the SAME leading layout, matching the Hessian
side's own block ordering:

```
flexible_cm:    x = [ ζ | λ_core (E, ncore-1) | λ_cm (C, ncm = nO*L) ]
common_frechet: x = [ ζ | λ_core (E, ncore-1) | λ_cm (C, ncm_cm = (D-1)*L) | λ_level (F, ncm_level = L) ]
```

`ζ` is the scalar/intercept term (task's "[scalar/intercept if present | E | C]" — here it is always
present, one scalar). `common_frechet.economic_range == flexible_cm.economic_range` (both are
`2:1+ncore-1`, `ncore` shared meaning in both) and `common_frechet.cm_range` uses the same
per-threshold `(nO, L)` block layout as `flexible_cm.cm_range` (`R`-congruence, cumulative-basis
suffix-sum lookup) — common Fréchet adds only the trailing `(F)` level range
`2+ncore-1+ncm_cm : 1+ncore-1+ncm_cm+L`.

This ordering was ALREADY correct pre-refactor (both `dual_index!` implementations sliced `x`
identically for the shared E/C prefix) — this task's contribution is making the CODE that consumes
this layout call the same functions for both families, not changing the layout itself.

## Ranges (as implemented after this task's refactor)

| range | flexible_cm | common_frechet |
|---|---|---|
| scalar | `x[1]` | `x[1]` |
| economic (E) | `x[2 : 1+ncore1]` | `x[2 : 1+ncore1]` (same `ncore1 = ncore-1`) |
| CM (C) | `x[2+ncore1 : 1+ncore1+ncm]` | `x[2+ncore1 : 1+ncore1+ncm_cm]` |
| Fréchet level (F) | n/a | `x[2+ncore1+ncm_cm : 1+ncore1+ncm_cm+ncm_level]` |
| total dimension | `1 + ncore1 + ncm` | `1 + ncore1 + ncm_cm + ncm_level` |

Both `dual_index!` methods (post-refactor) slice their own `λ_cm`/`λ_level` views from `x` according
to this table, then call the SAME shared functions on the sliced views:

```julia
# flexible_cm (cm_lookup_kernels.jl, post-refactor)
function dual_index!(st::CMLookupState, x)
    ncore1 = st.ncore - 1
    λ_cm = @view x[2+ncore1:1+ncore1+st.ncm]
    economic_forward_into_arg0!(st, x)      # [E]
    cm_forward_contribution!(st, λ_cm, st.method)   # [C]
    return st.arg0
end

# common_frechet (cm_frechet_lookup_kernels.jl, post-refactor)
function dual_index!(st::CMFrechetLookupState, x)
    ncore1 = st.ncore - 1
    λ_cm    = @view x[2+ncore1:1+ncore1+st.ncm_cm]
    λ_level = @view x[2+ncore1+st.ncm_cm:1+ncore1+st.ncm_cm+st.ncm_level]
    economic_forward_into_arg0!(st, x)          # [E] -- SAME function as flexible_cm
    cm_forward_contribution!(st, λ_cm, :suffix) # [C] -- SAME function as flexible_cm
    frechet_level_suffix_sums!(st.P_level, λ_level)         # [F] -- extension only
    frechet_level_forward_sum!(st.level_contrib, st.bins, st.D, st.P_level)   # [F]
    # ... const_term / arg0 -= level contribution ...        # [F]
    return st.arg0
end
```

`economic_forward_into_arg0!`/`cm_forward_contribution!` and their transpose-side counterparts
`economic_transpose_into_g1_and_gE!`/`cm_transpose_into_g!` (all four in `cm_lookup_kernels.jl`) are
duck-typed on the field subset both structs share (`obj`/`ncore`/`core_cf_ref`/`econ_ws`/
`econ_ws_for`/`econ_buf`/`xsub`/`arg0`/`nO`/`L`/`λmat_block`/`R`/`λmat_ext`/`bins`/`refIndex1`/
`origins`/`cm_contrib`/`arg1`/`hist_h`/`hist_partials`/`Hpre`/`g_block`/`g_stored`) — the same
no-abstract-supertype-needed pattern `hessian_cm_structured!` and
`_verify_inner_solution_operator_cm_core` already used for the Hessian/verification cores.

## Requirements checked

- `frechet.economic_range == flexible_cm.economic_range`: YES (`x[2:1+ncore1]`, same `ncore`
  convention in both — confirmed by both `dual_index!` methods slicing identically before calling
  the shared `economic_forward_into_arg0!`).
- `frechet.cm_range == flexible_cm.cm_range` (same per-threshold block structure): YES — both call
  `cm_forward_contribution!`/`cm_transpose_into_g!` with `method=:suffix` on an `(nO,L)`-shaped
  `λmat_block`; common Fréchet's range is simply shorter in absolute column count
  (`ncm_cm = (D-1)*L` vs flexible_cm's `ncm = nO*L`, since Fréchet's own CM sub-block excludes the
  reference origin the same way flexible_cm's does — `nO` is defined identically in both, `aug.ncm`
  differs only because Fréchet's total `aug.ncm` bundles CM+level together).
- Fréchet's shared prefix ordering matches flexible_cm's ordering: YES (see table above).
- Fréchet adds only a trailing (F) range: YES — `ncm_level = L` appended after the CM block, no
  interleaving.

No family-specific hard-coded offsets remain in the refactored `dual_index!`/FG-functor pair (each
computes its own `ncore1`/slice bounds from its own struct fields, which is not a "hard-coded
offset" in the sense the task warns against — those bounds are genuinely per-instance data, not
magic numbers).
