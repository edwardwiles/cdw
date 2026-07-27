# Five-Family Shared A-Gradient Gate — 2026-07-27

Task §4's ask: every production family calls the same shared `economic_A_gradient!` A-coordinate
loop, family wrappers append only their own family-specific outer-coordinate derivatives.

## Status: 3 of 5 wired (2 newly wired this session)

| Family | Entry point | Status | Evidence |
|---|---|---|---|
| ZC-only | `cm_originzc_production_gradient` | WIRED (inherited, pre-existing) | D=4 12/12, D=20 4/4, both bit-identical (inherited gates, re-confirmed by this session's own D=20 gate for CM+ZC using the identical pattern) |
| CM+ZC | `cm_meanzc_production_gradient` | **WIRED this session** | D=4 12/12 PASS (3 K_mean/K_pair configs x calib/legacy comparison), real D=20/W=80,000 4/4 PASS, bit-identical |
| Flexible-CM | `cm_production_gradient` | **WIRED this session** | D=4 16/16 PASS (2 L values x calib/perturbed x shared/legacy), real D=20/W=80,000 8/8 PASS, bit-identical |
| Common-Frechet | (its own gradient entry point) | NOT wired | Legacy `composite_gradient_at_fast` unchanged |
| Unrestricted | `c10_d20_production_driver.jl`'s own backend matrix (`_buffered`/`_pooled`/`_Aplus`/`_KBplus`) | NOT wired | Would require touching the large, actively-used production driver file; also structurally different (multiple parallel backend choices, not a single family wrapper) |

`get_or_build_econ_a_grad_ws` (the process-wide `EconomicAGradientWorkspace` cache keyed by `W`)
was moved from `cm_originzc_production.jl` to `shared_a_gradient.jl` this session — it was never
ZC-specific, and both newly-wired families needed the identical cache rather than a per-family
copy, consistent with this task's own "one shared implementation" principle applied to the
workspace cache itself, not just the gradient math.

## Why 2, not 5

Consistent with the priority order the inherited branch's own docs recommended ("pick the
simplest remaining family, gate it solidly, rather than spreading thin across all remaining
families with weaker gates on each"), this session wired the 2 next-simplest families (both follow
the identical established pattern: add `gradient_backend`/`econ_ws` kwargs, route the plain
`(g,A_od)` block through `economic_A_gradient!`, pass the family's own pre-folded `cache=`
unchanged) and gated each fully (D=4 AND real D=20, bit-identical) before moving to the next,
rather than wiring all remaining families with partial gating. Common-Frechet and unrestricted
remain honestly reported as not attempted.

## Verdict

```text
A_GRADIENT_WIRED =
    unrestricted:false
    flexible_cm:true (this session)
    common_frechet:false
    cm_plus_zc:true (this session)
    zc_only:true (inherited)
A_GRADIENT_GATES_D4 = flexible_cm:16/16 PASS, cm_plus_zc:12/12 PASS (both bit-identical)
A_GRADIENT_GATES_D20 = flexible_cm:8/8 PASS, cm_plus_zc:4/4 PASS (both bit-identical, real
    W=80,000, destination_sample=:exclude_row)
```
