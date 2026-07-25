# Core-Hessian family coverage table — 2026-07-25

Task §3-4. Every production family whose inner moment vector contains the common Ricardian
economic core `E`, discovered by source search of the canonical production tip (branch
`port/shared-winner-pair-core-hessian-production-2026-07-25`, based on `origin/production/fullA-exact`
@ `39b89c5`), with its resolved H_EE path before and after this port.

| Family | Public driver | Pre-port H_EE path | Post-port H_EE path | Verified active |
|---|---|---|---|---|
| Unrestricted | `run_profile_checkpointed`/`run_polish_checkpointed` (`c10_d20_production_driver.jl`) → `_callbackEvalH_inner_compressed!` (`compressed_live.jl`) | `CS.hessian!` — dense `BLAS.gemm!` over the whole moment block (H_EE **is** the whole Hessian here) | `hessian_core_winner_pair!`/`winner_pair_hessian!` (shared module), called directly on the packed output — no dense round-trip, no restriction columns to carve off | Yes — D=4 gates (5/5 dual points, serial+parallel+direct_packed) + D=20/W=80,000 real-data smoke (this doc's companion timing doc) |
| Flexible CM (`cm_extension=:cm_only`) | `run_cm_upper_checkpointed` (`cm_checkpoint.jl`) → `hessian_cm_structured!`/`_v2!` (`cm_hessian_architectures.jl`/`cm_hessian_threaded.jl`) | Small dense `BLAS.gemm!`/`syrk!` on `E` alone, inside the larger Architecture-C Hessian | Shared `fill_core_hessian_upper!`, writing into `Hfull[1:NCORE,1:NCORE]` via the new `_fill_cm_HEE!` helper (one helper, both serial and threaded callers) | Yes — D=4 gates (both serial + production-default threaded-bins path) + pre-existing D=20/W=80,000/L=50 regression test (`test_cm_compressed_core.jl`, unmodified, incidentally now exercises the winner-pair backend since it's the new default) |
| CM+mean/ZC | `run_cm_upper_checkpointed`, `cm_extension≠:cm_only` (`cm_meanzc_production.jl`/`cm_meanzc_moments.jl`) | Same small dense gemm as flexible CM, but over a WIDER `NCORE_ext = ncore_econ+n_mean+n_pair` block (mean/pair columns folded into the same "economic" corner by this family's own column layout — NOT a separate CM-grid-style restriction block) | `_fill_cm_HEE!` partitions the widened block itself: true core (`ncore_core = ncore_econ` columns) via the shared winner-pair backend; the (core×mean/pair) and (mean/pair×mean/pair) corners via the SAME dense BLAS this family used before (unchanged, task §4.3's "retain existing method for non-core blocks") | Yes — D=4 gates, K_mean=1/K_pair=0 and K_mean=1/K_pair=1 |
| Origin-specific ZC | `run_originzc_upper_checkpointed` (`cm_originzc_production.jl`) → new `archA_partitioned_hess_cb_builder` | ONE monolithic dense `gemm!` over core+restriction(η) columns combined (`cc_algo/PsiObjectiveBundle.jl::hessian!`, Architecture A) | Partitioned: H_EE via shared winner-pair backend; H_ER via dense `gemm!` restricted to (core×η) columns, computed once; H_RR via dense `gemm!` restricted to (η×η) columns; H_RE never independently computed (`transpose(H_ER)`) | Yes — D=4 gates, K_mean=1/K_pair=0 and K_mean=1/K_pair=1 |

## Other families searched for and found NOT to qualify

- **Fixed Fréchet**: repo-wide `grep -i "FixedFrechet\|fixed_frechet"` returns zero hits anywhere on
  this branch (production or diagnostic) — not canonical production, correctly excluded per the
  task's own instruction to include it "only if it has actually become canonical production."
- No other family (beyond the four above) was found to reference the common core `E` block in any
  live production Hessian callback.

## Design note: how "insertion into a larger family Hessian" is handled

Every family already materializes a DENSE symmetric scratch matrix before packing into KNITRO's
row-major upper-triangle format (`obj.∂∂f_∂∂x` for unrestricted/origin-ZC's generic `hessian!`;
`cctx.Hfull` for CM/CM+meanZC's Architecture C). The shared `fill_core_hessian_upper!` writes into
a `@view` of that EXISTING dense scratch at the correct offset — Julia's own view indexing is the
local-to-global map, exact and allocation-free, with no hand-rolled index table that could drift
from a family's real packing convention. Each family's own pre-existing "pack dense → KNITRO
triangle" step is completely untouched.
