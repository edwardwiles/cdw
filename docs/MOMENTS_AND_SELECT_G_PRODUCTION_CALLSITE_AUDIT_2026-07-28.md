# `moments!` / `select_G_from_H` production callsite audit — 2026-07-28

Method: `grep -rn "select_G_from_H(\|obj\.moments!(" --include=*.jl .` across the whole worktree,
then each hit's containing function traced to determine whether it is reachable from a real
production entry point (`run_cm_upper_checkpointed`, `cm_originzc_checkpoint.jl`,
`evaluate_fullA_fast_compressed`, or their direct callers), an explicit `:dense_reference` gate, a
test/benchmark script, or genuinely dead.

## PRODUCTION_OPERATOR (reached in a real production run, `moment_representation=:operator`)

| Call site | Family | Frequency | What it's for |
|---|---|---|---|
| `cm_lookup_production.jl:124` (`moments_fn(...)`) | Flexible CM | once/inner solve | Publishes `core_cf_ref[]`; builds K, gravity, economic block dense (unconditional); CM-grid block **skipped** (`skip_fill_safe` re-enabled this session, §6 of master report) |
| `cm_meanzc_lookup_production.jl:80` (`obj.moments!(...)`) | CM+ZC | once/inner solve | Publishes `core_cf_ref[]`; builds K, gravity, economic block, CM-grid block — **all unconditional**, no skip mechanism exists for this family |
| `cm_frechet_lookup_production.jl:96` (`moments_fn(...)`) | Common Fréchet | once/inner solve | Same as flexible-CM's closure, but `skip_fill_safe` is hardcoded `false` in production (`cm_frechet_cplus.jl:142`) — economic **and** CM-grid **and** level blocks all unconditional |
| `cm_originzc_lookup_production.jl:101` (`obj.moments!(...)`) | Origin-ZC | once/inner solve | Publishes `core_cf_ref[]`; builds K, gravity, mean/pair restriction block — **all unconditional**, no skip mechanism exists |

Each of these 4 calls is paired 1:1 with exactly one `select_G_from_H(obj, obj.H)` call (the `G`
argument passed into the same closure) — so `PRODUCTION_SELECT_G_FROM_H_CALLS` = 4, matching
`PRODUCTION_MOMENTS_CALLS` = 4 exactly, once per restricted family per inner solve.

**Not on this list**: unrestricted. Confirmed (`compressed_live.jl::inner_loop_internal_compressed`)
it never calls `obj.moments!`/`select_G_from_H` at all — K/gravity are built via direct scalar/O(W)
formulas (`fill_K_directgp!`-equivalent inline, `compressed_gravity_raw`), and the economic state via
`build_economic_moment_state!`/`cf_build` directly. This was already true before this session
(pre-existing architecture) and is the reference this task's fix generalizes to the other 4 families
on the *Hessian* side (not yet on the priming side, see master report §5).

## EXPLICIT_DENSE_REFERENCE (reached only under `moment_representation=:dense_reference` or
`use_compressed_core=false`, an explicit opt-out never used in a real production campaign)

| Call site | Notes |
|---|---|
| `cm_hessian_architectures.jl:1135` (`inner_loop_internal_archgeneric`) | The generic dense-only inner-solve driver; reached only when a caller does not dispatch to a family's own lookup/operator priming path |
| `oracle_fast.jl:180` (`inner_loop_internal_profiled`) | Same role, oldest/most generic driver in the codebase |
| `cc_algo/inner_loop_functions.jl`, `cc_algo/outer_loop_functions.jl`, `cc_algo/local_sensitivity.jl`, `cc_algo/PsiObjectiveBundle.jl`, `cc_algo/KLObjectiveBundle.jl` | Base library dense-generic code every family's `PsiObjectiveBundleImplicit` construction still inherits its type/interface from; not itself a production hot-path call site |

## TEST_ONLY / DEAD_CODE

The remaining ~40 files matched by the same grep (`c1*`/`c8*`/`c9*`/`c10*`-prefixed diagnostic
scripts, `bench_*.jl`, `profile_*.jl`, `test_*.jl`, `diag_*.jl`, `legacy/*.jl`) are standalone
diagnostic/benchmark/legacy scripts, never imported by a production entry point. Two scripts
(`bench_cm_bintable_decomposition.jl`, `profile_archC_hessian_d20.jl`) call `build_bin_tables!` with
its **old** 3-positional-argument signature (`(cctx, E, w)`) and will now error (a `BoundsError`, not
a silent miscomputation) since this task's §4 changed that function's second argument from a
pre-sliced `E` view to the full `H` matrix — flagged here as a known, low-priority breakage in
non-production diagnostic scripts, not fixed this session (out of scope, no production impact).

## Requirement check

```
PRODUCTION_OPERATOR moments! calls           = 4   (target was 0 -- NOT met, see master report §5)
PRODUCTION_OPERATOR select_G_from_H calls    = 4   (target was 0 -- NOT met, same reason)
```

Both counts are **greater than zero**, an honest gap against the task's stated requirement. The
Hessian-callback-side dependency (the decisive legacy dependency named in the task) is fully
eliminated and gated; the priming-side dense economic-block fill is not, after a same-session
attempt to remove it was caught by the gates as a real regression and reverted rather than shipped
un-debugged.
