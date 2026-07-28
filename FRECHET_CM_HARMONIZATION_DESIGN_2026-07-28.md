# Common Fréchet / flexible-CM harmonization: design (2026-07-28)

Branch: `refactor/harmonize-frechet-with-flexible-CM-2026-07-28`, from canonical production
`cdw/production/fullA-exact@7a185ec` (post Phase-1 no-H release).

## 0. Correction to the starting assumption

The prior session's architecture note (quoted in the task brief) reads as "flexible CM and common
Fréchet do not share one parameterized Hessian-callback implementation... genuinely duplicated
code." That is true of the **orchestration layer**, but a full inventory of all 18 named
pieces (methodology: read every function body on both sides, not just names/docstrings) found:

```
SHARED (one concrete function, both families dispatch to it):        7 / 18
SHARED kernels, DUPLICATED orchestration shell around them:           6 / 18
DUPLICATED (near-verbatim, no shared function exists):                4 / 18
Fréchet-only (the genuine CM-F piece):                                1 / 18
```

Already fully shared, no work needed: `economic_forward!`/`economic_transpose!`
(`economic_operator.jl`), `_fill_cm_HEE!` (H_EE), `economic_A_gradient!` (outer A-gradient
economic block), `CMBinHessCtx`/`build_cm_bin_ctx`/`compute_bin_indices` (CM bins+contrasts+
workspace), `winner_pair_cross_hessian_*` (H_EC kernels), and `prime_operator!` (the no-H bundle's
own priming, already literally one function for both families — Phase 1 did not need to touch this
and didn't break it).

**This changes the scope of the refactor**: it is not "extract shared code from two independent
implementations." It is "collapse ~12 near-duplicate orchestration functions down to one
parameterized set, keeping the state/type layer that is already shared exactly as it is." Much
smaller blast radius than the task brief's own framing implied.

## 1. What is genuinely duplicated (the actual refactor targets)

| # | Flexible-CM function | Common-Fréchet function | Divergence |
|---|---|---|---|
| CM transpose | `cumulative_backward_gradient!` (`cm_lookup_kernels.jl:284`) | `cumulative_backward_gradient_from_prefix!` (`cm_frechet_lookup_kernels.jl:66`) | identical 3-line loop; Fréchet hoists `prefix_sums!` out so the level block can reuse `Hpre` |
| `dual_index!` shell | `dual_index!(::CMLookupState,...)` (`cm_lookup_kernels.jl:386`) | `dual_index!(::CMFrechetLookupState,...)` (`cm_frechet_lookup_kernels.jl:197`) | economic block byte-identical (lines 394-408 / 206-220); Fréchet appends the level-block forward pass |
| H_EC assembly loop | `cm_hessian_architectures.jl:1019-1049` (+ threaded variant) | `cm_frechet_hessian.jl:97-115` (+ threaded variant) | same kernels, Fréchet's copy just omits flexible-CM's CM+ZC (`use_direct_hcz`) branch |
| H_CC block | `cm_hessian_architectures.jl:1059-1074` | `cm_frechet_hessian.jl:118-133` | verbatim, no shared function exists at all for this one |
| Hessian-callback builder | `archC_hess_cb_builder` (`cm_hessian_architectures.jl:1559`) | `archC_frechet_hess_cb_builder` (`cm_frechet_hessian.jl:265`) | identical 2-branch (`use_threaded_bins`) structure, identical shared `_prep_dual_index_for_archC!` call; only the fill-function name and an extra `level_targets` arg differ |
| Hessian fill body | `hessian_cm_structured!`/`_v2!` | `hessian_cm_frechet_structured!`/`_v2!` | steps 1-7 near-verbatim (unpack, `_fill_cm_HEE!`, bin tables, H_EC loop, H_CC loop); Fréchet appends 3 level blocks; flexible CM has the CM+ZC branch Fréchet lacks |
| Verification | `verify_inner_solution_operator_cm!` | `verify_inner_solution_operator_cm_frechet!` (`operator_verification.jl`) | CM portion verbatim; Fréchet appends the level block |
| FG entry point | `inner_loop_internal_cmlookup_production` | `inner_loop_internal_cmfrechetlookup_production` | steps (a)/(c)/(d)/(e) identical; only the lookup-state constructor and an `st.method` assert differ |
| KNITRO driver + FG callback | `inner_loop_KNITRO_cmlookup_production` + `_callbackEvalFG_inner_cmlookup!` | `inner_loop_KNITRO_cmfrechetlookup_production` + `_callbackEvalFG_inner_cmfrechetlookup!` | 36-line driver bodies line-for-line identical except the registered callback symbol; callbacks have identical 4-line bodies |

## 2. H_EC packing difference — checked directly, NOT a correctness divergence

Initial read of the two packing steps looked like a real algorithmic difference (flexible CM's
`pack_upper_cm_hessian!`, `cm_hessian_architectures.jl:945`, skips averaging for the H_EC block --
`i <= NCORE < j ? Hfull[i,j] : 0.5*(Hfull[i,j]+Hfull[j,i])` -- while Fréchet's own final loop,
`cm_frechet_hessian.jl:~243-248`, applies `0.5*(Hfull[i,j]+Hfull[j,i])` uniformly to every block).
**Checked directly against both full function bodies rather than assumed equivalent or different**:
Fréchet's H_EC block is filled by the identical pattern flexible CM uses
(`Hfull[1:NCORE,cols] .= block_ec; Hfull[cols,1:NCORE] .= transpose(block_ec)`,
`cm_frechet_hessian.jl:~211-212`) -- i.e. Fréchet ALSO explicitly writes the exact same value into
both triangles before packing, for H_EC and for its own new H_E,level/H_CM,level blocks alike
(`Hfull[j,level_off+l] = v; Hfull[level_off+l,j] = v`, etc.). Given that, `0.5*(v+v)` is bit-exact
equal to `v` in IEEE754 (no rounding difference from doubling then halving a finite value) --
Fréchet's blanket averaging and flexible CM's special-cased skip are **provably the same result**,
not two different algorithms. This is a harmless micro-optimization difference (flexible CM avoids
a few redundant flops), not a bug, and not something that needed root-causing.

**Consequence for this task**: still worth consolidating onto the one shared `pack_upper_cm_hessian!`
function (the task's literal-code-reuse goal), and the equivalence gates will still include an
explicit pre-refactor-Fréchet vs post-refactor-Fréchet row to confirm this in practice (not just by
this hand-argument) -- but it should be reported as a **confirmed no-op consolidation**, not framed
as fixing a latent correctness bug. (Recorded here specifically because the *initial* read looked
like a real divergence and the project's own standing lesson is to verify such things by
reconstructing and diffing before asserting a discrepancy, rather than reasoning about it from the
code's surface structure alone.)

## 3. Target architecture (mapping the task's abstract spec onto the real code)

```
CMOperatorBundle{EconomicState, CMState, Extension}
    EconomicState = the existing CompressedFactual / core_cf_ref plumbing (already shared)
    CMState       = CMBinHessCtx (already shared, ONE type/constructor for both families)
    Extension     = NoExtension()          -- flexible CM (and CM+ZC, unaffected by this task)
                  | CMFrechetExtension(level_targets, ...)  -- common Fréchet
```

`CMFrechetExtension` (new, narrow type) owns exactly the genuinely Fréchet-only state (item 18 in
the investigation): `level_targets::Vector{Float64}` (currently threaded as a bare positional
argument everywhere — this is the one piece of "should be a field, isn't yet" cleanup this task
does), plus the level-block scratch that `cm_frechet_hessian.jl`/`_threaded.jl` currently allocate
**fresh on every Hessian call** (`Wtab`, `T1`, `Esum_wb`, `colsum`, `Hraw_cmlevel` —
`cm_frechet_hessian.jl:142,149,168,170,201` and the threaded twins) instead of storing them
persistently the way `CMBinHessCtx` already does for the shared CM state. Moving these into
`CMFrechetExtension` as persistent fields is both the "no duplicated economic/CM state" requirement
(task §10) and a small, free allocation-reduction (these were being reallocated per Hessian call).

```julia
mutable struct CMFrechetExtension
    level_targets::Vector{Float64}
    # persistent scratch, sized once from cctx.L/nO at construction, reused across calls:
    Wtab::Matrix{Float64}
    T1::Vector{Float64}
    Esum_wb::Matrix{Float64}
    colsum::Vector{Float64}
    Hraw_cmlevel::Matrix{Float64}
end
```

Shared pipeline (task §11), concretely:

```
economic_forward!(...)                          [already shared, unchanged]
  -> shared CM forward (apply_contrast!/suffix_sums!/cumulative_forward_contribution!)  [already shared]
  -> extension forward: frechet_level_suffix_sums!/frechet_level_forward_sum! if Extension !== NoExtension()
  -> Psi/Psi'                                    [unchanged]
economic_transpose!(...)                         [already shared, unchanged]
  -> shared CM transpose (ONE cumulative_backward_gradient! -- see §1, consolidating the
     hoisted-prefix-sums variant into the single shared function, parameterized by whether the
     caller wants Hpre back)
  -> extension transpose: frechet_level_backward_gradient! if applicable
```

Hessian (task §11):

```
shared H_EE      : _fill_cm_HEE!                          [already shared -- no change]
shared H_EC      : winner_pair_cross_hessian_* + ONE pack_upper_cm_hessian! call
                    [consolidates the two near-duplicate assembly loops in item 1's table + the
                    packing divergence in §2]
optional H_EF     : the level block's E-cross term, dispatched only when Extension !== NoExtension()
shared H_CC      : NEW extraction -- item 13 has no shared function today; this task adds one
                    (`fill_cm_HCC!`, mirroring `_fill_cm_HEE!`'s naming), used by both families
optional H_CF     : level-CM cross term, extension-only
optional H_FF     : level-level term, extension-only
```

One shared Hessian-callback builder (`archC_hess_cb_builder`, extended to accept an
`extension::Union{Nothing,CMFrechetExtension}` kwarg) replaces both `archC_hess_cb_builder` and
`archC_frechet_hess_cb_builder` — this directly fixes the exact class of bug the no-H task's own
session already hit once (`archC_frechet_hess_cb_builder`'s serial branch missing a dispatcher fix
that was applied to the shared function but not propagated): after this task, there is only one
function to fix.

One shared FG entry point (`inner_loop_internal_cmlookup_production`, parameterized the same way)
replaces the two near-identical entry points + two near-identical KNITRO drivers + two near-identical
FG callbacks in item 17's table.

## 4. Fixed-theta immutability (task §12)

Already satisfied for the shared CM state (`CMBinHessCtx`'s bins/contrasts are built once by
`build_cm_bin_ctx` and never rebuilt inside a solve). The one place this task must actively fix
(not just verify) is the level-block scratch reallocation noted in §3 above (currently reallocated
per Hessian call, will become a persistent `CMFrechetExtension` field, built once). Gate:
`cm_feature_rebuilds_due_to_A_or_gp = 0` (already the shared-state's own tracked invariant, see
`cm_feature_immutability_counters.jl`) plus a new `frechet_level_rebuilds_due_to_A_or_gp = 0`
counter on the same struct, incremented if `CMFrechetExtension`'s scratch fields are ever
reallocated after construction.

## 5. No-H invariant during refactor (task §13)

Both families already construct `OperatorPsiBundle` (Phase 1). This refactor touches none of that
type or `prime_operator!` — it only consolidates the code that runs *after* priming (the Hessian
callback and the FG dispatch), which already operates on the no-H bundle for both families. No
dense bundle is introduced as a bridge; the existing `_dense_H_or_nothing`/dispatch pattern
(already shared, item 3/16) continues to serve both `OperatorPsiBundle` and the unchanged
`PsiObjectiveBundleImplicit` dense-reference path exactly as it does today.

## 6. Commit plan (task §16)

1. Extract `fill_cm_HCC!` (new shared H_CC function, item 13 — the one place with literally no
   existing shared function) and switch both families' Hessian bodies to call it.
2. Consolidate the H_EC assembly loop + packing (§2) onto flexible CM's `pack_upper_cm_hessian!`
   for both families; delete Fréchet's inline averaging loop.
3. Add `CMFrechetExtension` type + persistent scratch construction; thread it through
   `build_cm_frechet_production_context` in place of the bare `level_targets` argument.
4. Parameterize `archC_hess_cb_builder`/`hessian_cm_structured!`/`_v2!` with an
   `extension::Union{Nothing,CMFrechetExtension}` kwarg carrying the 3 level-block calls;
   delete `archC_frechet_hess_cb_builder`/`hessian_cm_frechet_structured!`/`_v2!`.
5. Parameterize `inner_loop_internal_cmlookup_production` + its KNITRO driver + FG callback the
   same way; delete the Fréchet-specific triplet.
6. Consolidate the two `dual_index!` methods and the two CM-transpose functions (§1) into one
   parameterized version each.
7. Delete now-dead code (`archC_frechet_hess_cb_builder`, `hessian_cm_frechet_structured!`/`_v2!`,
   `inner_loop_internal_cmfrechetlookup_production`, `inner_loop_KNITRO_cmfrechetlookup_production`,
   `_callbackEvalFG_inner_cmfrechetlookup!`, `cumulative_backward_gradient_from_prefix!`,
   the duplicate `dual_index!(::CMFrechetLookupState,...)` method, Fréchet's inline H_EC averaging
   loop) once nothing references it.
8. Tests + manifests + this task's equivalence/dispatch-proof gates.

Each commit keeps both families passing their existing D=4 gates before moving to the next step —
this is a mechanical consolidation of already-parallel code, not a redesign, so regressions should
surface immediately and locally rather than needing a full rebuild to diagnose.

## 7. Open question flagged, not resolved here

Per the task brief's own historical note (`SESSION_MASTER_VERDICT_2026-07-28.md`): common Fréchet's
priming-side `skip_fill_safe_frechet` gate is hardcoded `false`, unlike the other 3 restricted
families, because an earlier attempt to enable it produced inconclusive/contradictory D=20 evidence.
This harmonization does NOT flip that gate as a side effect — it only consolidates the
*orchestration* code. Whether making Fréchet call the exact same gated closure flexible CM already
uses (rather than its own copy) makes it safe to also enable the skip is a separate numerical
question this task will check (via the equivalence gates below) and report on, but not decide by
assumption.

## Verdict

```
FRECHET_STRUCTURE (pre-refactor, measured) = 7_shared + 6_shared_kernel_duplicated_shell + 4_duplicated + 1_frechet_only
REFACTOR_SCOPE = orchestration_layer_consolidation, not full reimplementation
H_EC_PACKING_DIFFERENCE = confirmed_no_op (bit-exact equivalent, verified by reading both fill
    sites' explicit both-triangle writes; consolidation is a style/perf cleanup, not a fix)
DESIGN_STATUS = ready_to_implement
```
