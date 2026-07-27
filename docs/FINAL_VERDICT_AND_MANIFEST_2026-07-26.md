# Shared Inner-FG Operator and Verification — Final Verdict — 2026-07-26

Branch `port/shared-inner-fg-operator-and-verification-2026-07-26`, base
`production/fullA-exact@f1fa8e7`. Not merged, not pushed to origin — a local branch, per this
project's standing rule to confirm before either.

## Deliverables produced

- `docs/ADDENDUM_SHARED_ECONOMIC_FG_2026-07-26.md` (verbatim task prompt, preserved)
- `docs/SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md`
- `docs/UNRESTRICTED_ALLOCATION_FREE_FG_FINAL_GATE_2026-07-26.md`
- `docs/FLEXIBLE_CM_OPERATOR_FG_FINAL_GATE_2026-07-26.md`
- `docs/COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md`
- `docs/CM_MEANZC_OPERATOR_FG_PORT_2026-07-26.md`
- `docs/ORIGIN_ZC_OPERATOR_FG_PORT_2026-07-26.md`
- `docs/OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md`
- `docs/NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md`
- `docs/FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`
- This file (final verdict + manifest + release structure)

New production code: `economic_operator.jl`, `zc_restriction_operator.jl`,
`cm_originzc_lookup_kernels.jl`/`cm_originzc_lookup_production.jl`,
`cm_meanzc_lookup_kernels.jl`/`cm_meanzc_lookup_production.jl`, `no_dense_g_counters.jl`,
`operator_verification.jl`, plus targeted edits to `cm_hessian_architectures.jl` (new
`OriginZCCoreHessCtx`/`CMBinHessCtx` fields, one include-order bugfix),
`cm_originzc_production.jl`/`cm_meanzc_production.jl` (dispatch wiring). New tests:
`test_originzc_operator_correctness.jl`, `test_meanzc_operator_correctness.jl`,
`test_operator_verification_originzc.jl`.

## Commits (release structure)

```
fea8e9a  Branch setup: adopt finish-five-family-optimization-stack chain, save addendum verbatim
b399de4  Fix include-order gap: cm_hessian_architectures.jl -> compressed_live.jl
74c1450  Job 1: shared economic operator + origin-ZC [E|Z] operator FG (new, opt-in)
47daf6f  Job 1: CM+ZC [E|C|Z] operator FG (new, opt-in) + D=20 perturbation-point fix
ae2a790  Real D=20/W=80,000 correctness gates: origin-ZC and CM+ZC operator FG both ALL PASS
3969acf  Job 2: operator-based verification proof-of-concept for origin-ZC
<this commit>  Final deliverable docs
```

Every commit above is individually gated (each carries its own D=4 and, where applicable, D=20
evidence in its message) — no commit depends on a later one for correctness.

## Final verdict (task's own required format)

```
ECONOMIC_FG_BACKEND =
    unrestricted:compressed_operator (inherited, unchanged this branch)
    flexible_cm:dense (unchanged this branch — CMLookupState's own E-block still dense BLAS.gemv!)
    common_frechet:dense (unchanged this branch — CMFrechetLookupState's own E-block still dense)
    cm_plus_zc:compressed_operator (NEW this branch, opt-in, NOT default)
    zc_only:compressed_operator (NEW this branch, opt-in, NOT default)

RESTRICTION_FG_BACKEND =
    unrestricted:not_applicable
    flexible_cm:cm_lookup (inherited default, unchanged)
    common_frechet:dense_reference (inherited default, unchanged -- cm_frechet_lookup available opt-in)
    cm_plus_zc:operator (NEW this branch, opt-in, NOT default -- :dense_reference remains default)
    zc_only:operator (NEW this branch, opt-in, NOT default -- :dense_reference remains default)

VERIFICATION_BACKEND =
    unrestricted:dense_reference (unchanged; no operator verification built for this family)
    flexible_cm:dense_reference (unchanged; skip_cm_fill_ref toggle intact, NOT removed)
    common_frechet:dense_reference (unchanged)
    cm_plus_zc:dense_reference (unchanged; no operator verification built)
    zc_only:dense_reference (production default, unchanged) | operator (validated standalone
        proof-of-concept, verify_inner_solution_operator_originzc!, NOT wired as the production
        dispatch default)

FULL_G_MATERIALIZATION =
    present_flexible_cm,common_frechet (economic-block E still dense for these two families)
    -- zero_all_production_families is FALSE; see NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md for the
       live counter proof of what IS zero (origin-ZC and CM+ZC's own operator-mode FG, when
       fg_backend/inner_fg_backend=:operator is explicitly selected -- not yet the default for
       either family) and the explicit list of what this proof does NOT cover.

GENERIC_DENSE_FG_CALLS = 2 (live-measured, one dense-reference inner solve each for origin-ZC and
    CM+ZC in a real D=4 sequence -- see NO_DENSE_G_RUNTIME_PROOF_2026-07-26.md; this is a
    demonstration count from this session's own validation run, not a cumulative production count)

DENSE_REFERENCE_VERIFICATION_CALLS = 0 (in the operator-mode sequences measured; dense verification
    remains the PRODUCTION DEFAULT for every family including origin-ZC, so in ordinary production
    use today this count would be non-zero -- see OPERATOR_BASED_INNER_VERIFICATION_2026-07-26.md)

PRODUCTION_MERGE = port_ready_not_merged
```

## Why `port_ready_not_merged`, not `partial_merge` or `merged_all`

Every commit passes its own real, gated correctness evidence (D=4 machine-precision agreement for
five families' worth of configs; real D=20/W=80,000 machine-precision agreement for the two new
operator families). Nothing has been merged into `production/fullA-exact` or pushed to `origin` —
per this project's own standing rule (`feedback-confirm-before-pushing-to-real-remote-2026-07-25`),
that requires explicit user confirmation, not inferred from a task description alone. The branch is
genuinely ready to be reviewed for merge (clean history, gated commits, honest gaps documented) but
merge itself was not authorized or attempted this session.

## What a next session should do first (priority order, mirrors this branch's own inherited
## discipline of smallest-scoped-item-first)

1. Performance A/B for origin-ZC and CM+ZC's operator FG (this session's largest gap — see
   `FIVE_FAMILY_OPERATOR_FG_PERFORMANCE_AB_2026-07-26.md`) — needed before either family's default
   can responsibly flip.
2. Root-cause `CMFrechetLookupState`'s real 14MB/callback allocation precisely (bisect the
   assembled function body, or try a type-parameterized struct instead of `obj::Any`) — see
   `COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md`'s own recommended follow-on.
3. Extend operator-based verification to CM+ZC (straightforward, reuses existing pieces), then to
   flexible CM/common Fréchet (requires the E-block retrofit below first).
4. Retrofit flexible CM's and common Fréchet's own economic-core block to the shared
   `economic_forward!`/`economic_transpose!` operator — the addendum's "harder ask", real
   correctness-sensitive surgery on already-shipped-default code, deliberately deferred by both
   this branch and its inherited predecessor.
5. Only after 1-4: the full-codebase dense-G-consumer audit (task §13) and the D=4 gate matrix's
   remaining untested dimensions (rectangular layout, non-last-omitted-destination, multiple
   restriction configs beyond what was tested here — task §14).

## Dropbox

Pushed to `dropbox:Gravity robustness/Analysis/Server Output/shared_inner_fg_operator_2026-07-26`
per this project's standing requirement (see provenance.txt in that folder for git branch/HEAD/log
at push time).
