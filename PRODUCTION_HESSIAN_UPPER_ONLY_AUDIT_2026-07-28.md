# Production Hessian upper-only audit — 2026-07-28

Base: `campaign/five-family-bounds-2026-07-28@93f26df7`. Scope: the packed-upper KNITRO Hessian
handoff for all 5 families — distinct from, and independent of, the legacy `obj.H = [K|ones|G]`
container audited separately (`LEGACY_CC_H_G_CONSTRUCTOR_AND_CALLSITE_AUDIT_2026-07-28.md`). Method:
a dedicated read-only research pass (`grep` for `symmetrize`/`mirror`/`transpose`/`0.5 *`/packed-upper
patterns across the whole worktree, every hit traced to its containing callback and classified) —
zero matches outside `cc_algo/` and `full_aod_diag/d4_exact/`.

## 0. What KNITRO actually receives

All 5 families register their Hessian callback via
`KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, cb_fn)` — KNITRO's dense **packed upper
triangle** convention (length `n(n+1)/2`, row-major over `i≤j`).

| Family | `hess_cb_builder` | Final structure before packing |
|---|---|---|
| Unrestricted | `_callbackEvalH_inner_compressed!` | packed vector written directly, no dense intermediate |
| Flexible-CM / CM+ZC | `archC_hess_cb_builder` → `hessian_cm_structured!`/`_v2!` | dense `n×n` scratch `Hfull`, block-assembled (mix of mirrored and independently-accumulated blocks), packed via a final `0.5*(Hfull[i,j]+Hfull[j,i])` averaging loop |
| Common-Fréchet | `archC_frechet_hess_cb_builder` → `hessian_cm_frechet_structured!`/`_v2!` | same pattern, `Hfull` extended with the level block |
| Origin-ZC | `archA_partitioned_hess_cb_builder` | dense `n×n` scratch `∂∂f_∂∂x`, block-assembled, packed via a **plain copy** (`evalResult.hess[k]=∂∂f_∂∂x[i,j]`, **no averaging step exists**) |

`cc_algo/PsiObjectiveBundle.jl::hessian!` (the `:dense_reference`-only base implementation) has
**no mirror pattern at all** — one `BLAS.gemm!('T','N',...)` fills the whole Gram matrix, both
triangles independently computed by BLAS, then packed upper-only.

## 1. Findings, by classification

### DEAD_CODE (unconditionally safe to delete, no coupled change needed)

**`cm_hessian_architectures.jl:1372`** (origin-ZC, production `archA_partitioned_hess_cb_builder`):
`@views ∂∂f_∂∂x[NCORE+1:n, 1:NCORE] .= transpose(HER)`. This family's packing loop
(lines 1403-1409) is a **plain copy that only ever reads `i≤j`** — no averaging step exists
downstream at all. The mirror writes into `∂∂f_∂∂x[NCORE+1:n, 1:NCORE]`, a region where
`row > NCORE ≥ col` for every entry — strictly lower-triangular, **never read** by the packing
loop or anything else in the callback body. Provably dead computation. **Fixed this session** —
see `PRODUCTION_HESSIAN_UPPER_ONLY_RELEASE_2026-07-28.md`.

### UPPER_ONLY_THEN_MECHANICALLY_MIRRORED (coupled to a downstream no-op average — both sides must change together)

13 sites across flexible-CM, CM+ZC, and common-Fréchet (serial + production-default threaded
variants): H_EE's dual-write unpack (`core_exact_hessian.jl:947-948`, shared by all 4 restricted
families), H_EC (`cm_hessian_architectures.jl:953`, `cm_hessian_threaded.jl:262`,
`cm_frechet_hessian.jl:108`, `cm_frechet_hessian_threaded.jl:103`), H_EM
(`cm_hessian_architectures.jl:756`), and common-Fréchet's H_E,level / H_CM,level dual-writes
(`cm_frechet_hessian.jl:170-172,188-190,211-212`, `cm_frechet_hessian_threaded.jl:157-158,175-176,197-198`).
Every one of these writes the *same scalar or block* into both the upper position and its
transpose — **by construction, not by later verification** — which every family's final
`0.5*(Hfull[i,j]+Hfull[j,i])` packing loop (`cm_hessian_architectures.jl:981-990`,
`cm_hessian_threaded.jl:285-292`, `cm_frechet_hessian.jl:233-242`,
`cm_frechet_hessian_threaded.jl:218-225`) then re-reads and averages back to exactly the same
value — a provable no-op **for these specific blocks**, at real (not negligible) compute cost:
one full extra write plus one extra add-and-multiply per matrix entry in these blocks, every
Hessian callback, every KNITRO iteration, for the production-default backends of 3 of the 5
families.

**Not independently deletable** — `cm_hessian_architectures.jl:946-950`'s own bug-fix comment
(from an earlier session) documents exactly why: before that fix, the lower half of the H_EC
block was left at `fill!(Hfull,0.0)`'s zero, and the average silently halved every entry — a real,
previously-shipped-then-fixed bug. Deleting a mirror write **without** simultaneously changing the
packing loop to stop re-reading the (now-zero) lower entry for that same block would reintroduce
that exact bug. The correct fix restructures the **packing loop**, not the mirror writes in
isolation: split it so blocks that are exactly mirror-symmetric by construction (H_EE, H_EC, H_EM,
H_E,level, H_CM,level) are packed as `h[k]=Hfull[i,j]` (no read of the lower entry, hence no need
to even write it), while blocks that are genuinely independently accumulated (H_CC, H_level,level
— see below) keep the full averaging. **Fixed this session for flexible-CM/CM+ZC (both serial and
production-default threaded variants)** — see the release doc. Common-Fréchet's four sites are
structurally identical and are the natural next target but were not reached this session (see
"not completed" below).

### UPPER_AND_LOWER_INDEPENDENTLY_ACCUMULATED (do not touch)

H_CC (`cm_hessian_architectures.jl:962-979`, `cm_hessian_threaded.jl:266-283`,
`cm_frechet_hessian.jl:111-127`, `cm_frechet_hessian_threaded.jl:106-122`) and H_level,level
(`cm_frechet_hessian.jl:216-231`, `cm_frechet_hessian_threaded.jl:202-216`): the outer loop runs
the **full** `l,lp ∈ 1:L × 1:L` grid, so the `(l,lp)` and `(lp,l)` blocks are each computed from an
independent prefix-sum evaluation (`CT[o,p,l,lp]` vs `CT[p,o,lp,l]`), not copied from one another.
Analytically equal, not mechanically identical — the averaging here absorbs genuine (if small)
floating-point path-order differences and must be retained. All BLAS-`gemm!`-based dense-fallback
blocks (H_EE/H_EM/H_MM's `:dense_reference`/tied-winner fallback, `zc_restriction_gram!`'s
H_ZZ/H_RR, origin-ZC's dense fallback) are classified here too, conservatively — a single
`BLAS.gemm!('T','N',A,A,...)` computes the full square from one Gram product, so both triangles
are almost certainly bit-identical in practice, but this is a BLAS-implementation property, not a
language guarantee, and confirming it would require a runtime check this audit did not perform.
Left untouched, as the task requires ("Do not remove a reconciliation step where upper and lower
entries were genuinely accumulated independently").

### TEST_ONLY / superseded

`cm_hessian_architecture_threaded.jl` ("Section 10 experimental" thread variant) and
`cm_hessian_architecture_interval.jl`'s mirror/average sites (the non-default `cm_basis=:interval`
config — no production driver was found to set this) both contain the same mirror+average pattern
but are not reachable from `cm_production_bundle.jl`/`cm_config.jl`/`cm_originzc_production.jl`.
`compressed_inner_alt_solvers.jl`'s `:denseaccum`/`:hvp` unrestricted-family options have no
callers found anywhere in the tree. `inner_diagnostics.jl`'s `Symmetric(...)` condition-number
check is a diagnostic, not a KNITRO callback. None of these were touched.

## 2. Not completed this session (honest gap)

Common-Fréchet's 4 mirror/average sites (H_EC, H_E,level ×2 branches, H_CM,level — both serial
`cm_frechet_hessian.jl` and production-default-threaded `cm_frechet_hessian_threaded.jl`) are
structurally identical to the flexible-CM/CM+ZC sites this session did fix, and are the direct
next target — left undone due to session time, not because they are harder or riskier. See
`PRODUCTION_HESSIAN_UPPER_ONLY_RELEASE_2026-07-28.md` for exactly what was and wasn't changed.
