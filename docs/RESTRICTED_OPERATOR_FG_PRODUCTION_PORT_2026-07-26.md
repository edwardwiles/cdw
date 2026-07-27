# Restricted Operator FG Production Port — 2026-07-26 (updated, this session)

## Status: flexible-CM allocation fix + default flip DONE and validated. Common Fréchet's
## matrix-free CM+level operator (`CMFrechetLookupState`) is DONE, validated D4+D20, but NOT
## flipped to default (allocation regressed ~21%, see §3). CM+ZC, origin-ZC operators NOT attempted
## this session (still scoped below). Two additional G-materialization findings surfaced and are
## documented in detail (§4) — one fixed (flexible CM), one deliberately NOT fixed this session
## (now fixed separately for the UNRESTRICTED family, see
## `docs/UNRESTRICTED_ALLOCATION_FREE_FG_PORT_2026-07-26.md`, addendum Part A).

This revises the prior session's draft of this doc (which reported the whole of task §5 as "NOT
ATTEMPTED"). This session picked up exactly the lowest-risk item that draft itself identified as
the right starting point — flexible CM's already-existing, allocation-regressed `:cm_lookup`
kernel — fixed its allocation regression, validated it at D=4 and real D=20/W=80,000, and flipped
it to the production default. Common Fréchet, CM+ZC, and origin-ZC operators remain unbuilt; see
§5 for the unchanged scoped follow-on for those three.

## 1. Flexible CM: allocation fix + default flip (DONE, validated)

### 1.1 What was wrong

The inherited `:cm_lookup` kernel (`cm_lookup_kernels.jl`/`cm_lookup_production.jl`,
`CMLookupState`) was numerically validated but shipped `AVAILABLE_BUT_NOT_DEFAULT`: at real
D=20/W=80,000/L=50, it was 9.6% faster than `:dense_reference` but allocated **12.6% MORE**
memory per complete inner solve — the allocation regression named in
`docs/RESTRICTED_LOOKUP_FG_PRODUCTION_PORT_2026-07-26.md` as the reason it was withheld.

Tracing `CMLookupState`'s per-FG-callback code path (`cm_lookup_kernels.jl`) found the allocation
sources named in the original task's own §5.5 checklist, concretely:
- `xsub = vcat(ζ, λ_core)` — a fresh vector on every FG callback.
- `apply_contrast(M, R) = R === nothing ? M : R * M` — a fresh matrix on every forward AND
  backward call (two per gradient-requested callback).
- `suffix_sums(nu)` — a fresh `(nO, L+1)` matrix every forward call (production's default
  `method=:suffix`/cumulative basis, not the `:interval` method the original file's comments
  focus on).
- `build_weighted_histogram(...)` — allocated its `nt` thread-local `(D, nbins)` partial buffers
  AND the final reduced `(D, nbins)` histogram FRESH every gradient callback.
- `interval_backward_gradient`/`cumulative_backward_gradient` — a fresh `(nO, L)` matrix every
  gradient callback (the latter also reallocating `prefix_sums`' own `(D, L)` buffer internally).
- `CMLookupState` itself was rebuilt from scratch on **every inner solve** (not every callback,
  but still avoidable per task §5.5's explicit "do not rebuild CMLookupState per solve" ask).

None of this was a new derivation — the identities were already correct and already validated;
the fix was purely mechanical: give every one of the above a persistent, pre-sized buffer on
`CMLookupState` and rewrite each step as an in-place `!`-suffixed operation. The original
allocating functions were kept unchanged (they remain the public API three other production files
call directly, once per outer point, not hot: `cm_meanzc_production.jl`, `cm_frechet_cplus.jl`,
`lfix_cm_aware.jl`); new `apply_contrast!`/`suffix_sums!`/`build_weighted_histogram!`/
`prefix_sums!`/`interval_backward_gradient!`/`cumulative_backward_gradient!` siblings were added
purely for `CMLookupState`'s own hot callback.

`CMLookupState` itself is now built **once per `cctx`** (cached on a new `cctx.cmlookup_st` field,
the same "build once per campaign, reuse every inner solve" pattern this struct's own
`core_ws`/`tls` fields already use) rather than rebuilt on every call to
`inner_loop_internal_cmlookup_production`; `st.n_fg_calls` is reset to 0 at the top of each solve
so the returned "FG calls this inner solve" count is unaffected by the caching.

Thread count (`nthreads_use`, sizing the histogram's thread-local buffers) now defaults to
`Threads.nthreads()` — the live worker policy, matching `cm_hessian_architecture_threaded.jl`'s
own pattern — instead of the previous hardcoded `1`.

### 1.2 Validation (real, this session)

**Correctness**, `test_phaseB1_cmlookup_production_correctness.jl`, both `d4` and `d20` modes,
through the real production entry points (`archC_verified_state`, D=4 square + real
D=20/W=80,000), both contrast bases, calibration + perturbed points:

```
D=4:  ALL PASS (0 failures) -- L in {10,20,50}, contrasts in {anchored,orthonormal}
D=20: ALL PASS (0 failures) -- L=50, contrasts in {anchored,orthonormal}, real W=80,000/KNITRO
```

ζ*/λ*/m_weights/Delta_dual/downstream-gradient agreement between `:dense_reference` and
`:cm_lookup` throughout: ~1e-13 to ~1e-18 (machine precision given different KNITRO iteration
paths — not `==`, per this codebase's own established tolerance convention).

**Performance**, `test_phaseB1_performance_gate.jl`, real D=20/W=80,000/L=50 calibration point,
5 reps, `workers ∈ {1, 4, 8, 10, 20}` (`julia -t N`):

| workers | dense median (s) | lookup median (s) | speedup | alloc ratio (dense/lookup) |
|---|---|---|---|---|
| 1  | 1.1488 | 0.9976 | 1.152x | 1.000x |
| 4  | 1.2922 | 1.0123 | 1.277x | 1.000x |
| 8  | 1.2056 | 0.9960 | 1.210x | 1.000x |
| 10 | 1.1085 | 1.0004 | 1.108x | 1.000x |
| 20 | 1.6218 | 1.0037 | 1.616x | 1.000x |

`:cm_lookup` is faster than `:dense_reference` at **every** tested thread count (previously only
1.096x at a single thread count), and allocation is now **exactly at parity** with the dense
reference (previously 12.6% *worse*) — the measured bytes are common `archC_verified_state`/
KNITRO overhead shared identically by both backends, i.e. the lookup kernel itself is now
allocation-free relative to dense. `Delta_dual` agreement at every thread count: ~1e-17-1e-18.

### 1.3 Flip decision

Task §5.6 criteria, all satisfied with real evidence: D=4 + D=20 correctness pass; complete inner
solve faster (not just "within 5%") at every tested thread count; allocation falls materially
(parity, from a 12.6% regression); no stability regression (identical `nStatus`, `Delta_dual`
agreement ~1e-17); `full_G_materializations=0` for the CM block by construction.

**`CM_INNER_FG_BACKEND_DEFAULT` flipped from `:dense_reference` to `:cm_lookup`**
(`core_exact_hessian.jl:124`). `:dense_reference` remains available as an explicit override for
replication/emergency-revert.

**Scope check (important):** the flip only affects flexible CM's own `build_cm_production_context`
(the only real production call site that reads this global default). Common Fréchet's own
`build_cm_bin_ctx` call (`cm_frechet_level.jl`) was found to omit the `inner_fg_backend` kwarg
entirely — it would have silently inherited the new `:cm_lookup` default as a **stored label**
on its `cctx` (harmless behaviorally, since common Fréchet's own inner-solve functions,
`archC_frechet_base_state`/`archC_frechet_verified_state`, never read `cctx.inner_fg_backend` at
all — they unconditionally call `inner_loop_internal_archgeneric`), but dishonest for
manifest/diagnostic reporting. Pinned explicitly to `:dense_reference` there. CM+ZC already
hardcoded `:dense_reference` explicitly (pre-existing, unaffected). Origin-ZC never touches
`CMBinHessCtx` at all (unaffected).

**Post-flip confirmation**: `phase8_transformed_a_default_smoke.jl` (default kwargs, not
overrides, all four restricted families, real D=20/W=80,000) — **ALL PASS**, including origin-ZC
(kappa=0.0558, feasible-or-timelimit, no callback error).

## 2. Follow-on fix: skip the now-wasted dense CM-column materialization

### 2.1 The finding

`obj.moments!` for flexible CM (`wrap_moments_with_cm_archB`'s closure, `cm_hessian_architectures.jl`)
unconditionally called `fill_cm_columns_from_bins!` on every inner solve, densely materializing
the full `(W, (D-1)·L)` CM block into `obj.H` — at D=20/W=80,000/L=50, roughly 608MB — **regardless
of which FG backend was registered afterward**.

Tracing every consumer of those columns:
- `:cm_lookup`'s own FG callback (`CMLookupState`) never reads them — it recomputes the CM
  contribution entirely from bin lookups against `obj.H`'s *core* columns plus `cctx.Bidx`.
- Architecture C's Hessian callback (`hessian_cm_structured!`) never reads them either — it builds
  its own bin tables directly from `cctx.Bidx` and per-solve weights, independent of `obj.H`.
- `archC_base_state` (the workhorse most gradient/dual-bank calls use) never reads `obj.H` at all
  after the solve — it returns only `ζ*`, `λ*`, and `obj.arg1`.
- `archC_verified_state`'s post-solve independent check **does** read them: `CS.select_G_from_H`
  for the KKT-residual diagnostic, and the explicit `obj(inner_x, constr=...)` recompute (which
  goes through `PsiObjectiveBundleImplicit`'s own dense callable, itself a `BLAS.gemv!` against
  the *full* `H[:, 2:1+outer_constr_index]`, core+CM columns together) for `Delta_dual`.

So under `:cm_lookup`, for the common case (`archC_base_state` calls — the majority of inner
solves during outer optimization), that ~608MB dense block was built every single inner solve and
then never read by anything. Only `archC_verified_state`'s diagnostic gate genuinely needs it.

### 2.2 The fix (implemented, validated)

Added an optional `skip_cm_fill_ref::Union{Nothing,Ref{Bool}}` kwarg to
`wrap_moments_with_cm_archB`, defaulting to `nothing` (= always fill, the original, fully
backward-compatible behavior for every caller that doesn't pass it — the archB diagnostic builder
and various `c13_*`/`c14_*` benchmark scripts included). `build_cm_production_context` now
threads a shared `Ref(false)` through this kwarg AND stores the same `Ref` on a new
`cctx.skip_cm_fill_ref` field (mirroring the existing `core_cf_ref` cross-cutting-state pattern),
so `archC_base_state`/`archC_verified_state` (which only have `cctx`, not the closure) can toggle
the shared box:

- `archC_base_state`: sets `cctx.skip_cm_fill_ref[] = true` immediately before dispatching to
  `inner_loop_internal_cmlookup_production` **only** when `cctx.inner_fg_backend == :cm_lookup`,
  and resets it to `false` in a `finally` block (so it can never leak `true` into any other caller
  sharing the same `cctx`, regardless of success/failure/exception).
- `archC_verified_state`: **defensively** forces `cctx.skip_cm_fill_ref[] = false` immediately
  before its own dispatch, every call, independent of whatever any prior call left the shared ref
  set to — its own post-solve recompute needs the columns correctly filled, and this must not
  depend on caller-ordering discipline elsewhere.
- CM+ZC's `build_cm_meanzc_bin_ctx` gets an inert `Ref(false)` (its own `moments!` wrapper,
  `wrap_moments_with_cm_meanzc`, is a completely separate closure that doesn't accept or check
  this kwarg at all — the field exists purely so every `CMBinHessCtx` is uniformly non-`nothing`).

**Validation**: re-ran the full `test_phaseB1_cmlookup_production_correctness.jl` gate (D=4 and
real D=20/W=80,000) after this change — chosen specifically because it exercises
`archC_verified_state` (Delta_dual, downstream-gradient agreement), the exact path this change
could have silently corrupted if the reset logic were wrong.

```
D=4  post-skip-fill: ALL PASS (0 failures)
D=20 post-skip-fill: ALL PASS (0 failures), Delta_dual agreement ~1e-16 to ~1e-17, unchanged from pre-change
```

No regression: the verified-state diagnostic path still gets correctly-filled CM columns every
time; only `archC_base_state`'s calls (which never read them) now skip the materialization.

### 2.3 Why this matters more than it looks

This is a second, independent instance of the same "full G must never be materialized in
production hot paths" invariant (task §4) that the flip in §1 does not by itself satisfy — fixing
the FG *callback's own* consumption of a dense block (§1) does not stop something else upstream
from building that block anyway "just in case." The lesson generalizes: whenever a matrix-free
operator replaces a dense consumer, the *producer* side (the thing that used to feed the dense
consumer) needs its own audit, not an assumption that it was already conditioned on which
consumer is active.

## 3. Common Fréchet: matrix-free CM+level operator (DONE, validated, NOT flipped to default)

### 3.1 Math

Common Fréchet is flexible CM's `(D-1)*L` CM restriction moments plus `L` common-level-anchor
moments (`docs/FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md`, a prior session's own proof).
The level column at threshold `l` is `u'f_l(ω) - (u'1)F*(H_l)`, `u = 1_D/√D` — i.e. `(1/√D)·Σ_o
1{z_o(ω)<H_l} - √D·p_l`, summed over **all** `D` origins (not reference-differenced, unlike the CM
block's `(D-1)` columns). Forward/backward were derived fresh and cross-checked against two
independent existing pieces of code before writing any new kernel:

- **Forward**: `Σ_l λ_level[l]·G_level[s,l] = invsqrtD·Σ_o P_level[bin(s,o)] - Σ_l λ_level[l]·targets[l]`,
  where `P_level[k] = Σ_{l≥k} λ_level[l]` (suffix sum). The `Σ_o P_level[bin(s,o)]` term is
  *exactly* what `frechet_level_forward_sum!` (`cm_frechet_cplus.jl`, pre-existing, used unchanged
  by the Lfix outer-gradient path at the converged `λ*`) already computes — reused directly rather
  than re-derived, for the live in-solve `λ_level` instead of the converged one.
- **Backward**: `g_level[l] = -(invsqrtD/M)·Σ_{o=1}^D Hpre[o,l] + (targets[l]/M)·Σ_s dPsi[s]`, where
  `Hpre` is the **same** `(D,L)` prefix-sum-of-weighted-histogram buffer the CM block's own
  cumulative backward gradient already computes — shared between the two, one `prefix_sums!` call
  per callback instead of two. This is genuinely new (no prior backward-gradient code existed for
  this block at all, since Fréchet's inner FG was 100% dense before this session).

Both were verified via a standalone unit-style comparison (`debug_frechet_lookup_unit.jl`) — the
new `CMFrechetLookupState(x,g)` callable directly against the dense `obj(x,g)` callable at a fixed
point, no KNITRO involved — **before** trusting any KNITRO-solve-level result, per this project's
own "verify math independently before blaming/trusting a downstream numerical discrepancy"
discipline: `f`/`g` agreed to ~1e-16 to ~4.8e-16 at two test points, confirming the kernel math
itself was correct from the start.

### 3.2 A real bug found and fixed (KNITRO callback protocol, not kernel math)

The FIRST full KNITRO-solve correctness run failed **every single case** (48/48 checks), with
`Delta_dual` exactly `-0.0` in every case despite `nStatus=0` ("success") — a strong tell that the
inner solve wasn't actually iterating rather than a small numerical slip. The standalone unit test
above (run *because* of this suspicious pattern, before assuming the kernel math was at fault)
showed the kernel math was exact — isolating the bug to the KNITRO wiring. Root cause: the FG
callback (`_callbackEvalFG_inner_cmfrechetlookup!`) branched on `evalRequest.evalRequestCode ==
KNITRO.KN_RC_EVALFC`/`KN_RC_EVALGA` and returned an error code on anything else — but this
codebase's actual `KN_add_eval_callback(kc, true, ...)` registration pattern (confirmed by reading
the real, working `_callbackEvalFG_inner_cmlookup!` in `cm_lookup_live_knitro.jl`, not assumed)
calls ONE combined callback for both `f` and `g` every time, with no such branching at all — every
callback call was hitting the error branch. Fixed by mirroring the real, proven callback exactly
(including its `lower_limit` infeasibility guard, which the first draft also omitted).

### 3.3 Validation

**Correctness** (`test_phase52_frechet_lookup_correctness.jl`, mirrors
`test_phaseB1_cmlookup_production_correctness.jl`'s structure, through the real
`archC_frechet_verified_state` entry point):

```
D=4:  ALL PASS (0 failures) -- L in {10,20,50}, contrasts in {anchored,orthonormal}
D=20: ALL PASS (0 failures) -- L=50, contrasts in {anchored,orthonormal}, real W=80,000/KNITRO
```

ζ*/λ*/m_weights/Delta_dual agreement: ~1e-13 to ~1e-17.

**Performance** (`test_phase52_frechet_performance_gate.jl`, real D=20/W=80,000/L=50, 5 reps,
`workers ∈ {1,4,8,10,20}`):

| workers | dense median (s) | lookup median (s) | speedup | alloc ratio (dense/lookup) |
|---|---|---|---|---|
| 1  | 1.2117 | 1.0684 | 1.134x | 0.825x |
| 4  | 1.2706 | 1.0802 | 1.176x | 0.825x |
| 8  | 1.1954 | 1.0749 | 1.112x | 0.825x |
| 10 | 1.3060 | 1.0975 | 1.190x | 0.825x |
| 20 | 1.2021 | 1.0842 | 1.109x | 0.825x |

Faster at every thread count (1.11x-1.19x), `Delta_dual` agreement ~1e-17 to ~2.4e-17 — but
allocation is **worse**, not better (~21% more: 80.5MB vs 66.4MB median), unlike flexible CM which
achieved exact parity. The CM block's own logic is identical code (`apply_contrast!`/
`suffix_sums!`/etc. reused unchanged from `cm_lookup_kernels.jl`), so the extra allocation is
either genuine extra work in the new level-block code, or (more likely, not yet confirmed) a
difference in the number of KNITRO iterations taken by the two solve paths (different but
equally-valid floating-point orderings can converge in different iteration counts) rather than a
per-callback allocation bug — not root-caused in this pass; flagged as a specific, bounded
follow-on rather than chased down, to stay within this session's actual priority (the addendum's
Part A was judged higher-value and more clearly scoped).

### 3.4 Flip decision

Per task §5.6 ("allocations fall materially" is one of five required criteria), this does **not**
meet the bar as-is — allocation regressed, not fell. **`CM_FRECHET_INNER_FG_BACKEND_DEFAULT`
stays `:dense_reference`.** `:cm_frechet_lookup` ships available, explicitly opt-in
(`build_cm_frechet_production_context(...; inner_fg_backend=:cm_frechet_lookup)`), same
`AVAILABLE_BUT_NOT_DEFAULT` treatment flexible CM's own kernel got before its allocation fix.

### 3.5 Implementation notes

New files: `cm_frechet_lookup_kernels.jl` (`CMFrechetLookupState`), `cm_frechet_lookup_production.jl`
(KNITRO wiring, mirrors `cm_lookup_production.jl`). Modified: `cm_frechet_level.jl`
(`inner_fg_backend`/`skip_cm_fill_ref` kwargs on `wrap_moments_with_cm_frechet_archB`/
`build_cm_frechet_production_context`), `cm_frechet_cplus.jl` (`archC_frechet_base_state`/
`archC_frechet_verified_state` now dispatch on `cctx.inner_fg_backend`, with the same
`skip_cm_fill_ref` toggle-and-defensive-reset discipline §2 established for plain CM — verified
harmless/inert for the `:dense_reference` default via the D4/D20 gates above, which exercise the
default path unchanged). `cctx.cmlookup_st` (the `Any`-typed cache field added for plain CM) is
reused as-is for `CMFrechetLookupState` too — a `cctx` only ever belongs to one family, so there is
no collision risk between the two struct types sharing one field.

## 4. Documented-but-NOT-fixed-this-session finding: `compressed_cc_value_grad` allocates heavily inside a real KNITRO hot loop

**Update**: this finding is now FIXED, for the unrestricted family, as addendum Part A — see
`docs/UNRESTRICTED_ALLOCATION_FREE_FG_PORT_2026-07-26.md` for the full writeup (20,081x allocation
reduction, bit-identical output, existing comprehensive dense-vs-compressed integration suite
ALL PASS). The finding as originally documented follows, unchanged, for the record.

**This was found, verified against the real call graph, and is explicitly NOT fixed this
session** — flagged for a dedicated follow-on, not attempted under time pressure, per this
project's own standing rule against shipping unvalidated numerical/plumbing changes quickly.

### 3.1 What it is and why it matters more than the CM finding above

`compressed_cc_value_grad` (`compressed_cc_inner.jl:72`) is the FG (function+gradient) evaluator
for the **unrestricted family's** compressed/winner-sparse inner dual solve — the family whose
backend manifest already reports `shared_exact_winner_pair`, i.e. the codebase's own flagship,
already-shipped-as-default "we solved the dense-matrix problem" success story for this project.
It is called, **directly, unconditionally, every single FG iterate**, by
`_callbackEvalFG_inner_compressed!` (`compressed_live.jl:173`) — the actual KNITRO callback
registered for the unrestricted family's production inner solve. This is not a diagnostic or
reference path; it is the real, default, hot loop.

Its body:

```julia
function compressed_cc_value_grad(ζ::Real, λ::AbstractVector, cf::CompressedFactual; Psi!, dPsi!)
    W = cf.W; M = W
    contr = compressed_dual_contraction(λ, cf)          # allocates κ (D×Ddest), C (Ddest), AND the returned length-W vector
    q = similar(contr)                                  # allocates
    @inbounds @. q = -ζ - contr
    Psq = similar(q); Psi!(Psq, q)                       # allocates
    dPsq = similar(q); dPsi!(dPsq, q)                    # allocates
    f = sum(Psq) / M + ζ
    g_ζ = 1.0 - sum(dPsq) / M
    g_λ = compressed_transpose_contraction(dPsq, cf)     # allocates B (D×Ddest)
    @. g_λ = -(1.0 / M) * g_λ
    return f, g_ζ, g_λ, q, dPsq
end
```

Tracing the two helper calls confirms the allocation count is **worse** than it looks from this
function alone: `compressed_dual_contraction` (`compressed_moments.jl:228`) itself allocates a
fresh `(D, Ddest)` matrix `κ` and a fresh length-`Ddest` vector `C` internally, in addition to the
length-`W` vector it returns as `contr`; `compressed_transpose_contraction`
(`compressed_cc_inner.jl:35`) allocates a fresh `(D, Ddest)` matrix `B` internally. That is at
least **7-8 separate heap allocations per FG callback** (several `W`-length, several
`(D, Ddest)`-sized), not the 4-5 visible in `compressed_cc_value_grad`'s own body — every one of
them avoidable via a persistent workspace (`compressed_dual_contraction!`/
`compressed_transpose_contraction!` writing into caller-supplied buffers, exactly the same
pattern this session applied to `CMLookupState` in §1), and every one of them repeated on **every
KNITRO iterate of every inner solve** for the unrestricted family — i.e. the single most-used
family in this codebase.

### 3.2 This was not already addressed by the existing "v2" optimization pass — checked, not assumed

The codebase has a `compressed_cc_kernels_v2.jl`/`compressed_live_v2.jl` pair, which could
plausibly have already fixed this. It has not: v2's own header states its motivation explicitly —
"a destination-major, cache-friendly reimplementation... motivated by Phase 3C's finding that the
compressed FG/Hessian loops lose to BLAS at D=20" (a *memory-access-pattern* fix, addressing
column-stride cache misses on `cf.winner`/`cf.wval`), and `compressed_cc_value_grad_v2` is
explicitly documented as a "drop-in mirror" of the original — it preserves the **exact same**
`similar()`-based allocation pattern verbatim. Neither v1 nor v2 has ever addressed the allocation
question at all.

### 3.3 Why this is worth emphasizing

This project has, this same session, twice independently found and fixed real allocation
regressions in restricted-family hot loops (§1, §2) via the same mechanical technique (persistent
workspace, in-place kernels). This finding shows the identical failure mode sitting inside the
codebase's own **flagship, already-default, most-used** inner-solve path (unrestricted,
`shared_exact_winner_pair`) — not a newly-built experimental kernel like `:cm_lookup`, but
long-standing production code that has been the default for a substantial part of this project's
history, survived at least one dedicated follow-on optimization pass (v2), and was never audited
for this. It suggests the allocation-hygiene discipline this session applied to the CM lookup
path (§1) is a systemic gap in this codebase's inner-FG layer generally, not a one-off. Fixing it
is the same well-understood, low-derivation-risk mechanical pattern as §1 — persistent
`CompressedCCWorkspace`-style buffers, in-place `compressed_dual_contraction!`/
`compressed_transpose_contraction!`/fused-`Psi!`+`dPsi!` variants — but touches the unrestricted
family's own default production path, so it needs its own dedicated D=4+D=20 correctness gate and
allocation/wall-clock benchmark (mirroring §1.2's methodology) before any flip, not a quick patch
under time pressure. **Recommended as the highest-value next allocation-focused item in this
codebase, ahead of any further restricted-family work**, given it is default/hot/unaudited rather
than experimental/off-by-default.

## 5. Scoped follow-on for CM+ZC / origin-ZC (unchanged in substance, still NOT attempted)

1. **CM+ZC** (task §5.3's `[E | C | Z]` partition): genuinely unbuilt — CM+ZC has **no**
   lookup/compressed FG alternative at all right now; its `cctx.inner_fg_backend` is hardcoded to
   `:dense_reference`, and its entire inner FG callback is one dense `BLAS.gemv!` against the full
   widened `obj.H` (core+mean/pair columns lumped into one dense block, per its own docs).
2. **Origin-ZC** (task §5.4's `[E | Z]` partition): genuinely new, no existing partial kernel; does
   not touch `CMBinHessCtx`/`build_cm_bin_ctx` at all currently.

(Common Fréchet was in this list in the prior draft of this doc — it is now DONE, see §3.)

Each requires the task's own §5.6 flip rule (D=4+D=20 correctness, non-inferior speed, material
allocation reduction, no stability regression, `full_G_materializations=0`) before any default
change — none of that gating work has started for these three.

## 6. Separately-scoped follow-on: winner-sparse economic-core FG for CM (not the CM block — the core block)

Even with §1's fix, flexible CM's economic-core columns (the non-CM part of the inner FG
forward/backward) still go through a real dense `BLAS.gemv!` against a densely-materialized
`obj.H[:, core columns]` (built cheaply, via winner-structure, by
`materialize_dense_factual_structured!`, but still materialized and still consumed via dense BLAS
on every callback) — this is the SAME class of gap as §3, just for CM's core block rather than the
unrestricted family's whole FG. Task §5.1 explicitly asks for "compressed winner gather/scatter
for the economic core," which has not been done for CM; §1 only replaced the CM-block half of the
computation. Natural next step once §3's fix establishes a validated, reusable
compressed-core-FG pattern.
