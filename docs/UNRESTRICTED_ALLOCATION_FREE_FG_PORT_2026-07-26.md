# Unrestricted Allocation-Free FG Port — Addendum Part A — 2026-07-26

## Status: DONE, validated, live in production (no default flag — this is unrestricted's only inner-FG path)

## 1. What this is

The unrestricted family's compressed inner-dual FG callback (`_callbackEvalFG_inner_compressed!`,
`compressed_live.jl`) — the actual, default, already-shipped-as-production KNITRO callback,
called on every FG iterate of every inner solve for the unrestricted family — called
`compressed_cc_value_grad` (`compressed_cc_inner.jl`), which allocated 7-8 fresh arrays every
single call (several `W`-scale, i.e. length-80,000 at real production scale):

- `compressed_dual_contraction` (`compressed_moments.jl`): allocates `κ` (`D x Ddest`), `C`
  (`Ddest`), and the returned length-`W` vector `t`.
- `compressed_cc_value_grad` itself: `q = similar(contr)`, `Psq = similar(q)`, `dPsq = similar(q)`
  — three more length-`W` arrays.
- `compressed_transpose_contraction`: allocates `B` (`D x Ddest`) and the returned length-`ncol`
  vector `v`.

This was flagged mid-session (unprompted, from a ChatGPT critique the user relayed) while tracing
this exact code path to answer a question about the compressed inner-FG mechanism, and confirmed
by checking the real call graph — this is not a diagnostic/reference path, it is literally what
KNITRO calls on every FG iterate of the unrestricted family's production inner solve. Checked
whether the existing "v2" optimization pass already fixed it (`compressed_cc_kernels_v2.jl`) — it
did not; v2 is a pure cache-locality (destination-major memory layout) rewrite, explicitly a
"drop-in mirror" of the original that preserves the same `similar()`-based allocation pattern
verbatim.

## 2. The fix

Same mechanical pattern as the flexible-CM allocation fix (persistent workspace, in-place
kernels) — no change to the divergence formula, no change to `Psi!`/`dPsi!`, no approximation.

- **`EconomicFGWorkspace`** (`compressed_cc_inner.jl`): persistent `κ`/`C`/`contr`/`q`/`Psq`/`dPsq`/`B`
  buffers, sized once from a `CompressedFactual`'s own `(D, D_dest, W)`.
- **`compressed_dual_contraction!`** (`compressed_moments.jl`): in-place analogue of
  `compressed_dual_contraction`, writes into a caller-supplied `t` using persistent `κ`/`C`.
- **`compressed_transpose_contraction!`** (`compressed_cc_inner.jl`): in-place analogue of
  `compressed_transpose_contraction`, writes into a caller-supplied `v` (at the real call site,
  this is a `@view` directly into KNITRO's own `evalResult.objGrad` buffer — no extra copy) using
  a persistent `B`.
- **`compressed_cc_value_grad!`** (`compressed_cc_inner.jl`): in-place analogue of
  `compressed_cc_value_grad`, composing the three above.

All three **allocating originals are kept unchanged** — they remain the API `dual_bank.jl`'s cheap
scorer, `theta_cplus.jl`, and various benchmark/test scripts call directly (none of them the hot
per-FG-callback path this exists to fix); the new `!`-suffixed siblings are used only by the real
KNITRO callback.

**`CompressedCBState`** (`compressed_live.jl`) gained a new `fg_ws::EconomicFGWorkspace` field,
built once per inner solve (same lifecycle as `cf`, which genuinely changes every outer point —
not cached across inner solves, though that further refinement is possible since a workspace's
own dimensions are campaign-stable; not done here, see §5). The one production call site
(`_callbackEvalFG_inner_compressed!`) now calls `compressed_cc_value_grad!` writing directly into
`@view(evalResult.objGrad[2:end])` instead of `compressed_cc_value_grad` plus a manual copy.

The 4-arg `CompressedCBState` outer constructor (the ONLY constructor path any of the 9 real call
sites in this codebase use — confirmed by grepping every `CompressedCBState(` call) now builds
`EconomicFGWorkspace(cf)` automatically, so every existing caller (`compressed_live.jl`,
`compressed_live_v2.jl`, `fast_range_screen.jl`, `infeasibility_screen.jl`,
`compressed_inner_alt_solvers.jl`, two benchmark scripts, one test file) picks up the fix with no
call-site changes needed.

## 3. Validation

**Correctness (real, existing gate)**: `test_compressed_live_integration.jl` — the pre-existing,
comprehensive dense-vs-compressed equivalence suite (calibration point at both incumbents, 15
random feasible perturbations, all 102 coordinate/sign/step-size finite-difference probes, warm
vs cold inner solves, a 5-step outer trajectory with warm-started inner solves comparing every
intermediate point, and an injected exact price tie confirming automatic dense fallback) — run
unmodified against the patched `_callbackEvalFG_inner_compressed!`:

```
ALL COMPRESSED-LIVE-INTEGRATION EQUIVALENCE TESTS PASSED
```

Every per-field worst-case abs diff across the whole suite unchanged from pre-patch expectations
(`Delta_dual` 5.6e-15, `zeta` 1.8e-13, `lambda` 3.5e-11, etc. — all at the same floating-point-order
tolerance this suite has always reported, not degraded by the patch).

**Allocation microbenchmark** (`bench_partA_economic_fg_allocation.jl`, real D=20/W=80,000, 200
reps, warm-started): direct comparison of the OLD allocating `compressed_cc_value_grad` against
the NEW in-place `compressed_cc_value_grad!`, both still present, called on the identical `(ζ,λ,cf)`:

```
agreement: f diff=0.000e+00  g_zeta diff=0.000e+00  g_lambda maxdiff=0.000e+00  q maxdiff=0.000e+00
old (compressed_cc_value_grad)  : median_bytes=2,570,376 (2570.4KB)  median_time=1.118e-02s
new (compressed_cc_value_grad!) : median_bytes=128 (0.1KB)          median_time=9.697e-03s
allocation reduction: 20081.1x  (0.0% of original)
time speedup: 1.153x
```

Output is **bit-for-bit identical** (0.000e+00, not just close) — this is a pure mechanical
refactor, not a numerically-different reformulation. Allocation drops from ~2.57MB to 128 bytes
per call (essentially eliminated — the residual 128 bytes is fixed small-object overhead, not
`W`-scale). At a real inner solve with, say, 5-20 FG callbacks, this removes on the order of
13-51MB of allocation per inner solve that used to happen, at zero cost to wall-clock (modestly
faster, not slower).

## 4. Runtime counter check

No `full_G_materializations`-style counter existed for this specific path before this session; the
compressed representation never materialized a dense `G` to begin with (that was the whole point
of the original `compressed_live.jl` port) — this fix is purely about the *repeated per-callback*
allocation of *smaller* intermediate arrays (`q`/`Psq`/`dPsq`/`κ`/`C`/`B`/`contr`/`v`), not about a
dense-matrix materialization. `dense_economic_G_materializations`-style counters per the addendum's
Part D request are not yet wired (see the Part B-I scoping note in
`docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md`).

## 5. Not done (explicitly, honestly)

- **Cross-inner-solve workspace caching**: `EconomicFGWorkspace` is rebuilt once per inner solve
  (cheap — 7 small allocations, not thousands) rather than cached once per campaign the way
  `cf_workspace`/`CompressedFactualWorkspace` already is on `ctx`. Given the fix already achieves a
  20,000x reduction at the dominant (per-callback) granularity, this further refinement was judged
  not worth the additional plumbing risk in this pass — flagged, not attempted.
- **Fused `Psi!`+`dPsi!` single-pass evaluation** (the addendum's own `psi_dpsi_sums!` suggestion,
  eliminating the separate `Psq` array and one full `O(W)` pass): NOT done. This would require
  replicating the piecewise hybrid-KL/quadratic divergence formula's simultaneous value+derivative
  computation, touching code this project's own standing rule says not to alter without independent
  verification (`cc_algo/PsiObjectiveBundle.jl`'s `Psi!`/`dPsi!`) — the addendum itself only asks for
  this "where the live divergence formula permits" and explicitly forbids introducing any
  approximation to it. Judged higher-risk than the buffer-reuse fix for a proportionally smaller
  remaining allocation (`Psq` is 1 of the 8 original allocations, already eliminated as a *fresh*
  array even without fusion — it now writes into `ws.Psq`, a persistent buffer, just not fused with
  `dPsq`'s own pass). Flagged as a legitimate further micro-optimization, not attempted.
- **`Part B` (shared economic operator across all 5 families)**, **Part C** (operator-based
  verification, removing `select_G_from_H`/dense `obj(...)` reads from `archC_verified_state`-style
  functions), **Part D** (the full runtime-counter/fail-fast-guard suite), **Parts E-I**: not
  attempted this session — see the scoping note in
  `docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md`.
