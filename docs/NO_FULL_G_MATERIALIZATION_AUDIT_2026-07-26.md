# No-Full-G-Materialization Audit — 2026-07-26

## Task requirement

Task §4: production hot paths must never materialize a full moment matrix `G`; `forward!`/
`transpose!`/Hessian blocks must be operator-based. The full dense matrix may exist only in
explicit reference/debug backends, correctness tests, or small D=4 diagnostics. Required runtime
invariant: `full_G_materializations = 0`, `generic_dense_FG_calls = 0`.

## Honest finding: this invariant does NOT currently hold, and this is the real remaining gap

Tracing each family's `moments!` closure:

### Flexible CM (`wrap_moments_with_cm_archB`, `cm_hessian_architectures.jl:191`)

Uses a **persistent, reused** scratch buffer `Gtmp_cache` (allocated once per closure lifetime,
resized only on a genuine `n` change), and builds the CM restriction columns via
`fill_cm_columns_from_bins!` — a **chunked** bin-fill (`chunk_size=2000` default), not an
O(W × total_moments) monolithic dense build. The economic core columns go through
`materialize_dense_factual_structured!` writing into a *view* of that same scratch buffer when
`use_compressed_core=true` (production default) — this uses the compressed winner-form `cf`
internally, not a from-scratch dense economic-core computation, but the **destination** is still a
dense `Matrix{Float64}` slab sized `(n, ncore_full)`. This is a bounded, reused, chunked
representation — much better than an unbounded per-call allocation, but it is still a dense
in-memory `G` slab being filled every callback, not a matrix-free operator (`forward!`/`transpose!`
acting on `lambda`/`q` without ever materializing `G` as a matrix). **Not compliant with the
strict letter of task §4**, though it is close in spirit (bounded, reused, chunked).

### Common Fréchet (`wrap_moments_with_cm_frechet_archB`, `cm_frechet_level.jl`)

Same architecture as flexible CM (shares `fill_cm_columns_from_bins!`) plus an additional
`fill_frechet_level_columns_from_bins!` for the level-anchor block — same chunked-dense pattern,
same non-compliance.

### CM+ZC (`cm_meanzc_production.jl`)

**Explicitly, deliberately dense** — Phase E part 2 of the inherited remediation (adopted this
session, `41e577c`) benchmarked CM+ZC's "persistent-dense-CM-columns" `moments!` against plain
flexible CM's bin-recompute approach at real D=20/W=80,000/L=50 and found it ~10.6% slower in wall
time with essentially identical allocation — and, per the task's own instruction not to change
representation "merely for stylistic uniformity" absent real benchmark evidence, **retained the
dense-column representation as-is**. This is the single clearest, most explicit instance of
`full_G_materializations > 0` in current production defaults, and it was a *documented, evidence-
based decision* by the immediately-preceding session, not an oversight.

### Origin-ZC (`cm_originzc_moments.jl`)

Has no CM grid (confirmed, §3 above) — its `Z`/`Zpair` restriction columns are raw power features
of a small, fixed dimension (`K_mean`/`K_pair`, not `L`-scaled), computed directly rather than via
a chunked bin-fill. Whether this counts as "dense" in the sense the task cares about is a matter of
scale: `K_mean`/`K_pair` are O(1)-O(10), not O(L·D), so even a literal dense materialization here
is a small, fixed-size block, not a moment-matrix-scale allocation. Lower priority than the other
three families.

## Verdict

`full_G_materializations = 0` is **not yet true** for any of the four restricted families under
current production defaults. The chunked-dense pattern (flexible CM, common Fréchet) is a
reasonable middle ground already in place, but is not the same as a true matrix-free operator.
CM+ZC is furthest from compliant, by an explicit, already-benchmarked decision.

**This is exactly the item the task itself flags as "the central unfinished item"** — genuinely
new numerical-kernel development (Phase 5: `forward!`/`transpose!`/Hessian-block operators per
family, replacing the chunked-dense-fill pattern) was **not attempted this session** given the
scope already covered (Phases 0, 1, 2, 3, 8) and the real risk of shipping an incorrect, unvalidated
Hessian/gradient kernel under time pressure. Recorded here as the top priority for a dedicated
follow-on session — see `HIGHEST_PRIORITY_REMAINING_GAP` in the master report's final verdict.

## What would be needed to close this (scoped, not attempted)

1. A `forward!(r, lambda, ctx)` / `transpose!(grad, q, ctx)` pair per family that walks `Bidx`
   directly (one bin lookup per origin/draw, exactly as the task's §7.2 interval-basis section
   describes) instead of filling a `(n, ncore_full)` scratch matrix and then multiplying against
   it.
2. For CM+ZC specifically: replace the dense CM-column block with the same bin-fill operator
   flexible CM already uses, THEN add the mean/pair block as a small separate operator — the
   Phase E part 2 benchmark that justified keeping dense columns compared apples-to-apples
   representations, not an operator-based alternative, so it does not settle whether an
   operator-based CM+ZC would out-perform the current dense-column design.
3. Real D=4 + D=20/W=80,000 correctness gates (operator output vs the current dense/chunked
   reference, bit-for-bit) before any default flip, per the task's own §5.6 flip rule.
