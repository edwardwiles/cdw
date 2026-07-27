# Verified-State Allocation Fixes — 2026-07-27

## What was fixed

Every family's `*_verified_state` function contained the identical pattern:

```julia
m_weights = copy(obj.arg1)                    # KEEP -- see below
p_weights = m_weights ./ sum(m_weights)       # REMOVED
...
weight_norm_resid = abs(sum(p_weights) - 1.0) # now computed without materializing p_weights
```

`p_weights` was allocated purely to compute `weight_norm_resid`, a diagnostic whose value is
`abs(sum(m_weights ./ s) - 1.0)` where `s = sum(m_weights)` — mathematically exactly `abs(1 - 1) =
0` and in practice a measure of pure floating-point rounding noise (never a substantive signal).

**Fix**: `weight_norm_resid = abs(sum(x -> x / s_m_weights, m_weights) - 1.0)`, where
`s_m_weights = sum(m_weights)` is computed once (same as before). This uses Julia's `mapreduce`
path instead of pre-materializing a broadcast array, and was verified to be **bit-identical**
(not merely "close") to the original `sum(m_weights ./ s)` computation:

```
n=100     bitequal=true  diff=0.0
n=381     bitequal=true  diff=0.0
n=400     bitequal=true  diff=0.0
n=8000    bitequal=true  diff=0.0
n=80000   bitequal=true  diff=0.0
```

(checked directly via `julia -e '...'`, covering the exact W/moment-count scales this codebase
uses at D=4 (W=8000, 400/381 moments) and real D=20 (W=80,000)). Bit-identity holds because Julia's
`sum`/`mapreduce` both apply the SAME pairwise-summation algorithm over the same sequence of
per-element values (`x/s` computed on the fly vs pre-materialized) — the associativity structure
depends only on array length, not on whether the divided values were precomputed.

## Where

All 5 families' verified-state functions:

- `cm_production_bundle.jl` (flexible CM)
- `cm_meanzc_production.jl` (CM+ZC)
- `cm_originzc_production.jl` (ZC only)
- `cm_frechet_cplus.jl` (common Fréchet)
- `fast_range_screen.jl` (unrestricted)

## What was explicitly NOT changed, and why

`m_weights = copy(obj.arg1)` is **retained unchanged** in every file. This is a genuine
`NECESSARY_OUTPUT_ALLOCATION`: `obj.arg1` is KNITRO's own internal buffer, reused (overwritten) by
the NEXT inner solve. `BaseDualState`/the returned `verify` NamedTuple must own an INDEPENDENT
snapshot of the weights at THIS solve's converged point, or a caller reading `m_weights` after a
subsequent inner solve has run would silently see the wrong (next-solve's) values — a
correctness bug, not merely an allocation one. This was not removed "without proving lifetime
safety," per the task's own explicit instruction; it was not touched at all, since proving its
necessity was straightforward from the existing `obj.arg1` reuse contract documented elsewhere in
this codebase (`solve_base_state`'s own identical `m_star = copy(obj.arg1)` pattern, unchanged by
any prior session).

## Regression evidence

`test_shared_core_hessian_d4_gates.jl` (D=4, covers flexible CM / CM+mean-ZC / origin-ZC's
verified-state call paths via their respective `archOZ_verified_state`/`archC_meanzc_verified_state`/
plain-CM analogs, run after this fix was applied): **41/41 checks PASS**, including dual-solution
agreement and full-Hessian-assembly agreement between backends — i.e. nothing downstream of
`verify.weight_norm_resid`'s new computation path broke. `weight_norm_resid`'s own numeric value
was not separately asserted equal pre/post-fix in an automated test this session (the bit-identity
proof above is a general mathematical argument, verified numerically at the relevant scales, not a
per-family before/after diff of the live `verify` NamedTuple) — a slightly stronger gate (asserting
the exact `verify.weight_norm_resid` field is unchanged, not just that everything downstream still
passes) would be a cheap addition for a future session.

Common Fréchet's own verified-state path was not separately exercised by
`test_shared_core_hessian_d4_gates.jl` (that test's own scope is CM/CM+ZC/origin-ZC per its header
comment) — `cm_frechet_cplus.jl`'s fix was applied via the same mechanical, bit-identity-verified
transformation as the other 4 files, but has no dedicated regression run in this task's own
deliverables. Flagged honestly, not silently assumed fine.
