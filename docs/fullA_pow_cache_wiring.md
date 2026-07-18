# Wiring `MuSigmaPowCache` into the live oracle path

Branch `diag/fullA-d4-exact` (HEAD `bb74649` at start). Continuation-6 handoff "What's left" item 2.
This is the follow-through on `docs/fullA_moment_construction_audit.md`, which built and
equivalence-tested `MuSigmaPowCache`/`EK_moments_gammanorm_directgp_fast!` **standalone** but left them
un-wired into the actual optimization path.

## What changed

One additive, opt-in helper in `full_aod_diag/d4_exact/moments_fast.jl`:

```julia
function enable_pow_cache!(ctx)
    obj = ctx.obj
    pow_cache = MuSigmaPowCache(obj.U, obj.γ.Uσ)
    obj.moments! = (K, G, θ, U, o) -> EK_moments_gammanorm_directgp_fast!(K, G, θ, U, o, pow_cache)
    return pow_cache
end
```

Why this is the whole wiring, and why it is low-risk:

- `ctx.obj` is a `PsiObjectiveBundleImplicit` (`cc_algo/PsiObjectiveBundle.jl:128`), a **mutable**
  `@with_kw` struct. Its `moments!::Function` field is invoked verbatim by *every* live consumer of the
  moment build — `inner_loop_internal` (via `oracle.jl:133`, `oracle_fast.jl:140`,
  `oracle_profiled.jl:102`), `lfix_incremental.jl:359`, and the KNITRO F+G callback path all call
  `obj.moments!(K, G, θ, obj.U, obj)` with the identical 5-arg signature. Rebinding that single field
  therefore threads the cache through the entire live path — **no context-builder mirror, no change to
  any call site**, exactly the field-mutation path the handoff identified.
- The closure signature `(K,G,θ,U,o)` matches the original `EK_moments_gammanorm_directgp!` exactly, so
  it is a drop-in.
- Opt-in by design: `d4_exact_setup()` is **unchanged**, so no existing caller or test is affected
  unless it explicitly calls `enable_pow_cache!`. (`lfix_incremental.jl` / `composite_gradient_fast.jl`
  were **not** modified, per the handoff's scoping constraint — they simply pick up the faster
  `moments!` automatically *if and only if* a caller opts in, and only in an equivalence-preserving way.)

No production files under `cc_algo/`, `full_aod_diag/moments_gammanorm.jl`, `moments/hFunction.jl`,
`lfix_incremental.jl`, or `composite_gradient_fast.jl` were touched.

New files, both additive: `verify_pow_cache_wiring.jl` (live-path equivalence),
`profile_pow_cache_wiring.jl` (realized-gain profile).

## Equivalence-test confirmation

Ran on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, `source .knitro_env.sh`.

**Baseline — every existing equivalence test still passes unchanged** (proves adding
`enable_pow_cache!` broke nothing; these do *not* opt into the cache):

| test | result |
|---|---|
| `test_oracle.jl` | ALL ORACLE TESTS PASSED |
| `test_oracle_profiled.jl` | ALL EQUIVALENCE CHECKS PASSED |
| `test_oracle_fast.jl` | ALL ORACLE_FAST EQUIVALENCE TESTS PASSED |
| `test_lfix_incremental.jl` | ALL PHASE 2 EQUIVALENCE TESTS PASSED (both tiers exact) |
| `test_composite_gradient_fast.jl` | ALL COMPOSITE_GRADIENT_FAST EQUIVALENCE TESTS PASSED |
| `test_moments_fast.jl` | 8/8 points, `max|G diff|=max|K diff|=0.0` exactly |

**With the cache wired in — `verify_pow_cache_wiring.jl`**: compares a cache-free `ctx` against a
cache-enabled `ctx` through the **full live** `evaluate_fullA` *and* `evaluate_fullA_fast`, field-for-field
(Delta_dual, Delta_primal, K_hard, gravity, winner_hash, λ, moment_resid, KKT residual, inner_status, …),
across calibration + `upper_lfixcomposite_sr1_60s` + `lower_stalled` + 5 random feasible perturbations:

- **All 8 points, both oracle paths: `worst|Δ| = 0.000e+00` (bit-identical).** Because the wiring only
  swaps the moment build for a byte-identical implementation, the entire downstream result — inner dual
  solve, divergence, gravity, winners — is unchanged to the last bit.
- **Cache behaviour correct**: `n_recompute=1`, then pure reuse (`n_reuse=13` on the `evaluate_fullA`
  path, `n_reuse=7` on the `evaluate_fullA_fast` path) — recomputed exactly once at first use, reused for
  every subsequent call at the fixed `μ`, as designed.

## Realized performance gain (with evidence, not wall-clock alone)

`profile_pow_cache_wiring.jl`, D=4 / W=8000, `JULIA_NUM_THREADS=20`, warmed. Two measurements:

**(A) Moment-build component in isolation** (the actual `obj.moments!` field call, N=300 reps):

| cache | median wall | min wall | alloc/call | GC/call |
|---|---|---|---|---|
| OFF (orig) | 8.071 ms | 7.902 ms | 4123.1 KB | 1.165 ms |
| ON (fast)  | 8.044 ms | 7.735 ms | 3601.3 KB | 0.632 ms |
| **delta** | **1.00× (median)** | 1.02× | **−521.9 KB (−12.7%)** | **−0.533 ms** |

**(B) Live `evaluate_fullA_fast` end-to-end** (its own `@prof "inner_moment_build"` timer + total +
KNITRO callback counts, N=50 reps, `warm=false`):

| cache | inner_moment_build median | inner_moment_build alloc | inner_moment_build GC | eval total | n_fg | n_hess |
|---|---|---|---|---|---|---|
| OFF | 9.647 ms | 4142.6 KB | 2.462 ms | 38.02 ms | 10 | 9 |
| ON  | 8.315 ms | 3620.0 KB | 0.782 ms | 33.76 ms | 10 | 9 |
| **delta** | **1.16×** | **−522.5 KB** | **−1.68 ms** | **1.13×** | identical | identical |

### What is actually driving the difference (verify-before-causal-claims)

The honest causal story, backed by allocation / GC / callback-count evidence — **not** wall-clock alone:

1. **The gain is allocation elimination, not raw compute.** The cache removes the fresh allocation of
   two `W×D` `Float64` matrices (`UPow`, `UσPow`) on every call: `8000×4×8 bytes × 2 = 512 KB`, which
   matches the measured **−522 KB/call** almost exactly. This is a robust, thread-count-independent win.
2. **At 20 threads the raw `.^(-μ)` broadcast is nearly free**, so the *isolated* moment-build wall time
   barely moves (median **1.00×**; min 1.02×). This is a genuine **correction** to
   `docs/fullA_moment_construction_audit.md`'s expectation: that audit measured 1.41× at 1 thread and
   already noted the wall-clock speedup shrinks with thread count (~1.1–1.2× at 8–16 threads); at 20
   threads the *wall-clock* component of the win has essentially vanished. The prior "1.1–1.4× on the
   moment-build component" number was a **1-thread** figure and does not hold at production thread counts.
3. **The realized live gain comes from reduced GC pressure**, not faster arithmetic. Eliminating ~522 KB
   of per-call allocation drops moment-build GC time from 2.46 ms → 0.78 ms (−1.68 ms) inside a real
   evaluation, which is where the live **1.16×** moment-build and **1.13×** end-to-end improvements
   come from.
4. **The delta is genuinely the moment build, not a changed solve trajectory.** The inner KNITRO dual
   solve callback counts are **bit-for-bit identical** cache-off vs cache-on (`n_fg=10`, `n_hess=9` at
   every rep) — the cache changes only the one moment build per evaluation, exactly as the design intends.

**Bottom line**: the wiring is correct (bit-identical results through the full live path) and delivers a
modest but real ~1.13× end-to-end / ~1.16× moment-build improvement at 20 threads, driven by eliminating
~522 KB of per-call allocation and the associated GC time — a documented correction to the 1-thread
1.1–1.4× expectation.

Machine-readable numbers: `results/fullA_d4/<hash>/pow_cache_wiring/pow_cache_wiring_profile.csv`.
(The `<hash>` subdir is mislabeled `53ffb58` because `d4_exact_setup`'s known `setwd`-style `cd` during
context construction makes `git rev-parse` resolve in a different directory; a cosmetic path quirk, not
a correctness issue — HEAD is `bb74649`.)
