# Winner accelerator live wiring (Continuation 8, workstream 3)

Branch `c8-winner-accel`, worktree `gravity-fullA-d4-c8-winner-accel`, base commit
`33c93ff` (tip of `diag/fullA-d4-exact` at start of this workstream). Picks up
Continuation 7's Workstream 4 (`docs/winner_certificate_report.md`), which built and
validated (but did not wire) two accelerators: the coordinate-specialized top-3 update
and the winner-margin certificate. This continuation wires both into the live hard
L_fix coordinate gradient (`composite_gradient.jl`/`composite_gradient_fast.jl`/
`lfix_incremental.jl`), following the standing brief's Section 3.

All work is additive: no existing function was deleted or had its default numerical
output changed without an equivalence test proving it identical; the original O(D)
fallbacks are kept and reachable via an explicit flag.

## 1. Coordinate-specialized top-3 update

**Target** (explicit in the brief): `composite_gradient.jl::count_winner_flips_multi`,
the same-destination-two-changed-origins O(D) rescan fallback (used by
`select_bandwidth`'s adaptive-bandwidth bisection). **Bonus** (same bug pattern, closes
the analogous fallback in the *value* tier, not just the flip-count tier): `lfix_
incremental.jl::dest_contrib_incremental`, `dest_contrib_incremental_o1`'s own
2-changed-origin fallback (used by the actual FD gradient probes, not just bandwidth
selection).

### What was built
- `LFixBaseCache` (lfix_incremental.jl) extended with `third0`/`third_price0`/
  `third_pTσ0` (rank-3 winner cache), computed for free from the already-built dense
  `price0`/`pTσ0` array in `build_lfix_base_cache` (an extra O(W·D²) *pass* over
  already-resident data, not an extra O(W·D²) *recompute*).
- `count_winner_flips_multi_top3` (composite_gradient.jl) and `dest_contrib_
  incremental_top3` (lfix_incremental.jl): exact O(1)-per-draw replacements for the
  O(D) rescan, using the same top-3-cache argument
  `winner_certificate.jl::coord_winner_update!` already established for the analogous
  `WinnerRefCache` (reused, not re-derived): with ≤2 changed origins the best
  surviving unchanged origin is at worst rank 3.
- Wired as the new **default** via `multi_method::Symbol=:top3` on `count_winner_
  flips`, `dest_contrib_incremental_o1`, and threaded through `select_bandwidth`,
  `a_block_fd_component`, `lfix_incremental_at`, `composite_gradient_at`, and
  `composite_gradient_at_fast`. The original O(D) rescans are unchanged and reachable
  via `multi_method=:generic` at every one of those call sites.

### API
```julia
count_winner_flips_multi_top3(cache::LFixBaseCache, ctx, θ_full, d, changed_origins)  # -> Int
dest_contrib_incremental_top3(cache::LFixBaseCache, ctx, θ_full, d, changed_origins)   # -> Vector{W}
count_winner_flips(cache, ctx, θ_full, d, changed_origins; multi_method::Symbol=:top3)
dest_contrib_incremental_o1(cache, ctx, θ_full, d, changed_origins; multi_method::Symbol=:top3)
select_bandwidth(cache, ctx, pe, w0, coord_idx; multi_method::Symbol=:top3, ...)
a_block_fd_component(cache, ctx, pe, w0, coord_idx, h; multi_method::Symbol=:top3)
lfix_incremental_at(cache, ctx, pe, w0, coord_idx, new_val; tier=:incremental, multi_method::Symbol=:top3)
composite_gradient_at(x_free0, ctx, pe; base=nothing, multi_method::Symbol=:top3)
composite_gradient_at_fast(x_free0, ctx, pe; ..., multi_method::Symbol=:top3)
```
`multi_method` accepts `:top3` (default) or `:generic` (original) at every level.

### Equivalence (`test_winner_top3_equivalence.jl`)
- **(A)+(B) synthetic 2-changed-origin cases** — every origin pair × every destination
  at D=4, 5 step sizes (1e-3 … 1.0), 2 base points (upper40, lower): **240 cases total,
  0 flip-count mismatches, 0 contribution-value mismatches** (max abs diff = 0.0,
  bit-identical, not just close).
- **(C) real coordinate sweep** through a live `composite_gradient_at` call: the 3
  coordinates that naturally hit the 2-changed-origin case (14, 15, 16 — all share the
  gravity pivot's destination 4, confirming Continuation 7's own audit) reproduce the
  original O(D) fallback bit-for-bit at both `count_winner_flips` and `dest_contrib_
  incremental_o1`, at 3 step sizes, on both base points.
- Pre-existing suites re-run and unaffected: `test_lfix_incremental.jl`, `test_
  composite_gradient.jl`, `test_composite_gradient_fast.jl`, `test_winner_
  certificate.jl` — all still pass with the new `:top3` default live.

### Measured end-to-end effect (`benchmark_winner_accelerator.jl`, Part 1)
Full `composite_gradient_at_fast` wall time (all 15 A-block coordinates, real
`upper40` point, D=4/W=8000), `:generic` (original) vs `:top3` (new default), min of
15 warm reps:

| config | `:generic` | `:top3` | ratio |
|---|---|---|---|
| threaded=true, h_mode=:fixed | 17.2–17.7 ms | 17.0–18.5 ms | 0.93–0.98× |
| threaded=true, h_mode=:adaptive | 26.3–28.5 ms | 24.4–24.5 ms | 1.08–1.17× |

Two independent runs both land in the 0.93×–1.17× range — **noise-dominated, no
measurable full-gradient speedup at D=4**. This is expected, not a negative result on
the fix's correctness: only 3 of 15 coordinates ever reach the fallback, and D=4's O(D)
rescan itself is tiny in absolute terms (a 4-element inner loop) next to the other
per-coordinate overhead (Dict allocation, `price_and_pTsigma_cell` calls, threading
dispatch) that dominates wall time at this scale. Continuation 7's own report already
flagged this: "the benefit is bounded at D=4 ... but grows with D." The fix remains
adopted as the default because it is **exact** (proven, not approximate) and removes a
documented structural gap, not because of a D=4 wall-clock win — the isolated
winner-computation component itself is 13.6–200× faster per Continuation 7's own
measurement, this benchmark's contribution is confirming that gain does not show up
end-to-end yet at the current D.

## 2. Winner-margin certificate: persistent value-eval accelerator

### Scope decision (load-bearing, read before extending)
The winner-margin certificate (`certified_winner_update`) is provably exact for the
**winner identity only** — its proof is that a certified cell's cached winner is the
unique strict argmin at the new point; it says nothing about whether the runner-up or
third-place identities are also unchanged (a lower-ranked competitor can cross the
runner-up without ever threatening the winner). The exact-top-3-dependent gradient
tiers (`dest_contrib_incremental_o1`'s O(1) update, this continuation's own `count_
winner_flips_multi_top3`/`dest_contrib_incremental_top3`) need *exact* runner-up/third,
so a stale/certified-but-unproven runner-up cannot be substituted there without
re-deriving a stronger top-3 certificate — explicitly out of scope per the brief
("your job is wiring... don't re-derive the certificate math").

Consequently the certificate is wired as an **L_fix value evaluator**
(`lfix_value_certified`), not a gradient-tier accelerator: it evaluates `L_fix`
*exactly* at an arbitrary new point (all D² A_od cells may move simultaneously — unlike
`lfix_incremental_at`'s ≤2-changed-cell coordinate tiers), which is precisely what a
line-search trial step or a profile-continuation step needs, and precisely what the
existing incremental machinery *cannot* do (it is coordinate-restricted by
construction).

### What was built
- `PersistentWinnerCache` (winner_certificate.jl): caller-owned, mutable, holds the
  current `WinnerRefCache` anchor plus cumulative `Ref`-style counters (certified/
  rescanned cells, full-fallback calls, rebuilds, cert-path vs. full-path wall time),
  built lazily and reused across many calls — not rebuilt per call.
- `winner_value_update!(wc, ctx, x_free′; threaded=false, rebuild_on_fallback=true)`:
  the core primitive — certifies/rescans against the persistent anchor, and on a
  `tol_far` full-scan fallback (still exact, just slower) rebases the anchor so future
  nearby calls certify cheaply again. Rebasing is a pure performance policy: every
  returned value (certified or fallback) is bit-identical to a full
  `compute_winners_fast` scan regardless of anchor history, so cache use never depends
  on callback order.
- `winners_from_certificate_threaded` (draw-level-threaded counterpart, for large-W·D
  standalone use only — see threading discipline below).
- `lfix_value_certified(cache::LFixBaseCache, wc::PersistentWinnerCache, ctx, x_free′;
  threaded=false) -> (Lfix_value, winner′, CertStats)` (composite_gradient_fast.jl):
  the value evaluator itself.
- `composite_gradient_at_fast` gained an **opt-in, gradient-neutral** diagnostic:
  `winner_cache_mode::Symbol=:none/:certificate` + `winner_cache::PersistentWinnerCache`.
  When `:certificate`, it additionally logs a `winner_cert_stats` field in `meta` via
  the persistent cache — **verified to never alter the returned gradient** (test B
  below).
- `winner_cache_report(wc)`: certified/rescanned/full-fallback-call fraction, rebuild
  count, cert-path vs. full-path wall time, implied speedup.

### API
```julia
PersistentWinnerCache(; tol_far::Float64=0.3)
winner_value_update!(wc, ctx, x_free′; threaded=false, rebuild_on_fallback=true)  # -> (winner′, wval′, CertStats)
lfix_value_certified(cache::LFixBaseCache, wc::PersistentWinnerCache, ctx, x_free′; threaded=false)  # -> (Lfix_value, winner′, CertStats)
winner_cache_report(wc)  # -> NamedTuple
composite_gradient_at_fast(x_free0, ctx, pe; ..., winner_cache_mode::Symbol=:none, winner_cache=nothing, winner_cache_threaded::Bool=false)
reset!(wc) / reset_counters!(wc)
```

### Threading discipline (per the brief's explicit requirement)
- `composite_gradient_at_fast`'s outer coordinate loop (`threaded=true`,
  `Threads.@threads` over 15 A-block coordinates) stays **inner-serial** — `count_
  winner_flips_multi_top3`/`dest_contrib_incremental_top3` never spawn threads
  internally, preserving Continuation 7's measured 7.33× advantage of
  outer-threaded+inner-serial over nested inner-threading.
- `winner_value_update!`/`lfix_value_certified`'s `threaded` kwarg (→ `certified_
  winner_update_threaded`/`winners_from_certificate_threaded`) is for a **standalone,
  non-nested** value evaluation only — default `false`. Continuation 7's own Context A
  table showed draw-level threading is a wash-to-slight-loss at D=4/W=8000 (0.83–1.52×)
  and only pays off at large W·D (1.86× at D=10/W=80000); this continuation did not
  re-measure that table, only preserved its conclusion in the API defaults and
  docstrings.

### Equivalence (`test_winner_accelerator_wiring.jl`)
- **(A) value + winner exactness, cold and warm**: at 6 step-size regimes (1e-3 … 5e-1)
  × 3 reps × 2 base points, `lfix_value_certified`'s winner matrix is **bit-identical**
  (max abs diff = 0) to the trusted full per-destination rescan
  (`dest_contrib_block_local`-based) in every case, both with a **fresh
  `PersistentWinnerCache` per point (cold)** and with **one `PersistentWinnerCache`
  reused across all points (warm)**. The L_fix *value* matched to **exactly 0 relative
  error observed** in all cases (well under the documented ~1e-8 tolerance budgeted for
  the known vectorized-vs-scalar floating-point-path difference — see the function's
  own docstring).
- **(B) gradient neutrality**: `composite_gradient_at_fast`'s returned gradient `g` is
  bit-identical (`max|g_none - g_certificate| = 0.0`) whether `winner_cache_mode=:none`
  or `:certificate`; the diagnostic is present in `meta` only when requested.
- **(C) warm 30-point sweep breakdown**: 98.08% certified, 1.92% rescanned, 0%
  full-fallback, 1 rebuild (the initial cold anchor).
- Pre-existing `test_winner_certificate.jl` re-run and unaffected.

### Measured end-to-end speedup (`benchmark_winner_accelerator.jl`, Part 2)
40-point simulated line-search/continuation sweep (random steps drawn from
{1e-3, 5e-3, 1e-2, 2e-2, 5e-2} magnitudes, real `upper40` base point, D=4/W=8000), one
`PersistentWinnerCache` reused across all 40 points, vs. the uncached trusted full
rebuild called fresh at every point (min of 8 warm reps):

| run | uncached full rebuild | warm certified | speedup | certified frac |
|---|---|---|---|---|
| run 1 | 335.1 ms (8.38 ms/point) | 54.4 ms (1.36 ms/point) | **6.16×** | 98.77% |
| run 2 | 345.1 ms (8.63 ms/point) | 53.2 ms (1.33 ms/point) | **6.49×** | 98.77% |

Full-fallback fraction: 0.0% in both runs (all 40 steps stayed within `tol_far=0.3` of
the single initial anchor); 1 anchor rebuild total (the lazy first-use build).

## Files touched (all owned by this workstream)
`composite_gradient.jl`, `composite_gradient_fast.jl`, `lfix_incremental.jl`,
`winner_certificate.jl` (extended, none deleted/replaced destructively); new files
`test_winner_top3_equivalence.jl`, `test_winner_accelerator_wiring.jl`,
`benchmark_winner_accelerator.jl`, this report.

## What's NOT done / left for a future continuation
- The top-3 fix's real payoff is D-scaling (Continuation 7's own note); a D=6/8/10
  rerun of `benchmark_winner_accelerator.jl` Part 1 would show whether the fallback's
  growing O(D) cost eventually shows up end-to-end — not run this session (out of
  scope: this workstream owns D=4-exact files only, per the standing multi-agent
  split).
- A genuine top-3-*exact* certificate (proving runner-up/third unchanged, not just the
  winner) would let `lfix_value_certified`-style persistence extend to the gradient
  tiers too — deliberately not attempted here (re-deriving certificate math was
  explicitly out of scope for this continuation).
- `winner_cache_mode=:certificate`'s diagnostic is currently a logging-only hook; no
  driver (KNITRO outer loop / profile-continuation script) yet consumes `winner_cert_
  stats` for an actual revalidation *policy* (analogous to `HybridGradientPolicy`) —
  the mechanism is wired and tested, the policy is left to the caller, matching this
  investigation's existing `h_mode=:cached` precedent (composite_gradient_fast.jl's own
  header: "this file does not impose a revalidation POLICY, only the mechanism").
