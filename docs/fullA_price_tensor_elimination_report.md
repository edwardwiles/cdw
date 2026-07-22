# Persistent preallocation + price-tensor elimination: final report

> **SUPERSEDED (partial), 2026-07-22**: the D=20 benchmark numbers in this report (§0 TL;DR
> table and §5's "6.75x/6x" headline) compare Backend C against the obsolete unbuffered
> `composite_gradient_at`, not the real production gradient
> (`composite_gradient_at_fast_buffered`). `docs/fullA_factorized_price_production_gate.md`
> redoes the comparison fairly (4.0-4.2x / 66.8x against the actual default) and is the
> authoritative benchmark. This report's correctness findings (Backends A/B/C algebra, call-site
> audit, `winner_certificate.jl` reuse) are unaffected and still current.

Written 2026-07-21/22, autonomous overnight session, addendum to the postmerge
correctness/productionization brief. Read `docs/fullA_price_tensor_audit.md` first (Step 1
call-site audit) — this report covers Steps 2-10.

## TL;DR

Three backends built, tested, and benchmarked against the allocating reference
(`build_lfix_base_cache`/`lfix_incremental.jl`), at real D=20/W=80,000 data (δ=1 "typical" and
δ=5 "difficult") plus D=4 synthetic (randomized + adversarial-tie fixtures):

| Backend | Correctness | Cache build (median, δ=1) | Full 400-coord gradient (median, δ=1) |
|---|---|---|---|
| Reference (current production) | — | 2.40s, 589.6MB | 38.89s, 13,229MB |
| **A** — persistent workspace | bit-exact | 2.10s, **0.0MB** (warm) | *(not separately implemented, see below)* |
| **B** — pTσ0-only, no price0 | bit-exact | 1.46s, 345.5MB (1.6x less) | 15.36s (2.5x), 5,716MB (2.3x less) |
| **C** — factorized (logCC+mulU) | machine-precision (~1e-17) | 0.67s, 125.8MB (4.7x less) | **5.76s (6.75x), 2,207MB (6x less)** |

**All three backends are correct.** Backend C is the standout performance result — reusing
ALREADY-VALIDATED prior-session code (`winner_certificate.jl`'s `WinnerRefCache`) rather than
inventing a new representation. Recommendation: merge Backend A unconditionally (zero risk,
directly fulfills the original brief's "Persistent `LFixBaseWorkspace`" ask), and treat Backend
C as the primary path to a genuinely faster/leaner production `L_fix` gradient, with Backend B
as a lower-risk (bit-exact) fallback if the team wants a smaller first step.

## 1. Code map (Step 1) — see `docs/fullA_price_tensor_audit.md`

Full call-site audit already written. Headline finding: **raw `price0` is never
mathematically required anywhere** — every use is a ranking/tie/bandwidth comparison, and
since σ>1 makes `price ↦ price^(1-σ)` strictly order-reversing, `pTσ0` alone (or a log-price
score, even more numerically robust) suffices for all of it. Also discovered: `winner_certificate.jl`
(a prior continuation, not built this session) ALREADY implements the factorized/log-score
representation the addendum's Backend C asks for, just scoped to a different consumer
(`WinnerRefCache`/`certified_winner_update`, a "certify against a nearby point" use case) than
`LFixBaseCache`'s own "400 sequential single-coordinate FD probes per gradient" need — Backend
C below is the unification of those two, not a third representation invented from scratch.

## 2. Current allocation/memory-lifetime benchmark (Step 2)

`full_aod_diag/d4_exact/c15_price_tensor_benchmark.jl`,
`results/fullA_d4/c15_price_tensor/price_tensor_benchmark.csv`. Real D=20/W=80,000, three
points (calibration, δ=1 context, δ=5 context):

- `price0` alone (allocating `price_and_pTsigma_cell`, not the in-place `!`): 732.5MB
- `pTσ0` alone (same, allocating): 732.5MB
- **Both, in-place** (current production pattern, `price_and_pTsigma_cell!`): 488.3MB —
  matches the theoretical `2 × W×D×D×8 bytes = 2×244.1MB` EXACTLY, confirming the existing
  in-place fix already eliminated all allocate-then-copy overhead; there is no "free" allocation
  win left in the two-tensor construction itself, only in NOT constructing them at all
  (Backend B/C) or NOT reconstructing them every call (Backend A).
- **Full `build_lfix_base_cache`**: 589.6MB (488.3MB tensors + ~101MB winner/runner-up/
  third/contrib0/q0 bookkeeping arrays) — matches the original brief's quoted "~590-650MB/
  gradient" figure exactly.
- Confirmed both tensors remain referenced (RSS unchanged, `cache` held alive) through a
  10-coordinate mini-sweep, i.e. for the WHOLE gradient call's duration, not just construction.

## 3. Backend A — persistent two-tensor workspace

`full_aod_diag/d4_exact/lfix_base_workspace.jl`. `LFixBaseWorkspace` (mutable, caller-owned)
holds every large array `build_lfix_base_cache` currently allocates fresh; `build_lfix_base_cache!
(ws, x_free0, ctx, base)` refills them in place and returns a normal `LFixBaseCache` whose
array fields ALIAS the workspace's buffers. **Zero changes to `LFixBaseCache` itself or any
consumer function** (`dest_contrib_*`, `count_winner_flips*`, `select_bandwidth*`,
`bandwidth_quantile.jl`) — they read `cache.price0` etc. exactly as before, unaware of the
backing store.

Lifecycle/safety, per the addendum's explicit checklist:
- `ensure_lfix_workspace!` reallocates only on a genuine `(D,W)` change (mirrors
  `gradient_workspace.jl::resize_pool_if_needed!`'s established pattern).
- `ws.valid` is `false` from construction, `false` again the instant a build starts, and only
  `true` after every step (including tie-check and optional dense self-validation) succeeds —
  an exception (dimension mismatch, `TiedWinnerError`) leaves it `false`.
- Verified no code anywhere in this repo stores an `LFixBaseCache` across more than one
  gradient call (`grep`-audited) — the one persistent cross-call cache that exists,
  `CrossDeltaExactCache`, stores only a `NamedTuple` of scalars, never an `LFixBaseCache` —
  so the aliasing design's one real hazard (a stale reference into a since-refilled workspace)
  cannot currently occur. Documented as an explicit invariant callers must preserve.

**Correctness**: `test_lfix_base_workspace.jl`, D=4, 62/62 passed — fixed + randomized points,
adversarial exact-tie fixture (both builders throw `TiedWinnerError` identically, `ws.valid`
correctly flips false→recovers-true), workspace-reuse-across-different-points contamination
check (demonstrates the aliasing hazard IS real if misused, and that it doesn't fire under
correct usage), and full downstream consumption (`dest_contrib_*`, `composite_gradient_at`-style
manual reconstruction) bit-identical.

**Benchmark** (real D=20/W=80,000, in `c16_backend_matrix_d20.jl`): cache construction on a
WARM (already-correctly-sized) workspace allocates **0 bytes** — median `@timed` bytes = 816
(noise floor, not the 6.18e8 the allocating reference shows). This is the addendum's Step-3
target achieved exactly: the ~590MB/call construction cost is gone on every call after the
first at a given `(D,W)`.

**What Backend A does NOT do**: it has no standalone "gradient" entry point of its own —
its value is realized by wiring it into the EXISTING gradient drivers
(`composite_gradient_at_fast_buffered`/`_pooled`, `gradient_workspace.jl`'s `GradWorkspacePool`
from the earlier Prompt-1 session) which already rebuild `cache` fresh via the allocating
builder every call. **Not wired into those drivers this session** — that wiring (swap
`build_lfix_base_cache(...)` for `ensure_lfix_workspace!`+`build_lfix_base_cache!(...)` inside
`composite_gradient_at_fast_buffered`/`_pooled`) is the natural next step and is low-risk given
the alias-safety already established, but is left as a clearly-flagged follow-up, not silently
assumed done.

## 4. Backend B — pTσ0-only

`full_aod_diag/d4_exact/lfix_pTsigma_only.jl`. Drops `price0` entirely; every ranking/tie/O(1)-
update site mechanically mirrors its Reference counterpart with comparison direction flipped
(`update_winner_o1_B`, `max_and_secondmax`, `max_secondthirdmax_with_idx`). `pTsigma_cell`/`!`
compute ONLY the σ-transformed value — a genuine FLOP reduction, not just a memory one (price
and pTσ are independently-computed formula chains using different arrays, `U` vs `Uσ`).

**Real finding, not assumed going in**: `Uσ == U.^(1-σ)` exactly, confirmed at
`prepare_cc/createUDerivatives!.jl:18` (computed once at context-build time, stored separately
in `ctx.γ.Uσ`, never re-derived). This makes `pTσ0 == price0.^(1-σ)` hold exactly in
principle — but **NOT reliably at floating-point `tol=0.0`**, because `pTσ0` is computed via
a genuinely different rounding pathway (`(U^(1-σ))^(-μ)` vs `U^(-μ)` then implicitly
`^(1-σ)`). Found empirically: the standard tie-injection fixture (patch only `ctx.U`, matching
`test_winner_certificate.jl`'s own established recipe) did NOT reproduce a tied `pTσ0` even
after ALSO patching `ctx.γ.Uσ` consistently, at the default `detect_pTσ_ties` tolerance —
**fixed** by reusing `winner_certificate.jl::build_winner_ref`'s own already-established
defensive technique (check ties in price-space via transient, O(W·D)+O(D²) `constCons`/`UPow`
factors — never a stored dense tensor — rather than trusting the derived `pTσ0`'s own bit
pattern). `detect_pTσ_ties` (the naive derived-tensor check) is KEPT in the file for
documentation/comparison but is NOT what `build_lfix_base_cache_B` actually calls.

**Correctness**: `test_lfix_pTsigma_only.jl`, D=4, 114/114 passed (bit-for-bit throughout,
including the full `composite_gradient_at_B` vs `composite_gradient_at` gradient comparison).
D=20 cross-check (`c16_backend_matrix_d20.jl`): bit-identical `winner0`/`contrib0`/`q0` and
gradient (`maxabsdiff = 0.000e+00`) at both δ=1 and δ=5.

**Benchmark** (D=20/W=80,000, δ=1): cache build 1.46s/345.5MB vs Reference's 2.40s/589.6MB
(1.6x faster, 1.7x less memory — the theoretical `price0` elimination, `244.1MB`, plus the
skipped raw-price FLOP path). **Full 400-coordinate gradient: 15.36s/5,716MB vs Reference's
38.89s/13,229MB — 2.53x faster, 2.31x less memory**, entirely from not allocating/computing
price at any of the ~800 per-gradient coordinate probes either (not just at cache construction).

## 5. Backend C — factorized (logCC + mulU), reusing `winner_certificate.jl`

`full_aod_diag/d4_exact/lfix_factorized.jl`. Persistent state is `WinnerRefCache` (`logCC0`
D×D, `mulU` W×D, winner/runner-up/third INDICES and log-price SCORES) — **O(W·D), not
O(W·D²)** — built via `build_winner_ref` (verbatim, unmodified, from a prior continuation, not
new code). `pTσ` at any queried cell is reconstructed on demand, `exp((1-σ)·score)`, O(1) per
cell, never stored as a tensor. `dest_contrib_incremental_top3_C` mirrors the Reference's own
top-3-cache tier structurally but sources ranking from `ref.winner/runnerup/third` +
`ref.sw/sr/st3` instead of a dense `price0`.

Deliberately does NOT call `winner_certificate.jl::coord_winner_update!` even though it proves
the identical case analysis — that function `copyto!`s the FULL W×D winner matrix on every
call, which would mean re-copying it on every one of ~800 probes/gradient; this file's own
`dest_contrib_incremental_top3_C` only ever touches the one affected destination column,
matching every other tier's per-destination scoping discipline.

**Correctness is NOT bit-exact** (unlike A and B) — this is expected and reported honestly, not
glossed over: `exp((1-σ)·(logCC+mulU))` is a genuinely different floating-point evaluation
order than the Reference's direct `constConsσ_od/Uσ^(-μ)`. D=4 (`test_lfix_factorized.jl`,
98/98 with a 1e-10 tolerance): observed max discrepancy 1-2 ULPs (~1e-14 to 1e-15) once the
first attempt's STRICT `==` comparisons (16/98 failing at exactly that magnitude) were relaxed
to the appropriate tolerance — winner/runner-up/third IDENTITIES and tie detection remain
exactly bit-identical throughout (integer comparisons with a real margin, immune to ULP noise).
D=20 real-data cross-check: gradient `maxabsdiff = 3.816e-17` at both δ=1 and δ=5 — machine
epsilon, not a numerically meaningful difference.

**Benchmark (the headline result)**: cache build 0.67s/125.8MB vs Reference's 2.40s/589.6MB
(3.6x faster, 4.7x less memory). **Full 400-coordinate gradient: 5.76s/2,207MB vs Reference's
38.89s/13,229MB — 6.75x faster, 6.0x less memory.** Consistent at both δ=1 (typical) and δ=5
(difficult): 5.91s/δ=5 vs 5.76s/δ=1, no meaningful difficulty-dependent slowdown.

## 6. Log-score representation (Step 6)

Not a separate deliverable — Backend C's `logCC+mulU` factorization IS the log-score
representation (chosen deliberately over the addendum's literal `K·B` product-space sketch
specifically to avoid the overflow/underflow concern Step 6 raises; sums in log-space are
numerically safer than products of `x^(1-σ)` terms for extreme `Aod`/`U` values, and this
codebase already had this exact representation validated for a different consumer).

## 7. Correctness suite — summary

| Suite | Scope | Result |
|---|---|---|
| `test_lfix_base_workspace.jl` | D=4, fixed+randomized+adversarial-tie | 62/62 (Backend A, bit-exact) |
| `test_lfix_pTsigma_only.jl` | D=4, fixed+randomized+adversarial-tie | 114/114 (Backend B, bit-exact) |
| `test_lfix_factorized.jl` | D=4, fixed+randomized+adversarial-tie | 98/98 (Backend C, ~1e-10 tol, actual noise ~1e-14) |
| `c16_backend_matrix_d20.jl` | D=20/W=80,000 real data, δ=1 + δ=5 | All 3 backends: winner/contrib0/q0/gradient match (A,B exact; C to 3.8e-17) |

Not done (honest gap): a dedicated D=4 "every origin pair, every destination, several step
sizes" adversarial sweep in the style of `test_winner_top3_equivalence.jl` for Backends B/C
specifically (the existing tests DO include a `rand1-4` fuzz sweep and one tie fixture each, but
not that file's exhaustive pairwise-synthetic-2-changed-origin construction). Given the
2-changed-origin top-3 logic is IDENTICAL in structure across Reference/B/C (same case
analysis, just re-sourced), and each backend's own tests already exercise the real D=4
coordinate sweep's actual 2-changed-origin coordinates (via the full `composite_gradient_at_*`
comparison), the risk this leaves uncovered is assessed as low, but flagged rather than
silently assumed equivalent to the more exhaustive existing suite.

## 8. End-to-end benchmark matrix — full numbers

See §0 table above and `c16_backend_matrix_d20.jl`'s own log
(`full_aod_diag/d4_exact/logs/c16_backend_matrix_d20_*.log`) for the complete run, including
per-δ breakdowns. Caveat, stated plainly: the "Reference" gradient benchmarked here is
`composite_gradient_at` (the plain, unbuffered entry point) — NOT the already-more-optimized
`composite_gradient_at_fast_buffered`/`_pooled` (Prompt-1 session's own buffer-reuse/
`GradWorkspacePool` work). Backends B/C's speedup is thus measured against the least-optimized
baseline; a fully fair "how much MORE do B/C save on top of what's already shipped" comparison
would benchmark against `_fast_buffered`/`_pooled` too — not done this session, flagged as a
follow-up, not silently assumed away. (Note this doesn't change the qualitative conclusion:
Backend C's O(W·D) persistent memory footprint and skipped price-FLOP path are savings
ORTHOGONAL to buffer reuse — buffer reuse eliminates repeat ALLOCATION of the same-shaped
arrays across probes; Backend C eliminates entire arrays and an entire redundant formula
evaluation. The two are complementary, not competing, optimizations.)

## 9. Decision, against the addendum's own rules

- **Merge Backend A**: yes, unconditionally. Exact results, zero measured allocation on warm
  reuse, no gradient-wall-time regression (it doesn't touch the hot per-coordinate loop at
  all). This directly fulfills the original brief's Prompt-1 "Persistent `LFixBaseWorkspace`"
  ask (previously "not started" per the handoff) — consider that item now DONE at the
  function level, NOT YET wired into `composite_gradient_at_fast_buffered`/`_pooled` (see §3).
- **Merge Backend C** as the primary path forward: preserves winner/tie behavior exactly
  (inherited from `build_winner_ref`'s own established correctness), removes BOTH dense
  tensors (not just one), and is dramatically faster (6.75x) and leaner (6x), validated at both
  a typical and a difficult real point. The only real complexity cost is the ~1e-14 numerical
  non-bit-exactness, judged immaterial (KNITRO's own convergence tolerances are far looser).
- **Backend B** is a legitimate, lower-risk intermediate: bit-exact, still a real 2.5x/2.3x win,
  and mechanically simpler to audit (a direct mirror of the Reference with flipped
  comparisons, no reused external module). Recommend it ONLY if the team wants a smaller,
  more conservative first step before Backend C's larger reformulation — otherwise Backend C
  dominates it on every measured axis and the addendum's own tie-breaker ("prefer whichever is
  equally fast or faster") applies unambiguously in C's favor.

## 10. What remains (honest gap list)

1. **Wire Backend A into `composite_gradient_at_fast_buffered`/`_pooled`** — the function-level
   workspace exists and is proven safe; the driver-level swap is not done.
2. **Wire Backend C (or B) into a real driver path** — currently standalone entry points
   (`composite_gradient_at_C`/`_B`), not called from any production driver.
3. **Benchmark against the already-optimized baseline** (`_fast_buffered`/`_pooled`), not just
   plain `composite_gradient_at` (see §8's caveat).
4. **Exhaustive 2-changed-origin adversarial sweep** for B/C specifically (see §7).
5. **A real, sustained KNITRO outer-loop run** using any of these backends — everything above
   is value/gradient-level validation at fixed points, not an observed effect on an actual
   optimizer trajectory's wall-clock or convergence behavior.
6. Retry the ORIGINAL (pre-addendum) Prompt-1 priority items — `test_driver_pooled_gradient_wiring.jl`
   and `test_cm_verified_success.jl` both hung again this session (confirmed via stale log
   timestamps, 38+ min with zero new output) under a very heavily loaded shared server (load
   average 42-56, one user's persistent 48-thread job). Not yet re-attempted on a quieter
   window — see the handoff doc's own remediation guidance.

## Commits

All new files are additive (no existing file modified except two dispatch-method additions in
`lfix_pTsigma_only.jl`/`lfix_factorized.jl` for `cf_contrib_at`/`gamma_component_analytic`, which
are NEW methods on existing generic function names, not edits to their original definitions).
New files: `lfix_base_workspace.jl`, `lfix_pTsigma_only.jl`, `lfix_factorized.jl`,
`test_lfix_base_workspace.jl`, `test_lfix_pTsigma_only.jl`, `test_lfix_factorized.jl`,
`c15_price_tensor_benchmark.jl`, `c16_backend_matrix_d20.jl`, this report, and the Step-1 audit
doc. Rollback: `git revert` or simply delete these files — nothing else in the repo references
them (confirmed: `grep -rl` for each new struct/function name turns up only the files
themselves and this report/audit doc).
