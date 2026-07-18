# Compressed live integration — report (continuation 8, workstream 2)

Branch `c8-compressed-live`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4/gravity-fullA-d4-c8-compressed-live`, built on
top of `diag/fullA-d4-exact` @ `33c93ff`. Files owned/touched: `full_aod_diag/d4_exact/oracle.jl`
(untouched — read-only reference), `oracle_fast.jl` (one additive branch), `context.jl` (untouched),
`compressed_moments.jl` (+1 additive function), `compressed_cc_inner.jl` (+1 additive function + a
documentation block), and three new files: `compressed_live.jl`, `test_compressed_live_integration.jl`,
`benchmark_compressed_live.jl`.

Continuation 7 built and validated `compressed_moments.jl`/`compressed_cc_inner.jl` as a standalone
bundle but never wired them into the live solver. This session does that wiring, tests it end to end,
and — importantly — finds that the standalone bundle's own benchmark numbers do **not** carry over
unmodified to the live D=4 setting; see §5.

## 1. Mode-flag API

```julia
evaluate_fullA_fast(x_free, ctx; moment_representation::Symbol = :dense, ...)
```

New kwarg on the EXISTING function (not a new function name), default `:dense`. Every existing caller
that does not pass this kwarg is byte-unaffected — the dense function body below the one new
early-return branch is untouched. `:compressed` dispatches to a new function,
`evaluate_fullA_fast_compressed` (`compressed_live.jl`), with the identical signature/return shape.

This is the "kwarg to `evaluate_fullA_fast`" option from the brief's three choices, chosen because it's
the single call site every consumer of the oracle already goes through (vs. a `ctx`-level flag, which
would need to be read inside `evaluate_fullA_fast` anyway, or a new `d4_exact_setup` option, which
would fix the mode for a whole ctx's lifetime rather than per-call).

**Default stays `:dense`.** Not fully confident enough in the compressed path's KNITRO-trajectory
robustness (see §5's noise finding) to flip the default; the equivalence suite passing everywhere it
was tested is necessary but not sufficient evidence for that with a live iterative solver, and the perf
picture at D=4 is mixed (§5), so there is no perf argument for flipping it here either. `:dense` remains
the trusted reference throughout.

## 2. What got wired

- **FG callback** (objective + gradient w.r.t. (ζ,λ), fused in this codebase, called on every KNITRO
  line-search/Newton step): replaced with `compressed_cc_value_grad` — no dense `G` ever touched.
- **Hessian callback**: stays dense (see §3) — materializes `obj.H`'s G columns from the already-built
  `CompressedFactual` **lazily, once per inner solve** (θ is fixed for the whole solve), then calls the
  UNCHANGED production `CS.hessian!`.
- **Primal-weight recovery / moment residual / KKT residual / gravity_raw**: all reuse the SAME
  lazily-materialized dense `obj.H` (materializing it at this point too, if the Hessian callback never
  fired — e.g. a warm-started already-converged point). These are one-time-per-outer-point costs
  regardless of representation, so there was no case for writing bespoke compressed formulas for them
  (the `compressed_moment_resid`/`compressed_transpose_contraction` primitive IS provided and available
  in `compressed_cc_inner.jl` for a future fully-compressed path, but the live wiring does not use it,
  documented as a deliberate choice, not an oversight).
- **Gravity moment column** (index `d == outer_constr_index`, the one G-column outside the inner-dual
  block): not compressed — it was never O(W·D²) to begin with (a single two-way-demeaned O(D²) scalar
  broadcast across all draws, per `moments/newGravityMoment!.jl`'s UoModel==1 branch). Computed by
  calling the EXISTING production `newGravityMoment!` on a throwaway 1-row buffer, then post-processed
  with the same SW/NormalizeMoments/usePMM formula every other column uses.
- **K column** (γ'_focal-direct counterfactual objective): also not compressed — `K[s] = θ[3+D]·SW[s]`,
  a trivial O(W) broadcast, computed directly.

## 3. Hessian-callback decision + exactness

KNITRO's registered inner Hessian callback is **dense** (`hessopt exact`, `KN_DENSE_ROWMAJOR`), not
HVP-style — it asks for the full `(1+ncol)×(1+ncol)` packed matrix every call. `compressed_cc_hvp`
computes exact HVPs in O(W·D), but assembling the dense Hessian from `1+ncol` (=18 at D=4) sequential
HVP calls would be **O(ncol·W·D)**, which loses to the dense path's single BLAS `gemm!` in `CS.hessian!`
at this scale (a highly-optimized multithreaded call vs. many sequential scalar loops). So the Hessian
callback stays dense, per the brief's own sanctioned fallback, refined to **lazy-once-per-inner-solve**
(not once-per-call): materialize `obj.H`'s G columns from `CompressedFactual` via the new
`materialize_dense_factual!` (in-place, reuses the already-computed `cf.winner`/`cf.wval` so it skips
the winner-search and per-loser σ-value work a from-scratch dense build would redo), cache the flag on
`CompressedCBState`, and call `CS.hessian!` unchanged thereafter.

**Exactness**: this makes the Hessian callback and everything downstream of it **bit-identical in
formula** to the dense path (same `CS.hessian!` call on an equivalent dense G) — not an approximation.
Measured tolerance across the whole equivalence suite (§4): every numeric field agrees to **1e-11 or
tighter**, almost all in the 1e-13–1e-17 range — floating-point re-association from summing in a
different order, not a formula difference. The genuinely-compressed piece (no dense G ever
materialized) is the FG callback alone.

## 4. Equivalence results (real numbers, not just PASS)

`test_compressed_live_integration.jl`, run on `demand.mit.edu`, `JULIA_NUM_THREADS=20`, D=4/W=8000.
**All phases PASS.** Per-field worst-case absolute difference across the ENTIRE suite (every phase
combined):

| field | max abs diff (dense vs compressed) |
|---|---|
| K_hard, gamma_focal_prime, gravity_value, gravity_R_sum/mean/beta | **0.0 (bit-identical)** |
| moment_resid | 8.3e-17 |
| max_abs_moment_resid | 5.6e-17 |
| weight_norm_resid | 2.2e-16 |
| gravity_raw | 2.0e-16 – 1.4e-20 |
| m_mean, mean_m_resid | ~4.9e-15 |
| max_abs_moment_kkt_resid | 4.5e-15 |
| Delta_dual, Delta_minus_delta | ~8–12e-15 |
| zeta | 7e-14 – 2e-13 |
| Delta_primal, primal_dual_gap | ~1–2e-13 |
| m_min | 3.1e-14 |
| lambda | 1.3–1.7e-11 (worst-case field, seen in Phase 3's h=1e-3 coordinate probes) |
| m_max | 5.2e-12 (cold-start-only) |
| logA | NaN (both dense and compressed hit the SAME "inner solve did not reach an accepted status" branch on the same hard point in Phase 3 — both fill logA with NaN, so `NaN-NaN=NaN` is expected, not a mismatch; `NaN > tol` is `false` in Julia so this correctly never fails the test) |

Suite coverage: calibration point; both named incumbents
(`upper_lfixcomposite_sr1_60s` κ=0.17245688540655113, `lower_lfixcomposite_fast_sr1_300s`
κ=0.005428799948779983, reconstructed from `candidate_registry.jl`'s `w` vectors via
`gravity_elimination.jl`'s pivot map); 5 gamma-profile-style points (A_od≡1, γ' ∈ {0.85,...,0.99}); 15
random feasible perturbations; **all 17 x_free coordinates × both FD signs × 3 h-values (1e-3/1e-5/1e-7)
= 102 probes** off the upper incumbent; warm and cold starts; a 5-step sequential warm-started
trajectory comparing every intermediate point (not just the endpoint); and an injected exact price tie
(reusing continuation 7's own proven `test_compressed_moments.jl` technique — perturbing one draw's `U`
so two origins tie bit-exactly) confirming `COMPRESSED_FALLBACK_COUNT` goes 0→1 and the fallen-back
result matches dense to 1e-9.

**One real bug found and fixed by this suite**: the Hessian callback originally called bare `hessian!`
instead of `CS.hessian!` (that function lives in module `CS`, not `Main`) — this produced a KNITRO
callback-error status (`nStatus=-500`) whenever the Hessian callback fired, silently masked in early ad
hoc testing because most warm-started points converged in a single FG call with zero Hessian calls. The
gamma-profile-style phase (γ' varied at A_od≡1, needing genuine iteration) and cold starts (which always
need Hessian calls) both caught it immediately.

## 5. Fallback mechanism

`COMPRESSED_FALLBACK_COUNT::Ref{Int}` (module-level in `compressed_live.jl`), incremented inside
`evaluate_fullA_fast_compressed`'s `catch e; if e isa TiedWinnerError` branch, alongside an `@warn` log
(tie count, example (draw,destination) pairs, running fallback count, hash of the x_free that triggered
it). On catch, it re-dispatches to the ordinary dense `evaluate_fullA_fast` for that one point and
returns dense's own result verbatim — never a different tie resolution (ties reuse
`lfix_incremental.jl`'s `TiedWinnerError`/`detect_price_ties` exactly, not reimplemented).
`reset_compressed_fallback_count!()` is provided for benchmark/test hygiene.

## 6. Measured speedup — the honest, mixed D=4 finding

`benchmark_compressed_live.jl`, D=4/W=8000, `JULIA_NUM_THREADS=20`, warmed medians.

**End-to-end** (`evaluate_fullA_fast`, full call):

| point | warm | dense | compressed | speedup |
|---|---|---|---|---|
| upper incumbent | true | 9.85 ms | 8.30 ms | **1.19x** |
| upper incumbent | false (cold) | 19.95 ms | 22.23 ms | **0.90x** |
| calibration | true | 8.93 ms | 7.56 ms | **1.18x** |
| calibration | false (cold) | 15.29 ms | 12.51 ms | **1.22x** |

**Per-phase breakdown** (median ms/call, `@prof` instrumentation, 10 cold solves each):

| phase | dense | compressed | ratio |
|---|---|---|---|
| inner_moment_build (the one-time build) | 5.90 | 2.89 | **2.04x faster** |
| inner_dual_fg_callback (per call, ~10 calls/solve) | 0.166 | 0.349 | **2.1x SLOWER** |
| inner_dual_hessian_callback (per call) | 0.964 | 1.019 | ~1.06x slower (noise-level) |
| inner_knitro_dual_solve (whole KN_solve, inclusive) | 11.4–13.2 | 14.2–15.2 | slower overall |

**This is a real, unflattering finding, reported honestly rather than cherry-picked**: at D=4, the
FG callback — the piece continuation 7's standalone report identified as the dominant win (its "CC
v+g" 2.15–2.31x) — is actually **slower** in the live wiring, not faster. Root cause, isolated
separately (`benchmark_compressed_live.jl`'s matched-work single-call test, both sides computing
objective+gradient): dense's FG callback is ONE BLAS `gemv!` on an 8000×17 matrix — at this tiny size
BLAS's vectorization/cache behavior is essentially free, while `compressed_dual_contraction`'s hand
-written Julia loop gathers through `cf.winner[s,d]` (indirect/scattered indexing into a small κ
matrix) — fewer total FLOPs (O(W·D) vs O(W·D²)) but a per-element overhead that dominates at D=4's
scale and isn't SIMD-friendly the way BLAS is. This is the SAME phenomenon continuation 7's own report
flagged for the isolated "transpose" row ("scalar bucket loop with scattered writes... competitive but
sometimes slower, 0.3x–2.7x") — but that report's "contract"/forward-direction row claimed a clean
2.2x win, which does NOT hold once compared against the REAL production baseline: their dense
comparator for "contract" was `[dot(G[s,:],β) for s in 1:W]` (W separate small BLAS `dot` calls, each
paying call overhead), not the single big `gemv!` the actual `PsiObjectiveBundleImplicit` callable uses
— an easier target than what's really in the callback's hot path. **Correcting this comparison is this
workstream's own finding, not something flagged before wiring it in.**

**Why the end-to-end number is still usually positive**: the ONE-TIME build saving (~3ms at D=4) is
large enough to outweigh the accumulated small per-call FG losses (~10 calls × ~0.18ms ≈ 1.8ms) in most
of the tested configurations, netting a modest 1.18–1.22x end-to-end win — except the cold upper-
incumbent case, where more FG/Hessian calls were needed and the accumulated losses flipped the net
effect to 0.90x (slower). This is a **narrow, call-count-sensitive win at D=4**, not the robust
multi-x speedup the standalone bundle's benchmark suggested. Per that same report's own documented
trend ("compressed advantage grows with D, smallest exactly where the problem is cheapest at D=4"), it
is plausible the FG-callback comparison flips in compressed's favor at larger D (their isolated
"contract" row showed 16.3x at D=10 even on the weaker dense baseline) — but this was NOT tested this
session (out of scope per the brief: "D=4 default is fine, this session doesn't need D-scaling from
you"), so it is reported as a documented hypothesis for a follow-up, not a verified claim.

**Isolated, apples-to-apples single-FG-call cost** (both sides computing objective+gradient,
matching the real callback's `length(g)>0` branch): dense 0.235 ms, compressed 0.272 ms, ratio
**0.87x** (compressed slower) — consistent with, and independently confirming, the live `@prof`
numbers above.

## 7. Base-state interface note for workstream 3 (verbatim, relay this)

`compressed_base_state(x_free0, ctx)` (`compressed_live.jl`) runs the compressed inner solve at
`x_free0` and returns a `BaseDualState` — **the exact, pre-existing struct from
`three_way_derivatives.jl`** (fields `x_free0`, `θ_full0`, `ζstar`, `λstar`, `m_star`,
`inner_status`), NOT a new/parallel type. `m_star` comes from one extra call to
`compressed_cc_value_grad` at the converged (ζ*,λ*) (its `dPsq` return value, exactly matching
`BaseDualState`'s own field comment `= dPsi(q_s*)`) — no dense G is needed for this step either, so
this part of the interface IS the fully-compressed O(W·D) path with no fallback-to-dense cost.
Drop-in compatible with `lfix_incremental.jl::build_lfix_base_cache(x_free0, ctx, base::BaseDualState)`
as-is — that function takes a `BaseDualState` regardless of how it was produced and does not care.

One thing NOT changed and not in scope here: `build_lfix_base_cache` itself still does its OWN full
dense `obj.moments!` call internally for its self-validation (`Gfull = zeros(W, obj.d);
obj.moments!(K, Gfull, ...)`), regardless of whether the `BaseDualState` it received came from the
dense or compressed path. This session did not touch `lfix_incremental.jl` (workstream 3's file) and
did not attempt to make that self-validation call compressed — if workstream 3 wants a genuinely
compressed base cache (i.e., also avoiding that one dense rebuild), `compressed_moments.jl`'s
`CompressedFactual` / `compressed_cc_inner.jl`'s `compressed_transpose_contraction` are the right
primitives to reuse for that, but building the compressed analog of `LFixBaseCache`'s per-destination
`contrib0`/`CONST_d` structure is new work, not something this session produced.

## 8. Files

New: `compressed_live.jl` (mode dispatch, callbacks, fallback, base-state interface),
`test_compressed_live_integration.jl` (equivalence suite), `benchmark_compressed_live.jl` (speedup
measurement). Modified additively: `compressed_moments.jl` (+`materialize_dense_factual!`),
`compressed_cc_inner.jl` (+Hessian-adapter-decision doc block, +`compressed_moment_resid`),
`oracle_fast.jl` (+1 kwarg, +1 early-branch in `evaluate_fullA_fast`, rest of that function untouched).
