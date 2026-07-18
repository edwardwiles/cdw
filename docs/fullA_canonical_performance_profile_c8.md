# Full-A D=4 exact formulation: canonical performance profile (Continuation 8, Wave 2)

Continuation 8, Wave 2, workstream A. Supersedes `docs/fullA_performance_profile_v2.md`
(continuation 3) as the canonical performance reference going forward — that document
predates both of Wave 1's live-wiring workstreams: **compressed-mode moments**
(`compressed_live.jl`, `docs/compressed_live_integration_report.md`) and the **winner
accelerators** (coordinate-specialized top-3 update + winner-margin certificate,
`docs/winner_accelerator_live_wiring.md`). Everything below is measured directly on
commit `d6e3b05` (tip of `diag/fullA-d4-exact`, includes all of Wave 1), branch
`c8-perf-profile`, machine `demand.mit.edu`, `JULIA_NUM_THREADS=20`.

## 0. What changed vs `fullA_performance_profile_v2.md`, and why this document exists

`fullA_performance_profile_v2.md` profiled only the **dense** value-callback path and the
**full-rebuild** `L_fix` gradient (32 complete moment-matrix rebuilds per gradient). Since
then: (1) `compressed_live.jl` wires an O(W·D) compressed winner-form representation into
the live inner-dual FG callback, with the dense path kept as the trusted default; (2) the
hard `L_fix` coordinate gradient (`composite_gradient_at_fast`) now uses genuinely
incremental O(1)/O(D)-per-draw tiers (not full rebuilds) plus a coordinate-specialized
top-3 winner update and an optional persistent winner-margin certificate. Neither existed
when v2 was written. This document profiles **both** paths side by side, at fine
granularity, across a D/W grid, in one warmed process — reusing the nested-`@prof`-timer
discipline `fullA_performance_profile_v2.md` §3 established (separate `prof_reset!()`
scopes per warm/cold/mode condition) rather than reinventing it.

**A genuine (small) infra improvement made along the way**: `audit_jach_d6d8.jl` (this
directory's existing D=6/8 profiling script) runs in a *separate process* from its D=4
counterpart because directly including `context.jl` twice in one process redefines the
`CounterfactualSensitivity` module's types and breaks type identity for objects already
built against the old type (see that script's own header comment). The hazard is
specifically *double-including* `context.jl`, not calling `d4_exact_setup`/
`d_exact_setup_scaled` multiple times with different `(D,W)` — those are ordinary function
calls. This harness includes `context_scaled.jl` exactly once as its sole root include
(which itself includes `context.jl` exactly once), then calls `d4_exact_setup` for D=4/W=8000
and `d_exact_setup_scaled` for every other grid point — all safely inside **one** warmed
process, covering D=4/W=8000, D=4/W=80000, and D=6/8/10/W=8000 together. All JIT/compile
costs are paid once during an untimed warm-up phase before any timed measurement.

Harness: `full_aod_diag/d4_exact/c8_perfprofile_harness.jl` (+ a small targeted follow-up,
`c8_perfprofile_certseq_recheck.jl`, §5.4). Raw log + CSVs:
`results/fullA_d4/d6e3b05/c8_perfprofile/`.

## 1. Part A — exact value callback, dense vs compressed, D=4/W=8000

`evaluate_fullA_fast(...; moment_representation=:dense|:compressed)`, upper incumbent
(`upper_lfixcomposite_sr1_60s`), N=50 warm reps / N=15 cold reps, each mode profiled in its
own `prof_reset!()` scope.

### Warmed (median ms)

| component | dense | compressed | ratio (d/c) |
|---|---|---|---|
| **TOTAL** | **11.047** | **8.223** | **1.343** (compressed faster end-to-end) |
| inner_moment_build | 5.531 | 2.353 | 2.350 (compressed faster) |
| winner_compute | 2.030 | 1.728 | 1.175 |
| materialize_dense_for_postproc (compressed-only, lazy) | — | 1.153 | n/a |
| inner_knitro_dual_solve (inclusive) | 0.797 | 0.654 | 1.217 |
| **inner_dual_fg_callback** | **0.211** | **0.251** | **0.839 (compressed SLOWER)** |
| kkt_residual_compute | 0.195 | 0.166 | 1.174 |
| moment_resid_compute | 0.192 | 0.171 | 1.123 |
| primal_weight_recovery | 0.171 | 0.156 | 1.096 |
| primal_divergence_compute | 0.148 | 0.134 | 1.106 |
| moments_reuse | 0.143 | 0.146 | 0.981 |
| gravity_compute | 0.011 | 0.007 | 1.494 |
| reconstruct_full | 0.001 | 0.001 | 1.779 |
| n_fg_calls/n_hess_calls per call | 1 / 0 | 1 / 0 | (both warm-started, single check) |

### Cold (median ms, n_fg≈10, n_hess≈9 calls/solve both modes)

| component | dense | compressed | ratio |
|---|---|---|---|
| **TOTAL** | **22.167** | **21.722** | **1.021** (near parity) |
| inner_knitro_dual_solve (inclusive) | 11.803 | 14.520 | 0.813 |
| inner_moment_build | 5.239 | 2.540 | 2.062 |
| winner_compute | 1.769 | 1.766 | 1.002 |
| inner_dual_hessian_callback | 0.988 | 1.045 | 0.946 |
| inner_dual_fg_callback | 0.167 | 0.268 | 0.623 (compressed markedly slower here, ~10 calls/solve) |

**This reproduces and refines Wave 1's finding at finer granularity, not just the coarse
end-to-end number**: `inner_moment_build` is robustly 2.0–2.35x faster compressed (the
one-time O(W·D) build vs O(W·D²) dense), consistent with Wave 1's report. The
`inner_dual_fg_callback` — the piece called on *every* KNITRO line-search/Newton
iteration — is genuinely **slower** compressed at D=4 (0.84x warm, 0.62x cold), exactly
matching Wave 1's root-cause explanation (dense's callback is one BLAS `gemv!` on an
8000×17 matrix, essentially free at this size; compressed's hand-written scattered-index
loop has fewer FLOPs but worse per-element/SIMD behavior at D=4's scale). The end-to-end
`TOTAL` still favors compressed warm (1.34x) because the one-time moment-build saving
dominates a warm solve's single FG call; cold solves (10 FG calls) erode that advantage
toward parity (1.02x), also matching Wave 1's own cold-case finding.

## 2. Part B — D=4/W=80000 (does the verdict flip with 10x more draws?)

Feasible point found via the "natural A_od theta" trick (see §5.2); N=25 warm reps.

| component | dense | compressed | ratio |
|---|---|---|---|
| **TOTAL** | **123.808** | **110.067** | **1.125** |
| inner_moment_build | 74.114 | 41.156 | 1.801 |
| winner_compute | 28.612 | 28.783 | 0.994 |
| **inner_dual_fg_callback** | **3.160** | **3.957** | **0.799 (still SLOWER)** |
| inner_knitro_dual_solve (inclusive) | 3.806 | 4.554 | 0.836 |

**Verdict does NOT flip at D=4/W=80000**: `inner_dual_fg_callback` ratio goes from 0.839
(W=8000) to 0.799 (W=80000) — compressed stays slower, if anything marginally more so.
This confirms the harness's own hypothesis was correctly stated as open (plausible either
way) and resolves it: at fixed D=4, scaling W alone does **not** favor compressed's
FG callback — both dense's BLAS `gemv!` and compressed's scattered loop scale the same way
in W (both are O(W·D)-ish per call at fixed D), so the D=4-specific per-element/SIMD
constant-factor disadvantage Wave 1 identified persists regardless of W. (See §4, Part D,
for the D-scaling axis, which behaves very differently.)

## 3. Part C — hard `L_fix` gradient breakdown, D=4/W=8000

### C1. Full 15-coordinate gradient wall time: top3 vs generic, threaded vs serial, adaptive vs fixed-h

Median ms, N=15 reps, `upper_lfixcomposite_sr1_60s`, shared base state:

| h_mode | threaded | top3 | generic | generic/top3 |
|---|---|---|---|---|
| adaptive | true | 36.08 | 32.63 | 0.904 |
| adaptive | false | 124.78 | 125.95 | 1.009 |
| fixed | true | 19.42 | 21.21 | 1.092 |
| fixed | false | 51.32 | 46.46 | 0.905 |

**Top-3's full-gradient contribution is noise-level at D=4** (ratios 0.90–1.09, no
consistent direction) — this directly reproduces Wave 1's own finding
(`docs/winner_accelerator_live_wiring.md`: "0.93×–1.17×, noise-dominated, no measurable
full-gradient speedup at D=4"; only 3/15 coordinates ever reach the fallback tier). Not a
new result, a confirmation at this harness's own N/point. **Threading is the dominant
lever here**, not the top-3 fix: adaptive-h threaded (36ms) vs serial (125ms) is a genuine
~3.4x; fixed-h threaded (19–21ms) vs serial (46–51ms) is ~2.4x.

### C2. Component decomposition (external timing around the existing public functions — no source edits)

Median ms, N=10 warmed reps, h_mode=:adaptive:

| component | top3 | generic |
|---|---|---|
| base_state solve (shareable with eval_F) | 6.549 | (same) |
| cache_build (`build_lfix_base_cache`) | 14.218 | 13.995 |
| gamma_analytic (closed-form, w[1]) | 0.012 | 0.008 |
| **bandwidth_selection** (`select_bandwidth`, 14 coords) | **45.305** | 45.497 |
| **fd_probes** (`a_block_fd_component` x2, 14 coords) | **66.379** | 65.733 |
| reconstructed total | 132.46 | 131.78 |

This localizes where the ~125–133ms serial adaptive-h cost actually goes (matching
`docs/fullA_p1_warmed_profile.md`'s Part A finding that most of a composite-gradient call
lives inside the per-coordinate bandwidth+FD loop, not the base solve): `bandwidth_selection`
(the mass-targeting bisection, up to 7 winner-flip counts per coordinate) and `fd_probes`
(2 incremental evaluations per coordinate) are comparable in size and together account for
essentially the entire non-base-solve cost. `cache_build` (one `build_lfix_base_cache`
call) is a fixed ~14ms overhead regardless of A-block loop method. top3 vs generic shows
no consistent direction inside either sub-component either — consistent with C1.

### C2b. Isolating the winner-update contribution at the known fallback coordinates

Per-coordinate cost (`select_bandwidth` + 2x `a_block_fd_component`), median of 30 reps,
at an ordinary single-changed-origin coordinate (k=2) vs the three coordinates that reach
the 2-changed-origin fallback (k=14,15,16 — sharing the gravity pivot's destination, per
`docs/winner_accelerator_live_wiring.md`'s own audit):

| coord | top3 (ms) | generic (ms) | generic/top3 |
|---|---|---|---|
| k=2 (ordinary) | 6.901 | 7.059 | 1.023 |
| k=14 (fallback) | 8.706 | 9.514 | 1.093 |
| k=15 (fallback) | 7.817 | 8.338 | 1.067 |
| k=16 (fallback) | 8.395 | 9.467 | 1.128 |

The fallback coordinates DO show a small, consistent top3 advantage (1.07–1.13x,
i.e. generic costs 7–13% more) exactly where the O(D) rescan is actually invoked — a real
but small signal (D=4's O(D) rescan is itself a 4-element inner loop) that gets buried in
noise at the full-15-coordinate level (C1) because it's diluted by 12 unaffected
coordinates plus the fixed base/cache overhead. This is consistent with, and slightly
sharper than, Wave 1's own characterization ("bounded at D=4 ... grows with D") — see §4
for whether this shows up more clearly as D grows.

### C3. Winner-margin certificate, in `composite_gradient_at_fast`'s own context

**(a) Single full 15-coordinate gradient call**, `winner_cache_mode=:none` vs
`:certificate` with a cold (freshly-reset) `PersistentWinnerCache`:

| | median ms (N) |
|---|---|
| `:none` | 35.08 (N=10) |
| `:certificate`, cold anchor | 36.99 (N=8) |
| overhead of the diagnostic | **+1.91ms (+5.4%)** |

Close to neutral, as the brief anticipated — a single cold anchor build costs a real but
small ~2ms extra on top of a ~35ms full-gradient call.

**(b) Sequence of 10 nearby points**, one `PersistentWinnerCache` reused across all 10,
same magnitudes as Wave 1's own line-search sweep ({1e-3, 5e-3, 1e-2, 2e-2, 5e-2}). The
first harness run showed a large first-point outlier in BOTH `:none` (504ms) and
`:certificate` (442/406ms) sequences — a genuine finding worth explaining rather than
burying: a targeted follow-up (`c8_perfprofile_certseq_recheck.jl`, forced `GC.gc(true)`
before each pass, 3 repeated passes) shows the outlier appears on a **different** point
index each pass (point 1, then point 1 again post-GC, then point 2 on a third pass with no
forced GC) for **both** `:none` and `:certificate` — i.e. it is per-point inner-CC-dual-solve
variability (or shared-machine contention on this 208-core box) uncorrelated with the
certificate flag, not a certificate-specific cost. **Honest conclusion: at the full
composite_gradient_at_fast level, embedding the certificate diagnostic in a sequence of
nearby calls is a wash, dominated by inner-solve noise** — each of the 10 points still
pays its own full `solve_base_state` + `build_lfix_base_cache` + FD-loop cost regardless of
the certificate flag (nothing in the current wiring shares those across sequence points),
so there was never a mechanism for the certificate's win to show up at this level; this is
the correct, non-cherry-picked answer to "does it help in this context," not a failure to
find one.

**Isolating just the certificate's own added call** (`winner_value_update!` alone, no
gradient math around it, same K=10 sequence) — this is where the real signal lives, free of
inner-solve noise:

```
cost per point (ms): [42.22, 1.00, 0.98, 0.99, 0.97, 0.98, 0.98, 0.98, 0.99, 0.96]
first (cold anchor build) vs subsequent (certified) ratio: 43.0x
certified_frac = 99.26%, rescanned_frac = 0.74%, full_fallback_frac = 0.0%
```

This is a clean, real, ~43x cold-vs-warm signal for the added diagnostic call itself
(even larger than Wave 1's own standalone 6.2–6.5x, likely because Wave 1's sweep used
threaded certificate calls at a coarser granularity — not re-derived here, just noted) —
confirming the certificate mechanism works exactly as designed **as an isolated call**,
while honestly reporting that this benefit does not (yet) propagate to
`composite_gradient_at_fast`'s own wall time, because nothing in that function's current
wiring lets a certified winner matrix substitute for the FD-loop's own per-coordinate work.

## 4. Part D — D-scaling grid (D=6, 8, 10; W=8000)

All three D values reached a feasible point via the "natural A_od theta" trick (§5.2) on
the first try — D=10 was **not** skipped; the standing brief's "D=4/6/8 partial is
acceptable" allowance was not needed.

| D | setup wall | TOTAL dense (ms) | TOTAL compressed (ms) | TOT ratio | fg_callback dense (ms) | fg_callback compressed (ms) | fg ratio | full gradient, top3+threaded (ms) |
|---|---|---|---|---|---|---|---|---|
| 4 (§1, for reference) | — | 11.05 | 8.22 | 1.34 | 0.211 | 0.251 | **0.84** | 36.08 |
| 6 | 0.36s | 23.24 | 14.69 | 1.58 | 0.801 | 0.355 | **2.26** | 67.83 |
| 8 | 0.35s | 45.10 | 27.79 | 1.62 | 0.421 | 0.586 | 0.72 | 118.40 |
| 10 | 0.68s | 62.40 | 41.59 | 1.50 | 0.533 | 0.763 | 0.70 | 150.30 |

**`inner_moment_build`'s dense-vs-compressed ratio grows monotonically and substantially
with D** (2.35x at D=4 → 3.22x at D=6 → 3.27x at D=8 → 3.64x at D=10), matching
Wave 1's own documented hypothesis exactly ("compressed advantage grows with D"). The
`TOTAL` end-to-end ratio also improves with D (1.34x → 1.58–1.62x) once D leaves 4,
though it does not increase monotonically past D=6 in this data (D=10's 1.50 is
slightly below D=6/D=8's ~1.6) — plausibly `winner_compute` (which does not benefit
from compression the way `inner_moment_build` does — both modes call the same
`compute_winners_fast`) growing as a larger absolute share at bigger D partially offsets
the moment-build gain; not chased further here (this document reports the realized
numbers, not a fitted trend).

**The `inner_dual_fg_callback` verdict is genuinely noisy at D=6/8/10**, not a clean
D-scaling story: it flips to compressed-FASTER at D=6 (2.26x) but flips BACK to
dense-faster at D=8 (0.72x) and D=10 (0.70x). This is very likely an artifact of the
callback's own absolute cost being tiny at this scale (0.3–0.8ms per call, single
FG-call-per-warm-solve, W=8000 fixed) — i.e. dominated by call overhead and scheduling
noise rather than a real, monotonic D-dependent crossover. **Reported honestly as an open,
noisy result** rather than forced into a clean narrative: the D=4/W=80000 result (§2, held
D fixed, varied W by 10x) is the cleaner, more trustworthy test of "does the FG-callback
verdict flip," and it did not flip. The D-axis result here is suggestive (D=6 flipped) but
not conclusive given D=8/10 reverted — a genuine D-scaling FG-callback study would need
more reps per D and probably a fixed W/D-normalized comparison, out of scope for this
document's time budget.

**Full-gradient wall time** (top3, threaded, adaptive-h) grows roughly linearly with D²
(the free-parameter count): 36ms (D=4, 15 coords) → 68ms (D=6, 35 coords) → 118ms (D=8,
63 coords) → 150ms (D=10, 99 coords) — consistent with a per-coordinate cost that is
roughly flat-to-mildly-growing in D, not a new finding but a useful confirmation that the
incremental machinery's O(1)/O(D)-per-draw design is holding up at these sizes rather than
degrading superlinearly.

## 5. Methodology notes

### 5.1 Warmed-process discipline
One Julia process (`c8_perfprofile_harness.jl`), `JULIA_NUM_THREADS=20`, all JIT/compile
costs paid in an untimed warm-up phase (every mode × threaded × h_mode × multi_method
combination invoked once, untimed, before Part A begins). Each timed condition uses its
own `prof_reset!()` scope, matching `fullA_performance_profile_v2.md` §3's methodology.

### 5.2 D-scaling feasible-start trick
Reused, not re-derived: `run_d6_pilot.jl`'s finding that the calibration point (A_od
theta≡1) is cold-infeasible at D≥6, and that the "natural" A_od theta baked into
`ctx.θ0_up` (the pre-gammanorm structural values) is feasible instead. This harness's
`find_feasible_point` helper tries calibration, then natural-theta, then up to 6 bounded
random perturbations (mirroring `audit_jach_d6d8.jl::try_scaled_feasible`'s own bounded-
attempts discipline) — generalized to be D/W-agnostic so it could serve any future grid
point. Natural-theta succeeded on the first try at every D/W tested (D=6, 8, 10, and the
D=4/W=80000 point).

### 5.3 Single-process D-scaling (infra note, §0 above)
Worth restating here as a methodology point: including `context_scaled.jl` once (instead
of `context.jl` directly) as the harness's sole root include let D=4/W=8000, D=4/W=80000,
and D=6/8/10/W=8000 all run in one warmed process — avoiding both the double-include
type-identity hazard `audit_jach_d6d8.jl` documents AND the cold-per-config-subprocess
JIT pollution the standing brief explicitly asked this document to avoid. Future D-scaling
scripts in this directory can reuse this pattern directly.

### 5.4 Certificate-sequence outlier recheck
`c8_perfprofile_certseq_recheck.jl` — a small, targeted, additive follow-up (not part of
the main harness run) that reruns Part C3(b)'s embedded K=10 certificate sequence three
times (with and without a forced `GC.gc(true)` before each pass) specifically to check
whether the harness's original 442ms first-point outlier was certificate-specific or
generic solve noise. Confirms the latter (§3, C3b) — included here for the "verify before
causal claims" discipline this investigation has flagged as a standing requirement.

## 6. Consolidated headline findings

1. **Dense vs compressed value callback, D=4**: `inner_moment_build` (the one-time build)
   is robustly 2.0–2.4x faster compressed; `inner_dual_fg_callback` (called every KNITRO
   iteration) is robustly ~0.6–0.84x — SLOWER compressed — at BOTH W=8000 and W=80000.
   The FG-callback verdict does **not** flip with a 10x increase in draws at fixed D=4.
   End-to-end `TOTAL` still favors compressed (1.02–1.34x) because the one-time build
   saving dominates for warm/low-call-count solves.
2. **D-scaling**: `inner_moment_build`'s compressed advantage grows monotonically and
   substantially with D (2.35x → 3.64x, D=4→10); the FG-callback verdict is noisy across
   D=6/8/10 (flips both directions) and should not be over-interpreted without more reps.
3. **Top-3 winner update**: noise-level at the full-15-coordinate-gradient level at D=4
   (reproduces Wave 1's own finding exactly), but shows a real, consistent 7–13% per-
   coordinate advantage isolated at the 3 actual fallback coordinates (14/15/16) — a small,
   genuine, correctly-scoped win that is simply diluted by 12 unaffected coordinates at
   this D.
4. **Winner-margin certificate, in composite_gradient_at_fast's context**: ~neutral
   overhead on a single cold call (+5.4%); a wash (not a win) embedded in a sequence of
   full gradient calls, because nothing in the current wiring lets the certificate replace
   the dominant per-point base-solve/cache-build/FD-loop costs — but the certificate's OWN
   isolated call cost shows a clean, real ~43x cold-vs-warm signal, confirming the
   mechanism itself works exactly as designed; the gap is a wiring-scope gap (documented
   already in `docs/winner_accelerator_live_wiring.md`'s own "what's not done" section:
   `lfix_value_certified` is the intended cheap-VALUE-evaluator consumer, not
   `composite_gradient_at_fast`), not a defect found this session.
5. **Threading dominates the top-3 fix at D=4** for full-gradient wall time (~2.4–3.4x from
   threading alone vs noise-level from top3).

## 7. Files

New (this workstream, prefix `c8_perfprofile_`, all under `full_aod_diag/d4_exact/`):
`c8_perfprofile_harness.jl` (main grid, Parts A–D), `c8_perfprofile_certseq_recheck.jl`
(targeted GC/noise recheck, §5.4). Raw log + CSVs:
`results/fullA_d4/d6e3b05/c8_perfprofile/` (`harness_log.txt`, `paired_*.csv`,
`c1_full_grad_top3_vs_generic.csv`, `c2_component_decomposition.csv`,
`c2b_percoord_top3_vs_generic.csv`, `c3_certificate_sequence.csv`, `d_scaling_grid.csv`,
`d4w8000_warm_{dense,compressed}_full.csv`). No existing `.jl` file in this directory was
modified.
