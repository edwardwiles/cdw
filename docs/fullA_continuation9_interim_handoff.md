# Continuation 9 interim handoff — Phases 1-5 complete

Written mid-session, before Phases 6-9. Branch `diag/fullA-d4-exact`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, machine `demand.mit.edu`.
This document exists because the session is long and the user asked for a
checkpoint write-up before continuing into the remaining (more expensive)
phases — it will be superseded by the final `fullA_continuation9_handoff.md`
once Phases 6-9 land.

## Starting premise vs. what was actually found

The standing continuation-9 brief assumed a real-data D=20 driver for the
full-A_od method already existed and needed benchmarking/optimizing. **Phase 1's
audit found this was false**: no genuine full-A_od real-data D=20 run had ever
been launched anywhere in this repo family. Two pieces existed separately —
real D=20 data + a D-generic driver on a sibling worktree (for the *sequential*
method only), and the modern full-A machinery (compressed moments,
`lfix_composite`, winner certificates) on this worktree with zero real-data
loading code. This session built the missing bridge rather than proceeding on
a false premise. Full detail: `docs/fullA_D20_production_path_audit.md`.

## Phase 1 — real-data D=20 port (commit `5c1baca`)

Ported `setup/importData.jl`'s `fakeData==3` branch and `setup/defineCounter.jl`'s
autarky `τ^Inf` bugfix from `trade_robustness_modular_perf` @ `778c362`, plus
the `real_data/noah_D20/*.csv` files (md5-verified). New
`full_aod_diag/d4_exact/context_real_d20.jl::d20_real_setup(; W)` builds a real
D=20 context (France focal, `baseIndex=2`, σ=2.5, μ estimated via gravity) with
the same field layout as `context_scaled.jl::d_exact_setup_scaled`, so every
existing D4-exact diagnostic function works unchanged on it.

**Validated**: κ_ACR = 0.0203135 matches the known real-data point estimate
(~0.020314) to 5 significant figures. D=20, n_free=401, France confirmed.
**W=8,000 is uniformly infeasible/unbounded at D=20 real data** (tested to
δ=1e6 — a draw-count problem, not a δ-scale problem). **W=80,000 solves
cleanly** at natural theta (`inner_status=0`, Δ_dual=0.00259, gravity at
machine-zero).

## A live incident, caught and fixed (commit `bd313a2`)

The very first W=800,000 probe (and, separately, the first W=80,000
microbenchmark run) triggered a **server-wide memory alert** on this shared
3TB machine — caught by the user, not by any internal safeguard. Root cause:
`d20_real_setup` inherited `context_scaled.jl`'s diagnostic default
`needs_outer_moment_jacobian=true`, which allocates a dense
`W × (nTotalMoments+2) × l_full` tensor (`jac_h`) — harmless at D=4-10
(`context_scaled.jl`'s intended scale) but **≈109GB at D=20/W=80,000** and
would have exceeded **1TB at W=800,000**. Both processes were killed; the
default was fixed to `false` (matching what the real production drivers
already do, since their gradient path never uses an analytic outer moment
Jacobian). Post-fix: W=80,000 peak 4.47GB (was killed pre-fix), W=800,000 peak
16.55GB (was climbing past 780GB pre-fix, killed). **Lesson for anyone
extending this work further: D=20's ~400 free parameters make dense structures
that were negligible at D≤10 potentially catastrophic — always sanity-check
memory on a small probe before scaling up new code paths.**

## Phase 2 — production microbenchmarks

- **W=80,000** (`docs/fullA_D20_W80k_microbenchmark.md`, full 4-outer-point
  breakdown + thread sweep): at the real calibration point, warm value callback
  is 1.86s, **65% in `inner_moment_build`** (dense mode) — the clear
  optimization target picked up by Phase 3. `evaluate_fullA` (plain oracle)
  costs ~2x `evaluate_fullA_fast` because it rebuilds the moment matrix twice —
  flagged as "don't use in performance-sensitive D=20 code." Full 400-coordinate
  `L_fix` gradient: 6.44s threaded / 35.3s serial (5.5x threading speedup, larger
  than D=4's 3.4x). All 4 outer points (calibration, gravity-tangent, upper-branch,
  lower-branch) feasible on the first try. γ'_focal's real-data bound is tight
  and asymmetric (`[0.9307, 1.0]`) — Δ_dual jumps ~90x for a 1% γ' move above
  calibration, a genuine feature of this calibration, not an artifact.
- **W=800,000** (`docs/fullA_D20_W800k_microbenchmark.md`, **partial** — memory
  safety only): confirmed safe post-fix (16.55GB peak, clean convergence,
  Δ_dual=0.00026, consistent with W=80k's 0.00259 shrinking as draws grow). The
  full 4-point/thread-sweep breakdown at W=800,000 was **deliberately deferred**
  to after Phase 3-5 landed (no point profiling a path about to change) — this
  is queued as part of the next phase of work.

## Phases 3, 4, 5 — production speedups (all landed and merged)

Dispatched as three coordinated agent workstreams (Phase 3 and Phase 4
sequentially in the main worktree since 4 builds on 3; Phase 5 concurrently in
an isolated worktree `gravity-fullA-d4-c9-bandwidth`, merged back via
`git merge --no-ff`). All at real D=20/W=80,000 — the first time any of this
machinery has been tested against real data or at this scale.

**Phase 3** (`docs/fullA_fully_compressed_inner_report.md`, commits `2722616`,
`35117ff`, `2573f3e`, `f23706e`, `39f9d30`):
- 3.1 `:compressed` moment mode ported to real D=20 with **zero code changes**
  needed — 1.73x warm value-callback speedup, 5.74x `inner_moment_build`
  speedup (continuing the D-scaling trend 2.35x→3.64x→5.74x, D=4→10→20).
  Honest negative: cold is a slight net loss (0.92x).
- 3.2 `build_lfix_base_cache`'s dense self-validation rebuild made opt-in
  (`validate_dense::Bool=false`, default off) — bit-identical gradients
  confirmed at D=4/8/20 — ~1.3x full-gradient speedup (7.0-7.2s → 5.4s).
- 3C (elevated from optional to required mid-task, per user request): three
  dense-Hessian-free inner CC dual solvers built on Continuation 8's
  previously-unused `compressed_cc_hvp` primitive — all converge to the exact
  same optimum as the dense baseline, but cost **2.7x-4.5x more wall-clock**
  (many small compressed calls losing to one BLAS `gemm!`) — a genuine,
  reported-not-forced negative result. **The existing dense-Hessian compressed
  baseline remains the right default.**

**Phase 4** (`docs/fullA_D20_winner_kernel_optimization_report.md`, commits
`97b410c`, `efcc7d3`, `b099424`, `e9cedcd`, `f2dad5d`):
- Winner-margin certificate: **20.9-21.2x** on a line-search sweep at D=20
  (vs D=4's 6.16-6.49x — the win scales with the rebuild cost it's avoiding,
  which grows with D). Order-independence explicitly verified (bit-identical
  under random permutation).
- Top-3 coordinate-specialized update: **1.16-1.38x** end-to-end at D=20 (was
  noise-level, 0.93-1.17x, at D=4) — confirms Continuation 7/8's "grows with D"
  prediction directly, the first D where it's a real win at the full-gradient
  level.
- Destination-batch loop reordering: honest null (0.99-1.01x) — the
  pivot-reduced coordinate order is already destination-major by construction.
- New destination-major compressed kernels (v2): mixed — cold is a wash
  (0.988x), warm is a modest real win (1.097x) — **not wired into production**
  given the mixed result.

**Phase 5** (`docs/fullA_D20_bandwidth_optimization_report.md`, commits
`cea3948`, `f5dd8ea`, `8db1cff`, `e09ccb5`, merged via `15ec0cb`):
- Closed-form quantile bandwidth selector: only 1.08x — bisection already
  converges fast in practice, so eliminating iterations saves little.
- **Staleness-aware bandwidth caching (recommended production lever)**: 1.86x
  fully warm, ~1.68x averaged over a 10-point simulated outer-loop trajectory
  with one deliberate large jump correctly triggering invalidation.
- Found (documented, not fixed by that agent) a pre-existing thread-safety bug
  in the cache: a plain `Dict` written concurrently from `Threads.@threads`,
  corrupting ~1.5% of trials. **Fixed directly by the coordinating session**
  (commit `a6ed25e`): `ReentrantLock` around just the dict read/write, expensive
  work stays outside the lock. Verified: 0/200 corrupted (was 3/200), gradient
  bit-identical to the reference (max diff 0.0).

## Compounding effect (not yet independently re-measured end-to-end)

Individually: 3.1 (1.73x value / 5.74x moment-build) × 3.2 (1.3x gradient) ×
4's top-3 (1.16-1.38x) × 4's certificate (up to ~21x on repeated nearby calls,
context-dependent) × 5's caching (1.68-1.86x). These multiply in different
parts of an outer-loop run (value calls vs gradient calls vs repeated-nearby-point
sequences), so a naive product is not a valid combined estimate — **the next
phase of work (D=10 gate rerun, Phase 7) is where a genuine end-to-end combined
number should come from**, not this document.

## Server safety note

This session triggered one real server-wide memory incident (see above) —
caught by the user, resolved within the session, and the root cause is now
fixed at the source (not just avoided). No further incidents. Every subagent
dispatched since the fix was instructed to sanity-check memory on a small probe
before scaling any new code path, per the lesson learned.

## What's left (Phases 6-9, in progress as of this writing)

1. **Phase 2 (deferred)**: full W=800,000 4-point/thread-sweep benchmark, now
   against the Phase 3-5 architecture (not the slow pre-optimization path).
2. **Phase 6**: production-scale A-block `L_fix` derivative validation at
   D=20/W=80,000 (≥20 directions × 3 points) — a scientific gate, especially
   important given Continuation 8's own unresolved finding of real A-block
   sub-gradient sign disagreement (73-93% coordinate agreement) between the
   fast and slow gradient methods at D=4/W=80,000.
3. **Phase 7**: rerun the D=10 upper gate with the full production architecture
   (compressed inner path + opt-in-only dense cache + selected bandwidth policy),
   longer budget than the prior 120s stall.
4. **Phase 8**: short supervised D=20/W=80,000 pilot (profile + constrained
   polish), gated on 6 and 7 passing.
5. **Phase 9**: W=800,000 fallback readiness — cost projection for 10 frontier
   points, checkpointing/recovery demonstration.

## Repo state

- Branch `diag/fullA-d4-exact`, all work merged, no open PRs.
- Worktree `gravity-fullA-d4-c9-bandwidth` (branch `c9-bandwidth`) is now fully
  merged and can be removed (`git worktree remove`) once nothing else needs it.
- Untracked leftover result directories from Continuation 8
  (`results/fullA_d4/{c4243c1,d547142}/d{8,10}_pilot_*`) remain from before this
  session started — not touched, flagged for cleanup by whoever picks this up
  next (per this investigation's "defer cleanup to the end" convention).
- KNITRO 14.2.0, Julia 1.12.6, `JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1` throughout.
