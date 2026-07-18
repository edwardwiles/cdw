# Continuation 7 handoff — read this FIRST if picking up this investigation

Written at end of session. Branch `diag/fullA-d4-exact`, worktree
`/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, **HEAD `65c0a19`** (clean tree). This session
picked up Continuation 6's handoff and ran 6 workstreams — mostly in parallel via isolated git
worktrees — covering everything Continuation 6 flagged as open plus two new user requests. All 6 are
merged into `diag/fullA-d4-exact` and independently re-verified post-merge (each workstream's own
equivalence-test file was re-run against the final merged tree, not just in its source worktree).

## Canonical headline (unchanged from Continuation 5/6)

- Best upper incumbent: `upper_lfixcomposite_sr1_60s`, κ=0.17245688540655113, γ'=0.8926359585,
  Δ=0.9999924058.
- Best lower incumbent: `lower_lfixcomposite_fast_sr1_300s`, κ=0.005428799948779983, Δ-δ=+8.5e-7.
- Nothing this session changed these numbers — all work was infrastructure/diagnostic (perf, exactness,
  and one genuine new scientific finding on the gamma profile's shape, which does NOT move κ).

## Workstream 1 — gamma-profile non-monotonicity: RESOLVED, real not artifact

Continuation 6's #1 priority. **Verdict: genuine, not a warm-start artifact.**
`profile_Delta(g) = min_A Delta(g,A)` is a smooth U-shaped curve with a single interior minimum at
**g*≈0.96 (Δ≈9.4e-5)**, rising monotonically on both sides. Verified with 9 independent starts per grid
point (g∈[0.94,0.99] every 0.005) — continuation plus path-free cold starts plus perturbations — all
converging to the same Δ at every g. The low-g `Δ=δ` crossing that pins the current best upper incumbent
(g≈0.8926) is **unique and unaffected** — the feasible set is a two-sided interval, not a half-line.
New substantive note: g*≈0.96 is a distinguished best-fit γ' (near-exact reproduction of the factual
estimand), distinct from and not competing with the robustness-boundary γ'≈0.8926.

Report: `results/fullA_d4/bb74649/gamma_profile_nonmonotonicity_report.md`. Code:
`full_aod_diag/d4_exact/gamma_profile_multistart.jl`, `analyze_multistart.jl`.

## Workstream 2 — compressed winner-form factual moments: BUILT + VALIDATED, real speedup growing with D

User's new ask (Continuation 6 addendum). **Verdict: compression works, equivalence-tested to machine
precision.** The factual bilateral moment matrix has exactly one nonzero winner entry per
(draw,destination) plus a draw-independent centering vector, so the entire fixed-dual pipeline evaluates
in **O(W·D)** instead of dense **O(W·D²)**, never materializing the W×D² matrix. Built: representation +
fixed-dual contraction (`compressed_moments.jl`) and a full compressed CC-inner dual bundle — objective,
gradient, **exact HVP** — from compressed moments only (`compressed_cc_inner.jl`).

**Corrected the task's own schematic**: the centering term is draw-independent, not scaled by the winner
value as the brief's `v_sd(e_w−λ̂)` suggested — exact contraction is
`Σ_d κ_{win,d}·v_{s,d} + Σ_d C_d`.

Equivalence (all PASS): dense-materialize ~1e-15, dual contraction ~1e-13, CC objective/gradient ~1e-15,
HVP-vs-FD ~1e-10 matched against production `PsiObjectiveBundleImplicit`.

Speedup grows with D exactly as O(W·D)/O(W·D²) predicts: **2.2× end-to-end at D=4, 5.2× at D=10** (up to
16× on the isolated contraction at D=10). Reused the tie-handling prior art
(`TiedWinnerError`/`detect_price_ties`) from the start rather than rediscovering it.

**Not yet done**: wiring `compressed_cc_inner` into the actual `L_fix`/inner-dual production evaluation
path (currently a validated standalone bundle) — flagged by the workstream's own agent as the natural
next step, behind a mode flag with dense-default + tie-fallback.

Report: `docs/compressed_winner_moments_report.md`. Code: `full_aod_diag/d4_exact/compressed_moments.jl`,
`compressed_cc_inner.jl`, `test_compressed_moments.jl`, `test_compressed_cc_inner.jl`,
`benchmark_compressed.jl`.

## Workstream 3 — MuSigmaPowCache wiring: DONE, corrected the prior audit's estimate

Continuation 6 priority item 2. Wired via opt-in `enable_pow_cache!(ctx)` (field-mutation on
`ctx.obj.moments!`, following the low-risk pattern Continuation 6 had sketched but not implemented).
Verified bit-identical against the full equivalence suite with and without the cache.

**Realized gain**: 1.13× end-to-end / 1.16× moment-build at 20 threads — driven by eliminating
~522 KB/call of allocation (two W×D Float64 matrices) and the resulting GC time, **not compute**. This
is a documented **correction** to `docs/fullA_moment_construction_audit.md`'s 1.1–1.4× estimate, which
was measured single-threaded; at 20 threads the raw broadcast is nearly free so isolated moment-build
wall time is ~1.00× and the win is purely allocation/GC.

Report: `docs/fullA_pow_cache_wiring.md`. Code change: `full_aod_diag/d4_exact/moments_fast.jl` (+37
lines, additive only).

## Workstream 4 — incremental hard-winner caching: BUILT + ADOPTED (2 of 5 sub-approaches)

New user ask, built on top of Workstreams 2/3's prior art. **Ranked verdict** (winner-computation
component, D=4/W=8000):

1. **Coordinate-specialized top-3 cache — ADOPT.** 13.6–200× vs full scan, exact. Closes the one
   remaining generic O(D) fallback in production (`composite_gradient.jl:97`, the same-destination
   two-changed-origins case) with an exact O(1) top-3 cache instead.
2. **Winner-margin certificate — ADOPT.** 8–18× vs full scan, exact. 97–100% of draws certified
   unchanged at accepted/line-search step sizes, still 66–95% at continuation/large steps; auto-falls-back
   to full scan beyond a `tol_far` guard. Certified cells are provably the unique strict argmin.
3. Trusted full scan — kept as reference/fallback (1.785 ms baseline).
4. Pairwise-breakpoint preprocessing — **REJECTED**, measured: 2.07ms warmed preprocessing + 1.02MB
   persistent state vs. the certificate's ~0.10ms/step and no memory, while the certificate already
   certifies 97–100%.
5. Draw-dominance partial order — **REJECTED**, measured: comparable-pair fraction 0.50→0.20 across
   D=4→10, longest chain 1–2 (no usable chains).
6. Origin pruning — **REJECTED**, measured: 0% prunable at every tested D (4/6/8/10), at-point and over
   a trust region.

Threading: outer-threaded + inner-serial confirmed **7.33× faster** than nested inner-threading (the
established discipline was correct, now with a number behind it).

Report: `docs/winner_certificate_report.md`. Code: `full_aod_diag/d4_exact/winner_certificate.jl` (+
test/bench/explore files).

## Workstreams 5+6 — focal autarky counterfactual moment: AUDITED, OPTIMIZED, further specialized

Two rounds, both new user asks.

**v1 audit finding**: the current autarky CF construction was **already** a direct O(W) domestic-only
broadcast (`hFunction.jl:201`) — no supplier search, no winner branching, no counterfactual price matrix.
The real (only) waste was in the `hFunctionCounter!` *wrapper*: a dead per-thread-chunk D×D `constCons`
rebuild plus a non-`@view` W-length column copy (+64.8KB/call). Fixed via opt-in `enable_autarky_cf!`.
Also verified: the CF numerator is bit-identical to the factual domestic numerator under
`wPrime=wHat=1, τPrime[bi,bi]=τ[bi,bi]=1, LPrime=L`; the only operative CF-vs-factual difference is the
additive constant `γ'[bi]^σ·LPrime[bi]`.

Result: **3.55× on the isolated CF component**, bit-identical (max|Δ|=0.0) through the full live oracle
(duals, divergence, weights, gradient). **Flat effect on full-build/value-call wall time at D=4** — CF is
only ~0.6% of the full moment build, so this is a D-scaling/allocation-hygiene win, not a current
wall-clock one.

**v2 (further specialization, user follow-up)**: precompute the entire draw-level domestic vector once,
excluding the `A_dd` factor; multiply by a single cached-reciprocal scalar per outer point.
**Corrected the task's own hint**: the exponent is **A_dd^(μ(σ−1))**, not the hinted bare `(σ−1)` — the
μ factor is real (0.25 at calibration μ=1/6, σ=5/2), verified from `hFunction.jl:201` +
`moments_gammanorm.jl:246-251`. Confirmed in code: μ, σ are pinned equal at `θ_lo==θ_hi` for the whole
run (never free parameters), so the cached base vector never needs invalidation. Confirmed independence
from all 15 other A_od entries: exactly 0.0 perturbation.

Equivalence: tight (not bit-identical by design — reciprocal-multiply vs divide), ~3.55e-15 absolute
against both the generic routine and v1; **downstream `Delta_dual` through the live inner KNITRO solve is
bit-identical**. Speedup: 35.7× on CF-only-from-raw-Uσ construction, but flat on the full build (same
0.6%-of-build reason as v1) — **recommended as an opt-in tool for CF-only sweeps
(`gamma_profile`/`delta_star_schedule`, which re-evaluate CF repeatedly at fixed draws while only A_dd/γ'
move), not as the production oracle default** (v1 stays default there — bit-identical, and v2 gives no
full-build gain).

Reports: `docs/autarky_cf_moment_audit.md`, `docs/autarky_cf_v2_cached_base.md`. Code:
`full_aod_diag/d4_exact/autarky_cf.jl`, `autarky_cf_v2.jl` (+ test/benchmark files for both).

## What's left — in priority order

1. **Wire `compressed_cc_inner` into the live `L_fix`/inner-dual production path** (Workstream 2's own
   flagged next step) — currently validated standalone, would give the O(W·D) inner-solve speedup live,
   growing with D.
2. **Wire `enable_autarky_cf_v2!` into `gamma_profile.jl`/`delta_star_schedule.jl` specifically** — that's
   where its 35.7× actually applies; it does nothing for the general oracle.
3. **D-scaling redo with everything now wired** (`enable_pow_cache!` + compressed moments/CC-inner once
   wired) — the D=6 pilot from Continuation 5 predates ALL of this session's work.
4. Everything else still open from Continuation 6's list that this session didn't touch: live bandwidth
   caching (`h_mode=:cached`) with a real revalidation policy in KNITRO; nested-W (W=8000/20000/80000 as
   prefixes of one draw pool); full gamma-profile deliverables (machine-readable table, lower-direction
   profile, deeper stationarity reassessment); fair hard-vs-smoothed comparison.

## Session mechanics (useful if this pattern recurs)

- 6 agents were run this session, 4 via the Agent tool's `isolation: "worktree"` and 2 via a **manual
  worktree workaround** (`git worktree add -b <branch> <path> <head>`, then instruct the agent to `cd`
  there explicitly) after `isolation: "worktree"` started failing mid-session with `Cannot create agent
  worktree: not in a git repository` — root cause not diagnosed, may be transient/environment-specific;
  worth checking if it recurs before assuming the manual path is required going forward.
- All 6 branches merged into `diag/fullA-d4-exact` cleanly (5 fast-forward/no-conflict octopus or simple
  merges) because every workstream was scoped to touch disjoint files (new files only, or one small
  additive diff to a file no other workstream touched) — this discipline is worth preserving explicitly
  in future multi-agent prompts here.
- Every merge was independently re-verified by re-running that workstream's own equivalence-test file
  against the final merged tree (not just trusting the source worktree's earlier pass) — caught nothing
  this session, but is the right discipline given how much this investigation depends on exact
  equivalence claims.
- `source .knitro_env.sh` inside a piped command (e.g. `source ... | tail`) runs in a subshell and
  silently fails to export `PATH`/KNITRO env vars into the parent shell — cost some time mid-session,
  now documented so it doesn't recur.

## Operational notes (carried forward from Continuation 6)

- Must run on `demand.mit.edu` (KNITRO license, demand-side only).
- `source .knitro_env.sh` before any Julia invocation touching KNITRO (see subshell warning above).
- Set `JULIA_NUM_THREADS` explicitly per run — 20 was used for this session's agent runs (machine has 208
  cores / ~2.2TB free RAM, no contention concerns even running several agents concurrently).
- Push new docs/handoffs to Dropbox at
  `dropbox:Gravity robustness/Analysis/Server Output/fullA_d4_continuation7_2026-07-18/` — done for this
  session's outputs (see below).
