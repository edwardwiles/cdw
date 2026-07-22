# Handoff: independent-audit correctness + allocation/cache-cleanup productionization

Written 2026-07-21 at the end of a long session, for the next Claude session to pick up cold.
Read this first, then the two deliverable docs it points to.

## TL;DR

- **Branch**: `audit/fullA-postmerge-correctness`, worktree
  `/bbkinghome/edav/gravity_robustness/gravity-fullA-postmerge-correctness`. Forked from
  `integration/fullA-final-production-merge` @ `2620097`.
- **22 commits** on top of that base (list at the bottom of this doc). Working tree clean.
- **This branch is NOT merged into `integration/fullA-final-production-merge` or anywhere else.**
  Don't assume these fixes are live on any other branch/worktree.
- Prompt 2 (independent-audit correctness remediation): **done**. All 14 findings resolved,
  tested, documented in `docs/fullA_independent_audit_remediation.md`.
- Prompt 1 (allocation/cache-cleanup productionization): **substantial progress, not complete**.
  Full state in `docs/fullA_postmerge_allocation_productionization.md` — read its §2-5 before
  doing anything else; it has the accurate feature-by-feature status and open-items list.

## Read these two docs before touching code

1. `docs/fullA_independent_audit_remediation.md` — AUD-01..14 disposition table, what was fixed,
   what's still a known gap (e.g. QMC RNG threading is a save/restore wrapper, not a full
   refactor; a handful of diagnostic scripts still bypass the AUD-03 cache-state fix).
2. `docs/fullA_postmerge_allocation_productionization.md` — the real state of the 3 target
   features (`GradWorkspacePool`, `CrossDeltaExactCache`, `run_cm_upper_checkpointed`'s gate),
   5 bugs found+fixed, and an explicit "what remains" list. This is the more important one for
   continuing Prompt 1.

Also useful: `/bbkinghome/edav/.claude/projects/-bbkinghome-edav-gravity-robustness/memory/` has
two memory files from this session —
`fullA-postmerge-correctness-and-productionization-2026-07-21.md` (project summary) and
`gravity-robustness-knitro-hang-past-timeout.md` (an operational gotcha, see below).

## What's actually wired vs just present (as of this commit)

| Feature | Wired? | Default behavior | What's unconfirmed |
|---|---|---|---|
| `GradWorkspacePool` / `composite_gradient_at_fast_pooled` | Yes, via `use_pooled_gradient::Bool=false` on `run_profile_checkpointed`/`run_polish_checkpointed` | OFF — old buffered path used | Real end-to-end KNITRO run with the flag on has never been observed to finish (see "KNITRO hang" below). Function-level correctness IS proven (`test_gradient_workspace.jl`, 9/9, bit-identical, 4.88x allocation win). |
| `CrossDeltaExactCache` | Yes, via `exact_cache_override=` (driver) / `cross_delta::Bool=false` (`run_staged_delta5_continuation`, requires `reuse_context=true`) | OFF | No real staged δ=2→3→4→5 continuation has been run with it. Function-level correctness proven (`test_cross_delta_cache.jl`, 36/36). No hit-rate/wall-savings numbers exist yet. |
| `run_cm_upper_checkpointed`'s AUD-04 gate | Yes, unconditionally inside that function (no flag — it's a correctness fix, not a performance toggle) | N/A, always on within that function | Gate logic itself confirmed live (`test_cm_verified_success.jl` steps 1-2: `classify_inner_result => VerifiedSolved` on a real converged point). Step 3 (full outer-solve smoke run) never finished. `run_cm_upper_checkpointed` is still not called from anywhere in production — no caller wires it in yet. |

**Do not flip any of these defaults to `true`/on without first getting a clean, completed
end-to-end run** (see "Immediate next step" below). The underlying functions are correct; what's
unconfirmed is the driver-integrated behavior under real, sustained KNITRO callback pressure.

## Immediate next step (highest priority)

Get one clean, completed real-D=20/W=80,000 end-to-end run of each of the two driver wirings.
Both attempts this session hit the same failure mode:

```
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-postmerge-correctness
source .knitro_env.sh
julia --project=. full_aod_diag/d4_exact/test_driver_pooled_gradient_wiring.jl
julia --project=. full_aod_diag/d4_exact/test_cm_verified_success.jl
```

Both got through context construction (~65-85s each, and each test does this 2-3 times) and into
the real KNITRO solve, then **hung** — process kept running well past even a 1500s/25-minute
`timeout` wrapper, with low CPU% (~116%, vs 300-500%+ during genuine active computation). This
matches a pattern already documented in this repo (`test_safe_exact_cache.jl`'s own section-4
comment) where real concurrent-ish KNITRO + Julia threading interactions can silently hang. See
memory `gravity-robustness-knitro-hang-past-timeout.md`.

**How to actually get a result**: don't trust `timeout` alone. Launch in the background, then
separately poll `ps -o pid,etime,pcpu -p <pid>` yourself; if `%CPU` stays low for a long stretch
(not actively computing) well past a reasonable budget, `kill -9` it and try again — possibly on
a quieter moment for this shared host (check `ps aux` for other users' concurrent Julia/KNITRO
jobs first; this server had at least 3-4 other heavy jobs running throughout this session, which
may be a contributing factor, not yet confirmed as the actual cause). If you get a clean pass,
flip the relevant default and re-run the full existing test suite to confirm nothing broke.

## Then, in roughly this priority order

1. **Persistent `LFixBaseWorkspace`** — not started. `build_lfix_base_cache` (`lfix_incremental.jl`)
   allocates ~590-650 MB/gradient in `price0`/`pTσ0` (W×D×D tensors) fresh every call. The design
   sketch from the original brief: caller-owned workspace, `build_lfix_base_cache!(workspace, ...)`
   in-place builder, refill-on-base-point-change, freeze-during-coordinate-sweep. This is real new
   engineering, not a wire-up — budget real time for it.
2. **`CrossDeltaExactCache` real benchmark** — run an actual staged δ=2→3→4→5 continuation with
   `cross_delta=true` and record hit rate / wall savings. The wiring already logs cache size
   before/after each stage (see `staged_delta5.jl`'s `lp("  stage ", i, " cross-delta cache: ...")`)
   as a proxy; a real hit/miss counter would be a nice-to-have addition to `CrossDeltaExactCache`
   itself if you want more precision.
3. **CM interrupted/resume production campaign** — the brief specifies D=20, W=80,000, L=50, δ=1.
   Not run. Do this only after the AUD-04 gate's tolerances (`VerifiedSuccessTolerances` in
   `oracle.jl` — explicitly marked provisional/uncalibrated in its own docstring) have been sanity
   checked against real output from this campaign; don't assume they're right.
4. **Fresh `Profile.Allocs` audit, two-tensor representation experiment (A-D), dense
   post-solve materialization audit, final matched benchmarks** — none started, all from the
   original brief's later sections.

## Lessons learned this session (read before repeating the mistakes)

- **Docstring-stacking silently breaks `@doc` macro expansion, and `Meta.parseall` does NOT catch
  it.** If you insert a new `"""docstring""" \n function foo() ... end` block directly before an
  EXISTING `"""docstring""" \n function bar() ... end`, and your edit doesn't include the existing
  docstring in its span, you can end up with two adjacent triple-quoted strings — Julia errors
  with "cannot document the following expression" at *load* time, not parse time. Always verify
  with a real `include()`/test run, not just a syntax check. Bit me on `composite_gradient.jl`,
  `lfix_buffer_reuse.jl`, and `draw_design.jl` this session.
- **A new file's `sha256_of_matrix`-style helper needs to live in the file its consumer actually,
  unconditionally depends on** — not wherever felt topically nearest. `context_fingerprint`
  (oracle.jl, AUD-08) unconditionally calls `sha256_of_matrix`; it was first placed in
  `draw_design.jl` (topically close to the AUD-11 fix that introduced it) and broke any test that
  includes `oracle.jl` without `draw_design.jl`. Moved to `oracle.jl` (the more universally-
  included file) with a defensive `isdefined`-guarded include left in `draw_design.jl`.
- **`zfree=0` (via `pivot_expand`) is an arbitrary reparameterization gauge reference (Aod≈1), NOT
  a feasible or calibrated starting point.** Already flagged in a prior-session memory
  (`feedback-gravity-elimination-zero-is-not-calibration.md`) and I still fell into it writing a
  new test this session. The real calibrated start is `ctx.θ0_up`'s own Aod block:
  `reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)`, pivot-reduced.
- **Check test files' own `include()`/hardcoded-path assumptions before trusting a "PASS".** A
  subagent found that `test_cm_checkpoint_original.jl`/`test_cm_checkpoint_resume.jl` hardcoded an
  absolute path to a completely different worktree (`gravity-fullA-alloc-cache-cleanup`) instead
  of `@__DIR__` — meaning this session's own earlier "PASS" runs of those two tests were silently
  validating a different checkout's code. Fixed, but worth a quick grep (`grep -rn
  "/bbkinghome/edav/gravity_robustness/gravity-fullA-" full_aod_diag/d4_exact/test_*.jl`) for any
  other stray absolute-path references before trusting other pre-existing tests' results.
- **`ScheduleWakeup`'s requested delay did not reliably match real elapsed wall-clock time in this
  environment** (observed firing back-to-back with far less real delay than requested, confirmed
  via `date`). For "wait until a real background compute job finishes," prefer a Bash background
  poll-loop (`until ! ps -p <pid>; do sleep 15; done`) launched with `run_in_background: true` —
  that worked reliably all session.
- **Real D=20/W=80,000 context construction costs ~65-85s each time**, and several driver
  functions rebuild it internally even if you already built an equivalent one in your own test
  script — a 3-context-build test can cost 250s+ before any actual solving starts. Budget
  generously (20-30+ minutes) for any real end-to-end test at this scale, and don't be alarmed by
  a slow start.
- **Subagents are worth it for well-scoped, isolated work** — the CM verified-success gate
  subagent (worktree-isolated) produced clean, well-reasoned code that matched this session's own
  patterns closely, AND independently caught the stale-worktree-path bug above. Review its diff
  before merging (it will branch from whatever commit existed when launched, which may be stale by
  the time it finishes — check `git diff --stat <your-current-branch>` vs `git status --short` in
  its worktree to distinguish "files it actually touched" from "files that diverged because your
  own branch moved on").

## Environment cheatsheet

```bash
export PATH="$HOME/.juliaup/bin:$PATH"   # NOT /opt/shared_sw -- that binary is broken
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-postmerge-correctness
source .knitro_env.sh                     # pins KNITRO 13.0.1, only licenses on demand.mit.edu
julia --project=. <script.jl>
```

This is a shared server with routinely 15-25+ parallel git worktrees and other users' concurrent
Julia/KNITRO jobs. Always confirm `pwd` before editing (a same-named file exists in nearly every
worktree with no error to catch a misdirected edit), and check `ps aux` before assuming a hang is
this session's own fault rather than resource contention.

## Full commit list (this session, base `2620097` → `HEAD`)

```
e9cdac6 Finalize productionization doc: all 3 wirings done, 5 bugs found+fixed, honest gap list
016abf8 AUD-04/AUD-10: CM checkpoint path verified-success gate (via subagent)
8f55315 Update productionization doc: pooled-gradient + cross-delta-cache wiring
1136af9 Wire CrossDeltaExactCache into staged delta continuation (opt-in)
8727d24 Wire GradWorkspacePool into the production gradient driver (opt-in)
aff8c13 Fix 3 real integration bugs surfaced by the merged branch's own tests
e23f71f AUD-08 follow-through: fix CrossDeltaExactCache's own key + start Prompt-1 doc
948a622 Merge branch 'perf/fullA-allocation-cache-cleanup' into audit/fullA-postmerge-correctness
9615895 Add independent-audit remediation doc: AUD-01..14 disposition table
b8a12c3 AUD-03/04/09/10: production driver cache-state, incumbent, and resume gates
8dce3c6 AUD-04/AUD-08: typed verified-success gate + context fingerprint (core)
64647b1 AUD-11: draw/checkpoint checksums use SHA-256, not Julia's hash()
a7f57d2 AUD-06: fused winning-range screen is now tie-safe
80fdff9 AUD-14: QMC draw generators no longer leak global RNG state to the caller
e14f8a3 AUD-12: never let two nonfinite L_fix probes silently become a zero gradient
5dfd80c AUD-13: rename moment_resid to benchmark_unweighted_moment_mean (mechanical)
62dde8f AUD-02: disable concurrent KNITRO evals; add cross-thread callback guard
```
(followed by the 5 commits from the `perf/fullA-allocation-cache-cleanup` merge itself, and the
original production-merge history below `2620097`.)
