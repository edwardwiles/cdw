# Full-A_od allocation/cache-cleanup productionization

Branch: `audit/fullA-postmerge-correctness` (correctness-hardening work, all 14 independent-audit
findings resolved — see `docs/fullA_independent_audit_remediation.md`), with
`perf/fullA-allocation-cache-cleanup` merged in on top.

**Status: substantial progress, not complete.** Activation audit, all 3 target-symbol wirings,
and the CM verified-success gate are done and committed. Full end-to-end wall-clock/allocation
benchmarking, `LFixBaseWorkspace`, and a long checkpoint/resume campaign were not reached this
session — see §4.

## 1. Corrected premise

The brief states the allocation/cache-cleanup branch "has now been merged into the main
production branch." Checked directly with git and found **false** at session start:
`integration/fullA-final-production-merge` @ `2620097` was an *ancestor* of
`perf/fullA-allocation-cache-cleanup` @ `1279ed6` (the perf branch forked forward from production
and was never merged back) — the reverse of "merged in." None of the four target symbols existed
anywhere on the production branch. User-confirmed: merged `perf/fullA-allocation-cache-cleanup`
into this branch as a prerequisite (clean merge, 9 purely additive files, zero conflicts).

## 2. Active-call-path map (final state this session)

| Feature | Source present | Called by production driver | Persistent across callbacks/stages | Tests |
|---|---:|---:|---:|---|
| `composite_gradient_at_fast_pooled` / `GradWorkspacePool` | Yes | **Wired** — `run_profile_checkpointed`/`run_polish_checkpointed` accept `use_pooled_gradient::Bool=false`; `cb_G!` dispatches to it when true, old buffered path byte-identical when omitted (default) | Yes — one `GradWorkspacePool` built per `ctx`, not per gradient call | `test_gradient_workspace.jl` 9/9 (function-level bit-identity + allocation, real D=20/W=80000). Driver-level `test_driver_pooled_gradient_wiring.jl`: parses/loads, dispatches through the intended code path (confirmed by code review), real end-to-end KNITRO run not observed to completion within a 1500s budget — see §4 |
| `CrossDeltaExactCache` | Yes | **Wired** — `run_profile_checkpointed`/`run_polish_checkpointed` accept `exact_cache_override`; `run_staged_delta5_continuation` accepts `cross_delta::Bool=false`, threads ONE cache through every stage (requires `reuse_context=true`, hard-errors otherwise) | Yes — one cache per staged continuation | `test_cross_delta_cache.jl` 36/36 (function-level). Driver-level staged-continuation hit-rate/wall-savings benchmark not run — see §4 |
| `run_cm_upper_checkpointed` | Yes | **Gated but not yet the default production path** — `cb_F!`/final checkpoint now require `is_verified_success` (AUD-04/AUD-10 parity with the unrestricted driver); still requires an explicit interrupted/resume production campaign (brief's own D=20/W=80,000/L=50/δ=1 spec) before being recommended as *the* CM path | Its own gate state (`best_feasible[]`) persists correctly across checkpoint/resume | `test_cm_verified_success.jl`: steps 1-2 (gate correctness, real D=20/W=80000) independently confirmed on this branch (`classify_inner_result => VerifiedSolved`, `archC_verified_state`/`archC_base_state` agree). Step 3 (full KNITRO outer-solve smoke run) not observed to completion — see §4. `test_cm_checkpoint_{original,resume}.jl`: a real bug was found here first — both hardcoded an absolute path to a *different* worktree (`gravity-fullA-alloc-cache-cleanup`), so this session's earlier "PASS" runs were silently exercising the wrong code; fixed to `@__DIR__` |

## 3. Bugs found (all fixed)

Beyond the correctness-hardening work in `docs/fullA_independent_audit_remediation.md`, this
productionization pass found and fixed **5 more real bugs**, all in code that either just got
merged or was written this session — found by applying the same scrutiny as the original audit,
never assumed correct because it was new or because a test file existed for it:

1. **`CrossDeltaExactCache`'s `FullAInnerKey` did not carry a context fingerprint.** Stripped
   `FullAEvalKey` down to `(x_free, find_smallest, inner_loop_opt, mode)` for its δ-independence
   design (sound on its own terms), but that also dropped the `ctx_fingerprint` field AUD-08 had
   just added — reintroducing the exact cross-context aliasing risk AUD-08 closed. Fixed:
   `FullAInnerKey` now also carries `ctx_fingerprint`.
2. **`sha256_of_matrix` (AUD-11) lived only in `draw_design.jl`**, but `oracle.jl`'s
   `context_fingerprint` (AUD-08) calls it unconditionally. `test_cross_delta_cache.jl` includes
   `oracle.jl` but not `draw_design.jl` — a real `UndefVarError` at runtime. Moved the canonical
   definition into `oracle.jl` (its real unconditional consumer, and the more universally-included
   file); `draw_design.jl` now has a defensive `isdefined`-guarded include.
3. **`test_cross_delta_cache.jl`'s own `FullAEvalKey(...)` calls and one field reference predated
   this session's AUD-08/AUD-13 fixes** (5-arg key with no `ctx_fingerprint`; `:moment_resid`
   instead of `:benchmark_unweighted_moment_mean`) — updated.
4. **My own first driver-level wiring smoke test used `zfree=0`** for the starting A-block —
   exactly the "z=0 is not calibration" trap already recorded in this session's own memory (an
   arbitrary reparameterization gauge reference, not a feasible point). Fixed to extract the real
   calibrated Aod block from `ctx.θ0_up`.
5. **`test_cm_checkpoint_original.jl`/`test_cm_checkpoint_resume.jl` hardcoded an absolute path to
   a different worktree** (`gravity-fullA-alloc-cache-cleanup`) instead of `@__DIR__` — found by
   the subagent while building the CM verified-success gate; this session's earlier "PASS" runs of
   those two tests were silently validating a different checkout's code, not this branch's. Fixed.

## 4. What remains (honest gaps, not attempted or not completed this session)

- ~~Full end-to-end KNITRO smoke runs for both new wirings STILL have not finished... Do not
  flip either default until ONE of these completes cleanly.~~ **RESOLVED, root cause found —
  this was never actually about `use_pooled_gradient`/`cross_delta` themselves.** The repeated
  hang (4 occurrences total across sessions) was a real regression: commit `62dde8f` ("AUD-02:
  disable concurrent KNITRO evals") changed `par_concurrent_evals` from `yes`→`no` on the outer
  solve's `.opt` files, which deadlocks this codebase's always-used nested-`KN_solve` pattern
  (outer solve's callback synchronously invokes the inner dual solve) via a non-reentrant KNITRO
  lock — confirmed via `SIGQUIT` stack traces and a direct A/B against a historical
  known-working commit. **Fixed** by reverting that one setting (kept AUD-02's *other*
  deliverable, the thread-aware `PsiObjectiveBundle.jl` guard, which already correctly handles
  the legitimate same-thread nesting). Full writeup: `docs/fullA_nested_knitro_solve_hang_rootcause.md`.

  **With the fix applied**, both originally-hung tests now complete:
  - `test_driver_pooled_gradient_wiring.jl`: both `use_pooled_gradient=false` and `=true` runs
    complete in ~90s total with real KNITRO iterations and gradient evaluations, and — the
    directly relevant check for this decision — **both arms reached the identical
    `kappa=0.0203135174923732`** (bit-for-bit the same outer-solve outcome regardless of which
    gradient path was used). Combined with the pre-existing function-level proof
    (`test_gradient_workspace.jl` 9/9, bit-identical, 4.88x lower allocation), this is now solid
    evidence for flipping `use_pooled_gradient`'s default to `true`.
  - `test_cm_verified_success.jl`: 11/11 passed cleanly (`checkpoint_reason=stage_complete`,
    `classify_inner_result => VerifiedSolved`) — validates the AUD-04 CM verified-success gate
    end-to-end, not `cross_delta`/`CrossDeltaExactCache` specifically.
  - **`cross_delta`/`CrossDeltaExactCache` was NOT independently re-run post-fix this session** —
    a real, disclosed gap. Its own function-level tests (`test_cross_delta_cache.jl`, 36/36) are
    unaffected by any of this (they don't hit the driver's outer `KN_solve` at all), but no
    driver-level staged δ-continuation with `cross_delta=true` has been observed to complete
    cleanly. Do not flip `cross_delta`'s default without that run.

  **Recommendation**: `use_pooled_gradient` has enough evidence now (function-level + fixed
  driver-level, matching outer-solve outcome) to default to `true`. `cross_delta` needs one more
  real run before its default should change — this is a small, bounded follow-up (one
  `run_staged_delta5_continuation(...; cross_delta=true)` call), not a re-opening of the whole
  investigation.
- ~~Persistent `LFixBaseWorkspace` for `build_lfix_base_cache`'s ~590-650 MB/gradient
  (price0/pTσ0 tensors) — not started.~~ **DONE** (continuation session, addendum): built,
  tested (D=4 62/62, D=20 real-data cross-check), and benchmarked — 0 bytes allocated on warm
  reuse vs 589.6MB/call. See `docs/fullA_price_tensor_elimination_report.md`. **Not yet wired
  into `composite_gradient_at_fast_buffered`/`_pooled`** — the workspace/builder exist and are
  proven safe to use, but no driver calls them yet.
- `CrossDeltaExactCache` stage-transition hit-rate / end-to-end wall-savings benchmark — wiring is
  done (cache size before/after is logged per stage) but no real staged δ=2→3→4→5 run has been
  executed to produce numbers.
- The CM interrupted/resume production campaign the brief specifies (D=20, W=80,000, L=50, δ=1) —
  not run. `run_cm_upper_checkpointed` should not be treated as *the* production CM path until it
  is.
- ~~Fresh line-level `Profile.Allocs` audit, two-tensor representation experiment (A-D), dense
  post-solve materialization audit, final matched benchmarks (unrestricted δ=1/δ=2/δ=5, CM L=50)
  — none started.~~ **The two-tensor representation experiment is DONE** (addendum, this
  continuation session) — went further than a simple A-D sketch: full call-site audit, THREE
  working+validated backends (persistent workspace / pTσ-only / factorized log-score), each
  benchmarked at real D=20/W=80,000 against the current allocating reference. Headline: the
  factorized backend is 6.75x faster and uses 6x less memory for a full 400-coordinate gradient
  call, to machine precision. See `docs/fullA_price_tensor_elimination_report.md` for the full
  writeup, numbers, and decision. Fresh `Profile.Allocs` line-level audit and dense
  post-solve-materialization audit remain NOT started.

## 5. What this document does NOT claim

No allocation, GC, wall-time, or peak-RSS numbers for the DRIVER-integrated paths have been
measured yet (only the function-level `test_gradient_workspace.jl` numbers: pooled 756.5 MB vs
buffered 3690.0 MB, 4.88x, independently reconfirmed on this branch). The prior session's own
"16-29x cache speedup" figure has not been independently re-verified on this branch. Do not cite
either driver-level wiring as "benchmarked in production" — they are wired and function-level-
correct, not yet performance-validated end-to-end.
