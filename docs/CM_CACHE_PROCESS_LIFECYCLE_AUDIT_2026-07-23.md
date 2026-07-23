# Complete-state cache: process-lifecycle and hit-opportunity audit — 2026-07-23

Part III.1 of the follow-up brief. Traced from source (`scripts/cm_production_supervisor.sh`,
`full_aod_diag/d4_exact/cm_production_stage_runner.jl`, `cm_cold_verify.jl`, `cm_checkpoint.jl`),
not assumed.

## Verdict up front

**The prior session's design doc names the prototype the "complete-state exact/cross-delta
cache" and lists "a later stage/restart revisits a point already solved" as a target use case
(`docs/COMPLETE_STATE_CACHE_DESIGN_2026-07-22.md` line 36). This is exactly the framing the
brief warns against**: every delta stage, every supervisor restart, and every next-delta seed
launches a brand-new OS process (confirmed below, not assumed), and the cache is a plain
in-process `Dict` with no serialization. **It cannot be "cross-delta" or "cross-restart" as
currently built — those words describe the intent, not the demonstrated capability.** This does
not mean the cache is worthless (see the within-process hit classes below), but the design doc's
own framing needed this correction before any further wiring decision.

## III.1.a: process-lifecycle facts, traced from source

1. **Each delta stage runs in a new Julia process.** `cm_production_supervisor.sh`'s
   `run_stage_with_watchdog` (line 200-206): `( cd "$REPO_ROOT" && "$JULIA_BIN" --project=. ...
   cm_production_stage_runner.jl "$stage_dir" "$delta" ... ) &` inside a `while true` loop --
   every attempt at every delta is a fresh `julia` invocation, no exceptions.
2. **Supervisor restarts run in a new Julia process.** The SAME `while true` loop handles both
   the first attempt at a stage AND any restart-after-stall/wall-limit -- a restart re-enters the
   loop and launches ANOTHER fresh `julia` process (`cur_mode` becomes `"resume"`, pointing at the
   stage's own `stage_latest.jls`, but the OS process itself is new every time; confirmed by
   direct read, lines 161-230).
3. **The next-delta seed is a new Julia process at a DIFFERENT stage_dir.**
   `cm_production_supervisor.sh` lines 350-375: after cold-verifying one delta's stage, `mode="seed_w0"` /
   `seed_arg=$verify_out` is passed to the NEXT `run_stage_with_watchdog` call for the next delta
   -- another fresh process, per fact 1.
4. **Cold verification itself is a THIRD, separate, one-shot process per stage.**
   `cm_production_supervisor.sh` line 366: `"$JULIA_BIN" --project=. full_aod_diag/d4_exact/cm_cold_verify.jl
   "$ckpt_path" "$verify_out"` -- a dedicated process, launched and torn down once per stage,
   entirely disjoint from the stage runner's own process.
5. **No exact/cross-delta scalar cache currently persists across ANY of these boundaries.**
   `last_F_state` (`cm_checkpoint.jl`, `run_cm_upper_checkpointed`'s local closure) is a
   `Ref{Any}` living only in that one process's heap -- not a `CMCheckpointV3` field, never
   serialized. It is reset to `nothing` at the start of every process (fact 1-3). The prototype
   `CompleteStateCache` (`complete_state_cache.jl`) is architecturally identical in this respect
   -- also a plain in-process `Dict`, also never serialized, also starts empty every time.
6. **`last_F_state`/accepted-point reuse already eliminates the common `cb_F!`→`cb_G!` duplicate
   solve WITHIN one process.** `cb_G!` (`cm_checkpoint.jl`) reuses `last_F_state[]` when the
   gradient callback's point matches the immediately-preceding `cb_F!` call's point exactly
   (confirmed by direct read, algebra trace Section 1) -- this is the single highest-value,
   already-solved hit class, and it needs no cache at all (single-slot memoization suffices
   because KNITRO always calls `cb_F!` then `cb_G!` at the SAME point in immediate succession).
7. **KNITRO's own algorithm class matters for how often EXACT point revisits occur at all.**
   Every real trajectory run this session (`results/cm_cplus_followup/matched_trajectory_*`)
   printed `Knitro using the Interior-Point/Barrier Direct algorithm`. Interior-point/barrier
   methods move continuously along a central path with a monotonically-adjusted barrier
   parameter; unlike a trust-region method that can reject a step and retry from the exact SAME
   base point with a different direction, an interior-point method's consecutive iterates are
   determined by a continuous Newton-type update and are not naturally exact-repeat-prone. This
   is a real, if second-order, reason to expect the within-process hit rate to be low even before
   measuring it (Part III.2).

## III.1.b: call-path diagram

```
cm_production_supervisor.sh (bash, one process, orchestrates via sequential sub-processes)
 |
 +-- [chain loop, per chain] -> run_stage_with_watchdog(delta, ...)
       |
       +-- [attempt loop, possibly >1 per stage] -> NEW julia process:
       |     cm_production_stage_runner.jl
       |       -> run_cm_upper_checkpointed(...)              [ONE process's lifetime]
       |            kc = KNITRO.KN_new()
       |            KNITRO internal iterate loop:
       |              cb_F!(x) -> archC_verified_state/cm_production_value_verified
       |                           last_F_state[] = (x, base, verify)     [in-process only]
       |              cb_G!(x) -> if x == last_F_state[].x: REUSE base    [same-point hit, ALREADY HANDLED]
       |                          else: archC_base_state(x) fresh solve  [line-search/restoration revisit -- POTENTIAL cache hit class]
       |              cb_newpt!(x) -> accepted-point bookkeeping          [does NOT itself solve -- no hit to count]
       |              do_checkpoint(...) periodically                    [writes CMCheckpointV3 to disk -- crosses NOTHING back in]
       |            KN_solve returns -> final checkpoint written, process EXITS
       |     [process boundary -- any in-memory cache is destroyed here]
       |
       +-- [after stage succeeds] -> NEW julia process: cm_cold_verify.jl
       |     loads stage_latest.jls, re-solves best_feasible.w from a COLD context
       |     [process boundary -- cannot benefit from the stage runner's own in-memory cache
       |      even in principle, since it never existed by the time this process starts]
       |     writes cold_verified_seed.jls
       |
       +-- [next delta] -> run_stage_with_watchdog(next_delta, mode="seed_w0", seed=cold_verified_seed.jls)
             -> ANOTHER NEW julia process, same structure as above, starting from a DIFFERENT
                point (the previous delta's cold-verified incumbent, not the previous delta's own
                trajectory) -- even if an in-memory cache had somehow survived, the NEW delta's
                own KNITRO trajectory starts from a point the OLD delta's cache almost certainly
                never visited exactly (different feasible region, delta bound moved).
```

## III.1.c: hit-class table

| Hit class | Already handled? | Can hit an in-memory cache? | Requires persistence? | Notes |
|---|---|---|---|---|
| same `F`→`G` point (cb_F! then cb_G! at the identical x) | **YES** (`last_F_state`, single-slot) | n/a -- already free | No | The dominant, highest-value case; a cache adds nothing here. |
| accepted new-point callback (`cb_newpt!`) | N/A -- doesn't solve | N/A | N/A | Pure bookkeeping (updates KNITRO's own internal state); never itself a solve to cache. |
| line-search revisit (same iterate re-probed at a rejected step) | No | **Yes, in principle** | No (same process) | Real for SOME algorithms; not clearly common for the barrier/interior-point algorithm this campaign actually uses (fact 7) -- unmeasured until Part III.2. |
| solver restoration/rejection revisit | No | **Yes, in principle** | No (same process) | Same caveat as above. |
| checkpoint verification (`cm_cold_verify.jl`) | No | **No** -- separate process by construction (fact 4) | **Yes** | Would need the cache serialized alongside the checkpoint and re-fingerprint-validated; not attempted this session (see Part IV gate below). |
| supervisor restart (stalled/wall-limited attempt retried) | No | **No** -- new process (fact 2) | **Yes** | Same as above; the restarted attempt's own KNITRO trajectory does not obviously revisit the PRIOR attempt's exact iterates anyway (different RNG-free but algorithmically-continued path from a checkpoint, not the same run replayed). |
| next-delta seed | No | **No** -- new process, different stage_dir (fact 3) | **Yes** | Even with persistence, the next delta starts from a DIFFERENT point (the cold-verified incumbent) with a DIFFERENT feasible region (delta bound changed) -- an exact-key match against the PREVIOUS delta's cached points is structurally unlikely regardless of persistence. This is the specific claim the design doc's "cross-delta" framing needs to earn, not assume. |
| cross-chain duplicate (two chains visit the identical point) | No | **No** -- separate processes AND separate starting points/perturbation seeds by design | **Yes**, and even then astronomically unlikely | `cm_production_stage_runner.jl`'s own chain-perturbation design (0.02-scale seeded offset) exists specifically to give each chain a genuinely different trajectory -- an exact collision is not a realistic target. |

## What this means for Part III.2 and Part IV

Only ONE hit class (line-search/restoration revisit, within a single process) is both (a) not
already handled by existing `last_F_state` memoization and (b) reachable by the CURRENT
(in-memory, unpersisted) `CompleteStateCache` prototype without further engineering. Every other
listed class requires disk persistence to even be theoretically reachable, and for the classes
that matter most in the design doc's own framing (next-delta seed, cross-chain), persistence
alone would likely not be sufficient either, because the NEW starting point differs from anything
previously cached.

Part III.2 measures the one reachable class directly (shadow instrumentation on real trajectories)
rather than continuing to reason about it in the abstract.
