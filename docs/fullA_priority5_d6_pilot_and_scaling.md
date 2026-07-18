# Continuation 5, Priority 5: D=6 pilot + D-scaling profile with the new incremental gradient

## 1. Finding a genuinely feasible D=6 starting point

The calibration point (`Aod_theta=1` everywhere) is **cold-infeasible at D=6/8/10**
(`inner_status=-300`), confirmed directly this session — matching
`docs/fullA_block_local_performance.md` sec 7's earlier D=6 finding, now also confirmed at D=8/10. 60
random perturbation trials around calibration (log-normal jitter, uniform γ') found nothing feasible.
An unconstrained-Delta-minimization attempt (60s, central-FD gradient) from calibration ALSO found
nothing feasible.

**What worked**: `ctx.θ0_up`'s own "natural" pre-gammanorm `Aod_theta` values (the structural values
baked into the synthetic economy's own construction, `θ0_up[Aod_offset+1:Aod_offset+D²]` — NOT
`ones(D,D)`) give a genuinely inner-feasible point directly: `inner_status=0`,
`Delta=0.0029` at D=6 (found by direct query, zero search needed). This is the correct D=6/8/10
analogue of what `d4_exact_setup`'s own `θ0_up` already provides at D=4 (where the SAME natural values
happen to coincide closely enough with `ones()` for D=4's specific calibration that this distinction
was never previously exposed) — a useful, reusable finding: **future D-scaling work should start from
`θ0_up`'s own natural Aod values, never `ones(D,D)`**, for D≥6.

Before trusting `composite_gradient_at_fast` at this new point, verified `build_lfix_base_cache`'s
self-validation passes here (it does) — confirming Priority 4's newly-found bug (sec below) is
specific to DECOUPLED (γ,A) pairs, not a general D=6 problem; this natural, jointly-consistent point is
safe to use.

## 2. D=6 pilot: both directions run, both genuinely feasible

`run_d6_pilot.jl`, `lfix_composite_fast`+SR1, 60s budget each direction, starting from the natural
feasible point above (`gp0=0.9541433952897307`, `Delta0=0.0029`, plenty of slack).

| direction | KNITRO status | outer iters | wall (KNITRO's own log) | n_eval | grad calls | grad wall total | best-feasible κ | Δ−δ |
|---|---|---|---|---|---|---|---|---|
| upper | -401 (iteration/time-limit, did not fully converge) | ~113* | **60.0s** (used full budget) | 692 | 259 | 27.53s | 0.19758 | −3.0e-4 (0.03% slack) |
| lower | **-103 (genuine convergence)** | 31 | **13.7s** | 215 | 32 | 3.21s | **0.007458** | **−9.7e-5** (essentially exactly on the boundary) |

(*upper's outer-iteration count wasn't printed directly by this driver; inferred range from KNITRO's
own iteration log, not exact — flagged.) Both best-feasible points passed a fresh **cold recheck**
(`evaluate_fullA(...; warm=false)`) reproducing the tracked Δ to the last printed digit — genuine,
reproducible feasibility, not a KNITRO-internal-state artifact.

**This is a clean, working D=6 pilot in BOTH directions** — the task's Priority 5 gate ("find or
construct a genuinely feasible D=6 benchmark point, then run one short upper and lower D=6 pilot")
is met. The lower direction converges cleanly within its budget; the upper direction is still making
progress at the 60s mark (would likely improve further with a longer budget, consistent with D=4's own
experience where SR1 needed ~27-43s to fully converge) — not claimed as a finished D=6 bound, a working
pilot per the task's own framing ("D=8/10 remain conditional on D=6... do not run D=20").

### A cosmetic bug, noted not chased

`run_d6_pilot.jl`'s own self-reported `wall` field (computed via `time()` around `KN_solve`) printed
`0.1s` for both directions -- clearly wrong given `grad_wall_total` alone was 27.53s (upper). Likely a
Julia closure-scoping interaction between the per-gradient-call `t0` inside `cb_G!` and the outer
`t0`/`wall` timer (not fully root-caused, diagnostic-only, does not affect any optimization result --
every reported numerical result was independently reproduced via the cold recheck and cross-checked
against KNITRO's own "Total program time" log line, 60.01s / 13.68s respectively, used in the table
above instead of the buggy self-reported field).

## 3. D-scaling profile with the new incremental gradient

Combines this session's moment-construction audit (`docs/fullA_moment_construction_audit.md`, pure
moment-BUILD cost only) with the D=6 pilot's LIVE gradient-call cost (the actual quantity Priority 5b
asks about: "redo the D-scaling profile using the new incremental live gradient... the old D^3.5-3.8
projection was measured before block-local/incremental reuse and is now stale"):

| D | n_free (A-block, D²) | moment-build (warmed, `JULIA_NUM_THREADS=1`) | live `lfix_composite_fast` gradient-call cost |
|---|---|---|---|
| 4 | 16 | 5.75ms | **71ms/call** (`docs/fullA_p2_p3_fast_gradient_and_comparison.md`, SR1, 60s live run) |
| 6 | 36 | 13.97ms | **~106ms/call** (27.53s / 259 calls, upper direction, live run above) |

**Gradient-call cost ratio D=4→D=6: ~1.49x, for a 2.25x increase in free A-block coordinates (16→36)**
— i.e., **sub-linear in n_free**, decisively better than a naive per-coordinate-cost-times-n_free
model would predict (which would suggest ~2.25x), and dramatically better than the old (pre-Phase-2)
`Delta_FD` D^3.5-3.8 projection. This is consistent with `docs/fullA_block_local_performance.md` sec
7's own qualitative prediction ("the case should strengthen at larger D... not yet quantified") — now
quantified, for the first D>4 live data point this investigation has produced with the validated
composite gradient. Moment-build cost itself grew ~2.4x (5.75→13.97ms) over the same D jump, roughly
tracking the O(D²) construction cost per Priority 1's own analysis — the GRADIENT's sub-linear growth
specifically reflects the block-local/incremental machinery's design (only 1-2 destination blocks touch
per coordinate, independent of total n_free), doing its job.

**Not established**: a clean power-law fit (only 2 data points, D=4 and D=6, both single live runs, not
repeated for a confidence interval) — flagged as directional evidence, not a precise scaling exponent.
D=8/10 live gradient-cost data (conditional on D=6 per the task's own gating, now cleared) is a natural
next step for a future continuation, using this same natural-Aod-start recipe.

## 4. What remains open

- **Priority 5a (nested W=8000/20000/80000)**: NOT attempted this continuation. Building a genuinely
  nested draw pool (shared random stream with W=8000/20000 as PREFIXES of the W=80000 draws, not
  independent redraws) is a distinct engineering task from anything else in Continuation 5 and was not
  reached given the time this session spent on Priority 4's gamma-profile bug investigation and the
  D=6 feasible-point search. Flagged explicitly, not silently skipped.
- D=8/10 live pilots: conditional on D=6 per the task's gating, which is now cleared (D=6 pilot works
  in both directions) — a natural next step, not attempted this session.
- The `build_lfix_base_cache` self-validation bug found in Priority 4 remains unfixed — any future
  D-scaling or gamma-profile work that evaluates the composite gradient at DECOUPLED (γ,A) pairs
  (rather than points reached by a genuine joint optimization trajectory, as both the D=6 pilot and
  every Priority 2/3 live run did) should expect to hit it and budget time for a real fix, not just
  the fallback.
