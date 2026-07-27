# A-Gradient Before/After Performance — 2026-07-27

## Scope and honesty note

This task's time went overwhelmingly into (a) the D=4 allocation reconciliation, (b) building and
correctness-gating the shared `economic_A_gradient!`, and (c) two unplanned but necessary bug
investigations (see `SHARED_OUTER_A_GRADIENT_ARCHITECTURE_2026-07-27.md`). **Wall-clock timing was
not this task's primary evidence axis and the numbers below should be read as a secondary,
lower-confidence signal** — the machine this session ran on was not confirmed idle (no load-average
check was run before/during timing, unlike the prior allocation audit which explicitly flagged its
own machine-load caveat). Allocation counts (`@allocated`) are unaffected by load and are the
primary evidence throughout this task's other documents.

## What was measured

The only wall-clock numbers captured this session are the `@time` outputs from
`test_shared_a_gradient_d20.jl`'s correctness section (D=20/W=80,000, `h_mode=:cached`, cold
bandwidth cache, SERIAL/`threaded=false`):

```
composite_gradient_at_fast (unbuffered, reference):  36.3 s  (6.14M allocations, 8.945 GiB, 16.75% compilation)
economic_A_gradient! (shared, this task):            31.6 s  (1.79M allocations, 703.8 MiB, 6.82% compilation)
```

Both numbers include a large one-time JIT-compilation fraction (16.8%/6.8% respectively, both
functions' first call on THIS process) since this was the correctness-gate call, not a dedicated
warmed timing loop. The compilation fraction itself is informative: `economic_A_gradient!`'s lower
compilation share is consistent with its allocation-free hot path touching fewer distinct
generic-dispatch call sites, but this is a plausibility argument, not a controlled measurement.

**No warmed, repeated, machine-load-controlled wall-clock benchmark was run this session** — the
allocation numbers in `A_GRADIENT_D20_ALLOCATION_RECONCILIATION_2026-07-27.md` are the trustworthy,
load-independent evidence; the two numbers above are reported for completeness but should not be
read as a rigorous "shared is X% faster than unbuffered" claim. Directionally, given the shared
path issues roughly 12x fewer GC-triggering allocations (1.79M vs 6.14M alloc COUNT, not just
bytes) and 12.7x fewer bytes at this scale, a genuine wall-clock improvement is plausible and would
be expected in a real KNITRO outer-loop run (where GC pressure compounds across thousands of
gradient calls), but this task did not isolate and confirm that effect with a controlled benchmark.

## What was NOT measured (honest gaps vs task §9's full list)

- Complete gradient wall time (properly warmed, repeated, GC-time-isolated) — not done.
- GC time specifically (the `@time` macro's own GC% field is reported above, but at only 1 sample
  each, under compilation-contaminated conditions) — not independently isolated.
- Base-cache construction time vs coordinate-loop time, split out — not measured (only bytes were
  split out, per the reconciliation doc; time was not).
- Family-wrapper overhead (the ZC-only `cm_originzc_production_gradient` wrapper's own cost beyond
  the shared `economic_A_gradient!` call) — not measured.
- Thread/task allocation (`threaded=true` path) — not exercised this session at all; every gate in
  this task's deliverables ran `threaded=false` only. The `economic_A_gradient!`/`TwoOriginScratch`
  machinery is written to support `threaded=true` (mirroring `composite_gradient_at_fast_pooled`'s
  own `:static`-scheduling discipline exactly, including the same per-thread-slot buffer-indexing
  safety argument), but this was NOT independently verified this session.
- Time/bytes attributed by source site (a profiler-based, not `@allocated`-based, breakdown) — not
  done, same gap noted in the reconciliation doc.

## Recommendation for the next session

Before trusting any wall-clock claim about this work, run a proper warmed benchmark (BenchmarkTools
`@benchmark` or a manual warm-then-time loop of >=20 repetitions) on an otherwise-idle machine
(check `uptime`'s load average first, per this project's own established practice in the prior
allocation audit), for both `threaded=false` and `threaded=true`, at both D=4 and real D=20.
