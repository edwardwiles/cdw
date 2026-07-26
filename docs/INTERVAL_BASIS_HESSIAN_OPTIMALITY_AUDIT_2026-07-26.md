# Interval-Basis Hessian Optimality Audit — 2026-07-26

## Status: NOT ATTEMPTED this session

Task §7.2 requires deriving the optimal interval-basis Hessian construction from scratch (one bin
lookup per origin/draw for the forward operation, one scatter for the transpose, the raw RR block
as the weighted joint-bin contingency table directly with no prefix sums, direct winner-bin
accumulation for the ER cross block) and explicitly warns not to assume the current cumulative
Architecture-C Hessian is optimal for interval moments.

This is a from-scratch derivation task, gated behind `INTERVAL_VS_CUMULATIVE_CM_BASIS_2026-07-26.md`
(the interval basis itself does not yet exist as a real, gated production moment construction) —
there is no interval-basis Hessian to audit for optimality until the interval-basis moments it
would differentiate exist and are validated. Not attempted this session; see that document's
scoped follow-on for the prerequisite work.

## What would need to happen, in order

1. Interval-basis moment construction, real and gated (`INTERVAL_VS_CUMULATIVE_CM_BASIS`'s own
   follow-on item 1).
2. Derive the interval-basis Hessian construction from scratch per the task's own architectural
   description (bin lookup, scatter, joint-bin contingency table, winner-bin ER accumulation) —
   independently, not by assuming the cumulative-basis Architecture-C derivation transfers.
3. Compare against the current cumulative-basis Hessian on: arithmetic complexity, allocations,
   callback time, KNITRO iteration count, conditioning, complete inner solve, outer progress (task
   §7.2's own list) — real D=4 and D=20/W=80,000 measurements, not asymptotic argument alone.
4. Apply the task §7.3 decision rule (scientific equivalence + non-regression on all of the above)
   before promoting interval/orthonormal as any kind of default. Retain cumulative as an explicit
   reference/replication mode regardless of outcome.
