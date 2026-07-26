# Winner-Aware Cross-Hessian Benchmark — 2026-07-26

## Status: NOT ATTEMPTED this session

Task §6 asks for an exact winner-bin cross-Hessian operator (`Q'SR - π(ν'SR)`) for CM/common-
Fréchet, an analogous winner-scatter operator for CM+ZC's mean/pair cross block, and a
winner-feature cross operator benchmarked (not forced) against origin-ZC's dense `H_EZ`.

This depends on Phase 5 (§ above) being real and validated first — the cross-block operator's
whole point is to avoid materializing the economic side densely, which only matters once the
*direct* `H_EE`/`H_RR` blocks are themselves operator-based rather than chunked-dense. Building the
cross-block operator before Phase 5 exists would mean benchmarking against a still-partially-dense
baseline, which would not answer the real question ("does winner-structure exploitation help once
the rest of the pipeline is also operator-based").

**Existing, relevant prior work** (not built this session, but already in the tree and worth
noting so this isn't re-derived from zero): the CM-side cross-block already exploits
bin/prefix-sum structure on the *restriction* side (per the task's own framing of the current
state) — the audit in `NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md` did not re-verify how far
that existing exploitation goes on the *economic* side; that would be the first concrete step of
this benchmark, before writing any new candidate operator.

## Scoped follow-on

1. Audit (not build) exactly which of `H_EE`/`H_ER`/`H_RR` per family currently use winner
   structure vs generic dense BLAS, at the same granularity `NO_FULL_G_MATERIALIZATION_AUDIT`
   used for the FG path — a natural first task for whoever picks this up, and itself useful
   independent of whether the operator work proceeds.
2. Only after Phase 5 lands: build and gate the winner-bin cross candidates per family, per task
   §6.1-6.3, with the §6.4 measurement discipline (isolated cross-block time, complete Hessian
   callback, complete inner solve, allocation, real short outer progress) before adopting.
