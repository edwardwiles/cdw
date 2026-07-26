# Restricted Operator FG Production Port — 2026-07-26

## Status: NOT ATTEMPTED this session — scoped, not superficially claimed

Task §5 asks for genuine `forward!`/`transpose!`/Hessian-block operator kernels for all four
restricted families, replacing the current chunked-dense-fill pattern documented in
`NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md`. This is new numerical-kernel development
comparable in scope to (or larger than) the inherited remediation's own Phase B1 lookup-FG port,
which took a full dedicated session on its own for **plain flexible CM alone** and, even then,
landed `AVAILABLE_BUT_NOT_DEFAULT` (allocation regressed) rather than a clean win.

This session prioritized: (0) auditing and correctly re-gating the inherited branch's own claimed
work at real D=20 scale for all four families (closing a real, disclosed gap — origin-ZC had never
been reached at D=20 for exact-cache/workspace), (2) a real dual-bank trajectory benchmark, and
(8) the transformed-A default promotion. Given real time constraints, attempting genuinely new
Hessian/gradient kernel derivation for four families in the remaining scope risked shipping
unvalidated numerical code under time pressure — explicitly the failure mode this project's own
CLAUDE.md and memory warn against repeatedly (do not explain away or rush past a real numerical
discrepancy; verify, don't assume).

## What exists today, concretely (from the audit)

- Flexible CM and common Fréchet: a **chunked**, reused-scratch dense-fill (`fill_cm_columns_from_
  bins!`, `chunk_size=2000`) — bounded and allocation-conscious, but still fills a dense
  `(chunk, ncore_full)` block per chunk rather than a true matrix-free operator.
- CM+ZC: **fully dense** CM columns, retained by an explicit, already-benchmarked decision (Phase
  E part 2, adopted this session) — furthest from compliant.
- Origin-ZC: small, fixed-size (`K_mean`/`K_pair` scale, not `L·D` scale) raw power features — the
  lowest-priority target since even a literal dense block here is small.

## Scoped follow-on (not attempted, but concretely defined)

1. **Flexible CM first** (lowest risk — the existing `:cm_lookup` kernel in
   `cm_lookup_production.jl` is already a validated, if allocation-regressed, O(W·(D-1)) FG
   kernel for exactly this family). The real remaining work there is the **allocation** fix task
   §5.5 describes (persistent per-thread scratch, no `CMLookupState` rebuild per solve), not a new
   derivation — re-profile after removing the allocation sources the inherited session's own
   report already diagnosed (Phase B1: "+9.6% wall-clock, allocation +12.6%" — the allocation
   increase is the blocker, and its causes are named, not mysterious).
2. **CM+ZC second**, reusing (1)'s bin-lookup for the CM block and adding a small, separate
   mean/pair operator (task §5.3's `[E | C | Z]` partition) — do NOT attempt to force this before
   (1) is solid, since CM+ZC's own moments! already reuses whatever the plain-CM path provides.
3. **Common Fréchet third** (reuses (1)'s CM operator plus the level-anchor direction).
4. **Origin-ZC last** (task §5.4's `[E | Z]` partition — genuinely new, since it has no existing
   `:cm_lookup`-style kernel to build from; the inherited remediation's own report explicitly
   flagged this as "Phase B2 — genuinely new kernel development, not begun").

Each step requires the task's own §5.6 flip rule (D=4+D=20 correctness, non-inferior speed,
material allocation reduction, no stability regression, `full_G_materializations=0`) before any
default change — none of that gating work has started.
