# Unrestricted Allocation-Free FG — Final Gate — 2026-07-26

This branch did not modify the unrestricted family's own economic FG kernel (Addendum Part A,
inherited from `port/finish-five-family-optimization-stack-2026-07-26@fbb7d79`, adopted verbatim
per `docs/ADDENDUM_SHARED_ECONOMIC_FG_2026-07-26.md`'s own Part A instruction: "Do not rewrite this
mathematics"). This branch's only change to the unrestricted family's own files was a real
**include-order bug fix** (see below) that the inherited commit's own D=4 gates never caught
because every prior test happened to include `compressed_cc_inner.jl` before `compressed_live.jl`
by caller convention.

## What was inherited (unchanged)

- `EconomicFGWorkspace`, `compressed_dual_contraction!`, `compressed_transpose_contraction!`,
  `compressed_cc_value_grad!` (`compressed_cc_inner.jl`/`compressed_moments.jl`).
- Direct writes into KNITRO's own gradient buffer (`_callbackEvalFG_inner_compressed!`,
  `compressed_live.jl`).
- Reduced warmed per-callback allocation from ~2.57MB to 128 bytes (20,081x reduction), bit-for-bit
  identical output vs the dense reference, validated against
  `test_compressed_live_integration.jl`'s existing comprehensive suite.

## What this branch found and fixed

`cm_hessian_architectures.jl`'s self-guarded include block (`compressed_moments.jl` ->
`structured_moment_build.jl` -> `compressed_live.jl`) never included `compressed_cc_inner.jl` —
but `compressed_live.jl` has needed `EconomicFGWorkspace`/`compressed_cc_value_grad!` since Addendum
Part A. Every pre-existing include site happened to load `compressed_cc_inner.jl` separately
before `compressed_live.jl` by caller convention, masking the gap. Any driver reaching
`compressed_live.jl` only through `cm_hessian_architectures.jl` (e.g.
`test_phaseB1_cmlookup_production_correctness.jl`, `test_phase52_frechet_lookup_correctness.jl` —
neither includes `compressed_cc_inner.jl` directly) hit `UndefVarError` at load time. Found by
re-running the inherited branch's own D=4 correctness gates as a baseline check before building on
top of them, per this project's own "check in on a job's early output" discipline — this was a
load-time error, not a subtle numerical one, so it surfaced in the first few seconds. Fixed with a
one-line additional self-guarded include; confirmed both affected gates load and pass after the fix
(see this branch's first commit).

## Validation this session

- `test_compressed_live_integration.jl` (the existing comprehensive dense-vs-compressed equivalence
  suite, unmodified): re-run as this branch's own baseline check, **ALL PASS**.
- `test_phaseB1_cmlookup_production_correctness.jl d4` / `test_phase52_frechet_lookup_correctness.jl
  d4`: re-run at the end of this session (after all of this branch's own additions/includes were
  layered on top) as a regression check — **ALL PASS**, confirming this branch's new files
  (`no_dense_g_counters.jl`, `economic_operator.jl`, `zc_restriction_operator.jl`, the new
  origin-ZC/CM+ZC operator files) introduced no regression to the unrestricted or already-shipped
  CM/Fréchet lookup paths.

## Backend status (unchanged from the inherited port)

`ECONOMIC_FG_BACKEND[unrestricted] = compressed_operator` — live, no flag, the unrestricted
family's only FG path (has been since the inherited Addendum Part A commit). No default decision
needed from this branch.
