# Five-Family Operator FG Performance A/B — 2026-07-26

## Scope statement

Task §16/§17 ask for per-callback and complete-inner-solve timing/allocation/GC/KNITRO-iteration
A/B for every family, plus a matched short real outer A/B for any family whose inner solve
improves materially. **Neither was run to completion for the two new operator families
(origin-ZC, CM+ZC) this session** — correctness gating (D=4 + real D=20/W=80,000, both ALL PASS)
consumed the available time; performance benchmarking was judged lower priority than closing out
correctness for the two genuinely-new operators, consistent with this branch's own inherited
priority order ("prefer closing out a validated, scoped piece over opening a new investigation").

## What IS known, directionally, from validated component measurements

- **Economic block (E)**: origin-ZC's and CM+ZC's operator FG reuse `economic_forward!`/
  `economic_transpose!` unchanged from the unrestricted family's own Addendum Part A kernel,
  independently measured at 20,081x allocation reduction (2.57MB -> 128 bytes/call) and 1.15x
  wall-clock speedup vs the equivalent dense economic-core `BLAS.gemv!`, bit-identical output. This
  same reduction should transfer directionally to origin-ZC's/CM+ZC's own economic block, since it
  is the identical code path against the identical `CompressedFactual` representation — not
  independently re-measured for these two families this session.
- **Z-restriction block**: `restriction_forward!`/`restriction_transpose!` read directly from the
  immutable `Zraw_all`/`Zpairraw_all` matrices, avoiding the dense `(W,D)`-or-`(W,npair)` centered
  copy (`dest = Z .- targets'`) the pre-existing `moments!` closures materialize every outer point
  — a real, structural allocation reduction relative to the status quo, not independently
  benchmarked at D=20 scale this session.
- **Common Fréchet**: real per-callback allocation measured this session (not a directional
  estimate) — see `COMMON_FRECHET_OPERATOR_FG_FINAL_GATE_2026-07-26.md` — 14,066,064 bytes/call,
  reproducible, NOT allocation-free, contradicting the inherited port's own "per-callback operator
  is allocation-free" framing. This is real evidence against flipping that family's default, not a
  performance number in the A/B sense the task asks for.
- **Flexible CM**: unchanged this branch — see the inherited port's own numbers
  (`RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md` §1.2), 1.108x-1.616x faster than dense at
  every tested thread count, allocation at exact parity.
- **Unrestricted**: unchanged this branch — 20,081x allocation reduction, 1.15x speedup (Addendum
  Part A's own numbers).

## What was NOT measured

- Origin-ZC and CM+ZC's own complete-inner-solve wall-clock/allocation/GC/KNITRO-iteration A/B
  against `:dense_reference`, at any thread count.
- Any short real direct-bound outer A/B for any family.
- CSVs/raw logs of timing data (task's own deliverable list asks for these) — none produced this
  session.

## Honest conclusion

Correctness is real and gated for both new operator families at both scales. Performance is
**directionally expected to improve** (component-level evidence above) but **not independently
measured or gated** for origin-ZC/CM+ZC — this is the single largest remaining gap before either
family's `fg_backend`/`inner_fg_backend` default could responsibly flip from `:dense_reference`,
and is flagged as the top follow-on item for the next session.
