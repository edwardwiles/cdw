# Goal 10 — `skip_cm_fill_ref` Removal — Full Root-Cause Report — 2026-07-27/28

## Summary

Task Goal 10 asked to remove `skip_cm_fill_ref` — a `Ref{Bool}` mutated around individual function
calls (a "mutable global fill-toggle" anti-pattern) — and replace it with explicit, non-mutating
dispatch. That mechanical refactor is **done and safe**: no code anywhere in this branch now reads
or writes a mutable `Ref{Bool}` to decide whether to fill CM/level moment columns.

Separately, and initially conflated with that refactor, a re-investigation was launched into
whether the *underlying optimization the Ref used to gate* — skipping the dense fill of CM-grid/
level moment columns under the `:cm_lookup`/`:cm_frechet_lookup` FG backends — was now safe to
re-enable, since it had previously been disabled after a real bug (`nStatus=-400`, commit
`5fd6347`, 2026-07-26/27) and a plausible-looking hypothesis suggested the thing that made it
unsafe might have since been fixed by later work. **That hypothesis was tested empirically at real
D=20 scale and found wrong for both families that could use it (flexible-CM and common-Fréchet).**
The skip is disabled again, and this document exists to leave a precise, dynamically-verified
record of *why*, so a future session does not have to re-derive it — and to specify exactly what a
real fix would require, since it is a well-defined and traceable gap, not a mystery.

## Timeline of this investigation (accurate, not retrospectively cleaned up)

1. Removed the `Ref{Bool}` mechanically for flexible-CM, threading `skip_fill::Bool` explicitly.
   Verified via a byte-identical git-stash A/B that this alone changed nothing.
2. Investigated `archC_verified_state`'s own skip decision (a *different*, narrower question: does
   the *post-solve verification* step read `obj.H`?). Found and fixed a real, confirmed-safe gap
   there: `verify_inner_solution_operator_cm!` genuinely never reads `obj.H` under the
   `:operator` verification backend (confirmed by reading it in full) — closing
   `dense_CM_G_materializations` to 0 for that call site was correct and remains so.
3. Separately, hypothesized that the *inner-solve-time* skip (a different mechanism, gating
   whether the CM/level columns get filled *before* the Hessian callback runs at all, not after)
   might now be safe for common-Fréchet, because the winner-bin `H_E,level` Hessian path
   (`winner_pair_cross_hessian_colsum!`/`_esum!`, `cm_frechet_hessian.jl`, commits
   `e3bce93`/`d458702`) is a git descendant of the original `nStatus=-400` bugfix commit
   (`5fd6347`) by about 7.5 hours — i.e. it postdates the state the original "unsafe" finding was
   made against. D=4 multi-point testing (calibration + 3 perturbed/hard points) supported the
   hypothesis: all feasible, values agreed to 8–11 significant figures.
4. **A real D=20/W=80,000 (`destination_sample=:exclude_row`) re-test at non-calibration points
   disproved the hypothesis.** At the calibration point, skip vs. fill agreed closely (a benign
   `nStatus` label difference, see below). At **both** tested perturbed points, the skip arm
   reproduced the *exact* original `nStatus=-400` failure; the fill arm solved the identical points
   cleanly (`nStatus=0`).
5. Root-caused this dynamically rather than accepting "it's unsafe, don't know why": traced the
   actual consumer chain (not a repeat of the earlier, incomplete static trace) and found
   `_archC_prep_for_hessian!` (§"Root cause" below).
6. Checked whether flexible-CM's own skip — already reported as "fully closed, verified safe" —
   had the identical bug, since it shares the exact same root-cause mechanism. **It did.** It had
   only ever been tested at the calibration point (isolated single-call checks) or at D=4 scale;
   never at a real D=20 non-calibration point. A dedicated re-test reproduced the identical
   `nStatus=-400` failure at the identical two perturbed points.
7. Reverted the inner-solve-time skip to unconditional fill for **both** families. Confirmed via a
   final real D=20 re-run that skip and fill arms now produce **bit-identical** results
   (`|Δζ*|=0.0` exactly, not just close) at all three tested points for both families, `nStatus=0`
   everywhere. The mechanical Ref-removal refactor (item 1) is retained; only the *decision* is
   now hardcoded `false`.
8. Confirmed, by direct comparison against the unrestricted family's own production Hessian
   callback, that this dense dependency is architecturally unnecessary in principle (§"Why
   unrestricted doesn't have this problem" below) — i.e. this is a well-specified, fixable gap, not
   an inherent property of the restricted math.

## Root cause, precisely

`_archC_prep_for_hessian!` (`cm_hessian_architectures.jl:963-968`):

```julia
function _archC_prep_for_hessian!(obj, x)
    @unpack H, arg0, arg1, outer_constr_index, Psi! = obj
    BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
    Psi!(arg1, arg0)
    return nothing
end
```

This is called **unconditionally, on every Hessian callback**, by both `archC_hess_cb_builder`
(flexible-CM, `cm_hessian_architectures.jl:1336-1357`) and `archC_frechet_hess_cb_builder`
(common-Fréchet, `cm_frechet_hessian.jl`) — the exact same call, `_archC_prep_for_hessian!(o,
xloc)`, immediately before the family's own structured Hessian fill.

`outer_constr_index` is **not** just the economic-column count for either family:
`cm_hessian_architectures.jl:312`/`cm_frechet_level.jl:160`: `outer_constr_index_new =
obj0.outer_constr_index + ncm`, and both sites assert `obj_cm.outer_constr_index == obj_cm.d`
(`cm_hessian_architectures.jl:329`, `cm_frechet_level.jl:172`) — i.e. it spans **every** restriction
column (CM-grid, and level for common-Fréchet), not just economic ones. So `H[:, 2:1+outer_constr_index]`
genuinely includes the CM-grid/level columns, and when the moments-closure skip leaves those
columns unfilled/stale, this `BLAS.gemv!` computes `arg0` from a mix of correct (economic) and
garbage (CM/level) column data — **discarding** whatever value `arg0` already held.

That matters because `arg0` feeds `w = ddPsi!(arg0)` inside the family's own structured Hessian
fill (`hessian_cm_frechet_structured!`/`hessian_cm_structured!`), and `w` is a **shared, per-draw
weight vector used by every block of the packed Hessian** — including the winner-bin-computed
`H_EE`/`H_EC` blocks that never read dense `H` themselves (`_fill_cm_HEE!(HEE, w, ...)`,
`build_bin_tables!(cctx, E, w; ...)` both take `w` as an argument). So corrupting `arg0` corrupts
the **entire** Hessian, not just CM/level-related entries — consistent with the severity actually
observed (a real `nStatus=-400` solve failure, not a subtle numerical discrepancy).

## Why unrestricted doesn't have this problem — precise, direct comparison

This was checked directly, not assumed, in response to the natural question: is this a general
property of how this codebase forms `G'SG`-type Hessians (unrestricted included), or specific to
the restricted families?

**Unrestricted's production Hessian callback** (`_callbackEvalH_inner_compressed!`,
`compressed_live.jl:225-258`) calls `hessian_core_winner_pair!(evalResult.hess, obj.arg2, obj,
st.core_ws.parallel_ws; ...)` directly — **no equivalent of `_archC_prep_for_hessian!` at all**.
Its own docstring (`compressed_live.jl:201-223`) states explicitly: *"nothing else in this inner
solve reads `obj.H`'s G columns; the FG callback above already gets everything it needs from
`st.cf` directly... The dense `obj.H` materialization this replaced is now SKIPPED for the
production default backend."*

Inside `hessian_core_winner_pair!` itself (`core_exact_hessian.jl:716-724`):

```julia
ddPsi! = obj.ddPsi!
ddPsi!(obj.arg2, obj.arg0)
S = obj.arg2; M = obj.M
```

This **trusts `obj.arg0` as already correct** — set by the immediately-preceding FG callback
(`_callbackEvalFG_inner_compressed!`, which computes it compressed-natively from `st.cf` — bin/
draw-level data, never dense `H` — and publishes `obj.arg0 .= st.fg_ws.q`) — and only applies
`ddPsi!` to derive the weight vector. **It never reads dense `H` to reconstruct `arg0`.** No
redundant recompute, no dense dependency, nothing to skip.

**This is the precise answer to "is unrestricted doing the same thing":** No. Unrestricted's
production path was *already* built the "right" way — the Hessian callback trusts the FG
callback's own correctly-computed `arg0` rather than redundantly reconstructing it from a dense
matrix. The restricted families' `_archC_prep_for_hessian!` does the *opposite*: it distrusts
`obj.arg0` and unconditionally reconstructs it from dense `H`, discarding the lookup-mode FG
callback's own already-correct bin-based computation of the exact same quantity. Given the
restricted lookup-mode FG evaluators (`(st::CMFrechetLookupState)(x,g)`, `cm_frechet_lookup_kernels.jl:191-266`,
and its plain-CM analogue) demonstrably compute `arg0`/`r` correctly from bins alone — confirmed by
direct code read: the CM and level contributions there are built entirely from
`cumulative_forward_contribution!`/`frechet_level_forward_sum!` over `st.bins`, never from `H` —
the dense recompute in `_archC_prep_for_hessian!` is **provably redundant under lookup mode**, not
a genuine mathematical requirement of the restricted formulation.

**Most likely explanation for why it exists as unconditional**: `_archC_prep_for_hessian!` predates
lookup-mode FG (it serves the original generic/dense Architecture-C path,
`inner_loop_internal_archgeneric`, where nothing else keeps `obj.arg0` fresh going into the
Hessian callback). When lookup mode was added later as an optimization, this prep step was reused
unchanged for both paths rather than given a lookup-aware counterpart.

## Real evidence (not summarized from memory — actual run output)

### D=4, common-Fréchet, calibration + 3 points (hypothesis-supporting stage, before the D=20 test)
All 4 points feasible, `dense_cross_hessian_calls=0` (winner_bin genuinely engaged), values agreed
to 8–11 significant figures. `nStatus` shifted `0→-103` under skip at every point — see "benign
status shift" note below.

### D=20/W=80,000/`:exclude_row`, common-Fréchet, BEFORE revert (real KNITRO output)
```
--- point: calibration ---
  SKIP: nStatus=-103  zeta*=-0.015822404017333463  dense_Frechet_G_materializations=0
  FILL: nStatus=0     zeta*=-0.015822403973620686  dense_Frechet_G_materializations=1
  COMPARE: |Δzeta*|=4.371277645409677e-11  max|Δlambda*|=1.2627217604865848e-9

--- point: perturbed_A ---
  SKIP: EXCEPTION: CMExpectedSolveFailure: archC_frechet_base_state: inner solve failed, nStatus=-400
  FILL: nStatus=0  zeta*=-0.3837874621544284  dense_Frechet_G_materializations=1

--- point: perturbed_B ---
  SKIP: EXCEPTION: CMExpectedSolveFailure: archC_frechet_base_state: inner solve failed, nStatus=-400
  FILL: nStatus=0  zeta*=-0.3803606909120858  dense_Frechet_G_materializations=1
```
Independently reproduced twice (two separate concurrent processes, identical seed) — deterministic,
not a race or a fluke.

### D=20/W=80,000/`:exclude_row`, flexible-CM, BEFORE revert (real KNITRO output) — the check that
### disproved "flexible-CM is already safe"
```
--- point: calibration ---
  SKIP: nStatus=-103  zeta*=-0.008661978886517567  dense_CM_G_materializations=0
  FILL: nStatus=0     zeta*=-0.00866197888560738   dense_CM_G_materializations=1
  COMPARE: |Δzeta*|=9.10186856439843e-13  max|Δlambda*|=1.4160408956520598e-11

--- point: perturbed_A ---
  SKIP: EXCEPTION: CMExpectedSolveFailure: archC_base_state: inner solve failed, nStatus=-400
  FILL: nStatus=0  zeta*=-0.1497868605656479  dense_CM_G_materializations=1

--- point: perturbed_B ---
  SKIP: EXCEPTION: CMExpectedSolveFailure: archC_base_state: inner solve failed, nStatus=-400
  FILL: nStatus=0  zeta*=-0.14856774540458445  dense_CM_G_materializations=1
```
Same points, same seed, as the common-Fréchet test — a direct, matched A/B.

### D=20/W=80,000/`:exclude_row`, BOTH families, AFTER revert (final verification)
Common-Fréchet: `ALL POINTS OK (feasible + winner_bin engaged + skip/fill agree)`, `EXIT=0`.
Flexible-CM:
```
--- point: calibration ---
  SKIP: nStatus=0  zeta*=-0.00866197888560738  dense_CM_G_materializations=1  winner_cross_hessian_calls=6
  FILL: nStatus=0  zeta*=-0.00866197888560738  dense_CM_G_materializations=1
  COMPARE: |Δzeta*|=0.0  max|Δlambda*|=0.0

--- point: perturbed_A ---
  SKIP: nStatus=0  zeta*=-0.1497868605656479  ...
  FILL: nStatus=0  zeta*=-0.1497868605656479
  COMPARE: |Δzeta*|=0.0  max|Δlambda*|=0.0

--- point: perturbed_B ---
  SKIP: nStatus=0  zeta*=-0.14856774540458445  ...
  FILL: nStatus=0  zeta*=-0.14856774540458445
  COMPARE: |Δzeta*|=0.0  max|Δlambda*|=0.0

ALL POINTS OK (feasible + winner_bin engaged + skip/fill agree)
```
Bit-identical, not merely close — confirms the "skip" code path and the "fill" code path are now
computing the literal same thing, i.e. the mechanism genuinely is a no-op post-revert, and
`winner_bin` (the separately-validated, already-safe cross-Hessian backend) remains correctly
engaged throughout (`dense_cross_hessian_calls=0` in every arm checked).

### The benign `nStatus` 0-vs--103 shift, explained
At the calibration point (only), skip and fill agree very closely in value
(`|Δzeta*|` at the `1e-11`–`1e-13` scale — the same order of magnitude every other winner-bin-vs-
dense Hessian comparison in this project shows, e.g. CM+ZC's H_CZ/H_ZZ gates: `6.4e-13`–`9.0e-13`)
but KNITRO reports a different status code. Looked this up directly in this codebase's own status
registry (`knitro_status.jl:48`): `-103 => KN_RC_FEAS_FTOL, :feasible_approx, true, "Primal
feasible; terminated because the relative objective change fell below ftol for ftol_iters
consecutive iterations."` — a well-defined, officially-acceptable KNITRO termination code (same
`:feasible_approx` family as `0`/`-100`/`-101`), not a failure. A tiny (floating-point-non-
associativity-scale) difference in Hessian values between two different-but-mathematically-
equivalent computation orders can shift which iteration KNITRO's objective-improvement check
crosses its tolerance on, without indicating a wrong answer. Confirmed as a **general**, not
common-Fréchet-specific, artifact via a control test: flexible-CM's skip shows the identical
`0→-103` shift at the calibration point too.

## What is NOT fixed (explicit scope boundary)

The **real** fix — giving `_archC_prep_for_hessian!` (or the lookup-mode call sites) a bin-aware
variant that computes `arg0` from `x` the same way the lookup FG evaluator already does, instead of
either (a) redundantly reading dense `H` (today's reverted-to-safe state) or (b) unsafely skipping
the fill without replacing the redundant read (the bug this document reports) — is **not
implemented in this session**, by explicit user direction, in favor of writing up this record
precisely and completely first. It is a well-specified, real piece of numerical-kernel work,
comparable in scope to this session's own `H_CZ`/`H_ZZ` work (Goals 7–8), requiring:

1. A new function, e.g. `_lookup_prep_for_hessian!(obj, st, x)`, that reproduces
   `_archC_prep_for_hessian!`'s contract (`obj.arg0`/`obj.arg1` populated correctly for the current
   `x`) by calling the SAME bin-based forward computation the lookup FG evaluator already uses
   (`cumulative_forward_contribution!`/`frechet_level_forward_sum!` for common-Fréchet;
   the plain-CM analogue for flexible-CM), rather than `BLAS.gemv!` against dense `H`.
2. Wiring `archC_hess_cb_builder`/`archC_frechet_hess_cb_builder` (or a lookup-specific variant) to
   call this instead of `_archC_prep_for_hessian!` when `inner_fg_backend===:cm_lookup`/
   `:cm_frechet_lookup`.
3. A machine-precision D=4 AND real D=20 gate (calibration + genuinely perturbed points, not just
   calibration — this exact gap is what let both of this session's bugs hide) comparing the new
   path's packed Hessian against the dense-reference construction, for both families.
4. Only then would `skip_fill`/`moments_skip!` (already-built, currently-dead-code infrastructure)
   become safe to actually engage in production.

## Current safe state (this document's own deliverable)

- `skip_cm_fill_ref` (the mutable `Ref{Bool}`) is fully removed from the codebase. No feature's
  behavior depends on mutable ambient state for this decision anymore (Goal 10's literal ask,
  satisfied).
- The CM-grid/level dense column fill is **unconditional** for both flexible-CM and common-Fréchet
  today — identical, safe, verified behavior to before this investigation began (and to every prior
  production release of this codebase).
- `dense_CM_G_materializations`/`dense_Frechet_G_materializations` are `1` per inner solve for
  these two families (an honestly-reported, currently-necessary cost — not a violation of any
  "zero" invariant this task defines, since those two counters were never in the required-zero set;
  only `full_G_materializations`/`dense_economic_G_materializations`/`generic_dense_FG_calls`/
  `dense_reference_verification_calls`/`dense_cross_hessian_calls` are, and all five remain `0`).
- `archC_verified_state`'s own independent fix (item 2 in the timeline — the post-solve
  verification no longer redundantly reads `obj.H` under `:operator`) remains in place and correct;
  it was never implicated in the D=20 failure (it does not affect KNITRO's live solve, only a
  post-hoc check) and its own isolated counter check (`dense_CM_G_materializations: 1→0` at that
  specific call site) still holds.
