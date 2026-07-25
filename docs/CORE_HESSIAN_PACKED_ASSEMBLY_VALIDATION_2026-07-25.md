# Packed-assembly validation — 2026-07-25

Task §2/§7's "local packed core output" / "insertion into the correct locations of a larger
family-level packed Hessian" / "precomputed local-to-global packed-index maps" requirements.

## Design decision (and the alternative rejected)

The diag branch's own gap analysis (`docs/HVP_OPERATOR_READINESS_MAP_2026-07-25.md`,
`docs/HESSIAN_BLOCK_OPERATOR_GAP_ANALYSIS_2026-07-25.md`, both in
`diag-compressed-hessian-operator-audit-2026-07-25`) flagged that the winner-pair kernels only ever
produced a STANDALONE packed Hessian (`n = 1 + wctx.ncolI` hardwired to the entire problem's
`outer_constr_index`) — no offset parameter, no support for writing into a sub-block of a larger
packed Hessian. This was the one genuine engineering gap this port had to close.

**Option considered and rejected**: hand-roll a `PackIdx`-style local-to-global lookup table
(the parallel kernel already has one internally, for its OWN standalone packed output) that maps
a local `(i,j)` pair to a global packed position within a LARGER Hessian's own triangle. Rejected
because every family in this codebase already materializes a DENSE symmetric scratch matrix
BEFORE packing into KNITRO's triangle (`obj.∂∂f_∂∂x`, `cctx.Hfull`) — a global packed-index table
would duplicate information Julia's own array-view offset arithmetic already provides for free,
and would need to be kept in sync with each family's own (different) packing/column-ordering
convention by hand.

**Design adopted**: `fill_core_hessian_upper!(Hdense, ...)` fills a SYMMETRIC DENSE
`(1+ncolI)×(1+ncolI)` block into `Hdense`, which the caller passes as either:
- the family's ENTIRE dense scratch (unrestricted — no restriction columns exist), or
- a `@view` into the top-left corner of a LARGER family scratch (CM: `cctx.Hfull[1:NCORE,1:NCORE]`;
  origin-ZC: `obj.∂∂f_∂∂x[1:NCORE,1:NCORE]`).

The view's own offset bookkeeping IS the local-to-global map — correct by construction (Julia's
own indexing semantics), not a hand-derived formula that could contain an off-by-one. Computation
of the exact core upper triangle (`hessian_core_winner_pair!`/`winner_pair_hessian!`, into a
persistent `packed_scratch` buffer owned by `CoreExactHessianWorkspace`) is kept strictly separate
from assembly into the caller's larger Hessian (the final unpack-into-`Hdense` loop), per task §2's
explicit instruction not to let family-specific code reimplement the core algebra.

## Validation

The D=4 correctness gates (`test_shared_core_hessian_d4_gates.jl`, task §7) directly validate this
design for all three insertion modes:
- **Standalone** (unrestricted): `max|Δh_serial|`, `max|Δh_par(workers=2)|`,
  `max|Δh_par(workers=4, storage=:direct_packed)|` vs `CS.hessian!`'s own packed output, at 6 dual
  points (zero, 4 random, 1 real KNITRO-solved) — all agree to machine precision after the
  off-by-one fix documented in the master summary.
- **Corner-of-larger-block, dense-scratch-then-pack** (flexible CM, both serial and
  production-default threaded-bins Architecture C): `max|ΔH_EE|` and `max|ΔH_full|` (the FULL
  assembled family Hessian, not just the extracted corner) both agree to ~1e-13–1e-16 against the
  pre-port dense-BLAS path.
- **Corner-of-widened-block** (CM+mean/ZC, where the "core" corner is narrower than the family's
  own widened "economic" block): `max|ΔH_EEcore|` (the TRUE core sub-block, `1:ncore_core`, not the
  wider `1:NCORE_ext`) and `max|ΔH_full|` both agree to ~1e-14–1e-15 at K_mean=1/K_pair∈{0,1}.
- **Corner-of-partitioned-block** (origin-ZC, H_EE+H_ER+H_RR computed as three separate BLAS/
  winner-pair calls instead of one monolithic gemm): `max|ΔH_EE|` and `max|ΔH_full|` both agree to
  ~1e-14–1e-15 at K_mean=1/K_pair∈{0,1} — confirming the partitioned reconstruction exactly matches
  the original monolithic dense contraction, i.e. no cross-term was dropped or double-counted at
  the H_ER/H_RR seam.

All of the above additionally confirm real KNITRO inner-solve agreement (feasible `nStatus`,
matching converged `(ζ*, λ*)` between the dense-reference and shared-backend arms), not merely
Hessian-value agreement in isolation — see `docs/full_correctness_log_2026-07-25.txt` for the raw
PASS/FAIL log (40/40 PASS).
