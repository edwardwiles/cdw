# Origin-ZC cross-block (H_ER) audit — 2026-07-25

Phase B, task §12. Audit-only, as instructed ("Do not turn this into a large redesign").

## 1. Pre-port state

Before this port, origin-ZC's entire Hessian (`H_EE`, `H_ER`, `H_RR` combined) was one monolithic
dense BLAS contraction — `cc_algo/PsiObjectiveBundle.jl:598-608`'s generic `hessian!`, called via
`archA_hess_cb_builder` (`cm_hessian_architectures.jl:734`). There was no separate "H_ER step" to
audit; the whole thing was a single `gemm!` over `H_copy[:, 2:1+outer_constr_index]` (core columns
and restriction/η columns combined, undifferentiated).

## 2. Post-Phase-A state (this port, task §4.4)

`archA_partitioned_hess_cb_builder` (`cm_hessian_architectures.jl`, new) now computes:
- `H_EE` via the shared winner-pair backend;
- `H_ER` via `BLAS.gemm!('T','N', 1/M, HC_core, HC_eta, 0.0, HER)` — the SAME dense-BLAS
  computation the pre-port monolithic gemm implicitly did for this sub-block, just isolated into
  its own call, computed once (`H_RE` obtained by `transpose(HER)`, never recomputed independently);
- `H_RR` via the analogous `BLAS.gemm!` restricted to the η columns.

This satisfies task §4.4's explicit instruction ("the existing exact dense method... for H_ER";
"the existing exact dense method for H_RR"; "Compute H_ER once and never calculate H_RE
independently") without introducing any new cross-block algorithm.

## 3. Can the E-side winner structure reduce the H_ER contraction?

In principle, yes — the same `E = Q - νπ'` factorization used for `H_EE` applies equally to
`H_ER = (1/M)E'S R` (`R` = the origin-specific mean/pairwise-ZC restriction columns,
`n_eta = K_mean·D + K_pair·D(D-1)/2` wide, typically small: e.g. `n_eta=4` at D=4/K_mean=1/K_pair=0
in this session's own D=4 gates, and on the order of tens to a couple hundred at D=20 for the
K≤2 configurations this codebase supports). Analogous to `H_EC`:

```
H_ER = (1/M)(Q'SR - π(ν'SR))
```

where `Q'SR` only needs each draw's WINNING core column (not all `NCORE` of them) scattered
against that draw's `R`-column values — an `O(W·n_eta)` winner-scatter accumulation instead of the
current `O(W·NCORE·n_eta)` dense `gemm!`.

## 4. Why not implemented this session

Per the task's own explicit constraint for this section ("Do not turn this into a large redesign.
Implement a candidate only if it can reuse the same winner-scatter primitives cleanly"):

- `R`'s columns (mean/pairwise-ZC target contrasts) are NOT bin-indexed or prefix-summed the way
  CM's `C` columns are — origin-ZC has no `Bidx`/`CScum` structure at all (confirmed: origin-ZC's
  own `wrap_moments_with_originzc` builds `R` directly from `Zraw_all`/`Zpairraw_all` power-level
  matrices, no binning). A winner-scatter `H_ER` would therefore need its OWN new accumulation
  structure (winning-column-keyed sums against each `R` column, or an extension of this port's
  `WinnerPairParallelWorkspace`/`WPGroup` machinery to accept an externally-supplied `R` block) —
  this is a genuine, non-trivial extension of the shared winner-pair primitives, not a
  "clean reuse" of what already exists for `H_EE`/`H_EC`.
- `n_eta` is small in every supported configuration (K_mean/K_pair ≤ 2 per this session's D=4/D=20
  gates), so `H_ER`'s dense `gemm!` is already cheap in absolute terms (`O(W·NCORE·n_eta)` with
  `n_eta` a small constant next to `NCORE`) — the same reasoning that made `H_EE`'s ORIGINAL small
  dense gemm cheap in isolation before this port (the point of the port was `H_EE`'s `O(NCORE²)`
  scaling, which `H_ER`/`H_RR` do not share since one dimension is bounded by `n_eta`, not `NCORE`).

## 5. Verdict

**`ORIGIN_ZC_H_ER = retained_dense`.** The candidate winner-scatter `H_ER` operator is real and
derivable (§3), but implementing it would require new accumulation machinery beyond what this
port's shared winner-pair workspace already provides (§4), for a block whose current dense-BLAS
cost is already small given `n_eta`'s bounded width — exactly the case the task's own "do not turn
this into a large redesign" instruction anticipates. `H_RR` is likewise retained dense
unconditionally (task §12: "Keep H_RR dense unless separate structure proves useful" — no such
structure was found).
