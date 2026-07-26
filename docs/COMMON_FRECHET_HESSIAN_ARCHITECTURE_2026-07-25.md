# Common-Fréchet Hessian architecture — 2026-07-25/26 (Part III)

See `FRECHET_AS_CM_PLUS_LEVEL_MATHEMATICS_2026-07-25.md` for the basis derivation this document
assumes. This document covers the Architecture-C (winner-pair-backed) Hessian extension:
`cm_frechet_hessian.jl`.

## 1. Design: reuse, don't rebuild, the bin-contingency tables

`CMBinHessCtx` (`cm_hessian_architectures.jl`, unchanged) already builds, per Hessian callback:
- `Ttab`/`CT`: `D × D × (L+1) × (L+1)` weighted bin-contingency table (prefix-summed to `CT`,
  `1:L` range), covering **all D origins** — needed by CM's own `(o,ref)`-difference formula,
  which reads `CT[o,p,l,l'] − CT[o,ref,l,l'] − CT[ref,p,l,l'] + CT[ref,ref,l,l']`.
- `Stab`/`CScum`: `D × NCORE × (L+1)` weighted economic-column table (prefix-summed), also over all
  D origins.

The level restriction is a **sum** over all D origins with weight `u = 1/√D` (not a difference), so
its Hessian contribution is a *different linear combination of the same tables* — no new `O(W)`
pass. Derivation (for two origin-coefficient vectors `α,β ∈ ℝ^D` at thresholds `l,l'`, the cross
Hessian term is `(1/M)Σ_{o,p} α_o β_p CT[o,p,l,l']`; CM's own `α/β` are `e_o − e_ref`, level's is
always `u`):

```
H_E,level[j,l]       = (1/√D)(1/M) Σ_o CScum[o,j,l]
H_CM,level[(o,l),l']  = (1/√D)(1/M) [Σ_p CT[o,p,l,l'] − Σ_p CT[ref,p,l,l']]
H_level,level[l,l']   = (1/D)(1/M)  Σ_{o,p} CT[o,p,l,l']
```

`H_EE` (the economic/core block) is completely unaffected — `_fill_cm_HEE!` (the shared
winner-pair backend) is called **unchanged**, and empirically confirmed byte-identical between the
CM and common-Fréchet paths (see gate results below).

## 2. The bug this design surfaced (and the general lesson)

CM's own raw restriction features have **zero target** by construction (`1{U_o≤z_l} −
1{U_ref≤z_l}`, no separate constant subtracted). The level feature does **not**:
`level_l(ω) = u'f_l(ω) − target_l`, `target_l = √D·p_l ≠ 0`. An additive per-draw *constant* in a
moment column changes the `E'diag(w)E`-type quadratic Hessian form (unlike the gradient/FG side,
where it only produces a harmless constant shift of `arg0`). The first implementation pass
computed only the `Σ α_o β_p CT[...]` term above and omitted three correction terms coming from
expanding `(A(s) − target_l)(B(s) − target_l')`:

```
H_E,level      -=  target_l · Esum[j] / M            (Esum[j] = Σ_s w_s E[s,j], one BLAS gemv)
H_CM,level     -=  target_l' · (T1[o,l] − T1[ref,l]) / M
H_level,level  -=  target_l'·(1/√D)ΣT1[·,l]/M + target_l·(1/√D)ΣT1[·,l']/M − target_l·target_l'·Wtot/M
```

`T1[x,l] = Σ_s w_s·1{bin(s,x)≤l}` is a new, cheap `O(W·D)` marginal weighted-count table (much
cheaper than `build_bin_tables!`'s own `O(W·(D·NCORE+D²))`).

**Symptom before the fix**: max block discrepancy vs the dense reference was `3.24` (vs the whole
Hessian's own `max|H|=1.0`) and the real structured KNITRO inner solve failed to converge
(`nStatus=-400`, iteration-limited) — a real, consequential bug, not cosmetic; fixing it also fixed
the convergence failure (structured solve then converged `nStatus=0` in 4 Hessian calls).

## 3. Validation

D=4 (`test_frechet_hessian_structured_vs_dense_d4.jl`, 20/20 PASS, both contrast modes): structured
vs dense Hessian at a shared `(θ,x)` point agree to `~1e-14`/`~1e-15` across every block (`H_EE`,
`H_EC`, `H_CC`, and the three new level blocks).

D=20/W=80,000 (`test_frechet_d20_gates.jl` Part 2): real KNITRO inner solves feasible for both
families; Hessian-callback allocation comparison **initially reported as flexible=52.8MB vs
Frechet=9.4MB (−82%)** — this was a measurement-order artifact (flexible CM's callback happened to
be measured first in that script, eating one-time JIT/compilation cost; Frechet's, measured
second, benefited from already-warmed shared machinery). Re-isolated with a dedicated diagnostic
(`diag_hessian_alloc.jl`, repeated calls on pinned state): once warmed, the two allocate essentially
identically — `9.38MB` vs `9.39MB` with an unchanged compressed factual, `29.499MB` vs `29.508MB`
(a `0.03%` difference) with a fresh compressed factual every call (the realistic pattern during a
real outer solve). **Corrected finding: no measurable per-callback efficiency gap between the two
Hessian paths; the level block costs only its proportionate share.**

## 4. Scope / not done

- Serial only. The threaded bin-table variant (`hessian_cm_structured_v2!`, used in production for
  plain CM) is **not** extended for the level block — `archC_frechet_hess_cb_builder` always uses
  the serial path, correct-but-slower rather than wrong. Disclosed follow-up.
- `CM_CROSS_BLOCK_FOLLOWUP` (Stab dominating ~68% of the CM Hessian callback per prior audits) is
  pre-existing, shared with plain CM, not introduced or worsened by this port.

## Verdict

`FRECHET_HESSIAN = validated_serial_architecture_c` — D=4 exact, D=20 real-scale confirmed feasible
and allocation-neutral. Threaded variant not implemented (disclosed gap, not a correctness issue).
