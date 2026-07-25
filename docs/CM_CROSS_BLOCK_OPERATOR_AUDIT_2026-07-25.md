# CM cross-block (H_EC) operator audit — 2026-07-25

Phase B, task §9-11. Independent of the shared H_EE release (Phase A) — this document is an
audit-and-decide pass on flexible CM's `H_EC` cross block, not a dependency of Phase A.

**CORRECTION (2026-07-25 continuation session)**: the original version of this document (prior
session) hedged that the `Stab` share of `build_bin_tables!` was "not isolated or measured finely
enough... unquantified." This continuation session actually instrumented it
(`bench_cm_bintable_decomposition.jl`, §2 below) — the result changes the verdict from
"not_justified" to **"justified_future_task."** `CM_CROSS_BLOCK_FOLLOWUP` below reflects the
corrected verdict; §9-11 of `SHARED_WINNER_PAIR_FINAL_PRODUCTION_GATE_2026-07-25.md` cites this
correction.

## 1. What H_EC actually does today (verified against the current port branch tip)

`hessian_cm_structured!` (`cm_hessian_architectures.jl`) computes `H_EC` entirely from
already-aggregated prefix-sum tables — it never reads a dense `W×m_E` core moment matrix at the
H_EC step itself:

```julia
CS_ = cctx.CScum
Hraw_EC = cctx.Hraw_EC
for l in 1:L
    for (oi, o) in enumerate(origins)
        for j in 1:NCORE
            Hraw_EC[j, oi] = (CS_[o, j, l] - CS_[refIndex1, j, l]) / M
        end
    end
    ...
end
```

This is `O(L·nO·NCORE)` — a read off `CScum` (already prefix-summed), not a `W`-scale loop. The
**only** `W`-scale operation anywhere in the callback is `build_bin_tables!`/
`build_bin_tables_threaded!`, which builds `Ttab` (bin×bin counts, needed for `H_CC`) **and**
`Stab` (bin×core-column sums, needed for `H_EC` via `CScum`) in the same per-draw pass. That pass
reads the dense `E` view (`@view H[:, 2:1+NCORE]`) once per draw — this is the step that touches
core data at `W` scale, not the `H_EC` assembly step itself.

## 2. Measured cost breakdown — CORRECTED, decomposed this session

Real D=20/L=50/W=80,000 measurement (`bench_cm_bintable_decomposition.jl`,
`docs/key_results/cm_bintable_decomposition_2026-07-25.txt`), replaying `Stab`'s and `Ttab`'s
per-draw loop blocks standalone (they are structurally separate blocks within
`build_bin_tables!`'s single `for s in 1:W` loop, not fused — separable without touching
production code):

| Stage | Wall time | Share of `build_bin_tables!` | Share of full Hessian callback |
|---|---|---|---|
| `Stab` (core-CM cross ingredient, `O(W·D·NCORE)`) | 1.716s | **80.7%** | **68.2%** |
| `Ttab` (CM-CM ingredient, `O(W·D²)`) | 0.107s | 5.0% | 4.2% |
| Combined `build_bin_tables!` (production) | 2.125s | — | 84.4% |
| `prefix_sum_tables!` | 0.014s | — | 0.5% |
| Full Hessian callback (dense-reference core) | 2.517s | — | 100% |

`D·NCORE=7,640` vs `D²=400` at D=20/NCORE=382 — an a-priori 19.1:1 per-draw operation-count ratio
that matches the measured ~15:1 wall-time ratio (1.716s vs 0.107s) closely. **`Stab`, not `Ttab`,
dominates `build_bin_tables!`** — the prior session's doc had this backwards (it speculated the
opportunity was bounded by "at most" an unmeasured `Stab` fraction of an 80.2%-of-callback line
"dominated by `Ttab`'s D²-scale counting" — measurement shows the reverse: `Ttab` is only 5% of
that line, `Stab` is 80.7% of it and 68.2% of the ENTIRE Hessian callback).

## 3. Does `E = Q - νπ'` already appear in the cross-block code?

No. `E` is read as a plain dense `E[s,j]` element inside `build_bin_tables!`'s per-draw loop — no
rank-one/winner-scatter shortcut is applied at that step. A related decomposition
(`materialize_dense_factual_structured!`, `structured_moment_build.jl`) already exists and is used
to **build** the dense `E` matrix cheaply for the moments/gradient step (`wrap_moments_with_cm_archB`,
now shared with H_EE via this port's own `core_cf_ref` plumbing) — but that is a different call
site (the moments closure), not the Hessian's own `build_bin_tables!` pass.

## 4. The candidate, and why it was NOT implemented this session (though now justified)

The task's suggested candidate (`H_EC = Q'SC - π(ν'SC)`, "winner-bin cross operator") targets
exactly the `Stab` computation just measured at 68.2% of the full Hessian callback:
`Stab[x,j,k] = Σ_w S[w]·1{bin(U[w,x])=k}·ν[w]·Q̃[w,j] - π[j]·Σ_w S[w]·1{bin(U[w,x])=k}·ν[w]`, where
the first term only needs the SINGLE winning `(w,x)`-pair's column per draw (not all `NCORE`
columns) and the second term is a rank-one correction computable once per bin. This is real,
derivable, and — per §2's correction — the underlying `Stab` pass it would replace is now known to
be MATERIAL (68.2% ≫ the task's own 10% threshold), not negligible.

**Still not implemented this session**, per the task's own explicit instruction ("Do not implement
a large new cross-block backend in this merge task unless it is trivial and independently
committed"): this is not trivial — it requires a new winner-scatter accumulation INTO `Stab`'s
`(D, NCORE, L+1)` bin-indexed layout (different from the winner-pair H_EE kernel's own `(D,Ddest)`-
indexed accumulation target), which needs its own design, implementation, and D=4/D=20 correctness
gates before it could be trusted — a genuinely separate, bounded follow-up task, not a same-session
addition on top of everything else this port already gates.

## 5. Verdict

**`CM_H_EC = retained_bin_prefix` for THIS release** (the shared H_EE port is not blocked on this),
**but `CM_CROSS_BLOCK_FOLLOWUP = justified_future_task`** (corrected from the prior session's
`not_justified` now that `Stab`'s real cost share — 68.2% of the full CM Hessian callback — is
measured, not speculated). A follow-up session should:
1. Design a winner-scatter `Stab` accumulator (bin-indexed target, not destination-indexed like
   the existing H_EE kernel — a genuinely different accumulation shape).
2. Validate it against the current dense-`E`-read `Stab` at D=4 and D=20 to machine/near-machine
   precision, the same discipline this session's H_EE gates used.
3. Re-run this session's `bench_cm_bintable_decomposition.jl` with the new accumulator substituted
   for `stab_only!` to measure the REAL achievable gain (this session did not implement the
   candidate, so no such number exists yet — do not assume a gain proportional to 68.2% without
   measuring the winner-scatter version's own real cost, which is not free either).
4. If validated and a genuine complete-callback/complete-solve gain is confirmed, merge as its own
   commit under its own tag (e.g. `cm-winner-bin-cross-hessian-production-ready-2026-07-25` or a
   later date), per the task's own separate-tag instruction.

**This does not block or gate the Phase A (shared H_EE) release** — it is an independent decision
per the task's own instruction (§Independent Phase B).
