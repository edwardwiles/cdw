# CM cross-block (H_EC) operator audit — 2026-07-25

Phase B, task §9-11. Independent of the shared H_EE release (Phase A) — this document is an
audit-and-decide pass on flexible CM's `H_EC` cross block, not a dependency of Phase A.

## 1. What H_EC actually does today (verified against the current port branch tip)

`hessian_cm_structured!` (`cm_hessian_architectures.jl:459-482`) computes `H_EC` entirely from
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
`build_bin_tables_threaded!` (`cm_hessian_architectures.jl:375-397`, `cm_hessian_threaded.jl:20-...`),
which builds `Ttab` (bin×bin counts, needed for `H_CC`) **and** `Stab` (bin×core-column sums,
needed for `H_EC` via `CScum`) in the same per-draw pass. That pass does read the dense `E` view
(`@view H[:, 2:1+NCORE]`) once per draw — this is the step that touches core data at `W` scale,
not the `H_EC` assembly step itself.

## 2. Measured cost breakdown

Production profile at D=20/L=50/W=80,000 (`docs/fullA_archC_hessian_profile_d20_L50_W80000.csv`,
cited in `docs/fullA_common_marginals_production_integration.md:214-223`, dated 2026-07-21 —
confirmed the referenced function names/line structure are still current on this branch tip):

| Stage | Share of Hessian callback |
|---|---|
| `build_bin_tables!` (bin accumulation, reads dense `E`) | 80.2% |
| H_EE (now: shared winner-pair backend; was: small dense gemm) | 19.1% |
| H_EC assembly (`CScum` reads only) | **0.2%** (≈5ms / 3.14s) |
| H_CC assembly | 0.2% |
| prefix sums | 0.6% |

No 2026-07-25 doc/CSV isolates H_EC further — none needed to; the 2026-07-21 breakdown already
shows H_EC assembly is negligible next to bin accumulation.

## 3. Does `E = Q - νπ'` already appear in the cross-block code?

No. `E` is read as a plain dense `E[s,j]` element inside `build_bin_tables!`'s per-draw loop
(`cm_hessian_architectures.jl:380-395`) — no rank-one/winner-scatter shortcut is applied at that
step. A related decomposition (`materialize_dense_factual_structured!`,
`structured_moment_build.jl`) already exists and is used to **build** the dense `E` matrix cheaply
for the moments/gradient step (`wrap_moments_with_cm_archB`, now shared with H_EE via this port's
own `core_cf_ref` plumbing) — but that is a different call site (the moments closure), not the
Hessian's own `build_bin_tables!` pass.

## 4. Derivation attempted, and why it is not worth merging

The task's suggested candidate (`Q'SC - π(ν'SC)`, "winner-bin cross operator") would target the
**bin-accumulation** pass's `Stab` computation, not the H_EC assembly step per se — i.e., it would
replace `build_bin_tables!`'s dense-`E` read (the 80.2% line) with a winner-scatter accumulation
into `Stab`, analytically split into "the winning column's contribution" plus the rank-one
`-π(ν'S·)` correction, mirroring how the winner-pair kernel avoids materializing `E` for `H_EE`.

This is a real, derivable operator (`E[w,j] = ν[w]·(Q̃[w,j] - π[j])`, so
`Stab[x,j,k] = Σ_w S[w]·1{bin(U[w,x])=k}·ν[w]·Q̃[w,j] - π[j]·Σ_w S[w]·1{bin(U[w,x])=k}·ν[w]`, and the
first term only needs the single winning `(w,x)`-pair's column per draw, not all `NCORE` columns).
**It was not implemented**, for a concrete, decisive reason:

- `build_bin_tables!` accumulates **both** `Ttab` (for H_CC, genuinely needs no core data) **and**
  `Stab` (for H_EC, the only piece a winner-scatter rewrite would touch) in the **same** per-draw
  loop, over the **same** bin lookups (`Bidx`). A winner-scatter rewrite of the `Stab` half alone
  would not reduce the loop's `W`-scale iteration count (still one pass over all `W` draws for
  `Ttab`); it would only reduce the per-draw work from `O(D·NCORE)` (all `NCORE` columns) to
  `O(D)` (the winning column only) **for the `Stab` half only**. Given `Stab`'s own share of the
  profiled 80.2% bin-accumulation time was not separately isolated in the 2026-07-21 profile (only
  the combined `Ttab`+`Stab` pass was measured), and H_EC's own assembly is already 0.2%, any gain
  from this rewrite is bounded by, at most, the `Stab`-only fraction of an 80.2% line that itself
  is dominated by `Ttab`'s `D²`-scale bin-pair counting (independent of `NCORE`) — i.e. the
  achievable upside is real but small and unquantified without instrumenting `Stab` separately,
  which was judged not worth the implementation and validation cost this session given H_EC's own
  assembly step (the thing actually named "H_EC" in the task) is already 0.2%.

## 5. Verdict

**`CM_H_EC = retained_bin_prefix`.** The current bin/contingency-table/prefix-sum implementation
is already effectively optimal for the H_EC assembly step itself (0.2% of callback time); the
`E = Q - νπ'` factorization is a real, derivable but SEPARATE opportunity inside
`build_bin_tables!`'s bin-accumulation pass (the 80.2% line), which shares its per-draw loop with
`Ttab` (H_CC's ingredient, not reducible by this factorization) and was not isolated or measured
finely enough this session to justify implementing and validating a change to it. Not merged; no
`CM_WINNER_BIN_CROSS_BENCHMARK_2026-07-25.md` produced (the task's own escape hatch: "if implemented").

**This does not block or gate the Phase A (shared H_EE) release** — it is an independent decision
per the task's own instruction (§Independent Phase B).
