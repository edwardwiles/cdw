# Production ZC/CM cross-Hessian algebra — 2026-07-28

Full derivation of `H_EC`, `H_EZ`, `H_CZ`, `H_ZZ` using `E = Q - νπ'` and `Z = Φ - 1t'`, including
sampling weights, centering, column order, and low-rank corrections, cross-referenced to where each
is actually implemented (file:function). This extends
`ZC_FEATURE_AND_CENTERING_ALGEBRA_2026-07-28.md` (which remains valid and is not superseded — its
own "already implemented, shared, and gated" status note still applies) with the sampling-weight/
column-order/low-rank details that document deliberately deferred, plus this session's own
production-relevant findings (a real bug found and fixed in the `H_EC` construction path, and the
`Zc` lifecycle release).

## Notation (unchanged from the earlier doc, restated for self-containedness)

- `w = 1..W` — draws (the sampling index; `M = W` = draw count, `obj.M`).
- `E = Q - νπ'` — the winner-conditioned economic block, `E[w,j] = ν[w]*(y[w,slot(j)]*1{winner(w,
  slot(j))=origin(j)} - π[j])`. Live representation: `core_exact_hessian.jl::WinnerPairHessCtx`
  (`nu`, `y`, `winner`, `pi_vec`, `cf_raw_scaled`/`has_cf` for the extra "cf"/gravity-common-factor
  column).
- `C` — the CM bin/contrast operator: `cctx.Bidx` (raw `W x D` bin-membership matrix, values in
  `1..L+1`) composed with `cctx.CScum`/`cctx.R` (cumulative-prefix-sum + optional origin-contrast
  transform).
- `Z = Φ - 1t'` — the centered mean/pairwise-ZC restriction feature block. `Φ` = raw,
  theta-INDEPENDENT feature matrix, one `W x D` block per mean level (`ZCRestrictionOperator.
  Zraw_all[k]`, `k=1..K_mean`) concatenated with one `W x npair` block per pair level
  (`Zpairraw_all[k]`, `k=1..K_pair`, `npair = D(D-1)/2`) — column order is **mean blocks first
  (level 1..K_mean, each `D` columns), then pair blocks (level 1..K_pair, each `npair` columns)**,
  matching `build_raw_mean_pair_matrix_levels`'s own construction order exactly (never re-derived
  or re-ordered downstream — `zc_gram_blas_candidates.jl`'s `build_zc_raw_weighted_workspace`
  documents and reuses this same order verbatim). `t` = the current outer point's target vector
  (`ZCRestrictionWorkspace.targets_mean`/`targets_pair`, refreshed via `refresh_zc_targets!`; see
  `ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md` for this release's correction to that function's
  own call-frequency documentation).
- `S = diag(Ψ''(r))` — the current dual point's Hessian weight vector, `obj.arg2` after
  `ddPsi!(obj.arg2, obj.arg0)`, length `W`. Refreshed once per Hessian callback (genuinely
  dual-dynamic — this is the ONE quantity in this whole algebra that cannot be cached across
  callbacks within one inner solve).
- Sampling weights: this D=4/D=20 synthetic/real setup does not carry a separate per-draw
  probability weight distinct from `S` and the uniform `1/M` normalization — every sum below is a
  plain `(1/M) Σ_w` (equivalently `(1/M) X'SY` in matrix form), `S` itself already carrying the
  entire weighting (both the outer Fréchet/CM curvature weight AND, for `has_cf`/economic terms,
  the `ν[w]` power). There is no additional importance-sampling or stratification weight layered on
  top anywhere in this codebase's D=4/D=20 gates — confirmed by reading every accumulator in
  `winner_pair_cross_hessian.jl`/`zc_restriction_operator.jl`, none of which reference a weight
  vector besides `S`/`Snu = S.*ν`/`Snu2 = S.*ν.^2`.

## H_EC = E'SC (flexible-CM / common-Fréchet CM-grid block; the true-economic sub-block of CM+ZC)

```
H_EC[j, o, l] = (1/M) Σ_w S[w] E[w,j] 1{bin(U[w,o]) <= l}     [contrast against refIndex1's own column when cctx.R is set]
```

Substituting `E[w,j] = ν[w]*(y[w,slot(j)]*1{winner(w,slot(j))=origin(j)} - π[j])`:

```
Σ_w S[w] E[w,j] 1{bin<=l}
  = Σ_w (S[w]ν[w]) y[w,slot(j)] 1{winner(w,slot(j))=origin(j)} 1{bin(U[w,o])<=l}   ["QCScum" term]
    - π[j] Σ_w (S[w]ν[w]) 1{bin(U[w,o])<=l}                                        ["NuCScum" term, rank-1 in j]
```

Row-1 (constant-1/ζ-paired moment) and cf/gravity-common-factor columns are not of the
`ν*(y*1{winner}-π)` shape and get dedicated `SOnlyTab`/`QCfTab` accumulators (single-`S`- or
`Snu`-weighted respectively) — same structure as the earlier doc, unchanged.

**Implementation, and a real bug found+fixed this session**: `winner_pair_cross_hessian_fill!`/
`_threaded!` (`winner_pair_cross_hessian.jl`) build the raw `QTab`/`NuTab`/`SOnlyTab`/`QCfTab`
tables into a persistent `WinnerBinCrossScratch` scratch object, obtained per Hessian callback via
`_ensure_cm_cross_scratch!` (`cm_hessian_architectures.jl:952`), which lazily constructs a fresh
`WinnerBinCrossScratch(ncolI, D, L)` the first time it's needed for a given `(ncolI,D,L)`. That
3-argument convenience constructor (added by an earlier phase of this task, alongside the new
`tasks_ec::Vector{Task}`/`EsumEcon::Vector{Float64}` fields for this session's threaded kernel) had
its **last two positional arguments swapped relative to the struct's own declared field order** —
passing `zeros(ncolI)` (a `Vector{Float64}`) into the `tasks_ec::Vector{Task}` slot and
`Vector{Task}(undef, Threads.nthreads())` into the `EsumEcon::Vector{Float64}` slot. Julia's
default memberwise constructor tries to `convert` each argument to its field's declared type, so
this raised `MethodError: Cannot convert Float64 to Task` (via `unsafe_copyto!`) the first time this
constructor was reached with a genuinely fresh `(ncolI,D,L)` — which, inside a KNITRO Hessian
callback, gets caught by KNITRO's own generic `_try_catch_handler` and surfaced only as
`nStatus=-500` (`KN_RC_CALLBACK_ERR`), indistinguishable at the KNITRO-status level from the
unrelated, previously-documented D=20 `archC_verified_state` direct-call issue. **This is a
distinct, genuine D=4 (and, since the buggy code path is `S`-count-independent, presumably D=20
too) construction bug, not a KNITRO/synthetic-data quirk** — reproduced deterministically by calling
`hessian_cm_structured_v2!` directly (bypassing KNITRO entirely) and confirmed via a full,
unswallowed Julia stacktrace pointing straight at the constructor. **Fixed** (swap the last two
positional args to `Vector{Task}(...), zeros(ncolI)`, matching field order) — see the dedicated
commit for the full root-cause writeup and reproduction evidence. Every other struct this task
added (`WinnerZCCrossScratch`, `BinZCrossScratch`, `ZCRawWeightedWorkspace`) was checked for the
same class of mismatch and found correctly ordered.

**Why `flexible_cm`'s own pre-existing D=4 gate never caught this**: that gate
(`test_threaded_cross_hessian_d4.jl`'s `flexible_cm` section) builds its context via
`build_cm_augmented_obj` (`common_marginals_moments.jl`) rather than a compressed-core-enabled
production constructor — that function never populates `core_cf_ref`, so `cf` stays literally
`nothing` throughout, `_fill_cm_HEE!`'s `cf isa CompressedFactual` gate is always false, and the
ENTIRE Hessian fill (H_EE and H_EC alike) takes the dense fallback path — `_ensure_cm_cross_scratch!`
/`WinnerBinCrossScratch` is never reached at all for that gate's exact configuration. This session's
`cm_meanzc` and `common_frechet` sections both use production-grade constructors
(`build_cm_meanzc_bin_ctx` / `build_cm_production_context_v2`) with `use_compressed_core=true` and
`cm_cross_hessian_backend=:winner_bin` defaults, so both genuinely exercise this path (confirmed via
an explicit `typeof(cf)`/`_cm_cross_hessian_wants_winner_bin` diagnostic print added to the new
`common_frechet` gate section) — which is exactly why `cm_meanzc`'s gate caught the bug and
`flexible_cm`'s did not.

Code: `winner_pair_cross_hessian_fill!`/`_fill_threaded!` (`winner_pair_cross_hessian.jl`),
`winner_pair_cross_hessian_cm_block!` (per-`l` slicing wrapper), dispatched from `_fill_cm_HEE!`'s
sibling call site inside `hessian_cm_structured_v2!`/`hessian_cm_structured!`
(`cm_hessian_threaded.jl:223-234`, `cm_hessian_architectures.jl`'s serial twin). Gated by
`_cm_cross_hessian_wants_winner_bin(cctx, cf)` (`cm_hessian_architectures.jl:885`).

## H_EZ = E'SZ (winner-aware economic × mean/pair-ZC cross; shared by CM+ZC and origin-ZC)

```
H_EZ = E'S(Φ - 1t') = E'SΦ - (E'S1)t' = E'SΦ - Esum·t'
```

`Z` (already centered) is passed in directly — `cs.hzz_centered.Zc`, built by
`refresh_zc_centered!` (see the Zc-caching release doc for its lifecycle) — rather than threading
`Φ`/`t`/`Esum` separately; algebraically identical to the expansion above. Substituting `E`'s own
decomposition exactly as for H_EC:

```
HEZ[j+1, x] = (1/M)( Σ_w (S[w]ν[w]) y[w,slot(j)] 1{winner=origin(j)} Z[w,x]  -- winner-conditioned scatter
                       - π[j] Σ_w (S[w]ν[w]) Z[w,x] )                          -- rank-1 pi_vec correction, NuZ[x]
row 1:  HEZ[1,x]      = (1/M) Σ_w S[w] Z[w,x]                                  -- plain gemv, S-only
cf row: HEZ[jcf+1,x]  = (1/M)( Σ_w (S[w]ν[w]) cf_raw_scaled[w] Z[w,x] - π[jcf]·NuZ[x] )
```

No threshold-binning here (`Z`'s columns are continuous features, not step functions of a bin
index) — one pass, `O(W*Ddest*n_x)` for the winner-conditioned scatter (loop order `slot -> x ->
w`, column-contiguous), plus `O(W*n_x)` BLAS `gemv!`s for the row-1/cf rows.

Code: `winner_pair_cross_hessian_zc_block!`/`_threaded!` (`winner_pair_cross_hessian.jl`),
`winner_pair_cross_hessian_zc_prep!` (refreshes `Snu[w]=S[w]ν[w]` once per callback, shared with
H_EC). Called from `_fill_cm_HEE!` for CM+ZC's `HEM` block (`cm_hessian_architectures.jl:824-826`)
and `archA_partitioned_hess_cb_builder` for origin-ZC's `HER` block (`:1593-1597`).

## H_CZ = C'SZ (CM-grid × mean/pair-ZC cross; CM+ZC only)

No winner-selection (`Z`'s columns are per-draw feature values, not a winner-argmin outcome) — only
a bin-membership test against the already-`S`-weighted, already-centered `ZcS[w,j] = S[w]*(Φ[w,j] -
t[j])` (built once per callback by `refresh_zc_centered!`, SAME scratch H_ZZ below reads — not
recomputed independently):

```
ZBinTab[x, j, b]    = Σ_{w: bin(U[w,x])=b} ZcS[w,j]                       [O(W*D*n_z)]
ZBinCScum[x, j, l]  = Σ_{b<=l} ZBinTab[x, j, b]                           [prefix sum, O(D*n_z*L)]
H_CZ[j, o, l]        = (1/M)( ZBinCScum[o,j,l] - ZBinCScum[refIndex1,j,l] )  [per-l, per-origin slice]
```

Code: `bin_zc_cross_hessian_fill!`/`_threaded!`/`_block!` (`winner_pair_cross_hessian.jl`). Called
from `hessian_cm_structured_v2!`/`hessian_cm_structured!`'s own `use_direct_hcz` branch
(`cm_hessian_threaded.jl:235-247`), gated by `_cm_cross_hessian_wants_direct_hcz`
(`cm_hessian_architectures.jl:904`).

## H_ZZ = Z'SZ (mean/pair-ZC self block; shared by CM+ZC and origin-ZC)

`Z'SZ = (Φ-1t')'S(Φ-1t')`, computed directly from the centered scratch (never term-by-term expanded
in the live code — `Zc = Φ - 1t'` is exact, so the direct Gram is algebraically identical to any
term-by-term `Φ'SΦ, Φ'S1, 1'SΦ, 1'S1` expansion plus rank-one corrections):

```
H_ZZ = (1/M) Zc' * ZcS         [BLAS.gemm!('T','N', 1/M, Zc, ZcS, 0.0, HZZ)]
```

Code (`:reference` backend): `zc_restriction_gram!` (`zc_restriction_operator.jl`). Shared verbatim
by CM+ZC's `HMM` (`_fill_cm_HEE!`'s `ncore < NCORE` branch, `:832`) and origin-ZC's `HRR`
(`archA_partitioned_hess_cb_builder`, `:1639`ff). Alternate BLAS/threaded backends
(`:blas_syrk`/`:blas_gemm`/`:threaded_packed`, `zc_gram_blas_candidates.jl`) build their own
row-`S`-weighted copy directly from the immutable `Phi` (`ZCRawWeightedWorkspace.Phi`, the SAME
concatenated-column-order raw matrix `refresh_zc_centered!` uses, built once per campaign) rather
than reading `cs.ZcS`, dispatched via `zc_gram_dispatch!` (`cctx.zc_gram_backend`, opt-in, default
`:reference`) — all four backends validated bit-exact/machine-precision equal at D=4 (this session's
`test_threaded_cross_hessian_d4.jl`, `zc_gram_backend` sweep, `cm_meanzc`/`origin_zc`, all configs).

## `Zc`'s own lifecycle (this session's Section 10 release)

`Zc` (and, when `fill_S=true`, `ZcS`) are built by `refresh_zc_centered!`
(`zc_restriction_operator.jl`), which — before this session — rebuilt `Zc` unconditionally on every
call, even though `Zc` depends only on `op`'s immutable raw features and `ws`'s current
outer-point targets (never on `S`). This session added an opt-in
`cache_across_callbacks`/`ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]` flag (default `false`, unchanged
behavior) that skips the rebuild when the outer point's targets haven't changed since `Zc` was last
built, tracked via a generation counter (`ZCRestrictionWorkspace.gen`, bumped only when the `νfull`
object identity changes — see that release's own doc for why an identity check, not a naive
"was `refresh_zc_targets!` called again" check, is required). `ZcS` (genuinely `S`-dependent) is
unaffected — always refreshed whenever `fill_S=true`. See
`ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md` for the full design and D=4 bit-exact + rebuild-count
validation evidence (`centered-Z rebuilds per Hessian callback = 0` confirmed under the new mode).

## D=4 correctness gates (status as of this session, all four restricted families)

| family | D=4 gate status | notes |
|---|---|---|
| `flexible_cm` | PASS, 6/6, bit-exact (maxdiff=0.0) | Gate's own construction (`build_cm_augmented_obj`) never populates `core_cf_ref`, so it does NOT exercise the `:winner_bin` H_EC path in practice (dense fallback throughout) — a pre-existing gap in the gate's own coverage, not a new regression; documented here for visibility, not fixed (out of this task's stated scope to change the gate's construction pattern for a family already marked PASSING). |
| `common_frechet` | PASS, 6/6, bit-exact (maxdiff=0.0) — **new this session** | Uses `build_cm_production_context_v2`/`CMConfig`, which DOES wire `core_cf_ref` (`use_compressed_core=true` default) and defaults `cm_cross_hessian_backend=:winner_bin` — genuinely exercises the fixed `WinnerBinCrossScratch` path (confirmed via diagnostic print). |
| `cm_meanzc` | PASS, 36/36 across K_mean1_pair0/K1/K2, both threaded-vs-serial and zc_gram_backend sweeps — **fixed this session** | Was FAILING ALL configs (nStatus=-500) before the `WinnerBinCrossScratch` fix above; root cause was a genuine constructor bug, not synthetic-data fragility. |
| `origin_zc` | PASS for K_mean1_pair0/K1 (26/26); K2 (K_mean=2,K_pair=1) genuinely infeasible (nStatus=-300) | Root-caused this session: the JOINT combination of K_mean=2 AND K_pair=1 (14 simultaneous origin-specific scalar targets) is infeasible at this D=4 synthetic dataset — confirmed robust across a wide sweep of perturbed economic points (6 random `x_free0` perturbations) and target scales (0.1x–3x, 7 values), i.e. not a knife-edge/floating-point boundary issue. Isolating the two pieces: K_mean=2 alone (K_pair=0) is FEASIBLE; K_pair=1 alone (K_mean=1, ="K1") is FEASIBLE; only the joint K_mean=2+K_pair=1 combination fails. This is genuine synthetic-data/capacity fragility for an over-parameterized origin-specific restriction set at D=4 (each origin separately targeted, unlike CM+ZC's `SharedByPowerLayout` which broadcasts one scalar target per level across all origins) — not a code bug, not fixable by seed choice (`d4_exact_setup` has no RNG/seed knob at all — the D=4 synthetic dataset is fully deterministic), and out of scope to "fix" (it is a genuine feasibility property of the model/data at this configuration, not a defect). |

Package/session provenance, full commit list, and Dropbox push manifest: see this session's
`provenance.txt` alongside this doc's Dropbox copy.
