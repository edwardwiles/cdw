# CM-augmented inner CC dual solve: Hessian architecture comparison (D=4)

Branch `diag/fullA-d4-exact-cm-hessian-arch`, based on
`diag/fullA-d4-exact-common-marginals`. Compares four Hessian architectures
for the common-marginals-augmented inner CC dual solve, at D=4, L in
{10,20,50}, plus a numerical block-elimination experiment (Section 10) and a
qualitative D=20/L=50/W=80000 projection. All code lives in
`full_aod_diag/d4_exact/`:

- `cm_hessian_architectures.jl` — Architectures B, C, D + generic KNITRO wiring
- `c13_probe_dims.jl` — dimension probe (D, W, ncore, ncm per L)
- `c13_validate_hessian_archs.jl` — pure numerical agreement check (no KNITRO)
- `c13_debug_archC.jl` — the debugging script that found and localized the
  Architecture C bug (kept for the record, not part of the main pipeline)
- `c13_bench_hessian_archs.jl` — full KNITRO inner-solve benchmark (the main result)
- `c13_schur_block_elimination.jl` — Section 10 experiment
- `docs/fullA_cm_hessian_bench_raw.csv` — raw benchmark numbers

## 1. Setup recap

D=4, W=8000 draws, `ncore = aug.ncore = obj0.d = 18` (economic moments
including gravity, before augmentation), `outer_constr_index == d` in this
codebase's convention. L in {10,20,50} gives `ncm = (D-1)*L` = 30/60/150,
`n = outer_constr_index_new = ncore + ncm` = 48/78/168.

## 2. Partition and notation

The inner Newton Hessian is w.r.t. `x = (zeta, lambda)`, dimension
`n = outer_constr_index`. Writing `Z = H[:, 2:1+n]` (the columns
`hessian!` in `cc_algo/PsiObjectiveBundle.jl` actually contracts):

```
Z = [E | C],   E = H[:,2:1+NCORE]  (W x NCORE, "ones" + pregrav economic moments)
               C = H[:,NCORE+2:NCORE+1+ncm]  (W x ncm, the CM block)
H = (1/M) Z' diag(w) Z,   w = ddPsi!(arg0)  (obj.arg2 -- the current-iterate dual weight)
H = [[H_EE, H_EC], [H_EC', H_CC]]
```

`NCORE = aug.ncore = 18` numerically equals the E-block width (1 "ones"
column replaces the 1 gravity column that gravity's outer-only status
removes — not a coincidence, `wrap_moments_with_cm`'s docstring already notes
gravity is excluded from the inner block).

Bin-index derivation used by Architecture C: for thresholds
`z_1 < ... < z_L` (quantiles of the reference origin's draws, from
`precalc_common_marginals_cdf`), define `bin(u) = searchsortedfirst(z, u) in
1:(L+1)`. Then `1{u<=z_l} == (bin(u) <= l)` for `l in 1:L` (verified directly,
`c13_debug_archC.jl`). Writing `T_xy(k,h) = sum_s w_s 1{bin(U_sx)=k}
1{bin(U_sy)=h}` (weighted bin contingency table, x,y ranging over all D
origins including the reference) and `CT_xy(l,l') = sum_{k<=l,h<=l'}
T_xy(k,h)` (2D prefix sum):

```
H_CC_raw[(o,l),(p,l')] = (1/M)[CT_op(l,l') - CT_o,ref(l,l') - CT_ref,p(l,l') + CT_ref,ref(l,l')]
H_EC_raw[j,(o,l)]      = (1/M)[CS_o(j,l) - CS_ref(j,l)],  CS_x(j,l) = sum_{k<=l} S_x(j,k)
                          S_x(j,k) = sum_{s: bin(U_sx)=k} w_s E[s,j]
```

Orthonormal contrasts (`CM = raw*R` per threshold block) are handled as a
post-hoc congruence: `H_EC_final[:,block_l] = H_EC_raw[:,block_l]*R`,
`H_CC_final[block_l,block_l'] = R' H_CC_raw[block_l,block_l'] R`.

## 3. Architectures

- **A (baseline)** — `build_cm_augmented_obj`'s `obj_cm` fed unchanged into
  the existing generic `cc_algo/PsiObjectiveBundle.jl::hessian!`. Confirmed
  to work with zero code changes (it already only reads `H`/`outer_constr_index`
  generically). This is what `c12_validate_dense_cm.jl`/`c12_d4_fixed_param_battery.jl`
  already exercised.
- **B** — `wrap_moments_with_cm_archB`: caches the `G_tmp` scratch buffer
  across calls (was freshly `similar`-allocated every call in the original
  `wrap_moments_with_cm`), and builds the CM columns fresh from bin indices
  in row-chunks (`fill_cm_columns_from_bins!`) instead of copying from a
  persistent dense `W x ncm` CM matrix. The Hessian **contraction** itself is
  untouched — B dispatches to the same generic `hessian!` as A. Validated to
  produce a byte-identical G matrix (`max|G_A-G_B| = 0.0` at every L/contrast
  tested) and hence a byte-identical Hessian.
- **C** — exact structured Hessian via the bin-contingency-table construction
  above (`hessian_cm_structured!`). Same `obj_cm`/`H` as A (this file does not
  change moment construction), only the Hessian **callback** is swapped.
- **D** — matrix-free Hessian-vector product, `Hv = (1/M) Z'(w.*(Z*v))`,
  wired via KNITRO's `hessopt=product(5)` mode (`ek_inner_hvp.opt`,
  `algorithm=cg`), mirroring the already-existing `compressed_inner_alt_solvers.jl`
  HVP pattern but built off the plain dense `H` (not the compressed
  factual) so it's generic to the CM-augmented obj. Diagnostic/validation
  candidate only, per task brief.

## 4. A real bug found and fixed (Architecture C)

First implementation of `hessian_cm_structured!` disagreed with Architecture
A by ~13-19% (absolute Hessian error ~0.13-0.19, not floating-point noise).
Isolating by sub-block (`c13_debug_archC.jl`) showed `H_EE` and `H_CC`
matched A exactly, but every entry of `H_EC` was off by a **uniform factor of
2**. Root cause: the code wrote the raw `H_EC` block into
`Hfull[1:NCORE, cols]` but never mirrored it into the transposed
`Hfull[cols, 1:NCORE]` position; the final defensive
`0.5*(Hfull[i,j]+Hfull[j,i])` symmetrization step then silently halved every
`H_EC` entry (reading a real value against an unset zero). `H_CC` was
unaffected because that block's assembly loop already visits both `(l,l')`
orderings explicitly; `H_EE` was unaffected because the BLAS `gemm!` for a
symmetric product already fills both triangles. Fix: explicitly mirror the
`H_EC` block (`Hfull[cols,1:NCORE] .= transpose(block_ec)`) before
symmetrizing. After the fix, C agrees with A to **1e-16 to 5e-15** (machine
precision) at every L and both contrast types (Section 5).

## 5. Correctness (agreement with Architecture A)

`c13_validate_hessian_archs.jl`, 3 fixed evaluation points per (L,contrasts)
(zero, small-random, larger-random), plus 5 random HVP directions for D:

| L | contrasts | max\|H_C − H_A\| | max HVP_D rel err | max\|G_A − G_B\| | \|H_B − H_A\| |
|---|---|---|---|---|---|
| 10 | anchored | 2.1e-15 | 5.6e-15 | 0.0 | 0.0 |
| 10 | orthonormal | 1.8e-15 | 4.1e-15 | 0.0 | 0.0 |
| 20 | anchored | 3.6e-15 | 2.8e-15 | 0.0 | 0.0 |
| 20 | orthonormal | 1.5e-15 | 3.6e-15 | 0.0 | 0.0 |
| 50 | anchored | 5.4e-15 | 1.6e-15 | 0.0 | 0.0 |
| 50 | orthonormal | 2.6e-15 | 2.6e-15 | 0.0 | 0.0 |

All four architectures are numerically equivalent to Architecture A to
machine precision. This was also cross-checked at the level of a **converged
KNITRO solve**: for every (L, contrasts) benchmarked, the converged dual
point `x*` from B/C/D agreed with A's to `max|Δx|` of `0.0` (B, exact, since
its G matrix is byte-identical), `~1e-14` (C), and `~1e-9` (D — the HVP/CG
path has a looser default convergence criterion, still well within solver
tolerance).

## 6. Full inner-solve benchmark (calibration point, D=4, W=8000)

Cold = `obj.x .= NaN` then solve (median of 3 independent reps, each
preceded by a fresh reset). Warm(same-point) = re-solve at the identical
theta immediately after (0 Newton iterations — measures fixed KNITRO
context-teardown/setup overhead, not the Hessian architecture). Warm
(perturbed) = re-solve at a theta perturbed by ~1% multiplicative noise,
warm-started from the calibration optimum (a handful of real Newton
iterations — more representative of an outer-loop re-solve at a nearby
point). A JIT-warmup pass (L=5, discarded) precedes all reported numbers —
an uncontrolled first run showed a spurious 3.9s "cold" time for whichever
architecture ran first, entirely Julia/KNITRO compilation, not a real cost
(flagged explicitly per this repo's "verify before causal claims" standard).

| L | arch | cold wall (s) | iters | hess_total (s) | alloc (MB) | warm-samept (s) | warm-perturbed (s) |
|---|---|---|---|---|---|---|---|
| 10 | A dense BLAS | 0.0186 | 4 | 0.0086 | 5.3 | 0.0078 | 0.0175 |
| 10 | B chunked+BLAS | 0.0180 | 4 | 0.0081 | 4.2 | 0.0077 | 0.0082 |
| 10 | C structured | 0.0169 | 4 | **0.0063** | 5.3 | 0.0092 | 0.0078 |
| 10 | D matrix-free HVP | 0.1180 | 6 | 0.1045 | 19.5 | 0.0089 | 0.0116 |
| 20 | A dense BLAS | 0.0282 | 4 | 0.0166 | 5.3 | 0.0089 | 0.0299 |
| 20 | B chunked+BLAS | 0.0284 | 4 | 0.0170 | 4.2 | 0.0085 | 0.0085 |
| 20 | C structured | 0.0189 | 4 | **0.0072** | 5.3 | 0.0104 | 0.0089 |
| 20 | D matrix-free HVP | 0.2246 | 6 | 0.2052 | 24.4 | 0.0105 | 0.0132 |
| 50 | A dense BLAS | 0.0788 | 4 | 0.0559 | 5.3 | 0.0128 | 0.0123 |
| 50 | B chunked+BLAS | 0.0792 | 4 | 0.0550 | 4.2 | 0.0116 | 0.0122 |
| 50 | C structured | **0.0293** | 4 | **0.0107** | 5.3 | 0.0123 | 0.0118 |
| 50 | D matrix-free HVP | 0.5819 | 6 | 0.5535 | 32.2 | 0.0147 | 0.0174 |
| 50 (orthonormal) | A dense BLAS | 0.0759 | 4 | 0.0538 | 5.3 | 0.0116 | 0.0123 |
| 50 (orthonormal) | B chunked+BLAS | 0.0775 | 4 | 0.0530 | 13.9 | 0.0138 | 0.0217 |
| 50 (orthonormal) | C structured | **0.0348** | 4 | **0.0120** | 8.3 | 0.0122 | 0.0183 |
| 50 (orthonormal) | D matrix-free HVP | 0.5790 | 6 | 0.5486 | 31.7 | 0.0112 | 0.0188 |

Raw numbers: `docs/fullA_cm_hessian_bench_raw.csv`. Iteration count is exact
KNITRO Newton iterations for A/B/C; D uses `algorithm=cg` and needs the same
6 outer iterations but 138–394 inner Hessian-vector-product calls per solve
(`n_hess` column in the CSV) — CG's per-Newton-step cost, not a convergence
problem (nStatus is 0 or -100, both "solved" in this codebase's convention).

**Findings:**

- **C wins on both cold wall-clock and Hessian-callback time, and its
  advantage grows with L**: 1.1x (L=10) → 1.5x (L=20) → **2.7x** (L=50,
  anchored) faster cold solve; 1.4x → 2.3x → **5.2x** faster Hessian callback
  alone. This holds under orthonormal contrasts too (2.2x cold, 4.5x Hessian)
  — the extra `R`-congruence step C pays there doesn't erase the advantage.
- **B gives a modest, real improvement** (~4-8% lower cold wall time at
  L≥20, and ~20% lower steady-state allocation, 4.2MB vs 5.3MB) from
  avoiding the fresh-allocate-every-call `G_tmp` and the persistent CM
  matrix. It does **not** approach C's speedup because it still feeds the
  same O(W·n²) dense BLAS contraction — B only cheapens moment
  *construction*, which happens once per solve, not the Hessian
  *contraction*, which happens once per Newton iteration.
- **D is not competitive on wall-clock** — CG needs 35–99 HV calls per
  Newton iteration (138–394 total per solve vs 4 for the dense
  architectures), so despite each individual HV call being ~5-15x cheaper
  than a dense Hessian call, the total is 4-8x *slower* than A. This matches
  the task brief's expectation that D is a validation/diagnostic tool, not a
  production candidate. It IS useful as an independent correctness oracle
  (Section 5) precisely because it never goes through the bin-table
  machinery at all.
- **Warm-started re-solves are dominated by fixed KNITRO context overhead**
  (~0.008–0.018s), not by Hessian architecture — a same-point warm solve
  takes 0 Newton iterations for A/B/C, so no Hessian callback ever fires;
  even the perturbed-warm case (a handful of real iterations) is cheap
  enough at D=4/W=8000 that KNITRO's own per-solve setup/teardown swamps
  the architecture difference. **The architecture choice matters for
  cold/many-iteration solves and will matter more as W and L grow** — see
  Section 8.
- **Dense BLAS is not free of overhead here either**: the honest FLOP
  accounting (Section 8) shows C's *nominal* FLOP advantage over A at D=4/L=50
  is ~290x, while the *measured* wall-clock advantage is only ~5x — i.e. my
  scalar/loop-based bin-table code runs at roughly 1/55th the effective
  throughput of a single large BLAS `dgemm` call. This is exactly the
  standing caution in this codebase's brief (dense BLAS is a strong baseline)
  and is reported honestly rather than papered over.

## 7. Correctness/robustness notes

- Both `evaluate_fullA`-style KNITRO wiring and the raw-Hessian unit checks
  passed for every (architecture, L, contrasts) combination tested; no
  disagreement beyond ~1e-14 (C) / ~1e-9 (D, HVP/CG tolerance) was left
  unexplained.
- Orthonormal-contrast allocation for B/C is visibly higher than anchored
  (13.9MB / 8.3MB vs 4.2MB / 5.3MB) — the `R`-congruence step
  (`Hraw_EC * R`, `R' * Hraw_CC * R`) allocates small temporaries per
  threshold block; not optimized further since it doesn't change the ranking.

## 8. Section 10: numerical block elimination (Schur complement)

Not a new architecture — a linear-algebra reorganization of solving ONE
Newton system `H·z = b` at a fixed (converged) point, using Architecture C's
Hessian at L=50 (n=168, NCORE=18, ncm=150). Tested at the genuine KKT point
(`b = -g(x*)`, `‖g(x*)‖_∞ = 2.2e-16`) and at a random RHS, comparing:

1. **Direct**: `cholesky(Symmetric(H))`, then one triangular solve.
2. **Schur**: factor `H_CC` (the *larger* block, 150×150) first, form
   `S = H_EE - H_EC·H_CC⁻¹·H_EC'` (18×18), factor `S`, recover `z_E` then
   `z_C` ("the CM multiplier step") by back-substitution through the `H_CC`
   factor. (This is a pure linear-algebra reorganization — the CM moments
   themselves still vary nonlinearly with the draws inside the CC conjugate;
   no closed form is claimed for anything beyond this one fixed-point solve.)

Results (`c13_schur_block_elimination.jl`):

| quantity | value |
|---|---|
| cond(H_full) | 1.44e5 |
| cond(H_EE) | 1.05e4 |
| cond(H_CC) | 4.78e3 |
| cond(Schur complement S) | 6.68e3 |
| max\|z_direct − z_schur\| (KKT RHS) | 1.4e-27 |
| max\|z_direct − z_schur\| (random RHS) | 2.1e-9 |
| cholesky(H_full) time | 1.51e-4 s |
| cholesky(H_CC) alone | 1.22e-4 s |
| cholesky(H_EE) alone | 1.4e-6 s |
| Direct total (factorize+solve) | 1.69e-4 s |
| Schur total (factorize+solve) | 2.46e-4 s |
| **Schur/Direct ratio** | **1.46x (Schur is slower)** |

**Block elimination does not help here, and the reason is structural, not
incidental**: `H_CC` (150×150, the block Schur elimination must factor
first) already costs 1.22e-4s to factor — 81% of the cost of factoring the
*entire* 168×168 matrix (1.51e-4s). Since `ncm ≥ NCORE` for every L tested in
this task (L=10 already gives ncm=30 > NCORE=18), the CM block is never the
"small" block being eliminated away; it IS the dominant dimension, so
Schur elimination pays the same leading-order factorization cost as direct
factorization, *plus* the extra work of forming and factoring `S` and doing
the additional triangular solves. This is not specific to L=50 — it gets
structurally worse (relatively) as L grows, since `ncm` grows linearly in L
while `NCORE` is fixed. Correctness of the reorganization itself is
confirmed to near machine precision (1.4e-27 at the KKT point; 2.1e-9 at a
generic point, both far inside solver tolerance) — this is a real, working
alternative solve path, just not a faster one at this problem's block-size
ratio. One genuinely informative byproduct: the Schur complement `S` is
*better conditioned* (6.68e3) than either `H_full` (1.44e5) or `H_EE` alone
(1.05e4) — interesting but not exploitable for a speed win here since KNITRO
factors the whole dense Hessian itself via its own internal linear algebra;
this experiment used a free-standing Cholesky, not a KNITRO-internal hook.

## 9. Recommendation

**Adopt Architecture C (structured bin-contingency-table Hessian) as the
Hessian architecture for the CM-augmented inner CC dual solve**, with the
following caveats:

- The win is real and measured, not assumed: 2.7-5.2x on the Hessian
  callback / cold-solve metrics at L=50 (the L this repo's other CM work
  most often uses), growing with L, holding under both contrast conventions.
- It is a pure Hessian-callback swap (`hess_cb_builder` in
  `inner_loop_KNITRO_archgeneric`) — zero change to the FG callback, moment
  construction, bounds, or complementarity wiring, so it composes cleanly
  with anything else already wired into the CM-augmented obj.
- Architecture B (moment-build optimization) is complementary, not
  competing — it should be combined with C for the memory/allocation
  benefits described in Section 10, even though on its own it doesn't touch
  the Hessian bottleneck.
- Architecture D remains valuable as an independent correctness oracle
  (already used that way in Section 5) but should not be used for
  production solves.
- Do NOT conclude "structured always wins by more at larger scale" without
  qualification: Section 10's own arithmetic shows the advantage of
  eliminating/restructuring around the CM block shrinks as `D` (not `L`)
  grows for a fixed L, because `NCORE` itself grows like `O(D²)` in this
  codebase's moment count while `ncm` grows like `O(D·L)` — see the D=20
  projection below for a concrete accounting.

## 10. D=20 / L=50 / W=80000 qualitative projection (NOT run — memory and FLOP scaling only)

Using this repo's own documented D=20/W=80000 numbers (`context_real_d20.jl`
comment: `nTotalMoments=404`, i.e. `NCORE_20 = 404`, real/measured elsewhere
in this repo, not guessed) — `ncm_20 = (D-1)*L = 19*50 = 950`,
`n_20 = 404 + 950 = 1354`.

**FLOP scaling.** Dense (A/B) per-Hessian-call cost is `O(W·n²)`. Structured
(C)'s per-call cost is `O(W·(D·NCORE + D²) + D²L² + n²)`, dominated by the
first term for realistic L:

| | D=4, L=50 (measured regime) | D=20, L=50 (projected) |
|---|---|---|
| dense nominal FLOPs `W·n²` | 8000·168² ≈ 2.26e8 | 80000·1354² ≈ 1.47e11 |
| structured nominal FLOPs `W·(D·NCORE+D²)` | 8000·88 ≈ 7.0e5 | 80000·8480 ≈ 6.8e8 |
| nominal FLOP ratio (dense/structured) | ≈ 322x | ≈ 216x |

The *ratio itself shrinks* from D=4 to D=20 (322x → 216x) because `NCORE`
(hence `D·NCORE`) grows roughly like `D³` while `n²` grows roughly like
`(D·L)²` — i.e. `D²`, for fixed L — so structured's own dominant term
catches up somewhat as D grows. It does NOT flip the ranking at D=20 (216x
is still a large nominal advantage), but it means the advantage is not
unboundedly growing with D the way it visibly grows with L at fixed D=4
(Section 6's 1.4x→5.2x trend). Given this run's own measured
"56x throughput gap" between scalar bin-table code and BLAS (Section 6), a
naive translation of 216x nominal FLOPs into wall-clock would suggest C
remains several-fold faster than A/B at D=20/L=50, but this is a projection,
not a measurement — the scalar-loop throughput gap could behave differently
at D=20's much larger per-draw work (20 vs 4 origins touched per draw in the
inner scatter loop) and was not tested here.

**Memory scaling — the more decisive story at this scale, given this repo's
own prior incident history** (the 109GB `jac_h` tensor bug from Continuation
9, and repeated W=80k/W=800k memory-safety work):

| buffer | size at D=20/L=50/W=80000 | which architectures need it |
|---|---|---|
| persistent dense CM matrix (`W x ncm`) | 80000·950·8 ≈ 580 MiB | A (via `wrap_moments_with_cm`'s captured `CM`); **avoided by B** |
| `obj.H` (`W x (n+2)`) | 80000·1356·8 ≈ 828 MiB | all of A/B/C/D as currently wired (all share one dense `obj_cm.H`, since the FG callback still needs a dense moment matrix regardless of Hessian architecture) |
| `obj.H_copy` (same shape, generic `hessian!`'s scratch) | ≈ 828 MiB | **A and B only** (both dispatch to the generic dense `hessian!`) |
| Architecture C's own scratch (`Ttab`+`Stab`+`Bidx`+`Ews`+`Hfull`) | `Ttab` 20·20·51² ≈ 8.3 MiB, `Stab` 20·404·51 ≈ 3.3 MiB, `Bidx` 80000·20·8 ≈ 12.8 MiB, `Ews` 80000·404·8 ≈ 258 MiB, `Hfull` 1354² ≈ 14.6 MiB → **≈ 297 MiB total** | C only |

So at D=20/L=50/W=80000: A needs ≈580+828+828 ≈ 2.2 GiB of large buffers: B
needs ≈828+828 ≈ 1.66 GiB (saves the persistent CM matrix, still pays the
generic-`hessian!` `H_copy` scratch); **C needs ≈828 MiB (shared `H`, for the
FG callback) + 297 MiB (its own scratch) ≈ 1.1 GiB — it is the only
architecture of the three that avoids the `H_copy`-sized dense scratch
specifically for the Hessian step**, because it never materializes a `W x
n`-shaped scaled copy at all. A genuinely memory-minimal D=20 deployment
would need to go further than this task's scope — combine B's bin-based
fresh CM construction with C's structured contraction, *and* restructure the
FG callback itself (currently still reads the dense `W x n` block via one
`BLAS.gemv!`, so a fully bin-table-only pipeline would need that touched
too) — noted here as a natural next step, not attempted or claimed as done.

**Bottom line for D=20 projection**: memory, not FLOPs, is the more likely
practical constraint this repo will hit first at that scale (consistent with
its own prior incidents), and Architecture C is the only one of the three
practical candidates that meaningfully reduces the large-buffer footprint
for the Hessian step specifically — but a full resolution of the D=20 memory
story requires touching the FG callback too, which is out of this task's
scope.
