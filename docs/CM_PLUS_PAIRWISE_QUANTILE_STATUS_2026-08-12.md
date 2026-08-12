# CM + pairwise-quantile (family #7): FG complete and gated, Hessian cross block complete, assembler outstanding

Session of 2026-08-12, continuing from `docs/CM_PLUS_PAIRWISE_QUANTILE_HANDOVER_2026-08-12.md`.
Worktree `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`, branch
`feature/pq-free-mass-reparam-2026-08-10`, commits `851cd78` → `3be1b30` → `6754df6`.
**Not pushed to any remote.**

Work order was the one the user set: FG right (mathematically, then in code), then the Hessian,
nothing downstream until both are gated. Target shape is a family that slots into the five-family
reproducible production campaign.

---

## 1. Status at a glance

| Piece | State | Gate |
|---|---|---|
| Moment rows, shared-μ kernels, redundancy claim | **done** | D=4 dense oracle, 116/116 |
| Composed inner FG (E + CM lookup + shared-μ PQ) | **done** | real D=4, 54/54, incl. per-coordinate FD |
| H_RR, H_E,R (by exact reuse of PQ's machinery) | **done** | dense Gram reference, ≤ 7.7e-15 |
| H_R,CM (the genuinely new block) | **done** | dense Gram reference, ≤ 3.5e-15 |
| H_CM,CM, H_E,CM (CM's own validated functions) | route identified, **not wired** | — |
| **Packed Hessian assembler + callback registration** | **NOT BUILT** | — |
| Outer gradient, checkpoint, campaign wiring | not started | — |

**The family cannot yet run with `hessopt=exact`.** That is the next task and it is the only thing
between here and a converging production inner solve — see §5.

---

## 2. What the family is

CM (CDW eq.35, optionally +eq.36) pins the origins' marginals *to each other* but says nothing about
the **level** of the reference marginal. This family adds `L-1` free outer parameters `mu_a` — the
binned masses of the reference marginal on the pairwise-quantile grid — and imposes

* `L-1` **level rows**: `1{b_ref = a} - mu_a`
* the **pair rows** keyed on the SHARED `mu`: `1{b_o=a, b_p=b} - mu_a*mu_b`, all `C(D,2)` pairs

and **drops** the standalone family's per-origin marginal rows `1{b_o=a} - mu_{o,a}`, which under CM
plus the level rows are implied.

The win is in the OUTER dimension: `D*(L-1) = 80` free mass coordinates collapse to `L-1 = 4` at
D=20, L=5. The inner solve gets *more* expensive, not less — CM's rows stack on top of the pair rows.

New files, all additive:
`cm_pairwise_quantile_config.jl`, `cm_pairwise_quantile_moments.jl`,
`cm_pairwise_quantile_lookup_kernels.jl`, `cm_pairwise_quantile_production.jl`,
`cm_pairwise_quantile_hessian.jl`, plus `test_cm_pairwise_quantile_d4_dense_oracle.jl`,
`test_cm_pairwise_quantile_real_d4_fg.jl`, `ek_inner_cmpq.opt`, `ek_inner_cmpq_fgonly.opt`.

---

## 3. Two corrections to the handover, both found numerically before writing code

### 3.1 The grid-superset condition is not "L divides 50" on CM's default grid

The redundancy of the dropped per-origin marginal rows rests on each PQ bin being a **union of CM
grid cells**. The handover stated the condition as "CM's grid is 50 equal-probability points, so this
holds iff L divides 50".

`precalc_common_marginals_cdf`'s **default** grid is `range(1/G,(G-1)/G,length=G)`, which at G=50 runs
`0.02, 0.039592, 0.059184, …, 0.98` — spacing `0.0195918`, **not** `0.02`. It is not the `k/G` grid
(that function's own docstring flags this). Measured: on the default grid the required levels are
present for **no** value of L — not L=5, not even L=2. On an explicit `k/50` grid all of
L ∈ {2,5,10,25} pass.

The user's intent was the `k/G` grid, so this family passes it explicitly through
`precalc_common_marginals_cdf`'s already-existing `probs=` kwarg (`cm_pq_probs_grid(G)`), and only
then is the condition `L | G`. `resolve_cm_pairwise_quantile_config` hard-errors on an L that does not
divide G, with a message saying what breaks.

**It is 49 levels at G=50, not 50.** `p = 1` must be dropped: its CDF contrast
`1{U_o≤∞} − 1{U_ref≤∞}` is identically zero, i.e. a structurally zero moment column and a singular
KKT. Keeping 50 levels via `k/51` would force `L | 51`, i.e. L ∈ {3,17}. **No change to CM is
required**, and passing `probs` explicitly also insulates the family from any later change to CM's
default. (Changing that default would move every existing CM run's cutoffs — a scientific change to
CM, not a refactor.)

### 3.2 The PQ cutoffs must be *selected from* CM's own threshold array, not recomputed

The PQ cutoff at z-level `r/L` and the CM threshold at U-level `1 − r/L` are the same number in exact
arithmetic. They are **not** the same Float64 by the two routes: `1 - (1 - 0.2) = 0.19999999999999996`
(measured). A draw landing between the two values would be binned inconsistently by CM and PQ,
silently breaking the superset identity for that draw.

So `cm_pq_u_cutoffs_from_cm_grid` **indexes** CM's own computed `z` array (checked with `===`), and
the PQ bin assignment is *also* derivable from CM's own bin index by a pure integer map
(`cm_pq_bin_map`) with no float comparison at all. `assert_cm_pq_bin_consistency` cross-checks the two
routes for every `(draw, origin)` cell and hard-errors on any disagreement — 16,000–32,000 cells per
gate case, all agreeing.

---

## 4. What is gated, with numbers

### 4.1 The redundancy claim — the one the whole family rests on (D=4/D=5, four configurations)

* `[CM | level | pair]` has **full column rank** (127/394/789/207 of 127/394/789/207).
* Every **dropped** per-origin marginal column lies in `span([CM | level])`: max relative
  least-squares residual **9.7e-15**.
* Re-adding those columns makes the stack rank deficient by **exactly `D*(L-1)`** (16/16/36/5) —
  `(D-1)*(L-1)` implied non-reference rows plus `(L-1)` that duplicate the level rows outright. The
  handover's trap #1 ("singular KKT, not a tolerance problem") reproduced on demand.

### 4.2 FG

* Forward/transpose vs naive dense indicator references: ≤ **5.7e-15**.
* CM's own production lookup kernels vs dense CM columns, eq.35 and eq.36: ≤ **3.8e-15**.
* Real D=4 context (`d4_exact_setup`, real `CompressedFactual`, real `KN_solve`): FD of `f`
  **coordinate by coordinate** over every block — ζ, λ_E, level, pair, CM — at ≤ **8.8e-10**.
  This is the gate that catches an offset/layout error; it is also the only feasible gate on the
  economic block, since the family is operator-native and has no dense path to compare against.
* Level-row and pair-row gradients vs dense indicator references rebuilt in the real context:
  ≤ **2.7e-15**.

### 4.3 The two-slot shared-μ term

`d(mu_a*mu_b)/d(mu_c) = δ_ac·mu_b + δ_bc·mu_a`. Gated at non-uniform μ vs FD (≤ 9.9e-10), with the
negative controls the handover asks for: both natural wrong transcriptions fail (rel 0.52, 0.67), and
the index-slip variant is shown to **pass** at uniform μ (1.3e-9) — the measured demonstration that a
uniform-μ gate would have been fooled. At L=2 the control is reported **SKIP**, not PASS: one free bin
means `mu[a]` and `mu[b]` are the same coordinate, so it cannot fire.

### 4.4 Hessian

`r` is **linear** in the inner variables, so the exact inner Hessian is the weighted Gram matrix
`H = (1/W)·M'diag(h)M` with `M = [1 | E | G_R | G_CM]`, `h = Ψ''(r)`. No second-order term exists.
That is both why every block is a contingency-table contraction and how the oracle gets a dense
reference with no derivative approximation.

* **H_RR / H_E,R need no new code, as an identity.** This family's rows *are* the standalone PQ
  family's rows at a μ common across origins, so replicating the shared μ into every origin row
  (`cmpq_replicate_shared_mu!`) makes PQ's gated machinery compute this family's blocks as a superset;
  `cmpq_to_pq_row` selects the sub-block. Measured vs dense: max abs **4.3e-16 – 7.7e-15** on a block
  of scale 0.21–0.57. Cost of the unread rows: ~5% at D=20/L=5 (3120 PQ rows vs 3044), paid
  deliberately to reuse code carrying 29 oracle checks, the threaded T3/T4 scatter, the canonical
  dedup and the `_lo_write!` convention.
* **H_R,CM — the new block.** Mixed resolution: CM's fine G-cell grid on one axis, PQ's L-bin grid on
  the other one or two. Two tables (`Y` level, `X` pair) accumulated at CM-cell resolution and
  prefix-summed once over that axis. Measured vs dense: **≤ 3.5e-15**, split by row family and CM
  feature family (level×CM 3.5e-15, pair×CM 1.6e-15, eq.35 1.7e-15, eq.36 1.4e-15). A negative control
  reading the CM axis one grid level off disagrees by 5.2e-2 – 5.6e-1, so the check is genuinely
  testing the CM axis.
* `X` ranges over **all** `x ∈ 1..D`, including `x` inside the pair, on purpose: restricting it and
  special-casing the overlaps against CM's own 2-way table saves ~10% of the build and buys a
  three-branch read path, which is where an index error hides.

### 4.5 Cost, in operation counts rather than wall-clock

Nothing here is a timing claim: the box was at load 55 all session with an unrelated campaign running
(`pq_L5_d1_long`), and per the standing rule a single-run wall-clock comparison on it is worthless.

* `X` build: `O(W·npair·D)` = **380M** increments at D=20/W=100k — same order as PQ's own T4 scatter
  (484M), and the dominant new Hessian cost. Threaded over `pidx`, which owns a disjoint `X` slice, so
  no reduction and no per-thread copy.
* `X` storage: 24 MB at L=5, 123 MB at L=10; doubled for two CM families.
* Inner row count at D=20, L=5, G=50, two families: `4 + 3040 + 1862 = 4906` restriction+CM rows
  versus the standalone family's `3120`. The Hessian factorization is `O(n^3)`, so expect roughly
  `(4906/3120)^3 ≈ 3.9x` per-callback KKT cost against standalone PQ. This is the inner cost the
  handover flagged; the payoff is the outer collapse, 80 → 4.

---

## 5. Next task, precisely: the packed assembler

Everything needed exists; what is missing is the routine that writes the six blocks into KNITRO's
packed upper triangle for the layout
`x = [ζ; λ_E(ncore1); λ_L(L-1); λ_P((L-1)²·npair); λ_CM(ncm)]`, plus its registration.

Model it on `pairwisequantile_hess_cb_builder` (`pairwise_quantile_production.jl`), which already
solves the same problem for three blocks and carries three measured performance fixes worth keeping:
read `HRR[j,i]` (column walk, the populated lower triangle), thread rows via the closed-form packed
offset `k0(i) = (i-1)*n - (i-1)*(i-2)/2`, and **hoist `evalResult.hess` once with a type assertion**
— that last one was the whole story at D=20/L=10 (86.16 s → measured cost was dynamic dispatch, ~660
ns/entry, not memory).

Block sources:

| Block | Source | Note |
|---|---|---|
| H_EE | `winner_pair_hessian!` + `WinnerPairHessCtx` | verbatim, as PQ's builder does |
| H_E,R | `pairwise_quantile_cross_hessian_block!` at replicated μ, then `extract_cmpq_HER!` | done, gated |
| H_RR | PQ raw fill + centering at replicated μ, then `extract_cmpq_HRR!` | done, gated |
| H_R,CM | `build_cmpq_cross_hess_tables!` + `fill_cmpq_cm_cross_block!` | done, gated |
| H_CM,CM | CM's `build_bin_tables!(…; fill_S=false)` → `prefix_sum_tables!` → `fill_cm_HCC!` | **to wire** |
| H_E,CM | `winner_pair_cross_hessian_cm_block!` per threshold block | **to wire** |

For the two CM blocks, build a `CMBinHessCtx` through `build_cm_bin_ctx(ctx, aug)` with a synthesized
`aug` NamedTuple (`L=Lcm`, `origins`, `refIndex1`, `z=z_cm`, `ncore=ncore_econ`, `ncm`, `contrasts`,
`core_cf_ref`, `n_families`) — the adapter pattern this repo already uses for the profiled-restricted
production bridge.

Specifics already traced, so the next session does not have to re-read
`cm_hessian_architectures.jl` (2237 lines) to find them:

* **H_CM,CM.** `build_bin_tables!(cctx, nothing, h; fill_S=false)` → `prefix_sum_tables!(cctx;
  fill_S=false)` → `fill_cm_HCC!(cctx.Hfull, cctx, W)`. Passing `H=nothing` is safe *only* with
  `fill_S=false` (that is the branch that never touches the dense economic columns; the function
  hard-errors otherwise). `fill_cm_HCC!` writes at CM's own offsets
  `rows = NCORE + (l-1)*nO+1 : NCORE + l*nO` (and `+ncm_cdf` for the eq.36 sub-block), and it fills
  **both** triangles. So read the `ncm×ncm` corner `cctx.Hfull[NCORE+1:NCORE+ncm, NCORE+1:NCORE+ncm]`
  out of it — do not try to redirect its offsets.
* **H_E,CM.** `winner_pair_cross_hessian_cm_block!(Hraw_EC, wctx, ws, l, origins, refIndex1, M;
  Hraw_EC_pow=…)` is **per threshold block** `l` (it fills `NCORE×nO`, not the whole block), and its
  `ws` is a `WinnerBinCrossScratch` that must first be populated by
  `winner_pair_cross_hessian_fill!` — it reads `ws.QCScum/NuCScum/SOnlyCScum/QCfCScum`. That
  prerequisite is the one remaining unread layer; budget for it. After each `l`, apply the contrast
  on the right (`Hraw_EC * R`, or `Hraw_EC` directly when `R === nothing`) exactly as
  `hessian_cm_structured!` does, and write into columns `(l-1)*nO+1 : l*nO` of this family's CM
  column range. Note `use_direct_hcz`/`ncore_core` splitting in that call site is CM+ZC's widened-
  layout concern and does **not** apply here (`ncore_core == NCORE` for this family).

**Free strong gate available:** H_EE is computed by *both* families' stacks. Assemble it from each and
require agreement — two independent routes to the same block, at no extra derivation.

Then, in order: D=4 dense oracle on the **full packed vector** vs `(1/W)M'diag(h)M` (extend
`test_cm_pairwise_quantile_d4_dense_oracle.jl`); real D=4 solve with `hessopt=exact`; real D=20
convergence and per-callback profile; only then the outer gradient (the closed form and its chain rule
are already written and gated: `d_delta_dual_d_mu_shared`, `chain_cmpq_mass_gradient_to_raw`), then
checkpoint/campaign wiring.

Measured reason this matters: the FG alone does **not** converge at production `n`. LBFGS hits
`maxit=5000` at `n_x=412` after 16,734 FG evals (`-400`) with Δ_dual finite at 1.92; the smaller
`n_x=145` case reaches `-102`. `-400` is accepted in the FG-only smoke *with that reasoning recorded
at the check* — the smoke's claim is "the composed FG drives a real `KN_solve` to a finite, sane Δ",
not "the FG alone converges".

---

## 6. Things worth not rediscovering

* `ek_inner_cmpq.opt` uses `opttol = opttol_abs = 1e-10`, not `ek_inner_pq.opt`'s `1e-12`, which sits
  below the ~1e-11 achievable floor (memory `inner-opttol-was-below-achievable-floor-2026-08-11`).
  `linsolver ma97` with 4 solver threads is kept; `linsolver auto` picks a serial solver.
* `ek_inner_cmpq_fgonly.opt` (`hessopt lbfgs`, `maxit 5000`) exists **only** for the FG gate. Running
  the production option file with no Hessian builder is a **hard error**, not a silent quasi-Newton
  downgrade — checked in the gate.
* The FG state holds its thread-local histogram scratch as campaign-lifetime fields. The standalone
  family's own state reallocates that scratch inside **every** FG callback (~608 KB/call at
  D=20/L=5/16 threads). That is a real defect there; it is deliberately **not** fixed in passing,
  since that family is in production and the fix needs its own gate.
* `cutoff_source` is deliberately absent from this family's config. `:empirical_quantile` puts each
  origin's cutoffs at its own empirical quantiles, which are not CM grid points and differ across
  origins, so the superset identity fails and a shared μ is not even well defined.
* CM's `L` parameter means "number of grid levels" and this family's `L` means "number of PQ bins" —
  a genuine collision between two established APIs. `resolve_cm_pairwise_quantile_config` returns
  `n_cm_levels` so no downstream site has to disambiguate it again.
