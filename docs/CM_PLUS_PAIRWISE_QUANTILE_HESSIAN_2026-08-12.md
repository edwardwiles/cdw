# CM + pairwise-quantile (family #7): the exact Hessian, D=20 convergence, and the outer gradient

Session of 2026-08-12, continuing from `docs/CM_PLUS_PAIRWISE_QUANTILE_STATUS_2026-08-12.md`.
Worktree `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`, branch
`feature/pq-free-mass-reparam-2026-08-10`, commits `d1992c0` → `c08d44d` → `<final>`.
**Not pushed to any remote.**

---

## 1. Status at a glance

| Piece | Before this session | Now | Gate |
|---|---|---|---|
| Moment rows, shared-μ kernels, redundancy | done | done | D=4 dense oracle |
| Composed inner FG | done | done | real D=4, 54/54 |
| `H_RR`, `H_E,R`, `H_R,CM` | done | done | dense Gram, ≤ 7.7e-15 |
| `H_CM,CM`, `H_E,CM` | **not wired** | **wired** | see §3 |
| **Packed assembler + callback registration** | **NOT BUILT** | **BUILT** | §3, 121/121 |
| Real D=4 solve, `hessopt=exact` | — | **`nStatus=0`** | §4 |
| Real D=20 solve, `hessopt=exact` | — | **`nStatus=0`** | §5 |
| Per-callback block profile | — | **measured** | §6 |
| Outer gradient (reoptimized FD) | closed form only | **gated end-to-end** | §7 |
| Checkpoint / campaign selectability | not started | not started | §9 |

The family now runs under `ek_inner_cmpq.opt` and converges at production scale. What is left
before it can enter the five-family campaign is wiring, not mathematics — §9.

---

## 2. What was built

Three files touched, one added, plus three test files.

* `cm_pairwise_quantile_hessian.jl` (+): `cmpq_pq_row_map`, `cmpq_pk_upper`, `pack_cmpq_hessian!`.
  These live here, not with the production wiring, so the **standalone dense oracle** — which loads
  no production stack at all — can still gate the packing.
* `cm_pairwise_quantile_hessian_assembly.jl` (**new**): `CMPQCoreHessCtx`, `build_cmpq_hess_ctx`,
  `cmpq_fill_hessian_blocks!`, `cmpq_hess_cb_builder`. This is where the production stack enters
  (`WinnerPairHessCtx`, `CMBinHessCtx`, `WinnerBinCrossScratch`).
* `cm_pairwise_quantile_production.jl`: `cm_pairwise_quantile_attach(...; build_hessian_ctx)` and
  `cmpq_hess_builder_for`.

### 2.1 Two wiring changes that break old call sites on purpose

* `cm_pairwise_quantile_attach(ctx, cmpq; build_hessian_ctx)` — required, not defaulted.
* `archCMPQ_base_state(...; hess_cb_builder)` — required, not defaulted. Previously it defaulted to
  `nothing`, which would have made "which Hessian is this solve using" an invisible property of a
  call site. Callers now pass `cmpq_hess_builder_for(ctx_cm)` or an explicit `nothing`.

Populating the Hessian context does **not** by itself register a callback, so the existing hard
error ("production option file requests `hessopt=exact` but no builder was supplied") still fires,
and is still checked in the FG gate.

### 2.2 One buffer released, checked rather than argued

`build_cm_bin_ctx` allocates `cctx.Ews`, a `W × NCORE` scratch — **321 MB** at D=20/W=100k/NCORE=401.
It feeds only `_fill_cm_HEE!`'s dense fallbacks, which this operator-native family never takes
(`ncore_core == NCORE`, and H_EE comes from `winner_pair_hessian!` directly). It is released to `0×0`
at context build. That is not left as an argument: the D=4 Hessian gate calls `_fill_cm_HEE!`
against the shrunken `cctx` on every case and checks it still produces the right H_EE.

---

## 3. The assembler, and how it is gated

`x = [ζ; λ_E(ncore1); λ_L(L-1); λ_P((L-1)²·npair); λ_CM(ncm)]`, `n = NCORE + n_restr + ncm`.

| Block | Source | Status |
|---|---|---|
| `H_EE` | `winner_pair_hessian!` + `WinnerPairHessCtx` | verbatim |
| `H_E,R` | `pairwise_quantile_cross_hessian_block!` @ replicated μ, `sig`-selected | reuse |
| `H_RR` | PQ raw fill + centering @ replicated μ, `sig`-selected | reuse |
| `H_R,CM` | `build_cmpq_cross_hess_tables!` + `fill_cmpq_cm_cross_block!` | the one new block |
| `H_CM,CM` | `build_bin_tables_threaded!` → `prefix_sum_tables_threaded!` → `fill_cm_HCC!` | wired |
| `H_E,CM` | `winner_pair_cross_hessian_fill!` → `winner_pair_cross_hessian_cm_block!` per `l` | wired |

Three details that were traced rather than guessed, and are now asserted in code:

* `build_bin_tables!(cctx, nothing, h; fill_S=false)` — `H === nothing` is admissible **only** on
  that branch. This family is operator-native and has no dense `H` at all, so `fill_S=false` is not
  an optimization, it is the only admissible call.
* `fill_cm_HCC!` writes at CM's own offsets and fills **both** triangles, so the `ncm×ncm` corner is
  read back out of `cctx.Hfull` rather than having its offsets redirected. `cctx.NCORE` is set to
  this family's economic width precisely so that corner lands where it is expected.
* `winner_pair_cross_hessian_cm_block!` is per threshold block and needs a `WinnerBinCrossScratch`
  populated once per callback by `winner_pair_cross_hessian_fill!`. CM+ZC's `use_direct_hcz` /
  `ncore_core` split does **not** apply (`ncore_core == NCORE`, asserted at build).

### 3.1 The `n_restr × n_restr` copy that is not made

`H_RR` and `H_E,R` arrive in the *standalone* PQ family's larger row space. Rather than copy the
sub-block out, the packed write reads through a precomputed monotone row map `sig`. The copy would
have been **74 MB written + 78 MB read per callback** at D=20/L=5. `extract_cmpq_HRR!` /
`extract_cmpq_HER!` still exist and are still gated (dense oracle check 10b); the real-context gate
additionally checks the direct `sig` read is **bit-identical** to them, so the two routes cannot
drift.

### 3.2 The packed write

Modelled on `pairwisequantile_hess_cb_builder`, carrying all three of its measured fixes: column
walks of the two symmetric blocks, the closed-form packed row offset `k0(i) = (i-1)n - (i-1)(i-2)/2`
with `:dynamic` scheduling, and `evalResult.hess` hoisted once with a type assertion. Measured cost
at D=20: **~26 ms of a 1.5 s callback (1.7%)** — i.e. the thing that was 46% of the standalone
family's inner solve before its own fix is a rounding error here.

`H_RC` and `H_EC` are the only blocks read by row. Left that way deliberately — see §6 for why the
profile says that is the right call, and for what would change the answer.

---

## 4. Real D=4, `hessopt=exact`: 121/121

`test_cm_pairwise_quantile_real_d4_hessian.jl`, five configurations
(L=5/G=10/fam1, L=5/G=50/fam2, L=5/G=10/fam2/**orthonormal**, L=2/G=50/fam1, L=10/G=50/fam2).

### 4.1 The reference is exact, not a finite difference

`r` is **linear** in the inner variables, so `H = (1/W) M' diag(h) M` exactly with
`M = [1 | E | G_R | G_CM]`. And because `r` is linear, `M` itself can be read out of the production
operator with no dense-G machinery: `r(0) = 0`, so `M[:,j] = -dual_index!(st, e_j)` **exactly**,
column by column. That gives an independent dense reference covering all six blocks — including the
economic ones the standalone oracle cannot see — with no derivative approximation anywhere. (`M` is
built in the test, at D=4/W=8000, purely as a reference; production never materializes it.)

The linearity premise is itself checked: `-M·x` reproduces `dual_index!(st,x)` to ≤ 4.9e-16.

### 4.2 What it measured

Worst relative deviation per block pair, over all five cases:

| Block pair | worst rel | | Block pair | worst rel |
|---|---|---|---|---|
| E × E | 3.6e-14 | | R × R | 8.3e-16 |
| E × R | 1.6e-14 | | R × CM | 2.8e-15 |
| E × CM | 4.9e-15 | | CM × CM | 4.1e-14 |

Four further gates, all passing at every case:

* **H_EE by two independent routes.** `winner_pair_hessian!` (this family's) vs
  `fill_core_hessian_upper!` / `_fill_cm_HEE!` (CM's, `:exact_winner_pair_parallel` — a different
  workspace and a different fill): ≤ **6.6e-15**. Free strong gate, two stacks, one block.
* **H·d == FD of the analytic gradient**, ≤ **6.0e-9**. This is the one that checks the assembled H
  is the Hessian *of the f the FG callback returns*, not merely of the Gram identity — it would
  catch the linearity premise being wrong, not just a mis-assembly.
* **Threaded vs serial CM bin tables** give the same packed vector, ≤ 2.1e-14.
* **`cmpq_pk_upper` == `_pk_upper` == a literal running counter** (three routes to the packed index).

### 4.3 Real KNITRO solves

| Case | nStatus | n_fg | n_hess | Δ_dual | \|grad\| |
|---|---|---|---|---|---|
| L=5, G=10, fam 1 | **0** | 11 | 10 | 1.6346 | 2.4e-13 |
| L=5, G=50, fam 2 (n_x=412) | **0** | 12 | 11 | 1.9622 | 5.8e-13 |
| L=5, G=10, fam 2, orthonormal | **0** | 11 | 10 | 1.6948 | 2.5e-14 |

The FG-only path at the same `n_x=412` point measured **16,734 FG evaluations and `-400`**. The
exact Hessian takes it to `nStatus=0` in twelve.

### 4.4 The dense oracle, extended: 136/136 (was 116)

Checks 13–15 gate the packed write against the full Gram identity with a *synthetic* economic block
(placement does not care what `E` is), **bit-identically** — it is pure data movement, so a tolerance
would be the wrong instrument. Two things make it more than a tautology:

* the two blocks that arrive in the standalone family's larger row space are handed in
  **NaN-poisoned** everywhere except the rows `sig` selects, so a single stray read of a dropped row
  surfaces as a NaN rather than as a small error;
* a negative control confirms a transposed `H_R,CM` is rejected, so check 14a is genuinely testing
  that block's orientation.

The reference had to be **explicitly symmetrized** first: `M'(h.*M)` via BLAS computes `[i,j]` and
`[j,i]` in different orders, so without that the bit-identity claim would have been testing BLAS's
rounding rather than the packing.

---

## 5. Real D=20: it converges

`d20_real_setup_design(W, δ=1.0, find_smallest=true, draw_design=:pseudorandom, draw_seed=20260719,
destination_sample=:exclude_row, σHat=3.0, inner_lower_limit=-10.0)`, L=5, G=50, families=2,
contrasts `:orthonormal`, `mass_start=:uniform`, at the calibration point.

`n_x = 5288 = NCORE 382 + n_restr 3044 + ncm 1862`; 13,984,116 packed entries; npair=190, Lcm=49,
nO=19.

| W | nStatus | n_fg | n_hess | wall | Δ_dual | \|grad\| | Hessian callbacks / solve |
|---|---|---|---|---|---|---|---|
| 20,000 | **0** | 8 | 7 | 29.6 s | 0.195205480388 | 1.0e-11 | 10.6 s of 29.6 s |
| **100,000** | **0** | 6 | 5 | 42.3 s | 0.028738400455 | 1.7e-12 | 27.6 s of 42.3 s |

Structural gates at W=100k: 2,000,000 (draw, origin) bin cells cross-checked between the z-space
route and CM's integer bin map, all agreeing; minimum joint-cell occupancy 3,800.

The **unread standalone-PQ rows cost 2.4%** at D=20/L=5 (3120 computed, 3044 read) — the prior
estimate was ~5%.

Production scale converges in **five Hessian callbacks**, and the residual `~15 s` outside the
callbacks is the O(n³) KKT factorization at n=5288. Δ_dual falls from 0.195 at W=20k to 0.0287 at
W=100k, which is the expected direction (see memory `d20-realdata-w-sensitivity`: small W overstates
the divergence at a fixed point); these are single points at the calibration θ, not frontier values.

---

## 6. The per-callback block profile — and a corrected cost model

3 warm repeats after a discarded JIT pass, D=20, L=5, G=50, families=2, 8 Julia threads. Both `W`
are shown because the ranking *changes* between them, and only the W=100,000 column is production.

| Block | W=20k s/cb | W=20k share | **W=100k s/cb** | **W=100k share** | scales with W? |
|---|---:|---:|---:|---:|---|
| `cmpq_H_ER` | 0.4355 | 29.3% | **2.5560** | **46.8%** | yes |
| `cmpq_H_CC_tables` | 0.3046 | 20.5% | 0.7883 | 14.4% | yes |
| `cmpq_H_RR_tables` | 0.1271 | 8.6% | 0.6336 | 11.6% | yes |
| `cmpq_H_ECM_fill` | 0.0822 | 5.5% | 0.5254 | 9.6% | yes |
| `cmpq_H_RCM_tables` | 0.0987 | 6.6% | 0.4750 | 8.7% | yes |
| `cmpq_H_RCM_fill` | 0.2822 | **19.0%** | 0.2249 | **4.1%** | no |
| `cmpq_H_CC_fill` | 0.1098 | 7.4% | 0.1231 | 2.3% | no |
| `cmpq_H_EE` | 0.0141 | 0.9% | 0.0916 | 1.7% | yes |
| `cmpq_H_RR_fill` | 0.0137 | 0.9% | 0.0191 | 0.4% | no |
| `cmpq_H_ECM_blocks` | 0.0150 | 1.0% | 0.0172 | 0.3% | no |
| `cmpq_H_RR_center` | 0.0026 | 0.2% | 0.0036 | 0.1% | no |
| packed write (unlabelled residual) | ~0.026 | ~1.7% | ~0.056 | ~1.0% | no |
| **total** | **1.511** | | **5.514** | | |

**This corrects the prior status doc's cost model.** That doc predicted the new `X`-table build
(`O(W·npair·D)`, 380M increments at W=100k) would be "the dominant new cost this family adds to the
Hessian callback". Measured at production `W` it is **8.7%**, sixth on the list.

At W=100k, 5 callbacks × 5.51 s = 27.6 s of the 42.3 s solve; the remaining ~15 s is the O(n³) KKT
factorization at n=5288.

### 6.2 Nothing was optimized off the back of this, and what the profile actually points at

Running both `W` was the point: `cmpq_H_RCM_fill` looks like a 19% target at W=20,000 and is 4.1% at
production scale, because it does not scale with `W` while five other blocks do. Optimizing it off
the small-`W` profile would have been work aimed at the wrong term — the exact trap the
"profile before optimizing" instruction exists to avoid.

The real target at production scale is **`cmpq_H_ER` at 46.8%** — `pairwise_quantile_cross_hessian_block!`,
whose winner-slot scatter is `O(Ddest·W·(D+npair))` = 399M increments at D=20/W=100k, the largest
single operation count in the callback. Two facts about it that decide what to do next:

* It is **not this family's code**. It is the standalone pairwise-quantile family's cross block,
  reused here at replicated μ. That family is in production. Changing it needs its own gate against
  its own results, exactly as the prior session declined to fix that family's per-callback scratch
  reallocation in passing.
* It is computed as a **superset**: 3120 rows for the 3044 this family reads, i.e. 2.4% waste — so
  the win is not in specializing it to this family, it is in the block itself.

So: identified, measured, and left alone. Any change to it should be gated against the standalone
family's own results and, per the box caveat below, benchmarked with interleaved arms.

**Wall-clock caveat, recorded in the driver itself.** This box was at load 74 during these runs and
hosts other campaigns. What is reported is a **within-run block share** (every block paying the same
load) plus operation counts and dimensions — not a cross-run speed claim. A single-run A/B here is
worthless; per the standing rule, arms must be interleaved.

**Wall-clock caveat, recorded in the driver itself.** This box was at load 74 during these runs and
hosts other campaigns. What is reported is a **within-run block share** (every block paying the same
load) plus operation counts and dimensions — not a cross-run speed claim. A single-run A/B here is
worthless; per the standing rule, arms must be interleaved.

---

## 7. The outer gradient, gated end to end

The closed form (`cm_pq_dC_dmu` → `d_delta_dual_d_mu_shared` → `chain_cmpq_mass_gradient_to_raw`) was
already gated against a **fixed-dual** finite difference in the dense oracle. That is necessary but
not sufficient: it tests the algebra at an arbitrary `(ζ, λ)`, not the envelope-theorem claim, whose
ground truth **re-solves** the inner problem at each probe.

`test_cm_pairwise_quantile_outer_gradient_fd.jl` supplies that. Every FD probe is a full, fresh
`hessopt=exact` KNITRO solve. Real D=4, L=5, G=50, families=2, orthonormal.

**Enforced point (non-uniform μ = [0.36986, 0.12329, 0.30822, 0.09863]):**

| h | rel L2 | max\|diff\| | cosine |
|---|---|---|---|
| 1e-3 | 1.141e-05 | 7.32e-05 | 0.999999999997 |
| 1e-4 | 1.140e-07 | 7.31e-07 | 1.000000000000 |
| 1e-5 | **1.690e-09** | 9.23e-09 | 0.9999999999999999 |

Clean `h²` convergence — the ladder is doing what it is for (truncation error falling as `h²`,
solver noise rising as `1/h`), and the threshold was not moved to meet the number. The standalone
family reaches 4.1e-10 on its own gate; this is the same order.

At uniform μ (reported, not enforced): 1.020e-09 at h=1e-5, same `h²` ladder.

**Both negative controls fire:**

* a sign-flipped gradient anti-correlates with FD at cosine **−1.000000000000**;
* the **one-slot product rule** — dropping the `b`-slot of `d(μ_a μ_b)/d(μ_c)`, the natural
  plausible-looking transcription error this family's own docstring warns about — fails the same
  gate at rel **1.075**.

The gate is enforced at non-uniform μ because at `μ = 1/L` the partner index is provably
unobservable (measured at 1.3e-9 in the dense oracle). At L=2 the one-slot control cannot fire and
is reported SKIP, never PASS.

Note this is where the family pays off: **4 outer mass coordinates at D=4** where the standalone
family needs 16, and **4 at D=20/L=5** where the standalone family needs 80.

---

## 8. Full gate inventory, as run on this machine

| Gate | Result |
|---|---|
| `test_cm_pairwise_quantile_d4_dense_oracle.jl` | **136/136** (was 116) |
| `test_cm_pairwise_quantile_real_d4_fg.jl` | **54/54** |
| `test_cm_pairwise_quantile_real_d4_hessian.jl` | **121/121** (new) |
| `test_cm_pairwise_quantile_outer_gradient_fd.jl` | **11/11** (new) |
| `test_cm_pairwise_quantile_d20_hessian_solve.jl` W=20k | **PASS**, nStatus=0 |
| `test_cm_pairwise_quantile_d20_hessian_solve.jl` W=100k | **PASS**, nStatus=0 |

---

## 9. What is left, in order

1. **Checkpoint layer.** Model on `pairwise_quantile_checkpoint.jl`. Nothing mathematical is
   outstanding — the outer gradient is exact and gated, and the inner solve converges.
2. **A verifier.** The standalone family has
   `verify_inner_solution_operator_pairwisequantile!` (independent forward/transpose recompute,
   per-block KKT residuals, probability tables). This family needs its analogue before
   `classify_inner_result`/`lfd_ok` can gate its campaign results — see memory
   `feedback-lfd-ok-verification-gate-required`: `FiniteSolved && within_budget` is **not** enough.
   The D=20 solves above report `|grad| ~ 1e-11`, which is a stationarity check, not a verification.
3. **Campaign selectability**, i.e. a `family_tag` and the orchestrator arm. Note memory
   `feedback-hzz-blas-thread-gate-and-family-tag-gates`: performance gates keyed on `family_tag`
   starve new families. Key on capability.
4. **Then** revisit the block profile at production `W` and decide on threading
   `fill_cmpq_cm_cross_block!` (§6.1).

---

## 10. Things worth not rediscovering

* The `M`-extraction trick (`M[:,j] = -dual_index!(st, e_j)`, exact because `r` is affine) gives any
  operator-native family a full dense Hessian reference at test scale with no dense-G machinery and
  no finite difference. It generalizes to every family in this codebase whose `r` is linear in the
  inner variables — which is all of them.
* A BLAS-computed `M' diag(h) M` is **not** bit-symmetric. Any test claiming bit-identity against a
  transposed read must symmetrize it explicitly first.
* `cctx.Ews` is dead weight for every operator-native family, not just this one — 321 MB at
  D=20/W=100k. The other families that build a `CMBinHessCtx` on a no-dense-H path may be carrying
  it too; worth a look, but it was out of scope here and is **not** claimed to have been checked.
* The packed write, which was 46% of the standalone family's inner solve before its own fix, is
  1.7% here. The three fixes carried over from it are load-bearing and should not be "simplified".
* `mean_m = 1.0` exactly at both D=4 outer-gradient points. That is not a bug — it is
  `(1/W) Σ Ψ'(r_w)` at a converged inner solution, where the ζ stationarity condition
  `∂f/∂ζ = 1 - (1/W) Σ Ψ'(r) = 0` pins it.
