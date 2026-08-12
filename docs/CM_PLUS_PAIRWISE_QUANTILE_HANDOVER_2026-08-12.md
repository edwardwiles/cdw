# Handover: build the CM + pairwise-quantile family (family #7), inner solve first

You are picking up a well-defined new family in the `cdw` full-A_od codebase. **Proceed
autonomously overnight.** The user is asleep and will read this in the morning.

Worktree: `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`
Branch: `feature/pq-free-mass-reparam-2026-08-10`, HEAD `d27ef7b`. **Not pushed to any remote; do
not push without asking.**

Read `CLAUDE.md` in the repo root first, and the memory files it points at. Everything there
applies. The highlights that will actually bite you are repeated at the bottom of this document.

---

## 1. What to build

A new restriction family combining **common marginals (CM)** with the **pairwise-quantile
independence** restriction. It sits exactly where a gap currently is:

|                        | per-origin free parameters | shared parameters (CM-combined) |
|------------------------|----------------------------|---------------------------------|
| powers-based (ZC)      | origin-ZC                  | CM+ZC (`cm_meanzc_*`)           |
| quantile-based (PQ)    | pairwise-quantile (exists) | **CM+PQ — DOES NOT EXIST, build it** |

### The math

**CM stays exactly as it is.** It is the pure-CDF common-marginals restriction (CDW eq. 35,
orthonormal anchored contrasts) in `common_marginals_moments.jl`, at `cm_grid_size = 50`,
`cm_grid_rule = :equal`. It imposes `F_o = F_ref` for every origin `o`. Critically it is
**theta-independent** (built once from the fixed baseline draws `U`) and **adds zero new outer
parameters**. Do not modify it. Do not re-derive it.

CM pins the marginals *to each other* but says nothing about the **level** of the reference
marginal. So:

**NEW — level parameterization.** Introduce `L-1` free outer parameters `mu_a`, the binned masses
of the reference marginal on the pairwise-quantile grid, with `mu_L = 1 - sum_a mu_a`. Use the
**existing** stick-breaking simplex transform in `pairwise_quantile_mass_transform.jl`
(`decode_origin_masses!`, `raw_from_origin_masses!`, `mass_jacobian_block!`,
`default_raw_mass_bounds`) — it already does exactly this for one origin. You need ONE simplex, not
`D` of them.

**NEW — level moment rows.** `L-1` rows pinning the reference marginal's binned masses to `mu`:

    1{b_ref = a} - mu_a         for a = 1 .. L-1

**Pair rows (the restriction proper), keyed on the SHARED mu:**

    1{b_o = a, b_p = b} - mu_a * mu_b      for all C(D,2) origin pairs (o,p), all (a,b) in (L-1)^2

Off-diagonals are included: `a` and `b` range independently. This matches the existing standalone
family, whose `n_pair_rows(D,L) = (L-1)^2 * C(D,2)`.

**DROP the per-origin marginal rows.** The standalone family has `1{b_o=a} - mu_{o,a}` for every
origin (`n_marginal_rows = (L-1)*D`). Under CM + the new reference-level rows these are **implied**,
so keeping them makes the stacked moment matrix **exactly rank-deficient** — a singular KKT, not a
tolerance problem. See §5 trap 1.

### A hard constraint on L

The whole construction rests on CM's grid being a superset of the PQ cutoffs, so that CM genuinely
forces every origin to share the *binned* marginal. CM's grid is 50 **equal-probability** points, so
this holds **iff L divides 50**: `L in {2, 5, 10, 25}` are valid; **L = 7, 8, 9 are NOT** (1/8 =
6.25/50 is not a grid point) and would silently break the argument. **Build and gate at L = 5.**
Hard-error on an L that does not divide `cm_grid_size`, with a message that says why.

---

## 2. Order of work — do NOT reorder this

The user was explicit:

1. **Get the FG callback right** — mathematically and in code — for the **inner** solve.
2. **Get the Hessian callback right** — same.
3. Only then the outer gradient, checkpoint/driver plumbing, and production wiring.

Do not start on the outer loop or a campaign until 1 and 2 are gated.

---

## 3. Reuse, do not reinvent — specific pointers

This is the single most important instruction in this document. **Everything below already exists
and is already optimized. Do not write new versions of any of it.** The user has repeatedly found
Claude sessions rewriting optimized machinery from scratch; do not add to that record.

### The shared-parameter outer gradient is already written, for ZC

CM+ZC's `nu` is **shared across origins**, exactly as `mu` is here. Its gradient is the template:

- `cm_meanzc_moments.jl:589  d_delta_dual_d_nu_vec` — **this is your model.** Note how it handles
  the mean (diagonal) block and the `K_pair` (off-diagonal) block, and how the pair term's
  derivative w.r.t. the SHARED parameter picks up the two-slot structure via `d_pair_dnu`. That is
  precisely the `d(mu_a*mu_b)/d(mu_c) = delta_{ac}*mu_b + delta_{bc}*mu_a` you need.
- `cm_meanzc_moments.jl:610  d_delta_dual_d_eta_nu_vec` — applies the transform chain rule.
- `cm_meanzc_moments.jl:635  d_delta_dual_d_eta_active_and_nustar_shared`.

The **per-origin** analogue, which the standalone PQ family already mirrors, is
`cm_originzc_moments.jl:317  d_delta_dual_d_eta_origin_vec`, and PQ's own copy is
`pairwise_quantile_mass_gradient.jl :: d_delta_dual_d_mu`. Read all three side by side: your job is
the shared-parameter version of the third, and the first two show exactly how that differs.

### The CM combination machinery

- `common_marginals_moments.jl` — `wrap_moments_with_cm` (splices the theta-independent CM block
  into the columns after `ncore`), `build_cm_augmented_obj` (wraps an existing full-A context's
  `obj` WITHOUT mutating it). This is how every CM+X family is built. Use it.
- `cm_meanzc_config.jl`, `cm_meanzc_moments.jl`, `cm_meanzc_production.jl`,
  `cm_meanzc_lookup_kernels.jl`, `cm_meanzc_cplus.jl`, `cm_meanzc_checkpoint.jl` — the complete
  worked example of "CM + a second restriction family", end to end. Your family is the same shape.

### The Hessian

The user specifically flagged **the CM x PQ cross block** as the genuinely tricky part. There is a
direct precedent — the CM x ZC cross block:

- `cm_hessian_architectures.jl` — `_cm_cross_hessian_wants_winner_bin`,
  `_cm_zc_cross_hessian_wants_winner_bin`, `_cm_cross_hessian_wants_direct_hcz`,
  `pack_upper_cm_hessian!`, `build_originzc_core_hess_ctx`. Read the CM x ZC cross path and follow
  its structure.
- `pairwise_quantile_hessian.jl` — PQ's own structured Hessian. Understand it fully before touching
  it: the T1/T2/T3/T4 table families, the 3x canonical dedup (`_assign_canonical_combos`,
  `C(D,3)=1140`, `C(D,4)=4845`), the threaded scatter, and `_lo_write!` which stores **only the
  lower triangle** (`H[max(i,j), min(i,j)]`) because the packed write walks it by column.
- `pairwise_quantile_operator.jl` — `build_pairwise_quantile_tables_threaded!`, the shared threaded
  histogram builder. Reuse it; do not write a second accumulator.
- `winner_pair_cross_hessian.jl`, `threaded_cross_hessian.jl` — the shared cross-Hessian layer.

**The structure to exploit.** The pair indicator factorizes:
`1{b_o=a, b_p=b} = 1{b_o=a} * 1{b_p=b}`, so the moment matrix's pair block is a row-wise
Khatri-Rao product of the per-origin indicator matrices, and every Hessian block is a contraction of
low-order joint histograms (T1..T4). That is why the assembly is O(tables), not O(W * n^2). The
CM x PQ cross block should reduce the same way: CM's block is a fixed function of the baseline draws
and PQ's is a bin indicator, so their cross terms are joint histograms of (CM grid cell, PQ bin)
over origins. Find the right table family before writing any loop.

### The economic block

Use **Backend C+** (`lfix_factorized.jl`) via the existing adapters
(`pairwise_quantile_cplus.jl`, `cm_meanzc_cplus.jl`, `cm_originzc_cplus.jl`). It is the factorized
representation that never materializes the `W x D x Ddest` price tensors and is 4.4x faster than the
dense path at real D=20/W=100,000. **Never build the dense path "for now".**

---

## 4. Hard prohibitions

- **No dense G, ever, anywhere.** No `dense_reference` fallback, silent or otherwise. If a backend
  is missing, find the backend-agnostic upstream fix. (memories:
  `feedback-no-dense-reduced-ever-anywhere`, `feedback-never-silently-fall-back-to-dense-reference`)
- **No new machinery where existing machinery covers it.** If you find yourself writing a histogram
  accumulator, a CM wrapper, a simplex transform, or a cross-Hessian scatter, stop and go find the
  one that already exists.
- **No defaults on any scientific parameter.** Bare Julia keyword arguments (`sigma::Float64`, not
  `sigma::Float64 = 2.5`) so an omission raises `UndefKeywordError`. This applies at EVERY layer of
  the call chain, not just the top one. See CLAUDE.md.

---

## 5. Known traps — each of these has already bitten this codebase

1. **Redundant moment rows produce a singular KKT, not a warning.** Keeping the per-origin marginal
   rows alongside CM + the reference-level rows is exact linear dependence. Precedent: the CM+ZC /
   origin-ZC K=2 (sigma=3) singularity, where the autarky moment was collinear with the
   k=(sigma-1) row (memory `cmzc-k2-singularity-root-cause-and-fix-2026-08-05`). **Run a rank /
   conditioning check on the stacked moment matrix early**, before you have built anything on top of
   it. Note also `feedback-solver-option-cannot-fix-genuine-rank-deficiency`: if it is rank
   deficient, no KNITRO option will save it.

2. **The shared-mu gradient is numerically INVISIBLE at uniform mu.** At `mu_a = 1/L` every mass is
   identical, so transposing a slot or an index changes nothing you can measure. The standalone
   family's gates are all enforced at a deliberately **non-uniform** mu and carry an explicit
   **negative control** that fires (0.55 / 0.38) if the partner index is written the natural wrong
   way. Do the same, and make the negative control specific to the new two-slot term. (memory
   `pairwise-quantile-free-mass-reparam-2026-08-10`, fact 2)

3. **Every new cross block needs its symmetrization mirror**, or it is silently halved (memory
   `feedback-check-hessian-symmetrization-mirror-for-every-new-cross-block`). PQ stores the lower
   triangle only; whatever you add must follow `_lo_write!`'s convention, and the packed write reads
   `HRR[j, i]`.

4. **Dedup storage without deduping the scatter overcounts.** T3/T4 store one table per canonical
   unordered tuple; the scatter must visit `canon_list` once, never all N role-variants. Doing it
   wrong gives a clean N-fold overcount that looks like a maths error. (memory
   `feedback-dedup-storage-without-deduping-scatter-overcounts`)

5. **Verify every analytic derivative against FD before building on it** (memory
   `feedback-always-verify-analytic-gradient-against-fd-before-trusting`). A self-cancelling test
   convention cannot gate a sign (memory
   `feedback-self-cancelling-test-convention-cannot-gate-a-sign`).

6. Do **not** explain an inner-solve failure by warm/cold start, and do **not** claim Delta* is
   unknown because `lower_limit` cut it off. Both are addressed at length at the top of CLAUDE.md.

---

## 6. Gates — all must pass before you call any part of this done

- **D=4 dense oracle** for the new family: the assembled Hessian vs an INDEPENDENT dense reference,
  and the packed vector vs the full dense matrix. Model it on
  `test_pairwise_quantile_d4_dense_oracle.jl` (29 checks; the standalone family agrees at 5.6e-17).
- **FG analytic vs FD.**
- **Rank / conditioning of the stacked CM + reference + pair moment matrix** (trap 1).
- **Non-uniform mu, with the negative control** (trap 2).
- **Outer gradient vs reoptimized FD** once you get there. The standalone family achieves 4.1e-10
  relative L2; aim for that order.
- **Real D=20 smoke** through the production driver, mirroring
  `smoke_pairwise_quantile_outer_driver.jl` (26 checks).

Pure performance changes must not move numbers at all: any numerical movement is a bug, not a
tolerance.

---

## 7. Current machine state — read before launching anything

- **An overnight job is running and must not be disturbed:** `screen -S pq_L5_d1_long`, campaign
  root `/bbkinghome/edav/repo_scratch/pq_L5_delta1_long_2026-08-12`, L=5 delta=1.0, finishing about
  07:30 local. Other unrelated campaigns are also on the box. Load average has been 20-55 on 208
  cores. **Wall-clock comparisons on this box are worthless unless you interleave arms** — a
  single-run A/B here produced an apparent 18% regression that was pure load drift.
- **Inner-solve KNITRO options** (measured this session): `ek_inner_pq.opt` for L < 8
  (`linsolver ma97`, 4 solver threads, `par_concurrent_evals no`) and `ek_inner_pq_ma97_t10.opt` for
  L >= 8. `linsolver auto` picks a SERIAL solver and costs 6.26x at L=10. Thread the callbacks, but
  do not raise the solver thread count: 16 threads measured slower than 4. Pass the file through
  `run_pairwise_quantile_upper_checkpointed(...; inner_opt_override = ...)`.
- The inner Hessian is **84.6% structurally dense**, so sparsity-pattern declaration is NOT a lever;
  do not spend time there. Reducing `n` is the only thing that moves the factorization.
- Note the new family will make the inner solve **more** expensive, not less: CM adds its own moment
  rows on top of the pair rows. The win is in the OUTER dimension (D*(L-1)=80 free mass coordinates
  collapse to L-1=4 at L=5).

## 8. Standing operational rules (CLAUDE.md — not optional)

- `export OPENBLAS_NUM_THREADS=1` under Julia, always.
- Long jobs under `screen`, never harness background. **Check `ps` + `tail` within ~60 seconds of
  launch.** This session caught a launcher that would have wasted an entire night, two minutes in.
- `flush(stdout)` after progress output in long-running scripts.
- Defer cleanup of scratch files to the end; batch any destructive commands into one request.
- **Push deliverables to Dropbox before you finish**, new dated subfolder:
  `rclone copy <dir> "dropbox:Gravity robustness/Analysis/Server Output/<subfolder>" --progress`.
  Docs + `provenance.txt` + a small `key_results/`. A few hundred KB, not raw logs.
- Write a status/handover doc in `docs/` and commit as you go, with honest commit messages that
  record what was measured versus what was assumed.
