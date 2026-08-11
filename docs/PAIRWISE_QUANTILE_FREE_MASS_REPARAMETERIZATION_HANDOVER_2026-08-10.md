# Handover: reparameterize the pairwise-independence restriction from FREE CUTOFFS to FIXED CUTOFFS + FREE MASSES

**Goal.** Replace the restriction's outer coordinates. Today the outer loop optimizes over the
quantile **cutoffs**, which live inside indicator functions, so `Delta*` is a genuine step function
of every outer coordinate and the gradient needs bandwidth/secant machinery. Instead: **fix the
cutoffs once per campaign and make the bin MASSES the free outer parameters.** The moment functions
then depend on the outer parameters only through a target-level shift, `Delta*` becomes smooth, and
the outer gradient has an exact closed form — which is *already implemented and production-validated
in this repo* as origin-ZC's `nu` gradient.

This is a user-directed design change (2026-08-10), not a refactor for its own sake. Read
`PAIRWISE_QUANTILE_OUTER_LOOP_STATUS_2026-08-10.md` first for what exists today and for two bugs
that must not be reintroduced.

---

## Orientation

- **Working tree**: `/bbkinghome/edav/cdw_worktrees/pq-outer-loop-2026-08-10`, branch
  `integration/pairwise-quantile-outer-loop-2026-08-10` (a clean merge of the restriction prototype
  and the `paper_upper_v1` orchestrator). Not pushed to any remote. Start a NEW branch off it.
- KNITRO: `source .knitro_env.sh` from the worktree root (`demand.mit.edu`, pinned 13.0.1). Julia via
  juliaup. Always `export OPENBLAS_NUM_THREADS=1`.
- A **live `paper_upper_v1` campaign runs on this host** (`screen -S paperupper_resume`). Do not
  touch `protocols/paper_upper_v1.toml`, and launch long jobs under `screen`, not harness background.

**Read first, in this order:**
1. `PAIRWISE_QUANTILE_OUTER_LOOP_STATUS_2026-08-10.md` — what exists, and the two bugs.
2. `PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md` — the equivalence proofs. Sections 2–3
   are stated in *cumulative* form (`P(z_o<q_r)=p_r`, `F(r,s)=p_r p_s`) and carry over verbatim with
   `p_r` free instead of `r/L`. This is the reason the change is clean: the math note already proves
   the bin-probability ↔ cumulative bijection you need.
3. `cm_originzc_moments.jl` (`d_delta_dual_d_eta_origin_vec`) and `cm_originzc_production.jl`
   (`d_delta_dual_d_eta_origin_fd`) — the gradient you are going to mirror, and its FD gate.

---

## 1. The reparameterization, precisely

**Today (version A).** Cutoffs `q_{o,r}` free (via a softplus-ordered transform,
`pairwise_quantile_cutoff_transform.jl`); targets fixed at `1/L` and `1/L²`:

```
g^M_{o,a}(z) = 1{b_o(z)=a} − 1/L                    a = 1..L-1
g^P_{op,ab}(z) = 1{b_o(z)=a, b_p(z)=b} − 1/L²       a,b = 1..L-1
```

**New (version B).** Cutoffs `q_{o,r}` FIXED for the whole campaign; masses `μ_{o,a}` free:

```
g^M_{o,a}(z) = 1{b_o(z)=a} − μ_{o,a}
g^P_{op,ab}(z) = 1{b_o(z)=a, b_p(z)=b} − μ_{o,a}·μ_{p,b}
```

Outer restriction coordinate count is **unchanged**: `(L-1)·D` either way.

Both versions are pure *dependence* restrictions — in A the marginal moments merely define `q` as the
reweighted distribution's own quantiles; in B they merely define `μ` as its own bin masses. Neither
restricts the marginals, provided **`μ` is origin-specific**. Do NOT make `μ` common across origins:
that silently adds a marginal restriction (all origins share bin masses) on top of the independence
one. The earlier `pairwise_grid_common_marginal` restriction in `trade_robustness_modular_perf` used
a common `μ` — useful prior art, but not the same restriction.

### Two modelling choices, both REQUIRED with no default (CLAUDE.md)

1. **Where the cutoffs are fixed.** The user's suggestion was Fréchet values. Candidates: the
   theoretical Fréchet quantiles of each origin's calibrated marginal, or the empirical quantiles of
   the calibration draws. Must be an explicit, named kwarg, recorded in the checkpoint (the whole
   restriction's meaning depends on it, and a run is not reproducible without it). Whatever is
   chosen, assert every bin is non-degenerate (a sane minimum draw count per marginal bin and per
   joint cell) and fail loudly otherwise.
2. **How `μ` is parameterized.** It must live on the simplex (`μ_{o,a} > 0`, `Σ_a μ_{o,a} < 1`, bin
   `L` taking the remainder) — a plain box cannot enforce that. Two good options:
   - **cumulative/stick-breaking**, mirroring the existing ordered-cutoff transform: free reals →
     increasing `0 < P_{o,1} < … < P_{o,L-1} < 1` (swap the current transform's `exp` for a
     logistic), then `μ_{o,a} = P_{o,a} − P_{o,a-1}`. Reuses machinery you already have, and the
     math note is *already written in cumulative form*, so the equivalence proofs transfer directly.
   - **softmax** over `L` unconstrained reals per origin (what `pairwise_grid_common_marginal` did).
     Beware the redundant dimension when writing the Jacobian.

   Pick one, document why, and provide the analytic Jacobian `∂μ/∂(raw)` plus a finite-difference
   check of that Jacobian alone (the existing `cutoff_jacobian_block!` has exactly such a test —
   copy that pattern).

---

## 2. Inner solve: FG and Hessian — **much smaller changes than you would expect**

This is the part to internalize before touching anything: **the indicator machinery does not change
at all.** The moment functions differ from version A only in the *constant* subtracted per row. So
every expensive kernel — the operator, the transpose, the T1–T4 Hessian tables — keeps working, and
the edits are to centering constants.

### 2.1 `pairwise_quantile_forward!` (`pairwise_quantile_operator.jl`) — one line

Today:

```julia
C_lambda = sum(lambda_M) / L + sum(lambda_P) / L^2      # <-- the ONLY L/target-dependent line
Rw = -C_lambda
for o …  a = bin[w,o];  a <= nlast && (Rw += lambda_M[o,a])
for pidx … (a<=nlast && b<=nlast) && (Rw += lambda_P[a,b,pidx])
arg0[w] -= Rw
```

Version B replaces `C_lambda` with

```
C_lambda = Σ_{o,a} λ^M_{o,a}·μ_{o,a}  +  Σ_{o<p,a,b} λ^P_{op,ab}·μ_{o,a}·μ_{p,b}
```

still a per-draw constant, so still hoisted out of the `w` loop. **Nothing else in that function
changes.**

> ⚠️ **SIGN.** `forward!` *subtracts* into its accumulator (`arg0[w] -= Rw`), i.e.
> `r = -ζ − E·λ_E − G_R·λ_R`. A sign error here was one of the two bugs found on 2026-08-10, and it
> survived an exactness test that passed to 1e-16. Respect this convention everywhere, and read
> `feedback-self-cancelling-test-convention-cannot-gate-a-sign` before writing any test for it.

### 2.2 `pairwise_quantile_transpose!` — the same one change

The transpose computes `-(1/W)·Gᵀw`, and `G[w,I] = ind_I(w) − c_I`, so
`Gᵀw|_I = (weighted histogram)_I − c_I·Σ_w w_w`. Only `c_I` changes: `1/L → μ_{o,a}` for marginal
rows, `1/L² → μ_{o,a}μ_{p,b}` for pair rows. The weighted-histogram pass
(`build_pairwise_quantile_tables_threaded!`) is untouched.

### 2.3 Hessian — the expensive part is **completely unchanged**

`H_RR[I,J] = (1/W) Σ_w h_w (ind_I(w) − c_I)(ind_J(w) − c_J)`, which expands to

```
(1/W)[ Σ_w h·ind_I·ind_J  −  c_J Σ_w h·ind_I  −  c_I Σ_w h·ind_J  +  c_I c_J Σ_w h ]
        \_______________/
         the T1–T4 tables — ~78% of inner-solve wall-clock, and INDEPENDENT of the targets
```

So:
- `build_pairwise_quantile_hessian_tables!` / `fill_pairwise_quantile_hessian_raw!` — **unchanged**.
- `center_and_scale_pairwise_quantile_hessian!` — takes `μ` instead of `1/L`, `1/L²`. This is
  essentially the whole Hessian edit.
- `pairwise_quantile_cross_hessian_block!` (H_E,R) — check whether it applies the centering; if so,
  the same substitution, and nothing else.
- `winner_pair_hessian!` (H_EE) — untouched, shared backend.
- `pairwise_quantile_hvp.jl` — the same centering substitution; its structure is unaffected.

**Set expectations honestly: this reparameterization does NOT make the inner solve faster.** Same
row count, same T1–T4 cost. Measured 2026-08-10 at real D=20, W=100,000: L=5 → 39.9 s/solve,
L=10 → 1084.6 s/solve (both `VerifiedSolved`). What this reparameterization buys is an **exact**
outer gradient, not speed.

> ⚠️ **`pairwise_quantile_hvp.jl` is NOT the fix for L=10 cost — it has already been measured and
> rejected.** An earlier draft of this handover said it was "the lever"; that was wrong, and the
> evidence was already in this repo. The 2026-08-09 session ran a decisive controlled A/B at real
> D=20/W=100,000 (`profile_pairwise_quantile_d20_hvp_ab.jl`, both variants verifier-confirmed
> correct, duals agreeing to ~1e-8):
>
> | variant | n_hess(-vec) calls | wall-clock |
> |---|---|---|
> | dense (`hessopt=exact`) | 9 | **343.72 s** |
> | HVP (`hessopt=5`, CG) | **6084** | **1512.30 s** |
>
> Each HVP callback is far cheaper, but CG needed 676× more of them — **4.4× slower overall**, and
> the verdict recorded there is "HVP is not adopted for production." That is a genuine
> algorithm-conditioning property of this problem, not a warm-start artifact. See
> `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md` Part 1 and memory
> `pairwise-quantile-hessian-optimization-2026-08-09`.
>
> Whether the ratio narrows at L=10 (per-HVP-call cost is roughly `L`-independent, while the dense
> callback grows steeply with `L`) is a *reasonable but unmeasured* hypothesis, and CG's iteration
> count would also be expected to grow with the larger, worse-conditioned system. Do not act on it
> without an A/B at L=10.

**What the measured evidence actually points at for L=10 cost.** The same 2026-08-09 session
delivered a real 7.0× via T3/T4 dedup + threading (407.4 s → 57.8 s at L=5/W=100k, 10 threads) and
identified what is left: *"the cross-block and final-packed-write blocks are NOT threaded and are
now the callback's new bottleneck (7.09 s combined) if further speedup is ever wanted."* Those two
blocks scale badly in `L` — the packed write is `O(n_rows²)`, and `n_rows` goes 3,120 → 15,570 from
L=5 to L=10. A sub-block profile at L=10 (`profile_pairwise_quantile_d20.jl 100000 10`, results in
`logs/pq_L10_blockprofile.log`) is the right way to confirm where the time actually goes before
optimizing anything.

### 2.4 A free simplification

With cutoffs fixed, `bin[w,o]` is campaign-lifetime constant: `refresh_pairwise_quantile_bins!` runs
once, not once per outer point, and `reset_for_solve!` no longer needs to rebuild it. The
`sorted_z`/`sorted_idx` arrays on `PairwiseQuantileOperator` exist *only* to support the
cutoff-crossing gradient and can go with it.

---

## 3. The outer gradient — closed form, and already validated in this repo

### 3.1 Derivation (short, do it yourself before coding, then check against this)

`f = (1/W) Σ_w Ψ(r_w) + ζ`, `Δ* = −f*`, and `μ` enters `r` **only** through `C_λ`:

```
∂C_λ/∂μ_{o,a} = λ^M_{o,a} + Σ_{p≠o} Σ_b λ^P_{op,ab}·μ_{p,b}   ≡  A_{o,a}      (draw-independent!)
∂r_w/∂μ_{o,a} = +A_{o,a}                                       (since r = … − (G_R·λ_R)_w)
∂f/∂μ_{o,a}   = (1/W) Σ_w Ψ'(r_w)·A_{o,a} = A_{o,a}·mean_m     (mean_m = (1/W)Σ_w m_w)
```

Envelope theorem at the converged `(ζ*, λ*)`:

```
   d(Delta_dual)/d(μ_{o,a})  =  −mean_m · ( λ^M_{o,a} + Σ_{p≠o} Σ_b λ^P_{op,ab}·μ_{p,b} )
```

then chain-rule through your `μ`-parameterization to the raw KNITRO coordinates.

**Compare to what already exists** (`d_delta_dual_d_eta_origin_vec`, `cm_originzc_moments.jl:300-347`):

```
   d(Delta_dual)/d(nu_{o,k})  =  −mean_m · ( λ_mean,o,k + Σ_{p≠o} nu_{p,k}·λ_pair,op,k )
```

Structurally identical, product term and all — the `nu_{p,k}·λ_pair` term is the same product rule.
So **read that function and mirror it**; do not invent a new one. The `1/L`-vs-`μ` distinction is
exactly the `nu` distinction. (Note origin-ZC's pair block is indexed by power `k`, yours by the bin
pair `(a,b)` — that is the one real indexing difference; get it right and assert it, see §4.)

### 3.2 Combined gradient

Unchanged in shape from what exists today: `vcat(g_econ, g_restriction)`, where `g_econ` is the
shared family-agnostic `economic_A_gradient!` block. Keep `pairwise_quantile_production_gradient`'s
overall structure and swap only the restriction half.

> ⚠️ **KEEP the `q0` restriction fold** (`build_lfix_base_cache_pairwise_quantile`). It is still
> required and for a reason that has nothing to do with which outer parameterization you use: `q0`
> is the per-draw *level* the economic block linearizes around, and it must include `G_R·λ_R`. This
> was the second bug of 2026-08-10; the tempting argument that "the restriction rows don't depend on
> theta" is true of the derivative and irrelevant to `q0`. See
> `feedback-q0-restriction-fold-is-a-level-not-a-derivative`.
> **Keep its exact cross-check too** — corrected `q0` must equal `verify.r_current` to ~1e-15 — it is
> free, it is exact, and it already caught a stale-bin-state bug as a by-product.

### 3.3 Delete, do not port

`pairwise_quantile_cutoff_gradient.jl` in full (`bandwidth_target`, `crossed_draw_range`,
`fixed_dual_delta_f`, `cutoff_probe_points`, `cutoff_secant_gradient!`), plus `matched_raw_steps`
and `d_delta_dual_d_cutoff_fd` in `pairwise_quantile_outer_production.jl`, plus the
matched-bandwidth logic in the FD gate. All of it exists solely to cope with a step-function
objective that version B does not have. Deleting it is the point of the exercise; carrying it along
"just in case" would keep the tuning burden you are removing.

---

## 4. Validation — the bar, and it is now HIGHER than before

Because `Delta*` is smooth in `μ`, a plain small-`h` reoptimized central FD is now a *valid* ground
truth. **Expect agreement to ~1e-5 relative or better, mirroring what `test_cm_originzc_pure_moments.jl`
holds origin-ZC's gradient to.** The old cutoff gradient could only be validated to ~20% at finite
bandwidth; if your new gradient only agrees to a few percent, something is wrong — do not accept it.

Required gates, in order:

1. **`μ`-Jacobian alone vs FD** — the parameterization only. Copy the existing
   `cutoff_jacobian_block!` FD test's pattern.
2. **Adapted D=4 dense oracle.** Take `test_pairwise_quantile_d4_dense_oracle.jl` and substitute
   `μ`-centering for `1/L`. `forward!` vs dense `G·λ`; `transpose!` vs dense `−(1/W)Gᵀw`; the
   centered Hessian block vs an independent centered-dense reference.
   ⚠️ Its `r_current`/`psi_scalar` convention was **corrected on 2026-08-10** to match production
   (`r_current = r0`, `psi_scalar.(a0)`, no negation). Do not "restore" the negations — that
   reintroduces a blind spot that hid a real sign bug.
3. **Version-A ↔ version-B equivalence anchor (do this — it is nearly free and very strong).** Set
   the fixed cutoffs to the empirical quantiles of the calibration draws AND pin `μ_{o,a} = 1/L`.
   The moment matrix is then *identical* to version A at its own start point, so `Delta*` must match
   version A's to machine precision. Real D=20, W=20,000, L=3 gives `Delta_dual = 0.005235`
   (`VerifiedSolved`, 4.4 s) as the recorded reference; W=100,000 gives 0.000706 at L=5 and 0.003011
   at L=10. Any mismatch means a centering/indexing error.
4. **Closed-form gradient vs reoptimized FD**, every probe re-solving the inner dual from scratch —
   the standard this repo already holds origin-ZC to. Mirror `d_delta_dual_d_eta_origin_fd` plus
   `test_cm_originzc_pure_moments.jl`'s testset, h-shrinking protocol included. Validate the
   **combined** `vcat(g_econ, g_μ)` vector, not just the restriction half, and keep the `q0`
   exactness check and its without-the-fold control.
   Include a **negative control** (e.g. sign-flipped, or the product term dropped) that must fail —
   a gate nobody has watched fail is not known to be a gate.
5. **Real D=20 end-to-end smoke** through the driver. Adapt
   `smoke_pairwise_quantile_outer_driver.jl`; it already checks the family-contract return shape,
   checkpoint round-trip, resume, hard-error resume guards, `objective_mode=:min_delta_fixed_gp`, and
   `UndefKeywordError` on every omitted scientific kwarg. **Keep its "did the restriction block
   actually MOVE" check and its `n_eval >= 1` guard** — without that guard it false-passed when every
   point was rejected.

---

## 5. Production outer loop

`pairwise_quantile_checkpoint.jl` already implements the full 23-step checkpointed driver
(mirroring `run_originzc_upper_checkpointed`) and its smoke passes at real D=20. Changes needed:

- Outer vector stays `w = vcat(gp, zfree, <restriction coords>)`; the tail is now `μ`-coords.
- Bounds: replace `default_raw_cutoff_bounds` with the `μ`-parameterization's own box. Follow that
  function's discipline — data-derived with a documented margin, never a hand-tuned magic constant.
- `PairwiseQuantileCheckpointV1` → a **new, uniquely-named** schema. It must record the FIXED CUTOFFS
  (the run is not reproducible without them) and how they were chosen, plus the `μ` coords, and drop
  `min_crossed`. ⚠️ `Serialization.deserialize` resolves types by NAME and this repo has been bitten
  by two different structs sharing one name — grep the whole tree before choosing it.
- Resume guards: hard-error on a cutoff-vector mismatch exactly as the current driver hard-errors on
  an `L` mismatch (different cutoffs = a different restriction, not a different search).
- `cb_G!`: closed form; no bandwidth cache for the restriction block (the economic block still needs
  its own).
- Family-#6 registration is already wired in code (`pairwise_quantile_family_spec`,
  `build_family`/`evaluate_family`, `paper_six_family_seed_specs`, `call_driver`). Update the spec's
  fields (`min_crossed` → whatever version B needs) and `evaluate_family`'s starting-value logic:
  the analog of "empirical-quantile cutoffs" is now a starting `μ`, and `μ = 1/L` is the natural one.
  Keep `nu_policy` honest about which it is.
- **Do not touch `protocols/paper_upper_v1.toml`** (frozen; live campaign running). That decision
  stands until the user revisits it.

---

## 6. What NOT to do

- Do **not** keep the cutoff-secant machinery "as a fallback". Delete it; it is the thing being
  replaced.
- Do **not** make `μ` common across origins — that adds a marginal restriction.
- Do **not** re-negate the D4 oracle's `r_current`/`psi_scalar` (see §4.2).
- Do **not** drop the `q0` fold or its exact cross-check (§3.2).
- Do **not** reimplement the economic `(g, A_od)` gradient, and do **not** finite-difference it at
  small `h` and treat disagreement as a bug — it is an adaptive bandwidth-selected secant whose own
  docstring says a sub-`h_floor=1e-4` probe reproduces a "known-wrong" gradient. Gate it by
  bit-identity against `composite_gradient_at_fast`, as the current gate does.
- Do **not** expect a speedup (§2.3), and do **not** reach for `pairwise_quantile_hvp.jl` as the
  L=10 performance fix — it was measured at real D=20/W=100k and rejected (4.4× slower, CG needs
  6084 calls vs dense's 9). Read `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md`
  Part 1 before forming any plan that involves it.
- Do **not** push any branch to a remote without asking.
- Per CLAUDE.md: push the session's deliverables to Dropbox under a new dated subfolder before
  finishing, and check any long-running job with `ps` + `tail` within ~60 s of launching it.
