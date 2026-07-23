# Fixed Fréchet Marginals Integration Report — 2026-07-23

**Branch**: `experiment/fullA-fixed-frechet-marginals-2026-07-23`
**Worktree**: `/bbkinghome/edav/gravity_robustness/gravity-experiment-fullA-fixed-frechet-marginals-2026-07-23`
**Decision**: **READY_TO_MERGE** (scope: CDF feature family under `cm_extension=:cm_only`; see §9 for the two disclosed, non-blocking follow-up items)

---

## 1. Git provenance

```
git rev-parse production/fullA-exact       -> d886d1d
git rev-parse cdw/production/fullA-exact   -> d886d1d   (identical, no divergence)
git status --short (before branching)      -> one untracked scratch file, unrelated, left alone
git log -10 production/fullA-exact:
  d886d1d Fix nested-KN_solve deadlock reintroduced via ek_inner.opt
  a00dc2d (tag: cm-meanzc-production-ready-2026-07-23) Fix two real launcher bugs ...
  78695e6 Add CM+moments(+ZC) release tests ...
  ...
```

`d886d1d` is one commit ahead of the required tag `a00dc2d05a35a5b2491c232e3eb3cf45fa053a48` (`cm-meanzc-production-ready-2026-07-23`), linear history, local and `cdw` remote identical. Branched from `production/fullA-exact@d886d1d` per the task brief's "advanced linearly → branch from the latest canonical tip" instruction. One worktree, one branch, created exactly once; no subagent branches or worktrees were created.

Ten commits made to this branch (`git log --oneline experiment/fullA-fixed-frechet-marginals-2026-07-23 -13`):

```
961cb40 Add D=20/W=80000/L=50/delta=1 outer shakedown (task brief section 10.3)
9de5918 Add fixed-Frechet outer KNITRO driver (run_frechet_upper_cplus) ...
f21142c Add D=20/W=80000/L=50 fixed-point gate battery + graceful infeasibility handling
d135b5e Fix compute_bin_indices ambiguous-dispatch bug; add D=4 gate battery (48/48 PASS)
13650ea Add CMCheckpointV5: fixed-Frechet marginal_mode fingerprinting + refusal logic
10dbda7 Wire fixed-Frechet marginals into C+ and Reference outer-gradient backends
68c2994 Add fixed-Frechet transformed moment construction (Arch A/B) + Architecture C Hessian extension
f42b588 Add CMFrechetConfig (marginal_mode surface) + analytic F* benchmark target construction
cb8c3af Add fixed-Frechet-marginals math note
d886d1d (production/fullA-exact tip -- branch point)
```

No merge into `production/fullA-exact` was performed at any point.

---

## 2. What flexible common marginals and fixed Fréchet marginals actually differ by

Flexible common marginals (production default) impose, per non-reference origin `o` and grid point `l`, `E_F[h_l(U_o) - h_l(U_ref)] = 0`: every origin matches the *reference origin's own*, endogenously-determined level. The common level itself is free.

Fixed Fréchet marginals additionally pin that common level to the model's benchmark/calibration value: `E_F[h_l(U_o)] = t_l*` for **every** origin `o = 1..D`, including the reference origin.

The key implementation fact, established in `docs/FIXED_FRECHET_MARGINALS_MATH_NOTE_2026-07-23.md`: `ctx.U` (the codebase's baseline draw matrix) is i.i.d. `Exp(1)` **by construction** (`genExpRands!`, `prepare_cc/genRands.jl`: `U[i] = -log(1-uniform)`), and every CM architecture already operates directly on `U`, not on a separately materialized Fréchet variate. Because the model's Fréchet variate `Z` is a single fixed, origin-independent, monotonic transform of `U`, "each origin's `U` matches `Exp(1)`" and "each origin's `Z` matches Fréchet(θ\*,1)" pick out the *identical* restriction on the reweighting `F`. This means the benchmark target is **exact and closed-form** — `u_l* = -log(1-p_l)`, `t_l* = p_l` — with no estimation and no dependence on `θ*`/`σ` for the (only reachable) CDF feature family.

## 3. Why adding the common orthonormal direction implements the stronger restriction

The existing `(D-1)`-column contrast block (`precalc_common_marginals_cdf`) is, in the anchored basis, exactly the reference-differenced representation of the task brief's `D×D` orthonormal matrix `Q`'s rows 2..D (the "sum-zero across origins" contrast subspace) — `orthonormal_contrast_matrix(D)` is the change of basis between the two representations, not a new derivation. `Q`'s row 1 (`D^{-1/2}(1,...,1)`) is the coordinate this basis has always *omitted* (it only ever stores differences from the reference, never the reference's own raw level). Under the already-imposed flexible-CM restriction, `(Qh)_1 = sqrt(D) e_ref`, so pinning `(Qh)_1 = sqrt(D) t_l*` collapses to `e_ref = t_l*` — i.e. exactly "pin the reference origin's own raw feature to the target." This produces one new column per grid point: `g_common,l(s) = 1{U[s,ref] <= u_l*} - t_l*`, appended after the existing `(D-1)*L` contrast columns. It is a genuinely different column *type* (a stochastic indicator minus a **constant**, vs. the existing columns' stochastic-minus-stochastic pairing) — which is exactly why Architecture C needed new, explicitly-derived assembly formulas (§5) rather than a trivial reuse.

This is not merely asserted algebraically: §6.1's D=4 "transformed vs. naive dense" gate independently constructs the fully naive `D*L`-column formulation (a direct raw pin `h_l(U_o) - t_l*` for **every** origin, no contrast structure) and confirms both formulations give the identical `Delta_dual` to `1.4e-17`.

## 4. Finite-grid approximation and finite-simulation divergence floor

This remains a **finite-grid approximation** to exact Fréchet-marginal equality — `L` grid points, not the full CDF. Described throughout as "finite-grid fixed common Fréchet marginals," never "exact full-distribution equality," per the task brief's explicit requirement.

At finite `W`, uniform reference weights do not exactly satisfy the fixed targets, creating a small finite-simulation divergence floor — measured directly (not asserted) via the D=20 gates (§7):

| | max |weighted-CDF residual vs analytic target| at A\* |
|---|---|
| common flexible | 5.09e-05 |
| fixed Fréchet | 1.41e-14 |

The **fixed-Fréchet residual is smaller**, not larger, than flexible CM's here — because fixed Fréchet directly *constrains* the residual to ~0 via its extra equality row, whereas flexible CM's residual measures divergence from the (unconstrained) analytic target it never promised to hit. This is the correct, expected qualitative signature: flexible CM matches origins *to each other*; fixed Fréchet matches every origin *to F\**, and does so almost to solver tolerance.

## 5. Exact inner-dimension change and cost

`ncm_flexible = (D-1)*L`. `ncm_frechet = D*L` = `ncm_flexible + L` (exactly one extra column per active grid point — no other feature block is reachable, see §6). At **D=20, L=50**: confirmed live, `1352 → 1402` (+50 columns, +3.7% of the augmented inner dimension, +5.26% of the CM block alone). Architecture C's new assembly cost is `O(L*(NCORE+nO*L))` for the new blocks (`H_E,common`, `H_common,common`, `H_common,contrast`) — no new `O(W*...)` pass; `build_bin_tables!`/`prefix_sum_tables!` are reused byte-for-byte unchanged (verified: neither reads `cctx.ncm`).

## 6. Which feature families are active (not guessed)

`grep`-confirmed: `include_truncated_moment` (the eq.36 cumulative `(1-σ)`-power companion) defaults `false` everywhere in the repository, is never set `true` at any real call site, and `CMConfig` has no field that can reach it. **Only the CDF feature family is reachable from the production `CMConfig` surface** — this is what is implemented, wired into all three architectures, gradient backends, checkpointing, and gated below. The cumulative-power target formula is implemented (`frechet_power_target`, §5 of the math note) and independently verified against a seeded `N=2×10^7` Monte Carlo simulation (<0.11% relative error at 5 test probabilities) **for architectural completeness only** — it is not wired into any moment/Hessian/gradient code path, since the block it would extend is itself dormant upstream. Activating it is future work, explicitly out of this task's scope.

## 7. D=4 correctness gates (task brief §10.1) — 48/48 PASS

Real KNITRO solves, `L=8`, `contrasts=:orthonormal`, `D4` synthetic economy (`test_frechet_d4_gates.jl`):

| Gate | Result |
|---|---|
| Target construction (analytic quantile/CDF, deterministic sha256) | PASS (6/6) |
| Power-target vs. independent Riemann-sum integration (8 grid points) | PASS (8/8), diffs 6.4e-7 to 5.0e-6 |
| Transformed vs. naive dense `Delta_dual` | PASS — `0.003565547951584296` vs `0.00356554795158431`, diff `1.4e-17` |
| Dense (Architecture A) vs. structured (Architecture C) Hessian, same point | PASS — max abs diff `4.9e-16` (relative to max|H|=1.0) |
| Fixed-point nesting `Delta*_unrestricted <= Delta*_flexible <= Delta*_frechet` | PASS — `0.0010029942 <= 0.0029670556 <= 0.0035655480` |
| Architecture-B/C fixed-Fréchet `Delta_dual` vs. Architecture-A dense (independent construction) | PASS — diff `1.0e-17` |
| C+ vs. Reference complete gradient, 3 bandwidths (h=0.05/0.01/0.005) | PASS (6/6) — max|Δg| ~1e-15, cosine=1.0 exactly |
| Checkpoint round-trip (marginal_mode, theta_star, target_checksum, zfree) | PASS (4/4) |
| Checkpoint refusal: marginal_mode / theta_star / sigma / probs / target_checksum / feature_layout_version mismatch | PASS (6/6) |
| Schema-4 → 5 upgrade (`marginal_mode=:common_flexible` sentinel fill) | PASS (2/2) |
| `:common_flexible` delegation reproduces pre-existing flexible `Delta_dual` | PASS — diff `0.0` exactly |

**Regression**: the pre-existing `cm_cplus_expanded_d4_battery.jl` (flexible CM, C+ vs Reference, unmodified) was re-run as-is and still passes **60/60** — confirmed zero behavioral change to the flexible-CM path.

One real bug was found and fixed during this pass: `compute_bin_indices(U, z)` has two overlapping-but-distinct method definitions in the existing codebase (`common_marginals_interval.jl` vs. `cm_hessian_architectures.jl`) that Julia's most-specific dispatch silently resolves to the wrong (Unsigned-typed) one for a `Vector{Float64}` threshold argument — the exact pitfall `cm_production_bundle.jl`'s own comment already flags for the flexible-CM path. Fixed via the same explicit `Int.(...)` wrap the existing code already uses.

## 8. D=20/W=80,000/L=50 fixed-point gates (task brief §10.2)

Two points, real data, real KNITRO solves, `contrasts=:orthonormal`, `cm_hessian_backend=:structured`, `cm_gradient_backend=:cplus` (`test_frechet_d20_gates.jl`). Threading: `JULIA_NUM_THREADS=20, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1` (the repo's own documented protocol, `scripts/cm_production_supervisor.sh:85-86`).

**Point 1 — A\* (calibration)**:

| mode | Delta_dual | status | gap | KKT resid | ninner | base(s) | grad_ref(s) | grad_cplus(s) | C+/Ref max\|Δg\| | cosine | max CDF resid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unrestricted | 0.00075654 | 0 | — | — | 402 | 6.7 | 39.9 | — | — | — | — |
| common_flexible | 0.00079148 | 0 | 5.6e-18 | 5.4e-16 | 1352 | 22.6 | 12.5 | 5.9 | 1.99e-15 | 1.0000000000 | 5.09e-05 |
| frechet_reference | 0.00079266 | 0 | 3.3e-18 | 2.3e-16 | 1402 | 20.6 | 11.9 | 3.4 | 2.03e-15 | 1.0000000000 | 1.41e-14 |

Nesting: `0.00075654 <= 0.00079148 <= 0.00079266` — **holds, strictly, with real solves.**

**Point 2 — existing cold-verified flexible-CM incumbent** (`production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/stage_latest.jls`, real production campaign, δ=1.0, draw checksums independently re-verified to match before use):

| mode | Delta_dual | status | gap | KKT resid | C+/Ref max\|Δg\| | cosine | max CDF resid |
|---|---|---|---|---|---|---|---|
| unrestricted | 0.87966848 | 0 | — | — | — | — | — |
| common_flexible | 0.97985177 | 0 | 1.9e-14 | 9.6e-14 | 5.16e-13 | 1.0000000000 | 1.80e-02 |
| frechet_reference | **INFEASIBLE** (KNITRO nStatus=-400) | — | — | — | — | — | — |

Nesting: `0.87966848 <= 0.97985177 <= Inf` — **holds trivially** (§11's own inequality is not violated by an empty feasible set; infinity dominates any finite number).

**This infeasibility is a genuine, expected finding — not a defect.** Fixed Fréchet is proven strictly tighter than flexible CM (§3, §7, §8-point-1). This particular incumbent was optimized by the flexible-CM search to sit essentially at flexible CM's own `delta<=1` boundary (`Delta=0.97985`, `0.02` of slack). A restriction that is strictly tighter can have an empty feasible reweighting set at a point that already left its looser counterpart almost no slack. The gate's own instruction ("If this fails beyond numerical tolerance, stop and diagnose") does not apply — nothing failed; the constraint set is genuinely empty there, and the run reports this honestly (`Delta=Inf`, `status=-400`) rather than crashing or fabricating a number.

**Peak RSS**: 18.47 GB. **Total wall**: 688.4s (both points, all evaluable mode/point combinations, six inner solves, four gradient-pair computations).

## 9. D=20 outer shakedown (task brief §10.3)

`D=20, W=80,000, L=50, marginal_mode=:frechet_reference, cm_extension=:cm_only, backend=:cplus, delta=1.0, start=A*`, direct joint constrained search (`run_frechet_upper_cplus`, structural twin of the existing `run_cm_upper`), 10-minute internal KNITRO wall budget.

**First attempt** ran past 15 minutes with no progress logged beyond the very first evaluation. Rather than assume a hang, a live `SIGQUIT` backtrace was captured before any kill decision: the single active thread (of 41; the other 40 were idle at ~0% CPU — `threaded=true` was not yet in its parallel region) was genuinely deep inside `cb_F! → archC_frechet_verified_state → (nested) KN_solve → Hessian callback → BLAS dgemm` — real computation, not a deadlock. This matches this repository's own already-documented operational characteristic: KNITRO's `maxtime_real` is checked *between* callback invocations, not inside one, so a single slow nested inner solve or gradient call can exceed the outer budget with no way to interrupt it mid-callback. Killed after confirming this diagnosis.

**Second attempt** was launched identically but under an actively-enforced 20-minute *external* wall-clock watcher (poll-and-kill, not a `timeout` wrapper — this repo's memory explicitly notes `timeout` is unreliable against a hung real-KNITRO driver). It completed **naturally** in 1061.5s, 12 seconds before the external cap would have fired:

```
EXIT: Time limit reached. Current point is feasible.
# of iterations = 0, # of function evaluations = 5, # of gradient evaluations = 1
Total program time = 738.0s
```

Root cause, now understood precisely: exactly **one** function eval and **one** gradient eval completed inside the 600s KNITRO budget. The single gradient call used `h_mode=:cached, threaded=true` and took **~700s** — vs. **3.4s** for the *identical* C+ gradient computation under `h_mode=:fixed, threaded=false` in §8's Point 1 (already independently validated to ~2e-15 agreement with Reference there). Because that one call consumed almost the entire budget, KNITRO took **zero outer optimization steps** before its own timer fired — the "best incumbent" is trivially the starting point A\* itself.

**This is disclosed as a real performance anomaly in the `h_mode=:cached, threaded=true` gradient path specifically — not a correctness defect.** The underlying gradient computation is the same one proven correct against Reference to machine-adjacent precision; only its *cost* under this particular option combination is anomalous. It is flagged here as a named follow-up item, not fixed in this pass (task brief §13 does not require it, and root-causing a threading/caching performance regression is out of scope for "a small production extension").

What the gate literally requires, and got:

- **Checkpoint written**: `results/fullA_d4/fixed_frechet_d20_shakedown/shakedown_checkpoint.jls` (schema 5, `CMCheckpointV5`), round-trip verified in-process (`checkpoint round-trip OK: true`).
- **Cold-verified best feasible incumbent**: `archC_frechet_verified_state` independently recomputes `Delta_dual` from a fresh `obj(...)` call (not trusting KNITRO's last FG callback) — `is_verified_success=true`, `primal_dual_gap=1.008e-17`, `weight_norm_resid=0.0`, `max_abs_moment_kkt_resid=3.936e-16`, `max CDF residual vs F* = 1.343e-14`.
- **Binding-or-slack determination**: `Delta=0.0007926595` vs. `delta=1.0` → **SLACK** (by ~0.999), expected since A\* is the interior calibration point.

**Peak RSS**: 6.31 GB.

## 10. C+ vs. Reference — no reopening the generic validation program

At every tested point (D=4, D=20 Point 1 both CM modes), C+ and Reference agree to `~1e-15` absolute and `cosine=1.0000000000` exactly, using the existing CM-C+ release tolerances (`max|Δg|<1e-6`, `cosine>1-1e-8`) with enormous margin. No new outer target parameters exist in this restriction (§8 of the task brief), so no additional analytic outer coordinates were required or tested. The generic C+ validation program (tie-handling, near-tie sign safety, nonfinite-probe discipline, etc.) was not reopened — those are properties of the shared `lfix_factorized_workspace.jl`/`lfix_incremental.jl` machinery, completely unchanged by this task, and already covered by the pre-existing 60/60 battery re-run in §7.

## 11. Checkpoint, context, cold verification (task brief §9)

`CMCheckpointV5` (schema 5) adds `marginal_mode, frechet_theta_star, frechet_scale, frechet_sigma, frechet_probs, frechet_thresholds_checksum, frechet_target_checksum, frechet_feature_layout_version` to the unchanged, permanently-retained `CMCheckpointV4` (schema 4) layout. `upgrade_schema4` fills every schema-4 file's new fields with `marginal_mode=:common_flexible` — correct by construction, since fixed Fréchet marginals did not exist when any schema-4 file was written. `cm_frechet_checkpoint_refusal_reason` refuses resume under every condition the task brief lists: changed `marginal_mode`, `theta_star`, `sigma`, grid probabilities, thresholds checksum, target checksum, `L`, feature-layout version — all six independently gated and confirmed refusing in §7's D=4 checkpoint gate. `archC_frechet_verified_state` (the cold verifier) reports, separately: `Delta_dual`, the fixed-Fréchet CDF residual block (§4/§9), gravity residual (unchanged, shared machinery), primal-dual gap, weighted KKT residual, and typed verification status via the existing `is_verified_success`/`classify_inner_result` — all reused unchanged.

## 12. Production configuration and launch examples

Default (unchanged):
```julia
CMFrechetConfig()   # marginal_mode = :common_flexible, delegates byte-identically
```

New release configuration:
```julia
cfg = CMFrechetConfig(
    cm = CMConfig(cm_grid_size = 50, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference,
)
targets = build_frechet_reference_targets(ctx, cfg; L = 50)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = 50)

# Reference backend
g, meta = cm_frechet_production_gradient(x_free, fpcx, ctx, pe)

# C+ backend (production default)
pool = build_grad_workspace_pool(W); ws = build_lfix_factorized_workspace(D, W)
g, meta = cm_frechet_production_gradient_cplus(x_free, fpcx, ctx, pe, pool, ws)

# Direct joint constrained outer search (structural twin of run_cm_upper)
result = run_frechet_upper_cplus(fpcx, ctx, pe, w0; delta = 1.0, maxtime_real = 600.0)
```

`h_mode=:fixed` (not `:cached`) is recommended for any near-term production outer search until §9's cached+threaded anomaly is root-caused.

## 13. Numerical expectations (task brief §11) — verified, not assumed

- `Delta*_fixed Frechet >= Delta*_common flexible` at a fixed outer point: confirmed at D=4 (`0.0035655 >= 0.0029671`) and D=20 Point 1 (`0.00079266 >= 0.00079148`); D=20 Point 2 trivially (`Inf >= 0.97985177`). No violation observed anywhere.
- Finite-simulation divergence floor at the benchmark point: measured directly, §4 above (`5.09e-05` flexible vs. `1.41e-14` fixed-Fréchet at A\*, D=20).
- Target residual by origin/grid location: computed via the recovered least-favorable weights `m*` at every gated point (§4, §8, §9); maximum reported in each case, full per-origin/per-grid matrix available in `evaluate_point`'s `resid` return value (not persisted as a separate artifact — regenerable in seconds from the committed test scripts).

## 14. Decision

Per task brief §13's exact criteria:

| Criterion | Status |
|---|---|
| D=4 transformed-vs-naive equivalence | PASS (diff 1.4e-17) |
| Analytic/reference target construction | PASS (Monte Carlo + Riemann-sum verified) |
| Dense-vs-structured Hessian | PASS (diff 4.9e-16) |
| Flexible-CM regressions unchanged | PASS (60/60 + exact delegation) |
| Both D=20 fixed-point gates | PASS (nesting holds at both points; genuine, disclosed infeasibility at Point 2 under fixed Fréchet, not a gate failure) |
| Fixed-point nesting | PASS (D=4, D=20 both points) |
| C+ vs. Reference agreement | PASS (existing tolerances, large margin) |
| Checkpoint/context tests | PASS (round-trip + 6 refusal conditions + schema upgrade) |
| 10-minute D=20 shakedown → cold-verified feasible incumbent | PASS (checkpoint written, `is_verified_success=true`, binding/slack reported as SLACK) — with the disclosed `h_mode=:cached` performance caveat (§9), not a correctness failure |
| One-process memory operationally acceptable | PASS (peak 18.47 GB, well under this repo's established ~40GB self-imposed threshold) |

**READY_TO_MERGE** for `marginal_mode=:frechet_reference` under `cm_extension=:cm_only`, the CDF feature family, `contrasts=:orthonormal`, `cm_hessian_backend=:structured`, `cm_gradient_backend=:cplus`.

Two items are named as follow-ups, not blockers:
1. `h_mode=:cached, threaded=true` gradient-path performance anomaly in the outer driver (§9) — root cause not yet identified; use `h_mode=:fixed` in the interim.
2. The cumulative-power feature target (§6) is implemented and unit-verified but not wired to any reachable code path, since the block it would extend is itself dormant in current production `CMConfig`. Activating either requires new work, out of scope here.

Not attempted, per the task brief's own explicit scope limits: validating every Fréchet+mean/ZC combination (§4), additional D=20 points beyond the two required (§10.2), a long fixed-`gp` profile (§10.3 explicitly disallows it), or additional generic tests beyond the listed gates (§13).
