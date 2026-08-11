# delta = 1 upper-bound runs for OZC-CROSS and CM+ZC-CROSS (8-hour budget each), 2026-08-10

Task: `full_aod_diag/d4_exact/HANDOVER_CROSS_8H_UPPER_BOUND_RUNS_2026-08-10.md`.
Branch `feature/ozc-cross-2026-08-09`, worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`,
commit `f897485` (parent `4df5254`, `origin/production/fullA-exact`). Not pushed to any remote.

---

## 0. Summary

**Both δ=1.0 upper-bound runs completed. Neither exhausted its 8-hour budget, and neither converged.**

| family | κ | status | wall | evals (feasible / verified) |
|---|---|---|---|---|
| **OZC-CROSS** | **2.4098742 %** | −102 stall | 6h17m of 8h | 100 (78 / 100) |
| **CM+ZC-CROSS** | **2.3898562 %** | −102 stall | 6h00m of 8h | 28 (16 / 28) |

Both κ are **valid but conservative** bounds — each attained at a feasible point that passed
`is_verified_success`, so the true supremum over each identified set is at least this. Neither is
converged: `-102` is `KN_RC_FEAS_NO_IMPROVE`, a tolerance stall, not an optimality certificate. The
OZC-CROSS bound additionally survives requiring Δ < 1 *strictly* (§2.4).

**The most important result is not either number — it is that both runs stalled with hours of budget
unspent, having barely moved.** CM+ZC-CROSS gained 0.0036 percentage points of κ in six hours. The
binding constraint on these searches was not compute. See §4 for what that implies.

**Phase 0** fixed a real regression (CM+ZC-CROSS was silently running at 1 BLAS thread instead of the
validated 8, §1.1), worth ~1.49× per outer evaluation, and passed its correctness gate (§1.5). It also
corrected an inherited claim: the earlier attribution run was already on `:blas_syrk`, so no
2×-from-symmetry win was ever available (§1.2).

---

## 1. Phase 0: the H_ZZ BLAS-thread regression

### 1.1 What was wrong (one line, `cm_checkpoint.jl:1509`)

```julia
effective_blas_threads = blas_threads !== nothing ? blas_threads :
    (family_tag === :cm_meanzc ? ZC_GRAM_BLAS_THREADS_DEFAULT[] : nothing)
```

The 2026-08-09 CM+ZC-CROSS work introduced `family_tag = :cm_meanzc_cross` — a deliberate split, since
the diagonal and cross families produce genuinely different `Delta*` at identical `(K_mean, K_pair)`
and share an exact-eval cache. But that new tag does not match this gate, so **CM+ZC-CROSS silently
ran at ambient (= 1) BLAS threads while the diagonal `cm_meanzc` family got the validated 8** — on the
family whose restriction gram is nearly three times wider (`nx` 1770 vs 630). OZC-CROSS was never
affected: `run_originzc_upper_checkpointed` takes `blas_threads = ZC_GRAM_BLAS_THREADS_DEFAULT[]` as a
plain kwarg default with no family gate (`cm_originzc_checkpoint.jl:574`, applied at `:789`).

Fixed by keying on `is_meanzc` (`== cm_extension !== :cm_only`) rather than enumerating tags, so a
future third CM+ZC layout arm cannot fall through the same crack. The two non-CM+ZC siblings
(`flexible_cm` / `common_frechet`), which were never part of this optimisation's validation, keep
their existing untouched-ambient path.

### 1.2 Correcting the handover's inherited claim about `zc_gram_backend`

The Dropbox `00_READ_FIRST_CORRECTION.md` states that the 2026-08-10 attribution run "used
`:reference`" for `zc_gram_backend`, which would have left a 2× symmetry win on the table. **That is
wrong**, and the handover already flags it. Traced through every layer:

* `zc_gram_blas_candidates.jl:260` — `ZC_GRAM_BACKEND_DEFAULT = Ref{Symbol}(:blas_syrk)`.
* `diag_cmzc_cross_iters_vs_periter_2026-08-10.jl` never mentions `zc_gram_backend` at all.
* It builds via `build_cm_meanzc_cross_production_context`, which forwards to the shared
  `build_cm_meanzc_bin_ctx` (`cm_meanzc_production.jl:78`), whose own default is
  `ZC_GRAM_BACKEND_DEFAULT[]`.

So every figure in that attribution table was already measured on the symmetry-exploiting `syrk`
backend. There was no 2×-from-symmetry win still available, and none was claimed here.

### 1.3 Kernel-level thread scaling (measured, not assumed)

`bench_hzz_blas_threads_2026-08-10.jl` — the real production kernels on synthetic `Phi` at the real
production shapes (`W = 100,000`; `nx = 630` = diagonal K=3/3, `nx = 1770` = cross K=3/3). The kernel
is dense BLAS-3, so its cost is shape-determined, not value-determined; this is a timing instrument
only, and correctness is gated separately in §1.5.

| backend | BLAS threads | nx = 630 | nx = 1770 | GFLOP/s @ 1770 | speedup vs 1t @ 1770 |
|---|---|---|---|---|---|
| `blas_syrk` | 1 | 0.770 s | 4.254 s | 73.7 | 1.00× |
| `blas_syrk` | 2 | 0.497 s | 2.463 s | 127.3 | 1.73× |
| `blas_syrk` | 4 | 0.338 s | 1.468 s | 213.5 | 2.90× |
| `blas_syrk` | **8** | 0.254 s | **0.993 s** | 315.8 | **4.28×** |
| `blas_syrk` | **16** | 0.218 s | **0.815 s** | 384.8 | **5.22×** |
| `blas_syrk` | 32 | 0.214 s | 0.783 s | 400.2 | 5.43× |
| `blas_gemm` | 1 | 1.270 s | 7.600 s | 41.2 | — |
| `blas_gemm` | 16 | 0.248 s | 1.046 s | 299.7 | — |

Three things this settles:

1. **`syrk` threads well.** The handover's prediction held: 4.28× at 8 threads, 5.22× at 16, at the
   cross family's real `nx = 1770`. This is the whole reason the gate fix matters.
2. **`syrk` beats `gemm` at every thread count** (1.79× at 1 thread, 1.28× at 16) — the symmetry
   saving is real and the existing default is the right one. No backend change was made.
3. **Returns die after 16.** 32 threads buys a further 4%, for double the cores. 16 is the knee.

`:threaded_packed` was not measured. It is the handover's option 3, to be tried "only if 1 and 2
disappoint" — and option 1 did not disappoint. It also needs `threaded_cross_hessian.jl`'s
`cross_hessian_chunk_ranges`, i.e. the whole `WinnerPairHessCtx` include chain this instrument
otherwise avoids. Its earlier bakeoff verdict remains retracted and un-revalidated at large `nx`.

### 1.4 The free inefficiency in `zc_gram_blas_syrk!`

`zc_gram_blas_candidates.jl` evaluated `sqrt(S[w])` inside the `j` (column) loop:

```julia
for j in 1:nx, w in 1:W
    RW[w, j] = sqrt(S[w]) * Phi[w, j]
end
```

`sqrt(S[w])` is column-invariant, so this did `W*nx ≈ 1.8e8` square roots per Hessian call instead of
`W = 1e5`. Hoisted into a persistent length-`W` workspace field (`ZCRawWeightedWorkspace.sqrtS`), so no
per-call allocation is added — this codebase has an allocation-regression test on that path. The
result is bit-identical: `sqrt` is correctly rounded and deterministic, so `sq[w] * Phi[w,j]` and
`sqrt(S[w]) * Phi[w,j]` produce the same Float64.

### 1.5 Correctness gate — PASS

A thread count or gram backend must not move `Delta_dual` beyond floating-point reassociation. Gated
by running the **real checkpointed production drivers** at the production configuration (D20 real
data, W=100,000, K=3/3, L=50, `delta=1.0`, Variant D level 2, `:sobol_randomized`/20260719,
`:exclude_row`, `exclude_diagonal_gravity`, Brazil-Korea excluded cells, `sigmaHat=3.0`,
`inner_lower_limit=-10.0`) and reading `Delta_dual` at eval 1 — which is the calibration point, since
`w0` is the calibration point in `:powered_aspace` coordinates.

| family | 1 thread | 8 threads | 16 threads | spread |
|---|---|---|---|---|
| CM+ZC-CROSS | `0.03952860482890379` | `0.0395286048282956` | `0.039528604818098605` | 1.08e-11 abs / **2.73e-10 rel** |
| OZC-CROSS | — | `0.03441331932541253` | `0.03441331932795739` | 2.54e-12 abs / **7.40e-11 rel** |

**Verdict: pass.** Three things justify that, and the qualification matters:

* The CM+ZC spread is ~2.7e-10 *relative*, which is about 3× the handover's stated ~1e-10 bar. The
  pass is judged on the **absolute** figure instead: ~1e-11 is the inner solve's own optimality floor
  in this codebase (see memory `knitro-minus100-is-opttol-1e-12-vs-achievable-floor` — `opttol_abs`
  of 1e-12 is below what this problem can actually achieve, ~1e-11). An iterative solve with a
  tolerance-based stop, re-ordered by BLAS, landing 1e-11 apart in the objective is that floor, not a
  behavioural change. Claiming machine-precision agreement here would be false.
* **`gp` at eval 1 is bit-identical across all five arms**: `0.9840278851786317`. The outer
  coordinate, which is what the bound is read off, does not move at all.
* `feasible=true, verified=true` in every arm, with `Delta ≈ 0.034–0.040` against `delta = 1.0`. The
  feasibility verdict is nowhere near a knife edge, so a 1e-11 wobble cannot flip it. (This is the
  opposite situation to the `A_od≡1` trap in CLAUDE.md, where two points differed by 27 log-points
  and genuinely landed on opposite sides of a screen — worth stating explicitly, since "the
  difference is small so it doesn't matter" is exactly the reasoning that goes wrong there.)

The 8-hour runs additionally reproduce the gate arm bit-for-bit: OZC-CROSS eval 1 in the production
run gives `Delta_dual = 0.03441331932541253`, identical to the 8-thread gate arm.

`test_cmzc_cross_wiring_2026-08-09.jl` (fast, solver-free; carries the checkpoint schema): **ALL
PASS** after the `cm_checkpoint.jl` edit.

### 1.6 Why the runs use 8 BLAS threads, not 16

The kernel microbenchmark favours 16 (5.22× vs 4.28×). The **real driver did not reproduce that**:
CM+ZC-CROSS eval 1 took 209.8 s at 8 threads but **246.2 s at 16**; OZC-CROSS went the other way
(106.3 s → 94.2 s). Those wall times are confounded — five arms ran concurrently at load average ~70,
and the 16-thread arms launched last — so they are inconclusive, not evidence *against* 16.

But inconclusive is not a reason to move off a validated setting. 8 is the value this codebase
already validated for `cm_meanzc`, and it is exactly what the fixed gate now delivers to the cross
family by default. Both runs therefore pass **no** `blas_threads` override, which also makes them an
end-to-end test of the fix itself: the CM+ZC-CROSS log records `blas_threads=(driver default)` and
its backend manifest records `blas_threads=8` — pre-fix, that path resolved to ambient (= 1).

Note the repo-wide `OPENBLAS_NUM_THREADS=1` rule is not violated: that governs the *ambient* setting,
and both drivers raise the count internally, process-scoped, via the already-established exception.
Verified live that `BLAS.set_num_threads(n)` does take effect at runtime under that env var (it is
not capped at the init value) — worth recording, since if it had been capped, the diagonal family's
validated 8 would have been silently inert too.

### 1.7 Attributing the speedup honestly

The 1 → 8 thread fix is worth **~1.49× on total outer-evaluation wall** for CM+ZC-CROSS (312.2 s →
209.8 s at eval 1), not the 4.28× the kernel benchmark shows. That is Amdahl, and it is the number
to budget with: `H_ZZ` is ~7.5 s of a ~13.6 s Hessian call, and the Hessian callback is itself only
part of an outer evaluation, which also pays the C+ outer gradient, the screens, and verification.

---

## 2. Phase 1: the two runs

### 2.1 Configuration (identical except the family)

Launched 2026-08-10 18:41 EDT as two separate OS processes, `-t 10` Julia threads each, separate
`ckpt_dir`s, via the existing `production_smoke_{ozc,cmzc}_cross_2026-08-09.jl` launchers — reusing
their `w0` construction rather than re-deriving it, with only the wall budget and run-level knobs
parameterised.

```
delta                    = 1.0              direction = upper (find_smallest = true)
K_mean = K_pair          = 3                Variant D  = level 2 (= sigma-1)
W                        = 100_000          L          = 50   (CM families only)
contrasts                = :orthonormal     probs      = nested_grid_sequence([10,20,50])[50]
include_truncated_moment = true             (CM families only)
draw_design              = :sobol_randomized ; draw_seed = 20260719
destination_sample       = :exclude_row     ; exclude_diagonal_gravity = true
gravity_exclude_cells    = default_gravity_exclude_cells_brazil_korea()
sigmaHat = 3.0 ; inner_lower_limit = -10.0 ; A_coordinate_mode = :powered_aspace
cm_gradient_backend      = :cplus (driver default)
maxtime_real             = 28800.0 (8 h)    checkpoint_interval_s = 600.0
blas_threads             = (driver default -> 8)  zc_gram_backend = :blas_syrk
```

* **OZC-CROSS** → `run_originzc_upper_checkpointed`,
  `distribution_restriction = :origin_specific_moments_zero_covariance`,
  `power_target_layout = :origin_by_power_cross`, `originzc_profiled_level = 2`.
* **CM+ZC-CROSS** → `run_cm_upper_checkpointed`, `cm_extension = :cm_plus_moments`,
  `meanzc_target_layout = :shared_by_power_cross`, `meanzc_profiled_level = 2`.

Start point: the calibration point in `:powered_aspace` coordinates, `eta0` from the theoretical
population mean `Gamma(1 - mu*k)` (**not** a sample average of the draws the restriction is imposed
on), with the Variant D omitted coordinate dropped.

These are **single-start** evaluations at one delta, as scoped. A multistart wave would be a better
search and is a separate, much larger task.

### 2.2 An incidental observation: `resolved_active_threshold = Inf`

Both runs print, at startup:

```
[threshold-config] mode=cm_plus_meanzc requested_delta=1.0 resolved_active_threshold=Inf ...
[threshold-config] mode=origin_zc      requested_delta=1.0 resolved_active_threshold=Inf ...
```

so `ThresholdAbortState`'s pointwise weak-duality early abort is **disabled**, even though `delta=1.0`
is far below the ~9 at which CLAUDE.md says it is normally active.

Controlled before reading anything into it (per CLAUDE.md's "control against the unmodified family
before bug-hunting"): this is **identical across both drivers and both families**, in the 8-hour runs
and in all five Phase 0 gate arms. It is pre-existing behaviour of these restricted-family drivers,
**not** something the 2026-08-09 cross work introduced.

It is also not costing these runs evaluations, because the two mechanisms cover different cases. The
early abort earns its keep on *infeasible* points; but an infeasible point drives the inner objective
to −∞ and is already cut off quickly by `inner_lower_limit = -10.0`, which both runs set. Recorded
here as an observation for a future session, deliberately not chased — it is outside this task.

### 2.3 Results — CM+ZC-CROSS

| | |
|---|---|
| `knitro_status` | **−102 `KN_RC_FEAS_NO_IMPROVE`** — *not* −401 |
| wall | 21 629.1 s (6h00m of an 8h budget; **stopped ~2 h early**) |
| `n_eval` / `n_grad` | 28 / 10 |
| verified evaluations | **28 of 28** (16 feasible, 12 infeasible; zero unverified) |
| best feasible | gp = 0.9840034814, Δ = 0.9993157737, at **eval 20** (t = 16 586.9 s) |
| **κ = 1 − gp^(σ/(σ−1))** | **2.3898562 %** |
| checkpoint | `reason=stage_complete`, `meanzc_target_layout=shared_by_power_cross`, `n_eta_stored=2` |
| backend manifest | `family=cm_meanzc_cross`, `blas_threads=8`, `zc_gram_backend=blas_syrk` |

**Termination is neither budget nor optimality.** −102 is classified by this repo's own
`knitro_status.jl:45-47` as `:feasible_approx` — "primal feasible; the solution estimate cannot be
improved further and the desired dual-feasibility accuracy could not be achieved". So it is a solver
**stall**, not a proven optimum, and equally not a wall-clock termination. It left two hours of budget
unused, which means **more wall-clock would not have helped this start**.

**The search barely moved.** Δ climbed steadily from 0.0395 to 0.9993 while gp fell only from
0.98402789 to 0.98400348 — six hours bought 2.4e-5 in gp, i.e. κ from 2.386225 % (the calibration
point) to 2.389856 %, a gain of **0.0036 percentage points**. Evaluations 25, 26 and 28 re-evaluate
the identical point, the `FEAS_NO_IMPROVE` signature.

`n_eta_stored=2` is correct and not a truncation: the CM+ZC (shared-mean) layout carries ONE shared
target per level, so `n_eta = K_mean = 3`, minus the Variant D omitted slot = 2 active. This differs
from the origin-specific OZC layout's 60/59 and is the intended difference between the two families.

### 2.4 Results — OZC-CROSS

| | |
|---|---|
| `knitro_status` | **−102 `KN_RC_FEAS_NO_IMPROVE`** — *not* −401 |
| wall | 22 603.6 s (6h17m of an 8h budget; **stopped ~1.7 h early**) |
| `n_eval` / `n_grad` | 100 / 32 |
| verified evaluations | **100 of 100** (78 feasible, 22 infeasible; zero unverified) |
| best feasible | gp = 0.9838689435, Δ = 1.0000005161, at **eval 72** (t = 15 775.9 s) |
| **κ** | **2.4098741679 %** |
| backend manifest | `family=origin_zc_cross`, `blas_threads=8`, `zc_gram_backend=blas_syrk` |

**The bound does not depend on the feasibility slack.** The driver's incumbent (eval 72) has
Δ = 1.0000005161, i.e. it exceeds 1.0 by 5.2e-7 and qualifies only under the `Δ ≤ delta + 1e-6` test
in `cm_originzc_checkpoint.jl:916`. That would normally require an asterisk. It does not here:
evaluation 100 is **strictly** feasible (Δ = 0.9999999999348741, inside by 6.5e-11, verified) and
attains κ = 2.4098741512 %, differing from the reported bound by **1.7e-8 percentage points**.
Evaluation 74 is likewise strictly feasible (Δ = 0.99999982) at the same gp to 8 decimals. So the
reported κ survives imposing Δ < 1 exactly.

### 2.5 Both runs stalled; neither exhausted its budget

| | CM+ZC-CROSS | OZC-CROSS |
|---|---|---|
| status | −102 `FEAS_NO_IMPROVE` | −102 `FEAS_NO_IMPROVE` |
| wall used | 6h00m of 8h | 6h17m of 8h |
| evals | 28 (16 feasible) | 100 (78 feasible) |
| verified | 28/28 | 100/100 |
| κ | 2.3898562 % | 2.4098742 % |
| κ gain over the calibration point | +0.0036 pp | +0.0237 pp |

**Neither run terminated on the wall budget, and neither proved an optimum.** −102 is
`:feasible_approx` in `knitro_status.jl` — "the solution estimate cannot be improved further and the
desired dual-feasibility accuracy could not be achieved". Both left 1.7–2 h unspent, so **more
wall-clock would not have helped either start**. The honest reading of both κ values is *valid but
conservative*: each is attained at a genuinely feasible, `is_verified_success`-verified point, so the
true supremum over each identified set is at least this — but the searches stopped on a tolerance
stall, not a certificate.

That both families stalled the same way, from the same start, at κ within 0.02 pp of each other, is a
mild consistency check on the cross implementation — and a strong hint that **the binding constraint
on these runs was the search, not the compute budget**. See §4.

### 2.6 On the `-100` inner-solve stall (the handover's open item)

The handover asked for the rate of `inner_status = -100` (`KN_RC_NEAR_OPT`) stalls, these runs being
the largest sample so far. **This cannot be answered from these logs.** The driver records the
*outer* KNITRO status and, per evaluation, only `(gp, Δ, feasible, verified)`; the inner status is
consumed inside `archOZ_verified_state`/`archC_meanzc_verified_state` and is not surfaced to the
trace or the checkpoint. Both runs ran with the inner solver at `outlev=0`, so no per-inner-solve
line was emitted either.

What the runs *do* establish on this point is weaker but not nothing: **228 of 228 outer evaluations
across both families passed `is_verified_success`** (28/28 and 100/100), and verification is exactly
the gate that a tolerance-class inner stall has to clear. So whatever the `-100` rate was, it did not
produce a single unverified evaluation. Getting the rate itself needs an instrumented run
(`CS.INNER_*` counters, or plumbing `base.inner_status` into the trace tuple) — recorded here as not
done, rather than estimated.

---

## 3. Interpretation — how to read these bounds

**Direction convention, confirmed in the code rather than taken on trust** (the handover asked for
this explicitly). `direction_bounds.jl:46-48` and `incumbent_logic.jl:57-59` both give
`find_smallest = true` ⇔ minimize gp ⇔ the **upper**-κ branch; `κ = 1 − gp^(σ/(σ−1))`, σ = 3, so a
smaller gp is a larger κ. Both runs used the `*_upper_checkpointed` drivers with `find_smallest=true`.

**Why a budget- or stall-terminated run still yields a valid bound.** The incumbent is updated only
where `feasible && verified && is_better_polish(...)`
(`cm_checkpoint.jl:1745-1747`, `cm_originzc_checkpoint.jl:918-920`), and `verified` is
`is_verified_success(verify)` — KNITRO's own tolerance-based stops are *not* sufficient. So every
reported κ is attained at a point that is both feasible and independently verified. An unconverged
search therefore still gives a **valid lower bound on the upper bound**: the true supremum over the
identified set is at least this. It is *not* a converged optimum and must not be presented as one.

**These specific runs.** Both terminated `-102` (`FEAS_NO_IMPROVE`), which is neither `-401` (budget)
nor a certificate. So:

> κ_OZC-CROSS ≥ 2.4098742 % and κ_CM+ZC-CROSS ≥ 2.3898562 % at δ = 1.0, K = 3/3, W = 100 000 —
> each attained at a verified feasible point, neither converged.

## 4. What these runs actually establish, and what they do not

**The searches, not the compute, were the binding constraint.** Both stalled with 1.7–2 h of budget
unspent; CM+ZC-CROSS moved κ by 0.0036 pp in six hours. Buying more wall-clock for the same single
start would have bought nothing. Three independent lines of evidence say the missing ingredient is
**multistart**, not time:

1. Both cross families stalled at `-102` from the same start, ~0.02 pp apart.
2. In the companion nesting experiment (`docs/NESTING_DIAGONAL_TO_CROSS_2026-08-10.md`), the
   *diagonal* family run single-start from the same calibration point reached κ = 2.955 % — while its
   own 6-start multistart best is κ = 6.321 %. **Multistart more than doubled the diagonal family's
   single-start κ.**
3. This repo's own history says the same (memory `paper-upper-v1-waves12-misseeded-delta-cells`, and
   the handoff README: "empirically in this campaign `CONVERGED` has repeatedly *not* meant optimal;
   multistart did the real work").

So these two κ values should be read as **single-start results at one δ**, exactly as the handover
scoped them — not as the cross families' δ=1 bounds. Given the diagonal precedent, the cross families'
true bounds under multistart are plausibly materially higher.

**What is solidly established** is the Δ\* cost of the cross restrictions, measured in a properly
controlled comparison (same point, same configuration, both families): **8.75×** at the calibration
point, and hard infeasibility (`-300`) at the diagonal family's own δ=1 optimum. That result does not
depend on any search converging. See the companion document.

## 5. Recommended next step

A **cross multistart wave** at δ = 1.0, K = 3/3, mirroring the diagonal family's protocol, using
`reference-multistart-seed-generator` (`full_aod_diag/d4_exact/multistart_seed_generator.jl`) rather
than a hand-rolled seeding — and noting that the campaign runner already accepts both cross families
(`production_five_family_cross_seed_specs`). Budget it by evaluations, not hours: these runs show a
single start exhausts its usable progress in ~6 h, so ~6 h × N starts, run as separate processes.

Two cheap improvements to fold in first:

* **Surface `inner_status` into the trace tuple**, so the `-100` rate (§2.6) becomes measurable
  instead of unanswerable.
* **Settle 8-vs-16 BLAS threads sequentially on an idle machine** (§1.6) — my Phase 0 comparison was
  confounded by 5-way concurrency, and CM+ZC-CROSS at 28 evaluations in 6 h is the family that would
  benefit most if 16 really is faster.
