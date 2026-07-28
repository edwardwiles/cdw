# CM+ZC / origin-ZC real D=20 profiling + H_ZZ backend benchmark (2026-07-28)

Branch: `agent/d20-profile-cmzc-originzc-hzz-2026-07-28`, based on
`optimize/production-structured-CM-ZC-hessian-2026-07-28@410c154`, itself based on
`origin/production/fullA-exact@5b4f9da` (CM/common-Fréchet harmonization merge). Scope: `cm_meanzc`
(CM+ZC) and `origin_zc` (ZC-only) ONLY — a sibling agent covers `flexible_cm`/`common_frechet` in a
separate worktree.

Real D=20/Ddest=19/W=100,000, fixed theta, `destination_sample=:exclude_row`, production K width
(`K_mean=1, K_pair=1`, matching `smoke_delta1_cmzc.jl`/`smoke_delta1_originzc.jl`'s own call
pattern — there is no separate `CM_MEANZC_K_...DEFAULT`/`ORIGINZC_K_...DEFAULT` Ref; K is caller-
supplied per driver call).

## 1. Headline verdict

```
origin_zc:  FULLY PROFILED, real solved economic states at 5 Julia-thread counts x 3 points.
            Dominant block: H_EZ(=HER) (crossprep), ~42% of the 4-block callback cost at t=20,
            H_ZZ ~34%, H_ZZ_centering_prep ~17%, H_EE ~7%.
            H_EZ threading: up to 5.3x (t=10) / 4.5x (t=20) vs serial.
            H_ZZ backend at production width (nx=210): :threaded_packed is NEVER the winner here
            (3-20x SLOWER than the BLAS candidates across every Julia-thread count tested) --
            REVERSES the prior session's nz=20 (K_pair=0) recommendation. :blas_gemm with BLAS
            threads>=8 is the clear winner (~0.08-0.09s vs 0.17-0.28s for every single-BLAS-thread
            candidate and 0.53-4.3s for threaded_packed).
            D=20 correctness gates: 135/135 PASS (0 failures) across workers {1,4,8,10,20} x
            backends x 3 points.

cm_meanzc:  BLOCKED at the source -- run_cm_upper_checkpointed (the real public driver, the ONLY
            path this task's brief identified as safe) reliably fails cm_meanzc's own real inner
            KNITRO solve on THIS branch base (nStatus=-500/-502, "Could not evaluate objective or
            constraints at the initial point"), reproduced 5-for-5 independent parameter variations
            and 15-for-15 (5 thread counts x 3 points) in the full sweep. This is a genuine,
            reproducible, PRE-EXISTING regression on `origin/production/fullA-exact@5b4f9da` --
            NOT something this task's own kernel work caused, and not something the task's own
            "route around it via the real driver" mitigation fixes (that mitigation assumed the
            real driver still worked for cm_meanzc, which was true at fb6ad2e per an earlier
            postmerge-smoke doc but is NOT true at the current 5b4f9da base -- see Sec 4).
            Sub-block TIMING and correctness-GATE data were still collected (12/15 planned blocks,
            all 5 thread counts, all 3 point labels) using the real production dims
            (n_E=382, n_Z=210, n_C=950, L=50, nO=19) but with S left at its unwritten default
            (all-ones) since no real Hessian callback ever fired -- this does NOT invalidate the
            TIMING (pure function of array shapes/threading) or the correctness GATES (algebraic
            equivalence between backends on whatever S is), only the "genuinely solved economic
            state" claim. Every affected row is tagged `*_UNSOLVED_TIMING_ONLY` — never silently
            presented as real. Dominant block: H_CZ_prep (CM+ZC-only, bin-feature table fill)
            ~53% at t=20, H_EZ ~13%, bintables ~14%, H_ZZ ~13%, H_EE ~3%.
```

## 2. cm_meanzc real-driver blocker — full evidence chain

The parent task brief's own point 4 states the real public driver
(`run_cm_upper_checkpointed`) is "the ONLY confirmed-reliable path for a genuine warmed D=20 CM+ZC
state." That claim is contradicted by fresh evidence gathered this session, on this exact branch
base. Five independent, targeted repro attempts, each isolating one hypothesis, ALL fail
identically:

| # | Variation from baseline | Result |
|---|---|---|
| 1 | Default config (K_mean=1,K_pair=1, calibration w0, `:operator` verification, `:cplus` gradient) | `nStatus=-500` inside `archC_meanzc_verified_state`'s inner solve, at the FIRST evaluation (`n_eval=0`) |
| 2 | `CM_MEANZC_VERIFICATION_BACKEND_DEFAULT[] = :dense_reference` (rules out a verification-layer regression) | Identical failure |
| 3 | `meanzc_K_pair = 0` (rules out the pairwise-ZC target specifically) | Identical failure |
| 4 | 1% random jitter on w0 (rules out an exact-calibration-point knife edge) | Identical failure |
| 5 | `cm_gradient_backend = :reference` (rules out the `:cplus` gradient path specifically, despite the failure being inside `grad_callback`) | Identical failure |

All five raise the SAME `CMExpectedSolveFailure("archC_meanzc_verified_state: inner solve failed,
nStatus=-500 ...")`, with the SAME KNITRO log signature (`ERROR: User routine for grad_callback
returned -500`, `Could not evaluate objective or constraints at the initial point`, `EXIT:
Evaluation error`), and the SAME downstream `verification class=ConfirmedNumericalNegative`
classification. The `x_free0` values printed in the error are genuine, real calibration-scale
economic values (A_od spanning ~719 to ~5e11, matching this repo's own documented real range —
**not** the `A_od≡1` degenerate reconstruction CLAUDE.md warns about), confirming this is not a
"passed a garbage w0" mistake.

The UNMODIFIED, pre-existing `smoke_delta1_cmzc.jl` script (previously reported successful in
`TRUE_OPERATOR_NO_H_POSTMERGE_SMOKES_2026-07-28.md`, at base commit `fb6ad2e`, 1 outer iter/3
evals/2 grads/wall=265.4s) was also run, byte-for-byte, on THIS worktree/branch — it fails
identically. This rules out any bug in this task's own scripts.

**Full sweep confirmation**: the main profiling run repeated this at all 5 Julia-thread counts
(1/4/8/10/20) x all 3 points (calibration/non_calibration/solver_trajectory) = 15 independent
driver calls — 15/15 failed with the identical `nStatus=-502`/`n_eval=0` signature (see
`CMZC_ORIGINZC_D20_DRIVER_SUMMARY_2026-07-28.csv`). Thread count has no bearing on the failure.

**Likely window**: `git diff fb6ad2e..5b4f9da -- cm_hessian_architectures.jl cm_hessian_threaded.jl
cm_meanzc_production.jl operator_verification.jl` shows all four files changed (98/35/3/124 lines
respectively) in the "CM/common-Fréchet harmonization merge" — the postmerge-smoke doc's one
successful CM+ZC run predates this merge; nothing in this session confirms which specific
harmonization commit introduced the regression, and per the task's own explicit instruction, this
was NOT further root-caused (out of scope for a profiling task; flagged as the top priority
open item for whoever owns `production/fullA-exact` next).

**What this does NOT mean**: it does not mean cm_meanzc's underlying math/kernels are wrong — the
sub-block-level correctness gates (threaded-vs-serial, backend-vs-reference), which do not depend
on a converged KNITRO state, all pass. It means the *outer inner-solve wiring* for this one family,
at this K config, on this exact branch base, cannot currently complete even its first evaluation.

## 3. Sub-block timing profile

Full data: `CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_cmzc_originzc.csv` (schema
`family,block,nthreads,t_min_s,t_mean_s,point` — `point` is an addition beyond the base schema the
task spec named, appended as a trailing column for backward-compatible parsing by name).

### origin_zc (real solved states, all 3 points, all 5 thread counts)

| block | t=1 | t=20 | speedup |
|---|---:|---:|---:|
| H_EE | 0.088s | 0.044s | 2.0x |
| H_ZZ_centering_prep | 0.078s | 0.105s | ~flat |
| H_EZ (serial baseline) | 1.141s | 1.333s | n/a (ignores threads, as expected) |
| H_EZ (threaded) | 1.082s | 0.253s | **4.5x** (peak 5.3x at t=10) |
| H_ZZ reference | 0.233s | 0.233s | 1.0x (single BLAS thread by design) |
| H_ZZ blas_syrk | 0.173s | 0.203s | ~flat |
| H_ZZ blas_gemm | 0.284s | 0.265s | ~flat |
| H_ZZ threaded_packed | 4.257s | 0.532s | 8.0x, but STILL 2.6x slower than blas_syrk at t=20 |

At t=20, solver_trajectory point, best-backend total = 0.604s: **H_EZ(threaded) 41.9%, H_ZZ(best)
33.6%, H_ZZ_centering_prep 17.3%, H_EE 7.3%**. This reconfirms H_EZ as the dominant block (matching
the prior session's own K_pair=0 finding of ~79% pre-threading share, now diluted to ~42%
post-threading — consistent, not contradictory, since threading is exactly what shrank it).

### cm_meanzc (production dims n_E=382/n_Z=210/n_C=950/L=50/nO=19, `*_UNSOLVED_TIMING_ONLY` — see
Sec 2 caveat; timing/gates still meaningful, "genuinely solved" claim is not)

| block | t=1 | t=20 | speedup |
|---|---:|---:|---:|
| H_EE(core-only) | 0.076s | 0.040s | 1.9x |
| bintables(threaded) | 0.174s | 0.199s | ~flat (this call is ALREADY the threaded variant) |
| H_ZZ_centering_prep | 0.075s | 0.074s | ~flat |
| H_EZ (serial) | 1.115s | — | n/a |
| H_EZ (threaded) | 1.132s | 0.185s | **6.0x** at t=20 |
| H_ZZ reference | 0.216s | 0.187s | ~flat |
| H_ZZ threaded_packed | 4.207s | 0.689s | 6.1x, still far slower than reference/blas at this width |
| H_CZ_prep (serial) | 1.063s | 1.016s | ~flat |
| H_CZ_prep (threaded) | 1.049s | 0.775s | **1.4x** at t=20 (weak — SLOWER than serial at t=4/t=8,
  only wins at t=20; not a clean win like H_EZ) |

At t=20, calibration point, sum of best-available blocks (H_EE+bintables+H_EZ(threaded)+
H_ZZ_prep+H_ZZ(best)+H_CZ(threaded)) = 1.460s: **H_CZ(threaded) 53.1%, bintables 13.6%, H_ZZ 12.8%,
H_EZ(threaded) 12.7%, H_ZZ_prep 5.0%, H_EE 2.7%**. H_CZ is the new dominant block at production
width (nx=210) — much larger than the prior session's rough ~46% combined H_EC+H_CZ estimate at
the narrower nz=20 config, and its threading gain is markedly weaker/less reliable than H_EZ's,
making it the clearest remaining optimization target for this family.

**Known gap**: `H_EC_prep(crossprep)` and `H_EC+H_CZ_asm` (3 of 15 planned cm_meanzc blocks) are
missing from every point/thread-count — a PRE-EXISTING, unrelated KNITRO.jl bug
(`MethodError: Cannot convert an object of type Float64 to an object of type Task`, an asynchronous
exception from KNITRO's own background `puts`-callback handler, matching the master report's own
documented observation) consistently interrupts `profile_cmzc_point!` right after `H_CZ_prep`,
before reaching the H_EC lines. Each point's already-collected 12 rows are preserved (per-point
`try`/`catch`, fixed during this session — the original design had one `try`/`catch` per family,
which lost ALL 3 points' data on this async exception; now isolated per point). Not pursued further
per the task's own "don't waste time re-diagnosing the cm_meanzc KNITRO bug" instruction — this is
a different, independently-confirmed pre-existing bug class, not the nStatus=-500 issue.

## 4. H_ZZ backend benchmark at production width (Section 9)

Full data: `HZZ_BACKEND_PRODUCTION_WIDTH_BENCHMARK_2026-07-28.csv`. Production width: `K_mean=1,
K_pair=1` → `nx=210` (origin_zc and cm_meanzc share the identical dispatcher, confirmed same nx).
Also tested a wider config `K_mean=2,K_pair=1` → `nx=230` for origin_zc (`K_mean=1,K_pair=0` is
**not a valid combination** under `distribution_restriction=:origin_specific_moments_zero_covariance`
— that restriction type requires `1<=K_pair<=K_mean`, a genuine config constraint, not a bug).

**Julia-thread axis** (BLAS threads=1, from Sec 3's own sweep): `:threaded_packed` ranges
0.53s (t=20) to 4.3s (t=1) at nx=210 — NEVER beats `:reference`/`:blas_syrk` (both flat at
0.17-0.28s regardless of Julia thread count, as expected since neither uses Julia threading).

**BLAS-thread axis** (Julia threads=1, this section's own sweep):

| blas_threads | blas_syrk | blas_gemm |
|---:|---:|---:|
| 1 | 0.192s | 0.234s |
| 4 | 0.174s | 0.136s |
| 8 | 0.121s | 0.106s |
| 10 | 0.106s | **0.084s** |
| 20 | 0.168s (non-monotone) | **0.078s** |

`:blas_gemm` beats `:blas_syrk` once BLAS threads >=4, and both comfortably beat `:reference`
(0.22s) and `:threaded_packed` (4.1-5.0s) at every BLAS-thread count tested. `blas_syrk` is
non-monotone past 10 threads (small-problem BLAS-threading overhead at nx~210-230) — `blas_gemm`
is the more robust choice. Machine-precision agreement with `:reference` throughout
(maxdiff 3e-14 to 4e-13, well inside the 1e-9 tolerance).

**Verdict — REVERSES the prior session's recommendation**: that session's finding
(`:threaded_packed` wins 4.0-4.5x at Julia-thread=20) was measured at `K_mean=1,K_pair=0` (nx=20),
an order of magnitude narrower than this task's own confirmed production width (`K_mean=1,
K_pair=1`, nx=210). At nx=210, `:threaded_packed` is uniformly the WORST candidate.
**`H_ZZ_BACKEND = :blas_gemm` at BLAS threads>=8-10** (Julia threads irrelevant to this backend) is
the new recommendation for both families at their actual production K width, ~0.08-0.09s vs the
prior recommendation's realistic best case of ~0.53s (Julia threads=20) — a further ~6x on top of
correcting a wrong prior default.

## 5. D=20 correctness gates (Section 12)

Full data: `CMZC_ORIGINZC_D20_CORRECTNESS_GATES_2026-07-28.csv`. **135/135 PASS, 0 failures.**
Covers: `cross_hessian_threaded` (H_EZ, H_CZ for cm_meanzc) at workers {1,4,8,10,20} vs serial,
`zc_gram_backend` ∈ {`:blas_syrk`, `:blas_gemm`, `:threaded_packed`} vs `:reference`, both
families, at calibration + non_calibration + solver_trajectory points, at each of the 5 thread
counts. Tolerances: 1e-12 (pure threading, bit-exact expected) / 1e-9 (BLAS/algebraic-identity
candidates) — every observed maxdiff was far inside tolerance (typically 1e-14 to 1e-13).

For origin_zc, KNITRO status/iterations/kappa were directly confirmed UNAFFECTED by any of this —
the driver runs that produced the profiled states are the SAME runs whose kappa/n_eval/n_grad are
recorded in `CMZC_ORIGINZC_D20_DRIVER_SUMMARY_2026-07-28.csv` (real progression 0.0203 →
0.0359/0.0524 → 0.0584 across calibration/non_calibration/solver_trajectory, `nStatus=-401`
KN_RC_TIME_LIMIT_FEAS throughout — a normal, expected stop on a feasible point given the short
budgets used, not an error). For cm_meanzc, this check is vacuous — no run ever reached a real
KNITRO iteration (Sec 2).

## 6. Caveats / open items for the parent task owner

1. **cm_meanzc's real driver is currently broken on `production/fullA-exact@5b4f9da`** for its own
   inner solve (Sec 2) — this blocks ANY genuine warmed-state measurement for this family, not just
   this task's own profiling. Recommend as the highest-priority follow-up: bisect the harmonization
   commits (`db786c0`..`5b4f9da`) against `smoke_delta1_cmzc.jl` to isolate the regression before
   trusting any other cm_meanzc D=20 result on this base.
2. `H_EC_prep`/`H_EC+H_CZ_asm` timing for cm_meanzc not captured (Sec 3, known async KNITRO.jl bug,
   unrelated to this task).
3. This benchmark did not reach the parent spec's optional "realistic five-family PROCESS-parallel
   resource plan" check (Section 9's own explicit "if you have time" item) — not attempted, out of
   time budget this session. All process launches in this session assumed exclusive access to the
   208-core host.
4. A CSV-serialization bug (unquoted block/key labels containing literal commas, e.g.
   `H_EZ(=HER,threaded)`, silently shifted downstream columns) was found and fixed mid-session —
   affected labels were renamed comma-free (`H_EZ(=HER)_threaded`, etc.) and all already-written
   CSVs (in both `results/` and `docs/`) were repaired and verified (`awk -F, 'NF != expected'`
   returns 0 rows on every final CSV). Flagged here in case the SAME labeling convention exists
   elsewhere in this codebase's other CSV writers.

## 7. Deliverables index

- `CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_cmzc_originzc.csv` (315 rows)
- `HZZ_BACKEND_PRODUCTION_WIDTH_BENCHMARK_2026-07-28.csv` (36 rows)
- `CMZC_ORIGINZC_D20_CORRECTNESS_GATES_2026-07-28.csv` (135 rows, 0 fail)
- `CMZC_ORIGINZC_D20_DRIVER_SUMMARY_2026-07-28.csv` (30 rows)
- This report.
- Source: `full_aod_diag/d4_exact/cross_hessian_live_stash_2026-07-28.jl` (new),
  `diag_cmzc_originzc_d20_profile_2026-07-28.jl` (new, main sweep),
  `diag_hzz_backend_benchmark_2026-07-28.jl` (new, Section 9 benchmark), plus one-line stash edits
  to `cm_meanzc_production.jl`/`cm_originzc_production.jl`.
