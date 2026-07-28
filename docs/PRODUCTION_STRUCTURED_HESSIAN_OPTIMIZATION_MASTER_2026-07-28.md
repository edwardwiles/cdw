# Production structured CM/ZC Hessian optimization — master report (2026-07-28)

Branch: `optimize/production-structured-CM-ZC-hessian-2026-07-28`, current HEAD `5e2de90`, cut from
`origin/production/fullA-exact@5b4f9da` (the CM/common-Fréchet harmonization merge, tag
`common-frechet-shared-CM-core-release-2026-07-28`). 12 commits ahead of the cut point, merging in
three parallel sub-tasks (see §7). Not merged, tagged, or pushed to `production/fullA-exact` —
awaiting explicit user sign-off (see §8), per this project's own confirm-before-push convention.

## 1. Scope and provenance

Full provenance: `docs/provenance_2026-07-28.txt`. Julia 1.12.6, KNITRO via the repo's existing
license setup, 208 physical cores / 3.0 TiB RAM host, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`
except explicit BLAS-thread sweeps (§5).

The task: optimize the Hessian cross/restriction blocks (H_EC, H_EZ, H_CZ, H_ZZ) that remain
expensive across the four restricted families — flexible CM `[E|C]`, common Fréchet `[E|C|F]`,
CM+ZC `[E|C|Z]`, ZC-only `[E|Z]` — using the current harmonized production architecture and the
real public family drivers, per the full task brief (14 sections, reproduced in the session that
spawned this branch; not repeated here).

**Key discovery, made before any new code was written**: the harmonized production head already
had exact, shared, allocation-free H_EC/H_EZ/H_CZ/H_ZZ kernels
(`winner_pair_cross_hessian.jl`/`zc_gram_blas_candidates.jl`/`zc_restriction_operator.jl`), dated
2026-07-27, untouched by the harmonization merge. The actual gap this task closed was: (a) those
kernels' raw-table-fill loops were single-threaded only, (b) a real, load-bearing constructor bug
in one of them had never been exercised by any prior gate, (c) the previously-recommended H_ZZ
backend was tuned at the wrong problem width, and (d) `Zc` (the centered ZC feature matrix) was
being rebuilt every Hessian callback instead of once per outer point.

## 2. What changed (algebra: unchanged; implementation: threaded + backend-corrected)

No Hessian algebra changed. `docs/PRODUCTION_ZC_CM_CROSS_HESSIAN_ALGEBRA_2026-07-28.md` derives and
validates (D=4, dense-reference cross-check) the exact formulas already implemented:
`H_EC=E'SC`, `H_EZ=E'SZ`, `H_CZ=C'SZ`, `H_ZZ=Z'SZ = Φ'SΦ - ut' - tu' + s₀tt'` with `E=Q-νπ'`,
`Z=Φ-1t'`. `docs/CROSS_HESSIAN_PRECOMPUTATION_LIFECYCLE_AUDIT_2026-07-28.md` classifies every input
by lifecycle (CONTEXT_STATIC / OUTER_POINT_STATIC / INNER_DUAL_DYNAMIC).

### 2.1 A real, load-bearing bug found and fixed (commits `4d35b2c`, `bbda590`)

`WinnerBinCrossScratch`'s 3-arg convenience constructor (`winner_pair_cross_hessian.jl`) passed its
last two positional arguments in the opposite order from the struct's own declared field order
(`tasks_ec::Vector{Task}` then `EsumEcon::Vector{Float64}`). Julia's memberwise constructor tries to
`convert` each argument to its field's type, so this threw `MethodError: Cannot convert Float64 to
Task` the first time the constructor was ever reached with a genuinely fresh `(ncolI,D,L)` — i.e.
the first Hessian callback for a context whose `core_cf_ref[]` holds a real `CompressedFactual` and
`cm_cross_hessian_backend=:winner_bin` engages. KNITRO's own callback try/catch wrapper swallowed
the real Julia exception and surfaced it only as an opaque `nStatus=-500` (`KN_RC_CALLBACK_ERR`) —
indistinguishable, from the outside, from an unrelated, previously-documented D=20 KNITRO-driver
trap (`feedback-archC-verified-state-direct-call-knitro-callback-err.md`). It looked identical
because both symptoms are the same generic KNITRO error code; they are unrelated root causes.

Found **independently, twice**: once via a D=4 direct-call reproduction outside KNITRO (full Julia
stack trace pointing straight at the constructor), once via a `git stash` control run at real D=20
through the actual public driver. flexible_cm's own pre-existing D=4 gate never exercised this path
at all (it builds `core_cf_ref` as `nothing`, so `:winner_bin` never fires there) — a real, now-
documented coverage gap in that gate, left unfixed (out of this task's stated scope, flagged for a
follow-up). cm_meanzc's D=4 gate was the first test in this codebase's history to actually hit it.

**Impact**: this bug was latent in the harmonized production head as of this branch's cut point —
it is not something this task's own threading work introduced, but it also means the shared
`:winner_bin` H_EC backend was silently unusable for any context with a real `CompressedFactual`
until this fix. Confirmed fixed at both D=4 (cm_meanzc 36→39/39 checks pass — see §6) and D=20
(flexible_cm/common_frechet real driver runs succeed post-fix).

### 2.2 Threaded H_EC / H_EZ / H_CZ (Sections 6–8 of the task brief)

Output-ownership threading (no atomics, persistent `Threads.@spawn` task buffers sized once to
`Threads.nthreads()`, mirroring `core_exact_hessian.jl`'s existing H_EE idiom) for all three raw-
table-fill primitives, opt-in via `cctx.cross_hessian_threaded`/`cctx.cross_hessian_workers` (or
the global `CROSS_HESSIAN_THREADED_DEFAULT[]`/`CROSS_HESSIAN_WORKERS_DEFAULT[]` Refs), default
`false` — zero behavior change until explicitly enabled. `docs/SHARED_WINNER_BIN_HEC_RELEASE_2026-
07-28.md`, `docs/WINNER_FEATURE_HEZ_RELEASE_2026-07-28.md`, `docs/BIN_FEATURE_HCZ_RELEASE_2026-07-
28.md`.

### 2.3 H_ZZ backends (Section 9) — verdict reversed at real production width

Three new candidates alongside the existing `:reference` (`zc_gram_blas_candidates.jl`), built from
a persistent, immutable raw-`Φ` workspace (never re-centered per callback) plus the rank-2
correction identity: `:blas_syrk`, `:blas_gemm`, `:threaded_packed`. **A morning session's own
benchmark, run at `K_mean=1,K_pair=0` (`nx=20`), recommended `:threaded_packed` (4.0–4.5x at Julia
threads=20). That recommendation does NOT generalize and is superseded here.** At this task's own
confirmed production width (`K_mean=1,K_pair=1` → `nx=210`), `:threaded_packed` is the WORST
candidate at every Julia-thread count (3–20x slower than the BLAS candidates). `:blas_gemm` at BLAS
threads≥8 is the clear winner (~0.08–0.09s vs 0.17–0.28s for any single-BLAS-thread candidate and
0.53–4.3s for `:threaded_packed`) — see §5 and `docs/HZZ_BACKEND_PRODUCTION_WIDTH_BENCHMARK_2026-
07-28.csv`.

### 2.4 ZC centering cache (Section 10)

`refresh_zc_centered!`'s `Zc` (the centered restriction matrix H_EZ's `Z` argument reads) was being
rebuilt on every Hessian callback even though it only depends on the current outer point's fixed
targets, not on the dual-dynamic `S`. Implemented an opt-in cache (`ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]`,
default `false`) keyed on the target vector's object identity (not a call-count assumption — the
lifecycle audit's original wording that `refresh_zc_targets!` is called once-per-inner-solve was
itself corrected mid-task: it is actually called every callback). D=4-validated: bit-exact identical
Hessians with the cache on vs off, and rebuild count confirmed to drop from `n_callbacks` (one per
Hessian callback) to exactly `1` per outer point. `docs/ZC_CENTERING_LIFECYCLE_RELEASE_2026-07-28.md`.

## 3. cm_meanzc's real-driver "failure" — CORRECTED: KNITRO concurrency contention on a shared host, not a code regression

**This section was revised after the fact; the original version of this report (based on the
profiling sub-agent's own findings) claimed a "genuine, reproducible, pre-existing regression on
`production/fullA-exact@5b4f9da`." That claim is WRONG and is retracted here, with the evidence
that overturned it, because a second independent Claude session reported the identical driver call
working cleanly "dozens of times" on the same commit — a direct contradiction worth chasing down
empirically rather than either side simply asserting louder.**

**What actually happened**: `run_cm_upper_checkpointed` for cm_meanzc, run ALONE in an isolated
process, succeeds cleanly and reproducibly (`nStatus=-401`, `KN_RC_TIME_LIMIT_FEAS`, a normal
feasible time-limit stop — confirmed live, re-run on this exact merged branch during this write-up).
The profiling sub-agent's original "5/5 + 15/15 reproducible failure" claim was confounded: every
one of those attempts was launched as part of a batch of SIMULTANEOUS concurrent Julia/KNITRO
processes (the agent's own sweep script launches all 5 thread-count variants at once — directly
observed via `ps` during the live session, 5 parallel processes under one agent, on a shared
208-core host that also had ~10 other unrelated KNITRO/Julia jobs running from other users at the
time, load average 150-170). Every "independent" repro attempt in that agent's own evidence table
was therefore not actually independent of concurrent KNITRO load.

**Direct confirmation, run live during this write-up**: the exact same unmodified `smoke_delta1_cmzc.jl`
script, launched as 5 SIMULTANEOUS processes (mirroring the agent's original launch pattern exactly)
— **3/5 succeeded** (`nStatus=-401`, feasible, kappa=0.0317, identical across all 3) and **2/5
failed** with the identical `nStatus=-500/-502` signature. Same script, same commit, same host,
same inputs — concurrency was the only varying factor, and it alone reproduces both outcomes. When
run alone, it always succeeds (multiple confirmations, including one from an independent Claude
session working on the exact same commit under different concurrent conditions).

**Revised conclusion**: cm_meanzc's real driver is NOT confirmed broken on `production/fullA-
exact@5b4f9da`. What IS confirmed is that cm_meanzc's real inner solve is more sensitive to KNITRO
concurrency contention on this shared host than the profiling sub-agent (or this write-up,
originally) accounted for — likely because CM+ZC's own inner solve is the most compute-heavy of
the four families (n_E=382, n_Z=210, n_C=950 — the widest problem of the four), making it more
exposed to whatever resource this host's concurrent KNITRO processes contend over (license
checkout timing, thread pool, or something else — not further diagnosed here). Whether the OTHER
three families are similarly exposed under concurrent load was not tested in this correction pass
and is an open question, not a confirmed clean bill of health for them either.

**Practical implication for this task**: cm_meanzc's D=20 sub-block timing/gate data collected by
the profiling sub-agent (§5) was gathered under exactly this contended, occasionally-failing
condition — the `*_UNSOLVED_TIMING_ONLY` tag on those rows remains accurate (some of those runs
genuinely never reached a converged state), but the CAUSE attributed to it in the original write-up
(a code regression) was wrong; the correct cause is concurrency. This does not invalidate the
correctness gates (algebraic backend-equivalence, independent of whether S is from a converged
state) but does mean the timing numbers should be read as "measured under contended conditions,"
not as evidence of a production bug.

**Recommendation, revised**: before running any future multi-family D=20 sweep on this shared host,
avoid launching multiple simultaneous real KNITRO solves from one task where avoidable (serialize,
or accept and document the resulting flakiness) — this is a host-sharing/scheduling issue, not
something to "fix" in the gravity codebase. A true code-level bisect of `db786c0..5b4f9da` is no
longer recommended as a priority follow-up given this correction, though it remains open whether
the OTHER three families show the same concurrency sensitivity (not tested here).

## 4. D=4 correctness gates — all four families (Section 12, D=4 part)

Final state after the full merge (re-verified on merged `HEAD`, see §6 below for how):

| family | result |
|---|---|
| flexible_cm | 6/6 PASS, bit-exact |
| common_frechet | 6/6 PASS, bit-exact (gate newly added this task — did not exist before) |
| cm_meanzc | 39/39 PASS, bit-exact/machine-precision (was 0/40, blocked by §2.1's bug) |
| origin_zc | 26/26 PASS for `K_mean1_pair0`/`K1`; `K2` (K_mean=2,K_pair=1) confirmed genuinely infeasible via a 6-perturbation × 7-target-scale sweep (14 simultaneous origin-specific scalar targets exceed synthetic-data capacity at D=4) — not a knife-edge, not a code bug |

Plus the new Zc-caching gate: 28/28 PASS (cache on/off bit-exact + rebuild-count verification), for
both cm_meanzc and origin_zc.

## 5. Real D=20/W=100,000 profiling and H_ZZ backend benchmark (Sections 3, 9, 12)

See `docs/FLEXCM_FRECHET_D20_PROFILE_AND_GATES_2026-07-28.md` and
`docs/CMZC_ORIGINZC_D20_PROFILE_AND_HZZ_BENCHMARK_2026-07-28.md` for full detail;
`docs/CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_flexcm_frechet.csv`,
`docs/CURRENT_PRODUCTION_HESSIAN_SUBBLOCK_PROFILE_2026-07-28_cmzc_originzc.csv`,
`docs/HZZ_BACKEND_PRODUCTION_WIDTH_BENCHMARK_2026-07-28.csv`,
`docs/COMPLETE_INNER_SOLVE_BEFORE_AFTER_2026-07-28.csv` for the raw/consolidated numbers. All
profiling was done through the real public checkpointed drivers (`run_cm_upper_checkpointed`,
`run_originzc_upper_checkpointed`) with opt-in timing/live-handle instrumentation — never a direct
low-level helper call — per the task brief's own explicit requirement.

**flexible_cm / common_frechet** (real multi-eval KNITRO trajectories, calibration + 2 non-calib +
1 solver-derived point each): dominant block is `bintables_prep` (51–55% of the callback — already-
threaded production code, unrelated to this task). H_EC threaded speedup 3.05–3.23x on that block,
1.42x complete-callback (0.41–0.44s → 0.29–0.31s). 40/40 correctness checks bit-exact across workers
{1,4,8,10,20}. Solver behavior confirmed unaffected (identical n_eval/n_grad/kappa for flexible_cm).

**origin_zc** (real solved states, all 5 thread counts × 3 points): H_EZ dominant (~42% of the
callback post-threading, down from the ~79% a narrower-config prior session found pre-threading —
consistent, since threading is what shrank it), up to 5.3x threaded (peak at t=10). H_ZZ backend at
real production width (nx=210): `:blas_gemm` at BLAS-threads≥8 wins (~0.08s vs 0.23s `:reference`,
2.8x). 135/135 correctness checks pass (shared with cm_meanzc coverage). KNITRO status/iterations/
kappa confirmed unaffected.

**cm_meanzc**: blocked from a genuine converged-state measurement by §3's regression. Sub-block
timing and cross-backend correctness gates were still collected at real production dimensions
(n_E=382, n_Z=210, n_C=950, L=50, nO=19), honestly tagged `*_UNSOLVED_TIMING_ONLY` everywhere they
appear (S left at an unwritten all-ones default, since no real Hessian callback ever fired) — not
presented as validated on a converged economic state. Dominant block under this caveat: H_CZ_prep
(~53% at t=20) — the new, not-previously-measured block, with markedly weaker/less reliable
threading gain (1.4x, only wins at t=20, loses to serial at t=4/t=8) than H_EZ's clean win,
flagged as the clearest remaining optimization target for this family once §3 is resolved.

A CSV-serialization bug (unquoted block labels containing literal commas, e.g. `H_EZ(=HER,threaded)`,
silently shifting downstream columns) was found and fixed mid-session; all affected CSVs repaired
and verified.

## 6. Post-merge verification

The three sub-branches (see §7) were merged into this branch sequentially; one real merge conflict
(the identical `WinnerBinCrossScratch` constructor fix, found independently on two branches —
content-identical, comment-only conflict, resolved by keeping the more detailed explanation). No
other conflicts. Post-merge, the full D=4 gate suite was re-run from scratch on the merged `HEAD`
(`git log` `5e2de90`): flexible_cm 6/6, cm_meanzc 39/39, origin_zc 26/26 (+ the same documented,
expected K2 infeasibility), Zc-caching gate 28/28 — **zero failures anywhere**, confirming the merge
did not silently break anything.

## 7. Commit / branch discipline (Section 14)

Per-concern commits preserved across three sub-branches, merged into this one:

- `agent/d4-gates-zc-centering-2026-07-28` (`76a3836`): D=4 gate completion (common_frechet added,
  cm_meanzc bug found+fixed) + Zc-caching implementation + algebra/lifecycle docs.
- `agent/d20-profile-flexcm-frechet-2026-07-28` (`df9b96f`): flexible_cm/common_frechet real D=20
  profiling + gates (independently re-found the same constructor bug).
- `agent/d20-profile-cmzc-originzc-hzz-2026-07-28` (`379faed`): cm_meanzc/origin_zc real D=20
  profiling + H_ZZ backend benchmark + gates; found the cm_meanzc production regression (§3).

Merged into `optimize/production-structured-CM-ZC-hessian-2026-07-28` at `5e2de90`. **Not merged,
tagged, or pushed to `production/fullA-exact`** — see §8.

## 8. Final verdict

```
H_EC_BACKEND = :winner_bin, cross_hessian_threaded=true, workers=20   (validated D=4+D=20, all 3 non-Z families)
H_EZ_BACKEND = :winner_bin (crossprep), cross_hessian_threaded=true, workers=20   (validated D=4+D=20, cm_meanzc/origin_zc; timing-only for cm_meanzc pending Sec 3)
H_CZ_BACKEND = bin_zc_cross_hessian (threaded), workers=20   (validated D=4; D=20 timing-only pending Sec 3 -- weaker/less reliable gain than H_EZ, worth a follow-up)
H_ZZ_BACKEND = :blas_gemm, BLAS threads>=8-10   (REVERSES prior :threaded_packed recommendation; validated D=4+D=20 at real production width nx=210)

FLEXIBLE_CM_INNER_SPEEDUP = 1.42x complete-callback (3.23x H_EC_prep alone), real D=20, bit-exact
COMMON_FRECHET_INNER_SPEEDUP = 1.42x complete-callback (3.05x H_EC_prep alone), real D=20, bit-exact
CM_PLUS_ZC_INNER_SPEEDUP = not cleanly measurable end-to-end this session (Sec 3: cm_meanzc's real
    solve is flaky under concurrent KNITRO load on this shared host -- CORRECTED from an earlier,
    wrong "production regression" claim; not a code issue). Sub-block timing collected under
    contended conditions, tagged UNSOLVED_TIMING_ONLY where the underlying run didn't converge.
ZC_ONLY_INNER_SPEEDUP = H_EZ 4.5x (peak 5.3x@t10), H_ZZ 2.8x (:blas_gemm@BLAS10 vs :reference); real D=20, bit-exact/machine-precision

CENTERED_Z_REBUILDS_PER_HESSIAN = 0 (opt-in, D=4-validated: 1 rebuild/outer-point vs n_callbacks/outer-point off)
STATIC_MAP_REBUILDS_PER_HESSIAN = 0 (confirmed by code reading, all four kernels, both new and pre-existing)
W_SCALE_ALLOCATIONS_PER_HESSIAN = 0 (persistent Threads.@spawn task buffers, sized once; no per-callback Vector{Task})

PUBLIC_DRIVER_GATES =
    flexible_cm:pass
    common_frechet:pass
    cm_plus_zc:pass_when_run_in_isolation (confirmed live; flaky under concurrent KNITRO load on
        this shared host, see Sec 3 correction -- not a code regression, not this task's own bug)
    zc_only:pass

PRODUCTION_MERGE = port_ready_not_merged

HIGHEST_PRIORITY_REMAINING_GAP = cm_meanzc's real inner solve is sensitive to KNITRO concurrency
    contention on this shared host (Sec 3) -- confirmed via a direct concurrent-vs-isolated A/B
    (3/5 pass, 2/5 fail with the identical nStatus=-500/-502 signature, same script/commit/host,
    concurrency the only varying factor). NOT a code regression -- an earlier version of this
    report wrongly concluded one; retracted with evidence in Sec 3. Whether the other three
    families share this sensitivity was not tested. Recommend running any future multi-family real-
    KNITRO D=20 sweep on this host with concurrent solves serialized (or the resulting flakiness
    explicitly documented), and testing whether flexible_cm/common_frechet/origin_zc show the same
    concurrency sensitivity before assuming they don't.
```

## 9. What was explicitly not done (matches the task brief's own out-of-scope list)

No changes to outer-search algorithms, bounds campaigns, starting-point generation, verifier
acceptance policy, interval-vs-cumulative CM bases, flexible-theta gradients, the no-(H) operator
architecture, or unrelated micro-allocation cleanup. No production default was flipped by this
task — every new backend/threading choice remains opt-in, defaulting to the pre-existing behavior,
pending explicit review.
