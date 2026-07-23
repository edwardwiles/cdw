# CM + exact equal means (+ pairwise zero covariance): production integration report

Date: 2026-07-23. Branch: `integration/fullA-cm-mean-zc-current`. Base: `production/fullA-exact`
@ `22683016b5a927d4952e52049145ad8d1f5a2b87`. Tip: `dde5e44a99de0357191cd63efd6ba659eb358aa3`
(8 commits). **Not merged into `production/fullA-exact`. Not promoted.** `production/sequential-linearized`
was not touched.

Naming used throughout, per the task's own instruction: **finite-grid CM**, **finite-grid CM +
exact equal means**, **finite-grid CM + exact equal means + pairwise zero covariance**. The last
arm is zero covariance, not independence, and not "zero correlation" without a finite-positive-variance
check (present in the gate suite, see §8).

## 1. Prototype preservation

- Old worktree/branch (`gravity-experiment-fullA-cm-pairwise-zero-cov`,
  `experiment/fullA-cm-pairwise-zero-cov`) still existed, untracked/uncommitted, exactly as the
  old report claimed: 15 untracked paths, 13 Julia files, 2,293 lines (verified by direct `wc -l`,
  matches the old report's own count exactly).
- Preserved unaltered to `archive/fullA-cm-mean-zc-prototype-2026-07-22` @
  `88c45cc4da57afc6fbd176b0c22b6f6586bc6e21`, base `82dd485bb01c58ec516fe56ed280eaa928fe07ee`
  (the branch's own HEAD at time of preservation). Annotated tag
  `archive-cm-mean-zc-prototype-2026-07-22`. Both pushed to `cdw` (the canonical remote;
  `origin`/habibiscoding is stale).
- SHA-256 manifest of every preserved source file:
  `docs/experiment_cm_pairwise_zero_cov/PROTOTYPE_MANIFEST_SHA256.txt` (on the archive branch).
- Excluded: nothing of substance -- the old `results/` directory's `.jls` checkpoints were only
  ~20KB each (not the "large checkpoints" the task worried about), so all 3 CSVs + 6 checkpoints
  were preserved too, as regression fixtures.

## 2. Audit of the old prototype (hazards)

| Finding | Verdict | Disposition |
|---|---|---|
| 3.1 Canonical divergence (`-zeta`) | The old work **already found and fixed** this itself (disclosed in its own report §6), in 4 files. Residual `-zeta` references today are only explanatory comments or a legitimate `zeta==zeta` repeat-call identity test. | Clean -- no active path uses it. |
| 3.2 Shared mutable `Ref` for ν | **Confirmed real hazard.** `nu_ref::Ref{Float64}` captured in a closure built once, mutated per outer eval -- unsafe under this codebase's `par_concurrent_evals` machinery. | **Not ported.** See §4. |
| 3.3 Checkpoint schema | Old work forked its own ad hoc schema=1 driver, reasonable at the time (predates the current production launcher, which didn't exist yet at the old base commit). | **Not ported.** New `CMMeanZCCheckpoint` (§9) extends the CURRENT canonical `CMCheckpoint` discipline instead. |
| 3.4 Current backend/cache drift | `oracle.jl`, `common_marginals_moments.jl`, `cm_hessian_architectures.jl`, `cm_production_bundle.jl`, `cm_checkpoint.jl`, `cm_outer_driver.jl`, `PsiObjectiveBundle.jl` are **byte-identical** between the old base and the current production tip -- none of the 28 intervening commits touched core CM math/Hessian machinery. What changed: a new chain-based supervisor (`cm_production_stage_runner.jl`) replaced the old driver style; a generic (CM-agnostic) `cross_delta_cache.jl` was added. CM-C+ and CM-specific complete-state cache exist only on an **unmerged** branch (`perf/fullA-cm-cplus-statecache-overnight-2026-07-22`), not in production. | Integrated against the current chain-supervisor pattern (§9); CM-C+ integration explicitly out of scope since it isn't in production yet. |

## 3. Mathematical derivation (independently re-derived, then generalized)

The archived prototype's math (one outer scalar ν suffices under CM; direct-vs-anchored basis
equivalence; column-layout argument for Hessian reuse; envelope derivative) was independently
re-checked against the current (unchanged) core files and confirmed sound -- full derivation in
`cm_meanzc_moments.jl`'s own docstrings, not repeated here.

**Generalization beyond the prototype** (added mid-task, at the user's request): from a single
scalar ν (order-1 mean, optionally pairwise ZC of the levels) to `K_mean` power levels
`E_F[z_o^k] = ν_k` (`k=1..K_mean`) plus `K_pair<=K_mean` levels of pairwise zero covariance of
the k-th powers, `E_F[z_o^k z_p^k] = ν_k^2`. `K_mean=1,K_pair=0/1` reproduce the original two
named arms bit-for-bit (`meanzc_extension_to_K` sugar). This generalizes cleanly because the
restriction's functional form in `ν_k` is identical at every level (only the underlying data
column changes, `z_o` -> `z_o^k`) and the outer Jacobian is block-diagonal in `k`.

**Design change from the prototype**: ν never touches a `Ref`. `θ_ext = vcat(θ_econ,
ν_1,...,ν_{K_mean})` rides through the same `θ` argument every other outer parameter already
flows through (a plain local variable per call, functionally). This is the change that removes
hazard 3.2 by construction rather than by review.

## 4. Column layout / outer-vector conventions (unchanged meanings, extended)

`[economic (ncore_econ-1) | mean_1..mean_{K_mean} (D each) | pair_1..pair_{K_pair} (D(D-1)/2 each)
| CM-grid (ncm) | gravity]`. Outer KNITRO vector: `w_ext = [gp; zfree; eta_nu_1;...;eta_nu_{K_mean}]`
-- `w_ext[1:end-K_mean]` is byte-identical in meaning to the existing `w=[gp;zfree]` convention.
Every checkpoint, cache key, and cold-verification path includes `eta_nu` explicitly (§8-9).

## 5. Memory design (D=20/L=50, real data)

Scoped single-point measurement (not a full campaign), `W=80000`, `draw_seed=20260719`,
`:pseudorandom`, `contrasts=:orthonormal` (matching the actual completed 2026-07-22 CM campaign's
own convention). Full log: `docs/experiment_cm_pairwise_zero_cov/d20_memory_and_nu_bounds_run_log.txt`.

| Object | Size |
|---|---|
| `Zraw_all[1]` (W x D) | 12.8 MB |
| `Zpairraw_all[1]` (W x 190) | **121.6 MB** (matches the task's own back-of-envelope estimate exactly) |
| CM dense reference matrix | 608.0 MB |
| `Bidx` | 12.8 MB |
| Architecture C `Hfull` scratch | 19.5 MB |
| `obj.H` (the dominant single buffer, W x (d+2)) | 1001.0 MB |
| **Process peak (VmHWM)** | **8.68 GB** |

vs. the archived prototype's reported 42.5GB at comparable L=50 ZC-arm settings -- **~5x lower**.
Root cause of the old figure was not independently re-derived (the old source predates this
investigation and was not re-run), but the design here avoids the two most likely culprits by
construction: `Zraw_all`/`Zpairraw_all` are built ONCE per context and reused (never rebuilt per
inner-solve callback), and no new persistent per-call buffers were introduced beyond what
Architecture C's existing `CMBinHessCtx` already allocates (only its `NCORE` field widens).

**Resource recommendations** (derived from the single-process 8.68GB peak measured above, with a
margin for real optimization runs touching more of the outer space than one calibration point,
and OS/Julia baseline overhead):
- **One process**: budget 12-16GB. Comfortable on any node with >=32GB.
- **Two parallel chains**: budget 24-32GB total. Comfortable on any node with >=64GB; do not
  assume linear scaling is exact (draws/context are per-process, not shared).
- **Three parallel chains**: budget 36-48GB total. This has NOT been measured directly (no 2- or
  3-process concurrent run was performed this session) -- the recommendation is an extrapolation,
  not a validated ceiling. Do not treat 3 as safe merely because the old prototype ran ONE process
  successfully; re-measure with an actual 2-3-process concurrent run before committing a real
  campaign to that concurrency level.

## 6. ν parameterization and bounds

`η_ν = log(ν)` (only option implemented, matches the prototype's own choice: draws are `Exp(1)`,
strictly positive, so any feasible ν must be positive). Default box:
`meanzc_default_nu_bounds` = ±4x the hard finite-support interval (`nu_feasible_interval`,
data-derived, not a guessed constant) -- deliberately wide, e.g. `eta_nu in [-11.66, 3.69]` at the
D=20/L=50 point tested.

**Empirical finding**: a first coarse 7-point profiling grid spanning `ν in [0.3, 3.5]` at that
point found only 1/7 points KNITRO-feasible; a finer 9-point grid concentrated near the natural
anchor (`ν≈1`, since `E[Exp(1)]=1`) found 5/9 feasible with a genuine interior minimum
(`Delta=0.043` at `η_ν=0.038`/`ν=1.039`, curving up to `Delta=6.4` at the profiled edges) -- real
curvature evidence for an interior optimum, not a single lucky point. This confirms the wide
DEFAULT box is not binding, but also confirms the task brief's own guidance that grid profiling
at a fixed arbitrary `(g,A)` is a feasibility diagnostic, not a substitute for joint optimization:
most of the wide box is simply infeasible at any FIXED `(g,A)` point, and the real search should
optimize `(g,A,η_ν)` jointly (as `run_cm_meanzc_upper` does; confirmed working end-to-end in
§9's smoke test, `nu_box_interior=[true]`).

## 7. Outer ν-derivative

Re-derived (not copied) from the current code's own sign/scaling convention (`cm_meanzc_moments.jl`
§6 docstring). `∂Delta_dual/∂ν_k = -mean_m·(Σ_o λ_mean,o,k* · (-1) + Σ_{o<p} λ_pair,op,k* · (-2ν_k))`,
`∂Delta_dual/∂η_{ν,k} = ν_k · (that)`. Validated at D=4 (K=1 and K=2) against BOTH a fixed-dual and
a reoptimized central FD in `η_ν`-space: matches to **~1e-9 to 1e-12** at every tested point (§8).
Not independently re-validated at D=20 this session (deferred; the D=4 validation exercises the
identical formula/code path, just at smaller scale).

## 8. Layered correctness gates (D=4)

All numbers below are from actual test runs this session (logs available via the commit history;
not re-pasted in full here).

| Gate | Scope | Result |
|---|---|---|
| 8.1 Pure moment tests | D=3/4/5/20, K∈{1,2}, no KNITRO | **746 assertions pass** (`test_cm_meanzc_pure_moments.jl`) |
| 8.2a Hessian equivalence (dense vs structured) | D=4, 4 configs `(K_mean,K_pair)∈{(1,0),(1,1),(2,0),(2,2)}`, both bases | **24/24 pass**, max diff 2.1e-15 to 5.6e-15 |
| 8.2b Inner-solve equivalence + canonical `Delta_dual` + KKT | same 4 configs x 2 bases | **90/90 pass** |
| 8.3 Fixed-point nesting | D=4: CM <= CM+mean <= CM+mean+ZC (K=1), AND K=1 <= K=2 (more restrictions) | **11/11 pass**, both nestings hold |
| 8.4 Full outer gradient vs trusted reference | D=4, all 4 configs, incl. eta_nu components | **24/24 pass**; (g,A) block cosine similarity 0.9999-0.99999 vs `full_rebuild_gradient_fallback_meanzc` (this codebase's own trusted fixed-dual full-rebuild reference -- see the methodology note below); eta_nu components match reoptimized FD to ~1e-9 to 1e-12 |
| 8.5 CM-only regression | D=4, extension files loaded but `cm_extension=:cm_only` | **7/7 pass**, bit-identical `Delta`/`ζstar`/`λstar` to the pre-existing path |

**Validation methodology note** (a real mistake made and corrected this session): the codebase's
`composite_gradient_at_fast` is itself an adaptive-bandwidth secant method around a possibly-
nonsmooth (winner-switching) objective -- a naive fixed-`h` central FD is NOT a valid ground
truth for it. Confirmed live: even PLAIN CM's own already-trusted analytic gradient disagreed
with a naive `h=1e-5` probe by up to 0.14. The correct reference, already established elsewhere
in this codebase (`full_rebuild_gradient_fallback`/`fixed_dual_L`), is a fixed-dual full-rebuild
central FD at `h=0.01`; meanzc-aware analogs (`fixed_dual_L_meanzc`/
`full_rebuild_gradient_fallback_meanzc`) were added and used instead, matching plain CM's own
~0.9999 cosine-similarity validation quality.

**D=20 fixed-point/gradient gates**: NOT run this session (deferred to the D=20 shakedown phase,
§13). The D=4 gates exercise the identical code paths; only scale/timing differ.

## 9. Production integration gates

### 9.1 Checkpoint/supervisor

`CMMeanZCCheckpoint` (new type, not a schema bump of `CMCheckpoint` -- Julia's `Serialization` is
not layout-tolerant across struct changes, so bumping in place would make the just-completed
2026-07-22 CM campaign's own schema-2 checkpoints undeserializable outright instead of cleanly
refusing). Provenance validated via `context_fingerprint(ctx)` (reused unmodified, already
handles both D=4 test and D=20 production contexts). `run_cm_meanzc_upper_checkpointed` is
generic over a `ctx_builder` function (production passes `d20_real_setup_design`, matching the
current canonical driver's own pattern) -- this is what let the schema/resume MECHANICS be
tested at D=4 speed.

**16/16 assertions pass**: clean completion + checkpoint write, resume continues
`n_eval`/`wall_elapsed` (never resets), schema-mismatch hard-refused, context-fingerprint-
mismatch hard-refused, cold re-verification of the exact stored best vector matches to **0.0
diff**.

Not run: a real D=20 checkpoint/interrupt/SIGKILL/resume cycle (promotion gate, §13, deferred).

### 9.2 Cache / accepted-point reuse

`CMMeanZCEvalKey` mirrors `CMEvalKey` exactly, reusing `SafeExactCache` unmodified. `x_free` AND
`eta_nu` are both part of the key -- "accepted-point reuse requires exact equality of the full
outer vector including eta_nu" holds by construction, verified empirically.

**20/20 assertions pass**: A/B/A restoration (bit-identical, no recompute), different-nu-same-
x_free is a distinct key (not a stale hit), changed L / CM cutpoints / basis / K_mean / K_pair /
contrasts / backend label / draw checksum / context fingerprint / delta -- each independently
verified to be a guaranteed cache miss using ONE deliberately-shared cache instance across
configurations (not relying on separate-instance isolation to hide a key-construction bug).
Unverified/infeasible points confirmed never cached.

### 9.3 Concurrency

**First attempt failed and is documented as a lesson, not hidden**: spawning many independent
full `KN_new()`/`KN_solve()` sessions via Julia `Threads.@threads` deadlocked for ~3.5 hours
(confirmed via `ps`: stuck in a non-Julia call, unresponsive to `SIGTERM`, required `SIGKILL`).
This is a real, pre-existing hazard class (non-reentrant KNITRO/OpenMP lock under concurrent
full-solve invocation), orthogonal to this extension's ν design, and not a pattern this
codebase's actual production concurrency model uses (real parallelism there is per-OS-process,
e.g. the 3-chain CM campaign, never concurrent `KN_solve` within one process).

**Corrected test**: the actual mechanism the task's hazard concern is about is KNITRO's own
C-level concurrent dispatch of `cb_F!`/`cb_G!` for different trial points WITHIN one outer solve,
controlled by `par_concurrent_evals` -- confirmed **set to `yes`** in production's own CM outer
opt file (`csw_outer_wallclock_sr1.opt`). The corrected test ran a real outer solve under that
exact production setting (D=4, K=1/K_pair=1), then cold-re-verified the returned incumbent from a
freshly rebuilt context/inner solve: **matched to 1.1e-16**. Repeated once more for run-to-run
sanity. **6/6 assertions pass.**

Every backgrounded run for the remainder of this session was OS-level `timeout`-wrapped after
this incident, so a hang can no longer run unbounded.

### 9.4 CM-C+ compatibility

Not applicable -- CM-C+ is not in production (§2), so there is nothing to integrate against yet.

## 10. Production driver / config

`CMMeanZCConfig` (sits alongside `CMConfig`, does not modify it) exposes
`cm_extension::Symbol ∈ (:cm_only, :cm_plus_equal_means, :cm_plus_equal_means_zero_covariance,
:cm_plus_moments)`, default `:cm_only`. `run_cm_meanzc_upper`'s `:cm_only` branch calls the
existing `run_cm_upper` directly (one line, no wrapper logic) -- verified both by code inspection
and empirically (first-callback-evaluation equivalence; full-run `n_eval`/`xsol` are NOT expected
to match between two separately wall-clock-timed runs of the same algorithm -- confirmed live,
29 vs 342 evals in "the same" 15s budget, purely a JIT-warmup/timing artifact, not a divergent
code path).

**11/11 smoke-test assertions pass** (D=4, short/bounded runs, no convergence claim): `:cm_only`
delegation consistency, and a real end-to-end `:cm_plus_equal_means_zero_covariance` outer solve
(`n_eval=90`, feasible verified incumbent found, `nu_box_interior=[true]`).

## 11. D=20 three-arm scientific shakedown (Step 10)

**Not run.** Explicitly paused per direct agreement before starting this task's expensive-compute
phase. What WAS run at D=20 is the scoped single-point memory/ν-bounds check in §5-6 (fixed-point
evaluation only, not an outer-loop campaign). No κ number from this integration should be treated
as reportable; none is claimed.

## 12. Promotion decision

**Not promoted.** Per the task's own promotion gate list:

| Gate | Status |
|---|---|
| Old prototype preserved | **Done** (§1) |
| Current-production regression gates pass | **Done at D=4** (§8.5); not re-run at D=20 |
| No shared-Ref race remains | **Done** (§3, §9.3 -- architectural elimination + empirical confirmation) |
| Checkpoint/supervisor/cache integration passes | **Done at D=4** (§9.1-9.2); not exercised against a real D=20 run |
| D=20 memory understood and operationally acceptable | **Done** (§5) -- 8.68GB single-process, well within typical node budgets; 2-3 concurrent chains extrapolated, not measured |
| Current Reference and C+ results agree where both supported | **N/A** -- C+ not in production (§2) |
| At least one real D=20 ZC checkpoint cold-verifies after interrupt/resume | **Not done** |
| All active paths use canonical `Delta_dual` | **Done** (§8, verified throughout) |
| Option off by default | **Done** (`cm_extension=:cm_only` default) |

Two gates remain open: a real D=20 checkpoint/interrupt/resume cycle, and the D=20 three-arm
shakedown itself (which several other gates, e.g. current-production regression at D=20, would
naturally be re-confirmed alongside). **The integration branch is left clean, all work
committed, nothing merged into `production/fullA-exact`.**

## 13. Blocker list for a future session

1. Real D=20 checkpoint interrupt/SIGKILL/resume cycle (9.1's remaining gate).
2. D=20 fixed-point nesting + full gradient-vs-reference re-confirmation at D=20 (the D=4 gates
   exercise identical code paths, but the task's own gate list asks for D=20 too).
3. Measured (not extrapolated) 2- and 3-concurrent-chain memory ceiling.
4. The D=20 three-arm shakedown itself (Step 10) -- matched start vectors across CM/CM+mean/
   CM+mean+ZC, cross-seeding between arms, exploration-quality reporting (not a convergence
   claim).
5. Only after 1-4: a genuine promotion decision.

## 14. Production invocation examples

```julia
# finite-grid CM only (current production path, unchanged, extension code never reached)
cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 50, cm_grid_rule = :nested_family,
                                    cm_grid_sizes = [10,20,50], contrasts = :orthonormal))
                                    # cm_extension defaults to :cm_only

# finite-grid CM + exact equal means
cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 50, contrasts = :orthonormal),
                      cm_extension = :cm_plus_equal_means, meanzc_basis = :direct)

# finite-grid CM + exact equal means + pairwise zero covariance
cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 50, contrasts = :orthonormal),
                      cm_extension = :cm_plus_equal_means_zero_covariance, meanzc_basis = :direct)

# generalized K_mean/K_pair (e.g. K=2, tested this session)
cfg = CMMeanZCConfig(cm = CMConfig(cm_grid_size = 50, contrasts = :orthonormal),
                      cm_extension = :cm_plus_moments, meanzc_K_mean = 2, meanzc_K_pair = 2,
                      meanzc_basis = :direct)

# single-shot outer solve (diagnostic/gate-testing entry point):
res = run_cm_meanzc_upper(cfg, ctx, pe, w0_ext; delta = 1.0, maxtime_real = 180.0)

# checkpointed production entry point (D=20, matching the current canonical driver's own pattern):
res = run_cm_meanzc_upper_checkpointed(d20_real_setup_design, w0_ext;
    W = 80_000, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 50, contrasts = :orthonormal, probs = nested_grid_sequence([10,20,50])[50],
    K_mean = 1, K_pair = 1, meanzc_basis = :direct,
    maxtime_real = 3600.0, ckpt_dir = "...", label = "stage", checkpoint_interval_s = 90.0)
```

## 15. Resource recommendations (repeated from §5 for convenience)

- **1 process**: 12-16GB.
- **2 parallel chains**: 24-32GB total (not independently measured; linear-scaling extrapolation).
- **3 parallel chains**: 36-48GB total (**not measured at all** -- do not treat as validated).
