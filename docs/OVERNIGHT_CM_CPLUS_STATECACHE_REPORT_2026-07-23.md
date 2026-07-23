# Overnight CM-C+ / complete-state cache — final report — 2026-07-22/23

Branch: `perf/fullA-cm-cplus-statecache-overnight-2026-07-22`
Worktree: `/bbkinghome/edav/gravity_robustness/gravity-perf-fullA-cm-cplus-statecache-overnight`
Base commit: `22683016b5a927d4952e52049145ad8d1f5a2b87` (`production/fullA-exact` == `cdw/production/fullA-exact`, tag `cm-production-ready-2026-07-22-r3`)
Final commit: `fbe0d17d890e2efdc05230c49a665533b4251ed3`

**Not merged into `production/fullA-exact` or `production/sequential-linearized`.** Neither
canonical trunk was touched. The live CM smoke-test process (pid tree rooted at 170719, chain 91,
`production_runs/cm_smoketest_2026-07-22/part1_clean`) was confirmed still running, untouched, on
its original cores (0-19) at every checkpoint through this session, including immediately before
writing this report (33m32s elapsed at last check).

## Commits on this branch

| Commit | Summary |
|---|---|
| `f0b3d3627c234df71469263c4d1276f027bce28d` | Baseline record + CM gradient algebra trace (no behavior change) |
| `a61af6662cbba8aa33d2075fb077147bf56134f2` | CM-aware C+ gradient backend behind `cm_gradient_backend` selector |
| `6da8a938fca4c5221a198509b04276d36be65031` | CM-C+ D=4 correctness battery (48/48 pass) |
| `b9c4e33daf2c9e3f0298fff9f1908a58c4f1ca97` | CM-C+ real D=20/W=80000/L=50 fixed-point gate (PASS, 2.79x speedup) |
| `fbe0d17d890e2efdc05230c49a665533b4251ed3` | Complete-state cache: design doc, opt-in prototype, D=4 tests (32/32 pass) |

Total diff vs base: 13 files changed, 1573 insertions(+), 2 deletions(-) (the 2 deletions are the
two lines in `cm_checkpoint.jl`/`cm_production_stage_runner.jl` that a new line was inserted
adjacent to — no line was removed).

---

## Feature 1: CM-aware C+ gradient backend

### Implementation status

**Complete and wired end-to-end**, opt-in, default off. `cm_gradient_backend::Symbol = :reference`
kwarg on `run_cm_upper_checkpointed` (`cm_checkpoint.jl`); `cb_G!` dispatches to
`cm_production_gradient_cplus` (new, `lfix_cm_cplus.jl`) only when `:cplus` is explicitly passed.
Resolved backend is logged explicitly at startup and recorded to a
`<label>_gradient_backend.txt` sidecar next to the checkpoint (deliberately does not touch
`CMCheckpoint`'s binary schema — see "Known risks" below). No existing function was modified
except `cm_checkpoint.jl` itself (the selector's only possible home) and
`cm_production_stage_runner.jl` (one new `include` line). `composite_gradient_at_Cplus` (the live
unrestricted-path production function) and every other pre-existing function are byte-for-byte
untouched.

The algebraic decomposition the implementation follows is derived, not assumed, in
`docs/CM_GRADIENT_ALGEBRA_TRACE_2026-07-22.md`: the CM-augmented base-point dual scalar
`q_s(θ) = -ζ* - λ_G*'G_s(θ) - λ_C*'C_s` has a θ-independent `λ_C*'C_s` term, computed once via the
UNMODIFIED `cm_fixed_contribution` (shared with the Reference CM path) and folded into whichever
economic-block builder's `q0` is in use — `build_lfix_base_cache` (Reference) or
`build_lfix_base_cache_C`/`build_lfix_base_cache_C!` (C+), both of which satisfy the same two
structural conditions (`λstar[1:D²]`-only indexing, identical counterfactual-tail check) that make
this decomposition valid.

### Tests actually run

1. **D=4 correctness battery** (`overnight_cm_cplus_d4_gate.jl`), both `contrasts ∈
   {:anchored, :orthonormal}`, 3 points (calibration + two perturbation scales chosen to explore
   different winner configurations): base-point `q0` agreement, full production-wiring gradient
   agreement (`cm_production_gradient` vs `cm_production_gradient_cplus`, matched `h=0.01`),
   independent-bandwidth-bisection agreement, and a fixed-dual-vs-independently-reoptimized
   directional check (Section 2.4-style, kept separate from the Section 2.3-style equivalence
   check). **48/48 pass.**
2. **D=20/W=80,000/L=50 real-data single fixed-point gate**
   (`overnight_cm_cplus_d20_fixedpoint.jl`), calibration point, `contrasts=:orthonormal`
   (production's actual decision), both backends sharing the identical `archC_verified_state`
   base. **PASS.**

### Tests NOT run (disclosed, not silently skipped)

- The brief's full "exhaustive or near-exhaustive" D=4 battery (every free coordinate
  individually with a dedicated targeted probe, forced exact ties, near-ties, nonfinite-probe
  retry-discipline exercise, CM-only-column-only configurations, top-three-fast-path vs
  generic-fallback isolation as *separate* explicit sub-tests) was **not** run in full — the
  battery actually run exercises representative points (including one perturbation scale
  deliberately chosen via a different-seed larger perturbation to be more likely to shift
  winners, though winner flips were not explicitly counted/forced) rather than every listed
  category individually.
- D=20 **multi-point** fixed-point gates (benchmark/calibration point was covered; a cold-verified
  δ≈0.1 point, a δ≈1 point via the outer-loop `delta` grid specifically, a winner-switch-heavy
  point, and a point recovered from the live campaign's own completed output) were **not** run —
  time/resource budget for this session; the live campaign's own completed CM output was not yet
  available to read safely during this session's window.
- **Trajectory comparison** (matched short KNITRO outer-loop runs at δ=0.1 and δ=1,
  `:reference` vs `:cplus`, same start/budget) was **not** run.
- **Same-state replay** (record states from one trajectory, time both backends at those exact
  states) was **not** run — depends on the trajectory run above.
- `contrasts=:anchored` at D=20 was skipped (only `:orthonormal`, production's actual decision,
  was run) to bound wall time within this session's resource-conscious window.

### D=4 maximum numerical differences (CM-C+ vs CM-Reference)

- Base-point `q0`: max `5.7e-14` (calibration `8.9e-16`).
- Full-vector gradient (matched `h=0.01`): max `|Δg| = 2.0e-13`, relative error `5.9e-14`,
  cosine `1.0` to `1e-10`, **zero A-block sign mismatches** across every point/contrast/coordinate
  tested.
- Own-bandwidth (independent bisections): cosine `1.0` to `5e-14`.
- Fixed-dual secant: `secant(C+) == secant(Reference)` exactly at every probed coordinate; both
  differ from the independently-reoptimized true secant by the same known, pre-existing AUD-05
  approximation gap (not a C+-specific error — this is the established "fixed-dual gradient is an
  approximation to the re-optimized derivative" property shared by every backend on this path,
  and this session's tests correctly keep that gate separate from the CM-C+-vs-CM-Reference
  equivalence gate rather than conflating the two).

### D=20 maximum numerical differences

Single point (calibration, `contrasts=:orthonormal`): `max|Δg| = 1.535e-16`, relative error
`8.9e-16`, cosine `1.0000000000`, zero sign mismatches across all 399 A-block coordinates.

### Complete callback timing and memory (D=20/W=80,000/L=50, calibration, `threaded=true`,
`JULIA_NUM_THREADS=12`, `taskset` pinned away from the live campaign's cores)

| Backend | Wall | Allocated | GC time |
|---|---|---|---|
| Reference | 14.850 s | 14,186.6 MB | 0.247 s |
| C+ | 5.322 s | 306.2 MB | 0.000 s |

**2.79x wall-clock speedup, ~46.3x less allocation.** One data point only (not a distribution
across many outer-loop evaluations); consistent in direction and rough magnitude with the
unrestricted-path C+ adoption decision already made in production (`c10_d20_production_driver.jl`,
4.0-4.2x vs Reference per that decision's own report) — the CM-specific overhead (one O(W·D)
`cm_fixed_contribution` call per gradient call) is evidently small relative to the backend
difference itself.

### Trajectory / same-state replay results

Not run this session (see "Tests NOT run" above).

### Known risks

- `cm_gradient_backend` is **not** part of `CMCheckpoint`'s persisted schema — a resumed run must
  be told the same backend explicitly by its caller (matches how `maxtime_real`,
  `checkpoint_interval_s`, `verbose` already work; deliberate, documented in the kwarg's own
  comment) — but this means a checkpoint file alone does not reveal which backend produced it;
  only the sidecar `<label>_gradient_backend.txt` (which is NOT versioned/schema-checked, could
  drift out of sync with a manually-edited checkpoint) does.
- The full near-exhaustive D=4 battery and the multi-point/trajectory D=20 gates are incomplete
  (see above) — the evidence gathered is strong but not the complete gate set the brief specifies.
- `build_lfix_base_cache_C!`'s own `validate_dense=true` internal self-check is incompatible with
  a CM-augmented `ctx` (documented in the D=4 test's commit message and inline in the test file)
  — this is a **pre-existing** property of unmodified code, not introduced by this session, but a
  future caller who naively passes `validate_dense=true` against `ctx_cm` will see a spurious
  failure and should know it is not a real bug.

### Suitable for further review?

**Yes.** The core algebraic claim is derived from the existing code (not assumed), matches at
machine precision in every test run so far (D=4 and the one real D=20 point), and the
implementation touches production code (`cm_checkpoint.jl`) minimally and reversibly (a single
`Symbol` kwarg, default-preserving).

### Suitable for production promotion after review?

**Not yet** — the correctness evidence, while strong, does not yet cover the full gate list (no
multi-point D=20, no trajectory, no same-state replay, no forced-tie/near-tie D=4 sub-tests). A
follow-up session completing those specific gaps (they are narrow, well-defined, and this
session's harness — `overnight_cm_cplus_d4_gate.jl`/`overnight_cm_cplus_d20_fixedpoint.jl` —
is directly extensible to them) would close the gap to a promotion-ready state.

### Classification

**READY_FOR_REVIEW_NOT_MERGED**

---

## Feature 2: complete-state exact/cross-delta cache

### Implementation status

**Opt-in prototype, NOT wired into `cm_checkpoint.jl`'s `cb_F!`/`cb_G!`.** Field-by-field
mutability audit (`docs/COMPLETE_STATE_CACHE_DESIGN_2026-07-22.md`) establishes that
`BaseDualState` and `archC_verified_state`'s `verify` NamedTuple are already fully-owned,
non-aliased data (every array field is a fresh `collect`/`copy`, never scratch — provable from
`solve_base_state`/`archC_verified_state`'s own source, and already relied upon by
`cm_checkpoint.jl`'s own `last_F_state[]` pattern), which materially simplifies the cache: entries
can be stored/returned by direct reference with no extra defensive copying. Implemented in
`complete_state_cache.jl`: `CompleteStateCache` (bounded, `Dict`-based LRU via a monotonic access
counter, no new external dependency), `complete_state_fingerprint` (draws, W/D/σ, CM
config/L/grid/basis/contrasts, `.opt`-file CONTENT hash, KNITRO release, schema version — `delta`
and `cm_gradient_backend` deliberately excluded, both with derived justification in the design
doc), `complete_state_lookup!`/`complete_state_store!`, and `archC_verified_state_cached!` (a
purely-additive wrapper around the unmodified `archC_verified_state`; `cache=nothing`, the default
everywhere, is behaviorally identical to calling `archC_verified_state` directly).

### Tests actually run (D=4 only)

`overnight_complete_state_cache_d4_gate.jl`, 32 checks, **32/32 pass**:

- A/B/A: cold-solve A (store), solve B (store), restore A (hit), compare every `BaseDualState`
  field + `verify.Delta_dual`/`inner_status` + a downstream gradient computed from the restored
  base, against a **fresh, cache-disabled** cold solve at A.
- No-aliasing: after 5 further (uncached) solves mutate `ctx_cm.obj`'s own live scratch, the
  already-cached entry's `m_star` is confirmed bit-identical to its value at store time.
- 7 independent context-mismatch fingerprint checks (different `L`, perturbed `probs`, different
  `contrasts`, different `cm_hessian_backend`, different KNITRO release string, same `.opt`-file
  *path* with different *content*, perturbed draws) — each produces a different fingerprint.
- Cross-delta key-independence: the fingerprint function has no `delta` parameter; demonstrated
  directly that two conceptually-different-δ callers get an identical fingerprint.
- Never-store-on-failure: an inner-solve failure (`CMExpectedSolveFailure`) at a deliberately
  infeasible point leaves the cache with zero new entries.
- Bounded LRU eviction: `max_entries=2`, 3 distinct inserts with a re-touch in between — correct
  entry evicted, `evictions` counter correct.
- Instrumentation sanity: `bytes_current`/`bytes_peak`/`fresh_base_solve_wall_counterfactual` all
  populated and self-consistent.

### Tests NOT run (disclosed)

- **Not wired into `cm_checkpoint.jl` at all** — no checkpoint/interrupt/resume interaction test
  was possible or run (there is nothing checkpoint-shaped for the cache to interact with yet).
- No D=20 measurement of hit rates or wall savings on real repeated-accepted-point / staged-δ /
  matched-trajectory patterns.
- No LRU eviction stress test at a realistic `max_entries` under real memory pressure.
- No test against the `ExactInfeasible` policy interaction described in the brief (§3.3's last
  bullet) — the never-store-on-failure test covers the general "don't invent a state" property,
  but not that specific policy's own edge cases.
- Combined CM-C+ + cache interaction gates (§4 of the brief) were **not** run — the cache is not
  wired into either gradient backend's production call path, so there is nothing to combine yet.

### D=4 maximum numerical differences

Restoration is **exact** (bit-for-bit) versus what was stored (no-aliasing test). Versus a FRESH,
independent re-solve at the same point, differences of `~1e-11` to `~1e-13` were observed in
`ζstar`/`λstar`/`m_star`/`Delta_dual` — **a real property of the underlying tolerance-based KNITRO
solver (two independent solves of the same problem are not bit-identical), not a cache defect.**
This is disclosed explicitly because it is a genuine, useful finding: any caller comparing a
cache-hit result against an expected "should equal a fresh solve exactly" invariant must use a
tolerance (`~1e-8` proved comfortably safe here, many orders of magnitude above the observed noise
floor and many orders below any correctness-relevant scale) rather than exact equality.

### D=20 maximum numerical differences

Not measured (cache not exercised at D=20 this session).

### Complete callback timing and memory

Not measured at D=20. At D=4, `fresh_base_solve_wall_counterfactual` accumulated `~4.0s` of real
inner-solve wall time avoided across the 1 real hit recorded in the A/B/A test — not a meaningful
production-scale number (D=4 solves are cheap; the interesting regime is D=20/W=80,000, not
measured).

### Cache hit rates and inner solves avoided

D=4 test only: 1 hit, 2 misses, 0 evictions in the main A/B/A sequence (by test construction, not
a realistic workload sample); the LRU test separately confirmed 1 eviction under a deliberately
tight `max_entries=2` bound. **No D=20/real-workload hit-rate or wall-value measurement was
made** — per the brief's own §3.4 instruction, a cache with unmeasured value must not be promoted
past prototype status, and none is claimed here.

### Known risks

- **Entirely unwired** — this is a tested library, not yet a production capability. Wiring it into
  `cm_checkpoint.jl`'s `cb_F!`/`cb_G!` (an `Union{Nothing,CompleteStateCache}` kwarg on
  `run_cm_upper_checkpointed`, mirroring `cm_gradient_backend`'s own pattern) is the next concrete
  step and was scoped but not attempted this session.
- The LRU eviction policy is O(`max_entries`) per store (linear scan for the minimum access
  counter) — fine at the brief's own expected scale ("tens, not thousands" of entries) but would
  need revisiting if a future use case wanted a much larger cache.
- `Base.summarysize` (used for the `bytes` instrumentation) is not free to call; at D=20 scale with
  a `BaseDualState` containing several length-D² vectors this is still cheap relative to an inner
  solve, but was not independently benchmarked.
- No interaction with checkpoint/resume has been tested, so a caller wiring this in for a
  real overnight campaign should NOT assume checkpoint-resume compatibility without first adding
  that specific test.

### Suitable for further review?

**Yes, as a design + prototype.** The field-level mutability argument is sound and directly
verifiable against the cited source lines; the D=4 test suite genuinely exercises every
correctness property the brief's §3.3 lists that is *possible* to exercise without wiring the
cache into the checkpoint driver.

### Suitable for production promotion after review?

**No.** Per the brief's own §3.4 ("a technically correct cache with negligible hits or negative
wall value should remain an optional prototype, not be promoted"), promotion requires a
demonstrated D=20 wall-time/hit-rate benefit this session did not measure. Correctness alone is
not a promotion argument.

### Classification

**PARTIAL_PROTOTYPE**

---

## Interaction between the two features (brief §4)

**Not tested.** The complete-state cache is not wired into either gradient backend's production
call path (`cm_production_gradient`/`cm_production_gradient_cplus` both call
`archC_base_state`/`archC_verified_state` directly, not the cache-aware wrapper), so there is
nothing to combine. This is a direct, disclosed consequence of the complete-state cache remaining
a standalone-tested prototype rather than a wired-in feature this session — see Feature 2's
"Tests NOT run" above.

---

## Overall session notes

- Production safety: verified at the start (exact tested commit match, live campaign identified
  and left untouched) and re-verified at the end (campaign still running on its original cores,
  production worktree still clean at the same commit). No production-run, checkpoint, cache, log,
  seed, or output directory was written into. Exactly one branch, one worktree, per the brief.
- Resource discipline: all D=4 work and the one D=20 run were explicitly `taskset`-pinned away from
  cores 0-19 (reserved for the live campaign); the D=20 run's core range was checked for idle
  capacity via `mpstat` before launch and ran in the background so it did not block other work.
- Two real (pre-existing, unrelated) bugs/gotchas were discovered and documented, not fixed (out
  of scope — this session only touches its own new files plus the two minimal `cm_checkpoint.jl`/
  `cm_production_stage_runner.jl` edits): `cm_config.jl` has a docstring-target parse error when
  included stand-alone (it is not part of the actual production include chain, so this has likely
  never been hit in production); `build_lfix_base_cache_C!`'s `validate_dense=true` self-check is
  structurally incompatible with a CM-augmented context (an omission, not a defect, in the
  pre-existing function — it was simply never exercised against a CM context with that flag set
  before this session).
- Push: the branch is clean (working tree matches the last commit, `git status --short` empty) and
  every commit is intentional and reviewed above. Will be pushed to `cdw` as the final action of
  this report, per the brief's explicit authorization ("push the single experimental branch to cdw
  only if it is clean and all commits are intentional"). No production tag was created.
