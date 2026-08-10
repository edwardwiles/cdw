# Handover: 8-hour delta=1 upper-bound runs for OZC-CROSS and CM+ZC-CROSS

Paste everything below the line as your task prompt in a fresh Claude Code session.

---

## Your goal

Launch and see through **two real production upper-bound searches at `delta = 1.0`**, one per new
cross-power restriction family, each with an **8-hour wall budget**, at the real D20 production
settings — then report the resulting bound honestly.

Before launching, spend a **strictly timeboxed ~90 minutes** on Phase 0 below: there is one verified
performance regression that would otherwise make one of the two runs several times slower than it
needs to be. If Phase 0 overruns its box, abandon it and launch anyway — the runs are the priority.

## Where things stand

Worktree `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`, branch
`feature/ozc-cross-2026-08-09`, based on `origin/production/fullA-exact`@`4df5254`.
**Everything is uncommitted working-tree state.**

Both families are already verified and wired end-to-end through the real checkpointed production
drivers and the real campaign runner (D4 gates 119 PASS/0 FAIL; production smokes with working
checkpoint/resume). Full write-up: `docs/CMZC_CROSS_AND_CAMPAIGN_INTEGRATION_2026-08-09.md`, plus
`00_READ_FIRST_CORRECTION.md` in `dropbox:.../cmzc_cross_and_campaign_2026-08-09/`. **Read both
before starting.** What has *not* been done is any converged search — every prior run was
budget-terminated (`-401`) after 5–24 outer evaluations. That is what these runs are for.

**Commit the working tree to the branch before launching** (do not push to any remote without
asking). An 8-hour run whose provenance is an uncommitted diff is not reproducible.

## Phase 0 — timeboxed ~90 min, do this first

### 0a. The regression (verified 2026-08-10, one line)

`cm_checkpoint.jl:1509-1510`:

```julia
effective_blas_threads = blas_threads !== nothing ? blas_threads :
    (family_tag === :cm_meanzc ? ZC_GRAM_BLAS_THREADS_DEFAULT[] : nothing)
```

The 2026-08-09 CM+ZC-CROSS work set `family_tag = :cm_meanzc_cross`, which no longer matches this
gate. So **CM+ZC-CROSS silently runs with ambient (=1) BLAS threads where the diagonal `cm_meanzc`
family gets the validated 8** — on the family with the *larger* `H_ZZ`. OZC-CROSS is unaffected:
`run_originzc_upper_checkpointed` takes `blas_threads::Union{Nothing,Int} = ZC_GRAM_BLAS_THREADS_DEFAULT[]`
as a plain kwarg default (`cm_originzc_checkpoint.jl:574`, applied at `:789`) with no family gate.

Fix the gate so both CM+ZC variants qualify. Prefer keying on "is this the CM+ZC family at all"
rather than adding a second literal, so a future third arm cannot fall through the same crack.

### 0b. Why this matters, and what is NOT worth trying

Measured 2026-08-10 (D20/W=100k/L=50, cold solve, both families at 1 BLAS thread): the Hessian
callback dominates the inner solve, and within it `H_ZZ` — the restriction gram — dominates and is
the only term growing quadratically in restriction width `nx`. `nx` 630 → 1770 took `H_ZZ`
1.41 → 7.52 s/call (**8.02×**, against the `O(W·nx²)` prediction 7.89×). The genuine economic
Hessian is ~0.03 s/call. Evidence: `key_results/15_iters_vs_per_iteration_2026-08-10.txt`.

**Already done, do not redo:** `ZC_GRAM_BACKEND_DEFAULT[] = :blas_syrk`, so the symmetry-exploiting
`syrk` backend (half the FLOPs of a full gemm) is *already* the default and was already in use in
every measurement above. There is no 2×-from-symmetry win still on the table.

**Not worth trying:** micro-optimising the kernel. `syrk` at ~42 GFLOP/s single-threaded is
reasonable for the operation; it is genuinely compute-bound `O(W·nx²)`. Do not go looking for a
better inner loop.

**Worth trying, in this order:**

1. **BLAS threads (the main lever).** After the gate fix, measure a single cold inner solve at
   K=3/3 for each family at `BLAS.set_num_threads` ∈ {1, 8, 16}. `syrk` is BLAS-3 and should thread
   well, but **measure it — do not assume**. Note `OPENBLAS_NUM_THREADS=1` is a repo-wide hard rule
   for ambient use; the 8-thread setting here is a *specific, validated, driver-scoped* exception
   already established for `cm_meanzc`, not a licence to raise it globally.
2. **A free inefficiency in `zc_gram_blas_syrk!`** (`zc_gram_blas_candidates.jl:137-141`): it
   evaluates `sqrt(S[w])` inside the `j` loop, i.e. `W*nx` ≈ 1.8e8 square roots per call instead of
   `W` = 1e5. Hoist it to a precomputed length-`W` vector. Small but free and safe.
3. **Only if 1 and 2 disappoint:** bakeoff `zc_gram_backend` ∈ `{:blas_syrk, :blas_gemm,
   :threaded_packed}` at the real `nx=1770`. `:threaded_packed` uses Julia threads rather than BLAS
   threads and may interact differently. Caveat: the earlier bakeoff of these backends
   (`docs/HZZ_BACKEND_BAKEOFF_VERDICT_2026-07-29.md`) was **confounded by the shared-`ZcS` bug** and
   its verdict was later retracted (`HZZ_HCZ_SHARED_ZCS_BUG_ROOT_CAUSE_2026-07-29.md`); the bug is
   fixed, but these backends have not been cleanly re-validated at large `nx`. Measure afresh.

### 0c. Mandatory correctness gate before launching

Any config change must leave the science untouched. **At the calibration point, K=3/3, D20/W=100k,
confirm `Delta_dual` agrees between the old and new configuration to solver tolerance** (~1e-10
relative) for *both* families. A BLAS-thread count or gram backend must not change `Delta_dual` at
all beyond floating-point reassociation. If it does, stop and investigate — do not launch.

Also re-run `test_cmzc_cross_wiring_2026-08-09.jl` (fast, solver-free) after any edit to
`cm_checkpoint.jl`, since that file carries the checkpoint schema.

## Phase 1 — the two runs

### Configuration (identical except the family)

```
delta            = 1.0            # the point being evaluated
direction        = upper          # find_smallest = true
K_mean = K_pair  = 3              # production spec
Variant D        = level 2 (= sigma-1)   # originzc_profiled_level / meanzc_profiled_level
W                = 100_000
L                = 50             # CM families only
contrasts        = :orthonormal
probs            = nested_grid_sequence([10,20,50])[50]
include_truncated_moment = true   # CM families only
draw_design      = :sobol_randomized ; draw_seed = 20260719
destination_sample = :exclude_row ; exclude_diagonal_gravity = true
gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea()
sigmaHat         = 3.0 ; inner_lower_limit = -10.0
A_coordinate_mode = :powered_aspace
cm_gradient_backend = :cplus      # driver default
maxtime_real     = 28800.0        # 8 hours
checkpoint_interval_s = 600.0     # so an interrupted run is recoverable
```

- **OZC-CROSS** → `run_originzc_upper_checkpointed`, with
  `distribution_restriction = :origin_specific_moments_zero_covariance`,
  `power_target_layout = :origin_by_power_cross`, `originzc_profiled_level = 2`.
- **CM+ZC-CROSS** → `run_cm_upper_checkpointed`, with `cm_extension = :cm_plus_moments`,
  `meanzc_target_layout = :shared_by_power_cross`, `meanzc_profiled_level = 2`.

**Start point:** the calibration point, in the driver's `:powered_aspace` coordinates, with `eta0`
from the theoretical population mean `Gamma(1 - mu*k)` and the Variant D omitted coordinate dropped
— exactly as `production_smoke_{ozc,cmzc}_cross_2026-08-09.jl` already construct it. Reuse those
scripts' `w0` construction rather than re-deriving it; parameterise the budget instead of writing
new launchers. (A multistart wave would be a better search, but is a different, much larger task —
these are single-start evaluations at one delta.)

**Run them as two separate OS processes**, `-t 10` Julia threads each, per the established campaign
convention. Do not run both inside one process. Separate `ckpt_dir`s.

### Launch discipline (this project has repeatedly lost hours here)

- **Check within 30–60 s of launch** that each process is alive (`ps`) and its log shows real early
  output — not an immediate argument/parse/env error. Do this *before* settling into any long wait.
- Then check in periodically (e.g. hourly): confirm the process is alive, evaluations are
  accumulating, and checkpoints are being written. `tail` the log for `eval N` lines.
- Do not `pkill -f <pattern>` in the same Bash call whose command line contains that pattern — it
  self-kills. Kill by PID in a separate call.
- Expect roughly 4–6 min per outer evaluation at K=3/3 before the Phase 0 fix, less after — so
  order ~100 evaluations in 8 hours. If you see *far* fewer, something is wrong; investigate rather
  than waiting out the budget.

### What to report

For each family: `knitro_status`, `n_eval`, `n_grad`, wall elapsed, and for the incumbent
`best_feasible`: `gp`, `Delta`, the evaluation index it was found at, and
`kappa = 1 - gp^(sigma/(sigma-1))`. Plus the evaluation trace (gp and Delta per eval) so the search
path is visible, and the count of verified vs unverified evaluations.

**Interpretation, stated explicitly in the write-up:** `best_feasible` is only updated at points that
are both feasible and pass `is_verified_success`, so the reported `kappa` is *attained at a genuinely
verified feasible point*. An unconverged search therefore still yields a **valid but conservative**
bound — the true supremum over the identified set is at least this. Say plainly whether each run
terminated on the budget (`-401`) or on an optimality criterion, and do not present a
budget-terminated `kappa` as converged. Confirm the direction convention against
`incumbent_logic.jl`/`direction_bounds.jl` rather than taking the above on trust.

Also record whether the cross inner solves are still stalling at `inner_status = -100` (a
`KN_RC_NEAR_OPT` stall, not an iteration cap) and how often, since that is the known open item and
these runs are the largest sample of it so far.

## Traps (all confirmed live in this codebase)

1. **`nu0` = theoretical population mean `Gamma(1 - mu*k)`**, never a sample average of the same draws
   the restriction is imposed on.
2. **`draw_seed` is INERT under `:pseudorandom`** — use `:sobol_randomized` (as configured above).
3. **`W = 8000` does not work at D20.** Use `W >= 80,000`.
4. **Checkpoint loaders differ**: `run_cm_upper_checkpointed` writes `CMCheckpointV11`
   (`load_cm_checkpoint`); `run_originzc_upper_checkpointed` writes `OriginZCCheckpointV10`
   (`load_cm_checkpoint_v10`). Field is `checkpoint_reason`, not `stop_reason`.
5. **A resume with a mismatched layout is hard-refused by design** — that guard is correct, do not
   work around it. If you resume, pass the identical layout/K/config.
6. **Do not benchmark with a warm re-solve at the same point**: the diagonal family's warm re-solve
   does *zero* iterations and returns in ~2 s, which makes any "steady-state" ratio meaningless.
   Compare cold solves.
7. **`-100` is `KN_RC_NEAR_OPT`** (a stall, "no further progress possible"), not an iteration cap;
   `-400/-401` are the limit codes. It is an accepted/feasible status in this codebase.
8. **Control against the unmodified diagonal family before bug-hunting** anything that looks wrong.
9. Long-running scripts should `flush(stdout)` after progress output, or an early check-in has
   nothing to see.

## Environment

```bash
export PATH="$HOME/.juliaup/bin:$PATH"     # NOT /opt/shared_sw — that Julia is broken here
export OPENBLAS_NUM_THREADS=1              # ambient hard rule; the driver raises it internally
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/13.0.1
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/13.0.1/lib:${LD_LIBRARY_PATH:-}
cd /bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09
julia --project=. -t 10 full_aod_diag/d4_exact/<script>.jl
```

## Deliverable

A `docs/*.md` write-up with: the Phase 0 outcome (what changed, the measured speedup, and the
`Delta_dual`-unchanged evidence); the two runs' results in the form above; and an explicit statement
of convergence status. Update memory `ozc-cross-kpair2-grid-build-2026-08-09`, and push a Dropbox
package to a **new** subfolder under `dropbox:Gravity robustness/Analysis/Server Output/` (written
deliverable, `provenance.txt` with the commit SHA, and a compact `key_results/` — do not push raw
KNITRO logs or whole `results/` trees).
