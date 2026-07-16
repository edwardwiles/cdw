# Handoff: D=20 real-data overnight run, sequential-linearized method only

Paste this whole file to a fresh Claude Code session as the task prompt.

## Where to work

`cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf` — branch
`feature/sequential-inversion-perf`. Check `git log --oneline -6` first; you should see (most
recent first):

```
987807a Make verify_batch_solutions.jl D/W/fakeData env-overridable; add maxit=200 KNITRO options
778c362 Fix autarky counterfactual bug (tau^Inf=1 for off-diagonal tau==1) and add real D=20 data support
2abaacb Add session summary for UoModel removal, universal gamma=1, FWL gravity simplification, mu-value caching
0951782 Add real in-run timing instrumentation, GRAVITY_SEED/WVAL flags to sequential driver; run full D=10 method comparison
9a4793f Cache U^(-mu) on mu's value, not just its type
ea35b01 Remove UoModel toggle and baseline gamma; apply verified FWL gravity-moment simplification
```

Do NOT touch `sequential-profiled-gravity`/`trade_robustness_modular` or any other worktree —
other work happens there asynchronously. Everything for this task lives in
`trade_robustness_modular_perf` on `feature/sequential-inversion-perf`.

## Environment (must run on demand.mit.edu — KNITRO license is machine-locked there)

```bash
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
export PATH="$HOME/.juliaup/bin:$PATH"
```

## Context: this is real data, freshly wired up and debugged this session

The user provided a real D=20-country dataset (`real_data/noah_D20/{pi,tau,L,countries}.csv`,
sourced from `Dropbox:Gravity robustness/Noah/prepare_clean_data/objects_for_julia`, France as
focal country, matching Noah's own gravity-regression analysis). Getting this to run correctly
took real debugging this session — read this section before assuming the pipeline "just works,"
and don't re-derive any of the following from scratch:

1. **A real, now-fixed bug**: `setup/defineCounter.jl`'s autarky counterfactual was built as
   `tauData .^ Inf`. Mathematically `1.0^Inf == 1.0`, not `Inf` — so any off-diagonal country pair
   with `tauData==1` exactly (a real occurrence here: a fully-connected EU-core clique — France,
   Germany, Spain, UK, Italy, Netherlands — has genuinely near-zero measured trade costs) silently
   stayed fully tradable even under "autarky." France (the focal country) sits in that clique, so
   its own autarky counterfactual was wrong, corrupting `gammaPrimeHat[focal]` enough to push the
   point-estimate `gamma'_focal` **above its theoretical ceiling of 1.0** — a real, internally
   inconsistent violation (not numerical noise): KNITRO's own outer-loop starting point was outside
   its own box constraints (`γp_hi=1.0`), causing status -514 and zero successful callbacks. Fixed
   by constructing `tauPrime` directly (`Inf` everywhere off-diagonal, own-cost preserved) instead
   of via exponentiation. **Verified**: the point estimate's κ now matches the closed-form
   Arkolakis-Costinot-Rodriguez-Clare check `1 - λ_dd^μ` (λ_dd = France's own-trade share, μ = the
   Frechet dispersion parameter estimated internally via the two-way-FE gravity regression in
   `master_prestep.jl`) to 6 significant figures: **κ_point_estimate = 0.020314**. This is
   confirmed clean data now — do not need to re-investigate this.
2. **A tolerance, not a bug**: destination-inversion's convergence tolerance was hardcoded at
   `1e-8`, tuned against synthetic data. On this real data, every failing destination was landing
   at `share_err` between 3.7e-8 and 7.1e-8 — a genuinely excellent match (1 part in ~15-30
   million), just narrowly missing the old threshold. Confirmed this is a real numerical plateau,
   not slow convergence (raising the Newton `maxit` from 150 to 500 produced bit-identical
   `share_err` — the LM-damping trial loop genuinely can't find a further improving step). User
   explicitly signed off on loosening this ("nobody cares about trade shares beyond 2dp"). Now
   `DEST_INV_TOL` (env override, default `1e-6`) in `sequential_gravity/run_profiled_production.jl`.
3. `setup/importData.jl` has a new `fakeData==3` branch that loads this real dataset from
   `real_data/noah_D20/` (override with `REAL_DATA_DIR` env if needed), and rescales `L` by 1e6
   purely for wage-calibration numerical conditioning — confirmed an exact no-op on every result
   (traced through `iterWagesPreStep!`/`computeGamma`: every use of L-dependent quantities
   downstream is ratio-based, so a uniform rescaling cancels exactly).
4. `sequential_gravity/run_profiled_production.jl` now has a `FAKEDATA` env override (default `1`
   = synthetic, unchanged for all prior uses; `3` = this real dataset) alongside the existing
   `DVAL`/`WVAL`/`DELTA_GRID`/`BOUND`/`OUTER_OPT_FILE`/`OUT_DIR`/`PARALLEL_INVERSION` conventions.
5. A full smoke test (D=20, `FAKEDATA=3`, `maxit=2`, both bounds) ran cleanly end-to-end this
   session: **zero non-converged destination inversions (76/76 each bound)**, both bounds
   `gravity-feasible=true`. The cold cross-verification script
   (`sequential_gravity/verify_batch_solutions.jl`, now also `DVAL`/`WVAL`/`FAKEDATA`-overridable)
   was also run against those smoke-test checkpoints and passed cleanly (max non-focal share error
   7.06e-08, all verdicts `true`). The pipeline is confirmed working — this task is "run it for
   real," not "get it working."

**σHat=2.5 is an assumed (not estimated) parameter, explicitly not in question for this task** —
user confirmed it's irrelevant to the γ'>1 investigation and shouldn't be revisited.
**baseIndex=2 (France) is the focal country** — matches Noah's own gravity-regression analysis
(`gravity.do`'s "France Only" specifications), already the default, don't change it.

## What's genuinely new work for this task

Essentially none on the code side — this is a straightforward "run the existing, now-validated
driver at real settings" task, NOT a build task. The only things to actually decide/do:

1. Confirm the fixes above are still in place (`git log` should show `778c362`/`987807a`; if not,
   STOP and ask the user rather than re-deriving the fixes from scratch — the full diagnostic
   trail is in this session's conversation history if you need to understand WHY they're needed).
2. Run the actual overnight batch (below).
3. Run the cold cross-verification afterward (same methodology as the D=10 comparison — see
   `D10_METHOD_COMPARISON_REPORT.md` §2 for what this checks and why it matters).
4. Write up the results.

## The actual run

**Sequential-linearized method only** (user explicitly does not want full-A or gravity-seeded for
this real-data run — just the production method). D=20, δ=0.1/1.0/10.0, both bounds, W=8000,
**maxit=200** (`full_aod_diag/csw_outer_200.opt`, already created — a byte-copy of `csw_outer_25.opt`
with only `maxit` changed), **19 threads** (`PARALLEL_INVERSION=true`, `julia -t 19` — D-1=19 for
D=20, so this is genuinely full destination-level parallelism, not partial).

```bash
FAKEDATA=3 DVAL=20 DELTA_GRID=0.1,1.0,10.0 BOUND=both PARALLEL_INVERSION=true \
  OUTER_OPT_FILE=full_aod_diag/csw_outer_200.opt OUT_DIR=sequential_gravity/batch_out_realD20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl
```

**Critical gotcha, learned the hard way in the D=10 comparison task — MUST use a fresh `OUT_DIR`**:
the resumable-checkpoint logic only checks for a `done=true` key in the saved JLD2 file: it does
NOT check that the checkpoint's D, W, or dataset match the current run. `sequential_gravity/batch_out/`
already has OLD D=10 synthetic-data checkpoints (including files literally named
`seq_lower_delta1.0.jld2`, `seq_upper_delta0.1.jld2`, etc. — the exact same filenames this D=20 run
would produce). If you don't set a distinct `OUT_DIR`, the script will silently treat the old D=10
results as "already done" for matching deltas and skip solving entirely. The command above already
uses `OUT_DIR=sequential_gravity/batch_out_realD20` — keep it that way, and don't reuse it for
anything else.

### Timing expectations

This is a bigger job than the D=10 comparison in every dimension: D=20 (not 10), maxit=200 (not
25, an 8x iteration budget), 3 deltas × 2 bounds = 6 solves. Per the D=10 comparison's own data,
a single D=10/maxit=25/one-bound/one-delta sequential solve took roughly 5-16 minutes depending on
delta; scaling isn't strictly linear (destination-inversion cost scales with the free-parameter
count differently than the outer iteration count), but budget for this taking **several hours at
minimum, plausibly much longer given the 8x maxit increase** — this is explicitly an overnight run,
not a quick check. Use the checkpoint/resume infrastructure (already built, validated) — it is safe
to launch in the background and let it run across a disconnect. On restart with the same `OUT_DIR`,
already-`done` (bound,delta) pairs are skipped automatically.

- Use the Bash tool's `run_in_background` for the launch (a single call, do NOT also append your
  own trailing `&` — double-backgrounding detaches the process from the harness's own tracking).
- Before committing to the full run, consider a quick smoke test first (e.g.
  `OUTER_OPT_FILE=full_aod_diag/csw_outer_smoke2.opt DELTA_GRID=1.0 BOUND=upper` with a throwaway
  `OUT_DIR`) just to confirm the environment/license/threading all work exactly as expected on this
  invocation before launching the real multi-hour job — this session's own smoke tests already
  validate the underlying pipeline, so this is just a final sanity check on your specific launch
  command, not re-validating the science.
- KNITRO's license is machine-locked to demand.mit.edu but floating (multiple concurrent processes
  can each acquire it — validated up to 4-way concurrency in the D=10 comparison task) — not a
  concern here since this is a single job, mentioned only in case you end up running the smoke test
  and the real run overlapping briefly.

## After the run: cold cross-verification

Exactly the same methodology as the D=10 comparison (see `D10_METHOD_COMPARISON_REPORT.md` §2 for
full rationale — production's own "gravity-feasible" bookkeeping can be warm-started/optimistic;
this check re-solves everything from scratch with no warm start):

```bash
DVAL=20 FAKEDATA=3 WVAL=8000 VERIFY_BATCH_DIR=sequential_gravity/batch_out_realD20 \
  julia --project=. sequential_gravity/verify_batch_solutions.jl
```

This reports, per (bound,delta), for both the KNITRO-own and best-feasible theta: divergence budget
check, focal moment residuals, **max non-focal share error across all 19 omitted destinations**
(the key number — should be small, comparable to the smoke test's 7e-8, not large), and the exact
gravity residual. Report these numbers alongside the raw batch results, not instead of them.

## Final deliverable

A short written report covering, for each δ=0.1/1.0/10.0 × {lower,upper}:
- κ (and γ'_focal) for both the KNITRO-own terminal value AND the best-feasible-tracker value,
  clearly labeled (don't collapse to one number — see `D10_METHOD_COMPARISON_REPORT.md` for why
  these can differ and both matter).
- `gravity_ok`/δ-feasibility flags for each, from both production's own bookkeeping and the cold
  cross-verification.
- The cold cross-verification's max non-focal share error for each point — flag anything that
  doesn't look like a clean match (comparable to ~1e-7 or better), don't just report "converged."
- opt_err/feas_err or KNITRO status for each solve, so solve quality is visible alongside the κ
  numbers (per the D=10 comparison's finding that not-fully-converged points can look superficially
  fine but shouldn't be trusted as validated bounds).
- How κ_point_estimate=0.020314 (already validated, closed-form-consistent) compares to the
  computed bounds — do they bracket it sensibly, does the interval widen with δ as expected?

Save the write-up in the repo (e.g. `D20_REALDATA_REPORT.md`, mirroring the D=10 report's
structure) and push it to the user's Dropbox via `rclone copy <file>
"dropbox:Gravity robustness/Analysis/Server Output/"` (the `dropbox:` remote is already configured
on this server) once done, matching how the D=10 report was delivered.
