# Full-A D=4: Continuation 3 resume audit

Phase 0 deliverable for the "hybrid solver / finish D=4 / gate scaling" continuation.

## 1. Repository state (verified, not assumed)

- Repo: `git@github.com:habibiscoding/Trade-Model-Robustness.git`.
- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`.
- Branch: `diag/fullA-d4-exact`.
- Local HEAD: `597e1617fb3d9e8bf4f7135ca2f1ddaf2520016b` — matches the task brief's stated
  "latest archived diagnostic HEAD" exactly.
- `origin/diag/fullA-d4-exact`: also `597e1617f...` — local is neither ahead nor behind
  (`git rev-list --left-right --count` = `0 0`). No drift.
- `git status --short`: clean (no uncommitted changes) at audit start.
- Production branch-off point cited in the codebase's own `environment.txt`/code-audit doc:
  `53ffb58e8d9b18498279fab25da4d1b7cc47556a` on `sequential-profiled-gravity` — matches task brief.
- `docs/00_READ_FIRST_CORRECTION.md` does **not exist** in this worktree (checked directly, not
  assumed) — the task's reading order item 1 is a no-op here; items 2-9 all exist and were read in
  full before any Phase 1+ work started.

## 2. Toolchain / environment

- Julia: `1.12.6` (via `~/.juliaup/bin/julia`; not on `PATH` by default — must
  `source .knitro_env.sh` first, which also puts `~/.juliaup/bin` on `PATH`).
- KNITRO: **runtime-linked library is `/opt/shared_sw/knitro/13.0.1/lib/libknitro.so`**
  (confirmed via `julia --project=. -e 'using KNITRO; println(KNITRO.libknitro)'`), matching the
  archived `environment.txt`'s original capture. **Drift noted, not fixed**: `.knitro_env.sh` in
  this worktree sets `KNITRODIR=/opt/shared_sw/knitro/14.2.0`, but that env var is evidently not
  what `KNITRO.jl` v1.2.1 actually uses to locate the shared library at this Manifest-pinned
  version (no `LocalPreferences.toml` present) — the 14.2.0 setting in the env script appears to be
  vestigial/ineffective, not a live configuration hazard, since both mandatory smoke tests below
  passed cleanly using the 13.0.1 library that's actually loaded. Left alone: changing
  `.knitro_env.sh` risks touching the one part of the environment (KNITRO licensing, working only on
  `demand.mit.edu` per memory `reference-knitro-license-demand`) that is out of scope for this
  investigation and currently works.
- Host: `demand.mit.edu` (license-compatible, confirmed).
- CPU: 208 cores (`nproc`).
- Julia threads: `Threads.nthreads() = 1` (default, unset `JULIA_NUM_THREADS`).
- BLAS threads: 104 (OpenBLAS default, i.e. `nproc/2`).
- Key package versions (`Pkg.status`): JuMP v1.30.1, HiGHS v1.24.1, ForwardDiff v1.4.1,
  KNITRO v1.2.1, Enzyme v0.13.182, Mooncake v0.5.37 — HiGHS/JuMP present as the task brief's Phase F
  primal-feasibility LP already added them in a prior continuation; no new dependency needed here.

## 3. Mandatory smoke tests (run before any new work, per task instruction)

Both run via `source .knitro_env.sh && julia --project=. full_aod_diag/d4_exact/<test>.jl`:

- `test_oracle.jl`: **ALL ORACLE TESTS PASSED** (single-eval sanity, bit-identical determinism at
  fixed x, cache hit/miss correctness, warm-vs-cold value agreement to `<1e-8`). No drift.
- `test_oracle_profiled.jl`: **ALL EQUIVALENCE CHECKS PASSED** (`evaluate_fullA_profiled` vs
  `evaluate_fullA`, warm/cold/cache-hit paths all field-for-field identical). No drift.

Conclusion: proceed with Phase 1+ as planned; no drift diagnosis/handoff-update detour needed.

## 4. Canonical candidate registry

Loaded from structured artifacts (not manually transcribed) by
`full_aod_diag/d4_exact/candidate_registry.jl` (new this phase), which re-evaluates each point
through the exact `evaluate_fullA` oracle and prints a fresh number next to the archived one.
**Verified by running the script** (not just reading artifacts) — output below.

| label | source artifact | γ'_focal | κ (fresh, matches archived to printed precision) | Δ | Δ−δ | classification |
|---|---|---|---|---|---|---|
| `calibration` | `test_oracle.jl` θ_initial (in-code), θ[3+D]=θ[7] | 0.9609650007 | 0.0642080946 | n/a — cold inner solve fails (`-300`) at z_free=0 without a warm-started continuation path; not claimed feasible by any prior artifact either | — | base point, not a candidate |
| `fixed_A_benchmark` | `results/fullA_d4/a377fff/movement_and_fixedA_check.txt` | 0.9109408706 | 0.1439805232 | n/a — same cold-solve caveat (this benchmark's own κ was produced via a warm-started continuation script, not a cold jump; reproducing its `Δ` needs the same continuation path, not attempted this phase, not needed since κ is a closed-form function of γ'_focal alone here) | — | trivial-A-fixed sanity floor |
| `upper_maxit15_productfd_control` | `results/fullA_d4/9e03706/optfd_upper_20260717_182444/summary.txt` (best_feasible_tracked) | 0.8938496736 | 0.1705806879 | 0.9989309115 | −0.0010690885 | Phase B control point, iteration-matched only |
| `upper_maxit40` (headline) | `results/fullA_d4/9e03706/optfd_upper_20260717_190946/summary.txt` (best_feasible_tracked) | 0.893083918 | 0.1717646139 | 0.9987689551 | −0.0012310449 | `EXACT_FEASIBLE_CANDIDATE`, `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)`, NOT `ROBUST_LOCAL_CANDIDATE` |
| `upper_poll1` (dir 18) | `results/fullA_d4/1b2a3a0/phaseA_upper_revalidation/step6_poll.csv` row (radius=0.001,dir_idx=18,sign=+1) | 0.893064593589446 | (γ'_focal↓ ⇒ κ↑ vs maxit40, ~+9e-6) | 0.9998092122907517 | — | poll-improved, genuine but tiny; **the full 16-dim `w` is not archived** (the poll used 36 random probe directions, not coordinate axes — regenerating it exactly requires re-running `phaseA_upper_revalidation.jl`'s RNG with the same seed, not done this phase; only γ'_focal/Δ are directly comparable) |
| `upper_poll2` (dir 20) | same file, dir_idx=20 | 0.8930651554888783 | ″ | 0.9998056125908148 | — | poll-improved, same w-reconstruction caveat |
| `upper_poll3` (dir 35) | same file, dir_idx=35 | 0.8930699916553521 | ″ | 0.9998512744233814 | — | poll-improved, same w-reconstruction caveat |
| `lower_stalled` | `results/fullA_d4/9e03706/optfd_lower_20260717_190831/summary.txt` (maxit=15, only lower run that reached `outer_iters=15` rather than stalling at `nStatus=-502`) | 0.9935715171 | 0.0106911632 | 0.8986353002 | −0.1013646998 | `BEST_FEASIBLE_STALLED` |
| `sequential_reconstructed` | not yet available — Phase 5 of this continuation | — | — | — | — | pending |

`x_free_hash` (a `hash(round.(x_free, digits=12))` value) is printed by the script for every
candidate that has a reconstructable full `w`, for exact-reproducibility cross-referencing in later
phases — not reproduced in this table for brevity, see script output.

## 5. Candidate loader script

`full_aod_diag/d4_exact/candidate_registry.jl` defines the table above as data (paths + expected
values) and, when run, re-loads each artifact from disk, re-evaluates it through the exact
`evaluate_fullA` oracle at D=4/W=8000/δ=1, and prints:

- the artifact's own reported γ'_focal/κ/Δ,
- a fresh cold `evaluate_fullA` recheck at the same point (γ'_focal, κ, Δ, gravity, inner_status),
- a hash of the full `x_free` vector (`hash(round.(x_free, digits=12))`) for exact-reproducibility
  cross-referencing across later phases,
- PASS/FAIL if the fresh recheck's κ disagrees with the artifact's own reported κ by more than
  `1e-8`.

This script is the canonical way later phases should load a starting point — never by re-typing a
`w` vector into a new script by hand.

## 6. Stale-claim sweep (retracted sequential number)

Grepped the full `docs/` tree and `full_aod_diag/` for `0.0779`, `beats sequential`, and `>2x`:

- `docs/fullA_next_handoff.md` §2a already contains the correction (written by the prior
  continuation after user review) — retained as history, correctly labeled as a correction, not
  live-stated as fact anywhere else in that file.
- `docs/fullA_d4_final_report.md` §9.3 and `docs/fullA_d4_recommendation.md` (both the "Update from
  continuation 2" section and the "RETRACTED" bullet) already carry the correction with the same
  framing.
- No other file in `docs/` or `full_aod_diag/` references `0.0779` or the ">2x" claim as a live,
  uncorrected statement. **No further edits needed this phase** — the two-continuations-ago retraction
  was already fully propagated; confirmed by direct grep, not assumed from memory alone.

## 7. What this phase did NOT do

- Did not re-verify KNITRO option fallback behavior fresh (relying on the code-audit doc's existing,
  directly-quoted evidence — `eval_fcga=yes` + `hessopt=4` → silent L-BFGS fallback — which is a
  static fact about KNITRO's option parser, not something that could have drifted since the last
  session touched the same `.opt` files, all still present and unmodified).
- Did not attempt to reconcile the `.knitro_env.sh` KNITRODIR=14.2.0 vs runtime-linked 13.0.1
  discrepancy noted in §2 beyond documenting it — out of scope, not a blocker (smoke tests pass).

## 8. Next command

```
cd /bbkinghome/edav/gravity_robustness/gravity-fullA-d4
source .knitro_env.sh
julia --project=. full_aod_diag/d4_exact/candidate_registry.jl   # confirms table above from fresh artifact reads
```

Then proceed to Phase 1 (`oracle_fast.jl` — reuse `inner_loop_internal`'s own `obj.H` moments output,
replace `sort()`-based winner computation, reduce KKT-residual allocations — all additive,
equivalence-tested against `oracle.jl`/`oracle_profiled.jl` before being trusted for timing).
