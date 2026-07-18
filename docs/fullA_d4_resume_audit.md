# D=4 exact full-A investigation: resume audit

Written at the start of this continuation session, before any new substantive work. Records the
verified state of the diagnostic worktree against the handoff prompt's expectations, and the
concrete discrepancies found while re-checking primary artifacts rather than trusting prose.

## 1. Repository / worktree state (verified, not assumed)

- Worktree: `/bbkinghome/edav/gravity_robustness/gravity-fullA-d4`, registered in the shared
  `git worktree list` alongside six other worktrees off the same repo — confirmed via
  `git worktree list` from inside this worktree.
- Branch: `diag/fullA-d4-exact` (`git branch --show-current`).
- HEAD: `f2fceb4d609c8f8e4e7a2b2925863492a1773af3` — matches the handoff prompt's expected HEAD
  exactly (not "a later continuation commit").
- Branch point: `git merge-base diag/fullA-d4-exact 53ffb58` returns `53ffb58e8d9...` itself,
  confirming the branch is a clean fast-forward descendant of production commit `53ffb58`, no
  divergence.
- `git status`: clean, nothing uncommitted.
- Production worktree `trade_robustness_modular` (branch `sequential-profiled-gravity`, also at
  `53ffb58`) was not touched.

## 2. Handoff document discrepancy

**No `HANDOFF.md` exists** anywhere in the worktree or in a separate handoff bundle (`find . -iname
'*handoff*'` returns nothing). This differs from the prompt's framing ("read `HANDOFF.md` if
present"). The two other named documents exist and were read in full instead:

- `docs/fullA_d4_code_audit.md` (code-level audit, 206 lines)
- `docs/fullA_d4_final_report.md` (interim findings report, 171 lines)
- `docs/reference/sequential_methodology.tex` (578 lines; read in place of the PDF — same content,
  easier to parse as text; `pdfinfo` confirms the PDF is 11 pages, consistent with the `.tex`
  source)

No other handoff artifact was found or needed — the two docs above are internally consistent with
each other and with the raw result files checked in §4 below, so they serve the same purpose a
`HANDOFF.md` would have.

## 3. Smoke test

`julia --project=. full_aod_diag/d4_exact/test_oracle.jl` — **all 4 sub-tests PASS**:
single-evaluation sanity, determinism (bit-identical repeated calls), cache hit/miss correctness,
warm-vs-cold agreement (diff 2.1e-17). Julia version reported by the run: `1.12.6`, matching
`environment.txt`. Note: `julia` is not on `PATH` by default in a fresh shell on this host: the
juliaup shims live at `~/.juliaup/bin`, which is added to `PATH` by the interactive shell profile
but not inherited by non-interactive tool invocations here. Every command in this investigation
explicitly prepends `export PATH="$HOME/.juliaup/bin:$PATH"`.

## 4. Environment / job / license check

- No unrelated Julia or KNITRO processes were running at the start of this session (`ps -eo
  pid,ppid,user,etimes,cmd | grep -iE 'julia|knitro'` returns only long-lived, harmless `juliaup
  self update` background helpers — not actual solver jobs). Nothing was killed.
- Host: `demand.mit.edu`, confirmed via `hostname` — this is the KNITRO-licensed host per
  memory `reference-knitro-license-demand` (KNITRO does **not** license on `supply`).
- `environment.txt` at the worktree root is **stale**: its `git commit` field reads `53ffb58`
  (the branch-off point), i.e. it was captured at branch creation, before any of the 17 diagnostic
  commits in `git log` were made. It is otherwise accurate (Julia/KNITRO versions, host, CPU count
  unchanged). Regenerating it is folded into the documentation-correction pass (§6 below).

## 5. Backup

Repository policy permits pushing (`origin` is `git@github.com:habibiscoding/Trade-Model-Robustness.git`,
reachable — `git ls-remote --heads origin diag/fullA-d4-exact` succeeded before the push, returning
empty, i.e. the branch did not yet exist on origin). Ran:

```
git push -u origin diag/fullA-d4-exact
```

Result: `* [new branch] diag/fullA-d4-exact -> diag/fullA-d4-exact`, tracking set up. No bundle was
needed since the push succeeded.

## 6. Discrepancies found between the handoff prompt's claims and the primary artifacts

These are the concrete, artifact-level confirmations of the five corrections the continuation
prompt flagged. Each was independently re-derived from the actual CSV/log/summary file, not taken
on the prompt's word.

### 6.1 `hessopt=4` mischaracterization (prompt correction #1)

Confirmed: `docs/fullA_d4_final_report.md` §6 literally writes `` `hessopt=4` (BFGS) ``. This is
wrong under KNITRO 13.0.1's own option mapping (`hessopt=2`=BFGS, `hessopt=3`=SR1, `hessopt=4`=
product finite-difference Hessian-vector, `hessopt=6`=L-BFGS). The report's own quoted KNITRO log
line — `"Option hessopt=4 not valid when eval_fcga=1. Changing hessopt to 6 (LBFGS)."` — already
contains the evidence that `hessopt=4` is a distinct mode from BFGS (KNITRO would not need to
"change" `hessopt=4` to `6` if `4` already meant BFGS); the report's prose just mislabeled it. Every
one of this session's short D=4 runs (`run_d4_optimized_fd.jl`, all `optfd_upper_*` /
`optfd_lower_*` results) used the opt files under `full_aod_diag/d4_exact/csw_outer_*.opt`, which
were checked directly (§ Phase B below) rather than assumed.

### 6.2 Stationarity-check point mismatch (prompt correction #2)

Confirmed by direct comparison of `results/fullA_d4/9e03706/stationarity_check_upper.txt` against
the two candidate summary files:

| | `stationarity_check_upper.txt` (archived) | `optfd_upper_20260717_182444/summary.txt` (maxit=15 best-feasible) | `optfd_upper_20260717_190946/summary.txt` (maxit=40 best-feasible) |
|---|---|---|---|
| `w[1]` = `gamma_focal_prime` | 0.8938496736355915 | 0.8938496736355915 | 0.8930839180420251 |
| kappa | (not stored directly; implied) | 0.17058068794444559 | 0.17176461388430053 |
| full `w` vector | matches maxit=15 exactly, entry-for-entry | — | differs from both other columns |

The archived stationarity check's `w` vector is a byte-for-byte match to the **maxit=15**
best-feasible point, not the maxit=40 point. `optfd_upper_20260717_190946/summary.txt` itself says
so explicitly: `status_label=BEST_FEASIBLE_STALLED_OR_VERIFIED (external KKT stationarity NOT
checked yet -- sec 22 not done this run)`. This is not an inference — it is what the artifact
says. A fresh external stationarity check on the maxit=40 point is required before any
classification of it; this is the first item of Phase A below.

### 6.3 "Verified stationary" overclaim (prompt correction #3)

`docs/fullA_d4_final_report.md` labels the maxit=15 point `VERIFIED_STATIONARY_FEASIBLE_CANDIDATE`.
Per the taxonomy the continuation prompt specifies, this conflates two distinct claims: (a) exact
hard feasibility (genuinely verified — cold recheck confirms gravity ~1e-18, KKT residual ~1e-16),
and (b) stationarity under a **single fixed-bandwidth (h=0.01) central-FD gradient** — which is a
`H_BANDWIDTH_KKT_CANDIDATE` claim, not an unqualified `VERIFIED_STATIONARY` claim, until multi-h
and directional robustness checks are run. No such multi-h check exists yet anywhere in the repo
for either upper candidate (`h_sweep.jl`/`test_h_sweep.jl` cover the three-way frozen-adjoint/
fixed-dual/optimized-value **distinction**, at h=0.2..0.00625, but that is a different h-grid built
around the calibration point for a different diagnostic purpose — not a stationarity-check h-grid
at either upper candidate). Relabeling and the actual multi-h/poll work is Phase A below.

### 6.4 `smoothing_check.csv` NaN vs. report's "successful jump near 1.5e-11" claim (prompt
correction #5)

Confirmed directly: `results/fullA_d4/9e03706/smoothing_check.csv` contains

```
quantity,value
hard_jump_at_threshold,8.274393141894211e-6
smoothed_tuner100_jump_at_threshold,NaN
```

The report's §3 claims "the same 1e-9-straddle jump shrinks from 8.27e-6 (hard) to 1.52e-11
(smoothed, tuner=-100)". The `hard_jump_at_threshold` value matches the report's `8.27e-6` exactly
— but the smoothed entry is `NaN` in the archived artifact, not `1.52e-11`. This is a real
inconsistency between the checked-in CSV and the report's prose, not a transcription rounding
issue. Re-running `smoothed_moments.jl`/`winner_switching.jl`'s smoothing check and regenerating one
canonical, reproducible artifact is required before the claim can be trusted either way; folded
into Phase A/D below (the smoothing check shares machinery with the derivative-method benchmark).

### 6.5 `-300` inner status "infeasibility" claims (prompt correction #4)

`check_multistart_feasibility.jl` / `check_multistart_warmstart_rescue.jl` (commits `b90e062`,
`f2fceb4`) report multistart failures as consistent with genuine primal infeasibility based on
KNITRO's own `-300` status persisting across warm-start rescue attempts. Per the prompt, this is
suggestive, not a certificate — no independent primal-feasibility LP exists yet in this repo for
this problem. This is Phase F below, not yet started.

## 7. What this audit does *not* re-litigate

Findings from `docs/fullA_d4_code_audit.md` that were re-derived from code (not just handoff prose)
in the prior session — the `n_free=17` count, the `γ_d≡1`-for-every-destination normalization, the
gravity-elimination machinery, the winner-boundary AD bug's presence — are treated as established
per that audit's own methodology (direct code inspection, cited line numbers) and are not
re-verified from scratch here, consistent with the continuation prompt's framing of them as "strong
but still reproducible findings." They remain reproducible on demand via the existing test files
(`test_free_param_and_gravity.jl`, `test_gravity_elimination.jl`, `test_winner_switching.jl`), which
were not re-run in this audit pass but are unchanged since the prior session (no commits touch them
between branch-off and `f2fceb4`).

## 8. Immediate next steps from this audit

1. Fix the five documentation corrections above in `docs/fullA_d4_code_audit.md` and
   `docs/fullA_d4_final_report.md` (or supersede the latter with a corrected non-interim report, per
   the continuation prompt's Phase/logging requirements).
2. Regenerate `environment.txt` at current HEAD.
3. Proceed to Phase A: fresh stationarity check + h-grid + directional/poll checks on the maxit=40
   point specifically (not re-using the maxit=15 numbers).
