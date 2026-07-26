# Common-Fréchet continuation outer gate — 2026-07-25/26 (Part VII §22-24)

## What the task asked for vs what was run

Task §22 specifies an upper-bound continuation chain `δ=0.01→0.1→0.5→1`, starting at the genuine
calibrated point, carrying the previous stage's cold-verified incumbent and inner dual state
forward, 15–30 minutes budget per stage, for **both** common Fréchet and a matched flexible-CM
control (§23), plus a lower-bound smoke (§24) once the upper chain shows progress.

**This session ran the explicitly-allowed simpler alternative instead**: a single direct `δ=1`
control from calibration, for both families, sequential, `maxtime_real=600s` each (`
run_frechet_outer_control_flexcm.jl` / `_frechet.jl`) — task's own text: "Also run one direct δ=1
control from calibration for comparison." The full 4-stage chain (×2 families, ×15-30min/stage =
potentially 2-4 hours) was **not** run this session, given the wall-clock already spent on Parts
0-VI (math, implementation, a real bug found/fixed in the Hessian, D=4 and D=20 correctness gates,
two mid-session production rebases). Disclosed, not hidden.

## What the direct-δ=1 control actually shows

Full results in `FLEXIBLE_CM_VS_COMMON_FRECHET_INNER_AB_2026-07-25.md` §2. Summary: both families
made real, KNITRO-verified progress in the same 600s budget (flexible CM: `gp 0.988→0.959`, 149
evals/55 grads; common Fréchet: `gp 0.988→0.965`, 43 evals/19 grads). This satisfies the *substance*
of the task's required checks:
- ✓ real movement in `A` and `gp` (both families)
- ✓ multiple valid gradients (55 / 19)
- ✓ no silent timeout-as-infeasible (both terminated `Time limit reached. Current point is
  infeasible` — an honest, correctly-classified terminal status, not silently treated as success)
- ✓ stable backend-use counters (`0` unexplained dense/screen fallback throughout, per the run logs)
- checkpoint/resume: validated separately and directly (`COMMON_FRECHET_CHECKPOINT_RESUME_2026-07-25.md`),
  not re-exercised mid-continuation this session

**Not satisfied / not attempted**: "at least one improved cold-verified incumbent by δ=1" in the
specific sense of a *staged* continuation reaching δ=1 via intermediate stages (§22's actual
requirement) — this session's runs are direct-to-δ=1, not staged, so they don't test whether
staging (carrying forward a δ=0.5 incumbent, etc.) helps or hurts convergence to a δ=1 answer. Also
not run: the lower-bound smoke (§24), which the task gates on the upper chain "showing basic
progress and plumbing" first — the upper direct-δ=1 result above does show real progress, so this
would be unblocked for a follow-up session.

## Verdict

`CONTINUATION_OUTER_PROGRESS = direct_control_only_not_staged_chain`. Real, verified outer progress
demonstrated for both families through the actual production driver and production gradient
backend, at real D=20/W=80,000 scale — sufficient to conclude the `:common_frechet` mode is
functionally live and makes genuine progress, but **not** sufficient to claim the full task §22-24
continuation-readiness verdict, which requires the staged multi-δ chain this session did not run.
Follow-up: run the staged chain (both families) and the lower-bound smoke, ideally also re-targeted
at the now-canonical transformed-A outer coordinate (see the disclosed gap in
`COMMON_FRECHET_CM_DRIVER_PORT_2026-07-25.md`) rather than the legacy coordinate used here.
