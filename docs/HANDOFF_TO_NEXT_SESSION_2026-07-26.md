# Handoff to next session — matrix-free operator FG, winner-aware Hessian, interval basis — 2026-07-26

This file **is** the prompt to hand to the next Claude Code session. Paste everything below the
`---` line as that session's opening message.

---

Continue the five-family production optimization stack task. The **full original task prompt**,
verbatim, is preserved at
`gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26/docs/ORIGINAL_TASK_PROMPT_2026-07-26.md`
— read it in full before doing anything else. It contains the exact mathematical specification
this handoff summarizes (operator partitions, the `Q'SR - π(ν'SR)` cross-Hessian formula, the
interval-basis derivation requirements, the required runtime counters, and the final-verdict
format) — do not rely on this handoff's paraphrase for those details, go to the original.

## What already happened (read before touching code)

A prior session completed Phases 0, 1, 2, 3, 4 (audit-only), 8, 9, and part of 10 of that original
prompt, on branch `port/finish-five-family-optimization-stack-2026-07-26`
(HEAD `08550a8`, base `production/fullA-exact @ f1fa8e770759c62b3f96c1024dd310f235ea463e`,
worktree at `gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26`).
**Nothing from that branch has been pushed to origin or merged to production** — it is still just
a local feature branch.

Read, in this order, before writing any code:

1. `docs/FIVE_FAMILY_OPTIMIZATION_COMPLETION_MASTER_REPORT_2026-07-26.md` — what was verified, the
   final verdict block, and exactly what's left (§12 = `HIGHEST_PRIORITY_REMAINING_GAP`).
2. `docs/NO_FULL_G_MATERIALIZATION_AUDIT_2026-07-26.md` — the **exact current state** of dense-G
   usage per family (flexible CM / common Fréchet: chunked-dense scratch fill via
   `fill_cm_columns_from_bins!`; CM+ZC: fully dense CM columns, kept by an explicit prior
   benchmark decision; origin-ZC: small fixed-size raw features, lower priority). This is your
   starting map for the work below — it names real file/function locations, not just concepts.
3. `docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md`,
   `docs/WINNER_AWARE_CROSS_HESSIAN_BENCHMARK_2026-07-26.md`,
   `docs/INTERVAL_VS_CUMULATIVE_CM_BASIS_2026-07-26.md`,
   `docs/INTERVAL_BASIS_HESSIAN_OPTIMALITY_AUDIT_2026-07-26.md` — the prior session's own scoped
   follow-on plan for exactly the three items below. Treat these as a starting outline, not gospel
   — verify claims (e.g. about existing kernels) against the code before relying on them.
4. `docs/IMMUTABLE_CM_FEATURE_OPERATOR_2026-07-26.md` — confirms bin indices/thresholds/contrasts
   are already built once at context-init time for all four families (verified empirically,
   `rebuilds_due_to_A_or_gp=0`). The operator work below can and should build on that existing
   immutability rather than re-deriving it.
5. `docs/FIVE_FAMILY_PUBLIC_DRIVER_GATE_MATRIX_2026-07-26.md` and
   `docs/FIVE_FAMILY_KILL_RESUME_REPORT_2026-07-26.md` — what gate coverage exists and what
   doesn't (D=4 square/rectangular, "hard point", and kill/resume for the 4 restricted families
   were never run).

**Start your own work on a new branch/worktree based on `port/finish-five-family-optimization-
stack-2026-07-26` at `08550a8`** (e.g. `git worktree add ... -b <new-branch>
port/finish-five-family-optimization-stack-2026-07-26`) — do NOT restart from
`production/fullA-exact` and redo the audit; that would throw away real, verified work (including a
found-and-fixed default-safety bug in the inherited dual bank, and a found-and-fixed masked KNITRO
callback-error bug in this session's own smoke tests).

## Priority order for this session (user's explicit instruction)

1. **Matrix-free FG operators eliminating full G** (original prompt §4-5). This is the task's own
   "central unfinished item" and the user's top priority. For each of the four restricted families,
   replace the current dense/chunked-dense `moments!` fill with a true `forward!`/`transpose!`
   operator pair that never materializes a `(n, ncore_full)` or similar dense block. Order of
   attack (lowest-risk first, per `RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md`):
   flexible CM (an existing `:cm_lookup` kernel in `cm_lookup_kernels.jl` / `cm_lookup_production.jl`
   is already numerically validated but currently off-by-default due to an allocation regression —
   diagnose and fix the allocation sources named in the original prompt's §5.5 before assuming a
   new kernel is needed) → common Fréchet (reuses the CM operator + level-anchor direction) →
   CM+ZC (partition `[E | C | Z]`, do not force this before the CM operator is solid) → origin-ZC
   (`[E | Z]`, genuinely new — no existing partial kernel to build from).
2. **Winner-aware sparse cross-Hessian** (original prompt §6): the `Q'SR - π(ν'SR)` operator for
   the E×R block, per family, gated behind (1) actually existing (benchmarking a winner-aware
   cross block against a still-partially-dense FG baseline doesn't answer the real question).
3. **Interval-basis diagnostics** (original prompt §7): implement the interval CM basis as a real,
   gated production candidate (not just reusing `cm_lookup_kernels.jl`'s existing `:interval`
   method variant without re-validating it — the prior session found that variant produced
   universal KNITRO infeasibility when first tried against a cumulative-basis context, i.e. it is
   NOT already a validated interval-basis implementation), derive its Hessian from scratch as the
   prompt's §7.2 requires, and run the anchored-vs-orthonormal × cumulative-vs-interval 4-arm
   comparison.
4. **Everything else the original prompt asked for that the prior session didn't reach**: D=4
   square/rectangular gates and a distinct "hard point" stress case for all five families;
   process-group hard-kill + fresh-process-resume gates for the four restricted families; the
   eleven-category wall-clock attribution instrumentation and the five real 300-second profiles
   (original prompt §11) — sequence that *after* items 1-3 land, since profiling a
   still-partially-dense pipeline doesn't produce the numbers the task actually wants, and (per the
   prior session's own `FIVE_FAMILY_FINAL_PROFILE_STATUS_2026-07-26.md`) a "final" profile should
   run against a canonical-or-near-canonical state, not an early draft.

Do not skip straight to item 4's profiling step to produce a quick-looking deliverable — the
whole point of items 1-3 is that the profile numbers are meaningless until the dense-G elimination
is real.

## Concrete gotchas this session hit (save yourself the rediscovery time)

- Julia must be launched with `--project=<worktree-root>`, NOT `--project=.` from inside
  `full_aod_diag/d4_exact/` — the dependency environment (e.g. `SpecialFunctions`) is declared at
  the worktree root's `Project.toml`.
- `julia` binary is at `$HOME/.juliaup/bin/julia` — export
  `PATH="$HOME/.juliaup/bin:$PATH"` first (per this project's own memory,
  `/opt/shared_sw`'s Julia is broken).
- Always `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` before launching (hard project rule).
- `cm_originzc_target_layout.jl` must be included **before** `cm_originzc_config.jl` (the latter
  references a type the former defines) — a real include-order bug this session hit twice.
- `run_originzc_upper_checkpointed`'s default `cm_gradient_backend=:cplus` requires
  `cm_originzc_cplus.jl` to be included, or KNITRO's gradient callback throws `UndefVarError` deep
  inside a `puts` callback — which KNITRO reports as a generic `nStatus=-500` at the *outer* level,
  easy to misdiagnose as a numerical infeasibility rather than a missing include. **Always check
  the actual KNITRO status/log for a callback error before trusting any smoke test's own
  "n_eval > 0" or similarly weak pass condition** — this session had a real false-positive PASS
  from exactly this pattern; the fix was tightening the check to require a genuinely
  feasible-or-timelimit status.
- Any two KNITRO-solved quantities that were reached via different warm-start paths (e.g. an "OFF
  vs ON" comparison after other calls have mutated the shared `obj.x` warm-start slot) should be
  compared with `isapprox(rtol=1e-7)`, not `==` — this is expected numerical behavior for a
  well-posed convex inner problem reached via different KNITRO iteration paths, not a bug, and this
  codebase's own `test_d20_restricted_full_hessian_gates.jl` already uses that same tolerance.
- `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed` require an **explicit** `w0` for a
  fresh (non-resumed) run — unlike the unrestricted family's own stage runner, there is no
  "`nothing` means calibration" convenience. Build it the same way
  `cm_production_stage_runner.jl`/`originzc_production_stage_runner.jl` do (`w_calib = vcat(gp,
  pivot_reduce(log(...)))`, plus `eta_nu0`/origin-power-moment append for the extended families).
  `distribution_restriction` for origin-ZC must be one of `:unrestricted`,
  `:origin_specific_moments`, `:origin_specific_moments_zero_covariance` — NOT `:origin_by_power`
  (that's the *layout* symbol, a different kwarg, easy to confuse).
- The real calibrated `A_od` block spans **~11 orders of magnitude** (this project's own CLAUDE.md,
  confirmed again this session in the dual-bank distance-scale finding) — any new operator or
  scoring logic that uses a raw/unscaled distance or magnitude over economic-core coordinates will
  likely misbehave; scale deliberately or work in log-space.
- This project's standing rule: **always check in on a background job within ~30-60 seconds**
  (`ps` + tail its log) before settling into a long wait — this session caught three separate
  launch-time bugs (missing includes, wrong include order, wrong symbol value) this way, each of
  which would otherwise have wasted a full run's wall-clock time before failing.
- **Confirm with the user before pushing to `origin` or merging to `production/fullA-exact`** —
  standing project rule, and nothing from the prior session's branch has been pushed/merged yet.
- **Push session deliverables to Dropbox** (`dropbox:Gravity robustness/Analysis/Server
  Output/<new-subfolder>`) before ending the session — standing project rule, not optional. Use a
  new subfolder name, don't overwrite `finish_five_family_optimization_stack_2026-07-26` (the prior
  session's own package).

## Mathematical/scientific grounding

- `papers/` in the repo root has local copies of the CDW paper and Christensen–Connault (2023) —
  these define the underlying economic/statistical model this codebase implements. Read them if
  the operator derivations in original-prompt §5-7 aren't immediately clear from the code alone.
- Existing code to mirror the *pattern* of (not necessarily reuse directly — verify each claim):
  `core_exact_hessian.jl` / `compressed_cc_inner.jl` (the existing winner-sparse `E = Q - νπ'`
  economic-core machinery the cross-Hessian work in item 2 needs to interface with),
  `cm_hessian_architectures.jl` (`build_cm_bin_ctx`, `fill_cm_columns_from_bins!` — the current
  chunked-dense pattern items 1 is replacing), `cm_lookup_kernels.jl` (the existing, partially-
  validated lookup-FG kernel for flexible CM), `common_marginals_interval.jl` (existing partial
  interval-basis code — audit before trusting).
- Do not assume any two "should be equivalent" reconstructions actually agree without independently
  building both and diffing them directly — this project's CLAUDE.md documents a recurring,
  multi-session mistake of exactly this kind (the "A_od≡1 is not calibration" trap) in a different
  part of this same codebase; the discipline generalizes to any new operator work here (e.g. don't
  assume the interval-basis lookup kernel's targets match the production interval-basis spec
  without checking).
