# Handoff to the omit-ROW (Part A) session — 2026-07-24

This release (`release/fullA-screens-threshold10-now-2026-07-24`, merged to
`production/fullA-exact` as tag `screens-threshold10-production-ready-2026-07-24`) shipped Part B
(restore exact infeasibility screens to CM/CM+mean-ZC/origin-ZC) and Part C (threshold=10
certified early-abort) independently of your omit-ROW work, per an explicit instruction not to
wait on it. Your branch/worktree (`release/fullA-omit-row-restore-screens-2026-07-23`) was **never
read past `git show <already-committed-commit>` and never modified** by this release — whatever
you have in progress there is untouched.

## What you need to do

1. **Rebase or selectively port onto the new production tip before your final merge.** The new
   tip supersedes the screens/threshold *implementation* your own branch already carries (you
   cherry-picked/authored the same `acae5c7` commit this release started from) — do not
   reintroduce an older copy of `cm_screen_bridge.jl` / `cc_algo/threshold_early_abort.jl` /
   `cc_algo/active_layout.jl` on top of the new tip. If you rebase, expect these files to already
   exist with additional fixes (see below) — take theirs (the new tip's), not your branch's older
   version, on any conflict in those specific files.

2. **Populate `active_destinations(ctx)` (and `active_origins(ctx)` if you ever reduce the origin
   side) with your reduced, named destination set.** `cc_algo/active_layout.jl` is new
   infrastructure this release added specifically so your omit-ROW context can plug in without
   touching any screen call site:
   ```julia
   active_origins(ctx)      # falls through to 1:ctx.D unless ctx.active_origins is set
   active_destinations(ctx) # falls through to 1:ctx.D unless ctx.active_destinations is set
   active_od_cells(ctx)     # Iterators.product of the two above
   ```
   Today's contexts (built by `context_real_d20.jl::d20_real_setup` and the CM/ZC builders on top
   of it) carry no such fields, so every accessor defaults to the full `1:D` range — zero behavior
   change. When your context drops a destination (e.g. ROW), set `ctx.active_destinations` (via
   whatever `merge(ctx, (active_destinations = ...,))` pattern fits your own context-construction
   code) to the reduced index set; **do not** hard-code `for o in 1:ctx.D, d in 1:ctx.D` anywhere
   new — use these accessors instead, matching the one call site this release already converted
   (`cm_screen_bridge.jl`'s witness-certificate loop).

   **Important scope note**: this accessor layer only covers the ONE loop this release's own code
   already had (the witness loop in `cm_screen_bridge.jl`). It does **not** make
   `pairwise_certificate`/`screen_hard_winners` (`infeasibility_screen.jl`) themselves
   rectangular — those still operate on full D×D matrices internally. Generalizing those two
   certificate functions to a genuinely reduced destination set is still your job, not something
   this release did for you. The accessor layer exists so that once you do that generalization,
   the screen *call sites* around it don't also need to change.

3. **Preserve every screen and threshold call path this release wired.** Specifically, do not
   revert or bypass:
   - `cm_screen_precheck!` being called before every restricted-family inner solve (via the
     `..._screened` wrapper functions in `cm_screen_bridge.jl`, wired into all 6+3 call sites in
     `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl`, plus the flexible-CM stage-runner preflight
     check — see the "second unwired-screen bug" finding in
     `docs/SCREEN_STACK_FINAL_AUDIT_2026-07-24.md` if you're wondering why that preflight call
     looks different from the mean-ZC branch next to it).
   - `threshold_state = obj0.threshold_state` (or `obj_cm.threshold_state`) being forwarded in
     **all seven** objective-bundle rebuild sites (`common_marginals_moments.jl`,
     `cm_production_bundle.jl`, `cm_hessian_architectures.jl`, `common_marginals_interval.jl`,
     `c12b_interval_common_marginals_moments.jl`, `cm_meanzc_moments.jl`,
     `cm_originzc_moments.jl`). If your omit-ROW work adds an EIGHTH rebuild site (e.g. a new
     rectangular-moments builder), it needs the same `threshold_state = obj0.threshold_state`
     line — this was a real, previously-undiscovered bug this release found and fixed (every one
     of those sites silently defaulted to `threshold=Inf`, disabling Part C, before the fix); do
     not reintroduce the class of bug by omitting it from a new site.
   - `with_screen_counters(pcx)` + `counters = pcx.screen_counters` threading through any new
     production entry point you add, if you want screen observability (calls/hits/wall) to show
     up for your omit-ROW campaigns the same way it now does for the existing four.

4. **Rerun only the bounded set the task specified — not a broader regression sweep:**
   - Non-square mask tests: `full_aod_diag/d4_exact/test_active_layout_accessors.jl` (construction-only, seconds) — extend Section 2/3 with a case shaped like your actual omit-ROW reduced set if you want extra confidence, but the existing synthetic cases already prove the accessor logic itself.
   - One restricted screen-hit test: reuse `test_cm_screen_restoration.jl`'s pattern (Group 3, pathological point) against your rectangular context once built.
   - One threshold test: reuse `test_threshold_propagation_regression.jl`'s pattern (construction-only field check, `pcx.ctx_cm.obj.threshold_state.threshold == 10.0` at delta=1) against your rectangular pcx builders.
   - One supervisor smoke: same shape as this release's four (see `docs/SCREEN_STACK_FINAL_AUDIT_2026-07-24.md`'s Part D table) for whichever mode your omit-ROW work targets first.

5. **Do not overwrite this release's B/C implementation with an older branch copy.** If your own
   `release/fullA-omit-row-restore-screens-2026-07-23` branch still has the pre-fix versions of
   the seven rebuild sites (missing `threshold_state` forwarding) or the pre-fix flexible-CM
   preflight check (missing the screened wrapper), a naive merge/rebase could silently
   reintroduce those two bugs. Diff against the new tip's `cc_algo/` and
   `full_aod_diag/d4_exact/cm_*.jl` files before finalizing your own merge.

## Known pre-existing gaps this release found but left unfixed (informational, not blocking)

- `test_cm_verified_success.jl` (D=4) fails with `UndefVarError: meanzc_resolve_K not defined` —
  confirmed pre-existing on a clean, untouched `production/fullA-exact@fd21f9f2` checkout, caused
  by that test file's own stale `include` list (predates `cm_meanzc_config.jl`). Unrelated to
  screens/threshold or omit-ROW; not this release's or your bounded scope to fix, but worth
  knowing about if you see it fail again.
- Four non-production diagnostic/shakedown scripts (`c33_phase4_cm_shakedown_control.jl`,
  `c33_phase4_cm_shakedown_interrupt.jl`, `c33_phase4_cm_shakedown_resume.jl`,
  `cm_cplus_matched_trajectory.jl`) call `run_cm_upper_checkpointed` without including
  `cm_screen_bridge.jl` — same class of stale-include bug as above, will `UndefVarError` if run
  as-is. The actual production entry points (`cm_production_stage_runner.jl`,
  `originzc_production_stage_runner.jl`) are unaffected.

## Where to find the details

`docs/SCREEN_STACK_FINAL_AUDIT_2026-07-24.md` in this release has the full audit: reachability
table, both new bugs found+fixed (threshold propagation, second screen bypass), full real
D=20/W=80,000 test/smoke results for all four current production modes. Read it before touching
any of the files it names.
