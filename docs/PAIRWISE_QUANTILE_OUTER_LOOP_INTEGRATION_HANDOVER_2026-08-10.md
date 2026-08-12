# Handover: build the outer loop for the pairwise-quantile-independence restriction, and slot it in as family #6 of the paper_upper_v1 production runner

## Orientation

Two separate trees are involved, and reconciling them is your first real task (see Step 0):

- **This restriction's worktree**: `/bbkinghome/edav/cdw_worktrees/pairwise-quantile-independence-2026-08-09`, branch `prototype/pairwise-quantile-independence-2026-08-09`, forked from `fix/zc-cmzc-exclude-row-k2-k3-2026-08-07` @ `c22d831`. Has uncommitted changes as of 2026-08-09 (a full Hessian-optimization + `L`-genericity pass — see docs below). This is where the restriction's INNER solve lives.
- **The production five-family orchestrator**: main repo `/bbkinghome/edav/cdw`, branch `protocol/paper-upper-v1-2026-08-08` (**NOT merged to `production/fullA-exact` or any branch this restriction's worktree descends from** — confirmed via `git log --all`). This is where you need this restriction to plug in as a 6th family.
- KNITRO env: `source .knitro_env.sh` from whichever worktree root you're running from (`demand.mit.edu` is the licensed host). Julia via juliaup.

**Read first, in this order** (all in the restriction's worktree's `docs/`):
1. `PAIRWISE_QUANTILE_INDEPENDENCE_MATH_NOTE_2026-08-09.md` — the restriction's math (equivalence proofs, why `:all_cross`).
2. `PAIRWISE_QUANTILE_STATUS_2026-08-09.md` — what was built and validated first (real D4 KNITRO solve, D20/W=100k profiling).
3. `PAIRWISE_QUANTILE_HESSIAN_OPTIMIZATION_RESULTS_2026-08-09.md` — the most recent pass: Hessian callback sped up 3.2x (dedup+threading, 7.0x with 10 threads), and **the whole restriction was made `L`-generic** (number of quantile bins is now a required parameter everywhere, not hardcoded to 5). If you need to change the number of bins for a production campaign, this is already done — just pass `L`.

**This document is the task.** Everything above is the INNER solve only: given a FIXED outer point (θ, cutoffs), solve the inner KNITRO dual problem and verify it. That part is real, correct, and fast. **Nothing about the OUTER loop exists** — no outer gradient wired to real data, no objective-value return from the inner solve, no checkpointed outer driver, no family registration. This document scopes building all of that, reusing existing architecture as aggressively as possible.

## Standing instruction, repeated because it matters here specifically

**This codebase has a documented history of Claude sessions reinventing machinery that already exists**, often at a lower quality than the original (see `CLAUDE.md`'s own callouts, and multiple `feedback-*` memory entries about this). Before writing ANY new function for this task, grep for it first. Section "Precise reuse map" below tells you exactly what to reuse and where it lives — that section is not optional background, it's the actual spec.

## Correction to the task's original framing — read this before touching the outer gradient

The task as originally framed to this session was: *"the outer gradient should not be very difficult, it's the same basic set of ideas as the existing ZC gradients, just slightly different formulas — there are still parameters called nu, they just mean slightly different things."* **This framing is wrong in a specific, load-bearing way, confirmed by reading the actual ZC gradient code — not a guess.**

ZC's `nu` outer-gradient (`d_delta_dual_d_eta_origin_vec`, `cm_originzc_moments.jl:300-347` in the main `cdw` repo) is an **exact closed-form envelope-theorem derivative**, evaluated once at the converged inner dual `(ζ*, λ*)`:

```
d(Delta_dual)/d(nu_{o,k}) = -mean_m * ( lambda_mean,o,k* + sum_{p!=o} nu_{p,k} * lambda_pair,op,k* )
d(Delta_dual)/d(eta_{o,k}) = nu_{o,k} * d(Delta_dual)/d(nu_{o,k})     [chain rule, nu = exp(eta)]
```

No bandwidth, no bisection, no crossing-counting anywhere. **This formula is exact only because `nu` enters the moment matrix `G` smoothly/affinely** (`g_mean,o,k(s) = z_o(s)^k - nu_{o,k}`, a continuous target-level shift) **and moving it never reassigns any draw to a different bin/regime.** The economic winner-takes-all assignment (which origin supplies which destination) is governed entirely by a *separate* block (θ/`A_od`), untouched by `nu`.

This restriction's cutoffs `q_{o,r}` are the opposite case: they are quantile-bin **boundaries**. Moving `q_{o,r}` does nothing to almost every draw until `q` crosses an actual draw value, at which point that draw's bin membership — and every moment it participates in — jumps discontinuously. **`Delta_dual` as a function of any single cutoff is a genuine step function**, not a smooth function with an ordinary derivative. A ZC-style closed-form envelope derivative is literally undefined at the jump points and misleadingly zero everywhere else. This isn't a subtlety previous sessions missed — it's stated explicitly in this restriction's own files (`pairwise_quantile_cutoff_gradient.jl:1-14`, and `test_pairwise_quantile_d4_dense_oracle.jl:11-14`: *"the fixed-dual objective is a genuine step function of any single cutoff between consecutive draws, so an infinitesimal-h comparison would be meaningless here"*).

**Do not try to force ZC's exact-envelope formula onto the cutoff gradient.** The actual matching precedent in this codebase for a boundary-crossing outer parameter is the **economic `A_od` block's own gradient machinery** — `select_bandwidth`/`count_winner_flips` in `composite_gradient.jl:73-328` (canonical) and its allocation-free mirror `select_bandwidth!`/`count_winner_flips!`/`TwoOriginScratch` in `shared_a_gradient.jl:98-354`. General recipe (this IS reusable as a pattern, not economics-specific): (a) a boundary-crossing counter as a function of probe step `h` ("does perturbing by `h` flip which regime an element falls into"); (b) geometric bisection of `h` targeting a switching-mass fraction of ~0.3%–3% of draws (too small reproduces the wrong smooth-cell gradient, too large averages over too many kinks — see the docstring at `composite_gradient.jl:196-213` for why); (c) central FD of the objective at the selected `h`, one-sided/shrink fallback on a non-finite probe; (d) persist the selected `h` across outer iterations (a `bandwidth_cache`, threaded through the outer gradient callback).

**The good news: this restriction already has a bespoke implementation of essentially that same pattern**, and it's arguably *better* than the A_od block's own version in one respect — see next section.

## What already exists for the cutoff outer-gradient — reuse it, don't rebuild it

`pairwise_quantile_cutoff_gradient.jl` (in the restriction's worktree) implements `cutoff_secant_gradient!`/`fixed_dual_delta_f`/`bandwidth_target`/`crossed_draw_range`. Confirmed via `grep -rl` across the ENTIRE main `cdw` repo: **nothing outside this restriction's own worktree references any of these functions** — it has never been called with real data.

What it does, concretely: for each cutoff `(o,r)`, `bandwidth_target` (`:50-72`) picks the smallest up/down step that crosses at least `min_crossed` draws (an `O(log W)` search on a presorted column — `min_crossed` has **no default**, consistent with this repo's no-silent-defaults rule; you must choose it deliberately per campaign `W`, the docstring suggests "a few hundred draws at W=100,000"). `fixed_dual_delta_f`/`crossed_draw_range` (`:31-37`, `:75-134`) then compute the **exact** fixed-dual objective change from only the crossed draws in `O(k_crossed·D)`, via the task's own closed-form `ΔR_w` update — **not** a full `O(W)` re-evaluation of Ψ, which is what the A_od block's own FD-based method has to do per probe. A central secant is taken when both directions cross enough draws, one-sided otherwise, then mapped from physical cutoffs to raw KNITRO coordinates via `cutoff_jacobian_block!` (already `L`-generic as of the Hessian-optimization pass).

**Validation state, precisely** (do not overstate this to yourself or in any status doc — check it, and add to it):
- The exact per-crossing formula (`fixed_dual_delta_f`) IS validated as a genuine exact identity (tol `1e-10`) against a slow brute-force `O(W)` recompute at the same explicit cutoff bump — `test_pairwise_quantile_d4_dense_oracle.jl:215-231`.
- The AGGREGATE `cutoff_secant_gradient!` orchestration (bandwidth choice, central/one-sided combination, Jacobian accumulation) is only checked for "finite and non-degenerate" in that same file (`:233-235`) — **never validated against a reoptimized-FD ground truth**, unlike ZC's own gradient, which has exactly that gate (`d_delta_dual_d_eta_origin_fd`, `cm_originzc_production.jl:318-330`, re-solves the inner dual at each perturbed point; consumed by `test_cm_originzc_pure_moments.jl:177-210`'s testset with a documented `h`-shrinking protocol when tolerances aren't met at the initial `h`). Building this analogous gate for the cutoff gradient is real, necessary work — see the task list.
- It has **never been called with a real converged `r_current`** from an actual KNITRO inner solve — only synthetic hand-built `λ_M`/`λ_P` in the D4 oracle test.

**Conclusion: do not rewrite this module and do not try to replace it with ZC's formula.** Wire it to real data, validate it properly, combine it with the (unrelated, already-shared) economic gradient. That's the actual remaining work.

## Precise reuse map

**Family-agnostic / reuse UNCHANGED** (confirmed explicitly reused-as-is by origin-ZC's own code, meaning every family including yours is expected to call these, not reimplement them):
- The economic `(g, A_od)` outer-gradient block: `composite_gradient_at_fast`/`composite_gradient_at_Cplus_from_cache` (`composite_gradient.jl`/`composite_gradient_fast.jl`), wrapped by `economic_A_gradient!` in `shared_a_gradient.jl`. Origin-ZC's own header states this is *"computed EXACTLY as in the CM-only path"* (`cm_originzc_production.jl:16-18`) — i.e. this block does not need to know or care that a restriction is active. `get_or_build_econ_a_grad_ws` (also `shared_a_gradient.jl`) is the shared per-process cache this needs; it was deliberately moved out of any family-specific file because *"nothing in its own body is ZC-specific"* (`shared_a_gradient.jl:36-39`).
- Outer-algorithm pinning: `set_production_outer_algorithm!`, called family-agnostically (`cm_originzc_checkpoint.jl:560`).
- The A-space coordinate transform (`cm_aspace_coordinate.jl`, `pe::PivotGravityElim`/`pivot_expand`/`pivot_reduce`) — explicitly family-agnostic (`cm_originzc_checkpoint.jl:578-582`).
- Shared economic-core Hessian backend for H_EE (`exact_winner_pair_parallel`) — this restriction already reuses this unchanged (`winner_pair_hessian!`, confirmed in the earlier Hessian-optimization pass).
- `CMExpectedSolveFailure` — reused, not redefined, by every family (`cm_originzc_production.jl:20-21`).
- Checkpoint I/O mechanics: plain `Serialization.serialize`/`deserialize` (NOT JLD2, NOT JSON) with an atomic tmp-then-`mv` write, identically implemented per checkpoint version (`cm_originzc_checkpoint.jl:139-148,272-282,362-372`) — copy this exact pattern, don't invent a new serialization approach.

**Family-specific — you must write these, but each has a precise sibling to model on**:
- The restriction's own outer-gradient combiner. Sibling: `cm_originzc_production_gradient` (`cm_originzc_production.jl:249-278`) — calls the shared economic gradient, then `vcat`s the restriction-specific piece: `return vcat(g_econ, d_eta)`. Your analog: `pairwise_quantile_production_gradient` (name it whatever fits this file's own conventions) should call the same shared economic gradient function for the θ piece, then `vcat` the output of `cutoff_secant_gradient!` (called on the REAL converged `r_current`/`λ_M`/`λ_P`) for the cutoff piece. **Verify this composition is even the right shape**: confirm the combined outer-variable vector layout you need (presumably `w = vcat(θ_free_coords, raw_cutoffs)`, mirroring origin-ZC's `w = vcat(gp, zfree, eta)`, `cm_originzc_checkpoint.jl:702-756,779`) before wiring the gradient to match it index-for-index.
- **An objective value for the inner solve to return.** THIS DOES NOT EXIST YET and is a prerequisite for everything else — `archPQ_base_state` (`pairwise_quantile_production.jl`) currently returns `(nStatus, x, obj, n_fg, n_hess)`, no `K_hard`/`Delta_dual` scalar at all. Compare to how other `OperatorPsiBundle`-based families set this (check `archOZ_base_state`/`archOZ_verified_state` in `cm_originzc_production.jl`, and `operator_psi_bundle.jl` for the `obj.H_save`-style convention) and mirror it exactly — do not invent a new convention for what the outer objective scalar is or where it's stored.
- The checkpointed outer driver itself. Sibling: `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl:510-975`) — read the full 23-step structure below and mirror it precisely (there is no reason for your driver's overall shape to differ).
- The checkpoint schema struct. Sibling: `OriginZCCheckpointV10` (`cm_originzc_checkpoint.jl:312-360`, 42 fields) plus its load/upgrade fallback chain (`load_cm_checkpoint_v10`, `:398-405`). Your version starts fresh (call it `PairwiseQuantileCheckpointV1` or similar — no need to fake a version history) but should carry the same STRUCTURAL field groups: run identity, draw provenance (`W,draw_seed,draw_design,draw_checksum_*`), outer-point/incumbent state (`best_feasible`, `n_eval`, `n_grad`, `wall_elapsed`, `bandwidth_cache`), plus this restriction's own fields (`L`, cutoff layout params, `min_crossed`).
- Seed-generator registration (see the family-6 section below) — a `FamilySeedSpec`/constructor function/`build_family`+`evaluate_family` arms, mirroring the origin-ZC pattern exactly.

**A concrete building block you already have for the seed generator's "good starting point" step**: origin-ZC's seed generator computes a companion-LFD-implied `nu` as a good outer-search starting value (`companion_implied_nu_originzc`, `multistart_seed_generator.jl:536-547`). You already have the direct analog for cutoffs — every test/profile script in this restriction's worktree builds starting cutoffs from the empirical quantiles of the calibration draws via a `quantile_naive` helper (see e.g. `test_pairwise_quantile_real_d4_knitro.jl`'s `raw_cutoffs` construction). That's exactly the right idea for the seed generator's own starting-point logic — adapt it, don't design a new one from scratch.

## The 23-step structure of `run_originzc_upper_checkpointed` (`cm_originzc_checkpoint.jl:510-975`) — mirror this

1. Resolve `ckpt_dir` to an absolute path before any real-data setup (a downstream `cd()` bit a prior run otherwise) — `:612-622`.
2. Hard-error validation of required/enumerated kwargs — `:624-635`.
3. Build the family's config struct, resolve any derived counts — `:640-642`.
4. If `resume_from` given, load and deserialize the checkpoint — `:644`.
5. Resume-mismatch guards: hard errors (not silent overrides) on direction/K/layout/moment-version/destination-sample/backend mismatches — `:651-680`.
6. Build the real-data economic context (`d20_real_setup_design`/`d4_exact_setup`), attach compressed-factual workspace, pivot elimination — `:683-685`.
7. Coordinate-mode branch setup (closures for encode/decode) — `:687-706`.
8. Build the restriction's own layout, validate bounds — `:707-708`.
9. Any profiled/eliminated-coordinate handling (origin-ZC has one for its focal k=σ-1 row; you likely don't need this) — `:709-727`.
10. Reconstruct `w0` on resume (dimension/checksum/coordinate-mode reconciliation) else require fresh `w0` — `:730-756`.
11. Build the production context (`prepare_production_run`), attach counters, dual bank, set BLAS threads, write a JSON backend-manifest artifact — `:758-778`.
12. Compute KNITRO outer-variable bounds — `:800-805`.
13. KNITRO outer solve setup (`KN_new`, load `.opt`, add vars/bounds/init values, add constraint) — `:807-822`.
14. Mutable outer-loop state (`last_F_state`, `best_feasible`, counters, `trace`, `bandwidth_cache`, timers) — `:824-837`.
15. `do_checkpoint(reason, w_current)` closure — `:839-857`.
16. `cb_F!` objective/constraint callback: unpack `w`, optional exact-cache lookup, compute value via the family's own verified-value function, checkpoint on new-best or wall-interval, push to trace — `:859-901`.
17. `cb_G!` gradient callback: reuse cached state if the point matches, call the family's own combined-gradient function, write `evalResult.jac` — `:902-935`.
18. Register callbacks (`KN_add_eval_callback` + `KN_set_cb_grad`) — `:937-938`.
19. Optional algorithm-pin assertions — `:940-943`.
20. `KN_solve`, pull results, `KN_free` — `:944-947`.
21. Compute any derived summary scalar (origin-ZC computes `κ`; you likely don't need an analog) — `:949-951`.
22. Final re-verification at the solution, checkpoint as `:stage_complete`/`:stage_complete_unverified` — `:953-969`.
23. Return `NamedTuple(knitro_status, wall, n_eval, n_grad, best, xsol, trace, final_checkpoint, ckpt_path, screen_summary, ...)` — `:970-975`.

## Slotting in as family #6 of `paper_upper_v1`

This is the "reproducible five-family upper bound" run. It lives on `protocol/paper-upper-v1-2026-08-08` in the main `cdw` repo (**unmerged** — see Step 0). Do not confuse it with the `docs/FIVE_FAMILY_*.md` files under `docs/` (11 files, dated 2026-07-26 to 07-28) — those describe an **earlier, superseded** "five-family" comparison with a different roster (it includes a real standalone mean-ZC family) and predate this protocol by 1-2 weeks. Also don't over-trust `docs/audits/reproducible-multistart-generator-2026-08-08/MASTER.md` — it documents the seed generator's ORIGINAL 3-kind API from before the same-day `paper_upper_v1` extension; the best current design doc is `paper_upper_v1_orchestrator/ORCHESTRATOR_STATUS.md`.

**The actual current 5 families** (from `protocols/paper_upper_v1.toml`, NOT a guess): `UNRESTRICTED` (driver `run_polish_checkpointed_unified`), `COMMON_MARGINALS` (`run_cm_upper_checkpointed`, `cm_extension=:cm_only`), `COMMON_FRECHET` (`run_cm_upper_checkpointed`, `marginal_restriction=:common_frechet`), `ORIGIN_ZC` (`run_originzc_upper_checkpointed`), `CM_PLUS_ZC` (`run_cm_upper_checkpointed`, `cm_extension=:cm_plus_moments`). There is no standalone mean-ZC family in this specific protocol.

**Dispatch is a plain `Symbol`-keyed `if/elseif` at multiple sites, NOT a registry** (worth knowing so you don't go looking for one, and worth matching in style rather than refactoring into something new — a registry would be cleaner, but changing the dispatch mechanism itself is out of scope for adding one family). Sites that need a new arm for family #6:
1. `full_aod_diag/d4_exact/multistart_seed_generator.jl:475-514` — `build_family(ctx, spec::FamilySeedSpec)`, dispatches on `spec.kind`.
2. `full_aod_diag/d4_exact/multistart_seed_generator.jl:630-663` — `evaluate_family(ctx, fb, x_free; eval_id)`, dispatches on `fb.spec.kind`.
3. `full_aod_diag/d4_exact/multistart_seed_generator.jl:247-255` — `paper_five_family_seed_specs` — register your new `FamilySeedSpec` in the returned vector.
4. `paper_upper_v1_orchestrator/family_start_chain.jl:135-146` — `call_driver`, dispatches on the TOML string driver name. If your driver doesn't fit the uniform kwargs convention (unlikely, but `UNRESTRICTED` needed its own wrapper — `call_unrestricted_driver`, `:184-215`, plus a branch in `run_stage`, `:262-274`), you'll need an analogous wrapper.
5. `paper_upper_v1_orchestrator/launch_wave.sh:27` — hardcoded bash array `FAMILIES=(...)` — hand-edit, not TOML-driven.
6. `protocols/paper_upper_v1.toml` — new `[families.PAIRWISE_QUANTILE]` block (pattern at lines 66-147: `driver`, `delta_restoration_driver`, a `[families.<ID>.kwargs]` sub-table — these are read generically by `fam_kwargs()`, `family_start_chain.jl:128-138`, so no per-family kwarg-NAME logic needs touching there). Also update `[concurrency]` (`phase1_families_per_start` 5→6, `phase1_jobs_per_wave`, `total_cores_required_per_wave` — currently 10 cores/process × 10 processes = 100 cores/wave at lines 189-206, recompute for 6 families). Also extend `[projection]` (`edges`/`inheritance_edges`, lines 218-231) with this family's nesting relationships — even though Phase II (cross-family projection) isn't built yet, the graph already exists structurally.

**The two-layer family contract, precisely:**
- **Seed-generator layer**: a `FamilySeedSpec` (struct fields at `multistart_seed_generator.jl:119-129`: `id, kind, K_mean, K_pair, L, contrasts, probs, include_truncated_moment, meanzc_basis` — most of these are generically reused, leave the ones that don't apply at their "off" sentinel, e.g. `0`/`false`/`:direct`) with a `kind::Symbol` you invent; a constructor function (pattern: `origin_zc_family_spec`, `multistart_seed_generator.jl:147-149`); `build_family`/`evaluate_family` arms returning a `FamilyBuild`/`FamilyLiftResult` (`FamilyLiftResult` fields at `:594-607`: `family_id, kind, full_outer_vector_digest, nu_values, nu_policy, derived_focal_nu, Delta_star, verified, inner_status, verification_class, layout_digest, wall_seconds` — most fields have an obvious null/N-A value for a family without ZC's specific "nu_policy"/"derived_focal_nu" concepts).
- **Production-driver layer**: your checkpointed driver function must accept the common kwarg set `family_start_chain.jl`'s `call_driver` merges in unconditionally (`:118-121`) — `ckpt_dir`, `label`, `checkpoint_interval_s`, `maxtime_real`, `resume_from`, `verbose`, plus every `[scientific]`-block kwarg (`draw_design`, `draw_seed`, `inner_lower_limit`, `destination_sample`, `exclude_diagonal_gravity`, `gravity_exclude_cells`, `σHat`, `find_smallest`, `W`, `z_halfwidth`) — and must return a `NamedTuple` with (at minimum) `.knitro_status::Int`, `.n_eval`, `.n_grad`, `.best` (either `nothing` or `NamedTuple(gp=..., w=..., Delta=..., n_eval=..., t=...)`, exact shape confirmed at `cm_checkpoint.jl:1746-1749`, read at `family_start_chain.jl:157,166-175,232-238`). Also accept `objective_mode::Symbol` (`:min_gp` default / `:min_delta_fixed_gp`) + `gp_fixed::Union{Nothing,Float64}` if you want Stage B/R0 fixed-gp restoration to work the same way it does for every other family (guard pattern: `cm_originzc_checkpoint.jl:645-648`).

**A correction to a claim in this restriction's own prior status doc**: `PAIRWISE_QUANTILE_STATUS_2026-08-09.md` refers to a "~200-kwarg surface" for production entry points. A dedicated search could not find this figure stated anywhere in this repo's text; the actual measured kwarg count for `run_originzc_upper_checkpointed` is 35, and `run_cm_upper_checkpointed` is "north of 45" but not close to 200. Don't be intimidated by a stale/exaggerated figure — the real surface is closer to ~35-50 kwargs, grouped into: scientific/model params (several required, no defaults), restriction-layout/moment options, solver/algorithm options, performance/backend options, checkpoint/resume/logging options.

**Scope check**: Phase II (cross-family projection) and Phase III (final polish) are **not implemented at all** in this orchestrator yet — confirmed absent, not just undocumented (`paper_upper_v1_orchestrator/ORCHESTRATOR_STATUS.md:24-27`; `run_paper_upper_bounds.jl` runs Phase I waves only and prints "not yet implemented" after). So "slotting in as family #6" concretely means: making this restriction runnable through **Phase I** (per-family/per-start/per-delta discovery via `family_start_chain.jl`), matching what the other 5 families currently do — not a complete final-bound pipeline, since that doesn't exist for anyone yet.

**`protocol_sha` freeze mechanism — flag to the user, don't decide unilaterally**: `protocols/paper_upper_v1.toml`'s own header (line 26) says *"Once any cell of this protocol has started, THIS FILE MUST NOT BE MUTATED. Any substantive change requires a new protocol version (paper_upper_v2.toml) with its own root output tree."* Adding a 6th family is a bigger change than any precedent — but actual practice on this branch has repeatedly just re-frozen `protocol_sha` in place after bug fixes (the many `Freeze protocol_sha` / `Reset protocol_sha` commit pairs in git log). Whether adding a new family should be a `paper_upper_v2.toml` or an in-place re-freeze is a real decision with consequences (a new protocol version means a fresh output root, i.e. the existing 5-family results so far aren't reused) — **raise this explicitly with the user before choosing**, don't pick one silently.

## Step 0: reconcile the two branches (do this first, verify it, don't rush it)

The restriction's worktree (`prototype/pairwise-quantile-independence-2026-08-09`) and the orchestrator's branch (`protocol/paper-upper-v1-2026-08-08`) diverged from a common ancestor and have never been combined. Before any of the above can be built, you need ONE working tree with both. Recommended approach (verify each step, don't assume it's clean):

1. `git log --all --oneline` in `/bbkinghome/edav/cdw` to find the actual merge-base of the two branches.
2. Create a NEW local branch/worktree (not pushed anywhere) that merges both — e.g. branch off `protocol/paper-upper-v1-2026-08-08`, then merge in the restriction's commits (`f8127bc`..`5c79730` plus this session's uncommitted Hessian-optimization/`L`-genericity changes, which you'll need to commit first in the source worktree or carry over as a patch). The restriction's own files are all NEW files (`pairwise_quantile_*.jl`) that the orchestrator branch never touches, so a clean merge is likely — but confirm, don't assume; check for conflicts especially in any shared file both branches might have touched (e.g. `multistart_seed_generator.jl`, `cm_checkpoint.jl`, `cm_originzc_checkpoint.jl`, `CLAUDE.md`).
3. This is a local, reversible operation — fine to just do it. **Do not push the merge result to any shared/remote branch without confirming with the user first**, per this repo's own standing rule (confirmed before in this project: "confirm before pushing to a real remote").
4. Once combined, re-run this restriction's own existing validation suite (the D4 dense oracle at a couple of `L` values, the real-KNITRO D4 A/B+verifier) in the new combined tree to confirm nothing broke in the merge, before starting new work.

## Suggested order of work

1. Step 0 (branch reconciliation), verified.
2. Add the missing objective-value return to the inner solve (`archPQ_base_state` / a `K_hard`-equivalent), matching the existing `OperatorPsiBundle`/`obj.H_save` convention used elsewhere — read `archOZ_base_state`/`operator_psi_bundle.jl` first to find the exact convention, don't invent one.
3. Wire `cutoff_secant_gradient!` to a REAL converged inner solve (real `r_current`/`λ_M`/`λ_P` from an actual `archPQ_base_state` call), choose `min_crossed` deliberately for at least one real campaign `W`.
4. Build the reoptimized-FD validation gate for the AGGREGATE cutoff gradient, mirroring `d_delta_dual_d_eta_origin_fd` + `test_cm_originzc_pure_moments.jl`'s testset (h-shrinking protocol included) — this is the standard this codebase already holds ZC's gradient to; hold this restriction to the same standard before calling it correct.
5. Build the combined outer-gradient function (θ via the shared economic block + cutoffs via step 3-4's output, `vcat`'d), and validate the COMBINED vector too (not just the cutoff piece in isolation) — confirm the economic piece is unaffected by this restriction being active (it should be, by construction, since it's family-agnostic, but verify, don't assume — this codebase's own memory has "always verify analytic gradient against FD before trusting" as a hard-won lesson).
6. Build the checkpointed outer driver (`pairwise_quantile_checkpoint.jl` or similar), mirroring the 23-step structure above precisely.
7. Register as family #6 at all the dispatch sites listed above; raise the `protocol_sha`/`paper_upper_v2.toml` question with the user.
8. Smoke-test end-to-end at small scale before claiming done — mirror `smoke_objective_mode_min_delta_fixed_gp.jl`'s pattern (real KNITRO, a few concrete families/points, not synthetic).
9. Update/add a dated status doc recording what was built and what was validated (this repo's own convention: new doc per session, never silently edit history), and push deliverables to Dropbox per `CLAUDE.md`'s standing instruction.

## What NOT to do

- Do not implement a ZC-style closed-form envelope gradient for the cutoffs — it does not apply (see the correction section above); this would very likely be silently wrong (zero almost everywhere, undefined at the crossings) rather than obviously broken.
- Do not rewrite `cutoff_secant_gradient!`/`fixed_dual_delta_f` from scratch, and do not port `select_bandwidth`/`count_winner_flips` wholesale to replace it — the existing module is a reasonable, already-partially-validated implementation of essentially that same pattern; extend and validate it, don't discard it.
- Do not reimplement the economic `(g, A_od)` outer gradient — call the shared `composite_gradient_at_fast`/`economic_A_gradient!` machinery, exactly as origin-ZC does.
- Do not invent a new checkpoint serialization format — plain `Serialization.serialize`/`deserialize` with atomic tmp-then-`mv`, matching every existing checkpoint struct.
- Do not refactor the family-dispatch mechanism (e.g. into a Dict registry) as part of adding one family — match the existing `if/elseif` style at each of the 4-5 sites listed.
- Do not silently default `min_crossed`, `L`, or any other genuine modeling choice — this repo's no-defaults rule applies here exactly as it did for the inner solve.
- Do not push branch-reconciliation or protocol-file changes to any shared/remote branch without confirming with the user first.
- Do not touch Melitz code (`src/melitz/`, `scripts/melitz_*`, `melitz/`, `test/melitz/`) — deliberately out of scope for this whole restriction, per `CLAUDE.md`.
- Do not claim the outer gradient is correct without a reoptimized-FD gate on the AGGREGATE combined vector, not just the isolated cutoff piece or the "finite and non-degenerate" smoke check that currently exists.
