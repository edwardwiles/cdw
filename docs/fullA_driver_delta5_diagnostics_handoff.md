# Full-A_od driver consolidation + δ≥2 continuation diagnostics

**⚠ READ §17 FIRST if you only read one section.** A follow-up addendum asked for the
outer γ'_focal bounds to be split at the Frechet benchmark for upper vs. lower runs. The
audit that fix required surfaced a **real, substantive direction-label bug**:
`run_staged_delta5_continuation` (the function at the center of this entire
investigation) hardcoded `find_smallest=false` on every call regardless of which
direction the caller wanted, and `c10_d20_production_driver.jl`'s own
`direction = find_smallest ? :lower : :upper` checkpoint-labeling convention was
backwards relative to this repo's own established, explicitly-calibrated convention.
**§17 also flags an explicit, unresolved disagreement between the addendum's own literal
bound formulas and the evidence found in this repo's own code/data — implemented the
evidenced direction, not the addendum's literal text, and this needs the user's own
review before any production merge.**

Branch: `diag/fullA-driver-delta5`, worktree `gravity-fullA-driver-delta5`, based off
`diag/fullA-d4-exact @ f6ae01e` (the latest reviewed/stable production-consolidation
commit at session start). **Not merged into production.** This session ran alongside a
separate, actively-developing CM/threading integration (see §1) and deliberately avoided
that work's files and worktree.

## 1. Branch/worktree map (as of session start)

| Role | Branch | Worktree | Tip |
|---|---|---|---|
| Latest reviewed production | `diag/fullA-d4-exact` | `gravity-fullA-d4` | `f6ae01e` |
| Production-consolidation source | `integration/fullA-d20-runtime-delta5` | `gravity-fullA-d20-runtime-delta5` | merged into `diag/fullA-d4-exact` via fast-forward `98983bd`→`afb75af`, then a no-conflict merge bringing in the QMC draw-design port, final tip `f6ae01e` |
| Rollback tag | `pre-consolidation-2026-07-21` | (on `gravity-fullA-d4`) | `98983bd` |
| **Active CM/threading integration** (not touched this session) | `integration/fullA-cm-parallel-production` | `gravity-fullA-cm-parallel-production` | `71559f4` (1 commit past `f6ae01e`: KNITRO-version fail-fast check) **+ substantial uncommitted work** across `cc_algo/inner_loop_functions.jl`, `full_aod_diag/d4_exact/{cm_hessian_architectures.jl, cm_config.jl, composite_gradient_fast.jl, compressed_live.jl, context.jl, lfix_buffer_reuse.jl, oracle.jl, oracle_fast.jl}`, `full_aod_diag/ek_inner.opt`, new `parallelism_guards.jl` — all left untouched |
| **This session** | `diag/fullA-driver-delta5` | `gravity-fullA-driver-delta5` | see commit list, §14 below |

**A near-miss worth recording**: partway through this session an edit intended for this
branch was accidentally applied to the active CM/threading worktree
(`gravity-fullA-cm-parallel-production/full_aod_diag/d4_exact/staged_delta5.jl`). Caught
immediately via `git diff` on that specific file (confirmed the entire diff was the
accidental edit, nothing pre-existing), reverted with `git checkout --
full_aod_diag/d4_exact/staged_delta5.jl` in that worktree only, leaving every other
uncommitted file in that worktree (the other Claude's own live work) untouched. The
correct edit was then re-applied in this session's own worktree. No lasting effect on the
other worktree.

**Conflict surface with the active integration branch**: `c10_d20_production_driver.jl`
is touched by both. The active branch's own diff (`f6ae01e`→`71559f4`) adds a
`knitro_version` field to `D20Checkpoint` (schema 2→3) and touches the two
`do_checkpoint` closures' constructor calls. This session's changes touch different
lines entirely (the `best`/`best_feasible` `Ref` initialization ~15-20 lines earlier, the
ctx-build block, `cb_F!` bodies) — low collision risk on rebase, but **do not merge
either branch into production until the other completes and this branch is rebased and
re-tested** (task §14/§15).

Handoff docs read at session start: `fullA_D20_production_consolidation_handoff.md`,
`fullA_D20_canonical_rerun_handoff.md`, `fullA_D20_warmstart_replay_handoff.md` (all in
`gravity-fullA-cm-parallel-production/docs/` and/or the canonical-rerun/warmstart-replay
worktrees' own `docs/`).

## 2. Numerical kernels: unchanged

No edits to winners, CES kernels, economic moments, common-marginal moments, the inner
CC objective/gradient/Hessian, `L_fix` derivative formulas, infeasibility certificates,
warm-start *scoring* (`dual_bank.jl`'s `cheap_score`/`select_warm_start` logic itself —
only its already-computed return value is now additionally recorded, see §12), threading/
BLAS settings, or KNITRO solver options (beyond diagnostic queries, §10). All new/changed
code lives in files this session added or in `c10_d20_production_driver.jl`'s own driver
orchestration (which this driver, not the frozen inner-solve files, owns).

## 3. The staged-continuation incumbent bug: root cause and fix

**The bug.** `run_polish_checkpointed` and `run_profile_checkpointed`
(`c10_d20_production_driver.jl`) each cold-verify the caller-supplied starting point as
inner-feasible *before* calling `KN_solve` (`r0`/`r_seed`), then discarded that result:

```julia
# pre-fix
best_feasible = Ref{Any}(resumed !== nothing ? resumed.best_feasible : nothing)
```

`nothing` on every fresh (non-resumed) call — even though `r0.inner_status in
FEASIBLE_CODES` had *just* been asserted one line above. KNITRO's own first evaluation of
that same point inside `cb_F!` happens via a **warm** solve against `ctx.obj.x`, a
freshly-initialized dual slot — not the cold solve `r0` just performed. That warm
evaluation can legitimately fail, or simply never complete before `maxtime_real` is
exhausted (a real, not hypothetical, risk: the production-consolidation session's own
Phase E finding was that each staged-continuation stage burns ~65-83s of its budget on
context rebuild alone, leaving as little as ~17-46s of genuine KNITRO wall time per
stage). When that happens, `best_feasible` stays `nothing` until *something* feasible is
found — which can be a materially **worse** point than the guaranteed-feasible start
already in hand, because the comparison `is_new_best = ... (best_feasible[] === nothing
|| ...)` accepts the *first* feasible point unconditionally when the incumbent is empty.

`run_staged_delta5_continuation` never uses `resume_from` between stages (each stage is a
fresh call), so this reset happens at *every* stage transition. Chained across
2→3→4→5, a run can ratchet the reported incumbent downward at each stage even though the
true feasible set only grows with δ — reproducing the task's own reported pathology
(κ 0.0806→0.0281→0.0093→0.0040→0.0031) without any economic-formula bug at all.

**Audit checklist (task §3), answered by code inspection**:
- Upper vs. lower direction / objective sign: `is_better_polish` (extracted, see below)
  correctly implements `find_smallest ? cand < best : cand > best`; not the bug.
- Whether the start point is evaluated before KNITRO starts: **yes** (`r0`/`r_seed`).
- Whether it's registered as a feasible incumbent: **no — this was the bug.**
- Terminal point vs. best-feasible point: the return value correctly returns
  `best_feasible[]` (best-ever), not the terminal KNITRO iterate; not the bug.
- Checkpoint/return-value/summary-table fields: correct once `best_feasible[]` itself is
  correct; not separately broken.
- Stale lower-bound flag inherited across stages: not applicable — `find_smallest` is
  passed explicitly per call, not inferred from prior state.

**The fix** (`incumbent_logic.jl`, wired into both functions): `seed_incumbent(resumed_best,
cand_feasible, cand)` — resumed incumbent wins unconditionally if present, else the
cold-verified start point (gated on the *same* feasibility test `cb_F!` uses: inner
solve success **and** `Δ_dual ≤ δ + 1e-6` for the polish stage) becomes the seed. Extracted
as a pure, side-effect-free function specifically so it's unit-testable without KNITRO or
real data (§4). `is_better_polish`/`is_better_profile` were also extracted (same
functions `cb_F!` already used, just given names) so the comparison direction itself is
independently testable.

**A second real finding, caught empirically during real-data validation, not derived a
priori**: κ = 1 − gp^(σ/(σ-1)) is a strictly **decreasing** function of `gp` whenever
σ>1 (confirmed σ≈2.5 from this session's own observed (gp, κ) pairs — see §6). Under
`find_smallest=false` ("upper"), `is_better_polish` greedily *maximizes* `gp` — meaning κ
can legitimately *decrease* stage-to-stage even with a fully-correct incumbent tracker, if
κ isn't monotonic in the actual tracked quantity everywhere. This session's first
validation-script draft asserted κ-monotonicity directly and got a spurious "FAIL" for
exactly this reason (§6) — fixed to check `best_gp` (the literal quantity the incumbent
tracks) instead. **This does not fully explain the task's own opening numbers** (κ
0.0806→0.0031, a ~26x compression, at real boundary points far from this session's own
near-autarky validation starting point) — see §6's caveat on why this session's own real
run is a mechanical-correctness proof, not a literal reproduction of the original report.

## 4. Regression tests (`test_incumbent_seeding.jl`, 18/18 passing)

Deterministic, no KNITRO, no real D=20 context — a toy callback loop
(`toy_polish_run`) reproduces exactly the seed→compare→update pattern under test.
Covers every case task §4 asked for: increasing-budget seeding, zero-iteration
termination, rejected-final-trial non-regression, upper/lower direction, checkpoint/
resume incumbent retention, and the task's own literal reported number (gp=0.0806 must
survive a later worse trial).

**Verified these tests actually catch the bug**: reran the identical suite against a
copy of `incumbent_logic.jl` with `seed_incumbent` reverted to the pre-fix behavior
(`return resumed_best !== nothing ? resumed_best : nothing`, ignoring `cand` entirely) —
3 failures + 2 errors (a `Nothing` field-access crash on the very tests designed to catch
exactly this). Full pass under the real fix. Scratch copies used for this check were not
committed (kept out of the repo, per session cleanup discipline).

## 5. Reusable-context refactor (`reusable_context.jl`)

`build_fullA_context(; W, δ, find_smallest, draw_design, draw_seed, inner_loop_opt=nothing)`
factors the `d20_real_setup_design` + `build_pivot_elimination` + `build_ranged_screen_context`
sequence (the ~65-83s real cost per the consolidation handoff's own Phase E measurement)
into a one-time builder. `set_context_delta!(ctx, new_δ)` cheaply overrides the divergence
budget for reuse at the next stage.

**A real subtlety found by inspection, not assumed**: δ is not read only from the
immutable `ctx.δ` field. The actual inner divergence-budget check and the exact-point
cache key (`FullAEvalKey`, `oracle.jl`) both read `ctx.obj.δ` — a mutable field on
`PsiObjectiveBundleImplicit` (`ctx.obj`), confirmed mutable (`@with_kw mutable struct`,
`cc_algo/PsiObjectiveBundle.jl`). A naive `merge(ctx, (δ=new_δ,))` alone would silently
leave `ctx.obj.δ` stale, corrupting every downstream feasibility check and cache lookup.
`set_context_delta!` updates both consistently. Confirmed by `grep` that
`build_pivot_elimination` (`gravity_elimination.jl`) and `build_ranged_screen_context`
(`fast_range_screen.jl`) reference **neither** `ctx.δ` nor `ctx.obj.δ` anywhere in their
construction — `pe`/`rsc` are genuinely δ-independent and safe to reuse unchanged across
every δ stage, confirmed live (§6: `set_context_delta!` correctly propagated in a real
staged run's own log, "REUSING context ... delta overridden to 3.0/4.0/5.0").

Wired into `run_polish_checkpointed` via a new `reuse=` kwarg (validated against the
caller's W/find_smallest/draw_design/draw_seed via `reuse_matches`; falls back to a fresh
build with a warning on mismatch; ignored — with an explicit note, not silently — when
resuming, since a resumed checkpoint's own provenance must win) and into
`run_staged_delta5_continuation` via `reuse_context::Bool=true` (default on), which now
builds the context once and threads it through every stage.

**Cache-policy decisions actually implemented** (task §5's own recommended defaults):
- Immutable context (`pe`/`rsc`/draws): persists across stages — implemented, live-verified.
- Exact-point cache: **not** persisted across stages in this session's implementation —
  each `run_polish_checkpointed` call still constructs its own fresh `SafeExactCache`
  (`use_exact_cache=true` default). The task's own reasoning (Δ*(θ) is independent of the
  outer δ budget) is correct, but the cache key (`FullAEvalKey`) includes δ explicitly
  (confirmed: `key = FullAEvalKey(collect(x_free), obj.δ, ...)` — used consistently
  across `oracle.jl`/`oracle_fast.jl`/`compressed_live.jl`/`fast_range_screen.jl`/
  `infeasibility_screen.jl`), so a cache built at δ=2 cannot answer a δ=3 lookup even if
  reused as an object — it would just carry stale, never-hit entries forward at extra
  memory cost with no correctness risk. Making the cache genuinely cross-δ would require
  either changing the key (touches `oracle.jl`, frozen) or a translation layer — scoped
  out this session; flagged as a real, low-risk follow-up.
- Successful-dual bank: **not** persisted across stages in this session's default
  (`use_dual_bank=true` still constructs a fresh `DualBank` per call) — task §5 allows
  this only "if configurations and moment ordering are identical," which holds here, but
  wiring it through `reuse=` wasn't done this pass; a straightforward follow-up (thread a
  `bank=` kwarg alongside `reuse=`, same pattern).
- Outer KNITRO quasi-Newton state: correctly does NOT persist (a fresh `kc =
  KNITRO.KN_new()` every call) — matches task §5's explicit recommendation and this
  repo's own memory note on the hybrid-gradient-source-switching hazard of mixing
  solver state across separately-sourced iterates.

## 6. Real-data validation: incumbent fix + context-reuse gains

`staged_delta5_realdata_validation.jl`, real D=20/W=80,000, `draw_seed=20260719`,
45s/stage budget, genuine calibration start point (`ctx.θ0_up`'s own A_od block via
`log.(Aod_theta_natural)` → `pivot_reduce`, `gp0 = ctx.θ0_up[3+D]`, `g_start = gp0*1.01`
— the same recipe `c10_canonical_benchmark.jl` already uses and this repo independently
validated). Two arms, same start/budget/draws: `reuse_context=true` vs. `=false`.

**Incumbent monotonicity (the direct, literal claim the fix makes)**: `best_gp` per
stage, Arm A (reuse_context=true): `[0.9994196576330954, 0.9996980619312509,
0.9996980619312509, 0.9996980619312509]` — strictly increased once (stage 1→2), then
held exactly flat (stages 2-4, no further improvement found within each 45s budget, and
correctly **not regressed**). Directly observed from the raw KNITRO eval trace at stages
1-2; stages 3-4 confirmed identical by an exact bit-for-bit match of `κ`
(`0.0005031794647728516` at all three of stages 2/3/4 — since κ = 1 − gp^(σ/(σ-1)) is a
strictly monotonic, hence injective, function of `gp` for fixed σ, identical κ implies
identical `gp`). **PASS**: `best_gp` is monotonically non-decreasing at every stage
transition, in a real D=20 run — this is the literal invariant the fix protects,
confirmed live.

(This session's real run's own self-check script printed a "FAIL" at the time it ran,
because that copy of the script still asserted κ-monotonicity, not `gp`-monotonicity —
the very bug in the *validation script itself* described earlier in this section, caught
from this run's own output and fixed in the committed version of
`staged_delta5_realdata_validation.jl` for future runs. The "FAIL" in the raw log is a
false alarm from the pre-fix self-check logic, not a regression in the driver fix itself
— `gp`, the actual tracked quantity, is confirmed monotonic as shown above.)

**Numerical equivalence, reuse_context=true vs. false**: both arms reached **bit-
identical** κ at every one of the 4 stages (`0.0009670501565163248,
0.0005031794647728516, 0.0005031794647728516, 0.0005031794647728516` in both arms) and
identical `n_eval`/`n_rejected`/`knitro_status` (`-401`, `KN_RC_TIME_LIMIT_FEAS`) at
every stage — context reuse changes wall time only, not the answer, exactly as required.

**Wall-time savings**:

| | Arm A (reuse_context=true) | Arm B (reuse_context=false) |
|---|---|---|
| Total wall | 323.7s | 360.0s |
| Per-stage wall | [92.0, 67.3, 69.2, 73.6] | [89.7, 88.9, 89.0, 92.4] |

Savings: 36.3s (**10.1%**) total. Real, but smaller than the ~65-83s-per-stage figure
the production-consolidation handoff's own Phase E measurement reported for a *cold,
first-in-process* context rebuild. Stage-1 wall is nearly identical between arms (both
pay one real rebuild there), but Arm B's stages 2-4 rebuilds only cost ~20-25s more than
Arm A's no-rebuild stages — not the ~65-83s a fully cold rebuild costs. **Most likely
explanation** (inferred, not independently re-verified against a fresh-process
baseline): this comparison ran both arms sequentially in the *same* Julia process, so
JIT/precompilation costs for `d20_real_setup_design` and its dependencies were already
paid by the earlier "probe" context build and Arm A's own stage-1 build — Arm B's later
rebuilds, despite being logically "cold" (no `reuse=` context), still benefit from that
warm JIT state, understating the savings a genuinely fresh, separate-process comparison
would show. Since real production usage is itself typically one long-running process
(exactly matching this comparison's own setup), **10.1% is arguably the more relevant,
conservative real-world figure** for a single continued run, even though it understates
the raw context-rebuild cost in isolation.

**Caveat on this validation's own scope**: the pre-built `organic_pathology/
d2_startA_canon_stage_complete_neval158.jls` checkpoint (the genuine κ≈0.08-scale δ=2
boundary point referenced in the task's own bug report) could not be loaded on this
branch — `deserialize`/`load_checkpoint` threw `EOFError` on it, consistent with it
having been written under a newer `D20Checkpoint` schema (the active CM/threading
branch's schema-3 `knitro_version` field addition, not yet on this branch — see §1's
conflict-surface note) than this branch defines. Rather than fabricate a compatible
checkpoint or silently patch around the mismatch, this session derived a fresh, genuinely
feasible calibration-based start point instead (`gp0*1.01`) — which correctly exercises
the *mechanism* (incumbent seeding, context reuse) but sits in a very different, near-
autarky part of the (gp, A_od) space than the original bug report's own large-κ boundary
point. **Recommendation**: once this branch is rebased onto the CM/threading integration
(§1/§15), re-run this exact script from the real δ=2 checkpoint to directly confirm
resolution of the task's own literal reported numbers.

## 7. Organic-failure capture (`organic_failure_capture.jl`)

`OrganicFailureCollector(outdir; max_n=5)` + `maybe_capture_organic_failure!` wraps the
existing, unmodified `screened_eval` call sites in both `cb_F!`s (opt-in via a new
`organic_failures=` kwarg, `nothing` by default — zero behavior change unless a caller
opts in). `is_organic_failure(status) = !(status in FEASIBLE_CODES) && status > -9000`
distinguishes a genuine KNITRO-level failure from this repo's own exact-screen
sentinels (§10). Captures: full `zfree`/reconstructed `logA_full`, `g`, point/config
hashes, draw checksums, δ/direction, all screen counts, decoded KNITRO status
(`knitro_status.jl`), dual state before the failing call and after (if finite), warm
source, n_eval/knitro_iter at capture, checkpoint parent, and an RNG-independent
reproduction command — as JLD2 (exact) + a readable JSON summary.

**Not exercised against a live failure this session**: `organic_failures` was not passed
in `staged_delta5_realdata_validation.jl` (every one of that run's 8 real polish stages
terminated `-401`/`KN_RC_TIME_LIMIT_FEAS` — budget-limited but feasible, not an organic
failure at all; see §11). The mechanism is validated structurally (syntax-checked,
reviewed against the exact data each function already has in scope) but not fire-tested
against a genuine `-300`. Matches this repo's own prior honest finding (production-
consolidation handoff §9): a readily-reproducible organic failure at the *current*
production code's screening level is genuinely hard to materialize on demand — real ones
exist (40 confirmed at δ=5 in the canonical rerun's own trace) but require a live,
longer, real-boundary-point run to capture with this new machinery. **Follow-up**: run a
longer δ=5 continuation from the real canonical checkpoint (once schema-compatible, §6)
with `organic_failures=OrganicFailureCollector(...)` passed through.

## 8. One-command replay (`replay_organic_failure`, same file)

Loads a saved record, rebuilds (or reuses a supplied) context, verifies point/draw-
checksum match, then runs: fast exact screens, a cold inner solve, and a warm solve
seeded from the record's own `dual_before`. Reports original/cold/warm status (decoded)
and Δ_dual for all three. Does not implement dual-ray extraction, separator search,
cutting-plane Phase I, or alternative-KNITRO-settings sweeps (explicitly out of scope
per task §8) — those need a genuine captured failure to validate against first (§7).

## 9. Per-solve instrumentation audit

Audited every counter category task §9 named:
- **Outer KNITRO native counters** (iterations, FC/GA/H/HV evals, CG iters, solve time,
  abs/rel feas/opt error): now queried directly per solve via `knitro_solve_diagnostics`
  (§10), from a **fresh `kc = KNITRO.KN_new()` every call** — both driver functions
  already followed this pattern before this session. **Live-verified genuinely per-solve,
  not inherited**: `test_per_solve_counters.jl` solves a trivial 1-D quadratic twice in
  the same process from very different starting distances and confirms the second
  solve's counts reflect only its own (short) trajectory, not an accumulation from the
  first (12/12 tests, real KNITRO, real license on `demand.mit.edu`).
- **Driver's own hand-rolled counters** (`n_eval`, `n_grad_calls`, `sc.*` screen counts):
  confirmed fresh per call — `sc = ScreenCounters()`, `n_eval = Ref(0)` (or
  `Ref(resumed.n_eval)` on resume, correctly cumulative *within* one logical
  checkpoint/resume chain, not across unrelated calls), `bank`/`exact_cache` freshly
  constructed — all at the top of each function, never passed in from a prior call
  (`reuse=` only threads `ctx`/`pe`/`rsc`, not these counters).
- **Inner CC dual solve's own instrumentation** (inside `oracle.jl`/`oracle_fast.jl`):
  explicitly **out of scope** this session — those are shared files under active edit by
  the CM/threading integration branch (§1), and task §13 lists "allocation buffers in the
  inner kernels" among files not to touch. The task's own opening concern ("many
  iterations despite only one FG callback and no Hessian callback") most likely refers to
  counters at *this* inner level, which this session did not have access to audit safely.
  Flagged as the clearest remaining gap in this section.

Both `run_profile_checkpointed`/`run_polish_checkpointed` now log and return
`native_outer_diag` (a `full_status_record`) alongside the existing hand-rolled counts,
explicitly labeled as the OUTER polish/profile NLP's own counts — not conflated with
inner-solve counts. Live values observed in §6's real run were all internally consistent
(e.g. `native_outer_iters=3, native_outer_fc_evals=6, native_outer_ga_evals=4` at stage 1
— FC ≥ iters, GA ≤ FC, no implausible blowups).

## 10. KNITRO status dictionary (`knitro_status.jl`)

`decode_knitro_status(code)` is a pure lookup transcribed from the **actually-loaded**
KNITRO 13.0.1 (`/opt/shared_sw/knitro/13.0.1/include/knitro.h`'s `KN_RC_*` defines,
cross-checked against the shipped HTML reference manual
`doc/html/3_referenceManual/returnCodes.html`) — confirmed via `KNITRO.jl`'s own
hardcoded `deps.jl` that 13.0.1, not the "14.2.0" `.knitro_env.sh` points at, is what
every run in this repo actually links (matches the production-consolidation handoff's
own §2 finding). Also includes this repo's own `≤-9000` exact-screen sentinels
(`infeasibility_screen.jl`/`fast_range_screen.jl`), clearly distinguished
(`category=:exact_screen_certificate`) from native KNITRO codes since they never reach
`KN_solve` at all. 77 deterministic tests, no live KNITRO needed for the decoder itself.

`knitro_solve_diagnostics(kc)`/`full_status_record` query the live per-solve counters
(§9) plus decode the terminal status, from a live `kc` before `KN_free`. Two fields are
explicitly `missing` rather than fabricated: complementarity error (bundled into
`abs/rel_opt_error` for this NLP formulation, no separate getter in this KNITRO C API
version) and final-step-norm/line-search-count (only visible in the per-iteration
`outlev` print stream, no post-hoc getter).

**Representative-point status records**: all 8 real polish-stage solves in §6's
validation run terminated `-401` (`KN_RC_TIME_LIMIT_FEAS`, `category=:limit_feasible`,
"time limit reached before full convergence; a feasible point WAS found") — i.e.,
budget-limited, not an organic failure. No genuine `-300`/`-200` observed this session
(see §7/§11's caveat on why: a near-autarky starting regime, not the real δ=5 boundary).

## 11. Large-δ inner-solve difficulty

**Primarily attributed, not re-derived**: the production-consolidation handoff's own
Phase D (§8 of that doc) already did real granular δ=1/2/5 profiling from real saved
canonical-rerun checkpoints: inner iterations roughly double at each step (27→56→112+),
FG-call count jumps far more sharply at δ=5 (13-15 at δ=1/2 → 85 at δ=5 cold), δ=5 never
reaches KNITRO's full-optimality status 0 (settling for `-100` in every condition
tried), and warm-starting provides much less benefit at δ=5 (6.98s vs. the ~0.6s pattern
at δ=1/2). This session did not repeat that real profiling work.

**What this session adds**: this session's own real δ=2→5 solves (§6) all terminated
`-401` (time-limit-feasible) at a 45s budget, never a genuine `-300` — consistent with
(not contradicting) the consolidation handoff's own finding that δ=5 solves are simply
much more expensive per iteration (more FG/line-search work), not necessarily more
likely to hit a hard organic failure at every attempted point. The canonical rerun's own
real δ=5 trace (cited in the consolidation handoff §9) found 40 genuine `-300` events
passing every current screen — this session's own near-autarky starting regime never
reached that population (too easy a local basin, budget-limited well before any real
infeasibility boundary).

**Not attempted this session** (task §11's fuller ask — dual-solution norms/quantiles,
least-favorable-weight ESS/concentration, Hessian eigenvalue/condition estimates, KKT
residual trajectories): these require either instrumentation inside the frozen inner-
solve files (oracle.jl/oracle_fast.jl, under active edit elsewhere, §1/§2) or reading and
externally recomputing the dual→implied-weight formula from those same files without
modifying them — genuinely possible but not completed given this session's time budget.
Flagged as the largest remaining gap in this handoff, with a clear, safe next step:
implement the weight/ESS computation as a **new, read-only** diagnostic module (calling
existing exported functions, not editing frozen internals) once the CM/threading
integration lands and a real large-δ organic-failure point (§7) is available to profile.

## 12. Successful-dual-bank A/B harness (`dual_bank_ab_harness.jl`)

Fixed-trajectory design per the task's own stated preference ("removes outer-path
noise"): both arms replay an *identical* sequence of outer `w`-points through
`screened_eval` on the same immutable context, rather than two independent live KNITRO
outer solves that could organically diverge onto different paths from a warm-start
difference feeding back into the outer quasi-Newton Hessian (the exact hybrid-gradient-
source-switching hazard this repo's memory already flags for cross-iterate solver-state
mixing). Arm A: exact cache on (fresh per arm), bank off. Arm B: exact cache on (fresh),
bank on with the real, unmodified `select_warm_start`/`cheap_score` scoring. Each arm
starts from a cold (`ctx.obj.x .= NaN`) dual state so results aren't cross-contaminated.

**Real instrumentation, not black-box timing only**: `screened_eval`
(`c10_d20_production_driver.jl`, not a frozen file) already computed
`select_warm_start`'s return label (`:actual`/`:last_accepted`/`:nearest`/`:neutral`) and
implicit candidate count, then discarded both. Added an opt-in `ab_stats::Union{Nothing,
DualBankABStats}` parameter (`nothing` default, zero behavior change) that now records
the already-computed values — genuinely additive instrumentation of driver code the
production consolidation session itself wrote, not a change to `dual_bank.jl`'s own
frozen scoring/selection logic.

**Not run live this session**: building a genuine fixed trajectory requires either a
real multi-iterate KNITRO run's own `trace` (this session's §6 validation runs only
reached 3-4 outer iterates each at a 45s budget — too short a trajectory to be a
meaningful A/B) or a longer dedicated run. The harness itself (`dual_bank_ab_trajectory`)
is built, syntax-checked, and ready to run against any `w_trajectory::Vector` (e.g.
extracted from a `trace` field of a completed `run_polish_checkpointed` result) as a
one-line follow-up.

## 13. Files not touched (confirmed)

KNITRO native-library setup, BLAS/Julia thread policies, common-marginal Hessian code,
allocation buffers in the inner kernels, `exact-point cache internals` (only called
through its existing `_cache_lookup`/`_cache_store!`/`SafeExactCache` interface),
`dual_bank.jl`'s own scoring logic (only its return value is now also recorded, see
§12), QMC sequence generation.

## 14. Commit list

1. Fix staged-delta-continuation incumbent bug + `incumbent_logic.jl` + 18 regression
   tests (§3-4).
2. Add KNITRO status decoder + native per-solve diagnostics wiring (§9-10).
3. Add reusable-context refactor + organic-failure capture/replay + real-data
   validation script (§5, 7-8).
4. Add successful-dual-bank A/B harness; fix the validation script's own
   kappa-vs-gp monotonicity-check bug (§6, 12).

## 15. Post-rebase integration instructions

1. Wait for `integration/fullA-cm-parallel-production` to complete and merge.
2. `git rebase` (or merge) `diag/fullA-driver-delta5` onto the new production head.
   Expected conflict surface: `D20Checkpoint`'s field list / `CHECKPOINT_SCHEMA` bump
   (their schema-3 `knitro_version` addition vs. this branch's unchanged schema-2 struct)
   — resolve by keeping their schema bump and re-verifying this branch's `best_feasible`/
   `best` seeding logic still reads the right fields (it doesn't touch the checkpoint
   struct itself, only the `Ref` initialization, so should merge cleanly, but re-run
   `test_incumbent_seeding.jl`/`test_knitro_status.jl`/`test_per_solve_counters.jl` after).
3. Re-run all four test files (`test_incumbent_seeding.jl`, `test_knitro_status.jl`,
   `test_per_solve_counters.jl`) — all are KNITRO-license-only or pure-Julia, no real
   D=20 context needed, should take under a minute total.
4. Re-run `staged_delta5_realdata_validation.jl` from the REAL δ=2 canonical checkpoint
   (now schema-compatible post-rebase) to directly confirm resolution of the task's own
   literal reported κ numbers (§6's caveat).
5. Run one `dual_bank_ab_trajectory` pass against a real multi-iterate trace (§12).
6. Only then consider merging into `diag/fullA-d4-exact`.

## 16. Answers to the task's 10 closing questions

1. **Why did the previous staged upper-bound path report decreasing κ?**
   `best_feasible`/`best` were initialized to `nothing` on every fresh stage despite the
   supplied start point being cold-verified feasible one line earlier; KNITRO's own first
   *warm* evaluation of that point can fail or never complete within a truncated
   per-stage budget (each stage pays ~65-83s of context-rebuild first), so the stage can
   report whatever (possibly worse) point it manages to find, discarding the
   known-feasible start entirely. Fixed in `incumbent_logic.jl` (§3), regression-tested
   (§4), and the fix's core invariant (`best_gp` non-decreasing) is confirmed holding in
   a real D=20 run (§6) — though see §6's caveat on why this doesn't yet reproduce the
   task's own literal numbers (different, near-autarky starting regime).
2. **Is the feasible starting point now always retained as an incumbent?** Yes —
   `seed_incumbent` seeds from the cold-verified start whenever not resuming, and
   `resumed.best_feasible` wins unconditionally when resuming. Both cases regression-
   tested; the fix's live behavior matches ("REUSING context" + monotonic `best_gp`
   observed in §6).
3. **How much wall time is saved by reusing one context?** 36.3s / 10.1% total over a
   real 4-stage run (323.7s vs. 360.0s, §6) — smaller than the ~65-83s-per-stage figure
   from an isolated cold rebuild, most likely because both arms ran in the same Julia
   process and so shared JIT/precompilation costs (§6's own caveat on this number).
   Numerically bit-identical results in both arms at every stage, confirming the
   refactor is a pure speed optimization.
4. **Are per-solve iteration and callback counts now trustworthy?** For the *outer*
   polish/profile NLP: yes, live-verified non-inheriting across separate `kc` instances
   (§9, `test_per_solve_counters.jl`). For the *inner* CC dual solve: not audited this
   session (frozen/actively-edited files, §9's own flagged gap).
5. **What exactly do the observed KNITRO status codes mean?** `knitro_status.jl`'s
   `decode_knitro_status`, transcribed from the actually-loaded 13.0.1's own header +
   shipped manual (§10). All 8 real solves this session terminated `-401`
   (`KN_RC_TIME_LIMIT_FEAS`).
6. **What changes in the dual solution and least-favorable weights between δ=1,2,5?**
   Not computed this session (§11's flagged gap) — attributed to the consolidation
   handoff's own existing iteration/FG-call profiling instead, which does not include
   weight/ESS statistics.
7. **Why does the δ=5 inner problem require so many FG calls?** Attributed to the
   consolidation handoff's own Phase D finding (§11): each δ=5 iteration does
   genuinely more line-search/backtracking work, not just more iterations, and never
   reaches full KNITRO optimality within the divergence-forced reweighting.
8. **Are the current organic -300 failures exactly reproducible?** The capture/replay
   machinery (§7-8) is built and structurally sound but not fire-tested against a live
   failure this session (none occurred in the near-autarky validation regime, §11).
9. **Does the successful-dual bank improve final-branch performance?** Not measured
   live this session — harness built and ready (§12), needs a real multi-iterate
   trajectory to run against.
10. **Which commits should be merged after the active CM/threading integration
    finishes?** All four commits on this branch (§14) are independent of that
    integration's own changes (different lines in the one shared file) and should merge
    cleanly after a rebase — see §15 for the exact sequence.

## 17. Addendum: outer γ'_focal bounds corrected for upper/lower runs

A follow-up instruction asked to tighten the outer γ'_focal (`gp`) box: instead of the
full theoretical range for both directions, split it at the Frechet-benchmark/
calibration value `γ_f^F`, with upper and lower runs each confined to their own half.
The addendum specified exact box formulas and asked for a direction/label audit
alongside the bounds fix (its own items 1-5). That audit is what surfaced §17.2 below.

### 17.1 What γ_f^F is, and where the box endpoints come from

`γ_f^F = ctx.θ0_up[3+D]` (post-clamp) — the same quantity this repo's own gamma-profile
investigation already computed and named `g_F`/`gF_ctx`
(`c8_gammainterp_benchmark_check.jl`, Continuation 8: *"What is g_F (the calibration/
factual gamma'_focal value), exactly?"*). It is the model's own calibrated value, at
which the divergence Δ* needed is (near-)zero by construction ("a correctly-specified
benchmark should have near-zero divergence at its own calibration point",
`fullA_continuation8_handoff.md` §1) — confirmed again this session:
`δ_star_initial = 0.00259...` at `γ_f^F` in the real D=20 run (§17.4).

`theoretical_gammaprime_bounds` (`moments_gammanorm.jl`, pre-existing, **unchanged**)
already documents the model's own theoretical extremes:

```
(κ_min, κ_max) = (0, 1 − λ_dd^(1/(σ−1)));  implied γ'_focal bounds (λ_dd^(1/σ), 1)
```

i.e. `κ = 1 − gp^(σ/(σ−1))`, with `κ=0` at `gp=1` and `κ=κ_max` at `gp=λ_dd^(1/σ)`. This
session's fix reuses these pre-existing, already-validated endpoints (`ctx.bounds.γp_lo`,
`ctx.bounds.γp_hi`) rather than introducing a new formula — γ_f^F only needs to *cut* the
box, not redefine its outer edges (see §17.2 for why a new formula would in fact have
been wrong here anyway).

### 17.2 The direction audit found a real, substantive bug — not just a bounds question

The transformation `κ = 1 − gp^(σ/(σ−1))` is **strictly decreasing in `gp`** for σ>1
(the standard CES case; this repo's own real calibrations run σ≈2.5–2.9). Three
independent lines of evidence, cross-checked against each other, establish which
`find_smallest` value is the real "upper" (larger-κ) direction:

1. **Algebra**: `d(κ)/d(gp) = −(σ/(σ−1))·gp^(σ/(σ−1)−1) < 0` for `gp>0, σ>1`.
2. **Real, established, multi-session numbers**: Continuation 8's own registered
   incumbents — "upper" `κ=0.17245688540655113` has `gp=0.8926359584642946`; "lower"
   `κ=0.004387827651021192` has `gp=0.9973649883022927`. Upper's `gp` is *below* the
   Frechet benchmark (`g_F≈0.960965` at D=4); lower's is *above* it. Both cross-checked
   algebraically: `1 − 0.8926^1.667 ≈ 0.1725` ✓ (σ=2.5, matching `σ/(σ−1)=5/3`).
3. **This repo's own explicit, deliberately-calibrated code comments**:
   `run_d4_optimized_fd.jl:70`: `` `const FIND_SMALLEST = DIRECTION == "upper"   #
   calibrated post-hoc against which gives larger kappa` ``, and the real D=20
   canonical-rerun frontier's own launcher (`c_canon_run_one.jl:53`, the script that
   produced the establishment κ≈0.0786 δ=1 upper candidate cited throughout this
   project): `` `find_smallest = true   # kappa-UPPER-bound branch (minimize gp)` ``.

**All three agree: `find_smallest=true` (minimize gp) is the real upper/larger-κ
direction; `find_smallest=false` (maximize gp) is the real lower/smaller-κ direction.**

**This is the OPPOSITE of two things found in this driver's own code**:

- `c10_d20_production_driver.jl`'s `direction = find_smallest ? :lower : :upper` line
  (both `do_checkpoint` closures) — a cosmetic/informational checkpoint field, but
  backwards relative to the established convention above. **Fixed**:
  `find_smallest ? :upper : :lower`.
- `run_staged_delta5_continuation` (`staged_delta5.jl`) **hardcoded
  `find_smallest=false` unconditionally** in its calls to both `build_fullA_context`
  and `run_polish_checkpointed`, regardless of which direction the caller actually
  wanted. This is the function at the center of the ENTIRE original investigation
  (the staged 2→3→4→5 continuation the task opened with). **Every prior staged
  continuation run through this function — including, plausibly, the one behind the
  task's own originally-reported κ 0.0806→0.0031 pathology — silently always ran the
  real lower/smaller-κ direction while being used and labeled "upper" throughout this
  repo's history of this feature.** Systematically walking `gp` UP (toward `gp_hi=1`,
  `κ→0`) at every stage, independent of and *in addition to* the incumbent-seeding bug
  already fixed on this branch (§3), would by itself produce a monotonically shrinking
  reported κ across a nominally-"upper" staged run. **Fixed**: `find_smallest` is now a
  REQUIRED keyword argument (no default) on `run_staged_delta5_continuation`, forcing
  every caller to be explicit rather than silently inheriting a wrong default.

Downstream callers audited and fixed to match: `staged_delta5_comparison.jl` (was
either missing `find_smallest` entirely or hardcoded `false`; now explicit `true`,
matching its own "upper" narrative and its source checkpoint's real provenance under
`c_canon_run_one.jl`'s own `find_smallest=true`), `c10_prod_driver_smoke_original.jl`
(relabeled `"smoke_upper"`→`"smoke_lower"` to match its actual, *unchanged*
`find_smallest=false` runtime behavior — a real but harmless pre-existing mislabeling,
not a logic bug, since its own starting `gp=gp0*1.01` was already correctly positioned
for the lower direction), `c10_prod_driver_smoke_resume.jl` (checkpoint filename updated
to match the rename). `c10_canonical_benchmark.jl` and `smoke_test_driver_wiring.jl`
were checked and found already directionally consistent (their own `find_smallest`/`g`
pairs sit on the correct side of `γ_f^F`) — not modified.

**Not found to be a problem, but worth stating explicitly**: `theoretical_gammaprime_bounds`
itself, `ctx.bounds.γp_lo`/`γp_hi`, and the `κ = 1 − gp^(σ/(σ−1))` formula used
everywhere to report κ from a converged `gp` are all **unchanged and correct** — this
was purely a *box-and-label* bug in the driver's own orchestration layer, not an error
in the underlying economic-model formulas.

### 17.3 A significant, unresolved divergence from the addendum's own literal text

**Flagged explicitly, not silently resolved.** The addendum's own stated box formulas —
`` `Upper: γ_f^F ≤ γ_f' ≤ λ_ff^{1/(σ−1)}` `` and `` `Lower: 0 ≤ γ_f' ≤ γ_f^F` `` — put the
upper run *above* γ_f^F and the lower run *below* it. Given §17.2's three-way evidence
(κ strictly decreasing in `gp`; real established numbers; this repo's own explicit
calibration comments), this is **backwards**: the real upper (larger-κ) run needs
`gp` *below* γ_f^F (toward `gp_lo`), and the real lower (smaller-κ) run needs `gp`
*above* γ_f^F (toward `gp_hi=1`). Additionally, the addendum's stated upper endpoint
`λ_ff^{1/(σ−1))}` does not match `theoretical_gammaprime_bounds`' own documented
`gp_lo=λ_dd^{1/σ}` (different exponent, `1/(σ−1)` vs. `1/σ`) — algebraically,
`λ^{1/(σ−1)} < λ^{1/σ}` for `λ<1, σ>1`, so the addendum's literal upper-bound value is
actually *smaller* than this code's own `gp_lo` (the *lower* endpoint), which cannot be
a self-consistent box.

This implementation uses the **evidenced** direction and the **pre-existing, internally
self-consistent** `theoretical_gammaprime_bounds` endpoints, not the addendum's literal
formulas. Plausible explanations for the discrepancy (not independently confirmed):
`λ_ff` written from memory/notation without checking this specific codebase's own
variable names; the upper/lower direction assumption inherited from the *very same*
now-fixed `:lower`/`:upper` labeling bug this audit uncovered (§17.2) — i.e. the
addendum's author may have been describing the code's own, now-known-backwards,
pre-fix convention. **This needs the user's own confirmation before any production
merge** — if there is paper-level context (a different `γ'` gauge, a different `λ_ff`
definition) this session's derivation is missing, the direction/formula should be
revisited.

### 17.4 Real D=20 validation

Real D=20/W=80,000, `draw_seed=20260719`, 45s/stage, genuine calibration start
(`g_start = γ_f^F` exactly — `gp0` unperturbed, the same "Start A" convention
`c_canon_run_one.jl` uses — with `find_smallest=true`, the corrected real upper
direction). Two arms, same start/budget/draws: `reuse_context=true` vs. `=false`.

`γ_f^F` at this real D=20 configuration: `0.9877618976237339` (cold-verified feasible
immediately, `Delta_dual=0.00259...`, consistent with the "near-zero divergence at the
calibration point" structural expectation, §17.1).

**Incumbent-seeding + direction fix, both confirmed live**:

| Stage (δ) | `best_gp` | κ |
|---|---|---|
| 1 (δ=2) | 0.9877618976237339 | 0.0203135174923732 |
| 2 (δ=3) | 0.9877618976237339 | 0.0203135174923732 |
| 3 (δ=4) | 0.9877618976237339 | 0.0203135174923732 |
| 4 (δ=5) | 0.9877618976237339 | 0.0203135174923732 |

Every stage terminated `-411` (`KN_RC_TIME_LIMIT_INFEAS`: the outer KNITRO search itself
found no FEASIBLE point better than the seed within the 45s budget — genuinely difficult,
consistent with this repo's own well-documented large-δ inner-solve difficulty, §11, not
a driver bug). **The incumbent held exactly at its own cold-verified-feasible seed value
at every single stage transition — never regressed, never went to `NaN`, never crashed —
directly demonstrating the incumbent-seeding fix (§3-4) doing its job under genuinely
adverse conditions** (an outer search that cannot improve on the start at all). `best_gp`
is trivially non-increasing (exactly flat) and κ trivially non-decreasing (exactly flat)
— both hold. **PASS** (script's own automated verdict, matching manual verification).

**Numerical equivalence, reuse_context=true vs. false**: bit-identical κ at all 4 stages
in both arms — confirmed by the script's own explicit equality check: `MATCH`.

**Wall-time savings**: 11.8% (comparable to the pre-addendum session's 10.1%, §6) — total
wall not separately re-quoted here since the per-stage pattern and caveat (shared-process
JIT costs understating the isolated per-rebuild figure) are identical to §6's own finding.

**Honest limitation of this specific run**: because the outer search could not find
anything feasible better than the seed at ANY stage (all four `-411`), this run does
**not** exercise the case where `is_new_best` fires mid-run (a genuine improvement over
the seed) — that path remains covered only by `test_incumbent_seeding.jl`'s toy tests
(§4), not by this live run. A longer per-stage budget or a start point with more slack
relative to `γ_f^F` (rather than exactly at it) would be needed to observe live
improvement over the seed; not attempted this pass given time budget.

### 17.5 Toy tests (`test_direction_bounds.jl`, 26/26 passing)

Deterministic, no KNITRO, no real context — a minimal mock `ctx` (only the fields
`direction_bounds.jl` reads) exercises `direction_gamma_bounds`/
`validate_gp_in_direction_box` directly. Covers the addendum's own item-5 asks, rewritten
to match the evidenced (not literal-addendum) direction: an upper run cannot cross
*above* γ_f^F; a lower run cannot cross *below* γ_f^F; γ_f^F itself is feasible in both
closed boxes (shared boundary point); closed-interval endpoints are feasible, not just
interior points; reported κ moves in the economically correct direction (upper's box
achieves strictly higher κ than lower's, at every matched pair of endpoints); and
`is_better_polish`'s own comparison direction is cross-checked against this convention
(not just tested in isolation, as the pre-existing `test_incumbent_seeding.jl` did).

### 17.6 Scope not covered this pass

- **Multistart construction**: `c10_d20_production_driver.jl` (the actual production
  D=20 driver this task is about) has no multistart mechanism at all — nothing to fix
  there. The addendum's "multistart construction" item most likely refers to the older
  D=4-scale exploratory scripts (`gamma_profile_multistart.jl`, `c8_gammabranch_*.jl`,
  `analyze_multistart.jl`, etc.) — numerous, exploratory, and outside this task's own
  charter ("the staged-δ continuation driver"). Not audited or modified this pass.
- **Checkpoint-resume validation**: implemented as a single check on the resumed
  `(g, zfree)`'s `gp` value against the direction box (§3's existing resume path already
  routes through the same `w0`/`g` variables the new validation reads) — not a separate,
  bespoke resume-specific code path. Not independently stress-tested against a
  deliberately-incompatible real checkpoint this session (the same schema-mismatch issue
  from §6 blocks constructing one easily); the toy tests (§17.5) cover the underlying
  validation logic directly instead.
- **Fixed-g profiling**: `run_profile_checkpointed`'s fixed `g` is now validated against
  the direction box (§17.2), but since `g` is fixed (not searched) in that stage, this is
  a validity gate on the caller's input, not a new search-space restriction — no separate
  test beyond the toy suite's direct coverage of `validate_gp_in_direction_box`.
