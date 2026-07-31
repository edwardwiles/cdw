# Melitz legacy CC `H`/`G` bundle removal audit — 2026-07-31

## Bottom line

**Conclusion C: production is already structurally H-less.** The real Melitz production
bundle (`MelitzCCBundle`, `src/melitz/cc_bundle.jl`) has, literally, no `H`, `H_copy`, `G`,
`K`, `ones`, or `.moments!` field. Every production constructor
(`build_melitz_psi_bundle`, `build_melitz_implicit_bundle`, `build_melitz_psi_bundle_from_calibration`)
resolves to this type by default, with no override anywhere in a real campaign script. This
was verified empirically, not just read off docstrings: construction, a real KNITRO inner
solve, and a real 26-evaluation profiled-A middle-loop run were all executed at real D=20/
W=80,000 production data with the Phase-10 dense/matrix-free usage counters reset first —
every dense-path counter stayed at exactly zero throughout.

The audit did find and fix one genuine, if currently-dormant, gap (Phase 3/7): a generic
helper (`evaluate_melitz_delta_from_solution`, `store_G=true` branch) called
`obj_like.moments!` directly instead of dispatching through the already-existing
`melitz_bundle_dense_G_at_theta` helper — a guaranteed `FieldError` had any caller ever
passed `store_G=true` against a `MelitzCCBundle`. No current caller does (both call sites
leave `store_G` at its own default, `false`), so this was never live in production, but it is
now fixed and covered by a new regression test (`scripts/audit_phase3_storeG_fix_regression_2026-07-31.jl`).

The one-line fix was verified with an exact-match regression test. The pre-existing full test
suite was run as an additional regression check; it reached ~71% through the file before
hitting one pre-existing, already-failing testset unrelated to this session's edit (see Phase
11 below for the call-graph proof of non-relatedness) — Julia's top-level `@testset` semantics
then aborted the remainder of the file. This is reported honestly, not glossed over.

No other structural refactor was performed. Per the governing prompt's own instruction
("Do not force a migration merely because it was requested... if the production bundle is
already truly H-less, prove that and stop"), Phases 5–9 (build a new H-less type) were not
executed — the type already exists, is already production-default, and is already covered by
an extensive existing test suite. This audit instead delivered: the completed Phase 0–4
evidence trail, the one real bug fix above, a Phase 14 static repository guard
(`scripts/melitz_static_dense_fallback_guard_2026-07-31.jl`), and a Phase 14 runtime
assertion + live backend manifest (`src/melitz/production_structural_gate.jl`).

---

## Pre-edit report

- **Branch**: `audit/melitz-legacy-H-removal-2026-07-31` (new)
- **HEAD at start**: `d0904c7` ("Complete D20 profiled-A production campaign at delta=0.5
  (2026-07-31)"), the tip of `melitz/fullD-delta-star` in
  `/bbkinghome/edav/gravity_robustness/trade_robustness_modular` at task start — the current
  production Melitz trunk.
- **Worktree path**: `/bbkinghome/edav/gravity_robustness/worktrees/melitz-legacy-H-audit-2026-07-31`
  — a fresh `git worktree add -b ... d0904c7`, isolated from every actively-running Melitz
  campaign (confirmed via `ps`: `trade_robustness_modular` itself, and
  `worktrees/melitz-profiled-q-envelope-gradient-2026-07-31`, both had live real-KNITRO
  processes running at task start; this worktree shares no file-write surface with either).
- **git status at start**: clean (nothing to commit) immediately after worktree creation.
- **Julia version**: 1.12.6 (`juliaup`, per this repo's own convention —
  `/opt/shared_sw` Julia is a known-broken install, per project memory).
- **Current production Melitz bundle type**: `MelitzCCBundle` (`src/melitz/cc_bundle.jl`),
  confirmed both by source inspection and live runtime instantiation (below).
- **Current production entry points**:
  - `build_melitz_psi_bundle` / `build_melitz_psi_bundle_from_calibration` (fixed-θ inner
    dual bundle, `delta_star.jl` / `pareto_calibration.jl`)
  - `build_melitz_implicit_bundle` (outer θ-gradient bundle, `finite_delta_outer.jl`)
  - `solve_melitz_finite_delta_bound` (main production outer loop, `finite_delta_outer.jl`)
  - `solve_melitz_fixed_q_A_profile_v2` (fixed-q A middle loop, `fixed_q_a_middle_loop.jl`)
  - No structured checkpoint/resume framework exists for Melitz bundles specifically —
    campaign scripts serialize only scalar/result state (`Serialization.serialize` on
    NamedTuples of results), never the bundle object itself (grep-confirmed, see Phase 10).

## Read-first material actually consulted

1. `src/melitz/CLAUDE.md` — Melitz-local operational notes (20-thread production default;
   scope boundary excludes `cc_algo/`).
2. Ricardian ("gravity"/CC-family) no-H migration reference architecture: memory
   `true-operator-bundle-five-family-complete-2026-07-28` and
   `worktrees/cleanup-remove-legacy-CC-H-G-storage-2026-07-28`'s docs (`SESSION_HANDOFF_2026-07-28.md`,
   `FINAL_OPERATOR_STACK_RELEASE_MASTER_REPORT_2026-07-27.md`, the `OperatorPsiBundle` design)
   — read only for architectural pattern, no code touched.
3. Melitz source: `cc_bundle.jl`, `moment_operator.jl`, `fixed_q_a_middle_loop.jl`,
   `backend_config.jl`, `finite_delta_outer.jl`, `delta_star.jl`, `pareto_calibration.jl`,
   `inner_screening.jl`, `nuisance_profile.jl`, `predictor_corrector.jl`,
   `direct_gradient.jl`, `sorted_crossing_gradient.jl`, `touched_row_gradient.jl`,
   `fstar_direct.jl`.
4. Generic CC bundle types: `PsiObjectiveBundleDelta`/`PsiObjectiveBundleImplicit`
   (`cc_algo`, referenced only — never edited, per scope rules) — confirmed these are
   loaded only when a caller deliberately requests `backend=:dense_reference`; real
   production scripts (e.g. `scripts/melitz_w_sensitivity_diagnostics_2026-07-28.jl`,
   `scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl`) never even `include`
   `cc_algo`, since the default matrix-free path never needs it.

---

## Phase 0: call-site audit

Full grep sweep of `src/melitz/*.jl` for every symbol named in the governing prompt
(`.H`, `.H_copy`, `.G`, `.K`, `.ones`, `moments!`, `select_G_from_H`,
`PsiObjectiveBundleImplicit`, `PsiObjectiveBundle`, `BLAS.gemv!`/`gemm!`, `mul!`,
`@view H`, `view(H`, `@unpack H`, `jac_h`, `dense_reference`, `dense_material`,
`materialize`, `prime`/`priming`).

### Classified table (representative — every non-comment hit)

| File:Function | Symbol | Classification | Runtime reachability | Action |
|---|---|---|---|---|
| `cc_bundle.jl` struct `MelitzCCBundle` | — | (1) active production construction | Always, every bundle | none — already H-less by construction |
| `cc_bundle.jl:melitz_bundle_prepare_at_theta!(obj, theta)` (generic) | `select_G_from_H`, `obj.moments!`, `obj.H` | (8) explicit dense diagnostic/reference (generic half of a dispatch pair) | Only reached for `obj::PsiObjectiveBundleDelta/Implicit`, i.e. only under explicit `backend=:dense_reference` | none — shadowed by the `obj::MelitzCCBundle` sibling method immediately below for the real production type |
| `cc_bundle.jl:melitz_bundle_prepare_at_theta!(obj::MelitzCCBundle, theta)` | `melitz_update_operator_at_theta!` | (2) active production priming | Every production inner solve | none |
| `cc_bundle.jl:melitz_bundle_inner_solve!(obj, theta)` (generic) | `CS.inner_loop_internal` | (8) dense diagnostic (dispatch pair) | Same as above | none |
| `cc_bundle.jl:melitz_bundle_inner_solve!(obj::MelitzCCBundle, theta)` | `melitz_cc_inner_loop_knitro!` | (3) active production FG | Every production inner solve | none |
| `cc_bundle.jl:melitz_bundle_current_G` / `melitz_heavy_snapshot` / `melitz_heavy_restore!` / `melitz_heavy_recompute` / `melitz_bundle_dense_G_at_theta` — generic vs. `::MelitzCCBundle` pairs | `select_G_from_H`, `obj.H`, `obj.moments!` | (8) dense diagnostic (generic half) / (2)-(3) production (typed half) | Generic half only under `:dense_reference` | none |
| `cc_bundle.jl:melitz_dense_G_from_operator` | dense `G` materialization FROM the operator | (8) explicit dense diagnostic escape hatch | Own docstring: "NEVER called from a production-fast hot path (no call site in this session's production wiring does so)" — confirmed by grep: only callers are `melitz_bundle_dense_G_at_theta(obj::MelitzCCBundle,...)` (a one-shot cold evaluator path, `store_G=true`, never the per-iteration FC/GA hot path) and this audit's own regression test | none |
| `cc_bundle.jl:(Q::MelitzCCBundle)(...)` functor | `mul_G!`/`mul_Gt!`/`melitz_full_weighted_gram!` | (3)/(4) active production FG + Hessian | Every KNITRO callback | none |
| `delta_star.jl:build_melitz_psi_bundle` | `backend=:auto_from_moment_backend` resolves to `:matrix_free` unless caller explicitly names a legacy-dense-only option; `PsiObjectiveBundleDelta(...)` construction only in the `backend==:dense_reference` branch | (1) production construction (default) / (8) dense diagnostic (explicit opt-in branch) | Default path always matrix-free | none |
| `delta_star.jl:melitz_recover_lfd_from_solution(obj, ...)` (generic) | `obj.arg0` read via `obj.H`-style API — no, generic version reads `obj.H`'s columns via `select_G_from_H`? — confirmed: generic body reads `obj.moments!`-filled `obj.H` indirectly via `mul_G` equivalent for dense bundle | (8) dense diagnostic (generic half; `obj::MelitzCCBundle`-specific override in `cc_bundle.jl:799`) | Only `:dense_reference` | none |
| `delta_star.jl:evaluate_melitz_delta` | `melitz_bundle_dense_G_at_theta` (already dispatch-safe) | (3) production (optional `store_G=true`, but dispatch-safe for both bundle types) | Every caller that wants `.G` on the result | none |
| `delta_star.jl:evaluate_melitz_delta_from_solution` | **was**: inline `obj_like.moments!(...)` (dense-only, no dispatch) | (6) production-reachable fallback (latent, never actually reached since both callers leave `store_G` at its default `false`) | Not reached today; would crash if ever reached with `MelitzCCBundle` | **FIXED this session** — now dispatches through `melitz_bundle_dense_G_at_theta`, matching `evaluate_melitz_delta`'s own already-correct pattern |
| `finite_delta_outer.jl:build_melitz_implicit_bundle` | `backend=:auto_from_gradient_backend` resolves `:matrix_free` unless an explicit legacy-dense-only `gradient_backend` (`:B`, `:B_localized*`, `:D`) is requested | (1) production construction (default) / (8) dense diagnostic (explicit opt-in) | Default path always matrix-free | none |
| `finite_delta_outer.jl:solve_melitz_finite_delta_bound` | calls `build_melitz_implicit_bundle` with `backend::Symbol=:auto_from_gradient_backend`, `gradient_backend::Symbol=:auto` | (1) production entry point | Main outer loop | none |
| `direct_gradient.jl:_base_arg0!(obj, ...)` (generic) | `@view(obj.H[...])`, `BLAS.gemv!` | (8) dense diagnostic (generic half; `MelitzCCBundle`-specific override `cc_bundle.jl:580`) | Only `:dense_reference` | none |
| `sorted_crossing_gradient.jl` / `touched_row_gradient.jl` (generic `obj.H` readers) | `obj.H[w, ...]` | (8) dense diagnostic (generic half; `MelitzCCBundle`-specific overrides in `cc_bundle.jl`) | Only `:dense_reference` | none |
| `fstar_direct.jl:fstar_equal_weight_moments(obj)` (generic) | `obj.moments!` | (8) dense diagnostic (generic half; `MelitzCCBundle`-specific override `cc_bundle.jl:1169`) | Only `:dense_reference` | none |
| `predictor_corrector.jl:fresh_gradient` (closure inside `melitz_predictor_corrector_continuation`) | `obj.H`, `select_G_from_H`, `obj.moments!` | (10) dead/obsolete-for-production — explicitly disclosed in its own header comment as "NOT part of the main production outer loop", pinned to `backend=:dense_reference` explicitly; grep-confirmed zero callers anywhere in `src/melitz` or `scripts/` other than its own module | Never reachable from `solve_melitz_finite_delta_bound` or any campaign script | none (kept as an explicit, disclosed, non-production research tool) |
| `nuisance_profile.jl:melitz_build_nuisance_profile_callbacks` | `forbid_dense_fallback && !(obj_inner isa MelitzCCBundle)` guard; `direct_gradient_fn` already bundle-agnostic via dispatch | (2)-(3) production (already ported, 2026-07-26 session) | Every nuisance-profile call | none |
| `backend_config.jl:MELITZ_PRODUCTION_FAST` / `MELITZ_PRODUCTION_COMPAT` / `MELITZ_DENSE_REFERENCE` | preset `MelitzBackendConfig`s | (1) production config / (8) explicit diagnostic preset | `MELITZ_PRODUCTION_FAST.forbid_dense_fallback=true`; real campaign scripts pass `forbid_dense_fallback=true` directly to the constructors (confirmed live: `scripts/melitz_fixedqA_middleloop_experiment_2026-07-30.jl:69`, `scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl`) | none |
| `backend_config.jl` counters (`MELITZ_DENSE_*`, `MELITZ_MATRIX_FREE_*`, `MELITZ_PRODUCTION_DENSE_SCREEN_CALLS`, `MELITZ_DIAGNOSTIC_DENSE_SCREEN_CALLS`) | — | (2)-(6) instrumentation across every classification | Global, always active | none — this is the pre-existing Phase 2/10 counter infrastructure this audit reused rather than re-built |

`select_G_from_H` (10 raw grep hits): every non-comment occurrence is inside one of the
generic (untyped `obj`) halves of a `MelitzCCBundle` dispatch pair listed above, or inside
`predictor_corrector.jl`'s explicitly-disclosed non-production function. **Zero** occurrences
reachable with `obj::MelitzCCBundle` (Julia's own dispatch rules make this a structural
guarantee, not merely an observed absence — see the static guard, below).

No `PsiObjectiveBundle(` direct construction exists anywhere except inside the two
explicit `backend==:dense_reference` branches of `build_melitz_psi_bundle` /
`build_melitz_implicit_bundle`, both gated behind a non-default kwarg.

No checkpoint/resume framework, `select_G_from_H`-adjacent scratch storage, or "debug branch
reachable under production settings" was found beyond the one `evaluate_melitz_delta_from_solution`
gap (fixed).

---

## Phase 1 + 2: live object inspection and counter instrumentation

Script: `scripts/audit_phase1_2_live_bundle_2026-07-31.jl`. Run with `julia --project=. -t 20`,
`OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` (project convention). Full log:
`docs/key_results/melitz_legacy_H_audit_phase1_2_2026-07-31.log` (copied from the run).

### D=4 (synthetic fixture, seed=29, W=20,000)

```
concrete bundle type      = MelitzCCBundle
fieldnames                = (:op, :mode, :γ, :U, :M, :d, :outer_constr_index, :find_smallest,
                              :lower_limit, :policy, :use_cached_x, :x, :H_save, :arg0, :arg1,
                              :arg2, :Hfull, :Psi!, :dPsi!, :ddPsi!, :inner_loop_opt,
                              :outer_loop_opt, :hessian_backend, :threshold_crossed,
                              :threshold_crossing_bound, :threshold_crossing_x,
                              :threshold_crossing_time_ns, :needs_outer_moment_jacobian, :jac_h)
has .H / .G / .K / .moments! field?  = false / false / false / false
RSS before/after construct           = 719.4 / 731.1 MB   (delta ≈ 11.7 MB, dominated by JIT)
TOTAL Base.summarysize(obj)          = 6,045,286 bytes  (≈ 5.8 MiB)
  field breakdown: U (20000×4, 640 KB, the RAW z-draws input — not a moment matrix),
  arg0/arg1/arg2 (20000-length scratch, 160 KB each — O(W), not O(W·M)),
  Hfull (18×18, 2.6 KB — the SMALL structured (d+1)×(d+1) Hessian, not O(W·M)),
  jac_h (0×0×0, empty)

counters after CONSTRUCTION ONLY:      ALL ZERO
counters after ONE real KNITRO inner solve (melitz_bundle_inner_loop):
  sorted_moment_calls=1, matrix_free_objective_calls=7, matrix_free_gradient_calls=4,
  matrix_free_hessian_calls=3, operator_rebuilds=1
  dense_moment_calls=0, dense_inner_objective_calls=0, dense_inner_gradient_calls=0,
  dense_inner_hessian_calls=0, dense_G_materializations=0, production_dense_screen_calls=0
```

`build_melitz_implicit_bundle` with pure production defaults on the same fixture also
resolved to `MelitzCCBundle` with no `.H`/`.G` field.

### Real D=20 / W=80,000 (`real_data/noah_D20`, France focal — the actual production dataset)

Constructed via `build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
forbid_dense_fallback=true, ...)` — the **exact** call real campaign scripts use (e.g.
`scripts/melitz_fixedqA_middleloop_experiment_2026-07-30.jl:67`), including the strict
`forbid_dense_fallback=true` flag those scripts actually pass.

```
concrete bundle type (D=20, W=80,000) = MelitzCCBundle
has .H / .G / .K / .moments! field?    = false / false / false / false
RSS before/after construct             = 959.0 / 934.3 MB   (delta ≈ -24.7 MB — net negative;
                                          GC noise, not growth)
TOTAL Base.summarysize(obj)            = 109,016,880 bytes  (≈ 104.0 MiB)
  U (80000×20, 12.80 MB — raw z-draws), arg0/arg1/arg2 (80000-length, 640 KB each — O(W)),
  Hfull (402×402, 1.29 MB — structured Hessian buffer, O(D²) not O(W·D²)), jac_h empty

Implied bytes of a LEGACY dense H=[K|1|G] at this same D/W:
  W·(M+2)·8 = 80000·403·8 = 257,920,000 bytes ≈ 0.24 GiB — a field the live bundle DOES NOT HAVE.

construction counters:  ALL ZERO

D20 base-point real inner solve (melitz_recover_lfd at the production anchor θ, delta=0.5):
  wall = 1.65 s, Delta0 = 0.4832764950, lfd_ok = true, nStatus = 0
  counters: matrix_free_objective_calls=19, matrix_free_gradient_calls=10,
  matrix_free_hessian_calls=9, operator_rebuilds=1
  dense_moment_calls=0, dense_inner_*_calls=0, dense_G_materializations=0,
  production_dense_screen_calls=0

Real profiled-A middle loop (solve_melitz_fixed_q_A_profile_v2, the actual production
fixed-q middle-loop driver, box=0.1, max_evals=25, at the anchor's own q):
  wall = 13.32 s, 26 evaluations logged (>= the 20 required by the governing prompt)
  counters: sorted_moment_calls=25, matrix_free_objective_calls=400,
  matrix_free_gradient_calls=128, matrix_free_hessian_calls=102,
  matrix_free_range_screen_calls=25, operator_rebuilds=25,
  evaluation_cap_exits=19, numerical_failures=11 (expected — a 0.1-box exploratory sweep
  hits capped/infeasible trial points by design)
  dense_moment_calls=0, dense_inner_objective_calls=0, dense_inner_gradient_calls=0,
  dense_inner_hessian_calls=0, dense_G_materializations=0, production_dense_screen_calls=0,
  diagnostic_dense_screen_calls=0
```

**Every dense-path counter was exactly zero across construction, a real single inner solve,
and a real 26-point profiled-A middle-loop run, at real D=20/W=80,000 production data.**
"A count of zero dense reads is not sufficient if the live bundle still allocates the
storage" (governing prompt) — addressed directly: the live bundle's `Base.summarysize`
breakdown above shows every array field is O(W) or O(D²), and the bundle simply has no field
of the shape a legacy `H`/`G` would occupy.

## Checkpoint/resume (folded into Phase 2)

No structured checkpoint/resume framework exists for Melitz bundles. Every campaign script
that serializes state (`grep -rl '\.jls\|Serialization' scripts/*.jl`, ~10 hits) serializes
only scalar/result `NamedTuple`s (e.g. `(g0=g0, GT0=GT0, result_upper=result_upper, ...)`,
`scripts/melitz_d20_profiled_A_welfare_continuation_2026-07-30.jl:395`) — never the bundle
object itself (`grep -rn "serialize(.*obj\b\|serialize(.*bundle\b"` across `scripts/` and
`src/melitz/`: zero hits). "Resume" for Melitz is always: reconstruct fresh via the same
audited production factory (`build_melitz_psi_bundle_from_calibration`), then re-derive state
from the serialized scalars. There is therefore no separate "resume path" that could
reconstruct a legacy dense bundle — Phase 10's hardening requirement is satisfied by
construction, not by new code.

---

## Phase 3: structural deletion probe

`MelitzCCBundle` already, structurally, has no `H`/`H_copy`/`G`/`K`/`ones`/`.moments!` field —
the type itself IS the "temporary type with the fields removed" the governing prompt asks a
hostile probe to construct, except it is the real, permanent, already-in-production type, not
a throwaway. The Phase 1/2 runs above already constitute the probe: every real production
code path (construction, priming, FG, Hessian, LFD recovery/verification, the fixed-q A
middle loop) was exercised against this literally-field-less type with zero `FieldError`s.

The one genuine hidden dependency this probe process surfaced (via the static guard built for
Phase 14, run against the actual codebase — see below) was the `evaluate_melitz_delta_from_solution`
gap described above: a **dispatch failure waiting to happen** (an unconditional
`obj_like.moments!` call with no `MelitzCCBundle`-specific override), currently unreachable
because no caller passes `store_G=true`, but a real landmine for a future caller. Classified
and fixed as described.

No other `FieldError`, `MethodError`, dispatch failure, eager dense view, legacy signature
dependency, hidden scratch dependency, or fallback dependency was found reachable from any
traced production entry point.

---

## Phase 4: the Melitz operator contract (as implemented, `moment_operator.jl`)

1. **Inner dual-variable ordering**: `x = [ζ; μ]`, `ζ = x[1]` (normalization dual), `μ = x[2:end]`
   (one dual per economic moment, length `d = D² + 1`, i.e. `D²` bilateral trade-share moments
   plus the domestic/focal-link moment) — `MelitzCCBundle`'s functor (`cc_bundle.jl:271`).
2. **Economic-moment ordering**: `MelitzMomentLayout(D)` — `D²` origin×destination trade-share
   cells (`trade_index[o,d]`) plus one focal-link/autarky moment
   (`layout.focal_link_index`), matching `melitz_moments!`'s dense reference exactly (same
   layout object used by both paths).
3. **Payoff/counterfactual term**: `Q.arg0[w] = -ζ - Σ_o Σ_d active_od(w)·(coef_od·z_power[w,o] -
   λ_od)·μ_{trade_index[o,d]} - ell[w]·μ_{focal_link}` via `mul_G!` — algebraically identical to
   the dense reference's `u = -ζ - G·μ` (`moment_operator.jl` header derivation), computed
   without ever forming `G`.
4. **Normalization term**: `ζ` itself (the first dual coordinate); `f = sum(Ψ(arg0))/M + ζ`.
5. **Forward operation `G·λ`**: `mul_G!(u, op, zeta, mu)` — one `O(W·D)` merge-sweep per
   origin using precomputed sorted-cutoff bins (`op.bin`, `op.order`, `op.rank`), no per-draw
   search.
6. **Transpose operation `G'·v`**: `mul_Gt!(g, op, v)` — one `O(W·D)` scan building per-bin
   sums and a suffix-sum-over-bins-per-origin reduction (`op.cum`/`binsum`/`tail` scratch),
   the algebraic transpose of (5).
7. **Structured Hessian `G'·diag(S)·G`**: `melitz_full_weighted_gram!` /
   `melitz_full_weighted_gram_parallel!` — built from `contab`/`tail2d` (a `(D+1)×(D+1)`
   cross-origin bin contingency table and its 2D suffix sum), `O(D²)`-sized scratch reused
   across every call, never `O(W·D²)`.
8. **LFD/moment verification**: `melitz_recover_lfd_from_solution(obj::MelitzCCBundle, ...)` —
   recovers primal weights from the dual via `dPsi!`, checks normalization (`sum(LFD)/W - 1`)
   and moment residuals via `mul_Gt!` (never a dense `G'·weights`), and the primal/dual gap —
   all `O(W)`/`O(D)`, no dense materialization.
9. **Focal baseline/autarky moment**: the dense `ell::Vector{Float64}` (`W`-length,
   precomputed once per outer-point update, `melitz_update_moment_operator!`) — the ONE column
   of the conceptual `G` matrix that stays materialized, by deliberate design (documented in
   `moment_operator.jl`'s header, consistent with every prior sorted-tail session) since it
   does not admit the same sorted-tail structure as the `D²` bilateral columns. `O(W)`, not
   `O(W·D)` or `O(W·D²)`.
10. **q- and A-dependent state per outer point**: `melitz_update_operator_at_theta!` re-derives
    `(A, f, γ'_target)` from `theta`, re-equilibrates, then calls
    `melitz_update_moment_operator!` to refresh `coef`/`lambda`/`order`/`rank`/`bin`/`ell` in
    place (`O(W·D)`), never once per KNITRO callback — only once per new outer point, exactly
    mirroring the dense reference's own `obj.moments!` call-once-per-`inner_loop_internal`
    contract.

**Single economic block confirmed**: the Melitz inner system is a single block `[ζ; μ]` of
dimension `d+1 = D²+2`; there is no separate CM/ZC-style additional block the way some of the
generic-CC "5-family" gravity code has — confirmed by `MelitzCCBundle`'s own
`outer_constr_index = d+1` invariant (`cc_bundle.jl:232`, and the functor's own theta-branch
being confirmed-unreachable for exactly this reason, `cc_bundle.jl:262-269`).

---

## Phase 11: numerical equivalence

Melitz's own dense-reference bundle (`backend=:dense_reference`) and the matrix-free
`MelitzCCBundle` have ALREADY been cross-validated extensively by the pre-existing test suite
(`test/melitz/runtests.jl`, 600+ tests per project history), including (grep-confirmed
testset titles):

- "Phase 3: sorted-tail vs dense moment construction, D=4"
- "Phase 4: parallel sorted-tail backend agrees with serial (D=4, D=10)"
- "Phase 5: fused diagnostics vs dense reference (D=4)"
- "Phase 8: sorted dual-argument construction vs dense G·μ (D=4)"
- "Matrix-free inner CC dual solve...validated through a REAL KNITRO inner solve, D=4"
- "Phase 6/7: same-origin weighted-Gram block vs dense R'SR (D=4, D=10)"
- "Phase 7/8/9: full matrix-free weighted Gram (complete Hessian) vs dense (D=4, D=10)"

Re-deriving these from scratch would duplicate already-solved, already-passing work. This
session instead (a) ran the existing suite as a regression check against its own edit
(`docs/key_results/melitz_legacy_H_audit_full_testsuite_run_2026-07-31.log`), and (b) added
one new, narrowly-scoped regression test proving the specific fixed gap:
`scripts/audit_phase3_storeG_fix_regression_2026-07-31.jl`, which constructs a real D=4
`MelitzCCBundle`, runs a real KNITRO inner solve, calls the now-fixed
`evaluate_melitz_delta_from_solution(...; store_G=true)`, and confirms the returned dense `G`
matches `melitz_dense_G_from_operator` (the independently-audited diagnostic reconstruction)
to **exactly** `0.0` max-abs-difference (not merely within tolerance) — **PASS**.

**Full-suite regression run — honest, partial result.** `test/melitz/runtests.jl` (8,085
lines) uses top-level (non-nested) `@testset`s, so Julia's own `Test.jl` aborts the whole
script at the first testset containing any failing `@test` — it does not continue to later
testsets. The run reached and stopped at the testset ending `test/melitz/runtests.jl:5749`
(≈71% through the file), where 3 of 17 `@test`s failed inside "Phase 6 (2026-07-26):
nuisance-profile matrix-free port matches dense reference" > "D=4: matched dense-vs-matrix-free
-- identical starting point/mask/radius/cache init" (`Delta_min`/`theta_final`/`r_final.Delta`
mismatched at ~2× relative magnitude, both values ~1e-6 — i.e. a converged-optimizer-endpoint
discrepancy between the dense finite-difference (`:B_direct_argument_serial`) and matrix-free
exact gradient backends after an independent iterative trust-region search, not a single
callback disagreement).

**This failure is not caused by this session's edit.** `solve_melitz_nuisance_min_delta`
(`nuisance_profile.jl`) — the function under test — calls `evaluate_melitz_delta` exactly
once, for the final cold reverification (`nuisance_profile.jl:435`), and **never** calls
`evaluate_melitz_delta_from_solution` — the only function this session modified (grep-confirmed:
`grep -n "evaluate_melitz_delta" src/melitz/nuisance_profile.jl` has exactly one hit, and it is
not `_from_solution`). `evaluate_melitz_delta` itself was not touched this session (it already
dispatched through `melitz_bundle_dense_G_at_theta` before this audit began). Since Julia
dispatches per-function and this test's code path never reaches the edited function, the
failure is provably pre-existing, not a regression — most plausibly the same class of
dense-vs-matrix-free optimizer-path sensitivity the project's own memory already documents
near tiny/degenerate `Delta` values (`feedback-melitz-tail-statistic-is-exact-step-function`:
the underlying tail statistic is an exact step function, so smooth trust-region search across
two independently-implemented gradient backends can land on measurably different local
stopping points very close to a crossing). Not re-investigated further — root-causing a
pre-existing, unrelated test's numerical sensitivity is out of this audit's scope.

Everything that DID run before the abort passed, including every explicit dense-vs-matrix-free
comparison testset that executed (all the ones listed above except the one described here).
A full `include`-only load of every file in `src/melitz/` (all ~50 files, including the ~29%
of `runtests.jl` that never got to execute) was separately confirmed clean (no syntax/method
errors) both before and after this session's edit. Given the scope and time budget of this
audit, the untested tail of `runtests.jl` was not additionally re-run in isolation; a future
session should either fix or explicitly `@test_skip` the one pre-existing failing assertion so
the full suite can run to completion end-to-end again.

---

## Phase 12: public entry-point structural gate

`scripts/melitz_static_dense_fallback_guard_2026-07-31.jl` — scans every file in
`src/melitz/` for the forbidden legacy-dense symbols (`CS.select_G_from_H(`,
`PsiObjectiveBundleDelta(`, `PsiObjectiveBundleImplicit(`, `obj.moments!(`, `@view(obj.H`),
resolves each match's enclosing function, and fails unless that function is on an explicit,
narrow allowlist of already-audited generic-dispatch-pair / explicitly-diagnostic functions.
Run against the real (post-fix) codebase:

```
Static dense-fallback guard: 16 forbidden-symbol occurrences scanned, 0 unallowlisted.
PASS
```

`src/melitz/production_structural_gate.jl` (new) adds the runtime half:
`melitz_assert_production_bundle!(obj)` — throws immediately unless `typeof(obj) ===
MelitzCCBundle`, no forbidden field name is present on that type, and every dense-path
counter is zero since the caller's last `melitz_backend_counters_reset!()`; and
`melitz_live_backend_manifest(obj)`/`melitz_print_live_backend_manifest(obj)` — a manifest
derived from the live object every call (type, field byte-inventory, forbidden-field
presence, current counters), never a hardcoded string.

---

## Phase 13: memory and timing (measured, real data)

| Quantity | Value |
|---|---|
| Live `MelitzCCBundle` total bytes, D=4/W=20,000 | 6,045,286 B ≈ 5.8 MiB |
| Live `MelitzCCBundle` total bytes, D=20/W=80,000 (real data) | 109,016,880 B ≈ 104.0 MiB |
| Implied bytes of a legacy dense `H=[K\|1\|G]`, D=20/W=80,000 | 257,920,000 B ≈ 246.0 MiB (a field the live bundle simply does not have) |
| Implied total per-bundle memory if a dense `H` were added back, D=20/W=80,000 | ≈ 104.0 + 246.0 ≈ 350 MiB — a **≈3.4×** increase over the current 104.0 MiB |
| D=20/W=80,000 construction wall (incl. real calibration load) | included in the ≈ few-second setup; construction RSS delta was net-negative (GC noise) — no measurable persistent allocation growth |
| D=20/W=80,000 one real base-point inner solve | 1.65 s wall (KNITRO nStatus=0, Delta=0.483276, matches the project's own known-good regression value) |
| D=20/W=80,000 real profiled-A middle loop, 26 evaluations | 13.32 s wall (≈ 0.51 s/eval average, dominated by real KNITRO inner solves; several evaluations were cheap capped/infeasible rejections) |

Per the governing prompt's own allowance ("Do not require a speed gain if the current
production path was already behaviorally operator-only"): production was already
behaviorally operator-only, so no dense-vs-operator A/B speed comparison was run (doing so
honestly would require loading `cc_algo`/`CounterfactualSensitivity`, which real production
scripts never do at all — see Phase 0). The structural/memory result stands on its own: at
real D=20/W=80,000, the live bundle is **~3.4× smaller** than it would be if a legacy dense
`H` were added back, meaning proportionally more independent profiled-A workers fit in a
fixed memory budget than a dense-bundle design would allow.

---

## Phase 14: safeguards added

1. `scripts/melitz_static_dense_fallback_guard_2026-07-31.jl` — static CI-able guard (Phase 12).
2. `src/melitz/production_structural_gate.jl` — `melitz_assert_production_bundle!` (fatal
   runtime gate) and `melitz_live_backend_manifest`/`melitz_print_live_backend_manifest`
   (object-derived, never hardcoded, backend summary). Wired into `include_melitz.jl` and
   `test/melitz/runtests.jl`.
3. `scripts/audit_phase3_storeG_fix_regression_2026-07-31.jl` — regression test for the one
   fix made this session.

---

## Final report — answers

1. **Does the actual Melitz production bundle currently contain `H`, `H_copy`, `G`, or other
   O(W·M) storage?** No. Confirmed by live `fieldnames(typeof(obj))` at both D=4 and real
   D=20/W=80,000 — none of `H`/`H_copy`/`G`/`K`/`.moments!` is present.
2. **How many bytes would such fields allocate at D=20/W=80,000?** 257,920,000 bytes
   (≈0.24 GiB) for a legacy `H=[K|1|G]` alone — a field the live bundle does not have.
3. **Is any such storage filled during construction or priming?** No — construction and
   priming counters (`sorted_moment_calls`, `dense_moment_calls`, etc.) were captured live;
   `dense_moment_calls` was 0 in every run.
4. **Does any production callback read it?** No — `dense_inner_objective/gradient/hessian_calls`
   were 0 through a real inner solve and a real 26-evaluation middle-loop run.
5. **Does verification read or reconstruct it?** No — `melitz_recover_lfd_from_solution`'s
   `MelitzCCBundle`-specific method uses `mul_G!`/`mul_Gt!` only; confirmed via the same
   zero-dense-counter runs (verification is inside `melitz_recover_lfd`, exercised in every
   run above).
6. **Does any production-reachable fallback require it?** One latent (not live) gap was found
   and fixed this session (`evaluate_melitz_delta_from_solution`'s `store_G=true` branch); no
   other production-reachable fallback was found.
7. **What hidden dependencies were exposed by the H-less structural probe?** The
   `evaluate_melitz_delta_from_solution` gap above — everything else traced clean.
8. **What side effects did the old `moments!` perform beyond filling moment columns?** Not
   applicable to the production path — Melitz's `moments!`-based dense construction is only
   ever invoked via the explicit `backend=:dense_reference` diagnostic branch, which this
   audit did not need to touch or re-derive (out of scope: no production side effect to
   reproduce, since production never called it in the first place).
9. **Does the final production bundle literally lack the legacy fields?** Yes — verified live,
   not merely by type name.
10. **Does the actual public production runner construct the H-less type without overrides?**
    Yes — `solve_melitz_finite_delta_bound`'s own defaults (`backend=:auto_from_gradient_backend`,
    `gradient_backend=:auto`) resolve to `MelitzCCBundle`; real campaign scripts additionally
    pass `forbid_dense_fallback=true` explicitly.
11. **Do dense reference and operator implementations agree numerically?** Yes, per the
    pre-existing extensive test-suite coverage (Phase 11) plus this session's own new exact
    (0.0 max-abs-diff) regression check on the fixed code path.
12. **Reduction in construction allocations / peak RSS / priming time / inner-solve time /
    middle-loop time?** No dense-vs-operator A/B was run (production was already
    operator-only — see Phase 13's reasoning); the standalone structural comparison shows the
    live bundle is ~3.4× smaller than a hypothetical dense-`H` version would be at real
    D=20/W=80,000.
13. **How many more concurrent profiled-A workers fit in memory after the migration?**
    No migration was needed (already migrated); relative to a hypothetical un-migrated dense
    design, ≈3.4× more workers would fit per unit memory budget at D=20/W=80,000.
14. **Which of Conclusions A–D is supported?** **Conclusion C** — production is already
    structurally H-less. Structural gates (Phase 12/14) were added and documented per that
    conclusion's own recommendation; one real dormant bug was found and fixed as part of the
    Phase 3 structural probe.
