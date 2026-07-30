# Dense-reference reachability audit (2026-07-30)

Method: `rg` for `moment_representation`, `dense_reference`, `PsiObjectiveBundleImplicit`,
`OperatorPsiBundle`, `select_G_from_H`, `MOMENT_REPRESENTATION`, `bundle_type=`,
`LIVE.*STASH` across `full_aod_diag/d4_exact/` (the only tree with real production code — `cc_algo/`
and `legacy/` are pre-restructure code the production families no longer call), then for every file
that references `moment_representation`/`dense_reference` with an actual value (not just a comment),
a Python AST-free brace-matching pass extracted the exact call to determine which of the 3 real
driver functions, if any, receives the argument directly. Full raw lists are reproducible via the
commands in the task brief; this document classifies by pattern, not a manual per-file read of all
~150 matches (that volume is the reachability problem itself — see Finding 1 below).

## Finding 1 (the headline finding): only 4 files pass `moment_representation` to a real driver

Exhaustively checked (brace-matched, not regex-adjacent-line-guessed) against the 3 real entry
points `run_cm_upper_checkpointed`, `run_originzc_upper_checkpointed`,
`run_polish_checkpointed_unified`:

| File | Classification |
|---|---|
| `cm_checkpoint.jl` (defines `run_cm_upper_checkpointed`) | production code — the kwarg's own definition/threading site |
| `cm_originzc_checkpoint.jl` (defines `run_originzc_upper_checkpointed`) | production code — same |
| `c10_d20_production_driver_unified.jl` (defines `run_polish_checkpointed_unified`) | production code — same |
| `test_unrestricted_operator_ctx_driver_wiring_2026-07-29.jl` | test only — explicitly exercises both values against the driver to prove the postmortem's first incident is fixed |
| `test_meanzc_originzc_driver_wiring_2026-07-29.jl` | test only — same, for the second incident |

**No campaign runner, standalone runner, or checkpoint-resume script anywhere in the repository
passes `moment_representation` to any of the 3 real drivers.** Every real production invocation
today resolves through the drivers' own defaults. This is good news about *today's* state and
exactly the postmortem's own point: it was true before both incidents too, and the switch still
existing at all is what let each family's default silently regress independently, undetected,
twice. The fix in this task (§2/§4) is to remove the kwarg's existence at the driver level, not
merely confirm no one currently passes it.

## Finding 2: the ~150 broader matches are overwhelmingly diagnostic/test/dead, by construction

The remaining ~145 files matching the broad patterns split as follows:

- **Builder-level `moment_representation` kwarg definitions** (5 files: `cm_production_bundle.jl`,
  `cm_frechet_level.jl`, `cm_meanzc_production.jl`, `cm_originzc_production.jl`,
  `compressed_live.jl`) — **accidentally-production-reachable in the structural sense** (nothing
  stops a future driver edit from passing `:dense_reference` explicitly, or a future call site
  bypassing the driver and calling the builder directly with production-shaped arguments) even
  though no current call site does either. This is the class this task's type-safety layer (§3) and
  shared factory (§4) close off directly.
- **Equivalence-gate test scripts** (`test_operator_no_H_bundle_equivalence_{cmzc,originzc,flexcm,frechet,unrestricted}[_d20].jl`,
  10 files) — **test only**. Each calls a builder directly with both `moment_representation` values
  explicitly, side by side, never through a real driver. Correctly scoped already; retained per
  task §11, but their claim (`REFERENCE_EQUIVALENCE`) must not be conflated with
  `PRODUCTION_DEFAULT_PATH` (the postmortem's own root-cause finding #2 — this doc's Finding 1
  addresses exactly that conflation).
- **`no_dense_g_counters.jl`, `production_backend_manifest.jl`** — **diagnostic infrastructure**,
  shared. Tracks a genuinely different axis (does an FG/Hessian callback ever materialize a dense
  matrix) from the bundle-struct-type axis this task is about (postmortem §3) — retained unchanged,
  extended by this task's manifest work (§7) rather than replaced.
- **Benchmark/profiling scripts** (`bench_*`, `profile_*`, `c8_*`/`c9_*`/`c10_*`/`c13_*`/`c14_*`
  numbered investigation scripts, ~50 files) — **diagnostic/reference only**, most calling a builder
  directly or an evaluate_fullA_screened*-family function with the *unrelated*
  `moment_representation=:compressed`/`:dense` kwarg (see Finding 3) rather than the
  `:operator`/`:dense_reference` axis this task governs.
- **Winner-bin/Hessian-architecture wiring tests** (`test_verification_backend_default_*.jl`,
  `test_operator_bundle_allocation_proof_flexcm.jl`, `test_backend_manifest_*.jl`, ~15 files) —
  **test only**.
- **`cc_algo/`, `legacy/`, `full_aod_diag/*.jl` (non-`d4_exact`)** — **dead code** relative to the 3
  real production drivers audited above; these predate the restricted-family/operator-bundle
  architecture and are not on any path `run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`/
  `run_polish_checkpointed_unified` reaches. `PsiObjectiveBundleImplicit`'s own struct definition
  (`cc_algo/PsiObjectiveBundle.jl:222`) lives here — genuinely still needed (it's the dense
  reference type), not itself dead, but the file's ~90 other matches for `select_G_from_H`/
  `PsiObjectiveBundleImplicit` in `cc_algo/`/`legacy/` are dead relative to the audited drivers.

## Finding 3: a naming collision that must not be conflated with this task's axis

`evaluate_fullA_screened`, `evaluate_fullA_screened_ranged`, and `evaluate_fullA_fast`
(`compressed_live.jl`, `fast_range_screen.jl`, and ~25 caller scripts) accept a **different**
kwarg that is also spelled `moment_representation`, valued `:compressed`/`:dense` — this governs
whether the *unrestricted* family's screening evaluator reads compressed vs. dense economic
moments, an orthogonal axis to `OperatorPsiBundle` vs. `PsiObjectiveBundleImplicit`. Roughly 15 of
the ~70 raw `moment_representation` matches in the initial grep are this unrelated kwarg. None of
this task's changes touch these call sites; flagged here explicitly so a future reader of the raw
`rg` output does not mistake one axis for the other (an easy mistake — the string is identical).

## Classification summary

| Category | Count (approx.) | Disposition |
|---|---|---|
| Accidentally-production-reachable (structural, not currently exercised) | 5 builder definitions + 3 driver definitions | Closed by §2-§4 (kwarg removed from drivers; builders remain but are no longer on any production call path) |
| Diagnostic/reference only | ~60 | Retained, allowlisted per §13 |
| Test only | ~30 | Retained per §11, several superseded/rewritten per §15 (the two driver-wiring tests whose entire purpose was validating a kwarg this task removes) |
| Dead code (pre-restructure, unreachable from real drivers) | ~50 (`cc_algo/`, `legacy/`) | Left in place — out of this task's scope to delete unrelated legacy code; confirmed not on any audited path |
| Unrelated (naming collision, Finding 3) | ~15 | Not touched |

No occurrence found in this audit is "accidentally production-reachable" in the sense of a live
call site an unmodified real campaign would hit today (Finding 1) — the risk this task closes is
structural/latent (Finding 2's first bucket), matching exactly how both real incidents happened:
not through a call site anyone could see was wrong, but through a private default silently
regressing with no test or type system in a position to catch it.
