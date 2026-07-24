# Screen Stack Audit — 2026-07-23

Branch: `release/fullA-omit-row-restore-screens-2026-07-23`. Base: `production/fullA-exact@fd21f9f2`
(== `cdw/production/fullA-exact`). Traces the ACTUAL production call path from every relevant
driver to the inner solve, per-screen, rather than assuming the "other Claude's" prior
disabled-screens statement.

## Method

For each screen implementation file, located its call sites via `grep -l screened_eval\|
evaluate_fullA_screened` across `full_aod_diag/d4_exact/*.jl`, then separately grepped every
`cb_F!`-style production outer-loop callback (`c10_d20_production_driver.jl`,
`cm_outer_driver.jl`, `cm_checkpoint.jl`, `cm_production_stage_runner.jl`,
`originzc_production_stage_runner.jl`) for which evaluation function they actually call at
runtime, to determine reachability rather than inferring it from a function's mere existence in
the repo.

## Per-screen inventory

| Screen | File | Certificate type | Valid families |
|---|---|---|---|
| Pairwise/zero-winner certificate | `infeasibility_screen.jl::pairwise_certificate` | Exact (draw-free; `S_sod=B_so+a_od` dominance test) | Unrestricted, and any restriction that only adds moments/constraints on top of the same winner structure |
| Hard-winner/support certificate | `infeasibility_screen.jl::screen_hard_winners` | Exact (zero win-count on fixed draw support) | Same as above; tie-safe (credits all exact-tied winners) |
| Extreme-draw witness certificate | `infeasibility_screen.jl::build_extreme_draw_witness` / `query_witness` | Exact | Same as above |
| Pre-winner envelope certificate | `fast_range_screen.jl::precompute_envelope` (`EXACT_INFEASIBLE_PREWINNER_ENVELOPE`) | Exact, requires `usePMM==0` (raises `EnvelopeUnsupportedContext` otherwise, not a silent wrong bound) | Unrestricted; asserts the usePMM/NormalizeMoments assumptions live rather than hardcoding them |
| Fused winning-range certificate | `fast_range_screen.jl` (`EXACT_INFEASIBLE_WINNING_RANGE`) | Exact | Unrestricted |
| General safety-net moment-range scan | `fast_range_screen.jl` (`EXACT_INFEASIBLE_MOMENT_RANGE`) | Exact | Unrestricted; final catch-all before the inner KNITRO call |

All six are genuinely mathematically exact one-sided infeasibility certificates (never a
heuristic/approximate rejection) — confirmed by reading each implementation's docstring and the
underlying derivation comments (§2 of `infeasibility_screen.jl`'s header, §2 of
`fast_range_screen.jl`'s header), not merely asserted by the file's own claim.

## Call-path reachability (traced, not assumed)

**Single production choke point in the base driver**: `c10_d20_production_driver.jl::screened_eval`
(line ~292) is documented in its own header comment as "EVERY call in this driver goes through
this, never raw `evaluate_fullA_fast`," and routes to `evaluate_fullA_screened_ranged`
(`fast_range_screen.jl`), which itself falls through to the older `screen_hard_winners`
(`infeasibility_screen.jl`) when the envelope screen is unsupported for the current context. This
is confirmed live by grep: `screened_eval` is the only function that constructs a `ScreenCounters`
and is called from every `cb_F!`/`cb_G!`/checkpoint-resume site inside that file.

**Restricted-family drivers bypass this choke point entirely**:

- `cm_outer_driver.jl::run_cm_upper` (deprecated single-shot form) calls
  `cm_production_value_verified` directly (line 88) — no `screened_eval`, no
  `evaluate_fullA_screened*` anywhere in the file.
- `cm_checkpoint.jl::run_cm_upper_checkpointed` — the **documented recommended CM production
  entry point** (`cm_outer_driver.jl`'s own docstring: "Prefer `run_cm_upper_checkpointed`") —
  also calls `cm_production_value_verified` at every evaluation site (lines 510, 597, 704) and
  never references `screened_eval`/`evaluate_fullA_screened*`.
- `cm_production_stage_runner.jl` (the actual top-level script the CM production supervisor
  launches) calls `run_cm_upper_checkpointed` and, for its own pre-flight check,
  `cm_production_value_verified` directly (line 197) — same bypass.
- `originzc_production_stage_runner.jl` (the origin-specific-ZC production supervisor entry
  point) calls `run_originzc_upper_checkpointed` and `cm_originzc_production_value_verified`
  (line 166) — the ZC analogue of the exact same bypass, confirming it is not CM-specific but a
  structural property of every restriction-family driver built on the CM production bundle
  (`cm_production_bundle.jl`).

`cm_production_value_verified` (`cm_production_bundle.jl:231`) traces down through
`archC_base_state`/`archC_verified_state` to `evaluate_fullA_fast` (oracle_fast.jl) — the raw,
unscreened dense evaluator `screened_eval`'s own header comment explicitly says callers should
never call directly.

## Verdict per family

| Restriction family | Screen status | Evidence |
|---|---|---|
| Unrestricted (`c10_d20_production_driver.jl`) | **ACTIVE** | `screened_eval` is the sole entry point; wired as "the production path, not opt-in" |
| Flexible common marginals (CM) | **UNREACHABLE** | `run_cm_upper_checkpointed`/`cm_production_stage_runner.jl` call `cm_production_value_verified` exclusively |
| CM + mean/ZC | **UNREACHABLE** | Built on the same `cm_production_bundle.jl` machinery as flexible CM; same call path |
| Origin-specific ZC | **UNREACHABLE** | `run_originzc_upper_checkpointed`/`originzc_production_stage_runner.jl` call `cm_originzc_production_value_verified` exclusively |
| Fixed Fréchet marginals | Not yet released to production (per-branch WIP, `experiment/fullA-fixed-frechet-*`) — out of scope for this audit | — |

**This is not the "screens were disabled during the CM merge" the task brief hypothesized —
it is more precise than that.** The screens were never disabled at the base/unrestricted level
(still `ACTIVE` there today). What actually happened: the CM production bundle
(`cm_production_bundle.jl`, `cm_outer_driver.jl`, `cm_checkpoint.jl`) was built as a **parallel,
independent evaluation path** that was never wired to call through `screened_eval` in the first
place — a gap introduced when the CM machinery was built, not a regression from previously-active
CM screening. Origin-specific ZC inherited the same gap because it was built on top of the CM
bundle rather than on the base driver.

## Restoration plan (Part B step 8)

`screened_eval`'s screens (pairwise/witness/hard-winner/envelope/winning-range/safety-net) are
each a pure function of `(xf, ctx)` — they only need the current outer point and the (already
shared) `ctx`/`RangedScreenContext`, not anything specific to the unrestricted moment layout.
`cm_production_value_verified` and `cm_originzc_production_value_verified` are additive wrappers
around the same underlying dense evaluator (`evaluate_fullA_fast`); the fix routes their
pre-inner-solve check through the existing base-context screen the same way `screened_eval`
already does, gated so the CM-specific extra moments never look for a "missing winner" on a
destination outside the active mask (Part B step 8's reduced-destination requirement) — the
screens already operate on `ctx`'s destination indices only, so this holds automatically once the
screens are called with the CM/ZC context's own base `ctx`, not a synthetic one.

See `cc_algo`/`full_aod_diag/d4_exact/cm_screen_bridge.jl` (this release) for the wiring and
`docs/key_results/screen_restoration_*` for validation.
