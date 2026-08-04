# profiled-outer-ab-completion-2026-08-04 — prior-evidence classification

Continuation of `profiled-outer-ab-readiness-2026-08-04` (tag `profiled-outer-ab-ready-2026-08-04`,
canonical prototype SHA `b6ac1c6`, verified live via `git fetch` at session start). This table
classifies every component that prior session's own `FINAL_CLOSEOUT_2026-08-04.md` claimed, per
this continuation's task §2. Source verified directly against current worktree
(`/bbkinghome/edav/cdw_worktrees/profiled-outer-ab-completion-2026-08-04`), not re-read from prose.

| Component | Status | Evidence |
|---|---|---|
| Canonical ZC free-nu dispatch (origin_zc + cm_meanzc, both formulations) | COMPLETE_AND_RETAIN | `profiled_zc_free_nu_production_driver_2026-08-04.jl` present; `bin/run_profiled_model.jl` dispatches through it; prior D4/D20w20k gates real (18/18, 16/16 checks) |
| Matched FULL/REDUCED gradient timer instrumentation | COMPLETE_AND_RETAIN (serial only) | `profiled_matched_gradient_instrumentation_2026-08-04.jl` present; serial `accounting_ratio` 0.9995 (REDUCED) / 0.9999 (FULL), both ≥0.98. Threaded semantics NOT valid (see task §7 — this continuation's own gate) |
| REDUCED outer-gradient coordinate-loop threading | COMPLETE_AND_RETAIN | gate: serial-vs-threaded bit-identical, `Threads.nthreads()=10` confirmed |
| REDUCED bandwidth-search reuse cache | PARTIAL_NEEDS_GATE | correctness (on/off equality, hit reuse, no stale reuse) passed at D20/W=20,000; **material benefit never measured** — task §8 of this continuation |
| `:profiled_powered_relative_A` encode/decode | PARTIAL_NEEDS_GATE | math derivation + bijection proof done, implemented in `profiled_powered_relative_a_2026-08-04.jl`; **not wired into any production context, CLI, checkpoint, or cache key** — task §6 of this continuation |
| Required-no-default arguments (`threaded`/`threaded_gradient`/`validity_radius`) | COMPLETE_AND_RETAIN | verified live this continuation (task §4): every kwarg added in the prior session's own new files (`profiled_matched_gradient_instrumentation_2026-08-04.jl`, `profiled_reduced_bandwidth_cache_2026-08-04.jl`) has no default; `bin/run_profiled_model.jl --threaded-gradient` is a required, validated CLI flag |
| FULL CLI include-list + `w0`/`probs` fixes (flexible_cm/common_frechet/cm_meanzc) | COMPLETE_AND_RETAIN | `cm_aspace_coordinate.jl` added to include list; `w0`/`probs` now genuinely constructed for fresh runs, mirroring `cm_production_stage_runner.jl`'s calibration branch |
| Decoded-state outer-gradient A/B | PARTIAL_NEEDS_GATE / INVALID_PRIOR_EVIDENCE for scope claims | only flexible_CM × calibration point × one direction tested; REDUCED vs FULL agree to 0.002%, matches one-off dense-G FD. **Not a 5-family × 3-point × 6-direction matrix** — task §9 |
| Short outer-search A/B "winner" labels (§9-10 of prior session) | INVALID_PRIOR_EVIDENCE | single unrepeated run per arm, W=20,000 only, judged from KNITRO eval/gradient counts and wall time, not verified objective progress. Per this continuation's task §3, these labels must NOT be reused as evidence |
| Coordinate-mode tournament (native vs powered) | NOT_DONE | explicitly not attempted by prior session (out of budget) — task §10 |
| Algorithmic-parity A/B mode | NOT_DONE | explicitly not attempted (no 1-thread/no-cache harness exists) — task §11 |
| W=100,000 outer/resource gates | NOT_DONE | prior session only ran W=20,000 for outer-search; W=100k warm-start/fast-reject gates that DO exist in `FamilyRegistry.jl` are from the separate functional-readiness task, not this outer-readiness lineage — task §13 |
| Fixed-state scientific equivalence (inner A/B) | NOT_DONE (external) | owned by `benchmark/profiled-fixed-state-inner-ab-2026-08-04`, still at its own step 2/13 (`9aa3b40`) per live worktree check this continuation's session start — out of scope here, must not be duplicated |
| `ProfiledLFixCache`/shared-workspace aliasing hazard | NOT_DONE | discovered, documented, not fixed by prior session (explicitly out of that session's "no inner kernel changes" scope, but this continuation's task §5 requires resolving it) |

## Reading

Nothing above is silently trusted from prose — each COMPLETE_AND_RETAIN row was re-verified by
grepping/reading the live source in this continuation's own worktree before being marked as such.
No component is re-derived or rewritten in this continuation unless a regression test on it fails.
