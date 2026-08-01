# Existing profiled-destination-scale gate reproduction (2026-08-01)

Diagnostic branch: `diagnostic/profiled-scales-unrestricted-knitro-2026-08-01`
Diagnostic worktree: `/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-knitro-2026-08-01`
Branched from: `architecture/profile-all-destination-scales-2026-07-31` @ `d84d392` (HEAD at branch-cut time)
Recorded prototype HEAD per task prompt: `b2a745b` — confirmed via `git diff --stat b2a745b..d84d392` that only
`PROFILED_DESTINATION_SCALES_MASTER_2026-07-31.md` changed between the two (122 lines of doc-only executive-summary
addition); zero code files differ. The diagnostic worktree is therefore code-identical to the recorded prototype
HEAD, with a more complete master report.
Base of the architecture branch: `production/fullA-exact @ cd17235`.
Archive `profile_all_destination_scales_2026-07-31.zip` was pulled from
`dropbox:Gravity robustness/Analysis/Server Output/` and cross-checked against the branch — the archive's
`key_results/code/*.jl` files are the same content as the branch's `full_aod_diag/d4_exact/*_2026-07-31.jl` files
(the archive is the doc/results package pushed off this branch, not a separate code source); the branch versions
were used directly, per task instruction "locate the branch versions."

## Julia environment note (first blocker, resolved)

The first run attempt (`julia --project=.` from inside `full_aod_diag/d4_exact/`) failed immediately with
`ArgumentError: Package SpecialFunctions not found in current path` — the Julia environment (`Project.toml`/
`Manifest.toml`) lives at the **worktree root**, not in `full_aod_diag/d4_exact/`. Fixed by invoking
`julia --project=<worktree-root>` from any working directory. `juliaup`'s `julia` (1.12.6) was used, per this
repo's standing note that `/opt/shared_sw`'s Julia is broken — `PATH="$HOME/.juliaup/bin:$PATH"`.
`OPENBLAS_NUM_THREADS=1`/`OMP_NUM_THREADS=1` set per this repo's standing hard-cap requirement (these gates are
single-threaded D=4 direct-evaluation tests, not a concurrency risk, but the env vars are exported unconditionally
per repo convention).

## Gate-by-gate reproduction (all 9 supplied D=4 gates + the static bundle guard)

All commands: `julia --project=<worktree-root> <test_file>.jl`, run from `full_aod_diag/d4_exact/`.

| Gate file | Exit | Result | Key reproduced number |
|---|---|---|---|
| `test_relative_a_coordinate_2026-07-31.jl` | 0 | PASS | round-trip exact; anchor cells confirmed unmoved at a retained-only perturbation |
| `test_gravity_pivot_on_retained_2026-07-31.jl` | 0 | PASS | fully-composed (anchor+gravity-pivot) point evaluates cleanly through `build_compressed_factual` |
| `test_outer_coordinate_layout_profiled_2026-07-31.jl` | 0 | PASS | round-trip diff `5.55e-17` at 5 random profiled points |
| `test_recover_full_a_2026-07-31.jl` | 0 | PASS | winner mismatches=0, max ratio diff=0.0 after gamma-normalizing recovery |
| `test_homogeneous_moments_2026-07-31.jl` | 0 | PASS | Σ_o homogeneous moment ≡ 0 confirmed; exact rescaling confirmed |
| `test_france_ratio_moment_2026-07-31.jl` | 0 | PASS | `max\|H1/H0 - predicted\| = 2.68e-12` (gp^σ homogeneity confirmed) |
| `test_homogeneous_contraction_2026-07-31.jl` | 0 | PASS | forward/transpose adjoint check `diff = 4.71e-14` |
| `test_homogeneous_hessian_2026-07-31.jl` | 0 | PASS | analytic vs finite-difference Hessian, worst entry diff `~4e-6` (relative, within FD tolerance) |
| `test_profiled_destination_scale_invariance_2026-07-31.jl` | 0 | PASS | winner/ratio/exponent/gravity all confirmed invariant, incl. baseIndex (France) destination |

**9/9 reproduced, all PASS, no discrepancies from the master report's claimed numbers.**

Static bundle guard (`scripts/static_bundle_guard_2026-07-30.sh`, pulled read-only into this branch's history —
present directly in this worktree, not fetched from elsewhere): run against `full_aod_diag/d4_exact/` —
**0 violations**. No hardcoded `:dense_reference` default, no direct `PsiObjectiveBundleImplicit` construction, no
`select_G_from_H` use, no static `bundle_type=OperatorPsiBundle` claim, outside the allowlisted test/diagnostic
files.

Full raw logs for all 9 runs are archived under `key_results/gate_logs_reproduction_2026-08-01/` in the final
deliverable package (see master report).

## Conclusion

The archive/branch reproduces cleanly and completely. Per the task's own §2 instruction ("stop if the archive
cannot be reproduced") — it reproduces, so this session proceeds to §3 (removing the anchor moments, the
prototype's main documented gap).

**Confirmed starting-point gap** (matches the branch's own master-report verdict, not a new finding): every
homogeneous-moment/FG/Hessian gate above operates on the **full** `ncolI = cf.oci - 1` moment dimension (all `D`
origins per destination, including the anchor cell) — `test_homogeneous_moments_2026-07-31.jl`'s own passing
result (`Σ_o` of the homogeneous moment ≡ 0 exactly) is direct numerical proof that the anchor column is exactly
linearly redundant given the other `D-1` retained columns, i.e. exactly the "one exact linear dependence per
destination" the task warns creates a nonidentified dual / singular Hessian if not removed. No KNITRO solve has
been attempted anywhere in the prototype (confirmed by reading every file — no `KNITRO.KN_new`/`KN_solve` call
outside the untouched, unrelated production files). This session's job is §3 onward.
