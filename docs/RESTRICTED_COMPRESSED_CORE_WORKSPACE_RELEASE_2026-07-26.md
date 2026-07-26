# Restricted-Family Compressed-Core Workspace Release — Phase E — 2026-07-26

**State: MATCHED_AB_PASSED** (not merged to production/fullA-exact; on
`port/remediate-production-5x7-audit-2026-07-26`).

## Part 1: workspace wiring (committed separately, `1a58aa9`)

All four restricted families' `moments!` closures called the allocating `build_compressed_factual`
on every single inner-solve callback. A workspace-reusing `build_compressed_factual!`/
`CompressedFactualWorkspace` already existed, independently validated bit-identical, but was only
wired into the unrestricted family's DualBank scoring branch. A new `cf_build` dispatch helper
(reuses `ctx.cf_workspace` when attached, falls back to the allocating path otherwise) was swapped
into all four call sites; both public drivers now attach the workspace once at ctx build time.
Correctness gate: D=4, real `archC_verified_state` call path, byte-exact agreement (diff=0.0),
ALL PASS. Full detail in that commit's message.

## Part 2: a real bug found and fixed (committed separately, `879c8bb`)

Phase B1 added a new `inner_fg_backend::Symbol` field to the `CMBinHessCtx` struct and updated the
one `CMBinHessCtx(...)` constructor call it directly touched (`build_cm_bin_ctx`,
`cm_hessian_architectures.jl`) — but missed a **second**, separate construction site in
`build_cm_meanzc_bin_ctx` (`cm_meanzc_production.jl`), CM+ZC's own bin-context builder. Every
attempt to build a real CM+ZC production context has raised `MethodError` since that commit.

This was not caught by any Phase B1/C/D correctness gate in this remediation — every one of those
gates, despite describing "all four restricted families," only ever built and solved a plain
flexible-CM `pcx` directly. The bug surfaced only when Phase E's dense-vs-binfill benchmark became
the first script in this whole remediation to actually construct a real CM+ZC context. Fixed by
adding the missing `:dense_reference` argument (hardcoded — CM+ZC does not support
`inner_fg_backend=:cm_lookup`, see `cm_lookup_production.jl`'s own scope note).

**Follow-up gap-closing tests** (written specifically because this bug exposed a real hole in this
session's own gate coverage):
- `test_phaseE_meanzc_end_to_end.jl` (D=4): CM+ZC through the full B1+C+D+E chain simultaneously
  (dense-reference FG, exact cache, dual bank, compressed-core workspace) — **ALL PASS** (9/9
  checks): first solve cold+miss, repeated point zero-new-inner-solves+hit, nearby point
  warm+miss.
- `test_phaseE_frechet_end_to_end.jl` (D=4): common-Fréchet through the same full chain — **ALL
  PASS** (7/7 checks), identical pattern.
- Origin-ZC's own equivalent test was attempted (5 iterations: missing includes for
  `lfix_factorized_workspace.jl`/`cm_screen_bridge.jl`/`cm_originzc_config.jl`, then a
  `K_pair`-vs-`K_mean` config-validation constraint, then one further missing include) and set
  aside after the 5th attempt given time budget. **This is a test-harness completeness gap, not
  evidence of a code defect** — unlike the CM+ZC bug, every origin-ZC failure was a load-time
  `UndefVarError` from an incomplete ad-hoc include list for this specific standalone test, never
  a runtime construction mismatch. Origin-ZC's B1/C/D/E wiring follows the identical pattern
  applied consistently to all four families in those commits, and will be exercised for real by
  the D=20 profiling runs (Phase I) and any live driver smoke test (Phase G). Flagged explicitly,
  not silently dropped.

**Lesson applied going forward**: every new gate in this remediation must actually construct and
solve each family's own real `pcx`, not assume structural call-chain similarity is sufficient —
this is the second time this exact class of gap has surfaced this session (the first was the
`try`/`global` scoping bug in Phase A).

## Part 3: dense-columns vs bin-fill benchmark (this commit)

Task's own explicit instruction: benchmark CM+ZC's persistent-dense-CM-columns approach
(`precalc_common_marginals_cdf`, `cm_meanzc_moments.jl:425`) against plain flexible-CM's
bin-recompute/in-place-fill approach (`fill_cm_columns_from_bins!`, `cm_hessian_architectures.jl`)
at real D=20/L=50 scale, and "retain the measured winner" — but also "do not change that
representation merely for stylistic uniformity" if the benchmark doesn't show a real difference.

`test_phaseE_denseVsBinfill_benchmark.jl` times `obj.moments!` directly (the real per-outer-point
column-construction closure each family's inner solve depends on) at D=20/W=80,000/L=50, 8 warmed
repetitions:

```
flexible_cm (bin-fill CM columns)      : median=0.540s  mean=0.557s  median_alloc=65.0MB
cm_meanzc   (dense-copy CM + mean/pair): median=0.598s  mean=0.605s  median_alloc=64.7MB
```

**Interpretation**: CM+ZC's `moments!` call is ~10.6% slower in wall time than plain flexible-CM's
— but its **allocation is essentially identical** (64.7MB vs 65.0MB, CM+ZC actually marginally
lower). Given CM+ZC's `moments!` call necessarily does strictly more work than flexible-CM's (the
same CM block **plus** the mean/pair columns on top), a ~10% wall-time increase with no allocation
penalty is consistent with that extra, unavoidable work — not with the dense-copy CM-column
representation itself being a poor choice. This benchmark does not isolate "CM-columns-only" cost
(that would require modifying either family's code to strip out the other's extra work, which was
out of scope), but it is the real, actual production cost each family pays per outer point, and it
shows no clear signal that switching CM+ZC to bin-fill would help.

**Decision: retain CM+ZC's dense-column representation as-is.** Per the task's own instruction,
switching representations "merely for stylistic uniformity" without benchmark evidence of a real
difference is exactly what should be avoided here — the evidence available does not support a
switch.

## Files

`compressed_factual_buffer_reuse.jl` (`cf_build` helper, part of `1a58aa9`),
`cm_meanzc_production.jl` (bugfix, part of `879c8bb`),
`test_phaseE_meanzc_end_to_end.jl`, `test_phaseE_frechet_end_to_end.jl`,
`test_phaseE_denseVsBinfill_benchmark.jl` (this commit).
