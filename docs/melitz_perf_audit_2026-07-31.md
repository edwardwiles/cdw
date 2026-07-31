# Melitz moment-construction / inner-callback performance audit — 2026-07-31

Follow-up to the same-day legacy-H removal audit, using spare capacity in that session.
Scope: `src/melitz/moment_operator.jl` (moment construction, forward/transpose operator,
structured Hessian) and `src/melitz/cc_bundle.jl` (the `MelitzCCBundle` functor — the real
KNITRO objective/gradient/Hessian callback). All measurements are empirical (`@allocated`,
`Profile.Allocs`, `@elapsed`), taken against the **real D=20/W=80,000 production dataset**
(`real_data/noah_D20`), not D=4 toy fixtures — per this project's own repeated lesson that
allocation/performance claims must be re-verified at real scale, not inferred from a smaller
test fixture or trusted from a docstring.

Scripts: `scripts/perf_audit_allocations_2026-07-31.jl` (the `@allocated` sweep),
`scripts/perf_audit_allocsites_2026-07-31.jl` (`Profile.Allocs` line/type attribution),
`scripts/perf_audit_mulG_fusion_verify_2026-07-31.jl` (correctness + speedup verification for
the `mul_G!` fix). Raw logs in `docs/key_results/`.

## Headline: this code is already unusually well-optimized

`moment_operator.jl`/`cc_bundle.jl` have already been through multiple documented
optimization passes (closure-boxing elimination, thread-local scratch, O(W·D)/O(D²·W)
algorithmic complexity that is already close to the theoretical minimum for a full D²-moment
Hessian). Two of the four things this audit measured were exactly as documented
(`mul_G!`/`mul_Gt!` and the serial `melitz_full_weighted_gram!` are genuinely zero-allocation).
The findings below are real but incremental: one clear allocation bug (now fixed), one real
constant-factor loop-fusion win (now fixed and verified), and two accurately-quantified,
documented-but-not-implemented improvement targets for a future session.

## Findings and status

| # | Location | Finding | Evidence | Status |
|---|---|---|---|---|
| 1 | `cc_bundle.jl`, `MelitzCCBundle` functor | Default keyword arguments (`g`/`θ`/`h`/`constr`/`jac`) were literal `Float64[]`/`Array{Float64}(undef,0,0)` **expressions**, re-evaluated (and freshly heap-allocated) on every call that omits them — on the single hottest call site in the codebase (KNITRO's own line-search calls this many times per Newton iteration) | `@allocated`: 176 B (objective-only) / 144 B (objective+gradient) per call at real D=20/W=80,000, confirmed via `Profile.Allocs` to be exactly 4-5 empty `Vector{Float64}`/`Matrix{Float64}` allocations | **FIXED** — module-level `const` empty singletons (`_MELITZ_CC_EMPTY_VEC`/`_MELITZ_CC_EMPTY_MAT`), safe because every use of these args is gated behind `length(...) > 0` before any read/write (never mutated). Re-measured: 0 bytes, both cases. |
| 2 | `moment_operator.jl`, `mul_G!` | Two separate `for s in 1:W` passes per origin (one unconditional `u[s] += const_o`, one conditional `u[s] -= z_power[s,o]*cum[b+1]`) — this function runs on every objective/gradient/Hessian callback, so the extra pass over `u`/`bin[:,o]` is paid `D` times per call | Fused into one pass; verified against a literal reimplementation of the original two-pass algorithm at real D=20/W=80,000: **not bit-identical** (floating-point reassociation), max abs diff `1.4e-14`, max relative diff `4.9e-16` (≈2 ULP — negligible against KNITRO's own `1e-6`–`1e-8` tolerances and the LFD verification's `1e-6` moment/normalization tolerances) | **FIXED** — 1.41× measured speedup on `mul_G!` alone (0.00405 s → 0.00287 s/call at D=20/W=80,000), zero behavior-relevant change. |
| 3 | `cc_bundle.jl`, `melitz_update_operator_at_theta!` (called once per new outer point, via `melitz_bundle_prepare_at_theta!`) | ~77 KB/call — its own docstring described this as "a small, O(D)-sized allocation" (referring only to `sortperm`), but the actual dominant contributor (`Profile.Allocs`-confirmed: `Memory{Int64}`/`Vector{Int64}` ≈12 KB, plus other structures) is `melitz_expand_theta`'s **non-mutating** return of fresh `A`/`f` matrices/state — this file never adopted the **already-existing** mutating alternative (`melitz_expand_theta!`/`MelitzExpandedState`/`MelitzThetaExpansionWorkspace`, `log_cutoff_param.jl`) that the coordinate-probe gradient code (`direct_gradient.jl` etc.) already uses for exactly this reason | `@allocated`: 76,896 B/call, consistent across warm calls (not JIT) | **DOCUMENTED, not fixed** (docstring corrected with the exact number and root cause). Porting would require threading a workspace through `MelitzPrimitives`/`MelitzEquilibrium`/`MelitzCounterfactual` construction too — a larger change than this session's scope, given this call happens once per outer point (not per KNITRO iteration): 25 times in the one real profiled-A middle-loop run measured in the companion legacy-H audit, vs. hundreds of objective/gradient/Hessian calls in the same run. Real but lower-urgency than findings 1-2. A secondary, smaller contributor: the per-origin `sortperm(cutoff_o)` inside `melitz_update_moment_operator!` could be replaced with `sortperm!` into a preallocated `Vector{Int}` buffer (`op` has no such scratch field today; one would need to be added). |
| 4 | `moment_operator.jl`, `melitz_full_weighted_gram_parallel!` | ~16.4 KB/call, contradicting the function's own docstring, which claimed "confirmed zero-alloc" — that claim was accurate about eliminating **closure-boxing** allocation (a real, documented, earlier fix) but not about achieving literal zero bytes | `Profile.Allocs`: 40 `Task` objects (8,960 B) + matching `SpinLock`/`GenericCondition`/`IntrusiveLinkedList`/scheduler-closure allocations — exactly `2 threaded regions × Threads.nthreads()=20`, i.e. **`Threads.@threads`'s own inherent per-iteration task-spawn overhead**, not a per-element or boxing allocation, and independent of `D`/`W` | **DOCUMENTED, not fixed** (docstring corrected). Eliminating this fully would mean replacing `Threads.@threads` with a lower-overhead primitive (a persistent worker pool/channel pattern, or a package like `Polyester.jl`/`OhMyThreads.jl`'s static scheduler) — an architectural change affecting every other `Threads.@threads` site in this codebase too (`direct_gradient.jl`, `sorted_crossing_gradient.jl`, etc. all use the same pattern), not something to do unilaterally inside one function without the user weighing in, since Julia's threading primitives in this codebase have a documented history of subtle bugs (`feedback-julia-threads-closure-boxing-surrounding-code` memory). At ~16 KB/call this is small in absolute terms — flagged for completeness/documentation accuracy, not as an urgent problem. |

## Things checked and found already good (no action needed)

- **Algorithmic complexity**: the structured-Hessian construction (`melitz_full_weighted_gram!`)
  is `O(D²·W)` — the same-origin (`D` blocks) and cross-origin (`D(D-1)/2` block-pairs) loops
  each cost `O(W)`, which is asymptotically necessary to fill a genuinely `O(D²)`-sized dense
  Hessian from `W` draws. No cheaper algorithm exists for the full Hessian; this is already
  the design the codebase's own prior sessions arrived at via a documented derivation.
- **Memory layout / cache behavior**: `bin`/`z_power` are `W×D`, column-major, and every hot
  loop iterates `o` (columns) outer, `s`/`w` (rows) inner — correctly aligned with Julia's
  column-major storage; no obvious cache-thrashing access pattern found.
- **`mul_G!`/`mul_Gt!`/`melitz_full_weighted_gram!` (serial)**: genuinely zero-allocation, as
  documented — confirmed, not merely trusted.
- **`@melitz_profile` macro**: verified zero-cost when `MELITZ_PROFILE[]` is off (literal
  pass-through, no wrapping overhead) — not a hidden allocation source.
- **`melitz_recover_lfd_from_solution`'s scratch reuse** (`obj.arg0`/`obj.arg1` instead of
  fresh `zeros(W)` per call) — already fixed in a prior session (2026-07-27, per its own
  comment); re-confirmed still in place and correct.

## Wall-clock numbers (real D=20/W=80,000, 20 Julia threads / 1 BLAS thread)

| Quantity | Before this session's fixes | After |
|---|---|---|
| `mul_G!` | 0.00405 s/call | 0.00287 s/call (**1.41×**) |
| Objective-only functor call | 176 B/call | **0 B/call** |
| Objective+gradient functor call | 144 B/call | **0 B/call** |
| Hessian-only functor call | 16,640 B/call | 16,528 B/call (residual is `Threads.@threads` overhead, finding #4) |
| Structured Hessian, serial | 0.0323–0.0360 s/call | unchanged (not touched) |
| Structured Hessian, parallel (20 threads) | 0.0122–0.0130 s/call | unchanged (not touched) |

## Regression check

The full Melitz test suite (`test/melitz/runtests.jl`) was re-run after both fixes
(`docs/key_results/melitz_perf_audit_regression_run_2026-07-31.log`). Like the companion
legacy-H audit's own regression run, `Test.jl`'s top-level-`@testset` abort-on-first-failure
semantics mean this run stops at the first failing testset, not at the end of the file.

**This run got further than the prior run** (same file, same abort semantics): it passed the
"Phase 6: nuisance-profile matrix-free port matches dense reference" testset that failed in
the legacy-H audit's regression run (that failure — a converged-optimizer-endpoint floating-
point discrepancy between two gradient backends after an independent iterative trust-region
search — is itself evidence of pre-existing run-to-run non-determinism in that specific test,
unrelated to any code in this file: nothing in `nuisance_profile.jl` or its dependencies was
touched by either session), and instead stopped later, at
`test/melitz/runtests.jl:6327`'s testset ("Governing prompt Phase 1.1/1.3/5 (2026-07-27
night): focal-link + plain-backend in-place expansion, allocation ceilings"), on two
allocation-ceiling assertions:

- "D=4 (FIXTURE): plain (non-sorted) direct backend now matches the sorted backend AND is
  allocation-free" — `@test b1 == 0` / `@test b2 == 0` for `make_melitz_gradient_delta_direct_parallel`
- "real D=20: full sorted-serial/parallel outer gradient (focal-link included) is post-warmup
  allocation-free" — same assertion for `make_melitz_gradient_delta_direct_sorted_parallel`

**Both are pre-existing, proven via a direct before/after comparison, not a regression from
this session's fixes.** Both parallel gradient-backend functions use `Threads.@threads
:static` internally (`direct_gradient.jl`) — the exact same construct behind finding #4
above. A minimal isolated repro (`@allocated` on `make_melitz_gradient_delta_direct_parallel`
at D=4, 8 Julia threads) was run against BOTH this session's modified code AND a second,
completely clean worktree checked out at the unmodified base commit (`d0904c7`, before any
change this session or the legacy-H audit made): **both measured exactly 8,912 bytes/call,
bit-for-bit identical.** This proves the allocation is `Threads.@threads`'s own inherent
task-spawn overhead, present identically before and after this session's edits — the
allocation-ceiling tests assuming these parallel backends are literally zero-allocation were
already inaccurate on this Julia version (1.12.6) before this session began; this session's
fixes did not introduce, worsen, or otherwise change this behavior.

No other test failed differently between the two runs. Every dense-vs-matrix-free numerical
equivalence testset that executed in either run passed in both.

## Recommendation summary

1. **Done, safe, verified**: keep fixes #1 (empty-array-default allocation) and #2 (`mul_G!`
   loop fusion).
2. **Worth a future session, not urgent**: port `melitz_update_operator_at_theta!` to the
   existing mutating `melitz_expand_theta!`/workspace machinery (finding #3) — real ~77 KB/call
   saving, but this function runs once per outer point, not per KNITRO iteration, so the
   aggregate win over a campaign is smaller than #1/#2's per-callback saving multiplied by
   hundreds of callbacks per outer point.
3. **Worth documenting accurately, not worth an isolated fix**: `melitz_full_weighted_gram_parallel!`'s
   ~16 KB/call `Threads.@threads` task-spawn overhead (finding #4) is architectural (shared by
   every `Threads.@threads` site in this codebase) and small in absolute terms — a genuine fix
   would be a deliberate, codebase-wide threading-primitive decision, not a one-function patch.
4. **New finding from the regression check, worth a future session**: the "Governing prompt
   Phase 1.1/1.3/5" allocation-ceiling tests (`test/melitz/runtests.jl:6327`) asserting
   `Threads.@threads`-based parallel gradient backends (`make_melitz_gradient_delta_direct_parallel`,
   `make_melitz_gradient_delta_direct_sorted_parallel`) are literally zero-allocation are
   currently failing on Julia 1.12.6, independent of any change in this session (proven via
   before/after comparison against the unmodified base commit — see Regression check below).
   Either these tests were written/verified against an older Julia version where
   `Threads.@threads :static` did not spawn per-iteration `Task` allocations, or they were
   never actually verified at real thread counts. A future session should either relax these
   assertions to an explicit small ceiling (documenting the `Threads.@threads` overhead, as
   this audit's docstring corrections now do for `melitz_full_weighted_gram_parallel!`) or
   replace `Threads.@threads` codebase-wide with a lower-overhead primitive if genuinely
   zero-allocation parallel callbacks are a hard requirement.
5. **Optional, very low priority**: `melitz_update_moment_operator!`'s focal-link (`ell`)
   computation has the same two-separate-`for w in 1:W`-passes pattern `mul_G!` had, fusable
   the same way — but this function also runs once per outer point (not per callback), so the
   wall-clock benefit is proportionally small. Not implemented this session to keep the
   verified fix set focused.
