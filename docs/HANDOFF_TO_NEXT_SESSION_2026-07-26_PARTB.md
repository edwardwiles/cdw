# Handoff to next session — shared economic-core FG operator (addendum Part B onward) — 2026-07-26

This file **is** the prompt to hand to the next Claude Code session. Paste everything below the
`---` line as that session's opening message.

---

Continue the five-family production optimization stack task, now under an **addendum** the user
gave mid-session that generalizes and formalizes what had already emerged organically. Read, in
this order, before doing anything else:

1. `gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26/docs/ORIGINAL_TASK_PROMPT_2026-07-26.md`
   — the original 13-section task spec (operator partitions, cross-Hessian formula, interval-basis
   requirements, acceptance invariants).
2. The **addendum** — ask the user to re-paste it if it isn't already in your context, or check
   recent conversation history; it is NOT yet saved as its own file in this repo (a gap this
   session should close: save it as
   `docs/ADDENDUM_SHARED_ECONOMIC_FG_2026-07-26.md` verbatim before doing anything else, so it
   isn't lost to context compaction). It specifies, in Parts A-I: (A) making the unrestricted
   family's compressed inner FG allocation-free [DONE, see below]; (B) one shared
   `EconomicCoreOperator`/`EconomicFGWorkspace`-style forward/transpose operator used by ALL FIVE
   families for their common economic-core block, each family only appending its own restriction
   operator; (C) operator-based post-solve verification (replacing dense-`G` reads in
   `archC_verified_state`-style functions); (D) runtime counters + fail-fast guards against any
   dense-`G` materialization; (E) preallocation for the restriction-specific operators; (F/G) D=4
   and D=20 correctness + performance gates per family; (H) a production backend manifest; (I) a
   long list of specific deliverable docs.
3. `docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md` — the current, up-to-date state of
   every restricted family's inner-FG backend (what's matrix-free, what's dense, what's flipped to
   default vs available-but-off, and why).
4. `docs/UNRESTRICTED_ALLOCATION_FREE_FG_PORT_2026-07-26.md` — addendum Part A, DONE this session.

## What already happened (read before touching code)

Branch `port/finish-five-family-optimization-stack-2026-07-26`, worktree
`gravity_robustness/worktrees/finish-five-family-optimization-stack-2026-07-26`, base
`production/fullA-exact @ f1fa8e770759c62b3f96c1024dd310f235ea463e`. **Nothing pushed to origin or
merged to production** — still a local feature branch; confirm with the user before either.

This session (a continuation of an earlier session that did Phases 0-4/8-10 of the original
prompt) did, in order:

1. **Flexible CM**: fixed a real allocation regression in the inherited `:cm_lookup` kernel
   (`CMLookupState`, `cm_lookup_kernels.jl`/`cm_lookup_production.jl`) via persistent per-callback
   scratch buffers and in-place kernels — was 12.6% *worse* allocation than dense, now exact
   parity; 1.1x-1.6x faster at every thread count tested (1/4/8/10/20). **Flipped to default**
   (`CM_INNER_FG_BACKEND_DEFAULT = :cm_lookup`, `core_exact_hessian.jl`). Also found and fixed a
   second, independent issue: the moments! closure was unconditionally dense-filling the CM
   columns even when the lookup FG path never reads them — added a `skip_cm_fill_ref` toggle
   (`archC_base_state` skips it, `archC_verified_state` defensively forces it back on since its
   own post-solve recompute needs the columns).
2. **Common Fréchet**: built a NEW matrix-free CM+level operator (`CMFrechetLookupState`,
   `cm_frechet_lookup_kernels.jl`/`cm_frechet_lookup_production.jl`) — genuinely new backward-
   gradient derivation for the level-anchor block (no prior kernel existed). D=4+D=20 correctness
   ALL PASS, 1.1x-1.2x faster, but allocation regressed ~21% (NOT fixed — root cause not chased
   down, likely either genuine extra level-block work or a KNITRO-iteration-count difference, not
   confirmed either way). **NOT flipped to default** (`CM_FRECHET_INNER_FG_BACKEND_DEFAULT` stays
   `:dense_reference`) — fails the flip rule's "allocation falls materially" criterion. Available
   opt-in. A real KNITRO-callback-protocol bug was found and fixed along the way (see
   `RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md` §3.2 for the full story and the
   debugging discipline that caught it — useful pattern to reuse).
3. **Addendum Part A** (mid-session addendum, unrestricted family): made
   `_callbackEvalFG_inner_compressed!`'s FG evaluation (`compressed_live.jl`, the actual default
   production callback, not a diagnostic path) allocation-free — `EconomicFGWorkspace`,
   `compressed_dual_contraction!`/`compressed_transpose_contraction!`/`compressed_cc_value_grad!`
   (new files/additions to `compressed_moments.jl`/`compressed_cc_inner.jl`/`compressed_live.jl`).
   **20,081x allocation reduction** (2.57MB → 128 bytes/call), bit-for-bit identical output,
   validated against the existing comprehensive `test_compressed_live_integration.jl` suite (ALL
   PASS, unmodified). This is live now — no flag, it's unrestricted's only FG path.

Also flagged, explicitly NOT fixed, per direct user instruction to document rather than chase:
`compressed_cc_hvp` (the Hessian-vector-product sibling of `compressed_cc_value_grad`, same file)
was not audited for the same allocation pattern — check it, it likely has the same issue
(`ddPsq = similar(q)`, `r = similar(q)` visible in the code read this session).

All work is in 4 commits on the branch (`git log --oneline c3c0073..HEAD`), each independently
gated before commit. Two Dropbox packages pushed
(`dropbox:Gravity robustness/Analysis/Server Output/cm_lookup_allocation_fix_2026-07-26` and
`.../frechet_lookup_and_unrestricted_allocfree_2026-07-26`).

## Priority order for next session (per the addendum + user's own redirect this session)

The user explicitly asked to stay focused and not drift into open-ended exploration — when in
doubt, prefer closing out a validated, scoped piece over opening a new investigation.

1. **Save the addendum itself as a doc** (see above) — it currently only exists in conversation
   history, a real risk of being lost.
2. **Part B, starting with origin-ZC's `[E | Z]` operator** (smaller than CM+ZC's `[E | C | Z]`,
   and genuinely unbuilt — no existing partial kernel at all, unlike CM+ZC which at least has a
   dense reference to extend). This is the natural next-smallest unit, matching the "lowest risk
   first" discipline that worked well this session (flexible CM → common Fréchet → unrestricted, in
   increasing/varying difficulty, each validated before moving on).
3. **CM+ZC's `[E | C | Z]` operator** next — reuses flexible CM's own bin-lookup kernels for the
   `C` block unchanged (same as common Fréchet did), needs a new small dense/structured operator
   for the `Z` (mean/pair) block.
4. **Only after B is real for at least one restricted family**: consider Part B's harder ask —
   literally ONE shared `EconomicCoreOperator` implementation used by all 5 families (currently,
   even after this session, unrestricted's compressed economic-core math
   (`compressed_dual_contraction!`/`compressed_transpose_contraction!`) and the restricted
   families' economic-core math (dense `BLAS.gemv!` against `materialize_dense_factual_structured!`
   output) are two SEPARATE implementations of conceptually the same `Eλ`/`E'v` operator — unifying
   them, per the addendum's explicit ask, is real, correctness-sensitive refactoring work, not
   wiring, and should not be rushed.
5. **Part C** (operator-based verification, removing `skip_cm_fill_ref`/dense-`G` reads from
   verification) only after B has at least one real shared operator to verify against.
6. Parts D/F/G/H (counters, gates, manifest) scale with B — do them per-family as each family's
   operator lands, not as a separate omnibus pass at the end.

## Concrete gotchas this session hit (save yourself the rediscovery time)

- **KNITRO FG callback protocol**: `KN_add_eval_callback(kc, true, Int32[], cb)` in this codebase
  calls ONE combined callback for both `f` and `g` on every invocation — do NOT branch on
  `evalRequest.evalRequestCode == KN_RC_EVALFC`/`KN_RC_EVALGA`; that branch will silently error on
  every call in a way that can look like a successful-but-degenerate solve (`nStatus=0`,
  `Delta_dual` exactly the value at the untouched initial point). Always mirror an existing,
  already-working callback (e.g. `_callbackEvalFG_inner_cmlookup!`,
  `_callbackEvalFG_inner_compressed!`) rather than reconstructing the protocol from general KNITRO
  knowledge. Include the `f <= obj.lower_limit ? -KN_INFINITY : f` guard every working callback
  has.
- **When a KNITRO-solve-level result looks wrong, write a standalone unit-style comparison first**
  (new kernel's `(x,g)` callable vs the dense reference callable, at a fixed `x`, no KNITRO
  involved) before suspecting the kernel math — this session's Fréchet bug was 100% a KNITRO-wiring
  issue, and the kernel math was exact from the first attempt; a unit test isolated that in one
  shot instead of a long dense-vs-lookup-through-KNITRO debugging session.
- **A dense moments! closure filling restriction columns doesn't know who's about to read them.**
  Any time you add a matrix-free FG operator, check whether the moments!-time dense fill for that
  same block is now partially wasted (only some callers still need it) — this was true for both
  flexible CM (fixed) and would be true for any new family's operator too; the
  `skip_cm_fill_ref`-toggled-by-caller pattern in `cm_production_bundle.jl`/`cm_frechet_cplus.jl`
  is the template, though the addendum explicitly wants this REMOVED in favor of explicit backend
  dispatch once operator verification exists (Part C) — don't extend the toggle pattern to new
  families if you're also doing Part C work; build the operator-verification path first for new
  families instead.
- **Always check in on a background job within ~30-60s** (`ps` + tail its log) before a long wait —
  this project's standing rule, and this session hit several include-order/missing-dependency
  errors this way that would otherwise have wasted a full D=20 run's wall-clock.
- Julia: `PATH="$HOME/.juliaup/bin:$PATH"`, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`,
  `--project=<worktree-root>` (not `--project=.` from inside `full_aod_diag/d4_exact/`).
- **Confirm with the user before pushing to origin or merging to production** — standing rule,
  nothing from this branch has been pushed/merged yet, 4+ sessions of work now sitting on it.
- **Push session deliverables to Dropbox before ending the session** — standing rule, use a new,
  clearly-named subfolder each time (see the two used this session for naming precedent).
