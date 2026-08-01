# Profiled economic block, all-families completion — master status (2026-08-01, continuation session)

## Branch / commits / worktree

- New branch: `architecture/profiled-economic-block-all-families-complete-2026-08-01`
- New worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profiled-economic-block-all-families-complete-2026-08-01`
- Forked from: `architecture/profiled-economic-block-all-families-2026-08-01` @ `79cc0bd` (recorded HEAD in the
  task prompt; confirmed via `git worktree add ... -b <new> architecture/profiled-economic-block-all-families-2026-08-01`,
  0 divergence at fork time)
- Source branch `79cc0bd` itself was **not modified** (worked entirely in the new worktree/branch).
- This session's new commits: see `git log --oneline 79cc0bd..HEAD` in this worktree at hand-off time.
- Not merged, not pushed to `origin`, no production default changed, no campaign launched.

## What this session actually is

The mission is an 18-section, ~2000-line specification asking for a complete, gated, five-family port
(genuine reduced economic layout + D4/D20 correctness gates + inner-solve equivalence + outer-gradient
sharing + performance profiling + W500k smoke tests + a production go/no-go). That is realistically
several more full sessions of work at this codebase's own historical pace — every comparable prior
piece of this port (H_EC, H_EZ, the threaded H_EZ twin, the France row) took one dedicated session each,
with real bugs found only by actually running KNITRO solves and independent brute-force references, not
by design alone (see the source branch's own master doc, "off-by-one row index", "missing kappa0 scale
factor", "latent OOB read"). This session's own honest contribution, in order of confidence:

1. **Resolved a real open design question** the source branch's own docs explicitly flagged as
   unresolved (`PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §2's "open question, to resolve during
   implementation, not assumed here" for H_EF, and the more general question of how a restricted
   family's `CMBinHessCtx`/`OriginZCCoreHessCtx` should ever come to hold a genuinely reduced economic
   block at all). **Verified by direct source read**, not assumed:
   - `build_cm_augmented_obj_archB`/`wrap_moments_with_cm_archB` (and, by the same construction pattern,
     `build_originzc_augmented_obj`/`build_cm_meanzc_augmented_obj`) already read the economic-block
     width **generically** off `ctx.obj.d`/`ctx.obj.outer_constr_index` — never a hardcoded
     `D*Ddest`-shaped constant.
   - The dense economic-column materialization (`materialize_dense_factual_structured!`) is **already
     unconditionally skipped** on the current production path (`skip_fill=true`,
     `moment_representation=:operator`, cm_hessian_architectures.jl:314/334/338's `if !skip_fill`
     guards) — the economic block's actual gradient/Hessian flow entirely through `cf` and the shared
     `economic_forward!`/`economic_transpose!`/`fill_core_hessian_upper!`/cross-block primitives, never
     through a dense `G` column read, on the path that matters.
   - **Consequence**: feeding these builders a `ctx.obj`-shaped object whose `.d`/`.outer_constr_index`
     already reflect the reduced width (`1 + layout.total_reduced_economic_moments`, the exact quantity
     `build_profiled_operator_bundle` already computes for the unrestricted family) requires **zero**
     changes to `wrap_moments_with_cm_archB`/`build_cm_augmented_obj_archB` on the production path. This
     collapses what the mission's §5 abstractly describes as "genuine reduced/anchor-omitting
     `CompressedFactual` construction" for the dimension-bookkeeping half of the problem into a single
     small additive helper, not a rewrite of the augmented-obj builders.
   - Separately, confirmed the H_EC/H_EF/H_EZ correction-term functions
     (`winner_pair_cross_hessian_cm_block!`/`_colsum!`/`_esum!`/`_zc_block!`) are **already fully generic
     over `ncolI`/`target_slot`** (they read `wctx.ncolI`/`wctx.target_slot[j]` and index
     `ws.QCScum`/`ws.MCScum`/etc. by the SAME `j`, with no hardcoded assumption about what `j` "means") —
     but their "keep"/winner term (`winner_pair_cross_hessian_fill!`'s `QTab`/`MTab` fill, and
     `_zc_block!`'s own inline scatter loop) hardcode the OLD full-index formula `j = slot + (o-1)*Ddest`
     directly from raw `wctx.winner[w,slot]`, so **do not** support a genuinely-reduced `ncolI` without
     either a new anchor-aware fill (mirroring `ReducedHomogeneousWinnerPairHessCtx`'s own
     `winner_reduced_col==0`-skip pattern) or a different wiring strategy. Resolved: the viable,
     **fully non-invasive** strategy is to keep building `wctx`/`ws` from the FULL (unreduced) `cf` via
     the existing, untouched `build_winner_pair_ctx(cf; bi_slot=...)`, call the existing, untouched
     `winner_pair_cross_hessian_fill!`/`_cm_block!(...; use_profiled_correction=true)`/etc. exactly as
     they are today (producing a FULL-width `Hraw_EC`/`HEZ`), and have the ORCHESTRATOR
     (`hessian_cm_structured!`/`archA_partitioned_hess_cb_builder`) **gather only the retained rows**
     (via `layout.retained_full_factual_j`/`full_factual_to_reduced`) when copying into the now-smaller
     `Hfull`. This touches **zero** lines inside any of the five functions the mission explicitly says
     not to replace (and zero lines inside `winner_pair_cross_hessian_fill!` either, though that one
     wasn't on the "do not replace" list) — the entire reduction happens in the packing/assembly step. It
     is not FLOP-optimal for the cross-block itself (still `O(W*Ddest*D)`, same as before — no cross-block
     compute savings), but it is correct, minimal-diff, and consistent with where this port's actual
     motivation lives (dual-DIMENSION reduction for KNITRO conditioning/speed, not cross-block FLOPs — the
     unrestricted family's own H_EE reduction already captures the compute-savings half of the story).

2. **Implemented and D4-verified** the dimension-bookkeeping half of that plan, for flexible CM
   specifically (the source doc's own "recommended next steps" #1 priority):
   - New file `full_aod_diag/d4_exact/profiled_restricted_family_base_2026-08-01.jl`:
     `build_reduced_base_obj_for_family(ctx, layout, CS)` — shallow-copies `ctx.obj` with
     `.d`/`.outer_constr_index` overridden to `1 + layout.total_reduced_economic_moments`, every other
     field (γ, δ, l, U, N, moments!, inner_loop_opt, ...) unchanged. ~15 lines, additive, zero change to
     any existing file.
   - `cm_hessian_architectures.jl::build_cm_augmented_obj_archB` gained one new optional keyword,
     `base_obj = nothing` (default preserves old behavior byte-for-byte — confirmed by the D4 gate below,
     `aug_full.ncore == ctx.obj.d` for every point tested).
   - **Along the way, found and fixed one genuine pre-existing bug**, unrelated to this task, that blocked
     even the very FIRST direct call to `build_cm_augmented_obj_archB` (baseline, no `base_obj`): two
     methods of `compute_bin_indices` exist (`common_marginals_interval.jl:72`, `z::Vector{Float64}`,
     returns a compact `UInt8`/`UInt16`-typed matrix; `cm_hessian_architectures.jl:104`,
     `z::AbstractVector{Float64}`, returns `Matrix{Int}`) — when both files are included (as every
     relevant test does), Julia's dispatch picks the MORE SPECIFIC first one, producing a
     `Matrix{UInt8}` that then fails `wrap_moments_with_cm_archB`'s declared `Bidx::Matrix{Int}`
     parameter with a hard `MethodError`. Latent because the only pre-existing direct caller,
     `build_cm_production_context`, apparently never triggers this exact dispatch path. Fixed with a
     single explicit `Matrix{Int}(...)` coercion at the `Bidx = ...` call site inside
     `build_cm_augmented_obj_archB` — value-preserving, zero behavior change for any correctly-typed
     input, confirmed by re-running the pre-existing `test_winner_pair_cross_hessian_cm_d4.jl`
     (20/20 PASS, byte-identical `max|Δ|=0.0` results, unchanged from before this fix).
   - New gate: `test_profiled_restricted_family_base_2026-08-01.jl` — **32/32 PASS**, real D4 calibration
     point, both contrast conventions (`:anchored`/`:orthonormal`), both `L∈{10,20}`. Verifies (against
     the SAME calibration point/context every existing D4 cross-Hessian test uses): the layout's own
     retained-moment count formula; `reduced_obj0.d == 1+layout.total_reduced_economic_moments` and `<
     ctx.obj.d` (a genuine reduction, not a no-op); `build_cm_augmented_obj_archB(...; base_obj=nothing)`
     is unchanged from before this session (`aug_full.ncore == ctx.obj.d`); with `base_obj=reduced_obj0`,
     `aug_reduced.ncore`/`obj_cm.d`/`outer_constr_index` all correctly reflect the reduced width, the CM
     restriction block itself (`ncm`) is untouched, and the resulting `cctx_reduced.NCORE`/`Hfull` size
     shrink accordingly.

3. **Wired and D4-verified the reduced H_EE kernel and the H_EC gather step for flexible CM** — the
   full mission §6/§4 Hessian-orchestrator wiring, for one family, with a genuine numeric gate (not
   just dimensions):
   - `CMBinHessCtx` gained 9 new fields (`profiled_layout`, `profiled_theta_ref`,
     `profiled_reduced_wctx`/`_for`, `profiled_hee_packed`, `profiled_full_wctx`/`_for`,
     `profiled_full_ws`, `profiled_hraw_ec_full`), all via the struct's own pre-existing
     "outer-constructor-appends-fields-with-defaults" idiom (matching `hcz_prep_backend`/
     `bin_zc_drawchunk`) — every existing `CMBinHessCtx(...)` call site is untouched.
   - `wrap_moments_with_cm_archB` gained a `theta_ref::Ref{Any}` keyword (mirrors `core_cf_ref`
     exactly), publishing `copy(θ)` alongside the existing `core_cf_ref[]=cf` line;
     `build_cm_augmented_obj_archB` creates and returns it; `build_cm_bin_ctx` picks it up via the
     same `hasproperty(aug,...)` discipline `core_cf_ref` already uses, and gained a `profiled_layout`
     keyword threading a caller-supplied `ProfiledEconomicMomentLayout` onto the new `cctx` field.
   - `_fill_cm_HEE!` gained an early, self-contained branch (zero lines of the pre-existing
     `cf isa CompressedFactual`/dense-fallback logic touched): when `cctx.profiled_layout !== nothing`,
     builds (cf-identity-cached) a `ReducedHomogeneousWinnerPairHessCtx` via
     `build_reduced_homogeneous_winner_pair_ctx(cf, cctx.econ_ctx, cctx.profiled_theta_ref[], layout)`
     and calls `reduced_homogeneous_winner_pair_hessian!` — the UNRESTRICTED family's own,
     already-validated reduced H_EE kernel, reused verbatim — into a persistent packed scratch buffer,
     then unpacks (row-major upper-triangle, same convention `fill_core_hessian_upper!` itself uses)
     into the caller's dense `HEE` view. Guarded to error (not silently misbehave) if a caller ever
     combines `profiled_layout` with CM+ZC's mean/pair widening (`ncore_core < NCORE`) — not ported.
   - `hessian_cm_structured!` gained an early branch implementing item 1's "gather" design exactly:
     rebuilds (cf-identity-cached) a FULL, unreduced `WinnerPairHessCtx`/`WinnerBinCrossScratch` from
     the same `cf`, calls the untouched `winner_pair_cross_hessian_fill!`/`_cm_block!(...;
     use_profiled_correction=true)` per threshold block, and copies only
     `layout.retained_full_factual_j`-selected rows (+ row 1 + the France row) into the reduced
     `Hfull`, then proceeds through the pre-existing (untouched) `fill_cm_HCC!`/packing tail. Guarded
     to error if combined with a common-Fréchet `extension` (H_EF's own gather is not built — see
     below). `hessian_cm_structured_v2!` (the threaded twin, `cm_hessian_threaded.jl`) was **not**
     touched — the serial path only, this session.
   - **New D4 numeric gate**, `test_profiled_flexcm_d4_hessian_gate_2026-08-01.jl` — **ALL PASS**
     (16 checks: both contrasts × both `L` × 2 points, feasibility/cf checks + H_EE + H_EC +
     finite/symmetric), `max|Δ|=0.0` (bit-identical, not merely close) for BOTH H_EE and H_EC at every
     combination. Design: solves the FULL (unreduced) model normally to get a real
     `(ζ*,λ*)`, then zeroes the ANCHOR entries of `λ*` (one per destination) — this makes the full
     model's own `q = -ζ-Σ_j E_jλ_j` collapse to exactly the reduced model's `q_reduced` (the anchor's
     contribution drops out because its coefficient, not its formula, is zero), giving a dual point at
     which the two models are provably contracting the same way. H_EC is checked by gathering the
     FULL model's own `winner_pair_cross_hessian_cm_block!(...;use_profiled_correction=true)` output
     (the prior session's already-D4-validated primitive) at this point — **this passed on the first
     correct attempt** (`max|Δ|=0.0`), confirming the gather-index logic (`layout.retained_full_factual_j`
     row selection) is exactly right. H_EE required one real methodology correction, documented in
     detail in the test file's own header: an initial attempt gathered from `core_exact_hessian.jl`'s
     OLD `winner_pair_hessian!` kernel and found a genuine, reproducible ~0.22 discrepancy — traced
     (by reading `winner_pair_hessian!`'s body directly, not guessed) to that kernel's own "keep"/
     correction structure being **destination-independent** (scalar `t0`/`s0`, the exact H_EE analog
     of H_EC's OLD `nu_diff` correction this whole port replaces) — i.e. gathering from the
     *unprofiled* full H_EE is comparing two genuinely different formulas, not testing the same thing
     with fewer columns (no bug in the new code). Corrected to compare against directly invoking
     `build_reduced_homogeneous_winner_pair_ctx`/`reduced_homogeneous_winner_pair_hessian!` (the
     already-independently-validated reduced kernel) on the identical `(cf,ctx,θ_full,layout,arg0)` —
     a genuine, still-rigorous check of exactly what is NEW this session (the struct-field
     threading/caching/unpacking), not a re-derivation of the kernel's own math (out of scope, already
     done by the unrestricted family's prior session). Re-ran all three pre-existing/prior D4 test
     files after these changes — zero regression (`test_winner_pair_cross_hessian_cm_d4.jl` 20/20,
     `test_winner_pair_cross_hessian_zc_d4.jl` all pass, `test_profiled_restricted_family_base_2026-08-01.jl`
     32/32, all still `max|Δ|=0.0` / unchanged).

4. **Closed the FG-callback gap** flagged immediately below (item 5 as originally written), then
   diagnosed — but did **not** fully resolve — a real numerical-conditioning issue blocking an actual
   live KNITRO solve:
   - New `materialize_dense_factual_structured_reduced!(Gview, cf, layout; scratch_full=...)`
     (`profiled_restricted_family_base_2026-08-01.jl`) — the reduced analog of
     `materialize_dense_factual_structured!`, built via the SAME "gather from the existing validated
     full function into a persistent scratch buffer" design this session already used for H_EC — not
     a re-derived formula.
   - `wrap_moments_with_cm_archB` gained a `profiled_layout::Any=nothing` keyword: when set, `G` is
     **always** fully filled (both the reduced economic block AND the CM-grid columns — `skip_fill` is
     ignored/irrelevant for a profiled context, since there is no separate "`:cm_lookup` priming"
     concept for it); `nothing` (every existing caller) preserves every line byte-for-byte.
     `build_cm_augmented_obj_archB` threads this through as its own `profiled_layout` keyword.
   - **New gate, Part 1**: `test_profiled_flexcm_d4_fg_and_solve_gate_2026-08-01.jl` — calls
     `obj_reduced.moments!` **directly** (bypassing KNITRO entirely) at the calibration θ and compares
     every piece of its output against a gather from the FULL model's own (already-validated)
     `moments!` output at the same θ: **K matches exactly, bilateral columns match exactly, the France
     column matches exactly, the CM-grid columns match exactly, and the gravity column matches
     exactly** (all `max|Δ| < 1e-12`) — **ALL PASS**. This is a genuine, real closure of the gap: a
     reduced family's FG callback now produces numerically correct output.
   - **Part 2 (real KNITRO solve) does NOT pass** — diagnosed in detail, not merely observed:
     - First found the TRUE production default is `inner_fg_backend=:cm_lookup`
       (`CM_INNER_FG_BACKEND_DEFAULT[]`, `core_exact_hessian.jl:228`), **not** `:dense_reference` as
       every pre-existing D4 test's explicit override had suggested — a separate, not-yet-touched
       O(W·(D-1)) operator-based FG evaluator (`cm_lookup_kernels.jl`). Forcing
       `inner_fg_backend=:dense_reference` (matching what this session's new dense-G reduction
       actually supports) was necessary just to reach the `:dense_reference` code path at all.
     - Then found `build_cm_bin_ctx`'s own `threaded_bins=true` default routes the Hessian callback to
       `hessian_cm_structured_v2!` (the threaded twin) — never given a profiled/gather branch this
       session (see below) — forcing `threaded_bins=false` was necessary to reach the serial path that
       WAS wired.
     - With both forced, hit a `nStatus=-400` (KNITRO's generic "callback error" code) on the real
       `archC_base_state` solve. Root-caused as far as time allowed, via three escalating manual
       checks rather than accepting the code at face value:
       1. Manually called the Hessian callback and the dense-BLAS `arg0`-prep step
          (`_prep_dual_index_for_archC!`/the object's own `PsiObjectiveBundleImplicitMethodBFullA`
          functor, `H[:,2:1+outer_constr_index]*(-x)`) at `x=0` and at a small random `x` — both ran
          without error, and (once `obj.H` was filled the SAME way `inner_loop_internal_archgeneric`
          itself fills it — an earlier attempt using separate scratch buffers gave a false-alarm
          discrepancy from stale/undef memory, corrected) matched an independent hand-derived `q`
          formula to **machine precision** (`max|Δ|≈2.2e-16`). This also settled a real, initially
          confusing question: `H`'s gravity column (the LAST column) is deliberately **excluded** from
          both the dual contraction `q` and the gradient formula (`H[:,3:1+outer_constr_index]`, one
          column short of `G`'s own full width) — confirmed this is the FULL model's OWN pre-existing
          convention too (same "one column short" pattern, `H_FULL` has 50 columns, `λstar` has 47
          entries, not 48), not a bug introduced by this session's reduced-width construction.
       2. Ran a crude, deliberately unsophisticated damped-Newton trace (5 steps, step size 0.5, no
          line search) using ONLY this session's own validated FG/Hessian machinery, starting from
          `x=0`: the objective **diverges catastrophically** (`f: 0 → 1731 → 1.9e34 → 3.2e65 →
          7.5e97` in 5 steps, `|x|` growing to `4e49`) — even though every individual FG/Hessian
          evaluation along the way returned finite, correctly-formed numbers. This strongly suggests
          the reduced dual problem's Hessian is **very poorly conditioned** near `x=0` for this
          particular D4 test point/anchor choice (a large Newton step from a small gradient implies a
          near-singular curvature direction) — KNITRO's own trust-region/line-search machinery is far
          more robust than this crude trace, but a `nStatus=-400` (consistent with an uncaught
          `DomainError` from `Psi!`/`dPsi!`/`ddPsi!` being evaluated at an extreme, out-of-domain `q`
          during KNITRO's own step) is fully consistent with this diagnosis.
     - **Two further, decisive checks (same session, after the above)** narrow this down sharply:
       1. **Warm start does NOT fix it.** Built the exact "zeroed-anchor" point from the FULL model's
          own real converged solve (the SAME point this session's D4 Hessian gate already validated
          numerically) and set it as `obj_reduced.x` (consumed by `CS.inner_loop_initial_values`
          whenever `use_cached_x && norm(obj.x)<1e6`) instead of a cold `zeros` start — still
          `nStatus=-400`, byte-identical failure. This is exactly the outcome this repo's own
          CLAUDE.md standing guidance predicts ("the inner solve's warm/cold start affects speed,
          never whether it converges") — strong evidence this is a genuine problem-level issue, not a
          starting-point artifact, and specifically rules out "just needs a better initial guess" as a
          fix.
       2. **The UNRESTRICTED family's own already-validated reduced solve converges FINE at the exact
          same θ and `AnchorSpec`.** Called `build_profiled_operator_bundle`/`inner_loop_KNITRO_profiled`
          (this branch's own pre-existing, merged, validated unrestricted-family machinery) directly at
          the identical calibration θ and `spec` used throughout this session's flexible-CM work:
          `nStatus=0`, converges normally. **This is the decisive finding**: it proves the reduced
          economic layout itself, this exact anchor choice, and this exact D4 test point are **not**
          inherently ill-conditioned — a model containing ONLY the reduced economic block (no
          restriction columns at all) solves fine. The failure is therefore specific to something
          about how flexible CM's OWN restriction block (the 30 CM-grid columns) interacts with the
          now-smaller (13-column, vs the unrestricted family's own economic-only problem which is
          solved standalone) reduced economic block — most plausibly a genuine relative-scale/
          conditioning interaction between `H_EE`(now smaller) and `H_CC`(unchanged, still full CM-grid
          size) in the assembled Hessian, rather than a further formula bug (`pack_upper_cm_hessian!`/
          `fill_cm_HCC!` were re-read and confirmed fully generic over `NCORE`/`n`, no hardcoded
          assumption found; both are also exercised, unmodified, inside this session's own already-
          bit-identical D4 Hessian gate, via the packed `h` output that gate compares).
     - **Conclusion**: NOT a formula-correctness bug (every formula independently validated to machine
       precision, at multiple hand-set points, including via a completely independent reference
       kernel); NOT a cold-start artifact (warm start doesn't help, consistent with this repo's own
       standing guidance); NOT inherent to the reduced economic layout alone (the unrestricted family's
       own reduced solve converges fine at the identical point). The remaining, unresolved, and now
       much more precisely bounded question is a **numerical-conditioning interaction specific to
       flexible CM's combination of a reduced economic block with its unchanged CM-grid restriction
       block**, at this particular D4 test point — genuinely open, not yet closed. Left as an
       honestly-failing, diagnostic-rich test (not deleted, not silently downgraded) for whoever
       continues this.

## Important correction to this session's own earlier claim (found while building the H_EE/H_EC gate)

Item 1 above states "the dense economic-column materialization is already unconditionally skipped on
the current production path" and concluded no changes were needed to `wrap_moments_with_cm_archB` for
a real FG solve. **This is only true for the dimension-bookkeeping half of the problem** (confirmed by
the dimension gate) — it does **not** mean a reduced family's `obj_cm.moments!` can actually run an
end-to-end FG callback yet. Found while trying to build a fully-wired KNITRO-driven gate (not merely a
Hessian-formula gate): `wrap_moments_with_cm_archB`'s `skip_fill=true` path skips **all** of `G`
(economic block AND the CM-grid restriction columns both — confirmed by direct re-read of
`cm_hessian_architectures.jl`'s `if !skip_fill` guards, both the economic copy and
`fill_cm_columns_from_bins!` are inside the SAME guard), not just the economic columns as a narrower
reading might suggest — `skip_fill=true` is a narrow priming mechanism for the separate `:cm_lookup`/
operator inner-FG-backend machinery (`cm_lookup_kernels.jl`), used only for a specific priming call,
**not** installed as any family's primary `obj_cm.moments!` in the default (`:dense_reference`
inner-FG-backend) configuration. In the DEFAULT configuration (`CM_INNER_FG_BACKEND_DEFAULT[]`), the
primary `moments!` is the `skip_fill=false` variant, which genuinely does call
`materialize_dense_factual_structured!` every callback — and that function, for the reduced case,
would be asked to fill a `pregrav`-width (small, reduced) view while itself requiring
`size(Gview)==(W,cf.oci-1)` (the FULL, unreduced width) — an immediate dimension-mismatch error, not a
silent no-op. **Consequence**: this session's D4 Hessian gate above is legitimate and rigorous for what
it tests (the Hessian-callback formula and wiring, at a hand-set valid dual point) but does **not**
demonstrate a working end-to-end reduced-family KNITRO FG+Hessian solve — that requires either (a) a
new, small, layout-aware dense-G materialization function (mirroring
`materialize_dense_factual_structured!` but writing only `layout`-retained columns), or (b) wiring the
reduced case through the `:cm_lookup`/operator inner-FG-backend instead (a separate, larger
subsystem). Neither is done. This is a real, previously-understated gap — flagged prominently here
rather than left implicit, since acting on the OLD phrasing ("zero changes needed") would wrongly
suggest the reduced family is closer to a real solve than it is.

## What is NOT done — honest accounting against the mission's 18 sections

Everything below is genuinely `not_started`/`not_run`, not "quietly assumed to work":

- **A converging live KNITRO solve for the reduced economic layout** — the FG callback itself is now
  CLOSED and validated correct (item 4 above, Part 1 of the new gate, `:dense_reference` inner-FG-
  backend only), but a real solve from a cold start still fails (`nStatus=-400`), diagnosed as likely
  a conditioning issue (item 4's Part 2) rather than a formula bug, and not yet resolved. The TRUE
  production default (`:cm_lookup`) is separately, entirely unaddressed.
- **`hessian_cm_structured_v2!`** (the threaded twin, `cm_hessian_threaded.jl`): not given the
  analogous profiled/gather branch — only the serial `hessian_cm_structured!` was touched this session.
- **H_EF (`colsum!`/`esum!`) gather for common Fréchet**: not implemented (explicitly guarded to
  `error()` rather than silently mishandled if `extension!==nothing` is combined with
  `profiled_layout!==nothing`); still needs the "open question" from
  `PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §4 resolved by reading `CMFrechetExtension`'s actual
  `Wtab`/`T1` construction.
- **H_EZ gather for ZC-only and CM+ZC**: not implemented — `OriginZCCoreHessCtx`/
  `archA_partitioned_hess_cb_builder` are architecturally separate from `CMBinHessCtx`/
  `hessian_cm_structured!` (per the earlier research pass) and were not touched this session; would
  need their own (structurally analogous, but separately written and separately gated) field
  additions and gather branch.
- **CM+ZC's mean/pair-widened H_EE/H_EM**: explicitly out of scope this session (`_fill_cm_HEE!`'s new
  branch asserts `ncore_core==NCORE`, i.e. errors fast rather than silently mishandling the widened
  case).
- **Dual bounds / initial dual / moment names / verification slices** (mission §5's explicit checklist):
  not audited. `outer_constr_index`/`d` bookkeeping is confirmed generic and correctly reduces (this
  session's own gate), but KNITRO variable-bound arrays, moment-name diagnostics, and the
  `_verify_inner_solution_operator_cm_core`-style verification path were not checked for hardcoded
  `D*Ddest`-shaped assumptions — likely candidates for a further latent bug, on the pattern of this
  session's own `Bidx` type-dispatch finding.
- **Every D20/inner-equivalence/outer-gradient/performance/W500k gate the mission asks for (§9-21)**:
  status as of THIS section's own scope is `not_run` for all of them; see the verdict block below for
  whether a D20 cross-block gate was reached later in this same session (it reflects the true final
  state; this section is a narrative walkthrough of the flexible-CM H_EE/H_EC work specifically).
- **`FULL_FORMULATION_NO_OVERHEAD_GATE_2026-08-01.csv`** (mission §3): not run. Worth noting the
  profiled-only scratch fields (`MTab`/`MCScum`/`SnuWval`/`TZ_buf`/etc.) were already confirmed, by the
  SOURCE branch's own prior session, to be filled unconditionally but cheaply (one extra
  multiply-add per existing loop iteration, no new O(W) pass) — this session did not re-verify that
  claim or measure allocations/wall-time directly. The NEW `profiled_full_ws`/`profiled_hraw_ec_full`
  scratch this session added is sized to the FULL (unreduced) width and lives on `cctx`, but is only
  ever touched when `cctx.profiled_layout!==nothing` — a non-profiled `cctx` never allocates or fills
  it, preserving the "no overhead on the old path" property by construction, though this was not
  independently allocation-profiled.

## Verdict block (mission's own format)

```text
PRIMITIVE_CROSS_BLOCK_FORMULAS =
    validated   # unchanged from source branch: H_EC/H_EF/H_EZ D4-validated vs independent brute force

OLD_FULL_PATH_OVERHEAD =
    not_remeasured_this_session   # source branch reasoned this is fine (same complexity class); not empirically re-checked here

GENUINE_REDUCED_LAYOUT =
    unrestricted:      pass                          # pre-existing, source branch, unchanged
    flexible_CM:        fg_and_hessian_formulas_pass_live_solve_fails_400_isolated_to_HEE_HCC_interaction
    common_Frechet:      not_started   # shares flexible CM's plumbing but H_EF gather not built
    ZC_only:             not_started
    CM_plus_ZC:          not_started

HESSIAN_ORCHESTRATOR_WIRING =
    flexible_CM: pass_serial_only_D4_verified   # hessian_cm_structured! only; _v2! (threaded) not touched
    common_Frechet: fail_H_EF_not_implemented
    ZC_only: fail_not_started
    CM_plus_ZC: fail_not_started_and_out_of_scope_widened_case

D4_COMPLETE_HESSIAN =
    flexible_CM: pass   # test_profiled_flexcm_d4_hessian_gate_2026-08-01.jl, H_EE+H_EC, max|Δ|=0.0,
                         # both contrasts x both L x 2 points; H_EF/H_EZ/other families not_run
    common_Frechet: not_run
    ZC_only: not_run
    CM_plus_ZC: not_run

D20_CROSS_BLOCK_REFERENCE =
    not_run

INNER_EQUIVALENCE =
    not_run   # the FG-callback gap is now closed (moments! validated correct, Part 1) but a live
              # KNITRO solve of the reduced dual problem does not yet converge (nStatus=-400,
              # diagnosed as likely a conditioning issue, not a formula bug -- see above) -- no
              # reduced-family inner solve exists yet to test equivalence of

SHARED_PROFILED_A_GP_GRADIENT =
    not_started   # unchanged from source branch

RESTRICTION_ONLY_CODE_CHANGED =
    none   # confirmed: only cm_hessian_architectures.jl (additive base_obj/theta_ref/profiled_layout
           # kwargs on build_cm_augmented_obj_archB/wrap_moments_with_cm_archB/build_cm_bin_ctx, new
           # CMBinHessCtx fields via its own append idiom, new early branches in _fill_cm_HEE!/
           # hessian_cm_structured! gated on profiled_layout!==nothing, one pre-existing Bidx-dtype
           # bugfix) and two new additive files touched; H_CC/H_CF/H_FF/H_CZ/H_ZZ/CM bins/Fréchet
           # level moments/Z features/restriction FG/verification/restriction-parameter gradients:
           # byte-for-byte untouched -- confirmed by zero regression on all 3 pre-existing/prior D4
           # test files after every change this session

ZC_OPTIMIZATION_INTEGRATION =
    pending   # this session made no contact with that separate workstream; nothing to integrate yet

W500K_PUBLIC_ENTRY_SMOKE =
    not_run

PRODUCTION_RECOMMENDATION =
    insufficient_evidence   # unchanged from the source branch's own honest verdict; this session did
                             # not add outer-loop A/B evidence, only infrastructure/correctness gates

PRODUCTION_DEFAULT_CHANGED = false
CAMPAIGN_LAUNCHED = false
```

## Recommended next steps (in order, for whoever continues this)

1. **Isolate the nStatus=-400 conditioning issue** (the single largest remaining blocker now) —
   narrowed THIS session to "an interaction between the reduced H_EE and flexible CM's unchanged
   CM-grid H_CC block", via two decisive checks already run (warm start does not help; the
   unrestricted family's own reduced solve converges fine at the identical θ/AnchorSpec — see above).
   Next: (a) compare `eigvals`/condition number of the assembled reduced Hessian's `H_EE` block vs its
   `H_CC` block at the SAME dual point, to see if one dominates/vanishes relative to the other once
   the economic block shrinks; (b) try a SMALLER `L` (fewer CM-grid columns, e.g. `L=2`) to see if the
   failure is sensitive to the CM-grid's own relative size; (c) try `:orthonormal` vs `:anchored`
   contrasts (both already exercised in this session's Hessian-formula gates, but not in the live
   solve) in case one is better-conditioned; (d) as a structural check, try whether `KN_set_var_scalings_all`
   (used successfully elsewhere in this repo per memory `melitz-real-d20-scaled-knitro-native-scaling`)
   changes the outcome — but only AFTER (a)-(c) narrow the actual cause, not as a first resort.
2. Once (1) yields a converging solve: re-run this session's
   `test_profiled_flexcm_d4_hessian_gate_2026-08-01.jl`-style comparison but at a GENUINELY SOLVED
   reduced-model point (not a hand-set one) — a real, if narrow, version of the mission's §9
   inner-equivalence gate for flexible CM specifically.
3. Thread the profiled/gather branch into `hessian_cm_structured_v2!` (threaded twin,
   `cm_hessian_threaded.jl`) — mechanical once (1)-(2) are solid, but not yet done or gated.
4. Wire the reduced layout through `cm_lookup_kernels.jl`'s `:cm_lookup` operator FG evaluator (the
   TRUE production default, `CM_INNER_FG_BACKEND_DEFAULT[]`) — this session only closed the
   `:dense_reference` FG path; production itself does not default to that path.
5. Resolve H_EF's open Wtab/T1 question (`PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md` §4) and
   implement the analogous gather branch for common Fréchet (which already gets the dimension/H_EE
   plumbing for free via shared `CMBinHessCtx`, per this session's design).
6. Port ZC-only (`OriginZCCoreHessCtx`/`archA_partitioned_hess_cb_builder`) and then CM+ZC's widened
   case — structurally analogous to this session's flexible-CM work but a SEPARATE codebase surface,
   not automatically covered by anything done so far.
7. Only after (1)-(6): D20 cross-block/inner-equivalence gates, outer-gradient sharing, performance
   profiling, W500k smokes, per the mission's own ordering — each is real, separately gate-able work.

Given the source branch's own prior verdict (`PORT_TO_RESTRICTED_FAMILIES = insufficient_evidence`, a
real-but-small 3.85-7.99% unrestricted-only outer-loop edge from 3 points/one seed/upper-direction-only)
was already judged too thin to justify the FULL remaining campaign before this session started, and
this session's own contribution is infrastructure/correctness-gate work rather than new outer-loop
evidence, that judgment call — whether to continue investing in the full five-family port before more
unrestricted evidence exists — still stands open for the user, not resolved here.
