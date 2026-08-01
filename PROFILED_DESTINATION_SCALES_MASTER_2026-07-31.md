# Profiled Destination-Scale Reparameterization — Master Report (Phase 1)

Branch: `architecture/profile-all-destination-scales-2026-07-31`
Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profile-all-destination-scales-2026-07-31`
Base: `production/fullA-exact @ cd17235`

## CORRECTION NOTICE (2026-07-31, mid-session, user stop)

Every D=4 gate in this session was originally implemented against the **legacy dense `G`/`H`/`K`
moment matrix** (`ctx.obj.moments!`, only present because `d4_exact_setup()` attaches a
pre-hardening `PsiObjectiveBundleImplicit`), and repeatedly, wrongly, described as "live production
code" in this report and in commit messages. The user stopped the session to correct this: *"The
code does not use G or PsiObjectiveBundleImplicit. We use newer bundles that do not define G or H
or K... I worked very hard to remove all reference to H and G and those old bundles."* This
session's **own earlier audit** (§3a(i) below) had already found exactly this — "no dense G/H...
operator-only hardening 2026-07-25 through 2026-07-30" — and should have changed which interface
every gate used from the start; it didn't, which is the actual failure here, not a lack of
information.

**Fix, commit `408a4b1`**: every helper and every test file rebuilt to read
`build_compressed_factual(θ_full, ctx)` (`compressed_moments.jl`) directly — the real
winner-compressed representation production's operator path uses — instead of reconstructing
values by inverting dense-`G` post-processing. Every numerically-asserted finding was re-run end to
end and reproduced **identically or better** on the corrected path (exponent test 4.4e-16 vs. the
original 6.7e-16; recovery gate 2.2e-16 exact, unchanged; homogeneous-moment identity 1.78e-15,
unchanged; France ratio-moment rescaling 2.7e-12, unchanged). The underlying mathematics was never
wrong — the verification code path was. All narrative below is left as originally written except
where explicitly marked corrected, since the mathematical content and conclusions did not change;
only the phrase "live production code" throughout should now be read as referring to the
*corrected* gates.

## 0. Scope of this session, and why

The full task specification is 26 sections covering: theory proof, a full call-graph audit, an
anchor manifest, a ground-up reduced economic-coordinate reparameterization touching the FG
forward/transpose operators, `H_EE`, every economic×restriction cross-Hessian block, the outer C+
gradient, every feasibility screen, calibration conversion, exact full-A recovery, independent
full-system re-verification, checkpoint/export schema, a campaign driver, D=4 gates, D=20 gates at
W=100k and W=500k, a performance profile, and a numerical-conditioning study — each with a
dedicated written derivation or gate-result deliverable, machine-precision equivalence
requirements, and (for the D=20 gates) real KNITRO solves that this project's own history shows
take from minutes to tens of minutes each, often across multiple starts and five distributional
families.

This is, by the standards of this project's own history (see e.g. the "Winner-aware H_ER phase"
and "structured cross-Hessian" efforts in prior sessions, each a full dedicated session covering
one Hessian sub-block and still landing "not merged"), a multi-day engineering effort. One
non-interactive session cannot respons­ibly claim to complete it with the correctness guarantees
the task itself demands (machine-precision D4/D20 equivalence, real full-system recovery checks)
without either fabricating results or taking exactly the kind of unverified-assumption shortcut
this repository's own `CLAUDE.md` explicitly and repeatedly warns against (the standing
"`A_od≡1` is not calibration" note exists precisely because past sessions asserted numerical
equivalences without actually reconstructing and diffing the objects involved).

Given explicit instruction to push as far as possible and accept the session ending short of full
completion, this session's actual scope was: **Phase 1 — theory, audit, and anchor
specification**, done rigorously and code-grounded, with every downstream implementation section
left honestly marked incomplete rather than stubbed or faked.

## 1. Starting-condition note: the "Brazil→Korea gravity exclusion" premise

The task's opening instruction assumes a prior "Brazil→Korea gravity exclusion" merge as a
starting precondition. A full search of this repo (`git log --all`, all branches/tags, every
`.jl` file) at session start found **no trace of it** on the locally-fetched `production/fullA-exact`
ref. A subsequent `git fetch` revealed `origin/production/fullA-exact` had in fact moved ahead
(commits `e622366`/`ac00051` etc., "Brazil-Korea gravity-exclusion task") — i.e. the work exists,
but on a tip the user identified as belonging to another, concurrently-running Claude session that
had "broken something ... still trying to fix it," and the user explicitly said to ignore it as
unnecessary for this task's goal.

**Decision taken**: this branch is cut from the last **locally stable** `production/fullA-exact`
point, `cd17235`, predating that concurrent work, rather than from the live (potentially unstable)
`origin` tip. Section 3's own anchor-manifest requirements (own-cell + Brazil→Korea gravity
ineligibility) are satisfied by this task's own from-scratch anchor spec regardless of that other
branch's state — see `DESTINATION_SCALE_ANCHOR_MANIFEST_2026-07-31.json`. That other branch's
`gravity_sample_mask()` utility was inspected read-only (`git show`, not merged) and is the exact
right shape of primitive to reuse once stable — flagged as a concrete follow-up rather than
depended upon.

## 2. What was verified this session (see `PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md` for full detail)

- Production is already fixed-theta by default at real D=20 (`μ`,`σ` box-constrained equal,
  neither in `free_idx`) — no gating decision needed for task §6's fixed/flexible fork.
- The exact homogeneity exponent governing a common destination-column rescale of the working
  `z=log(Aod_theta)` coordinate is **`μ(σ-1)`**, re-derived from and cross-checked across three
  independent files (`fast_range_screen.jl`, `compressed_moments.jl`, `gravity_elimination.jl`).
  This **replaces** the task brief's own speculative, explicitly-flagged-as-unverified aside
  (`σ-1` / `1/(σ-1)`) with the code-derived value.
- Winner identities and factual share ratios are proven exactly invariant to a destination-column
  gauge shift, using that exponent (§2.1 of the theory doc) — the core mathematical claim the
  whole reparameterization rests on.
- **`ρ_f` — corrected mid-session.** An earlier finding here claimed `ρ_f = gp` (identity), based
  on `K[:] = γ_prime_bi`. That was wrong: tracing `K`'s actual consumer
  (`cc_algo/PsiObjectiveBundle.jl`) shows `K` is the **outer KNITRO objective value** (`gp` itself,
  being extremized), never read by the inner-dual/constraint computation at all — not an economic
  moment. The real France autarky moment is `G`'s own `cf_col` (confirmed an inner-dual column at
  D=4: `outer_constr_index=obj.d=18`, gravity alone is the outer column), whose target is
  `gp^σ·wPrime_bi·LPrime_bi`. **The task brief's original `ρ_f=gp^σ` hypothesis was closer to
  right than the earlier "correction" claimed** — the homogeneous-ratio-moment coefficient this
  implies is exactly `gp^σ` (theory doc §2.5, D=4-gated to 4e-12: `homogeneous_france_moment`
  scales exactly by `κ^{μ(σ-1)}` under a France-column rescale). Left visible in the theory doc as
  a documented correction rather than silently edited.
- The existing gravity-pivot mechanism (`gravity_elimination.jl`) removes one **global** scalar
  coordinate via a linear constraint across the whole `D×D_dest` block; the destination-anchor
  reduction this task implements removes **19** coordinates via a *different*, per-column
  multiplicative-invariance argument. The two compose (19 anchors removed, then the existing
  1-coordinate gravity pivot applied to what remains) but are not the same mechanism and must not
  be conflated.
- Task §7's required "exact weighted identity" (a destination-column shift is exactly
  gravity-invisible) is now **fully proved**, not just asserted for anchor cells: it follows
  directly from `within_transform_rect`'s two-way fixed-effects annihilator being symmetric,
  idempotent, and annihilating any destination-constant vector, combined with the FWL orthogonality
  identity `gravity_tariff.jl` already establishes for a different purpose. See theory doc §2.1(c).
- Live dimension audit (D=20, `destination_sample=:exclude_row` ⟹ `D_dest=19`, both figures read
  from `real_data/noah_D20/countries.csv` and `context_real_d20.jl`, not hardcoded): active A
  `20×19=380`; after removing 19 anchors, `361`; after the existing 1-coordinate gravity pivot,
  `360`. This independently reproduces the task brief's own predicted `380→361`, `379→360` figures:
  the brief's `379` is the *pre-anchor-reduction* free-A count (`380-1`, already net of the existing
  gravity pivot), reducing to `379-19=360` — a different subtraction order than this audit's
  `380-19=361`, `361-1=360`, but both orders agree on the two totals that matter (`380` active
  cells, `360` final free coordinates), so there is no real discrepancy once the order is made
  explicit.
- Country indices resolved live from `real_data/noah_D20/countries.csv`: France=2, Brazil=3,
  Korea=14, ROW=20 (excluded destination).

## 3. Deliverables produced this session

- `PROFILED_DESTINATION_SCALE_THEORY_2026-07-31.md` — formal proof, §2.1 fully derived and
  code-verified; §2.2–2.4 stated formally with exact exponents but not yet numerically executed.
- `DESTINATION_SCALE_ANCHOR_MANIFEST_2026-07-31.json` — complete 19-entry anchor map
  (France→France, Korea→Brazil, all others own-cell), live-resolved indices, requirements
  checklist with honest per-item status (2 pass-by-construction, 3 pending/not-yet-implemented).
- `FULL_TO_PROFILED_PIPELINE_CALL_GRAPH_2026-07-31.md` (493 lines, 16 topics) /
  `FULL_TO_PROFILED_CHANGE_MATRIX_2026-07-31.csv` (100 rows) — full call-graph audit, covering
  every topic the task specified. Read in full and cross-checked, not taken on faith; see §3a below
  for the two findings that materially change the implementation plan.
- This master report.

### 3a. Two audit findings that change the implementation plan

**(i) CORRECTED (see task §11 commit `2194c8a`, later in this session): this finding was WRONG for
`H_EE`, and the `H_EC`/`H_EZ`/`H_CZ` part is unverified, not confirmed.** This section originally
claimed Topics 6–8 (economic operator forward/transpose, `H_EE`/winner-pair, and every
`H_EC`/`H_EZ`/`H_CZ` cross-block) all classify as **unchanged**, reasoning they operate entirely on
`CompressedFactual`/`WinnerPairHessCtx`, decoupled from A-coordinate layout. That reasoning
overlooked that `build_winner_pair_ctx` (the `H_EE` constructor) reads `cf.denom[slot]` directly —
the exact same fixed-target dependency the FG kernels turned out to have (§9, commit `4178527`).
Once actually implemented and gated, `H_EE` needed genuinely new accumulators (not present in the
original kernel at all), a materially bigger fix than the one-line FG swap. **The `H_EC`/`H_EZ`/
`H_CZ` cross-blocks have NOT been re-checked against this corrected understanding** — do not trust
their "unchanged" classification either without a direct read of their construction, the same way
`H_EE`'s classification turned out not to hold up. Economic-operator forward/transpose (Topic 6,
`economic_forward!`/`economic_transpose!` themselves, as opposed to the kernels they call) remains
accurately classified as thin wrappers — that part of the finding was correct.

**(ii) Bad news — the real complexity is a pre-existing duplication problem, not a new derivation.**
The "outer A-coordinate → gauge-normalized level A" reconstruction formula
(`Aod = Aod_θ .* cHat .* ((wHat·τ)/(wHat[1,1]·τ[1,:]'))^(1/μ) .* (λ/λ[1,:]')`) is **copy-pasted
independently in at least 15 files** (`moments_gammanorm.jl`, `moments_fast.jl`, `autarky_cf.jl`/
`_v2.jl`, `gravity_elimination.jl`, `winners.jl`, `cm_aspace_coordinate.jl`, screen/oracle files),
not centralized behind one function the way the gravity pivot is. It also already contains an
**existing, different** per-destination gauge device — dividing every destination-`d` column by
`λ[1,d]` (origin 1's factual share) — which must not be confused with the *new* per-destination
anchor this task introduces (France→France, Korea→Brazil, else own-cell): they are different
objects, and the existing one does not remove any free coordinate today (all `D·D_dest` A cells
remain in `free_idx` currently; the `λ[1,d]` divide is purely a reconstruction-formula detail, not
a coordinate-count reduction). Similarly, `free_idx` construction itself (`[gp; every A_od cell]`)
is independently duplicated in exactly 4 context builders (`context.jl`, `context_scaled.jl`,
`context_real_d20.jl`, `qmc_context_real_d20.jl`). **Practical implication**: before writing the
relative-A coordinate layer (§6), there is a real choice between (a) implementing the new
per-destination anchor logic independently in all ~15+4 sites (correctness risk: the same logic
must be replicated correctly that many times), or (b) consolidating the gauge-reconstruction
formula into one shared function first (mirroring what `gravity_elimination.jl`/
`outer_coordinate_layout.jl` already did for the pivot), then building the anchor layer on top of
that single site. (b) is recommended but is itself extra, not-yet-scoped work.

Two smaller items from the audit: no `rho_from_gp`/`GT_from_gp` functions exist anywhere in this
repo (task brief's assumed names were unfounded, independently confirming this session's own
earlier finding); and the concurrent Brazil-Korea task's `gravity_sample_mask` utility is confirmed
**absent** from this worktree's base (`cd17235`) by a repo-wide search — consistent with this
branch predating that work, as documented in §1 above.

## 4. Implementation progress this session (task §§6–7, continuation after initial Phase 1)

Following user direction to continue, this session went beyond audit/theory into real, additively
implemented and D=4-gated code — all following this repo's established "new function, verify vs
trusted path, then wire in" convention, touching zero existing production files:

- **§6 relative-A coordinate layer** (`relative_a_coordinate_2026-07-31.jl`): `AnchorSpec` (type
  itself enforces "exactly one anchor per destination," task §2.4, not just a runtime check),
  `decode_relative_A`/`encode_relative_A`, `build_anchor_gauge`. Deliberately operates purely in
  `z=log(Aod_theta)` space so it never touches or duplicates the >15-site gauge-reconstruction
  formula §3a(ii) flagged as this reparameterization's main risk. D=4 gate
  (`test_relative_a_coordinate_2026-07-31.jl`, all tests pass): round-trip exact to 1e-16;
  gauge-from-genuine-calibration round-trip exact to **0.0**; end-to-end integration through a real
  `θ` vector into the live `ctx.obj.moments!` reproduces genuine calibration **bit-for-bit**;
  tested with a non-own anchor (D=4 analog of Korea→Brazil), not just the own-cell default.
- **§7 gravity pivot composition** (`gravity_pivot_on_retained_2026-07-31.jl`): pivot selection
  restricted to retained (non-anchor) cells, operating in the reduced `r`-space by composing two
  exact affine maps (gravity-in-z, proven in `gravity_elimination.jl`; z-in-r, proven in the
  relative-A layer above). D=4 gate (`test_gravity_pivot_on_retained_2026-07-31.jl`, all pass):
  **dimension audit confirms `D·Ddest-Ddest-1=11` free coordinates at D=4, live-derived and matching
  the general formula exactly**; composed pivot cell confirmed never an anchor cell; round-trip
  exact; gravity residual at true machine precision (~1e-18) over 5 random composed points; anchor
  cells confirmed fixed under every composed point; a fully-composed point feeds the live
  `moments!` cleanly end-to-end.

- **§16/theory §2.2 exact full-A recovery** (`recover_full_a_2026-07-31.jl`): `destination_M_d`
  (reconstructs per-draw `M_d(ω)` from the live `moments!` output) and
  `recover_gamma_normalized_full_A` (`c_d = γ̃_d^{-1/(μ(σ-1))}` rescale). D=4 gate
  (`test_recover_full_a_2026-07-31.jl`, all pass): built a genuinely non-gamma-normalized working
  point (`γ̃_d` scattered `0.976`–`1.029`), confirmed recovery drives `γ̃_d → 1` to **2.2e-16**
  (exact) for every destination, while winner identities and per-draw share ratios are unchanged to
  `~1e-16`–`1e-19`.

These three gates are real evidence (not just derivation) that the composed anchor-reduction +
gravity-pivot reparameterization, and the recovery map back to the full formulation, are internally
consistent against production code at machine precision. This closes theory §2.2's previously
open "recovery construction unexecuted" gap.

- **§8 homogeneous factual moment** (`homogeneous_moments_2026-07-31.jl`): `Q_od(ω)-λ_od·M_d(ω)`,
  replacing the old formulation's fixed `denom[d]` with the model's own `M_d(ω)`. D=4 gate (all
  pass): `Σ_o` of the moment is **exactly** 0 for every draw (1.8e-15, confirms task §1.2's
  "omitted anchor share follows automatically" algebraically); the moment rescales by **exactly**
  `κ^{μ(σ-1)}` under a destination shift (6.7e-16–8.9e-16). This also resolved an open question:
  the OLD absolute moments (fixed `denom[d]`) actually *do* pin the destination scale uniquely —
  it's specifically swapping to this homogeneous form that makes the scale genuinely unidentified.
- **§1.3 France ratio moment** (`homogeneous_moments_2026-07-31.jl`'s `homogeneous_france_moment`)
  — built after finding and correcting a real mistake mid-session: an earlier claim that
  `ρ_f = gp` was wrong (it conflated `K`, the **outer KNITRO objective value**, with the actual
  France moment's target, `G`'s own `cf_col`). The corrected target has the `gp^σ` structure the
  task brief originally guessed. D=4 gate: cross-checked against the real production `G` column;
  exact proportional rescaling under a France-column shift confirmed to 4e-12.

- **§6/§10 (Topic 1/10) KNITRO-facing profiled outer vector**
  (`outer_coordinate_layout_profiled_2026-07-31.jl`): `decode_outer_profiled`/`reduce_to_w_profiled`,
  layered on top of the unchanged `outer_coordinate_layout.jl`, producing the exact `xf` shape
  `decode_outer_unified`/screened-evaluation entry points already expect from a genuinely shorter
  outer vector. D=4 gate: profiled outer vector is exactly `Ddest=4` shorter than full (`12` vs
  `16`); round trip from genuine calibration reproduces `Aod_theta` to `7e-15`; the resulting `xf`
  feeds `build_compressed_factual` cleanly; round trip and gravity-feasibility hold at random
  (non-calibration) points too. This closes the loop on making the reduced coordinate system
  genuinely usable by production code, not just testable in isolation.

- **§9 FG forward/transpose** (`homogeneous_contraction_2026-07-31.jl`) — implemented after
  correcting a second mischaracterization mid-session: an earlier response described the remaining
  work as "assembling a reduced `moments!`-equivalent function," which is still legacy-bundle
  thinking (the user correctly pushed back). The actual production kernels
  (`economic_forward!`/`economic_transpose!`) call `compressed_dual_contraction!`/
  `compressed_transpose_contraction!` (`compressed_moments.jl`), which encode the OLD absolute
  target as one fixed scalar `cf.denom[slot]` multiplied by a global weighted count, pulled outside
  the per-draw loop. The homogeneous moment requires moving that term inside the loop and
  multiplying by the model's own per-draw `cf.wval[w,slot]` (`== M_d(w)` exactly, already a
  `CompressedFactual` field) instead — a small, targeted, well-understood change to two existing
  kernels, not a new `moments!`-shaped function. D=4 gate (`test_homogeneous_contraction_2026-07-31.jl`):
  both new kernels cross-checked against direct `G_new*β`/`G_new'*weights` computation (using the
  independently-verified `homogeneous_factual_moment`/`homogeneous_france_moment` functions to build
  the reference), diff `~1e-15`–`2e-13`; forward/transpose independently confirmed exact mutual
  adjoints (`~4.7e-14`), a check with no dependency on the hand-rolled reference at all.

Together, §6/§7/§8/§1.3/§9/§10 now cover the coordinate layer, the per-cell homogeneous moment
definitions, AND their integration into the real compressed-contraction kernels the production
operator path actually calls. What remains is wiring these kernels behind
`economic_forward!`/`economic_transpose!` themselves (gated behind an explicit parameterization
choice, not default-on), the Hessian side (§11–§12, likely largely unchanged per the audit),
the outer gradient (§13), screens (§14), and an actual KNITRO inner-solve comparison (§2.3).

## 5. Recommended next steps (in dependency order)

Done this session (all D=4-gated on the corrected `build_compressed_factual` path): consolidation
question resolved for the coordinate layer (§6 operates purely in z-space, independent of the
Topic-2 duplicated formula); §16 exact recovery; §6/§7 composed anchor+gravity-pivot coordinate
layer; §10 KNITRO-facing outer vector; §8 homogeneous moment + §1.3 France ratio moment; §9 FG
forward/transpose wired against the real `compressed_dual_contraction!`/
`compressed_transpose_contraction!` kernels.

Remaining:

1. Wire `homogeneous_dual_contraction`/`homogeneous_transpose_contraction!` *behind*
   `economic_forward!`/`economic_transpose!` themselves, gated by an explicit
   `economic_parameterization` choice (task §5) — currently they're correct, tested, callable
   functions, but a real driver still calls the original (unchanged) kernels by default.
2. Resolve the `gravity_sample_mask` reuse-vs-reimplement question once the concurrent Brazil-Korea
   task's branch is stable (own-cell + Brazil→Korea eligibility masking is needed by both efforts).
3. Re-scope §§11–12 (`H_EE`, cross-Hessian) against the audit's finding that those kernels are
   largely `unchanged` already (they consume `cf`/`WinnerPairHessCtx` generically) — needs a direct
   check now that §9's kernels exist, not just the audit's structural argument.
4. §13 outer gradient (`composite_gradient_at_Cplus` — the one genuinely new derivative-bookkeeping
   site) and §14 screens (contained per the audit, inherit the Topic-2 dependency but need no
   anchor-aware logic of their own).
5. Execute theory §2.3 (comparison theorem) — requires an actual KNITRO inner solve, not just
   direct evaluation (which is all every gate this session needed), a materially larger lift than
   anything done so far, and depends on (1)-(4) existing first to have a real reduced inner problem
   to solve.

## 6. Final verdict block

```
THEORY =
    partial_verified | core_invariance_exponent_gravity_zero_contribution_AND_recovery_construction_all_NUMERICALLY_confirmed_D4 | comparison_theorem_unexecuted_requires_KNITRO

ANCHORS =
    France:France
    Korea:Brazil
    all_other_active_destinations:own
    one_per_destination:pass (enforced at the TYPE level in AnchorSpec, not just a runtime check)

MOMENT_SYSTEM =
    factual_homogeneous:pass_D4 (Q_od-lambda_od*M_d; Sigma_o==0 exact 1.8e-15; rescales exactly by
        kappa^mu(sigma-1), 6.7e-16-8.9e-16)
    anchor_share_implied:pass_D4 (same gate: Sigma_o H==0 exactly proves the omitted anchor share
        follows automatically, not just asymptotically)
    no_factual_gamma_moment:not_applicable_by_construction (the homogeneous moment never
        references gamma_d at all, so there is nothing to omit)
    France_ratio_moment:pass_D4 (Phi_ff-gp^sigma*M_f; corrects an earlier in-session mistake, see
        master report §4 for the correction; exact rescaling confirmed to 4e-12)
    NOTE: components are built and individually gated; not yet assembled into one drop-in
        moments!-equivalent function with the reduced column count

DIMENSIONS =
    active_A:380->361 (live-derived, D=20/D_dest=19; D=4 analog 16->12 numerically confirmed)
    free_A_after_gravity:360 (D=4 analog 12->11 numerically confirmed via test_gravity_pivot_on_retained)
    economic_moments:380->361

COORDINATE_LAYER (task section 6) =
    relative_A_encode_decode:pass_D4 (round-trip exact 1e-16; calibration round-trip exact 0.0;
        bit-exact end-to-end through build_compressed_factual, the operator-representative path)
    gravity_pivot_composition (task section 7):pass_D4 (dimension audit exact; pivot never an
        anchor cell; gravity residual ~1e-18 at 5 random points)
    KNITRO_facing_outer_vector (task section 10, Topic 1/10):pass_D4 (decode_outer_profiled
        produces the exact xf shape decode_outer_unified/screened-evaluation code already expects,
        from a vector exactly Ddest shorter; round trip 7e-15 at calibration, exact at 5 random
        points; xf feeds build_compressed_factual cleanly -- the coordinate layer is now genuinely
        usable by production-shaped evaluation code, not just tested in isolation)

FG =
    forward:pass_D4 (homogeneous_dual_contraction, real compressed kernel algebra -- see
        homogeneous_contraction_2026-07-31.jl; cross-checked against direct G_new*beta at
        ~1e-15-3e-15, three random beta draws)
    transpose:pass_D4 (homogeneous_transpose_contraction!, same file; cross-checked against direct
        G_new'*weights at ~4e-14-2e-13; forward/transpose independently confirmed exact mutual
        adjoints, diff ~4.7e-14)
    NOTE: implemented as new parallel functions alongside the UNCHANGED real production kernels
        (compressed_dual_contraction!/compressed_transpose_contraction!, compressed_moments.jl)
        that economic_forward!/economic_transpose! actually call -- not a new moments!-shaped
        function (a mischaracterization corrected mid-session, see the user exchange this commit
        follows). Not yet wired behind economic_forward!/economic_transpose! themselves (that
        wiring would be the next step, gated behind an explicit parameterization choice per task
        §5, not a default-on change).

HESSIAN =
    H_EE:pass_D4 (CORRECTS the audit's "likely unchanged" claim -- build_winner_pair_ctx DOES
        read cf.denom[slot] directly via pi_vec, same dependency as FG. Fix is materially bigger
        than FG's one-line swap: two new O(W*Ddest)+O(W*Ddest^2) accumulators (T1 for the
        zeta-lambda row, R2/R4 for the lambda-lambda block), QQ itself unchanged. D4 gate
        (test_homogeneous_hessian_2026-07-31.jl): verified against a finite-difference Hessian of
        the already-verified homogeneous_dual_contraction's own objective (an independent method,
        not a second hand-derived formula) -- caught a real bug on first run (missing kappa0[j]
        scale factor, 30% error), fixed, re-verified to 4.05e-9 vs FD)
    H_E_CM:not_implemented (audit's "unchanged" claim for cross-Hessian blocks NOT yet re-checked
        against H_EE's corrected finding -- do not trust the audit's classification here either
        without a direct check)
    H_E_Frechet:not_implemented (same caveat)
    H_E_ZC:not_implemented (same caveat)

OUTER_GRADIENT =
    Cplus:not_implemented
    gravity_pivot:not_implemented
    gp:not_implemented

FULL_RECOVERY =
    gamma_one_all_destinations:pass_D4 (2.2e-16, non-focal destinations + baseIndex both tested;
        real D=20/all-19-destinations not yet run)
    all_full_shares:pass_D4 (share ratios unchanged by recovery, ~1e-16-1e-19; CompressedFactual
        object itself not yet touched, see theory doc caveat)
    all_anchor_shares:not_run (algebraically immediate from Sigma_o lambda_od=1 but not
        numerically checked against a CompressedFactual target vector)
    France_autarky_ratio:not_run
    gravity:pass_D4 (via the gravity-pivot-composition gate, ~1e-18)
    objective:not_run (requires an actual inner solve, not just moments! evaluation)

EQUIVALENCE =
    D4:partial (coordinate-layer, gravity-pivot-composition, recovery, homogeneous-moment,
        France-ratio-moment, KNITRO-facing-outer-vector, FG forward/transpose, AND H_EE sub-gates
        ALL pass at machine/FD precision on the corrected build_compressed_factual path, 10 gate
        files, 0 failures on re-run; full inner-solve equivalence, theory section 2.3, not yet
        attempted -- requires wiring the new FG/H_EE kernels behind the real production callbacks,
        H_EC/H_EZ/H_CZ (unverified, do not trust the audit's "unchanged" label), the outer
        gradient, AND a real KNITRO inner solve, the remaining pieces
        needed to call D4 fully pass)
    D20_W100k:not_run
    D20_W500k:not_run
    all_families:not_run

OPERATOR_ONLY_GATE =
    pass_informal (the repo's own static_bundle_guard_2026-07-30.sh, pulled read-only from the
        concurrent Brazil-Korea branch where it lives -- not merged onto this branch's base --
        was run by hand against this branch's full_aod_diag/d4_exact/ and reports 0 violations:
        no hardcoded :dense_reference default, no direct PsiObjectiveBundleImplicit construction,
        no select_G_from_H use, no static bundle_type=OperatorPsiBundle claim, in any non-test
        file this session added. Not "formal" because the guard script itself isn't part of this
        branch's own history/CI -- but it is the actual repo-authored check, run for real,
        confirming the post-correction (commit 408a4b1) state.)

PRODUCTION_DEFAULT =
    full_gamma_normalized_reference

BRANCH_STATUS =
    incomplete_theory_audit_coordinate_layer_moment_pieces_outer_vector_FG_and_H_EE_kernels_done_
    and_corrected_three_times_mid_session_wiring_H_EC_H_EZ_H_CZ_gradient_screens_KNITRO_comparison_
    D20_not_started
```
