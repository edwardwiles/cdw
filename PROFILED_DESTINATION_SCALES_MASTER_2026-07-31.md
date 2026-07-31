# Profiled Destination-Scale Reparameterization — Master Report (Phase 1)

Branch: `architecture/profile-all-destination-scales-2026-07-31`
Worktree: `/bbkinghome/edav/gravity_robustness/worktrees/architecture-profile-all-destination-scales-2026-07-31`
Base: `production/fullA-exact @ cd17235`

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
- **`ρ_f = gp` (identity map), not `ρ_f = gp^σ`.** The task brief explicitly flagged this relation
  as needing verification before hard-coding; it is verified here, from two independent code paths
  (`autarky_cf.jl`'s diagnostic dense path and `compressed_moments.jl`'s live compressed builder),
  to be the identity map — `gp^σ` was a plausible-looking but incorrect guess (a different,
  genuinely-σ-powered occurrence of the same variable, as a CES price deflator *inside* the
  moment's own formula, exists nearby and is presumably what motivated the guess).
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

**(i) Good news — the Hessian machinery needs no changes, or close to it.** Topics 6–8 (economic
operator forward/transpose, `H_EE`/winner-pair, and every `H_EC`/`H_EZ`/`H_CZ` economic×restriction
cross-block) all classify as **unchanged**: they already operate entirely on the compressed
`CompressedFactual`/`WinnerPairHessCtx` abstraction, agnostic to how many outer A-coordinates are
free or how they're laid out. This is a direct consequence of this repo's own prior "no dense
G/H, operator-only" hardening (2026-07-25 through 2026-07-30, confirmed from file headers). **This
means task §§11–12 (H_EE derivation, economic×restriction cross-Hessian rework) are very likely
much smaller than the task brief assumes** — the brief's derivations there describe what the
*moment values feeding into* these kernels look like, not new Hessian math, since the kernels
themselves don't need to change. This should be re-confirmed once the moment-state builder (§8) is
actually reduced, but it substantially de-risks what looked like the task's largest single
implementation section.

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

## 5. Recommended next steps (in dependency order)

1. ~~Decide on consolidation vs. per-site duplication~~ — resolved: the relative-A layer above
   avoids the question entirely by operating in z-space, independent of the Topic-2 formula. The
   consolidation question still applies to *other* sections (screens, `OperatorPsiBundle`
   construction, `outer_coordinate_layout.jl` generalization) that must read the FULL reconstructed
   `AodPow`, not just the anchor bookkeeping — still open for those.
2. ~~Implement §16's exact full-A recovery scalar~~ — done, D=4 gate passing, see §4 above.
3. Resolve the `gravity_sample_mask` reuse-vs-reimplement question once the concurrent Brazil-Korea
   task's branch is stable (own-cell + Brazil→Korea eligibility masking is needed by both efforts).
4. Execute theory §2.3 (comparison theorem) — requires an actual KNITRO inner solve (not just
   direct `moments!` evaluation, which is all sections 6/7/16's gates needed), a materially larger
   lift than anything done so far this session.
5. Generalize `OuterCoordinateLayout`/`outer_dim`/`decode_outer_unified`/`reduce_to_w_unified`/
   `layout_fingerprint` (Topic 1) and `free_idx` construction in all 4 context builders (Topic 10)
   to actually wire the relative-A layer into a real KNITRO-facing outer vector — the pieces built
   this session are correct and tested in isolation but not yet wired into a live driver.
6. Only after the above: proceed to §8 (moment state) and §9 (FG forward/transpose, mostly
   `unchanged` per the audit — see §3a(i)), then re-scope §§11–12 (`H_EE`, cross-Hessian) in light
   of the audit's finding that those kernels are largely `unchanged` already, then §13 (outer
   gradient — the one genuinely new derivative-bookkeeping site, `composite_gradient_at_Cplus`) and
   §14 (screens, contained per the audit — they inherit the Topic-2 dependency but don't need
   anchor-aware logic of their own).

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
    factual_homogeneous:not_implemented
    anchor_share_implied:not_implemented
    no_factual_gamma_moment:not_implemented
    France_ratio_moment:not_implemented

DIMENSIONS =
    active_A:380->361 (live-derived, D=20/D_dest=19; D=4 analog 16->12 numerically confirmed)
    free_A_after_gravity:360 (D=4 analog 12->11 numerically confirmed via test_gravity_pivot_on_retained)
    economic_moments:380->361

COORDINATE_LAYER (task section 6) =
    relative_A_encode_decode:pass_D4 (round-trip exact 1e-16; calibration round-trip exact 0.0;
        bit-exact end-to-end through live moments!)
    gravity_pivot_composition (task section 7):pass_D4 (dimension audit exact; pivot never an
        anchor cell; gravity residual ~1e-18 at 5 random points; not yet wired into a live
        KNITRO-facing outer vector -- OuterCoordinateLayout/free_idx generalization still open)

FG =
    forward:not_implemented
    transpose:not_implemented

HESSIAN =
    H_EE:not_implemented (audit finding: likely unchanged/no-op once moment state is reduced --
        see master report section 3a(i); not yet confirmed)
    H_E_CM:not_implemented (same audit finding applies)
    H_E_Frechet:not_implemented (same audit finding applies)
    H_E_ZC:not_implemented (same audit finding applies)

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
    D4:partial (coordinate-layer, gravity-pivot-composition, AND recovery sub-gates all pass at
        machine precision; full inner-solve equivalence, theory section 2.3, not yet attempted --
        this is the one remaining piece needed to call D4 fully pass)
    D20_W100k:not_run
    D20_W500k:not_run
    all_families:not_run

OPERATOR_ONLY_GATE =
    not_run

PRODUCTION_DEFAULT =
    full_gamma_normalized_reference

BRANCH_STATUS =
    incomplete_theory_audit_coordinate_layer_and_recovery_done_moment_system_and_beyond_not_started
```
