```text
CM_FEATURE_IMMUTABILITY = pass
CM_BASIS_DEFAULT = cumulative
ORIGIN_CONTRAST_DEFAULT = inconclusive
INTERVAL_HESSIAN = correct_not_faster
EQUIVALENCE_VERIFIED = rank: PASS (D4, both bases full column rank at every L/contrasts checked) |
  inverse-transform: PASS (D4, two independent constructions of the exact cumulative<->interval
  linear map agree with each other and with the dense reference to <1.2e-16, both contrasts,
  L in {10,20,50}) | zero-sets: PASS (D4, a structurally-infeasible point is rejected -300 by BOTH
  bases at every L/contrasts combination, never one feasible/one not) | Delta*: PASS (D4: <1.9e-9
  agreement including at the -300 boundary point where KNITRO's own infeasibility-certificate
  arithmetic differs slightly between bases; <1.4e-15 at every genuinely feasible D4 point. D20/L=50:
  PASS, <2.3e-16 at all 4 basis-x-contrast arms x 2 points) | gradient congruence: not separately
  computed as a standalone check this session -- implied by, and consistent with, the Delta*
  agreement above (Delta_dual is the negative of the solved dual objective, so machine-precision
  agreement there is machine-precision agreement in the function value the gradient differentiates;
  a literal `[gradient interval] = M' * [gradient cumulative]` check via the transform matrix `M`
  from `full_transform_matrix` was not run standalone -- flagged as the one item in this list not
  independently re-verified this session, see HIGHEST_PRIORITY_REMAINING_GAP) | Hessian congruence:
  PASS -- proved twice: (1) analytically, `hessian_cm_structured_interval!` differentiates the
  interval moments' own defining indicator directly (no re-derivation-by-assumption from the
  cumulative Hessian), and (2) numerically, `c13_validate_interval_native_archC.jl` (re-run fresh
  this session) matches the dense interval-basis reference Hessian to <8.4e-16 worst case across
  L in {10,20,50} x both contrasts x 2 points
HIGHEST_PRIORITY_REMAINING_GAP = D20 four-arm run only covers flexible CM at 2 points (calibration +
  perturbed_2pct); common Fréchet and CM+ZC were not separately re-run at D20 this session (see
  Section 6) -- though the shared H_EE/H_EC/H_CC machinery makes a differing verdict there unlikely.
```

# Interval vs. Cumulative CM Basis — Final Diagnosis — 2026-07-27

Phase C, item 13 (+16's decision rule for the basis axis). This document supersedes the two
same-topic 2026-07-26 documents (`INTERVAL_VS_CUMULATIVE_CM_BASIS_2026-07-26.md`,
`INTERVAL_BASIS_HESSIAN_OPTIMALITY_AUDIT_2026-07-26.md`), both of which say "NOT ATTEMPTED this
session" — **that claim was accurate for the session that wrote it, but is stale relative to this
session's HEAD.** The interval basis and its from-scratch Hessian were, in fact, already built,
validated, and wired into a production configuration surface by an earlier, separate lineage
("Continuation 12/13", commits `4b02f60`/`4bf7d1c`/`bf766b0`/`f31d1a1`) that is an ancestor of this
session's HEAD (`a69b32d`) — the 2026-07-26 session simply was not aware that lineage had already
landed on the branch it was working from. This is exactly the kind of cross-session state confusion
CLAUDE.md's own standing feedback (`feedback-check-for-existing-calibration-before-explaining-away-
nonconvergence`, `feedback-grep-all-consumers-not-just-named-files`) warns about — noted here
explicitly so the next session does not repeat either the original omission or a mirror-image error
of assuming this document's own claims are still current without checking.

## 1. What already existed (read, not rederived, this session)

- `common_marginals_interval.jl` — interval-basis moment construction (`precalc_common_marginals_
  interval`), the explicit cumulative<->interval transform matrix (`cumulative_interval_transform_
  matrix`, `full_transform_matrix`), and two independent reconstructions of it
  (`interval_to_cumulative_dense` via block-cumsum vs. the Kronecker matrix) that are cross-checked
  against each other AND against the trusted dense cumulative reference
  (`common_marginals_moments.jl::precalc_common_marginals_cdf`, unmodified).
- `cm_hessian_architecture_interval.jl` — the interval-basis Hessian **derived from scratch**, not
  mechanically reusing cumulative Architecture C: `hessian_cm_structured_interval!` reads the RAW
  bin-`l` contingency table entry (`Ttab[o,p,l,lp]`) directly, with **no prefix-sum step at all**
  (`CMBinHessCtxInterval` has no `CT`/`CScum` fields — "nothing to accidentally fall back on," per
  its own docstring). This satisfies item 15's "derive from scratch" requirement; see the dedicated
  Hessian-optimality doc for the full writeup.
- `cm_config.jl` — the actual production option surface: `CMConfig(cm_basis=:cumulative|:interval,
  contrasts=:anchored|:orthonormal, cm_hessian_backend=:structured|:dense_reference, ...)`,
  `build_cm_production_context_v2` dispatching all 4 (basis x backend) combinations onto the
  already-validated builders.
- `docs/fullA_common_marginals_production_integration.md` Section 4 — the **carried-forward
  verdict this session reconfirms, not overturns**: interval-native's Hessian is dramatically
  worse-conditioned than cumulative's at D20 (2.6x-39x worse across 5 checkpointed real-trajectory
  points), reversing the D4 finding (interval 1.2x-11.3x *better* at D4). Recommendation there:
  retain cumulative for production, keep interval available and validated behind `cm_basis =
  :interval`.

## 2. What this session verified fresh, on the current HEAD

Re-ran (not re-derived) the existing equivalence/correctness scripts to confirm nothing regressed
across the intervening Phase A commits (economic-operator retrofit touched adjacent lookup-state
structs, not this machinery directly, but re-confirming is cheap and this codebase has a documented
history of exactly this kind of unnoticed regression):

- `c12i_validate_interval_equiv.jl` (`docs/key_results/c12i_validate_interval_equiv_rerun_2026-07-27.log`):
  transform-matrix proof PASS at L in {10,20,50} x both contrasts (max discrepancy 1.11e-16); D4
  end-to-end inner-solve equivalence PASS at calibration/upper/perturbed points (Delta_dual err
  <2.7e-15, max relative moment-weight discrepancy <2.9e-14); the deliberately-infeasible point is
  rejected `nStatus=-300` by **both** bases at every L, consistently (not one feasible / one not).
- `c13_validate_interval_native_archC.jl` (`docs/key_results/c13_validate_interval_native_archC_
  rerun_2026-07-27.log`): interval-native Hessian matches the dense interval-basis reference to
  **8.36e-16 worst case** across L in {10,20,50} x both contrasts x 2 points — the from-scratch
  interval Hessian derivation is correct, not merely plausible-looking.

## 3. New this session: four-arm (basis x contrast) comparison, D4 and real D20

Neither prior lineage had run the basis comparison crossed with the contrast axis (prior D4/D20
basis comparisons only ever used `contrasts=:anchored`). `c15_d4_four_arm_basis_contrast_comparison.jl`
and `c15_d20_four_arm_basis_contrast_comparison.jl` (new this session, reusing the existing,
already-validated `compare_bases` function verbatim — no new numerical kernel code) close that gap.
Full data: `docs/key_results/four_arm_basis_contrast_summary_2026-07-27.csv`,
`docs/key_results/c15_d4_four_arm_basis_contrast_comparison_2026-07-27.log`,
`docs/key_results/c15_d20_four_arm_basis_contrast_comparison_2026-07-27.log`.

**D4** (W=8000, 2 L values x 2 contrasts x 4 points): interval strictly better conditioned than
cumulative at every feasible point (ratio 0.076-0.83, i.e. cumulative's Hessian is 1.2x-13x worse),
consistent with the pre-existing D4 finding. The one deliberately-infeasible point (`difficult_
15pct`, a 15% multiplicative perturbation) is rejected `-300` by both bases at both contrasts, at
both L — the zero-set equivalence holds even at the boundary, not just in the feasible interior.

**Real D=20/W=80,000** (seed 20260719, `destination_sample=:exclude_row`, L=50, calibration +
perturbed_2pct, both contrasts — built via `d20_real_setup_design`, which correctly seeds the
draws; see Section 5 for why this matters): interval is **32.9x-38.1x worse conditioned** than
cumulative at every one of the 4 (point x contrasts) combinations checked. This is a *stronger*, not
weaker, version of the pre-existing D20 finding (which only checked `contrasts=:anchored`) — the
interval basis's conditioning disadvantage at D20 is not an artifact of one particular contrast
choice, it holds under orthonormal contrasts too.

`Delta_dual` agrees between bases to **1.6e-16 to 2.2e-16** at all 4 D20 arms, and both architectures
converge (`nStatus=0`) at every point — the exact-reparameterization equivalence is not merely
preserved but reconfirmed under an axis (contrasts) it had not previously been checked against at
D20.

## 4. Decision rule (task item 16) applied honestly

Per item 16's own text: do not select the basis from kernel timing alone, and require complete
inner solve to improve or stay within 5%, with acceptable conditioning/failure rates, before
promoting a non-default basis. On these criteria:

- **Complete inner solve**: cold solves at D20/L=50 are statistically indistinguishable between
  bases (cumulative 19.0-25.4s vs. interval 18.6-21.9s across the 4 arms — differences are within
  run-to-run KNITRO noise, not a systematic advantage either way; warm solves likewise ~2.4-3.0s
  both). **Timing alone would not block interval.**
- **Conditioning**: interval fails this gate decisively at D20 (33-38x worse, not "within 5%" by
  any reading) — this is the actual disqualifying criterion, exactly the numerical-conditioning
  concern item 16 asks to weigh, not raw wall-clock.
- **Equivalence**: passes cleanly at both D4 and D20 (Section 2-3).

**Verdict: `CM_BASIS_DEFAULT = cumulative`.** This is a reconfirmation of the existing carried-
forward verdict (`fullA_common_marginals_production_integration.md` Section 4), not a new decision
— this session adds the missing orthonormal-contrast cross-check and finds it does not change the
conclusion. Interval remains available, validated, and correct behind `cm_basis = :interval` for
replication/comparison, per item 16's "retain all reference modes" instruction — it is not being
removed or deprecated, only not promoted to default.

## 5. A methodological note: draw-design seeding

This session's new D20 script deliberately used `d20_real_setup_design(W=80000, δ=1.0,
find_smallest=true)` (which defaults to `draw_design=:pseudorandom, draw_seed=20260719,
destination_sample=:exclude_row` and explicitly seeds `Random.seed!(draw_seed)` immediately before
context construction) rather than the bare `d20_real_setup(...)` the prior lineage's own D20 scripts
(`c13_cumulative_vs_interval_native_comparison.jl`, `section7_cm_basis_recheck_d20.jl`) used. Per
this project's own prior finding (`gateC-d20-real-setup-vs-d20-real-setup-design-seeding.md`), bare
`d20_real_setup` is unseeded — the prior D20 comparisons' *relative* basis comparisons remain valid
(both architectures were compared within one unseeded-but-fixed process, so the comparison itself
is apples-to-apples), but this session's numbers are the first on this exact topic built from a
reproducibly-seeded context matching the task's specified real-D20 recipe exactly.

## 6. Honest gaps

- Common Fréchet and CM+ZC were **not** independently re-run at D20 for this basis axis this
  session (time budget went to flexible CM per the task's own prioritization: "flexible CM
  (simplest family with a CM block)... time-permitting, common Fréchet and CM+ZC too"). Structural
  argument for why the verdict likely transfers: both families share the exact same `CMBinHessCtx`/
  Architecture C H_EE/H_EC/H_CC machinery for their CM block (confirmed via
  `production_backend_manifest.jl`'s `resolve_common_frechet_manifest`, which reports
  `cm_restriction_basis = :cumulative` identically to the flexible-CM resolver) — but this is an
  argument, not a measurement, and is flagged as the top remaining gap.
- `cm_config.jl` itself documents (and this session's `_cm_validate` reading confirms) that
  `marginal_restriction = :common_frechet` **only supports `cm_basis = :cumulative` so far** — an
  interval-basis common-Fréchet Hessian was never built at all, in either lineage. This is a
  pre-existing, disclosed scope limit, not a new gap this session found, but it means the "4 arms"
  framing is flexible-CM-specific; common Fréchet currently has only 2 arms (anchored/orthonormal
  under cumulative).
- Gradient congruence was checked only via the Delta_dual (dual objective value) proxy, not via an
  explicit `g_interval = M' * g_cumulative` numerical check using `full_transform_matrix`'s own `M`.
  Low risk (Delta_dual agreement to 1e-16 at a KNITRO-converged stationary point is strong indirect
  evidence the gradients are consistent — an inconsistent gradient basis would generally not permit
  both solves to converge to the same objective value to machine precision) but not the same as a
  direct check, and is this document's named `HIGHEST_PRIORITY_REMAINING_GAP`.
