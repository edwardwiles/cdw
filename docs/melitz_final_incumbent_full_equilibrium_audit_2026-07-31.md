# Independent adversarial full-equilibrium audit of the final D20 Melitz (δ=0.5) incumbents (2026-07-31)

**Scope**: an independent verification of whether the two final checkpointed D20 Melitz
incumbents (`docs/melitz_d20_profiledA_overnight_qpoll_delta0p5_2026-07-31.md`) satisfy the
*complete* theoretical Melitz equilibrium system, not merely the `D^2+1` moments actually sent
to KNITRO. This is not an optimization session; nothing was re-searched, improved, or modified.
The Ricardian implementation was not touched.

Primary states audited (both loaded from their canonical checkpoints, fingerprints verified):

| | GT | Δ* | checkpoint |
|---|---:|---:|---|
| **Upper** | 8.84934137951131% | 0.49812250980571005 | `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/phase6_upper_m1.jls` |
| **Lower** | 0.2722655222019532% | 0.11376383122706066 | `docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase5/checkpoints/phase6_lower_m-1.jls` |

Repo: `trade_robustness_modular`, branch `melitz/fullD-delta-star`, HEAD `6e803d4` (unchanged by
this audit). D=20, real data (`real_data/noah_D20`), focal country = France, σ=2.5,
θ*=8.75177334 (estimated), W=80,000, seed=1. Fingerprints and SHA-256 checksums of the exact
files read: `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/provenance_2026-07-31.txt`.

**All calculations were run in fresh, isolated Julia processes** (`scripts/melitz_audit_acr_quicklook_2026-07-31.jl`,
`scripts/melitz_independent_full_equilibrium_audit_2026-07-31.jl`), never reusing a live campaign
process's state.

## Headline finding

The two checkpoints are **internally consistent with everything the D²+1 moment system + the
gravity pivots + the LFD-feasibility gate actually check** — every one of those directly-imposed
or algebraically-enforced equations holds to machine precision or better, independently
reconfirmed below. **But the model's own documented, theoretically-required cross-check between
the reported gains-from-trade and the ACR/Chaney sufficient statistic
(`acr_gains_from_trade`, `equilibrium.jl`) — which the model's own Gate A2 record calls "a
blocking economic error, not a documented discrepancy" if it fails — was never evaluated at
either D20 incumbent (grep-confirmed: `acr_gains_from_trade` is called in zero D20 production or
overnight scripts) and, when independently evaluated here, fails by 6.82 percentage points
(upper) and 1.76 percentage points (lower)** — millions of times larger than the ~1e-14
"agreement" the model's own D=4 calibration-point tests document as the expected standard.

This is **Conclusion B, with a documented ambiguity flagged for Conclusion D** (see "Decision"
below) — not because any directly-imposed equation failed, but because a theoretically-required,
previously-documented cross-check was silently dropped from the D20 pipeline and, when restored,
fails materially.

---

## Phase 1: authoritative equation inventory

Full table: `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/equation_inventory_2026-07-31.csv`
(19 rows, columns: equation ID, formula, indices, economic meaning, theory source, code source,
how enforced, category, tolerance, independently tested, result at each incumbent).

Category legend (per the governing prompt): **1** = directly imposed inner moment; **2** =
outer gravity restriction; **3** = imposed algebraically by state reconstruction; **4** = implied
by another equation, redundant; **5** = not imposed but theoretically required; **6** =
diagnostic only, not required by the maintained model.

Summary of classifications:

| # | equation | category | why |
|---|---|---|---|
| E1 | `D^2` factual bilateral trade shares | **1** | directly imposed KNITRO moment columns |
| E2 | trade-share adding-up by destination | **4** | implied by E1 + data's own column-sum consistency (verified: data columns sum to 1 ± 7.06e-8) |
| E3 | nonfocal-origin baseline free entry (D−1 origins) | **6** | `f_entry_o` for o≠j is *recovered*, not imposed — no data target exists for it under the active `N_o≡1` closure |
| E4 | focal baseline free entry (alone) | **6** | definitional; only the *difference* (E8) carries content |
| E5 | zero-profit cutoff identity | **3** | `f` is derived from `q` via the exact algebraic inverse (`melitz_log_f_from_q`) |
| E6 | focal autarky free entry (alone) | **6** | definitional; only the *difference* (E8) carries content |
| E7 | autarky domestic-cutoff normalization (`ẑ'_jj≡1`) | **3** | imposed by construction (`derive_fjj_from_autarky_cutoff`) |
| E8 | focal baseline-vs-autarky free-entry **link** | **1** | the ONE directly-imposed free-entry moment |
| E9 | autarky price-index / N′ two-formula cross-check | **5 as documented, but near-tautological in practice** | the model's own "KEY omitted moment" (equilibrium.jl comment) — but at `ẑ'_jj≡1` under a Pareto(1,·) reference distribution, the participation gate in both formulas is a no-op regardless of γ'_j, so the two formulas coincide almost by algebraic necessity here, independent of whether γ'_j is economically sensible |
| E10 | **ACR/Chaney sufficient-statistic cross-check** | **5 — required, omitted, and FAILS** | see headline finding |
| E11 | baseline CES price-index identity (all D destinations) | **4** | implied by E1+E2, same as E2 |
| E12 | income = expenditure (data) | **4/6** | a data-integrity property, invariant to the search |
| E13 | income = expenditure (model-predicted) | **4** | implied by E1 holding tightly + E12 |
| E14 | LFD nonnegativity + normalization | **1** | directly enforced by the `lfd_ok` gate |
| E15 | A-gravity restriction | **3** | imposed by the A-pivot construction |
| E16 | q/f-gravity restriction | **3 (q) / 4 (f)** | q-gravity imposed by pivot; f-gravity algebraically implied given q-gravity+A-gravity |
| E17 | cutoff support/feasibility restrictions | **1** | hard screen, rejects infeasible candidates before acceptance |
| E18 | "only the link ties baseline/autarky together" | n/a | structural note, not a separate equation |
| E19 | baseline normalizations (`N_o≡1`, `γ_d≡1`) | **3** | universal, not searched |

**The single equation classified 5 that both (a) the model's own documentation calls required
and (b) is not near-tautological in this construction is E10, the ACR cross-check.** E9 is
nominally "required" by the same documentation section but, on independent inspection, is
almost a tautology at this specific construction (see caveat above) — so its passing here is
much weaker evidence of a genuine standalone equilibrium than its "PASS" status might suggest at
first glance. This distinction — a check that passes at machine precision but for structural,
not economic, reasons — is exactly the kind of gap the governing prompt asked this audit to
surface, not just report as green.

---

## Phase 2-3: independent oracle + LFD verification

`scripts/melitz_independent_full_equilibrium_audit_2026-07-31.jl` builds every result below from
its own loops (Kahan-compensated Float64 summation; BigFloat for the autarky-cutoff and focal-link
residuals) over the real reference draws (`obj.U`, Pareto(1,θ*), scrambled Halton) and the
recovered LFD weights. It never calls `melitz_moments!`, `mul_G!`, `mul_Gt!`, the production
moment-column constructors, or a cached production moment residual. It reuses only one-line
primitive economic formulas (`melitz_firm`/`melitz_C`/`melitz_cutoff`, identical building blocks
used everywhere else in this codebase) and the state-reconstruction function
(`expand_free_theta_logcutoff`, how `(A,f,q,γ)` are rebuilt from `theta_free` — this is reading
state, not the moment operator). A separate call to `check_profiled_melitz_equilibrium(...;
full=true)` — the codebase's own existing, but never invoked in any D20 production/overnight
script, ex-post equilibrium checker — is also run and reported, clearly labeled REFERENCE, purely
as a cross-check against the independent numbers.

Fresh solve + LFD, both incumbents:

| | Upper | Lower |
|---|---:|---:|
| Fresh cold `Δ*` vs checkpoint | drift = 0.0 (bit-identical) | drift = 0.0 |
| `lfd_ok` | true | true |
| Reported `Δ` vs independently-recomputed primal divergence | 0.4981225098057301 vs 0.49812250980571 (diff 2.0e-14) | 0.1137638312 vs 0.1137638312 (diff 5.1e-14) |
| LFD sum of weights | 1.000000000000 | 1.000000000000 |
| min / max weight | 3.51e-160 / 1.61e-3 | 3.95e-271 / 7.64e-5 |
| Effective sample size (1/Σw²) | 40,188.6 (50.24% of W) | 65,304.4 (81.63% of W) |
| # weights > 10/W | 1 | 0 |
| # weights > 100/W | 1 | 0 |
| Overflow/underflow/clipping | none observed (all weights finite, nonneg) | none observed |

Both LFDs are well-behaved re-weightings (not degenerate point masses); the upper LFD concentrates
somewhat more (ESS ≈ 50% of W vs 82%), consistent with it sitting closer to its `Δ*=0.5` budget
ceiling.

## Phase 4: bilateral trade shares (independent, all D²=400 cells)

Full residual matrices: `bilateral_share_residuals_{upper,lower}.csv`.

| | Upper | Lower |
|---|---:|---:|
| max\|predicted share − data share\| over 400 cells | 8.44e-15 | 2.91e-14 |
| max\|Σ_o data share − 1\| (adding-up, by destination) | 7.06e-8 | 7.06e-8 |
| max\|Σ_o predicted share − 1\| | 7.06e-8 | 7.06e-8 |

All D² trade-share moments hold at machine precision independently, confirming `lfd_ok`'s own
gate is not being fooled. Trade-share adding-up (E2/E11) holds to 7e-8 at every destination —
tiny, and identical between data and predicted (as it must be given E1 holds this tightly) — this
is a property of the raw data, not sensitive to which incumbent is evaluated.

Adversarial check (Phase 11 below): perturbing `A[5,12]` by one log point blows the cell-(5,12)
residual up to 0.146 while cell (1,1) stays at −7.4e-15 — the independent check localizes the
fault to the correct cell.

## Phase 5: cutoff / zero-profit identities

| | Upper | Lower |
|---|---:|---:|
| max\|independent forward cutoff (from A,f via `melitz_C`/`melitz_cutoff`) − stored q\| | 1.25e-15 | 1.39e-15 |
| min domestic log-cutoff `q[o,o]` (feasible iff ≥0) | 0.024553 | 0.002726 (thin margin) |
| min export-minus-domestic `q[o,d]−q[o,o]`, d≠o (feasible iff ≥0) | 0.083719 | 0.083719 |

Both incumbents are comfortably feasible on export-selection; the **lower** incumbent's domestic
participation margin (0.0027 log-points) is thin — a small further push in the lower direction
would plausibly hit this constraint, worth flagging for anyone continuing this search.

## Phase 6: free-entry system

| | Upper | Lower |
|---|---:|---:|
| Focal link residual (E8, independent) | 1.07e-14 | 1.10e-13 |
| Autarky cutoff identity, BigFloat (E7) | 1.85e-16 | −2.84e-16 |
| Recovered baseline `f_entry` range across all 20 origins | [0.3315, 85.09] | [0.3303, 85.21] |
| Recovered focal `f_entry[j]` (baseline) | 2.6456 | 1.5384 |
| Recovered focal `f_entry_j` (autarky) | 2.6456 | 1.5384 |

The link moment (the one thing actually imposed) holds at machine precision, as expected. **The
recovered nonfocal baseline entry costs span a ~257× range across the 20 origins** — this is not
a "failure" under the current design (E3/E4/E6 are Category 6, no data target exists to compare
against), but it is a real, checkable, and previously-unreported plausibility signal: the profiled
A/q search is not required to, and apparently does not, deliver an economically narrow/uniform
implied entry-cost structure across origins.

## Phase 7: price index, autarky N′, and welfare/GT reconstruction — the central finding

| | Upper | Lower |
|---|---:|---:|
| max\|baseline price-index residual `γ_d − 1`\|, all D destinations | 7.06e-8 | 7.06e-8 |
| `N′` via market clearing | 0.786445 | 1.165585 |
| `N′` via price index | 0.786445 | 1.165585 |
| relative diff (E9) | 1.64e-14 | 1.62e-14 |
| **GT_model (recomputed, matches checkpoint exactly)** | **8.849341%** | **0.272266%** |
| **GT_ACR (data-anchored domestic trade share, `λ_jj = X_data[j,j]/expenditure_j = 0.835676`)** | **2.030281%** | **2.030281%** |
| GT_ACR using the independently-*predicted* domestic share (cross-checks E1 at the (j,j) cell) | 2.030281% (diff from data-based: 2.2e-8pp) | 2.030281% |
| **\|GT_model − GT_ACR\|** | **6.819060 percentage points** | **1.758016 percentage points** |

`GT_ACR` is **structurally invariant across the entire campaign**: it depends only on
`X_data[j,j]/expenditure[j]` and `θ*`, both fixed data objects never touched by the outer search
over `(A, f, q, γ'_target)`. `GT_model`, by contrast, is driven entirely by the searched
`γ'_target` (equivalently, the searched autarky price power/cutoff), tied to the data only through
the single E8 link moment — not through the standard general-equilibrium relationship that would
force it back toward `GT_ACR`. The model's own Gate A record
(`docs/melitz_delta_star.md` §14.2) verified `|GT_model − GT_ACR| ≈ 3.6e-14` at the D=4 exact
Pareto calibration benchmark and explicitly called ACR agreement **"the required cross-check"**
whose failure would be **"a blocking economic error, not a documented discrepancy."** That check
was never re-run at real D20, and never at any point in the profiled-A/cutoff-vector search that
produced these two incumbents (confirmed by grepping every D20 production/overnight script:
`acr_gains_from_trade` appears in zero of them). Independently evaluated here, it fails by
6.8 and 1.8 percentage points — several orders of magnitude past the documented standard, in
**both** directions of the reported bound.

**Two readings, both stated plainly (per the instruction not to silently pick the favorable one):**

1. **As a bug/gap**: the D20 "profiled-A + cutoff-vector search" pipeline dropped a
   theoretically-required verification step (E10) that the model's own D=4 Gate A record
   established as diagnostic of "blocking economic error." Under this reading, the reported
   8.85%/0.27% bound is not verified against the model's own documented completeness standard,
   and should not be reported as such until E10 is either satisfied or explicitly waived in
   writing.
2. **As intended partial-identification design**: the entire point of a Christensen–Connault
   `Δ≤δ` bound is to search over economically plausible-but-not-point-identified values of
   `γ'_target` (the autarky counterfactual), subject only to the *imposed* moments + a bounded
   divergence budget on the reference-draw reweighting — in which case moving away from the
   ACR/GE point value is not an error, it *is* the bound. Nothing in any of the reviewed documents
   (`docs/melitz_delta_star.md`, the production/overnight reports, `src/melitz/CLAUDE.md`)
   explicitly states this reinterpretation, however — the only place ACR agreement is discussed
   treats it as a required, not a relaxable, check.

This audit does **not** resolve which reading is correct — that is a modeling-intent question for
the model's authors/user, not something this verification pass can decide from the code and
documents alone. See "Decision" below.

## Phase 8: aggregate equilibrium identities

| | Upper / Lower (identical — data/calibration-invariant) |
|---|---:|
| max\|Σ_d X_data[o,d] − w_o·L_o\|, all 20 origins (income=expenditure, DATA) | 1.43e-5 |
| max\|Σ_d predicted_share[o,d]·expenditure_d − w_o·L_o\|, all 20 origins (MODEL-predicted) | 1.43e-5 (identical to 5 sig figs, as expected given E1 holds to 1e-14) |

Labor-market clearing / resource constraints are **never separately re-verified** during the outer
search because `w`, `τ`, `expenditure` are treated as fixed calibration data throughout (a
documented design choice, `equilibrium.jl` comment). Checked here directly for the first time
against the actual profiled A/q states: holds at ~1.4e-5, essentially unchanged from the
underlying data's own consistency — not a new failure, but also not something the search actively
guarantees; it is inherited entirely from calibration, not from anything the D20 campaign itself
verified.

## Phase 9: gravity restrictions (independent, own `withinTransform`)

| | Upper | Lower |
|---|---:|---:|
| A-gravity, independent | −2.240e-15 | −2.294e-15 |
| A-gravity, production `gravity_residuals()` | −2.584e-15 | −2.318e-15 |
| f-gravity, independent | −6.124e-16 | −1.107e-15 |
| f-gravity, production `gravity_residuals()` | −4.970e-16 | −1.435e-15 |

Both restrictions hold at machine precision, independently confirmed with a from-scratch
`withinTransform`/gravity-coefficient reimplementation, not calling `misc/doubleDiff.jl`. The
f-gravity restriction (algebraically implied given the q-gravity pivot + A-gravity, per
`log_cutoff_param.jl`'s own derivation) is confirmed to actually hold this tightly in practice,
not just in the symbolic derivation.

## Phase 10: reconciliation table (production moment inventory vs. theory)

| equation | theoretically required | directly imposed | algebraically enforced | independently satisfied | status |
|---|---|---|---|---|---|
| E1 trade shares (D²) | yes | **yes** | — | yes (8e-15/3e-14) | OK |
| E2 trade adding-up | yes | no | yes (via E1+data) | yes (7e-8) | OK |
| E3/E4 nonfocal/focal baseline free entry (standalone) | **no** (by design; N_o≡1, no target) | no | no (definitional) | n/a | as designed, but see E3 plausibility flag |
| E5 zero-profit cutoff | yes | no | **yes** | yes (1e-15) | OK |
| E6 autarky free entry (standalone) | **no** (by design) | no | no (definitional) | n/a | as designed |
| E7 autarky cutoff normalization | yes | no | **yes** | yes, BigFloat (2e-16) | OK |
| E8 focal link | yes | **yes** | — | yes (1e-13) | OK |
| E9 autarky N′ cross-check | yes (documented) | no | no (ex-post only) | yes, but **near-tautological** given ẑ'_jj≡1 + Pareto(1,·) support | PASSES for structural, not economic, reasons |
| **E10 ACR sufficient statistic** | **yes (documented Gate A2 standard)** | **no** | **no** | **NO — fails by 6.82pp / 1.76pp** | **MATERIAL GAP — never checked at D20, fails when checked** |
| E11 price-index adding-up | yes | no | yes (via E1+E2) | yes (7e-8) | OK |
| E12/E13 income=expenditure | yes | no | yes (data+E1) | yes (1.4e-5) | OK, but inherited from calibration, not the search |
| E14 LFD validity | yes | **yes** | — | yes | OK |
| E15 A-gravity | yes | no | **yes** | yes (2e-15) | OK |
| E16 q/f-gravity | yes | no | **yes** | yes (1e-15) | OK |
| E17 cutoff support | yes | **yes** (hard screen) | — | yes (comfortable margins, lower's domestic margin thin) | OK |

**Explicitly identified**:
- **Required equation not imposed, and failing when independently checked**: E10 (ACR
  sufficient-statistic cross-check).
- **Equation verified only through the same production code before this audit**: none of the
  "OK" rows above had ever been independently recomputed at these specific D20 incumbents before
  this session — the production final-verify scripts checked only `Δ` drift, `lfd_ok`,
  `A`-gravity residual, and `q`/`f` reconstruction drift (grep-confirmed against
  `scripts/melitz_overnight_final_verify_2026-07-31.jl`); everything else in this table (E2–E13,
  E16 f-gravity) was checked here for the first time at these incumbents, via `check_profiled_melitz_equilibrium`
  (existing code, never called) and this audit's own independent oracle.
- **No missing country/cell**: all D=20 origins/destinations were covered in every D²-scale check.

## Phase 11: adversarial perturbation tests

All six planted perturbations (on the upper incumbent's reconstructed state) were correctly
detected by the independent oracle; the unperturbed incumbent passes every check, and each
perturbed state fails the specific equation it should:

| perturbation | equation targeted | residual before | residual after |
|---|---|---:|---:|
| `A[5,12] *= e` | trade share, cell (5,12) | −4.3e-17 | **0.1456** (other cells unaffected: cell (1,1) stays −7.4e-15) |
| `f[5,12] *= e` | cutoff/zero-profit forward check | 0.0 (exact) | **0.667** |
| `q[5,12] += 0.1` (A,f unchanged) | cutoff/zero-profit forward check | 0.0 | **−0.100** (exact detection of the injected offset) |
| one LFD weight ×10 (no renormalization) | probability normalization | 0.0 | **1.28e-4** deviation from sum=1 |
| `g=log(γ'_target) += 0.05` (LFD held fixed) | focal link moment | 1.0e-14 | **0.129** |
| A-gravity pivot cell `×= e^0.5` | A-gravity restriction | −2.2e-15 | **0.244** |

Every perturbation moved its targeted residual by 6–13 orders of magnitude while (where checked)
leaving unrelated cells/equations unaffected, confirming the independent oracle is a meaningful,
non-vacuous check, not one that would pass a broken state.

---

## Final report questions

**1. Full theoretically required Melitz equation set?** 19 candidate equations (E1–E19,
`equation_inventory_2026-07-31.csv`), spanning trade shares, free entry (baseline/autarky, focal
and nonfocal), zero-profit cutoffs, autarky normalization/price-index, the ACR sufficient
statistic, aggregate/CES identities, gravity (both A and f), LFD validity, and cutoff-support
feasibility.

**2. Which are directly imposed?** E1 (D² trade shares), E8 (focal free-entry link), E14 (LFD
validity), E17 (cutoff-support screen) — 4 of 19.

**3. Which are algebraically enforced?** E5 (zero-profit cutoff, via `f` derived from `q`), E7
(autarky cutoff normalization, via `derive_fjj_from_autarky_cutoff`), E15/E16-q (A- and
q-gravity, via the two pivots), E19 (universal normalizations) — 5 of 19, all independently
reconfirmed to hold at machine precision.

**4. Which were previously not being checked (at these specific D20 incumbents)?** E2, E3/E4/E6,
E9, **E10**, E11, E12/E13, E16-f — i.e., everything outside the 4 directly-imposed + 5
algebraically-enforced equations. The D20 final-verify scripts checked only Δ-drift, `lfd_ok`,
A-gravity, and q/f-reconstruction drift.

**5. Do all D² factual trade shares match independently?** Yes — max residual 8.4e-15 (upper) /
2.9e-14 (lower) across all 400 cells, both directions.

**6. Do all required factual free-entry conditions hold?** The one that is actually required and
imposed (E8, the link) holds at machine precision. The per-origin baseline conditions (E3/E4) are
**not required by the maintained (reduced) closure** — no data target exists for them — so "hold"
is not a well-posed question for them; their recovered values are reported as a plausibility
diagnostic (a 257× spread across origins).

**7. Do all required autarky free-entry conditions hold?** Same structure: only the link (E8,
which is jointly baseline+autarky) is required and it holds; the standalone autarky condition
(E6) is definitional under this closure.

**8. Do all cutoff/zero-profit identities hold?** Yes, to 1.25e-15 (upper) / 1.39e-15 (lower),
independently recomputed via the forward `melitz_C`/`melitz_cutoff` formulas.

**9. Does the autarky price index independently reproduce the reported GT?** The `N′`
cross-check (E9) agrees to 1.6e-14 but is **near-tautological** at this construction (participation
gate is a no-op given ẑ'_jj≡1 and Pareto(1,·) support). The **substantively independent**
cross-check — the data-anchored ACR sufficient statistic (E10) — does **not** reproduce the
reported GT: it gives 2.030281% at both incumbents, vs. the reported 8.849341% (upper) and
0.272266% (lower).

**10. Do the two gravity restrictions hold independently?** Yes — A-gravity ~2e-15, f-gravity
~1e-15 at both incumbents, via an independent `withinTransform` reimplementation.

**11. Are normalization, nonnegativity, and divergence correct?** Yes — LFD weights are all
finite, nonnegative, sum to 1.000000000000, and the independently-recomputed primal divergence
matches the reported `Δ` to ~2–5e-14 at both incumbents.

**12. Are there any theoretically required but omitted moments?** Yes: **E10, the ACR
sufficient-statistic cross-check**, is the one clear case — required by the model's own Gate A2
documentation, omitted from every D20 script, and materially violated when independently
evaluated (6.82pp / 1.76pp). E9 is nominally in the same category by the documentation's own
wording but is near-tautological in this specific construction, so its omission is lower-stakes
than E10's.

**13. Did the adversarial verifier catch the planted perturbations?** Yes — all six, cleanly, with
the correct cell/equation localized and 6–13 orders of magnitude of residual movement.

**14. Which of Conclusions A–D is supported?**

## Decision

**B, with an explicit D-flavored caveat that must not be silently resolved.**

The directly-imposed system (E1, E8, E14, E17) and every algebraically-enforced equation (E5, E7,
E15, E16-q) pass independently at strict tolerance, at both incumbents. But one theoretically
required equation that the model's own documentation calls a "blocking economic error" if
violated — the ACR/Chaney sufficient-statistic cross-check (E10) — was never evaluated anywhere
in the D20 profiled-A/cutoff-vector search pipeline that produced these incumbents, and fails
materially (6.82 and 1.76 percentage points, vs. a documented ~1e-14 standard) when independently
checked here, in **both** directions of the reported bound.

**Recommendation**: do not report GT=8.849341% / 0.272266% as fully-verified Melitz equilibrium
states without one of the following, made explicit in writing by the model's authors/user (this
is the Conclusion-D-style ambiguity that must not be silently resolved in either direction):

- *(a)* an explicit statement that the ACR cross-check is understood to be relaxable away from
  the calibration point in this partial-identification/robustness-bound design — in which case
  Conclusion A would follow once that scope is documented, and the reported bound should be
  captioned accordingly (e.g., "gains-from-trade bound over autarky counterfactuals satisfying
  the focal free-entry link and a Δ≤0.5 divergence budget, not constrained to satisfy the
  standard ACR relationship"); or
- *(b)* reinstating E10 (and, at lower priority given its near-tautological character here, E9)
  as an enforced or at least monitored restriction in the outer search, and re-running the
  campaign, per Conclusion B's standard recommendation.

Everything else independently checked in this audit — all D² trade shares, both gravity
restrictions, the zero-profit cutoff system, the focal free-entry link, LFD validity, cutoff
feasibility, and the aggregate income/expenditure identities inherited from calibration — passes
cleanly and was, in most cases, being verified at these specific incumbents for the first time.

---

## Required outputs (this audit)

1. This document.
2. `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/equation_inventory_2026-07-31.csv`
3. `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/bilateral_share_residuals_{upper,lower}.csv`
4. `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/equation_summary_{upper,lower}.csv` (free-entry, cutoff, price-index/welfare, aggregate, gravity, LFD residuals)
5. `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/pass_fail_summary_2026-07-31.csv` (machine-readable per-check pass/fail)
6. `docs/key_results/melitz_final_incumbent_full_equilibrium_audit_2026-07-31/provenance_2026-07-31.txt` (git HEAD, checkpoint/data SHA-256 fingerprints, Julia version)
7. Scripts (reproducible): `scripts/melitz_audit_acr_quicklook_2026-07-31.jl` (minimal ACR check),
   `scripts/melitz_independent_full_equilibrium_audit_2026-07-31.jl` (full independent oracle,
   Phases 3-9 and 11).
