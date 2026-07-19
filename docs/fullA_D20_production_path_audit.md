# Full-A_od D=20 real-data production-path audit

Read-only research audit. No code was edited or run; this file is the only artifact written.
Scope: does a real-data (not synthetic) D=20 production driver exist for the **full-A_od**
(free-A outer-loop) method specifically, as opposed to the **sequential/profiled** method — and
if not, what exists that a coordinating session would need to assemble one.

**Headline finding, stated up front**: a real-data D=20 driver exists and has been run
extensively — but it is the **sequential/profiled** method's driver
(`sequential_gravity/run_profiled_production.jl`), living on sibling worktrees/branches
(`trade_robustness_modular_perf` @ `feature/sequential-inversion-perf`,
`trade_robustness_cm_gradient_fix` @ `fix/cm-fixed-dual-gradient`,
`trade_robustness_common_marginals` @ `feature/common-marginals`). **No genuine full-A_od
real-data D=20 run has ever been launched or logged anywhere in the `gravity_robustness` tree.**
A D-configurable full-A production driver that *could* do this
(`full_aod_diag/run_fullA_D10_production.jl`, in `trade_robustness_modular_perf`) was extended on
2026-07-16 to accept `FAKEDATA=3`/real data specifically "for a fair comparison" with the
sequential method, but no shell script, log file, or saved `.jld2` output anywhere shows it was
ever actually invoked at `DVAL=20`. The `gravity-fullA-d4` worktree under audit here (branch
`diag/fullA-d4-exact`) — the home of all the extensively-developed D4-exact full-A diagnostic
machinery (`full_aod_diag/d4_exact/`) — has **no real-data loading code at all**: its
`setup/importData.jl` only knows `fakeData ∈ {0, 1, 2}` (WIOD-path/synthetic/synthetic), never
`fakeData==3`. The real-data branch (`fakeData==3`, reading `real_data/noah_D20/`) was added on a
different branch (`feature/common-marginals`, commit `778c362`) and never merged into
`diag/fullA-d4-exact`.

---

## 1. Newest real-data D=20 driver and its context-building function

**Method**: sequential/profiled (linearized-gravity outer loop over `(gamma'_focal, A_od-column)`
via `outer_loop_cached` + `FreeParamMap`), **not** full-A_od.

**File**: `sequential_gravity/run_profiled_production.jl`. Newest/most-current copies (identical
commit `a6b2264`, 2026-07-16, "Extend GRADIENT_METHOD=fixed_dual_fd_full to support the
common-marginals restriction (CM_L>0)"):
- `/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity/run_profiled_production.jl` (branch `feature/sequential-inversion-perf`)
- `/bbkinghome/edav/gravity_robustness/trade_robustness_cm_gradient_fix/sequential_gravity/run_profiled_production.jl` (branch `fix/cm-fixed-dual-gradient`, a direct descendant of the above)

An earlier ancestor (commit `778c362`, 2026-07-13, the commit that *first* added real-D20 support
+ fixed the autarky `tau^Inf` bug) lives at:
- `/bbkinghome/edav/gravity_robustness/trade_robustness_common_marginals/sequential_gravity/run_profiled_production.jl` (branch `feature/common-marginals`)

**Context-building function**: real data is loaded through the ordinary `setup/importData.jl` →
`master_setup` → `master_prestep` → `master_prepare_cc` pipeline (same pipeline `master.jl` and
the D4-exact diagnostics use), **not** `d_exact_setup_scaled` — that function
(`full_aod_diag/d4_exact/context_scaled.jl`, line 40) is diagnostic-only and always synthetic (see
§13 below). The driver is invoked as:
```
FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true DELTA_GRID=1.0 BOUND=upper \
  OUTER_OPT_FILE=full_aod_diag/csw_outer_1000.opt \
  REAL_DATA_DIR=real_data/noah_D20 \
  julia -t 19 --project=. sequential_gravity/run_profiled_production.jl
```
Confirmed live launcher scripts (13 of them) at the root of `trade_robustness_modular_perf`:
`run_d20_coldstart_delta{01,1,2,5,10}_W{80000,800000}.sh`,
`run_d20_upper_delta1_W80000_fixeddualfdfull.sh`, `run_d20_overnight.sh`,
`run_d20_rerun_delta1_2.sh`, `run_d20_bbo_global_upper_delta1_W80000.sh`, etc. All `cd` into
`trade_robustness_modular_perf`, set the KNITRO env vars, and call
`sequential_gravity/run_profiled_production.jl` (never a "fullA" script — verified no `.sh` in
this family invokes `run_fullA_D10_production.jl`).

**"Synthetic D-scaling" vs "real D=20" scripts, explicitly distinguished**:
- `run_d6_pilot.jl`, `c8_perfprofile_harness.jl`, `profile_D_W_scaling.jl` (all in
  `gravity-fullA-d4/full_aod_diag/d4_exact/`) build contexts via `d_exact_setup_scaled(D=..., W=...)`
  → `master_setup` with `fakeData=1` (synthetic Frechet draws, `createFakeData`) — **synthetic**,
  confirmed by reading `context_scaled.jl` (only `fakeData`-driven synthetic generation, no
  `readdlm`/real-data path exists in that file or anywhere in `full_aod_diag/d4_exact/`).
- `sequential_gravity/run_profiled_production.jl` with `FAKEDATA=3` — **real data**, confirmed by
  `setup/importData.jl`'s `fakeData==3` branch (below) actually calling `readdlm` on real CSVs.

## 2. Data files and year

Real dataset lives at `real_data/noah_D20/{countries,L,pi,tau}.csv` inside each of the three
worktrees above (byte-identical copies, diffed: `countries.csv` and `pi.csv` are identical to the
repo-family-wide copy at `/bbkinghome/edav/gravity_robustness/data/{countries,L,pi,tau,xi}.csv` —
confirmed via `diff`, zero output). `xi.csv` is not present in the `real_data/noah_D20` copies
(only in the top-level `data/` dir) and is not read by `importData.jl`'s `fakeData==3` branch — it
appears to be used by an unrelated investigation (`src/iid_certificate.jl`, the common-marginals
certificate work — see below).

Per the handoff doc `trade_robustness_modular_perf/HANDOFF_D20_realdata_overnight_run.md` (§"Context"):
source is `Dropbox:Gravity robustness/Noah/prepare_clean_data/objects_for_julia`, exported
2026-03-04 (file mtimes on `/bbkinghome/edav/gravity_robustness/data/*.csv` confirm: `pi.csv`/`tau.csv`
dated Mar 4 14:30, `countries.csv`/`L.csv`/`xi.csv` dated Mar 4 15:22-15:23). No explicit trade-year
is stated in any file read during this audit — the handoff doc does not name a year, and neither
`importData.jl` nor the CSVs themselves carry a year field. **This is a gap**: the source year is
not documented anywhere found in this audit; would need to be pulled from the original
`prepare_clean_data`/Noah gravity-regression Stata files (`tariff_build/raw/noah/`,
`prepare_clean_data_old/lambda_noah.csv`, `tau_noah.csv` — not opened in this pass) or asked of the
user.

`pi.csv` = expenditure-share matrix λ[o,d] (column-normalized, diagonal ≈0.88 for e.g. Australia —
`pi.csv` row 1 col 1 = `.88389701`); `tau.csv` = bilateral trade-cost matrix; `L.csv` = country
labor/size (rescaled `./1e6` purely for wage-calibration conditioning, confirmed a no-op
downstream per the handoff doc's own derivation); `countries.csv` = 20 ISO3 codes: `aus, fra, bra,
can, che, chn, deu, esp, gbr, idn, ind, ita, jpn, kor, mex, nld, rus, tur, usa, row`.

## 3. Focal country

**France (`fra`), `baseIndex=2`** (1-indexed, second row of `countries.csv`). Confirmed explicitly
in `HANDOFF_D20_realdata_overnight_run.md`: "**baseIndex=2 (France) is the focal country** —
matches Noah's own gravity-regression analysis (`gravity.do`'s "France Only" specifications),
already the default, don't change it." `run_profiled_production.jl`'s `params` tuple line 180
carries `baseIndex` through unchanged from the D=4 synthetic default (`baseIndex=2` is
coincidentally also the D4 default, so no override was needed).

## 4. Delta (δ) grid

No single canonical grid — different scripts/phases used different subsets. Observed across all
`run_d20_*.sh` launchers and `derivative_diagnostics/verify_d20_deltagrid.jl`:
- Individual coldstart runs, one δ each: **δ ∈ {0.1, 1.0, 2.0, 5.0, 10.0}**, each also run at
  **W ∈ {80000, 800000}** for several δ values (`run_d20_coldstart_delta{01,1,2,5,10}_W{80000,800000}.sh`).
- Batch/combined runs: `run_d20_overnight.sh` → `DELTA_GRID=0.1,1.0,10.0` (W=8000, both bounds);
  `run_d20_rerun_delta1_2.sh` → `DELTA_GRID=0.1,1.0,2.0,10.0`.
- The independent verification script `derivative_diagnostics/verify_d20_deltagrid.jl` (line 24):
  `const DELTAS = [0.1, 1.0, 2.0, 5.0]`.
- Most of these runs are `BOUND=upper` only (`BOUND=upper` explicit in 8 of the 13 scripts); only
  `run_d20_overnight.sh` and `run_d20_rerun_delta1_2.sh` request `BOUND=both`.

No literal "ten frontier points" / "five upper, five lower" language was found anywhere in
`trade_robustness_modular_perf`, `trade_robustness_cm_gradient_fix`, or `gravity-fullA-d4`
(grepped `.md`/`.jl` for those phrases, zero hits) — that description does not correspond to any
artifact found in this pass; the actual grid is the informal δ∈{0.1,1,2,5,10} list above, mostly
upper-only.

**Paper's own δ range** (from `papers/CDW_Draft_June_2026.pdf`, text-extracted): the paper discusses
δ=0 (ACR/gravity point estimate, GT=0.026 in their empirical example), δ=1, and δ=2 explicitly in
prose ("even at δ=1 the trade data support Ricardian models with gains from trade that are four
times larger. And at δ=2 ... common marginals achieves ... a reduction in sensitivity"), with
δ→∞ as the "full/unconstrained" bound limit. δ=2 is called out as "the value of divergence that
... we [use]" for headline comparisons.

## 5. Calibrated μ, σ, wages, tariffs, trade shares, benchmark A* for D=20 real data

Computed by the **same shared code** as the D=4 synthetic case (`master_prestep.jl` /
`master_prepare_cc.jl`, `include`d verbatim by `run_profiled_production.jl` lines 30-32) — no
D=20-specific calibration code exists; it's the general-D pipeline applied to real inputs.
- `σHat = 2.5` fixed (assumed, not estimated) — `params.σHat=2.5` in `run_profiled_production.jl`
  (matches `master.jl`'s D=4 default). Handoff doc: "σHat=2.5 is an assumed (not estimated)
  parameter, explicitly not in question for this task."
- `μ` (Fréchet dispersion θ*) is estimated internally via the two-way-FE gravity regression in
  `master_prestep.jl` (θHat=0 ⇒ "estimate theta via gravity" — same flag/path as D=4).
  Point-estimate validation: κ_point_estimate = 0.020314, matching the closed-form ACR check
  `1 - λ_dd^μ` (λ_dd = France's own-trade share from `pi.csv`) to 6 significant figures, per the
  handoff doc.
- Wages solved via `iterWagesPreStep!`'s fixed point given `(λ, L, τ)` — unchanged code path.
- Benchmark A* is the calibrated Frechet/EK baseline (`γ.cHat`, `Aod_θ==1` at calibration),
  computed by `moments!.jl`/`buildObjectsForMoments.jl`, same as D=4.
- One real, now-fixed **bug** specific to the real data (not present in synthetic D=4): the
  autarky counterfactual (`setup/defineCounter.jl`) built `tauData .^ Inf`, and since
  `1.0^Inf == 1.0` in Julia, any off-diagonal country pair with **exactly** `tau==1` (a real
  occurrence: the France-Germany-Spain-UK-Italy-Netherlands EU-core clique has measured trade
  costs of exactly 1 for some pairs) stayed "tradable" even under autarky, corrupting France's
  own `gammaPrimeHat` enough to push the point-estimate `gamma'_focal` above its theoretical
  ceiling of 1.0. Fixed by constructing `tauPrime` directly (`Inf` off-diagonal, own-cost
  preserved) rather than via exponentiation — commit `778c362`,
  `trade_robustness_common_marginals`/`trade_robustness_modular_perf`.

## 6. Factual and autarky moment definitions at D=20

Identical code to D=4 — `EK_moments!`/`EK_moments_Jacobian!` (`moments/` directory, same
`include_moments.jl`), parameterized generically by `D` (no D=20-specific branch found anywhere
in `moments/`). The only D=20-specific adjustment found is the destination-inversion tolerance
(§12) and the autarky bug fix above (a data-triggered edge case, not a dimension-triggered one).

## 7. Normalization (γ_d≡1?)

`run_profiled_production.jl`'s `params` tuple (line ~199) still literally carries `UoModel=1` —
this is at odds with the memory note "UoModel deleted everywhere; baseline gamma≡1 made
UNIVERSAL," which describes work landing on this *same* branch (`feature/sequential-inversion-perf`,
per that memory file's own text). Two readings are consistent with what was directly observed:
either (a) the `UoModel` field is now a vestigial/ignored entry in the params NamedTuple (kept for
structural compatibility with `checkParams`/downstream code that still destructures it, but no
longer branched on now that gamma≡1 is unconditional), or (b) the cleanup commit and this driver's
own params tuple have diverged. **Not resolved in this pass** — would need a direct read of
`checkParams.jl`/`moments!.jl` on the `feature/sequential-inversion-perf` branch to confirm which.
Separately, the D4-exact full-A diagnostic machinery in `gravity-fullA-d4` (this worktree) **does**
confirm γ_d≡1 as the live, unconditional gauge: `full_aod_diag/d4_exact/parameter_table.jl` line 26
prints `"n_A_entries_free = D^2  (ALL free -- no A[1,d]=1 pins under gamma_d=1-for-all-d gauge)"` —
i.e. the D4-exact code has no `A[1,d]=1` normalization row at all, replaced by the universal
γ_d≡1 gauge, consistent with the `gamma-d-normalization-and-direct-gp` memory finding. Whether the
D=20 real-data sequential driver uses the *same* gauge could not be fully confirmed from the
params tuple alone within this audit's scope.

## 8. Free-parameter count at D=20

- **Full-A_od method** (not run at D=20, but its parameter-counting logic is generic in
  `full_aod_diag/d4_exact/parameter_table.jl`): natural candidate before any elimination is
  `1 + D^2` (line 23: `n_free (FreeParamMap) = ..., natural candidate: 1 + D^2`); with `D=20` this
  is `1 + 400 = 401`. After the pivot gravity-elimination (`gravity_elimination.jl`,
  `build_pivot_elimination`), the A-block drops from `D^2` to `D^2 - 1` free log-A entries (one
  pivot solved exactly to satisfy the linear gravity constraint), giving **399** free A-entries at
  D=20, matching the continuation-9 brief's ~400-dimension mention (`D^2 - 1 = 399` at D=20,
  `D^2 = 400` before elimination) — this generalizes the D=4 case (`D^2-1=15` free A-entries)
  directly; the elimination code has no D-specific branch.
- **Sequential/profiled method** (the one actually run at D=20): dramatically smaller by design —
  outer loop is over `(gamma'_focal, A_od[:,focal])` only, i.e. `1 + D` free parameters (1 for
  γ'_focal + D for the focal column of A_od; μ is frozen unless `FREE_MU=true`) — at D=20 that's
  **21** free outer parameters (or 22 for the CM-restricted variant, `θ[3+D+1]` slots depending on
  `CM_L`), not ~400. This is the entire point of the "profiled reformulation" per the memory note
  `profiled-reformulation-fix-gp-min-deltastar`.

## 9. Gravity pivot/nullspace choice

Full-A_od diagnostic code (`gravity_elimination.jl`, `build_pivot_elimination`, lines 51-57):
pivot is chosen **dynamically** as the A-block entry with the **largest |gravity coefficient|**
(`pivot = argmax(abs.(c))` where `c = gravity_linear_coeffs(ctx) = μ .* q_tilde ./ N_obs`) — not a
fixed coordinate, and not tied to the focal country. An alternative `NullspaceGravityElim` struct
(orthonormal nullspace basis of the linear gravity constraint, `Z: D^2 x (D^2-1)`) is also
implemented in the same file as a second option. Both are generic in D — no D=20-specific pivot
logic exists or was needed to be written; would apply unchanged if the full-A method were ever run
at D=20.

The sequential method sidesteps this entirely by construction (it never has off-focal-column A_od
parameters in the outer loop to begin with — see the paper excerpt in §8/§2 of this doc: "the
parameters `A_od'` for `d'≠d` do not appear in any equations relevant for determining `GT_d`... we
can omit `⟨γ_o, γ_o', w_o⟩_{o≠d}`" — this is exactly CDW's own dimensionality-reduction argument,
§3.4 of the paper, which the sequential method implements directly rather than eliminating gravity
via a pivot).

## 10. Intended production W

**Paper's own target, confirmed by direct PDF text extraction**
(`papers/CDW_Draft_June_2026.pdf`, page ~22, §3.4 "Implementation Details"): *"We compute
expectations using Monte Carlo integration with **800,000 draws**."* — this is **W=800,000**, not
80,000. (The CC 2023 method paper, `Christensen Connault 2023 ECMA.pdf`, separately reports
50,000/120,000 Halton draws for its own numerical examples — a different paper's number, not this
project's target.)

In executable code:
- `master.jl` (D=4 synthetic default): `W = 8000, Jac_W = 8000`, with an explicit code comment
  "paper uses ~80000" (line 75) — this comment **understates** the paper's actual figure by 10x
  (800,000, not 80,000; see above) and should be treated as stale/imprecise.
- `SETUP_AND_FINDINGS.md` (this worktree, line 55/59): "the paper uses ~80000 ... For paper
  precision: set `W=80000, Jac_W=25000`" — **also states 80,000, not 800,000** — same
  understatement, propagated into human-readable docs.
- Actual D=20 real-data runs launched with **both** `WVAL=80000` and `WVAL=800000`:
  `run_d20_coldstart_delta{1,5,10}_W800000.sh` in `trade_robustness_modular_perf` explicitly set
  `WVAL=800000`, confirming W=800,000 has been operationally exercised at D=20, real data — this
  is the closest evidence found that "the paper's real W" is actively being targeted, not merely
  discussed.
- The D4-exact diagnostic machinery (`gravity-fullA-d4`) never uses W=800,000 — its nested-W
  continuation (Continuation 8 §9) tops out at **W=80,000** (`c8_nestedw_run_grid.jl`:
  `W_GRID = [8000, 20000, 80000]`), one order of magnitude short of the paper's real target.

## 11. Does current source/paper text still say W=800,000 — historical or active?

**Active, not historical** — see §10: the paper text itself (§3.4) states 800,000, and the most
recent real-D20 shell launchers (`run_d20_coldstart_delta{1,5,10}_W800000.sh`, all dated with the
`trade_robustness_modular_perf` worktree's recent commits) are actively running W=800,000 jobs
against real data. What **is** stale/historical is the *documentation's* number: both `master.jl`'s
inline comment and `SETUP_AND_FINDINGS.md` say "the paper uses ~80000," which is a 10x
understatement relative to the paper's actual stated figure of 800,000. This appears to be a
documentation drift (someone dropped a zero at some point and it propagated across two files) —
worth fixing if anyone touches those files next, but does not affect any executed run since the
real W=800,000 launchers hardcode the correct number directly via `WVAL=800000`, not by reading
the stale comment.

## 12. Bounds/safeguards on reconstructed log A

Two **different, non-interchangeable** conventions exist depending on method:
- **Full-A_od D4-exact diagnostic** (`gravity-fullA-d4`): free log-A coordinates bounded to
  **`[-8, 8]`** via `KNITRO.KN_set_var_lobnds_all(kc, fill(-8.0, n))` / `fill(8.0, n)` —
  consistently applied across ~15 files in `full_aod_diag/d4_exact/` (`gamma_profile.jl`,
  `c8_gammabranch_core.jl`, `phaseA_*_revalidation.jl`, `run_d6_pilot.jl`, etc.). Confirmed as a
  pure numerical safeguard, not an active constraint, in
  `docs/fullA_d4_final_candidate_verification_c8.md` line 92: "`[-8,8]` is confirmed (again) to be
  a numerical safeguard nowhere near active (max|z_free| 1.46–1.75)." This convention has never
  been exercised at D=20 (never run there).
- **Sequential/profiled D=20 real-data driver** (`run_profiled_production.jl`, `focal_bounds`,
  lines 523-531): bounds are in **level space** (not log-A), set relative to the calibrated
  starting point: `lo = θr .* 1e-4; hi = θr .* 1e4` for the A_od-column block, with `γ'_focal`
  bounded by `theoretical_kappa_bounds` (`KBOUNDS.γp_lo`/`γp_hi`, `(0,1]`-type economic bound) and
  `μ` either frozen (`FREEZE_MU`, default) or bounded `[0.001, 1/(σ-1)-0.001]` if `FREE_MU=true`.
  No `[-8,8]`-style bound exists in this driver — the two methods' safeguard conventions are
  independently implemented and were not found to be cross-validated against each other anywhere
  in this audit.

## 13. Hard-value / L_fix / optimized-FD / smoothed evaluation paths — D=20 availability

All of the advanced gradient/evaluation machinery from Continuations 4-8
(`lfix_composite`/`lfix_composite_fast`, compressed winner-form moments, winner-margin
certificates, the smoothed-Dirac homotopy) is **D4-exact-diagnostic-only**: it lives entirely
under `gravity-fullA-d4/full_aod_diag/d4_exact/` (`oracle_fast.jl`, `compressed_live.jl`,
`winner_certificate.jl`, `run_smoothed_homotopy.jl`), is included by nothing outside that
directory, and — critically — **none of these files are `include`d by
`sequential_gravity/run_profiled_production.jl`** (confirmed by reading that driver's full
`include(...)` list, lines 27-40: it pulls in `focal_moments.jl`, `profiled_gravity.jl`,
`hardmax_verify.jl`, `common_marginals_moments.jl`, `PsiObjectiveBundleImplicitMethodB.jl`, and
the `derivative_diagnostics/` gradient-method family (`fixed_dual_criterion.jl`,
`fixed_dual_fd.jl`, `full_fixed_dual_criterion.jl`, `fixed_A_incumbent.jl`,
`boundary_derivative.jl`, `gradient_method_wiring.jl`) — no `lfix`/`oracle_fast`/`compressed_live`
anywhere in that list).

What the D=20 real-data driver **does** have, natively:
- `hardmax_verify.jl` — the rho-continuation homotopy hard-max verification (memory:
  `hardmax-inversion-validation`), wired in via `VERIFY_HARDMAX` env var, default on.
- `GRADIENT_METHOD` dispatch supporting `pointwise_ad` (default-historical),
  `fixed_dual_fd`/`fixed_dual_fd_full` (the winner-boundary-derivative-bug fix, memory:
  `sequential-winner-boundary-derivative-fix`), used in the D=20 runs
  (`run_d20_upper_delta1_W80000_fixeddualfdfull.sh` explicitly sets
  `GRADIENT_METHOD=fixed_dual_fd_full`).
- Common-marginals restriction moments (`CM_L`, `CM_REF`, `CM_EQ36`), off by default at D=20 in
  every launcher script inspected (`CM_L` never set in any `run_d20_*.sh`).

Conversely, the standalone D-configurable **full-A** production driver
(`full_aod_diag/run_fullA_D10_production.jl`, `trade_robustness_modular_perf`) does carry an
analogous fixed-dual-FD fix for its own method (`fixed_dual_fd_fullA.jl`, referenced in its header
as fixing "the winner-boundary/Dirac term" bug for the full-A gradient specifically), and does
accept `FAKEDATA=3`/`DVAL=20` as of 2026-07-16 — but, as stated in the headline finding, **no
evidence was found that it has ever actually been run at `DVAL=20`** (no matching `.sh` launcher,
no log file, no `.jld2` output under any plausible `OUT_DIR` naming convention was found in any of
the six worktrees searched).

---

## Docs read per the standing brief

- `docs/fullA_continuation8_handoff.md` — read in full (already summarized in the task prompt;
  confirms explicitly, §10: "D=20 NOT launched, per the brief").
- `docs/fullA_d4_section10_dimension_scaling_c8.md` — not re-read in full per instruction (already
  read by the coordinating session); confirmed via targeted grep that it contains no D=20 driver
  discussion beyond what continuation8_handoff.md already summarizes (D=10 upper budget-stalled,
  D=20 out of scope).
- `docs/reference/sequential_methodology.pdf` / `sequential_methodology.tex` — **both present**
  at `docs/reference/` in this worktree; not opened in full (out of scope for this audit's
  13-item checklist, and the `.tex` source makes `pdftotext` unnecessary if a future session needs
  its contents — grep the `.tex` directly).
- Files with "methodology" or "parameter"/"param_map" in the name: searched `docs/` — no
  `*methodology*.md` or `*param_map*.md` files exist in `gravity-fullA-d4/docs/`; the closest
  analogues are `full_aod_diag/d4_exact/parameter_table.jl` (code, not docs — used directly as
  evidence in §7-9 above) and `docs/fullA_d4_code_audit.md` (referenced by `gravity_elimination.jl`'s
  own header as the source of the "gravity is a second explicit KNITRO constraint" framing this
  file supersedes).

## Summary table: what exists where

| Component | Full-A_od method | Sequential/profiled method |
|---|---|---|
| D4-exact diagnostic machinery | `gravity-fullA-d4/full_aod_diag/d4_exact/` (extensive, this worktree) | not applicable (D4-exact is full-A-only) |
| D=6/8/10 pilots | synthetic only, gated/partial (`run_d6_pilot.jl` etc.) | N/A |
| D-configurable production driver | `full_aod_diag/run_fullA_D10_production.jl` (`trade_robustness_modular_perf`) — exists, D=20-capable in principle | `sequential_gravity/run_profiled_production.jl` — exists, D=20-capable |
| Real D=20 data support wired in | **Yes** (`FAKEDATA=3` added 2026-07-16) but **never invoked** at D=20 | **Yes**, and extensively run (13 shell launchers, multiple δ/W combos) |
| Actually run at D=20 with real data | **No — confirmed gap** | **Yes** |
| Lives on branch `diag/fullA-d4-exact` (this worktree)? | No real-data code at all here | No — lives only on sibling branches |

## The gap, stated plainly

If a coordinating session needs a genuine full-A_od D=20 real-data run, the pieces to assemble
are: (1) `full_aod_diag/run_fullA_D10_production.jl` from `trade_robustness_modular_perf` (already
D-configurable and `FAKEDATA=3`-capable), (2) the `real_data/noah_D20/` CSVs (already present in
that worktree, and in `trade_robustness_cm_gradient_fix`/`trade_robustness_common_marginals`), and
(3) a new launcher script analogous to `run_d20_upper_delta1_W80000_fixeddualfdfull.sh` but
invoking `run_fullA_D10_production.jl` instead of `run_profiled_production.jl` with `DVAL=20`. None
of this exists on `diag/fullA-d4-exact` (this worktree) — the D4-exact continuations 4-8 machinery
(`lfix_composite`, compressed moments, winner certificates) would need to be either ported to
`run_fullA_D10_production.jl`'s D-generic code path (it currently only has the older
`fixed_dual_fd_fullA` gradient, not the newer composite/compressed machinery) or the two branches
would need a real merge — neither has happened. Expect the D=20 free-A_od outer loop to be
**~400 free parameters** (§8) vs the sequential method's ~21 — a qualitatively harder KNITRO
problem that none of this repo family's D=20 experience (all sequential) speaks to directly.
