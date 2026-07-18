# Sequential/profiled vs. full-A exact comparison (Phase 5)

Continuation of the D=4 exact full-A investigation. This phase re-runs the production
sequential/profiled solver **correctly** (materially larger `maxit`, `eval_fcga=no`, genuine
5-start multistart — none of which the retracted `docs/fullA_d4_final_report.md` §9.3 run did),
reconstructs the full-A competitiveness matrix from the winning sequential solution, converts it
into the full-A `gamma_d≡1` gauge, and evaluates the reconstructed point in the exact full-A
oracle. The comparison is a genuine, mandatory sanity check the investigation has been missing
throughout (`docs/fullA_d4_recommendation.md` priority #3).

## 0. Repository state

- Diagnostic worktree: this worktree, checked out from `diag/fullA-d4-exact` at `b5c109d`, new work
  committed to `diag/fullA-d4-exact-phase5-sequential`.
- Production worktree: `/bbkinghome/edav/gravity_robustness/trade_robustness_modular`, branch
  `sequential-profiled-gravity`, confirmed **unchanged** at `53ffb58e8d9b18498279fab25da4d1b7cc47556a`
  before and after this session's runs (`git log --oneline -1` re-checked at the end).
- Synthetic economy (both scripts' own params, byte-identical): D=4, W=8000, δ=1,
  seedFakeData=889, seedU=888, baseIndex=2, σ=2.5. Confirmed by direct comparison of
  `run_profiled_production.jl`'s `params` tuple against `full_aod_diag/ad_benchmark/setup_context.jl`'s
  `AD_PARAMS` tuple (identical, field for field) — not assumed.
- Candidate registry values used below (loaded via `full_aod_diag/d4_exact/candidate_registry.jl`,
  not hand-transcribed): `κ_fixedA = 0.1439805232`, `κ_fullA_incumbent = 0.17176461388430053`
  (`γ'_focal = 0.8930839180420251`, maxit=40, labeled `EXACT_FEASIBLE_CANDIDATE` /
  `H_BANDWIDTH_KKT_CANDIDATE(h=0.01)` but explicitly **NOT** `ROBUST_LOCAL_CANDIDATE` — see
  `docs/fullA_d4_final_report.md` §4).

## 1. Part A — a reliable sequential/profiled production run

### 1.1 What was invalid about the earlier attempt (recap, per the required reading)

`docs/fullA_d4_final_report.md` §9.3: the retracted run used (a) the default `OUTER_OPT_FILE`
(`full_aod_diag/csw_outer_25.opt`), `maxit=25`, terminating at `nStatus=-400` (iteration cap, not
converged); (b) `eval_fcga=yes` + `hessopt=4`, which KNITRO **silently downgrades to L-BFGS**
(`"WARNING: Option hessopt=4 not valid when eval_fcga=1. Changing hessopt to 6 (LBFGS)."`) — so the
requested Hessian mode was never actually used; (c) **one** calibration-anchored start, not the
production-specified 5-start multistart. The reported κ=0.0779 was *below* the trivial
`κ_fixedA=0.1440` floor, which a converged sequential search cannot be — direct internal proof of
non-convergence.

### 1.2 Corrected configuration used this session

- New KNITRO options file: `full_aod_diag/d4_exact/csw_outer_seqprod_fcga_no_maxit150.opt` — a copy
  of the already-existing, already-validated `csw_outer_fcga_no_maxit40.opt` (used earlier in this
  investigation's own Phase B Hessian-mode comparison) with `maxit` raised from 40 to **150** (6x
  the invalid run's 25, and well above the maxit=40 full-A headline candidate's own budget) and
  `eval_fcga no` / `hessopt 4` / `outlev iter` preserved.
- **Grepped every one of the 5 run logs for the exact fallback string
  `"WARNING:  Option 'hessopt"` — zero matches in all 5 logs.** `hessopt=4` (product finite-difference
  Hessian-vector, per `docs/fullA_d4_final_report.md` §2's corrected characterization) was genuinely
  honored throughout, not silently downgraded.
- **No run hit `nStatus=-400`.** Terminal KNITRO statuses observed across the 10 (5 starts × 2
  bounds) sub-runs: `-101`, `-102`, `-103`, `-502` — all genuine KNITRO stopping conditions (relative
  function/point-change tolerance, or a feasible-but-slow-progress condition for `-502`), not the
  iteration-cap code.
- Genuine 5-start multistart per `sequential_methodology.tex`'s "Checks, validation, and multistart"
  section: one calibration-anchored start (`START_ID=0`, unperturbed `θ_r0`, i.e. the "$A^*$-anchored"
  start the spec calls for) plus four independent random perturbations (`START_ID=1..4`, fixed seeds
  `20260718+id` for reproducibility): `γ'_focal` drawn **uniformly within its true theoretical bounds**
  (not a small jitter), each `A_{o,focal}` entry drawn as a log-normal jitter around calibration
  (`exp(0.5·randn())`, clipped to the existing ±1e4 numerical safety box). This is implemented as a new,
  **additive** script in the production worktree,
  `sequential_gravity/run_profiled_production_phase5_multistart.jl` — a byte-for-byte diff against the
  tracked `run_profiled_production.jl` shows the only changes are: a `START_ID` env var, the
  `phase5_perturbed_start` function, and using its output as the per-bound-direction starting point
  instead of the raw calibration `θ_r0`. The tracked production file itself was **not modified**; this
  new file remains untracked in the production worktree (ground rules: no commits there).
- Fresh, uniquely-named `OUT_DIR`s: `sequential_gravity/batch_out_phase5_multistart_20260718_065235/start{0..4}/`
  — did not touch `batch_out`, `batch_out_v2`, `batch_out_d4x_continuation`, `batch_out_gravityseed`,
  or `batch_out_w80000`.
- Verified `D`, `W`, seeds, `baseIndex`, σ inside the actual JLD2 result files (not filenames) — each
  result dict carries `"D"=>4`, `"seedFakeData"=>889`, `"seedU"=>888`, `"W"=>8000`,
  `"baseIndex"=>2` explicitly (added to the new script's save block precisely to make this
  in-file-verifiable, per the task's warning about a past stale-D=10-file mixup).

### 1.3 Results — all 5 starts × 2 bound directions

| start | bound | KNITRO status | raw κ | raw feasible | best-feasible κ | best-feasible feasible | wall (s) |
|---|---|---|---|---|---|---|---|
| 0 (A*-anchored) | lower | -102 | 0.028583 | true | **0.005280** | true | 138.1 |
| 0 (A*-anchored) | upper | -102 | 0.119570 | true | **0.122005** | true | 167.6 |
| 1 (random) | lower | -102 | 0.009341 | true | 0.009341 | **false** | 593.5 |
| 1 (random) | upper | -102 | 0.126873 | true | **0.126979** | true | 147.0 |
| 2 (random) | lower | -101 | 0.008365 | true | **0.005570** | true | 281.9 |
| 2 (random) | upper | -102 | 0.138703 | false | **0.157907** | true | 163.8 |
| 3 (random) | lower | -103 | 0.022080 | false | NaN | false (none found) | 71.1 |
| 3 (random) | upper | -103 | 0.106567 | false | NaN | false (none found) | 140.9 |
| 4 (random) | lower | -502 | 0.016426 | true | **0.015312** | true | 461.9 |
| 4 (random) | upper | -101 | 0.153933 | true | **0.154320** | true | 81.3 |

Per `sequential_methodology.tex`'s multistart rule ("the reported number at each δ is the best
across all five starts"): **best-of-5 upper-bound κ = 0.157907 (START_ID=2)**; best-of-5
lower-bound κ = 0.005280 (START_ID=0). One start (`START_ID=3`) found no feasible point at all in
either direction within `maxit=150` — a genuine multistart miss, not a bug (its random `A_od`
perturbation, up to `[8.6, 6.8]`-scale, evidently landed the search somewhere it could not recover
gravity feasibility in the iteration budget; exactly the failure mode multistart exists to guard
against).

**This alone is already dramatically different from, and internally consistent unlike, the retracted
run**: 0.157907 clears the `κ_fixedA=0.1440` floor by a comfortable margin (as a genuinely converged
sequential search must), unlike the retracted 0.0779.

## 2. Part B — reconstructing the full-A point

### 2.1 Locating the destination-share inversion machinery

Confirmed by direct search (not assumed): `sequential_gravity/profiled_gravity.jl::invert_destination`
(module `ProfiledGravity`) is the **only** destination-share inversion machinery in the codebase; it
is called from `run_profiled_production.jl`'s own `seq_gravcol` for every omitted (non-focal)
destination, at softmax temperature ρ=2×10⁻³, gauge `u[ref=1,d]=0`. This is existing, validated
production code — not re-derived.

### 2.2 Reconstruction methodology

New script: `full_aod_diag/d4_exact/phase5_sequential_reconstruction.jl` (full derivation and
citations in its header comment). Summary:

**Focal column — no conversion needed.** Direct algebraic comparison of
`sequential_gravity/focal_moments_directgp.jl::EK_moments_focal_norm_directgp!`'s `AodPow[o]` formula
against `full_aod_diag/moments_gammanorm.jl::EK_moments_gammanorm_directgp!`'s `AodPow[o,d]` formula
(both quoted verbatim in the script header) shows they are **algebraically identical** with
`Acol[o] ≡ Aod_theta[o,focal]` — the sequential script's `θ[4:3+D]` free block already **is** the
full-A `γ_d≡1`-gauge free parameter for the focal column, bit for bit. Verified numerically, not just
algebraically: an independent round-trip (compute `u_focal` via the sequential script's own
`focal_u`, convert back to `Aod_theta` via the inverse formula below, compare to `Acol`) agrees to
**2.7×10⁻¹⁵ – 5.3×10⁻¹⁵** absolute error across all tested candidates — floating-point-exact.

**Omitted columns — genuine gauge conversion required.** `invert_destination` returns `u[·,d]` in the
*destination-inversion gauge* (`sequential_methodology.tex`'s own §"Gauge": `u[ref,d]=0`), a
different convention from full-A's `γ_d≡1`/`Aod_theta` parameterization. Conversion path (existing,
validated formulas, not re-derived): the gravity section's own
`log A_od = log(w_o) + log(τ_od) + u_od/(σ-1)` (identical to what
`profiled_gravity.jl::gravity_residual` already uses internally) gives the absolute level; inverting
full-A's own `Aod = Aod_theta·cHat·((w_o τ_od)/(w_1 τ_1d))^(1/μ)·(λ_od/λ_1d)` /
`AodPow=(Aod/cHat)^(-μ)` construction (`full_aod_diag/moments_gammanorm.jl`,
`full_aod_diag/d4_exact/winners.jl::factual_prices`) for `Aod_theta` — `cHat` cancels exactly (shown
algebraically in the script header) — gives:

```
AodPow[o,d] = exp(-(u[o,d]/(σ-1) + log(w_o) + log(τ_od)))
Aod_theta[o,d] = AodPow[o,d]^(-1/μ) / ( ((w_o τ_od)/(w_1 τ_1d))^(1/μ) · (λ_od/λ_1d) )
```

The full `θ_full` vector is then assembled (`γ_d≡1` slots inert, `γ'_focal` direct, `Aod_theta`
matrix) and packed to `x_free` via the codebase's own `CS.FreeParamMap`/`pack_free` (not manual
indexing).

### 2.3 Exact-feasibility verification — result: **NOT exactly feasible, and why**

`evaluate_fullA` (`full_aod_diag/d4_exact/oracle.jl`) was run on the reconstructed `x_free` for both
the best-of-5 upper (START_ID=2, κ_seq=0.157907) and lower (START_ID=0, κ_seq=0.005280) points.

**Result: `inner_status = -300` (unbounded dual / primal infeasible) for both.** This was
**independently corroborated**, not just read off KNITRO's own status code, using the same
HiGHS/JuMP phase-I LP machinery this investigation already built and validated for exactly this
purpose (`full_aod_diag/d4_exact/phaseF_primal_feasibility_lp.jl`'s `lp_feasibility_check`, reused
verbatim in a new `full_aod_diag/d4_exact/phase5_lp_diagnose.jl`): **CERTIFIED_INFEASIBLE**, phase-I
minimum achievable max-|moment-residual| = **0.0876** (upper, START_ID=2) / **0.0487** (upper,
START_ID=0) — a real, nonzero floor, not numerical noise (Phase F's own reference range for known
genuinely-infeasible random-start points was 9.7×10⁻⁵–0.176, so this sits squarely in "genuinely
infeasible" territory by that same yardstick).

**Root-cause diagnosis (not left as an unexplained failure):**

1. **The focal-column and winner-selection algebra are independently confirmed correct** — the
   1e-15 round-trip above, plus a direct derivation showing `hFunction.jl`'s hard-max winner rule
   (`argmin` over `w_o·AodPow[o,d]·τ_od·U_o^μ`) and `profiled_gravity.jl`'s softmax winner rule
   (`argmax` over `u[o,d]+log_x[s,o]`) are the same ordering up to a positive rescaling by `(σ-1)` —
   so this is not a units/sign bug in the conversion formula itself.
2. **A direct, Newton-independent check** (`winners.jl::compute_winners`, already validated against
   `hFunction!`'s own `G` output) of the **hard-max**, `p_seq`-weighted trade shares implied by the
   reconstructed `Aod_theta` shows they are **close** to the data targets at the *share* level — e.g.
   destination 4 (the country with by far the largest domestic/own-trade share, λ₄₄=0.859):
   hard-max shares `[0.020, 0.094, 0.025, 0.861]` vs. target `[0.026, 0.077, 0.038, 0.859]` — gaps of
   0.2–1.7 percentage points, not tens of percent.
3. **But the corresponding *raw moment* residual is large (≈1.05–1.44) specifically for the
   origin=destination=4 (domestic) moment**, ~30× every neighboring moment in the same destination
   column. Tracing this into `moments/hFunction.jl`'s own formula
   (`G[ω,d1] = pricesTempσ[o]·1{winner} − λ_od·γ[d]^σ·gdp[d]`) shows the *scale* of this one moment is
   set by `pricesTempσ[o] ∝ (AodPow[o,d]·τ_od)^{1−σ}`, and the reconstructed `Aod_theta[4,4]` is
   **economically extreme** (≈41–122, vs. the calibration value of 1) — i.e. this specific point
   requires an implied domestic-technology level for country 4 far outside the calibrated region, which
   inflates the *raw-unit* scale of exactly this one moment enough to turn a 0.2–1.7-point share gap
   into a large absolute residual.
4. **A ρ→0 (hard-max) continuation re-inversion of the omitted columns, warm-started from the
   ρ=0.002 solution** (`invert_destination(...; ρ=0.0, u_init=...)`, already-existing machinery, no
   new formula) was attempted as a candidate fix. It **did not fully converge** (share error plateaus
   at ≈5×10⁻⁴ after 300 iterations rather than reaching the 1×10⁻¹⁰ tolerance — the exact-line-search
   Newton cycling at a dense finite-sample kink this investigation's memory has repeatedly documented
   as a known hard-max fragility), and, tellingly, **the resulting moment residual barely moved**
   (1.0539→1.0538) despite the u-vector itself shifting only ≈10⁻⁴–10⁻³ — ruling out "softmax vs.
   hard-max numerical gap, fixable by tightening ρ" as the explanation. The residual is a genuine
   share-level/scale-interaction effect at this specific extreme point, not a smoothing artifact.

**Conclusion for this section**: the gauge-conversion *formula* is verified correct (focal round-trip
to 1e-15, winner-rule algebra, and share-level agreement within ~1–2 percentage points at the omitted
destination checked in detail). What is **not** established is that the reconstructed point is
*exactly* feasible in full-A's own hard-max, D²-bilateral-moment oracle — it is not, both by KNITRO's
own `-300` and by an independent LP certificate, because the sequential search (in achieving a higher
κ than the fixed-A floor) drove the implied `Aod_theta` for the omitted columns to economically
extreme levels at which small, real share-level gaps get amplified by scale into a nonzero
moment-matching floor of ≈0.05–0.09 (raw units) that even the optimal reweighting cannot close under
δ=1's own additional constraint.

## 3. Part C — interpretation

**The properly-run sequential result does NOT exceed the full-A incumbent, and does not need the
exact-feasibility question resolved to reach that conclusion**, since the reconstructed point's own
`κ` (0.157907, read directly off `γ'_focal` — a closed-form, gauge-invariant function of `θ[3]`
alone, unaffected by whether the inner divergence-minimization re-solve succeeds) is already below
`0.17176461388430053`, and the LP diagnosis in §2.3 gives no reason to think a corrected/exactly-feasible
version of this point would score *higher* (if anything, tightening the point to satisfy the extra
D²-moment constraints full-A imposes should be expected to cost some κ, not gain it, since the
sequential search never had to satisfy the omitted-column moments as *hard* constraints during its own
search — it only had to satisfy them well enough for its own destination-inversion convergence
criterion).

**Final comparison table** (D=4, W=8000, δ=1, seedFakeData=889, seedU=888, baseIndex=2, σ=2.5):

| candidate | κ | note |
|---|---|---|
| `fixed_A_benchmark` (A held at calibration A\*, only γ'_focal optimized) | 0.1439805232 | trivial floor |
| **full-A incumbent** (maxit=40, `EXACT_FEASIBLE_CANDIDATE`=true, `ROBUST_LOCAL_CANDIDATE`=**FALSE**) | **0.17176461388430053** | current headline; not yet independently re-verified as a robust local optimum |
| sequential, properly run + reconstructed (**upper**, best of 5 genuine starts) | **0.1579066256** | beats fixed-A floor by +0.0139; falls short of full-A incumbent by −0.0139; exact full-A-oracle feasibility **not certified** (§2.3) |
| sequential, properly run + reconstructed (**lower**, best of 5 genuine starts) | 0.0052797574 | for reference only; not the direction being compared against the incumbent |

**Plain statement, as required**: the properly-run sequential result (κ=0.157907) **falls short** of
the current full-A incumbent (κ=0.171765) by about 0.014 in κ (roughly 8% relative). It does,
however, **clearly beat** the trivial fixed-A floor (κ=0.144) by about 0.014, confirming — unlike the
retracted 0.0779 run — that this is now a genuinely converged, internally consistent sequential
search. No rounding or hedging: **0.157907 < 0.171765**. This does **not** support flagging the
full-A incumbent as nonoptimal, and the sequential result should **not** be recommended as a
mandatory warm-start for the next full-A attempt on the grounds of exceeding it — it doesn't. (It
could still be a *reasonable* diagnostic warm-start on general "diverse starting point" grounds, but
that is a much weaker claim than what finding it superior would have supported, and is not pursued
further here.)

## 4. What this phase did **not** complete / open caveats

- **Exact full-A-oracle feasibility of the reconstructed sequential point was not achieved** — see
  §2.3. The gap is diagnosed (concentrated in the dominant domestic-trade-share moment at an
  economically extreme implied `Aod_theta` entry) but not closed. A follow-up could try: (a) a
  proper ρ-continuation *schedule* (several small steps rather than one direct ρ=0.002→0 jump) for
  the omitted-column re-inversion, since the single-step attempt here got stuck at a kink; (b)
  restricting the multistart's random `A_od` perturbations to a narrower range, trading off some of
  the κ gain for a reconstruction more likely to land in a numerically well-conditioned region; (c)
  running full-A's own outer search *warm-started* from the reconstructed (inexactly-feasible) point
  directly and letting KNITRO's own inner solve pull it onto the feasible manifold, rather than
  insisting the raw reconstruction itself be exactly feasible.
- **START_ID=3 found no feasible point in either direction** within `maxit=150` — consistent with
  expected multistart behavior (not every start should be expected to succeed), but not
  investigated further (e.g. whether a larger `maxit` would rescue it).
- **δ grid**: only δ=1 was run (matching the candidate registry's synthetic economy), per the task
  scope; the production methodology's own δ grid (0.1, 1.0, 2.0, 5.0) was not swept here.
- **W=8000 only** (matching the synthetic economy used throughout this investigation) — the standing
  W-stability caveat from `docs/fullA_d4_recommendation.md` (Phase G, not yet resolved) applies
  equally to this comparison.

## 5. Artifacts

- Production worktree (untracked, not committed, per ground rules):
  - `sequential_gravity/run_profiled_production_phase5_multistart.jl` — additive 5-start wrapper.
  - `sequential_gravity/batch_out_phase5_multistart_20260718_065235/start{0..4}/seq_{lower,upper}_delta1.jld2`
    — 5-start × 2-bound raw results.
  - `sequential_gravity/run_phase5_start{0..4}_20260718_065235.log` — full KNITRO logs (grepped for
    the hessopt-fallback string with zero matches; grepped for `-400` with zero matches).
- Diagnostic worktree (this branch):
  - `full_aod_diag/d4_exact/csw_outer_seqprod_fcga_no_maxit150.opt` — corrected KNITRO options file.
  - `full_aod_diag/d4_exact/phase5_sequential_reconstruction.jl` — reconstruction + gauge-conversion
    methodology (context builder, `seq_gravcol` re-host, `aod_theta_from_AodPow`,
    `reconstruct_fullA_point`).
  - `full_aod_diag/d4_exact/phase5_run_comparison.jl` — driver: loads all 5×2 results, picks best
    per bound, reconstructs, evaluates in the exact oracle, prints the final comparison table.
  - `full_aod_diag/d4_exact/phase5_lp_diagnose.jl` — independent LP feasibility diagnostic (reuses
    `phaseF_primal_feasibility_lp.jl`'s `lp_feasibility_check`) plus the share-level/root-cause
    breakdown in §2.3.
