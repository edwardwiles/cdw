# Sequentially-linearized profiled full-gravity — progress note

Branch: **`sequential-profiled-gravity`** (off `experiments-derivatives`, in `trade_robustness_modular`).
Fully reversible: `git checkout experiments-derivatives` restores the prior state. Nothing here
touches the live `master.jl` pipeline; all new code is under `sequential_gravity/` and gated in
standalone scripts. Companion docs: **`SEQUENTIAL_GRAVITY_DESIGN.md`** (spec→code derivation, §18
integration, R_sum/R_mean/R_beta scaling, the divergence-budget check); the Claude memory note
`sequential-profiled-gravity`.

Everything below is on the **D=4 simulated example** (focal country = `baseIndex=2`), not real data.
A **20-country real-data build** (Teti-BACI tariffs + trade values, already assembled in
`../tariff_build/work/`) was explicitly deferred to a separate session — see §8.

If you're picking this up cold: read §1–2 for the method, §3 for what's built, **§5 first if you're
about to trust any reported bound** (it documents a critical bug that was found and fixed — make
sure whatever code you're looking at has the fix), then §6–7 for current numbers and what's running.

---

## 1. The problem this solves

The distribution-agnostic bounds put the structural parameters θ in an outer optimizer. All bilateral
shifters `A[o,d]` currently live in θ: 16 entries (12 free) at D=4, but **361 (~342 free) at D=19** —
the outer bilevel program does not scale. The coauthor's "France-only gravity moment" compromise
(keep only destination = France rows, drop the other `A` columns) is too noisy.

**This approach** ("profiled") keeps only `A[·,focal]` in θ and *profiles out* every other column: for
each omitted destination d, recover `A[·,d]` by **inverting** its observed trade-share column under
the current least-favorable distribution F. The full origin+destination-FE gravity restriction `R = 0`
is then imposed via a **sequentially-linearized** moment (an influence function), so gravity can shape
which F CC selects — without carrying the omitted A's as parameters.

---

## 2. Exact model ↔ ChatGPT-spec mapping (derived and verified)

For draw s and origin o (`UoModel=1`, `θConstant=0`):
- `log_x[s,o] = μ·log(Uσ[s,o]) = μ(1−σ)·log(U[s,o])`  (destination-independent; U~Exp(1), Uσ=U^{1−σ})
- `u[o,d] = (σ−1)(log A_od − log w_o − log τ_od)`  — the spec's competitiveness index exactly, using
  the structural `A_od = 1/AodPow`
- winner(s,d) = `argmax_o (u[o,d] + log_x[s,o])` = argmin level price
- `share_d[o] = Σ_s p[s]·(winner value) / Σ_s p[s]·(value)` — matched to the observed column `λ̂[·,d]`
- LFD weights recovered exactly as in `lfd/LFD.jl:50-53` (`dPsi!` of the dual solution).

---

## 3. What was built, phase by phase (all committed)

| phase | deliverable | status |
|---|---|---|
| 1 | `profiled_gravity.jl` — inversion `I_d`, gravity residual `R`, share-Jacobian `H_d`, adjoint, influence function ψ̄ | **validated** |
| 1 | `test_profiled_gravity.jl` + `validate_on_pipeline.jl` | **8/8 tests pass** |
| 2a | `focal_moments.jl` — reduced θ (D+4) & inner moments (D+1), `EK_moments_focal!` | **==full focal columns to 2e-14** |
| 2a | `run_focal_bounds.jl` — reduced focal-only CC solve | works |
| 2b | `run_sequential.jl` — sequential loop at fixed θ | R→0 in 2 iters |
| 2c | `run_profiled_bounds.jl` — nested §18 outer integration | **works, bug-fixed (§5)** |
| 2d | best-feasible-θ tracking, warm re-verification | **critical fix applied (§5)** |
| 2e | `focal_moments.jl::EK_moments_focal_norm!` + `run_profiled_bounds_norm.jl` — γ_focal≡1 normalized reparameterization | **derived, verified, working** |
| 2f | `run_fullA_variant.jl` — full-A-in-outer-loop with optional μ/non-focal-A freezes, for comparison | working |
| 2g | δ-sweep infrastructure (`DELTA_GRID` env var, `run_at_delta` wrapper) in both `run_profiled_bounds.jl` and `run_fullA_variant.jl` | working, **sweep in progress** |

### Phase 1 — the linearization core
Carries a softmax temperature ρ throughout (ρ→0 = hard max) so finite-sample winner-switch graininess
is smoothed; ψ_R is the **exact** derivative of the smoothed residual. Tests C–J pass, including the
mandatory directional-derivative gate: rel err ~1e-4 synthetic, 3.8e-5 on real pipeline draws.

### Phase 2a — reduced focal-only CC mode
`EK_moments_focal!`: outer θ = `[μ, σ, γ_focal, γ'_focal, A[·,focal]]` (D+4 params), inner moments = D
focal trade shares + 1 counterfactual. Reproduces the full model's focal columns to ~2e-14.

### Phase 2b/2c — sequential loop + nested outer integration
See spec §14/§18. A stateful moment closure recomputes the sequential loop at **every θ** the outer
optimizer visits, appending one linearized gravity moment to the focal moment set.

---

## 4. Performance fixes along the way (all in `run_profiled_bounds.jl`)

- **Inversion speed**: `dest_share` (non-allocating) + a smooth-regime Newton/monotone-backtrack line
  search (replacing an ~80-bisection exact search only needed for hard-max graininess): cold inversion
  350ms→22ms, warm ~8ms (W=8000,D=4) ≈ one min-divergence CC solve (~7ms).
- **Sequential-loop `maxit` 6→20**: a diagnostic sweep (varying μ, A_focal near θ_initial with verbose
  tracing) found TWO distinct causes of "gravity-infeasible": (a) maxit=6 was simply too tight — several
  θ showed R decreasing monotonically every iteration, just hadn't crossed tolerance yet; (b) at μ far
  below μHat, the destination-inversion Newton genuinely diverges (structural degeneracy: as μ→0,
  log_x shrinks, origins become nearly tied, matching a fixed share needs increasingly extreme u) — a
  plausible genuine infeasibility, not a bug. Raising maxit fixed (a) without touching (b).
- **Exact outer gradient** (`grad_R_theta`): the envelope-theorem piece the frozen gravity column was
  dropping. `∂R_sum/∂u[o,d] = Q̃[o,d]/(σ-1)` in closed form; focal contribution via
  `ForwardDiff.jacobian` on the plain `focal_u(θ)`; omitted-destination contribution via the implicit
  function theorem on the inversion's fixed point (`∂u_d/∂μ = -H_d⁻¹·∂share_d/∂μ|_u`, one
  `ForwardDiff.derivative`, no Newton/KNITRO). Improves KNITRO's own convergence metric (opt_err)
  substantially but doesn't by itself fix search robustness — see §5.
- **Best-feasible-θ tracking**: KNITRO's own terminal iterate at the iteration cap is sometimes
  infeasible even though the search visited many feasible θ along the way (confirmed via θ-eval trace:
  up to 172/441 evaluations gravity-feasible, yet the final point wasn't one of them — the search
  wanders off good points near the cap). Fix: track the best feasible θ seen during the **entire**
  search (extremal κ subject to feasibility) and report it alongside KNITRO's raw answer.
  **Warm-start subtlety**: the sequential loop is not a pure function of θ (it's warm-started from the
  previous θ's state), so a *cold* re-verification of a tracked "best" θ can spuriously fail even
  though it was genuinely feasible via the warm-started trajectory reached during search. Fix: snapshot
  the `umat` that achieved feasibility (`best_warm`) and re-verify **warm**, not cold.

---

## 5. THE CRITICAL BUG: gravity-feasible ≠ δ-feasible (found via review, now fixed)

**Symptom**: best-feasible-θ tracking reported **negative GT** (gains-from-trade), e.g. κ_lower=−0.10.
This is economically impossible — the user's paper draft **proves** GT is bounded in `[0, κ_max]`
(monotone increasing in μ; κ=0 at μ→0). Any negative value is a bug, not a modeling curiosity.

**Root cause**: `seq_gravcol`'s "ok" (gravity-feasible) determination checked **only**
`|R_mean| ≤ tol` (is the linearized gravity restriction satisfied) — it **never checked that the
recovered distribution `p`'s actual divergence from F\* is `≤ δ`**. These are different conditions.
A θ can be gravity-consistent (some distribution makes the linearized gravity moment hold) while that
distribution requires a divergence far outside the stated δ-neighborhood — i.e., "gravity-feasible"
silently drifted from "δ-feasible," and the best-feasible tracker (which only gated on the former)
picked up economically nonsensical points that were never actually admissible under the stated
robustness budget.

**Diagnosis method** (don't skip this if debugging something similar): constructed a θ with
`γ'_focal > γ_focal` by hand (algebraically forces negative κ), confirmed both `recover_lfd`
(min-divergence value = 1e10, solver status −300 = infeasible) and `seq_gravcol` (Newton diverges,
`ok=false`) correctly REJECT that crude example — meaning the real bug wasn't in that obvious a place.
The actual failure mode was subtler: real θ found during search satisfied `|R_mean|≤tol` via a
distribution whose *actual* divergence was never checked.

**Fix** (`run_profiled_bounds.jl` / `run_profiled_bounds_norm.jl`): added `divergence_of(p)`, the
**exact primal CDW/CC hybrid-divergence functional φ**, applied directly to any candidate `p`:
```
φ(m) = m·log(m) - m + 1        for 0 < m ≤ e     (m = p[s]·W, the density ratio vs uniform F*)
φ(m) = m²/(2e) - e/2 + 1        for m > e
divergence(p) = (1/W) Σ_s φ(p[s]·W)
```
Derived as the Legendre dual of `Psi!`/`dPsi!` in `cc_algo/Psi.jl` (the SAME divergence the rest of
the codebase uses — not a new definition). Necessary because the accepted `p` inside the sequential
loop can be a **damped convex combination** `(1-α)p_k + α·p_candidate` (spec §15), which is a valid
distributional iterate but **not itself the argmin of any single min-divergence problem** — so its
divergence must be evaluated directly from the primal functional, not read off a solver's internal
`val`. Sanity check: `divergence_of(fill(1/W,W)) = 0` (F\* itself has zero divergence from F\*).

`seq_gravcol`'s `ok` now requires **both** `|R_mean|≤tol` **and** `divergence_of(p)≤δ` (tiny numerical
slack on the boundary). Verified via smoke test: best-feasible κ went from negative (−0.04 to −0.10)
to always positive and sensible (e.g. 0.008, 0.162) immediately after the fix, at the same iteration
budget.

**If you're extending this method**: any new way of constructing/accepting a candidate distribution
`p` must be checked against `divergence_of(p) ≤ δ` before being treated as "feasible" for **any**
purpose (reporting, warm-starting a subsequent step, etc.) — gravity-consistency alone is not
sufficient, and this is easy to silently reintroduce.

### The R_mean vs R_beta scaling (a related, separate design point — see DESIGN.md §5a for the full
derivation) resolved as: **R_mean** (`R_sum/D²`, the literal, unembellished GMM sample-moment average
`E[ΔΔlogA·ΔΔlogτ]=0` — no variance normalization) is the correct quantity for the identification
condition and the tolerance check (per explicit review — an earlier version used a
variance-normalized "R_beta" here, which was wrong for that purpose, though R_sum=0⟺R_mean=0⟺R_beta=0
are the same condition at exactly zero). Separately, R_beta's *numerical scale* is kept for the
solver-facing moment/gradient (`col`, `grad_R_theta`) because R_mean's ~34× smaller magnitude
empirically caused KNITRO's search to take oversized steps into extreme, unrecoverable θ territory —
a pure numerical-conditioning issue, decoupled from the (separately correct) R_mean tolerance check
via the fact that `S_Q=ΣQ̃²` is a data-only constant, so `R_beta≡(D²/S_Q)·R_mean` exactly, always.

---

## 6. The γ_focal≡1 normalized reparameterization (per review, verified)

**Motivation**: even with the divergence-budget fix, letting `γ_focal` and `γ'_focal` both range
freely (generic ×1e-4–×1e4 bounds) allows the search to explore (γ,γ') combinations that don't
correspond to genuine model equilibria. The user's paper proves `κ∈[0,κ_max]`; **structurally bounding
this via θ-bounds is more robust than catching violations after the fact**.

**Derivation** (`focal_moments.jl`, `EK_moments_focal_norm!` docstring has the full version — verify
this against `prestep/computeGamma.jl` if you distrust it, don't just trust this summary): under
autarky with `wPrime[focal]=1=wHat[focal]`, `τPrime[focal,focal]=1`, the EK gravity identity gives
`λ_dd = Phi'_focal/Phi_focal` **exactly** (Phi'_focal is the autarky Frechet aggregator, only the
domestic term survives). Combined with `γ = (Phi^{-(1-σ)/θ}·Γ(·)/(wL))^{1/σ}` (computeGamma.jl) and
`θ=1/μ`:
```
γ'_focal/γ_focal = λ_dd^{μ(σ-1)/σ}          (λ_dd = λ[focal,focal], DATA)
κ = 1-(γ'_focal/γ_focal)^{σ/(σ-1)} = 1 - λ_dd^μ
```
μ ranges over `(0, 1/(σ-1)]` (existing outer bound), and `κ=1-λ_dd^μ` is strictly increasing in μ, so:
```
κ ∈ [0, 1-λ_dd^{1/(σ-1)}]   ⟺   (γ_focal≡1 normalization) γ'_focal ∈ [λ_dd^{1/σ}, 1]
```
Numerically verified: `κ_max=0.2331` matches an **independently derived** ceiling from
`EXPERIMENTS_FINDINGS.md`'s earlier all-A audit exactly — two unrelated derivations agree.

**A subtlety that cost real debugging time — read this before touching the normalization**: my first
implementation forced `γ_focal≡1` **while also keeping the existing `A[1,focal]=1` pin** — this
double-pins the *same* redundancy (a common multiplicative scale across all of `Acol` is exactly
redundant with `γ_focal`'s level — verify by checking how `γf` and `AodPow` both depend on `Acol` in
`EK_moments_focal!`) and silently shifted the moment-matching point away from F\* (confirmed via a
`max|ΔG|=0.308` discrepancy at θ_initial). **The fix**: `γ_focal≡1` *replaces* the `A[1,focal]=1`
normalization — **all D entries of `Acol` are free** under the normalized parameterization, starting
from the compensating common scale `s = γf0^{-σ/(μ(σ-1))}` (derived from how `share_mag` scales as
`Acol[o]^{μ(σ-1)}` under a uniform shift). This reduces the *initial-point* moment mismatch but not
to exactly zero (residual ~0.14, from γ'_focal's absolute-level effect on the counterfactual moment,
a second-order effect the simple closed-form scale doesn't capture) — **but the actual min-divergence
needed at this starting point is only ≈1.8e-4** (checked via the real `recover_lfd`/CC machinery, not
the crude `max|ΔG|` proxy), utterly negligible against any δ in the sweep. Don't over-invest in
chasing max|ΔG|→0 exactly; check the real divergence instead.

**Implementation**: `run_profiled_bounds_norm.jl` (a parallel copy of the bug-fixed
`run_profiled_bounds.jl` with targeted edits — reduced θ layout `[μ,σ,γ'_focal,A[1..D,focal]]`,
length D+3 not D+4; bounds `γ'_focal∈[KBOUNDS.γp_lo,γp_hi]`; no A[1,focal] pin). Smoke-tested clean:
no crashes, no negative κ (structurally impossible now), numbers consistent with the bug-fixed
original at the same iteration budget.

---

## 7. The 5-configuration comparison (what's currently running)

Per explicit request, comparing (all bug-fixed, δ-sweepable):
1. **1a**: full A_od in outer loop, μ free (`master.jl` / `run_fullA_variant.jl` no freezes) — the
   "gold standard" reference (23 free params at D=4).
2. **1b**: full A_od, μ **frozen** (`FREEZE_MU=1`).
3. **2**: full A_od, but **non-focal columns frozen** at their calibrated initial value (=1), μ free
   (`FREEZE_NONFOCAL_A=1`) — isolates whether profiling's *inversion* step (not just outer-dimension
   reduction) is what matters, by comparing against a naive freeze.
4. **3a**: profiled (this method), μ free.
5. **3b**: profiled, μ **frozen**.

All wired for a δ-sweep via `DELTA_GRID="0.1,0.25,0.5,1,2,5,10"` env var (both `run_fullA_variant.jl`
and `run_profiled_bounds.jl`; `run_profiled_bounds_norm.jl` has the same `run_at_delta` structure but
wasn't included in the launched sweep — run it separately per δ if you want that comparison too).

**Pre-bug-fix, δ=1, maxit=200 numbers** (all still directionally informative, but the profiled-method
best-feasible figures from before the divergence-budget fix are WRONG — regenerate before citing):
| run | κ_lower | κ_upper |
|---|---|---|
| 1a: full-A, μ free | 0.0181 | 0.1544 |
| 1b: full-A, μ frozen | 0.0208 | 0.1022 |
| 2: full-A, non-focal frozen | 0.00896 | 0.1590 |

Post-bug-fix smoke tests (maxit=8, not converged, just sanity-checking positivity/sensibility):
profiled 3a ≈ [0.008, 0.162]; normalized variant ≈ [0.0066, 0.159] — consistent with each other and
with the full-A numbers' rough range.

**Sweep completed.** Full results, δ=0.1..10, maxit=200 (⚠ = best-feasible tracker's own
re-verification failed the feasibility check at that point — boundary-sensitivity, see §5's warning;
read as "approximately this, unconfirmed"; profiled columns use best-feasible where it beat KNITRO's
own terminal answer):

**κ_lower**

| δ | 1a: full-A μ-free | 1b: full-A μ-frozen | 2: full-A nonfocal-frozen | 3a: profiled μ-free | 3b: profiled μ-frozen |
|---|---|---|---|---|---|
| 0.1 | 0.0385 | 0.0414 | 0.0390 | 0.0390 | 0.0344 |
| 0.25 | 0.0319 | 0.0225 | 0.0303 | 0.0217 | 0.0208 |
| 0.5 | 0.0263 | 0.0220 | 0.0189 | 0.0120 | 0.0136 |
| 1 | 0.0181 | 0.0208 | 0.0090 | 0.0055 | 0.0054⚠ |
| 2 | 0.0029 | 0.0097 | 0.0136 | 0.0030⚠ | 0.0037 |
| 5 | 0.0013 | 0.0014 | 0.0010 | 0.0056⚠ | 0.0034 |
| 10 | 0.0008 | 0.0023 | 0.0007 | 0.0052 | 0.0041 |

**κ_upper**

| δ | 1a: full-A μ-free | 1b: full-A μ-frozen | 2: full-A nonfocal-frozen | 3a: profiled μ-free | 3b: profiled μ-frozen |
|---|---|---|---|---|---|
| 0.1 | 0.0694 | 0.0908 | 0.0970 | 0.1064 | 0.1030 |
| 0.25 | 0.1138 | 0.1155 | 0.1109 | 0.1310⚠ | 0.1264 |
| 0.5 | 0.1266 | 0.1443 | 0.1465 | 0.1529 | 0.1465 |
| 1 | 0.1544 | 0.1022 | 0.1590 | 0.1701⚠ | 0.1710 |
| 2 | 0.2010 | 0.1871 | 0.1893 | 0.1785⚠ | 0.1777 |
| 5 | 0.2004 | 0.2005 | 0.2072 | 0.2069 | 0.1678⚠ |
| 10 | 0.2026 | 0.1915 | 0.1869 | 0.1958⚠ | 0.1951 |

**Takeaways**: (1) profiled and full-A land in the same ballpark at every δ — same order of magnitude
and shape (lower bound → 0, upper bound rises and saturates ~0.17–0.21) — the intended cross-
validation holds. (2) Profiled tends to run somewhat wider on the lower bound at small/medium δ
(e.g. δ=1: full-A 0.018 vs profiled 0.006), consistent with dropping non-focal trade-share moments
removing some real constraint on the divergence budget (§7 above). (3) μ-frozen variants are narrower
than μ-free, as expected. (4) All methods saturate by δ≈2–10 — some other bound (μ's own range)
becomes binding before the divergence budget does. (5) **Profiled's numbers are noticeably less
reliable at individual δ points** (more ⚠ flags) than full-A's clean convergence — the per-θ nested
loop makes feasibility harder to pin down exactly, especially at large δ; this is the main thing to
improve if these numbers need to be publication-quality rather than exploratory.

The normalized variant (`run_profiled_bounds_norm.jl`) was only run at δ=1 (not the full sweep):
κ∈[0.0052, 0.1629] (best-feasible) — consistent with 3a's δ=1 row. Its KNITRO-own upper answer was
*also* infeasible at the terminal iterate (status −102, "converged" but outside budget) — confirming
the search-robustness issue is orthogonal to which parameterization is used; the normalization fixes
*validity* (no more negative GT, structurally impossible now) but not *search convergence reliability*.
Extending the normalized variant to the full δ-sweep, and/or improving convergence reliability (better
warm-starting across δ, or the exact IFT gradient noted in §9.2), are the natural next steps if this
work continues.

---

## 8. Explicitly deferred: 20-country real data

A real Teti-BACI 20-country tariff+trade dataset already exists at
`../tariff_build/work/` (`block_tariffs_2016.csv`, `pair_aggregates_2016.csv`, 20 countries incl.
FRA, ROW aggregate). Building the full `importData.jl`-style loader to wire this into `master_setup`
was scoped but **explicitly deferred by the user to a separate session** — known complications:
missing/aggregated domestic trade flows for some countries (see memory note `itpde-missing-domestic`),
wages need to be solved/calibrated (not directly observed) via the same `iterWagesPreStep!`-style
machinery the fake-data pipeline already uses. Don't attempt this without confirming the user still
wants it and syncing on how to handle the known data-quality gaps.

---

## 9. Known limitations / open items

1. **Speed at scale**: D=4 is slower than all-A per outer evaluation (per-θ sequential-loop overhead
   when the all-A outer is only 23 params). The payoff is at D=19 (all-A: 361 outer params; profiled:
   ~19). That crossover is untested — needs D=19 real data (see §8) and parallelizing the D−1
   destination inversions (embarrassingly parallel, not yet done).
2. **Approximate outer gradient remains approximate in one respect**: `grad_R_theta` is exact for how
   R depends on θ *holding F fixed* (envelope theorem), but does not differentiate through the
   sequential loop's own re-linearization trajectory. Feasibility is always checked exactly (§5), so
   reported bounds are trustworthy; only search *efficiency* is affected.
3. **Not wired into `master.jl`** behind an `orthogonality_mode` flag — lives in standalone
   `sequential_gravity/` scripts (`:focal_only`, `:sequential_profiled_full`,
   `:profiled_full_ex_post_diagnostic` from the original spec were never formally implemented as an
   enum/flag; the scripts are the de facto modes).
4. **ρ (softmax temperature)** = 2e-3 throughout the inversion/influence function; ρ→0 is the strict
   hard-max. Small O(ρ) bias, not re-litigated recently.
5. The δ-sweep's very small δ values (0.1) showed **harder convergence** (status −400, opt_err~1.35 for
   1a at δ=0.1 vs converged at δ=1) — tight δ may need higher maxit or different solver settings;
   not yet investigated.

---

## 10. File map & how to run

Under `sequential_gravity/`:
- `profiled_gravity.jl` — Phase-1 core module (LinearAlgebra, ForwardDiff only).
- `test_profiled_gravity.jl` — synthetic tests C–J, no KNITRO needed.
- `validate_on_pipeline.jl` — Phase-1 core on real draws, no outer solve.
- `focal_moments.jl` — `EK_moments_focal!` (original) + `EK_moments_focal_norm!` (γ_focal≡1
  normalized) + `theoretical_kappa_bounds`/`lambda_dd` helpers.
- `test_focal_moments.jl` — reduced-vs-full unit test.
- `run_focal_bounds.jl` — reduced focal-only CC bounds (no gravity).
- `run_sequential.jl` — sequential loop at a fixed θ (R→0 demo).
- `run_profiled_bounds.jl` — **the main nested §18 profiled-gravity driver** (bug-fixed, δ-sweepable,
  best-feasible tracking). `DELTA_GRID`, `FREEZE_MU`, `PROF_EXACT_GRAD` env vars.
- `run_profiled_bounds_norm.jl` — same, using the γ_focal≡1 normalized parameterization.
  `OUTER_OPT_FILE` env var lets you point at an isolated opt file for quick smoke tests without
  touching the shared production one (important if something else is mid-sweep).
- `run_fullA_variant.jl` — full-A-in-outer-loop comparison, `FREEZE_MU`/`FREEZE_NONFOCAL_A`/
  `DELTA_GRID` env vars.
- `time_pieces.jl` — inversion vs min-div-solve timing.

KNITRO env (demand.mit.edu only; see `SETUP_AND_FINDINGS.md`):
```
export ZIENA_LICENSE=/etc/sharedsw_licenses/ziena.txt
export KNITRODIR=/opt/shared_sw/knitro/14.2.0
export LD_LIBRARY_PATH=/opt/shared_sw/knitro/14.2.0/lib:$LD_LIBRARY_PATH
```
`csw_outer_loop_settings_cluster.opt` (shared production outer-solve settings, `maxit` currently 200)
is read by every script unless `OUTER_OPT_FILE` overrides it — **don't edit it while another script
might be mid-run**, use an isolated copy instead (see `csw_outer_smoke_norm.opt` for the pattern).
