# Continuation 10, Section 7: randomized QMC precision investigation

Branch `c10-qmc-is` (worktree `gravity-fullA-d4-c10-qmc-is`, forked from
`diag/fullA-d4-exact` @ `690b8f5`), machine `demand.mit.edu`,
`JULIA_NUM_THREADS=20`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Real
D=20 France-focal data (`real_data/noah_D20/`), production W=80,000.
Isolated — no production file touched (see `full_aod_diag/d4_exact/
qmc_context_real_d20.jl`'s header for the exact fork boundary; confirmed at
production scale via `c10_phase7_qmc_wiring_smoketest.jl`, EQUIVALENCE=PASS,
WIRING=PASS). Harness: `full_aod_diag/d4_exact/
c10_phase7_qmc_precision_comparison.jl` (72 fixed-point evaluations, raw
output `results/fullA_d4/c61317a/c10_phase7_qmc_precision_20260719_112614/`)
+ `c10_phase7_short_continuation.jl` (capped continuation, §5).

## 1. Background and hypothesis

CDW's own draft (`papers/CDW_Draft_June_2026.pdf`, sec 3.4) computes
expectations via "Monte Carlo integration with 800,000 draws" — plain
pseudorandom MC, the scheme this repository's production code
(`prepare_cc/drawU.jl::drawU` -> `genExpRands!`, ordinary `rand!` +
inverse-CDF) already implements. The underlying method paper, Christensen &
Connault (2023 ECMA), states in its own "Practical Considerations" section
(sec 3.1) that **in their empirical applications they instead used "a
randomized quasi-Monte Carlo approach based on scrambled Halton sequences as
in Owen (2017)"** — this task's QMC hypothesis is literally what the
method's own authors did. `cc_algo/rhalton.jl` (ported from Owen's own R
code) is already in this repository, unused for this purpose until this
task; it was validated independently first
(`c10_phase7_rhalton_validate.jl`) and passed every check — domain, per-dim
mean/variance vs Uniform(0,1), same-seed reproducibility, cross-seed
difference, low cross-dimension correlation (worst |corr|=0.022 at the real
target width d=20), a bin-variance discrepancy proxy far below plain
pseudorandom's (0.49 vs 87.1 at n=5,000), no index-0 degeneracy, and fast
production-scale generation (0.3s for W=80,000×d=20) — before being trusted
here.

The caveat given going in: this model's hard winner indicators (`argmin`
over origins per draw/destination, `winners.jl::compute_winners`) make the
moment integrand DISCONTINUOUS in the underlying uniforms. Classical QMC
error bounds are proved for integrands of bounded variation/sufficient
smoothness; a hard threshold does not obviously satisfy that, so "QMC beats
MC" does not mechanically transfer here.

**Headline finding: it does not transfer uniformly, and the actual pattern
is more informative than a flat null result. At the calibration point
(Δ_dual ≈ 0.001–0.003, very close to its own theoretical floor), both QMC
methods show a large, tightly-replicated (6/6 scrambles each) DOWNWARD LEVEL
BIAS relative to plain Monte Carlo (Halton 32.5% lower, Sobol 72.5% lower)
— AND, strikingly, at that same point the outer-loop gradient direction is
essentially unstable NOISE regardless of draw type (mean cosine similarity
across scrambles: 0.29 pseudorandom, 0.05 Halton, −0.07 Sobol — none of the
three methods produces a stable gradient there). At the delta=1 upper
(large-kappa) candidate — the actual decision-relevant, near-δ-boundary
region this investigation reports κ from — NEITHER effect replicates: Δ_dual
differences across draw types (≤4% at the candidate itself, ≤12% at two
small perturbations around it) are not clearly distinguishable from Monte
Carlo's own substantial seed-to-seed noise there (~8%), AND the gradient
direction is extremely stable and virtually IDENTICAL across all three draw
types (cosine similarity 0.999999, both within- and cross-drawtype).**

Key setup facts, confirmed by reading the code (not assumed): the F* draw
matrix is W×D = 80,000×20 (origin-indexed only — `UoModel=1`, this
investigation's fixed default, means `U[ω,o]` is shared across destinations,
NOT a W×D² matrix — confirmed via `prepare_cc/drawU.jl` and independently via
`winners.jl::factual_prices`). U is drawn exactly once per context build
(`master_prepare_cc.jl`) and never redrawn inside the KNITRO callback loop —
**common random numbers hold automatically** within any one optimization,
confirmed by grepping every `rand`/`Random` call in the `prepare_cc/`/
`cc_algo/` chain (none besides the one `drawU` call).

## 2. Method

`qmc_context_real_d20.jl` forks the production chain
(`build_ad_context_real_d20` → `master_prepare_cc`) into parallel `*_qmc`
functions accepting an externally-supplied, already-Exp(1)-transformed W×D
draw matrix instead of drawing it internally. Three draw generators
(`qmc_draws.jl`), all routed through `exp_from_uniform01` — a literal,
unedited copy of `genExpRands!`'s own `-log(1-u)` inverse-CDF transform:

- **pseudorandom_U**: Julia's default RNG, exactly production's own path,
  seed-controllable for independent replicates.
- **halton_U**: `cc_algo/rhalton.jl`'s scrambled Halton sequence.
- **sobol_U**: `Sobol.jl`'s `SobolSeq` (no built-in digital/Owen scrambling)
  + an independent Cranley-Patterson random shift (mod 1) per replicate —
  the standard simple RQMC randomization when digital scrambling isn't
  available. **This is a strictly weaker randomization than Halton's
  per-digit permutation scrambling — flagged explicitly.** `Sobol.jl` was
  added as a project dependency; the resolve was verified clean (only Sobol
  + its 2 already-present stdlib deps, `Manifest.toml` diffed to confirm).

Five fixed points (natural-theta `w=[gamma', pivot-reduced zfree]`, 399 free
A-directions after the exact gravity-elimination pivot):

1. **calibration**: `gp0=0.987762`, A at natural-theta calibration.
2. **upper candidate**: `gamma'=0.955701` (delta=1, large-kappa branch, this
   investigation's headline result), A from the **actual optimized point**
   — parsed verbatim from `results/fullA_d4/b200eda/
   c9_phase8_d20_pilot_20260719_081058/summary.txt`'s `polish2.best_feasible`
   text (`qmc_fixed_points/upper_candidate_w.csv`), not re-derived.
3. **extreme_2x**: `w_calib + 2·(w_upper - w_calib)` — a directional
   extrapolation double the calibration→upper displacement, an explicit
   judgment call (cheap to construct, no new optimization needed).
4-5. **perturb1/perturb2**: two random directional perturbations around the
   upper candidate in pivot-reduced z-space (h=0.05).

4–6 independent scrambles per draw type per point (6 at calibration and
upper candidate, 4 at the other three, tiered by cost/importance — each
combo costs ~25–40s wall at W=80,000). At calibration and upper candidate
ONLY, the full 400-coordinate `composite_gradient_at_fast` gradient is also
computed per scramble for cross-scramble cosine similarity. No full
outer-loop optimization is run per scramble; a separate short (maxit=30
-capped) continuation from the upper candidate is run for 4 replicates only
(§5). Total wall time for the 72-combo fixed-point sweep: ~38 minutes.

## 3. Results: Δ_dual across draw types

| point | draw type | n | mean Δ | std Δ | rel. std | ratio vs pseudorandom |
|---|---|---|---|---|---|---|
| calibration | pseudorandom | 6 | 0.002625 | 0.000127 | 4.85% | 1.000 |
| calibration | halton | 6 | 0.001772 | 0.000125 | 7.04% | **0.675 (−32.5%)** |
| calibration | sobol | 6 | 0.000722 | 0.000053 | 7.36% | **0.275 (−72.5%)** |
| upper_candidate | pseudorandom | 6 | 1.174478 | 0.098867 | 8.42% | 1.000 |
| upper_candidate | halton | 6 | 1.156230 | 0.153496 | 13.28% | 0.985 (−1.6%) |
| upper_candidate | sobol | 6 | 1.127445 | 0.081725 | 7.25% | 0.960 (−4.0%) |
| perturb1 | pseudorandom | 4 | 1.208228 | 0.088056 | 7.29% | 1.000 |
| perturb1 | halton | 4 | 1.063196 | 0.064800 | 6.09% | 0.880 (−12.0%) |
| perturb1 | sobol | 4 | 1.148049 | 0.088256 | 7.69% | 0.950 (−5.0%) |
| perturb2 | pseudorandom | 4 | 1.214355 | 0.092422 | 7.61% | 1.000 |
| perturb2 | halton | 4 | 1.064206 | 0.064747 | 6.08% | 0.876 (−12.4%) |
| perturb2 | sobol | 4 | 1.150744 | 0.090312 | 7.85% | 0.948 (−5.2%) |
| extreme_2x | all 3 | 4 each | — | — | — | **all 12/12 evaluations inner_status=-300 (infeasible)** |

`inner_status=0` and `winner_zero_incidence=0` (every origin wins at least
one draw at every destination, out of 80,000) for all 60 feasible
evaluations — the calibration-point bias is not a feasibility/degeneracy
artifact.

**Calibration**: a large, tightly-replicated (6/6 scrambles agree within
their own ~5–7% spread) systematic LEVEL DIFFERENCE, not a variance
question. Both QMC methods' own relative spreads (7.0%, 7.4%) are actually
slightly WORSE than plain MC's (4.9%) — no variance-reduction benefit here;
the dominant effect is a large downward bias that GROWS with the sequence's
own discrepancy properties (Sobol, with only a simple shift, shows a much
larger bias than Halton's per-digit-scrambled sequence). A caveat: Δ_dual is
intrinsically tiny at calibration (the model is calibrated to fit well, so
Δ_dual sits close to its own floor) — small absolute differences read as
large percentage differences here more than they would further from that
floor.

**Upper candidate and its two perturbations**: no comparable systematic
bias. The original saved production point (seed 888) has Δ=0.999883,
essentially exactly at the δ=1 boundary — but under every one of the 6
OTHER pseudorandom seeds tested here, Δ lands well above 1.0 (mean 1.17,
range 1.03–1.32), i.e. mostly infeasible relative to δ=1 under a fresh seed.
This is expected, not a bug: the outer-loop optimizer specifically pushed
the A-block tight against δ=1 UNDER seed-888's own draw realization; a fresh
seed generally does not reproduce that exact tightness. **This is itself an
important, separate finding about how sensitive the reported κ is to the
specific draw realization it was optimized against** — worth flagging for
anyone citing a single-seed κ value literally, independent of the QMC
question. Against this ~8–9% pseudorandom-seed-noise backdrop, Halton/Sobol
sit within 1.6–12.4% of the pseudorandom mean at these four points — some of
that (the 12% at perturb1/2) may be a real, smaller-magnitude version of the
calibration bias, but it is not clearly separable from ordinary MC noise at
this replicate count, and does not grow with distance from the boundary the
way calibration's bias might suggest if extrapolated naively.

**extreme_2x**: uniformly infeasible (`inner_status=-300`) under all 3 draw
types × 4 scrambles = 12/12 evaluations — the 2× extrapolation pushes
gamma' (0.9236) below this model's feasible range (the c9 phase8 pilot
recorded `γp_lo≈0.9307` at this same context). A clean, consistent result:
the draw source does not change WHETHER this point is feasible, only
draw-dependent quantities would if it were.

## 4. Results: gradient direction stability (the most important single finding here)

Cosine similarity of the full 400-coordinate `composite_gradient_at_fast`
gradient across independent scrambles (15 pairs from 6 scrambles), and
across draw types (36 cross-pairs):

| point | comparison | n pairs | mean cosine | min cosine |
|---|---|---|---|---|
| calibration | pseudorandom (within) | 15 | 0.287 | −0.997 |
| calibration | halton (within) | 15 | 0.053 | −0.982 |
| calibration | sobol (within) | 15 | −0.066 | −0.989 |
| calibration | halton vs pseudorandom | 36 | −0.263 | −0.999 |
| calibration | sobol vs pseudorandom | 36 | 0.198 | −0.995 |
| upper_candidate | pseudorandom (within) | 15 | 0.9999987 | 0.9999966 |
| upper_candidate | halton (within) | 15 | 0.9999973 | 0.9999894 |
| upper_candidate | sobol (within) | 15 | 0.9999989 | 0.9999960 |
| upper_candidate | halton vs pseudorandom | 36 | 0.9999980 | 0.9999900 |
| upper_candidate | sobol vs pseudorandom | 36 | 0.9999988 | 0.9999952 |

**At calibration, the A-block gradient direction is essentially UNSTABLE
NOISE — for every one of the three draw types, not just QMC.** Mean
within-type cosine similarity across independent scrambles is 0.29
(pseudorandom), 0.05 (Halton), and −0.07 (Sobol) — i.e. even redrawing the
SAME draw type with a different seed gives an almost UNCORRELATED gradient
direction at this point, and cross-type comparisons are similarly poor
(as negative as −0.26 on average). This is not a QMC-specific defect: it
is evidence that Δ_dual's gradient is not well-identified by ANY
finite-W=80,000 estimator at the calibration point — plausibly because
Δ_dual sits at/near a genuine local minimum there (consistent with §3's
observation that Δ_dual itself is very small and close to its floor), where
the true gradient is near zero and dominated by estimation noise regardless
of the sampling method.

**At the upper candidate, by sharp contrast, the gradient direction is
extremely stable and virtually IDENTICAL across every draw type and every
scramble** — cosine similarity ≥0.9999894 in every one of 90+ pairwise
comparisons tested (within-type AND cross-type). This is the actual
decision-relevant region (this is where κ is read off, near the δ=1
constraint), and it says the outer-loop optimizer's gradient information
would be essentially unaffected by switching from pseudorandom MC to either
QMC method at W=80,000 — the calibration point's instability does not
transfer to the region that matters.

## 5. Short (maxit=30-capped) continuation from the upper candidate

*(Filled in once `c10_phase7_short_continuation.jl` completes — see
`results/fullA_d4/.../c10_phase7_short_continuation_*/summary.txt`.)*

## 6. Assessment: does W=80,000 randomized QMC match a larger MC run's precision?

**Reasoning from theory, not directly measured** (no large-W MC run was
executed as ground truth, per task scope): classical QMC variance-reduction
guarantees (Koksma-Hlawka-type bounds) require integrands of bounded
variation; this model's hard-argmin winner selection is discontinuous in the
underlying uniforms, so those guarantees do not formally apply, and this
investigation's own empirical results are consistent with that theoretical
gap rather than contradicting it — at calibration, neither Halton nor Sobol
showed LOWER within-type relative spread than plain MC (both were
slightly higher, 7.0–7.4% vs 4.9%), the opposite of the usual QMC
expectation. **We would NOT extrapolate from this evidence that a larger MC
run (e.g. W=800,000, matching CDW's own choice) is well-approximated by
randomized QMC at W=80,000** — if anything, the large, replicated bias at
calibration suggests a low-discrepancy sequence at this W is systematically
answering a SLIGHTLY DIFFERENT question (interacting with the discontinuous
winner-selection map) rather than a lower-noise version of the same one, at
least at points where Δ_dual is very close to its own floor.

At the decision-relevant upper-candidate region, however, both Δ_dual
(within ~4–12% of pseudorandom, itself noisy at ~8%) and — far more
convincingly — the gradient direction (cosine ≥0.99999 across every draw
type tested) show no meaningful difference from plain MC. This is a
genuinely two-sided finding: **QMC shows no demonstrated precision
advantage anywhere in this investigation, but it also shows no demonstrated
harm at the actual decision-relevant point**, only a real, replicated,
unexplained bias at a diagnostic anchor point (calibration) that this
investigation does not itself optimize over.

## 7. Verdict, per this task's explicit instruction

**Do NOT switch the production draw scheme based on this evidence.** The
brief requires REPLICATED evidence (multiple scrambles agreeing) before any
positive recommendation, and while the evidence here IS replicated (6/6
scrambles agree with each other within each draw type, at both points
tested), it does not point toward adopting QMC — at the one point where a
difference is large and consistent (calibration), it runs the WRONG
direction (a bias, not an improvement); at the point that matters for
reported κ values (upper candidate), there is no distinguishable difference
in either direction. **The honest summary is a null result on the
"does QMC help" question, plus an unexpected, real, and worth-flagging
finding that this model's own gradient/objective are simply noisy relative
to any W=80,000 estimator (of any draw type) very close to the divergence
floor** — a fact about this model's identification at that specific regime,
not about QMC per se.

## 8. Files

New, all under `full_aod_diag/d4_exact/`: `c10_phase7_rhalton_validate.jl`
(Halton validation), `qmc_context_real_d20.jl` + `qmc_draws.jl` (isolated
draw-injection infra), `c10_phase7_qmc_wiring_smoketest.jl` (equivalence/
wiring check), `c10_phase7_qmc_precision_comparison.jl` (main 72-combo
sweep), `qmc_fixed_points/{upper,lower}_candidate_w.csv` (parsed A-block
data, provenance noted in-file), `c10_phase7_short_continuation.jl` (§5).
Project.toml/Manifest.toml: added `Sobol.jl` (clean resolve, diffed). No
production file modified. Raw output:
`results/fullA_d4/c61317a/c10_phase7_qmc_precision_20260719_112614/`
(`fixed_point_comparison.csv`, `gradient_cosine.csv`, `harness_log.txt`).
