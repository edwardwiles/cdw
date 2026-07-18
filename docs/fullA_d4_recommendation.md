# D=4 exact full-A: viability recommendation

Written at the end of the continuation session covering Phases A, B, D, F (mandatory setup +
corrections #1-#5 all resolved with primary evidence; Phases C, E, G not attempted — see
`docs/fullA_d4_final_report.md` §8). This is the concise answer the continuation prompt asks for;
`docs/fullA_d4_final_report.md` has the full evidence trail.

## Verdict

**(b) Viable only with a hybrid/corrected-gradient method — and even that verdict is provisional,
pending the W-stability and lower-direction work this session did not reach.**

Not (a) viable for scaling as-is: the exact optimized-value-FD approach that produced the upper
candidate is not demonstrated to scale — Phase D shows the inner-solve-skipping savings that would
make a cheap gradient method pay off are modest (2-4x) at this W=8000/D=4 scale, meaning the
$O(n\_free)$ FD-probe cost (not the inner solve) is the current bottleneck, and that only gets worse
as $n\_free$ grows with D. Not (d) not viable: the method does reach a genuine, exactly-feasible,
economically sensible candidate (κ≈0.1718) that survives fresh cold rechecks at two tolerances and a
real (if imperfect) KKT check — this is real signal, not noise.

## The four questions this continuation was organized around

**1. Is the maxit=40 upper point robustly locally optimal under exact hard-value checks?**
**No — but it is close.** `EXACT_FEASIBLE_CANDIDATE`: yes (fresh cold+warm recheck, two tolerances,
both pass cleanly). `H_BANDWIDTH_KKT_CANDIDATE` at the original h=0.01: yes. `ROBUST_LOCAL_CANDIDATE`:
**no** — the KKT residual is not stable across a bandwidth grid (0.12% at h=0.01 vs 1.9% at h=0.001,
and outright failure at h=0.02), literally every one of 20 random probe directions crosses a winner
boundary, and a real deterministic poll found 3 small-but-genuine exact-feasible improvements. The
practical reading: κ≈0.1718 is a good, defensible, nearly-optimal number for this synthetic economy at
this δ — not a number that should be quoted as an exact, provably-stationary bound without that
caveat.

**2. Can the lower direction be solved with profiling and continuation?**
**Unknown — not attempted this session.** The existing short run is honestly `BEST_FEASIBLE_STALLED`
(feasible, far from the divergence budget, simply out of iterations at maxit=15) — there is no
evidence it is intrinsically harder than the upper direction, only that it was not given the same
iteration budget or a profiling/continuation treatment. This is the single highest-value next step:
it is plausible (not yet shown) that a longer run or a `profile_Δ(g)` continuation from g=1 reaches a
comparably strong lower candidate.

**3. Can `L_fix` or a hybrid gradient reproduce optimized-value guidance at much lower cost?**
**Directionally yes, quantitatively not yet a clean win at this scale.** `L_fix` FD tracks the
optimized-value gradient's *direction* almost exactly (cosine 0.998-1.000) at every point tested
except the fragile lower-stalled point — a genuinely encouraging result for using it as a search
direction. But its *magnitude* is systematically off (norm ratio 0.4-0.75, needs rescaling, not a
drop-in replacement), and the wall-clock savings from skipping the inner re-solve are only 2-4x here,
because the inner CC dual solve is already cheap at W=8000. **The case for `L_fix`/hybrid strengthens
exactly where this session did not test it**: larger W (where the inner solve cost grows) and larger D
(where $n\_free$ and thus the FD-probe count grows). Phase G would settle this.

**4. Does the method remain stable as W rises enough to be relevant for production?**
**Untested this session.** All numbers here (this session's and the prior session's) are W=8000. Per
memory `d20-realdata-w-sensitivity`, W=8000 is known to understate κ relative to W≥80,000 at δ≥1 on
the *real* economy — whether the same bias, or a different one, applies to this synthetic D=4 economy
and to the specific candidates found here is completely open. **No D=10/D=20 claim should be attempted
before this is resolved** (per the continuation prompt's own explicit ordering).

## What to do next, in priority order

1. **Phase G (W-stability)** first, not last — it directly determines whether Phase D's "cost savings
   are modest at this scale" finding is a permanent verdict against hybrid methods or an artifact of
   testing at the smallest, least representative W. Cheap to start (re-run the existing maxit=40
   candidate's exact recheck at W=20,000 and see if it stays feasible/near-κ before committing to a
   full sweep).
2. **Phase C (lower-direction profiling)** — the lower direction has had roughly a third of the
   upper direction's iteration budget and zero profiling attention; this is very likely undersold, not
   genuinely harder.
3. **Phase E (sequential-solution continuation)** — cheapest of the three remaining phases once a
   comparable sequential/profiled run is located, and provides a mandatory sanity-check incumbent this
   investigation has been missing throughout (comparing full-A's κ against the production method's own
   number for the same economy/δ/seeds is the single most direct external validity check available).
4. Only after 1-3: revisit whether a wall-clock-matched (not iteration-matched) Hessian-mode
   comparison changes Phase B's picture, and whether an `L_fix`-primary/`Δ`-refresh hybrid scheme
   (task §12G's `should_refresh` policy, implemented but not wired into a live solve this session)
   closes the gap Phase D found.

## Update from continuation 2 (performance profiling / D=4 completion / staged scaling)

Full detail: `docs/fullA_performance_profile.md`, `docs/fullA_scaling_projection.md`,
`docs/fullA_d4_W_stability.md`, `docs/fullA_d4_final_report.md` §9. Three developments change the
picture above:

1. **Question 3 (can a cheap gradient replace optimized-value FD) is now answered more precisely, not
   just "directionally yes"**: blockwise re-analysis (Phase 6) shows `L_fix_FD` specifically — not
   cheap methods generally — survives scrutiny in the economically important gravity-tangent A-block
   (cosine 0.997-0.999, norm ratio ≈1.0 at both upper candidates), while hard pathwise AD's A-block
   gradient is actually **anti-correlated** with the truth near the optimum (cosine -0.67 to -0.79) and
   `Q_adj_FD` overestimates A-block magnitude by 14-17x. The prior "directionally yes, quantitatively
   not yet a clean win" verdict undersold `L_fix_FD` specifically and oversold the other two cheap
   methods.
2. **The D-scaling cost problem is now quantified, not just asserted**: full-gradient cost scales
   empirically as D^3.5-3.8 (4 points, D=4..10, W=8000 fixed) — projecting ~18 minutes per
   `Delta_FD` gradient at D=20 versus ~4.2 minutes per `L_fix_FD` gradient. This makes the case for a
   validated `L_fix`-hybrid solver strictly stronger at higher D, and makes an unoptimized D=20 attempt
   with the current architecture concretely, not just qualitatively, inadvisable.
3. **A genuine external validity check now exists**: the sequential/profiled production method, run
   fresh on the identical synthetic economy, gives a directly comparable (gauge-invariant) upper bound
   of κ=0.0779 (genuinely feasible) — full-A's κ=0.1718 candidate clears it by more than 2x, with the
   caveat that the sequential run used one start, not the production 5-start multistart, and did not
   fully converge (`nStatus=-400`).

## Final decision labels

Per the continuation task's required taxonomy:

- **D=4 mathematical/numerical viability**: `VIABLE_WITH_HYBRID_AND_PERFORMANCE_WORK`. The exact
  method reaches a genuine, exactly-feasible, externally-corroborated (beats the sequential method's
  own comparable number) candidate — but the profiling in this continuation identifies concrete,
  unimplemented performance work (redundant moments recomputation, `L_fix`-hybrid gradient, block
  reuse) needed before the method is efficient enough to iterate on freely, and Phase 4's gamma-
  profile/upper-polish/lower-completion work remains undone.
- **D=4, W=80,000 viability**: `NOT_CURRENTLY_VIABLE` (specific blocker: untested, not
  intrinsically hard) — only cost-scaling was measured at W=80,000 (calibration point, ~14.4s per
  `Delta_FD` gradient, tractable), no actual candidate was re-evaluated or re-optimized there. The
  blocker is time-budget, not evidence of a problem — the one real W=20,000 candidate check available
  was reassuring (more feasibility margin, not less).
- **D=6/8 pilot viability**: `VIABLE_WITH_VALIDATED_METHOD` for a computational pilot specifically
  (not a headline bound) — Phase 1C's cost data (`Delta_FD` gradients of 15-38s at D=6/8) is
  compatible with a short supervised pilot using the existing architecture as-is, no additional
  performance work strictly required to attempt one.
- **Projected D=10 viability**: `VIABLE_ONLY_AS_BENCHMARK` currently — `Delta_FD` gradients of ~100s
  make casual iteration impractical, but a single supervised computational pilot (per Phase 8A) is
  plausible; a headline D=10 bound should wait for the `L_fix`-hybrid to be wired into a live solver
  (Phase 6/7 follow-up) given the ~5x cost gap that gradient method offers at this D.
- **Projected D=20 viability**: `NOT_CURRENTLY_VIABLE`, precise blocker: projected ~18 minutes per
  `Delta_FD` gradient evaluation with the current (no block-reuse, no hybrid-gradient-in-solver)
  architecture makes even a short pilot prohibitively expensive; this is a concrete, evidence-based
  blocker (Phase 1C), not a qualitative caution, and it is specifically the redundant-moments-
  recomputation and missing-hybrid-gradient gaps identified in Phase 1/6 that need closing first, per
  the task's own Phase 8 gating criteria (none of which currently pass).

## Standing caveats carried forward

- Every κ number in this report and its predecessor is for the **synthetic D=4 economy, W=8000,
  δ=1** — not a calibrated economic estimate, and not yet checked against the analogous real-economy
  or larger-W results.
- The `smoothing_check.csv` non-determinism bug (§6 of the final report) is unresolved and,
  while scoped away from every number in this report, should be fixed before Method E is used for
  anything beyond the qualitative "smoothing removes discontinuities" check it was originally built
  for.
- No claim in this report or its predecessor is a global bound — every "candidate" label is a local
  claim, explicitly.
