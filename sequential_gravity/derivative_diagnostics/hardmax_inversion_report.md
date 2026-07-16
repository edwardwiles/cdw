# Hard-max destination inversion: validation, methods, and a production solver swap

Status: hard-max validation COMPLETE on all 3 example points (D=20, W=80000, real data). A
correct, fast method (rho-continuation/homotopy, reusing `invert_destination` unmodified) was
found and validated; a first attempt (LP duality) was tried, found mathematically wrong for
this model, and abandoned. As a related but separate deliverable (user request mid-session),
`invert_destination`'s rho>0 (smoothed) branch was ALSO given a new default solver
(Optim.jl `NewtonTrustRegion`, replacing the hand-rolled LM-damped Newton as the default,
which remains available via `method=:handrolled`) -- this is now live in production
(`sequential_gravity/profiled_gravity.jl`). The hard-max verifier itself is ALSO now wired into
production (`sequential_gravity/hardmax_verify.jl`, called from `run_one_bound`) -- see Section 9
-- so every best-feasible point the batch loop saves is independently re-checked against the
true hard-max model before being reported, not just diagnosed after the fact.

## 1. The question

Do the smoothed-inversion (`rho=2e-3`) solutions this project's whole outer-loop machinery
produces also satisfy the TRUE hard-argmax economic model, or is there a meaningful,
economically-relevant gap? Three checks per converged theta: (1) focal trade shares under
hard-argmin -- already confirmed hard by construction (`EK_moments_focal_norm_directgp!`), (2)
non-focal ("omitted") destinations' trade shares under a GENUINE hard-max (rho=0) inversion for
`u`, not just the existing rho=2e-3 solution re-checked at rho=0, and (3) whether the resulting
full `u_mat` still satisfies the gravity-equation orthogonality moment.

## 2. Method 1 (tried, WRONG): transportation-LP duality

**Hypothesis**: `share(u)` at rho=0 assigns each draw's full probability mass to its
hard-argmax winner -- this looked, at first, exactly like the dual of a max-weight
transportation LP (source = draws with mass `p[s]`, sinks = origins with target mass
`lambda_hat[o]`), which would guarantee an EXACT solution always exists (strong LP duality) and
be solvable with a single canned LP call (JuMP+HiGHS, no KNITRO).

**Why it's wrong**: `dest_stats`/`dest_share`'s actual formula is
`share[o] = sum_s rweight[s]*W[s,o]`, where `rweight[s] = p[s]*exp(V[s](u)) / sum_s' p[s']*exp(V[s'](u))`
-- a SECOND, always-on (temperature fixed at 1, independent of rho) softmax reweighting of
draws by their own realized best value `V[s](u) = max_o(u[o]+log_x[s,o])`. This is a genuine,
permanent part of the Eaton-Kortum expenditure-share formula (converts "probability origin o
wins draw s" into "expenditure share"), present at rho=0 exactly as much as rho>0 -- NOT
smoothing of the origin-choice rule itself (that part IS genuinely hard, 0/1, at rho=0). The
naive LP treats the mass moved from draw `s` as the FIXED constant `p[s]`, silently dropping
this u-dependent reweighting -- turning the true entropy-regularized (partial) optimal
transport problem into a plain transportation LP, which is a different, wrong problem.

**Empirical confirmation**: on a synthetic problem with a KNOWN true `u`, the LP's recovered
`u` did NOT reproduce the target shares (hard-max share error 8.3e-2, both sign conventions
tried). On real data (D=20, W=80000, Point 1's first omitted destination) the LP took **833s**
and gave hard-max share error **2.26e-2** -- WORSE than simply evaluating the EXISTING rho=2e-3
solution's own hard-max gap (1.35e-4) with zero extra work. Both wrong and impractical;
abandoned.

## 3. Method 2 (works): rho-continuation (homotopy)

Decrease rho geometrically from the production value (2e-3) down to ~1e-8 (3 substeps/decade),
warm-starting `invert_destination` from the previous step's solution at each step, then attempt
one final native rho=0 solve warm-started from the rho=1e-8 endpoint. No new solver math --
100% reuse of the existing, already-validated `invert_destination`.

**Single-destination validation** (Point 1, first omitted destination): the rho=0 hard-max gap
(checked via `dest_share(...;rho=0)`) reaches a FLOOR very quickly (already at rho~1e-3) and
stays EXACTLY flat for 7+ further orders of magnitude of rho down to rho=1e-10 -- strong
evidence of a genuine STRUCTURAL floor (the target share vector, having been matched under the
smoothed model, is not exactly achievable by any strict no-split hard-argmax partition -- the
handoff prompt's hypothesis 2, confirmed), not a numerical artifact. Confirmed path-independent:
a much coarser 5-point rho schedule landed on the exact same floor value (8.322e-05).

A native COLD-started rho=0 solve (`u_init=nothing`) reproduces the ORIGINAL "just stalls"
observation -- it plateaus at a WORSE share error (2.5e-4) than the true floor within the
iteration budget. A WARM-started rho=0 solve (from the homotopy endpoint) converges in ONE
Newton iteration to the exact same floor (8.322e-05). So rho=0 is not fundamentally unsolvable
-- it just needs a good starting point, which the homotopy path supplies for free. This directly
resolves the open question the handoff prompt posed about whether "naive rho=0" genuinely fails
or just needs a different entry point: it's the latter.

Implementation: `hardmax_invert_destination` in `derivative_diagnostics/hardmax_inversion.jl`.

## 4. Full 3-step validation, all 3 example points (D=20, W=80000, real data)

| Point | budget | focal_err (Step 1) | hard-max err, max/mean (Step 2) | R_mean smoothed | R_mean hard-max |
|---|---|---|---|---|---|
| 1 (LC T1, mild) | 0.1 | 5.59e-08 | 8.59e-05 / 3.96e-05 | -2.264e-04 | -2.251e-04 |
| 2 (LU T2, medium) | 1.0 | 5.59e-08 | 1.56e-03 / 5.26e-04 | -2.054e-04 | -2.048e-04 |
| 3 (LU-multistart T3, hardest) | ~2.0 | 5.59e-08 | 1.78e-03 / 6.76e-04 | -1.601e-04 | -1.619e-04 |

(`||u_hardmax - u_rho2e-3||`: max 1.32e-2 / 1.33e-2 / 2.27e-2 across Points 1/2/3 respectively --
grows with stress, consistent with the LFD `p` becoming more concentrated/skewed at higher
delta, per prior project findings.)

**Step 1 (focal)**: essentially machine-precision (5.59e-08) at all 3 points, as expected --
sanity check passes trivially, no bug.

**Step 2 (non-focal, genuine hard-max inversion)**: the achieved gap is small in absolute terms
(under 0.2 percentage points of trade share even at the hardest point) but genuinely NONZERO and
grows with stress -- confirming the structural-floor hypothesis, not a solver weakness. One
nuance worth flagging: at Points 2 and 3, the DEDICATED hard-max inversion's error is actually
slightly LARGER than just taking the existing rho=2e-3 solution and checking its own gap at
rho=0 directly (informational row, not shown in the table above but present in the raw run
logs: Point 2 smoothgap_maxerr=1.47e-03 vs hardmax_maxerr=1.56e-03; Point 3
smoothgap_maxerr=1.13e-03 vs hardmax_maxerr=1.78e-03). This means the homotopy path does NOT
always land at the single best-achievable hard-max fixed point once the LFD weights are
concentrated/skewed enough (higher delta) -- there can be more than one locally-stable floor,
and which one you reach depends on the path. Both are still small in absolute terms, but this is
a genuine, honest limitation of the homotopy approach at higher stress, not something to gloss
over.

**Step 3 (gravity moment)**: R_mean barely moves between the smoothed and true hard-max
competitiveness matrices at any of the 3 points (differences in the 4th-5th decimal, an order
of magnitude smaller than R_mean itself) -- the gravity-equation identifying restriction, which
is what the whole outer-loop optimization machinery is actually driving toward, is essentially
unaffected by using the genuinely-hard-max-inverted `u_mat` instead of the smoothed one.

## 5. Honest read: are the smoothed-inversion solutions trustworthy?

**Yes, for the purpose this whole project cares about.** The focal side is exact by
construction. The non-focal hard-max gap is real but small (well within the project's own
established "nobody cares about trade shares beyond 2dp" tolerance convention), and -- most
importantly -- the gravity moment itself, which is the actual identifying restriction the outer
optimization targets, is essentially unchanged whether you use the smoothed or the genuinely
hard-max-inverted competitiveness matrix. The smoothing (rho=2e-3) is not hiding an
economically meaningful discrepancy from the true model at the delta/W scales tested. The one
caveat: the hard-max gap itself (not the gravity consequence of it) grows with stress and is not
perfectly path-independent at higher delta -- worth keeping in mind if a future use case cares
about the PER-DESTINATION trade shares themselves at high delta, rather than the gravity moment.

## 6. Related, separate finding: `invert_destination`'s smoothed (rho>0) solver

(User-directed exploration mid-session, not part of the original handoff prompt, but changes
production code so documented here too.)

The user asked for a canned-solver replacement for the hand-rolled LM-damped Newton used in the
rho>0 branch, specifically preserving its damping/trust-region-like robustness (a plain,
undamped Newton step provably overshoots on this problem -- confirmed both via NLsolve's
`:newton` and Optim's `Newton()`, both fail in ~2 iterations with a non-finite result,
regardless of whether a line search -- BackTracking or StrongWolfe -- is added: a line search
only rescales the SAME bad direction, it doesn't fix it).

**What worked**: Optim.jl's `NewtonTrustRegion`, given the exact same analytic gradient
(`share(u)-lambda_hat`) and Hessian (`share_jacobian_smoothed`) already in the codebase.
Tuning sweep (5 destinations, D=20/W=80000 real data, tol~1e-8): the hand-rolled solver actually
FAILED to satisfy its own convergence criterion on 3/5 destinations (hit its 150-iteration cap,
~28-29s each); `NewtonTrustRegion` converged cleanly on all 5, 3-15x faster. Tolerance sweep
(1e-6 vs 1e-7 vs 1e-10): IDENTICAL iteration count/wall time/share error at every tolerance
tested on every destination -- quadratic convergence means the last accepted step already lands
near machine precision regardless of the requested tolerance, so there is no cost to a tight
default. `initial_delta` sensitivity: robust across 0.1-3.0; 10.0 caused an outright non-finite
failure on one destination -- do not enlarge beyond the Optim default (1.0).

**What did NOT work**: NLsolve.jl's OWN `:trust_region` implementation (a DIFFERENT trust-region
algorithm than Optim's) did not converge reliably cold-started on this same problem across 3
different `factor` (initial radius) settings tried -- "trust region" is not a monolithic
guarantee; the specific implementation matters, and Optim's happened to work well here while
NLsolve's did not (not exhaustively tuned further).

**A real bug found and fixed during full-scale validation**: the first wrapper implementation
gated the returned `converged` flag on BOTH `share_err < tol` AND Optim's own `Optim.converged
(res)` (which checks x-space step-size/function-value convergence criteria). On one destination
(D=20/W=80000, Point 1 destination 7), the objective was flat enough near the optimum that
`Optim.converged(res)` returned false (x_abstol not satisfied) even though `share_err=3.7e-8`
-- WELL inside tolerance and BETTER than the hand-rolled solver's own answer for the same
destination (5.5e-8). Fixed by using `share_err < tol` alone as the convergence criterion,
matching the hand-rolled solver's own semantics exactly (its `converged` flag is likewise just
"did the share error drop below tol", not any Optim-internal notion). A separate, genuine
failure mode was also caught by the full-3-point validation run and handled by a fallback (not
silently swallowed): one destination in Point 3 produced an actual `NaN` gradient inside
`NewtonTrustRegion` (Optim's own "Terminated early due to NaN in gradient" warning); the
validation script detects non-convergence and automatically falls back to `method=:handrolled`
for that destination, which converged cleanly (share_err=1.7e-08).

**Production change**: `invert_destination`'s default is now `method=:trustregion` for rho>0;
`method=:handrolled` restores the original solver byte-for-byte unchanged; the rho<=0 (hard-max)
branch is completely untouched either way. Live in
`trade_robustness_modular_perf/sequential_gravity/profiled_gravity.jl` (working-tree change, not
committed). `Optim` added to `Project.toml`/`Manifest.toml` (+2 small transitive deps). This
changes the DEFAULT behavior for every existing caller of `invert_destination` with rho>0,
including `run_profiled_production.jl`'s own `seq_gravcol`/`invert_all` -- not just the hard-max
diagnostics in this file.

**W=800,000 scaling check** (10x the example points' W=80,000, same theta/destination as
Point 1's first omitted destination): trust-region remains both faster (33.3s vs 36.9s cold)
and more accurate (share_err 3.7e-8 vs 3.3e-7) than the hand-rolled solver at this scale too,
though the margin is smaller than the dramatic W=80,000 cases -- this particular destination
doesn't happen to trigger the hand-rolled solver's maxit-stall pathology at W=800,000 (both
converge in a similar ~18-21 iterations); the trust-region advantage is largest specifically on
destinations that would otherwise stall, and merely "modestly better" on ones that wouldn't
have anyway. The hard-max homotopy costs ~123s cold / ~94s warm-started from the trust-region
solution at this scale (each of its ~18 rho-schedule steps costs O(S*D), so total cost scales
with W roughly as expected). Notably, the hard-max FLOOR itself shrinks substantially at higher
W -- 2.66e-06 at W=800,000 vs 8.32e-05 at W=80,000 for the identical destination -- a satisfying
independent confirmation of the structural-floor theory in Section 3: more draws means a finer
achievable-share discretization, so the target vector can be matched more closely even though
it's still never exactly hit.

## 7. Files

- `derivative_diagnostics/hardmax_inversion.jl`: `hardmax_invert_destination`, the validated
  rho-continuation hard-max inversion (reusable, KNITRO-free).
- `derivative_diagnostics/hardmax_full_validation.jl`: the full 3-point/3-step validation
  driver (threaded across destinations, KNITRO needed once per point for `recover_lfd`).
- `derivative_diagnostics/tune_trustregion.jl`, `canned_softmax_solver*.jl`,
  `canned_softmax_optim.jl`: the canned-solver comparison/tuning scripts (Section 6).
- `derivative_diagnostics/hardmax_lp_single_dest.jl`, `hardmax_lp_smoketest.jl`: the abandoned
  LP-duality attempt (Section 2), kept for the record.
- `profiled_gravity.jl`: production file, modified (Section 6).

## 8. A note on the recover_lfd bug found concurrently (not this session's finding)

A separate Claude session, working in this same live repo concurrently, found and fixed a real
bug in `recover_lfd` (`run_profiled_production.jl`): it only checked `all(isfinite,x)` on the
KNITRO inner solve's result, not `nStatus` -- silently accepting a REJECTED (e.g. -300 =
UNBOUNDED) solve's raw, non-NaN'd garbage as if it were a legitimate LFD. This session's
isolated working copy was patched with the identical fix before any of the validation above was
run. All 3 example points used in this validation were independently re-confirmed GENUINE under
the fixed `recover_lfd` (Points 1 and 2 by the other session's own audit; Point 3 -- in
`out_lu_multistart50/`, a directory that audit does not cover -- independently by this session,
`worst_share_err=7.50e-07`, GENUINE).

## 9. Wired into production: `hardmax_verify.jl`

Follow-up request (user, same session): don't just leave this as a diagnostic -- run it
automatically as part of the production batch loop, on the actual point about to be reported,
so a point that can't be independently hard-max-verified is never silently reported without at
least a visible flag.

**What was moved to production** (`sequential_gravity/hardmax_verify.jl`, NOT
`derivative_diagnostics/`): `hardmax_invert_destination` (unchanged, Section 3) plus a new
`verify_hardmax_point(θ, umat, p; ref=1, tol=5e-4)` that reuses the `umat`/`p` a `seq_gravcol`
call already produced (no extra KNITRO call needed), runs the hard-max homotopy for every
omitted destination warm-started from its existing smoothed column, and checks the resulting
hard-max `u_mat`'s gravity residual with the SAME `abs(R_mean)<=tol` convention (`tol=5e-4`)
`seq_gravcol`'s own smoothed `gravity_ok` uses, so the two are directly comparable.
`derivative_diagnostics/hardmax_inversion.jl` now just `include`s this production file, so
there's a single source of truth (no more duplicated diagnostics-vs-production solver copies).

**Wiring** (`run_profiled_production.jl`, `run_one_bound`): the existing call
`_, Rb, _, _, _, okb = seq_gravcol(bθ; δ=δval, warm=bwarm)` was discarding `umat`/`p` -- changed
to capture them and, when `VERIFY_HARDMAX` (env var, default true), immediately call
`verify_hardmax_point` on the best-feasible point, print a `HARD-MAX VERIFY:` line (with a loud
`*** WARNING ***` if unverified), and thread the result through to the batch loop's `JLD2.save`.

**Design choice, confirmed with the user**: the new `hardmax_verified` field (plus
`hardmax_R_mean`, `hardmax_focal_err`, `hardmax_max_share_err`, `hardmax_mean_share_err`,
`hardmax_homotopy_all_ok`, `hardmax_verify_wall`) is ADDITIVE -- it does NOT change the meaning
or value of the existing `gravity_feasible`/`best_feasible_gravity_ok` fields (still the
smoothed-model check exactly as before), and a failed hard-max verification does not block or
retry anything in the outer loop. This was a deliberate choice over folding it into the existing
feasibility flag: it avoids silently changing behavior for any of the many existing downstream
consumers of those fields (warm-start chaining between delta values, `head_to_head` summaries,
`comprehensive_reaudit.jl`, etc.), and a hard gate isn't obviously well-founded anyway given
Section 4's finding that a nonzero hard-max share-error floor is EXPECTED/structural, not itself
a defect -- gating on it could reject genuinely fine points. `verified` is deliberately based
only on the gravity-moment consequence (what the outer optimization actually targets), not on
`max_hard_err` directly.

**End-to-end tested** (D=4, both the isolated working copy and the live repo, confirming no
interaction with the other session's concurrent dual-warm-start caching work): the new fields
appear correctly in the saved `.jld2`, and behave independently of the existing check as
designed -- one test point had `gravity-feasible=false` (failed the smoothed model's divergence-
budget constraint) but `hardmax_verified=true`, demonstrating the two checks are genuinely
answering different questions, not duplicating each other. Adds a few seconds at D=4; expect
roughly the Section 4/6 per-point wall times (tens of seconds to a few minutes) at D=20.
