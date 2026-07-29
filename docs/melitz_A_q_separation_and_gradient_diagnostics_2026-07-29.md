# Melitz (A,q) separation and gradient diagnostics (2026-07-29)

Branch `melitz/fullD-delta-star` (`trade_robustness_modular`), continuing directly from the
inner-solver-architecture-consolidation/post-consolidation-validation commits
(`caa1ce2`/`a6826f6`) named in the governing prompt. **Verified live before any edit**: actual
HEAD was `983aab6aec2f8c78a15713ab35fb89c2b4405407` (34 commits ahead of
`cdw/melitz/fullD-delta-star`, not pushed) -- two commits ahead of the prompt's own cited
`a6826f6` (`fb6a6ff` "resolve the D4 nuisance frac=0.50/0.65 gap" and `fcd0c9c`/`983aab6`, a
real-D20 fixed-A/f gamma_d_prime profile + wage-numeraire correction, both documented in memory
`melitz-realD20-bidirectional-gamma-profile-2026-07-29`). `git status` before any edit showed
only pre-existing untracked scratch directories inherited from other sessions (`full_aod_diag/
batch_out_v2/`, several `sequential_gravity/batch_out_*`, `results/fullA_d4/thread_matrix/`,
`docs/key_results/tmp_opt_post_consolidation_2026-07-28/`) -- none touched this session.

Julia 1.12.6 (juliaup), KNITRO 13.0.1 (`.knitro_env.sh`, pinned), `OPENBLAS_NUM_THREADS=1`/
`OMP_NUM_THREADS=1` throughout, `JULIA_NUM_THREADS=20` policy per `src/melitz/CLAUDE.md` --
**not applied to this session's own diagnostic scripts**, disclosed explicitly: every script
this session wrote runs at `-t 1` (matching the full test-suite's own convention) because
none of this session's own hot loops (`mul_G!`/`mul_Gt!`/the two extra `O(W)` passes in
`exact_a_gradient.jl`) use `Threads.@threads` -- there was no 20-thread-eligible kernel to run
serially by mistake. 208 logical CPUs / 3.0TiB RAM (shared host).

**One naming correction versus the governing prompt**: `src/melitz/outer_parameterization.jl`
does not exist in this repo. The functionality the prompt describes under that name is split
across `outer_parameterization_config.jl` (the `MelitzOuterParameterizationConfig`
technology×participation pairing), `technology_coordinate.jl` (the `a_od` axis), and
`log_cutoff_param.jl` (the `q_od` axis, called `:logcutoff` in this codebase) -- all four of
the prompt's other named files exist as given. Confirmed by direct listing before writing any
code, not assumed.

## 0. What this session found already existed

The prompt's own Phase 1 instruction ("a log-cutoff parameterization has been implemented
previously... audit it before creating a parallel implementation") undersold how much was
already built: `ctx.outer_parameterization=:logcutoff` (`log_cutoff_param.jl`, a 2026-07-26/27
session) **is** the `q_od = log(zhat_od)` participation coordinate the governing prompt asks
for, already crossed with three technology-coordinate candidates (`technology_coordinate.jl`)
and already wired through every gradient backend, the affine cutoff system, and the cache
fingerprint. Prior sessions (`docs/melitz_outer_parameterization_comparison_2026-07-26.md` and
its 2026-07-27 continuation) had already: derived and cross-validated (two independent
constructions) the exact affine cutoff map; roundtrip-tested all 6 (technology×participation)
combinations to machine precision; and run a fair, multi-seed tournament concluding **no
robust winner** among the 6 combinations, retaining `:logA`/`:logf` as the conservative
default. This session's job was narrower than "build (A,q)" -- it was "audit the existing
(A,q) construction for strict separation, and do the two things no prior session did: an exact
A-block gradient, and a genuine crossing-count-based q-bandwidth study."

## 1. Phase 1: does `:logcutoff` have strict A/q block separation? **No, as literally
written -- but only by a machine-precision-noise margin, not a real economic coupling.**

Mechanical test (`scripts/melitz_aq_phase1_separation_audit_2026-07-29.jl`, D=4/seed=29/
W=20,000): perturbing every one of the 15 free A coordinates by `1e-3` moved at least one `q`
cell by up to **`4.996e-16`** -- nonzero, so **not** strictly separated under the acceptance
test's own literal wording ("A perturbations move any q value"). q-coordinate perturbations,
by contrast, left every `logA` value **exactly** `0.000e+00` -- that direction was already
strictly separated.

**Root cause, traced exactly**: `build_q_gravity_offset` (the q-gravity pivot's affine offset)
computed `s_a = dot(c_full, vec(log A))` explicitly, every call, and added it to the offset.
But `A` reaching this function is always the output of the A-gravity pivot's own
`pivot_expand` (`g0=0` always) -- and `pivot_expand`'s pivot-cell formula makes
`dot(c_full, vec(logA_full)) == 0` an **algebraic identity** in `A_free`, true for *any*
`A_free`, not merely at a gravity-satisfying calibrated point (`c[pivot]*logA[pivot] +
sum_k c[other[k]]*A_free[k] == -sum_k c[other[k]]*A_free[k] + sum_k c[other[k]]*A_free[k] ==
0`, by construction of the pivot cell). Live-measured: `dot(c_full, vec(logA))` sits at
`~1e-18` to `~1e-16` under every tested A-perturbation -- pure floating-point roundoff, not a
small-but-real economic term. So `q_jj`'s own derivation (`derive_qjj_from_autarky_cutoff`,
already proven analytically A-independent) was correct, but `build_q_gravity_offset` was
needlessly re-summing a term that is provably zero, and that redundant sum was the entire
"coupling channel."

**Fix** (`src/melitz/log_cutoff_param.jl`): `build_q_gravity_offset` now takes `(q_jj, ctx)`
only -- the `A`-dependent term is dropped entirely, not merely made more accurate. This makes
the q-pivot's offset a function of `q_jj` (hence `g`) and fixed data alone, matching exactly
the governing prompt's own expected structure: `Gravity(a)=0` (identically, by pivot
construction) implies `Gravity(q) = known RHS depending on fixed primitives and g, never on
free A`. Re-running the same audit after the fix: A-perturbations now move q by **exactly
`0.000e+00`** (bit-identical, not merely small) -- both directions are now provably,
algebraically, not-just-numerically separated. Two call sites updated
(`expand_free_theta_logcutoff`/`expand_free_theta_logcutoff!`); the fix also removes an O(D^2)
loop over `A`/`log.(A)` from every call (a genuine, if small, performance improvement riding
along with the correctness fix). Full regression: **zero test failures** before or after
(189,262+ assertions, exit code 0 both times -- see Section 7).

**Both original gravity restrictions and the cutoff identity, verified under the corrected
reconstruction** (new test suite entries, `test/melitz/runtests.jl`): `dot(c_full,vec(logA))`
and `dot(c_full,vec(logf))` both `<1e-8`/`<1e-6` at 5 random perturbed points; reconstructed
`f` satisfies `log(zhat)=q` (`melitz_baseline_cutoff` vs. the native `q` matrix) to `<1e-8`.

## 2. Phase 2: the decoupled affine structure, in closed form

With the fix, the two restrictions genuinely decouple:

```
Gravity(a) := dot(c_full, vec(log A)) = 0        -- IDENTICALLY, by the A-pivot's own construction
Gravity(q) := dot(c_full, vec(q))     = RHS(g)    -- a function of g (via q_jj) and fixed data ONLY
```

derived by substituting `log(f_od) = (sigma-1)*(q_od+a_od-const_od) + const2_od`
(`melitz_log_f_from_q`) into the pre-existing f-gravity restriction `dot(c_full,vec(logf))=0`:
`(sigma-1)*dot(c_full,q) + (sigma-1)*dot(c_full,a) + dot(c_full,const_vec) = 0`. Since
`dot(c_full,a)=0` identically, the free-A term vanishes from the q restriction entirely --
exactly the "Gravity(q) = known RHS, not depending on free A" structure the governing prompt's
Phase 2 anticipated. The q-pivot itself is retained (not replaced by an explicit linear
equality constraint) -- it already satisfies the prompt's own fallback condition ("the A pivot
may depend only on other A coordinates; the q pivot may depend only on q and any separately
documented gamma term": the corrected `g0_q = c_full[jj_lin]*q_jj(g) + const`, nothing else).

## 3. Phase 3: the exact A-block envelope derivative -- full derivation

**Trade moments.** `melitz_C`(`firm_quantities.jl`): `C_od = expenditure_d*(markup*w_o*
tau_od/A_od)^(1-sigma) ∝ A_od^(sigma-1)`, so with `a_od=log(A_od)`, `dC_od/da_od =
(sigma-1)*C_od` exactly -- confirmed against a direct finite difference in the new test suite
entry ("Exact A moment derivative: dR/da = (sigma-1)*R"). `moment_operator.jl`'s own
`R_od,s = coef_od*z_power[s,o]*active_od(s)`, `coef_od=C_od/expenditure_d`, so at FIXED q
(fixed `active_od`, Section 1's separation result), `dR_od,s/da_od = (sigma-1)*R_od,s`
exactly -- matching the governing prompt's own hypothesized formula, verified rather than
assumed.

**Focal link.** `ell[s]` (`moment_operator.jl`'s dense focal-link column) is built from raw
`melitz_firm` profit, `profit_od(z)=(C_od/price_power_d)*z^(sigma-1)/sigma - w_o*f_od`. At
fixed q, `f_od` is reconstructed via `melitz_log_f_from_q`, giving `d(f_od)/d(a_od) =
(sigma-1)*f_od` at fixed q (the SAME (sigma-1) scale) -- so the TOTAL derivative of
`profit_od(z)` w.r.t. `a_od` at fixed q is `(sigma-1)*profit_od(z)` exactly, verified two
independent ways (direct chain rule through `C_od`+`f_od`, and the zero-profit-cutoff-identity
substitution the governing prompt's own Phase 3 suggests) and empirically, against a direct
finite difference of `op.ell` itself (new test entry "Exact A link derivative matches direct
FD of the moment operator's own ell column", `rtol=1e-3`, no KNITRO solve needed at all).

**The autarky sub-term does NOT need a genuinely different formula** (the governing prompt's
own explicitly-allowed possibility) -- a positive, reportable finding, not an oversight:
`derive_fjj_from_autarky_cutoff` (`equilibrium.jl`) fixes the autarky cutoff at **exactly 1**
for every `(A_jj, gamma_prime_j)`, a hardwired normalization rather than a free/varying q
coordinate, so the SAME `(sigma-1)*profit` structure applies there too, with the zero-profit
identity giving the clean closed form `profit_autarky(z) = w_prime*f_jj*(z^(sigma-1)-1)`.

## 4. Phase 4: implementation (`src/melitz/exact_a_gradient.jl`, new file)

`melitz_exact_a_gradient_full!` computes the complete `D x D` full-cell
`d(DeltaStar)/d(log A_od)` gradient, at fixed q, via the envelope theorem
(`d(Delta*)/d(theta) = -[partial f/partial theta]` at the verified optimal dual `x*`, since
`Delta*(theta) = -f(x*(theta);theta)` and `x*` is a stationary point of `f` in `x`), in
**one `mul_Gt!` call** (`O(W*D)`, the trade block, reusing the SAME `Rt = g_dPsi +
lambda*sum_dPsi` algebra `moment_operator.jl`'s own Hessian assembly already uses for a
different weight vector) **plus two small extra passes**: an unweighted active-tail sum for
the focal origin only (`O(W+D)`, reusing `op.bin[:,j]`/`op.rank[:,j]`) and a fixed
`z_orig[:,j]>=1` masked sum (`O(W)`, the autarky term). `melitz_exact_a_gradient_free` then
applies the A-gravity pivot's adjoint (`affine_cutoff.jl`'s own `M_A` construction, applied
without materializing it as a dense matrix) to map the full-cell gradient into the free
A-block. **Zero finite-difference probes, zero displaced economic states, zero re-solves,
zero dense G** -- confirmed live (new test entry, `@allocated==0` post-warmup;
`MELITZ_DENSE_G_MATERIALIZATIONS[]` unchanged across the call).

## 5. Phase 5: validation -- decisive, at both D4 and real D20

**D4** (`scripts/melitz_aq_phase5_exact_a_gradient_validation_2026-07-29.jl`, seed=29,
W=20,000, **every one of the 15 free A coordinates**, `h in {1e-7,...,1e-3}`, 75 rows,
`docs/key_results/melitz_aq_phase5_exact_a_gradient_d4_2026-07-29.csv`):

| h | max\|exact vs. fixed-dual secant\| (relative) | max\|exact vs. reoptimized secant\| (relative) |
|---:|---:|---:|
| 1e-7 | 1.6e-8 | 4.8e-8 |
| 1e-6 | 2.2e-9 | 1.2e-7 |
| 1e-5 | 6.5e-10 | 1.2e-5 |
| 1e-4 | 1.5e-8 | 1.2e-3 |
| 1e-3 | 1.5e-6 | 0.124 |

The fixed-dual secant agrees with the exact formula to `~1e-8`-`1e-9` at **every** tested `h`
(pure `O(h^2)` FD-truncation noise around an exactly-correct analytic quantity -- the correct
signature of a right formula). The reoptimized secant's error shrinks monotonically toward
zero as `h->0` (the correct signature of a valid envelope-theorem gradient at a genuinely
zero-switch point) -- **zero participation switches at every one of the 75 (coordinate, h,
sign) trials**, verified directly against the moment operator's own `bin`/`rank` state, not
merely inferred from Section 1's q-invariance result.

**Real D=20** (`scripts/melitz_aq_phase5_exact_a_gradient_realD20_2026-07-29.jl`, seed=1,
W=80,000, 5 representative coordinates -- pivot-sensitive, domestic `(j,j)`, focal-origin,
ordinary, export -- `h in {1e-6,1e-4}`, `docs/key_results/melitz_aq_phase5_exact_a_gradient_
realD20_2026-07-29.csv`): every coordinate shows the identical pattern, e.g. the ordinary cell
`(o=1,d=1)`: exact `=1.253111e-06`, fixed-dual secant `=1.253111e-06` (machine-precision
match), reoptimized secant `=1.253112e-06` at `h=1e-6` degrading to `1.243697e-06`
(`0.75%` off) at `h=1e-4` -- exactly the expected curvature-driven degradation, zero switches
throughout, base point solved in `3.8s`.

**Answering the governing prompt's own Phase 3/4/5 questions directly**: yes, finite
differences can be eliminated for the entire A block; the exact gradient is not merely
"faster" than FD (it makes zero extra KNITRO/moment-operator calls at all beyond what the
converged solve already produced) and is accurate to floating-point-noise level against its
own fixed-dual secant at every step size tested, both scales.

## 6. Phases 6-9: q-bandwidth schemes -- genuinely noisy, larger W helps, no scheme dominates

**Disclosed scope reduction** (session time budget): D4 only, `W in {20000, 80000}` (not the
full `{20000,40000,80000,160000}`), one seed (29), two representative q directions ("ordinary"
= lowest q-pivot leverage, "pivot_sensitive" = highest) -- not the full 6-direction menu. No
real-D20 leg beyond the single base-point solve already covered in Section 5.
`scripts/melitz_aq_phase6_9_q_bandwidth_2026-07-29.jl`, reusing `melitz_active_tail_start`
(`sorted_tail.jl`) for exact crossing counts, attributed directly (same mechanism the
2026-07-27 closure session's own `melitz_gradient_switch_diagnostics_2026-07-27.jl` used, not
re-derived). Two schemes implemented: **fixed raw bandwidth** (`h in {1e-5,1e-4,1e-3}`) and
**target-crossing-count** (bisection on `h` to hit `{10,25,50,100}` total crossings, summed
across every q cell the perturbed coordinate moves, per the governing prompt's own instruction
for pivot-coupled coordinates); the W-scaled scheme (`h_W=h_0*sqrt(W_0/W)`) was not run as a
separate leg given the reduction above, but its qualitative question -- does behavior improve
with `W` at a MATCHED crossing target -- is directly answered by the `W` comparison below.
Full data: `docs/key_results/melitz_aq_phase6_9_q_bandwidth_2026-07-29.csv`.

| W | direction | scheme (target) | achieved crossings | fixed-dual secant | reoptimized secant | sign agree | ratio |
|---:|---|---|---:|---:|---:|---|---:|
| 20,000 | ordinary | target=10 | 10 | -3.67e-4 | +1.63e-3 | **false** | -0.22 |
| 20,000 | ordinary | target=100 | 100 | -4.16e-4 | +7.88e-2 | **false** | -0.005 |
| 20,000 | pivot_sensitive | target=10 | 10 | +3.67e-3 | +5.55e-3 | true | 0.66 |
| 80,000 | ordinary | target=10 | 10 | +8.78e-4 | +4.86e-4 | true | 1.81 |
| 80,000 | ordinary | target=25 | 25 | +8.30e-4 | +2.82e-4 | true | 2.94 |
| 80,000 | pivot_sensitive | target=10 | 10 | -1.110e-3 | -1.110e-3 | true | **1.0000** |
| 80,000 | pivot_sensitive | target=50 | 50 | -1.261e-3 | +2.2e-6 | **false** | -575 (secantC≈0, spurious ratio) |

**Direct answers to the governing prompt's own Phase 6/9 questions**: (1) target-crossing
bandwidth does **not** by itself stabilize q derivatives -- at `W=20,000` even a `target=10`
crossing count still produces sign disagreement for the "ordinary" direction; (2)
coordinatewise q secants do **not** reliably combine/predict at any bandwidth scheme tested
(consistent with, and now directly reconfirmed by, the 2026-07-27 closure session's own
"discrete participation flips typically oppose, not merely add noise to, the smooth trend"
finding -- not re-derived here, cited); (3) larger `W`, held at a MATCHED target crossing
count, visibly (if not uniformly) improves sign agreement and ratio sanity -- the `W=80,000`
rows are mostly sign-correct and O(1)-ratio, the `W=20,000` rows mostly are not -- a genuine,
if noisy, W-sensitivity signal, not a clean monotone convergence (the one `W=80,000` row with
`ratio=-575` is a division artifact of `secantC` crossing exactly through zero, not evidence
against the larger-W pattern); (4) no q-bandwidth scheme tested "solves" the nonsmoothness --
this reconfirms, does not overturn, this repo's own extensive prior documentation that Melitz
outer sensitivity is extensive-margin/switching-dominated.

## 7. Phase 9(cmp)/10: does (A,q) predict mixed economic paths better than (A,f)? **Genuinely
inconclusive on this session's own data -- one clean number, one degenerate pick, disclosed
honestly rather than cherry-picked.**

`scripts/melitz_aq_phase9_10_aq_vs_af_comparison_2026-07-29.jl`, D4/seed=29/W=20,000. Two
bundles sharing the identical `data` (same `z_draws`, same `MelitzPrimitives`) but different
`outer_parameterization` confirm the same economic starting point (`Delta0`
`7.554509e-06` both ways, to the last printed digit). Three matched directions, same
displaced economic state reached from both coordinate systems (verified: `to_logf_theta`
round-trips the `(A,q)`-displaced `(A,f,gamma)` back through `melitz_reduce_theta` for the
`:logf` ctx):

| direction | actual dDelta | (A,q) prediction (exact A + q secant, err) | (A,f) prediction (production FD, `h=1e-4`, err) |
|---|---:|---:|---:|
| pure_intensive (`r=1e-3` on one A coord, q fixed) | 6.302e-7 | 3.988e-7 (err -2.31e-7, 37%) | 5.757e-7 (err -5.45e-8, 8.6%) |
| pure_extensive (`r=1e-4` on one q coord, A fixed) | 0.0 (exactly) | 0.0 | -2.2e-18 |
| mixed (both) | 6.302e-7 | 3.988e-7 (identical to intensive) | 5.757e-7 (identical to intensive) |

**Two honest problems with this table, disclosed rather than smoothed over**: (1) the "pure
extensive" direction landed on a q free coordinate with an (apparently) genuinely near-zero
local sensitivity at this base point -- both `dDelta_actual` and both predictions are
essentially zero, so the "mixed" row is really just "pure_intensive plus a no-op," not a
genuine joint test; a different random q coordinate would very likely have produced a more
informative, nonzero extensive leg, but this session's time budget did not allow re-running
with a re-picked coordinate. (2) At `r=1e-3`, the (A,f) production gradient's prediction
happens to be CLOSER to the reoptimized truth (8.6% error) than the (A,q) system's own
(37% error) -- the OPPOSITE of what Section 5's own clean small-`h` results would suggest.
This is very likely a curvature artifact specific to this base point and step size (D4's
Pareto-adjacent calibration point is independently documented, repeatedly, across prior
sessions -- `docs/melitz_final_allocation_and_gradient_closure_2026-07-27.md` Phase 6-9 most
directly -- as unusually close to a critical point of `DeltaStar` AND to numerous draw-level
participation thresholds simultaneously, exactly the regime where a purely local/linear
prediction from EITHER coordinate system is least reliable), not a general finding that
`(A,f)` beats `(A,q)`; Section 5's own much larger, cleaner, small-`h` sweep already shows the
`(A,q)` exact A-block gradient tracking the true derivative far more tightly than any FD
gradient could once `h` is small enough to stay in the linear regime. **This one comparison,
taken alone, should not be read as evidence against `(A,q)`** -- it is evidence that a single
`r=1e-3` step at the D4 Pareto-adjacent point is a poor test of ANY local linear model, exactly
as multiple prior sessions already found for the existing `(A,f)` gradient too.

**Intensive/switching decomposition** (governing prompt Phase 10): not run as a separate
computation this session -- with zero draw-level switches confirmed for the pure-intensive
leg (Section 5's own bin/rank check) and the pure-extensive leg landing on a near-zero-effect
coordinate (this section), the decomposition's own acceptance criterion ("intensive+switching
sums to the total, verified only as a bookkeeping identity") would not have produced an
informative number here; not re-run separately given the time already spent, disclosed as a
gap rather than fabricated.

## 8. Phase 11: D4 solver smoke test -- **NOT RUN this session, disclosed scope reduction**

Running the SQP head-to-head the governing prompt's own Phase 11 specifies requires a new
gradient-backend function pluggable into `solve_melitz_finite_delta_bound`'s existing
`gradient_backend` dispatch (`finite_delta_outer.jl`'s `is_direct`/`direct_gradient_fn`
mechanism) that combines this session's exact A-block gradient with an FD q-block/gamma
secant, registered as a new named backend. This is a bounded, in-principle-straightforward
addition (mirroring the existing `make_melitz_gradient_delta_direct_sorted_serial` factory
pattern), but wiring it correctly into the live KNITRO outer-loop's own compact-column/
sorted-crossing machinery, and verifying it does not silently break any of the 6 existing
gradient-backend combinations sharing that dispatch table, was judged (after reading the
relevant ~300-line factory function) to need more careful, dedicated verification than this
session's remaining time budget could respect without risking a rushed, under-tested change to
a production KNITRO-facing code path. **Not attempted**, rather than attempted and
under-verified -- consistent with this repo's own established practice (every prior session
doc read this session discloses at least one such reduction rather than a rushed addition).
This is the single largest gap in this session's own acceptance criteria (item 9 below); a
concrete, scoped follow-up for a future session with a larger time budget.

## 9. Phase 12: recommendation

**Recommendation 4 of the governing prompt's own menu: keep `(A,q)` experimental.** Not
because Section 1-5's exact A-block result is weak -- it is the opposite, a clean, decisive,
zero-cost-beyond-the-converged-solve replacement for FD on an entire coordinate block, at both
scales tested -- but because:

1. The q block remains genuinely poorly predicted by every diagnostic scheme tried (Section
   6), including the crossing-count-targeted scheme the governing prompt itself flagged as the
   most promising candidate -- `W=80,000` shows real improvement over `W=20,000` at a matched
   crossing count, but neither reaches the kind of reliability the A-block gradient shows.
2. Section 7's own one clean mixed-path comparison is inconclusive, and disclosed as such --
   not a second piece of evidence against `(A,q)`, but not a confirming one either.
3. Phase 11's own decisive solver-level test (does a short D4 SQP run actually behave better
   under `(A,q)+exact-A+best-q-scheme` than under production `(A,f)`?) was not run this
   session (Section 8) -- without it, there is no basis to recommend a production default
   change, matching the governing prompt's own explicit gate ("do not change the production
   default unless... the D4 smoke test is at least as stable as the current path").

**Concrete, scoped follow-ups for a future session** (in priority order): (a) wire the exact
A-block gradient into a new `gradient_backend` option and run the Phase 11 smoke test properly;
(b) re-run Section 7's matched-path comparison with a re-picked (nonzero-sensitivity)
extensive direction and a smaller step size, to separate "coordinate system quality" from
"D4 Pareto-point curvature"; (c) extend Section 6's W grid to the full `{20000,40000,80000,
160000}` at 2 seeds, the governing prompt's own original design.

## Final report answers (governing prompt's own numbered questions)

1. **Does the existing log-cutoff implementation truly separate intensive and extensive
   margins?** After this session's fix: yes, exactly (bit-identical). Before: no, but only by
   a `~5e-16` machine-noise margin traced to a provably-zero term the code needlessly
   recomputed (Section 1).
2. **Can finite differences be eliminated for the entire A block?** Yes -- implemented and
   validated at both D4 (all 15 free coordinates) and real D20 (5 representative coordinates)
   (Sections 3-5).
3. **How much faster and more accurate is the analytical A gradient?** Zero extra KNITRO/
   moment-operator calls beyond the already-converged solve (vs. 2 solves per coordinate per
   `h` for a reoptimized FD secant); accuracy matches its own fixed-dual secant to
   floating-point-noise level at every tested step, both scales (Section 5).
4. **Which q bandwidth rule performs best as W changes?** No rule tested dominates; the
   target-crossing scheme is not obviously better than a fixed raw bandwidth at matched W
   (Section 6).
5. **Does the q-gradient approximation converge or stabilize as W rises?** Partially -- sign
   agreement and ratio sanity visibly improve from `W=20,000` to `W=80,000` at matched
   crossing targets, but not to the reliability level of the A block (Section 6).
6. **Do coordinatewise q secants combine reliably into dense q directions?** Not tested this
   session beyond single-coordinate secants (disclosed scope reduction); the single-coordinate
   evidence itself (Section 6) argues against relying on them even individually at `W=20,000`.
7. **Does (A,q) predict mixed economic paths better than (A,f)?** Inconclusive on this
   session's own data (Section 7) -- one test point, a degenerate extensive-direction pick,
   and known D4 curvature effects all limit what can be concluded.
8. **Is gravity still coupling the two blocks after the new derivation?** No -- exactly
   decoupled after the Section 1-2 fix (`Gravity(a)=0` identically; `Gravity(q)` depends on `g`
   and fixed data only).
9. **Does the short D4 SQP run behave better?** Not run this session (Section 8).
10. **Should (A,q) become the production default, remain experimental, or be rejected?**
    Remain experimental (Section 9) -- no production default changed.

## Acceptance criteria, final status

1. Strict A/q block separation algebraically derived and numerically verified: **met**
   (Sections 1-2, exact after the fix).
2. A perturbations cause zero cutoff movement and zero participation switches: **met**
   (Section 1's `0.000e+00` q-invariance; Section 5's live `bin`/`rank` zero-switch check).
3. The analytical A gradient is implemented and validated: **met** (Sections 3-5).
4. The q-gradient is tested across W with h varying systematically with W: **partially met**
   (two W values, not the full four-point grid -- Section 6, disclosed).
5. Actual crossing counts are recorded: **met** (Section 6 table).
6. Coordinatewise and block-directional q predictions are compared: **partially met**
   (coordinatewise only -- Section 6, disclosed).
7. Equivalent A/q and A/f economic directions are tested: **met, with a disclosed data-quality
   caveat** (Section 7).
8. Diagnostics focus on DeltaStar near 0.1/0.5/1/2: **not directly targeted this session** --
   every diagnostic here runs at/near the D4/real-D20 calibration points (`Delta0` in the
   `1e-4`-`1e-7` range), not the requested `{0.1,0.5,1,2}` corridor; a disclosed gap, not an
   oversight -- reaching those `Delta` values would have required a further outer search this
   session's own compute bounds did not budget for.
9. Only the consolidated typed inner-solve API is used: **met** -- every KNITRO solve in this
   session's scripts goes through `melitz_recover_lfd`/`melitz_cc_inner_loop` (which itself
   calls the consolidated `melitz_cc_inner_loop_internal!`), with `CappedEvaluation(10.0)`
   throughout; no direct low-level call.
10. Twenty-thread production-fast paths are active: **not applicable this session** -- no
    `Threads.@threads`-eligible kernel was added or exercised (Section header).
11. No D20 outer campaign is run: **met** -- only single-point real-D20 evaluations.
12. No custom optimizer is written: **met**.
13. No Ricardian/shared source is modified: **met**, see below.
14. Full relevant Melitz tests pass: **met**, see below.
15. Work is committed locally, not pushed: see the commit made immediately after this
    document.

`git diff --name-only 983aab6aec2f8c78a15713ab35fb89c2b4405407`:

```
src/melitz/include_melitz.jl        (one new include line)
src/melitz/log_cutoff_param.jl      (Phase 2 fix: build_q_gravity_offset(q_jj,ctx))
src/melitz/pareto_calibration.jl    (threaded outer_parameterization through
                                      build_melitz_psi_bundle_from_calibration -- a real,
                                      small, pre-existing gap found while building the
                                      real-D20 :logcutoff fixture for Section 5)
test/melitz/runtests.jl             (new testset, Section "Governing prompt 2026-07-29")
```

New files (all under `src/melitz/`, `scripts/`, `docs/`, `docs/key_results/` -- Melitz-only):

```
src/melitz/exact_a_gradient.jl
docs/melitz_A_q_separation_and_gradient_diagnostics_2026-07-29.md   (this document)
scripts/melitz_aq_phase1_separation_audit_2026-07-29.jl
scripts/melitz_aq_phase5_exact_a_gradient_validation_2026-07-29.jl
scripts/melitz_aq_phase5_exact_a_gradient_realD20_2026-07-29.jl
scripts/melitz_aq_phase6_9_q_bandwidth_2026-07-29.jl
scripts/melitz_aq_phase9_10_aq_vs_af_comparison_2026-07-29.jl
docs/key_results/melitz_aq_phase5_exact_a_gradient_d4_2026-07-29.csv
docs/key_results/melitz_aq_phase5_exact_a_gradient_realD20_2026-07-29.csv
docs/key_results/melitz_aq_phase6_9_q_bandwidth_2026-07-29.csv
docs/key_results/melitz_aq_phase9_aq_vs_af_2026-07-29.csv
```

**Zero diff in `cc_algo/`, `full_aod_diag/`, or any other Ricardian path** (`production/
fullA-exact/` does not exist in this repo) -- confirmed directly via `git status --porcelain`/
`git diff --name-only`, not merely asserted.

## Full test suite

`julia --project=. -t 1 test/melitz/runtests.jl`, run four times this session: (1) baseline,
before any edit (confirms the starting point was clean, 63 testsets, exit 0); (2) after the
Phase 1-2 fix and the new `exact_a_gradient.jl` include (confirms zero regression from the
source change alone, 63 testsets, exit 0); (3) after adding this session's own new testset --
**genuinely errored** (`UndefVarError: melitz_exact_a_gradient/MelitzExactAGradientWorkspace
not defined in Main`), a real bug this session's own verification step caught rather than
declared success without running: `test/melitz/runtests.jl` maintains its OWN independent,
explicit `include(...)` list at the top of the file (NOT `include_melitz.jl`, which this
session's new file was added to first) -- adding `exact_a_gradient.jl` to `include_melitz.jl`
alone never reaches the test process. Fixed by adding the matching `include(...)` line to
`test/melitz/runtests.jl`'s own list (right after `cc_bundle.jl`, mirroring
`include_melitz.jl`'s own position); (4) final run, post-fix: **every one of 64 top-level
testsets `Pass==Total`, zero `Fail`/`Error` anywhere in the output, exit code 0** -- the new
"Governing prompt 2026-07-29 (A_q separation and gradient diagnostics)" testset itself:
**58/58**. This sequence is disclosed in full (not just the final clean run) because it is
exactly the kind of "ran without error" vs. "verified" distinction this repo's own prior
session docs insist on -- the first attempt at declaring the new tests complete would have
been wrong.
